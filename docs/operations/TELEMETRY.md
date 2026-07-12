# Telemetry

token-miser can record what it routes and what it (would) cost, so you can
decide whether tier routing actually saves money **before** trusting it. It is
**off by default** and adds no overhead until enabled.

## Why

Model routing only saves money if the cheaper tier is good enough for the work
sent to it. A wrong downroute means the user retries on the expensive model
anyway — you pay twice. Telemetry plus **shadow mode** lets you measure the
downroute rate and cost on real traffic while every request still gets the safe
(baseline) model, so there is no quality risk during evaluation.

## Enabling

Add a `[telemetry]` section to your config file (`config.toml`, loaded
automatically, or any file named by `TOKEN_MISER_CONFIG`). A complete working
config is in [`config.example.toml`](../config.example.toml).

```toml
[telemetry]
enabled = true          # master switch; nothing is recorded unless true
shadow_mode = true      # route everything to shadow_tier, log what would've been chosen
shadow_tier = "complex" # baseline tier: "free" | "standard" | "complex"
log_path = "telemetry.jsonl"  # optional; appends one JSON object per request
```

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `false` | Master switch. When false, no records are produced and routing is untouched. |
| `shadow_mode` | `false` | Route every request to `shadow_tier` while recording the tier the router *would* have chosen. Only takes effect when `enabled` is also true. |
| `shadow_tier` | — | The baseline tier used while shadowing. If unset/invalid, shadow routing is disabled (a warning is logged) and live routing is used. |
| `log_path` | — | If set, appends one JSON record per request. Records always also go to the `telemetry` tracing target regardless of this. |

Records are also emitted as structured `tracing` events on the `telemetry`
target, so you can route them with `RUST_LOG` (e.g. `RUST_LOG=telemetry=info`)
even without a log file.

## What gets recorded

One record per request, e.g.:

```json
{
  "endpoint": "/v1/chat/completions",
  "model": "gpt-4o-mini",
  "classified_tier": "Standard",
  "effective_tier": "Complex",
  "shadow": true,
  "stream": false,
  "input_tokens": 1843,
  "output_tokens": 512,
  "estimated_cost_usd": 0.013209,
  "latency_ms": 2241
}
```

| Field | Notes |
|-------|-------|
| `endpoint` | `/v1/chat/completions` or `/v1/messages`. |
| `model` | The model the client requested. |
| `classified_tier` | The tier the router chose from token count + intent. |
| `effective_tier` | The tier actually used. Differs from `classified_tier` only under shadow mode. |
| `shadow` | Whether shadow routing overrode the route for this request. |
| `stream` | Whether the response was streamed. |
| `input_tokens` | Actual prompt tokens for non-streaming responses; the router's estimate for streaming. |
| `output_tokens` | Completion tokens. For streams, read from the provider's final usage chunk; `null` if the provider doesn't return one. |
| `estimated_cost_usd` | `input_tokens · input_cost_per_1m + output_tokens · output_cost_per_1m`, using the **effective** tier's configured pricing. `null` when `output_tokens` is unknown. |
| `latency_ms` | Time to the upstream response (time to first byte for streams). |

Set `input_cost_per_1m` / `output_cost_per_1m` per provider in the config; the
defaults reflect gpt-4o-mini and Claude Sonnet list prices.

## The shadow-mode validation recipe

1. Set `enabled = true`, `shadow_mode = true`, `shadow_tier = "complex"`,
   `log_path = "telemetry.jsonl"`. Every request now gets the high-quality
   baseline, so users are unaffected.
2. Run on real traffic for a representative period (a week is typical).
3. Analyze `telemetry.jsonl`:

```bash
# How often would the router downroute off the Complex baseline?
jq -r 'select(.shadow) | .classified_tier' telemetry.jsonl | sort | uniq -c

# Baseline spend actually incurred while shadowing:
jq -s 'map(.estimated_cost_usd // 0) | add' telemetry.jsonl

# Rough savings upper bound: re-price each request at the tier the router WOULD
# have used (approximation — a smaller model may produce a different length).
#   tier1_free=$0, tier2_standard=$0.15/$0.60, tier3_complex=$3/$15 per 1M
jq -s '
  def price(t): if t=="Free" then {i:0,o:0}
                elif t=="Standard" then {i:0.15,o:0.60}
                else {i:3,o:15} end;
  map(price(.classified_tier) as $p
      | (.input_tokens/1e6)*$p.i + ((.output_tokens//0)/1e6)*$p.o)
  | add' telemetry.jsonl
```

Compare the baseline spend to the routed estimate. If the gap is meaningfully
positive **and** the downroute rate is high enough to matter, routing is worth
turning on (set `shadow_mode = false`). If not, the routing overhead isn't
paying off on your workload.

## Limitations

- **Streaming output tokens depend on the provider.** token-miser requests a
  final usage chunk (`stream_options.include_usage`) and reads the count from
  the streamed response; providers that ignore it leave `output_tokens` (and
  cost) `null`. For streaming `/v1/messages`, `input_tokens` is the router's
  estimate, since the upstream reveals the exact prompt count only at the end.
- **Shadow savings are an estimate, not a measurement.** While shadowing, the
  recorded tokens come from the baseline model; a cheaper model could produce a
  different output length. Treat the re-priced figure as a rough upper bound.
- File writes are best-effort: a failed append is logged on the `telemetry`
  target and never fails the request.
