# Measurement Runbook — does token-miser actually save money?

Use this runbook to validate token-miser on real traffic with a real embedding
model, real provider pricing, and the actual escalation path you plan to use.

The goal is to answer two questions:

1. Would the router choose cheaper tiers often enough to matter?
2. When live routing is enabled, does the saving survive quality recovery
   overhead such as escalation?

---

## What already exists

All of this is implemented, tested, and on `main`:

- **OpenAI-compatible proxy** (`/v1/chat/completions`) and Anthropic-compatible
  (`/v1/messages`) that routes each request to a tier (Free / Standard / Complex).
- **Telemetry + shadow mode** — records routing decisions and cost per request to
  a JSONL file. See `docs/operations/TELEMETRY.md`.
- **Semantic routing** — routes the ambiguous middle by embedding similarity.
  See `docs/architecture/SEMANTIC-ROUTING.md`.
- **Escalation + LLM judge** — recovers misroutes automatically. See
  `docs/architecture/ESCALATION.md`.

Config reference: `config.toml` (auto-loaded from the working dir, or set
`TOKEN_MISER_CONFIG=/path`). `${VAR}` in the config expands from the environment.

---

## Hard requirement: OpenAI-compatible providers

The proxy POSTs **OpenAI-format** requests to `{endpoint}/chat/completions` for
every provider, and `{endpoint}/embeddings` for the semantic router. So every
tier — and the embedding endpoint and the judge — must speak the OpenAI API.

- ✅ OpenAI, Azure OpenAI, OpenRouter, Together, Groq, vLLM, **Ollama**
  (`http://localhost:11434/v1`), most private clusters.
- ❌ Anthropic's **native** API (`api.anthropic.com`) is *not* OpenAI-format at
  `/chat/completions` — don't point a tier's `endpoint` directly at it. Use an OpenAI-compatible route to a
  strong model instead (e.g. `gpt-4o`/`gpt-4.1`, or Claude **via OpenRouter**).

---

## Phase plan

Two phases. Phase A is risk-free and measures *routing decisions*. Phase B runs
live with the safety net and measures *real net savings + recovery*.

| | Phase A — Shadow | Phase B — Live + safety net |
|---|---|---|
| Routing served | always `shadow_tier` (safe baseline) | the router's choice |
| Measures | what semantic routing *would* pick | actual cost, escalation/recovery rate |
| Quality risk | none (users always get the baseline) | low (escalation+judge recover misroutes) |
| Escalation/judge | **inert** (baseline is top tier) | active |

Run A first. If its projected savings and routing distribution look good, run B.

---

## Step 0 — Setup

```bash
git clone <this repo> token-miser && cd token-miser
cargo build --release        # installs nothing else; needs a Rust toolchain
export OPENAI_API_KEY=sk-...  # and any other provider keys you'll use
```

Decide your three tiers, embedding model, and judge model before you start. A
simple all-OpenAI starting point is:

- `tier1_free`: local Ollama `llama3.2` (free) — or omit Ollama and use
  `gpt-4o-mini` here too.
- `tier2_standard`: `gpt-4o-mini`.
- `tier3_complex`: `gpt-4o` (or `gpt-4.1`).
- embedding: `text-embedding-3-small` (OpenAI).
- judge: the `standard` tier (`gpt-4o-mini`) — cheap and adequate.

Put **real per-1M prices** in `input_cost_per_1m` / `output_cost_per_1m` for each
tier — the cost numbers are only as accurate as these.

---

## Step 1 — Phase A config (shadow)

Create a private local config such as `.local/config/config.shadow.toml`.
Adjust endpoints, models, and prices to your choices.

```toml
[server]
host = "127.0.0.1"
port = 8080

[routing]
tier1_threshold = 2000
tier2_threshold = 32000
complexity_keywords = ["architect", "refactor", "system design", "redesign", "migrate"]

[providers.tier1_free]
endpoint = "http://localhost:11434/v1"   # Ollama; or a cheap OpenAI-compatible endpoint
auth_type = "None"
input_cost_per_1m = 0.0
output_cost_per_1m = 0.0
[providers.tier1_free.model_mapping]
default = "llama3.2"

[providers.tier2_standard]
endpoint = "https://api.openai.com/v1"
auth_type = "ApiKey"
api_key = "${OPENAI_API_KEY}"
input_cost_per_1m = 0.15
output_cost_per_1m = 0.60
[providers.tier2_standard.model_mapping]
default = "gpt-4o-mini"

[providers.tier3_complex]
endpoint = "https://api.openai.com/v1"
auth_type = "ApiKey"
api_key = "${OPENAI_API_KEY}"
input_cost_per_1m = 2.50
output_cost_per_1m = 10.00
[providers.tier3_complex.model_mapping]
default = "gpt-4o"

[telemetry]
enabled = true
shadow_mode = true
shadow_tier = "complex"            # everyone is served by tier3 → full quality, no risk
log_path = "telemetry-shadow.jsonl"

[semantic_router]
enabled = true
endpoint = "https://api.openai.com/v1"
model = "text-embedding-3-small"
api_key = "${OPENAI_API_KEY}"

[escalation]
enabled = false                    # inert under shadow; leave off in Phase A
```

> **Cost note:** in Phase A every request is served by `shadow_tier` (Complex),
> so you pay the **baseline** price for everything plus one cheap embedding per
> request. That's the price of risk-free measurement. To halve it, set
> `shadow_tier = "standard"` (cheaper baseline, slightly more risk that a genuinely
> hard task gets a mid model).

---

## Step 2 — Run Phase A and generate real traffic

```bash
TOKEN_MISER_CONFIG=.local/config/config.shadow.toml RUST_LOG=info,telemetry=info ./target/release/token_miser
```

Point a real client at it so your normal work *is* the traffic:

- **OpenAI-style tools** (Aider, Continue, Cursor, scripts): set the base URL to
  `http://127.0.0.1:8080/v1` and any API key value.
- **Anthropic-style tools**: set the base URL to `http://127.0.0.1:8080` (the
  proxy serves `/v1/messages` too).

Then work normally for a representative period (a day or a week — the more
varied the traffic, the more meaningful the numbers). Every request appends a
line to `telemetry-shadow.jsonl`.

---

## Step 3 — Analyze Phase A

Use `bench/analyze.py` or an equivalent local script to inspect the telemetry.
If you write your own analyzer, make sure `PRICE` matches your config.

```python
import json, sys
from collections import Counter

PRICE = {  # USD per 1M tokens (input, output) — MATCH YOUR CONFIG
    "Free": (0.0, 0.0), "Standard": (0.15, 0.60), "Complex": (2.50, 10.00),
}
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = len(recs); print(f"requests: {n}")

dist = Counter(r["classified_tier"] for r in recs)
print("\nrouter would choose:")
for t in ("Free", "Standard", "Complex"):
    c = dist.get(t, 0); print(f"  {t:9} {c:4}  {100*c/n:5.1f}%")

def cost(tier, i, o):
    pin, pout = PRICE[tier]; return i/1e6*pin + o/1e6*pout

baseline = sum(r.get("estimated_cost_usd") or 0 for r in recs)  # served at shadow_tier
routed = sum(cost(r["classified_tier"], r.get("input_tokens",0), r.get("output_tokens") or 0) for r in recs)
print(f"\nbaseline spend (served):        ${baseline:.4f}")
print(f"projected spend (router choice): ${routed:.4f}")
if baseline:
    print(f"projected saving:                ${baseline-routed:.4f}  ({100*(baseline-routed)/baseline:.1f}%)")
print("\n(projected = upper bound: assumes the cheaper tier's answers are acceptable)")
```

Also useful (`docs/operations/TELEMETRY.md` has more):
```bash
jq -r .classified_tier telemetry-shadow.jsonl | sort | uniq -c   # distribution
```

Spot-check classifier quality by hand: grep a sample where
`classified_tier == "Free"` and read the prompts — would a cheap/local model
actually have nailed them? That manual check is your ground truth that the
projected saving isn't fantasy.

---

## Step 4 — Phase B config (live + safety net)

Only proceed if Phase A looks favorable. Create a private local config such as
`.local/config/config.live.toml` — same as shadow but with these changes:

```toml
[telemetry]
enabled = true
shadow_mode = false                # route to the router's actual choice now
log_path = "telemetry-live.jsonl"

# semantic_router: same as Phase A (keep it enabled)

[escalation]
enabled = true
max_escalations = 2
on_empty_response = true
on_truncation = true
on_refusal = true

[escalation.judge]
enabled = true
tier = "standard"                  # gpt-4o-mini judges the cheap answer
min_score = 3
```

Run it the same way (`TOKEN_MISER_CONFIG=.local/config/config.live.toml ...`) and generate the
same kind of real traffic.

---

## Step 5 — Analyze Phase B (the real result)

```python
import json, sys
from collections import Counter
PRICE = {"Free": (0.0, 0.0), "Standard": (0.15, 0.60), "Complex": (2.50, 10.00)}  # MATCH CONFIG
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
n = len(recs)
def cost(t, i, o): pin, pout = PRICE[t]; return i/1e6*pin + o/1e6*pout

served = sum(r.get("estimated_cost_usd") or 0 for r in recs)            # what you actually paid (final tier)
all_complex = sum(cost("Complex", r.get("input_tokens",0), r.get("output_tokens") or 0) for r in recs)
esc = sum(1 for r in recs if r.get("escalations", 0) > 0)
print(f"requests: {n}")
print("served tier:", Counter(r["served_tier"] for r in recs))
print(f"escalated: {esc}/{n}  ({100*esc/n:.1f}%)   (how often the cheap tier wasn't good enough)")
print(f"\nactual spend (served):      ${served:.4f}")
print(f"all-Complex baseline:       ${all_complex:.4f}")
if all_complex:
    print(f"REAL net saving:            ${all_complex-served:.4f}  ({100*(all_complex-served)/all_complex:.1f}%)")
print("\nNote: served cost omits discarded escalation attempts, so true cost is")
print("slightly higher; use the escalation rate to gauge how often that happens.")
```

This is the bottom line: actual spend with the safety net vs. paying
for the top tier on everything. The **escalation rate** is your built-in quality
signal — it's how often the router guessed too cheap and the judge/heuristics
caught it.

---

## How to read the result / decision

- **High projected saving (A) + low escalation rate (B)** → routing pays off; the
  cheap tiers are good enough often enough. Consider running live for real.
- **High escalation rate (B)** → the router downroutes too aggressively for your
  traffic; you're paying for cheap attempts *and* escalations. Tune the semantic
  exemplars (`src/semantic.rs`) or raise the thresholds, or accept a smaller-but-
  safer routing spread.
- **Saving ≈ 0** → your traffic is mostly genuinely hard (long context / complex);
  routing can't help much. That's a real finding — stop here.

---

## Caveats

1. **No human-acceptance signal.** The proxy can't see whether *you* accepted an
   answer or silently retried. In Phase B the **judge verdict** (and escalation
   rate) is the proxy for quality; a true acceptance signal needs client-side
   instrumentation and is out of scope.
2. **Projected saving (A) is an upper bound** — it assumes cheap-tier answers are
   acceptable. The manual spot-check and Phase B's escalation rate are what keep
   it honest.
3. **Cost under-reports escalations.** `estimated_cost_usd` is the final served
   tier only; an escalated request also paid for the discarded attempt(s).
4. **Semantic quality = embedding quality.** A weak embedder routes poorly; use a
   real model (the runbook does). Exemplars are built-in and tuned for coding
   traffic — adjust in `src/semantic.rs` for another domain.
5. **First request after boot** embeds ~17 exemplars (cached after). Send one
   throwaway request to warm it before timing latency.

---

## Short checklist

1. Confirm provider, model, and pricing choices.
2. Build a private shadow config and collect representative traffic.
3. Analyze projected routing and manually spot-check free-tier prompts.
4. If Phase A looks good, build a private live config and collect equivalent traffic.
5. Compare actual spend, escalation rate, and practical quality.
