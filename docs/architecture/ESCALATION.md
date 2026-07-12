# Quality-aware escalation

Routing on token count and keywords is a guess at difficulty — it can send a
substantive task to a cheap tier whose answer isn't good enough. Escalation makes
that guess **safe to be wrong**: when the routed (cheaper) tier returns a bad or
failed response, the proxy automatically retries on the next tier up instead of
handing the client a degenerate answer.

This directly addresses the misroute-risk zone surfaced by shadow-mode analysis
(short, substantive coding requests routed to Free on length alone).

## Enabling

```toml
[escalation]
enabled = false           # master switch
max_escalations = 1       # how many tiers to climb (Free -> Standard -> Complex)
on_empty_response = true  # escalate on an empty/whitespace response body
on_truncation = false     # escalate when finish_reason == "length" (cut off)
on_refusal = false        # escalate on a refusal / "I can't help" response

[escalation.judge]        # optional: catch plausible-but-wrong answers
enabled = false
tier = "standard"         # tier whose model scores the response
min_score = 3             # escalate when the 1-5 score is below this
```

When `enabled = false` (the default), routing is single-shot — no behavior
change.

## What triggers an escalation

The tier ladder is `Free → Standard → Complex`; escalation climbs one step at a
time, capped at `Complex`, up to `max_escalations` times.

- **Transient/overload errors** — a request failure (network/timeout), a `5xx`,
  or a `429` from the routed tier. Deterministic errors (`4xx` other than `429`,
  auth failures, provider-not-found) do **not** escalate, since a higher tier
  won't fix them.
- **Degenerate responses** — a non-streaming response that is empty/whitespace
  (`on_empty_response`), filtered (`finish_reason == content_filter`, always),
  truncated (`on_truncation`, `finish_reason == length`), or a short refusal
  (`on_refusal`). Refusal detection is conservative: it only flags short
  responses dominated by a known "I can't help"-style phrase, so long answers
  that merely mention one are not escalated.
- **Judge verdict** (when `[escalation.judge].enabled`) — for responses that pass
  the heuristics, a judge model scores 1-5 how well the response answers the
  request; a score below `min_score` escalates. This catches *plausible-but-wrong*
  answers that no heuristic can. The judge runs on the configured `tier`, adds one
  model call per checked response, and **fails open** (no escalation) if the judge
  tier is misconfigured or unreachable.

Streaming responses escalate only on the **initial** connection error (a non-2xx
before any bytes flow); their content can't be inspected without buffering, so
content-based escalation does not apply once a stream has started.

## Observability

Each telemetry record carries:

- `served_tier` — the tier that actually produced the response (may be higher
  than `effective_tier` if it escalated).
- `escalations` — how many times routing climbed for this request.

```bash
# How often did escalation kick in, and from where to where?
jq -r 'select(.escalations > 0) | "\(.classified_tier) -> \(.served_tier)"' telemetry.jsonl \
  | sort | uniq -c
```

## Cost note

An escalated request also paid for the discarded cheaper attempt(s).
`estimated_cost_usd` currently reflects the **final** (served) tier only, so the
true cost of an escalated request is slightly higher than recorded. Use the
`escalations` count to gauge how often this happens; summing per-attempt cost is
a future refinement.
