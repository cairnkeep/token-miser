# token-miser — Overview

token-miser is an OpenAI- and Anthropic-compatible reverse proxy written in
Rust. It routes each request to the cheapest tier that should handle it, then
optionally escalates if the cheap answer is clearly bad.

## Request flow

Each request is classified in this order:

1. Token threshold and complexity keyword checks
2. Optional semantic routing for the ambiguous middle
3. Optional private-cluster-backed intent classification fallback
4. Optional escalation if the served answer is empty, truncated, refusal-like,
   or judge-scored too low

The three logical tiers map to provider keys:

- `tier1_free`
- `tier2_standard`
- `tier3_complex`

Each tier points to an OpenAI-compatible HTTP endpoint.

## Interfaces

- `POST /v1/chat/completions`
- `POST /v1/messages`
- `GET /v1/models`
- `GET /health`
- SSE streaming on both chat endpoints

## Enterprise/private-cluster fit

A common enterprise setup is to keep cheap tiers on a single private,
OpenAI-compatible endpoint, optionally use `[private_cluster]` for intent
classification, and reserve the complex tier for a stronger model on that
same endpoint or another OpenAI-compatible API provider. See
[architecture/MODEL-SELECTION.md](architecture/MODEL-SELECTION.md).

## Observability

Telemetry records routing decisions, tokens, cost estimates, and escalation
count. Shadow mode lets you measure routing decisions without changing the
served tier.

See:

- [operations/TELEMETRY.md](operations/TELEMETRY.md)
- [operations/MEASUREMENT-RUNBOOK.md](operations/MEASUREMENT-RUNBOOK.md)

## Status

- Routing, translation, streaming, telemetry, semantic routing, and escalation
  are implemented
- HTTP and CLI-backed providers are supported
- Current validation baseline: `cargo test` and `cargo clippy --all-targets --all-features -- -D warnings`
