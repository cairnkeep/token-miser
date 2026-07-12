# Token Miser

token-miser is a Rust reverse proxy for LLM traffic. It accepts OpenAI- and
Anthropic-style chat requests, classifies each request, and forwards it to the
cheapest tier that should handle it.

It is a good fit for hybrid setups where cheap traffic should stay on a local
model or a private/enterprise endpoint, while expensive traffic goes to a
stronger provider only when needed.

## What it does

- Serves `POST /v1/chat/completions` and `POST /v1/messages`
- Routes requests across three tiers: `tier1_free`, `tier2_standard`, `tier3_complex`
- Classifies by token thresholds, complexity keywords, optional semantic routing,
  and optional private-cluster-backed intent classification
- Forwards to any OpenAI-compatible HTTP backend
- Logs routing decisions, token estimates, and cost estimates to JSONL
- Supports shadow mode and escalation to validate routing safely
- Optional upstream FastContext stage: gathers small, capped repo context before
  routing to lower the premium-tier (Opus) escalation rate
  (see [docs/architecture/FASTCONTEXT-EXPLORE.md](docs/architecture/FASTCONTEXT-EXPLORE.md))

## Current status

- Rust proxy, Anthropic/OpenAI translation, streaming, semantic routing,
  escalation, telemetry, and the optional upstream FastContext explore stage
  are implemented
- `cargo test` passes
- `cargo clippy --all-targets --all-features -- -D warnings` passes

Token Miser is pre-1.0 software. Configuration and routing behavior may change
between minor releases.

## Quick start

```bash
cp .env.example .env
cp config.example.toml config.local.toml
export TOKEN_MISER_CONFIG=config.local.toml

cargo run --release
```

Then point a client at the proxy:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 opencode
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

For a fuller setup guide, see [QUICKSTART.md](QUICKSTART.md).

## Config layout

- `config.example.toml`: generic local starting point (local model + paid cloud tier)
- `config.enterprise.example.toml`: minimal single-endpoint example for routing
  all tiers to one private/enterprise OpenAI-compatible endpoint
- `.local/`: your machine-specific notes, configs, and service files (gitignored
  except this note)

Keep personal paths, local endpoints, and operator notes under `.local/`, not in
the shared repo surface.

## Documentation

- [docs/OVERVIEW.md](docs/OVERVIEW.md): one-page technical overview
- [docs/README.md](docs/README.md): doc index
- [docs/architecture/ROUTING-ARCHITECTURE.md](docs/architecture/ROUTING-ARCHITECTURE.md): routing flow
- [docs/operations/TELEMETRY.md](docs/operations/TELEMETRY.md): routing and cost logging

## Security

Token Miser has no inbound authentication layer. It binds to `127.0.0.1` by
default and should remain behind an authenticated gateway or a trusted network
boundary. Do not expose it directly to an untrusted network, because callers
can consume the provider credentials configured for the proxy.

See [SECURITY.md](SECURITY.md) for supported versions and private reporting.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow. This
project is licensed under the [Apache License 2.0](LICENSE).
