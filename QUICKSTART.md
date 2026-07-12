# Quick Start Guide

## 1. Prepare a config

Use the generic example as a starting point and keep your edits in a private copy.

```bash
cp config.example.toml config.local.toml
export TOKEN_MISER_CONFIG=config.local.toml
```

## 2. Prepare environment variables

```bash
cp .env.example .env
```

Then fill in only what your chosen providers need. Config values support
`${VAR}` expansion.

Common variables:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `ENTERPRISE_API_KEY` (only if using `config.enterprise.example.toml`)
- `RUST_LOG`

## 3. Run the proxy

```bash
cargo run --release
```

Default address: `http://127.0.0.1:8080`

## 4. Point a client at it

OpenAI-style clients:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8080/v1 opencode
```

Anthropic-style clients:

```bash
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 claude
```

## 5. Smoke-test the endpoints

```bash
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "messages": [{"role": "user", "content": "hello"}]
  }'

curl -X POST http://127.0.0.1:8080/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test-model",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "hello"}]
  }'
```

## 6. Watch routing decisions

If telemetry is enabled in your config, inspect the JSONL log configured under
`[telemetry].log_path`.

For interactive debugging:

```bash
RUST_LOG=debug cargo run
```

## Shared vs local files

- Shared examples stay in the repo root
- Personal notes, host-specific configs, and service files belong under `.local/`
- Repo bootstrap wrappers and overrides, if you use them, belong in your local `.ai/` layer such as `.ai/.env.local` or `.ai/env.local`

## systemd

Use your own local service file under `.local/systemd/` or your normal systemd
user directory. The shared repo no longer tracks a canonical unit file because
the useful values are usually machine-specific.

## Validation

```bash
cargo fmt --all -- --check
cargo test --locked
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo package --locked
./scripts/verify-no-private-references.sh
```

## Network exposure

Token Miser does not authenticate inbound requests. Keep the default loopback
binding unless an authenticated reverse proxy or equivalent trusted network
boundary protects the service.
