# Repository Guide

Token Miser is a Rust reverse proxy for cost-aware LLM request routing.

## Commands

- Format: `cargo fmt --all -- --check`
- Test: `cargo test --locked`
- Lint: `cargo clippy --locked --all-targets --all-features -- -D warnings`
- Package: `cargo package --locked`
- Public-surface check: `./scripts/verify-no-private-references.sh`

## Conventions

- Keep public examples generic and bind to loopback by default.
- Never commit credentials, private endpoints, personal paths, telemetry, or
  machine-specific service files.
- Put local configuration under `.local/` or in ignored `config.local.toml`.
- Update tests and documentation when routing or configuration behavior changes.
