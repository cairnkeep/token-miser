# Contributing

Contributions are welcome through GitHub issues and pull requests.

## Development

Install a Rust toolchain compatible with the `rust-version` in `Cargo.toml`,
then run:

```bash
cargo fmt --all -- --check
cargo test --locked
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo package --locked
./scripts/verify-no-private-references.sh
```

Keep each change focused and include tests for behavior changes. Never commit
credentials, private endpoints, local paths, telemetry, or machine-specific
configuration.

## Reporting security issues

Do not open a public issue for a suspected vulnerability. Follow
[SECURITY.md](SECURITY.md) instead.
