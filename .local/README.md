# Local Files

Use `.local/` for machine-specific notes, configs, and service files that should
stay in your working copy but not in the shared repo.

Current preserved files:

- `.local/config/config.homelab.toml`
- `.local/config/config.hybrid-claude.toml`
- `.local/config/config.hybrid-codex.toml`
- `.local/config/config.hybrid-copilot.toml`
- `.local/config/config.shadow.toml`
- `.local/config/config.live.toml`
- `.local/bin/run-hybrid.sh`
- `.local/systemd/token-miser.service`

Optional private variants can also live here, for example a Copilot HTTP config
used during validation work.

Shared examples stay in the repo root. Personal paths, local endpoints, cookie
files, and host-specific setup belong here.
