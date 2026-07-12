#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

tracked_files=()
while IFS= read -r -d '' file; do
  tracked_files+=("$file")
done < <(git ls-files -z)

if ((${#tracked_files[@]} == 0)); then
  echo "No tracked files found" >&2
  exit 1
fi

# Keep organization-specific identifiers out of the public repository. Split
# literals prevent this guard from triggering on its own source.
private_markers=(
  "sie""mens"
  "code.""sie""mens"
  "PARA/""Sie""mens"
  "cairnkeep-""sie""mens"
  "work-""laptop"
)

for marker in "${private_markers[@]}"; do
  if rg --fixed-strings --ignore-case --line-number -- "$marker" "${tracked_files[@]}"; then
    echo "Private organization reference found: $marker" >&2
    exit 1
  fi
done

if rg --line-number --glob '!scripts/verify-no-private-references.sh' \
  '(AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9_]{20,}|glpat-[A-Za-z0-9_-]{20,}|npm_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)' \
  "${tracked_files[@]}"; then
  echo "Possible credential found in tracked files" >&2
  exit 1
fi

for path in .env config.toml config.local.toml token-miser.service; do
  if git ls-files --error-unmatch "$path" >/dev/null 2>&1; then
    echo "Local-only file is tracked: $path" >&2
    exit 1
  fi
done

echo "Public-surface check passed"
