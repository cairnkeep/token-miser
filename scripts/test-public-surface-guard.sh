#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
verifier="$repo_root/scripts/verify-no-private-references.sh"
fixture=$(mktemp -d)
trap 'rm -rf "$fixture"' EXIT

git -C "$fixture" init --quiet
mkdir -p "$fixture/scripts"
cp "$verifier" "$fixture/scripts/verify-no-private-references.sh"
chmod +x "$fixture/scripts/verify-no-private-references.sh"
printf '%s\n' 'safe public content' >"$fixture/README.md"
git -C "$fixture" add README.md scripts/verify-no-private-references.sh

run_check() {
  (cd "$fixture" && ./scripts/verify-no-private-references.sh >/dev/null 2>&1)
}

run_check

printf '%s\n' 'tracked but missing' >"$fixture/missing.txt"
git -C "$fixture" add missing.txt
rm "$fixture/missing.txt"
if run_check; then
  echo "Guard accepted a scan failure" >&2
  exit 1
fi
git -C "$fixture" reset --quiet -- missing.txt

private_marker="work-""laptop"
printf '%s\n' "$private_marker" >"$fixture/private.txt"
git -C "$fixture" add private.txt
if run_check; then
  echo "Guard accepted a private identifier" >&2
  exit 1
fi
git -C "$fixture" reset --quiet -- private.txt
rm "$fixture/private.txt"

credential="npm_""ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"
printf '%s\n' "$credential" >"$fixture/credential.txt"
git -C "$fixture" add credential.txt
if run_check; then
  echo "Guard accepted a credential-shaped string" >&2
  exit 1
fi

echo "Public-surface guard regression tests passed"
