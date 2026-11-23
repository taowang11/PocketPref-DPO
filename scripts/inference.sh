#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${PYTHON_BIN:-python}" "$repo_root/generate.py" \
  --checkpoint "${CHECKPOINT:-$repo_root/ckpts/dpo/pocketpref-dpo.pt}" "$@"
