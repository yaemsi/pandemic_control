#!/usr/bin/env bash
set -euo pipefail

if command -v pre-commit >/dev/null 2>&1; then
  PRE_COMMIT_BIN="pre-commit"
elif [ -x ".venv/bin/pre-commit" ]; then
  PRE_COMMIT_BIN="./.venv/bin/pre-commit"
else
  echo "pre-commit not found."
  echo "Either activate the venv (source .venv/bin/activate) or run: .venv/bin/pre-commit install ..."
  exit 1
fi

"$PRE_COMMIT_BIN" install --hook-type pre-commit --install-hooks
"$PRE_COMMIT_BIN" install --hook-type pre-push --install-hooks

echo "Installed pre-commit and pre-push hooks."

