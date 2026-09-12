#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  echo "Usage: $0 /absolute/path/to/tp2-snapshot" >&2
  exit 2
fi

# Preserve the legacy TP2 environment names as aliases.
export DOWNLOAD_MIN_FREE_GIB="${DOWNLOAD_MIN_FREE_GIB:-${TP2_DOWNLOAD_MIN_FREE_GIB:-550}}"
export MODEL_HF_HOME="${MODEL_HF_HOME:-${TP2_HF_HOME:-}}"
export PACK_RESERVE_GIB="${PACK_RESERVE_GIB:-${TP2_POST_DOWNLOAD_RESERVE_GIB:-32}}"

exec "$SCRIPT_DIR/materialize_model.sh" 2 "$DEST"
