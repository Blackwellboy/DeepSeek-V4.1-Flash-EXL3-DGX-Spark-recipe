#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

TP="${1:-}"
if [[ "$TP" != "2" && "$TP" != "4" ]]; then
  echo "Usage: $0 2|4 [--json]" >&2
  exit 2
fi
shift || true

exec python3 "$RECIPE_ROOT/scripts/doctor.py" \
  --tp "$TP" \
  --model-dir "${MODEL_DIR:-}" \
  --model "${MODEL:-}" \
  --reserve-gib "${PACK_RESERVE_GIB:-32}" \
  "$@"
