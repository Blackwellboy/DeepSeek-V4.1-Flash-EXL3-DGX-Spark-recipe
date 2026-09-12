#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

DEST="${1:-}"
if [[ -z "$DEST" ]]; then
  echo "Usage: $0 /absolute/path/to/tp2-snapshot" >&2
  echo "Use a filesystem with enough space; do not materialize onto a nearly-full Spark root disk." >&2
  exit 2
fi

case "$DEST" in
  /*) ;;
  *) echo "ERROR: destination must be an absolute path" >&2; exit 2 ;;
esac

REPO="${MODEL_TP2:-vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw}"
MIN_FREE_GIB="${TP2_DOWNLOAD_MIN_FREE_GIB:-500}"
mkdir -p "$DEST"

FREE_GIB="$(df -PB1 "$DEST" | awk 'NR==2 {printf "%.2f", $4/1024/1024/1024}')"
python3 - "$FREE_GIB" "$MIN_FREE_GIB" <<'PY'
import sys
free = float(sys.argv[1]); need = float(sys.argv[2])
if free < need:
    raise SystemExit(f"ERROR: destination has {free:.1f} GiB free; conservative TP2 full-snapshot gate requires {need:.1f} GiB. Set TP2_DOWNLOAD_MIN_FREE_GIB only if you know the exact snapshot/staging budget.")
PY

echo "=== TP2 Hugging Face materialization ==="
echo "Repo:        $REPO"
echo "Destination: $DEST"
echo "Free space:  $FREE_GIB GiB"
echo

# Do not use git clone as the model-weight materializer. Large HF objects may be
# represented by LFS/Xet pointer stubs when the matching transport is absent.
if command -v hf >/dev/null 2>&1; then
  hf download "$REPO" --local-dir "$DEST"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$REPO" --local-dir "$DEST"
else
  python3 - "$REPO" "$DEST" <<'PY'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit("ERROR: install huggingface_hub or use the recipe runtime image") from exc
snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
fi

echo
echo "Validating materialized snapshot..."
python3 "$RECIPE_ROOT/scripts/check_tp2_pack.py" "$DEST" --reserve-gib "${TP2_POST_DOWNLOAD_RESERVE_GIB:-32}"

echo
echo "TP2 snapshot is materially present at: $DEST"
echo "Set on BOTH Sparks:"
echo "  MODEL_DIR=$(dirname "$DEST")"
echo "  MODEL=/models/$(basename "$DEST")"
