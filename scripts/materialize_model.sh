#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

TP="${1:-}"
DEST="${2:-}"
if [[ "$TP" != "2" && "$TP" != "4" ]] || [[ -z "$DEST" ]]; then
  echo "Usage: $0 2|4 /absolute/path/to/model-snapshot" >&2
  exit 2
fi
case "$DEST" in
  /*) ;;
  *) echo "ERROR: destination must be an absolute path" >&2; exit 2 ;;
esac

REPO="$(resolve_model_for_tp "$TP")"
REVISION="$(resolve_model_revision_for_tp "$TP")"
TOPOLOGY="tp$TP"
MIN_FREE_GIB="${DOWNLOAD_MIN_FREE_GIB:-550}"
POST_RESERVE_GIB="${PACK_RESERVE_GIB:-32}"
mkdir -p "$DEST"

# Keep all Hugging Face/Xet staging on the same large filesystem as the model.
MODEL_HF_HOME="${MODEL_HF_HOME:-$(dirname "$DEST")/.hf-dsv41-cache}"
export HF_HOME="$MODEL_HF_HOME"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_XET_CACHE"

DEST_FS="$(df -P "$DEST" | awk 'NR==2 {print $1}')"
CACHE_FS="$(df -P "$HF_HOME" | awk 'NR==2 {print $1}')"
if [[ "$DEST_FS" != "$CACHE_FS" ]]; then
  echo "ERROR: HF/Xet cache must be on the same large filesystem as the model." >&2
  echo "Destination: $DEST ($DEST_FS)" >&2
  echo "HF_HOME:     $HF_HOME ($CACHE_FS)" >&2
  exit 2
fi

FREE_GIB="$(df -PB1 "$DEST" | awk 'NR==2 {printf "%.2f", $4/1024/1024/1024}')"
python3 - "$FREE_GIB" "$MIN_FREE_GIB" <<'PY'
import sys
free = float(sys.argv[1]); need = float(sys.argv[2])
if free < need:
    raise SystemExit(
        f"ERROR: destination has {free:.1f} GiB free; conservative download/staging gate requires {need:.1f} GiB. "
        "Override DOWNLOAD_MIN_FREE_GIB only after measuring the exact filesystem budget."
    )
PY

echo "=== DeepSeek V4.1 EXL3 materialization ==="
echo "Topology:    $TOPOLOGY"
echo "Repo:        $REPO"
echo "Revision:    ${REVISION:-<unlocked/quarantined>}"
echo "Destination: $DEST"
echo "HF_HOME:     $HF_HOME"
echo "Free space:  $FREE_GIB GiB"
echo

DOWNLOAD_ARGS=("$REPO" --local-dir "$DEST")
if [[ -n "$REVISION" ]]; then
  DOWNLOAD_ARGS+=( --revision "$REVISION" )
fi

if command -v hf >/dev/null 2>&1; then
  hf download "${DOWNLOAD_ARGS[@]}"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "${DOWNLOAD_ARGS[@]}"
else
  python3 - "$REPO" "$REVISION" "$DEST" <<'PY'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError as exc:
    raise SystemExit("ERROR: install huggingface_hub or use the recipe runtime image") from exc
repo, revision, dest = sys.argv[1:]
snapshot_download(repo_id=repo, revision=revision or None, local_dir=dest)
PY
fi

echo
echo "Validating physical checkpoint contract..."
python3 "$RECIPE_ROOT/scripts/validate_pack.py" \
  "$DEST" --topology "$TOPOLOGY" --reserve-gib "$POST_RESERVE_GIB"

echo
echo "Materialized and validated: $DEST"
echo "Set on EVERY Spark:"
echo "  MODEL_DIR=$(dirname "$DEST")"
echo "  MODEL=/models/$(basename "$DEST")"
echo "  HF_HOME=$HF_HOME"
