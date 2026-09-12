#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

TP="${1:-}"
if [[ "$TP" != "2" && "$TP" != "4" ]]; then
  echo "Usage: $0 2|4" >&2
  exit 2
fi

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "ERROR: head container '$CONTAINER_NAME' is not running." >&2
  exit 2
fi

exec docker exec -i "$CONTAINER_NAME" \
  python /recipe/scripts/cluster_collective.py \
  --tp "$TP" \
  --megabytes "${COLLECTIVE_MEGABYTES:-16}" \
  --warmup "${COLLECTIVE_WARMUP:-2}" \
  --iterations "${COLLECTIVE_ITERATIONS:-5}" \
  --master-port "${COLLECTIVE_MASTER_PORT:-29557}" \
  ${COLLECTIVE_JSON:+--json}
