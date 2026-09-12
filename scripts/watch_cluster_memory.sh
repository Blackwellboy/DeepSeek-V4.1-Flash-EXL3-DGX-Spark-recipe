#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "ERROR: head container '$CONTAINER_NAME' is not running on this host." >&2
  exit 2
fi

INTERVAL="${INTERVAL:-2}"
exec docker exec -i "$CONTAINER_NAME" \
  python /recipe/scripts/watch_ray_memory.py --interval "$INTERVAL" "$@"
