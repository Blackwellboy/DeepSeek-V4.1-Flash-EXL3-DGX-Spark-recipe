#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Stopped local recipe container '$CONTAINER_NAME'."
