#!/usr/bin/env bash
# Summarize the runtime/kernel paths actually advertised or observed by the server.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
source "$SCRIPT_DIR/lib.sh"

if ! docker ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
  echo "ERROR: container '$CONTAINER_NAME' is not running." >&2
  exit 2
fi

cat <<'EOF'
=== vllm-exl3 runtime diagnostics ===
EOF
docker exec "$CONTAINER_NAME" python - <<'PY'
import json
import vllm_exl3
vllm_exl3.register()
print(json.dumps(vllm_exl3.runtime_diagnostics(), indent=2, sort_keys=True))
PY

cat <<'EOF'

=== observed server kernel/dispatch evidence ===
EOF
# This is intentionally evidence collection, not a parser that invents a PASS.
# Keep the raw matching lines in qualification receipts so changes in upstream
# vLLM logging do not silently change our interpretation.
docker logs "$CONTAINER_NAME" 2>&1 | grep -Ei \
  'EXL3|mixed[_ -]?K|fused_moe|python_loop|DeepGEMM|MXFP8|FlashInfer|FlashMLA|Engram.*DISK|disk-backed|kernel|trellis arena' \
  | tail -n "${KERNEL_RECEIPT_LINES:-400}" || true
