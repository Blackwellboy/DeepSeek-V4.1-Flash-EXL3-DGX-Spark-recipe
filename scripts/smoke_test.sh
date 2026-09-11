#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

URL="${API_URL:-http://127.0.0.1:${PORT}}"

echo "Checking model list at $URL ..."
curl -fsS "$URL/v1/models" | python -m json.tool

echo
echo "Sending chat completion ..."
curl -fsS "$URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python - <<PY
import json
print(json.dumps({
  "model": "${SERVED_MODEL_NAME}",
  "messages": [{"role": "user", "content": "Reply with exactly: EXL3 Spark OK"}],
  "temperature": 0,
  "max_tokens": 64,
  "chat_template_kwargs": {"thinking": False}
}))
PY
)" | python -m json.tool
