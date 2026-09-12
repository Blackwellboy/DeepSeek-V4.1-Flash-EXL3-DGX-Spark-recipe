#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

URL="${API_URL:-http://127.0.0.1:${PORT}}"
EXPECTED="${SMOKE_EXPECTED:-EXL3 Spark OK}"

echo "Checking model list at $URL ..."
MODELS_JSON="$(curl -fsS "$URL/v1/models")"
python3 - "$SERVED_MODEL_NAME" "$MODELS_JSON" <<'PY'
import json, sys
name, raw = sys.argv[1:]
data = json.loads(raw)
ids = {str(item.get('id')) for item in data.get('data', []) if isinstance(item, dict)}
if name not in ids:
    raise SystemExit(f"SMOKE FAIL: served model {name!r} not present in /v1/models: {sorted(ids)}")
print(json.dumps(data, indent=2))
PY

echo
echo "Sending deterministic chat completion ..."
REQUEST_JSON="$(python3 - "$SERVED_MODEL_NAME" <<'PY'
import json, sys
print(json.dumps({
  "model": sys.argv[1],
  "messages": [{"role": "user", "content": "Reply with exactly: EXL3 Spark OK"}],
  "temperature": 0,
  "max_tokens": 64,
  "chat_template_kwargs": {"thinking": False}
}))
PY
)"
RESPONSE_JSON="$(curl -fsS "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQUEST_JSON")"

python3 - "$EXPECTED" "$RESPONSE_JSON" <<'PY'
import json, sys
expected, raw = sys.argv[1:]
data = json.loads(raw)
if data.get("error"):
    raise SystemExit(f"SMOKE FAIL: API returned error: {data['error']}")
try:
    content = data["choices"][0]["message"]["content"]
except (KeyError, IndexError, TypeError) as exc:
    raise SystemExit(f"SMOKE FAIL: malformed chat completion: {exc}: {data}")
if not isinstance(content, str) or content.strip() != expected:
    raise SystemExit(
        f"SMOKE FAIL: deterministic sentinel mismatch: expected {expected!r}, got {content!r}"
    )
print(json.dumps(data, indent=2))
print(f"SMOKE PASS: {content.strip()}")
PY
