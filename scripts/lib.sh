#!/usr/bin/env bash
set -euo pipefail

RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK_TOOL="$RECIPE_ROOT/scripts/runtime_lock.py"

# Capture the process environment BEFORE sourcing .env. Explicit shell-prefix
# overrides must win over .env values.
declare -A _EXPLICIT_ENV=()
while IFS='=' read -r key value; do
  [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
  _EXPLICIT_ENV["$key"]="$value"
done < <(env)

if [[ -f "$RECIPE_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$RECIPE_ROOT/.env"
  set +a
fi

for key in "${!_EXPLICIT_ENV[@]}"; do
  printf -v "$key" '%s' "${_EXPLICIT_ENV[$key]}"
  export "$key"
done
unset _EXPLICIT_ENV key value

python3 "$LOCK_TOOL" validate >/dev/null
LOCK_MODEL_TP4="$(python3 "$LOCK_TOOL" get models.tp4.repo_id)"
LOCK_MODEL_TP2="$(python3 "$LOCK_TOOL" get models.tp2.repo_id)"
LOCK_MODEL_REVISION_TP4="$(python3 "$LOCK_TOOL" get models.tp4.revision 2>/dev/null || true)"
LOCK_MODEL_REVISION_TP2="$(python3 "$LOCK_TOOL" get models.tp2.revision 2>/dev/null || true)"

IMAGE="${IMAGE:-deepseek-v41-exl3:spark}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-exl3}"
RAY_PORT="${RAY_PORT:-6379}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v41-exl3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

MODEL_TP4="${MODEL_TP4:-$LOCK_MODEL_TP4}"
MODEL_TP2="${MODEL_TP2:-$LOCK_MODEL_TP2}"

mkdir -p "$HF_HOME"

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: $name must be set" >&2
    exit 2
  fi
}

is_true() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_model_for_tp() {
  local tp="$1"
  if [[ -n "${MODEL:-}" ]]; then
    printf '%s\n' "$MODEL"
  elif [[ "$tp" == "4" ]]; then
    printf '%s\n' "$MODEL_TP4"
  elif [[ "$tp" == "2" ]]; then
    printf '%s\n' "$MODEL_TP2"
  else
    echo "ERROR: unsupported topology TP$tp" >&2
    return 2
  fi
}

resolve_model_revision_for_tp() {
  local tp="$1"
  if [[ -n "${MODEL_REVISION:-}" ]]; then
    printf '%s\n' "$MODEL_REVISION"
  elif [[ -n "${MODEL:-}" ]]; then
    # User supplied a different model without an explicit revision. Do not
    # silently apply the lock's revision for the recipe's canonical model.
    printf '%s\n' ""
  elif [[ "$tp" == "4" ]]; then
    printf '%s\n' "$LOCK_MODEL_REVISION_TP4"
  elif [[ "$tp" == "2" ]]; then
    printf '%s\n' "$LOCK_MODEL_REVISION_TP2"
  else
    echo "ERROR: unsupported topology TP$tp" >&2
    return 2
  fi
}
