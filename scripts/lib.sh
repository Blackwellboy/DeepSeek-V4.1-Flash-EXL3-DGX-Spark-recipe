#!/usr/bin/env bash
set -euo pipefail

RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Capture the process environment BEFORE sourcing .env. Explicit shell-prefix
# overrides such as MAX_MODEL_LEN=8192 ./scripts/serve_tp4.sh must win over
# values stored in .env. The old order sourced .env last and silently replaced
# those overrides, which could turn an intended 8K/seq1 qualification run into
# the 65K/seq4 defaults.
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

# Restore everything that was explicitly present in the caller environment.
# This gives us conventional precedence: command environment > .env > defaults.
for key in "${!_EXPLICIT_ENV[@]}"; do
  printf -v "$key" '%s' "${_EXPLICIT_ENV[$key]}"
  export "$key"
done
unset _EXPLICIT_ENV key value

IMAGE="${IMAGE:-deepseek-v41-exl3:spark}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-exl3}"
RAY_PORT="${RAY_PORT:-6379}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v41-exl3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

MODEL_TP4="${MODEL_TP4:-vcruz305/DSV4.1-Flash-EXL3-4.75bpw}"
MODEL_TP2="${MODEL_TP2:-vcruz305/DSV4.1-Flash-SAGE-EXL3-3.30bpw}"

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
