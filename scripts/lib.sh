#!/usr/bin/env bash
set -euo pipefail

RECIPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$RECIPE_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$RECIPE_ROOT/.env"
  set +a
fi

IMAGE="${IMAGE:-deepseek-v41-exl3:spark}"
CONTAINER_NAME="${CONTAINER_NAME:-dsv41-exl3}"
RAY_PORT="${RAY_PORT:-6379}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-deepseek-v41-exl3}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

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
