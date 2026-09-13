#!/usr/bin/env bash
# Verify the host-side UMA guards without starting/stopping anything.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED="${1:-4}"
if [[ ! "$EXPECTED" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: expected host count must be a positive integer" >&2
  exit 2
fi

# shellcheck disable=SC2206
HOSTS=( ${OOM_GUARD_HOSTS:-} )
if (( ${#HOSTS[@]} != EXPECTED )); then
  echo "ERROR: OOM_GUARD_HOSTS must contain exactly $EXPECTED SSH hosts; got ${#HOSTS[@]}." >&2
  echo "Example: OOM_GUARD_HOSTS=\"spark-a spark-b spark-c spark-d\"" >&2
  exit 2
fi

TARGET="${OOM_GUARD_CONTAINER_NAME:-${CONTAINER_NAME:-dsv41-exl3}}"
STATUS="$(OOM_GUARD_CONTAINER_NAME="$TARGET" "$SCRIPT_DIR/watch_oom_guard.sh" status)"
printf '%s\n' "$STATUS"

armed="$(grep -c '^OOM_GUARD_ARMED=YES$' <<<"$STATUS" || true)"
alive="$(grep -c '^OOM_GUARD_ALIVE=YES ' <<<"$STATUS" || true)"
targets="$(grep -c "^OOM_GUARD_CONTAINER_NAME=$TARGET$" <<<"$STATUS" || true)"

if (( armed != EXPECTED || alive != EXPECTED || targets != EXPECTED )); then
  cat >&2 <<EOF
ERROR: OOM guard qualification failed.
  expected hosts: $EXPECTED
  armed:          $armed
  alive:          $alive
  exact target:   $targets ($TARGET)

Start/repair the guards before model load:
  OOM_GUARD_HOSTS="$OOM_GUARD_HOSTS" \\
  OOM_GUARD_CONTAINER_NAME="$TARGET" \\
    bash scripts/watch_oom_guard.sh start
EOF
  exit 2
fi

echo "OOM_GUARD_CLUSTER=PASS hosts=$EXPECTED target=$TARGET"
