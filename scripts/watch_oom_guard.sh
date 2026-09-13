#!/usr/bin/env bash
# Portable install/start/stop/status helper for scripts/oom_guard.sh across Spark hosts.
#
# Usage:
#   OOM_GUARD_HOSTS="host1 host2 host3 host4" bash scripts/watch_oom_guard.sh install|start|stop|status|uninstall
#
# Thresholds (defaults match disk-Engram / TP4 UMA qualification):
#   OOM_GUARD_WARN_GIB=24
#   OOM_GUARD_ABORT_GIB=16
#
# Do not hard-code private host inventories in this recipe.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACTION="${1:?usage: $0 install|start|stop|status|uninstall}"
GUARD_SCRIPT="$SCRIPT_DIR/oom_guard.sh"
REMOTE_DIR="${OOM_GUARD_REMOTE_DIR:-/var/tmp/recipe-oom-guard}"
SCRIPT_NAME=oom_guard.sh

if [[ ! -f "$GUARD_SCRIPT" ]]; then
  echo "ERROR: missing $GUARD_SCRIPT" >&2
  exit 2
fi

# shellcheck disable=SC2206
HOSTS=( ${OOM_GUARD_HOSTS:-} )
if (( ${#HOSTS[@]} == 0 )); then
  cat >&2 <<EOF
ERROR: set OOM_GUARD_HOSTS to a space-separated list of SSH hosts.
Example:
  OOM_GUARD_HOSTS="spark-a spark-b spark-c spark-d" $0 start
EOF
  exit 2
fi

deploy() {
  local h="$1"
  ssh -o BatchMode=yes "$h" "mkdir -p $REMOTE_DIR/receipts"
  scp -o BatchMode=yes "$GUARD_SCRIPT" "$h:$REMOTE_DIR/$SCRIPT_NAME"
}

for h in "${HOSTS[@]}"; do
  case "$ACTION" in
    install)
      deploy "$h"
      echo "$h INSTALL_OK"
      ;;
    start)
      WARN="${OOM_GUARD_WARN_GIB:-24}"
      ABORT="${OOM_GUARD_ABORT_GIB:-16}"
      MATCH="${OOM_GUARD_CONTAINER_MATCH:-dsv41-exl3}"
      deploy "$h"
      ssh -o BatchMode=yes "$h" "mkdir -p $REMOTE_DIR/receipts
        if [[ -f $REMOTE_DIR/oom_guard.pid ]] && kill -0 \$(cat $REMOTE_DIR/oom_guard.pid) 2>/dev/null; then
          echo ALREADY_RUNNING; exit 0; fi
        nohup env OOM_GUARD_DIR=$REMOTE_DIR \
          OOM_GUARD_WARN_GIB=$WARN \
          OOM_GUARD_ABORT_GIB=$ABORT \
          OOM_GUARD_CONTAINER_MATCH=$MATCH \
          OOM_GUARD_DRY_ABORT=${OOM_GUARD_DRY_ABORT:-0} \
          OOM_GUARD_SYNTHETIC_TEST=${OOM_GUARD_SYNTHETIC_TEST:-0} \
          bash $REMOTE_DIR/$SCRIPT_NAME >/dev/null 2>&1 &
        sleep 0.5
        cat $REMOTE_DIR/state.env"
      ;;
    stop)
      ssh -o BatchMode=yes "$h" "
        if [[ -f $REMOTE_DIR/oom_guard.pid ]]; then
          kill \$(cat $REMOTE_DIR/oom_guard.pid) 2>/dev/null || true
          rm -f $REMOTE_DIR/oom_guard.pid
        fi
        rm -f $REMOTE_DIR/state.env
        echo STOPPED"
      ;;
    status)
      ssh -o BatchMode=yes "$h" "
        echo HOST=\$(hostname -s)
        if [[ -f $REMOTE_DIR/state.env ]]; then cat $REMOTE_DIR/state.env; else echo OOM_GUARD_ARMED=NO; fi
        if [[ -f $REMOTE_DIR/oom_guard.pid ]] && kill -0 \$(cat $REMOTE_DIR/oom_guard.pid) 2>/dev/null; then
          echo OOM_GUARD_ALIVE=YES PID=\$(cat $REMOTE_DIR/oom_guard.pid)
        else
          echo OOM_GUARD_ALIVE=NO
        fi
        ls -1t $REMOTE_DIR/receipts 2>/dev/null | head -3 || true
        echo ---
      "
      ;;
    uninstall)
      ssh -o BatchMode=yes "$h" "
        if [[ -f $REMOTE_DIR/oom_guard.pid ]]; then
          kill \$(cat $REMOTE_DIR/oom_guard.pid) 2>/dev/null || true
        fi
        rm -rf $REMOTE_DIR
        echo UNINSTALLED"
      ;;
    *)
      echo "usage: $0 install|start|stop|status|uninstall" >&2
      exit 2
      ;;
  esac
done
