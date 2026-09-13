#!/usr/bin/env bash
# Fail-closed host MemAvailable watchdog for Spark / GB10 UMA qualification.
# Removable: delete this script + PID file + receipt dir; no systemd unit by default.
#
# Defaults for disk-backed Engram / TP4 load experiments:
#   WARN_GIB=24  ABORT_GIB=16
# Abort targets one exact container name only (recipe default: dsv41-exl3).
set -euo pipefail

HOSTNAME_S="$(hostname -s)"
GUARD_DIR="${OOM_GUARD_DIR:-/var/tmp/recipe-oom-guard}"
RECEIPT_DIR="${GUARD_DIR}/receipts"
PID_FILE="${GUARD_DIR}/oom_guard.pid"
STATE_FILE="${GUARD_DIR}/state.env"
LOG_FILE="${GUARD_DIR}/guard.log"
POLL_SEC="${OOM_GUARD_POLL_SEC:-1}"
WARN_GIB="${OOM_GUARD_WARN_GIB:-24}"
ABORT_GIB="${OOM_GUARD_ABORT_GIB:-16}"
# Swap growth: record WARN if SwapFree drops by this many GiB from baseline.
SWAP_GROW_GIB="${OOM_GUARD_SWAP_GROW_GIB:-2}"
TARGET_CONTAINER="${OOM_GUARD_CONTAINER_NAME:-dsv41-exl3}"
SYNTHETIC_TEST="${OOM_GUARD_SYNTHETIC_TEST:-0}"
DRY_ABORT="${OOM_GUARD_DRY_ABORT:-0}"

mkdir -p "$RECEIPT_DIR"
exec >>"$LOG_FILE" 2>&1

ts() { date -Is; }

read_meminfo() {
  local k="$1"
  awk -v k="$k" '$1==k":" {print $2; exit}' /proc/meminfo
}

gib_from_kb() {
  awk -v kb="$1" 'BEGIN{printf "%.3f", kb/1024/1024}'
}

list_target_containers() {
  # Never pattern-match unrelated DeepSeek jobs. A guard that can stop/kill a
  # container must be bound to one exact recipe container name.
  docker ps --format '{{.Names}}' 2>/dev/null | grep -Fx "$TARGET_CONTAINER" || true
}

abort_local_targets() {
  local reason="$1"
  local names
  names="$(list_target_containers | tr '\n' ' ')"
  if [[ -z "${names// }" ]]; then
    echo "$(ts) ABORT_NO_TARGET reason=$reason name=$TARGET_CONTAINER"
    return 0
  fi
  echo "$(ts) ABORT_STOPPING reason=$reason names=$names dry=$DRY_ABORT"
  if [[ "$DRY_ABORT" == "1" ]]; then
    echo "$(ts) DRY_ABORT=YES skipped docker stop"
    return 0
  fi
  docker stop -t 5 "$TARGET_CONTAINER" || true
  if docker ps --format '{{.Names}}' | grep -Fxq "$TARGET_CONTAINER"; then
    docker kill "$TARGET_CONTAINER" || true
  fi
}

write_receipt() {
  local kind="$1"
  local reason="$2"
  local mem_avail_kb="$3"
  local swap_free_kb="$4"
  local cached_kb="$5"
  local receipt="${RECEIPT_DIR}/$(date +%Y%m%dT%H%M%S)_${kind}_${HOSTNAME_S}.txt"
  {
    echo "KIND=$kind"
    echo "REASON=$reason"
    echo "HOST=$HOSTNAME_S"
    echo "TS=$(ts)"
    echo "MemAvailable_kB=$mem_avail_kb"
    echo "MemAvailable_GiB=$(gib_from_kb "$mem_avail_kb")"
    echo "SwapFree_kB=$swap_free_kb"
    echo "Cached_kB=$cached_kb"
    echo "WARN_GIB=$WARN_GIB"
    echo "ABORT_GIB=$ABORT_GIB"
    echo "TARGET_CONTAINER=$TARGET_CONTAINER"
    echo "TARGETS=$(list_target_containers | tr '\n' ',')"
    echo "--- /proc/meminfo subset ---"
    grep -E '^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapTotal|SwapFree|Active\(file\)|Inactive\(file\)|Dirty|Writeback|AnonPages|Mapped|Shmem):' /proc/meminfo
  } >"$receipt"
  # Log the human-readable line on stderr so command substitution captures only
  # the pathname. This keeps OOM_GUARD_TRIGGER_RECEIPT a valid one-line env value.
  echo "$(ts) RECEIPT=$receipt" >&2
  printf '%s\n' "$receipt"
}

if [[ -f "$PID_FILE" ]]; then
  old="$(cat "$PID_FILE" || true)"
  if [[ -n "$old" ]] && kill -0 "$old" 2>/dev/null; then
    echo "$(ts) already running pid=$old"
    exit 0
  fi
fi
echo $$ >"$PID_FILE"
trap 'rm -f "$PID_FILE"' EXIT

BASE_SWAP_FREE_KB="$(read_meminfo SwapFree)"
{
  echo "OOM_GUARD_ARMED=YES"
  echo "OOM_GUARD_ABORT_THRESHOLD_GIB=$ABORT_GIB"
  echo "OOM_GUARD_WARN_THRESHOLD_GIB=$WARN_GIB"
  echo "OOM_GUARD_HOST=$HOSTNAME_S"
  echo "OOM_GUARD_PID=$$"
  echo "OOM_GUARD_STARTED=$(ts)"
  echo "OOM_GUARD_CONTAINER_NAME=$TARGET_CONTAINER"
  echo "OOM_GUARD_SYNTHETIC_TEST=$SYNTHETIC_TEST"
} >"$STATE_FILE"

echo "$(ts) ARMED host=$HOSTNAME_S warn=${WARN_GIB}GiB abort=${ABORT_GIB}GiB poll=${POLL_SEC}s target=$TARGET_CONTAINER synthetic=$SYNTHETIC_TEST dry=$DRY_ABORT"

warned_mem=0
warned_swap=0
while true; do
  mem_avail_kb="$(read_meminfo MemAvailable)"
  swap_free_kb="$(read_meminfo SwapFree)"
  cached_kb="$(read_meminfo Cached)"
  mem_avail_gib="$(gib_from_kb "$mem_avail_kb")"
  swap_delta_kb=$(( BASE_SWAP_FREE_KB - swap_free_kb ))
  if (( swap_delta_kb < 0 )); then swap_delta_kb=0; fi
  swap_delta_gib="$(gib_from_kb "$swap_delta_kb")"

  if [[ "$SYNTHETIC_TEST" == "1" ]]; then
    :
  fi

  need_abort="$(awk -v a="$mem_avail_gib" -v t="$ABORT_GIB" 'BEGIN{print (a<t)?1:0}')"
  need_warn="$(awk -v a="$mem_avail_gib" -v t="$WARN_GIB" 'BEGIN{print (a<t)?1:0}')"
  swap_warn="$(awk -v d="$swap_delta_gib" -v t="$SWAP_GROW_GIB" 'BEGIN{print (d>=t)?1:0}')"

  if [[ "$need_warn" == "1" && "$warned_mem" == "0" ]]; then
    write_receipt WARN "MemAvailable_below_${WARN_GIB}GiB" "$mem_avail_kb" "$swap_free_kb" "$cached_kb" >/dev/null
    echo "$(ts) WARN MemAvailable=${mem_avail_gib}GiB"
    warned_mem=1
  fi

  if [[ "$need_abort" == "1" ]]; then
    receipt="$(write_receipt ABORT "MemAvailable_below_${ABORT_GIB}GiB" "$mem_avail_kb" "$swap_free_kb" "$cached_kb")"
    abort_local_targets "MemAvailable=${mem_avail_gib}GiB<${ABORT_GIB}"
    echo "OOM_GUARD_TRIGGERED=YES" >>"$STATE_FILE"
    echo "OOM_GUARD_TRIGGER_RECEIPT=$receipt" >>"$STATE_FILE"
    echo "$(ts) HARD_ABORT done receipt=$receipt"
    exit 2
  fi

  # Swap growth by itself is evidence, not a kill condition. The hard stop is
  # MemAvailable crossing the explicit UMA abort threshold above.
  if [[ "$swap_warn" == "1" && "$warned_swap" == "0" ]]; then
    write_receipt WARN "Swap_growth_${swap_delta_gib}GiB" "$mem_avail_kb" "$swap_free_kb" "$cached_kb" >/dev/null
    echo "$(ts) WARN SwapGrowth=${swap_delta_gib}GiB MemAvailable=${mem_avail_gib}GiB"
    warned_swap=1
  fi

  sleep "$POLL_SEC"
done
