#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ROLE="${1:-}"
if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
  echo "Usage: $0 head|worker" >&2
  exit 2
fi

require_env HEAD_IP
require_env NODE_IP

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "ERROR: image '$IMAGE' is not present. Run ./scripts/build_runtime.sh first." >&2
  exit 2
fi

# Never destroy an existing runtime implicitly. This is especially important on
# long Spark qualification runs where a second shell may call the launcher by mistake.
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  if ! is_true "${REPLACE_CONTAINER:-0}"; then
    echo "ERROR: container '$CONTAINER_NAME' already exists." >&2
    echo "Set REPLACE_CONTAINER=1 only when you intentionally want to replace it." >&2
    exit 2
  fi
  echo "REPLACE_CONTAINER=1: removing existing '$CONTAINER_NAME'." >&2
  docker rm -f "$CONTAINER_NAME" >/dev/null
fi

# RDMA passthrough adapted from PR #1 by @fattchris.
# ENABLE_RDMA=auto: map RDMA when the device exists; otherwise use normal NCCL selection.
# ENABLE_RDMA=1: require the device and hard-fail if it is missing.
# ENABLE_RDMA=0: do not map RDMA and do not inject IB/RoCE overrides.
ENABLE_RDMA="${ENABLE_RDMA:-auto}"
RDMA_DEVICE="${RDMA_DEVICE:-/dev/infiniband}"
case "$ENABLE_RDMA" in
  auto|1|0) ;;
  *) echo "ERROR: ENABLE_RDMA must be auto, 1, or 0 (got '$ENABLE_RDMA')" >&2; exit 2 ;;
esac

rdma_enabled=0
if [[ "$ENABLE_RDMA" == "1" ]]; then
  if [[ ! -e "$RDMA_DEVICE" ]]; then
    echo "ERROR: ENABLE_RDMA=1 but RDMA device '$RDMA_DEVICE' does not exist." >&2
    exit 2
  fi
  rdma_enabled=1
elif [[ "$ENABLE_RDMA" == "auto" ]]; then
  if [[ -e "$RDMA_DEVICE" ]]; then
    rdma_enabled=1
  else
    echo "INFO: RDMA device '$RDMA_DEVICE' not found; using normal NCCL/Gloo network selection." >&2
  fi
fi

DOCKER_ARGS=(
  run -d --rm
  --name "$CONTAINER_NAME"
  --gpus all
  --network host
  --ipc host
  --ulimit memlock=-1
  --ulimit stack=67108864
  --entrypoint ray
  -v "$RECIPE_ROOT:/recipe:ro"
  -v "$HF_HOME:/root/.cache/huggingface"
  -e HF_HOME=/root/.cache/huggingface
  -e VLLM_HOST_IP="$NODE_IP"
  -e RAY_DEDUP_LOGS=0
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600
)

if [[ "$rdma_enabled" == "1" ]]; then
  DOCKER_ARGS+=( --device "$RDMA_DEVICE" )

  # Use the tested DGX Spark defaults only when the corresponding host devices
  # actually exist. Otherwise leave selection to NCCL unless the user overrides it.
  NCCL_IB_HCA_EFFECTIVE="${NCCL_IB_HCA:-}"
  if [[ -z "$NCCL_IB_HCA_EFFECTIVE" && -e /sys/class/infiniband/rocep1s0f1 ]]; then
    NCCL_IB_HCA_EFFECTIVE="rocep1s0f1"
  fi

  NCCL_SOCKET_IFNAME_EFFECTIVE="${NCCL_SOCKET_IFNAME:-}"
  if [[ -z "$NCCL_SOCKET_IFNAME_EFFECTIVE" && -e /sys/class/net/enp1s0f1np1 ]]; then
    NCCL_SOCKET_IFNAME_EFFECTIVE="enp1s0f1np1"
  fi
  GLOO_SOCKET_IFNAME_EFFECTIVE="${GLOO_SOCKET_IFNAME:-$NCCL_SOCKET_IFNAME_EFFECTIVE}"

  DOCKER_ARGS+=(
    -e NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
    -e NCCL_IB_ROCE_VERSION_NUM=${NCCL_IB_ROCE_VERSION_NUM:-2}
    -e NCCL_IB_ADDR_FAMILY=${NCCL_IB_ADDR_FAMILY:-AF_INET}
    -e NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
    -e NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0}
  )
  if [[ -n "${NCCL_NET:-}" ]]; then
    DOCKER_ARGS+=( -e NCCL_NET="$NCCL_NET" )
  fi
  if [[ -n "$NCCL_IB_HCA_EFFECTIVE" ]]; then
    DOCKER_ARGS+=( -e NCCL_IB_HCA="$NCCL_IB_HCA_EFFECTIVE" )
  fi
  if [[ -n "$NCCL_SOCKET_IFNAME_EFFECTIVE" ]]; then
    DOCKER_ARGS+=( -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME_EFFECTIVE" )
  fi
  if [[ -n "$GLOO_SOCKET_IFNAME_EFFECTIVE" ]]; then
    DOCKER_ARGS+=( -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME_EFFECTIVE" )
  fi
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  DOCKER_ARGS+=( -e HF_TOKEN="$HF_TOKEN" )
fi

if [[ -n "${MODEL_DIR:-}" ]]; then
  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: MODEL_DIR '$MODEL_DIR' does not exist on this node." >&2
    exit 2
  fi
  DOCKER_ARGS+=( -v "$MODEL_DIR:/models" )
fi

if [[ "$ROLE" == "head" ]]; then
  if [[ "$NODE_IP" != "$HEAD_IP" ]]; then
    echo "ERROR: head NODE_IP ($NODE_IP) must equal HEAD_IP ($HEAD_IP)." >&2
    exit 2
  fi
  RAY_ARGS=(start --head --node-ip-address="$NODE_IP" --port="$RAY_PORT" --num-gpus=1 --disable-usage-stats --block)
else
  RAY_ARGS=(start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$NODE_IP" --num-gpus=1 --disable-usage-stats --block)
fi

echo "Starting $ROLE container '$CONTAINER_NAME' on $NODE_IP (RDMA=$rdma_enabled)"
docker "${DOCKER_ARGS[@]}" "$IMAGE" "${RAY_ARGS[@]}"

echo "Container started. Recent logs:"
sleep 2
docker logs --tail 30 "$CONTAINER_NAME" || true
