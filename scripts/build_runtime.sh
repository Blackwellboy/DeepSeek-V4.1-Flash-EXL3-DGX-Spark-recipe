#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:deepseekv41-flash-0909}"
VLLM_EXL3_REF="${VLLM_EXL3_REF:-8f4517e80416466fa4a3ad2eb28685021d39e95f}"
EXLLAMAV3_REF="${EXLLAMAV3_REF:-be57335b087e4f001c5caae061544df3c06ba01e}"

echo "Building $IMAGE"
echo "  base:        $BASE_IMAGE"
echo "  vllm-exl3:  $VLLM_EXL3_REF"
echo "  ExLlamaV3:  $EXLLAMAV3_REF"

docker build \
  -f "$RECIPE_ROOT/Dockerfile.spark" \
  --build-arg BASE_IMAGE="$BASE_IMAGE" \
  --build-arg VLLM_EXL3_REF="$VLLM_EXL3_REF" \
  --build-arg EXLLAMAV3_REF="$EXLLAMAV3_REF" \
  -t "$IMAGE" \
  "$RECIPE_ROOT"

echo
echo "Built runtime. Local image identity:"
docker image inspect "$IMAGE" --format 'ID={{.Id}} RepoDigests={{json .RepoDigests}}'
