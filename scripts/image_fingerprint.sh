#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"

ARCHIVE="${1:-}"

if [[ -n "$ARCHIVE" ]]; then
  if [[ ! -f "$ARCHIVE" ]]; then
    echo "ERROR: archive not found: $ARCHIVE" >&2
    exit 2
  fi
  echo "archive_sha256=$(sha256sum "$ARCHIVE" | awk '{print $1}')"
fi

echo "docker_version=$(docker version --format '{{.Server.Version}}' 2>/dev/null || true)"
echo "storage_driver=$(docker info --format '{{.Driver}}' 2>/dev/null || true)"
echo "image_ref=$IMAGE"
echo "image_id=$(docker image inspect "$IMAGE" --format '{{.Id}}')"
echo "repo_digests=$(docker image inspect "$IMAGE" --format '{{json .RepoDigests}}')"
echo "rootfs_layers=$(docker image inspect "$IMAGE" --format '{{json .RootFS.Layers}}')"

cat <<'EOF'

Note: Docker/Moby issue #51934 reports docker load producing different image IDs
across hosts/storage integrations even when the saved image content/layers match.
For cross-node verification, compare the archive SHA256 (when available), the full
RootFS layer list, the pinned source revisions, and runtime_identity output. Do not
reject an otherwise identical Spark solely because the short docker images ID differs.
EOF
