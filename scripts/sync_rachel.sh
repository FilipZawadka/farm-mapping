#!/usr/bin/env bash
# Sync Rachel's CAFO data: Google Drive -> RunPod volume (via S3 API) -> extract on pod.
#
# Prerequisites (one-time):
#   1. rclone installed: `curl https://rclone.org/install.sh | sudo bash`
#   2. `drive` remote configured (rclone config) with root_folder_id pinned to Rachel's folder.
#   3. .env in repo root with RUNPOD_API_KEY, RUNPOD_NETWORK_VOLUME_ID, POD_HOST, POD_PORT.
#
# Re-run any time. Idempotent: --update transfers only changed files, unzip -n never overwrites.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found" >&2; exit 1
fi

set -a; source "$ENV_FILE"; set +a

: "${RUNPOD_NETWORK_VOLUME_ID:?missing in .env}"
: "${POD_HOST:?missing in .env}"
: "${POD_PORT:?missing in .env}"
: "${RUNPOD_S3_ACCESS_KEY_ID:?missing in .env — create S3 keys in RunPod console -> Settings -> S3 API Keys}"
: "${RUNPOD_S3_SECRET_ACCESS_KEY:?missing in .env}"
: "${RUNPOD_S3_ENDPOINT:?missing in .env — RunPod console shows endpoint when creating S3 key}"
: "${RUNPOD_S3_REGION:?missing in .env}"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

VOLUME_PATH="data/rachel_geometry_candidates/all_countries_zips"
EXTRACT_DST="/workspace/farm-mapping/data/rachel_geometry_candidates/all_countries"

# Pass RunPod S3 creds via env so rclone uses them without saving to config
export RCLONE_CONFIG_RUNPOD_TYPE=s3
export RCLONE_CONFIG_RUNPOD_PROVIDER=Other
export RCLONE_CONFIG_RUNPOD_ENV_AUTH=false
export RCLONE_CONFIG_RUNPOD_ACCESS_KEY_ID="$RUNPOD_S3_ACCESS_KEY_ID"
export RCLONE_CONFIG_RUNPOD_SECRET_ACCESS_KEY="$RUNPOD_S3_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_RUNPOD_ENDPOINT="$RUNPOD_S3_ENDPOINT"
export RCLONE_CONFIG_RUNPOD_REGION="$RUNPOD_S3_REGION"

echo "=== 1. Drive -> RunPod volume (zip staging) ==="
rclone sync \
  drive: \
  "runpod:${RUNPOD_NETWORK_VOLUME_ID}/${VOLUME_PATH}/" \
  --progress --transfers=4 --checkers=8 --update

echo
echo "=== 2. Extract any new ZIPs on the pod ==="
ssh -o StrictHostKeyChecking=no -p "$POD_PORT" -i "$SSH_KEY" "root@$POD_HOST" "bash -s" <<EOF
set -e
SRC=/workspace/farm-mapping/${VOLUME_PATH}
DST=${EXTRACT_DST}
mkdir -p "\$DST"
n=0
for z in "\$SRC"/*.zip; do
  [ -f "\$z" ] || continue
  unzip -nq "\$z" -d "\$DST" && n=\$((n+1)) || echo "WARN: \$(basename \$z) failed"
done
echo "Extracted \$n zip(s). all_countries/ has \$(ls "\$DST" | wc -l) entries."
EOF

echo
echo "=== Done. ==="
