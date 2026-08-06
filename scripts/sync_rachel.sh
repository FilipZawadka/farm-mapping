#!/usr/bin/env bash
# Sync Rachel's per-country for_analysis parquets: Google Drive -> RunPod network volume.
#
# WHY THIS IS NOT A PLAIN `rclone sync drive: runpod:...` (all three are real, hit 2026-07):
#   1. Rachel's Drive folder is now the WHOLE WORLD (~204 country folders, multi-GB each — e.g.
#      Brazil has a 4.2GB brazil_osm.gpkg + a building-geojson tree). An unscoped `rclone sync`
#      would mirror terabytes and, being `sync` not `copy`, DELETE non-matching files on the dest.
#      We only need the small *_selected_clusters_for_analysis.parquet files (~72MB total; 167
#      countries have one) — the input to merge_clusters_v5 and the cnn/if_split_assigned splits.
#   2. RunPod's S3 API rejects the `x-amz-acl` header. The system rclone (v1.60) ALWAYS sends
#      `X-Amz-Acl: private` even with --s3-acl="" (501 NotImplemented); a modern rclone (v1.74)
#      omits it but then CANNOT authenticate to RunPod (SDK v2 -> SignatureDoesNotMatch). So there
#      is NO working laptop-rclone -> RunPod-S3 UPLOAD path.
#   3. The `drive` remote uses rclone's shared OAuth client (no personal client_id) -> Google
#      403 rateLimitExceeded. Throttled below with --tpslimit; set a personal client_id to remove
#      it for good (see RUNBOOK at the bottom of this file).
#
# APPROACH: pull Drive -> a local staging dir (read-only; local writes have no ACL), then rsync
# that tree to the pod's mounted volume over SSH (plain filesystem write — bypasses S3/ACL).
# Idempotent & non-destructive (rsync without --delete; merges into existing country dirs).
#
# Prerequisites (one-time):
#   1. rclone with a `drive` remote whose root_folder_id is pinned to Rachel's folder.
#   2. .env in repo root with POD_HOST, POD_PORT (RunPod pod SSH). SSH_KEY optional (defaults below).
#   3. The pod running with the network volume mounted at /workspace.
#
# Re-run any time. Env overrides: RACHEL_INCLUDE (glob), RACHEL_STAGE (local staging dir).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/.env"

if [ ! -f "$ENV_FILE" ]; then
  echo "ERROR: $ENV_FILE not found" >&2; exit 1
fi

set -a; source "$ENV_FILE"; set +a

: "${POD_HOST:?missing in .env}"
: "${POD_PORT:?missing in .env}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"

# What to pull. Cluster for_analysis is the pipeline input. To also grab the buildings parquets,
# run with RACHEL_INCLUDE='*_for_analysis.parquet'.
INCLUDE_PATTERN="${RACHEL_INCLUDE:-*_selected_clusters_for_analysis.parquet}"

# Local staging dir (kept between runs so re-syncs are incremental).
STAGE="${RACHEL_STAGE:-$REPO_ROOT/data/rachel_geometry_candidates/for_analysis_stage}"

# Destination on the pod. The network volume is mounted at /workspace, so this path also equals
# the S3 key  <RUNPOD_NETWORK_VOLUME_ID>/farm-mapping/data/rachel_geometry_candidates/all_countries/
POD_DST="/workspace/farm-mapping/data/rachel_geometry_candidates/all_countries"

SSH_OPTS=(-o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$POD_PORT" -i "$SSH_KEY")

echo "=== 0. Check pod reachable ==="
ssh "${SSH_OPTS[@]}" "root@$POD_HOST" 'test -d /workspace/farm-mapping' \
  || { echo "ERROR: cannot ssh to pod or /workspace/farm-mapping missing (is the pod up?)" >&2; exit 1; }

echo
echo "=== 1. Drive -> local staging ($STAGE) ==="
# --max-depth 2: the files live at drive:{Country}/{ISO}_...parquet, so depth-2 finds them all
# WITHOUT descending into the huge per-country *_buildings/ geojson trees (fewer API calls, less
# rate-limit exposure). Preserves the {Country}/ layout. --tpslimit rides out the shared-client 403s.
mkdir -p "$STAGE"
rclone copy drive: "$STAGE/" \
  --include "$INCLUDE_PATTERN" \
  --max-depth 2 \
  --tpslimit 6 --transfers 4 --checkers 8 \
  --low-level-retries 10 --retries 5 \
  --stats 15s --stats-one-line -v
SRC_N=$(find "$STAGE" -name "$INCLUDE_PATTERN" -type f | wc -l)
echo "Local staging now has $SRC_N file(s) matching '$INCLUDE_PATTERN'."

echo
echo "=== 2. Staging -> pod volume ($POD_DST) via rsync (non-destructive) ==="
ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "mkdir -p '$POD_DST'"
# include filters: send only the matching parquets (+ the dirs needed to hold them), nothing else.
rsync -a --info=stats1 \
  --include='*/' --include="$INCLUDE_PATTERN" --exclude='*' --prune-empty-dirs \
  -e "ssh ${SSH_OPTS[*]}" \
  "$STAGE/" "root@$POD_HOST:$POD_DST/"

echo
echo "=== 3. Verify pod count matches source ==="
POD_N=$(ssh "${SSH_OPTS[@]}" "root@$POD_HOST" "find '$POD_DST' -name '$INCLUDE_PATTERN' -type f | wc -l")
echo "Source (local staging): $SRC_N    On pod volume: $POD_N"
if [ "$SRC_N" -eq "$POD_N" ] && [ "$SRC_N" -gt 0 ]; then
  echo "=== Done. All $POD_N file(s) synced to the volume. ==="
else
  echo "WARNING: count mismatch (source=$SRC_N pod=$POD_N) — investigate." >&2
  exit 1
fi

# ============================================================================================
# RUNBOOK — eliminate the Google Drive 403 rateLimitExceeded (optional but recommended)
# --------------------------------------------------------------------------------------------
# The `drive` remote currently uses rclone's SHARED OAuth client, whose per-minute quota is
# globally exhausted (hence intermittent 403s that --tpslimit only mitigates). Give it a personal
# Google OAuth client to get your own quota:
#   1. https://console.cloud.google.com -> create/select a project.
#   2. APIs & Services -> Library -> enable "Google Drive API".
#   3. APIs & Services -> Credentials -> Create Credentials -> OAuth client ID -> type "Desktop app".
#      (Configure the OAuth consent screen if prompted; add yourself as a test user.)
#   4. Copy the Client ID and Client secret, then:
#        rclone config reconnect drive:            # re-runs OAuth against the new client
#      or `rclone config` -> edit `drive` -> set client_id / client_secret -> reconnect (browser).
#   A reconnect is REQUIRED: the existing token was minted by the shared client and won't refresh
#   under a new client_id. Verify with:  rclone about drive:   (should stop 403-ing).
# ============================================================================================
