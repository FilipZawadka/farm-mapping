#!/usr/bin/env bash
# Round-5 balancing campaign, end to end, from the laptop.
#
#   bash scripts/run_round5_campaign.sh                 # every default stage, in order
#   bash scripts/run_round5_campaign.sh sync merge      # only these stages
#   bash scripts/run_round5_campaign.sh --dry-run        # print the commands, run nothing
#   bash scripts/run_round5_campaign.sh --list
#
# Default stages (each idempotent; re-run any of them on its own):
#   sync      Rachel's round_5 per-country parquets: Google Drive -> local staging -> pod volume
#             (scripts/sync_rachel.sh with RACHEL_INCLUDE='*_selected_clusters_round_5.parquet')
#   merge     all_clusters_v11.parquet from the staged round_5 files (scripts/merge_clusters_v7.py:
#             asserts cluster_id stability + zero geometry drift vs v10, carries viz_status /
#             viz_label / template_score_if over by id)
#   upload    scp all_clusters_v11.parquet onto the pod volume
#   audit     per-country x class composition of the round_5 train split and what each sampler
#             would do to it (scripts/audit_country_balance.py) -- read this BEFORE spending
#   baseline  launch world_v10_fourclass_r5_a_s44 WITH the candidates step: it builds
#             candidates_world_v10_r5 on the volume, which every balanced arm reads
#   wait      block until that candidates dir is complete (_COMPLETE.json written by
#             training/rachel_to_candidates.py) -- launching the arms earlier would train on a
#             partially written directory
#   arms      launch g / h / i through experiments/launch_fleet.py (train + inference only)
#   collect   pull metrics, scored parquets and sampling reports while pods are alive
#             (also reaps finished pods; --watch until all four runs are in)
#   evaluate  experiments/evaluate_balancing.py on the four round_5 runs
# Optional (not run by default):
#   score     full-world scoring passes for publishing (gen_score_configs.py --prefix ...)
#
# Needs, on the laptop: .env with RUNPOD_API_KEY (+ RUNPOD_NETWORK_VOLUME_ID); the SSH key
# RunPod knows (~/.ssh/id_ed25519 or SSH_KEY); rclone with the `drive` remote (sync stage);
# one RUNNING pod with the volume mounted for sync / upload / wait (POD_HOST/POD_PORT are
# taken from .env if set there, otherwise resolved from the RunPod API); and
# data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet (merge stage).
# Everything the pods run is STAGED from this working tree, so be on `develop` at or after
# the commit that added training/balancing.py.
#
# Env overrides: BUDGET (launch_fleet balance reserve, default 5), MAX_CONCURRENT (3),
# RACHEL_STAGE (local staging dir), SSH_KEY, WAIT_MINUTES (40).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
REPO=$PWD

ROUND=r5
PATTERN='*_selected_clusters_round_5.parquet'
ALL=data/rachel_geometry_candidates/all_countries
V10=$ALL/all_clusters_v10.parquet
V11=$ALL/all_clusters_v11.parquet
STAGE="${RACHEL_STAGE:-$REPO/data/rachel_geometry_candidates/for_analysis_stage}"
REMOTE_ROOT=/workspace/farm-mapping
REMOTE_ALL=$REMOTE_ROOT/$ALL
REMOTE_CAND=$REMOTE_ROOT/data/rachel_geometry_candidates/candidates_world_v10_r5
ORDER=experiments/balancing_order_$ROUND.txt
STATE=experiments/results/${ROUND}_balancing_fleet_state.json
BASELINE=world_v10_fourclass_${ROUND}_a_s44
BASELINE_CFG=configs/rachel_clusters/$BASELINE.yaml
AUDIT_TXT=experiments/results/${ROUND}_train_country_audit.txt
BUDGET="${BUDGET:-5}"
MAX_CONCURRENT="${MAX_CONCURRENT:-3}"
WAIT_MINUTES="${WAIT_MINUTES:-40}"
DEFAULT_STAGES=(sync merge upload audit baseline wait arms collect evaluate)

DRY=0
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
run() { printf '   $ %s\n' "$*"; [ "$DRY" = 1 ] || "$@"; }
need_file() { [ "$DRY" = 1 ] && return 0; [ -f "$1" ] || { echo "   missing $1 -- $2" >&2; return 1; }; }

if [ -f .env ]; then set -a; . ./.env; set +a; fi
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
ssh_opts() { echo -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$POD_PORT" -i "$SSH_KEY"; }

resolve_pod() {
  if [ "$DRY" = 1 ]; then
    if [ -z "${POD_HOST:-}" ] || [ -z "${POD_PORT:-}" ] || [ "$POD_HOST" = '<pod-ip>' ]; then
      POD_HOST='<pod-ip>'; POD_PORT='<port>'; export POD_HOST POD_PORT
    fi
    echo "   pod: $POD_HOST:$POD_PORT (dry-run: not verified; resolved from the RunPod API at run time)"
    return 0
  fi
  # Prefer an explicit POD_HOST/POD_PORT (from .env or the environment); verify it is alive.
  if [ -n "${POD_HOST:-}" ] && [ -n "${POD_PORT:-}" ]; then
    if ssh $(ssh_opts) "root@$POD_HOST" 'test -d /workspace/farm-mapping' 2>/dev/null; then
      echo "   pod: $POD_HOST:$POD_PORT (from .env / environment)"; return 0
    fi
    echo "   POD_HOST=$POD_HOST:$POD_PORT is set but unreachable -- fix or remove it in .env" >&2
    return 1
  fi
  local ep
  ep=$(python3 - <<'PY'
import sys
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
from collect_results import live_pod
ep = live_pod()
print(f"{ep[0]} {ep[1]}" if ep else "")
PY
)
  if [ -z "$ep" ]; then
    echo "   no RUNNING pod with public SSH. Start one (any pod with the network volume), e.g." >&2
    echo "   python3 -m training.runpod_launch --config $BASELINE_CFG --steps candidates train inference" >&2
    return 1
  fi
  POD_HOST=${ep% *}; POD_PORT=${ep#* }
  export POD_HOST POD_PORT
  echo "   pod: $POD_HOST:$POD_PORT (resolved from the RunPod API)"
}

stage_sync() {
  say "sync: Rachel's round_5 files -> staging -> volume"
  [ "$DRY" = 1 ] || command -v rclone >/dev/null || { echo "rclone not installed" >&2; return 1; }
  resolve_pod
  run env RACHEL_INCLUDE="$PATTERN" RACHEL_STAGE="$STAGE" POD_HOST="$POD_HOST" POD_PORT="$POD_PORT" \
      bash scripts/sync_rachel.sh
}

stage_merge() {
  say "merge: $PATTERN -> $V11"
  if [ "$DRY" = 0 ]; then
    local n
    n=$(find "$STAGE" -name "$PATTERN" -type f 2>/dev/null | wc -l)
    [ "$n" -gt 0 ] || { echo "   no $PATTERN under $STAGE -- run the sync stage first" >&2; return 1; }
    echo "   $n round_5 country files staged"
  fi
  need_file "$V10" "needed for the id-stability check and carry-over columns"
  run python3 scripts/merge_clusters_v7.py --round2-dir "$STAGE" --pattern "$PATTERN" --prev "$V10" --out "$V11"
}

stage_upload() {
  say "upload: $V11 -> volume"
  need_file "$V11" "run the merge stage first"
  resolve_pod
  run ssh $(ssh_opts) "root@$POD_HOST" "mkdir -p '$REMOTE_ALL'"
  run scp -o StrictHostKeyChecking=no -i "$SSH_KEY" -P "$POD_PORT" "$V11" "root@$POD_HOST:$REMOTE_ALL/"
  if [ "$DRY" = 0 ]; then
    local local_sz remote_sz
    local_sz=$(stat -c %s "$V11" 2>/dev/null || stat -f %z "$V11")
    remote_sz=$(ssh $(ssh_opts) "root@$POD_HOST" "stat -c %s '$REMOTE_ALL/$(basename "$V11")'")
    [ "$local_sz" = "$remote_sz" ] && echo "   ok: $remote_sz bytes on the volume" \
      || { echo "   size mismatch local=$local_sz remote=$remote_sz" >&2; return 1; }
  fi
}

stage_audit() {
  say "audit: round_5 train split composition + sampler what-if"
  need_file "$V11" "run the merge stage first"
  mkdir -p experiments/results
  run python3 scripts/audit_country_balance.py --parquet "$V11" --split train --schemes all \
      --json "experiments/results/${ROUND}_train_country_audit.json"
  [ "$DRY" = 1 ] || python3 scripts/audit_country_balance.py --parquet "$V11" --split train --schemes all \
      > "$AUDIT_TXT" 2>&1 || true
  echo "   saved $AUDIT_TXT -- check: single-label countries, NMI before/after per scheme, ESS ratio of G"
  echo "   (ESS < 0.25 for G => set temperature: 2 in gen_balancing_configs.py and regenerate)"
}

stage_baseline() {
  say "baseline: $BASELINE with the candidates step (builds candidates_world_v10_r5)"
  need_file "$BASELINE_CFG" "run: python3 experiments/gen_balancing_configs.py"
  # Drop any stale completion marker so `wait` waits for THIS conversion.
  if resolve_pod 2>/dev/null; then
    run ssh $(ssh_opts) "root@$POD_HOST" "rm -f '$REMOTE_CAND/_COMPLETE.json'"
  else
    echo "   (no live pod yet; nothing to clean)"
  fi
  run python3 -m training.runpod_launch --config "$BASELINE_CFG" --steps candidates train inference
}

stage_wait() {
  say "wait: candidates_world_v10_r5 complete on the volume (up to $WAIT_MINUTES min)"
  [ "$DRY" = 1 ] && return 0
  local i=0
  until resolve_pod >/dev/null 2>&1; do
    i=$((i + 1)); [ "$i" -le "$WAIT_MINUTES" ] || { echo "no reachable pod after $WAIT_MINUTES min" >&2; return 1; }
    echo "   no reachable pod yet ($i/$WAIT_MINUTES min)"; sleep 60
  done
  i=0
  while :; do
    if ssh $(ssh_opts) "root@$POD_HOST" \
        "test -f '$REMOTE_CAND/_COMPLETE.json' && ! ls '$REMOTE_CAND'/*.csv.tmp >/dev/null 2>&1" 2>/dev/null; then
      echo "   complete:"; ssh $(ssh_opts) "root@$POD_HOST" "cat '$REMOTE_CAND/_COMPLETE.json'"
      return 0
    fi
    i=$((i + 1)); [ "$i" -le "$WAIT_MINUTES" ] || { echo "candidates dir still incomplete after $WAIT_MINUTES min" >&2; return 1; }
    echo "   not yet ($i/$WAIT_MINUTES min)"; sleep 60
  done
}

stage_arms() {
  say "arms: g / h / i via launch_fleet (train inference), state $STATE"
  [ -f "$ORDER" ] || { echo "missing $ORDER -- run: python3 experiments/gen_balancing_configs.py" >&2; return 1; }
  local arms_order
  arms_order=$(mktemp)
  grep -v "_a_s" "$ORDER" > "$arms_order"
  echo "   runs: $(tr '\n' ' ' < "$arms_order")"
  run python3 experiments/launch_fleet.py --order-file "$arms_order" --state "$STATE" \
      --max-concurrent "$MAX_CONCURRENT" --budget "$BUDGET" --steps train inference
}

stage_collect() {
  say "collect: metrics + scored parquets + sampling reports for $(tr '\n' ' ' < "$ORDER")"
  run python3 experiments/collect_results.py --watch --names-file "$ORDER"
}

stage_evaluate() {
  say "evaluate: round_5 arms vs $BASELINE"
  need_file "$V11" "the evaluation slices come from it (merge stage)"
  run python3 experiments/evaluate_balancing.py --prefix "world_v10_fourclass_$ROUND" --v10 "$V11"
}

stage_score() {
  say "score (optional): full-world scoring passes for publishing"
  run python3 experiments/gen_score_configs.py --prefix "world_v10_fourclass_$ROUND"
  echo "   first scoring run must build the *_scoreall candidates dir:"
  echo "     python3 -m training.runpod_launch --config configs/rachel_clusters/${BASELINE}_score.yaml --steps candidates inference"
  echo "   then the rest with --steps inference through launch_fleet (order file of the *_score names)"
}

stages=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --list) printf '%s\n' "${DEFAULT_STAGES[@]}" score; exit 0 ;;
    sync|merge|upload|audit|baseline|wait|arms|collect|evaluate|score) stages+=("$a") ;;
    *) echo "unknown argument: $a (stages: ${DEFAULT_STAGES[*]} score; flags: --dry-run --list)" >&2; exit 2 ;;
  esac
done
[ "${#stages[@]}" -gt 0 ] || stages=("${DEFAULT_STAGES[@]}")

[ "$DRY" = 1 ] || : "${RUNPOD_API_KEY:?RUNPOD_API_KEY missing (put it in .env)}"
for s in "${stages[@]}"; do "stage_$s"; done
say "done: ${stages[*]}"
