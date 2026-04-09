#!/bin/bash
# Check training logs on the current RunPod machine.
#
# Configure the pod connection in .env or pass as arguments:
#   ./scripts/check_logs.sh                     # uses POD_HOST/POD_PORT from .env
#   ./scripts/check_logs.sh 213.173.105.69 21634
#
# Usage:
#   ./scripts/check_logs.sh              # show all recent training progress
#   ./scripts/check_logs.sh -f           # follow latest train log (live)
#   ./scripts/check_logs.sh -a           # show all pipeline logs
#   ./scripts/check_logs.sh -e <name>    # show specific experiment (e.g. rachel_baseline)

set -euo pipefail

# Load connection from .env or args
if [ -f .env ]; then
    source .env 2>/dev/null || true
fi

HOST="${1:-${POD_HOST:-}}"
PORT="${2:-${POD_PORT:-}}"
MODE="${3:-summary}"

if [ -z "$HOST" ] || [ -z "$PORT" ]; then
    echo "Usage: $0 <host> <port> [-f|-a|-e <experiment>]"
    echo "   or: set POD_HOST and POD_PORT in .env"
    echo ""
    echo "Current .env values:"
    echo "  POD_HOST=${POD_HOST:-<not set>}"
    echo "  POD_PORT=${POD_PORT:-<not set>}"
    exit 1
fi

# Handle flag args when host/port come from .env
if [[ "$HOST" == -* ]]; then
    MODE="$HOST"
    HOST="${POD_HOST}"
    PORT="${POD_PORT}"
    shift 2>/dev/null || true
fi

SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p $PORT -i $SSH_KEY root@$HOST"

echo "Connecting to root@$HOST:$PORT ..."
echo ""

case "${MODE}" in
    -f|--follow)
        echo "Following latest train log (Ctrl+C to stop)..."
        $SSH 'latest=$(find /workspace/farm-mapping/runs -name "train.log" -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1 | cut -d" " -f2); echo "==> $latest"; tail -f "$latest"'
        ;;
    -a|--all)
        $SSH 'find /workspace/farm-mapping/runs -maxdepth 4 -name "pipeline.log" -exec echo "=== {} ===" \; -exec cat {} \; 2>/dev/null'
        ;;
    -e|--experiment)
        EXP="${4:-${2:-}}"
        if [ -z "$EXP" ]; then
            echo "Usage: $0 -e <experiment_name>"
            exit 1
        fi
        $SSH "echo '=== Pipeline ===' && cat /workspace/farm-mapping/runs/$EXP/latest/pipeline.log 2>/dev/null && echo '=== Train (last 20) ===' && tail -20 /workspace/farm-mapping/runs/$EXP/latest/train.log 2>/dev/null && echo '=== Inspected ===' && cat /workspace/farm-mapping/runs/$EXP/latest/inspected_metrics.json 2>/dev/null"
        ;;
    *)
        # Summary: show latest train log lines for each experiment
        $SSH '
echo "========================================"
echo "TRAINING STATUS — $(date -u +"%Y-%m-%d %H:%M UTC")"
echo "========================================"
echo ""

for run_dir in /workspace/farm-mapping/runs/*/; do
    exp=$(basename "$run_dir")
    latest=$(readlink -f "$run_dir/latest" 2>/dev/null)
    [ -z "$latest" ] && continue

    train_log="$latest/train.log"
    pipeline_log="$latest/pipeline.log"

    [ ! -f "$pipeline_log" ] && continue

    # Check if completed
    if grep -q "Pipeline completed" "$pipeline_log" 2>/dev/null; then
        status="DONE"
    elif grep -q "failed" "$pipeline_log" 2>/dev/null; then
        status="FAILED"
    elif grep -q "Step: train" "$pipeline_log" 2>/dev/null; then
        status="TRAINING"
    else
        status="RUNNING"
    fi

    echo "--- $exp [$status] ---"

    if [ -f "$train_log" ]; then
        last_epoch=$(grep "Epoch" "$train_log" | tail -1)
        test_line=$(grep "Test metrics" "$train_log" | tail -1)
        inspected_line=$(grep "Inspected metrics" "$train_log" | tail -1)
        [ -n "$last_epoch" ] && echo "  $last_epoch"
        [ -n "$test_line" ] && echo "  $test_line"
        [ -n "$inspected_line" ] && echo "  $inspected_line"
    fi
    echo ""
done
'
        ;;
esac
