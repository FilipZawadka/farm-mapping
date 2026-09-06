#!/usr/bin/env bash
# Bootstrap a fresh environment (Claude Code web, a new laptop, a CI box) so the
# experiment tooling works: deps, .env, the SSH key the fleet needs, and the
# gitignored data that evaluation reads.
#
#   bash scripts/bootstrap_cloud.sh              # deps + .env + ssh + verify
#   bash scripts/bootstrap_cloud.sh --pull-data  # also fetch parquets from a live pod
#
# Reads these from the environment (set them as secrets in the web UI):
#   RUNPOD_API_KEY            required for anything touching pods
#   RUNPOD_NETWORK_VOLUME_ID  required to launch runs onto the shared volume
#   RUNPOD_SSH_PRIVATE_KEY    private key whose public half is registered in
#                             RunPod → Settings → SSH Public Keys. Either raw
#                             PEM text or base64 of it. Without this you can
#                             create pods but never talk to them.
#   CARTO_BASEMAP_API_KEY     optional, only for building the web app
#   CLOUDFLARE_API_TOKEN      optional, only for Access allowlist admin
#   GEE_SERVICE_ACCOUNT       optional, only for candidate/patch extraction
#   GEE_PRIVATE_KEY_JSON      optional, ditto (JSON text; written to a file)
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
PULL_DATA=0
[ "${1:-}" = "--pull-data" ] && PULL_DATA=1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()  { printf '   \033[32mok\033[0m  %s\n' "$*"; }
bad() { printf '   \033[31mMISSING\033[0m  %s\n' "$*"; }

say "1/5 Python dependencies"
python3 -m pip install --quiet --upgrade pip >/dev/null 2>&1
if python3 -m pip install --quiet -r requirements-cpu.txt; then
  ok "requirements-cpu.txt installed"
else
  echo "   pip install failed — see output above"; exit 1
fi
python3 - <<'PY' || exit 1
import importlib, sys
missing=[m for m in ("pandas","pyarrow","numpy","sklearn","yaml","runpod","dotenv") if not importlib.util.find_spec(m)]
print("   import check:", "all present" if not missing else f"MISSING {missing}")
sys.exit(1 if missing else 0)
PY

say "2/5 .env"
# Rebuilt from the environment every run; never committed (.env is gitignored).
: > .env
add() { [ -n "${2:-}" ] && { printf '%s=%s\n' "$1" "$2" >> .env; ok "$1"; } || bad "$1"; }
add RUNPOD_API_KEY           "${RUNPOD_API_KEY:-}"
add RUNPOD_NETWORK_VOLUME_ID "${RUNPOD_NETWORK_VOLUME_ID:-}"
add CARTO_BASEMAP_API_KEY    "${CARTO_BASEMAP_API_KEY:-}"
add CLOUDFLARE_API_TOKEN     "${CLOUDFLARE_API_TOKEN:-}"
add GEE_SERVICE_ACCOUNT      "${GEE_SERVICE_ACCOUNT:-}"
if [ -n "${GEE_PRIVATE_KEY_JSON:-}" ]; then
  mkdir -p secrets && printf '%s' "$GEE_PRIVATE_KEY_JSON" > secrets/gee_key.json
  chmod 600 secrets/gee_key.json
  echo "GEE_KEY_FILE=secrets/gee_key.json" >> .env; ok "GEE_KEY_FILE"
fi
chmod 600 .env

say "3/5 SSH key for pod access"
if [ -n "${RUNPOD_SSH_PRIVATE_KEY:-}" ]; then
  mkdir -p ~/.ssh && chmod 700 ~/.ssh
  # accept raw PEM or base64
  if printf '%s' "$RUNPOD_SSH_PRIVATE_KEY" | grep -q "BEGIN .*PRIVATE KEY"; then
    printf '%s\n' "$RUNPOD_SSH_PRIVATE_KEY" > ~/.ssh/id_ed25519
  else
    printf '%s' "$RUNPOD_SSH_PRIVATE_KEY" | base64 -d > ~/.ssh/id_ed25519 2>/dev/null
  fi
  chmod 600 ~/.ssh/id_ed25519
  if ssh-keygen -y -f ~/.ssh/id_ed25519 >/dev/null 2>&1; then
    ok "private key installed ($(ssh-keygen -l -f ~/.ssh/id_ed25519 | awk '{print $1" "$4}'))"
    echo "   public half (must be in RunPod → Settings → SSH Public Keys):"
    echo "     $(ssh-keygen -y -f ~/.ssh/id_ed25519)"
  else
    bad "RUNPOD_SSH_PRIVATE_KEY is set but not a valid key"
  fi
else
  bad "RUNPOD_SSH_PRIVATE_KEY — pods will launch but be unreachable"
fi

say "4/5 Connectivity"
python3 - <<'PY'
import os, sys, json, subprocess
sys.path.insert(0, "experiments"); sys.path.insert(0, ".")
try:
    from training.env_loader import load_dotenv; load_dotenv()
except Exception as e:
    print("   env_loader failed:", e)
key = os.environ.get("RUNPOD_API_KEY")
if not key:
    print("   \033[31mno RUNPOD_API_KEY\033[0m — pod tooling will not work"); sys.exit(0)
q = '{"query":"query { myself { clientBalance pods { id name desiredStatus } } }"}'
try:
    r = subprocess.run(["curl","-s","--max-time","25","-H",f"Authorization: Bearer {key}",
                        "-H","Content-Type: application/json","-X","POST",
                        "https://api.runpod.io/graphql","-d",q],
                       capture_output=True, text=True, timeout=40)
    d = json.loads(r.stdout).get("data",{}).get("myself")
    if d is None:
        print("   \033[31mRunPod API rejected the key or egress is blocked\033[0m:", r.stdout[:180])
    else:
        print(f"   \033[32mok\033[0m  RunPod API reachable — balance ${d['clientBalance']:.2f}, "
              f"{len(d['pods'])} pod(s) running")
        for p in d["pods"]: print("        ", p["name"].split("/")[-1], p["desiredStatus"])
except Exception as e:
    print("   \033[31mcould not reach api.runpod.io\033[0m:", str(e)[:160])
PY

say "5/5 Data (gitignored — not in a fresh checkout)"
V10=data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet
N_SCORED=$(ls -d experiments/gpu_results/*/scored_candidates.parquet 2>/dev/null | wc -l)
[ -f "$V10" ] && ok "$V10" || bad "$V10  (needed by evaluate_r4.py)"
echo "   scored parquets present: $N_SCORED"
if [ "$PULL_DATA" = "1" ]; then
  echo "   pulling from a live pod (requires one RUNNING pod + the SSH key)…"
  python3 - <<'PY'
import sys, subprocess, os
sys.path.insert(0,"experiments"); sys.path.insert(0,".")
from collect_results import _api, live_pod, REMOTE_OUT
ep = live_pod()
if not ep:
    print("   no RUNNING pod with public SSH — start one, then rerun with --pull-data"); sys.exit(0)
host, port = ep
print(f"   via {host}:{port}")
# the label parquet lives on the volume next to the candidates
remote = "/workspace/farm-mapping/data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet"
os.makedirs("data/rachel_geometry_candidates/all_countries", exist_ok=True)
r = subprocess.run(["scp","-o","StrictHostKeyChecking=no","-P",str(port),
                    f"root@{host}:{remote}",
                    "data/rachel_geometry_candidates/all_countries/all_clusters_v10.parquet"],
                   capture_output=True, text=True, timeout=1800)
print("   v10 parquet:", "ok" if r.returncode==0 else f"failed: {r.stderr[:160]}")
PY
  python3 experiments/collect_results.py --names-file experiments/results/score_order.txt 2>&1 | tail -2
fi

say "Done"
echo "Next: python3 experiments/launch_fleet.py --status"
echo "      python3 experiments/evaluate_r4.py        (needs the data above)"
