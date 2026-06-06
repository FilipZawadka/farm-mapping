#!/usr/bin/env bash
# One-time setup for the remote GPU box (e.g. RTX 3080 desktop).
#
# Run THIS SCRIPT ON THE TARGET BOX (the 3080 machine), not your laptop.
# It will:
#   1. Verify CUDA is available
#   2. Clone the repo if missing
#   3. Create a Python venv with training dependencies
#   4. Verify torch can see the GPU
#   5. Set up .env from .env.example (you fill in secrets after)
#   6. Print the values to add to .env on YOUR LAPTOP
#
# Prereqs on the box:
#   - Ubuntu 22.04+ (or compatible)
#   - NVIDIA driver + CUDA already installed (verify with `nvidia-smi`)
#   - Python 3.11+ (`python3 --version`)
#   - git, openssh-server, tmux
#   - Tailscale OR a stable LAN IP reachable from your laptop
#
# Optional (recommended): install Tailscale before running this:
#   curl -fsSL https://tailscale.com/install.sh | sh
#   sudo tailscale up
#   # Then `tailscale status` shows the device's stable hostname.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/FilipZawadka/farm-mapping.git}"
TARGET_DIR="${TARGET_DIR:-$HOME/farm-mapping}"
VENV_DIR="${VENV_DIR:-$TARGET_DIR/.venv}"

echo "=== 1. CUDA check ==="
if ! command -v nvidia-smi >/dev/null; then
  echo "ERROR: nvidia-smi not found. Install NVIDIA driver first." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo
echo "=== 2. System prereqs ==="
for pkg in git tmux python3-venv python3-pip; do
  if ! dpkg -s "$pkg" >/dev/null 2>&1; then
    echo "Installing $pkg ..."
    sudo apt-get update -qq && sudo apt-get install -y -qq "$pkg"
  fi
done

echo
echo "=== 3. Clone repo to $TARGET_DIR ==="
if [ ! -d "$TARGET_DIR/.git" ]; then
  git clone "$REPO_URL" "$TARGET_DIR"
else
  echo "Repo already exists; pulling latest"
  ( cd "$TARGET_DIR" && git pull --ff-only )
fi

echo
echo "=== 4. Python venv + deps ==="
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$TARGET_DIR/requirements-train.txt"

echo
echo "=== 5. Verify torch sees the GPU ==="
"$VENV_DIR/bin/python" - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f"torch.cuda.is_available(): {ok}")
print(f"torch.__version__: {torch.__version__}")
if ok:
    print(f"device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")
else:
    raise SystemExit(
        "torch can't see CUDA. Check driver/CUDA install. "
        "If torch was installed CPU-only, reinstall with: "
        "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121"
    )
EOF

echo
echo "=== 6. .env scaffold ==="
ENV_FILE="$TARGET_DIR/.env"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<EOF
# Earth Engine (paste service account email + path to JSON key)
GEE_SERVICE_ACCOUNT=
GEE_KEY_FILE=

# Optional: Google Maps Places API
GOOGLE_MAPS_API_KEY=
EOF
  echo "Created $ENV_FILE — fill in GEE_SERVICE_ACCOUNT and GEE_KEY_FILE before training."
else
  echo ".env already present at $ENV_FILE"
fi

echo
echo "============================================================"
echo " ON YOUR LAPTOP add these lines to YOUR .env:"
echo "------------------------------------------------------------"
TS_HOST="$(tailscale status --self --peers=false --json 2>/dev/null \
  | python3 -c 'import json,sys;d=json.load(sys.stdin);print(d["Self"]["DNSName"].rstrip("."))' 2>/dev/null \
  || hostname -I | awk '{print $1}')"
USERNAME="$(whoami)"
echo "LOCAL_HOST=$TS_HOST"
echo "LOCAL_PORT=22"
echo "LOCAL_USER=$USERNAME"
echo "LOCAL_CODE_DIR=$TARGET_DIR"
echo "LOCAL_VENV=$VENV_DIR"
echo "LOCAL_SSH_KEY=~/.ssh/id_ed25519   # path on your LAPTOP"
echo "============================================================"
echo
echo "Then on your laptop test the connection:"
echo "  ssh $USERNAME@$TS_HOST 'echo OK'"
echo "  python -m training.local_launch --list"
echo
echo "And launch a run with:"
echo "  python -m training.local_launch --config configs/rachel_clusters/baseline_v2_multiclass.yaml"
