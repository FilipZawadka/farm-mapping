# Running the experiment tooling from Claude Code web

The web environment is a fresh cloud container with a clone of this repo. Three
things that exist on the laptop do **not** exist there, and each breaks a
different part of the workflow:

| Missing | Breaks | Fix |
|---|---|---|
| `.env` (gitignored) | everything touching RunPod | set secrets in the web UI, §1 |
| `~/.ssh/id_ed25519` | talking to pods (launch, collect, drive) | §2 |
| `data/` + `experiments/gpu_results/` (gitignored, ~400 MB) | `evaluate_r4.py`, publishing | §4 |

One command does all of it once the secrets are set:

```bash
bash scripts/bootstrap_cloud.sh            # deps, .env, ssh key, connectivity check
bash scripts/bootstrap_cloud.sh --pull-data  # …and fetch the parquets from a live pod
```

It is idempotent and prints a per-item ok/MISSING report, so it doubles as a
diagnostic when something stops working.

---

## 1. Secrets

In the Claude Code web UI, add these as environment variables for the project.
`bootstrap_cloud.sh` materialises `.env` from them on every run.

| Variable | Needed for | Notes |
|---|---|---|
| `RUNPOD_API_KEY` | all pod operations | RunPod → Settings → API Keys |
| `RUNPOD_NETWORK_VOLUME_ID` | launching runs | currently `r8nyom4e4e` (EU-RO-1) |
| `RUNPOD_SSH_PRIVATE_KEY` | reaching pods | see §2 — the big one |
| `CARTO_BASEMAP_API_KEY` | building the web app | basemap key, not a platform token |
| `CLOUDFLARE_API_TOKEN` | Access allowlist admin | optional |
| `GEE_SERVICE_ACCOUNT`, `GEE_PRIVATE_KEY_JSON` | candidate/patch extraction only | optional; the JSON is written to `secrets/gee_key.json` |

Nothing else in `.env` is required: `POD_HOST`/`POD_PORT` are stale
conveniences, and `GOOGLE_MAPS_API_KEY` is unused (imagery previews moved to
Esri tiles).

## 2. SSH access to pods — the part that actually blocks you

`launch_fleet.py`, `collect_results.py` and the on-pod drivers all shell out to
plain `ssh root@<ip> -p <port>`. RunPod injects whatever public keys are on the
account into every pod, so the container needs the matching **private** key.

**Recommended: generate a dedicated key for the cloud environment** rather than
copying your personal one.

```bash
ssh-keygen -t ed25519 -f /tmp/runpod_cloud -N "" -C "claude-code-web"
cat /tmp/runpod_cloud.pub   # → paste into RunPod → Settings → SSH Public Keys
base64 -w0 /tmp/runpod_cloud # → paste as the RUNPOD_SSH_PRIVATE_KEY secret
```

The bootstrap script accepts either raw PEM text or base64, installs it at
`~/.ssh/id_ed25519` with `600`, and prints the derived public key so you can
confirm it matches what RunPod has. Pods created **before** the key was added
won't accept it — only new pods pick up account keys at creation.

### Why the API key alone is not enough

The RunPod API key authenticates the *control plane* — create, list, terminate,
read balance. It gives no way to run a command inside a pod. As currently coded
the GPU path needs SSH twice:

- **starting work**: `_ssh_run_startup()` SSHes the startup script in and runs it
  under tmux;
- **getting results out**: `collect_results.py` streams tar over SSH (the image
  has no rsync).

So with `RUNPOD_API_KEY` only, you can launch, watch and terminate pods — but
they will sit idle, and nothing comes back.

**This is a design choice, not a platform limit.** The CPU prep path already
starts work without SSH by passing the script at creation time
(`_DOCKER_ARGS = "/bin/bash -lc 'eval $STARTUP_SCRIPT'"` plus the script in the
`STARTUP_SCRIPT` env var, `runpod_launch.py:32,404`). Porting that to
`_build_create_kwargs` would make **launching** API-only and would also remove
the fragile staging step that once stalled 20 minutes with a pod idle. Result
collection would still need SSH, unless results are left on the network volume
for a later pod to serve.

## 3. Network egress — verify before trusting it

The cloud sandbox may restrict outbound traffic. Two different things must work,
and the second is the one likely to be blocked:

```bash
# a) HTTPS to the RunPod API — bootstrap step 4 already checks this
curl -s -o /dev/null -w '%{http_code}\n' https://api.runpod.io/graphql   # expect 400

# b) SSH to an arbitrary host on a high port (how pods are reached)
python3 experiments/launch_fleet.py --status     # get a pod's ip:port, then:
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -p <port> root@<ip> 'echo reachable'
```

If (a) works but (b) does not, the environment can still **launch, monitor
and terminate** pods through the API, but cannot stage code, drive on-pod
loops, or collect results. In that case run experiments from the laptop and use
the web environment for analysis and writing only.

## 4. Data

`data/` and `experiments/gpu_results/` are gitignored, so a fresh checkout has
neither. Two ways to get them:

- **From a live pod** (canonical copies live on the network volume):
  `bash scripts/bootstrap_cloud.sh --pull-data` — pulls `all_clusters_v10.parquet`
  (69 MB) by scp and the scored parquets via `collect_results.py`. Needs one
  RUNNING pod and a working §2.
- **Upload** the two paths directly if no pod is running.

`evaluate_r4.py` needs `all_clusters_v10.parquet` plus at least one arm's
scored parquets; `publish_r4.py` needs the `*_score` parquets (~337 MB for all 18).

## 5. What works where

| Task | Web env | Notes |
|---|---|---|
| Read code, write docs/paper | ✅ | no setup needed |
| Re-run evaluation, per-country analysis | ✅ | needs §1 + §4 |
| Launch / monitor / terminate pods | ✅ | needs `RUNPOD_API_KEY` (API only) |
| Stage code to pods, drive on-pod runs, collect | ⚠️ | needs §2 **and** SSH egress (§3b) |
| Publish datasets + push web repo | ✅ | needs §4 and git credentials |
| Patch extraction (Earth Engine) | ⚠️ | needs GEE secrets and EE quota |

## 6. Long-running work

Training campaigns run for hours. The fleet is designed to survive the
controller going away: pods are created via the API and keep running, the
launcher reaps finished pods, and `collect_results.py --watch` re-pulls
anything it missed. If the web session ends mid-campaign, reconnect later and
run `python3 experiments/launch_fleet.py --status` plus
`python3 experiments/collect_results.py --names-file …` to catch up — no state
is lost, because it all lives on the network volume and in
`experiments/results/*_state.json`.

Do not start a second launcher against the same state file while one is
running; both will try to fill the same concurrency slots.
