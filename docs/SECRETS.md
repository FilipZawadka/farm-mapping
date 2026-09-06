# Secrets and network access

Everything the tooling needs to authenticate, where each credential comes from,
and exactly which hosts it talks to — so an egress allowlist (Claude Code web,
CI, a locked-down VM) can be written without trial and error.

`.env` is gitignored and is rebuilt from environment variables by
`scripts/bootstrap_cloud.sh`. Nothing here belongs in a commit.

---

## 1. Required — nothing works without these

### `RUNPOD_API_KEY`
- **Get it**: runpod.io → Settings → API Keys → create (read/write).
- **Used by**: `experiments/launch_fleet.py`, `experiments/collect_results.py`,
  `training/runpod_launch.py`, `training/auto_terminate.py`.
- **Talks to**: `api.runpod.io` (HTTPS, GraphQL).
- **Can do**: create pods, terminate pods, read balance — i.e. **it can spend
  money**. Treat as a payment credential, rotate if exposed.
- **Cannot do**: run anything inside a pod. See `RUNPOD_SSH_PRIVATE_KEY`.

### `RUNPOD_NETWORK_VOLUME_ID`
- **Value**: `r8nyom4e4e` (EU-RO-1). Not really a secret, but it lives with the
  others; a wrong value silently launches pods with no data attached.
- **Used by**: pod creation, to mount the shared volume at `/workspace`.
- **Talks to**: nothing directly.

### `RUNPOD_SSH_PRIVATE_KEY`
- **Get it**: generate a **dedicated** keypair, don't reuse a personal one:
  ```bash
  ssh-keygen -t ed25519 -f runpod_cloud -N "" -C "claude-code-web"
  cat runpod_cloud.pub    # → RunPod → Settings → SSH Public Keys
  base64 -w0 runpod_cloud # → this secret's value
  ```
- **Used by**: every `ssh`/`scp`/`tar-over-ssh` call — starting work on a pod
  (`_ssh_run_startup`), driving on-pod loops, collecting results.
- **Talks to**: **pod IPs directly, on high non-standard ports** (e.g.
  `213.173.98.13:16228`). These are ephemeral and cannot be allowlisted by
  hostname — an egress policy must permit outbound TCP to arbitrary
  addresses/ports, or SSH-dependent steps will not work.
- **Note**: only pods created *after* the public key is registered will accept it.

---

## 2. Optional — needed only for specific jobs

### `CARTO_BASEMAP_API_KEY`
- **Get it**: <https://carto.com/basemaps/apikey/> — the free *basemap* key,
  emailed back instantly. **Not** a CARTO platform/account token; a platform
  token is silently ignored and the map keeps its "API KEY REQUIRED" watermark.
- **Used by**: the web build only (`web/vite.config.ts` exposes it,
  `web/src/lib/mapStyle.ts` appends `?key=`). Baked into the bundle at build
  time, so it must also be set in **Railway → web service → Variables**.
- **Talks to**: `basemaps.cartocdn.com` — from the **viewer's browser**, not
  from the sandbox. No egress needed to build.
- **Limits**: 5M tiles/month, non-commercial; restrict it to
  `animalfarmingatlas.org` in CARTO's dashboard.

### `CLOUDFLARE_API_TOKEN`
- **Get it**: dash.cloudflare.com → My Profile → API Tokens. Scope it to
  *Access: Apps and Policies* (edit) for the zone. The current token can read
  Access apps but **cannot** read DNS or mint service tokens.
- **Used by**: managing the site's email allowlist (Zero Trust Access).
- **Talks to**: `api.cloudflare.com`.

### `GEE_SERVICE_ACCOUNT` + `GEE_PRIVATE_KEY_JSON`
- **Get it**: Google Cloud console → service account with Earth Engine access;
  register it at <https://signup.earthengine.google.com/>. `GEE_SERVICE_ACCOUNT`
  is the `…@….iam.gserviceaccount.com` address;`GEE_PRIVATE_KEY_JSON` is the
  key file's JSON text (bootstrap writes it to `secrets/gee_key.json` and sets
  `GEE_KEY_FILE`).
- **Used by**: candidate generation and patch extraction only
  (`training/patch_extraction.py`, `training/imagery/earth_engine_s2.py`).
  **Not** needed for training, evaluation, or publishing — the patch store
  already exists on the volume.
- **Talks to**: `earthengine.googleapis.com`, `oauth2.googleapis.com`,
  `storage.googleapis.com`.
- **Note**: on pods these arrive as RunPod *secrets* (`RUNPOD_SECRET_*`), not
  from `.env` — see `_RUNPOD_SECRETS_ENV` in `runpod_launch.py`.

### Git push access
- **Get it**: an SSH deploy key or a PAT for `github.com`. Both repos use SSH
  remotes (`git@github.com:FilipZawadka/farm-mapping.git` and
  `…/farm-mapping-web.git`).
- **Needed for**: committing work, and **publishing the site** — Railway
  deploys on push to the web repo's `main`.
- **Talks to**: `github.com`.

---

## 3. Not secrets, but the network must allow them

| Host | Why | When |
|---|---|---|
| `pypi.org`, `files.pythonhosted.org` | `pip install -r requirements-cpu.txt` | environment setup |
| `github.com` | git ops; pods also clone the repo from here | setup, every pod launch |
| `api.runpod.io` | all pod control | always |
| arbitrary IPs on high TCP ports | SSH to pods | launching, collecting |
| `registry.npmjs.org` | `npm ci` for the web build | only when building the site |
| `download.pytorch.org` | torch wheels **on pods**, not in the sandbox | pod setup only |
| `overpass-api.de` | OSM building footprints | candidate generation only |

**Browser-only** (the viewer's machine fetches these, never the sandbox):
`basemaps.cartocdn.com`, `server.arcgisonline.com`, `fonts.openmaptiles.org`,
`fonts.googleapis.com`, `www.google.com` (per-point map deep links).

---

## 4. Minimum sets by task

| I want to… | Secrets | Hosts |
|---|---|---|
| Read code, write docs/paper | none | `github.com` |
| Re-run evaluation / analysis | none¹ | `pypi.org`, `github.com` |
| Launch, monitor, terminate pods | `RUNPOD_API_KEY`, `RUNPOD_NETWORK_VOLUME_ID` | + `api.runpod.io` |
| Actually run training end to end | + `RUNPOD_SSH_PRIVATE_KEY` | + outbound TCP to any IP/port |
| Publish datasets to the site | + git push access | + `github.com` |
| Fix the basemap watermark | + `CARTO_BASEMAP_API_KEY` (also in Railway) | build-time only |
| Manage the site allowlist | + `CLOUDFLARE_API_TOKEN` | + `api.cloudflare.com` |
| Extract new imagery patches | + `GEE_SERVICE_ACCOUNT`, `GEE_PRIVATE_KEY_JSON` | + `*.googleapis.com` |

¹ needs the gitignored data present — `bash scripts/bootstrap_cloud.sh --pull-data`,
which itself needs the RunPod secrets and a live pod.

---

## 5. Handling

- The RunPod key is passed in a `curl` argv, so it used to land in error text and
  from there into logs. `_api` now redacts `rpa_…` before building any message;
  `experiments/results/*.log` is gitignored. Verified: no key in git history.
- Two entries in the local `.env` are *optional*, not required (an earlier draft
  of this file wrongly called them dead — both are still referenced):
  - `GOOGLE_MAPS_API_KEY` is still injected into pods via `_RUNPOD_SECRETS_ENV`
    (`runpod_launch.py:359`), but the local value is empty and imagery previews
    now use Esri tiles, so nothing depends on it in practice.
  - `POD_HOST` / `POD_PORT` are used by `scripts/check_logs.sh` to tail a fixed
    pod's logs. The fleet resolves pod addresses from the API per call, so these
    matter only for that convenience script.
- Rotate `RUNPOD_API_KEY` and the RunPod SSH key independently; that separation
  is the reason for a dedicated cloud keypair.
