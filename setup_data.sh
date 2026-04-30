#!/usr/bin/env bash
# Download the datasets used by this repo into ./data/.
#
# Datasets:
#   - STAR (RasaHQ/STAR on GitHub) -> data/STAR
#   - thousand-voices-trauma (yenopoya, gated HF dataset) -> data/thousand-voices-trauma
#
# The thousand-voices-trauma dataset is GATED. You must:
#   1. Visit https://huggingface.co/datasets/yenopoya/thousand-voices-trauma
#      while logged in and accept the access terms (auto-approved).
#   2. Create a token at https://huggingface.co/settings/tokens (read scope).
#   3. Export it before running this script:
#        export HF_TOKEN=hf_xxx
#      or add HF_TOKEN=hf_xxx to .env (auto-loaded below).
#
# This script is idempotent: it skips datasets that are already populated.
#
# Usage:
#   bash setup_data.sh             # both datasets
#   bash setup_data.sh star        # STAR only
#   bash setup_data.sh tvot        # thousand-voices-trauma only

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
ENV_FILE="${SCRIPT_DIR}/.env"

mkdir -p "${DATA_DIR}"

# -----------------------------------------------------------------------------
# STAR (Rasa) — public GitHub repo, plain git clone.
# -----------------------------------------------------------------------------
STAR_DIR="${DATA_DIR}/STAR"
STAR_REPO="https://github.com/RasaHQ/STAR.git"

setup_star() {
  # Idempotency: a successful clone leaves a populated dialogues/ folder.
  if [ -d "${STAR_DIR}/dialogues" ] && \
     [ -n "$(ls -A "${STAR_DIR}/dialogues" 2>/dev/null || true)" ]; then
    echo "[STAR] already present at ${STAR_DIR}; skipping."
    return 0
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "[STAR] error: git is required." >&2
    return 1
  fi

  # Clean any partial dir so git clone can create it fresh.
  if [ -d "${STAR_DIR}" ] && [ -z "$(ls -A "${STAR_DIR}" 2>/dev/null || true)" ]; then
    rmdir "${STAR_DIR}" 2>/dev/null || true
  fi

  echo "[STAR] cloning ${STAR_REPO} -> ${STAR_DIR} ..."
  git clone --depth 1 "${STAR_REPO}" "${STAR_DIR}"
  echo "[STAR] done."
}

# -----------------------------------------------------------------------------
# thousand-voices-trauma — gated HF dataset.
# Strategy: download README + Scores/* + ThousandVoicesOfTrauma.zip (~12 files)
# and unpack the zip locally. Fetching the ~6000 loose conversation JSONs
# individually trips HF's 5000-requests-per-5-minutes rate limit.
# -----------------------------------------------------------------------------
TVOT_REPO_ID="yenopoya/thousand-voices-trauma"
TVOT_DIR="${DATA_DIR}/thousand-voices-trauma"

load_hf_token_from_env() {
  [ -f "${ENV_FILE}" ] || return 0
  for key in HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
    if [ -z "${!key:-}" ]; then
      local val
      val="$( { grep -E "^${key}=" "${ENV_FILE}" || true; } | tail -n1 | cut -d= -f2- | tr -d '\r' | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
      if [ -n "${val}" ]; then export "${key}=${val}"; fi
    fi
  done
}

setup_tvot() {
  # Idempotency: a successful extract leaves >100 conversation JSONs.
  if [ -d "${TVOT_DIR}/ThousandVoicesOfTrauma/conversations" ] && \
     [ "$(find "${TVOT_DIR}/ThousandVoicesOfTrauma/conversations" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')" -gt 100 ]; then
    echo "[TVoT] already extracted at ${TVOT_DIR}/ThousandVoicesOfTrauma; skipping."
    return 0
  fi

  load_hf_token_from_env
  local token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
  if [ -z "${token}" ]; then
    echo "[TVoT] error: HF_TOKEN is not set." >&2
    echo "  This dataset is gated. Get a token at https://huggingface.co/settings/tokens" >&2
    echo "  after accepting terms at https://huggingface.co/datasets/${TVOT_REPO_ID}," >&2
    echo "  then run:  export HF_TOKEN=hf_xxx  (or add it to .env)" >&2
    return 1
  fi

  local py_bin=""
  for candidate in python python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      if "${candidate}" -c "import huggingface_hub" >/dev/null 2>&1; then
        py_bin="${candidate}"; break
      fi
    fi
  done
  if [ -z "${py_bin}" ]; then
    echo "[TVoT] error: no python with huggingface_hub installed." >&2
    echo "  Install with:  pip install -U huggingface_hub" >&2
    return 1
  fi

  mkdir -p "${TVOT_DIR}"
  echo "[TVoT] downloading ${TVOT_REPO_ID} (zip + scores) via huggingface_hub ..."
  HF_TOKEN="${token}" "${py_bin}" - "${TVOT_REPO_ID}" "${TVOT_DIR}" <<'PY'
import os, sys, zipfile, pathlib
from huggingface_hub import snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

repo_id, target = sys.argv[1], sys.argv[2]
target = pathlib.Path(target)

try:
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target),
        token=os.environ.get("HF_TOKEN") or None,
        allow_patterns=["README.md", "Scores/*", "ThousandVoicesOfTrauma.zip"],
        max_workers=2,
    )
except GatedRepoError:
    print(
        f"\nGated repo error: your HF_TOKEN does not have access to {repo_id}.\n"
        f"Accept terms at https://huggingface.co/datasets/{repo_id} (logged in as "
        f"the same account that owns the token), then re-run.\n",
        file=sys.stderr,
    )
    sys.exit(2)
except RepositoryNotFoundError as e:
    print(f"\nRepo not found or token invalid: {e}\n", file=sys.stderr)
    sys.exit(3)

zip_path = target / "ThousandVoicesOfTrauma.zip"
if not zip_path.exists():
    print(f"Error: {zip_path} did not download.", file=sys.stderr)
    sys.exit(4)

extracted = target / "ThousandVoicesOfTrauma"
conv_dir = extracted / "conversations"
already = conv_dir.exists() and len(list(conv_dir.glob("*.json"))) > 100
if already:
    print(f"Already extracted at {extracted}; skipping unzip.")
    sys.exit(0)

print(f"Extracting {zip_path.name} ...")
with zipfile.ZipFile(zip_path) as z:
    members = [m for m in z.namelist()
               if not m.startswith("__MACOSX/") and not m.endswith(".DS_Store")]
    for m in members:
        z.extract(m, path=target)

conv_count = len(list(conv_dir.glob("*.json"))) if conv_dir.exists() else 0
meta_dir = extracted / "metadata"
meta_count = len(list(meta_dir.glob("*.json"))) if meta_dir.exists() else 0
print(f"Extracted {conv_count} conversations and {meta_count} metadata files to {extracted}")
PY
  echo "[TVoT] done."
}

# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
target="${1:-all}"
case "${target}" in
  all)  setup_star; setup_tvot ;;
  star) setup_star ;;
  tvot|thousand-voices-trauma) setup_tvot ;;
  *)
    echo "Usage: bash setup_data.sh [all|star|tvot]" >&2
    exit 64
    ;;
esac
