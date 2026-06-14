# Getting the data back on a new machine

`data/`, `.llm_cache/`, `logs/`, and `.env` are **gitignored** — they are *not* in this repo.
This note is how to restore a working `data/` directory after a fresh `git clone`.

There are two kinds of missing data, and they are restored differently:

| Kind | Restore by | Re-downloadable? |
|---|---|---|
| **Raw datasets** (STAR, Thousand Voices) | `setup_data.sh` | ✅ yes |
| **Generated / paid artifacts** (splits, LLM labels, LLM DAGs) | **copy from old machine** | ❌ **no** |

---

## 0. First: environment

```bash
git clone https://github.com/ratneshwaran/fudge_validate.git
cd fudge_validate

# Python env (≥3.10). The repo was developed in a conda env named "fudge".
conda create -n fudge python=3.11 -y && conda activate fudge
pip install -e .                 # installs deps from pyproject.toml
pip install -U huggingface_hub   # needed by setup_data.sh (not a runtime dep)

# Secrets — recreate .env (it is gitignored). Template is committed:
cp .env.example .env
# then fill in:
#   OPENROUTER_API_KEY=...   (LLM DAG generation via OpenRouter)
#   OPENAI_API_KEY=...       (STAR/TV labelling pipeline)
#   HF_TOKEN=hf_...          (gated Thousand-Voices download)
```

> Experiments import `fudge.*` and are run with `PYTHONPATH=src`. See `HANDOVER.md` §6 for the
> exact run commands and Windows gotchas.

---

## 1. Raw datasets — re-downloadable (`setup_data.sh`)

These come straight from source, so they only need the script:

```bash
bash setup_data.sh        # both; idempotent (skips what's already present)
# or: bash setup_data.sh star   |   bash setup_data.sh tvot
```

- **STAR** → `data/STAR/` (~125 MB) — public `git clone` of RasaHQ/STAR. No token.
- **Thousand Voices of Trauma** → `data/thousand-voices-trauma/ThousandVoicesOfTrauma/` (~78 MB) —
  **gated** HF dataset. You must (1) accept the terms at
  <https://huggingface.co/datasets/yenopoya/thousand-voices-trauma> while logged in, (2) put a
  read token in `.env` as `HF_TOKEN=hf_...`. The script downloads the zip and unpacks it.

---

## 2. Generated / paid artifacts — NOT re-downloadable ⚠️

These are **not** produced by `setup_data.sh`. They were created by seeded scripts and **paid LLM
API calls**, and the DAGs are **non-deterministic** — regenerating them would cost money *and*
produce different graphs, which would no longer match the committed result JSONs in `experiments/`.
**Copy them from the old machine** (cloud drive / USB / `scp`). Total ≈ **40 MB**.

| Path | Size | What it is | If lost |
|---|---|---|---|
| `data/splits/TV_v1.json`, `data/splits/STAR_v2.json` | 188 KB | **The locked 70/30 splits.** Everything is scored against these. | Comparability breaks — must be byte-identical. |
| `data/TV_llm_labels/` | 6.6 MB | LLM intent labels, TV (P5/P6/P7/P8/P10). | ~$5–10 + hours to relabel; may differ. |
| `data/STAR_llm_labels/` | 2.1 MB | LLM intent labels, STAR. | Re-pay to relabel. |
| `data/dags/<model>/<variant>/<phase>/` | 31 MB | Generated DAGs (`dag.json`), aligned flows (`aligned_r5.json`), coverage, transcripts. | Re-pay to regenerate **and** results no longer reproduce (LLM sampling). |

**One-liner to bundle them on the OLD machine** (everything in `data/` except the two raw datasets):

```bash
# run in the repo root on the OLD laptop
tar czf fudge_generated_data.tgz \
    data/splits data/dags data/TV_llm_labels data/STAR_llm_labels
# move fudge_generated_data.tgz to the new machine (cloud/USB), then in the repo root:
tar xzf fudge_generated_data.tgz
```

After this + step 1, `data/` is fully restored.

---

## 3. Optional but useful — the LLM cache

- `.llm_cache/` (~50 MB) — cached LLM responses keyed on (model, stage, prompt). **Not required**,
  but copying it makes identical re-runs **free** (no re-billing). Skip it and any re-run just
  re-calls the API.

```bash
# optional, on the OLD machine:
tar czf fudge_llm_cache.tgz .llm_cache    # then extract on the new one
```

`logs/` is disposable — don't bother copying.

---

## 4. Sanity check

```bash
PYTHONPATH=src python -c "from fudge.splits import load_split; \
  s=load_split('data/splits/TV_v1.json'); print('split OK:', s['version'])"
pytest -q          # smoke + oracle tests should pass
```

If the split loads and tests pass, you're ready — see `HANDOVER.md` for what to do next.
