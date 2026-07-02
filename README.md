# fudge-validate

Validation of **FuDGE** — a discrimination metric for dialogue flows — on
labelled (STAR) and unlabelled (Thousand Voices of Trauma) datasets, plus an
LLM-based intent-labeling pipeline that lets the metric work without gold
annotations.

The formal write-up lives in `progress_summary.tex` and `VALIDATION_REPORT.md`.
`archive/PROGRESS.md` tracks day-to-day status.

## Layout

```
src/fudge/             FuDGE implementation
scripts/               LLM labeling pipeline
experiments/           Discrimination + significance experiments
tests/                 Pytest suite (smoke + oracle tests)
data/                  Datasets (gitignored, populated by setup_data.sh)
```

## Setup

Requires Python ≥ 3.10 (developed with a `.venv` on 3.12).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env  # then fill in the keys (see below)
PATH="$PWD/.venv/bin:$PATH" bash setup_data.sh   # downloads datasets into ./data
```

API keys (all read from `.env`): `OPENAI_API_KEY` (labelling), `HF_TOKEN` (gated TV dataset
download), `OPENROUTER_API_KEY` (DAG generation + LLM judge).

Result JSONs in `experiments/` are tied to specific data generations — see `PROVENANCE.md`
before comparing numbers.

`setup_data.sh` is idempotent — re-running skips datasets already present.
Use `bash setup_data.sh star` or `bash setup_data.sh tvot` to fetch one.

### Datasets

- **STAR** (`data/STAR/`) — public, cloned from
  [RasaHQ/STAR](https://github.com/RasaHQ/STAR).
- **Thousand Voices of Trauma**
  (`data/thousand-voices-trauma/ThousandVoicesOfTrauma/`) — gated HF dataset
  ([yenopoya/thousand-voices-trauma](https://huggingface.co/datasets/yenopoya/thousand-voices-trauma)).
  Accept the access terms on the dataset page, then put a read token in `.env`
  as `HF_TOKEN=hf_...` before running the script.

### Environment

`.env` is auto-loaded by both the LLM pipeline and `setup_data.sh`. Required
keys:

- `OPENAI_API_KEY` — for `scripts/llm_label_star.py` / `scripts/llm_label_tv.py`.
- `HF_TOKEN` — for the gated thousand-voices-trauma download.
- `OPENROUTER_API_KEY` — for `scripts/generate_llm_dags.py` and `scripts/llm_judge.py`.

## Running

```bash
pytest                                      # full test suite
python -m experiments.significance          # discrimination + bootstrap CI
python scripts/llm_label_star.py --help     # LLM labeling options
```
