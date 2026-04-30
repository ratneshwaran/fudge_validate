# fudge-validate

Validation of **FuDGE** — a discrimination metric for dialogue flows — on
labelled (STAR) and unlabelled (Thousand Voices of Trauma) datasets, plus an
LLM-based intent-labeling pipeline that lets the metric work without gold
annotations.

The formal write-up lives in `progress_summary.tex` and `VALIDATION_REPORT.md`.
`PROGRESS.md` tracks day-to-day status.

## Layout

```
src/fudge/             FuDGE implementation
scripts/               LLM labeling pipeline
experiments/           Discrimination + significance experiments
tests/                 Pytest suite (smoke + oracle tests)
data/                  Datasets (gitignored, populated by setup_data.sh)
```

## Setup

Requires Python ≥ 3.10.

```bash
pip install -e .
cp .env.example .env  # then fill in OPENAI_API_KEY and HF_TOKEN
bash setup_data.sh    # downloads both datasets into ./data
```

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

- `OPENAI_API_KEY` — for `scripts/llm_label_star.py`.
- `HF_TOKEN` — for the gated thousand-voices-trauma download.

## Running

```bash
pytest                                      # full test suite
python -m experiments.significance          # discrimination + bootstrap CI
python scripts/llm_label_star.py --help     # LLM labeling options
```
