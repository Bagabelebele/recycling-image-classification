# Smart Sort — GA-Optimized Scratch CNN for Recycling Stream Classification
AIM 350 — Programming for AI · Term Group Project (Topic 3)

Team: Toni Davis (Project Leader) · Yahiya Bagayoko (Data Leader) · Elsa Baires (AI/Programming Leader) · Tara Dickerson (Research Leader)

Three-class image classifier (`plastic`, `paper_cardboard`, `aluminum`). A genetic algorithm (PyGAD) searches the
architecture and hyperparameters of a **scratch CNN**, and is compared against a hand-specified baseline CNN and an
equal-budget random-search control. All three share one split, input size, augmentation and epoch budget.

## Repository layout
```
README.md             this file
DATA_ACCESS.md        how to obtain TrashNet and the team's original photos (images are not committed)
requirements.txt      pinned Python dependencies
src/
  prepare_data.py       class mapping, corrupt-file check, near-duplicate grouping, stratified 70/15/15 split
  train_baseline.py     baseline scratch CNN (3 seeds) -> results/baseline_metrics.json
  baselines_classical.py majority-class and HSV-histogram logistic-regression floors
  ga_search.py          PyGAD search over the scratch CNN + random-search control (not yet run)
  check_overlap.py      exact / perceptual-hash overlap test between two image folders
  eval_own_photos.py    tests the baseline on the group's own photos (transfer + 5-fold fine-tuned)
notebooks/
  01_baseline_colab.ipynb  runs the pipeline end to end on Google Colab
results/                metrics JSON, training logs, figures
```

## Quick start
```bash
pip install -r requirements.txt
python src/prepare_data.py --source data/raw/dataset-resized --out data/manifest.csv
python src/baselines_classical.py --manifest data/manifest.csv --out results
python src/train_baseline.py --manifest data/manifest.csv --seeds 42 7 2026 --out results
```

## Status
| Milestone | Status |
|---|---|
| Stage 1 baseline on TrashNet-3 proxy | done — see `results/baseline_metrics.json` |
| Own photo set | 84 collected (31 plastic / 24 paper-cardboard / 29 aluminum); target 160 |
| Baseline on own photos | done — see `results/own/own_photo_metrics.json` (`src/eval_own_photos.py`) |
| GA search + random-search control | code ready, run scheduled |
| Live demo (Streamlit, offline) | not started |

## Contribution rule
Every member commits under their own GitHub account. Branch per task (`data/…`, `model/…`, `docs/…`), pull request, one reviewer.
