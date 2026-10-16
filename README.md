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
  ga_search.py          PyGAD search over the scratch CNN + random-search control
  check_overlap.py      exact / perceptual-hash overlap test between two image folders
  final_analysis.py     McNemar test, GA model on own photos, matched GA vs random-search comparison
  make_figures.py       report figures 1-5 -> results/figures/
  eval_own_photos.py    tests the baseline on the group's own photos (transfer + 5-fold fine-tuned)
notebooks/
  01_baseline_colab.ipynb  runs the baseline pipeline end to end on Google Colab
  02_ga_search_and_final_model.ipynb  GA search, final model training, and new-photo predictions (Colab)
results/                metrics JSON, training logs, saved models, GA + random-search records, figures
  final_optimized_cnn.keras, final_test_metrics.json   GA-selected model and its test metrics
  ga_search_recovered.json, random_search_partial_28.json   every search evaluation
  new_photo_predictions.json/.png   three-photo demonstration
  demo_photos/   the three demonstration photos
  final/final_analysis.json, figures/   final-report analyses and figures
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
| Baseline CNN on TrashNet (3 seeds) | done: 78.5% accuracy, 0.763 macro-F1 (`results/baseline_metrics.json`) |
| Own photo set | 84 photos (31 plastic / 24 paper-cardboard / 29 aluminum) |
| Baseline on own photos | done (`results/own/own_photo_metrics.json`) |
| GA search | done: 67 evaluations; selected model 81.14% accuracy, 0.790 macro-F1 (`results/final_test_metrics.json`) |
| Random-search control | partial: 28 of 67 evaluations (`results/random_search_partial_28.json`) |
| Final analyses + figures | done (`results/final/`, `results/figures/`) |
| Three-photo demonstration | done: 1 of 3 correct (`results/new_photo_predictions.json`) |

## Contribution rule
Every member commits under their own GitHub account. Branch per task (`data/…`, `model/…`, `docs/…`), pull request, one reviewer.
