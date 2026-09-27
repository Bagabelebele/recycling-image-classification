# Data access

Images are **not** committed to this repository (size, and our original photos stay inside the team Drive).

## 1. TrashNet (public proxy used for the Stage 1 baseline)
1. Download `data/dataset-resized.zip` from https://github.com/garythung/trashnet (2,527 images, 512x384, 6 classes).
2. Unzip to `data/raw/` so the class folders sit at `data/raw/dataset-resized/<class>/`.
3. Only `plastic`, `paper`, `cardboard`, `metal` are used (1,889 images). `paper` + `cardboard` -> `paper_cardboard`; `metal` -> `aluminum`. `glass` and `trash` are dropped.

## 2. Smart Sort Original Set (our photos, Stage 2)
1. Team Drive folder: `Smart Sort / original_set /` (request access from the Data Leader).
2. Download into `data/original/` keeping the class folders: `plastic/`, `paper_cardboard/`, `aluminum/`.
3. File naming: `class_member_sequence.jpg`, e.g. `aluminum_tdavis_007.jpg`. No faces, documents, or private property.
4. Run `python src/prepare_data.py --source data/original --out data/manifest_original.csv`.

## 3. Kaggle "Garbage Classification" (asdasdasasdas)
**Not used as a separate source.** Its class counts are each exactly 10 lower than TrashNet's (2,467 vs 2,527) with the
same six classes, consistent with a re-upload of TrashNet. To confirm, download it to `data/kaggle/` and run
`python src/check_overlap.py --a data/raw/dataset-resized --b data/kaggle`.
