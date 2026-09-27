"""Build the Smart Sort 3-class manifest and a stratified, near-duplicate-grouped 70/15/15 split.

Stage 1 (pilot): source images are the public TrashNet release, mapped to our three streams.
Stage 2 (final): point --source at the Smart Sort Original Set (class-named folders) and rerun.

Usage:
    python src/prepare_data.py --source data/raw/dataset-resized --out data/manifest.csv
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image, ImageOps

CLASSES = ["plastic", "paper_cardboard", "aluminum"]           # 0, 1, 2
TRASHNET_MAP = {"plastic": "plastic", "paper": "paper_cardboard",
                "cardboard": "paper_cardboard", "metal": "aluminum"}  # glass/trash dropped


def dhash(path, size=8):
    """Difference hash used to group near-duplicate photos so they never straddle a split."""
    img = ImageOps.exif_transpose(Image.open(path)).convert("L").resize((size + 1, size))
    a = np.asarray(img, dtype=np.int16)
    return "".join("1" if b else "0" for b in (a[:, 1:] > a[:, :-1]).flatten())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="data/manifest.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hamming", type=int, default=4, help="max bit distance treated as near-duplicate")
    args = ap.parse_args()

    rows = []
    for folder in sorted(os.listdir(args.source)):
        label = TRASHNET_MAP.get(folder, folder if folder in CLASSES else None)
        if label is None:
            continue
        for fn in sorted(os.listdir(os.path.join(args.source, folder))):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            p = os.path.join(args.source, folder, fn)
            try:
                with Image.open(p) as im:
                    im.verify()
            except Exception:
                print("corrupt, skipped:", p)
                continue
            rows.append({"path": p, "source_folder": folder, "label": label,
                         "y": CLASSES.index(label), "hash": dhash(p)})

    # union-find over near-duplicate hashes -> group ids
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    bits = np.array([[c == "1" for c in r["hash"]] for r in rows])
    for i in range(len(rows)):
        d = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.where(d <= args.hamming)[0] + i + 1:
            if rows[i]["y"] == rows[j]["y"]:
                parent[find(j)] = find(i)
    for i, r in enumerate(rows):
        r["group"] = find(i)

    # stratified group split: shuffle groups within each class, fill 70/15/15 by image count
    rng = random.Random(args.seed)
    by_class = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(rows):
        by_class[r["y"]][r["group"]].append(i)
    for y, groups in by_class.items():
        gids = list(groups)
        rng.shuffle(gids)
        n = sum(len(groups[g]) for g in gids)
        acc = 0
        for g in gids:
            frac = acc / n
            split = "train" if frac < 0.70 else ("val" if frac < 0.85 else "test")
            for i in groups[g]:
                rows[i]["split"] = split
            acc += len(groups[g])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "source_folder", "label", "y", "group", "split"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    n_groups = len({r["group"] for r in rows})
    print(f"images={len(rows)} near-dup groups={n_groups} (merged {len(rows) - n_groups})")
    for s in ["train", "val", "test"]:
        c = Counter(r["label"] for r in rows if r["split"] == s)
        print(s, sum(c.values()), dict(c))


if __name__ == "__main__":
    main()
