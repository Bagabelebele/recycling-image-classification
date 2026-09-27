"""Reference floors for the CNN: majority-class predictor and a colour-histogram logistic regression.

Usage:
    python src/baselines_classical.py --manifest data/manifest.csv --out results
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CLASSES = ["plastic", "paper_cardboard", "aluminum"]


def hsv_hist(path, bins=(8, 4, 4)):
    im = ImageOps.exif_transpose(Image.open(path)).convert("HSV").resize((128, 128))
    h, _ = np.histogramdd(np.asarray(im).reshape(-1, 3), bins=bins, range=((0, 256),) * 3)
    return (h.flatten() / h.sum()).astype(np.float32)


def score(yt, yp):
    return dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
                per_class_recall=dict(zip(CLASSES, recall_score(yt, yp, average=None, zero_division=0).tolist())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    df = pd.read_csv(args.manifest)
    X = np.stack([hsv_hist(p) for p in df.path])
    tr, te = (df.split == "train").values, (df.split == "test").values
    y = df.y.values
    res = {}
    maj = DummyClassifier(strategy="most_frequent").fit(X[tr], y[tr])
    res["majority_class"] = score(y[te], maj.predict(X[te]))
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    lr.fit(X[tr], y[tr])
    res["hsv_histogram_logreg"] = score(y[te], lr.predict(X[te]))
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "classical_baselines.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
