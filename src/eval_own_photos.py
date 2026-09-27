"""Evaluate the baseline on the Smart Sort Original Set (the group's own photos).

Test A (transfer): the baseline CNN trained on TrashNet (3 seeds, identical CFG) is tested on every own photo.
Test B (fine-tuned): 5-fold stratified CV on the own photos. For each fold, the TrashNet-trained model of each seed
is fine-tuned on the 4 training folds and tested on the held-out fold; every photo is predicted exactly once per seed.
Non-neural floors (majority class, HSV-histogram logistic regression) are evaluated the same way.

Usage:
    python src/eval_own_photos.py --manifest data/manifest.csv --own data/original --out results/own
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import train_baseline as tb
from baselines_classical import hsv_hist

CL = tb.CLASSES


def metrics(yt, yp, probs=None):
    out = dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
               per_class_recall=dict(zip(CL, recall_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0).tolist())),
               per_class_precision=dict(zip(CL, precision_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0).tolist())),
               confusion_matrix=confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist())
    if probs is not None:
        keep = probs.max(1) >= 0.60
        out["abstain_rate_at_0_60"] = float(1 - keep.mean())
        out["accuracy_on_accepted"] = float((yp[keep] == yt[keep]).mean()) if keep.any() else None
    return out


def summarize(runs, key):
    v = [r[key] for r in runs]
    return float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--own", default="data/original")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 2026])
    ap.add_argument("--ft_epochs", type=int, default=10)
    ap.add_argument("--out", default="results/own")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    tn = pd.read_csv(args.manifest)
    Xtr, ytr = tb.load(tn[tn.split == "train"].path.tolist()), tn[tn.split == "train"].y.values
    Xva, yva = tb.load(tn[tn.split == "val"].path.tolist()), tn[tn.split == "val"].y.values
    cw = {i: float(w) for i, w in enumerate(compute_class_weight("balanced", classes=np.arange(3), y=ytr))}

    own = [(os.path.join(args.own, c, f), CL.index(c)) for c in CL for f in sorted(os.listdir(os.path.join(args.own, c)))]
    Xo, yo = tb.load([p for p, _ in own]), np.array([y for _, y in own])
    names = [os.path.basename(p) for p, _ in own]
    print("own photos:", len(yo), np.bincount(yo), flush=True)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(Xo, yo))

    res = {"n_own": int(len(yo)), "own_class_counts": dict(zip(CL, np.bincount(yo).tolist()))}

    # floors
    Htn = np.stack([hsv_hist(p) for p in tn[tn.split == "train"].path])
    Ho = np.stack([hsv_hist(p) for p, _ in own])
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5)).fit(Htn, ytr)
    res["A_majority"] = metrics(yo, np.full_like(yo, np.bincount(ytr).argmax()))
    res["A_hsv_logreg"] = metrics(yo, lr.predict(Ho))
    oof = np.zeros_like(yo)
    for trn, te in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        m.fit(np.vstack([Htn, Ho[trn]]), np.concatenate([ytr, yo[trn]]))
        oof[te] = m.predict(Ho[te])
    res["B_hsv_logreg"] = metrics(yo, oof)
    res["B_majority"] = metrics(yo, np.concatenate([np.full(len(te), np.bincount(yo[trn]).argmax()) for trn, te in folds])[np.argsort(np.concatenate([te for _, te in folds]))])

    runsA, runsB, per_image = [], [], {}
    for seed in args.seeds:
        m = tb.build(seed)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=tb.CFG["patience"], restore_best_weights=True)
        m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=tb.CFG["max_epochs"], batch_size=tb.CFG["batch"],
              class_weight=cw, callbacks=[es], verbose=0)
        w0 = m.get_weights()
        pA = m.predict(Xo, verbose=0)
        rA = metrics(yo, pA.argmax(1), pA); rA["seed"] = seed; runsA.append(rA)
        print("A seed", seed, round(rA["accuracy"], 3), round(rA["macro_f1"], 3), flush=True)
        pB = np.zeros_like(pA)
        for trn, te in folds:
            m.set_weights(w0)
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            ocw = {i: float(w) for i, w in enumerate(compute_class_weight("balanced", classes=np.arange(3), y=yo[trn]))}
            m.fit(Xo[trn], yo[trn], epochs=args.ft_epochs, batch_size=16, class_weight=ocw, verbose=0)
            pB[te] = m.predict(Xo[te], verbose=0)
        rB = metrics(yo, pB.argmax(1), pB); rB["seed"] = seed; runsB.append(rB)
        print("B seed", seed, round(rB["accuracy"], 3), round(rB["macro_f1"], 3), flush=True)
        per_image[seed] = {"A": pA.argmax(1).tolist(), "B": pB.argmax(1).tolist()}

    for tag, runs in (("A_cnn", runsA), ("B_cnn", runsB)):
        res[tag] = {"runs": runs}
        for k in ("accuracy", "macro_f1"):
            res[tag][k + "_mean"], res[tag][k + "_sd"] = summarize(runs, k)
        for part in ("per_class_recall", "per_class_precision"):
            res[tag][part + "_mean"] = {c: float(np.mean([r[part][c] for r in runs])) for c in CL}
        res[tag]["abstain_mean"] = float(np.mean([r["abstain_rate_at_0_60"] for r in runs]))
        res[tag]["accepted_acc_mean"] = float(np.mean([r["accuracy_on_accepted"] for r in runs if r["accuracy_on_accepted"] is not None]))
    res["per_image"] = {"files": names, "true": yo.tolist(), "pred_by_seed": per_image}
    with open(os.path.join(args.out, "own_photo_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("DONE", {k: (round(res[k]["accuracy_mean"], 3), round(res[k]["macro_f1_mean"], 3)) for k in ("A_cnn", "B_cnn")})


if __name__ == "__main__":
    main()
