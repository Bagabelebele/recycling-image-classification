"""Final-report analyses on top of the saved models (run after train_baseline.py, eval_own_photos.py and the GA).

1. Reproduces the GA-optimized model's test metrics from results/final_optimized_cnn.keras.
2. Paired comparison with the seed-42 baseline on the same 281 TrashNet test images (exact McNemar test).
3. GA model on the group's own photographs: Test A (unchanged) and Test B (5-fold fine-tuning, same protocol
   as the baseline in eval_own_photos.py).
4. Matched-evaluation comparison of the GA and the partial random-search control.

Usage:
    python src/final_analysis.py --out results/final
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import train_baseline as tb

CL = tb.CLASSES


def summary(yt, yp):
    return dict(accuracy=accuracy_score(yt, yp), macro_f1=f1_score(yt, yp, average="macro"),
                per_class_recall=dict(zip(CL, recall_score(yt, yp, average=None, labels=[0, 1, 2], zero_division=0).tolist())),
                confusion_matrix=confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--own", default="data/original")
    ap.add_argument("--ga_model", default="results/final_optimized_cnn.keras")
    ap.add_argument("--baseline_model", default="results/baseline_cnn_seed42.keras")
    ap.add_argument("--ga_json", default="results/ga_search_recovered.json")
    ap.add_argument("--rs_json", default="results/random_search_partial_28.json")
    ap.add_argument("--out", default="results/final")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}

    df = pd.read_csv(args.manifest)
    te = df[df.split == "test"]
    Xte, yte = tb.load(te.path.tolist()), te.y.values
    ga = tf.keras.models.load_model(args.ga_model, compile=False)
    base = tf.keras.models.load_model(args.baseline_model, compile=False)
    pg, pb = ga.predict(Xte, verbose=0).argmax(1), base.predict(Xte, verbose=0).argmax(1)
    res["ga_test"] = summary(yte, pg) | {"params": int(ga.count_params())}
    res["baseline_seed42_test"] = summary(yte, pb) | {"params": int(base.count_params())}
    b = int(((pg == yte) & (pb != yte)).sum())   # GA right, baseline wrong
    c = int(((pg != yte) & (pb == yte)).sum())   # GA wrong, baseline right
    res["mcnemar_ga_vs_baseline_seed42"] = {"ga_only_correct": b, "baseline_only_correct": c,
                                            "exact_p_value": binomtest(b, b + c, 0.5).pvalue if b + c else 1.0}
    print("test", res["ga_test"]["accuracy"], res["baseline_seed42_test"]["accuracy"], res["mcnemar_ga_vs_baseline_seed42"], flush=True)

    # same photo order and folds as eval_own_photos.py, so Test B is directly comparable with the baseline
    own = [(os.path.join(args.own, c_, f), CL.index(c_)) for c_ in CL for f in sorted(os.listdir(os.path.join(args.own, c_)))]
    Xo, yo = tb.load([p for p, _ in own]), np.array([y for _, y in own])
    res["ga_own_A"] = summary(yo, ga.predict(Xo, verbose=0).argmax(1))
    tf.keras.utils.set_random_seed(42)
    w0 = ga.get_weights()
    pB = np.zeros(len(yo), dtype=int)
    for trn, tst in StratifiedKFold(5, shuffle=True, random_state=42).split(Xo, yo):
        ga.set_weights(w0)
        ga.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy")
        cw = {i: float(w) for i, w in enumerate(compute_class_weight("balanced", classes=np.arange(3), y=yo[trn]))}
        ga.fit(Xo[trn], yo[trn], epochs=10, batch_size=16, class_weight=cw, verbose=0)
        pB[tst] = ga.predict(Xo[tst], verbose=0).argmax(1)
    res["ga_own_B"] = summary(yo, pB)
    print("own", res["ga_own_A"]["accuracy"], res["ga_own_B"]["accuracy"], flush=True)

    g, r = json.load(open(args.ga_json)), json.load(open(args.rs_json))
    best = lambda ev: np.maximum.accumulate([e["fitness"] for e in ev]).tolist()  # selection criterion
    res["search_trajectories"] = {"ga_best_so_far": best(g["evaluations"]), "random_best_so_far": best(r["evaluations"])}
    n = len(r["evaluations"])
    res["matched_at_random_count"] = {"n": n, "ga_best": best(g["evaluations"])[n - 1], "random_best": best(r["evaluations"])[n - 1]}
    with open(os.path.join(args.out, "final_analysis.json"), "w") as f:
        json.dump(res, f, indent=1)
    print("DONE", res["matched_at_random_count"])


if __name__ == "__main__":
    main()
