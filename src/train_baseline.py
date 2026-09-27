"""Hand-specified baseline scratch CNN (the reference point the genetic algorithm must beat).

Every GA candidate is trained with this same data, split, input size, augmentation and epoch budget;
only the genes listed in the proposal (Section 4.2) change.

Usage:
    python src/train_baseline.py --manifest data/manifest.csv --seeds 42 7 2026 --out results
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight

CLASSES = ["plastic", "paper_cardboard", "aluminum"]
IMG = 128
CFG = dict(blocks=[32, 64, 128, 128], kernel=3, dropout=0.3, dense=128, lr=1e-3, batch=32,
           optimizer="adam", bn_momentum=0.9, max_epochs=25, patience=6)
# Manual tuning log (decisions made on VALIDATION metrics only):
#   trial 1: bn_momentum=0.99 (Keras default), max_epochs=15, patience=4 -> val loss erratic, early stop at epoch 5
#   trial 2: bn_momentum=0.9,  max_epochs=25, patience=6  -> adopted as the baseline


def load(paths):
    out = np.zeros((len(paths), IMG, IMG, 3), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
        out[i] = np.asarray(im)
    return out


def build(seed):
    tf.keras.utils.set_random_seed(seed)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(25 / 360),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
    ], name="augment")
    inp = tf.keras.Input((IMG, IMG, 3))
    x = tf.keras.layers.Rescaling(1 / 255.0)(inp)
    x = aug(x)
    for f in CFG["blocks"]:
        x = tf.keras.layers.Conv2D(f, CFG["kernel"], padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=CFG["bn_momentum"])(x)
        x = tf.keras.layers.ReLU()(x)
        x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(CFG["dropout"])(x)
    x = tf.keras.layers.Dense(CFG["dense"], activation="relu")(x)
    x = tf.keras.layers.Dropout(CFG["dropout"])(x)
    out = tf.keras.layers.Dense(len(CLASSES), activation="softmax")(x)
    m = tf.keras.Model(inp, out, name="smart_sort_baseline_cnn")
    m.compile(optimizer=tf.keras.optimizers.Adam(CFG["lr"]),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--out", default="results")
    ap.add_argument("--set", nargs="*", default=[], help="override CFG, e.g. --set max_epochs=15 patience=4 bn_momentum=0.99")
    args = ap.parse_args()
    for kv in args.set:
        k, v = kv.split("=")
        CFG[k] = type(CFG[k])(v)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.manifest)
    data = {s: (load(df[df.split == s].path.tolist()), df[df.split == s].y.values) for s in ["train", "val", "test"]}
    cw = compute_class_weight("balanced", classes=np.arange(3), y=data["train"][1])
    cw = {i: float(w) for i, w in enumerate(cw)}
    print("sizes", {s: len(v[1]) for s, v in data.items()}, "class_weights", cw, flush=True)

    runs = []
    for seed in args.seeds:
        t0 = time.time()
        m = build(seed)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=CFG["patience"], restore_best_weights=True)
        h = m.fit(*data["train"], validation_data=data["val"], epochs=CFG["max_epochs"],
                  batch_size=CFG["batch"], class_weight=cw, callbacks=[es], verbose=2)
        vprobs = m.predict(data["val"][0], verbose=0)
        val_acc = accuracy_score(data["val"][1], vprobs.argmax(1))
        val_f1 = f1_score(data["val"][1], vprobs.argmax(1), average="macro")
        probs = m.predict(data["test"][0], verbose=0)
        yt, yp = data["test"][1], probs.argmax(1)
        conf = probs.max(1)
        keep = conf >= 0.60
        run = dict(
            seed=seed, epochs_run=len(h.history["loss"]),
            best_epoch=int(np.argmin(h.history["val_loss"])) + 1,
            train_seconds=round(time.time() - t0, 1), params=int(m.count_params()),
            val_accuracy=val_acc, val_macro_f1=val_f1,
            test_accuracy=accuracy_score(yt, yp), test_macro_f1=f1_score(yt, yp, average="macro"),
            per_class_recall=dict(zip(CLASSES, recall_score(yt, yp, average=None).tolist())),
            per_class_f1=dict(zip(CLASSES, f1_score(yt, yp, average=None).tolist())),
            confusion_matrix=confusion_matrix(yt, yp).tolist(),
            abstain_rate_at_0_60=float(1 - keep.mean()),
            accuracy_on_accepted=float((yp[keep] == yt[keep]).mean()) if keep.any() else None,
            history={k: [float(v) for v in vals] for k, vals in h.history.items()},
        )
        print(classification_report(yt, yp, target_names=CLASSES, digits=3), flush=True)
        print(json.dumps({k: v for k, v in run.items() if k != "history"}, indent=1), flush=True)
        runs.append(run)
        if seed == args.seeds[0]:
            m.save(os.path.join(args.out, "baseline_cnn_seed%d.keras" % seed))

    summary = dict(config=CFG, img_size=IMG, sizes={s: int(len(v[1])) for s, v in data.items()},
                   class_weights=cw, runs=runs)
    for k in ["val_accuracy", "val_macro_f1", "test_accuracy", "test_macro_f1"]:
        vals = [r[k] for r in runs]
        summary[k + "_mean"], summary[k + "_sd"] = float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    with open(os.path.join(args.out, "baseline_metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print("DONE", summary["test_accuracy_mean"], summary["test_macro_f1_mean"])


if __name__ == "__main__":
    main()
