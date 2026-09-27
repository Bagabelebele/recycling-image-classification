"""Genetic-algorithm search over the SCRATCH CNN only (MobileNetV2 is not part of this search).

Fairness protocol (see proposal Section 4.2):
  * same manifest/split, input size (128x128), augmentation, class weights and epoch budget as the baseline
  * fitness = validation macro-F1 minus a small parsimony penalty; the test split is never touched here
  * a random-search control draws the same number of candidates (96) from the same search space

Status: implemented, not yet run (scheduled milestone). Usage:
    python src/ga_search.py --manifest data/manifest.csv --mode ga      --out results/ga
    python src/ga_search.py --manifest data/manifest.csv --mode random  --out results/random
"""
import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import pygad
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import train_baseline as tb  # reuses load(), CLASSES, IMG and the shared training budget

SPACE = {
    "n_blocks": [2, 3, 4, 5],
    "filters": [16, 32, 64, 128],
    "kernel": [3, 5],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "dense": [64, 128, 256],
    "lr": list(np.logspace(-4, -2, 7)),
    "batch": [8, 16, 32],
    "optimizer": ["adam", "rmsprop", "sgd"],
    "aug": [0.5, 1.0, 1.5],  # multiplier on the baseline augmentation magnitudes
}
KEYS = list(SPACE)
POP, GENS, PARSIMONY = 12, 8, 0.02  # penalty per million parameters


def decode(sol):
    return {k: SPACE[k][int(g)] for k, g in zip(KEYS, sol)}


def build(c, seed=42):
    tf.keras.utils.set_random_seed(seed)
    a = c["aug"]
    aug = tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),
                               tf.keras.layers.RandomRotation(a * 25 / 360),
                               tf.keras.layers.RandomZoom(min(0.2 * a, 0.5)),
                               tf.keras.layers.RandomBrightness(min(0.2 * a, 0.5), value_range=(0.0, 1.0))])
    inp = tf.keras.Input((tb.IMG, tb.IMG, 3))
    x = aug(tf.keras.layers.Rescaling(1 / 255.0)(inp))
    for i in range(c["n_blocks"]):
        f = min(c["filters"] * 2 ** i, 256)
        x = tf.keras.layers.Conv2D(f, c["kernel"], padding="same", use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=tb.CFG["bn_momentum"])(x)
        x = tf.keras.layers.MaxPooling2D()(tf.keras.layers.ReLU()(x))
    x = tf.keras.layers.Dropout(c["dropout"])(tf.keras.layers.GlobalAveragePooling2D()(x))
    x = tf.keras.layers.Dropout(c["dropout"])(tf.keras.layers.Dense(c["dense"], activation="relu")(x))
    m = tf.keras.Model(inp, tf.keras.layers.Dense(3, activation="softmax")(x))
    opt = {"adam": tf.keras.optimizers.Adam, "rmsprop": tf.keras.optimizers.RMSprop,
           "sgd": lambda lr: tf.keras.optimizers.SGD(lr, momentum=0.9)}[c["optimizer"]](c["lr"])
    m.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/manifest.csv")
    ap.add_argument("--mode", choices=["ga", "random"], default="ga")
    ap.add_argument("--out", default="results/ga")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(args.manifest)
    Xtr, ytr = tb.load(df[df.split == "train"].path.tolist()), df[df.split == "train"].y.values
    Xva, yva = tb.load(df[df.split == "val"].path.tolist()), df[df.split == "val"].y.values
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.arange(3), y=ytr)))
    log = []

    def fitness(sol):
        c = decode(sol)
        m = build(c)
        es = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=tb.CFG["patience"], restore_best_weights=True)
        m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=tb.CFG["max_epochs"], batch_size=c["batch"],
              class_weight=cw, callbacks=[es], verbose=0)
        f1 = f1_score(yva, m.predict(Xva, verbose=0).argmax(1), average="macro")
        fit = f1 - PARSIMONY * m.count_params() / 1e6
        log.append({**{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in c.items()},
                    "val_macro_f1": f1, "params": int(m.count_params()), "fitness": fit})
        print(len(log), log[-1], flush=True)
        return fit

    if args.mode == "ga":
        ga = pygad.GA(num_generations=GENS - 1, sol_per_pop=POP, num_parents_mating=4, num_genes=len(KEYS),
                      gene_space=[list(range(len(SPACE[k]))) for k in KEYS], gene_type=int,
                      fitness_func=lambda g, s, i: fitness(s), parent_selection_type="tournament", K_tournament=3,
                      crossover_type="single_point", mutation_type="random", mutation_probability=0.15,
                      keep_elitism=2, random_seed=42, save_solutions=False)
        ga.run()
        best, _, _ = ga.best_solution()
        result = {"best": decode(best), "fitness_by_generation": [float(v) for v in ga.best_solutions_fitness]}
    else:
        rng = random.Random(42)
        sols = [[rng.randrange(len(SPACE[k])) for k in KEYS] for _ in range(POP * GENS)]
        best = max(sols, key=fitness)
        result = {"best": decode(best)}
    result["evaluations"] = log
    with open(os.path.join(args.out, f"{args.mode}_search.json"), "w") as f:
        json.dump(result, f, indent=1, default=float)
    print("best:", result["best"])


if __name__ == "__main__":
    main()
