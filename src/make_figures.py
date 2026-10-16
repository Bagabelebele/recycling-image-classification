"""Figures for the final report, generated only from files in results/.

Usage:
    python src/make_figures.py --out results/figures
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a63", "#e4e3dd"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, out, name):
    fig.savefig(os.path.join(out, name), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline(out):
    fig, ax = plt.subplots(figsize=(9.5, 3.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 40)
    boxes = [(1, 24, "Image sources", "TrashNet (1,889 mapped)\nGroup photos (84)"),
             (21, 24, "Preparation", "class mapping, EXIF fix,\nnear-duplicate grouping,\n70/15/15 split"),
             (41, 24, "Input pipeline", "resize 128x128, rescale,\nflip/rotate/zoom/\nbrightness (train only)"),
             (61, 24, "Scratch CNN", "conv-BN-ReLU-pool blocks,\nGAP, dense, softmax (3)"),
             (81, 24, "Evaluation", "accuracy, macro-F1,\nper-class recall, McNemar,\n0.60 abstention"),
             (36, 3, "Genetic algorithm", "PyGAD, 9 genes\nfitness = validation\nmacro-F1 - size penalty"),
             (58, 3, "Random search", "control: same space\nand training rules\n(28 runs completed)")]
    for x, y, t, s in boxes:
        ax.add_patch(FancyBboxPatch((x, y), 18, 14, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#f3f6fb", ec=BLUE, lw=1.2))
        ax.text(x + 9, y + 11.2, t, ha="center", va="center", fontsize=8.6, weight="bold", color=INK)
        ax.text(x + 9, y + 5.2, s, ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.25)
    for x in (19.6, 39.6, 59.6, 79.6):
        ax.annotate("", xy=(x + 1.6, 31), xytext=(x - 0.2, 31), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2))
    for x0 in (45, 67):
        ax.annotate("", xy=(70, 23.4), xytext=(x0, 17.8), arrowprops=dict(arrowstyle="<|-|>", color=ORANGE, lw=1.1))
    ax.text(79, 20.4, "each candidate is trained and\nscored on validation data", fontsize=7, color=ORANGE, ha="left", va="center")
    ax.text(80, 8, "Held-out test data is used\nonly after selection is final.", fontsize=7.4, color=MUTED, ha="left", va="center")
    save(fig, out, "fig1_pipeline.png")


def search(out, fa):
    ga, rs = fa["search_trajectories"]["ga_best_so_far"], fa["search_trajectories"]["random_best_so_far"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.step(range(1, len(ga) + 1), ga, where="post", color=BLUE, lw=2, label="Genetic algorithm (67 evaluations)")
    ax.step(range(1, len(rs) + 1), rs, where="post", color=ORANGE, lw=2, label="Random search (28 of 67 completed)")
    n = len(rs)
    ax.axvline(n, color=MUTED, lw=1, ls=":")
    ax.text(n + 1, min(ga) + 0.012, f"Matched at {n} evaluations:\nGA {ga[n-1]:.3f} vs random {rs[n-1]:.3f}", fontsize=8, color=INK)
    ax.text(len(ga) + 0.6, ga[-1], f"{ga[-1]:.3f}", color=INK, fontsize=8, va="center")
    ax.set_xlabel("Candidate evaluations"); ax.set_ylabel("Best fitness so far\n(validation macro-F1 - size penalty)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out, "fig2_search.png")


def comparison(out, B, C, fa):
    labels = ["Majority\nclass", "Color histogram +\nlogistic regression", "Baseline CNN\n(mean of 3 seeds)", "GA-optimized\nCNN"]
    acc = [C["majority_class"]["accuracy"], C["hsv_histogram_logreg"]["accuracy"], B["test_accuracy_mean"], fa["ga_test"]["accuracy"]]
    f1 = [C["majority_class"]["macro_f1"], C["hsv_histogram_logreg"]["macro_f1"], B["test_macro_f1_mean"], fa["ga_test"]["macro_f1"]]
    err_a = [0, 0, B["test_accuracy_sd"], 0]; err_f = [0, 0, B["test_macro_f1_sd"], 0]
    fig, ax = plt.subplots(figsize=(7, 3.5)); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    x = range(len(labels)); w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], acc, w, color=BLUE, label="Accuracy")
    ax.bar([i + w / 2 + 0.01 for i in x], f1, w, color=ORANGE, label="Macro-F1")
    ax.errorbar([2 - w / 2 - 0.01, 2 + w / 2 + 0.01], [acc[2], f1[2]], yerr=[err_a[2], err_f[2]], fmt="none", ecolor=INK, lw=1, capsize=3)
    for i in x:
        ax.text(i - w / 2, acc[i] + err_a[i] + 0.015, f"{acc[i]:.3f}", ha="center", fontsize=7.5, color=INK)
        ax.text(i + w / 2, f1[i] + err_f[i] + 0.015, f"{f1[i]:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylim(0, 1); ax.set_ylabel("Score (TrashNet test, n = 281)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    save(fig, out, "fig3_comparison.png")


def confusions(out, fa):
    names = ["Plastic", "Paper/\ncardboard", "Aluminum"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    for ax, key, title in zip(axes, ["baseline_seed42_test", "ga_test"], ["Baseline CNN (seed 42)", "GA-optimized CNN"]):
        cm = fa[key]["confusion_matrix"]
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=150)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=10, color="white" if cm[i][j] > 80 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3)); ax.set_xticklabels(names, fontsize=8); ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Predicted"); ax.set_title(title, fontsize=9.5, color=INK)
        for s in ax.spines.values(): s.set_visible(False)
    axes[0].set_ylabel("Actual")
    fig.tight_layout()
    save(fig, out, "fig4_confusion.png")


def generalization(out, B, O, fa):
    groups = ["TrashNet test\n(n = 281)", "Own photos, unchanged\n(Test A, n = 84)", "Own photos, fine-tuned\n(Test B, n = 84)"]
    base = [B["test_macro_f1_mean"], O["A_cnn"]["macro_f1_mean"], O["B_cnn"]["macro_f1_mean"]]
    ga = [fa["ga_test"]["macro_f1"], fa["ga_own_A"]["macro_f1"], fa["ga_own_B"]["macro_f1"]]
    fig, ax = plt.subplots(figsize=(7, 3.3)); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    x = range(3); w = 0.36
    ax.bar([i - w / 2 - 0.01 for i in x], base, w, color=BLUE, label="Baseline CNN")
    ax.bar([i + w / 2 + 0.01 for i in x], ga, w, color=ORANGE, label="GA-optimized CNN")
    for i in x:
        ax.text(i - w / 2, base[i] + 0.015, f"{base[i]:.3f}", ha="center", fontsize=7.5, color=INK)
        ax.text(i + w / 2, ga[i] + 0.015, f"{ga[i]:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.axhline(1 / 3, color=MUTED, lw=1, ls=":"); ax.text(2.62, 1 / 3 + 0.012, "chance", fontsize=7.5, color=MUTED, ha="right")
    ax.set_xticks(list(x)); ax.set_xticklabels(groups, fontsize=8.5); ax.set_ylim(0, 1); ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, out, "fig5_generalization.png")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/figures"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = json.load(open("results/baseline_metrics.json")); C = json.load(open("results/classical_baselines.json"))
    O = json.load(open("results/own/own_photo_metrics.json")); fa = json.load(open("results/final/final_analysis.json"))
    pipeline(a.out); search(a.out, fa); comparison(a.out, B, C, fa); confusions(a.out, fa); generalization(a.out, B, O, fa)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
