"""Check whether two image folders share images (exact MD5 match or near-duplicate perceptual hash).

Used to test whether the Kaggle "Garbage Classification" upload is a re-distribution of TrashNet.
Usage:
    python src/check_overlap.py --a data/raw/dataset-resized --b data/kaggle/Garbage\ classification
"""
import argparse
import hashlib
import os

import numpy as np
from PIL import Image, ImageOps


def files(root):
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                yield os.path.join(d, f)


def dhash(p, size=8):
    a = np.asarray(ImageOps.exif_transpose(Image.open(p)).convert("L").resize((size + 1, size)), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--hamming", type=int, default=4)
    args = ap.parse_args()
    A, B = list(files(args.a)), list(files(args.b))
    md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
    ma = {md5(p) for p in A}
    exact = sum(md5(p) in ma for p in B)
    ha = np.stack([dhash(p) for p in A])
    near = sum(int(((ha != dhash(p)).sum(1) <= args.hamming).any()) for p in B)
    print(f"A={len(A)} B={len(B)} exact_matches={exact} near_duplicates(<= {args.hamming} bits)={near} "
          f"-> {100 * near / max(len(B), 1):.1f}% of B also appears in A")


if __name__ == "__main__":
    main()
