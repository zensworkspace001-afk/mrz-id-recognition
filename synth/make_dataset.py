"""產生合成資料集：python -m synth.make_dataset --n 20 --out data/synth"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .identities import SUPPORTED
from .render import generate_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每個國家張數")
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    n = 0
    with (out / "annotations.jsonl").open("w", encoding="utf-8") as f:
        for c in SUPPORTED:
            for i in range(a.n):
                rec = generate_sample(c, i, out / "images", a.seed)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    print(f"產生 {n} 筆 -> {out}")


if __name__ == "__main__":
    main()
