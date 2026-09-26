"""把 run.py 的結果與合成資料的正確答案比對：python evaluate.py data/synth/annotations.jsonl out/records.jsonl"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from idpipe.mrz import parse_td3


def main(ann_path, rec_path):
    truth = {}
    for l in open(ann_path, encoding="utf-8"):
        a = json.loads(l)
        truth[a["image"]] = a
    recs = {Path(r["source_image"]).name: r for r in map(json.loads, open(rec_path, encoding="utf-8"))}
    per = defaultdict(lambda: {"n": 0, "read": 0, "valid": 0, "exact": 0, "review": 0})
    for img, a in truth.items():
        s = per[a["country"]]
        s["n"] += 1
        r = recs.get(img)
        if not r:
            continue
        s["review"] += r["needs_review"]
        if r["mrz"].get("line1"):
            s["read"] += 1
            s["valid"] += bool(r["mrz"]["checksum_valid"])
            t = parse_td3(*a["mrz"])
            s["exact"] += (r["surname"], r["given_names"], r["document_number"], r["date_of_birth"],
                           r["date_of_expiry"], r["nationality"]) == (
                t.surname, t.given_names, t.document_number, t.date_of_birth, t.date_of_expiry, t.nationality)
    print(f"{'國家':<5}{'張數':>5}{'讀到MRZ':>9}{'檢查碼通過':>11}{'欄位全對':>9}{'需複核':>7}")
    tot = defaultdict(int)
    for c, s in sorted(per.items()):
        print(f"{c:<7}{s['n']:>4}{s['read']:>9}{s['valid']:>11}{s['exact']:>9}{s['review']:>7}")
        for k, v in s.items():
            tot[k] += v
    print(f"{'合計':<6}{tot['n']:>4}{tot['read']:>9}{tot['valid']:>11}{tot['exact']:>9}{tot['review']:>7}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
