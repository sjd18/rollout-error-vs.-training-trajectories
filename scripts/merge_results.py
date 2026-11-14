"""Merge shard CSVs (results/sweep.shard*.csv) into results/sweep.csv.

python scripts/merge_results.py [--delete-shards]
"""

import argparse
import csv
from pathlib import Path

from rollscale import FIELDS, read_results

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stem", default="sweep")
    ap.add_argument("--delete-shards", action="store_true")
    a = ap.parse_args()
    out = ROOT / "results" / f"{a.stem}.csv"
    shards = sorted(out.parent.glob(f"{a.stem}.shard*.csv"))
    rows = {}
    for path in [out, *shards]:
        for r in read_results(path):
            rows[(r["n_traj"], r["width"], r["seed"])] = r
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])
    print(f"wrote {len(rows)} runs to {out} (from {len(shards)} shard files)")
    if a.delete_shards:
        for path in shards:
            path.unlink()


if __name__ == "__main__":
    main()
