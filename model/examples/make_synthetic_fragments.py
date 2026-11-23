"""Write a small synthetic FinaleDB-style fragment file (chrom, start, end, MAPQ, strand) for testing the feature
extractor: fragments are drawn around a few reference sites of chr21/chr22 (module sites and footprint sites) plus
uniform background. Not real data.
usage: python model/examples/make_synthetic_fragments.py [out.tsv.gz]"""
import gzip, sys
from pathlib import Path
import numpy as np

REF = Path(__file__).resolve().parents[1] / "weights" / "reference"


def make(out, n_bg=20000, seed=0):
    rng = np.random.default_rng(seed); rows = []
    ms = np.load(REF / "module_sites.npz", allow_pickle=True); fs = np.load(REF / "footprint_sites.npz", allow_pickle=True)
    for chrom in ("21", "22"):
        cen = np.r_[ms["center"][np.char.replace(ms["chrom"].astype(str), "chr", "") == chrom][:40],
                    fs["center"][fs["chrom"].astype(str) == chrom][:40]]
        lo, hi = cen.min() - 5000, cen.max() + 5000
        mids = np.r_[np.repeat(cen, 60) + rng.integers(-1600, 1600, 60 * len(cen)), rng.integers(lo, hi, n_bg)]
        L = rng.integers(40, 420, len(mids)); s = mids - L // 2; e = s + L; q = rng.choice([0, 20, 60], len(mids), p=[.1, .1, .8])
        for a, b, c in zip(s, e, q):
            rows.append(f"{'chr' + chrom if rng.random() < .5 else chrom}\t{a}\t{b}\t{c}\t{'+' if rng.random() < .5 else '-'}\n")
    rows.append("chrX\t1000\t1167\t60\t+\n")       # non-autosomal, ignored
    with gzip.open(out, "wt") as f:
        f.writelines(rows)
    return out


if __name__ == "__main__":
    print(make(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "data" / "synthetic" / "synthetic_fragments.tsv.gz")))
