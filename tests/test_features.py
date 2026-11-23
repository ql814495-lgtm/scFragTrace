"""Feature extractor on the synthetic fragment file, checked against a direct per-site computation; ichorCNA segment ->
arm copy number on the shipped example segment files."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from scfragtrace.features import extract_fragment_features, arm_copy_number_from_seg

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "model" / "weights" / "reference"
EX = ROOT / "model" / "examples" / "data"


def _read(path):
    d = pd.read_csv(path, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["c", "s", "e", "q"], dtype={"c": str})
    d["c"] = d.c.str.replace("^chr", "", regex=True)
    return d[(d.q >= 30) & d.c.isin([str(i) for i in range(1, 23)])]


def test_extractor_matches_direct_count():
    frag = EX / "synthetic" / "synthetic_fragments.tsv.gz"
    out = extract_fragment_features(frag, REF)
    d = _read(frag); ms = np.load(REF / "module_sites.npz"); fs = np.load(REF / "footprint_sites.npz")
    mch = np.char.replace(ms["chrom"].astype(str), "chr", ""); M = int(ms["module"].max()) + 1
    exp = np.zeros((M, 6)); n = 0
    for c, g in d.groupby("c"):
        s, e = g.s.values, g.e.values; L = e - s; k = (L >= 60) & (L <= 400); s, e = s[k], e[k]; n += k.sum(); mid = (s + e) // 2
        for i in np.where(mch == c)[0]:
            cc = ms["center"][i]; m = ms["module"][i]; dm = mid - cc
            exp[m] += [np.sum(np.abs(dm) <= 75), np.sum((np.abs(dm) >= 1000) & (np.abs(dm) <= 1500)),
                       np.sum((s - cc >= -70) & (s - cc < -50)), np.sum((e - cc >= -70) & (e - cc < -50)),
                       np.sum((s - cc >= 50) & (s - cc < 70)), np.sum((e - cc >= 50) & (e - cc < 70))]
    assert out["n_frag_total"] == n
    assert np.allclose(out["module_counts"], exp) and exp[:, 0].sum() > 0 and exp[:, 2:].sum() > 0
    C = np.zeros(len(fs["set_names"])); F = np.zeros_like(C)
    for c, g in d.groupby("c"):
        s, e = g.s.values, g.e.values; L = e - s; k = (L >= 50) & (L <= 400); mid = (s[k] + e[k]) // 2
        for i in np.where(fs["chrom"].astype(str) == c)[0]:
            dm = mid - fs["center"][i]; C[fs["set"][i]] += np.sum(np.abs(dm) <= 75); F[fs["set"][i]] += np.sum((np.abs(dm) >= 1000) & (np.abs(dm) <= 1500))
    assert np.allclose(out["C"], C) and np.allclose(out["F"], F) and C.sum() > 0


def test_arm_copy_number_from_ichorcna_segments():
    arms = list(pd.read_csv(REF / "tcga_arm_templates.csv", index_col=0).columns)
    tab = pd.read_csv(EX / "snyder_gse71378" / "arm_copy_number.csv", dtype={"sample_id": str}).set_index("sample_id")
    files = sorted((EX / "ichorcna_seg").glob("*.cna.seg"))
    assert files
    for f in files:
        a = arm_copy_number_from_seg(f, REF / "arm_bins.tsv", arms)
        assert np.allclose(a, tab.loc[f.name.split(".")[0], arms].values, atol=1e-9)
