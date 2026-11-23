"""Fragment file -> per-sample model inputs (a compact, single-file implementation of the extraction steps).

Input: a fragment file as distributed by FinaleDB (``*.frag.tsv.bgz`` / ``.gz`` / plain TSV; columns chrom, start, end,
MAPQ, ...; 0-based half-open hg19 coordinates). Outputs, per sample:

  module counts   (764 x 6) central, flank, UL, DL, UR, DR summed over each module's sites, and n_frag_total.
                  Fragments with MAPQ >= 30 on autosomes, length 60-400 bp; central = midpoint |d| <= 75 bp, flank = 1000 <= |d| <= 1500;
                  UL / UR = fragment starts in [-70, -50) / [50, 70) of the site centre, DL / DR = fragment ends there.
                  The paper weighted every fragment by a GC weight (length x GC bin, Griffin-style, from the hg19 2bit
                  genome) and dropped fragments with > 10% N bases; pass ``gc_weight`` (a function of start, end, length
                  arrays per chromosome -> weights) to do the same. Without it all weights are 1 (unweighted counts).
  footprint counts (19 site sets) C = midpoints with |d| <= 75 bp, F = midpoints with 1000 <= |d| <= 1500 bp, unweighted,
                  MAPQ >= 30, autosomes, length 50-400 bp.
  arm copy number from an ichorCNA ``<id>.cna.seg`` file: 1 Mb logR, NA -> excluded, sample-median centred over
                  autosomal bins, mean per chromosome arm.
  tumour fraction from the ichorCNA ``<id>.params.txt`` (run settings: model/docs/INPUTS.md).

The global fragmentomic features (optional input, weight 0 in the shipped configuration) are defined in
model/docs/INPUTS.md (section 2.1, step 5)."""
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

AUTO = [str(i) for i in range(1, 23)]
MINQ = 30


def read_fragments(path, chunksize=2_000_000):
    """Yield (chrom, start, end) arrays per chromosome chunk (autosomes, MAPQ >= 30)."""
    opener = gzip.open if str(path).endswith((".gz", ".bgz")) else open
    with opener(path, "rt") as fh:
        for ch in pd.read_csv(fh, sep="\t", header=None, usecols=[0, 1, 2, 3], names=["c", "s", "e", "q"], comment="#",
                              dtype={"c": str, "s": np.int64, "e": np.int64, "q": np.int64}, chunksize=chunksize):
            ch = ch[ch.q >= MINQ]; ch["c"] = ch.c.str.replace("^chr", "", regex=True)
            for c, g in ch.groupby("c", sort=False):
                if c in AUTO:
                    yield c, g.s.to_numpy(np.int64), g.e.to_numpy(np.int64)


def _window_sum(pos_sorted, cw, centres, lo, hi):
    """Sum of weights of sorted positions p with lo <= p - c <= hi (inclusive) for every centre c; cw = cumulative weights."""
    return cw[np.searchsorted(pos_sorted, centres + hi, "right")] - cw[np.searchsorted(pos_sorted, centres + lo, "left")]


def _sites(ref_dir, name, key):
    z = np.load(Path(ref_dir) / name, allow_pickle=True)
    ch = np.array([str(c).replace("chr", "") for c in z["chrom"]])
    return ch, z["center"].astype(np.int64), z[key].astype(np.int64), z


def extract_fragment_features(frag_path, ref_dir, gc_weight=None):
    """Module counts (764 x 6), n_frag_total, footprint C and F (19 sets) for one fragment file."""
    mch, mcen, mmod, _ = _sites(ref_dir, "module_sites.npz", "module")
    fch, fcen, fset, fz = _sites(ref_dir, "footprint_sites.npz", "set"); set_names = [str(s) for s in fz["set_names"]]
    M = int(mmod.max()) + 1; NS = len(set_names)
    site = np.zeros((len(mcen), 6)); C = np.zeros(NS); F = np.zeros(NS); ntot = 0.0
    per = {}
    for c, s, e in read_fragments(frag_path):
        per.setdefault(c, []).append((s, e))
    for c, parts in per.items():
        s = np.concatenate([p[0] for p in parts]); e = np.concatenate([p[1] for p in parts]); L = e - s
        # module counts: 60-400 bp, optional GC weight
        k = (L >= 60) & (L <= 400); sm, em = s[k], e[k]
        w = np.ones(len(sm)) if gc_weight is None else np.asarray(gc_weight(c, sm, em), float)
        ntot += float(w.sum())
        si = np.where(mch == c)[0]
        if si.size:
            cen = mcen[si]
            mid = (sm + em) // 2; o = np.argsort(mid); cw = np.r_[0, np.cumsum(w[o])]; ms = mid[o]
            site[si, 0] += _window_sum(ms, cw, cen, -75, 75)
            site[si, 1] += _window_sum(ms, cw, cen, 1000, 1500) + _window_sum(ms, cw, cen, -1500, -1000)
            for col, pos in ((2, sm), (4, sm), (3, em), (5, em)):
                o = np.argsort(pos); cw = np.r_[0, np.cumsum(w[o])]
                lo, hi = (-70, -51) if col in (2, 3) else (50, 69)          # [-70, -50) and [50, 70) on integer positions
                site[si, col] += _window_sum(pos[o], cw, cen, lo, hi)
        # footprint counts: 50-400 bp, unweighted
        k = (L >= 50) & (L <= 400); mid = np.sort((s[k] + e[k]) // 2); cw = np.arange(len(mid) + 1, dtype=float)
        fi = np.where(fch == c)[0]
        if fi.size:
            cen = fcen[fi]
            C += np.bincount(fset[fi], _window_sum(mid, cw, cen, -75, 75), NS)
            F += np.bincount(fset[fi], _window_sum(mid, cw, cen, 1000, 1500) + _window_sum(mid, cw, cen, -1500, -1000), NS)
    counts = np.stack([np.bincount(mmod, site[:, j], M) for j in range(6)], 1)
    return dict(module_counts=counts, n_frag_total=ntot, C=C, F=F, set_names=set_names)


def arm_copy_number_from_seg(seg_path, arm_bins_path, arm_names):
    """Arm-level copy number from an ichorCNA .cna.seg file (column 6 = logR)."""
    d = pd.read_csv(seg_path, sep="\t"); a0 = pd.read_csv(arm_bins_path, sep="\t")
    ch = d.iloc[:, 0].astype(str).str.replace("^chr", "", regex=True).values
    auto = np.isin(ch, AUTO); v = pd.to_numeric(d.iloc[:, 5], errors="coerce").values.astype(float)
    arm = np.array([""] * len(d), dtype=object)
    st = d.iloc[:, 1].values; en = d.iloc[:, 2].values
    for _, r in a0.iterrows():
        m = (ch == str(r["chrom"])) & (st - 1 >= r["start"]) & (en <= r["stop"] + 1)
        arm[m] = r["arm"]
    v, arm = v[auto], arm[auto]
    v = v - np.nanmedian(v)
    out = []
    for a in arm_names:
        m = arm == a
        out.append(np.nanmean(v[m]) if m.any() and np.isfinite(v[m]).any() else 0.0)
    return np.nan_to_num(np.array(out))


def tumour_fraction_from_params(params_path):
    """Tumour fraction of the selected solution from an ichorCNA <id>.params.txt."""
    for line in open(params_path):
        if line.lower().startswith("tumor fraction"):
            return float(line.split(":")[1])
    raise ValueError(f"no 'Tumor Fraction' line in {params_path}")


def write_inputs(out_dir, records, batch, reference_noncancer, tumour_fraction, arm_names, extra=None):
    """Write a model input folder from extract_fragment_features() records (dict sample_id -> features) and arm rows."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); ids = list(records)
    tab = pd.DataFrame(dict(sample_id=ids, batch=[batch[i] for i in ids], tumour_fraction=[tumour_fraction[i] for i in ids],
                            reference_noncancer=[bool(reference_noncancer[i]) for i in ids]))
    if extra is not None:
        tab = tab.merge(extra, on="sample_id", how="left")
    tab.to_csv(out / "samples.csv", index=False)
    np.savez_compressed(out / "module_counts.npz", ids=np.array(ids), counts=np.stack([records[i]["module_counts"] for i in ids]),
                        n_frag_total=np.array([records[i]["n_frag_total"] for i in ids]), columns=np.array(["central", "flank", "UL", "DL", "UR", "DR"]))
    np.savez_compressed(out / "footprint_counts.npz", ids=np.array(ids), set_names=np.array(records[ids[0]]["set_names"]),
                        C=np.stack([records[i]["C"] for i in ids]), F=np.stack([records[i]["F"] for i in ids]))
    arm = pd.DataFrame(np.stack([records[i]["arm"] for i in ids]), columns=arm_names); arm.insert(0, "sample_id", ids)
    arm.to_csv(out / "arm_copy_number.csv", index=False, float_format="%.17g")
    return out
