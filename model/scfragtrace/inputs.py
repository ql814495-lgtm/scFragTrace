"""Reading and checking a folder of per-sample model inputs (format: model/docs/INPUTS.md).

<dir>/samples.csv             sample_id, batch, tumour_fraction, reference_noncancer  (+ any other columns, kept as is)
<dir>/module_counts.npz       ids (n), counts (n x 764 x 6: central, flank, UL, DL, UR, DR), n_frag_total (n)
<dir>/footprint_counts.npz    ids (n), set_names (>= the 19 footprint sets), C (n x sets), F (n x sets)
<dir>/arm_copy_number.csv     sample_id + 39 arm columns (1p ... 22q): ichorCNA 1 Mb logR, sample-median centred, arm mean
<dir>/global_features.npz     optional: ids, X (n x 1005: 121 length bins, 256 end motifs, 628 5 Mb bin ratios; NaN = missing)
<dir>/delfi_features.npz      optional: ids, X (n x 1063 DELFI features from FinaleToolkit: 5 Mb GC-corrected short-to-long
                              ratios, 5 Mb coverage shares, arm coverage shares); input of the coverage component of the cancer score

A *batch* is a set of samples sequenced and processed together (a cohort or a run). Footprint, node-feature and global
feature referencing are done within each batch, against its known non-cancer samples (reference_noncancer = True) when
the batch has enough of them (footprint >= 10, node and global features >= 3), otherwise as described in INPUTS.md."""
import os
import numpy as np
import pandas as pd

from .priors import FOOTPRINT_SETS
from .reference import N_MODULES

REQUIRED_COLUMNS = ["sample_id", "batch", "tumour_fraction", "reference_noncancer"]


def _bool(col):
    if col.dtype == bool:
        return col.values
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"]).values


def _align(ids_file, ids, what):
    pos = {str(s): i for i, s in enumerate(ids_file)}
    miss = [s for s in ids if s not in pos]
    if miss:
        raise ValueError(f"{what}: {len(miss)} samples of samples.csv are missing, e.g. {miss[:3]}")
    return np.array([pos[s] for s in ids])


def load_inputs(path):
    """Read an input folder; returns a dict of aligned arrays (rows in samples.csv order)."""
    p = lambda f: os.path.join(path, f)
    tab = pd.read_csv(p("samples.csv"), dtype={"sample_id": str, "batch": str})
    miss = [c for c in REQUIRED_COLUMNS if c not in tab.columns]
    if miss:
        raise ValueError(f"samples.csv lacks columns {miss}")
    if tab.sample_id.duplicated().any():
        raise ValueError("duplicated sample_id in samples.csv")
    ids = tab.sample_id.tolist()
    out = dict(samples=tab, ids=np.array(ids), batch=tab.batch.values.astype(str),
               tf=np.nan_to_num(tab.tumour_fraction.values.astype(float)), ref=_bool(tab.reference_noncancer))
    z = np.load(p("module_counts.npz"), allow_pickle=True); ix = _align(z["ids"].astype(str), ids, "module_counts.npz")
    out["module_counts"] = np.asarray(z["counts"], np.float64)[ix]; out["n_frag_total"] = np.asarray(z["n_frag_total"], np.float64)[ix]
    if out["module_counts"].shape[1:] != (N_MODULES, 6):
        raise ValueError(f"module_counts.npz: counts must be n x {N_MODULES} x 6, got {out['module_counts'].shape}")
    z = np.load(p("footprint_counts.npz"), allow_pickle=True); ix = _align(z["ids"].astype(str), ids, "footprint_counts.npz")
    nm = [str(s) for s in z["set_names"]]; lack = [s for s in FOOTPRINT_SETS if s not in nm]
    if lack:
        raise ValueError(f"footprint_counts.npz lacks site sets {lack}")
    cols = [nm.index(s) for s in FOOTPRINT_SETS]
    out["C"] = np.asarray(z["C"], float)[ix][:, cols]; out["F"] = np.asarray(z["F"], float)[ix][:, cols]; out["set_names"] = list(FOOTPRINT_SETS)
    arm = pd.read_csv(p("arm_copy_number.csv"), dtype={"sample_id": str}).set_index("sample_id")
    _align(arm.index.values, ids, "arm_copy_number.csv")
    out["arm"] = arm.loc[ids]
    if os.path.exists(p("global_features.npz")):
        z = np.load(p("global_features.npz"), allow_pickle=True); ix = _align(z["ids"].astype(str), ids, "global_features.npz")
        out["global"] = np.asarray(z["X"], float)[ix]
    if os.path.exists(p("delfi_features.npz")):
        z = np.load(p("delfi_features.npz"), allow_pickle=True); ix = _align(z["ids"].astype(str), ids, "delfi_features.npz")
        out["delfi"] = np.asarray(z["X"], float)[ix]
    return out
