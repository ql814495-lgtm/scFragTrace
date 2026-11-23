"""scFragTrace demo: load the trained model, run it on 58 open-access plasma samples of Snyder et al. 2016 (GEO GSE71378;
48 cancers of 13 types, 10 non-cancer controls), write predictions and the evidence decomposition, score them against
the known diagnoses and compare with the shipped expected output.

usage (from the repository root):
    python model/examples/run_demo.py                      # bundle cristiano_only (trained on the Cristiano cohort only;
                                                           # Snyder is a new cohort for it). Tissue of origin: label-free
                                                           # 47-feature model; cancer score: five components.
    python model/examples/run_demo.py --mode full          # tissue of origin with the residual graph network added
                                                           # (the model used within the discovery cohort)
    python model/examples/run_demo.py --prior-only         # label-free model only (no trained weights needed)

Outputs: model/examples/output/<bundle>/predictions.csv, class_scores.csv, evidence_terms.csv, metrics.json.
The same prediction from the command line:  scfragtrace predict --inputs model/examples/data/snyder_gse71378 --out out/"""
import argparse
import warnings
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
sys.path.insert(0, str(HERE.parent))                # model/ -> scfragtrace importable without installation
from scfragtrace import load_model, load_inputs  # noqa: E402
from scfragtrace.model import WEIGHTS_ROOT, predict_prior_only  # noqa: E402


def metrics(tab, samples):
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    s = samples.set_index("sample_id").loc[tab.sample_id]
    lab = s.true_class.fillna("").values; m = lab != ""
    cancer = ~s.reference_noncancer.astype(bool).values
    tfx5 = m & (tab.tumour_fraction.values >= 0.05)
    out = dict(n_samples=int(len(tab)), n_cancer=int(cancer.sum()), n_noncancer=int((~cancer).sum()),
               n_tissue_of_origin_evaluable=int(m.sum()),
               tissue_of_origin_balanced_accuracy=float(balanced_accuracy_score(lab[m], tab.predicted_class.values[m])),
               tissue_of_origin_top1=float(np.mean(lab[m] == tab.predicted_class.values[m])),
               tissue_of_origin_top2=float(np.mean((lab[m] == tab.predicted_class.values[m]) | (lab[m] == tab.second_class.values[m]))),
               n_tfx_ge_5pct=int(tfx5.sum()),
               tissue_of_origin_balanced_accuracy_tfx_ge_5pct=float(balanced_accuracy_score(lab[tfx5], tab.predicted_class.values[tfx5])),
               detection_auroc=(float(roc_auc_score(cancer, tab.detection_score.values)) if tab.detection_score.notna().all() else None),
               mean_evidence_share={k: float(tab[f"share_{k}"][m].mean()) for k in ("footprint", "copy_number", "gnn", "global")})
    return out


def compare(out_dir, exp_dir, atol=1e-4):
    a = pd.read_csv(out_dir / "predictions.csv"); b = pd.read_csv(exp_dir / "predictions.csv")
    assert list(a.sample_id) == list(b.sample_id), "sample order differs"
    same_class = (a.predicted_class == b.predicted_class).all()
    num = [c for c in b.columns if c in a.columns and pd.api.types.is_numeric_dtype(b[c]) and b[c].notna().any()]
    diff = max(float(np.nanmax(np.abs(a[c].values.astype(float) - b[c].values.astype(float)))) for c in num)
    return same_class, diff, diff <= atol


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", default="cristiano_only"); ap.add_argument("--inputs", default=str(HERE / "data" / "snyder_gse71378"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", default=None, choices=["label_free", "full"],
                    help="tissue-of-origin mode; default: the bundle's default (label_free for cristiano_only)")
    ap.add_argument("--prior-only", action="store_true", help="label-free prior only (no trained weights needed)")
    a = ap.parse_args()
    bundle_ok = (Path(a.bundle) / "manifest.json").exists() or (WEIGHTS_ROOT / a.bundle / "manifest.json").exists()
    prior_only = a.prior_only or not bundle_ok
    tag = "prior_only" if prior_only else (a.bundle if a.mode in (None, "label_free") else f"{a.bundle}_{a.mode}")
    out = Path(a.out or HERE / "output" / tag)
    x = load_inputs(a.inputs)
    print(f"inputs: {len(x['ids'])} samples, {int(x['ref'].sum())} non-cancer references, batch(es) {sorted(set(x['batch']))}")
    if prior_only:
        if not a.prior_only:
            print(f"bundle '{a.bundle}' not found under {WEIGHTS_ROOT}: running the label-free prior only (lineage footprint + copy-number template)")
        res = predict_prior_only(x)
    else:
        print(f"loading bundle '{a.bundle}' ...")
        model = load_model(a.bundle)
        tr = model.manifest["training"]["tissue_of_origin"]
        de = model.manifest["training"]["detection"]
        print(f"  trained on: tissue-of-origin graph network {', '.join(tr['train_cohorts'])}; cancer score {', '.join(de['train_cohorts'])}")
        res = model.predict(x, mode=a.mode)
        print(f"  tissue-of-origin mode: {res.table.tissue_mode.iloc[0]}; cancer score components: {res.table.detection_components.iloc[0]}")
    res.save(out)
    m = metrics(res.table, x["samples"]); json.dump(m, open(out / "metrics.json", "w"), indent=1)
    show = res.table.merge(x["samples"][["sample_id", "diagnosis"]], on="sample_id")
    cols = ["sample_id", "diagnosis", "tumour_fraction", "detection_score", "predicted_class", "share_footprint", "share_copy_number", "share_gnn"]
    print(show[cols].sort_values("tumour_fraction", ascending=False).head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n... {len(show)} samples; full table: {out / 'predictions.csv'}")
    print(f"tissue of origin ({m['n_tissue_of_origin_evaluable']} cancers of the 8 classes): balanced accuracy {m['tissue_of_origin_balanced_accuracy']:.3f}, "
          f"top-1 {m['tissue_of_origin_top1']:.3f}, top-2 {m['tissue_of_origin_top2']:.3f}; TFx >= 5% (n={m['n_tfx_ge_5pct']}): "
          f"balanced accuracy {m['tissue_of_origin_balanced_accuracy_tfx_ge_5pct']:.3f}")
    if m["detection_auroc"] is not None:
        print(f"detection (48 cancers vs 10 non-cancer): AUROC {m['detection_auroc']:.3f}")
    print("mean evidence share at the predicted class: " + ", ".join(f"{k} {v:.2f}" for k, v in m["mean_evidence_share"].items()))
    exp = HERE / "expected_output" / tag
    if (exp / "predictions.csv").exists() and a.inputs == str(HERE / "data" / "snyder_gse71378"):
        same, diff, ok = compare(out, exp)
        print(f"\ncomparison with {exp.relative_to(HERE.parent.parent)}: predicted classes identical: {same}; max abs difference of numeric columns {diff:.2e} -> {'OK' if ok and same else 'DIFFERENT'}")
        if not (ok and same):
            sys.exit(1)


if __name__ == "__main__":
    main()
