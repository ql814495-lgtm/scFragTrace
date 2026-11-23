"""Command line interface.

    scfragtrace predict --inputs <input dir> --out <output dir> [--bundle cristiano_only] [--mode label_free|full] [--w-global W]
    scfragtrace extract --manifest <samples.tsv> --out <input dir>
    scfragtrace info [--bundle cristiano_only]

``extract`` reads a tab-separated manifest with columns sample_id, batch, reference_noncancer, fragments (fragment file),
seg (ichorCNA .cna.seg) and tumour_fraction (or params, an ichorCNA .params.txt) and writes a model input folder."""
import argparse
import json
import sys
from pathlib import Path


def _predict(a):
    from .model import load_model
    m = load_model(a.bundle)
    res = m.predict(a.inputs, w_global=a.w_global, mode=a.mode)
    out = res.save(a.out)
    cols = ["sample_id", "detection_score", "predicted_class", "second_class"] + [f"share_{k}" for k in ("footprint", "copy_number", "gnn")]
    print(res.table[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nwritten: {out}/predictions.csv, class_scores.csv, evidence_terms.csv (bundle {m.manifest['name']})")


def _extract(a):
    import pandas as pd
    from .features import extract_fragment_features, arm_copy_number_from_seg, tumour_fraction_from_params, write_inputs
    ref = Path(a.reference or Path(__file__).resolve().parents[1] / "weights" / "reference")
    arm_names = list(pd.read_csv(ref / "tcga_arm_templates.csv", index_col=0).columns)
    man = pd.read_csv(a.manifest, sep="\t", dtype={"sample_id": str, "batch": str})
    rec, tf, batch, refnc = {}, {}, {}, {}
    for _, r in man.iterrows():
        f = extract_fragment_features(r["fragments"], ref)
        f["arm"] = arm_copy_number_from_seg(r["seg"], ref / "arm_bins.tsv", arm_names)
        rec[r.sample_id] = f; batch[r.sample_id] = r["batch"]; refnc[r.sample_id] = str(r["reference_noncancer"]).lower() in ("true", "1", "yes")
        tf[r.sample_id] = float(r["tumour_fraction"]) if "tumour_fraction" in man and pd.notna(r.get("tumour_fraction")) else tumour_fraction_from_params(r["params"])
        print("extracted", r.sample_id, flush=True)
    write_inputs(a.out, rec, batch, refnc, tf, arm_names)
    print("written", a.out)


def _info(a):
    from .model import load_model
    m = load_model(a.bundle)
    man = dict(m.manifest); man.pop("files", None)
    print(json.dumps(man, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scfragtrace", description="scFragTrace: cfDNA cancer detection and tissue of origin")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("predict", help="run a trained bundle on an input folder")
    p.add_argument("--inputs", required=True); p.add_argument("--out", required=True); p.add_argument("--bundle", default="cristiano_only")
    p.add_argument("--w-global", type=float, default=None, help="fusion weight of the global LR in full mode (default: manifest, 0)")
    p.add_argument("--mode", default=None, choices=["label_free", "full"], help="tissue-of-origin score (default: the bundle's manifest)")
    p.set_defaults(fn=_predict)
    p = sub.add_parser("extract", help="fragment files + ichorCNA outputs -> input folder")
    p.add_argument("--manifest", required=True); p.add_argument("--out", required=True); p.add_argument("--reference", default=None)
    p.set_defaults(fn=_extract)
    p = sub.add_parser("info", help="print a bundle's manifest"); p.add_argument("--bundle", default="cristiano_only"); p.set_defaults(fn=_info)
    a = ap.parse_args(argv); a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
