"""Train a scFragTrace weight bundle (final configuration of the paper) from a folder of labelled per-sample inputs.

usage: python model/training/train_model.py --inputs <input dir> --config model/training/configs/cristiano_only.json \
           --out model/weights/cristiano_only [--stages too,global,det] [--threads 2]

The input folder has the format of model/docs/INPUTS.md plus the training columns cancer, true_class and train_too.
The training inputs of the shipped bundles (1,124 samples of Cristiano, Adalsteinsson, Jiang, Snyder and Sun) are not
distributed: they derive from controlled-access data (EGA, dbGaP). The training inputs are computed from fragment
files as described in model/docs/INPUTS.md (``scfragtrace extract``). Wall time on 2 CPU threads: see the
manifest of each bundle (training.*.seeds[].seconds)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scfragtrace import load_inputs  # noqa: E402
from scfragtrace.train import train_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--inputs", required=True); ap.add_argument("--config", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--reference", default=str(Path(__file__).resolve().parents[1] / "weights" / "reference"))
    ap.add_argument("--stages", default="too,global,det"); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--save-outputs", default="", help="optional .npz with the in-training predictions (for verification)")
    a = ap.parse_args(); cfg = json.load(open(a.config))
    x = load_inputs(a.inputs)
    man, out = train_bundle(x, a.out, a.reference, cfg["name"], cfg["too_train_cohorts"], cfg["too_val_order"], cfg["too_unlabelled_cohorts"],
                            cfg["det_train_cohorts"], cfg["det_weights"], cfg["w_global"], cfg["global_reference_batch"], tuple(cfg["seeds"]),
                            a.threads, tuple(a.stages.split(",")), cfg["description"], log=lambda *m: print(*m, flush=True),
                            too_val_split=cfg.get("too_val_split", "cohort"), too_default_mode=cfg.get("too_default_mode", "full"))
    if a.save_outputs:
        import numpy as np
        np.savez(a.save_outputs, **{k: v for k, v in out.items() if v is not None})
    print("saved", a.out)


if __name__ == "__main__":
    main()
