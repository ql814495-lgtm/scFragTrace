"""End-to-end demo in a fresh Python process on the 58 open-access Snyder et al. 2016 samples.
* prior-only demo (no trained weights): must reproduce model/examples/expected_output/prior_only/ and the paper's
  label-free tissue-of-origin model on Snyder (balanced accuracy 0.554, TFx >= 5% 0.806).
* bundle cristiano_only (every trained component fitted on the Cristiano cohort): must reproduce
  expected_output/cristiano_only/ (label-free tissue of origin + five-component cancer score) and
  expected_output/cristiano_only_full/ (residual graph network added); the command-line entry point must give the
  same table."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "model" / "examples"
W = ROOT / "model" / "weights"


def _run(args, tmp):
    r = subprocess.run([sys.executable, str(EX / "run_demo.py"), *args, "--out", str(tmp)], cwd=ROOT, capture_output=True, text=True, timeout=1800)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_prior_only_demo(tmp_path):
    out = _run(["--prior-only"], tmp_path)
    assert "-> OK" in out, out
    m = json.load(open(tmp_path / "metrics.json"))
    assert round(m["tissue_of_origin_balanced_accuracy"], 3) == 0.554
    assert round(m["tissue_of_origin_balanced_accuracy_tfx_ge_5pct"], 3) == 0.806
    t = pd.read_csv(tmp_path / "predictions.csv")
    assert len(t) == 58 and np.allclose(t.filter(like="P_").sum(1), 1)


B = "cristiano_only"


def test_bundle_present():
    assert (W / B / "manifest.json").exists(), "trained bundle cristiano_only is part of the release"
    m = json.load(open(W / B / "manifest.json"))
    assert m["training"]["tissue_of_origin"]["train_cohorts"] == ["Cristiano"]
    assert m["training"]["detection"]["train_cohorts"] == ["Cristiano"]


def test_bundle_demo_label_free(tmp_path):
    out = _run(["--bundle", B], tmp_path)
    assert "-> OK" in out, out
    t = pd.read_csv(tmp_path / "predictions.csv")
    assert len(t) == 58 and np.allclose(t.filter(like="P_").sum(1), 1)
    assert (t.tissue_mode == "label_free").all()
    m = json.load(open(tmp_path / "metrics.json"))
    # label-free tissue of origin has no trained parameter: identical to the prior-only demo and to the paper
    assert round(m["tissue_of_origin_balanced_accuracy"], 3) == 0.554
    assert round(m["tissue_of_origin_balanced_accuracy_tfx_ge_5pct"], 3) == 0.806
    p = pd.read_csv(EX / "expected_output" / "prior_only" / "predictions.csv")
    assert (t.predicted_class == p.predicted_class).all()
    assert t.detection_score.notna().all()
    assert set(t.detection_components.iloc[0].split("+")) == {"tf", "module_lr", "gnn", "global_lr", "delfi_lr"}


def test_bundle_demo_full(tmp_path):
    out = _run(["--bundle", B, "--mode", "full"], tmp_path)
    assert "-> OK" in out, out
    t = pd.read_csv(tmp_path / "predictions.csv")
    assert len(t) == 58 and np.allclose(t.filter(like="P_").sum(1), 1)
    assert np.allclose(t[[c for c in t.columns if c.startswith("share_")]].sum(1), 1)
    assert (t.tissue_mode == "full").all()


def test_cli_matches_api(tmp_path):
    r = subprocess.run([sys.executable, "-m", "scfragtrace", "predict", "--inputs", str(EX / "data" / "snyder_gse71378"),
                        "--out", str(tmp_path), "--bundle", B], cwd=ROOT, capture_output=True, text=True, timeout=1800,
                       env={**os.environ, "PYTHONPATH": str(ROOT / "model")})
    assert r.returncode == 0, r.stdout + r.stderr
    a = pd.read_csv(tmp_path / "predictions.csv"); b = pd.read_csv(EX / "expected_output" / B / "predictions.csv")
    assert (a.predicted_class == b.predicted_class).all() and np.allclose(a.detection_score, b.detection_score, atol=1e-4)
