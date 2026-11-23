"""The trained scFragTrace model: loading a weight bundle and predicting on a folder of per-sample inputs.

    from scfragtrace import load_model, load_inputs
    m = load_model()                         # model/weights/cristiano_only (or a bundle name / folder)
    res = m.predict(load_inputs("model/examples/data/snyder_gse71378"))
    res.table                                # one row per sample: cancer score, predicted class, class scores, evidence shares
    res.save("out/")                         # predictions.csv, class_scores.csv, evidence_terms.csv

Tissue of origin (per class k; formulas in model/docs/ARCHITECTURE.md):
    PRIOR_k  = zs(S)_k + zs(CNA)_k                                         label-free prior (8 footprints + 39 arms)
  mode "label_free" (new cohorts, the paper's setting): score_k = PRIOR_k
  mode "full" (discovery cohort components):
    final_k  = PRIOR_k + mean_s lambda_s * G_s,k                           residual graph network, 3 seeds
    score_k  = final_k + w * log p_global(k)                               late fusion with the global LR
    P(k)     = softmax_k(score)                                            normalised score, not calibrated
Cancer score: D = sum_m z(component m) over ichorCNA tumour fraction (TFx), module LR, detection graph network, global
fragmentomics LR and DELFI coverage LR, each z within the batch over all its samples, with the (equal) weights of the
manifest; a component whose input file is absent is left out. A batch-relative score, not a calibrated probability.
Evidence (LCC decomposition): for the predicted class, each term (footprint, copy-number template, graph network,
global) centred over classes, |t_pred - mean_c t_c|, as a share of their sum."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import priors, preprocess
from .reference import graph_tensors, lineage_mask, module_node_features

WEIGHTS_ROOT = Path(os.environ.get("SCFRAGTRACE_WEIGHTS", Path(__file__).resolve().parents[1] / "weights"))
TERMS = ["footprint", "copy_number", "gnn", "global"]
DET_COMPONENTS = ["tf", "module_lr", "gnn", "global_lr", "delfi_lr"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _npz(path):
    z = np.load(path, allow_pickle=False); return {k: z[k] for k in z.files}


class Prediction:
    def __init__(self, table, scores, terms):
        self.table = table; self.scores = scores; self.terms = terms

    def save(self, out):
        out = Path(out); out.mkdir(parents=True, exist_ok=True)
        self.table.to_csv(out / "predictions.csv", index=False, float_format="%.6g")
        cs = pd.DataFrame({"sample_id": self.table.sample_id})
        for name in ["prior", "gnn_residual", "final", "global_logp", "fused"]:
            for j, c in enumerate(priors.CLASSES):
                cs[f"{name}_{c}"] = self.scores[name][:, j]
        cs.to_csv(out / "class_scores.csv", index=False, float_format="%.6g")
        rows = []
        for t in TERMS:
            df = pd.DataFrame(self.terms[t], columns=priors.CLASSES); df.insert(0, "term", t); df.insert(0, "sample_id", self.table.sample_id.values)
            rows.append(df)
        pd.concat(rows).to_csv(out / "evidence_terms.csv", index=False, float_format="%.6g")
        return out


class ScFragTrace:
    """A trained bundle: reference files (model/weights/reference) + one bundle folder with manifest.json."""

    def __init__(self, bundle_dir, reference_dir=None, verify_checksums=True, device="cpu"):
        import torch
        from . import gnn
        self.dir = Path(bundle_dir)
        self.manifest = json.load(open(self.dir / "manifest.json"))
        ref = Path(reference_dir) if reference_dir else (self.dir / self.manifest.get("reference_dir", "../reference")).resolve()
        self.ref_dir = ref
        if verify_checksums:
            for rel, h in self.manifest["files"].items():
                p = (self.dir / rel).resolve()
                if sha256(p) != h:
                    raise ValueError(f"checksum mismatch for {p}")
        self.classes = list(self.manifest["classes"]); K = len(self.classes)
        g = _npz(ref / "module_graph.npz"); self.module_set = g["module_set"]
        A, Dh = graph_tensors(g["adj_sc"]); mask, hasmod = lineage_mask(self.module_set, K)
        T = pd.read_csv(ref / "tcga_arm_templates.csv", index_col=0); self.arm_names = list(T.columns); self.TT = T.loc[self.classes].values
        fp = _npz(ref / "footprint_pooled_reference.npz"); self.fp_fallback = (fp["coef"], fp["resid_sd"])
        arch = self.manifest["architecture"]; tor = self.manifest["tissue_of_origin"]
        self.too_nets = []
        for f in tor["gnn_weights"]:
            net = gnn.ResidualLineageGNN(A, Dh, arch["node_features"], mask, hasmod, nco=arch["n_cohorts_adversary"], d=arch["d"], heads=arch["heads"])
            self.too_nets.append(gnn.load_weights(net, torch.load(self.dir / f, map_location=device, weights_only=True)).eval())
        nz = _npz(self.dir / tor["node_standardisation"]); self.too_mu, self.too_sd = nz["mu"], nz["sd"]
        self.lambdas = np.asarray(tor["lambdas"], float)
        self.too_mode = tor.get("default_mode", "full")
        self.w_global = float(self.manifest["fusion"]["w_global"])
        self.glr = _npz(self.dir / self.manifest["global_lr"]["file"])
        det = self.manifest["detection"]; self.det_weights = det["weights"]
        self.det_nets = []
        for f in det["gnn_weights"]:
            net = gnn.DetectionGNN(A, Dh, arch["node_features"], d=arch["d"], heads=arch["heads"])
            self.det_nets.append(gnn.load_weights(net, torch.load(self.dir / f, map_location=device, weights_only=True)).eval())
        self.det_lr = _npz(self.dir / det["module_lr"])
        self.det_glr = _npz(self.dir / det["global_lr"]) if det.get("global_lr") else None
        self.det_delfi = _npz(self.dir / det["delfi_lr"]) if det.get("delfi_lr") else None
        self._gnn = gnn

    # ---------------------------------------------------------------- components
    def footprint(self, x):
        return priors.lineage_footprint(x["C"], x["F"], x["set_names"], x["batch"], ~x["ref"], fallback=self.fp_fallback)

    def copy_number(self, x):
        return priors.corr_t(x["arm"][self.arm_names].values.astype(float), self.TT)

    def global_logp(self, x):
        K = len(self.classes)
        if "global" not in x:
            return np.zeros((len(x["ids"]), K)), False
        g = self.glr; X = preprocess.global_recentre(x["global"], x["batch"], x["ref"], g["hmean"])
        z = (X - g["mean"]) / g["scale"]; lg = z @ g["coef"].T + g["intercept"]
        p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        P = np.full((len(X), K), 1e-6); P[:, g["classes"]] = p
        return np.log(np.clip(P, 1e-6, 1)), True

    def tissue_of_origin(self, x, node):
        tf = x["tf"]; S = np.nan_to_num(self.footprint(x)); CP = self.copy_number(x)
        PRI = priors.prior_inputs(tf, S, CP); PRIOR = priors.label_free_prior(tf, S, CP)
        Nz = preprocess.too_node_features(node, x["batch"], x["ref"])
        G = []
        for net, mu, sd in zip(self.too_nets, self.too_mu, self.too_sd):
            Xs = np.clip((Nz - mu) / sd, -5, 5).astype(np.float32)
            G.append(self._gnn.run(net, Xs, PRI)["gnn"])
        G = np.array(G); res = (self.lambdas[:, None, None] * G).mean(0)
        return dict(S=S, CP=CP, PRIOR=PRIOR, T_fp=priors.footprint_term(tf, S), T_cn=priors.zs(CP), G_seeds=G, gnn_residual=res, final=PRIOR + res)

    def detection(self, x, node):
        Xd = preprocess.det_node_features(node, x["batch"])
        flat = Xd.reshape(len(Xd), -1); L = self.det_lr
        mod = ((flat - L["mean"]) / L["scale"]) @ L["coef"].ravel() + float(L["intercept"].ravel()[0])
        gd = np.mean([self._gnn.run(net, Xd)["rdet"] for net in self.det_nets], 0)
        w = self.det_weights; b = x["batch"]; nan = np.full(len(Xd), np.nan)
        comp = {"tf": x["tf"], "module_lr": mod, "gnn": gd, "global_lr": nan, "delfi_lr": nan}
        if self.det_glr is not None and "global" in x:
            L = self.det_glr; Xg = np.where(np.isnan(x["global"]), L["impute"], x["global"])
            comp["global_lr"] = ((Xg - L["mean"]) / L["scale"]) @ L["coef"].ravel() + float(L["intercept"].ravel()[0])
        if self.det_delfi is not None and "delfi" in x:
            L = self.det_delfi; Zd = preprocess.batch_standardise(x["delfi"], b); Zd = np.where(np.isnan(Zd), L["impute"], Zd)
            lg = ((Zd - L["mean"]) / L["scale"]) @ L["coef"].ravel() + float(L["intercept"].ravel()[0])
            comp["delfi_lr"] = 1.0 / (1.0 + np.exp(-lg))
        used = [k for k in DET_COMPONENTS if float(w.get(k, 0)) != 0 and np.isfinite(comp[k]).all()]
        score = sum(float(w[k]) * preprocess.batch_z(comp[k], b) for k in used)
        return dict(score=score, used=used, **comp)

    # ---------------------------------------------------------------- prediction
    def predict(self, x, w_global=None, mode=None):
        """x: dict from scfragtrace.load_inputs (or a folder path).
        mode: "label_free" (class score = label-free prior; the paper's setting for new cohorts) or "full" (prior +
        graph-network residual + w * global log-probability); default: the bundle's manifest (tissue_of_origin.default_mode).
        w_global overrides the fusion weight of the manifest in full mode."""
        if isinstance(x, (str, Path)):
            from .inputs import load_inputs
            x = load_inputs(x)
        w = self.w_global if w_global is None else float(w_global)
        node = module_node_features(x["module_counts"], x["n_frag_total"])
        mode = mode or self.too_mode
        if mode not in ("label_free", "full"):
            raise ValueError("mode must be 'label_free' or 'full'")
        T = self.tissue_of_origin(x, node); glp, has_glob = self.global_logp(x); D = self.detection(x, node)
        if mode == "label_free":
            w = 0.0; T["gnn_residual"] = np.zeros_like(T["PRIOR"]); T["final"] = T["PRIOR"]
        fused = T["final"] + w * glp
        P = np.exp(fused - fused.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
        terms = {"footprint": T["T_fp"], "copy_number": T["T_cn"], "gnn": T["gnn_residual"], "global": w * glp}
        pred = fused.argmax(1); n = len(pred)
        cen = {k: v - v.mean(1, keepdims=True) for k, v in terms.items()}
        A = np.stack([np.abs(cen[k][np.arange(n), pred]) for k in TERMS], 1); sh = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
        order = np.argsort(-fused, 1)
        tab = pd.DataFrame(dict(sample_id=x["ids"], batch=x["batch"], tumour_fraction=x["tf"], detection_score=D["score"],
                                predicted_class=[self.classes[i] for i in pred], second_class=[self.classes[i] for i in order[:, 1]]))
        for j, c in enumerate(self.classes):
            tab[f"P_{c}"] = P[:, j]
        for j, k in enumerate(TERMS):
            tab[f"share_{k}"] = sh[:, j]
        tab["detection_module_lr"] = D["module_lr"]; tab["detection_gnn"] = D["gnn"]
        tab["detection_global_lr"] = D["global_lr"]; tab["detection_delfi_lr"] = D["delfi_lr"]
        tab["detection_components"] = "+".join(D["used"]); tab["tissue_mode"] = mode
        scores = dict(prior=T["PRIOR"], gnn_residual=T["gnn_residual"], final=T["final"], global_logp=glp, fused=fused)
        return Prediction(tab, scores, terms)


def predict_prior_only(x, reference_dir=None):
    """Label-free part of the model only (no trained weights needed): PRIOR = zs(TFx*TS) + zs(CNA) from the reference files.
    Returns a Prediction whose final score is the prior (GNN residual 0, global weight 0, no detection score)."""
    if isinstance(x, (str, Path)):
        from .inputs import load_inputs
        x = load_inputs(x)
    ref = Path(reference_dir or WEIGHTS_ROOT / "reference")
    T = pd.read_csv(ref / "tcga_arm_templates.csv", index_col=0); fp = _npz(ref / "footprint_pooled_reference.npz")
    tf = x["tf"]; S = np.nan_to_num(priors.lineage_footprint(x["C"], x["F"], x["set_names"], x["batch"], ~x["ref"], fallback=(fp["coef"], fp["resid_sd"])))
    CP = priors.corr_t(x["arm"][list(T.columns)].values.astype(float), T.loc[priors.CLASSES].values)
    PRIOR = priors.label_free_prior(tf, S, CP); K = len(priors.CLASSES); n = len(tf); zero = np.zeros((n, K))
    terms = {"footprint": priors.footprint_term(tf, S), "copy_number": priors.zs(CP), "gnn": zero, "global": zero}
    P = np.exp(PRIOR - PRIOR.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); pred = PRIOR.argmax(1); order = np.argsort(-PRIOR, 1)
    cen = {k: v - v.mean(1, keepdims=True) for k, v in terms.items()}
    A = np.stack([np.abs(cen[k][np.arange(n), pred]) for k in TERMS], 1); sh = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    tab = pd.DataFrame(dict(sample_id=x["ids"], batch=x["batch"], tumour_fraction=tf, detection_score=np.nan,
                            predicted_class=[priors.CLASSES[i] for i in pred], second_class=[priors.CLASSES[i] for i in order[:, 1]]))
    for j, c in enumerate(priors.CLASSES):
        tab[f"P_{c}"] = P[:, j]
    for j, k in enumerate(TERMS):
        tab[f"share_{k}"] = sh[:, j]
    scores = dict(prior=PRIOR, gnn_residual=zero, final=PRIOR, global_logp=zero, fused=PRIOR)
    return Prediction(tab, scores, terms)


def load_model(bundle="cristiano_only", weights_root=None, **kw):
    """Load a trained bundle by name (folder under model/weights) or by path."""
    p = Path(bundle)
    if not (p / "manifest.json").exists():
        p = Path(weights_root or WEIGHTS_ROOT) / bundle
    if not (p / "manifest.json").exists():
        raise FileNotFoundError(f"no manifest.json in {p}")
    return ScFragTrace(p, **kw)
