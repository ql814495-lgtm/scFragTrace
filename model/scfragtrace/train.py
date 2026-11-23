"""Training a weight bundle from a folder of per-sample inputs with labels (the final configuration of the paper).

Extra columns of samples.csv used for training (model/docs/INPUTS.md):
    cancer        1 = cancer, 0 = non-cancer (detection label)
    true_class    one of the 8 classes for tissue-of-origin training samples, empty otherwise
    train_too     True for the tissue-of-origin training rows (one sample per patient; the paper kept the deepest sample)

Recipe (tissue of origin: 'wdat' variant on the single-cell module graph):
  * tissue of origin, seed s = 0, 1, 2: validation cohort = val_order[s % len(val_order)], fit = the other training
    cohorts; node features standardised on fit; residual GNN trained with the wdat loss, transductive distillation on the
    train_too rows of the unlabelled cohorts (their labels are never read); lambda in {0, .25, .5, 1} maximising the
    balanced accuracy of PRIOR + lambda * G on the validation cohort (ties -> smaller lambda).
  * global fragmentomic LR: L2 logistic regression (C = 0.3, balanced) on the batch-recentred global features of the
    tissue-of-origin training rows.
  * detection, seed s: 20% of each (cohort, label) group of the detection training samples as validation; detection GNN
    on batch-standardised node features; module LR (C = 0.01, balanced) on the flattened node features.
Each network is constructed right after torch.manual_seed(seed), so the initialisation is reproducible; in the runs
of the paper it followed the random state of the process."""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__, priors, preprocess
from .model import sha256
from .reference import graph_tensors, lineage_mask, module_node_features

LAMS = [0, 0.25, 0.5, 1.0]


def _bool(col):
    return col.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"]).values


def _lr(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X); m = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, m


def train_bundle(x, out_dir, ref_dir, name, too_train_cohorts, too_val_order, too_unlabelled_cohorts, det_train_cohorts,
                 det_weights, w_global=0.0, global_reference_batch="Cristiano", seeds=(0, 1, 2), threads=1, stages=("too", "global", "det"),
                 description="", log=print, too_val_split="cohort", too_default_mode="full"):
    import torch
    from sklearn.metrics import balanced_accuracy_score
    from . import gnn
    torch.set_num_threads(threads)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True); ref_dir = Path(ref_dir)
    tab = x["samples"]; batch = x["batch"]; tf = x["tf"]; K = len(priors.CLASSES)
    g = np.load(ref_dir / "module_graph.npz"); A, Dh = graph_tensors(g["adj_sc"]); mask, hasmod = lineage_mask(g["module_set"], K)
    TT = pd.read_csv(ref_dir / "tcga_arm_templates.csv", index_col=0).loc[priors.CLASSES]; arm_names = list(TT.columns)
    fp = np.load(ref_dir / "footprint_pooled_reference.npz")
    node = module_node_features(x["module_counts"], x["n_frag_total"])
    S = np.nan_to_num(priors.lineage_footprint(x["C"], x["F"], x["set_names"], batch, ~x["ref"], fallback=(fp["coef"], fp["resid_sd"])))
    CP = priors.corr_t(x["arm"][arm_names].values.astype(float), TT.values)
    PRI_all = priors.prior_inputs(tf, S, CP); PRIOR_all = priors.label_free_prior(tf, S, CP)
    D = np.where(_bool(tab.train_too))[0]
    y = tab.true_class.iloc[D].map({c: i for i, c in enumerate(priors.CLASSES)}).values
    if np.isnan(y.astype(float)).any():
        raise ValueError("train_too rows need a true_class among " + ", ".join(priors.CLASSES))
    y = y.astype(int); coh = batch[D]; tfD = tf[D]; PRI = PRI_all[D]; PRIOR = PRIOR_all[D]
    XN = preprocess.too_node_features(node, batch, x["ref"])[D]
    COH = {c: i for i, c in enumerate(sorted(set(coh)))}; cid = np.array([COH[c] for c in coh])
    arch = dict(node_features=3, d=32, heads=4, n_cohorts_adversary=len(COH), n_modules=int(len(g["module_set"])))
    man = dict(name=name, description=description, package_version=__version__, created=time.strftime("%Y-%m-%d"),
               classes=priors.CLASSES, reference_dir="../reference", architecture=arch, files={})
    outputs = {}
    mf = out / "manifest.json"
    if mf.exists():
        old = json.load(open(mf)); man.update({k: v for k, v in old.items() if k in ("tissue_of_origin", "global_lr", "detection", "fusion", "training")})
    man.setdefault("training", {})
    if "too" in stages:
        man["training"]["tissue_of_origin"] = dict(train_cohorts=list(too_train_cohorts), unlabelled_cohorts=list(too_unlabelled_cohorts),
            n_train=pd.crosstab(coh[np.isin(coh, too_train_cohorts)], np.array(priors.CLASSES)[y[np.isin(coh, too_train_cohorts)]]).to_dict(orient="index"),
            n_unlabelled=int(np.isin(coh, too_unlabelled_cohorts).sum()), cohort_index=COH, seeds=[])
        files, MU, SD, lam, G = [], [], [], [], []
        for s in seeds:
            if too_val_split == "stratified_20pct":            # single training cohort: validation split inside it
                from sklearn.model_selection import train_test_split
                trn = np.where(np.isin(coh, too_train_cohorts))[0]; vc = "stratified 20% of " + "+".join(too_train_cohorts)
                fit, val = train_test_split(trn, test_size=0.2, stratify=y[trn], random_state=s)
            else:                                              # multi-cohort reference design: one training cohort as validation
                vc = too_val_order[s % len(too_val_order)]
                fit = np.where(np.isin(coh, too_train_cohorts) & (coh != vc))[0]; val = np.where(coh == vc)[0]
            unl = np.where(np.isin(coh, too_unlabelled_cohorts))[0]
            mu = XN[fit].mean(0); sd = XN[fit].std(0) + 1e-6; Xs = np.clip((XN - mu) / sd, -5, 5).astype(np.float32)
            torch.manual_seed(s)
            net = gnn.ResidualLineageGNN(A, Dh, 3, mask, hasmod, nco=len(COH))
            t0 = time.time(); gS, ep = gnn.train_wdat(net, Xs, PRI, y, tfD, cid, fit, val, unl, seed=s)
            sc = {l: balanced_accuracy_score(y[val], (PRIOR[val] + l * gS[val]).argmax(1)) for l in LAMS}
            lb = max(LAMS, key=lambda l: (round(sc[l], 3), -l))
            f = f"too_gnn_seed{s}.pt"; torch.save(gnn.state_dict_for_saving(net), out / f); files.append(f)
            MU.append(mu); SD.append(sd); lam.append(lb); G.append(gS)
            man["training"]["tissue_of_origin"]["seeds"].append(dict(seed=s, val_cohort=vc, n_fit=int(len(fit)), n_val=int(len(val)), epochs=int(ep),
                                                                      val_balanced_accuracy={str(k): round(v, 4) for k, v in sc.items()}, chosen_lambda=lb,
                                                                      seconds=round(time.time() - t0)))
            log(f"[too] seed {s} val {vc} fit {len(fit)} epochs {ep} val bal {sc} -> lambda {lb} ({time.time() - t0:.0f}s)")
        np.savez(out / "too_node_standardisation.npz", mu=np.array(MU), sd=np.array(SD))
        man["tissue_of_origin"] = dict(gnn_weights=files, node_standardisation="too_node_standardisation.npz", lambdas=lam,
                                       default_mode=too_default_mode, validation_split=too_val_split,
                                       mode_note="label_free: class score = label-free prior (47 features; lambda = 0, w = 0), the setting of the "
                                                 "paper for new cohorts. full: prior + mean lambda * graph-network residual + w * log p_global.",
                                       recipe="wdat: TFx-weighted CE + 0.5 KL to prior + 0.3 batch adversary (GRL) + distillation towards the prior")
        outputs.update(too_ids=x["ids"][D], too_gnn_seeds=np.array(G), too_prior=PRIOR, too_lambdas=np.array(lam))
    if "global" in stages:
        Xraw = x["global"]; rb = (batch == global_reference_batch) & x["ref"]
        hmean = np.nanmean(Xraw[rb], 0); Xc = preprocess.global_recentre(Xraw, batch, x["ref"], hmean)[D]
        tr = np.where(np.isin(coh, too_train_cohorts))[0]; sc, m = _lr(Xc[tr], y[tr], 0.3)
        np.savez(out / "global_lr.npz", hmean=hmean, mean=sc.mean_, scale=sc.scale_, coef=m.coef_, intercept=m.intercept_, classes=m.classes_)
        man["global_lr"] = dict(file="global_lr.npz", train_cohorts=list(too_train_cohorts), n_train=int(len(tr)), C=0.3,
                                reference_mean_from=f"non-cancer samples of {global_reference_batch}", n_features=int(Xraw.shape[1]))
        man["fusion"] = dict(w_global=float(w_global), note="full mode: score = final + w_global * log p_global; w = 0 in new cohorts")
        outputs.update(global_logp_train=None)
        log(f"[global] LR on {len(tr)} samples, classes {m.classes_.tolist()}")
    if "det" in stages:
        ydet = tab.cancer.astype(int).values; Xd = preprocess.det_node_features(node, batch)
        tr = np.where(np.isin(batch, det_train_cohorts))[0]
        sc, m = _lr(Xd[tr].reshape(len(tr), -1), ydet[tr], 0.01)
        np.savez(out / "det_module_lr.npz", mean=sc.mean_, scale=sc.scale_, coef=m.coef_, intercept=m.intercept_)
        files, info, GD = [], [], []
        for s in seeds:
            r = np.random.default_rng(s); val = []
            for c in np.unique(batch[tr]):
                for lab in (0, 1):
                    ii = tr[(batch[tr] == c) & (ydet[tr] == lab)]
                    if len(ii): val += list(r.choice(ii, max(1, int(round(0.2 * len(ii)))), replace=False))
            val = np.array(sorted(val)); fit = np.setdiff1d(tr, val)
            torch.manual_seed(s); net = gnn.DetectionGNN(A, Dh, 3)
            t0 = time.time(); gd, ep = gnn.train_detection(net, Xd, ydet, fit, val, seed=s)
            f = f"det_gnn_seed{s}.pt"; torch.save(gnn.state_dict_for_saving(net), out / f); files.append(f); GD.append(gd)
            info.append(dict(seed=s, n_fit=int(len(fit)), n_val=int(len(val)), epochs=int(ep), seconds=round(time.time() - t0)))
            log(f"[det] seed {s} fit {len(fit)} val {len(val)} epochs {ep} ({time.time() - t0:.0f}s)")
        man["detection"] = dict(gnn_weights=files, module_lr="det_module_lr.npz", weights=det_weights,
                                note="cancer score = sum_m weight_m * z(component m), z within batch over all its samples (no labels); "
                                     "components: tf (ichorCNA tumour fraction, TFx), module_lr, gnn, global_lr, delfi_lr; a component whose "
                                     "input is absent is left out")
        man["training"]["detection"] = dict(train_cohorts=list(det_train_cohorts), n_train=pd.crosstab(batch[tr], ydet[tr]).rename(columns={0: "non_cancer", 1: "cancer"}).to_dict(orient="index"),
                                            seeds=info)
        if "global" in x and det_weights.get("global_lr", 0):    # global fragmentomics detection LR (raw features, C = 0.3)
            Xg = x["global"].astype(float); imp = np.nanmean(Xg[tr], 0); imp = np.where(np.isfinite(imp), imp, 0.0); Xg = np.where(np.isnan(Xg), imp, Xg)
            scg, mg = _lr(Xg[tr], ydet[tr], 0.3)
            np.savez(out / "det_global_lr.npz", impute=imp, mean=scg.mean_, scale=scg.scale_, coef=mg.coef_, intercept=mg.intercept_)
            man["detection"]["global_lr"] = "det_global_lr.npz"; man["training"]["detection"]["global_lr"] = dict(C=0.3, n_features=int(Xg.shape[1]))
            outputs.update(det_global_lr=mg.decision_function(scg.transform(Xg)))
        if "delfi" in x and det_weights.get("delfi_lr", 0):      # genome-wide coverage score: LR on the DELFI features
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import roc_auc_score
            Zd = preprocess.batch_standardise(x["delfi"], batch); imp = np.nanmean(Zd[tr], 0); imp = np.where(np.isfinite(imp), imp, 0.0)
            Zd = np.where(np.isnan(Zd), imp, Zd); grid = (0.001, 0.01, 0.1, 1.0); cv = {}
            for C in grid:                                        # penalty by 5-fold cross-validation inside the training cohort
                o = np.zeros(len(tr))
                for a_, b_ in StratifiedKFold(5, shuffle=True, random_state=0).split(tr, ydet[tr]):
                    s_, m_ = _lr(Zd[tr[a_]], ydet[tr[a_]], C); o[b_] = m_.predict_proba(s_.transform(Zd[tr[b_]]))[:, 1]
                cv[C] = float(roc_auc_score(ydet[tr], o))
            Cb = max(cv, key=cv.get); scd, md = _lr(Zd[tr], ydet[tr], Cb)
            np.savez(out / "det_delfi_lr.npz", impute=imp, mean=scd.mean_, scale=scd.scale_, coef=md.coef_, intercept=md.intercept_)
            man["detection"]["delfi_lr"] = "det_delfi_lr.npz"
            man["training"]["detection"]["delfi_lr"] = dict(C=Cb, cv_auroc={str(k): round(v, 4) for k, v in cv.items()}, n_features=int(Zd.shape[1]))
            outputs.update(det_delfi_lr=md.predict_proba(scd.transform(Zd))[:, 1])
            log(f"[det] DELFI LR: C by 5-fold CV {cv} -> {Cb}")
        outputs.update(det_ids=x["ids"], det_gnn_seeds=np.array(GD), det_module_lr=((Xd.reshape(len(Xd), -1) - sc.mean_) / sc.scale_) @ m.coef_.ravel() + m.intercept_[0])
    import sklearn, torch as _t
    man["software"] = dict(python_package=__version__, numpy=np.__version__, torch=_t.__version__, scikit_learn=sklearn.__version__, pandas=pd.__version__)
    for f in sorted(p.name for p in out.iterdir() if p.suffix in (".pt", ".npz")):
        man["files"][f] = sha256(out / f)
    for f in sorted(p.name for p in ref_dir.iterdir()):
        man["files"]["../reference/" + f] = sha256(ref_dir / f)
    json.dump(man, open(mf, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return man, outputs
