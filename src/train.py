"""Fit the preprocessing pipeline, baselines and LightGBM; pick the decision
threshold on validation; serialise everything the serving layer needs.

Run:  python -m src.train            (full run)
      python -m src.train --quick    (smoke test on a subsample, not for reporting)
"""
import argparse
import json
import logging
import platform
import time
import warnings
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from . import config as C
from .data_pipeline import build_splits
from .evaluate import select_threshold_by_cost
from .features import add_features, build_preprocessor
from .monitoring import build_reference

log = logging.getLogger(__name__)
# LightGBM >= 4.7 warns that eval_set is being renamed; the old name still works
warnings.filterwarnings("ignore", message=".*eval_set.*")


def _lgbm(params, metric):
    return lgb.LGBMClassifier(
        n_estimators=500, metric=metric, subsample_freq=1, random_state=C.RANDOM_STATE,
        n_jobs=-1, verbose=-1, **params,
    )


def _fit_lgbm(params, Xt, yt, Xv, yv):
    # average_precision is the metric we care about; fall back to AUC only if the
    # installed LightGBM build does not know it.
    for metric in ("average_precision", "auc"):
        try:
            clf = _lgbm(params, metric)
            clf.fit(Xt, yt, eval_set=[(Xv, yv)],
                    callbacks=[lgb.early_stopping(30, first_metric_only=True, verbose=False)])
            return clf, metric
        except lgb.basic.LightGBMError as e:
            log.warning("metric %s failed (%s)", metric, e)
    raise RuntimeError("LightGBM could not be trained")


def sample_params(rng, neg_pos_ratio):
    return {
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "learning_rate": float(rng.choice([0.03, 0.05, 0.1])),
        "min_child_samples": int(rng.choice([20, 50, 100, 300])),
        "subsample": float(rng.choice([0.7, 0.85, 1.0])),
        "colsample_bytree": float(rng.choice([0.6, 0.8, 1.0])),
        "reg_lambda": float(rng.choice([0.0, 1.0, 10.0])),
        "scale_pos_weight": float(rng.choice([1.0, np.sqrt(neg_pos_ratio), neg_pos_ratio])),
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="subsample negatives; dev smoke test only")
    args = ap.parse_args()
    t0 = time.time()
    C.ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
    rng = np.random.default_rng(C.RANDOM_STATE)

    train, val, test, summary = build_splits()
    del test  # the test partition is not used anywhere in training
    if args.quick:
        neg = train[train[C.TARGET] == 0].sample(200_000, random_state=C.RANDOM_STATE)
        train = pd.concat([train[train[C.TARGET] == 1], neg]).sample(frac=1, random_state=1)
        args.trials = min(args.trials, 2)
        log.warning("QUICK mode: train negatives subsampled, results are not reportable")

    ytr, yv = train[C.TARGET].to_numpy(), val[C.TARGET].to_numpy()
    Xtr_f, Xv_f = add_features(train), add_features(val)

    # fitted on TRAIN only; val/test/inference only ever call .transform
    prep = build_preprocessor().fit(Xtr_f)
    names = list(prep.get_feature_names_out())
    Xtr, Xv = prep.transform(Xtr_f), prep.transform(Xv_f)
    log.info("train %s rows (%d fraud), val %s rows (%d fraud), %d features",
             f"{len(ytr):,}", ytr.sum(), f"{len(yv):,}", yv.sum(), Xtr.shape[1])

    # ---- baseline: class-weighted logistic regression
    logreg = LogisticRegression(class_weight="balanced", max_iter=300, C=1.0)
    logreg.fit(Xtr, ytr)
    lr_ap = average_precision_score(yv, logreg.predict_proba(Xv)[:, 1])
    log.info("logreg val PR-AUC = %.4f", lr_ap)
    joblib.dump(Pipeline([("prep", prep), ("clf", logreg)]), C.ARTIFACT_DIR / C.BASELINE_FILE)

    # ---- LightGBM: small random search, selected on validation PR-AUC
    ratio = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    trials, best = [], None
    for i in range(args.trials):
        params = sample_params(rng, ratio)
        t1 = time.time()
        clf, metric = _fit_lgbm(params, Xtr, ytr, Xv, yv)
        ap_val = average_precision_score(yv, clf.predict_proba(Xv)[:, 1])
        trials.append({"trial": i, **params, "best_iteration": int(clf.best_iteration_ or 0),
                       "val_pr_auc": float(ap_val), "seconds": round(time.time() - t1, 1)})
        log.info("trial %d  val PR-AUC=%.4f  %s", i, ap_val, params)
        if best is None or ap_val > best["ap"]:
            best = {"ap": ap_val, "clf": clf, "params": params, "metric": metric}

    clf = best["clf"]
    p_val = clf.predict_proba(Xv)[:, 1]
    sel = select_threshold_by_cost(yv, p_val, val["amount"].to_numpy())

    # ---- imbalance ablation: same params, no class weighting
    flat_params = {**best["params"], "scale_pos_weight": 1.0}
    flat, _ = _fit_lgbm(flat_params, Xtr, ytr, Xv, yv)
    p_flat = flat.predict_proba(Xv)[:, 1]
    sel_flat = select_threshold_by_cost(yv, p_flat, val["amount"].to_numpy())
    ablation = {
        "weighted": {"scale_pos_weight": best["params"]["scale_pos_weight"], "val_pr_auc": float(best["ap"]),
                     "val_min_cost": sel["val_cost"]},
        "unweighted": {"scale_pos_weight": 1.0, "val_pr_auc": float(average_precision_score(yv, p_flat)),
                       "val_min_cost": sel_flat["val_cost"]},
    }

    # ---- feature-availability ablation: same params, but allowed to see the
    # post-transaction balances. Not served; it shows how much of the signal in
    # PaySim comes from simulator bookkeeping rather than from anything a live
    # system could know at authorisation time.
    prep_post = build_preprocessor("post_settlement").fit(Xtr_f)
    post_clf, _ = _fit_lgbm(best["params"], prep_post.transform(Xtr_f), ytr, prep_post.transform(Xv_f), yv)
    post_ap = average_precision_score(yv, post_clf.predict_proba(prep_post.transform(Xv_f))[:, 1])
    joblib.dump(Pipeline([("prep", prep_post), ("clf", post_clf)]), C.ARTIFACT_DIR / C.ABLATION_FILE, compress=3)
    log.info("post-settlement ablation val PR-AUC = %.4f (pre-auth champion %.4f)", post_ap, best["ap"])

    # ---- hard-mode ablation: hide the sender's balance so the exact-drain
    # signature (amount == oldbalanceOrg) is unavailable
    prep_blind = build_preprocessor("sender_blind").fit(Xtr_f)
    blind_clf, _ = _fit_lgbm(best["params"], prep_blind.transform(Xtr_f), ytr, prep_blind.transform(Xv_f), yv)
    blind_ap = average_precision_score(yv, blind_clf.predict_proba(prep_blind.transform(Xv_f))[:, 1])
    joblib.dump(Pipeline([("prep", prep_blind), ("clf", blind_clf)]), C.ARTIFACT_DIR / C.BLIND_FILE, compress=3)
    log.info("sender-blind ablation val PR-AUC = %.4f", blind_ap)

    model = Pipeline([("prep", prep), ("clf", clf)])
    joblib.dump(model, C.ARTIFACT_DIR / C.MODEL_FILE, compress=3)

    version = "lgbm-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    (C.ARTIFACT_DIR / C.THRESHOLD_FILE).write_text(json.dumps({
        "threshold": sel["threshold"], "model_version": version,
        "selection_rule": "minimise expected cost on validation",
        "review_cost": C.REVIEW_COST, "friction_rate": C.FRICTION_RATE,
        "val_pr_auc": float(best["ap"]), "val_cost": sel["val_cost"],
        "val_no_model_cost": sel["val_no_model_cost"],
    }, indent=2))

    gains = clf.booster_.feature_importance(importance_type="gain")
    (C.ARTIFACT_DIR / "feature_importance.json").write_text(
        json.dumps({n: float(g) for n, g in zip(names, gains)}, indent=2))

    # reference distributions for drift monitoring (train features, val scores)
    ref = build_reference(Xtr_f, p_val)
    (C.ARTIFACT_DIR / C.REFERENCE_STATS_FILE).write_text(json.dumps(ref))
    Xtr_f.sample(min(20_000, len(Xtr_f)), random_state=C.RANDOM_STATE).to_csv(
        C.ARTIFACT_DIR / C.REFERENCE_SAMPLE_FILE, index=False)

    (C.ARTIFACT_DIR / "train_summary.json").write_text(json.dumps({
        "model_version": version, "quick_mode": args.quick,
        "train_rows": int(len(ytr)), "train_fraud_rate": float(ytr.mean()),
        "best_params": best["params"], "best_iteration": int(clf.best_iteration_ or 0),
        "early_stopping_metric": best["metric"],
        "val_pr_auc": {"logreg": float(lr_ap), "lightgbm": float(best["ap"]),
                       "lightgbm_post_settlement_ablation": float(post_ap),
                       "lightgbm_sender_blind_ablation": float(blind_ap)},
        "imbalance_ablation": ablation, "trials": trials,
        "feature_names": names, "train_seconds": round(time.time() - t0, 1),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "lightgbm": lgb.__version__, "numpy": np.__version__},
        "split_summary": summary["splits"],
    }, indent=2))
    log.info("done in %.0fs  threshold=%.5f  version=%s", time.time() - t0, sel["threshold"], version)


if __name__ == "__main__":
    main()
