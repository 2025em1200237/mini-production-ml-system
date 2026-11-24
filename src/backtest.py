"""Rolling-origin backtest: is the result stable over time, or an accident of one split?

Run:  python -m src.backtest        (a couple of minutes)

Time is cut into blocks of 75 hours. For each fold the model is retrained on all
earlier blocks, its threshold is picked on the next block, and it is scored on the
block after that, so every test block lies strictly in the future of everything
used to build the model. Hyper-parameters are the ones selected by train.py
(class weight scaled to each fold's own imbalance). The last two test blocks
overlap the main experiment's test window (hours 601-743); the earlier three
are new evidence, so this complements that single split rather than replacing it.
"""
import json
import logging

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .data_pipeline import load_raw, scope_filter  # noqa: E402
from .evaluate import metrics_at_threshold, select_threshold_by_cost  # noqa: E402
from .features import add_features, build_preprocessor, drain_rule_score  # noqa: E402
from .train import _fit_lgbm  # noqa: E402

log = logging.getLogger(__name__)
BLOCK = 75


def _steps(b):
    return [b * BLOCK + 1, min((b + 1) * BLOCK, 743)]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ts = json.loads((C.ARTIFACT_DIR / "train_summary.json").read_text())
    params = dict(ts["best_params"])
    r = ts["train_fraud_rate"]
    weight_multiplier = params["scale_pos_weight"] / ((1 - r) / r)

    scoped = scope_filter(load_raw())
    X = add_features(scoped)
    y = scoped[C.TARGET].to_numpy()
    amt = scoped["amount"].to_numpy()
    blk = (scoped["step"].to_numpy() - 1) // BLOCK
    drain = drain_rule_score(scoped)

    folds = []
    for k in range(3, 8):                      # train blocks 0..k, val k+1, test k+2
        tr, va, te = blk <= k, blk == k + 1, blk == k + 2
        ytr, yva, yte = y[tr], y[va], y[te]
        p = {**params, "scale_pos_weight": float(weight_multiplier * (1 - ytr.mean()) / ytr.mean())}
        row = {"fold": len(folds) + 1, "train_steps": [1, (k + 1) * BLOCK], "val_steps": _steps(k + 1),
               "test_steps": _steps(k + 2), "train_fraud_rate": float(ytr.mean()),
               "test_rows": int(te.sum()), "test_fraud": int(yte.sum()), "test_fraud_rate": float(yte.mean())}

        for name, fs in (("lightgbm", "pre_auth"), ("sender_blind", "sender_blind")):
            prep = build_preprocessor(fs).fit(X.loc[tr])
            clf, _ = _fit_lgbm(p, prep.transform(X.loc[tr]), ytr, prep.transform(X.loc[va]), yva)
            pv = clf.predict_proba(prep.transform(X.loc[va]))[:, 1]
            pe = clf.predict_proba(prep.transform(X.loc[te]))[:, 1]
            res = {"pr_auc": float(average_precision_score(yte, pe))}
            if name == "lightgbm":
                thr = select_threshold_by_cost(yva, pv, amt[va])["threshold"]
                m = metrics_at_threshold(yte, pe, amt[te], thr)
                res.update({k2: m[k2] for k2 in ("precision", "recall", "fp", "fn", "alert_rate",
                                                 "cost_reduction_vs_no_model")})
                res["threshold"] = float(thr)
            row[name] = res

        m = metrics_at_threshold(yte, drain[te], amt[te], 0.5)
        row["drain_rule"] = {"pr_auc": float(average_precision_score(yte, drain[te])),
                             **{k2: m[k2] for k2 in ("precision", "recall", "fp", "fn", "alert_rate",
                                                     "cost_reduction_vs_no_model")}}
        folds.append(row)
        log.info("fold %d test %s: LightGBM PR-AUC %.4f (P %.3f R %.3f) | drain rule PR-AUC %.4f | sender-blind %.4f",
                 row["fold"], row["test_steps"], row["lightgbm"]["pr_auc"], row["lightgbm"]["precision"],
                 row["lightgbm"]["recall"], row["drain_rule"]["pr_auc"], row["sender_blind"]["pr_auc"])

    summary = {
        "block_hours": BLOCK, "folds": folds,
        "lightgbm_pr_auc_min": min(f["lightgbm"]["pr_auc"] for f in folds),
        "lightgbm_pr_auc_mean": float(np.mean([f["lightgbm"]["pr_auc"] for f in folds])),
        "lightgbm_recall_min": min(f["lightgbm"]["recall"] for f in folds),
        "sender_blind_pr_auc_range": [min(f["sender_blind"]["pr_auc"] for f in folds),
                                      max(f["sender_blind"]["pr_auc"] for f in folds)],
    }
    (C.REPORT_DIR / "backtest.json").write_text(json.dumps(summary, indent=2))

    x = np.arange(len(folds))
    plt.figure(figsize=(6.4, 3.6))
    for i, (key, lab) in enumerate((("lightgbm", "LightGBM (served)"), ("drain_rule", "exact-drain rule"),
                                    ("sender_blind", "sender balance hidden"))):
        plt.bar(x + (i - 1) * 0.26, [f[key]["pr_auc"] for f in folds], 0.26, label=lab)
    plt.plot(x, [f["test_fraud_rate"] for f in folds], "k--", marker="o", ms=3, label="no-skill (prevalence)")
    plt.xticks(x, [f"{f['test_steps'][0]}-{f['test_steps'][1]}" for f in folds])
    plt.xlabel("Test block (hours)"); plt.ylabel("PR-AUC"); plt.ylim(0, 1.05)
    plt.title("Rolling-origin backtest: five forward folds"); plt.legend(fontsize=7, loc="lower left")
    plt.grid(alpha=.3, axis="y")
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "backtest.png", dpi=160); plt.close()
    log.info("wrote reports/backtest.json and figures/backtest.png")


if __name__ == "__main__":
    main()
