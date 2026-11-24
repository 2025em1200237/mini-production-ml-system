"""Metrics, cost-based threshold selection and the final held-out test report.

Run:  python -m src.evaluate
The test partition is touched only here, after the model and its threshold were
frozen by train.py using the validation partition.
"""
import json
import logging

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, roc_auc_score)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .data_pipeline import build_splits  # noqa: E402
from .features import add_features, drain_rule_score, heuristic_score  # noqa: E402

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# cost model and threshold selection
# --------------------------------------------------------------------------
def total_cost(y, flagged, amount, review_cost=C.REVIEW_COST, friction=C.FRICTION_RATE):
    y, flagged = np.asarray(y).astype(bool), np.asarray(flagged).astype(bool)
    missed = amount[y & ~flagged].sum()
    reviews = review_cost * flagged.sum()
    friction_cost = friction * amount[~y & flagged].sum()
    return float(missed + reviews + friction_cost)


def select_threshold_by_cost(y, score, amount, review_cost=C.REVIEW_COST, friction=C.FRICTION_RATE):
    """Exact cost-minimising threshold from one sorted pass (no grid needed).
    Only cut points between distinct scores are considered."""
    y = np.asarray(y).astype(float)
    score = np.asarray(score, dtype=float)
    amount = np.asarray(amount, dtype=float)
    order = np.argsort(-score, kind="stable")
    s, yy, a = score[order], y[order], amount[order]
    fraud_cum = np.cumsum(a * yy)
    legit_cum = np.cumsum(a * (1 - yy))
    total_fraud = fraud_cum[-1]
    k = np.arange(1, len(s) + 1)
    cost = total_fraud - fraud_cum + review_cost * k + friction * legit_cum
    boundary = np.ones(len(s), dtype=bool)
    boundary[:-1] = s[:-1] != s[1:]
    cand = np.flatnonzero(boundary)
    best_i = cand[np.argmin(cost[cand])]
    no_flag_cost = total_fraud
    if cost[best_i] >= no_flag_cost:
        thr, best_cost = float(np.nextafter(1.0, 2.0)), float(no_flag_cost)
    else:
        nxt = s[best_i + 1] if best_i + 1 < len(s) else 0.0
        thr, best_cost = float((s[best_i] + nxt) / 2.0), float(cost[best_i])
    return {"threshold": thr, "val_cost": best_cost, "val_no_model_cost": float(no_flag_cost),
            "curve_scores": s[cand], "curve_costs": cost[cand]}


def recall_at_precision(y, score, min_precision=0.90):
    p, r, _ = precision_recall_curve(y, score)
    ok = p >= min_precision
    return float(r[ok].max()) if ok.any() else 0.0


def metrics_at_threshold(y, score, amount, thr):
    y = np.asarray(y).astype(int)
    flagged = np.asarray(score) >= thr
    tn, fp, fn, tp = confusion_matrix(y, flagged.astype(int), labels=[0, 1]).ravel()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    cost = total_cost(y, flagged, amount)
    no_model = float(amount[y.astype(bool)].sum())
    return {
        "threshold": float(thr), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "f1_macro": float(f1_score(y, flagged.astype(int), average="macro")),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "alert_rate": float(flagged.mean()),
        "expected_cost": cost, "no_model_cost": no_model,
        "cost_reduction_vs_no_model": float(1 - cost / no_model) if no_model else 0.0,
    }


def budget_metrics(y, score, budgets=(0.01, 0.02, 0.05)):
    """What an analyst team with limited review capacity would see: precision
    when reviewing exactly as many alerts as there are frauds (R-precision) and
    recall when only the top x% of transactions can be reviewed. Skipped for
    binary rules, where ties make 'top k' ambiguous."""
    y = np.asarray(y).astype(int)
    if len(np.unique(score)) <= 2:
        return {"r_precision": None, **{f"recall_at_{int(b * 100)}pct_alerts": None for b in budgets}}
    order = np.argsort(-np.asarray(score), kind="stable")
    ys = y[order]
    n_fraud = int(y.sum())
    out = {"r_precision": float(ys[:n_fraud].mean()) if n_fraud else 0.0}
    for b in budgets:
        k = max(int(round(b * len(y))), 1)
        out[f"recall_at_{int(b * 100)}pct_alerts"] = float(ys[:k].sum() / max(n_fraud, 1))
    return out


def ranking_metrics(y, score):
    return {
        "pr_auc": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(score)) > 1 else 0.5,
        "recall_at_precision_90": recall_at_precision(y, score, 0.90),
        "recall_at_precision_50": recall_at_precision(y, score, 0.50),
        **budget_metrics(y, score),
    }


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _savefig(name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / name, dpi=160)
    plt.close()


def plot_pr_curves(y, scores: dict, prevalence):
    plt.figure(figsize=(6, 4.4))
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y, s)
        plt.plot(r, p, label=f"{name} (AP={average_precision_score(y, s):.3f})", lw=1.8)
    plt.axhline(prevalence, color="grey", ls=":", label=f"no-skill ({prevalence:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.ylim(0, 1.02)
    plt.title("Precision-recall on held-out test (steps 601-743)")
    plt.legend(fontsize=8); plt.grid(alpha=.3)
    _savefig("pr_curves.png")


def plot_cost_curve(sel, chosen_thr):
    plt.figure(figsize=(6, 4))
    plt.plot(sel["curve_scores"], sel["curve_costs"], lw=1.5)
    plt.axvline(chosen_thr, color="crimson", ls="--", label=f"chosen = {chosen_thr:.4f}")
    plt.axhline(sel["val_no_model_cost"], color="grey", ls=":", label="flag nothing")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("Decision threshold (log)"); plt.ylabel("Validation cost (log)")
    plt.title("Cost vs threshold on validation"); plt.legend(fontsize=8); plt.grid(alpha=.3)
    _savefig("cost_curve.png")


def plot_confusion(m):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(4.2, 3.6))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, f"{v:,}", ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["pred legit", "pred fraud"]); plt.yticks([0, 1], ["legit", "fraud"])
    plt.title("Test confusion matrix (LightGBM)")
    _savefig("confusion_matrix.png")


def plot_importance(names, gains):
    order = np.argsort(gains)[::-1][:12][::-1]
    plt.figure(figsize=(6, 4))
    plt.barh(np.array(names)[order], np.array(gains)[order] / np.sum(gains))
    plt.xlabel("Share of total gain"); plt.title("LightGBM feature importance")
    _savefig("feature_importance.png")


def plot_score_hist(y, score):
    plt.figure(figsize=(6, 3.8))
    bins = np.linspace(0, 1, 41)
    plt.hist(score[y == 0], bins=bins, alpha=.7, label="legit")
    plt.hist(score[y == 1], bins=bins, alpha=.7, label="fraud")
    plt.yscale("log"); plt.xlabel("Predicted fraud probability"); plt.ylabel("Count (log)")
    plt.title("Score distribution on test"); plt.legend(); plt.grid(alpha=.3)
    _savefig("score_hist.png")


# --------------------------------------------------------------------------
# uncertainty, calibration and model/rule complementarity
# --------------------------------------------------------------------------
def _pct(a):
    a = np.asarray(a)
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]


def bootstrap_ci(y, scores, thresholds, n_boot=500, seed=0):
    """Row bootstrap of the test window: 95% intervals for PR-AUC, precision and
    recall, plus paired differences between the served model and the drain rule
    (computed on the same resamples, so the difference interval is meaningful)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    n = len(y)
    keys = list(scores)
    acc = {k: {"pr_auc": [], "precision": [], "recall": []} for k in keys}
    d_rec, d_ap = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.sum() == 0:
            continue
        rec, apv = {}, {}
        for k in keys:
            sb = scores[k][idx]
            f = sb >= thresholds[k]
            tp = int((f & (yb == 1)).sum())
            fp = int((f & (yb == 0)).sum())
            fn = int((~f & (yb == 1)).sum())
            apv[k] = average_precision_score(yb, sb)
            rec[k] = tp / max(tp + fn, 1)
            acc[k]["pr_auc"].append(apv[k])
            acc[k]["precision"].append(tp / max(tp + fp, 1))
            acc[k]["recall"].append(rec[k])
        d_rec.append(rec["lightgbm"] - rec["drain_rule"])
        d_ap.append(apv["lightgbm"] - apv["drain_rule"])
    out = {k: {m: _pct(v) for m, v in acc[k].items()} for k in keys}
    out["paired_lightgbm_minus_drain_rule"] = {"recall": _pct(d_rec), "pr_auc": _pct(d_ap)}
    out["n_boot"] = n_boot
    return out


CAL_BINS = [0, 0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99, 1.0000001]


def _reliability(y, p):
    rows, ece, n = [], 0.0, len(y)
    which = np.digitize(p, CAL_BINS[1:-1])
    for b in range(len(CAL_BINS) - 1):
        m = which == b
        if not m.any():
            continue
        rows.append({"bin": f"{CAL_BINS[b]:g}-{min(CAL_BINS[b + 1], 1.0):g}", "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()), "observed": float(y[m].mean())})
        ece += m.sum() / n * abs(p[m].mean() - y[m].mean())
    return rows, float(ece)


def calibration_report(yv, pv, yt, pt):
    """Class weighting inflates raw probabilities. Isotonic regression fitted on the
    validation window shows how far they are from real frequencies on the test window.
    The calibrator is analysis only: the served score is the raw model score."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(pv, yv)
    pc = iso.predict(pt)
    yt = np.asarray(yt).astype(float)
    rows_raw, ece_raw = _reliability(yt, pt)
    rows_cal, ece_cal = _reliability(yt, pc)
    return {
        "brier_raw": float(np.mean((pt - yt) ** 2)), "brier_isotonic": float(np.mean((pc - yt) ** 2)),
        "ece_raw": ece_raw, "ece_isotonic": ece_cal,
        "reliability_raw": rows_raw, "reliability_isotonic": rows_cal,
        "val_fraud_rate": float(np.mean(yv)), "test_fraud_rate": float(yt.mean()),
    }


def plot_calibration(cal):
    plt.figure(figsize=(4.8, 4.0))
    plt.plot([0, 1], [0, 1], color="grey", ls=":", label="perfect")
    for key, lab in (("reliability_raw", "raw score"), ("reliability_isotonic", "isotonic (fit on val)")):
        r = cal[key]
        plt.plot([x["mean_pred"] for x in r], [x["observed"] for x in r], marker="o", ms=4, lw=1.3,
                 label=f"{lab}")
    plt.xlabel("Mean predicted probability (bin)"); plt.ylabel("Observed fraud rate (bin)")
    plt.title("Reliability on test window"); plt.legend(fontsize=7); plt.grid(alpha=.3)
    _savefig("calibration.png")


def complementarity(test, y, model_flag, rule_flag):
    """Where the model and the exact-drain rule agree and differ."""
    y = np.asarray(y).astype(bool)
    model_flag, rule_flag = np.asarray(model_flag).astype(bool), np.asarray(rule_flag).astype(bool)
    rule_missed = y & ~rule_flag
    rule_caught = y & rule_flag
    return {
        "frauds": int(y.sum()),
        "caught_by_both": int((y & rule_flag & model_flag).sum()),
        "caught_only_by_rule": int((y & rule_flag & ~model_flag).sum()),
        "caught_only_by_model": int((y & ~rule_flag & model_flag).sum()),
        "missed_by_both": int((y & ~rule_flag & ~model_flag).sum()),
        "model_false_alarms": int((~y & model_flag).sum()),
        "rule_false_alarms": int((~y & rule_flag).sum()),
        "median_amount_frauds_rule_missed": float(test.loc[rule_missed, "amount"].median()) if rule_missed.any() else None,
        "median_amount_frauds_rule_caught": float(test.loc[rule_caught, "amount"].median()) if rule_caught.any() else None,
        "rule_missed_by_type": {str(t): int(c) for t, c in test.loc[rule_missed, "type"].value_counts().items()},
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def error_analysis(test, score, thr):
    df = test[["type", "amount", C.TARGET]].copy()
    df["flagged"] = score >= thr
    fr = df[df[C.TARGET] == 1]
    by_type = {t: {"fraud": int(len(g)), "caught": int(g["flagged"].sum()),
                   "recall": float(g["flagged"].mean())}
               for t, g in fr.groupby("type", observed=True)}
    edges = np.array(fr["amount"].quantile([0, .25, .5, .75, 1.0]), dtype=float)  # writable copy
    edges[0] -= 1
    edges[-1] += 1
    bucket = pd.cut(fr["amount"], bins=np.unique(edges))
    by_amount = {str(k): {"fraud": int(len(g)), "recall": float(g["flagged"].mean())}
                 for k, g in fr.groupby(bucket, observed=True)}
    return {"by_type": by_type, "by_fraud_amount_quartile": by_amount}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _, val, test, summary = build_splits()
    lgbm = joblib.load(C.ARTIFACT_DIR / C.MODEL_FILE)
    logreg = joblib.load(C.ARTIFACT_DIR / C.BASELINE_FILE)
    post = joblib.load(C.ARTIFACT_DIR / C.ABLATION_FILE)
    blind = joblib.load(C.ARTIFACT_DIR / C.BLIND_FILE)
    thr_info = json.loads((C.ARTIFACT_DIR / C.THRESHOLD_FILE).read_text())

    Xv, Xt = add_features(val), add_features(test)
    yv, yt = val[C.TARGET].to_numpy(), test[C.TARGET].to_numpy()
    av, at = val["amount"].to_numpy(), test["amount"].to_numpy()
    prior = float(json.loads((C.ARTIFACT_DIR / "train_summary.json").read_text())["train_fraud_rate"])

    abl = "lightgbm_post_settlement_ablation"
    bld = "lightgbm_sender_blind_ablation"
    val_scores = {"logreg": logreg.predict_proba(Xv)[:, 1], "lightgbm": lgbm.predict_proba(Xv)[:, 1],
                  abl: post.predict_proba(Xv)[:, 1], bld: blind.predict_proba(Xv)[:, 1]}
    test_scores = {
        "dummy_prior": np.full(len(test), prior),
        "flag_everything": np.ones(len(test)),
        "naive_rule": heuristic_score(test),
        "drain_rule": drain_rule_score(test),
        "logreg": logreg.predict_proba(Xt)[:, 1],
        "lightgbm": lgbm.predict_proba(Xt)[:, 1],
        bld: blind.predict_proba(Xt)[:, 1],
        abl: post.predict_proba(Xt)[:, 1],
    }
    thresholds = {
        "dummy_prior": 1.0,  # flags nothing; kept for the table
        "flag_everything": 0.5,  # flags everything; shows how forgiving the cost metric is
        "naive_rule": 0.5,
        "drain_rule": 0.5,
        "logreg": select_threshold_by_cost(yv, val_scores["logreg"], av)["threshold"],
        "lightgbm": thr_info["threshold"],
        bld: select_threshold_by_cost(yv, val_scores[bld], av)["threshold"],
        abl: select_threshold_by_cost(yv, val_scores[abl], av)["threshold"],
    }

    results = {}
    for name, s in test_scores.items():
        row = ranking_metrics(yt, s)
        row.update(metrics_at_threshold(yt, s, at, thresholds[name]))
        results[name] = row
        log.info("%-15s PR-AUC=%.4f P=%.3f R=%.3f cost_red=%.1f%%", name, row["pr_auc"],
                 row["precision"], row["recall"], 100 * row["cost_reduction_vs_no_model"])

    # alternative operating points for the champion (all picked on validation)
    pv, rv, tv = precision_recall_curve(yv, val_scores["lightgbm"])
    f1v = 2 * pv[:-1] * rv[:-1] / np.clip(pv[:-1] + rv[:-1], 1e-12, None)
    alt = {"f1_max_val": float(tv[np.argmax(f1v)])}
    ok = pv[:-1] >= 0.90
    alt["precision_90_val"] = float(tv[ok][0]) if ok.any() else 1.0
    operating_points = {k: metrics_at_threshold(yt, test_scores["lightgbm"], at, v) | {"threshold": v}
                        for k, v in alt.items()}

    # cost-assumption sensitivity: re-pick threshold on val under each friction rate
    sens = []
    for fr in C.FRICTION_RATES_SENSITIVITY:
        sel = select_threshold_by_cost(yv, val_scores["lightgbm"], av, friction=fr)
        flagged = test_scores["lightgbm"] >= sel["threshold"]
        c = total_cost(yt, flagged, at, friction=fr)
        base = float(at[yt.astype(bool)].sum())
        sens.append({"friction_rate": fr, "threshold": sel["threshold"],
                     "recall": float(flagged[yt == 1].mean()), "alert_rate": float(flagged.mean()),
                     "cost_reduction_vs_no_model": float(1 - c / base)})

    sel_final = select_threshold_by_cost(yv, val_scores["lightgbm"], av)

    log.info("bootstrapping 95%% intervals (500 resamples) ...")
    boot = bootstrap_ci(yt, {k: test_scores[k] for k in ("lightgbm", "drain_rule", "logreg", bld)}, thresholds)
    cal = calibration_report(yv, val_scores["lightgbm"], yt, test_scores["lightgbm"])
    comp = complementarity(test, yt, test_scores["lightgbm"] >= thresholds["lightgbm"],
                           test_scores["drain_rule"] >= 0.5)
    log.info("complementarity: %s", comp)
    out = {
        "bootstrap_95ci": boot, "calibration": cal, "complementarity": comp,
        "split_summary": summary["splits"],
        "test_results": results,
        "operating_points_lightgbm": operating_points,
        "cost_sensitivity": sens,
        "error_analysis": error_analysis(test, test_scores["lightgbm"], thr_info["threshold"]),
        "cost_assumptions": {"review_cost": C.REVIEW_COST, "friction_rate": C.FRICTION_RATE},
    }
    C.REPORT_DIR.mkdir(exist_ok=True, parents=True)
    (C.REPORT_DIR / "metrics.json").write_text(json.dumps(out, indent=2))

    plot_pr_curves(yt, {k: test_scores[k] for k in ("naive_rule", "logreg", "lightgbm", bld)}, float(yt.mean()))
    plot_cost_curve(sel_final, thr_info["threshold"])
    plot_confusion(results["lightgbm"])
    plot_calibration(cal)
    plot_score_hist(yt, test_scores["lightgbm"])
    imp = json.loads((C.ARTIFACT_DIR / "feature_importance.json").read_text())
    plot_importance(list(imp.keys()), list(imp.values()))

    def _f(v, spec=".3f"):
        return "-" if v is None else format(v, spec)

    lines = ["| Model | PR-AUC | ROC-AUC | Recall@P>=0.9 | R-precision | Recall@5% alerts | Precision | Recall | F1 | F1-macro | Alert rate | Cost reduction |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, r in results.items():
        lines.append(f"| {k} | {r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['recall_at_precision_90']:.3f} | "
                     f"{_f(r['r_precision'])} | {_f(r['recall_at_5pct_alerts'])} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {r['f1_macro']:.3f} | "
                     f"{100 * r['alert_rate']:.2f}% | {100 * r['cost_reduction_vs_no_model']:.1f}% |")
    (C.REPORT_DIR / "results_table.md").write_text("\n".join(lines))
    log.info("wrote reports/metrics.json and figures")


if __name__ == "__main__":
    main()
