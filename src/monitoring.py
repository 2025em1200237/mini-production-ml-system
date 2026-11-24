"""Drift monitoring: PSI and KS against reference statistics saved at training time.

Library use (from train.py / a scheduled job):
    ref = build_reference(X_train_features, reference_scores)
    report = compute_drift(ref, reference_sample_df, X_live_features, live_scores)

CLI:
    python -m src.monitoring --simulate              # train->val, train->test, injected shock
    python -m src.monitoring --input new_batch.csv   # raw PaySim-style rows vs. reference
"""
import argparse
import json
import logging

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402

log = logging.getLogger(__name__)
EPS = 1e-4


def psi_from_props(expected, actual, eps=EPS):
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_edges(x: pd.Series, bins=10):
    qs = np.unique(np.nanquantile(x.to_numpy(dtype=float), np.linspace(0, 1, bins + 1)[1:-1]))
    return qs.tolist()


def _numeric_props(x: pd.Series, edges):
    idx = np.searchsorted(np.asarray(edges), x.to_numpy(dtype=float), side="right")
    counts = np.bincount(idx[~np.isnan(x.to_numpy(dtype=float))], minlength=len(edges) + 1)
    return counts / max(counts.sum(), 1)


def build_reference(features: pd.DataFrame, scores=None) -> dict:
    ref = {"numeric": {}, "categorical": {}, "score": None}
    for col in C.NUM_FEATURES:
        edges = _numeric_edges(features[col])
        ref["numeric"][col] = {"edges": edges, "props": _numeric_props(features[col], edges).tolist()}
    for col in C.FLAG_FEATURES + C.CAT_FEATURES:
        vc = features[col].astype(str).value_counts(normalize=True)
        ref["categorical"][col] = {str(k): float(v) for k, v in vc.items()}
    if scores is not None:
        s = pd.Series(np.asarray(scores, dtype=float))
        edges = _numeric_edges(s)
        ref["score"] = {"edges": edges, "props": _numeric_props(s, edges).tolist()}
    return ref


def _cat_psi(ref_props: dict, live: pd.Series) -> float:
    live_props = live.astype(str).value_counts(normalize=True).to_dict()
    levels = sorted(set(ref_props) | set(live_props))   # unseen levels get their own bin
    return psi_from_props([ref_props.get(k, 0.0) for k in levels],
                          [live_props.get(k, 0.0) for k in levels])


def _status(psi, ks=None):
    if psi >= C.PSI_ALERT:
        return "alert"
    if psi >= C.PSI_WARN or (ks is not None and ks >= C.KS_ALERT):
        return "warn"
    return "ok"


def compute_drift(ref: dict, ref_sample: pd.DataFrame, live: pd.DataFrame, live_scores=None) -> dict:
    out = {"features": {}, "n_rows": int(len(live))}
    for col, r in ref["numeric"].items():
        psi = psi_from_props(r["props"], _numeric_props(live[col], r["edges"]))
        ks = float(ks_2samp(ref_sample[col].dropna(), live[col].dropna()).statistic)
        out["features"][col] = {"psi": psi, "ks": ks, "status": _status(psi, ks)}
    for col, r in ref["categorical"].items():
        psi = _cat_psi(r, live[col])
        out["features"][col] = {"psi": psi, "ks": None, "status": _status(psi)}
    if live_scores is not None and ref.get("score"):
        s = ref["score"]
        psi = psi_from_props(s["props"], _numeric_props(pd.Series(live_scores), s["edges"]))
        out["score"] = {"psi": psi, "status": _status(psi)}
    stat = [v["status"] for v in out["features"].values()]
    out["n_alert"], out["n_warn"] = stat.count("alert"), stat.count("warn")
    score_alert = out.get("score", {}).get("status") == "alert"
    out["retrain_recommended"] = bool(out["n_alert"] >= 3 or score_alert)
    return out


def load_reference():
    ref = json.loads((C.ARTIFACT_DIR / C.REFERENCE_STATS_FILE).read_text())
    sample = pd.read_csv(C.ARTIFACT_DIR / C.REFERENCE_SAMPLE_FILE)
    return ref, sample


def _plot(report: dict):
    scen = list(report.keys())
    feats = list(report[scen[0]]["features"].keys())
    x = np.arange(len(feats))
    w = 0.8 / len(scen)
    plt.figure(figsize=(9, 4.2))
    for i, s in enumerate(scen):
        plt.bar(x + i * w, [report[s]["features"][f]["psi"] for f in feats], w, label=s)
    plt.axhline(C.PSI_WARN, color="orange", ls="--", lw=1)
    plt.axhline(C.PSI_ALERT, color="red", ls="--", lw=1)
    plt.yscale("symlog", linthresh=0.01)
    plt.xticks(x + w * (len(scen) - 1) / 2, feats, rotation=60, ha="right", fontsize=8)
    plt.ylabel("PSI (symlog)"); plt.title("Feature drift vs. training reference")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "drift_psi.png", dpi=160); plt.close()


def _plot_timeline(tl):
    x = np.arange(len(tl))
    labels = [f"{t['steps'][0]}-{t['steps'][1]}" for t in tl]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].plot(x, [t["fraud_rate"] for t in tl], marker="o", label="label fraud rate (known late)")
    ax[0].plot(x, [t["alert_rate"] for t in tl], marker="s", label="alert rate (known now)")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_xlabel("Block (hours)"); ax[0].set_ylabel("Share of transactions")
    ax[0].set_title("Alert rate tracks the prior shift", fontsize=9); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].bar(x - 0.2, [t["max_feature_psi"] for t in tl], 0.4, label="max feature PSI")
    ax[1].bar(x + 0.2, [t["score_psi"] for t in tl], 0.4, label="score PSI")
    ax[1].axhline(C.PSI_WARN, color="orange", ls="--", lw=1); ax[1].axhline(C.PSI_ALERT, color="red", ls="--", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_xlabel("Block (hours)"); ax[1].set_title("Drift against the training reference", fontsize=9)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3, axis="y")
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "drift_timeline.png", dpi=160); plt.close()


def simulate():
    import joblib
    from .data_pipeline import build_splits
    from .features import add_features

    _, val, test, _ = build_splits(write_summary=False)
    ref, sample = load_reference()
    model = joblib.load(C.ARTIFACT_DIR / C.MODEL_FILE)

    shocked = test.copy()
    for col in ["amount"] + C.BALANCE_COLS:
        shocked[col] = shocked[col] * 3.0     # e.g. sudden inflation / new high-value segment

    report = {}
    for name, frame in (("train_vs_val", val), ("train_vs_test", test), ("train_vs_test_x3_shock", shocked)):
        feats = add_features(frame)
        scores = model.predict_proba(feats)[:, 1]
        report[name] = compute_drift(ref, sample, feats, scores)
        report[name]["fraud_rate_label"] = float(frame[C.TARGET].mean())
        log.info("%-24s alerts=%d warns=%d retrain=%s", name, report[name]["n_alert"],
                 report[name]["n_warn"], report[name]["retrain_recommended"])
    # ---- what a rolling monitor would have shown over the post-training period:
    # per 75-hour block, label fraud rate (only known later), alert rate (known now),
    # worst feature PSI and score PSI
    thr = json.loads((C.ARTIFACT_DIR / C.THRESHOLD_FILE).read_text())["threshold"]
    later = pd.concat([val, test], ignore_index=True)
    timeline = []
    for lo in range(C.TRAIN_END_STEP + 1, 744, 75):
        hi = min(lo + 74, 743)
        blk = later[(later["step"] >= lo) & (later["step"] <= hi)]
        if len(blk) < 500:
            continue
        feats = add_features(blk)
        scores = model.predict_proba(feats)[:, 1]
        d = compute_drift(ref, sample, feats, scores)
        timeline.append({
            "steps": [lo, hi], "rows": int(len(blk)), "fraud_rate": float(blk[C.TARGET].mean()),
            "alert_rate": float((scores >= thr).mean()),
            "max_feature_psi": float(max(v["psi"] for v in d["features"].values())),
            "score_psi": float(d["score"]["psi"]), "n_alert": d["n_alert"], "n_warn": d["n_warn"]})
    report["timeline"] = timeline
    (C.REPORT_DIR / "drift_report.json").write_text(json.dumps(report, indent=2))
    _plot({k: v for k, v in report.items() if k != "timeline"})
    _plot_timeline(timeline)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--simulate", action="store_true")
    ap.add_argument("--input", help="CSV of raw transactions to compare with the reference")
    args = ap.parse_args()
    if args.simulate:
        simulate()
    elif args.input:
        import joblib
        from .features import add_features
        ref, sample = load_reference()
        raw = pd.read_csv(args.input)
        raw = raw[raw["type"].isin(C.SCOPE_TYPES)]
        feats = add_features(raw)
        scores = joblib.load(C.ARTIFACT_DIR / C.MODEL_FILE).predict_proba(feats)[:, 1]
        print(json.dumps(compute_drift(ref, sample, feats, scores), indent=2))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
