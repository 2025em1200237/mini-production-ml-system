"""Ingestion, cleaning and leak-free temporal splitting for PaySim."""
import json
import logging
from pathlib import Path

import pandas as pd

from . import config as C

log = logging.getLogger(__name__)

_DTYPES = {
    "step": "int32", "type": "category", "amount": "float64",
    "oldbalanceOrg": "float64", "newbalanceOrig": "float64",
    "oldbalanceDest": "float64", "newbalanceDest": "float64", "isFraud": "int8",
}


def find_data_file(path=None) -> Path:
    if path:
        p = Path(path)
        if p.exists():
            return p
        raise FileNotFoundError(f"data file not found: {p}")
    for p in C.DATA_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "paysim.csv not found. Looked in: " + ", ".join(str(p) for p in C.DATA_CANDIDATES)
    )


def load_raw(path=None) -> pd.DataFrame:
    """Read only the columns we are allowed to use. nameOrig/nameDest (ids) and
    isFlaggedFraud (a rule output from the simulator) are never loaded."""
    p = find_data_file(path)
    usecols = C.RAW_FEATURE_COLS + [C.TARGET]
    df = pd.read_csv(p, usecols=usecols, dtype=_DTYPES)
    log.info("loaded %s rows from %s", f"{len(df):,}", p.name)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "null_cells": int(df.isna().sum().sum()),
        "negative_amounts": int((df["amount"] < 0).sum()),
        "negative_balances": int((df[C.BALANCE_COLS] < 0).any(axis=1).sum()),
        "zero_amount_rows": int((df["amount"] == 0).sum()),
        "fraud_rate_all": float(df[C.TARGET].mean()),
        "fraud_by_type": {
            t: {"rows": int(g.shape[0]), "fraud": int(g[C.TARGET].sum())}
            for t, g in df.groupby("type", observed=True)
        },
        "step_min": int(df["step"].min()),
        "step_max": int(df["step"].max()),
    }


def scope_filter(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"].isin(C.SCOPE_TYPES)].reset_index(drop=True)


def temporal_split(df: pd.DataFrame):
    """Strict time cutoff on `step`. Nothing from the future is visible to an
    earlier partition, which is how the model will actually be deployed."""
    train = df[df["step"] <= C.TRAIN_END_STEP]
    val = df[(df["step"] > C.TRAIN_END_STEP) & (df["step"] <= C.VAL_END_STEP)]
    test = df[df["step"] > C.VAL_END_STEP]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def leakage_audit(train, val, test, feature_cols=None) -> dict:
    """Explicit checks that back the leakage guarantees."""
    feature_cols = feature_cols or C.MODEL_FEATURES
    checks = {
        "train_max_step_lt_val_min_step": bool(train["step"].max() < val["step"].min()),
        "val_max_step_lt_test_min_step": bool(val["step"].max() < test["step"].min()),
        "no_forbidden_columns_in_features": not (set(feature_cols) & C.FORBIDDEN_COLS),
        "raw_frames_have_no_identifier_columns": not ({"nameOrig", "nameDest"} & set(train.columns)),
        "target_not_in_features": C.TARGET not in feature_cols,
        "serving_model_uses_no_post_settlement_features": not (set(feature_cols) & set(C.POST_SETTLEMENT_ONLY)),
        "all_partitions_non_empty": min(len(train), len(val), len(test)) > 0,
    }
    checks["all_passed"] = all(checks.values())
    return checks


def build_splits(path=None, write_summary=True):
    raw = load_raw(path)
    quality = data_quality_report(raw)
    scoped = scope_filter(raw)
    train, val, test = temporal_split(scoped)
    audit = leakage_audit(train, val, test)
    if not audit["all_passed"]:
        raise RuntimeError(f"leakage audit failed: {audit}")

    summary = {"data_quality": quality, "leakage_audit": audit, "splits": {}}
    for name, part in (("train", train), ("val", val), ("test", test)):
        summary["splits"][name] = {
            "rows": int(len(part)),
            "fraud": int(part[C.TARGET].sum()),
            "fraud_rate": float(part[C.TARGET].mean()),
            "step_min": int(part["step"].min()),
            "step_max": int(part["step"].max()),
        }
    if write_summary:
        C.ARTIFACT_DIR.mkdir(exist_ok=True, parents=True)
        (C.ARTIFACT_DIR / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return train, val, test, summary
