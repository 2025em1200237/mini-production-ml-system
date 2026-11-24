"""Feature engineering shared by training, batch scoring and the API.

add_features() is stateless on purpose: it only does row-wise arithmetic, so
nothing is "fitted" here and it cannot leak. Everything that does learn from
data (imputer medians, scaler, one-hot categories) sits inside the sklearn
ColumnTransformer from build_preprocessor() and is fitted on train only.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C


def _log(x: pd.Series) -> pd.Series:
    # clip so a stray negative value cannot produce NaN/-inf in log1p
    return np.log1p(x.clip(lower=0))


def _signed_log(x: pd.Series) -> pd.Series:
    return np.sign(x) * np.log1p(np.abs(x))


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready frame (C.MODEL_FEATURES) for raw PaySim-style rows."""
    amt = df["amount"].astype("float64")
    oo, no = df["oldbalanceOrg"].astype("float64"), df["newbalanceOrig"].astype("float64")
    od, nd = df["oldbalanceDest"].astype("float64"), df["newbalanceDest"].astype("float64")

    out = pd.DataFrame(index=df.index)
    out["amount_log"] = _log(amt)
    out["old_org_log"] = _log(oo)
    out["new_org_log"] = _log(no)
    out["old_dest_log"] = _log(od)
    out["new_dest_log"] = _log(nd)
    # Balance bookkeeping errors. For an honest transfer both are ~0; in PaySim
    # fraud the destination balance often does not move, so err_dest != 0.
    out["err_orig"] = _signed_log(no + amt - oo)
    out["err_dest"] = _signed_log(od + amt - nd)
    out["amount_to_balance"] = (amt / (oo.clip(lower=0) + 1.0)).clip(upper=10.0)
    out["amount_to_dest_log"] = _log(amt / (od.clip(lower=0) + 1.0))
    out["hour_of_day"] = (df["step"].astype("float64") - 1) % 24
    # NaN comparisons evaluate to False, so a missing balance simply gives flag 0
    out["org_would_empty"] = ((amt >= oo) & (oo > 0)).astype("int8")     # pre-authorisation
    out["dest_empty_before"] = (od == 0).astype("int8")
    out["org_emptied"] = ((no == 0) & (oo > 0)).astype("int8")           # post-settlement
    out["dest_unchanged"] = ((nd == od) & (amt > 0)).astype("int8")
    out["type"] = df["type"].astype(str)
    return out[C.ALL_FEATURES]


def build_preprocessor(feature_set: str = "pre_auth") -> ColumnTransformer:
    """Columns not in the chosen feature set are dropped by the transformer, so
    add_features() can always return the full superset."""
    fs = C.FEATURE_SETS[feature_set]
    num = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        [
            ("num", num, fs["num"]),
            ("flag", "passthrough", fs["flag"]),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), C.CAT_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def drain_rule_score(df: pd.DataFrame) -> np.ndarray:
    """Exact-drain rule: an in-scope transaction that moves exactly the sender's
    whole balance. On PaySim this is a near-perfect rule (see report); it is
    the strong interpretable baseline and the API's fallback when the model is
    unavailable. Uses only pre-authorisation fields. Returns 0/1."""
    in_scope = df["type"].isin(C.SCOPE_TYPES)
    drains = (df["oldbalanceOrg"] > 0) & ((df["amount"] - df["oldbalanceOrg"]).abs() < 0.005)
    return (in_scope & drains).to_numpy().astype("float64")


def heuristic_score(df: pd.DataFrame) -> np.ndarray:
    """Interpretable rule baseline: an in-scope transaction that would empty the
    sender's account and moves a large amount gets flagged. Uses only
    pre-authorisation fields. Returns 0/1."""
    in_scope = df["type"].isin(C.SCOPE_TYPES)
    emptied = (df["amount"] >= df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)
    big = df["amount"] >= C.HEURISTIC_MIN_AMOUNT
    return (in_scope & emptied & big).to_numpy().astype("float64")
