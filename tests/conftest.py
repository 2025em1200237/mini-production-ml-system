import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.features import add_features, build_preprocessor  # noqa: E402


def make_synthetic(n=4000, seed=0):
    """Small PaySim-shaped frame. Amounts drift upward over time so that a
    'fit on everything' mistake would be visible in tests."""
    rng = np.random.default_rng(seed)
    step = rng.integers(1, 744, n)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n, p=[.3, .3, .2, .2])
    amount = rng.lognormal(9 + step / 400, 1.2, n)
    old_o = amount * rng.uniform(1, 3, n)
    fraud = (rng.random(n) < 0.06) & np.isin(types, ["TRANSFER", "CASH_OUT"])
    new_o = np.where(fraud, 0.0, old_o - amount)
    old_d = np.where(fraud, 0.0, rng.lognormal(10, 1, n))
    new_d = np.where(fraud, 0.0, old_d + amount)
    return pd.DataFrame({
        "step": step, "type": types, "amount": amount, "oldbalanceOrg": old_o,
        "newbalanceOrig": new_o, "oldbalanceDest": old_d, "newbalanceDest": new_d,
        "isFraud": fraud.astype(int),
    })


@pytest.fixture(scope="session")
def synthetic():
    return make_synthetic()


@pytest.fixture(scope="session")
def tiny_artifacts(tmp_path_factory, synthetic):
    d = tmp_path_factory.mktemp("artifacts")
    scoped = synthetic[synthetic["type"].isin(C.SCOPE_TYPES)]
    X = add_features(scoped)
    prep = build_preprocessor().fit(X)
    clf = lgb.LGBMClassifier(n_estimators=30, num_leaves=7, verbose=-1, n_jobs=1, random_state=0)
    clf.fit(prep.transform(X), scoped["isFraud"])
    joblib.dump(Pipeline([("prep", prep), ("clf", clf)]), d / C.MODEL_FILE)
    (d / C.THRESHOLD_FILE).write_text(json.dumps({"threshold": 0.5, "model_version": "test-model"}))
    return d


@pytest.fixture()
def client(tiny_artifacts, monkeypatch):
    monkeypatch.setenv("FRAUD_ARTIFACT_DIR", str(tiny_artifacts))
    monkeypatch.delenv("FRAUD_FORCE_FALLBACK", raising=False)
    from serving.app import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


VALID = {"transaction_id": "t1", "step": 10, "type": "TRANSFER", "amount": 1000.0,
         "oldbalanceOrg": 1000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}
