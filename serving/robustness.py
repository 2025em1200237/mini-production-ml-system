"""Edge-case checks for the inference path. Writes reports/robustness.json.

    python -m serving.robustness
"""
import json
import logging

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src import config as C
from .app import app

log = logging.getLogger("robustness")

BASE = {"transaction_id": "t-1", "step": 700, "type": "TRANSFER", "amount": 181000.0,
        "oldbalanceOrg": 181000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0}


def _case(name, expect, **overrides):
    return {"name": name, "expect": expect, "payload": {**BASE, **overrides}}


def run():
    cases = [
        _case("typical fraud-like transfer", 200),
        _case("all balances missing", 200, oldbalanceOrg=None, newbalanceOrig=None,
              oldbalanceDest=None, newbalanceDest=None),
        _case("destination balances missing", 200, oldbalanceDest=None, newbalanceDest=None),
        _case("zero amount", 200, amount=0.0),
        _case("extreme amount (9.9e9)", 200, amount=9.9e9, oldbalanceOrg=9.9e9),
        _case("PAYMENT (out of scope)", 200, type="PAYMENT"),
        _case("lowercase type normalised", 200, type="cash_out"),
        _case("unseen category 'CRYPTO'", 422, type="CRYPTO"),
        _case("negative amount", 422, amount=-5.0),
        _case("amount is a string", 422, amount="abc"),
        _case("NaN-like string amount", 422, amount="NaN"),
        _case("negative balance", 422, oldbalanceOrg=-1.0),
        _case("step = 0", 422, step=0),
        _case("empty transaction_id", 422, transaction_id=""),
    ]
    rows = []
    with TestClient(app, raise_server_exceptions=False) as client:
        for c in cases:
            r = client.post("/predict", json=c["payload"])
            body = r.json()
            rows.append({"case": c["name"], "expected_status": c["expect"], "status": r.status_code,
                         "mode": body.get("mode"), "fraud_probability": body.get("fraud_probability"),
                         "passed": r.status_code == c["expect"]})

        r = client.post("/predict", json={**BASE, "oldbalanceOrg": None})
        body = r.json()
        rows.append({"case": "missing sender balance -> needs_review flagged", "expected_status": 200,
                     "status": r.status_code, "mode": body.get("mode"),
                     "fraud_probability": body.get("fraud_probability"),
                     "passed": r.status_code == 200 and body.get("needs_review") is True})

        bad = dict(BASE); bad["unexpected"] = 1
        r = client.post("/predict", json=bad)
        rows.append({"case": "unknown extra field", "expected_status": 422, "status": r.status_code,
                     "mode": None, "fraud_probability": None, "passed": r.status_code == 422})
        missing = {k: v for k, v in BASE.items() if k != "amount"}
        r = client.post("/predict", json=missing)
        rows.append({"case": "required field missing", "expected_status": 422, "status": r.status_code,
                     "mode": None, "fraud_probability": None, "passed": r.status_code == 422})
        r = client.post("/predict", content="not json", headers={"content-type": "application/json"})
        rows.append({"case": "malformed JSON body", "expected_status": 422, "status": r.status_code,
                     "mode": None, "fraud_probability": None, "passed": r.status_code == 422})
        r = client.post("/predict/batch", json={"records": [BASE] * 1001})
        rows.append({"case": "batch of 1001 (limit 1000)", "expected_status": 422, "status": r.status_code,
                     "mode": None, "fraud_probability": None, "passed": r.status_code == 422})

        # failure injection: force the scoring path to fail and check graceful degradation
        pred = app.state.predictor
        was = pred.force_fallback
        pred.force_fallback = True
        r = client.post("/predict", json=BASE)
        b = r.json()
        rows.append({"case": "model failure -> heuristic fallback", "expected_status": 200,
                     "status": r.status_code, "mode": b.get("mode"), "fraud_probability": b.get("fraud_probability"),
                     "passed": r.status_code == 200 and b.get("mode") == "fallback_rule"})
        h = client.get("/health").json()
        rows.append({"case": "/health reports degraded during failure", "expected_status": 200,
                     "status": 200, "mode": h["status"], "fraud_probability": None,
                     "passed": h["status"] == "degraded"})
        pred.force_fallback = was

        # the model path itself must tolerate an unseen category (batch jobs bypass the API schema)
        frame = pd.DataFrame([{**BASE, "type": "CRYPTO"}]).drop(columns=["transaction_id"])
        try:
            res = pred.score_frame(frame)
            ok = bool(np.isfinite(res["fraud_probability"]).all())
            rows.append({"case": "model path: unseen category (one-hot ignore)", "expected_status": 200,
                         "status": 200, "mode": res["mode"].iloc[0],
                         "fraud_probability": float(res["fraud_probability"].iloc[0]), "passed": ok})
        except Exception as exc:  # report, do not hide
            rows.append({"case": "model path: unseen category (one-hot ignore)", "expected_status": 200,
                         "status": 500, "mode": str(exc)[:80], "fraud_probability": None, "passed": False})

        # outlier scaling must keep probabilities inside [0, 1]
        scales = [1, 10, 100, 1e4]
        probs = []
        for s in scales:
            f = pd.DataFrame([{**BASE, "amount": BASE["amount"] * s, "oldbalanceOrg": BASE["oldbalanceOrg"] * s}])
            probs.append(float(pred.score_frame(f.drop(columns=["transaction_id"]))["fraud_probability"].iloc[0]))
        rows.append({"case": "amount x1,10,100,1e4 stays in [0,1]", "expected_status": 200, "status": 200,
                     "mode": "model", "fraud_probability": probs, "passed": all(0 <= p <= 1 for p in probs)})

    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (C.REPORT_DIR / "robustness.json").write_text(json.dumps(rows, indent=2))
    n_ok = sum(r["passed"] for r in rows)
    log.info("robustness: %d/%d cases behaved as specified", n_ok, len(rows))
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
