from tests.conftest import VALID


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] and body["model_version"] == "test-model"


def test_predict_returns_contract(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"transaction_id", "fraud_probability", "is_fraud", "decision_threshold",
                         "mode", "model_version", "needs_review", "warnings"}
    assert body["needs_review"] is False
    assert 0.0 <= body["fraud_probability"] <= 1.0 and body["mode"] == "model"
    assert "X-Process-Time-ms" in r.headers


def test_missing_balances_scored_with_warning(client):
    r = client.post("/predict", json={**VALID, "oldbalanceDest": None, "newbalanceDest": None})
    assert r.status_code == 200
    assert any("missing_fields_imputed" in w for w in r.json()["warnings"])


def test_missing_sender_balance_requests_review(client):
    r = client.post("/predict", json={**VALID, "oldbalanceOrg": None})
    body = r.json()
    assert r.status_code == 200 and body["needs_review"] is True
    assert "sender_balance_missing_review_recommended" in body["warnings"]
    assert client.get("/stats").json()["review_recommended"] >= 1


def test_out_of_scope_type_never_needs_review(client):
    r = client.post("/predict", json={**VALID, "type": "PAYMENT", "oldbalanceOrg": None})
    assert r.json()["needs_review"] is False


def test_malformed_payloads_get_422(client):
    for bad in ({**VALID, "amount": -1}, {**VALID, "type": "CRYPTO"}, {}, {**VALID, "extra": 1}):
        assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", content="nope", headers={"content-type": "application/json"}).status_code == 422


def test_out_of_scope_type_is_rule_scored_zero(client):
    r = client.post("/predict", json={**VALID, "type": "PAYMENT"})
    assert r.json()["mode"] == "out_of_scope_rule" and r.json()["fraud_probability"] == 0.0


def test_fallback_when_model_fails(client):
    client.app.state.predictor.force_fallback = True
    r = client.post("/predict", json={**VALID, "amount": 500000.0, "oldbalanceOrg": 500000.0})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "fallback_rule" and body["is_fraud"] is True
    assert client.get("/health").json()["status"] == "degraded"


def test_batch_endpoint(client):
    r = client.post("/predict/batch", json={"records": [VALID, {**VALID, "transaction_id": "t2"}]})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_stats_counts_requests_and_errors(client):
    client.post("/predict", json=VALID)
    client.post("/predict", json={})
    s = client.get("/stats").json()
    assert s["requests"] >= 2 and s["errors_4xx"] >= 1
