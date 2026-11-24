import numpy as np
import pandas as pd

from src import config as C
from src.data_pipeline import leakage_audit, scope_filter, temporal_split
from src.features import add_features, build_preprocessor, heuristic_score


def test_feature_columns_and_finiteness(synthetic):
    X = add_features(synthetic)
    assert list(X.columns) == C.ALL_FEATURES
    assert set(C.MODEL_FEATURES).isdisjoint(C.POST_SETTLEMENT_ONLY)
    num = X[C.NUM_ALL + C.FLAG_ALL]
    assert np.isfinite(num.to_numpy()).all()


def test_features_survive_edge_values():
    df = pd.DataFrame({
        "step": [1, 743], "type": ["TRANSFER", "CASH_OUT"], "amount": [0.0, 9e9],
        "oldbalanceOrg": [np.nan, -3.0], "newbalanceOrig": [0.0, np.nan],
        "oldbalanceDest": [0.0, 1e11], "newbalanceDest": [np.nan, 0.0],
    })
    X = add_features(df)
    assert X.shape[0] == 2
    assert not np.isinf(X[C.NUM_ALL].to_numpy(dtype=float)).any()


def test_org_emptied_flag():
    df = pd.DataFrame({"step": [5, 5], "type": ["TRANSFER"] * 2, "amount": [10.0, 10.0],
                       "oldbalanceOrg": [10.0, 50.0], "newbalanceOrig": [0.0, 40.0],
                       "oldbalanceDest": [0.0, 0.0], "newbalanceDest": [0.0, 10.0]})
    assert add_features(df)["org_emptied"].tolist() == [1, 0]


def test_heuristic_ignores_out_of_scope_types():
    df = pd.DataFrame({"type": ["PAYMENT", "TRANSFER"], "amount": [5e5, 5e5],
                       "oldbalanceOrg": [5e5, 5e5], "newbalanceOrig": [0.0, 0.0]})
    assert heuristic_score(df).tolist() == [0.0, 1.0]


def test_drain_rule_needs_exact_full_balance_and_scope():
    from src.features import drain_rule_score
    df = pd.DataFrame({"type": ["TRANSFER", "TRANSFER", "CASH_OUT", "PAYMENT", "TRANSFER"],
                       "amount": [100.0, 99.0, 50.0, 70.0, 10.0],
                       "oldbalanceOrg": [100.0, 100.0, 50.0, 70.0, 0.0]})
    assert drain_rule_score(df).tolist() == [1.0, 0.0, 1.0, 0.0, 0.0]


def test_sender_blind_set_hides_sender_balance():
    blind = set(C.FEATURE_SETS["sender_blind"]["num"] + C.FEATURE_SETS["sender_blind"]["flag"])
    assert blind.isdisjoint({"old_org_log", "amount_to_balance", "org_would_empty"} | set(C.POST_SETTLEMENT_ONLY))


def test_temporal_split_has_no_overlap(synthetic):
    scoped = scope_filter(synthetic)
    tr, va, te = temporal_split(scoped)
    assert tr["step"].max() <= C.TRAIN_END_STEP < va["step"].min()
    assert va["step"].max() <= C.VAL_END_STEP < te["step"].min()
    assert len(tr) + len(va) + len(te) == len(scoped)
    assert leakage_audit(tr, va, te)["all_passed"]


def test_preprocessor_is_fitted_on_train_only(synthetic):
    scoped = scope_filter(synthetic)
    tr, _, _ = temporal_split(scoped)
    prep = build_preprocessor().fit(add_features(tr))
    scaler = prep.named_transformers_["num"].named_steps["scale"]
    train_mean = add_features(tr)[C.NUM_FEATURES].mean().to_numpy()
    all_mean = add_features(scoped)[C.NUM_FEATURES].mean().to_numpy()
    assert np.allclose(scaler.mean_, train_mean)
    assert not np.allclose(scaler.mean_, all_mean)   # amounts drift with time, so these must differ


def test_unseen_category_does_not_break_transform(synthetic):
    scoped = scope_filter(synthetic)
    prep = build_preprocessor().fit(add_features(scoped))
    odd = scoped.head(3).copy()
    odd["type"] = "CRYPTO"
    out = prep.transform(add_features(odd))
    assert out.shape[0] == 3 and np.isfinite(out).all()
