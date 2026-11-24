import numpy as np
import pandas as pd
import pytest

from serving.predict_batch import InputError, main, score_file
from serving.predictor import Predictor
from src import config as C
from src.features import add_features
from src.monitoring import build_reference, compute_drift, psi_from_props
from tests.conftest import make_synthetic


def _csv(tmp_path, df, name="in.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_batch_scores_good_rows_and_rejects_bad_ones(tmp_path, tiny_artifacts, synthetic):
    good = synthetic.head(50).drop(columns=["isFraud"]).copy()
    good.insert(0, "transaction_id", [f"id{i}" for i in range(50)])
    bad = pd.DataFrame([{"transaction_id": "bad1", "step": 5, "type": "CRYPTO", "amount": 10.0},
                        {"transaction_id": "bad2", "step": 5, "type": "TRANSFER", "amount": -4.0}])
    src = _csv(tmp_path, pd.concat([good, bad], ignore_index=True))
    out = tmp_path / "scores.csv"
    stats = score_file(src, out, chunksize=20, predictor=Predictor(artifact_dir=tiny_artifacts))
    assert stats["rows_in"] == 52 and stats["scored"] == 50 and stats["rejected"] == 2
    scored = pd.read_csv(out)
    assert list(scored.columns) == ["transaction_id", "fraud_probability", "is_fraud", "mode", "needs_review"]
    assert set(scored["transaction_id"]) == set(good["transaction_id"])
    assert scored["fraud_probability"].between(0, 1).all()
    rejects = pd.read_csv(tmp_path / "scores.rejects.csv")
    assert set(rejects["reason"]) == {"unknown_type", "bad_amount"}


def test_batch_matches_row_by_row_scoring(tmp_path, tiny_artifacts, synthetic):
    sub = synthetic.head(30).drop(columns=["isFraud"]).copy()
    sub.insert(0, "transaction_id", range(30))
    out = tmp_path / "o.csv"
    pred = Predictor(artifact_dir=tiny_artifacts)
    score_file(_csv(tmp_path, sub), out, predictor=pred)
    batch = pd.read_csv(out).set_index("transaction_id")["fraud_probability"]
    single = [pred.score_frame(sub.iloc[[i]])["fraud_probability"].iloc[0] for i in range(30)]
    assert np.allclose(batch.sort_index().to_numpy(), single, atol=1e-5)


def test_missing_required_column_is_input_error(tmp_path):
    p = _csv(tmp_path, pd.DataFrame({"transaction_id": [1], "amount": [5.0]}))
    with pytest.raises(InputError):
        score_file(p, tmp_path / "o.csv")
    assert main(["--input", str(p), "--output", str(tmp_path / "o.csv")]) == 2


def test_psi_basics():
    same = [0.1] * 10
    assert psi_from_props(same, same) == pytest.approx(0, abs=1e-9)
    assert psi_from_props(same, [0.5] + [0.5 / 9] * 9) > 0.25


def test_drift_quiet_on_same_distribution_and_loud_on_shift():
    ref_df = make_synthetic(6000, seed=1)
    ref_df = ref_df[ref_df["type"].isin(C.SCOPE_TYPES)]
    ref_feat = add_features(ref_df)
    ref = build_reference(ref_feat)
    other = make_synthetic(6000, seed=2)
    same = add_features(other[other["type"].isin(C.SCOPE_TYPES)])
    quiet = compute_drift(ref, ref_feat, same)
    assert quiet["n_alert"] == 0 and not quiet["retrain_recommended"]

    raw3 = make_synthetic(6000, seed=3)
    shifted_raw = raw3[raw3["type"].isin(C.SCOPE_TYPES)].copy()
    for col in ["amount"] + C.BALANCE_COLS:
        shifted_raw[col] *= 8
    loud = compute_drift(ref, ref_feat, add_features(shifted_raw))
    assert loud["features"]["amount_log"]["status"] == "alert"
    assert loud["retrain_recommended"]
