"""Model loading and scoring, kept separate from the web layer and the CLI so the
API and the batch job are guaranteed to give the same answer for the same row."""
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src.features import add_features, drain_rule_score

log = logging.getLogger(__name__)

FALLBACK_FLAG_PROB = 0.5   # reported probability when the heuristic fires
RAW_COLS = C.RAW_FEATURE_COLS


class Predictor:
    def __init__(self, artifact_dir=None, n_jobs=1, force_fallback=None):
        self.artifact_dir = Path(artifact_dir or os.environ.get("FRAUD_ARTIFACT_DIR", C.ARTIFACT_DIR))
        if force_fallback is None:
            force_fallback = os.environ.get("FRAUD_FORCE_FALLBACK", "0") == "1"
        self.force_fallback = force_fallback
        self.model = None
        self.threshold = None
        self.version = "unavailable"
        self.prior = 0.002
        self._load(n_jobs)

    def _load(self, n_jobs):
        try:
            self.model = joblib.load(self.artifact_dir / C.MODEL_FILE)
            meta = json.loads((self.artifact_dir / C.THRESHOLD_FILE).read_text())
            self.threshold = float(meta["threshold"])
            self.version = meta.get("model_version", "unknown")
            # one-row requests: extra threads only add overhead
            self.model.named_steps["clf"].set_params(n_jobs=n_jobs)
            summ = self.artifact_dir / "train_summary.json"
            if summ.exists():
                self.prior = float(json.loads(summ.read_text()).get("train_fraud_rate", self.prior))
            log.info("model %s loaded, threshold=%.5f", self.version, self.threshold)
        except Exception as exc:  # missing/corrupt artifacts must not kill the service
            log.error("could not load model artifacts (%s); running in fallback mode", exc)
            self.model = None

    @property
    def ready(self) -> bool:
        return self.model is not None and not self.force_fallback

    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """df has the raw columns (balances may be NaN). Returns one row per input."""
        n = len(df)
        prob = np.zeros(n)
        flag = np.zeros(n, dtype=bool)
        mode = np.full(n, "out_of_scope_rule", dtype=object)

        # known non-fraud-prone types never reach the model
        route = ~df["type"].isin(C.OUT_OF_SCOPE_TYPES).to_numpy()
        # Without the sender's balance the model cannot see the drain signature
        # (see report, section 4.2), so a low score is not trustworthy: ask for review.
        needs_review = route & df["oldbalanceOrg"].isna().to_numpy()
        if route.any():
            sub = df.loc[route, RAW_COLS]
            try:
                if not self.ready:
                    raise RuntimeError("model unavailable or fallback forced")
                p = self.model.predict_proba(add_features(sub))[:, 1]
                if not np.all(np.isfinite(p)):
                    raise ValueError("non-finite scores")
                prob[route], flag[route] = p, p >= self.threshold
                mode[route] = "model"
            except Exception as exc:
                log.warning("scoring failed, using heuristic fallback: %s", exc)
                h = drain_rule_score(sub) > 0
                prob[route] = np.where(h, FALLBACK_FLAG_PROB, self.prior)
                flag[route] = h
                mode[route] = "fallback_rule"
        return pd.DataFrame({"fraud_probability": prob, "is_fraud": flag, "mode": mode,
                             "needs_review": needs_review}, index=df.index)

    def predict_records(self, records):
        """records: list of dicts with the request fields."""
        df = pd.DataFrame(records)
        for col in RAW_COLS:
            if col not in df:
                df[col] = np.nan
        return self.score_frame(df)
