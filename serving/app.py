"""Online scoring service.

    uvicorn serving.app:app --port 8000

Endpoints: GET /health, POST /predict, POST /predict/batch, GET /stats
"""
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .predictor import Predictor
from .schemas import (BatchRequest, BatchResponse, HealthResponse, PredictionResponse,
                      TransactionRequest)

log = logging.getLogger("fraud_api")


class Stats:
    """Tiny in-process counters; a real deployment would export these to Prometheus."""

    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.requests = 0
        self.err_4xx = 0
        self.err_5xx = 0
        self.fallback_predictions = 0
        self.review_recommended = 0
        self.flagged = 0
        self.scored = 0
        self.latency_ms = deque(maxlen=5000)

    def snapshot(self):
        with self.lock:
            lat = np.array(self.latency_ms) if self.latency_ms else np.array([0.0])
            return {
                "requests": self.requests, "errors_4xx": self.err_4xx, "errors_5xx": self.err_5xx,
                "fallback_predictions": self.fallback_predictions,
                "review_recommended": self.review_recommended,
                "alert_rate": self.flagged / self.scored if self.scored else 0.0,
                "latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)),
                               "p99": float(np.percentile(lat, 99))},
            }


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app.state.predictor = Predictor(n_jobs=1)
    app.state.stats = Stats()
    yield


app = FastAPI(title="Fraud scoring service", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def timing_and_counters(request: Request, call_next):
    t0 = time.perf_counter()
    stats: Stats = request.app.state.stats
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error on %s", request.url.path)
        response = JSONResponse({"error": "internal_error", "detail": "unexpected failure"}, status_code=500)
    ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.3f}"
    with stats.lock:
        stats.requests += 1
        stats.latency_ms.append(ms)
        if 400 <= response.status_code < 500:
            stats.err_4xx += 1
        elif response.status_code >= 500:
            stats.err_5xx += 1
    return response


def _to_response(req: TransactionRequest, row, pred: Predictor) -> PredictionResponse:
    warns = []
    missing = req.missing_fields()
    if missing and row["mode"] == "model":
        warns.append("missing_fields_imputed: " + ",".join(missing))
    if row["mode"] == "fallback_rule":
        warns.append("model_unavailable_heuristic_used")
    if bool(row["needs_review"]):
        warns.append("sender_balance_missing_review_recommended")
    return PredictionResponse(
        transaction_id=req.transaction_id,
        fraud_probability=float(row["fraud_probability"]),
        is_fraud=bool(row["is_fraud"]),
        decision_threshold=float(pred.threshold) if pred.threshold is not None else -1.0,
        mode=row["mode"], model_version=pred.version, needs_review=bool(row["needs_review"]),
        warnings=warns,
    )


def _record_stats(request: Request, rows):
    s: Stats = request.app.state.stats
    with s.lock:
        s.scored += len(rows)
        s.flagged += int(rows["is_fraud"].sum())
        s.fallback_predictions += int((rows["mode"] == "fallback_rule").sum())
        s.review_recommended += int(rows["needs_review"].sum())


@app.get("/health", response_model=HealthResponse)
def health(request: Request):
    pred: Predictor = request.app.state.predictor
    return HealthResponse(
        status="ok" if pred.ready else "degraded", model_loaded=pred.model is not None,
        model_version=pred.version, decision_threshold=pred.threshold,
        uptime_seconds=round(time.time() - request.app.state.stats.started, 1),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TransactionRequest, request: Request):
    pred: Predictor = request.app.state.predictor
    rows = pred.predict_records([req.model_dump()])
    _record_stats(request, rows)
    return _to_response(req, rows.iloc[0], pred)


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, request: Request):
    pred: Predictor = request.app.state.predictor
    rows = pred.predict_records([r.model_dump() for r in body.records])
    _record_stats(request, rows)
    out = [_to_response(r, rows.iloc[i], pred) for i, r in enumerate(body.records)]
    return BatchResponse(count=len(out), predictions=out)


@app.get("/stats")
def stats(request: Request):
    return request.app.state.stats.snapshot()
