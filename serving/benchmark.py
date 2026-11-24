"""Empirical latency / throughput profiling.

    python -m serving.benchmark                 # full run
    python -m serving.benchmark --requests 300  # quicker

Starts the real FastAPI app under uvicorn in a subprocess and drives it over
HTTP, so the numbers include serialisation, validation and the network stack
(loopback). Results go to reports/benchmark.json and reports/figures/latency_hist.png.
"""
import argparse
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402
from .predict_batch import score_file  # noqa: E402
from .predictor import Predictor  # noqa: E402

log = logging.getLogger("benchmark")
FIELDS = ["transaction_id", "step", "type", "amount"] + C.BALANCE_COLS


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    def __init__(self, env_extra=None, workers=1):
        self.port = _free_port()
        env = {**os.environ, **(env_extra or {})}
        cmd = [sys.executable, "-m", "uvicorn", "serving.app:app", "--port", str(self.port),
               "--log-level", "warning"]
        if workers > 1:
            cmd += ["--workers", str(workers)]
        self.proc = subprocess.Popen(cmd, cwd=C.ROOT, env=env)
        self.url = f"http://127.0.0.1:{self.port}"
        for _ in range(120):
            try:
                if httpx.get(self.url + "/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.5)
        self.stop()
        raise RuntimeError("server did not start")

    def stop(self):
        if os.name == "nt":
            # kill the whole process tree so multi-worker children do not outlive the parent
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"], capture_output=True)
        else:
            self.proc.terminate()
        try:
            self.proc.wait(10)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def pct(lat):
    a = np.asarray(lat)
    return {"n": int(len(a)), "mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max())}


def load_payloads():
    df = pd.read_csv(C.SAMPLE_PATH)
    df = df[df["type"].isin(C.KNOWN_TYPES)]
    recs = df[FIELDS].astype(object).where(df[FIELDS].notna(), None).to_dict("records")
    for r in recs:
        r["step"] = int(r["step"])
        r["transaction_id"] = str(r["transaction_id"])
    return recs


def run_sequential(url, payloads, n, warmup=50):
    lat, bad = [], 0
    with httpx.Client(base_url=url, timeout=10) as c:
        for i in range(warmup):
            c.post("/predict", json=payloads[i % len(payloads)])
        for i in range(n):
            t = time.perf_counter()
            r = c.post("/predict", json=payloads[i % len(payloads)])
            lat.append((time.perf_counter() - t) * 1000)
            bad += r.status_code != 200
    return lat, bad


def run_concurrent(url, payloads, n, workers):
    def worker(chunk):
        out = []
        with httpx.Client(base_url=url, timeout=30) as c:
            for p in chunk:
                t = time.perf_counter()
                r = c.post("/predict", json=p)
                out.append(((time.perf_counter() - t) * 1000, r.status_code))
        return out

    work = [payloads[i % len(payloads)] for i in range(n)]
    chunks = [work[i::workers] for i in range(workers)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        res = [x for part in ex.map(worker, chunks) for x in part]
    wall = time.perf_counter() - t0
    lat = [x[0] for x in res]
    out = pct(lat)
    out.update(workers=workers, requests_per_sec=n / wall, errors=sum(x[1] != 200 for x in res))
    return out


def run_soak(url, payloads, seconds, workers, window=10):
    """Closed-loop load for `seconds`; reports latency per `window`-second slice so a
    slow degradation (leak, queue build-up) would show as a trend."""
    t0 = time.perf_counter()
    stop = t0 + seconds

    def worker(i):
        out, k = [], i
        with httpx.Client(base_url=url, timeout=30) as c:
            while time.perf_counter() < stop:
                p = payloads[k % len(payloads)]
                k += workers
                t = time.perf_counter()
                r = c.post("/predict", json=p)
                out.append((t - t0, (time.perf_counter() - t) * 1000, r.status_code))
        return out

    with ThreadPoolExecutor(workers) as ex:
        res = [x for part in ex.map(worker, range(workers)) for x in part]
    rows = []
    for w in range(int(np.ceil(seconds / window))):
        sl = [x for x in res if w * window <= x[0] < (w + 1) * window]
        if not sl:
            continue
        lat = np.array([x[1] for x in sl])
        rows.append({"window": f"{w * window}-{(w + 1) * window}s", "requests": len(sl),
                     "rps": len(sl) / window, "p50_ms": float(np.percentile(lat, 50)),
                     "p99_ms": float(np.percentile(lat, 99)), "errors": int(sum(x[2] != 200 for x in sl))})
    allat = np.array([x[1] for x in res])
    return {"seconds": seconds, "clients": workers, "total_requests": len(res),
            "total_errors": int(sum(x[2] != 200 for x in res)), "overall": pct(allat), "windows": rows}


def run_batch_endpoint(url, payloads, size, reps=20):
    body = {"records": [payloads[i % len(payloads)] for i in range(size)]}
    lat = []
    with httpx.Client(base_url=url, timeout=60) as c:
        c.post("/predict/batch", json=body)
        for _ in range(reps):
            t = time.perf_counter()
            r = c.post("/predict/batch", json=body)
            r.raise_for_status()
            lat.append((time.perf_counter() - t) * 1000)
    out = pct(lat)
    out.update(batch_size=size, records_per_sec=size / (np.mean(lat) / 1000))
    return out


def offline_throughput(rows):
    sample = pd.read_csv(C.SAMPLE_PATH)
    reps = int(np.ceil(rows / len(sample)))
    big = pd.concat([sample] * reps, ignore_index=True).iloc[:rows].copy()
    big["transaction_id"] = np.arange(len(big)).astype(str)   # unique ids; rows are tiled copies
    pred = Predictor(n_jobs=-1)
    scored = big.drop(columns=["transaction_id"]).copy()
    scored = scored[scored["type"].isin(C.KNOWN_TYPES)]
    t = time.perf_counter()
    pred.score_frame(scored)
    score_only = len(scored) / (time.perf_counter() - t)
    with tempfile.TemporaryDirectory() as d:
        src_csv, out_csv = Path(d) / "in.csv", Path(d) / "out.csv"
        big.to_csv(src_csv, index=False)
        stats = score_file(src_csv, out_csv, predictor=pred)
    return {"rows": int(len(big)), "score_only_records_per_sec": float(score_only),
            "end_to_end_csv_records_per_sec": stats["records_per_sec"], "seconds": stats["seconds"]}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=2000)
    ap.add_argument("--offline-rows", type=int, default=500_000)
    ap.add_argument("--soak-seconds", type=int, default=60, help="0 skips the soak test")
    args = ap.parse_args()
    payloads = load_payloads()
    result = {"environment": {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                              "python": platform.python_version()}}

    srv = Server()
    try:
        lat, bad = run_sequential(srv.url, payloads, args.requests)
        seq = pct(lat) | {"errors": int(bad), "requests_per_sec": 1000 / np.mean(lat)}
        result["online_sequential"] = seq
        log.info("sequential: p50=%.2f p95=%.2f p99=%.2f ms", seq["p50_ms"], seq["p95_ms"], seq["p99_ms"])
        result["online_concurrent"] = [run_concurrent(srv.url, payloads, args.requests, w) for w in (4, 16)]
        result["batch_endpoint"] = [run_batch_endpoint(srv.url, payloads, s) for s in (100, 500)]
        result["server_stats"] = httpx.get(srv.url + "/stats").json()
    finally:
        srv.stop()

    # scale-out: same 16 clients, more uvicorn worker processes; soak on the 4-worker setup
    scale = []
    for w in (2, 4):
        srv = Server(workers=w)
        try:
            row = run_concurrent(srv.url, payloads, args.requests, 16)
            row["uvicorn_workers"] = w
            scale.append(row)
            log.info("%d workers, 16 clients: %.0f req/s, p99 %.1f ms", w, row["requests_per_sec"], row["p99_ms"])
            if w == 4 and args.soak_seconds > 0:
                result["soak_4_workers"] = run_soak(srv.url, payloads, args.soak_seconds, 8)
                log.info("soak: %d requests, %d errors, p99 %.1f ms",
                         result["soak_4_workers"]["total_requests"], result["soak_4_workers"]["total_errors"],
                         result["soak_4_workers"]["overall"]["p99_ms"])
        finally:
            srv.stop()
    result["scale_out_16_clients"] = scale

    srv = Server({"FRAUD_FORCE_FALLBACK": "1"})
    try:
        flat, fbad = run_sequential(srv.url, payloads, min(args.requests, 1000))
        result["online_fallback_mode"] = pct(flat) | {"errors": int(fbad)}
    finally:
        srv.stop()

    result["offline_batch"] = offline_throughput(args.offline_rows)
    log.info("offline: %.0f rec/s score-only, %.0f rec/s end-to-end",
             result["offline_batch"]["score_only_records_per_sec"],
             result["offline_batch"]["end_to_end_csv_records_per_sec"])

    C.REPORT_DIR.mkdir(exist_ok=True, parents=True)
    (C.REPORT_DIR / "benchmark.json").write_text(json.dumps(result, indent=2))
    plt.figure(figsize=(6, 3.8))
    plt.hist(lat, bins=60)
    for q, col in ((50, "green"), (95, "orange"), (99, "red")):
        plt.axvline(np.percentile(lat, q), color=col, ls="--", label=f"p{q}={np.percentile(lat, q):.1f} ms")
    plt.xlabel("Latency per /predict request (ms)"); plt.ylabel("Requests")
    plt.title("Single-client latency distribution"); plt.legend(fontsize=8)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "latency_hist.png", dpi=160); plt.close()


if __name__ == "__main__":
    main()
