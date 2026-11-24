"""Batch scoring CLI.

    python -m serving.predict_batch --input data/sample_transactions.csv --output out/scores.csv

Reads a CSV in chunks, scores it with the same Predictor the API uses, and
writes one output row per accepted input row (with its record id). Bad rows are
not allowed to crash the job: they go to a rejects file with a reason.

Exit codes: 0 ok, 1 runtime failure, 2 unusable input file.
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import config as C
from .predictor import Predictor

log = logging.getLogger("predict_batch")
REQUIRED = ["step", "type", "amount"]
NUMERIC = ["step", "amount"] + C.BALANCE_COLS


class InputError(Exception):
    pass


def _validate(chunk: pd.DataFrame):
    """Return (clean_chunk, rejects) — coerces types and flags unusable rows."""
    chunk = chunk.copy()
    chunk["type"] = chunk["type"].astype(str).str.strip().str.upper()
    for col in NUMERIC:
        if col in chunk:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        else:
            chunk[col] = np.nan            # optional balance column absent -> imputed

    reason = pd.Series("", index=chunk.index)
    reason[~chunk["type"].isin(C.KNOWN_TYPES)] = "unknown_type"
    reason[chunk["amount"].isna() | ~np.isfinite(chunk["amount"]) | (chunk["amount"] < 0)] = "bad_amount"
    reason[chunk["step"].isna() | (chunk["step"] < 1)] = "bad_step"
    reason[(chunk[C.BALANCE_COLS] < 0).any(axis=1)] = "negative_balance"
    bad = reason != ""
    return chunk[~bad], pd.DataFrame({"reason": reason[bad]})


def score_file(input_path, output_path, rejects_path=None, chunksize=200_000,
               id_col="transaction_id", predictor=None):
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise InputError(f"input file not found: {input_path}")
    head = pd.read_csv(input_path, nrows=0).columns
    missing = [c for c in REQUIRED if c not in head]
    if missing:
        raise InputError(f"input is missing required columns: {missing}")
    if id_col not in head:
        log.warning("id column '%s' not found; using row numbers as record ids", id_col)

    predictor = predictor or Predictor(n_jobs=-1)
    if not predictor.ready:
        log.warning("model not available: whole job will run on the heuristic fallback")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejects_path = Path(rejects_path) if rejects_path else output_path.with_suffix(".rejects.csv")

    stats = {"rows_in": 0, "scored": 0, "rejected": 0, "flagged": 0, "fallback_rows": 0}
    first_out = first_rej = True
    t0 = time.perf_counter()
    offset = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        chunk.index = np.arange(offset, offset + len(chunk))
        offset += len(chunk)
        stats["rows_in"] += len(chunk)
        ids = chunk[id_col].astype(str) if id_col in chunk else pd.Series(chunk.index.astype(str), index=chunk.index)
        ok, rej = _validate(chunk)
        if len(rej):
            rej.insert(0, id_col, ids.loc[rej.index])
            rej.to_csv(rejects_path, mode="w" if first_rej else "a", header=first_rej, index=False)
            first_rej = False
            stats["rejected"] += len(rej)
        if len(ok):
            res = predictor.score_frame(ok)
            res.insert(0, id_col, ids.loc[ok.index])
            res["fraud_probability"] = res["fraud_probability"].round(6)
            res.to_csv(output_path, mode="w" if first_out else "a", header=first_out, index=False)
            first_out = False
            stats["scored"] += len(res)
            stats["flagged"] += int(res["is_fraud"].sum())
            stats["fallback_rows"] += int((res["mode"] == "fallback_rule").sum())
        log.info("processed %s rows (%s scored, %s rejected)", f"{stats['rows_in']:,}",
                 f"{stats['scored']:,}", f"{stats['rejected']:,}")
    if first_out:  # nothing scored: still leave a valid, empty output with headers
        pd.DataFrame(columns=[id_col, "fraud_probability", "is_fraud", "mode", "needs_review"]).to_csv(
            output_path, index=False)

    secs = time.perf_counter() - t0
    stats.update(seconds=round(secs, 3), records_per_sec=round(stats["rows_in"] / secs, 1) if secs else None,
                 output=str(output_path))
    return stats


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Score a CSV of transactions in batch.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--rejects", help="where to write rejected rows (default: <output>.rejects.csv)")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--id-col", default="transaction_id")
    ap.add_argument("--summary", help="optional path for a JSON run summary")
    args = ap.parse_args(argv)
    try:
        stats = score_file(args.input, args.output, args.rejects, args.chunksize, args.id_col)
    except InputError as exc:
        log.error("%s", exc)
        return 2
    except Exception:
        log.exception("batch scoring failed")
        return 1
    log.info("done: %s", json.dumps(stats))
    if args.summary:
        Path(args.summary).write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
