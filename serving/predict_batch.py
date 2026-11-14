import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from serving.inference import ChurnScorer, validate_rows
from src.config import ARTIFACT_DIR, ID_COL, REQUIRED

log = logging.getLogger("churn.batch")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score a CSV of customers for churn risk.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--artifacts", default=str(ARTIFACT_DIR))
    ap.add_argument("--chunk-size", type=int, default=50000)
    ap.add_argument("--log-file", default=None)
    args = ap.parse_args(argv)

    handlers = [logging.StreamHandler(sys.stdout)] + ([logging.FileHandler(args.log_file)] if args.log_file else [])
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers, force=True)

    start = time.perf_counter()
    try:
        scorer = ChurnScorer(args.artifacts)
        raw = pd.read_csv(args.input)
    except Exception as exc:
        log.error("startup failed: %s", exc)
        return 1
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        log.error("input is missing required columns: %s", missing)
        return 2
    log.info("read %d rows from %s (model %s)", len(raw), args.input, scorer.version)

    ids = raw[ID_COL].astype(str) if ID_COL in raw.columns else pd.Series(raw.index.astype(str), index=raw.index)
    reasons = validate_rows(raw)
    bad = reasons != ""
    if bad.any():
        rej = Path(args.output).with_suffix(".rejects.csv")
        pd.concat([ids[bad].rename("record_id"), reasons[bad].rename("reason")], axis=1).to_csv(rej, index=False)
        log.warning("rejected %d invalid rows, details in %s", bad.sum(), rej)

    good = raw[~bad]
    parts = []
    for i in range(0, len(good), args.chunk_size):
        parts.append(scorer.score_frame(good.iloc[i:i + args.chunk_size]))
    scored = pd.concat(parts) if parts else pd.DataFrame()
    out = pd.concat([ids[~bad].rename("record_id"), scored], axis=1)
    out.to_csv(args.output, index=False)

    elapsed = time.perf_counter() - start
    fallback = int((out["scoring_source"] != "model").sum()) if len(out) else 0
    log.info("scored %d rows (%d via fallback) in %.2fs = %.0f records/sec, flagged %d, output %s",
             len(out), fallback, elapsed, len(out) / max(elapsed, 1e-9), int(out["churn_flag"].sum()) if len(out) else 0,
             args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
