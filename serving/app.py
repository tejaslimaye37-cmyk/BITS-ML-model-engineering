import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from serving.inference import ChurnScorer
from serving.schemas import CustomerRecord, HealthResponse, PredictionResponse
from src.config import ARTIFACT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("churn.api")
state = {"scorer": None}


@asynccontextmanager
async def lifespan(app):
    try:
        state["scorer"] = ChurnScorer(Path(os.environ.get("ARTIFACT_DIR", ARTIFACT_DIR)))
        log.info("artifacts loaded, model version %s", state["scorer"].version)
    except Exception:
        log.exception("could not load artifacts")
        state["scorer"] = None
    yield
    state["scorer"] = None


app = FastAPI(title="Churn scoring service", version="1.0.0", lifespan=lifespan)


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    rid = uuid.uuid4().hex[:8]
    log.exception("unhandled error request_id=%s", rid)
    return JSONResponse(status_code=500, content={"detail": "internal error", "request_id": rid})


@app.get("/health", response_model=HealthResponse)
def health():
    scorer = state["scorer"]
    body = HealthResponse(status="ok" if scorer else "unavailable", model_loaded=scorer is not None,
                          model_version=scorer.version if scorer else None, fallback_available=scorer is not None)
    return JSONResponse(status_code=200 if scorer else 503, content=body.model_dump())


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord):
    start = time.perf_counter()
    scorer = state["scorer"]
    if scorer is None:
        raise HTTPException(status_code=503, detail="model artifacts are not loaded")
    out = scorer.score_frame(pd.DataFrame([record.model_dump()])).iloc[0]
    latency = (time.perf_counter() - start) * 1000
    log.info("predict source=%s score=%.4f latency_ms=%.2f", out["scoring_source"], out["churn_probability"], latency)
    return PredictionResponse(customer_id=record.customerID, churn_probability=float(out["churn_probability"]),
                              churn_flag=bool(out["churn_flag"]), threshold=float(out["threshold"]),
                              scoring_source=out["scoring_source"], model_version=scorer.version,
                              warnings=[w for w in out["warnings"].split(";") if w], latency_ms=round(latency, 3))
