import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import httpx
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from serving.inference import ChurnScorer
from src.config import DATA_PATH, FIG_DIR, REPORT_DIR, FEATURES


def pct(a, q):
    return float(np.percentile(a, q))


def api_benchmark(records, n, port):
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.app:app", "--port", str(port), "--log-level", "warning"])
    base = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=base, timeout=10) as client:
            for _ in range(100):
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.HTTPError:
                    time.sleep(0.3)
            for r in records[:50]:
                client.post("/predict", json=r)
            client_ms, server_ms, errors = [], [], 0
            t0 = time.perf_counter()
            for r in records[:n]:
                s = time.perf_counter()
                resp = client.post("/predict", json=r)
                client_ms.append((time.perf_counter() - s) * 1000)
                if resp.status_code == 200:
                    server_ms.append(resp.json()["latency_ms"])
                else:
                    errors += 1
            wall = time.perf_counter() - t0
    finally:
        proc.terminate(); proc.wait(timeout=10)
    c = np.array(client_ms)
    return {"requests": n, "errors": errors, "requests_per_sec": n / wall,
            "client_ms": {"mean": float(c.mean()), "p50": pct(c, 50), "p95": pct(c, 95), "p99": pct(c, 99), "max": float(c.max())},
            "server_ms": {"mean": float(np.mean(server_ms)), "p50": pct(server_ms, 50), "p95": pct(server_ms, 95), "p99": pct(server_ms, 99)}}, c


def batch_benchmark(df):
    scorer = ChurnScorer()
    rng = np.random.default_rng(0)
    out = {}
    for size in (1_000, 10_000, 100_000):
        frame = df.sample(size, replace=True, random_state=1).reset_index(drop=True)
        runs = []
        for _ in range(3):
            s = time.perf_counter(); scorer.score_frame(frame); runs.append(time.perf_counter() - s)
        out[f"in_process_{size}"] = {"records": size, "seconds": float(np.median(runs)), "records_per_sec": size / float(np.median(runs))}
    big = df.sample(100_000, replace=True, random_state=2).reset_index(drop=True)
    big["customerID"] = [f"BENCH-{i}" for i in range(len(big))]
    with tempfile.TemporaryDirectory() as d:
        inp, outp = os.path.join(d, "in.csv"), os.path.join(d, "out.csv")
        big.to_csv(inp, index=False)
        s = time.perf_counter()
        subprocess.run([sys.executable, "-m", "serving.predict_batch", "--input", inp, "--output", outp],
                       check=True, capture_output=True)
        sec = time.perf_counter() - s
    out["cli_end_to_end_100000"] = {"records": 100_000, "seconds": sec, "records_per_sec": 100_000 / sec}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    df = pd.read_csv(DATA_PATH).drop(columns=["Churn"])
    records = json.loads(df.sample(args.requests + 50, replace=True, random_state=3).to_json(orient="records"))
    for r in records:
        if r.get("TotalCharges") in ("", " ", None):
            r["TotalCharges"] = None
        else:
            r["TotalCharges"] = float(r["TotalCharges"])
    api, lat = api_benchmark(records, args.requests, args.port)
    batch = batch_benchmark(df)
    result = {"environment": {"platform": platform.platform(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
                              "note": "single uvicorn worker, client and server on the same machine"},
              "api": api, "batch": batch}
    json.dump(result, open(REPORT_DIR / "benchmark_results.json", "w"), indent=2)

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    ax.hist(lat, bins=40, color="#2A7F8E")
    for q, c in ((50, "#1F2A44"), (95, "#E4A11B"), (99, "#C8553D")):
        ax.axvline(np.percentile(lat, q), color=c, ls="--", label=f"p{q} = {np.percentile(lat, q):.1f} ms")
    ax.set_xlabel("Request latency (ms)"); ax.set_ylabel("Requests"); ax.set_title("POST /predict latency, 1000 requests")
    ax.legend(fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "latency_hist.png", dpi=200); plt.close(fig)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
