import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import ARTIFACT_DIR, CATEGORICAL, NUMERIC

PSI_WARN, PSI_ALERT, KS_ALERT_P = 0.10, 0.25, 0.01
EPS = 1e-4


def psi(ref_props, cur_props):
    r, c = np.clip(np.asarray(ref_props), EPS, None), np.clip(np.asarray(cur_props), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_numeric(ref, values):
    edges = np.concatenate(([-np.inf], ref["edges"], [np.inf]))
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    return psi(ref["ref_props"], np.histogram(v, bins=edges)[0] / max(len(v), 1)), v


def status(value):
    return "ALERT" if value >= PSI_ALERT else "WARN" if value >= PSI_WARN else "OK"


def drift_report(current, scores=None, profile=None):
    """PSI per feature (plus KS for numerics) against the training reference. Optional score drift."""
    profile = profile or json.load(open(ARTIFACT_DIR / "reference_profile.json"))
    rows = []
    for c in NUMERIC:
        value, v = psi_numeric(profile["numeric"][c], pd.to_numeric(current[c], errors="coerce"))
        ks = ks_2samp(profile["numeric"][c]["sorted_values"], v)
        rows.append({"feature": c, "type": "numeric", "psi": value, "ks_stat": float(ks.statistic),
                     "ks_pvalue": float(ks.pvalue), "status": status(value)})
    for c in CATEGORICAL:
        ref = profile["categorical"][c]
        cur = current[c].astype(str).value_counts(normalize=True).to_dict()
        levels = sorted(set(ref) | set(cur))
        value = psi([ref.get(k, 0) for k in levels], [cur.get(k, 0) for k in levels])
        rows.append({"feature": c, "type": "categorical", "psi": value, "ks_stat": None, "ks_pvalue": None,
                     "status": status(value)})
    if scores is not None:
        value, v = psi_numeric(profile["score"], scores)
        ks = ks_2samp(profile["score"]["sorted_values"], v)
        rows.append({"feature": "model_score", "type": "score", "psi": value, "ks_stat": float(ks.statistic),
                     "ks_pvalue": float(ks.pvalue), "status": status(value)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Compare a new CSV of customers against the training reference.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    rep = drift_report(pd.read_csv(args.input))
    print(rep.round(4).to_string(index=False))
    if args.output:
        rep.to_csv(args.output, index=False)
    return 0 if (rep["status"] != "ALERT").all() else 3


if __name__ == "__main__":
    raise SystemExit(main())
