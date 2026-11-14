import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, CATEGORICAL, FEATURES, ID_COL, NUMERIC, REQUIRED
from src.heuristic import heuristic_score

log = logging.getLogger("churn.inference")


def _clean_cell(v):
    if isinstance(v, str):
        v = v.strip()
        return v if v else np.nan
    return np.nan if v is None else v


class ChurnScorer:
    """Loads serialized artifacts once and scores raw customer frames. Shared by the API and the batch CLI."""

    def __init__(self, artifact_dir=ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        self.meta = json.load(open(artifact_dir / "metadata.json"))
        self.pre = joblib.load(artifact_dir / "preprocessor.joblib")
        self.model = joblib.load(artifact_dir / "model.joblib")
        self.version = self.meta["model_version"]
        self.threshold = self.meta["threshold"]

    def prepare(self, raw):
        """Coerce types, clip numeric outliers to the training range, and record data-quality warnings."""
        n = len(raw)
        num = raw.reindex(columns=NUMERIC).apply(pd.to_numeric, errors="coerce").astype(float)
        arr = num.to_numpy()
        lo = np.array([self.meta["numeric_ranges"][c][0] for c in NUMERIC])
        hi = np.array([self.meta["numeric_ranges"][c][1] for c in NUMERIC])
        num_missing = np.isnan(arr)
        num_clipped = (arr < lo) | (arr > hi)
        Xn = pd.DataFrame(np.clip(arr, lo, hi), columns=NUMERIC, index=raw.index)

        cat = raw.reindex(columns=CATEGORICAL).astype(object).map(_clean_cell)
        cat_missing = cat.isna().to_numpy()
        seen = cat.isin(self.meta["categorical_levels"]).to_numpy()
        cat_unseen = ~cat_missing & ~seen

        warns = [[] for _ in range(n)]
        for i, j in np.argwhere(num_missing):
            warns[i].append(f"missing_or_invalid:{NUMERIC[j]}")
        for i, j in np.argwhere(num_clipped):
            warns[i].append(f"clipped:{NUMERIC[j]}")
        for i, j in np.argwhere(cat_missing):
            warns[i].append(f"missing:{CATEGORICAL[j]}")
        for i, j in np.argwhere(cat_unseen):
            warns[i].append(f"unseen_category:{CATEGORICAL[j]}={cat.iat[i, j]}")
        return pd.concat([Xn, cat], axis=1)[FEATURES], warns

    def fallback(self, X):
        table, prior = self.meta["fallback"]["table"], self.meta["fallback"]["prior"]
        p = np.array([table.get(str(k), prior) for k in heuristic_score(X)])
        return p, self.meta["fallback"]["threshold"]

    def score_frame(self, raw):
        X, warns = self.prepare(raw)
        source, threshold = "model", self.threshold
        try:
            p = self.model.predict_proba(self.pre.transform(X))[:, 1]
            if not np.all(np.isfinite(p)):
                raise ValueError("non-finite model output")
        except Exception:
            log.exception("model scoring failed, switching to heuristic fallback")
            p, threshold = self.fallback(X)
            source = "fallback_heuristic"
        return pd.DataFrame({"churn_probability": np.round(p, 5), "churn_flag": p >= threshold,
                             "threshold": threshold, "scoring_source": source,
                             "warnings": [";".join(w) for w in warns]}, index=raw.index)


def validate_rows(raw):
    """Vectorised row validation for batch input. Returns a Series of reject reasons ('' means valid)."""
    reasons = pd.Series("", index=raw.index)
    for c in REQUIRED:
        reasons = reasons.where(raw[c].notna() & (raw[c].astype(str).str.strip() != ""), reasons + f"missing:{c};")
    for c in ("tenure", "MonthlyCharges"):
        num = pd.to_numeric(raw[c], errors="coerce")
        reasons = reasons.where(~(num.isna() & raw[c].notna()), reasons + f"non_numeric:{c};")
        reasons = reasons.where(~(num < 0), reasons + f"negative:{c};")
    return reasons
