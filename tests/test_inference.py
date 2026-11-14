import numpy as np
import pandas as pd

from serving.inference import validate_rows


def frame(record, **patch):
    return pd.DataFrame([{**record, **patch}])


def test_scores_are_probabilities_and_deterministic(scorer, record):
    a = scorer.score_frame(frame(record))
    b = scorer.score_frame(frame(record))
    assert 0 <= a["churn_probability"].iloc[0] <= 1
    assert a["churn_probability"].iloc[0] == b["churn_probability"].iloc[0]
    assert a["scoring_source"].iloc[0] == "model"


def test_missing_optional_values_are_imputed(scorer, record):
    out = scorer.score_frame(frame(record, InternetService=None, TotalCharges=None))
    assert np.isfinite(out["churn_probability"].iloc[0])
    assert "missing:InternetService" in out["warnings"].iloc[0]


def test_unseen_category_is_flagged_not_fatal(scorer, record):
    out = scorer.score_frame(frame(record, PaymentMethod="Crypto"))
    assert "unseen_category:PaymentMethod=Crypto" in out["warnings"].iloc[0]
    assert out["scoring_source"].iloc[0] == "model"


def test_extreme_outlier_is_clipped(scorer, record):
    out = scorer.score_frame(frame(record, MonthlyCharges=1e7, tenure=9999))
    assert "clipped:MonthlyCharges" in out["warnings"].iloc[0]
    assert 0 <= out["churn_probability"].iloc[0] <= 1


def test_fallback_when_model_fails(scorer, record, monkeypatch):
    monkeypatch.setattr(scorer.model, "predict_proba", lambda X: (_ for _ in ()).throw(RuntimeError("boom")))
    out = scorer.score_frame(frame(record))
    assert out["scoring_source"].iloc[0] == "fallback_heuristic"
    assert 0 <= out["churn_probability"].iloc[0] <= 1


def test_batch_row_validation(record):
    df = pd.DataFrame([record, {**record, "tenure": -1}, {**record, "MonthlyCharges": "x"}, {**record, "Contract": None}])
    reasons = validate_rows(df)
    assert reasons.iloc[0] == "" and all(reasons.iloc[1:] != "")
