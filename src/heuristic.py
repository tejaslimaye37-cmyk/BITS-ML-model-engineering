import numpy as np


def heuristic_score(df):
    """Rule-based risk score from 0 to 4: one point per well-known churn risk factor."""
    score = ((df["Contract"] == "Month-to-month").astype(int)
             + (df["tenure"] < 12).astype(int)
             + (df["InternetService"] == "Fiber optic").astype(int)
             + (df["PaymentMethod"] == "Electronic check").astype(int))
    return np.asarray(score, dtype=int)
