import json
import platform
from datetime import datetime, timezone

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from src.config import (ARTIFACT_DIR, CATEGORICAL, FEATURES, ID_COL, MODEL_VERSION, NUMERIC, REPORT_DIR, SEED,
                        TARGET)
from src.data_pipeline import build_preprocessor, load_splits
from src.evaluate import best_cost_threshold
from src.heuristic import heuristic_score


def fit_lgbm(params, X_tr, y_tr, X_va, y_va, spw):
    model = lgb.LGBMClassifier(n_estimators=1000, subsample_freq=1, scale_pos_weight=spw, metric="average_precision",
                               random_state=SEED, n_jobs=1, verbose=-1, **params)
    model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va, callbacks=[lgb.early_stopping(50, verbose=False)])
    return model


def sample_params(rng):
    return {"learning_rate": float(rng.choice([0.02, 0.03, 0.05, 0.08])),
            "num_leaves": int(rng.choice([6, 8, 12, 16, 24, 32])),
            "min_child_samples": int(rng.choice([10, 20, 30, 50, 80])),
            "subsample": float(rng.choice([0.6, 0.8, 1.0])),
            "colsample_bytree": float(rng.choice([0.5, 0.7, 0.9, 1.0])),
            "reg_lambda": float(rng.choice([0.0, 1.0, 5.0, 10.0]))}


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    train, val, test = load_splits()
    assert not set(train[ID_COL]) & set(val[ID_COL]) and not set(train[ID_COL]) & set(test[ID_COL])
    print("rows:", len(train), len(val), len(test), "| churn rate:",
          [round(d[TARGET].mean(), 4) for d in (train, val, test)])

    pre = build_preprocessor().fit(train[FEATURES])           # fitted on the training partition only
    X_tr, X_va, X_te = (pre.transform(d[FEATURES]) for d in (train, val, test))
    y_tr, y_va, y_te = (d[TARGET].to_numpy() for d in (train, val, test))
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

    rng = np.random.default_rng(SEED)
    best = (-1, None)
    for _ in range(30):
        params = sample_params(rng)
        m = fit_lgbm(params, X_tr, y_tr, X_va, y_va, spw)
        score = m.best_score_["valid_0"]["average_precision"]
        if score > best[0]:
            best = (score, params, m)
    print("best validation PR-AUC:", round(best[0], 4), best[1], "| trees:", best[2].best_iteration_)
    params, lgbm = best[1], best[2]

    lgbm_unw = fit_lgbm(params, X_tr, y_tr, X_va, y_va, 1.0)
    logreg = LogisticRegression(class_weight="balanced", max_iter=2000).fit(X_tr, y_tr)
    final = CalibratedClassifierCV(FrozenEstimator(lgbm), method="sigmoid").fit(X_va, y_va)

    heur_tr = heuristic_score(train)
    table = {str(k): float(train[TARGET][heur_tr == k].mean()) for k in sorted(set(heur_tr))}
    prior = float(y_tr.mean())

    def scores(d, X):
        h = np.array([table.get(str(k), prior) for k in heuristic_score(d)])
        return {"dummy": np.full(len(d), prior), "heuristic": h, "logreg": logreg.predict_proba(X)[:, 1],
                "lgbm_unweighted": lgbm_unw.predict_proba(X)[:, 1], "lgbm": final.predict_proba(X)[:, 1]}

    s_va, s_te = scores(val, X_va), scores(test, X_te)
    thresholds, val_pr = {}, {}
    for key, p in s_va.items():
        thresholds[key], _ = best_cost_threshold(y_va, p)
        val_pr[key] = float(average_precision_score(y_va, p))
    print("validation PR-AUC:", {k: round(v, 4) for k, v in val_pr.items()})
    print("cost-optimal thresholds (validation):", thresholds)

    for name, d, s in (("val", val, s_va), ("test", test, s_te)):
        out = pd.DataFrame({ID_COL: d[ID_COL], "y": d[TARGET]})
        for key, p in s.items():
            out[f"p_{key}"] = p
        out.to_csv(REPORT_DIR / f"{name}_predictions.csv", index=False)

    joblib.dump(pre, ARTIFACT_DIR / "preprocessor.joblib")
    joblib.dump(final, ARTIFACT_DIR / "model.joblib")
    joblib.dump(lgbm, ARTIFACT_DIR / "lgbm_raw.joblib")

    ranges = {c: [float(train[c].min()), float(train[c].max())] for c in NUMERIC}
    levels = {c: sorted(train[c].unique().tolist()) for c in CATEGORICAL}
    profile = {"numeric": {}, "categorical": {}}
    for c in NUMERIC:
        v = train[c].dropna().to_numpy()
        edges = np.unique(np.quantile(v, np.linspace(0, 1, 11)))
        inner = np.concatenate(([-np.inf], edges[1:-1], [np.inf]))
        profile["numeric"][c] = {"edges": edges[1:-1].tolist(),
                                 "ref_props": (np.histogram(v, bins=inner)[0] / len(v)).tolist(),
                                 "sorted_values": np.sort(v).round(4).tolist()}
    for c in CATEGORICAL:
        profile["categorical"][c] = train[c].value_counts(normalize=True).to_dict()
    sv = np.sort(s_va["lgbm"])
    sedges = np.unique(np.quantile(sv, np.linspace(0, 1, 11)))[1:-1]
    profile["score"] = {"edges": sedges.tolist(), "sorted_values": sv.round(5).tolist(),
                        "ref_props": (np.histogram(sv, bins=np.concatenate(([-np.inf], sedges, [np.inf])))[0]
                                      / len(sv)).tolist()}
    json.dump(profile, open(ARTIFACT_DIR / "reference_profile.json", "w"))

    meta = {"model_version": MODEL_VERSION, "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL, "numeric_ranges": ranges,
            "categorical_levels": levels, "thresholds": thresholds, "threshold": thresholds["lgbm"],
            "fallback": {"table": table, "prior": prior, "threshold": thresholds["heuristic"]},
            "lgbm_params": params, "best_iteration": int(lgbm.best_iteration_), "scale_pos_weight": spw,
            "validation_pr_auc": val_pr,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "lightgbm": lgb.__version__}}
    json.dump(meta, open(ARTIFACT_DIR / "metadata.json", "w"), indent=2)
    print("artifacts written to", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
