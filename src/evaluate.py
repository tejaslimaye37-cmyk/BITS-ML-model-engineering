import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
                             roc_auc_score)

from src.config import ARTIFACT_DIR, COST_FN, COST_FP, FIG_DIR, MIN_PRECISION, REPORT_DIR, TARGET
from src.data_pipeline import load_splits

GRID = np.round(np.linspace(0.02, 0.98, 97), 2)


def cost_at(y, p, thr, c_fn=COST_FN, c_fp=COST_FP):
    pred = p >= thr
    return float(c_fn * np.sum((y == 1) & ~pred) + c_fp * np.sum((y == 0) & pred))


def best_cost_threshold(y, p):
    costs = np.array([cost_at(y, p, t) for t in GRID])
    return float(GRID[int(np.argmin(costs))]), costs


def recall_at_precision(y, p, min_precision=MIN_PRECISION):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def summarize(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"pr_auc": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(p)) > 1 else 0.5,
            "recall_at_precision50": recall_at_precision(y, p),
            "threshold": float(thr),
            "f1_macro": float(f1_score(y, pred, average="macro")),
            "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
            "cost": cost_at(y, p, thr), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


MODELS = [("dummy", "Dummy prior"), ("heuristic", "Heuristic rule score"),
          ("logreg", "Logistic Regression (balanced)"), ("lgbm_unweighted", "LightGBM (unweighted)"),
          ("lgbm", "LightGBM (weighted + calibrated)")]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = pd.read_csv(REPORT_DIR / "test_predictions.csv")
    meta = json.load(open(ARTIFACT_DIR / "metadata.json"))
    y = preds["y"].to_numpy()

    # independent check: the saved production artifacts must reproduce the stored test scores
    _, _, test = load_splits()
    pre, model = joblib.load(ARTIFACT_DIR / "preprocessor.joblib"), joblib.load(ARTIFACT_DIR / "model.joblib")
    from src.config import FEATURES
    p_art = model.predict_proba(pre.transform(test[FEATURES]))[:, 1]
    assert np.allclose(p_art, preds["p_lgbm"].to_numpy(), atol=1e-9), "artifacts do not reproduce test scores"

    results = {}
    for key, name in MODELS:
        thr = meta["thresholds"][key]
        results[key] = {"name": name, **summarize(y, preds[f"p_{key}"].to_numpy(), thr)}
    pos, neg = int(y.sum()), int((y == 0).sum())
    results["contact_nobody"] = {"name": "Contact nobody", "cost": COST_FN * pos}
    results["contact_everyone"] = {"name": "Contact everyone", "cost": COST_FP * neg}
    json.dump(results, open(REPORT_DIR / "metrics.json", "w"), indent=2)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    styles = {"dummy": ("#9CA3AF", ":"), "heuristic": ("#C8553D", "--"), "logreg": ("#2A7F8E", "-."),
              "lgbm_unweighted": ("#E4A11B", "--"), "lgbm": ("#1F2A44", "-")}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for key, name in MODELS:
        pr, rc, _ = precision_recall_curve(y, preds[f"p_{key}"])
        c, ls = styles[key]
        ax.plot(rc, pr, color=c, ls=ls, lw=2, label=f"{name} (AP {results[key]['pr_auc']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall curves on the test set")
    ax.legend(fontsize=8, loc="upper right"); ax.set_ylim(0, 1.02)
    fig.tight_layout(); fig.savefig(FIG_DIR / "pr_curves.png", dpi=200); plt.close(fig)

    fin = results["lgbm"]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    cm = np.array([[fin["tn"], fin["fp"]], [fin["fn"], fin["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Test confusion matrix (t={fin['threshold']:.2f})")
    fig.tight_layout(); fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=200); plt.close(fig)

    _, val, _ = load_splits()
    pv = pd.read_csv(REPORT_DIR / "val_predictions.csv")
    _, vcosts = best_cost_threshold(pv["y"].to_numpy(), pv["p_lgbm"].to_numpy())
    tcosts = np.array([cost_at(y, preds["p_lgbm"].to_numpy(), t) for t in GRID])
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(GRID, vcosts, color="#1F2A44", lw=2, label="Validation")
    ax.plot(GRID, tcosts, color="#E4A11B", lw=2, ls="--", label="Test")
    ax.axvline(meta["thresholds"]["lgbm"], color="#C8553D", ls=":", label="Chosen threshold")
    ax.set_xlabel("Decision threshold"); ax.set_ylabel("Expected cost (cost units)")
    ax.set_title("Cost versus threshold (FN = 5, FP = 1)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG_DIR / "cost_threshold.png", dpi=200); plt.close(fig)

    raw = joblib.load(ARTIFACT_DIR / "lgbm_raw.joblib")
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    imp = pd.Series(raw.booster_.feature_importance("gain"), index=names).sort_values(ascending=False)[:10][::-1]
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.barh(imp.index, imp.values / imp.values.sum() * 100, color="#2A7F8E")
    ax.set_xlabel("Share of total gain (%)"); ax.set_title("Top 10 features by gain")
    fig.tight_layout(); fig.savefig(FIG_DIR / "feature_importance.png", dpi=200); plt.close(fig)
    print(json.dumps({k: {m: round(v, 4) if isinstance(v, float) else v for m, v in r.items() if m != "name"}
                      for k, r in results.items()}, indent=1))


if __name__ == "__main__":
    main()
