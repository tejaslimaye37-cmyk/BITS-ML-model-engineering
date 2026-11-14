import json

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from serving.inference import ChurnScorer
from src.config import FIG_DIR, REPORT_DIR, FEATURES
from src.data_pipeline import load_splits
from src.monitor import PSI_ALERT, PSI_WARN, drift_report


def main():
    _, _, test = load_splits()
    scorer = ChurnScorer()
    rng = np.random.default_rng(7)

    shifted = test.copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 1.25 + 5
    mask = rng.random(len(shifted)) < 0.5
    shifted.loc[mask, "Contract"] = "Month-to-month"
    shifted.loc[mask, "tenure"] = (shifted.loc[mask, "tenure"] * 0.5).astype(int)

    scenarios = {"Test data (no drift)": test, "Simulated price and contract shift": shifted}
    results, table = {}, {}
    for name, d in scenarios.items():
        scores = scorer.score_frame(d[FEATURES])["churn_probability"].to_numpy()
        rep = drift_report(d, scores)
        results[name] = rep
        table[name] = rep.round(4).to_dict(orient="records")
    json.dump(table, open(REPORT_DIR / "drift_demo.json", "w"), indent=1)

    base, shift = results["Test data (no drift)"], results["Simulated price and contract shift"]
    order = shift.sort_values("psi", ascending=False).head(6)["feature"].tolist()
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(x - w / 2, base.set_index("feature").loc[order, "psi"], w, color="#9CA3AF", label="Test data (no drift)")
    ax.bar(x + w / 2, shift.set_index("feature").loc[order, "psi"], w, color="#1F2A44", label="Simulated shift")
    ax.axhline(PSI_WARN, color="#E4A11B", ls="--", lw=1.2, label=f"Warn ({PSI_WARN})")
    ax.axhline(PSI_ALERT, color="#C8553D", ls="--", lw=1.2, label=f"Alert ({PSI_ALERT})")
    ax.set_xticks(x, order, rotation=30, ha="right", fontsize=8); ax.set_ylabel("PSI")
    ax.set_title("Population Stability Index by feature"); ax.legend(fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "drift_psi.png", dpi=200); plt.close(fig)
    for name, rep in results.items():
        print(name, "->", rep[rep.status != "OK"][["feature", "psi", "status"]].round(3).to_dict(orient="records"))


if __name__ == "__main__":
    main()
