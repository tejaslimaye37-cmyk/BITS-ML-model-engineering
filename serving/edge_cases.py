import json

from fastapi.testclient import TestClient

from serving.app import app, state
from src.config import REPORT_DIR
from tests.conftest import RECORD


def main():
    cases = {
        "Valid record": {},
        "Optional fields missing": {"drop": ["InternetService", "TotalCharges", "PaymentMethod", "gender"]},
        "Unseen category (PaymentMethod=Crypto)": {"set": {"PaymentMethod": "Crypto"}},
        "Extreme outlier (MonthlyCharges=1e7, tenure=9999)": {"set": {"MonthlyCharges": 1e7, "tenure": 9999}},
        "Negative tenure": {"set": {"tenure": -5}},
        "Non-numeric MonthlyCharges": {"set": {"MonthlyCharges": "abc"}},
        "Required field missing (Contract)": {"drop": ["Contract"]},
        "Unknown extra field": {"set": {"favourite_colour": "blue"}},
        "Empty body": {"body": {}},
    }
    rows = []
    with TestClient(app) as client:
        for name, spec in cases.items():
            body = spec.get("body", {k: v for k, v in RECORD.items() if k not in spec.get("drop", [])})
            body = {**body, **spec.get("set", {})} if "body" not in spec else body
            r = client.post("/predict", json=body)
            j = r.json()
            rows.append({"case": name, "status": r.status_code,
                         "outcome": (f"score {j['churn_probability']:.3f}, source {j['scoring_source']}, warnings: "
                                     f"{', '.join(j['warnings']) or 'none'}") if r.status_code == 200
                         else f"rejected: {j['detail'][0]['loc'][-1] if isinstance(j['detail'], list) else j['detail']}"})
        model = state["scorer"].model
        original = model.predict_proba
        model.predict_proba = lambda X: (_ for _ in ()).throw(RuntimeError("simulated failure"))
        r = client.post("/predict", json=RECORD); j = r.json()
        model.predict_proba = original
        rows.append({"case": "Model scoring raises an exception", "status": r.status_code,
                     "outcome": f"score {j['churn_probability']:.3f}, source {j['scoring_source']}"})
    json.dump(rows, open(REPORT_DIR / "edge_case_results.json", "w"), indent=1)
    for r in rows:
        print(r["status"], "|", r["case"], "|", r["outcome"])


if __name__ == "__main__":
    main()
