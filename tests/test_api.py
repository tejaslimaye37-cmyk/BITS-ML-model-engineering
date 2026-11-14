from fastapi.testclient import TestClient

from serving.app import app, state


def test_health_and_predict(record):
    with TestClient(app) as client:
        h = client.get("/health")
        assert h.status_code == 200 and h.json()["model_loaded"] is True
        r = client.post("/predict", json=record)
        body = r.json()
        assert r.status_code == 200 and 0 <= body["churn_probability"] <= 1
        assert body["scoring_source"] == "model" and body["customer_id"] == "7590-VHVEG"


def test_malformed_payloads_get_422(record):
    with TestClient(app) as client:
        assert client.post("/predict", json={**record, "tenure": -5}).status_code == 422
        assert client.post("/predict", json={"tenure": 3}).status_code == 422
        assert client.post("/predict", content="not json", headers={"content-type": "application/json"}).status_code == 422


def test_api_falls_back_when_model_raises(record, monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setattr(state["scorer"].model, "predict_proba", lambda X: (_ for _ in ()).throw(RuntimeError("x")))
        body = client.post("/predict", json=record).json()
        assert body["scoring_source"] == "fallback_heuristic"


def test_503_when_artifacts_missing(record, monkeypatch, tmp_path):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json=record).status_code == 503
