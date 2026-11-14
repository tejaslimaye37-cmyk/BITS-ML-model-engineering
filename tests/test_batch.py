import pandas as pd

from serving.predict_batch import main
from src.config import DATA_PATH


def test_batch_cli_scores_and_rejects(tmp_path):
    df = pd.read_csv(DATA_PATH).drop(columns=["Churn"]).head(50)
    df.loc[3, "tenure"] = -4
    inp, out = tmp_path / "in.csv", tmp_path / "out.csv"
    df.to_csv(inp, index=False)
    assert main(["--input", str(inp), "--output", str(out)]) == 0
    scored = pd.read_csv(out)
    assert len(scored) == 49 and scored["churn_probability"].between(0, 1).all()
    assert (tmp_path / "out.rejects.csv").exists()


def test_batch_cli_fails_cleanly_on_missing_columns(tmp_path):
    inp = tmp_path / "bad.csv"
    pd.DataFrame({"a": [1]}).to_csv(inp, index=False)
    assert main(["--input", str(inp), "--output", str(tmp_path / "o.csv")]) == 2
