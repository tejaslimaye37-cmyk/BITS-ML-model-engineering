import numpy as np

from src.config import FEATURES, NUMERIC, TARGET, ID_COL
from src.data_pipeline import build_preprocessor, load_splits


def test_splits_are_disjoint_and_stratified():
    train, val, test = load_splits()
    assert not (set(train[ID_COL]) & set(val[ID_COL])) and not (set(train[ID_COL]) & set(test[ID_COL]))
    assert not (set(val[ID_COL]) & set(test[ID_COL]))
    rates = [d[TARGET].mean() for d in (train, val, test)]
    assert max(rates) - min(rates) < 0.005
    assert abs(len(train) / (len(train) + len(val) + len(test)) - 0.6) < 0.01


def test_preprocessor_is_fitted_on_training_data_only():
    train, val, test = load_splits()
    pre = build_preprocessor().fit(train[FEATURES])
    scaler = pre.named_transformers_["num"].named_steps["scale"]
    train_mean = train[NUMERIC].apply(lambda s: s.fillna(s.median())).mean().to_numpy()
    assert np.allclose(scaler.mean_, train_mean, atol=1e-6)
    assert not np.allclose(scaler.mean_, val[NUMERIC].mean().to_numpy())
