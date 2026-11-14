import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL, DATA_PATH, ID_COL, NUMERIC, SEED, TARGET


def load_raw(path=DATA_PATH):
    return pd.read_csv(path)


def clean(df):
    df = df.copy()
    df[ID_COL] = df[ID_COL].astype(str)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    return df


def make_splits(df, seed=SEED):
    """Stratified 60/20/20 split. Churn prevalence is preserved in every partition."""
    train_val, test = train_test_split(df, test_size=0.2, stratify=df[TARGET], random_state=seed)
    train, val = train_test_split(train_val, test_size=0.25, stratify=train_val[TARGET], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor():
    """Unfitted preprocessor. It must be fitted on the training partition only."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def load_splits():
    return make_splits(clean(load_raw()))
