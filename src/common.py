"""Shared helpers used by training, tests and the Streamlit app."""
from __future__ import annotations

import pandas as pd

FEATURES = [
    "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
    "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
    "pH", "sulphates", "alcohol",
]
TARGET = "quality"
WINE_TYPES = ["red", "white"]
BANDS = ["Low", "Standard", "Premium"]


def score_to_band(score: float, low: float = 5.5, high: float = 6.5) -> str:
    """Map a continuous quality score (3-9 scale) to a quality band."""
    if score < low:
        return "Low"
    if score >= high:
        return "Premium"
    return "Standard"


def true_band(quality: int) -> str:
    """Band of an actual lab/taster score (integer)."""
    return "Low" if quality <= 5 else ("Standard" if quality == 6 else "Premium")


def load_raw(data_dir: str = "data/raw") -> dict[str, pd.DataFrame]:
    out = {}
    for t in WINE_TYPES:
        df = pd.read_csv(f"{data_dir}/winequality-{t}.csv", sep=";")
        out[t] = df
    return out
