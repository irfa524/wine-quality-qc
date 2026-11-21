import json
from pathlib import Path

import pandas as pd
import pytest

from src.common import FEATURES, TARGET, load_raw, score_to_band, true_band


def test_band_mapping():
    assert score_to_band(5.4) == "Low"
    assert score_to_band(5.5) == "Standard"
    assert score_to_band(6.49) == "Standard"
    assert score_to_band(6.5) == "Premium"
    assert [true_band(q) for q in (4, 5, 6, 7, 8)] == ["Low", "Low", "Standard", "Premium", "Premium"]


def test_raw_schema():
    for t, df in load_raw().items():
        assert list(df.columns) == FEATURES + [TARGET], t
        assert df.isna().sum().sum() == 0
        assert df[TARGET].between(3, 9).all()


@pytest.mark.skipif(not Path("models/model_meta.json").exists(), reason="model not trained yet")
def test_model_bundle_predicts():
    import joblib
    meta = json.loads(Path("models/model_meta.json").read_text())
    assert meta["model_version"] and meta["mlflow_run_id"]
    for t in ("red", "white"):
        m = joblib.load(f"models/model_{t}.joblib")
        ref = pd.read_csv(f"models/reference_{t}.csv")
        p = m.predict(ref[FEATURES].head(5))
        assert ((p > 2) & (p < 10)).all()
