"""Vineyard Quality Control Desk - Streamlit app."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.common import FEATURES, TARGET, score_to_band, true_band  # noqa: E402

MODELS = ROOT / "models"
st.set_page_config(page_title="Vineyard QC Desk", page_icon="🍷", layout="wide")

BAND_STYLE = {"Low": ("#c0392b", "Hold back / re-blend"),
              "Standard": ("#d68910", "Approve for standard bottling"),
              "Premium": ("#1e8449", "Candidate for premium line")}


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODELS / "model_meta.json").read_text())
    arts = {}
    for t in ("red", "white"):
        model = joblib.load(MODELS / f"model_{t}.joblib")
        ref = pd.read_csv(MODELS / f"reference_{t}.csv")
        scaler = StandardScaler().fit(ref[FEATURES])
        nn = NearestNeighbors(n_neighbors=5).fit(scaler.transform(ref[FEATURES]))
        arts[t] = {"model": model, "ref": ref, "scaler": scaler, "nn": nn}
    return meta, arts


meta, arts = load_artifacts()
th = meta["band_thresholds"]

# ---------------- sidebar: model version ----------------
with st.sidebar:
    st.header("Model version")
    st.code(meta["model_version"])
    st.caption(f"Git commit: `{meta['git_commit'][:7]}`")
    st.caption(f"MLflow run: `{meta['mlflow_run_id'][:8]}`")
    st.caption(f"DVC data hash: `{meta['dvc_data_md5'][:8]}`")
    st.caption(f"Trained: {meta['trained_at_utc']}")
    m = meta["metrics"]
    st.subheader("Held-out test performance")
    st.dataframe(pd.DataFrame({
        "red": {k: round(v, 3) for k, v in m["separate_red"].items() if k != "cv_mae"},
        "white": {k: round(v, 3) for k, v in m["separate_white"].items() if k != "cv_mae"},
    }))

st.title("🍷 Vineyard Quality Control Desk")
st.write("Enter lab chemistry for a batch to get a predicted quality band before bottling.")

wine_type = st.radio("Wine type", ["red", "white"], horizontal=True)
A = arts[wine_type]
ref = A["ref"]

# ---------------- chemistry sliders ----------------
cols = st.columns(3)
values = {}
for i, f in enumerate(FEATURES):
    lo, hi, med = float(ref[f].min()), float(ref[f].max()), float(ref[f].median())
    step = max((hi - lo) / 200, 0.0001)
    with cols[i % 3]:
        values[f] = st.slider(f, lo, hi, med, step=step, format="%.4f" if hi < 2 else "%.2f")

x = pd.DataFrame([values])[FEATURES]
score = float(A["model"].predict(x)[0])
band = score_to_band(score, th["low"], th["high"])
colour, action = BAND_STYLE[band]

# ---------------- result ----------------
st.divider()
c1, c2, c3 = st.columns([1, 1, 2])
c1.metric("Predicted score (3-9)", f"{score:.2f}")
c2.markdown(f"**Quality band**<br><span style='font-size:2rem;color:{colour}'>{band}</span>",
            unsafe_allow_html=True)
c3.info(action)
st.caption(f"Bands: Low < {th['low']} ≤ Standard < {th['high']} ≤ Premium. "
           "Scores are tasting-panel scale predictions from chemistry only; treat as decision support.")

# ---------------- nearest similar samples ----------------
st.subheader("Most similar historical samples")
dist, idx = A["nn"].kneighbors(A["scaler"].transform(x))
sim = ref.iloc[idx[0]].copy()
sim.insert(0, "distance", np.round(dist[0], 2))
sim.insert(1, "actual band", [true_band(q) for q in sim[TARGET]])
st.dataframe(sim.reset_index(drop=True), width="stretch")
agree = (sim[TARGET].mean())
st.caption(f"Average actual quality of these neighbours: {agree:.2f}")

# ---------------- importance ----------------
with st.expander("What drives quality? (permutation importance)"):
    img = ROOT / "reports" / f"permutation_importance_{wine_type}.png"
    if img.exists():
        st.image(str(img))
