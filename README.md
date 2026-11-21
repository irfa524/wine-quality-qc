# Vineyard Quality Control Desk

Predict a wine quality band (Low / Standard / Premium) from lab chemistry before bottling.
Data: [UCI Wine Quality](https://archive.ics.uci.edu/dataset/186) (red 1,599 + white 4,898 rows).

* **Models:** `HistGradientBoostingRegressor` per wine type + a combined model for comparison, permutation importance.
* **App:** Streamlit - chemistry sliders, predicted band, 5 nearest historical samples, model version in sidebar.
* **MLOps:** data versioned with **DVC**, every commit gets an **MLflow** run (tags: `git_commit`, `dvc_data_md5`),
  CI fails if the committed model is stale.

## Run locally
```bash
pip install -r requirements-train.txt
dvc pull            # or copy the csv files into data/raw/
python -m src.train # or: dvc repro
mlflow ui           # (local: mlflow ui --backend-store-uri sqlite:///mlflow.db)
streamlit run app/streamlit_app.py
```

## Workflow
1. Change data / `params.yaml` / `src/` -> `dvc repro` -> `git add dvc.lock models reports data/raw.dvc` -> commit -> push.
2. CI: `dvc pull` -> tests -> `dvc status` (must be clean) -> retrain -> MLflow run linked to the commit SHA.
3. Streamlit Cloud redeploys from `main` automatically; sidebar shows the model version `<git7>.<mlflow-run7>`.
