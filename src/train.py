"""Train red / white / combined gradient-boosted regressors, log to MLflow.

Run:  python -m src.train
Each run is tagged with the Git commit and the DVC data hash, and writes
models/model_meta.json that the Streamlit app displays as its model version.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from src.common import FEATURES, TARGET, WINE_TYPES, load_raw, score_to_band, true_band

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]  # our own model
MODELS = Path("models")
REPORTS = Path("reports")


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def git_info() -> dict:
    sha = os.getenv("GITHUB_SHA") or _git("rev-parse", "HEAD") or "nogit"
    dirty = bool(_git("status", "--porcelain", "--untracked-files=no", "--", ".", ":!models", ":!reports", ":!dvc.lock"))  # ignore DVC outputs
    branch = os.getenv("GITHUB_REF_NAME") or _git("rev-parse", "--abbrev-ref", "HEAD")
    return {"sha": sha, "short": sha[:7], "dirty": dirty, "branch": branch}


def data_hash() -> str:
    """md5 of the DVC-tracked raw data directory (from data/raw.dvc)."""
    p = Path("data/raw.dvc")
    if p.exists():
        d = yaml.safe_load(p.read_text())
        return d["outs"][0]["md5"]
    return "untracked"


def make_model(p: dict, seed: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(random_state=seed, **p)


def evaluate(y_true: pd.Series, pred: np.ndarray, th: dict) -> dict:
    pred = np.asarray(pred)
    rounded = np.clip(np.rint(pred), 3, 9).astype(int)
    bands_p = [score_to_band(s, th["low"], th["high"]) for s in pred]
    bands_t = [true_band(q) for q in y_true]
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
        "qwk": float(cohen_kappa_score(y_true, rounded, weights="quadratic")),
        "within_1": float(np.mean(np.abs(rounded - y_true.values) <= 1)),
        "band_accuracy": float(np.mean(np.array(bands_p) == np.array(bands_t))),
    }


def main() -> None:
    params = yaml.safe_load(open("params.yaml"))
    seed, th = params["seed"], params["band_thresholds"]
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)

    raw = load_raw()
    frames = {}
    for t, df in raw.items():
        if params["drop_duplicates"]:
            df = df.drop_duplicates().reset_index(drop=True)
        frames[t] = df

    splits = {}
    for t, df in frames.items():
        tr, te = train_test_split(
            df, test_size=params["test_size"], random_state=seed, stratify=df[TARGET]
        )
        splits[t] = (tr.reset_index(drop=True), te.reset_index(drop=True))

    g = git_info()
    dhash = data_hash()

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")  # local fallback
    mlflow.set_experiment("wine-quality-qc")
    with mlflow.start_run(run_name=f"{g['short']}{'-dirty' if g['dirty'] else ''}") as run:
        run_id = run.info.run_id
        mlflow.set_tags({
            "git_commit": g["sha"], "git_branch": g["branch"], "git_dirty": str(g["dirty"]),
            "dvc_data_md5": dhash,
            "ci_run_url": (
                f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}"
                f"/actions/runs/{os.getenv('GITHUB_RUN_ID')}" if os.getenv("GITHUB_RUN_ID") else "local"
            ),
        })
        mlflow.log_params({f"model_{k}": v for k, v in params["model"].items()})
        mlflow.log_params({"seed": seed, "test_size": params["test_size"],
                           "drop_duplicates": params["drop_duplicates"],
                           "band_low": th["low"], "band_high": th["high"]})
        mlflow.log_artifact("params.yaml")

        metrics, bundle = {}, {}

        # 1) separate red / white models
        for t in WINE_TYPES:
            tr, te = splits[t]
            model = make_model(params["model"], seed)
            cv = -cross_val_score(model, tr[FEATURES], tr[TARGET], cv=5,
                                  scoring="neg_mean_absolute_error")
            model.fit(tr[FEATURES], tr[TARGET])
            m = evaluate(te[TARGET], model.predict(te[FEATURES]), th)
            m["cv_mae"] = float(cv.mean())
            metrics[f"separate_{t}"] = m
            bundle[t] = model
            mlflow.sklearn.log_model(model, name=f"model_{t}", skops_trusted_types=TRUSTED_TYPES)

            # permutation importance on held-out test set (MAE scoring)
            pi = permutation_importance(
                model, te[FEATURES], te[TARGET], n_repeats=params["permutation_importance"]["n_repeats"],
                random_state=seed, scoring="neg_mean_absolute_error")
            imp = (pd.DataFrame({"feature": FEATURES, "importance_mean": pi.importances_mean,
                                 "importance_std": pi.importances_std})
                   .sort_values("importance_mean", ascending=False))
            imp.to_csv(REPORTS / f"permutation_importance_{t}.csv", index=False, lineterminator="\n")
            top = imp.iloc[::-1]
            plt.figure(figsize=(6, 4))
            plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
            plt.xlabel("Increase in MAE when feature is shuffled")
            plt.title(f"Permutation importance - {t} wine")
            plt.tight_layout()
            plt.savefig(REPORTS / f"permutation_importance_{t}.png", dpi=130)
            plt.close()

        # 2) one combined model (+ is_red flag) evaluated on each type's test set
        def with_flag(df, t):
            d = df[FEATURES].copy()
            d["is_red"] = int(t == "red")
            return d
        Xc = pd.concat([with_flag(splits[t][0], t) for t in WINE_TYPES])
        yc = pd.concat([splits[t][0][TARGET] for t in WINE_TYPES])
        comb = make_model(params["model"], seed).fit(Xc, yc)
        for t in WINE_TYPES:
            te = splits[t][1]
            metrics[f"combined_on_{t}"] = evaluate(te[TARGET], comb.predict(with_flag(te, t)), th)

        for name, m in metrics.items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in m.items()})

        # save the app bundle (small -> kept in Git, not DVC)
        for t in WINE_TYPES:
            joblib.dump(bundle[t], MODELS / f"model_{t}.joblib", compress=3)
            frames[t].to_csv(MODELS / f"reference_{t}.csv", index=False, lineterminator="\n")

        meta = {
            "model_version": f"{g['short']}{'-dirty' if g['dirty'] else ''}.{run_id[:7]}",
            "git_commit": g["sha"],
            "mlflow_run_id": run_id,
            "mlflow_experiment": "wine-quality-qc",
            "dvc_data_md5": dhash,
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "band_thresholds": th,
            "metrics": metrics,
        }
        (MODELS / "model_meta.json").write_text(json.dumps(meta, indent=2), newline="\n")
        (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), newline="\n")
        mlflow.log_artifact(str(MODELS / "model_meta.json"))
        mlflow.log_artifacts(str(REPORTS), artifact_path="reports")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()