import argparse
import json
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import mlflow
from src.common.config import Config
from src.features.build_features import infer_feature_types, build_preprocessor


def build_pipeline(C: float, max_iter: int, num_cols, cat_cols) -> Pipeline:
    pre = build_preprocessor(num_cols, cat_cols)
    clf = LogisticRegression(C, max_iter, n_jobs=None, solver="lbfgs")
    pipe = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
    return pipe


def run(cfg: Config):
    os.makedirs(os.path.dirname(cfg.model_path), exist_ok=True)

    df = pd.read_csv(cfg.train_path)
    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    num_cols, cat_cols = infer_feature_types(df, cfg.target_col)
    pipe = build_pipeline(cfg.C, cfg.max_iter, num_cols, cat_cols)

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment)
    with mlflow.start_run(run_name="train"):
        mlflow.log_params(
            {
                "model": "logreg",
                "C": cfg.C,
                "max_iter": cfg.max_iter,
                "num_cols": num_cols,
                "cat_cols": cat_cols,
            }
        )
        pipe.fit(X, y)

        try:
            p = pipe.predict_proba(X)[:, 1]
            mlflow.log_metric("train_auc", float(roc_auc_score(y, p)))
        except Exception:
            pass

        joblib.dump(pipe, cfg.model_path)
        schema = {"target": cfg.target_col, "num_cols": num_cols, "cat_cols": cat_cols}
        with open(cfg.feature_schema_path, "w") as f:
            json.dump(schema, f, indent=2)

        mlflow.log_artifact(cfg.model_path)
        mlflow.log_artifact(cfg.feature_schema_path)
        print(f"Saved model to {cfg.model_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="params.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    run(cfg)
