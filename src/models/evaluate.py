import argparse
import json
import joblib
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from src.common.config import Config


def run(cfg: Config):
    df = pd.read_csv(cfg.test_path)
    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    model = joblib.load(cfg.model_path)
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = float(roc_auc_score(y, proba))
    report = classification_report(y, preds, output_dict=True)
    metrics = {"auc", auc, "report", report}

    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({"auc": auc}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="params.yaml")
    args = ap.parse_args()
    cfg = Config(args.config)
    run(cfg)
