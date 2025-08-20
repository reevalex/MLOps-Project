import argparse
import os
import pandas as pd
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
from src.common.config import Config


def run(cfg: Config):
    ref = pd.read_csv(cfg.test_path)
    cur_path = "data/production/current.csv"
    if not os.path.exists(cur_path):
        raise FileNotFoundError("Put a CSV at data/production.current.csv for monitoring")
    cur = pd.read_csv(cur_path)

    report = Report(metrics=[DatasetDriftMetric(), ColumnDriftMetric(column_name=cfg.target_col)])
    report.run(reference_data=ref, current_data=cur)

    os.makedirs("artifacts", exist_ok=True)
    out_html = "artifacts/evidently_report.html"
    report.save_html(out_html)
    print(f"Saved drift report to {out_html}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="params.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    run(cfg)
