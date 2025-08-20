import argparse
import pandas as pd
from src.common.config import Config


def _basic_checks(df: pd.DataFrame, target: str):
    assert target in df.columns, f"Missing target column: {target}"
    assert df[target].notna().all(), "Target column contains NaNs"

    unique = set(df[target].unique())
    assert unique.issubset({0, 1}), f"Target should be 0/1 only, got {unique}"


def run(cfg: Config):
    train = pd.read_csv(cfg.train_path)
    test = pd.read_csv(cfg.test_path)
    _basic_checks(train, cfg.target_col)
    _basic_checks(test, cfg.target_col)
    assert set(train.columns) == set(test.columns), "Train/Test column mismatch"
    print("Data validation passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="params.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    run(cfg)
