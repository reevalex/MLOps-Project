import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from src.common.config import Config


def run(cfg: Config):
    df = pd.read_csv(cfg.raw_path)
    if cfg.id_col and cfg.id_col in df.columns:
        df = df.drop(columns=cfg.id_col)
    assert cfg.target_col in df.columns, f"Target '{cfg.target_col}' not in data"
    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.target_col],
    )
    train_df.to_csv(cfg.train_path, index=False)
    test_df.to_csv(cfg.test_path, index=False)
    print(f"Wrote {cfg.train_path} and {cfg.test_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="params.yaml")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.config)
    run(cfg)
