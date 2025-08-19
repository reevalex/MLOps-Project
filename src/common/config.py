import yaml
from dataclasses import dataclass


@dataclass
class Config:
    raw_path: str
    train_path: str
    test_path: str
    id_col: str
    target_col: str
    test_size: float
    random_state: int
    model_type: str
    C: float
    max_iter: int
    mlflow_tracking_uri: str
    experiment: str
    model_path: str
    feature_schema_path: str

    @staticmethod
    def from_yaml(path: str) -> "Config":
        with open(path, "r") as f:
            y = yaml.safe_load(f)
        return Config(
            raw_path=y["data"]["raw_path"],
            train_path=y["data"]["train_path"],
            test_path=y["data"]["test_path"],
            id_col=y["data"]["id_col"],
            target_col=y["data"]["target_col"],
            test_size=y["data"]["test_size"],
            random_state=y["data"]["random_state"],
            model_type=y["model"]["type"],
            C=float(y["model"]["C"]),
            max_iter=y["model"]["max_iter"],
            mlflow_tracking_uri=y["tracking"]["mlflow_tracking_uri"],
            experiment=y["tracking"]["experiment"],
            model_path=y["serving"]["model_path"],
            feature_schema_path=y["serving"]["feature_schema_path"],
        )
