from src.common.config import Config


def test_config_loads():
    cfg = Config.from_yaml("params.yaml")
    assert cfg.target_col
