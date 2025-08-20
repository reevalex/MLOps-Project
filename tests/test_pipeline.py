import pandas as pd
from src.features.build_features import infer_feature_types
from src.models.train import build_pipeline


def test_build_pipeline_on_synth():
    df = pd.DataFrame(
        {"f_num1": [1.0, 2.0, 3.0], "f_cat1": ["a", "b", "c"], "churn": [0, 1, 0]}
    )
    num_cols, cat_cols = infer_feature_types(df, "churn")
    pipe = build_pipeline(1.0, 200, num_cols, cat_cols)
    y = df["churn"]
    X = df.drop(columns=["churn"])
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(y)
