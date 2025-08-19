from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def infer_feature_types(df: pd.DataFrame, target: str) -> Tuple[List[str], List[str]]:
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in df.select_dtypes(include=["number"]).columns if c != target]
    cat_cols = [c for c in cat_cols if c != target]
    return num_cols, cat_cols


def build_preprocessor(num_cols, cat_cols):
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer"),
            SimpleImputer(strategy="most_frequent"),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    pre = ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)]
    )
    return pre
