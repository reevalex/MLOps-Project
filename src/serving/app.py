import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

MODEL_PATH = os.getenv("SERVING_MODEL_PATH", "artifacts/model.joblib")
SCHEMA_PATH = os.getenv("FEATURE_SCHEMA_PATH", "artifacts/feature_schema.json")

app = FastAPI(title="Churn Model API", version="1.0")


class Records(BaseModel):
    records: List[Dict[str, Any]]


def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _load_schema():
    if not os.path.exists(SCHEMA_PATH):
        raise None
    with open(SCHEMA_PATH) as f:
        return json.load(f)


model = _load_model()
schema = _load_schema()


@app.get("/")
def root():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict")
def predict(payload: Records):
    if not payload.records:
        raise HTTPException(status_code=400, detail="No records provided")

    if schema:
        expected = set(schema.get("num_cols", []) + schema.get("cat_cols", []))
        missing = [c for c in expected if any(c not in r for r in payload.records)]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    X = pd.DataFrame(payload.records)
    try:
        proba = model.predict_proba(X)[:, 1]
        preds = (proba >= 0.5).astype(int).tolist()
        return {"predictions": preds, "probabilities": proba.astype(float).tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=(str(e)))
