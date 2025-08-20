FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -q
COPY artifacts/ ./artifacts/
COPY src/ ./src/
ENV SERVING_MODEL_PATH=artifacts/model.joblib \
    FEATURE_SCHEMA_PATH=artifacts/feature_schema.json
EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]