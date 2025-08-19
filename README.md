```
mlops-e2e-churn/
├── README.md
├── params.yaml
├── dvc.yaml
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── data/
│   ├── raw/                # put churn.csv here (not committed)
│   ├── processed/          # train/test splits (DVC-managed)
│   └── production/         # live data + predictions for monitoring
├── artifacts/
│   ├── model.joblib        # trained model
│   ├── feature_schema.json # features learned at train-time
│   └── metrics.json
├── src/
│   ├── common/
│   │   └── config.py
│   ├── data/
│   │   ├── split_dataset.py
│   │   └── validate_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── serving/
│   │   └── app.py
│   └── monitoring/
│       └── monitor.py
└── tests/
    ├── test_config.py
    └── test_pipeline.py
```