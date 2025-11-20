# Real-Time Portfolio Anomaly Detection Service

This repository implements the Isolation Forest–based ML service described in `model_requirement.md`.

## Structure

- `app.py` – FastAPI + Socket.IO service exposing `/analyze` and `/ws`.
- `model_train.py` – CLI trainer that persists `models/iso_model.pkl`.
- `feature_extractor.py` – Converts nested backend payloads into numeric features.
- `data/training_data.csv` – Minimal example dataset for local training runs.
- `requirements.txt` – Python dependencies.

## Quickstart

1. **Install dependencies**
   ```bash
   python -m venv .venv
   .venv/Scripts/activate
   pip install -r requirements.txt
   ```
2. **Train the model**
   ```bash
   python model_train.py
   ```
3. **Run the API**
   ```bash
   uvicorn app:app --reload --port 8001
   ```

Incoming backend posts to `POST /analyze` will emit `ml_recommendations` via Socket.IO mounted at `/ws`.

