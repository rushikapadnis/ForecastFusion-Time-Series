
# End-to-End Time Series Forecasting System

## Features
- ARIMA / SARIMA
- Prophet
- XGBoost
- LSTM
- Automatic Best Model Selection
- FastAPI REST API
- Lag Features
- Rolling Statistics
- Missing Date Handling
- Weekly Forecasting

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Train Models

```bash
python app/train.py
```

---

## Run API

```bash
uvicorn app.api:app --reload
```

---

## API Endpoint

POST `/forecast`

Request:
```json
{
  "state": "California"
}
```

Response:
```json
{
  "state": "California",
  "best_model": "XGBoost",
  "forecast": [.....]
}
```
