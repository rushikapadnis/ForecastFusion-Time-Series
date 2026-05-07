
from fastapi import FastAPI
from pydantic import BaseModel
import json

from app.predict import generate_forecast

app = FastAPI(
    title='Sales Forecast API'
)

class ForecastRequest(BaseModel):
    state: str

@app.get('/')
def home():
    return {
        'message': 'Forecast API Running'
    }

@app.post('/forecast')
def forecast(request: ForecastRequest):

    state = request.state

    forecast_values = generate_forecast(state)

    with open(
        'saved_models/best_models.json'
    ) as f:
        best_models = json.load(f)

    return {
        'state': state,
        'best_model': best_models.get(state, 'XGBoost'),
        'forecast_next_8_weeks': forecast_values
    }
