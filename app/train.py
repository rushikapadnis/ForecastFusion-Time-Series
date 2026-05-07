import json
import os
import joblib
import pandas as pd

from app.preprocessing import load_and_prepare_data
from app.feature_engineering import create_features
from app.utils import calculate_rmse

from models.xgboost_model import train_xgb
from models.arima_model import train_arima
from models.prophet_model import train_prophet
from models.lstm_model import train_lstm

DATA_PATH = 'data/Forecasting Case.csv'

FEATURE_COLUMNS = [
    'lag_1',
    'lag_7',
    'lag_30',
    'rolling_mean_7',
    'rolling_std_7',
    'month',
    'week',
    'day_of_week',
    'holiday_flag'
]

os.makedirs('saved_models', exist_ok=True)

def main():

    print("Loading dataset...")

    df = load_and_prepare_data(DATA_PATH)

    results = []

    best_models = {}

    for state in df['State'].unique():

        print(f"Training models for: {state}")

        state_df = df[df['State'] == state].copy()

        state_df = create_features(state_df)

        if len(state_df) < 40:
            print(f"Skipping {state} due to insufficient rows")
            continue

        X = state_df[FEATURE_COLUMNS]
        y = state_df['Total']

        split_index = int(len(state_df) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        scores = {}

        # ---------------- XGBoost ----------------
        try:
            xgb = train_xgb(X_train, y_train)

            xgb_pred = xgb.predict(X_test)

            xgb_rmse = calculate_rmse(y_test, xgb_pred)

            scores['XGBoost'] = float(xgb_rmse)

            joblib.dump(
                xgb,
                f'saved_models/{state}_xgb.pkl'
            )

            print(f"XGBoost RMSE: {xgb_rmse:.2f}")

        except Exception as e:
            print(f"XGBoost failed for {state}: {e}")

        # ---------------- ARIMA / SARIMA ----------------
        try:
            y_train_arima = y_train.copy()
            y_train_arima.index = pd.date_range(
                start='2020-01-01',
                periods=len(y_train_arima),
                freq='W'
            )

            arima = train_arima(y_train_arima)

            arima_pred = arima.forecast(len(y_test))

            arima_rmse = calculate_rmse(y_test, arima_pred)

            scores['ARIMA/SARIMA'] = float(arima_rmse)

            print(f"ARIMA RMSE: {arima_rmse:.2f}")

        except Exception as e:
            print(f"ARIMA failed for {state}: {e}")

        # ---------------- Prophet ----------------
        try:
            prophet = train_prophet(
                state_df[['Date', 'Total']]
            )

            future = prophet.make_future_dataframe(
                periods=len(y_test),
                freq='W'
            )

            forecast = prophet.predict(future)

            prophet_pred = forecast['yhat'].tail(len(y_test))

            prophet_rmse = calculate_rmse(
                y_test.values,
                prophet_pred.values
            )

            scores['Prophet'] = float(prophet_rmse)

            print(f"Prophet RMSE: {prophet_rmse:.2f}")

        except Exception as e:
            print(f"Prophet failed for {state}: {e}")

        # ---------------- LSTM ----------------
        try:
            lstm = train_lstm(
                X_train.values,
                y_train.values
            )

            lstm_pred = lstm.predict(
                X_test.values.reshape(
                    (X_test.shape[0], X_test.shape[1], 1)
                ),
                verbose=0
            )

            lstm_rmse = calculate_rmse(
                y_test,
                lstm_pred.flatten()
            )

            scores['LSTM'] = float(lstm_rmse)

            print(f"LSTM RMSE: {lstm_rmse:.2f}")

        except Exception as e:
            print(f"LSTM failed for {state}: {e}")

        if len(scores) == 0:
            continue

        best_model = min(scores, key=scores.get)

        best_models[state] = best_model

        results.append({
            'State': state,
            'Best_Model': best_model,
            'Best_RMSE': scores[best_model]
        })

        print(f"Best model for {state}: {best_model}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        'saved_models/model_results.csv',
        index=False
    )

    with open(
        'saved_models/best_models.json',
        'w'
    ) as f:
        json.dump(best_models, f)

    print("\nTraining Completed Successfully")

if __name__ == "__main__":
    main()
