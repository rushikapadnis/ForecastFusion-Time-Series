
import joblib
import pandas as pd

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

def generate_forecast(state):

    model = joblib.load(
        f'saved_models/{state}_xgb.pkl'
    )

    future_rows = []

    for i in range(8):

        row = {
            'lag_1': 100,
            'lag_7': 100,
            'lag_30': 100,
            'rolling_mean_7': 100,
            'rolling_std_7': 10,
            'month': 5,
            'week': 20 + i,
            'day_of_week': 1,
            'holiday_flag': 0
        }

        future_rows.append(row)

    future_df = pd.DataFrame(future_rows)

    preds = model.predict(
        future_df[FEATURE_COLUMNS]
    )

    return preds.tolist()
