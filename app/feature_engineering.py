
import numpy as np

def create_features(df):

    df = df.copy()

    df['lag_1'] = df['Total'].shift(1)
    df['lag_7'] = df['Total'].shift(7)
    df['lag_30'] = df['Total'].shift(30)

    df['rolling_mean_7'] = df['Total'].rolling(7).mean()
    df['rolling_std_7'] = df['Total'].rolling(7).std()

    df['month'] = df['Date'].dt.month
    df['week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['Date'].dt.dayofweek

    df['holiday_flag'] = np.where(df['month'].isin([11, 12]), 1, 0)

    df = df.dropna()

    return df
