from prophet import Prophet

def train_prophet(df):

    prophet_df = df[['Date', 'Total']].rename(
        columns={'Date': 'ds', 'Total': 'y'}
    )

    prophet_df = prophet_df.dropna()

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    model.fit(prophet_df)

    return model
