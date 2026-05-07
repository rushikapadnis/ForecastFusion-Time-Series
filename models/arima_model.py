from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_arima(series):

    series = series.copy()

    # Ensure proper datetime index
    if not hasattr(series.index, 'freq'):
        try:
            series = series.asfreq('W')
        except Exception:
            pass

    model = SARIMAX(
        series,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    fitted = model.fit(disp=False)

    return fitted
