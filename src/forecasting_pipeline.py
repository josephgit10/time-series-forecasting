import os
from prophet import Prophet
import pandas as pd

def prophet_forecast(store_df):
    # Here store_df['Date'] is already a Timestamp thanks to dayfirst parsing
    prophet_df = store_df[['Date', 'Weekly_Sales']].rename(
        columns={'Date': 'ds', 'Weekly_Sales': 'y'}
    )
    model = Prophet()
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=52, freq='W')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

def run_forecasting():
    df = pd.read_csv('/app/data/processed/merged_data.csv', parse_dates=['Date'])
    forecasts = []

    for store in df['Store'].unique():
        store_df = df[df['Store'] == store]
        forecast = prophet_forecast(store_df)
        forecast['Store'] = store
        forecasts.append(forecast)

    forecast_df = pd.concat(forecasts)
    os.makedirs('/app/data/processed', exist_ok=True)
    forecast_df.to_csv('/app/data/processed/forecast_results.csv', index=False)

if __name__ == "__main__":
    run_forecasting()
