from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/forecast/{store_id}")
def get_forecast(store_id: int):
    df = pd.read_csv('/app/data/processed/forecast_results.csv', parse_dates=['ds'])
    store_forecast = df[df['Store'] == store_id]
    return store_forecast.to_dict(orient='records')