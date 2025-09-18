import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from pathlib import Path

# --- Data Loading and Preparation ---

def load_data():
    base_path = Path("data/processed")
    actuals_path = base_path / "merged_data.csv"

    actuals_df = pd.read_csv(actuals_path, parse_dates=["Date"])

    forecast_files = {
        'prophet': base_path / "forecast_results_prophet.csv",
        'xgboost': base_path / "forecast_results_xgboost.csv",
        'lstm': base_path / "forecast_results_lstm.csv",
    }

    all_forecasts = []
    for model_name, path in forecast_files.items():
        try:
            df = pd.read_csv(path, parse_dates=["ds"])
            all_forecasts.append(df)
        except FileNotFoundError:
            print(f"Warning: {path} not found. Run the forecasting pipeline for {model_name}.")

    if not all_forecasts:
        raise Exception("No forecast files found. Please run the forecasting pipelines.")

    forecasts_df = pd.concat(all_forecasts)

    return actuals_df, forecasts_df

actuals, all_forecasts = load_data()

# --- App Layout ---

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Demand Forecasting Dashboard", style={'textAlign': 'center'}),

    dcc.Dropdown(
        id='store-dropdown',
        options=[{'label': f"Store {s}", 'value': s} for s in sorted(actuals['Store'].unique())],
        value=1,
        clearable=False
    ),

    dcc.Graph(id='forecast-graph'),

    html.Div(id='rmse-scores')
])

# --- Callbacks ---

@app.callback(
    [Output('forecast-graph', 'figure'),
     Output('rmse-scores', 'children')],
    [Input('store-dropdown', 'value')]
)
def update_dashboard(selected_store):
    store_actuals = actuals[actuals['Store'] == selected_store]
    store_forecasts = all_forecasts[all_forecasts['Store'] == selected_store]

    fig = go.Figure()

    # Add actual sales trace
    fig.add_trace(go.Scatter(
        x=store_actuals['Date'],
        y=store_actuals['Weekly_Sales'],
        mode='lines',
        name='Actual Sales',
        line=dict(color='black', width=3)
    ))

    # Add forecast traces
    model_colors = {'prophet': 'blue', 'xgboost': 'green', 'lstm': 'red'}
    best_model = store_forecasts.loc[store_forecasts['rmse'].idxmin()]

    for model_name, group in store_forecasts.groupby('model'):
        is_best = model_name == best_model['model']
        fig.add_trace(go.Scatter(
            x=group['ds'],
            y=group['yhat'],
            mode='lines',
            name=f"{model_name.capitalize()} Forecast",
            line=dict(
                color=model_colors.get(model_name, 'grey'),
                width=4 if is_best else 2,
                dash='dash'
            )
        ))

    fig.update_layout(
        title=f"Sales Forecast for Store {selected_store}",
        xaxis_title="Date",
        yaxis_title="Weekly Sales",
        legend_title="Models"
    )

    # RMSE scores display
    rmse_html = [html.H3("Model Performance (RMSE)")]
    for _, row in store_forecasts.groupby('model').first().iterrows():
        is_best = row['model'] == best_model['model']
        style = {'fontWeight': 'bold'} if is_best else {}
        rmse_html.append(html.P(f"{row['model'].capitalize()}: {row['rmse']:.2f}", style=style))

    return fig, html.Div(rmse_html)

if __name__ == '__main__':
    app.run(debug=True)
