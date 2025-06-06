# app.py

import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from pathlib import Path

#
# STEP A: LOAD & PREPARE DATA
#

# 1. Define paths (adjust these if your folder structure is different)
base = Path("data/processed")
actuals_path  = base / "merged_data.csv"
forecast_path = base / "forecast_results.csv"

# 2. Read CSVs
actuals = pd.read_csv(actuals_path, parse_dates=["Date"])
forecast = pd.read_csv(forecast_path, parse_dates=["ds"])

# 3. Rename columns so they match for merging
actuals = actuals.rename(columns={"Date": "ds", "Weekly_Sales": "actual_sales"})
forecast = forecast.rename(columns={"yhat": "forecast_sales"})

# 4. Merge on (Store, ds)
df = pd.merge(
    actuals[["Store", "ds", "actual_sales"]],
    forecast[["Store", "ds", "forecast_sales"]],
    on=["Store", "ds"],
    how="inner",
)

# 5. Compute errors
df["abs_error"] = (df["actual_sales"] - df["forecast_sales"]).abs()
df["pct_error"] = df["abs_error"] / df["actual_sales"] * 100

# 6. Add a “week_of_year” column for the heatmap
df["week_of_year"] = df["ds"].dt.isocalendar().week.astype(int)

# 7. Prepare a melted DataFrame for line chart
df_melted = df.melt(
    id_vars=["Store", "ds"],
    value_vars=["actual_sales", "forecast_sales"],
    var_name="kind",
    value_name="sales",
)

# 8. Precompute MAE by store
mae_by_store = (
    df.groupby("Store")["abs_error"]
      .mean()
      .reset_index()
      .rename(columns={"abs_error": "MAE"})
)
avg_pct_by_store = (
    df.groupby("Store")["pct_error"]
      .mean()
      .reset_index()
      .rename(columns={"pct_error": "AvgPctError"})
)
mae_by_store = mae_by_store.merge(avg_pct_by_store, on="Store")
mae_by_store = mae_by_store.sort_values("MAE", ascending=False)

# 9. Build a “long” DataFrame for the heatmap
heatmap_df = df.pivot_table(
    index="Store",
    columns="week_of_year",
    values="abs_error",
    aggfunc="mean",
    fill_value=0
).reset_index()
heatmap_long = heatmap_df.melt(
    id_vars="Store",
    var_name="Week",
    value_name="AbsError"
)

#
# STEP B: CREATE PLOTLY FIGURES (SOME AS TEMPLATES, OTHERS VIA CALLBACKS)
#

# ---- FIGURE 1: Base (empty) template for Actual vs Forecast line chart ----

# We’ll create a “blank” template with no data. Then a Dash callback will fill it once the user picks a store.
fig1_template = px.line(
    pd.DataFrame({"ds": [], "sales": [], "kind": [], "Store": []}),
    x="ds",
    y="sales",
    color="kind",
    title="Actual vs. Forecast: Weekly Sales (Select a Store Below)",
    labels={"ds": "Week (Date)", "sales": "Sales ($)", "kind": "Series"},
)

# Style adjustments for Fig1 template
fig1_template.update_layout(
    xaxis=dict(title="Week (Date)"),
    yaxis=dict(title="Sales ($)"),
    legend_title="Series",
)
fig1_template.update_traces(line=dict(width=3))
# We'll recolor the two possible kinds inside the callback.

# ---- FIGURE 2: MAE by Store (static, since it doesn't depend on any dropdown) ----

fig2 = px.bar(
    mae_by_store,
    x="Store",
    y="MAE",
    color="MAE",
    color_continuous_scale="Reds",
    labels={"MAE": "Mean Absolute Error ($)", "Store": "Store ID"},
    title="Average Absolute Forecast Error (MAE) by Store",
    hover_data={
        "MAE": ":,.0f",
        "AvgPctError": ":.2f"
    },
)
fig2.update_layout(coloraxis_showscale=False)

# ---- FIGURE 3: Error Heatmap (static) ----

fig3 = px.density_heatmap(
    heatmap_long,
    x="Week",
    y="Store",
    z="AbsError",
    color_continuous_scale="OrRd",
    labels={"AbsError": "Absolute Error ($)", "Store": "Store ID"},
    title="Weekly Absolute Forecast Error by Store (Heatmap)",
)
fig3.update_xaxes(dtick=4)
fig3.update_yaxes(autorange="reversed")

#
# STEP C: SET UP THE DASH APP & CALLBACK
#

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1(
        "Supply Chain Demand Forecast Dashboard (Dash)", 
        style={"textAlign": "center", "marginBottom": "20px"}
    ),

    # Dropdown to select a Store
    html.Div([
        html.Label("Select Store:"),
        dcc.Dropdown(
            id="store-dropdown",
            options=[{"label": f"Store {s}", "value": s} for s in sorted(df["Store"].unique())],
            value=1,            # default to Store 1
            clearable=False,
            style={"width": "200px"}
        ),
    ], style={"textAlign": "center"}),

    # Actual vs Forecast line chart (will be populated via callback)
    html.Div([
        dcc.Graph(id="actual-vs-forecast", figure=fig1_template),
    ], style={"width": "100%", "display": "inline-block", "marginTop": "20px"}),

    # Below, side-by-side: MAE bar chart (static) and error heatmap (static)
    html.Div([
        html.Div([
            dcc.Graph(id="mae-by-store", figure=fig2)
        ], style={"width": "48%", "display": "inline-block"}),

        html.Div([
            dcc.Graph(id="error-heatmap", figure=fig3)
        ], style={"width": "48%", "display": "inline-block", "float": "right"}),
    ], style={"marginTop": "40px"}),
])

#
# STEP D: CALLBACK TO UPDATE FIGURE 1 WHEN STORE DROPDOWN CHANGES
#

@app.callback(
    Output("actual-vs-forecast", "figure"),
    Input("store-dropdown", "value")
)
def update_line_chart(selected_store):
    # Filter df_melted to only the selected store:
    filtered = df_melted[df_melted["Store"] == selected_store]

    # Build a fresh line chart for THIS store:
    fig = px.line(
        filtered,
        x="ds",
        y="sales",
        color="kind",
        labels={"ds": "Week (Date)", "sales": "Sales ($)", "kind": "Series"},
        title=f"Actual vs. Forecast: Weekly Sales (Store {selected_store})",
    )

    # Style actual_sales vs forecast_sales differently
    fig.update_traces(
        selector=dict(name="actual_sales"),
        line=dict(color="royalblue", width=3, dash="solid"),
        name="Actual Sales"
    )
    fig.update_traces(
        selector=dict(name="forecast_sales"),
        line=dict(color="darkorange", width=3, dash="dash"),
        name="Forecast Sales"
    )

    fig.update_layout(
        xaxis=dict(title="Week (Date)"),
        yaxis=dict(title="Sales ($)"),
    )
    return fig

#
# STEP E: RUN THE APP
#

if __name__ == "__main__":
    app.run(debug=True)   # <-- replaces app.run_server(debug=True) in newer Dash versions
