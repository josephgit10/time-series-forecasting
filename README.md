# End-to-End Demand Forecasting System

> An advanced, containerised system for multi-store weekly demand forecasting, featuring automated hyperparameter tuning and an interactive dashboard.

---

## ✨ Key Features

| Area                     | Tech                                     | Highlights                                                                                             |
| ------------------------ | ---------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Data Prep & Features** | `pandas`, `numpy`                        | ETL script with advanced feature engineering (lags, rolling stats, time-based features).               |
| **Models**               | **Prophet**, **LSTM/Keras**, **XGBoost** | Three powerful models for robust forecasting.                                                          |
| **Hyperparameter Tuning**| **Optuna**                               | Automated hyperparameter search for each model to minimize RMSE and improve accuracy.                  |
| **Orchestration**        | **Apache Airflow 2.9**                   | DAG to preprocess data and run all three forecasting models in parallel.                               |
| **Dashboard**            | **Plotly Dash**                          | Interactive dashboard to visualize forecasts, compare models, and view performance metrics (RMSE).     |
| **Serving (Optional)**   | **FastAPI + Uvicorn**                    | A microservice to serve predictions (can be extended).                                                 |
| **Reproducibility**      | **Docker**                               | Fully containerised environment for consistent setup and execution.                                    |

---

## 🚀 Quick Start

This guide will walk you through setting up the project and running the forecasting pipeline.

### 1. Initial Setup

Clone the repository and set up the Python environment.

```bash
# Clone the repository
git clone https://github.com/josephgit10/time-series-forecasting.git
cd time-series-forecasting

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

### 2. Add Raw Data

The project expects three CSV files with raw data in the `data/raw/` directory. This directory is not tracked by Git.

```bash
# Create the directory if it doesn't exist
mkdir -p data/raw

# Copy your data files into the directory
# e.g., cp ~/Downloads/sales.csv data/raw/sales_data.csv
```
*See `data/README.md` for details on the required data format.*

### 3. Preprocess Data

Run the preprocessing script to merge the raw data and create engineered features.

```bash
python src/preprocessing.py
```
This will generate a `merged_data.csv` file in the `data/processed/` directory.

### 4. Run the Forecasting Pipeline

The forecasting pipeline will perform hyperparameter tuning for each model and generate forecast files. You can run the models individually.

```bash
# Run Prophet model
python src/forecasting_pipeline.py --model prophet

# Run XGBoost model
python src/forecasting_pipeline.py --model xgboost

# Run LSTM model
python src/forecasting_pipeline.py --model lstm
```
This will generate `forecast_results_<model_name>.csv` files in `data/processed/`, which include the 52-week forecast and the RMSE score for each store.

*Note: The hyperparameter tuning process can be time-consuming, especially for the LSTM model.*

### 5. Visualize Results with the Dashboard

Launch the interactive Plotly Dash application to visualize the results.

```bash
python notebooks/scripts/app.py
```
Open your web browser and navigate to **http://127.0.0.1:8050/** to see the dashboard.

---

## 📊 Interactive Dashboard

The dashboard provides an intuitive interface to analyze the forecasting results:
-   **Store Selection:** Use the dropdown menu to select a store.
-   **Forecast Visualization:** The main graph displays the historical sales data along with the forecasts from all three models.
-   **Best Model Highlight:** The forecast line of the model with the lowest RMSE is highlighted for easy identification.
-   **Performance Metrics:** The RMSE scores for each model are displayed, allowing for a quick comparison of their performance.

---

## ⚙️ Hyperparameter Tuning with Optuna

The forecasting pipeline uses **Optuna** to automatically find the best hyperparameters for each model, aiming to minimize the Root Mean Squared Error (RMSE). This ensures that the models are optimized for the best possible performance on the given data.

---

## 🐳 Running with Docker and Airflow

For a fully automated and orchestrated workflow, you can use Docker and Airflow.

```bash
# Build and start the Docker containers
docker-compose up --build
```

-   **Airflow UI:** [http://localhost:8080](http://localhost:8080)
-   The `daily_forecast_pipeline` DAG will automatically run the preprocessing and forecasting for all three models.

---

## 🧰 Development & Tests

```bash
# Lint and format code
black src notebooks/scripts airflow/dags

# Run tests (if any)
pytest -q
```

---

## 📜 Data Folder Policy

The `data/raw/` and `data/processed/` directories are not tracked by Git. Add your CSVs locally or mount them via a Docker volume. See `data/README.md` for column documentation.
