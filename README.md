# Time‑Series Forecasting for Supply Chain Logistics

> **Multi‑store weekly demand forecasting with Prophet, LSTM, XGBoost, Airflow orchestration, and a FastAPI micro‑service – fully containerised with Docker.**

---

## ✨ Key Features

| Area                | Tech                                          | Highlights                                                                     |
| ------------------- | --------------------------------------------- | ------------------------------------------------------------------------------ |
| **Data prep**       | `pandas`, `numpy`                             | ETL script normalises three raw Walmart‑style CSVs into a single weekly panel. |
| **Models**          | **Prophet**, **LSTM/Keras**, **XGBoost**      | Switchable back‑ends, hyper‑param search via Keras Tuner / Optuna.             |
| **Orchestration**   | **Apache Airflow 2.9**                        | DAG schedules daily refresh → preprocessing · training · inference · upload.   |
| **Serving**         | **FastAPI + Uvicorn**                         | `/predict` endpoint returns forecast for any (Store, Date).                    |
| **Reproducibility** | Docker, `requirements.txt`, `environment.yml` | One‑command spin‑up via `docker‑compose`.                                      |
| **Dashboard**       | Plotly Dash (optional)                        | Actual vs Forecast lines, MAE rank bar, error heat‑map.                        |

---

## 🚀 Quick start

```bash
# 1 Clone
$ git clone https://github.com/<you>/time-series-forecasting.git
$ cd time-series-forecasting

# 2 Create env
$ python -m venv venv && source venv/bin/activate  # or use conda
$ pip install -r requirements.txt

# 3 Add raw data (NOT tracked by Git)
$ cp ~/Downloads/*.csv data/raw/

# 4 Preprocess → merged_data.csv
$ python src/preprocessing.py

# 5 Generate Prophet forecasts → forecast_results.csv
$ python scripts/regenerate_forecasts.py

# 6 Spin‑up full stack (Airflow + API)
$ docker-compose up --build
```

### Airflow

* Web UI: [http://localhost:8080](http://localhost:8080)  (default creds in `docker-compose.yml`)
* DAG: **daily\_forecast\_pipeline** → preprocess → forecast → …

### FastAPI service

```
GET /health # → {"status":"ok"}
GET /forecast/{store_id} # e.g., /forecast/3
# → Returns a JSON array of all weekly forecasts for the given store.
```

Running standalone (without Docker):

```bash
uvicorn src.api:app --reload --port 8000
```

### Plotly Dash (optional)

```bash
python app.py   # localhost:8050
```

---

## 🧰 Development & Tests

```bash
# lint & tests
black src scripts airflow/dags
pytest -q
```

Create a feature branch ➜ PR to `main`; CI (GitHub Actions) runs style‑check & unit tests.

---

## 📜 Data Folder Policy

`data/raw/` & `data/processed/` stay **out of Git**. Add your CSVs locally or mount via Docker volume. See `data/README.md` for column documentation.

---

## 🔧 Configuration

All tweakables (paths, model params) live in **`src/config.py`**. Environment variables override sensitive fields (DB URLs, secrets). Example:

```bash
export FASTAPI_SECRET="super‑secret‑token"
export AIRFLOW__CORE__FERNET_KEY="..."
```

---

## 🪄 Makefiles / Convenience scripts

```bash
make preprocess   # runs src/preprocessing.py
make forecast     # regenerates Prophet forecasts
make airflow-up   # docker-compose up airflow stack
make api          # runs FastAPI locally for dev
```

*(Requires GNU Make on macOS/Linux; optional on Windows.)*

---
