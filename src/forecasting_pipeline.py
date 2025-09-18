import os
import argparse
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import xgboost as xgb
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam

# --- Utility Functions ---

def create_train_val_split(store_df, test_size=0.2):
    """Splits store data into training and validation sets."""
    train, val = train_test_split(store_df, test_size=test_size, shuffle=False)
    return train, val

def get_features_and_target(df):
    """Separates features and target variable from a dataframe."""
    features = [col for col in df.columns if col not in ['Date', 'Weekly_Sales', 'Store', 'Type']]
    target = 'Weekly_Sales'
    return features, target

# --- Prophet Model ---

def tune_prophet(train_df):
    def objective(trial):
        params = {
            'changepoint_prior_scale': trial.suggest_float('changepoint_prior_scale', 0.001, 0.5, log=True),
            'seasonality_prior_scale': trial.suggest_float('seasonality_prior_scale', 0.01, 10, log=True),
            'holidays_prior_scale': trial.suggest_float('holidays_prior_scale', 0.01, 10, log=True),
        }
        model = Prophet(**params)
        model.fit(train_df)

        # Dynamic CV settings based on data size
        data_duration_days = (train_df['ds'].max() - train_df['ds'].min()).days
        horizon_days = min(365, data_duration_days // 4)
        initial_days = min(730, data_duration_days // 2)
        period_days = min(180, data_duration_days // 8)

        if initial_days < horizon_days or not horizon_days or not period_days:
            # Skip CV if not enough data
            return 99999

        df_cv = cross_validation(model, initial=f'{initial_days} days', period=f'{period_days} days', horizon=f'{horizon_days} days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        return df_p['rmse'].values[0]

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    return study.best_params, study.best_value

def prophet_forecast(store_df, best_params):
    prophet_df = store_df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'})
    model = Prophet(**best_params).fit(prophet_df)
    future = model.make_future_dataframe(periods=52, freq='W')
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]

# --- XGBoost Model ---

def tune_xgboost(train_df, val_df):
    features, target = get_features_and_target(train_df)
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': trial.suggest_float('eta', 1e-3, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }

        bst = xgb.train(param, dtrain, evals=[(dval, 'validation')], early_stopping_rounds=50, verbose_eval=False)
        preds = bst.predict(dval)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        return rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)

    # Returning eta as learning_rate for compatibility with XGBRegressor
    best_params = study.best_params
    best_params['learning_rate'] = best_params.pop('eta')

    return best_params, study.best_value

def xgboost_forecast(store_df, best_params):
    features, target = get_features_and_target(store_df)
    X_train, y_train = store_df[features], store_df[target]

    model = xgb.XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)

    future_dates = pd.to_datetime(store_df['Date']).max() + pd.to_timedelta(np.arange(1, 53), 'W')
    future_df = pd.DataFrame({'Date': future_dates})

    last_known_features = store_df.iloc[-1]
    for col in features:
        future_df[col] = last_known_features[col]

    predictions = []
    for i in range(52):
        current_features = future_df[features].iloc[[i]]
        pred = model.predict(current_features)[0]
        predictions.append(pred)
        if i + 1 < 52:
            future_df.loc[future_df.index[i + 1], 'Lag_1'] = pred
            sales_history = list(y_train.values[-3:]) + predictions
            future_df.loc[future_df.index[i + 1], 'Rolling_Mean_4'] = np.mean(sales_history[-4:])
            future_df.loc[future_df.index[i + 1], 'Rolling_Std_4'] = np.std(sales_history[-4:])

    future_df['yhat'] = predictions
    return future_df.rename(columns={'Date': 'ds'})[['ds', 'yhat']]

# --- LSTM Model ---

def create_lstm_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), 0])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

def tune_lstm(train_df, val_df):
    seq_length = min(8, len(val_df) - 1)
    if seq_length < 1:
        return {}, 99999 # Not enough data

    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[['Weekly_Sales']])
    val_scaled = scaler.transform(val_df[['Weekly_Sales']])

    X_train, y_train = create_lstm_sequences(train_scaled, seq_length)
    X_val, y_val = create_lstm_sequences(val_scaled, seq_length)

    if X_train.size == 0 or X_val.size == 0:
        return {}, 99999 # Not enough data to create sequences

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))

    def objective(trial):
        model = Sequential()
        model.add(LSTM(units=trial.suggest_int('units', 32, 128),
                       input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(trial.suggest_float('dropout', 0.1, 0.5)))
        model.add(Dense(1))

        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')

        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=50, batch_size=32, verbose=0,
                  callbacks=[optuna.integration.KerasPruningCallback(trial, 'val_loss')])

        val_loss = model.evaluate(X_val, y_val, verbose=0)
        return val_loss

    study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    best_params['seq_length'] = seq_length
    return best_params, np.sqrt(study.best_value)


def lstm_forecast(store_df, best_params):
    seq_length = best_params.get('seq_length', 8)
    if len(store_df) <= seq_length:
        return pd.DataFrame(columns=['ds', 'yhat'])

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_sales = scaler.fit_transform(store_df[['Weekly_Sales']])

    X, y = create_lstm_sequences(scaled_sales, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=best_params['units'], input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(best_params['dropout']))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
    model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    last_sequence = scaled_sales[-seq_length:]
    predictions_scaled = []
    for _ in range(52):
        input_seq = last_sequence.reshape((1, seq_length, 1))
        next_pred = model.predict(input_seq, verbose=0)[0][0]
        predictions_scaled.append(next_pred)
        last_sequence = np.append(last_sequence[1:], [[next_pred]], axis=0)

    predicted_sales = scaler.inverse_transform(np.array(predictions_scaled).reshape(-1, 1)).flatten()

    future_dates = pd.to_datetime(store_df['Date']).max() + pd.to_timedelta(np.arange(1, 53), 'W')
    return pd.DataFrame({'ds': future_dates, 'yhat': predicted_sales})

# --- Main Execution ---

def run_forecasting(model_name):
    df = pd.read_csv('/app/data/processed/merged_data.csv', parse_dates=['Date'])

    all_forecasts = []

    for store in df['Store'].unique():
        print(f"Processing Store {store} with {model_name}...")
        store_df = df[df['Store'] == store].copy()
        train_df, val_df = create_train_val_split(store_df)

        best_params, rmse = {}, np.nan

        if model_name == 'prophet':
            best_params, rmse = tune_prophet(train_df.rename(columns={'Date': 'ds', 'Weekly_Sales': 'y'}))
            forecast = prophet_forecast(store_df, best_params)
        elif model_name == 'xgboost':
            best_params, rmse = tune_xgboost(train_df, val_df)
            forecast = xgboost_forecast(store_df, best_params)
        elif model_name == 'lstm':
            best_params, rmse = tune_lstm(train_df, val_df)
            forecast = lstm_forecast(store_df, best_params)
        else:
            raise ValueError("Unsupported model.")

        if not forecast.empty:
            forecast['Store'] = store
            forecast['model'] = model_name
            forecast['rmse'] = rmse
            all_forecasts.append(forecast)

    if all_forecasts:
        forecast_df = pd.concat(all_forecasts)
        output_path = f'/app/data/processed/forecast_results_{model_name}.csv'
        os.makedirs('/app/data/processed', exist_ok=True)
        forecast_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Model: 'prophet', 'xgboost', or 'lstm'")
    args = parser.parse_args()
    run_forecasting(args.model)
