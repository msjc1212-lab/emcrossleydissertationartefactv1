import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib

from arch import arch_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "models/deployment"
RF_MODEL_PATH = MODEL_ROOT / "random_forest"
LSTM_MODEL_PATH = MODEL_ROOT / "lstm"
SCALER_PATH = MODEL_ROOT / "scalers"
GARCH_MODEL_PATH = MODEL_ROOT / "garch"
METADATA_PATH = MODEL_ROOT / "metadata"
DEPLOYMENT_OUTPUT_PATH = PROJECT_ROOT / "outputs/deployment"
LIVE_DATA_PATH = PROJECT_ROOT / "data/live"

MANIFEST_FILE = METADATA_PATH / "deployment_manifest.csv"
MANIFEST_OUTPUT_FILE = DEPLOYMENT_OUTPUT_PATH / "deployment_manifest.csv"
GARCH_PARAMETER_FILE = GARCH_MODEL_PATH / "garch_deployment_parameters.csv"
CHECKS_FILE = DEPLOYMENT_OUTPUT_PATH / "deployment_model_checks.csv"
REFIT_HISTORY_FILE = DEPLOYMENT_OUTPUT_PATH / "refit_history.csv"

RANDOM_STATE = 42
EPSILON = 1e-8
REFIT_INTERVAL = 63
MARKET_TIMEZONE = ZoneInfo("America/New_York")

RF_LAGS = [1, 2, 3, 5, 10, 21]
RF_ROLLING_WINDOWS = [5, 10, 21, 63]
RF_FEATURES = [
    "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag5", "Return_Lag10", "Return_Lag21",
    "Squared_Return_Lag1", "Squared_Return_Lag2", "Squared_Return_Lag3", "Squared_Return_Lag5",
    "Squared_Return_Lag10", "Squared_Return_Lag21", "Rolling_Vol_5", "Rolling_Vol_10",
    "Rolling_Vol_21", "Rolling_Vol_63"
]
LSTM_FEATURES = ["Log_Return", "Squared_Return"]


def safe_ticker_name(ticker):
    return str(ticker).replace("^", "")


def parse_depth(value):
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in ["none", "nan"]:
        return None
    return int(float(value))


def atomic_csv_write(dataframe, destination):
    destination = Path(destination)
    temp = destination.with_name(destination.stem + "_tmp" + destination.suffix)
    dataframe.to_csv(temp, index=False)
    os.replace(temp, destination)


def live_history_file(ticker):
    return LIVE_DATA_PATH / f"{safe_ticker_name(ticker)}_live_processed.csv"


def load_live_history(ticker):
    path = live_history_file(ticker)
    if not path.exists():
        raise FileNotFoundError(f"No live history exists for {ticker}: {path}. Refresh the asset first.")
    data = pd.read_csv(path)
    required = ["Date", "Adj_Close", "Log_Return", "Squared_Return", "Ticker", "Asset", "Role"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"{ticker}: live history is missing columns: {missing}")
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    if len(data) < 64:
        raise ValueError(f"{ticker}: at least 64 live return observations are required for deployment refitting.")
    return data


def load_manifest():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Missing deployment manifest: {MANIFEST_FILE}")
    manifest = pd.read_csv(MANIFEST_FILE)
    for c in ["GARCH_Training_Through", "RF_Target_Through", "LSTM_Target_Through"]:
        if c in manifest.columns:
            manifest[c] = pd.to_datetime(manifest[c])
    return manifest


def get_manifest_row(ticker, manifest=None):
    if manifest is None:
        manifest = load_manifest()
    match = manifest[manifest["Ticker"] == ticker]
    if len(match) != 1:
        raise ValueError(f"{ticker}: expected exactly one deployment manifest row.")
    return match.iloc[0]


def get_refit_status(ticker):
    manifest = load_manifest()
    row = get_manifest_row(ticker, manifest)
    training_through = pd.to_datetime(row["GARCH_Training_Through"])

    history_path = live_history_file(ticker)
    live_history_available = history_path.exists()

    if live_history_available:
        history = load_live_history(ticker)
        latest_date = history["Date"].max()
        new_observations = int((history["Date"] > training_through).sum())
    else:
        latest_date = training_through
        new_observations = 0

    remaining = max(REFIT_INTERVAL - new_observations, 0)
    due = new_observations >= REFIT_INTERVAL
    last_refit_time = None
    last_trigger = None
    if REFIT_HISTORY_FILE.exists():
        log = pd.read_csv(REFIT_HISTORY_FILE)
        if "Ticker" in log.columns:
            ticker_log = log[log["Ticker"] == ticker].copy()
            if len(ticker_log) > 0:
                ticker_log["Refit_Timestamp"] = pd.to_datetime(ticker_log["Refit_Timestamp"])
                latest_log = ticker_log.sort_values("Refit_Timestamp").iloc[-1]
                last_refit_time = latest_log["Refit_Timestamp"]
                last_trigger = latest_log["Trigger"]
    return {
        "Ticker": ticker,
        "Training_Through": training_through,
        "Latest_Data_Date": latest_date,
        "New_Observations": new_observations,
        "Refit_Interval": REFIT_INTERVAL,
        "Remaining": remaining,
        "Due": due,
        "Last_Refit_Time": last_refit_time,
        "Last_Trigger": last_trigger,
        "Live_History_Available": live_history_available,
    }


def build_supervised_data(history):
    data = history.copy().sort_values("Date").reset_index(drop=True)
    data["Target_Date"] = data["Date"].shift(-1)
    data["Target_Squared_Return"] = data["Squared_Return"].shift(-1)
    return data


def build_rf_training_data(history):
    data = build_supervised_data(history)
    for lag in RF_LAGS:
        shift_amount = lag - 1
        data[f"Return_Lag{lag}"] = data["Log_Return"].shift(shift_amount)
        data[f"Squared_Return_Lag{lag}"] = data["Squared_Return"].shift(shift_amount)
    for window in RF_ROLLING_WINDOWS:
        data[f"Rolling_Vol_{window}"] = data["Log_Return"].rolling(window=window, min_periods=window).std(ddof=1)
    data = data.dropna(subset=RF_FEATURES + ["Target_Date", "Target_Squared_Return"]).reset_index(drop=True)
    if data[RF_FEATURES].isna().any().any():
        raise ValueError("Missing RF deployment features after live-data reconstruction.")
    if (data["Target_Squared_Return"] < 0).any():
        raise ValueError("Negative RF variance target.")
    return data


def build_lstm_training_data(history):
    data = build_supervised_data(history)
    data = data.dropna(subset=["Target_Date", "Target_Squared_Return", "Log_Return", "Squared_Return"]).reset_index(drop=True)
    if data[LSTM_FEATURES].isna().any().any():
        raise ValueError("Missing LSTM deployment source data.")
    if (data["Target_Squared_Return"] < 0).any():
        raise ValueError("Negative LSTM variance target.")
    return data


def build_lstm_model(lookback, number_features, units, dropout, learning_rate):
    model = keras.Sequential([
        keras.Input(shape=(lookback, number_features)),
        layers.LSTM(units=units, dropout=dropout, recurrent_dropout=0.0),
        layers.Dense(1, activation="softplus"),
    ])
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="mse")
    return model


def create_lstm_sequences(asset_df, scaled_features, lookback):
    X, y = [], []
    targets = asset_df["Target_Squared_Return"].to_numpy(dtype=np.float32)
    for i in range(lookback - 1, len(asset_df)):
        start = i - lookback + 1
        end = i + 1
        X.append(scaled_features[start:end])
        y.append([targets[i]])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def append_refit_history(row):
    new_row = pd.DataFrame([row])
    if REFIT_HISTORY_FILE.exists():
        existing = pd.read_csv(REFIT_HISTORY_FILE)
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    atomic_csv_write(combined, REFIT_HISTORY_FILE)


def refit_selected_asset(ticker, trigger="Forced", lstm_callbacks=None):
    if trigger not in ["Forced", "Scheduled"]:
        raise ValueError("trigger must be 'Forced' or 'Scheduled'.")

    print("\n" + "=" * 100)
    print(f"DEPLOYMENT REFIT: {ticker} ({trigger})")
    print("=" * 100)

    manifest = load_manifest()
    manifest_row = get_manifest_row(ticker, manifest)
    history = load_live_history(ticker)
    latest_date = history["Date"].max()
    current_training_through = pd.to_datetime(manifest_row["GARCH_Training_Through"])
    new_observations = int((history["Date"] > current_training_through).sum())

    if trigger == "Scheduled" and new_observations < REFIT_INTERVAL:
        raise ValueError(
            f"{ticker}: scheduled refit is not due yet. "
            f"{new_observations}/{REFIT_INTERVAL} new observations are available."
        )
    if latest_date <= current_training_through:
        raise ValueError(f"{ticker}: there are no new observations beyond the current deployment training date.")

    asset_name = str(manifest_row["Asset"])
    role = str(manifest_row["Role"])
    safe_ticker = safe_ticker_name(ticker)

    ## GARCH
    print("\n[1/3] Refitting fixed GARCH specification...")
    garch_returns = history.set_index("Date")["Log_Return"].astype(float)
    if garch_returns.isna().any():
        raise ValueError(f"{ticker}: missing GARCH return.")
    garch_p = int(manifest_row["GARCH_p"])
    garch_q = int(manifest_row["GARCH_q"])
    garch_dist = str(manifest_row["GARCH_Distribution"])
    garch_model_name = str(manifest_row["GARCH_Selected_Model"])
    garch_model = arch_model(garch_returns, mean="Constant", vol="GARCH", p=garch_p, o=0, q=garch_q, dist=garch_dist, rescale=False)
    garch_result = garch_model.fit(disp="off", update_freq=0, show_warning=False)
    if garch_result.convergence_flag != 0:
        raise ValueError(f"{ticker}: deployment GARCH failed to converge.")

    persistence = 0.0
    new_garch_parameter_rows = []
    for parameter_order, (parameter_name, parameter_value) in enumerate(garch_result.params.items()):
        new_garch_parameter_rows.append({
            "Ticker": ticker, "Asset": asset_name, "Role": role, "Selected_Model": garch_model_name,
            "p": garch_p, "q": garch_q, "Distribution": garch_dist,
            "Parameter_Order": parameter_order, "Parameter": parameter_name,
            "Estimate": float(parameter_value), "Training_Through": latest_date,
        })
        if parameter_name.startswith("alpha[") or parameter_name.startswith("beta["):
            persistence += float(parameter_value)

    garch_forecast_object = garch_result.forecast(horizon=1, method="analytic", reindex=False)
    garch_smoke_forecast = float(garch_forecast_object.variance["h.1"].iloc[-1])
    if not np.isfinite(garch_smoke_forecast) or garch_smoke_forecast <= 0:
        raise ValueError(f"{ticker}: invalid refitted GARCH smoke forecast.")

    ## RF
    print("\n[2/3] Refitting fixed Random Forest specification...")
    rf_asset = build_rf_training_data(history)
    X_rf = rf_asset[RF_FEATURES].to_numpy(dtype=float)
    y_rf = rf_asset["Target_Squared_Return"].to_numpy(dtype=float)
    rf_n_estimators = int(manifest_row["RF_N_Estimators"])
    rf_depth = parse_depth(manifest_row["RF_Max_Depth"])
    rf_leaf = int(manifest_row["RF_Min_Samples_Leaf"])
    rf_max_features = float(manifest_row["RF_Max_Features"])
    rf_model = RandomForestRegressor(
        n_estimators=rf_n_estimators, criterion="squared_error", max_depth=rf_depth,
        min_samples_leaf=rf_leaf, max_features=rf_max_features, bootstrap=True,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf_model.fit(X_rf, y_rf)
    rf_probe = X_rf[-5:]
    rf_probe_prediction = rf_model.predict(rf_probe)
    if not np.isfinite(rf_probe_prediction).all():
        raise ValueError(f"{ticker}: non-finite RF smoke prediction.")

    ## LSTM
    print("\n[3/3] Refitting fixed LSTM specification...")
    lstm_asset = build_lstm_training_data(history)
    lstm_lookback = int(manifest_row["LSTM_Lookback"])
    lstm_units = int(manifest_row["LSTM_Units"])
    lstm_dropout = float(manifest_row["LSTM_Dropout"])
    lstm_learning_rate = float(manifest_row["LSTM_Learning_Rate"])
    lstm_batch_size = int(manifest_row["LSTM_Batch_Size"])
    lstm_epochs = int(manifest_row["LSTM_Epochs"])
    lstm_scaler = RobustScaler()
    lstm_scaler.fit(lstm_asset[LSTM_FEATURES])
    scaled_lstm_features = lstm_scaler.transform(lstm_asset[LSTM_FEATURES]).astype(np.float32)
    X_lstm, y_lstm = create_lstm_sequences(lstm_asset, scaled_lstm_features, lstm_lookback)
    if len(X_lstm) == 0:
        raise ValueError(f"{ticker}: no LSTM sequences were generated.")
    keras.backend.clear_session()
    keras.utils.set_random_seed(RANDOM_STATE)
    lstm_model = build_lstm_model(
        lookback=lstm_lookback, number_features=len(LSTM_FEATURES), units=lstm_units,
        dropout=lstm_dropout, learning_rate=lstm_learning_rate,
    )
    lstm_model.fit(X_lstm, y_lstm, epochs=lstm_epochs, batch_size=lstm_batch_size, shuffle=False, verbose=0, callbacks=lstm_callbacks)
    lstm_probe = X_lstm[-5:]
    original_lstm_prediction = lstm_model(lstm_probe, training=False).numpy().reshape(-1)
    if not np.isfinite(original_lstm_prediction).all() or (original_lstm_prediction <= 0).any():
        raise ValueError(f"{ticker}: invalid LSTM smoke prediction.")

    ## Save to temp and verify
    rf_file = RF_MODEL_PATH / f"{safe_ticker}_rf.joblib"
    lstm_file = LSTM_MODEL_PATH / f"{safe_ticker}_lstm.keras"
    scaler_file = SCALER_PATH / f"{safe_ticker}_lstm_scaler.joblib"
    rf_temp = RF_MODEL_PATH / f"{safe_ticker}_rf_tmp.joblib"
    lstm_temp = LSTM_MODEL_PATH / f"{safe_ticker}_lstm_tmp.keras"
    scaler_temp = SCALER_PATH / f"{safe_ticker}_lstm_scaler_tmp.joblib"

    for temp_path in [rf_temp, lstm_temp, scaler_temp]:
        if temp_path.exists():
            temp_path.unlink()

    joblib.dump(rf_model, rf_temp, compress=3)
    lstm_model.save(lstm_temp)
    joblib.dump(lstm_scaler, scaler_temp, compress=3)

    loaded_rf = joblib.load(rf_temp)
    loaded_rf_prediction = loaded_rf.predict(rf_probe)
    rf_save_load_match = bool(np.allclose(rf_probe_prediction, loaded_rf_prediction, rtol=1e-12, atol=1e-12))
    if not rf_save_load_match:
        raise ValueError(f"{ticker}: refitted RF save/load check failed.")

    loaded_lstm = keras.models.load_model(lstm_temp, compile=False)
    loaded_scaler = joblib.load(scaler_temp)
    original_scaled_probe = lstm_scaler.transform(lstm_asset[LSTM_FEATURES].tail(lstm_lookback))
    loaded_scaled_probe = loaded_scaler.transform(lstm_asset[LSTM_FEATURES].tail(lstm_lookback))
    scaler_save_load_match = bool(np.allclose(original_scaled_probe, loaded_scaled_probe, rtol=1e-12, atol=1e-12))
    if not scaler_save_load_match:
        raise ValueError(f"{ticker}: refitted LSTM scaler save/load check failed.")
    loaded_lstm_prediction = loaded_lstm(lstm_probe, training=False).numpy().reshape(-1)
    lstm_save_load_match = bool(np.allclose(original_lstm_prediction, loaded_lstm_prediction, rtol=1e-5, atol=1e-6))
    if not lstm_save_load_match:
        raise ValueError(f"{ticker}: refitted LSTM save/load check failed.")

    ## Replace deployment files
    os.replace(rf_temp, rf_file)
    os.replace(lstm_temp, lstm_file)
    os.replace(scaler_temp, scaler_file)

    ## Update GARCH parameter table
    if GARCH_PARAMETER_FILE.exists():
        existing_garch = pd.read_csv(GARCH_PARAMETER_FILE)
        existing_garch = existing_garch[existing_garch["Ticker"] != ticker].copy()
    else:
        existing_garch = pd.DataFrame()
    new_garch_df = pd.DataFrame(new_garch_parameter_rows)
    updated_garch = pd.concat([existing_garch, new_garch_df], ignore_index=True)
    updated_garch = updated_garch.sort_values(["Ticker", "Parameter_Order"]).reset_index(drop=True)
    atomic_csv_write(updated_garch, GARCH_PARAMETER_FILE)

    ## Update manifest
    rf_target_through = rf_asset["Target_Date"].max()
    lstm_target_through = lstm_asset["Target_Date"].max()
    ticker_mask = manifest["Ticker"] == ticker
    updates = {
        "GARCH_Training_Through": latest_date,
        "GARCH_Observations": len(garch_returns),
        "GARCH_Persistence": persistence,
        "GARCH_Convergence_Flag": garch_result.convergence_flag,
        "GARCH_Smoke_Forecast": garch_smoke_forecast,
        "RF_Training_Rows": len(rf_asset),
        "RF_Target_Through": rf_target_through,
        "RF_Model_Path": str(rf_file.relative_to(PROJECT_ROOT)),
        "LSTM_Training_Sequences": len(X_lstm),
        "LSTM_Target_Through": lstm_target_through,
        "LSTM_Model_Path": str(lstm_file.relative_to(PROJECT_ROOT)),
        "LSTM_Scaler_Path": str(scaler_file.relative_to(PROJECT_ROOT)),
    }
    for column, value in updates.items():
        manifest.loc[ticker_mask, column] = value
    manifest = manifest.sort_values("Ticker").reset_index(drop=True)
    atomic_csv_write(manifest, MANIFEST_FILE)
    atomic_csv_write(manifest, MANIFEST_OUTPUT_FILE)

    ## Update checks
    new_check_row = pd.DataFrame([{
        "Ticker": ticker,
        "Asset": asset_name,
        "GARCH_Converged": True,
        "GARCH_Forecast_Positive": True,
        "RF_Save_Load_Match": rf_save_load_match,
        "LSTM_Save_Load_Match": lstm_save_load_match,
        "LSTM_Scaler_Save_Load_Match": scaler_save_load_match,
    }])
    if CHECKS_FILE.exists():
        checks = pd.read_csv(CHECKS_FILE)
        checks = checks[checks["Ticker"] != ticker].copy()
        checks = pd.concat([checks, new_check_row], ignore_index=True)
    else:
        checks = new_check_row
    checks = checks.sort_values("Ticker").reset_index(drop=True)
    atomic_csv_write(checks, CHECKS_FILE)

    ## Log refit
    refit_timestamp = datetime.now(MARKET_TIMEZONE)
    append_refit_history({
        "Refit_Timestamp": refit_timestamp.isoformat(),
        "Ticker": ticker,
        "Asset": asset_name,
        "Trigger": trigger,
        "Previous_Training_Through": current_training_through,
        "New_Training_Through": latest_date,
        "New_Observations_Used": new_observations,
        "Refit_Interval": REFIT_INTERVAL,
        "GARCH_Selected_Model": garch_model_name,
        "GARCH_p": garch_p,
        "GARCH_q": garch_q,
        "GARCH_Distribution": garch_dist,
        "RF_N_Estimators": rf_n_estimators,
        "RF_Max_Depth": "None" if rf_depth is None else rf_depth,
        "RF_Min_Samples_Leaf": rf_leaf,
        "RF_Max_Features": rf_max_features,
        "LSTM_Lookback": lstm_lookback,
        "LSTM_Units": lstm_units,
        "LSTM_Epochs": lstm_epochs,
        "GARCH_Success": True,
        "RF_Success": True,
        "LSTM_Success": True,
        "Scaler_Success": True,
    })

    print("\n" + "=" * 100)
    print(f"{ticker}: DEPLOYMENT REFIT COMPLETE")
    print("=" * 100)
    print("Training through:", latest_date.date())
    print("Trigger:", trigger)

    keras.backend.clear_session()

    return {
        "Ticker": ticker,
        "Asset": asset_name,
        "Trigger": trigger,
        "Previous_Training_Through": current_training_through,
        "New_Training_Through": latest_date,
        "New_Observations_Used": new_observations,
        "GARCH_Smoke_Forecast": garch_smoke_forecast,
        "RF_Save_Load_Match": rf_save_load_match,
        "LSTM_Save_Load_Match": lstm_save_load_match,
        "LSTM_Scaler_Save_Load_Match": scaler_save_load_match,
    }
