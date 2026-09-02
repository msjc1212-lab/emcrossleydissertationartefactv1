from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import joblib
from tensorflow import keras

st.set_page_config(
    page_title="Volatility Forecasting System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make package imports available when Streamlit launches this nested script.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
LIVE_OUTPUT_PATH = PROJECT_ROOT / "outputs/live"
COMPARISON_PATH = PROJECT_ROOT / "outputs/comparison"
DEPLOYMENT_PATH = PROJECT_ROOT / "outputs/deployment"
DEPLOYMENT_METADATA_PATH = PROJECT_ROOT / "models/deployment/metadata"
LIVE_DATA_PATH = PROJECT_ROOT / "data/live"

LIVE_FORECAST_FILE = LIVE_OUTPUT_PATH / "latest_live_forecasts.csv"
LIVE_QUALITY_FILE = LIVE_OUTPUT_PATH / "live_data_quality.csv"
FORECAST_HISTORY_FILE = LIVE_OUTPUT_PATH / "live_forecast_history.csv"
STOCK_AVERAGE_FILE = COMPARISON_PATH / "stock_average_model_comparison.csv"
QLIKE_COMPARISON_FILE = COMPARISON_PATH / "qlike_comparison_by_asset.csv"
ALL_METRICS_FILE = COMPARISON_PATH / "all_model_test_metrics.csv"
TEST_FORECAST_FILE = COMPARISON_PATH / "combined_test_forecasts_all_models.csv"
DM_QLIKE_FILE = COMPARISON_PATH / "diebold_mariano_qlike.csv"
MANIFEST_FILE = DEPLOYMENT_METADATA_PATH / "deployment_manifest.csv"
REFIT_HISTORY_FILE = DEPLOYMENT_PATH / "refit_history.csv"


@st.cache_resource(show_spinner=False)
def load_cached_lstm_model(model_path_string, version_token):
    print("STREAMLIT CACHE: loading LSTM model...")
    model = keras.models.load_model(model_path_string, compile=False)
    print("STREAMLIT CACHE: LSTM model loaded.")
    return model


@st.cache_resource(show_spinner=False)
def load_cached_lstm_scaler(scaler_path_string, version_token):
    print("STREAMLIT CACHE: loading LSTM scaler...")
    scaler = joblib.load(scaler_path_string)
    print("STREAMLIT CACHE: LSTM scaler loaded.")
    return scaler


def read_csv(path, parse_dates=None):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found:\n{path}")
    return pd.read_csv(path, parse_dates=parse_dates)


def safe_ticker(ticker):
    return str(ticker).replace("^", "")


def format_price(value):
    if pd.isna(value):
        return "N/A"
    return f"${value:,.2f}"


def file_version(path):
    path = Path(path)
    if not path.exists():
        return 0
    return path.stat().st_mtime_ns


def resolved_forecast_history(ticker, model_name):
    live_history_file = LIVE_DATA_PATH / f"{safe_ticker(ticker)}_live_processed.csv"

    if not FORECAST_HISTORY_FILE.exists() or not live_history_file.exists():
        return pd.DataFrame()

    history = pd.read_csv(FORECAST_HISTORY_FILE)
    if len(history) == 0:
        return pd.DataFrame()

    history["Latest_Data_Date"] = pd.to_datetime(
        history["Latest_Data_Date"],
        errors="coerce"
    )
    history = history[
        (history["Ticker"] == ticker) &
        (history["Model"] == model_name)
    ].copy()

    market = pd.read_csv(live_history_file)
    market["Date"] = pd.to_datetime(market["Date"], errors="coerce")
    market = (
        market
        .dropna(subset=["Date", "Log_Return", "Squared_Return"])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    resolved_rows = []

    for _, forecast_row in history.sort_values("Latest_Data_Date").iterrows():
        origin_date = forecast_row["Latest_Data_Date"]
        future_market = market[market["Date"] > origin_date]

        if len(future_market) == 0:
            continue

        realised = future_market.iloc[0]
        forecast_variance = float(forecast_row["Forecast_Variance"])
        realised_squared_return = float(realised["Squared_Return"])

        resolved_rows.append({
            "Forecast_Origin_Date": origin_date,
            "Realised_Date": realised["Date"],
            "Model": model_name,
            "Forecast_Variance": forecast_variance,
            "Realised_Squared_Return": realised_squared_return,
            "Predicted_Daily_Volatility_Pct": float(
                forecast_row["Forecast_Daily_Volatility_Pct"]
            ),
            "Realised_Absolute_Return_Pct": abs(float(realised["Log_Return"])),
            "Live_QLIKE": (
                np.log(max(forecast_variance, 1e-8)) +
                realised_squared_return / max(forecast_variance, 1e-8)
            )
        })

    return pd.DataFrame(resolved_rows)


def clear_deployment_caches():
    load_cached_lstm_model.clear()
    load_cached_lstm_scaler.clear()

    try:
        from artefact import live_prediction_engine
        if hasattr(live_prediction_engine, "_RF_MODEL_CACHE"):
            live_prediction_engine._RF_MODEL_CACHE.clear()
    except Exception:
        pass


def run_refit_subprocess(ticker, trigger, asset_name):
    command = [
        sys.executable,
        "-m",
        "artefact.refit_cli",
        ticker,
        "--trigger",
        trigger,
    ]

    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    output_lines = []
    output_box = st.empty()

    if process.stdout is None:
        raise RuntimeError("Could not capture deployment refit output.")

    for line in process.stdout:
        clean_line = line.rstrip()
        output_lines.append(clean_line)
        output_box.code("\n".join(output_lines[-24:]), language="text")

    return_code = process.wait()

    if return_code != 0:
        tail = "\n".join(output_lines[-30:])
        raise RuntimeError(
            f"{asset_name} deployment refit exited with code {return_code}.\n\n{tail}"
        )

    return output_lines


st.title("Financial Volatility Forecasting System")
st.caption("Live one-step-ahead volatility forecasting using GARCH, Random Forest and LSTM models.")

if "refresh_message" in st.session_state:
    st.success(st.session_state.pop("refresh_message"))
if "refit_message" in st.session_state:
    st.success(st.session_state.pop("refit_message"))

try:
    live_forecasts = read_csv(
        LIVE_FORECAST_FILE,
        parse_dates=["Latest_Data_Date", "Deployment_Model_Training_Through"]
    )
    live_quality = read_csv(
        LIVE_QUALITY_FILE,
        parse_dates=["Latest_Data_Date", "Deployment_Training_Through"]
    )
    stock_average = read_csv(STOCK_AVERAGE_FILE)
    qlike_comparison = read_csv(QLIKE_COMPARISON_FILE)
    all_metrics = read_csv(ALL_METRICS_FILE)
    test_forecasts = read_csv(
        TEST_FORECAST_FILE,
        parse_dates=["Date", "Target_Date"]
    )
    dm_qlike = read_csv(DM_QLIKE_FILE)
    deployment_manifest = read_csv(
        MANIFEST_FILE,
        parse_dates=["GARCH_Training_Through", "RF_Target_Through", "LSTM_Target_Through"]
    )
except Exception as error:
    st.error("The artefact could not load its required output files.")
    st.exception(error)
    st.stop()

asset_lookup = (
    live_forecasts[["Ticker", "Asset", "Role"]]
    .drop_duplicates()
    .sort_values(["Role", "Ticker"])
    .reset_index(drop=True)
)
asset_labels = {
    row["Ticker"]: f"{row['Asset']} ({row['Ticker']})"
    for _, row in asset_lookup.iterrows()
}
ticker_list = asset_lookup["Ticker"].tolist()

st.sidebar.header("Controls")
selected_ticker = st.sidebar.selectbox(
    "Asset",
    options=ticker_list,
    format_func=lambda ticker: asset_labels[ticker]
)
model_selection = st.sidebar.selectbox(
    "Forecast model",
    options=[
        "Best research model (QLIKE)",
        "GARCH",
        "Random Forest",
        "LSTM",
        "Historical Variance"
    ]
)

st.sidebar.divider()
st.sidebar.subheader("Where to find things")
st.sidebar.markdown(
    """
**Live Predictor** — current next-session forecast  
**Asset Analysis** — asset-level research metrics and out-of-sample analysis  
**Forecast Check** — compare matured live forecasts with the next realised session  
**Refit History** — deployment maintenance and refit records  
**Methodology** — how the research and live system work  
**Research Results** — frozen out-of-sample model comparison
    """
)

selected_asset_rows = live_forecasts[live_forecasts["Ticker"] == selected_ticker].copy()
if len(selected_asset_rows) == 0:
    st.error("No live forecast is available for the selected asset.")
    st.stop()

selected_asset = selected_asset_rows["Asset"].iloc[0]
research_best_model = selected_asset_rows["Research_Best_Model"].iloc[0]
active_model = research_best_model if model_selection == "Best research model (QLIKE)" else model_selection
selected_model_rows = selected_asset_rows[selected_asset_rows["Model"] == active_model]
if len(selected_model_rows) == 0:
    st.error(f"No saved forecast is available for {active_model}.")
    st.stop()
selected_forecast = selected_model_rows.iloc[0]

manifest_match = deployment_manifest[deployment_manifest["Ticker"] == selected_ticker]
if len(manifest_match) != 1:
    st.error(f"Could not identify exactly one deployment configuration for {selected_ticker}.")
    st.stop()
manifest_row = manifest_match.iloc[0]
lstm_model_path = PROJECT_ROOT / Path(str(manifest_row["LSTM_Model_Path"]))
lstm_scaler_path = PROJECT_ROOT / Path(str(manifest_row["LSTM_Scaler_Path"]))

from artefact.deployment_refit import get_refit_status

try:
    refit_status = get_refit_status(selected_ticker)
except Exception as error:
    st.error("Could not calculate deployment refit status.")
    st.exception(error)
    st.stop()

refit_due = bool(refit_status["Due"])
live_history_available = bool(refit_status.get("Live_History_Available", True))
new_refit_observations = int(refit_status["New_Observations"])
refit_interval = int(refit_status["Refit_Interval"])
refit_remaining = int(refit_status["Remaining"])

st.sidebar.divider()
st.sidebar.subheader("Data & model maintenance")

refresh_selected = st.sidebar.button(
    f"↻ Refresh {selected_ticker}",
    type="primary",
    width="stretch",
    help=(
        "Download newly completed daily market observations and generate fresh forecasts "
        "using the currently fitted deployment models."
    )
)

with st.sidebar.expander("Deployment status", expanded=False):
    st.metric(
        "Trained through",
        str(pd.to_datetime(refit_status["Training_Through"]).date())
    )
    st.metric(
        "New labelled observations",
        f"{new_refit_observations} / {refit_interval}"
    )
    if refit_due:
        st.metric("Refit status", "DUE")
    else:
        st.metric(
            "Next scheduled refit",
            f"{refit_remaining} trading observations"
        )

    if not live_history_available:
        st.info(
            f"Refresh {selected_ticker} to initialize its live market history and refit counter."
        )
    elif refit_due:
        st.warning(
            f"{selected_ticker} is due for a scheduled deployment refit."
        )
    else:
        st.caption(
            f"Scheduled refit becomes due after {refit_interval} new trading observations."
        )

scheduled_refit = st.sidebar.button(
    f"Refit {selected_ticker}",
    width="stretch",
    disabled=not refit_due,
    help=(
        "Refit the already-selected GARCH, Random Forest and LSTM specifications "
        "once 63 new trading observations are available."
    )
)

force_refit = st.sidebar.button(
    f"Force refit {selected_ticker}",
    width="stretch",
    disabled=not live_history_available,
    help=(
        "Refit the already-selected deployment specifications now, even before the "
        "scheduled 63-observation threshold."
    )
)

if refresh_selected:
    try:
        if not lstm_model_path.exists():
            raise FileNotFoundError(f"LSTM model not found:\n{lstm_model_path}")
        if not lstm_scaler_path.exists():
            raise FileNotFoundError(f"LSTM scaler not found:\n{lstm_scaler_path}")

        model_version = file_version(lstm_model_path)
        scaler_version = file_version(lstm_scaler_path)

        with st.spinner(f"Preparing {selected_asset} LSTM model..."):
            cached_lstm_model = load_cached_lstm_model(str(lstm_model_path), model_version)
            cached_lstm_scaler = load_cached_lstm_scaler(str(lstm_scaler_path), scaler_version)

        with st.spinner(f"Updating {selected_asset} market data and forecasts..."):
            from artefact.live_prediction_engine import generate_single_live_forecast
            generate_single_live_forecast(
                ticker=selected_ticker,
                preloaded_lstm_model=cached_lstm_model,
                preloaded_lstm_scaler=cached_lstm_scaler
            )

        st.session_state["refresh_message"] = (
            f"{selected_asset} market data and forecasts updated successfully."
        )
        st.rerun()
    except Exception as error:
        st.error(f"{selected_asset} refresh failed.")
        st.exception(error)

if scheduled_refit:
    try:
        st.info(
            f"Refitting {selected_asset} in an isolated deployment process. "
            "Progress will appear below."
        )
        run_refit_subprocess(
            ticker=selected_ticker,
            trigger="Scheduled",
            asset_name=selected_asset,
        )
        clear_deployment_caches()
        updated_status = get_refit_status(selected_ticker)
        st.session_state["refit_message"] = (
            f"{selected_asset} deployment models were refitted successfully through "
            f"{pd.to_datetime(updated_status['Training_Through']).date()}."
        )
        st.rerun()
    except Exception as error:
        st.error(f"{selected_asset} scheduled refit failed.")
        st.exception(error)

if force_refit:
    st.session_state[f"confirm_force_{selected_ticker}"] = True
    st.rerun()

confirmation_key = f"confirm_force_{selected_ticker}"
if st.session_state.get(confirmation_key, False):
    st.warning(
        "Forced refitting keeps the dissertation's selected model specifications fixed but "
        "re-estimates deployment parameters/weights using all currently labelled live data. "
        "Frozen research results are not changed."
    )
    confirm_col1, confirm_col2 = st.columns(2)
    with confirm_col1:
        confirm_force = st.button("Confirm forced refit", type="primary", width="stretch")
    with confirm_col2:
        cancel_force = st.button("Cancel", width="stretch")

    if cancel_force:
        st.session_state[confirmation_key] = False
        st.rerun()

    if confirm_force:
        try:
            st.info(
                f"Force-refitting {selected_asset} in an isolated deployment process. "
                "Progress will appear below."
            )
            run_refit_subprocess(
                ticker=selected_ticker,
                trigger="Forced",
                asset_name=selected_asset,
            )
            clear_deployment_caches()
            updated_status = get_refit_status(selected_ticker)
            st.session_state[confirmation_key] = False
            st.session_state["refit_message"] = (
                f"{selected_asset} deployment models were force-refitted successfully through "
                f"{pd.to_datetime(updated_status['Training_Through']).date()}."
            )
            st.rerun()
        except Exception as error:
            st.error(f"{selected_asset} forced refit failed.")
            st.exception(error)

latest_data_date = pd.to_datetime(selected_forecast["Latest_Data_Date"]).date()
training_through = pd.to_datetime(refit_status["Training_Through"]).date()
st.caption(
    f"**{selected_asset} ({selected_ticker})** • Active model: **{active_model}** • "
    f"Market data through **{latest_data_date}** • Deployment trained through **{training_through}**"
)

predictor_tab, asset_tab, verification_tab, refit_tab, methodology_tab, results_tab = st.tabs([
    "Live Predictor",
    "Asset Analysis",
    "Forecast Check",
    "Refit History",
    "Methodology",
    "Research Results"
])

with predictor_tab:
    st.subheader(f"{selected_asset} — Next-Session Volatility Forecast")
    st.caption("Use this tab for the current forecast and a side-by-side comparison of all four models.")

    live_history_file = LIVE_DATA_PATH / f"{safe_ticker(selected_ticker)}_live_processed.csv"
    if live_history_file.exists():
        live_history = (
            pd.read_csv(live_history_file, parse_dates=["Date"])
            .sort_values("Date")
            .reset_index(drop=True)
        )

        st.subheader("Latest live market history")
        max_live_observations = len(live_history)

        if max_live_observations >= 20:
            live_observations_to_show = st.slider(
                "Number of most recent live observations to display",
                min_value=20,
                max_value=max_live_observations,
                value=min(120, max_live_observations),
                step=1
            )
            recent_live = live_history.tail(live_observations_to_show)
        else:
            recent_live = live_history.copy()

        live_price_chart = go.Figure()
        live_price_chart.add_trace(go.Scatter(
            x=recent_live["Date"],
            y=recent_live["Adj_Close"],
            mode="lines",
            name="Adjusted close"
        ))
        live_price_chart.update_layout(
            xaxis_title="Date",
            yaxis_title="Adjusted close ($)",
            hovermode="x unified"
        )
        st.plotly_chart(live_price_chart, width="stretch")
    else:
        st.info("Refresh the selected asset to load its latest live market history.")

    st.divider()
    st.write(f"Forecast generated using **{active_model}**.")
    if model_selection == "Best research model (QLIKE)":
        st.caption(
            f"{active_model} achieved the lowest frozen test-period QLIKE for {selected_asset}."
        )

    market_col1, market_col2, market_col3 = st.columns(3)
    with market_col1:
        st.metric("Latest adjusted close", format_price(float(selected_forecast["Latest_Adj_Close"])))
    with market_col2:
        latest_return = float(selected_forecast["Latest_Log_Return_Pct"])
        st.metric("Latest daily return", f"{latest_return:+.2f}%")
    with market_col3:
        st.metric("Latest completed observation", str(latest_data_date))

    st.divider()
    variance = float(selected_forecast["Forecast_Variance"])
    daily_volatility = float(selected_forecast["Forecast_Daily_Volatility_Pct"])
    annualised_volatility = float(selected_forecast["Forecast_Annualised_Volatility_Pct"])
    variance_col, daily_col, annual_col = st.columns(3)
    with variance_col:
        st.metric("Predicted variance", f"{variance:.3f}")
    with daily_col:
        st.metric("Predicted daily volatility", f"{daily_volatility:.2f}%")
    with annual_col:
        st.metric("Annualised volatility", f"{annualised_volatility:.1f}%")

    st.caption(
        "Daily volatility is the square root of the predicted conditional variance. "
        "Annualised volatility uses 252 trading days."
    )

    with st.expander("How to read this forecast"):
        st.markdown(
            """
- **Predicted variance** is the model's forecast of the next session's squared percentage return.
- **Daily volatility** is the square root of that variance, expressed as a percentage.
- **Annualised volatility** scales the daily forecast by √252 to give a familiar annualised measure.
- The system forecasts the **magnitude of volatility**, not whether the price will rise or fall.
            """
        )

    st.subheader("Current model comparison")
    live_chart_data = selected_asset_rows[["Model", "Forecast_Daily_Volatility_Pct"]].sort_values(
        "Forecast_Daily_Volatility_Pct"
    )
    live_fig = px.bar(
        live_chart_data,
        x="Model",
        y="Forecast_Daily_Volatility_Pct",
        text="Forecast_Daily_Volatility_Pct",
        labels={"Forecast_Daily_Volatility_Pct": "Predicted daily volatility (%)", "Model": "Model"}
    )
    live_fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    live_fig.update_layout(yaxis_title="Daily volatility (%)", xaxis_title="Model", showlegend=False)
    st.plotly_chart(live_fig, width="stretch")

    predictor_table = selected_asset_rows[[
        "Model", "Forecast_Variance", "Forecast_Daily_Volatility_Pct",
        "Forecast_Annualised_Volatility_Pct", "Research_Test_QLIKE", "Research_QLIKE_Winner"
    ]].copy()
    predictor_table = predictor_table.rename(columns={
        "Forecast_Variance": "Variance",
        "Forecast_Daily_Volatility_Pct": "Daily Volatility (%)",
        "Forecast_Annualised_Volatility_Pct": "Annualised Volatility (%)",
        "Research_Test_QLIKE": "Research Test QLIKE",
        "Research_QLIKE_Winner": "Best Research QLIKE"
    })
    st.dataframe(predictor_table, width="stretch", hide_index=True)

    quality_match = live_quality[live_quality["Ticker"] == selected_ticker]
    if len(quality_match) > 0:
        quality_row = quality_match.iloc[0]
        if quality_row["Research_Overlap_Status"] == "OK":
            st.success(
                "Live-data consistency check passed: fresh market history is consistent with the frozen research dataset."
            )
        else:
            st.warning(str(quality_row["Research_Overlap_Status"]))

with verification_tab:
    st.subheader(f"{selected_asset} — Previous Forecast Check")
    st.caption("Use this tab to see whether earlier live forecasts aligned with the next completed trading session.")
    st.write(
        "This view compares an earlier live forecast with what was observed in the next completed trading session."
    )

    resolved_history = resolved_forecast_history(
        selected_ticker,
        active_model
    )

    if len(resolved_history) == 0:
        st.info("No live forecasts have matured yet.")
    else:
        latest_check = resolved_history.sort_values("Realised_Date").iloc[-1]

        check_col1, check_col2, check_col3, check_col4 = st.columns(4)
        with check_col1:
            st.metric(
                "Forecast origin",
                str(pd.to_datetime(latest_check["Forecast_Origin_Date"]).date())
            )
        with check_col2:
            st.metric(
                "Realised session",
                str(pd.to_datetime(latest_check["Realised_Date"]).date())
            )
        with check_col3:
            st.metric(
                "Predicted daily volatility",
                f"{latest_check['Predicted_Daily_Volatility_Pct']:.2f}%"
            )
        with check_col4:
            st.metric(
                "Realised |return| proxy",
                f"{latest_check['Realised_Absolute_Return_Pct']:.2f}%"
            )

        variance_col1, variance_col2 = st.columns(2)
        with variance_col1:
            st.metric(
                "Forecast variance",
                f"{latest_check['Forecast_Variance']:.3f}"
            )
        with variance_col2:
            st.metric(
                "Realised squared return",
                f"{latest_check['Realised_Squared_Return']:.3f}"
            )

        recent_checks = resolved_history.sort_values("Realised_Date").tail(30).copy()
        comparison_chart = go.Figure()
        comparison_chart.add_trace(go.Scatter(
            x=recent_checks["Realised_Date"],
            y=recent_checks["Predicted_Daily_Volatility_Pct"],
            mode="lines+markers",
            name="Predicted daily volatility"
        ))
        comparison_chart.add_trace(go.Scatter(
            x=recent_checks["Realised_Date"],
            y=recent_checks["Realised_Absolute_Return_Pct"],
            mode="lines+markers",
            name="Realised |return| proxy"
        ))
        comparison_chart.update_layout(
            xaxis_title="Realised session (date)",
            yaxis_title="Daily volatility / absolute return (%)",
            hovermode="x unified"
        )
        st.plotly_chart(comparison_chart, width="stretch")

        st.caption(
            "True daily volatility is latent and cannot be observed directly. "
            "The absolute daily return is shown as an intuitive one-session proxy; "
            "the model itself is evaluated against squared return, consistent with the dissertation target."
        )

        with st.expander("Recent resolved forecasts"):
            check_table = recent_checks[[
                "Forecast_Origin_Date",
                "Realised_Date",
                "Forecast_Variance",
                "Realised_Squared_Return",
                "Predicted_Daily_Volatility_Pct",
                "Realised_Absolute_Return_Pct",
                "Live_QLIKE"
            ]].copy()
            check_table = check_table.rename(columns={
                "Forecast_Origin_Date": "Forecast Origin",
                "Realised_Date": "Realised Session",
                "Forecast_Variance": "Forecast Variance",
                "Realised_Squared_Return": "Realised Squared Return",
                "Predicted_Daily_Volatility_Pct": "Predicted Daily Volatility (%)",
                "Realised_Absolute_Return_Pct": "Realised |Return| Proxy (%)",
                "Live_QLIKE": "One-Step QLIKE"
            })
            st.dataframe(check_table, width="stretch", hide_index=True)


with results_tab:
    st.subheader("Frozen Out-of-Sample Research Results")
    st.caption("Use this tab for the final dissertation model comparison and statistical evaluation.")
    st.write(
        "These results come from the predefined 2024–2026 test period and remain separate from live deployment."
    )

    with st.expander("What do the evaluation metrics mean?"):
        st.markdown(
            """
- **QLIKE** is the primary variance-forecast loss used in this project. It compares forecast variance with realised squared return; **lower is better**.
- **RMSE** measures the typical size of forecast errors but gives extra weight to large misses; **lower is better**.
- **MAE** is the average absolute forecast error and is less dominated by extreme misses than RMSE; **lower is better**.
- **Diebold–Mariano (DM) test** compares the predictive loss of two models. A small Holm-adjusted p-value (typically **< 0.05**) provides evidence that their forecast performance differs statistically.
- **Squared return** is used as the observable proxy for the otherwise latent daily variance.
            """
        )

    best_rmse = stock_average.sort_values("Mean_RMSE").iloc[0]
    best_mae = stock_average.sort_values("Mean_MAE").iloc[0]
    best_qlike = stock_average.sort_values("Mean_QLIKE").iloc[0]
    result_col1, result_col2, result_col3 = st.columns(3)
    with result_col1:
        st.metric("Lowest mean RMSE", f"{best_rmse['Mean_RMSE']:.3f}")
        st.caption(str(best_rmse["Model"]))
    with result_col2:
        st.metric("Lowest mean MAE", f"{best_mae['Mean_MAE']:.3f}")
        st.caption(str(best_mae["Model"]))
    with result_col3:
        st.metric("Lowest mean QLIKE", f"{best_qlike['Mean_QLIKE']:.3f}")
        st.caption(str(best_qlike["Model"]))

    st.subheader("Six-stock average performance")
    st.dataframe(stock_average, width="stretch", hide_index=True)

    metric_choice = st.selectbox(
        "Compare average metric",
        options=["Mean_QLIKE", "Mean_RMSE", "Mean_MAE"],
        format_func=lambda value: value.replace("Mean_", "")
    )
    metric_axis_labels = {
        "Mean_QLIKE": "Mean QLIKE (loss score)",
        "Mean_RMSE": "Mean RMSE (squared percentage points)",
        "Mean_MAE": "Mean MAE (squared percentage points)",
    }
    average_chart_data = stock_average[["Model", metric_choice]].sort_values(metric_choice)
    average_fig = px.bar(average_chart_data, x="Model", y=metric_choice, text=metric_choice)
    average_fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    average_fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric_axis_labels[metric_choice],
        showlegend=False
    )
    st.plotly_chart(average_fig, width="stretch")

    st.subheader("QLIKE by asset")
    qlike_long = qlike_comparison.melt(
        id_vars=["Ticker", "Asset", "Role"],
        value_vars=["Historical Variance", "GARCH", "Random Forest", "LSTM"],
        var_name="Model",
        value_name="QLIKE"
    )
    qlike_fig = px.bar(
        qlike_long,
        x="Ticker",
        y="QLIKE",
        color="Model",
        barmode="group",
        hover_data=["Asset", "Role"],
        labels={"Ticker": "Asset (ticker)", "QLIKE": "QLIKE (loss score)"}
    )
    st.plotly_chart(qlike_fig, width="stretch")
    st.caption("Lower QLIKE indicates better variance-forecast performance.")

with asset_tab:
    st.subheader(f"{selected_asset} — Asset Analysis")
    st.caption("Use this tab for asset-level research metrics, out-of-sample forecast history and statistical comparison.")
    asset_metrics = all_metrics[all_metrics["Ticker"] == selected_ticker].copy().sort_values("QLIKE")
    st.dataframe(asset_metrics, width="stretch", hide_index=True)

    st.subheader("Out-of-sample forecast history")
    asset_test = test_forecasts[test_forecasts["Ticker"] == selected_ticker].sort_values("Target_Date").copy()
    max_observations = len(asset_test)
    if max_observations >= 60:
        observations_to_show = st.slider(
            "Number of most recent test observations to display",
            min_value=60,
            max_value=max_observations,
            value=min(180, max_observations),
            step=1
        )
        asset_test = asset_test.tail(observations_to_show)

    forecast_chart = go.Figure()
    if "Target_Squared_Return" in asset_test.columns:
        forecast_chart.add_trace(go.Scatter(
            x=asset_test["Target_Date"],
            y=asset_test["Target_Squared_Return"],
            mode="lines",
            name="Actual squared return"
        ))

    forecast_column_map = {
        "Historical Variance": "Historical_Variance_Forecast",
        "GARCH": "GARCH_Forecast",
        "Random Forest": "RF_Forecast",
        "LSTM": "LSTM_Forecast"
    }
    for model_name, column_name in forecast_column_map.items():
        if column_name in asset_test.columns:
            forecast_chart.add_trace(go.Scatter(
                x=asset_test["Target_Date"],
                y=asset_test[column_name],
                mode="lines",
                name=model_name
            ))

    forecast_chart.update_layout(
        xaxis_title="Target date",
        yaxis_title="Variance / squared return (squared percentage points)",
        hovermode="x unified"
    )
    st.plotly_chart(forecast_chart, width="stretch")
    st.caption(
        "Variance, squared returns, RMSE and MAE use squared percentage points: "
        "a 1% log return has a squared return of 1 on this scale."
    )

    st.subheader("Diebold–Mariano QLIKE tests")
    asset_dm = dm_qlike[dm_qlike["Ticker"] == selected_ticker].copy()
    st.dataframe(asset_dm, width="stretch", hide_index=True)



with methodology_tab:
    st.subheader("Forecasting Framework")
    st.caption("Use this tab to understand the research design, live deployment and refit process.")
    st.markdown(
        """
### Frozen research experiment

The empirical experiment uses a fixed chronological design:

- **Training:** 2010–2021
- **Validation:** 2022–2023
- **Test:** 2024–31 July 2026
- **Forecast horizon:** one trading session ahead
- **Target:** next-day squared percentage log return

The frozen research results are never overwritten by the deployment system.

### Live deployment

The live system retrieves newly completed daily market observations and generates current next-session volatility forecasts.

### Periodic deployment refitting

Deployment specifications remain fixed to those chosen during the original validation process. The application does **not** re-run model selection.

After every **63 new trading observations** (approximately one quarter), the selected deployment specifications become eligible for refitting using all currently available labelled observations. A forced refit is also available for demonstration or manual maintenance.

A deployment refit:

1. re-estimates the already-selected GARCH specification;
2. refits the already-selected Random Forest configuration;
3. refits the existing LSTM architecture using its selected lookback, units, dropout, learning rate, batch size and deployment epoch count;
4. refits the LSTM RobustScaler;
5. replaces deployment-only model files after persistence checks succeed;
6. updates the deployment training-through date;
7. records the event in the persistent refit history.

Deployment LSTM refitting runs in an isolated process and uses eager TensorFlow execution for runtime reliability. This changes the execution mechanism only; the selected deployment architecture and hyperparameters remain fixed.

None of these operations alter the dissertation's frozen test forecasts, model comparison results or Diebold–Mariano tests.
        """
    )

    st.subheader("Deployment specifications")
    st.dataframe(deployment_manifest, width="stretch", hide_index=True)
    st.warning(
        "The system forecasts the magnitude of market volatility, not the direction of future returns. "
        "Forecasts are for academic research and demonstration and should not be interpreted as investment advice."
    )

with refit_tab:
    st.subheader("Deployment Refit History")
    st.caption("Use this tab to review when deployment models were updated and why.")
    st.write(
        "This log records changes to deployment models only. Frozen research results remain unchanged."
    )
    if REFIT_HISTORY_FILE.exists():
        refit_history = pd.read_csv(REFIT_HISTORY_FILE)
        if "Refit_Timestamp" in refit_history.columns:
            refit_history["Refit_Timestamp"] = pd.to_datetime(refit_history["Refit_Timestamp"])
            refit_history = refit_history.sort_values("Refit_Timestamp", ascending=False)
        st.dataframe(refit_history, width="stretch", hide_index=True)
    else:
        st.info("No deployment refits have been recorded yet.")

st.divider()
st.caption("MSc data science dissertation artefact")
