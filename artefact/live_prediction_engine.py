import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
import time as time_module

from pathlib import Path
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

from arch import arch_model


## Project paths

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

MODEL_ROOT = (
    PROJECT_ROOT /
    "models/deployment"
)

METADATA_PATH = (
    MODEL_ROOT /
    "metadata"
)

GARCH_MODEL_PATH = (
    MODEL_ROOT /
    "garch"
)

PROCESSED_RESEARCH_FILE = (
    PROJECT_ROOT /
    "data/processed/"
    "combined_stock_data.csv"
)

QLIKE_RESULTS_FILE = (
    PROJECT_ROOT /
    "outputs/comparison/"
    "qlike_comparison_by_asset.csv"
)

LIVE_DATA_PATH = (
    PROJECT_ROOT /
    "data/live"
)

LIVE_OUTPUT_PATH = (
    PROJECT_ROOT /
    "outputs/live"
)

LIVE_DATA_PATH.mkdir(
    parents=True,
    exist_ok=True
)

LIVE_OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


## Deployment files

MANIFEST_FILE = (
    METADATA_PATH /
    "deployment_manifest.csv"
)

FEATURE_DEFINITION_FILE = (
    METADATA_PATH /
    "feature_definition.json"
)

GARCH_PARAMETER_FILE = (
    GARCH_MODEL_PATH /
    "garch_deployment_parameters.csv"
)


## Settings

DOWNLOAD_START = "2010-06-29"

EPSILON = 1e-8

TRADING_DAYS_PER_YEAR = 252

RECENT_OVERLAP_DAYS = 14

YAHOO_TIMEOUT_SECONDS = 12

YAHOO_DOWNLOAD_ATTEMPTS = 2

YAHOO_RETRY_DELAY_SECONDS = 2

MARKET_TIMEZONE = (
    ZoneInfo(
        "America/New_York"
    )
)

CURRENT_DAY_ACCEPT_TIME = (
    time(
        hour=16,
        minute=20
    )
)

OVERLAP_WARNING_THRESHOLD = 0.05


## Feature definitions

RF_LAGS = [
    1,
    2,
    3,
    5,
    10,
    21
]

RF_ROLLING_WINDOWS = [
    5,
    10,
    21,
    63
]

EXPECTED_RF_FEATURES = [
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "Return_Lag10",
    "Return_Lag21",

    "Squared_Return_Lag1",
    "Squared_Return_Lag2",
    "Squared_Return_Lag3",
    "Squared_Return_Lag5",
    "Squared_Return_Lag10",
    "Squared_Return_Lag21",

    "Rolling_Vol_5",
    "Rolling_Vol_10",
    "Rolling_Vol_21",
    "Rolling_Vol_63"
]

EXPECTED_LSTM_FEATURES = [
    "Log_Return",
    "Squared_Return"
]


## Random forest cache

_RF_MODEL_CACHE = {}


def load_rf_model_cached(path):

    path = str(path)

    if path not in _RF_MODEL_CACHE:

        print(
            "Loading Random Forest model..."
        )

        _RF_MODEL_CACHE[path] = (
            joblib.load(path)
        )

    return (
        _RF_MODEL_CACHE[path]
    )


## Load deployment metadata

def load_deployment_metadata():

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(
            f"Missing deployment manifest: "
            f"{MANIFEST_FILE}"
        )

    if not GARCH_PARAMETER_FILE.exists():

        raise FileNotFoundError(
            f"Missing GARCH parameters: "
            f"{GARCH_PARAMETER_FILE}"
        )

    manifest = pd.read_csv(
        MANIFEST_FILE
    )

    manifest[
        "GARCH_Training_Through"
    ] = pd.to_datetime(
        manifest[
            "GARCH_Training_Through"
        ]
    )

    garch_parameters = (
        pd.read_csv(
            GARCH_PARAMETER_FILE
        )
    )

    garch_parameters[
        "Training_Through"
    ] = pd.to_datetime(
        garch_parameters[
            "Training_Through"
        ]
    )

    if FEATURE_DEFINITION_FILE.exists():

        with open(
            FEATURE_DEFINITION_FILE,
            "r"
        ) as file:

            feature_definition = (
                json.load(file)
            )

        rf_features = (
            feature_definition[
                "random_forest_features"
            ]
        )

        lstm_features = (
            feature_definition[
                "lstm_features"
            ]
        )

    else:

        rf_features = (
            EXPECTED_RF_FEATURES
        )

        lstm_features = (
            EXPECTED_LSTM_FEATURES
        )

    if (
        rf_features
        != EXPECTED_RF_FEATURES
    ):

        raise ValueError(
            "Saved RF feature definition "
            "does not match the live "
            "prediction engine."
        )

    if (
        lstm_features
        != EXPECTED_LSTM_FEATURES
    ):

        raise ValueError(
            "Saved LSTM feature definition "
            "does not match the live "
            "prediction engine."
        )

    return (
        manifest,
        garch_parameters,
        rf_features,
        lstm_features
    )


## Safe ticker name

def safe_ticker_name(ticker):

    return (
        str(ticker)
        .replace(
            "^",
            ""
        )
    )


## Load existing live history

def load_existing_live_history(
    ticker
):

    safe_ticker = (
        safe_ticker_name(
            ticker
        )
    )

    live_file = (
        LIVE_DATA_PATH /
        f"{safe_ticker}_live_processed.csv"
    )

    if not live_file.exists():

        return (
            None,
            live_file
        )

    existing = pd.read_csv(
        live_file
    )

    if (
        "Date"
        not in existing.columns
    ):

        return (
            None,
            live_file
        )

    existing[
        "Date"
    ] = pd.to_datetime(
        existing[
            "Date"
        ]
    )

    existing = (
        existing
        .sort_values(
            "Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    if len(existing) == 0:

        return (
            None,
            live_file
        )

    return (
        existing,
        live_file
    )


## Determine download start

def determine_download_start(
    existing_history
):

    if (
        existing_history is None
        or
        len(existing_history) == 0
    ):

        return (
            DOWNLOAD_START,
            False
        )

    latest_existing_date = (
        pd.to_datetime(
            existing_history[
                "Date"
            ].max()
        )
        .date()
    )

    recent_start = (
        latest_existing_date
        -
        timedelta(
            days=RECENT_OVERLAP_DAYS
        )
    )

    return (
        recent_start.isoformat(),
        True
    )


## Download recent market data

def download_market_data(
    ticker,
    start_date
):

    print(
        f"Downloading {ticker} data "
        f"from {start_date}..."
    )

    last_error = None

    data = None

    for attempt in range(
        1,
        YAHOO_DOWNLOAD_ATTEMPTS + 1
    ):

        print(
            f"Yahoo request attempt "
            f"{attempt}/"
            f"{YAHOO_DOWNLOAD_ATTEMPTS}"
        )

        try:

            data = yf.download(
                ticker,
                start=start_date,
                interval="1d",
                auto_adjust=False,
                actions=True,
                repair=False,
                keepna=False,
                progress=False,
                threads=False,
                multi_level_index=False,
                timeout=
                    YAHOO_TIMEOUT_SECONDS
            )

            if (
                data is not None
                and
                len(data) > 0
            ):

                print(
                    f"Yahoo download complete: "
                    f"{len(data)} raw rows."
                )

                break

            last_error = RuntimeError(
                "Yahoo Finance returned "
                "an empty dataset."
            )

        except Exception as error:

            last_error = error

            data = None

        if (
            attempt
            <
            YAHOO_DOWNLOAD_ATTEMPTS
        ):

            print(
                "Download failed or returned "
                "no rows. Retrying..."
            )

            time_module.sleep(
                YAHOO_RETRY_DELAY_SECONDS
            )

    else:

        raise RuntimeError(
            f"{ticker}: Yahoo Finance "
            f"download failed after "
            f"{YAHOO_DOWNLOAD_ATTEMPTS} "
            f"attempts. Last error: "
            f"{last_error}"
        )

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = [
            column[0]
            if isinstance(
                column,
                tuple
            )
            else column

            for column
            in data.columns
        ]

    data = (
        data
        .reset_index()
    )

    if (
        "Date"
        not in data.columns
    ):

        first_column = (
            data.columns[0]
        )

        data = data.rename(
            columns={
                first_column:
                    "Date"
            }
        )

    data[
        "Date"
    ] = pd.to_datetime(
        data[
            "Date"
        ]
    )

    try:

        data[
            "Date"
        ] = (
            data[
                "Date"
            ]
            .dt
            .tz_localize(
                None
            )
        )

    except TypeError:

        pass

    data[
        "Date"
    ] = (
        data[
            "Date"
        ]
        .dt
        .normalize()
    )

    data = data.rename(
        columns={
            "Adj Close":
                "Adj_Close",

            "Stock Splits":
                "Stock_Splits"
        }
    )

    if (
        "Adj_Close"
        not in data.columns
    ):

        raise ValueError(
            f"{ticker}: Adj Close "
            "was not returned."
        )

    data = (
        data
        .sort_values(
            "Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    return data


## Remove incomplete current-day bar

def keep_completed_daily_bars(
    data
):

    now_new_york = (
        datetime.now(
            MARKET_TIMEZONE
        )
    )

    current_date = (
        now_new_york.date()
    )

    current_time = (
        now_new_york
        .time()
        .replace(
            tzinfo=None
        )
    )

    dates = (
        data[
            "Date"
        ]
        .dt
        .date
    )

    if (
        current_time
        <
        CURRENT_DAY_ACCEPT_TIME
    ):

        completed = (
            data[
                dates
                <
                current_date
            ]
            .copy()
        )

    else:

        completed = (
            data[
                dates
                <=
                current_date
            ]
            .copy()
        )

    completed = (
        completed
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )

    if len(completed) == 0:

        raise ValueError(
            "No completed daily bars available."
        )

    return (
        completed,
        now_new_york
    )


## Create returns

def create_processed_returns(
    raw_data,
    ticker,
    asset,
    role
):

    data = raw_data.copy()

    data = (
        data[
            data[
                "Adj_Close"
            ].notna()
        ]
        .copy()
    )

    if (
        data[
            "Adj_Close"
        ]
        <= 0
    ).any():

        raise ValueError(
            f"{ticker}: non-positive "
            "adjusted price detected."
        )

    data[
        "Log_Return"
    ] = (
        100
        *
        np.log(
            data[
                "Adj_Close"
            ]
            /
            data[
                "Adj_Close"
            ].shift(1)
        )
    )

    data[
        "Squared_Return"
    ] = (
        data[
            "Log_Return"
        ]
        ** 2
    )

    data[
        "Ticker"
    ] = ticker

    data[
        "Asset"
    ] = asset

    data[
        "Role"
    ] = role

    data = (
        data[
            data[
                "Log_Return"
            ].notna()
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return data


## Merge existing + recent history

def merge_live_history(
    existing_history,
    recent_processed
):

    if (
        existing_history is None
        or
        len(existing_history) == 0
    ):

        combined = (
            recent_processed.copy()
        )

    else:

        existing = (
            existing_history.copy()
        )

        existing[
            "Date"
        ] = pd.to_datetime(
            existing[
                "Date"
            ]
        )

        recent = (
            recent_processed.copy()
        )

        recent[
            "Date"
        ] = pd.to_datetime(
            recent[
                "Date"
            ]
        )

        overlap_start = (
            recent[
                "Date"
            ].min()
        )

        old_non_overlap = (
            existing[
                existing[
                    "Date"
                ]
                <
                overlap_start
            ]
            .copy()
        )

        combined = pd.concat(
            [
                old_non_overlap,
                recent
            ],
            ignore_index=True
        )

    combined = (
        combined
        .sort_values(
            "Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last"
        )
        .reset_index(
            drop=True
        )
    )

    return combined


## Update complete live history

def update_live_history(
    ticker,
    asset,
    role
):

    (
        existing_history,
        live_file
    ) = load_existing_live_history(
        ticker
    )

    (
        start_date,
        incremental_mode
    ) = determine_download_start(
        existing_history
    )

    if incremental_mode:

        print(
            "Using incremental refresh."
        )

        print(
            "Existing local history through: "
            f"{existing_history['Date'].max().date()}"
        )

    else:

        print(
            "No usable local live history found. "
            "A full history download is required."
        )

    raw_recent = (
        download_market_data(
            ticker=ticker,
            start_date=start_date
        )
    )

    (
        completed_recent,
        refresh_time
    ) = keep_completed_daily_bars(
        raw_recent
    )

    recent_processed = (
        create_processed_returns(
            completed_recent,
            ticker=ticker,
            asset=asset,
            role=role
        )
    )

    combined_history = (
        merge_live_history(
            existing_history=
                existing_history,

            recent_processed=
                recent_processed
        )
    )

    if len(combined_history) < 63:

        print(
            "Local history is insufficient. "
            "Downloading full history once..."
        )

        raw_full = (
            download_market_data(
                ticker=ticker,
                start_date=
                    DOWNLOAD_START
            )
        )

        (
            completed_full,
            refresh_time
        ) = keep_completed_daily_bars(
            raw_full
        )

        combined_history = (
            create_processed_returns(
                completed_full,
                ticker=ticker,
                asset=asset,
                role=role
            )
        )

    combined_history.to_csv(
        live_file,
        index=False
    )

    print(
        f"Complete local history: "
        f"{len(combined_history)} returns."
    )

    print(
        "Latest completed observation: "
        f"{combined_history['Date'].max().date()}"
    )

    return (
        combined_history,
        refresh_time
    )


## Research overlap check

def compare_with_frozen_research_data(
    live_processed,
    ticker
):

    if not (
        PROCESSED_RESEARCH_FILE.exists()
    ):

        return (
            np.nan,
            "Research file unavailable"
        )

    research = pd.read_csv(
        PROCESSED_RESEARCH_FILE
    )

    research[
        "Date"
    ] = pd.to_datetime(
        research[
            "Date"
        ]
    )

    research = (
        research[
            research[
                "Ticker"
            ]
            == ticker
        ][
            [
                "Date",
                "Log_Return"
            ]
        ]
        .rename(
            columns={
                "Log_Return":
                    "Frozen_Log_Return"
            }
        )
    )

    live = (
        live_processed[
            [
                "Date",
                "Log_Return"
            ]
        ]
        .rename(
            columns={
                "Log_Return":
                    "Live_Log_Return"
            }
        )
    )

    overlap = (
        research
        .merge(
            live,
            on="Date",
            how="inner"
        )
    )

    if len(overlap) == 0:

        return (
            np.nan,
            "No overlapping dates"
        )

    overlap = (
        overlap
        .sort_values(
            "Date"
        )
        .tail(
            60
        )
    )

    max_difference = (
        np.abs(
            overlap[
                "Live_Log_Return"
            ]
            -
            overlap[
                "Frozen_Log_Return"
            ]
        )
        .max()
    )

    if (
        max_difference
        >
        OVERLAP_WARNING_THRESHOLD
    ):

        status = (
            "WARNING: fresh vendor data differs "
            "from frozen research history"
        )

    else:

        status = "OK"

    return (
        float(
            max_difference
        ),
        status
    )


## Random forest features

def build_rf_feature_history(
    processed
):

    data = processed.copy()

    for lag in RF_LAGS:

        shift_amount = (
            lag - 1
        )

        data[
            f"Return_Lag{lag}"
        ] = (
            data[
                "Log_Return"
            ]
            .shift(
                shift_amount
            )
        )

        data[
            f"Squared_Return_Lag{lag}"
        ] = (
            data[
                "Squared_Return"
            ]
            .shift(
                shift_amount
            )
        )

    for window in (
        RF_ROLLING_WINDOWS
    ):

        data[
            f"Rolling_Vol_{window}"
        ] = (
            data[
                "Log_Return"
            ]
            .rolling(
                window=window,
                min_periods=window
            )
            .std(
                ddof=1
            )
        )

    return data


## Historical variance

def forecast_historical_variance(
    processed
):

    if len(processed) < 21:

        raise ValueError(
            "At least 21 returns are required "
            "for historical variance."
        )

    forecast = (
        processed[
            "Log_Return"
        ]
        .tail(
            21
        )
        .var(
            ddof=1
        )
    )

    return max(
        float(forecast),
        EPSILON
    )


## GARCH forecast

def forecast_garch(
    processed,
    ticker,
    manifest_row,
    garch_parameter_df
):

    print(
        "Generating GARCH forecast..."
    )

    returns = (
        processed
        .set_index(
            "Date"
        )[
            "Log_Return"
        ]
        .astype(
            float
        )
    )

    p = int(
        manifest_row[
            "GARCH_p"
        ]
    )

    q = int(
        manifest_row[
            "GARCH_q"
        ]
    )

    distribution = str(
        manifest_row[
            "GARCH_Distribution"
        ]
    )

    model = arch_model(
        returns,
        mean="Constant",
        vol="GARCH",
        p=p,
        o=0,
        q=q,
        dist=distribution,
        rescale=False
    )

    parameter_rows = (
        garch_parameter_df[
            garch_parameter_df[
                "Ticker"
            ]
            == ticker
        ]
        .copy()
        .sort_values(
            "Parameter_Order"
        )
    )

    if len(parameter_rows) == 0:

        raise ValueError(
            f"{ticker}: no saved GARCH "
            "parameters found."
        )

    parameters = (
        parameter_rows[
            "Estimate"
        ]
        .to_numpy(
            dtype=float
        )
    )

    fixed_result = (
        model.fix(
            parameters
        )
    )

    forecast_object = (
        fixed_result.forecast(
            horizon=1,
            method="analytic",
            reindex=False
        )
    )

    forecast = float(
        forecast_object
        .variance[
            "h.1"
        ]
        .iloc[-1]
    )

    if (
        not np.isfinite(
            forecast
        )
        or
        forecast <= 0
    ):

        raise ValueError(
            f"{ticker}: invalid live "
            "GARCH forecast."
        )

    return max(
        forecast,
        EPSILON
    )


## Random forest forecast

def forecast_random_forest(
    processed,
    manifest_row,
    rf_features
):

    print(
        "Generating Random Forest forecast..."
    )

    feature_history = (
        build_rf_feature_history(
            processed
        )
    )

    latest = (
        feature_history
        .iloc[-1]
    )

    missing_features = [
        feature
        for feature
        in rf_features
        if pd.isna(
            latest[
                feature
            ]
        )
    ]

    if len(missing_features) > 0:

        raise ValueError(
            "Latest RF observation is "
            f"missing features: "
            f"{missing_features}"
        )

    model_path = (
        PROJECT_ROOT
        /
        Path(
            str(
                manifest_row[
                    "RF_Model_Path"
                ]
            )
        )
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Missing RF model: "
            f"{model_path}"
        )

    model = (
        load_rf_model_cached(
            model_path
        )
    )

    X_latest = (
        latest[
            rf_features
        ]
        .to_numpy(
            dtype=float
        )
        .reshape(
            1,
            -1
        )
    )

    forecast = float(
        model.predict(
            X_latest
        )[0]
    )

    if not np.isfinite(
        forecast
    ):

        raise ValueError(
            "RF produced non-finite forecast."
        )

    return max(
        forecast,
        EPSILON
    )


## LSTM forecast

def forecast_lstm(
    processed,
    manifest_row,
    lstm_features,
    preloaded_model=None,
    preloaded_scaler=None
):

    print(
        "Generating LSTM forecast..."
    )

    lookback = int(
        manifest_row[
            "LSTM_Lookback"
        ]
    )

    if len(processed) < lookback:

        raise ValueError(
            f"LSTM requires at least "
            f"{lookback} observations."
        )

    if preloaded_model is None:

        raise ValueError(
            "No preloaded LSTM model was "
            "provided to the live engine."
        )

    if preloaded_scaler is None:

        raise ValueError(
            "No preloaded LSTM scaler was "
            "provided to the live engine."
        )

    print(
        "Using preloaded LSTM scaler."
    )

    scaler = (
        preloaded_scaler
    )

    print(
        "Using preloaded LSTM model."
    )

    model = (
        preloaded_model
    )

    latest_sequence = (
        processed[
            lstm_features
        ]
        .tail(
            lookback
        )
        .copy()
    )

    scaled_sequence = (
        scaler.transform(
            latest_sequence
        )
        .astype(
            np.float32
        )
    )

    X_latest = (
        scaled_sequence
        .reshape(
            1,
            lookback,
            len(
                lstm_features
            )
        )
    )

    prediction = (
        model(
            X_latest,
            training=False
        )
    )

    forecast = float(
        np.asarray(
            prediction
        )
        .reshape(
            -1
        )[0]
    )

    if not np.isfinite(
        forecast
    ):

        raise ValueError(
            "LSTM produced non-finite forecast."
        )

    return max(
        forecast,
        EPSILON
    )


## Variance to volatility

def variance_to_volatility(
    variance
):

    variance = max(
        float(variance),
        EPSILON
    )

    daily_volatility = (
        np.sqrt(
            variance
        )
    )

    annualised_volatility = (
        daily_volatility
        *
        np.sqrt(
            TRADING_DAYS_PER_YEAR
        )
    )

    return (
        float(
            daily_volatility
        ),
        float(
            annualised_volatility
        )
    )


## Research QLIKE

def load_research_qlike():

    if not (
        QLIKE_RESULTS_FILE.exists()
    ):

        return None

    return pd.read_csv(
        QLIKE_RESULTS_FILE
    )


def research_qlike_for_model(
    qlike_df,
    ticker,
    model_name
):

    if qlike_df is None:

        return np.nan

    row = (
        qlike_df[
            qlike_df[
                "Ticker"
            ]
            == ticker
        ]
    )

    if len(row) == 0:

        return np.nan

    if (
        model_name
        not in row.columns
    ):

        return np.nan

    return float(
        row[
            model_name
        ].iloc[0]
    )


def research_winner(
    qlike_df,
    ticker
):

    if qlike_df is None:

        return (
            None,
            np.nan
        )

    row = (
        qlike_df[
            qlike_df[
                "Ticker"
            ]
            == ticker
        ]
    )

    if len(row) == 0:

        return (
            None,
            np.nan
        )

    candidates = {}

    for model in [
        "Historical Variance",
        "GARCH",
        "Random Forest",
        "LSTM"
    ]:

        if model in row.columns:

            candidates[
                model
            ] = float(
                row[
                    model
                ].iloc[0]
            )

    if len(candidates) == 0:

        return (
            None,
            np.nan
        )

    winner = min(
        candidates,
        key=candidates.get
    )

    return (
        winner,
        candidates[
            winner
        ]
    )


## Generate forecasts for one asset

def generate_asset_forecasts(
    ticker,
    manifest,
    garch_parameters,
    rf_features,
    lstm_features,
    qlike_df=None,
    preloaded_lstm_model=None,
    preloaded_lstm_scaler=None
):

    matching = (
        manifest[
            manifest[
                "Ticker"
            ]
            == ticker
        ]
    )

    if len(matching) != 1:

        raise ValueError(
            f"{ticker}: expected exactly "
            "one manifest row."
        )

    manifest_row = (
        matching.iloc[0]
    )

    asset = str(
        manifest_row[
            "Asset"
        ]
    )

    role = str(
        manifest_row[
            "Role"
        ]
    )

    print("\n")
    print(
        f"Updating {asset} ({ticker})"
    )

    (
        processed,
        refresh_time
    ) = update_live_history(
        ticker=ticker,
        asset=asset,
        role=role
    )

    if len(processed) < 63:

        raise ValueError(
            f"{ticker}: fewer than "
            "63 completed returns."
        )

    latest_date = (
        processed[
            "Date"
        ].max()
    )

    latest_adj_close = float(
        processed[
            "Adj_Close"
        ].iloc[-1]
    )

    latest_return = float(
        processed[
            "Log_Return"
        ].iloc[-1]
    )

    latest_squared_return = float(
        processed[
            "Squared_Return"
        ].iloc[-1]
    )

    deployment_training_through = (
        pd.to_datetime(
            manifest_row[
                "GARCH_Training_Through"
            ]
        )
    )

    new_observations = int(
        (
            processed[
                "Date"
            ]
            >
            deployment_training_through
        )
        .sum()
    )

    (
        overlap_difference,
        overlap_status
    ) = compare_with_frozen_research_data(
        processed,
        ticker
    )

    print(
        "Generating Historical Variance forecast..."
    )

    historical_variance = (
        forecast_historical_variance(
            processed
        )
    )

    garch_variance = (
        forecast_garch(
            processed=processed,
            ticker=ticker,
            manifest_row=
                manifest_row,
            garch_parameter_df=
                garch_parameters
        )
    )

    rf_variance = (
        forecast_random_forest(
            processed=processed,
            manifest_row=
                manifest_row,
            rf_features=
                rf_features
        )
    )

    lstm_variance = (
        forecast_lstm(
            processed=processed,
            manifest_row=
                manifest_row,
            lstm_features=
                lstm_features,
            preloaded_model=
                preloaded_lstm_model,
            preloaded_scaler=
                preloaded_lstm_scaler
        )
    )

    forecasts = {
        "Historical Variance":
            historical_variance,

        "GARCH":
            garch_variance,

        "Random Forest":
            rf_variance,

        "LSTM":
            lstm_variance
    }

    (
        research_best_model,
        research_best_qlike
    ) = research_winner(
        qlike_df,
        ticker
    )

    rows = []

    for (
        model_name,
        forecast_variance
    ) in forecasts.items():

        (
            daily_volatility,
            annualised_volatility
        ) = variance_to_volatility(
            forecast_variance
        )

        model_qlike = (
            research_qlike_for_model(
                qlike_df=
                    qlike_df,
                ticker=
                    ticker,
                model_name=
                    model_name
            )
        )

        rows.append(
            {
                "Forecast_Generated_At":
                    refresh_time.isoformat(),

                "Ticker":
                    ticker,

                "Asset":
                    asset,

                "Role":
                    role,

                "Latest_Data_Date":
                    latest_date,

                "Latest_Adj_Close":
                    latest_adj_close,

                "Latest_Log_Return_Pct":
                    latest_return,

                "Latest_Squared_Return":
                    latest_squared_return,

                "Deployment_Model_Training_Through":
                    deployment_training_through,

                "New_Observations_Since_Deployment":
                    new_observations,

                "Model":
                    model_name,

                "Forecast_Horizon":
                    "Next trading session",

                "Forecast_Variance":
                    forecast_variance,

                "Forecast_Daily_Volatility_Pct":
                    daily_volatility,

                "Forecast_Annualised_Volatility_Pct":
                    annualised_volatility,

                "Research_Test_QLIKE":
                    model_qlike,

                "Research_QLIKE_Winner":
                    (
                        model_name
                        ==
                        research_best_model
                    ),

                "Research_Best_Model":
                    research_best_model,

                "Research_Best_QLIKE":
                    research_best_qlike,

                "Recent_Research_Overlap_Max_Abs_Return_Diff":
                    overlap_difference,

                "Research_Overlap_Status":
                    overlap_status
            }
        )

    print(
        f"{ticker}: all four forecasts complete."
    )

    return (
        pd.DataFrame(
            rows
        ),
        processed
    )


## Build wide table

def build_wide_forecast_table(
    long_df
):

    wide = (
        long_df
        .pivot(
            index=[
                "Ticker",
                "Asset",
                "Role",
                "Latest_Data_Date",
                "Latest_Adj_Close",
                "Latest_Log_Return_Pct",
                "New_Observations_Since_Deployment",
                "Research_Best_Model"
            ],
            columns="Model",
            values=
                "Forecast_Daily_Volatility_Pct"
        )
        .reset_index()
    )

    wide.columns.name = None

    return wide


## Save selected-asset forecast

def generate_single_live_forecast(
    ticker,
    preloaded_lstm_model=None,
    preloaded_lstm_scaler=None
):

    print("\n")
    print("=" * 90)
    print(
        f"SELECTED ASSET REFRESH: {ticker}"
    )
    print("=" * 90)

    (
        manifest,
        garch_parameters,
        rf_features,
        lstm_features
    ) = load_deployment_metadata()

    if ticker not in set(
        manifest[
            "Ticker"
        ]
    ):

        raise ValueError(
            f"Unknown deployment ticker: "
            f"{ticker}"
        )

    qlike_df = (
        load_research_qlike()
    )

    (
        asset_forecasts,
        processed
    ) = generate_asset_forecasts(
        ticker=ticker,
        manifest=manifest,
        garch_parameters=
            garch_parameters,
        rf_features=
            rf_features,
        lstm_features=
            lstm_features,
        qlike_df=
            qlike_df,
        preloaded_lstm_model=
            preloaded_lstm_model,
        preloaded_lstm_scaler=
            preloaded_lstm_scaler
    )

    ## Forecast history

    history_file = (
        LIVE_OUTPUT_PATH /
        "live_forecast_history.csv"
    )

    if history_file.exists():

        forecast_history = pd.read_csv(
            history_file
        )

        forecast_history = pd.concat(
            [
                forecast_history,
                asset_forecasts
            ],
            ignore_index=True
        )

    else:

        forecast_history = asset_forecasts.copy()

    forecast_history["Latest_Data_Date"] = pd.to_datetime(
        forecast_history["Latest_Data_Date"]
    )

    forecast_history = (
        forecast_history
        .sort_values(
            [
                "Latest_Data_Date",
                "Ticker",
                "Model",
                "Forecast_Generated_At"
            ]
        )
        .drop_duplicates(
            subset=[
                "Ticker",
                "Model",
                "Latest_Data_Date"
            ],
            keep="last"
        )
        .reset_index(drop=True)
    )

    forecast_history["Latest_Data_Date"] = (
        forecast_history["Latest_Data_Date"]
        .dt.strftime("%Y-%m-%d")
    )

    forecast_history.to_csv(
        history_file,
        index=False
    )

    ## Update long file

    long_file = (
        LIVE_OUTPUT_PATH /
        "latest_live_forecasts.csv"
    )

    if long_file.exists():

        existing_long = (
            pd.read_csv(
                long_file
            )
        )

        existing_long = (
            existing_long[
                existing_long[
                    "Ticker"
                ]
                != ticker
            ]
            .copy()
        )

        combined_long = (
            pd.concat(
                [
                    existing_long,
                    asset_forecasts
                ],
                ignore_index=True
            )
        )

    else:

        combined_long = (
            asset_forecasts.copy()
        )

    combined_long = (
        combined_long
        .sort_values(
            [
                "Ticker",
                "Model"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    combined_long.to_csv(
        long_file,
        index=False
    )

    ## Rebuild wide file

    wide = (
        build_wide_forecast_table(
            combined_long
        )
    )

    wide_file = (
        LIVE_OUTPUT_PATH /
        "latest_live_forecasts_wide.csv"
    )

    wide.to_csv(
        wide_file,
        index=False
    )

    ## Update quality file

    first_row = (
        asset_forecasts
        .iloc[0]
    )

    quality_row = pd.DataFrame(
        [
            {
                "Ticker":
                    ticker,

                "Asset":
                    first_row[
                        "Asset"
                    ],

                "Latest_Data_Date":
                    first_row[
                        "Latest_Data_Date"
                    ],

                "Completed_Return_Observations":
                    len(
                        processed
                    ),

                "Deployment_Training_Through":
                    first_row[
                        "Deployment_Model_Training_Through"
                    ],

                "New_Observations_Since_Deployment":
                    first_row[
                        "New_Observations_Since_Deployment"
                    ],

                "Research_Overlap_Max_Abs_Return_Diff":
                    first_row[
                        "Recent_Research_Overlap_Max_Abs_Return_Diff"
                    ],

                "Research_Overlap_Status":
                    first_row[
                        "Research_Overlap_Status"
                    ]
            }
        ]
    )

    quality_file = (
        LIVE_OUTPUT_PATH /
        "live_data_quality.csv"
    )

    if quality_file.exists():

        existing_quality = (
            pd.read_csv(
                quality_file
            )
        )

        existing_quality = (
            existing_quality[
                existing_quality[
                    "Ticker"
                ]
                != ticker
            ]
            .copy()
        )

        combined_quality = (
            pd.concat(
                [
                    existing_quality,
                    quality_row
                ],
                ignore_index=True
            )
        )

    else:

        combined_quality = (
            quality_row.copy()
        )

    combined_quality = (
        combined_quality
        .sort_values(
            "Ticker"
        )
        .reset_index(
            drop=True
        )
    )

    combined_quality.to_csv(
        quality_file,
        index=False
    )

    print("\n")
    print("=" * 90)
    print(
        f"{ticker}: SELECTED-ASSET "
        "REFRESH COMPLETE"
    )
    print("=" * 90)

    return (
        asset_forecasts,
        wide[
            wide[
                "Ticker"
            ]
            == ticker
        ].copy(),
        quality_row
    )