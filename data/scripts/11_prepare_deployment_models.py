import os

# Reduce TensorFlow console noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import sys
import json
import platform

from pathlib import Path
from importlib.metadata import (
    version,
    PackageNotFoundError
)

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from arch import arch_model

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler

from tensorflow import keras
from tensorflow.keras import layers


## Purpose
# Builds deployment-only models using the specifications selected during validation.
# Deployment models are refitted on all currently labelled data for use by the live artefact.
# Frozen research outputs from Steps 1–10 are not overwritten.


## Paths

PROCESSED_PATH = Path(
    "data/processed"
)

FEATURE_PATH = Path(
    "data/features"
)

GARCH_OUTPUT_PATH = Path(
    "outputs/garch"
)

RF_OUTPUT_PATH = Path(
    "outputs/random_forest"
)

LSTM_OUTPUT_PATH = Path(
    "outputs/lstm"
)


# New deployment-only directories

MODEL_ROOT = Path(
    "models/deployment"
)

RF_MODEL_PATH = (
    MODEL_ROOT /
    "random_forest"
)

LSTM_MODEL_PATH = (
    MODEL_ROOT /
    "lstm"
)

SCALER_PATH = (
    MODEL_ROOT /
    "scalers"
)

GARCH_MODEL_PATH = (
    MODEL_ROOT /
    "garch"
)

METADATA_PATH = (
    MODEL_ROOT /
    "metadata"
)

DEPLOYMENT_OUTPUT_PATH = Path(
    "outputs/deployment"
)


for path in [

    RF_MODEL_PATH,
    LSTM_MODEL_PATH,
    SCALER_PATH,
    GARCH_MODEL_PATH,
    METADATA_PATH,
    DEPLOYMENT_OUTPUT_PATH

]:

    path.mkdir(
        parents=True,
        exist_ok=True
    )


## Input files

PROCESSED_FILE = (
    PROCESSED_PATH /
    "combined_stock_data.csv"
)


RF_FEATURE_FILE = (
    FEATURE_PATH /
    "combined_rf_features.csv"
)


LSTM_SOURCE_FILE = (
    FEATURE_PATH /
    "combined_lstm_source.csv"
)


GARCH_SELECTION_FILE = (
    GARCH_OUTPUT_PATH /
    "garch_selected_models.csv"
)


RF_SELECTION_FILE = (
    RF_OUTPUT_PATH /
    "rf_selected_models.csv"
)


LSTM_SELECTION_FILE = (
    LSTM_OUTPUT_PATH /
    "lstm_selected_models.csv"
)


## Settings

RANDOM_STATE = 42

EPSILON = 1e-8


## Random forest features

RF_FEATURES = [

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


## LSTM features

LSTM_FEATURES = [

    "Log_Return",
    "Squared_Return"

]


## Set random seed

keras.utils.set_random_seed(
    RANDOM_STATE
)


## Load helper

def load_csv(
    path
):

    if not path.exists():

        raise FileNotFoundError(
            f"Required file not found: {path}"
        )


    data = pd.read_csv(
        path
    )


    if (
        "Date"
        in data.columns
    ):

        data[
            "Date"
        ] = pd.to_datetime(
            data[
                "Date"
            ]
        )


    if (
        "Target_Date"
        in data.columns
    ):

        data[
            "Target_Date"
        ] = pd.to_datetime(
            data[
                "Target_Date"
            ]
        )


    return data


## Load source data

processed_df = load_csv(
    PROCESSED_FILE
)

rf_df = load_csv(
    RF_FEATURE_FILE
)

lstm_df = load_csv(
    LSTM_SOURCE_FILE
)


garch_selection = load_csv(
    GARCH_SELECTION_FILE
)

rf_selection = load_csv(
    RF_SELECTION_FILE
)

lstm_selection = load_csv(
    LSTM_SELECTION_FILE
)


## Ticker check

expected_tickers = set(
    garch_selection[
        "Ticker"
    ].unique()
)


if (
    len(
        expected_tickers
    )
    != 7
):

    raise ValueError(
        "Expected exactly seven "
        "deployment assets."
    )


if set(
    rf_selection[
        "Ticker"
    ].unique()
) != expected_tickers:

    raise ValueError(
        "RF selected ticker set does "
        "not match GARCH."
    )


if set(
    lstm_selection[
        "Ticker"
    ].unique()
) != expected_tickers:

    raise ValueError(
        "LSTM selected ticker set does "
        "not match GARCH."
    )


## Source data checks

for ticker in expected_tickers:

    if ticker not in set(
        processed_df[
            "Ticker"
        ]
    ):

        raise ValueError(
            f"{ticker}: missing from "
            "processed data."
        )


    if ticker not in set(
        rf_df[
            "Ticker"
        ]
    ):

        raise ValueError(
            f"{ticker}: missing from "
            "RF feature data."
        )


    if ticker not in set(
        lstm_df[
            "Ticker"
        ]
    ):

        raise ValueError(
            f"{ticker}: missing from "
            "LSTM source data."
        )


## RF data checks

if rf_df[
    RF_FEATURES
].isna().any().any():

    raise ValueError(
        "Missing Random Forest "
        "deployment features detected."
    )


if rf_df[
    "Target_Squared_Return"
].isna().any():

    raise ValueError(
        "Missing Random Forest targets."
    )


if (
    rf_df[
        "Target_Squared_Return"
    ] < 0
).any():

    raise ValueError(
        "Negative Random Forest "
        "variance target."
    )


## LSTM data checks

if lstm_df[
    LSTM_FEATURES
].isna().any().any():

    raise ValueError(
        "Missing LSTM source data."
    )


if lstm_df[
    "Target_Squared_Return"
].isna().any():

    raise ValueError(
        "Missing LSTM target."
    )


if (
    lstm_df[
        "Target_Squared_Return"
    ] < 0
).any():

    raise ValueError(
        "Negative LSTM target."
    )


## Build deployment LSTM

def build_lstm_model(
    lookback,
    number_features,
    units,
    dropout,
    learning_rate
):

    model = keras.Sequential(
        [

            keras.Input(
                shape=(
                    lookback,
                    number_features
                )
            ),

            layers.LSTM(

                units=
                    units,

                dropout=
                    dropout,

                recurrent_dropout=
                    0.0
            ),

            # Variance forecast must remain
            # positive.

            layers.Dense(
                1,
                activation=
                    "softplus"
            )
        ]
    )


    optimizer = (
        keras.optimizers.Adam(

            learning_rate=
                learning_rate,

            clipnorm=
                1.0
        )
    )


    # No custom QLIKE metric is required
    # in the saved deployment model.
    # This keeps later model loading simple.

    model.compile(

        optimizer=
            optimizer,

        loss=
            "mse"
    )


    return model


## LSTM sequence creation

def create_lstm_sequences(
    asset_df,
    scaled_features,
    lookback
):

    X = []
    y = []


    targets = (

        asset_df[
            "Target_Squared_Return"
        ]

        .to_numpy(
            dtype=np.float32
        )
    )


    for i in range(

        lookback - 1,

        len(
            asset_df
        )
    ):


        start = (
            i
            -
            lookback
            +
            1
        )


        end = (
            i + 1
        )


        X.append(

            scaled_features[
                start:end
            ]
        )


        y.append(
            [
                targets[i]
            ]
        )


    return (

        np.asarray(
            X,
            dtype=np.float32
        ),

        np.asarray(
            y,
            dtype=np.float32
        )
    )


## Helper: parse RF depth

def parse_depth(
    value
):

    if pd.isna(
        value
    ):

        return None


    if (
        str(
            value
        )
        .strip()
        .lower()
        in [
            "none",
            "nan"
        ]
    ):

        return None


    return int(
        float(
            value
        )
    )


## Storage

manifest_rows = []

garch_parameter_rows = []

deployment_checks = []


## Process each asset

for ticker in sorted(
    expected_tickers
):


    print("\n")
    print("=" * 100)

    print(
        f"CREATING DEPLOYMENT MODELS: "
        f"{ticker}"
    )

    print("=" * 100)


    ## Load selected configurations

    garch_config = (

        garch_selection[
            garch_selection[
                "Ticker"
            ] == ticker
        ]

        .iloc[0]
    )


    rf_config = (

        rf_selection[
            rf_selection[
                "Ticker"
            ] == ticker
        ]

        .iloc[0]
    )


    lstm_config = (

        lstm_selection[
            lstm_selection[
                "Ticker"
            ] == ticker
        ]

        .iloc[0]
    )


    asset_name = (
        garch_config[
            "Asset"
        ]
    )


    role = (
        garch_config[
            "Role"
        ]
    )


    safe_ticker = (
        ticker.replace(
            "^",
            ""
        )
    )


    ## GARCH deployment model

    print(
        "\n[1/3] Refitting GARCH..."
    )


    # GARCH uses the processed return dataset,
    # rather than combined_garch_input.csv.
    # This means the deployment fit includes
    # the latest OBSERVED return as well,
    # including the final day that had no
    # t+1 target in the research dataset.

    garch_asset = (

        processed_df[
            processed_df[
                "Ticker"
            ] == ticker
        ]

        .copy()

        .sort_values(
            "Date"
        )

        .reset_index(
            drop=True
        )
    )


    if garch_asset[
        "Log_Return"
    ].isna().any():

        raise ValueError(
            f"{ticker}: missing "
            "deployment GARCH return."
        )


    garch_returns = (

        garch_asset

        .set_index(
            "Date"
        )[
            "Log_Return"
        ]

        .astype(
            float
        )
    )


    garch_p = int(
        garch_config[
            "p"
        ]
    )


    garch_q = int(
        garch_config[
            "q"
        ]
    )


    garch_dist = str(
        garch_config[
            "Distribution"
        ]
    )


    garch_model_name = str(
        garch_config[
            "Selected_Model"
        ]
    )


    garch_model = arch_model(

        garch_returns,

        mean=
            "Constant",

        vol=
            "GARCH",

        p=
            garch_p,

        o=
            0,

        q=
            garch_q,

        dist=
            garch_dist,

        rescale=
            False
    )


    garch_result = (

        garch_model.fit(

            disp=
                "off",

            update_freq=
                0,

            show_warning=
                False
        )
    )


    if (
        garch_result.convergence_flag
        != 0
    ):

        raise ValueError(

            f"{ticker}: deployment "
            "GARCH failed to converge."
        )


    garch_training_through = (

        garch_asset[
            "Date"
        ]
        .max()
    )


    # GARCH persistence

    persistence = 0.0


    for (
        parameter_order,
        (
            parameter_name,
            parameter_value
        )
    ) in enumerate(

        garch_result.params.items()
    ):


        garch_parameter_rows.append(
            {
                "Ticker":
                    ticker,

                "Asset":
                    asset_name,

                "Role":
                    role,

                "Selected_Model":
                    garch_model_name,

                "p":
                    garch_p,

                "q":
                    garch_q,

                "Distribution":
                    garch_dist,

                "Parameter_Order":
                    parameter_order,

                "Parameter":
                    parameter_name,

                "Estimate":
                    float(
                        parameter_value
                    ),

                "Training_Through":
                    garch_training_through
            }
        )


        if (
            parameter_name.startswith(
                "alpha["
            )
            or
            parameter_name.startswith(
                "beta["
            )
        ):

            persistence += float(
                parameter_value
            )


    # Produce one-step forecast as deployment
    # smoke test.

    garch_forecast_object = (

        garch_result.forecast(

            horizon=
                1,

            method=
                "analytic",

            reindex=
                False
        )
    )


    garch_smoke_forecast = float(

        garch_forecast_object

        .variance[
            "h.1"
        ]

        .iloc[-1]
    )


    if (
        not np.isfinite(
            garch_smoke_forecast
        )
        or
        garch_smoke_forecast
        <= 0
    ):

        raise ValueError(

            f"{ticker}: invalid "
            "deployment GARCH forecast."
        )


    print(
        "  Training through:",
        garch_training_through.date()
    )

    print(
        "  Persistence:",
        round(
            persistence,
            6
        )
    )

    print(
        "  Smoke-test next variance:",
        round(
            garch_smoke_forecast,
            6
        )
    )


    ## Random forest deployment model

    print(
        "\n[2/3] Refitting Random Forest..."
    )


    rf_asset = (

        rf_df[
            rf_df[
                "Ticker"
            ] == ticker
        ]

        .copy()

        .sort_values(
            "Date"
        )

        .reset_index(
            drop=True
        )
    )


    X_rf = (

        rf_asset[
            RF_FEATURES
        ]

        .to_numpy(
            dtype=float
        )
    )


    y_rf = (

        rf_asset[
            "Target_Squared_Return"
        ]

        .to_numpy(
            dtype=float
        )
    )


    rf_n_estimators = int(
        rf_config[
            "N_Estimators"
        ]
    )


    rf_depth = parse_depth(
        rf_config[
            "Max_Depth"
        ]
    )


    rf_leaf = int(
        rf_config[
            "Min_Samples_Leaf"
        ]
    )


    rf_max_features = float(
        rf_config[
            "Max_Features"
        ]
    )


    rf_model = RandomForestRegressor(

        n_estimators=
            rf_n_estimators,

        criterion=
            "squared_error",

        max_depth=
            rf_depth,

        min_samples_leaf=
            rf_leaf,

        max_features=
            rf_max_features,

        bootstrap=
            True,

        random_state=
            RANDOM_STATE,

        n_jobs=
            -1
    )


    rf_model.fit(

        X_rf,
        y_rf
    )


    rf_file = (

        RF_MODEL_PATH /
        f"{safe_ticker}_rf.joblib"
    )


    joblib.dump(

        rf_model,

        rf_file,

        compress=
            3
    )


    # Reload immediately to prove persistence works.

    loaded_rf = joblib.load(
        rf_file
    )


    rf_probe = (

        X_rf[
            -5:
        ]
    )


    original_rf_prediction = (

        rf_model.predict(
            rf_probe
        )
    )


    loaded_rf_prediction = (

        loaded_rf.predict(
            rf_probe
        )
    )


    if not np.allclose(

        original_rf_prediction,

        loaded_rf_prediction,

        rtol=
            1e-12,

        atol=
            1e-12
    ):

        raise ValueError(

            f"{ticker}: saved RF "
            "predictions do not match."
        )


    rf_training_target_through = (

        rf_asset[
            "Target_Date"
        ]
        .max()
    )


    print(
        "  Training rows:",
        len(
            rf_asset
        )
    )

    print(
        "  Targets through:",
        rf_training_target_through.date()
    )

    print(
        "  Saved:",
        rf_file
    )


    ## LSTM deployment model

    print(
        "\n[3/3] Refitting LSTM..."
    )


    lstm_asset = (

        lstm_df[
            lstm_df[
                "Ticker"
            ] == ticker
        ]

        .copy()

        .sort_values(
            "Date"
        )

        .reset_index(
            drop=True
        )
    )


    lstm_lookback = int(
        lstm_config[
            "Lookback"
        ]
    )


    lstm_units = int(
        lstm_config[
            "LSTM_Units"
        ]
    )


    lstm_dropout = float(
        lstm_config[
            "Dropout"
        ]
    )


    lstm_learning_rate = float(
        lstm_config[
            "Learning_Rate"
        ]
    )


    lstm_batch_size = int(
        lstm_config[
            "Batch_Size"
        ]
    )


    lstm_epochs = int(
        lstm_config[
            "Best_Epoch"
        ]
    )


    # Fit deployment scaler using all labelled
    # forecast-origin observations.
    # This is post-evaluation deployment training,
    # so validation/test distinctions no longer apply.

    lstm_scaler = RobustScaler()


    lstm_scaler.fit(

        lstm_asset[
            LSTM_FEATURES
        ]
    )


    scaled_lstm_features = (

        lstm_scaler

        .transform(

            lstm_asset[
                LSTM_FEATURES
            ]
        )

        .astype(
            np.float32
        )
    )


    (
        X_lstm,
        y_lstm
    ) = create_lstm_sequences(

        asset_df=
            lstm_asset,

        scaled_features=
            scaled_lstm_features,

        lookback=
            lstm_lookback
    )


    if (
        len(
            X_lstm
        )
        == 0
    ):

        raise ValueError(

            f"{ticker}: no deployment "
            "LSTM sequences generated."
        )


    keras.backend.clear_session()


    keras.utils.set_random_seed(
        RANDOM_STATE
    )


    lstm_model = build_lstm_model(

        lookback=
            lstm_lookback,

        number_features=
            len(
                LSTM_FEATURES
            ),

        units=
            lstm_units,

        dropout=
            lstm_dropout,

        learning_rate=
            lstm_learning_rate
    )


    lstm_model.fit(

        X_lstm,
        y_lstm,

        epochs=
            lstm_epochs,

        batch_size=
            lstm_batch_size,

        shuffle=
            False,

        verbose=
            0
    )


    # Save complete Keras model

    lstm_file = (

        LSTM_MODEL_PATH /
        f"{safe_ticker}_lstm.keras"
    )


    lstm_model.save(
        lstm_file
    )


    # Save scaler

    scaler_file = (

        SCALER_PATH /
        f"{safe_ticker}_lstm_scaler.joblib"
    )


    joblib.dump(

        lstm_scaler,

        scaler_file,

        compress=
            3
    )


    # Reload model + scaler to confirm persistence.

    loaded_lstm = (

        keras.models.load_model(

            lstm_file,

            compile=
                False
        )
    )


    loaded_scaler = joblib.load(
        scaler_file
    )


    # Scaler consistency check

    original_scaled_probe = (

        lstm_scaler.transform(

            lstm_asset[
                LSTM_FEATURES
            ]
            .tail(
                lstm_lookback
            )
        )
    )


    loaded_scaled_probe = (

        loaded_scaler.transform(

            lstm_asset[
                LSTM_FEATURES
            ]
            .tail(
                lstm_lookback
            )
        )
    )


    if not np.allclose(

        original_scaled_probe,

        loaded_scaled_probe,

        rtol=
            1e-12,

        atol=
            1e-12
    ):

        raise ValueError(

            f"{ticker}: saved LSTM "
            "scaler mismatch."
        )


    # Model consistency check

    lstm_probe = (

        X_lstm[
            -5:
        ]
    )


    original_lstm_prediction = (

        lstm_model.predict(

            lstm_probe,

            verbose=
                0
        )

        .reshape(
            -1
        )
    )


    loaded_lstm_prediction = (

        loaded_lstm.predict(

            lstm_probe,

            verbose=
                0
        )

        .reshape(
            -1
        )
    )


    if not np.allclose(

        original_lstm_prediction,

        loaded_lstm_prediction,

        rtol=
            1e-5,

        atol=
            1e-6
    ):

        raise ValueError(

            f"{ticker}: saved LSTM "
            "predictions do not match."
        )


    if (
        loaded_lstm_prediction
        <= 0
    ).any():

        raise ValueError(

            f"{ticker}: non-positive "
            "LSTM prediction detected."
        )


    lstm_training_target_through = (

        lstm_asset[
            "Target_Date"
        ]
        .max()
    )


    print(
        "  Lookback:",
        lstm_lookback
    )

    print(
        "  Units:",
        lstm_units
    )

    print(
        "  Epochs:",
        lstm_epochs
    )

    print(
        "  Training sequences:",
        len(
            X_lstm
        )
    )

    print(
        "  Targets through:",
        lstm_training_target_through.date()
    )

    print(
        "  Saved:",
        lstm_file
    )


    ## Manifest entry

    manifest_rows.append(
        {

            "Ticker":
                ticker,

            "Safe_Ticker":
                safe_ticker,

            "Asset":
                asset_name,

            "Role":
                role,


            ## GARCH

            "GARCH_Selected_Model":
                garch_model_name,

            "GARCH_p":
                garch_p,

            "GARCH_q":
                garch_q,

            "GARCH_Distribution":
                garch_dist,

            "GARCH_Training_Through":
                garch_training_through,

            "GARCH_Observations":
                len(
                    garch_returns
                ),

            "GARCH_Persistence":
                persistence,

            "GARCH_Convergence_Flag":
                garch_result.convergence_flag,

            "GARCH_Smoke_Forecast":
                garch_smoke_forecast,


            # Random Forest

            "RF_N_Estimators":
                rf_n_estimators,

            "RF_Max_Depth":
                (
                    "None"
                    if rf_depth is None
                    else rf_depth
                ),

            "RF_Min_Samples_Leaf":
                rf_leaf,

            "RF_Max_Features":
                rf_max_features,

            "RF_Training_Rows":
                len(
                    rf_asset
                ),

            "RF_Target_Through":
                rf_training_target_through,

            "RF_Model_Path":
                str(
                    rf_file
                ),


            ## LSTM

            "LSTM_Lookback":
                lstm_lookback,

            "LSTM_Units":
                lstm_units,

            "LSTM_Dropout":
                lstm_dropout,

            "LSTM_Learning_Rate":
                lstm_learning_rate,

            "LSTM_Batch_Size":
                lstm_batch_size,

            "LSTM_Epochs":
                lstm_epochs,

            "LSTM_Training_Sequences":
                len(
                    X_lstm
                ),

            "LSTM_Target_Through":
                lstm_training_target_through,

            "LSTM_Model_Path":
                str(
                    lstm_file
                ),

            "LSTM_Scaler_Path":
                str(
                    scaler_file
                )
        }
    )


    ## Deployment check entry

    deployment_checks.append(
        {

            "Ticker":
                ticker,

            "Asset":
                asset_name,

            "GARCH_Converged":
                (
                    garch_result.convergence_flag
                    == 0
                ),

            "GARCH_Forecast_Positive":
                (
                    garch_smoke_forecast
                    > 0
                ),

            "RF_Save_Load_Match":
                bool(

                    np.allclose(

                        original_rf_prediction,

                        loaded_rf_prediction
                    )
                ),

            "LSTM_Save_Load_Match":
                bool(

                    np.allclose(

                        original_lstm_prediction,

                        loaded_lstm_prediction,

                        rtol=
                            1e-5,

                        atol=
                            1e-6
                    )
                ),

            "LSTM_Scaler_Save_Load_Match":
                bool(

                    np.allclose(

                        original_scaled_probe,

                        loaded_scaled_probe
                    )
                )
        }
    )


    # Reduce memory before next asset

    del rf_model
    del loaded_rf

    del lstm_model
    del loaded_lstm

    keras.backend.clear_session()


## Create output tables

manifest_df = pd.DataFrame(
    manifest_rows
)


garch_parameters_df = pd.DataFrame(
    garch_parameter_rows
)


deployment_checks_df = pd.DataFrame(
    deployment_checks
)


## Sort outputs

manifest_df = (

    manifest_df

    .sort_values(
        "Ticker"
    )

    .reset_index(
        drop=True
    )
)


garch_parameters_df = (

    garch_parameters_df

    .sort_values(
        [
            "Ticker",
            "Parameter_Order"
        ]
    )

    .reset_index(
        drop=True
    )
)


deployment_checks_df = (

    deployment_checks_df

    .sort_values(
        "Ticker"
    )

    .reset_index(
        drop=True
    )
)


## Save GARCH parameters

garch_parameter_file = (

    GARCH_MODEL_PATH /
    "garch_deployment_parameters.csv"
)


garch_parameters_df.to_csv(

    garch_parameter_file,

    index=
        False
)


## Save deployment manifest

manifest_file = (

    METADATA_PATH /
    "deployment_manifest.csv"
)


manifest_df.to_csv(

    manifest_file,

    index=
        False
)


# Also save in outputs for convenient inspection.

manifest_df.to_csv(

    DEPLOYMENT_OUTPUT_PATH /
    "deployment_manifest.csv",

    index=
        False
)


## Save deployment checks

deployment_checks_df.to_csv(

    DEPLOYMENT_OUTPUT_PATH /
    "deployment_model_checks.csv",

    index=
        False
)


## Save feature definitions

feature_definition = {

    "random_forest_features":
        RF_FEATURES,

    "lstm_features":
        LSTM_FEATURES,

    "target":
        "next-trading-day Squared_Return",

    "forecast_horizon":
        1,

    "random_seed":
        RANDOM_STATE
}


with open(

    METADATA_PATH /
    "feature_definition.json",

    "w"

) as file:

    json.dump(

        feature_definition,

        file,

        indent=
            4
    )


## Save software environment

def safe_version(
    package_name
):

    try:

        return version(
            package_name
        )

    except PackageNotFoundError:

        return (
            "Not installed / "
            "version unavailable"
        )


environment = {

    "python":
        sys.version,

    "platform":
        platform.platform(),

    "numpy":
        safe_version(
            "numpy"
        ),

    "pandas":
        safe_version(
            "pandas"
        ),

    "scikit-learn":
        safe_version(
            "scikit-learn"
        ),

    "joblib":
        safe_version(
            "joblib"
        ),

    "tensorflow":
        safe_version(
            "tensorflow"
        ),

    "keras":
        safe_version(
            "keras"
        ),

    "arch":
        safe_version(
            "arch"
        )
}


with open(

    METADATA_PATH /
    "deployment_environment.json",

    "w"

) as file:

    json.dump(

        environment,

        file,

        indent=
            4
    )


## Final quality checks

if (
    len(
        manifest_df
    )
    != 7
):

    raise ValueError(

        "Expected seven rows in "
        "deployment manifest."
    )


if (
    len(
        deployment_checks_df
    )
    != 7
):

    raise ValueError(

        "Expected seven rows in "
        "deployment checks."
    )


boolean_check_columns = [

    "GARCH_Converged",

    "GARCH_Forecast_Positive",

    "RF_Save_Load_Match",

    "LSTM_Save_Load_Match",

    "LSTM_Scaler_Save_Load_Match"
]


for column in (
    boolean_check_columns
):

    if not deployment_checks_df[
        column
    ].all():

        raise ValueError(

            f"Deployment quality "
            f"check failed: {column}"
        )


# Check all required RF files

for ticker in (
    manifest_df[
        "Safe_Ticker"
    ]
):

    rf_file = (

        RF_MODEL_PATH /
        f"{ticker}_rf.joblib"
    )


    if not rf_file.exists():

        raise FileNotFoundError(

            f"Missing saved RF model: "
            f"{rf_file}"
        )


# Check all required LSTM files

for ticker in (
    manifest_df[
        "Safe_Ticker"
    ]
):

    lstm_file = (

        LSTM_MODEL_PATH /
        f"{ticker}_lstm.keras"
    )


    scaler_file = (

        SCALER_PATH /
        f"{ticker}_lstm_scaler.joblib"
    )


    if not lstm_file.exists():

        raise FileNotFoundError(

            f"Missing saved LSTM model: "
            f"{lstm_file}"
        )


    if not scaler_file.exists():

        raise FileNotFoundError(

            f"Missing saved LSTM scaler: "
            f"{scaler_file}"
        )


## Display results

print("\n")
print("=" * 110)
print(
    "DEPLOYMENT MODEL MANIFEST"
)
print("=" * 110)


print(

    manifest_df[
        [
            "Ticker",
            "GARCH_Selected_Model",
            "GARCH_Training_Through",
            "RF_Target_Through",
            "LSTM_Target_Through",
            "LSTM_Lookback",
            "LSTM_Units",
            "LSTM_Epochs"
        ]
    ]

    .to_string(
        index=
            False
    )
)


print("\n")
print("=" * 110)
print(
    "DEPLOYMENT MODEL CHECKS"
)
print("=" * 110)


print(

    deployment_checks_df

    .to_string(
        index=
            False
    )
)


print("\n")
print("=" * 110)
print(
    "DEPLOYMENT MODEL CREATION COMPLETE"
)
print("=" * 110)


print(
    "\nModels saved under:"
)

print(
    MODEL_ROOT
)


print(
    "\nGARCH parameters:"
)

print(
    garch_parameter_file
)


print(
    "\nManifest:"
)

print(
    manifest_file
)