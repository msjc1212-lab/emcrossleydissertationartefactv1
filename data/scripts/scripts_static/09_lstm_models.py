import os

# Reduce unnecessary TensorFlow console output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import pandas as pd
import numpy as np
import tensorflow as tf

from pathlib import Path

from sklearn.preprocessing import RobustScaler

from tensorflow import keras
from tensorflow.keras import layers


## Paths

FEATURE_PATH = Path("data/features")

OUTPUT_PATH = Path(
    "outputs/lstm"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILE = (
    FEATURE_PATH /
    "combined_lstm_source.csv"
)


## Settings

RANDOM_STATE = 42

EPSILON = 1e-8


# LSTM candidate settings

LOOKBACK_VALUES = [
    10,
    21,
    63
]


LSTM_UNITS_VALUES = [
    16,
    32
]


DROPOUT_RATE = 0.10

LEARNING_RATE = 0.001

BATCH_SIZE = 32

MAX_EPOCHS = 100

PATIENCE = 10


EXPECTED_VALIDATION_ROWS = 501

EXPECTED_TEST_ROWS = 647


## Input features

LSTM_FEATURES = [
    "Log_Return",
    "Squared_Return"
]


## Set initial random seed

keras.utils.set_random_seed(
    RANDOM_STATE
)


## Load data

df = pd.read_csv(
    INPUT_FILE
)


df["Date"] = pd.to_datetime(
    df["Date"]
)


df["Target_Date"] = pd.to_datetime(
    df["Target_Date"]
)


df = df.sort_values(
    [
        "Ticker",
        "Date"
    ]
).reset_index(
    drop=True
)


## Basic data checks

if df[
    LSTM_FEATURES
].isna().any().any():

    raise ValueError(
        "Missing LSTM input values detected."
    )


if df[
    "Target_Squared_Return"
].isna().any():

    raise ValueError(
        "Missing LSTM target values detected."
    )


if (
    df[
        "Target_Squared_Return"
    ] < 0
).any():

    raise ValueError(
        "Negative variance target detected."
    )


if not (
    df["Target_Date"]
    >
    df["Date"]
).all():

    raise ValueError(
        "Target-date leakage detected."
    )


## QLIKE metric for Keras

def qlike(
    y_true,
    y_pred
):

    y_pred = tf.maximum(
        y_pred,
        tf.cast(
            EPSILON,
            y_pred.dtype
        )
    )


    return (
        tf.math.log(
            y_pred
        )
        +
        y_true / y_pred
    )


## Standard evaluation metrics

def calculate_metrics(
    actual,
    forecast
):

    actual = np.asarray(
        actual,
        dtype=float
    ).reshape(-1)


    forecast = np.asarray(
        forecast,
        dtype=float
    ).reshape(-1)


    forecast = np.maximum(
        forecast,
        EPSILON
    )


    error = (
        forecast - actual
    )


    ## RMSE

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )


    ## MAE

    mae = np.mean(
        np.abs(
            error
        )
    )


    ## QLIKE

    qlike_losses = (
        np.log(
            forecast
        )
        +
        actual / forecast
    )


    qlike_value = np.mean(
        qlike_losses
    )


    return {
        "RMSE": rmse,
        "MAE": mae,
        "QLIKE": qlike_value
    }


## Create LSTM model

def build_lstm_model(
    lookback,
    number_features,
    units
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
                units=units,

                dropout=
                    DROPOUT_RATE,

                recurrent_dropout=
                    0.0
            ),

            # Softplus guarantees a
            # positive variance forecast.

            layers.Dense(
                1,
                activation="softplus"
            )
        ]
    )


    optimizer = (
        keras.optimizers.Adam(
            learning_rate=
                LEARNING_RATE,

            # Mild gradient clipping
            # protects against unstable
            # updates from extreme observations.

            clipnorm=1.0
        )
    )


    model.compile(

        optimizer=
            optimizer,

        # RF also used squared-error
        # regression, so MSE is retained
        # as the neural-network training loss.

        loss="mse",

        # QLIKE is monitored on validation
        # and controls model selection.

        metrics=[
            qlike
        ]
    )


    return model


## Sequence creation function

def create_sequences(
    asset_df,
    scaled_features,
    lookback,
    allowed_splits
):

    X = []
    y = []

    metadata = []


    target_values = (
        asset_df[
            "Target_Squared_Return"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )


    for i in range(
        lookback - 1,
        len(asset_df)
    ):

        current_split = (
            asset_df.iloc[i][
                "Split"
            ]
        )


        if (
            current_split
            not in allowed_splits
        ):

            continue


        # Sequence ends at forecast origin t.
        # Target belongs to t+1.

        start_index = (
            i - lookback + 1
        )


        end_index = (
            i + 1
        )


        sequence = (
            scaled_features[
                start_index:
                end_index
            ]
        )


        X.append(
            sequence
        )


        y.append(
            [
                target_values[i]
            ]
        )


        metadata.append(
            {
                "Date":
                    asset_df.iloc[i][
                        "Date"
                    ],

                "Target_Date":
                    asset_df.iloc[i][
                        "Target_Date"
                    ],

                "Ticker":
                    asset_df.iloc[i][
                        "Ticker"
                    ],

                "Asset":
                    asset_df.iloc[i][
                        "Asset"
                    ],

                "Role":
                    asset_df.iloc[i][
                        "Role"
                    ],

                "Split":
                    current_split,

                "Target_Squared_Return":
                    target_values[i]
            }
        )


    X = np.asarray(
        X,
        dtype=np.float32
    )


    y = np.asarray(
        y,
        dtype=np.float32
    )


    metadata_df = pd.DataFrame(
        metadata
    )


    return (
        X,
        y,
        metadata_df
    )


## Storage

validation_results = []

validation_forecasts = []

selected_models = []

test_forecasts = []

test_metrics = []

scaler_results = []


## Process each asset

for ticker in df[
    "Ticker"
].unique():

    print("\n")
    print("=" * 80)
    print(
        f"LSTM MODELLING: {ticker}"
    )
    print("=" * 80)


    asset_df = (
        df[
            df["Ticker"] == ticker
        ]
        .copy()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


    asset_name = (
        asset_df[
            "Asset"
        ].iloc[0]
    )


    role = (
        asset_df[
            "Role"
        ].iloc[0]
    )


    ## Train-only scaler for validation

    train_mask = (
        asset_df[
            "Split"
        ] == "Train"
    )


    scaler = RobustScaler()


    scaler.fit(
        asset_df.loc[
            train_mask,
            LSTM_FEATURES
        ]
    )


    # Transform the entire chronological
    # source series using TRAINING
    # statistics only.
    # This lets validation sequences use
    # earlier historical observations
    # without allowing validation data to
    # influence scaling parameters.

    scaled_features = (
        scaler.transform(
            asset_df[
                LSTM_FEATURES
            ]
        )
        .astype(
            np.float32
        )
    )


    ## Validate each LSTM candidate

    candidate_number = 0


    for lookback in (
        LOOKBACK_VALUES
    ):

        # Sequences only have to be generated
        # once per lookback length.

        (
            X_train,
            y_train,
            train_metadata
        ) = create_sequences(

            asset_df=
                asset_df,

            scaled_features=
                scaled_features,

            lookback=
                lookback,

            allowed_splits=[
                "Train"
            ]
        )


        (
            X_validation,
            y_validation,
            validation_metadata
        ) = create_sequences(

            asset_df=
                asset_df,

            scaled_features=
                scaled_features,

            lookback=
                lookback,

            allowed_splits=[
                "Validation"
            ]
        )


        # Validation period must remain complete.

        if (
            len(
                X_validation
            )
            != EXPECTED_VALIDATION_ROWS
        ):

            raise ValueError(

                f"{ticker}, lookback "
                f"{lookback}: expected "
                f"{EXPECTED_VALIDATION_ROWS} "
                f"validation sequences but "
                f"found "
                f"{len(X_validation)}."
            )


        for units in (
            LSTM_UNITS_VALUES
        ):

            candidate_number += 1


            model_name = (

                f"LSTM_"
                f"Lookback{lookback}_"
                f"Units{units}"
            )


            print(
                f"Testing candidate "
                f"{candidate_number}/6: "
                f"{model_name}"
            )


            ## Reset Keras state

            keras.backend.clear_session()


            keras.utils.set_random_seed(
                RANDOM_STATE
            )


            ## Build model

            model = build_lstm_model(

                lookback=
                    lookback,

                number_features=
                    len(
                        LSTM_FEATURES
                    ),

                units=
                    units
            )


            ## Early stopping

            # Training weights are optimized
            # using MSE.
            # Early stopping is based on
            # validation QLIKE because QLIKE
            # is our primary variance-forecast
            # selection metric.

            early_stopping = (
                keras.callbacks.EarlyStopping(

                    monitor=
                        "val_qlike",

                    mode=
                        "min",

                    patience=
                        PATIENCE,

                    min_delta=
                        1e-4,

                    restore_best_weights=
                        True,

                    verbose=
                        0
                )
            )


            ## Train

            history = model.fit(

                X_train,
                y_train,

                validation_data=(
                    X_validation,
                    y_validation
                ),

                epochs=
                    MAX_EPOCHS,

                batch_size=
                    BATCH_SIZE,

                shuffle=
                    False,

                callbacks=[
                    early_stopping
                ],

                verbose=
                    0
            )


            ## Best epoch

            validation_qlike_history = (
                history.history[
                    "val_qlike"
                ]
            )


            best_epoch = (
                int(
                    np.argmin(
                        validation_qlike_history
                    )
                )
                + 1
            )


            epochs_trained = len(
                history.history[
                    "loss"
                ]
            )


            ## Validation forecasts

            validation_prediction = (
                model.predict(
                    X_validation,
                    verbose=0
                )
                .reshape(-1)
            )


            validation_prediction = (
                np.maximum(
                    validation_prediction,
                    EPSILON
                )
            )


            ## Validation metrics

            metrics = calculate_metrics(

                y_validation,

                validation_prediction
            )


            validation_results.append(
                {
                    "Ticker":
                        ticker,

                    "Asset":
                        asset_name,

                    "Role":
                        role,

                    "Model":
                        model_name,

                    "Lookback":
                        lookback,

                    "LSTM_Units":
                        units,

                    "Dropout":
                        DROPOUT_RATE,

                    "Learning_Rate":
                        LEARNING_RATE,

                    "Batch_Size":
                        BATCH_SIZE,

                    "Train_Sequences":
                        len(
                            X_train
                        ),

                    "Validation_Observations":
                        len(
                            X_validation
                        ),

                    "Epochs_Trained":
                        epochs_trained,

                    "Best_Epoch":
                        best_epoch,

                    "Model_Parameters":
                        model.count_params(),

                    "RMSE":
                        metrics[
                            "RMSE"
                        ],

                    "MAE":
                        metrics[
                            "MAE"
                        ],

                    "QLIKE":
                        metrics[
                            "QLIKE"
                        ]
                }
            )


            ## Store validation forecasts

            candidate_forecasts = (
                validation_metadata[
                    [
                        "Date",
                        "Target_Date",
                        "Ticker",
                        "Asset",
                        "Role",
                        "Target_Squared_Return"
                    ]
                ]
                .copy()
            )


            candidate_forecasts[
                "Model"
            ] = (
                model_name
            )


            candidate_forecasts[
                "Lookback"
            ] = (
                lookback
            )


            candidate_forecasts[
                "LSTM_Units"
            ] = (
                units
            )


            candidate_forecasts[
                "LSTM_Forecast"
            ] = (
                validation_prediction
            )


            validation_forecasts.append(
                candidate_forecasts
            )


            print(
                f"  Best epoch: "
                f"{best_epoch}"
            )


            print(
                f"  RMSE: "
                f"{metrics['RMSE']:.6f}"
            )


            print(
                f"  MAE: "
                f"{metrics['MAE']:.6f}"
            )


            print(
                f"  QLIKE: "
                f"{metrics['QLIKE']:.6f}"
            )


    ## Select best LSTM

    ticker_results = pd.DataFrame(
        [
            x
            for x
            in validation_results

            if (
                x[
                    "Ticker"
                ] == ticker
            )
        ]
    )


    ticker_results = (
        ticker_results
        .sort_values(
            [
                "QLIKE",
                "RMSE",
                "MAE"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    best = (
        ticker_results.iloc[0]
    )


    best_lookback = int(
        best[
            "Lookback"
        ]
    )


    best_units = int(
        best[
            "LSTM_Units"
        ]
    )


    best_epoch = int(
        best[
            "Best_Epoch"
        ]
    )


    print("\n")
    print(
        "Selected LSTM:"
    )


    print(
        "Lookback:",
        best_lookback
    )


    print(
        "Units:",
        best_units
    )


    print(
        "Final epochs:",
        best_epoch
    )


    print(
        "Validation QLIKE:",
        best["QLIKE"]
    )


    selected_models.append(
        {
            "Ticker":
                ticker,

            "Asset":
                asset_name,

            "Role":
                role,

            "Lookback":
                best_lookback,

            "LSTM_Units":
                best_units,

            "Dropout":
                DROPOUT_RATE,

            "Learning_Rate":
                LEARNING_RATE,

            "Batch_Size":
                BATCH_SIZE,

            "Best_Epoch":
                best_epoch,

            "Validation_RMSE":
                best[
                    "RMSE"
                ],

            "Validation_MAE":
                best[
                    "MAE"
                ],

            "Validation_QLIKE":
                best[
                    "QLIKE"
                ]
        }
    )


    ## Final scaler
    ## Train + validation

    train_validation_mask = (
        asset_df[
            "Split"
        ].isin(
            [
                "Train",
                "Validation"
            ]
        )
    )


    final_scaler = (
        RobustScaler()
    )


    final_scaler.fit(

        asset_df.loc[
            train_validation_mask,
            LSTM_FEATURES
        ]
    )


    final_scaled_features = (

        final_scaler.transform(
            asset_df[
                LSTM_FEATURES
            ]
        )

        .astype(
            np.float32
        )
    )


    # Save scaler values for reproducibility

    for feature_index, feature in enumerate(
        LSTM_FEATURES
    ):

        scaler_results.append(
            {
                "Ticker":
                    ticker,

                "Asset":
                    asset_name,

                "Feature":
                    feature,

                "Center_Median":
                    final_scaler.center_[
                        feature_index
                    ],

                "Scale_IQR":
                    final_scaler.scale_[
                        feature_index
                    ]
            }
        )


    ## Final train + validation sequences

    (
        X_train_validation,
        y_train_validation,
        train_validation_metadata
    ) = create_sequences(

        asset_df=
            asset_df,

        scaled_features=
            final_scaled_features,

        lookback=
            best_lookback,

        allowed_splits=[
            "Train",
            "Validation"
        ]
    )


    ## Final test sequences

    (
        X_test,
        y_test,
        test_metadata
    ) = create_sequences(

        asset_df=
            asset_df,

        scaled_features=
            final_scaled_features,

        lookback=
            best_lookback,

        allowed_splits=[
            "Test"
        ]
    )


    if (
        len(
            X_test
        )
        != EXPECTED_TEST_ROWS
    ):

        raise ValueError(

            f"{ticker}: expected "
            f"{EXPECTED_TEST_ROWS} "
            f"test sequences but found "
            f"{len(X_test)}."
        )


    ## Refit final LSTM

    keras.backend.clear_session()


    keras.utils.set_random_seed(
        RANDOM_STATE
    )


    final_model = build_lstm_model(

        lookback=
            best_lookback,

        number_features=
            len(
                LSTM_FEATURES
            ),

        units=
            best_units
    )


    # We do NOT use the test set for early stopping.
    # The number of final epochs is frozen using the
    # best epoch identified during validation.

    final_model.fit(

        X_train_validation,

        y_train_validation,

        epochs=
            best_epoch,

        batch_size=
            BATCH_SIZE,

        shuffle=
            False,

        verbose=
            0
    )


    ## Final test forecasts

    test_prediction = (

        final_model.predict(
            X_test,
            verbose=0
        )

        .reshape(-1)
    )


    test_prediction = (
        np.maximum(
            test_prediction,
            EPSILON
        )
    )


    if not np.isfinite(
        test_prediction
    ).all():

        raise ValueError(
            f"{ticker}: non-finite "
            "LSTM forecasts detected."
        )


    ## Final test metrics

    final_metrics = (
        calculate_metrics(

            y_test,

            test_prediction
        )
    )


    test_metrics.append(
        {
            "Ticker":
                ticker,

            "Asset":
                asset_name,

            "Role":
                role,

            "Model":
                "LSTM",

            "Observations":
                len(
                    X_test
                ),

            "Lookback":
                best_lookback,

            "LSTM_Units":
                best_units,

            "Dropout":
                DROPOUT_RATE,

            "Final_Epochs":
                best_epoch,

            "RMSE":
                final_metrics[
                    "RMSE"
                ],

            "MAE":
                final_metrics[
                    "MAE"
                ],

            "QLIKE":
                final_metrics[
                    "QLIKE"
                ],

            "Mean_Actual_Variance":
                np.mean(
                    y_test
                ),

            "Mean_Forecast_Variance":
                np.mean(
                    test_prediction
                )
        }
    )


    print("\n")
    print(
        "FINAL TEST PERFORMANCE"
    )


    print(
        "RMSE:",
        final_metrics[
            "RMSE"
        ]
    )


    print(
        "MAE:",
        final_metrics[
            "MAE"
        ]
    )


    print(
        "QLIKE:",
        final_metrics[
            "QLIKE"
        ]
    )


    ## Save individual forecast file

    ticker_test = (

        test_metadata[
            [
                "Date",
                "Target_Date",
                "Ticker",
                "Asset",
                "Role",
                "Target_Squared_Return"
            ]
        ]

        .copy()
    )


    ticker_test[
        "LSTM_Forecast"
    ] = (
        test_prediction
    )


    ticker_test[
        "Model"
    ] = (
        "LSTM"
    )


    ticker_test[
        "Lookback"
    ] = (
        best_lookback
    )


    ticker_test[
        "LSTM_Units"
    ] = (
        best_units
    )


    safe_ticker = (
        ticker.replace(
            "^",
            ""
        )
    )


    ticker_test.to_csv(

        OUTPUT_PATH /
        f"{safe_ticker}_lstm_test_forecasts.csv",

        index=False
    )


    test_forecasts.append(
        ticker_test
    )


## Create output dataframes

validation_results_df = (
    pd.DataFrame(
        validation_results
    )
)


selected_models_df = (
    pd.DataFrame(
        selected_models
    )
)


test_metrics_df = (
    pd.DataFrame(
        test_metrics
    )
)


scaler_results_df = (
    pd.DataFrame(
        scaler_results
    )
)


## Combine validation forecasts

combined_validation_forecasts = (

    pd.concat(
        validation_forecasts,
        ignore_index=True
    )

    .sort_values(
        [
            "Ticker",
            "Model",
            "Target_Date"
        ]
    )

    .reset_index(
        drop=True
    )
)


## Combine test forecasts

combined_test_forecasts = (

    pd.concat(
        test_forecasts,
        ignore_index=True
    )

    .sort_values(
        [
            "Ticker",
            "Target_Date"
        ]
    )

    .reset_index(
        drop=True
    )
)


## Stock-only average

stock_test = (

    test_metrics_df[
        test_metrics_df[
            "Role"
        ] == "Stock"
    ]

    .copy()
)


stock_average = pd.DataFrame(
    {
        "Model": [
            "LSTM"
        ],

        "Mean_RMSE": [
            stock_test[
                "RMSE"
            ].mean()
        ],

        "Mean_MAE": [
            stock_test[
                "MAE"
            ].mean()
        ],

        "Mean_QLIKE": [
            stock_test[
                "QLIKE"
            ].mean()
        ],

        "Number_of_Stocks": [
            stock_test[
                "Ticker"
            ].nunique()
        ]
    }
)


## Save outputs

validation_results_df.to_csv(

    OUTPUT_PATH /
    "lstm_validation_results.csv",

    index=False
)


selected_models_df.to_csv(

    OUTPUT_PATH /
    "lstm_selected_models.csv",

    index=False
)


combined_validation_forecasts.to_csv(

    OUTPUT_PATH /
    "combined_lstm_validation_forecasts.csv",

    index=False
)


combined_test_forecasts.to_csv(

    OUTPUT_PATH /
    "combined_lstm_test_forecasts.csv",

    index=False
)


test_metrics_df.to_csv(

    OUTPUT_PATH /
    "lstm_test_metrics.csv",

    index=False
)


stock_average.to_csv(

    OUTPUT_PATH /
    "lstm_stock_average.csv",

    index=False
)


scaler_results_df.to_csv(

    OUTPUT_PATH /
    "lstm_scaler_parameters.csv",

    index=False
)


## Final quality checks

# One selected model per asset

if (
    len(
        selected_models_df
    )
    != 7
):

    raise ValueError(
        "Expected exactly seven "
        "selected LSTM models."
    )


# Six candidate models per asset

for ticker in (
    validation_results_df[
        "Ticker"
    ].unique()
):

    ticker_results = (

        validation_results_df[
            validation_results_df[
                "Ticker"
            ] == ticker
        ]
    )


    if (
        len(
            ticker_results
        )
        != 6
    ):

        raise ValueError(

            f"{ticker}: expected "
            "6 LSTM candidates."
        )


    if not (
        ticker_results[
            "Validation_Observations"
        ]
        == EXPECTED_VALIDATION_ROWS
    ).all():

        raise ValueError(

            f"{ticker}: invalid "
            "validation forecast count."
        )


# Test forecast counts

for ticker in (
    test_metrics_df[
        "Ticker"
    ]
):

    ticker_forecasts = (

        combined_test_forecasts[
            combined_test_forecasts[
                "Ticker"
            ] == ticker
        ]
    )


    if (
        len(
            ticker_forecasts
        )
        != EXPECTED_TEST_ROWS
    ):

        raise ValueError(

            f"{ticker}: expected "
            f"{EXPECTED_TEST_ROWS} "
            f"test forecasts."
        )


# Missing forecasts

if (
    combined_test_forecasts[
        "LSTM_Forecast"
    ]
    .isna()
    .any()
):

    raise ValueError(
        "Missing LSTM test forecast."
    )


# Positive forecasts

if (
    combined_test_forecasts[
        "LSTM_Forecast"
    ]
    <= 0
).any():

    raise ValueError(
        "Non-positive LSTM "
        "variance forecast detected."
    )


# Chronology

if not (
    combined_test_forecasts[
        "Target_Date"
    ]
    >
    combined_test_forecasts[
        "Date"
    ]
).all():

    raise ValueError(
        "Target-date leakage "
        "detected in LSTM forecasts."
    )


# Expected combined test rows

expected_combined_test = (
    7 * EXPECTED_TEST_ROWS
)


if (
    len(
        combined_test_forecasts
    )
    != expected_combined_test
):

    raise ValueError(
        "Incorrect combined LSTM "
        "test forecast count."
    )


# Expected combined validation rows

expected_combined_validation = (
    7
    * 6
    * EXPECTED_VALIDATION_ROWS
)


if (
    len(
        combined_validation_forecasts
    )
    != expected_combined_validation
):

    raise ValueError(
        "Incorrect combined LSTM "
        "validation forecast count."
    )


## Display results

print("\n")
print("=" * 100)
print(
    "SELECTED LSTM MODELS"
)
print("=" * 100)


print(
    selected_models_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print(
    "FINAL LSTM TEST PERFORMANCE"
)
print("=" * 100)


print(

    test_metrics_df[
        [
            "Ticker",
            "Observations",
            "Lookback",
            "LSTM_Units",
            "Final_Epochs",
            "RMSE",
            "MAE",
            "QLIKE"
        ]
    ]

    .to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print(
    "STOCK-ONLY LSTM AVERAGE"
)
print("=" * 100)


print(
    stock_average.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print(
    "LSTM MODELLING COMPLETE"
)
print("=" * 100)