import pandas as pd
import numpy as np

from pathlib import Path
from itertools import product

from sklearn.ensemble import RandomForestRegressor


## Paths

FEATURE_PATH = Path("data/features")
OUTPUT_PATH = Path("outputs/random_forest")

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_FILE = (
    FEATURE_PATH /
    "combined_rf_features.csv"
)


## Settings

EPSILON = 1e-8
RANDOM_STATE = 42
N_ESTIMATORS = 500


# Hyperparameter search
# This gives:
# 3 max_depth values
# x
# 3 min_samples_leaf values
# x
# 2 max_features values
# = 18 candidate models per asset.
# Hyperparameters are selected using validation
# QLIKE only.

MAX_DEPTH_VALUES = [
    5,
    10,
    None
]

MIN_SAMPLES_LEAF_VALUES = [
    1,
    5,
    10
]

MAX_FEATURES_VALUES = [
    0.5,
    1.0
]


## Random forest features

RF_FEATURES = [

    # Returns
    "Return_Lag1",
    "Return_Lag2",
    "Return_Lag3",
    "Return_Lag5",
    "Return_Lag10",
    "Return_Lag21",

    # Squared returns
    "Squared_Return_Lag1",
    "Squared_Return_Lag2",
    "Squared_Return_Lag3",
    "Squared_Return_Lag5",
    "Squared_Return_Lag10",
    "Squared_Return_Lag21",

    # Historical volatility
    "Rolling_Vol_5",
    "Rolling_Vol_10",
    "Rolling_Vol_21",
    "Rolling_Vol_63"
]


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
).reset_index(drop=True)


## Initial data checks

missing_features = (
    df[RF_FEATURES]
    .isna()
    .sum()
    .sum()
)

if missing_features > 0:

    raise ValueError(
        f"RF input contains "
        f"{missing_features} missing features."
    )


if (
    df["Target_Squared_Return"] < 0
).any():

    raise ValueError(
        "Negative variance targets detected."
    )


if not (
    df["Target_Date"]
    >
    df["Date"]
).all():

    raise ValueError(
        "Target-date leakage detected."
    )


## Metric function

def calculate_metrics(
    actual,
    forecast
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    forecast = np.asarray(
        forecast,
        dtype=float
    )


    # Variance forecasts must be positive
    # for QLIKE.

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


    qlike = np.mean(
        qlike_losses
    )


    return {
        "RMSE": rmse,
        "MAE": mae,
        "QLIKE": qlike
    }


## Storage

validation_results = []
validation_forecasts = []

selected_models = []

test_forecasts = []
test_metrics = []

feature_importance_results = []


## Process each asset

for ticker in df["Ticker"].unique():

    print("\n")
    print("=" * 80)
    print(f"RANDOM FOREST MODELLING: {ticker}")
    print("=" * 80)


    asset_df = (
        df[
            df["Ticker"] == ticker
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
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


    ## Train / validation / test

    train_df = (
        asset_df[
            asset_df["Split"]
            == "Train"
        ]
        .copy()
    )


    validation_df = (
        asset_df[
            asset_df["Split"]
            == "Validation"
        ]
        .copy()
    )


    test_df = (
        asset_df[
            asset_df["Split"]
            == "Test"
        ]
        .copy()
    )


    print(
        "Train:",
        len(train_df)
    )

    print(
        "Validation:",
        len(validation_df)
    )

    print(
        "Test:",
        len(test_df)
    )


    ## Create training matrices

    X_train = (
        train_df[
            RF_FEATURES
        ]
        .to_numpy()
    )


    y_train = (
        train_df[
            "Target_Squared_Return"
        ]
        .to_numpy()
    )


    X_validation = (
        validation_df[
            RF_FEATURES
        ]
        .to_numpy()
    )


    y_validation = (
        validation_df[
            "Target_Squared_Return"
        ]
        .to_numpy()
    )


    ## Hyperparameter search

    candidate_number = 0


    for (
        max_depth,
        min_samples_leaf,
        max_features
    ) in product(

        MAX_DEPTH_VALUES,
        MIN_SAMPLES_LEAF_VALUES,
        MAX_FEATURES_VALUES

    ):

        candidate_number += 1


        model_name = (
            "RF_"
            f"Depth{max_depth}_"
            f"Leaf{min_samples_leaf}_"
            f"Features{max_features}"
        )


        print(
            f"Testing candidate "
            f"{candidate_number}/18: "
            f"{model_name}"
        )


        ## Build model

        model = RandomForestRegressor(

            n_estimators=
                N_ESTIMATORS,

            criterion=
                "squared_error",

            max_depth=
                max_depth,

            min_samples_leaf=
                min_samples_leaf,

            max_features=
                max_features,

            bootstrap=
                True,

            random_state=
                RANDOM_STATE,

            n_jobs=
                -1
        )


        ## Fit training period only

        model.fit(
            X_train,
            y_train
        )


        ## Validation forecasts

        validation_prediction = (
            model.predict(
                X_validation
            )
        )


        # Variance predictions should not
        # be negative.
        # Random Forest averages terminal-node
        # target values, but numerical protection
        # is retained for QLIKE.

        validation_prediction = np.maximum(
            validation_prediction,
            EPSILON
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

                "N_Estimators":
                    N_ESTIMATORS,

                "Max_Depth":
                    (
                        "None"
                        if max_depth is None
                        else max_depth
                    ),

                "Min_Samples_Leaf":
                    min_samples_leaf,

                "Max_Features":
                    max_features,

                "Observations":
                    len(
                        validation_df
                    ),

                "RMSE":
                    metrics["RMSE"],

                "MAE":
                    metrics["MAE"],

                "QLIKE":
                    metrics["QLIKE"]
            }
        )


        ## Store validation forecasts

        candidate_forecasts = (
            validation_df[
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
        ] = model_name


        candidate_forecasts[
            "RF_Forecast"
        ] = validation_prediction


        validation_forecasts.append(
            candidate_forecasts
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


    ## Select best random forest

    ticker_results = pd.DataFrame(
        [
            x

            for x in validation_results

            if x["Ticker"] == ticker
        ]
    )


    # Primary criterion:
    # lowest validation QLIKE
    # RMSE then MAE are only used
    # as tie-breakers.

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


    # Convert selected depth
    # back to Python None if necessary.

    if (
        str(
            best["Max_Depth"]
        )
        == "None"
    ):

        best_depth = None

    else:

        best_depth = int(
            best["Max_Depth"]
        )


    best_leaf = int(
        best[
            "Min_Samples_Leaf"
        ]
    )


    best_features = float(
        best[
            "Max_Features"
        ]
    )


    print("\n")
    print("Selected Random Forest:")

    print(
        "Max depth:",
        best_depth
    )

    print(
        "Min samples leaf:",
        best_leaf
    )

    print(
        "Max features:",
        best_features
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

            "N_Estimators":
                N_ESTIMATORS,

            "Max_Depth":
                (
                    "None"
                    if best_depth is None
                    else best_depth
                ),

            "Min_Samples_Leaf":
                best_leaf,

            "Max_Features":
                best_features,

            "Validation_RMSE":
                best["RMSE"],

            "Validation_MAE":
                best["MAE"],

            "Validation_QLIKE":
                best["QLIKE"]
        }
    )


    ## Refit using train + validation

    train_validation_df = (
        asset_df[
            asset_df["Split"].isin(
                [
                    "Train",
                    "Validation"
                ]
            )
        ]
        .copy()
    )


    X_train_validation = (
        train_validation_df[
            RF_FEATURES
        ]
        .to_numpy()
    )


    y_train_validation = (
        train_validation_df[
            "Target_Squared_Return"
        ]
        .to_numpy()
    )


    final_model = RandomForestRegressor(

        n_estimators=
            N_ESTIMATORS,

        criterion=
            "squared_error",

        max_depth=
            best_depth,

        min_samples_leaf=
            best_leaf,

        max_features=
            best_features,

        bootstrap=
            True,

        random_state=
            RANDOM_STATE,

        n_jobs=
            -1
    )


    final_model.fit(

        X_train_validation,

        y_train_validation
    )


    ## Final test forecasts

    X_test = (
        test_df[
            RF_FEATURES
        ]
        .to_numpy()
    )


    y_test = (
        test_df[
            "Target_Squared_Return"
        ]
        .to_numpy()
    )


    test_prediction = (
        final_model.predict(
            X_test
        )
    )


    test_prediction = np.maximum(

        test_prediction,

        EPSILON
    )


    ## Final test metrics

    final_metrics = calculate_metrics(

        y_test,

        test_prediction
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
                "Random Forest",

            "Observations":
                len(test_df),

            "N_Estimators":
                N_ESTIMATORS,

            "Max_Depth":
                (
                    "None"
                    if best_depth is None
                    else best_depth
                ),

            "Min_Samples_Leaf":
                best_leaf,

            "Max_Features":
                best_features,

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
    print("FINAL TEST PERFORMANCE")

    print(
        "RMSE:",
        final_metrics["RMSE"]
    )

    print(
        "MAE:",
        final_metrics["MAE"]
    )

    print(
        "QLIKE:",
        final_metrics["QLIKE"]
    )


    ## Save individual test forecasts

    ticker_test_forecasts = (
        test_df[
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


    ticker_test_forecasts[
        "RF_Forecast"
    ] = test_prediction


    ticker_test_forecasts[
        "Model"
    ] = "Random Forest"


    safe_ticker = (
        ticker.replace(
            "^",
            ""
        )
    )


    ticker_test_forecasts.to_csv(

        OUTPUT_PATH /
        f"{safe_ticker}_rf_test_forecasts.csv",

        index=False
    )


    test_forecasts.append(
        ticker_test_forecasts
    )


    ## Feature importance

    importance = (
        final_model
        .feature_importances_
    )


    for feature, value in zip(
        RF_FEATURES,
        importance
    ):

        feature_importance_results.append(
            {
                "Ticker":
                    ticker,

                "Asset":
                    asset_name,

                "Feature":
                    feature,

                "Importance":
                    value
            }
        )


## Create output tables

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


feature_importance_df = (
    pd.DataFrame(
        feature_importance_results
    )
)


## Combine validation forecasts

combined_validation_forecasts = (
    pd.concat(
        validation_forecasts,
        ignore_index=True
    )
)


combined_validation_forecasts = (
    combined_validation_forecasts
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
)


combined_test_forecasts = (
    combined_test_forecasts
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


## Save output files

validation_results_df.to_csv(

    OUTPUT_PATH /
    "rf_validation_results.csv",

    index=False
)


selected_models_df.to_csv(

    OUTPUT_PATH /
    "rf_selected_models.csv",

    index=False
)


combined_validation_forecasts.to_csv(

    OUTPUT_PATH /
    "combined_rf_validation_forecasts.csv",

    index=False
)


combined_test_forecasts.to_csv(

    OUTPUT_PATH /
    "combined_rf_test_forecasts.csv",

    index=False
)


test_metrics_df.to_csv(

    OUTPUT_PATH /
    "rf_test_metrics.csv",

    index=False
)


feature_importance_df.to_csv(

    OUTPUT_PATH /
    "rf_feature_importance.csv",

    index=False
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
            "Random Forest"
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


stock_average.to_csv(

    OUTPUT_PATH /
    "rf_stock_average.csv",

    index=False
)


## Quality checks

EXPECTED_VALIDATION_ROWS = 501
EXPECTED_TEST_ROWS = 647


# Selected model count

if len(
    selected_models_df
) != 7:

    raise ValueError(
        "Expected one selected RF model "
        "for each of seven assets."
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
            f"test forecasts, found "
            f"{len(ticker_forecasts)}."
        )


# Validation forecast counts

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
        != 18
    ):

        raise ValueError(
            f"{ticker}: expected "
            "18 RF candidates."
        )


    if not (
        ticker_results[
            "Observations"
        ]
        == EXPECTED_VALIDATION_ROWS
    ).all():

        raise ValueError(
            f"{ticker}: incorrect "
            "validation observation count."
        )


# Positive forecasts

if (
    combined_test_forecasts[
        "RF_Forecast"
    ] <= 0
).any():

    raise ValueError(
        "Non-positive RF forecast detected."
    )


# Missing forecasts

if combined_test_forecasts[
    "RF_Forecast"
].isna().any():

    raise ValueError(
        "Missing RF forecasts detected."
    )


# Target chronology

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
        "Target-date leakage detected "
        "in RF forecasts."
    )


## Display results

print("\n")
print("=" * 100)
print("SELECTED RANDOM FOREST MODELS")
print("=" * 100)

print(
    selected_models_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("FINAL RANDOM FOREST TEST PERFORMANCE")
print("=" * 100)

print(
    test_metrics_df[
        [
            "Ticker",
            "Observations",
            "Max_Depth",
            "Min_Samples_Leaf",
            "Max_Features",
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
print("STOCK-ONLY RANDOM FOREST AVERAGE")
print("=" * 100)

print(
    stock_average.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("RANDOM FOREST MODELLING COMPLETE")
print("=" * 100)