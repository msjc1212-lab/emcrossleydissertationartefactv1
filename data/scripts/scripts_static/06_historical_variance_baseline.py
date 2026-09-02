import pandas as pd
import numpy as np
from pathlib import Path


## Paths

FEATURE_PATH = Path("data/features")
OUTPUT_PATH = Path("outputs/baseline")

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


## Settings

INPUT_FILE = (
    FEATURE_PATH /
    "combined_lstm_source.csv"
)

WINDOW = 21

# Small positive value used only to ensure that
# variance forecasts remain valid for QLIKE.
EPSILON = 1e-8


## Load data

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df["Target_Date"] = pd.to_datetime(
    df["Target_Date"]
)

df = df.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)


## Storage

forecast_datasets = []
metric_results = []


## Process each asset

for ticker in df["Ticker"].unique():

    print("\n" + "=" * 70)
    print(f"Historical variance baseline: {ticker}")
    print("=" * 70)

    asset_df = (
        df[df["Ticker"] == ticker]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )


    ## 21-day historical variance

    # Uses returns available through day t.
    # This becomes the forecast of variance
    # for trading day t+1.

    asset_df[
        "Historical_Variance_Forecast"
    ] = (
        asset_df["Log_Return"]
        .rolling(
            window=WINDOW,
            min_periods=WINDOW
        )
        .var(ddof=1)
    )


    ## Remove initial rows without 21 days

    asset_df = asset_df.dropna(
        subset=[
            "Historical_Variance_Forecast"
        ]
    ).reset_index(drop=True)


    ## Ensure positive forecast

    asset_df[
        "Historical_Variance_Forecast"
    ] = np.maximum(
        asset_df[
            "Historical_Variance_Forecast"
        ],
        EPSILON
    )


    ## Save validation + test forecasts

    evaluation_df = asset_df[
        asset_df["Split"].isin(
            ["Validation", "Test"]
        )
    ].copy()


    evaluation_df = evaluation_df[
        [
            "Date",
            "Target_Date",
            "Ticker",
            "Asset",
            "Role",
            "Split",
            "Target_Squared_Return",
            "Historical_Variance_Forecast"
        ]
    ]


    safe_ticker = ticker.replace("^", "")

    evaluation_df.to_csv(
        OUTPUT_PATH /
        f"{safe_ticker}_historical_variance_forecasts.csv",
        index=False
    )

    forecast_datasets.append(
        evaluation_df
    )


    ## Calculate metrics

    for split_name in [
        "Validation",
        "Test"
    ]:

        split_df = evaluation_df[
            evaluation_df["Split"]
            == split_name
        ].copy()

        actual = (
            split_df[
                "Target_Squared_Return"
            ]
            .to_numpy()
        )

        forecast = (
            split_df[
                "Historical_Variance_Forecast"
            ]
            .to_numpy()
        )


        ## Forecast errors

        error = forecast - actual


        ## RMSE

        rmse = np.sqrt(
            np.mean(
                error ** 2
            )
        )


        ## MAE

        mae = np.mean(
            np.abs(error)
        )


        ## QLIKE

        # Gaussian quasi-likelihood loss:
        # log(forecast variance)
        # +
        # actual variance / forecast variance
        # Lower values indicate better forecasts.

        qlike_losses = (
            np.log(forecast)
            +
            actual / forecast
        )

        qlike = np.mean(
            qlike_losses
        )


        ## Store results

        metric_results.append(
            {
                "Ticker": ticker,

                "Role": (
                    split_df[
                        "Role"
                    ].iloc[0]
                ),

                "Model":
                    "Historical Variance (21-day)",

                "Split":
                    split_name,

                "Observations":
                    len(split_df),

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "QLIKE":
                    qlike,

                "Mean_Actual_Variance":
                    np.mean(actual),

                "Mean_Forecast_Variance":
                    np.mean(forecast)
            }
        )


    print(
        "Validation forecasts:",
        (
            evaluation_df["Split"]
            == "Validation"
        ).sum()
    )

    print(
        "Test forecasts:",
        (
            evaluation_df["Split"]
            == "Test"
        ).sum()
    )


## Combine all forecasts

all_forecasts = pd.concat(
    forecast_datasets,
    ignore_index=True
)

all_forecasts = all_forecasts.sort_values(
    [
        "Ticker",
        "Target_Date"
    ]
).reset_index(drop=True)


all_forecasts.to_csv(
    OUTPUT_PATH /
    "combined_historical_variance_forecasts.csv",
    index=False
)


## Metrics table

metrics_df = pd.DataFrame(
    metric_results
)

metrics_df.to_csv(
    OUTPUT_PATH /
    "historical_variance_metrics.csv",
    index=False
)


## Stock-only average performance

stock_metrics = metrics_df[
    metrics_df["Role"] == "Stock"
].copy()


stock_average = (
    stock_metrics
    .groupby(
        ["Model", "Split"],
        as_index=False
    )
    .agg(
        Mean_RMSE=("RMSE", "mean"),
        Mean_MAE=("MAE", "mean"),
        Mean_QLIKE=("QLIKE", "mean"),
        Number_of_Stocks=("Ticker", "nunique")
    )
)


stock_average.to_csv(
    OUTPUT_PATH /
    "historical_variance_stock_average.csv",
    index=False
)


## Quality checks

if (
    all_forecasts[
        "Historical_Variance_Forecast"
    ] <= 0
).any():

    raise ValueError(
        "Non-positive variance forecasts detected."
    )


if all_forecasts[
    "Target_Squared_Return"
].isna().any():

    raise ValueError(
        "Missing targets detected."
    )


## Display results

print("\n")
print("=" * 100)
print("HISTORICAL VARIANCE BASELINE METRICS")
print("=" * 100)

print(
    metrics_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("STOCK-ONLY AVERAGE")
print("=" * 100)

print(
    stock_average.to_string(
        index=False
    )
)

print(
    "\nHistorical variance baseline complete."
)