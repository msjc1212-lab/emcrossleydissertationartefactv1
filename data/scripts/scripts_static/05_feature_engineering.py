import pandas as pd
import numpy as np
from pathlib import Path


## Paths

EXPERIMENT_PATH = Path("data/experiment")
FEATURE_PATH = Path("data/features")

FEATURE_PATH.mkdir(
    parents=True,
    exist_ok=True
)


## Assets

FILES = {
    "AAPL": "AAPL_experiment.csv",
    "JPM": "JPM_experiment.csv",
    "XOM": "XOM_experiment.csv",
    "JNJ": "JNJ_experiment.csv",
    "WMT": "WMT_experiment.csv",
    "TSLA": "TSLA_experiment.csv",
    "^GSPC": "GSPC_experiment.csv"
}


## Feature settings

# Lag numbers are defined relative to the target.
# Lag 1 = current trading day's value (t)
# Lag 2 = previous trading day's value (t-1)
# etc.

LAGS = [
    1,
    2,
    3,
    5,
    10,
    21
]


# Historical rolling-volatility windows

VOL_WINDOWS = [
    5,
    10,
    21,
    63
]


## Storage

combined_rf = []
combined_garch = []
combined_lstm = []

quality_results = []


## Process each asset

for ticker, filename in FILES.items():

    print("\n" + "=" * 70)
    print(f"Feature engineering: {ticker}")
    print("=" * 70)

    df = pd.read_csv(
        EXPERIMENT_PATH / filename
    )


    ## Date conversion

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df["Target_Date"] = pd.to_datetime(
        df["Target_Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)


    ## Basic leakage check

    if not (
        df["Target_Date"] > df["Date"]
    ).all():

        raise ValueError(
            f"{ticker}: target-date leakage detected."
        )


    ## Random forest features

    ## Return lags

    for lag in LAGS:

        # Because Lag1 represents today's return,
        # the required pandas shift is lag - 1.

        shift_amount = lag - 1

        df[
            f"Return_Lag{lag}"
        ] = df[
            "Log_Return"
        ].shift(
            shift_amount
        )


    ## Squared-return lags

    for lag in LAGS:

        shift_amount = lag - 1

        df[
            f"Squared_Return_Lag{lag}"
        ] = df[
            "Squared_Return"
        ].shift(
            shift_amount
        )


    ## Historical rolling volatility

    for window in VOL_WINDOWS:

        df[
            f"Rolling_Vol_{window}"
        ] = (
            df["Log_Return"]
            .rolling(
                window=window
            )
            .std()
        )


    ## Define random forest feature list

    rf_feature_columns = (

        [
            f"Return_Lag{lag}"
            for lag in LAGS
        ]

        +

        [
            f"Squared_Return_Lag{lag}"
            for lag in LAGS
        ]

        +

        [
            f"Rolling_Vol_{window}"
            for window in VOL_WINDOWS
        ]
    )


    ## Create random forest dataset

    rf_columns = [
        "Date",
        "Target_Date",
        "Ticker",
        "Asset",
        "Role",
        "Split",

        *rf_feature_columns,

        "Target_Squared_Return"
    ]


    rf_df = df[
        rf_columns
    ].copy()


    ## Remove early rows without enough history

    # The 63-day rolling volatility feature requires
    # at least 63 historical observations.
    # These missing rows occur only at the beginning
    # of the sample and are NOT imputed.

    rf_df = rf_df.dropna(
        subset=rf_feature_columns
    ).reset_index(drop=True)


    ## Random forest validation

    if rf_df[
        rf_feature_columns
    ].isna().any().any():

        raise ValueError(
            f"{ticker}: missing RF features remain."
        )


    if (
        rf_df["Target_Squared_Return"] < 0
    ).any():

        raise ValueError(
            f"{ticker}: invalid negative target."
        )


    ## Save random forest dataset

    safe_ticker = ticker.replace("^", "")

    rf_df.to_csv(
        FEATURE_PATH /
        f"{safe_ticker}_rf_features.csv",
        index=False
    )

    combined_rf.append(
        rf_df
    )


    ## Create GARCH input dataset

    # GARCH does not need artificially engineered
    # lag features.
    # It receives the chronological return series.

    garch_df = df[
        [
            "Date",
            "Target_Date",
            "Ticker",
            "Asset",
            "Role",
            "Split",
            "Log_Return",
            "Target_Squared_Return"
        ]
    ].copy()


    garch_df.to_csv(
        FEATURE_PATH /
        f"{safe_ticker}_garch_input.csv",
        index=False
    )

    combined_garch.append(
        garch_df
    )


    ## Create LSTM source dataset

    # We do NOT create fixed sequences yet.
    # Sequence length will be tuned later using the
    # validation period.

    lstm_df = df[
        [
            "Date",
            "Target_Date",
            "Ticker",
            "Asset",
            "Role",
            "Split",
            "Log_Return",
            "Squared_Return",
            "Target_Squared_Return"
        ]
    ].copy()


    lstm_df.to_csv(
        FEATURE_PATH /
        f"{safe_ticker}_lstm_source.csv",
        index=False
    )

    combined_lstm.append(
        lstm_df
    )


    ## Quality summary

    quality_results.append(
        {
            "Ticker": ticker,

            "RF_Features": len(
                rf_feature_columns
            ),

            "RF_Total_Rows": len(
                rf_df
            ),

            "RF_Train_Rows": (
                rf_df["Split"]
                == "Train"
            ).sum(),

            "RF_Validation_Rows": (
                rf_df["Split"]
                == "Validation"
            ).sum(),

            "RF_Test_Rows": (
                rf_df["Split"]
                == "Test"
            ).sum(),

            "GARCH_Rows": len(
                garch_df
            ),

            "LSTM_Source_Rows": len(
                lstm_df
            ),

            "RF_Missing_Features": (
                rf_df[
                    rf_feature_columns
                ]
                .isna()
                .sum()
                .sum()
            ),

            "Negative_Targets": (
                df[
                    "Target_Squared_Return"
                ] < 0
            ).sum()
        }
    )


    print(
        f"RF observations: {len(rf_df)}"
    )

    print(
        f"GARCH observations: {len(garch_df)}"
    )

    print(
        f"LSTM source observations: {len(lstm_df)}"
    )


## Combined random forest dataset

combined_rf_df = pd.concat(
    combined_rf,
    ignore_index=True
)

combined_rf_df.to_csv(
    FEATURE_PATH /
    "combined_rf_features.csv",
    index=False
)


## Combined GARCH dataset

combined_garch_df = pd.concat(
    combined_garch,
    ignore_index=True
)

combined_garch_df.to_csv(
    FEATURE_PATH /
    "combined_garch_input.csv",
    index=False
)


## Combined LSTM source dataset

combined_lstm_df = pd.concat(
    combined_lstm,
    ignore_index=True
)

combined_lstm_df.to_csv(
    FEATURE_PATH /
    "combined_lstm_source.csv",
    index=False
)


## Quality summary

quality_df = pd.DataFrame(
    quality_results
)

quality_df.to_csv(
    FEATURE_PATH /
    "feature_quality_summary.csv",
    index=False
)


## Feature dictionary

feature_dictionary = []


for lag in LAGS:

    feature_dictionary.append(
        {
            "Feature": f"Return_Lag{lag}",
            "Model": "Random Forest",
            "Description":
                f"Daily log return at target lag {lag}"
        }
    )


for lag in LAGS:

    feature_dictionary.append(
        {
            "Feature":
                f"Squared_Return_Lag{lag}",

            "Model":
                "Random Forest",

            "Description":
                f"Squared daily return at target lag {lag}"
        }
    )


for window in VOL_WINDOWS:

    feature_dictionary.append(
        {
            "Feature":
                f"Rolling_Vol_{window}",

            "Model":
                "Random Forest",

            "Description":
                f"Historical {window}-trading-day "
                "standard deviation of log returns"
        }
    )


feature_dictionary_df = pd.DataFrame(
    feature_dictionary
)

feature_dictionary_df.to_csv(
    FEATURE_PATH /
    "rf_feature_dictionary.csv",
    index=False
)


## Display summary

print("\n")
print("=" * 100)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 100)

print(
    quality_df.to_string(
        index=False
    )
)

print(
    "\nRandom Forest feature count:",
    len(rf_feature_columns)
)