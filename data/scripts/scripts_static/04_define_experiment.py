import pandas as pd
from pathlib import Path


## Paths

PROCESSED_DATA_PATH = Path("data/processed")

EXPERIMENT_PATH = Path("data/experiment")

EXPERIMENT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


## Assets

FILES = {
    "AAPL": "AAPL_processed.csv",
    "JPM": "JPM_processed.csv",
    "XOM": "XOM_processed.csv",
    "JNJ": "JNJ_processed.csv",
    "WMT": "WMT_processed.csv",
    "TSLA": "TSLA_processed.csv",
    "^GSPC": "GSPC_processed.csv"
}


## Experiment dates

TRAIN_END = pd.Timestamp("2021-12-31")

VALIDATION_START = pd.Timestamp("2022-01-01")
VALIDATION_END = pd.Timestamp("2023-12-31")

TEST_START = pd.Timestamp("2024-01-01")


## Storage

all_datasets = []
split_summary = []


## Process each asset

for ticker, filename in FILES.items():

    print(f"\nPreparing experiment dataset for {ticker}...")

    df = pd.read_csv(
        PROCESSED_DATA_PATH / filename
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.sort_values(
        "Date"
    ).reset_index(drop=True)


    ## Define next-day target

    # The current row contains information
    # available at trading day t.
    # The target is the squared return
    # observed on the next trading day t+1.

    df["Target_Date"] = (
        df["Date"].shift(-1)
    )

    df["Target_Squared_Return"] = (
        df["Squared_Return"].shift(-1)
    )


    ## Drop final row

    # The final observation cannot have a
    # next-day target because no future
    # observation exists in the dataset.

    df = df.dropna(
        subset=[
            "Target_Date",
            "Target_Squared_Return"
        ]
    ).reset_index(drop=True)


    ## Define dataset role

    if ticker == "^GSPC":
        df["Role"] = "Benchmark"
    else:
        df["Role"] = "Stock"


    ## Define forecast horizon

    df["Forecast_Horizon"] = 1


    ## Assign train / validation / test

    df["Split"] = None

    df.loc[
        df["Target_Date"] <= TRAIN_END,
        "Split"
    ] = "Train"

    df.loc[
        (
            (df["Target_Date"] >= VALIDATION_START)
            &
            (df["Target_Date"] <= VALIDATION_END)
        ),
        "Split"
    ] = "Validation"

    df.loc[
        df["Target_Date"] >= TEST_START,
        "Split"
    ] = "Test"


    ## Validation checks

    if df["Split"].isna().any():
        raise ValueError(
            f"{ticker}: some observations "
            "were not assigned to a split."
        )


    if not (
        df["Target_Date"] > df["Date"]
    ).all():

        raise ValueError(
            f"{ticker}: target-date leakage detected."
        )


    if (
        df["Target_Squared_Return"] < 0
    ).any():

        raise ValueError(
            f"{ticker}: negative squared "
            "return target detected."
        )


    ## Create split summary

    for split_name in [
        "Train",
        "Validation",
        "Test"
    ]:

        split_df = df[
            df["Split"] == split_name
        ]

        split_summary.append(
            {
                "Ticker": ticker,
                "Role": (
                    "Benchmark"
                    if ticker == "^GSPC"
                    else "Stock"
                ),
                "Split": split_name,
                "Observations": len(split_df),

                "First_Input_Date": (
                    split_df["Date"].min()
                ),

                "Last_Input_Date": (
                    split_df["Date"].max()
                ),

                "First_Target_Date": (
                    split_df["Target_Date"].min()
                ),

                "Last_Target_Date": (
                    split_df["Target_Date"].max()
                )
            }
        )


    ## Save individual experiment dataset

    safe_ticker = ticker.replace("^", "")

    output_file = (
        EXPERIMENT_PATH /
        f"{safe_ticker}_experiment.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    all_datasets.append(df)

    print(
        df["Split"]
        .value_counts()
        .to_string()
    )


## Combine all assets

combined = pd.concat(
    all_datasets,
    ignore_index=True
)

combined = combined.sort_values(
    ["Ticker", "Date"]
).reset_index(drop=True)

combined.to_csv(
    EXPERIMENT_PATH /
    "combined_experiment_data.csv",
    index=False
)


## Save split summary

summary_df = pd.DataFrame(
    split_summary
)

summary_df.to_csv(
    EXPERIMENT_PATH /
    "experiment_split_summary.csv",
    index=False
)


## Print summary

print("\n")
print("=" * 100)
print("EXPERIMENT SPLIT SUMMARY")
print("=" * 100)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nExperiment dataset preparation complete.")