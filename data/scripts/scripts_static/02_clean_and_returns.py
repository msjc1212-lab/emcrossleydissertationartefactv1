import pandas as pd
import numpy as np
from pathlib import Path


## Paths

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)


## Assets

FILES = {
    "AAPL": "AAPL_raw.csv",
    "JPM": "JPM_raw.csv",
    "XOM": "XOM_raw.csv",
    "JNJ": "JNJ_raw.csv",
    "WMT": "WMT_raw.csv",
    "TSLA": "TSLA_raw.csv",
    "^GSPC": "GSPC_raw.csv"
}


## Storage

processed_datasets = []
quality_summary = []


## Process each asset

for ticker, filename in FILES.items():

    print(f"\nProcessing {ticker}...")

    file_path = RAW_DATA_PATH / filename

    # Load raw data
    df = pd.read_csv(file_path)


    ## Date cleaning

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)


    ## Validation

    duplicate_dates = df["Date"].duplicated().sum()

    missing_adj_close = df["Adj Close"].isna().sum()

    invalid_prices = (df["Adj Close"] <= 0).sum()


    if duplicate_dates > 0:
        raise ValueError(
            f"{ticker}: duplicate dates detected."
        )

    if missing_adj_close > 0:
        raise ValueError(
            f"{ticker}: missing adjusted prices detected."
        )

    if invalid_prices > 0:
        raise ValueError(
            f"{ticker}: zero or negative adjusted prices detected."
        )


    ## Daily log returns

    df["Log_Return"] = (
        100 *
        np.log(
            df["Adj Close"] /
            df["Adj Close"].shift(1)
        )
    )


    ## Squared returns

    df["Squared_Return"] = (
        df["Log_Return"] ** 2
    )


    ## Drop first observation

    # The first observation has no previous trading
    # day from which a return can be calculated.

    df = df.dropna(
        subset=["Log_Return"]
    ).reset_index(drop=True)


    ## Rename adjusted price

    df = df.rename(
        columns={
            "Adj Close": "Adj_Close",
            "Stock Splits": "Stock_Splits"
        }
    )


    ## Select variables

    df = df[
        [
            "Date",
            "Ticker",
            "Asset",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj_Close",
            "Volume",
            "Dividends",
            "Stock_Splits",
            "Log_Return",
            "Squared_Return"
        ]
    ]


    ## Quality summary

    quality_summary.append(
        {
            "Ticker": ticker,
            "Start_Date": df["Date"].min(),
            "End_Date": df["Date"].max(),
            "Observations": len(df),
            "Missing_Values": df.isna().sum().sum(),
            "Duplicate_Dates": df["Date"].duplicated().sum(),
            "Mean_Return": df["Log_Return"].mean(),
            "Return_Std_Dev": df["Log_Return"].std(),
            "Min_Return": df["Log_Return"].min(),
            "Max_Return": df["Log_Return"].max(),
            "Mean_Squared_Return": df["Squared_Return"].mean()
        }
    )


    ## Save individual dataset

    safe_ticker = ticker.replace("^", "")

    output_file = (
        PROCESSED_DATA_PATH /
        f"{safe_ticker}_processed.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    processed_datasets.append(df)

    print(
        f"{ticker}: "
        f"{len(df)} return observations saved."
    )


## Create combined dataset

combined = pd.concat(
    processed_datasets,
    ignore_index=True
)

combined = combined.sort_values(
    ["Date", "Ticker"]
).reset_index(drop=True)

combined.to_csv(
    PROCESSED_DATA_PATH /
    "combined_stock_data.csv",
    index=False
)


## Save data quality summary

quality_df = pd.DataFrame(
    quality_summary
)

quality_df.to_csv(
    PROCESSED_DATA_PATH /
    "data_quality_summary.csv",
    index=False
)


## Display summary

print("\n--------------------------------")
print("DATA PROCESSING COMPLETE")
print("--------------------------------\n")

print(quality_df.to_string(index=False))

print(
    "\nCombined observations:",
    len(combined)
)