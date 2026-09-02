import yfinance as yf
import pandas as pd
from pathlib import Path


## Settings

START_DATE = "2010-06-29"

# yfinance treats the end date as EXCLUSIVE.
# Therefore 2026-08-01 includes observations
# through 2026-07-31.
END_DATE = "2026-08-01"


# Individual stocks used in the modelling analysis
STOCKS = {
    "AAPL": "Apple",
    "JPM": "JPMorgan Chase",
    "XOM": "Exxon Mobil",
    "JNJ": "Johnson & Johnson",
    "WMT": "Walmart",
    "TSLA": "Tesla"
}

# Market benchmark
BENCHMARK = {
    "^GSPC": "S&P 500"
}


## Create output folder

RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)


## Combine tickers

ASSETS = {**STOCKS, **BENCHMARK}


## Download each asset

for ticker, company_name in ASSETS.items():

    print(f"Downloading {company_name} ({ticker})...")

    df = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",

        # Keep original and adjusted prices separately.
        auto_adjust=False,

        # Retain dividends and stock split information.
        actions=True,

        # Do not automatically alter suspicious values.
        repair=False,

        progress=False
    )

    # Flatten columns if yfinance returns a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Reset date from index to normal column
    df = df.reset_index()

    # Add identifying information
    df["Ticker"] = ticker
    df["Asset"] = company_name

    # Save one untouched CSV per asset
    safe_ticker = ticker.replace("^", "")

    filename = RAW_DATA_PATH / f"{safe_ticker}_raw.csv"

    df.to_csv(filename, index=False)

    print(
        f"Saved {len(df)} observations "
        f"to {filename}"
    )


print("\nRaw data download complete.")