import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import (
    skew,
    kurtosis,
    jarque_bera
)

from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)
from statsmodels.graphics.tsaplots import plot_acf


## Paths

PROCESSED_DATA_PATH = Path("data/processed")
OUTPUT_PATH = Path("outputs/eda")

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


## Files

FILES = {
    "AAPL": "AAPL_processed.csv",
    "JPM": "JPM_processed.csv",
    "XOM": "XOM_processed.csv",
    "JNJ": "JNJ_processed.csv",
    "WMT": "WMT_processed.csv",
    "TSLA": "TSLA_processed.csv",
    "^GSPC": "GSPC_processed.csv"
}


## Storage for results

summary_results = []
diagnostic_results = []


## Process each series

for ticker, filename in FILES.items():

    print(f"\nAnalysing {ticker}...")

    df = pd.read_csv(
        PROCESSED_DATA_PATH / filename
    )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date").reset_index(drop=True)

    returns = df["Log_Return"].dropna()
    squared_returns = df["Squared_Return"].dropna()


    ## Descriptive statistics

    jb_result = jarque_bera(returns)

    summary_results.append(
        {
            "Ticker": ticker,
            "Observations": len(returns),

            "Mean_Return": returns.mean(),
            "Median_Return": returns.median(),

            "Std_Dev": returns.std(),

            "Minimum": returns.min(),
            "Maximum": returns.max(),

            "Skewness": skew(
                returns,
                bias=False
            ),

            "Excess_Kurtosis": kurtosis(
                returns,
                fisher=True,
                bias=False
            ),

            "Jarque_Bera_Statistic": jb_result.statistic,
            "Jarque_Bera_p_value": jb_result.pvalue
        }
    )


    ## Augmented dickey-fuller test

    adf_result = adfuller(
        returns,
        autolag="AIC"
    )

    adf_statistic = adf_result[0]
    adf_p_value = adf_result[1]


    ## Ljung-box test: returns

    lb_returns = acorr_ljungbox(
        returns,
        lags=[5, 10, 20],
        return_df=True
    )


    ## Ljung-box test: squared returns

    lb_squared = acorr_ljungbox(
        squared_returns,
        lags=[5, 10, 20],
        return_df=True
    )


    ## ARCH-lm test

    arch_result = het_arch(
        returns,
        nlags=10
    )

    arch_lm_stat = arch_result[0]
    arch_lm_p = arch_result[1]


    ## Save diagnostic results

    diagnostic_results.append(
        {
            "Ticker": ticker,

            "ADF_Statistic": adf_statistic,
            "ADF_p_value": adf_p_value,

            "LB_Returns_Lag5_p": (
                lb_returns.loc[5, "lb_pvalue"]
            ),

            "LB_Returns_Lag10_p": (
                lb_returns.loc[10, "lb_pvalue"]
            ),

            "LB_Returns_Lag20_p": (
                lb_returns.loc[20, "lb_pvalue"]
            ),

            "LB_Squared_Lag5_p": (
                lb_squared.loc[5, "lb_pvalue"]
            ),

            "LB_Squared_Lag10_p": (
                lb_squared.loc[10, "lb_pvalue"]
            ),

            "LB_Squared_Lag20_p": (
                lb_squared.loc[20, "lb_pvalue"]
            ),

            "ARCH_LM_Statistic": arch_lm_stat,
            "ARCH_LM_p_value": arch_lm_p
        }
    )


    ## Plots

    safe_ticker = ticker.replace("^", "")

    ticker_output = OUTPUT_PATH / safe_ticker
    ticker_output.mkdir(
        parents=True,
        exist_ok=True
    )


    ## Adjusted price

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Date"],
        df["Adj_Close"]
    )

    plt.title(
        f"{ticker} Adjusted Closing Price"
    )

    plt.xlabel("Date")
    plt.ylabel("Adjusted Price")

    plt.tight_layout()

    plt.savefig(
        ticker_output /
        "01_adjusted_price.png",
        dpi=300
    )

    plt.close()


    ## Daily log returns

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Date"],
        df["Log_Return"],
        linewidth=0.6
    )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.title(
        f"{ticker} Daily Log Returns"
    )

    plt.xlabel("Date")
    plt.ylabel("Log Return (%)")

    plt.tight_layout()

    plt.savefig(
        ticker_output /
        "02_log_returns.png",
        dpi=300
    )

    plt.close()


    ## Squared returns

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Date"],
        df["Squared_Return"],
        linewidth=0.6
    )

    plt.title(
        f"{ticker} Squared Daily Returns"
    )

    plt.xlabel("Date")
    plt.ylabel("Squared Return")

    plt.tight_layout()

    plt.savefig(
        ticker_output /
        "03_squared_returns.png",
        dpi=300
    )

    plt.close()


    ## 21-day rolling volatility

    rolling_volatility = (
        df["Log_Return"]
        .rolling(window=21)
        .std()
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        df["Date"],
        rolling_volatility
    )

    plt.title(
        f"{ticker} 21-Day Rolling Volatility"
    )

    plt.xlabel("Date")
    plt.ylabel("Rolling Standard Deviation (%)")

    plt.tight_layout()

    plt.savefig(
        ticker_output /
        "04_rolling_volatility.png",
        dpi=300
    )

    plt.close()


    ## Return histogram

    plt.figure(figsize=(8, 5))

    plt.hist(
        returns,
        bins=80
    )

    plt.title(
        f"{ticker} Distribution of Daily Returns"
    )

    plt.xlabel("Log Return (%)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        ticker_output /
        "05_return_distribution.png",
        dpi=300
    )

    plt.close()


    ## ACF of returns

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    plot_acf(
        returns,
        lags=40,
        zero=False,
        ax=ax
    )

    ax.set_title(
        f"{ticker} ACF of Daily Returns"
    )

    plt.tight_layout()

    plt.savefig(
        ticker_output /
        "06_acf_returns.png",
        dpi=300
    )

    plt.close()


    ## ACF of squared returns

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    plot_acf(
        squared_returns,
        lags=40,
        zero=False,
        ax=ax
    )

    ax.set_title(
        f"{ticker} ACF of Squared Returns"
    )

    plt.tight_layout()

    plt.savefig(
        ticker_output /
        "07_acf_squared_returns.png",
        dpi=300
    )

    plt.close()


## Create results tables

summary_df = pd.DataFrame(
    summary_results
)

diagnostics_df = pd.DataFrame(
    diagnostic_results
)


## Save results

summary_df.to_csv(
    OUTPUT_PATH /
    "descriptive_statistics.csv",
    index=False
)

diagnostics_df.to_csv(
    OUTPUT_PATH /
    "diagnostic_tests.csv",
    index=False
)


## Print results

print("\n")
print("=" * 80)
print("DESCRIPTIVE STATISTICS")
print("=" * 80)

print(
    summary_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 80)
print("TIME-SERIES DIAGNOSTICS")
print("=" * 80)

print(
    diagnostics_df.to_string(
        index=False
    )
)

print("\nEDA and diagnostic analysis complete.")