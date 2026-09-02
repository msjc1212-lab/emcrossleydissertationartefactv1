import pandas as pd
import numpy as np

from pathlib import Path

from arch import arch_model

from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    het_arch
)


## Paths

FEATURE_PATH = Path("data/features")
OUTPUT_PATH = Path("outputs/garch")

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_FILE = (
    FEATURE_PATH /
    "combined_garch_input.csv"
)


## Settings

EPSILON = 1e-8


# Candidate GARCH specifications
# p = ARCH order
# q = GARCH order
# Both Normal and Student-t innovation
# distributions are compared.

CANDIDATES = [

    {
        "Model": "GARCH(1,1)-Normal",
        "p": 1,
        "q": 1,
        "dist": "normal"
    },

    {
        "Model": "GARCH(1,1)-Student-t",
        "p": 1,
        "q": 1,
        "dist": "t"
    },

    {
        "Model": "GARCH(1,2)-Normal",
        "p": 1,
        "q": 2,
        "dist": "normal"
    },

    {
        "Model": "GARCH(1,2)-Student-t",
        "p": 1,
        "q": 2,
        "dist": "t"
    },

    {
        "Model": "GARCH(2,1)-Normal",
        "p": 2,
        "q": 1,
        "dist": "normal"
    },

    {
        "Model": "GARCH(2,1)-Student-t",
        "p": 2,
        "q": 1,
        "dist": "t"
    }
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


## Storage

validation_results = []

validation_forecasts = []

selected_models = []

test_forecasts = []

test_metrics = []

parameter_results = []

diagnostic_results = []


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


    # Variance forecasts must remain positive
    # for QLIKE.

    forecast = np.maximum(
        forecast,
        EPSILON
    )


    # Forecast error

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
        np.abs(error)
    )


    ## QLIKE

    qlike_losses = (
        np.log(forecast)
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


## Process each asset

for ticker in df["Ticker"].unique():

    print("\n")
    print("=" * 80)
    print(f"GARCH MODELLING: {ticker}")
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
        asset_df["Asset"].iloc[0]
    )

    role = (
        asset_df["Role"].iloc[0]
    )


    ## Create return series

    returns = (
        asset_df
        .set_index("Date")[
            "Log_Return"
        ]
        .astype(float)
    )


    ## Identify validation period

    validation_df = (
        asset_df[
            asset_df["Split"]
            == "Validation"
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )


    first_validation_origin = (
        validation_df[
            "Date"
        ].min()
    )

    first_validation_target = (
        validation_df[
            "Target_Date"
        ].min()
    )


    ## Validate each candidate model

    for candidate in CANDIDATES:

        model_name = (
            candidate["Model"]
        )

        p = (
            candidate["p"]
        )

        q = (
            candidate["q"]
        )

        dist = (
            candidate["dist"]
        )


        print(
            f"Testing {model_name}..."
        )


        try:

            ## Create model

            am = arch_model(

                returns,

                mean="Constant",

                vol="GARCH",

                p=p,

                o=0,

                q=q,

                dist=dist,

                # Returns are already percentage
                # returns rather than decimals.

                rescale=False
            )


            ## Fit using training period

            # last_obs follows Python slicing logic.
            # first_validation_target is therefore
            # excluded from parameter estimation.
            # This means the model is estimated using
            # data available before validation begins.

            result = am.fit(

                last_obs=
                    first_validation_target,

                disp="off",

                update_freq=0,

                show_warning=False
            )


            ## Convergence check

            convergence_flag = (
                result.convergence_flag
            )


            ## One-step validation forecasts

            forecasts = result.forecast(

                horizon=1,

                start=
                    first_validation_origin,

                align="origin",

                method="analytic",

                reindex=True
            )


            ## Corrected for ARCH forecast object
            # result.forecast() returns an
            # ARCHModelForecast object.
            # Variance forecasts are therefore
            # accessed using:
            # forecasts.variance
            # NOT:
            # forecasts["variance"]

            variance_forecasts = (
                forecasts
                .variance["h.1"]
            )


            ## Align forecast with target

            candidate_validation = (
                validation_df.copy()
            )


            candidate_validation[
                "GARCH_Forecast"
            ] = (
                variance_forecasts
                .reindex(
                    candidate_validation[
                        "Date"
                    ]
                )
                .to_numpy()
            )


            ## Forecast quality checks

            if candidate_validation[
                "GARCH_Forecast"
            ].isna().any():

                raise ValueError(
                    "Missing validation forecasts."
                )


            if (
                candidate_validation[
                    "GARCH_Forecast"
                ] <= 0
            ).any():

                raise ValueError(
                    "Non-positive variance forecast."
                )


            ## Calculate validation metrics

            metrics = calculate_metrics(

                candidate_validation[
                    "Target_Squared_Return"
                ],

                candidate_validation[
                    "GARCH_Forecast"
                ]
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

                    "p":
                        p,

                    "q":
                        q,

                    "Distribution":
                        dist,

                    "Observations":
                        len(
                            candidate_validation
                        ),

                    "RMSE":
                        metrics["RMSE"],

                    "MAE":
                        metrics["MAE"],

                    "QLIKE":
                        metrics["QLIKE"],

                    "AIC":
                        result.aic,

                    "BIC":
                        result.bic,

                    "Log_Likelihood":
                        result.loglikelihood,

                    "Estimation_Observations":
                        result.nobs,

                    "Convergence_Flag":
                        convergence_flag
                }
            )


            ## Store validation forecasts

            temp_forecast = (
                candidate_validation[
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


            temp_forecast[
                "Model"
            ] = model_name


            temp_forecast[
                "GARCH_Forecast"
            ] = (
                candidate_validation[
                    "GARCH_Forecast"
                ]
            )


            validation_forecasts.append(
                temp_forecast
            )


            print(
                f"  QLIKE = "
                f"{metrics['QLIKE']:.6f}"
            )


        except Exception as e:

            print(
                f"FAILED: {model_name}"
            )

            print(
                f"Reason: {e}"
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

                    "p":
                        p,

                    "q":
                        q,

                    "Distribution":
                        dist,

                    "Observations":
                        np.nan,

                    "RMSE":
                        np.nan,

                    "MAE":
                        np.nan,

                    "QLIKE":
                        np.nan,

                    "AIC":
                        np.nan,

                    "BIC":
                        np.nan,

                    "Log_Likelihood":
                        np.nan,

                    "Estimation_Observations":
                        np.nan,

                    "Convergence_Flag":
                        np.nan
                }
            )


    ## Select best validation model

    ticker_results = pd.DataFrame(

        [
            x

            for x in validation_results

            if x["Ticker"] == ticker
        ]
    )


    # Only models with:
    ## Valid QLIKE
    ## Successful optimizer convergence
    # can be selected.

    valid_results = ticker_results[
        (
            ticker_results[
                "QLIKE"
            ].notna()
        )
        &
        (
            ticker_results[
                "Convergence_Flag"
            ] == 0
        )
    ].copy()


    if len(
        valid_results
    ) == 0:

        raise ValueError(
            f"{ticker}: no valid "
            "GARCH candidate converged."
        )


    # Primary selection metric:
    # lowest validation QLIKE
    # RMSE acts only as a tie-breaker.

    valid_results = (
        valid_results
        .sort_values(
            [
                "QLIKE",
                "RMSE"
            ]
        )
        .reset_index(
            drop=True
        )
    )


    best = (
        valid_results.iloc[0]
    )


    best_model_name = (
        best["Model"]
    )

    best_p = int(
        best["p"]
    )

    best_q = int(
        best["q"]
    )

    best_dist = (
        best["Distribution"]
    )


    print("\nSelected model:")
    print(best_model_name)

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

            "Selected_Model":
                best_model_name,

            "p":
                best_p,

            "q":
                best_q,

            "Distribution":
                best_dist,

            "Validation_RMSE":
                best["RMSE"],

            "Validation_MAE":
                best["MAE"],

            "Validation_QLIKE":
                best["QLIKE"]
        }
    )


    ## Define final test period

    test_df = (
        asset_df[
            asset_df["Split"]
            == "Test"
        ]
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )


    first_test_origin = (
        test_df[
            "Date"
        ].min()
    )

    first_test_target = (
        test_df[
            "Target_Date"
        ].min()
    )


    ## Build final selected model

    final_am = arch_model(

        returns,

        mean="Constant",

        vol="GARCH",

        p=best_p,

        o=0,

        q=best_q,

        dist=best_dist,

        rescale=False
    )


    ## Refit using train + validation

    # first_test_target is excluded.
    # Parameter estimation therefore uses
    # all information available before the
    # final test period begins.

    final_result = final_am.fit(

        last_obs=
            first_test_target,

        disp="off",

        update_freq=0,

        show_warning=False
    )


    if (
        final_result.convergence_flag
        != 0
    ):

        print(
            f"WARNING: final model for "
            f"{ticker} did not report "
            f"successful convergence."
        )


    ## Generate final test forecasts

    final_forecast_object = (
        final_result.forecast(

            horizon=1,

            start=
                first_test_origin,

            align="origin",

            method="analytic",

            reindex=True
        )
    )


    ## Corrected for ARCH forecast object

    final_variance_forecasts = (
        final_forecast_object
        .variance["h.1"]
    )


    ## Align test forecasts

    test_df[
        "GARCH_Forecast"
    ] = (
        final_variance_forecasts
        .reindex(
            test_df["Date"]
        )
        .to_numpy()
    )


    ## Test forecast checks

    if test_df[
        "GARCH_Forecast"
    ].isna().any():

        raise ValueError(
            f"{ticker}: missing "
            "test forecasts."
        )


    if (
        test_df[
            "GARCH_Forecast"
        ] <= 0
    ).any():

        raise ValueError(
            f"{ticker}: non-positive "
            "GARCH variance forecast."
        )


    # Numerical protection for QLIKE

    test_df[
        "GARCH_Forecast"
    ] = np.maximum(

        test_df[
            "GARCH_Forecast"
        ],

        EPSILON
    )


    ## Test metrics

    final_metrics = calculate_metrics(

        test_df[
            "Target_Squared_Return"
        ],

        test_df[
            "GARCH_Forecast"
        ]
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
                best_model_name,

            "Observations":
                len(
                    test_df
                ),

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
                test_df[
                    "Target_Squared_Return"
                ].mean(),

            "Mean_Forecast_Variance":
                test_df[
                    "GARCH_Forecast"
                ].mean(),

            "Convergence_Flag":
                final_result.convergence_flag,

            "AIC":
                final_result.aic,

            "BIC":
                final_result.bic
        }
    )


    ## Save test forecasts

    ticker_test = (
        test_df[
            [
                "Date",
                "Target_Date",
                "Ticker",
                "Asset",
                "Role",
                "Target_Squared_Return",
                "GARCH_Forecast"
            ]
        ]
        .copy()
    )


    ticker_test[
        "Model"
    ] = (
        best_model_name
    )


    safe_ticker = (
        ticker.replace(
            "^",
            ""
        )
    )


    ticker_test.to_csv(

        OUTPUT_PATH /
        f"{safe_ticker}_garch_test_forecasts.csv",

        index=False
    )


    test_forecasts.append(
        ticker_test
    )


    ## Save final model parameters

    params = (
        final_result.params
    )


    # GARCH persistence:
    # sum(alpha parameters)
    # +
    # sum(beta parameters)

    persistence = 0.0


    for parameter_name, value in params.items():

        parameter_results.append(
            {
                "Ticker":
                    ticker,

                "Asset":
                    asset_name,

                "Model":
                    best_model_name,

                "Parameter":
                    parameter_name,

                "Estimate":
                    value
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

            persistence += (
                value
            )


    parameter_results.append(
        {
            "Ticker":
                ticker,

            "Asset":
                asset_name,

            "Model":
                best_model_name,

            "Parameter":
                "Persistence",

            "Estimate":
                persistence
        }
    )


    ## Standardized residual diagnostics

    std_resid = (
        final_result
        .std_resid
        .dropna()
    )


    # Ljung-Box:
    # standardized residuals

    lb_resid = acorr_ljungbox(

        std_resid,

        lags=[
            10,
            20
        ],

        return_df=True
    )


    # Ljung-Box:
    # squared standardized residuals

    lb_squared = acorr_ljungbox(

        std_resid ** 2,

        lags=[
            10,
            20
        ],

        return_df=True
    )


    # ARCH-LM:
    # remaining conditional heteroskedasticity

    arch_test = het_arch(

        std_resid,

        nlags=10
    )


    diagnostic_results.append(
        {
            "Ticker":
                ticker,

            "Asset":
                asset_name,

            "Model":
                best_model_name,

            "LB_StdResid_Lag10_p":
                lb_resid.loc[
                    10,
                    "lb_pvalue"
                ],

            "LB_StdResid_Lag20_p":
                lb_resid.loc[
                    20,
                    "lb_pvalue"
                ],

            "LB_Squared_StdResid_Lag10_p":
                lb_squared.loc[
                    10,
                    "lb_pvalue"
                ],

            "LB_Squared_StdResid_Lag20_p":
                lb_squared.loc[
                    20,
                    "lb_pvalue"
                ],

            "ARCH_LM_Statistic":
                arch_test[0],

            "ARCH_LM_p_value":
                arch_test[1],

            "Persistence":
                persistence
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


parameter_df = (
    pd.DataFrame(
        parameter_results
    )
)


diagnostics_df = (
    pd.DataFrame(
        diagnostic_results
    )
)


## Save validation forecasts

if len(
    validation_forecasts
) > 0:

    validation_forecasts_df = (
        pd.concat(
            validation_forecasts,
            ignore_index=True
        )
    )


    validation_forecasts_df = (
        validation_forecasts_df
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


    validation_forecasts_df.to_csv(

        OUTPUT_PATH /
        "combined_garch_validation_forecasts.csv",

        index=False
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


## Save all output tables

validation_results_df.to_csv(

    OUTPUT_PATH /
    "garch_validation_results.csv",

    index=False
)


selected_models_df.to_csv(

    OUTPUT_PATH /
    "garch_selected_models.csv",

    index=False
)


combined_test_forecasts.to_csv(

    OUTPUT_PATH /
    "combined_garch_test_forecasts.csv",

    index=False
)


test_metrics_df.to_csv(

    OUTPUT_PATH /
    "garch_test_metrics.csv",

    index=False
)


parameter_df.to_csv(

    OUTPUT_PATH /
    "garch_final_parameters.csv",

    index=False
)


diagnostics_df.to_csv(

    OUTPUT_PATH /
    "garch_final_diagnostics.csv",

    index=False
)


## Stock-only average test performance

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
            "Selected GARCH"
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
    "garch_stock_average.csv",

    index=False
)


## Final quality checks

EXPECTED_VALIDATION_FORECASTS = 501
EXPECTED_TEST_FORECASTS = 647


# Test forecast counts

for ticker in (
    test_metrics_df[
        "Ticker"
    ]
):

    ticker_count = len(

        combined_test_forecasts[
            combined_test_forecasts[
                "Ticker"
            ] == ticker
        ]
    )


    if (
        ticker_count
        != EXPECTED_TEST_FORECASTS
    ):

        raise ValueError(

            f"{ticker}: expected "
            f"{EXPECTED_TEST_FORECASTS} "
            f"test forecasts but found "
            f"{ticker_count}."
        )


# Validation candidate counts

successful_validation = (
    validation_results_df[
        validation_results_df[
            "QLIKE"
        ].notna()
    ]
)


for ticker in (
    successful_validation[
        "Ticker"
    ].unique()
):

    ticker_models = (
        successful_validation[
            successful_validation[
                "Ticker"
            ] == ticker
        ]
    )


    for _, row in (
        ticker_models.iterrows()
    ):

        if (
            row["Observations"]
            != EXPECTED_VALIDATION_FORECASTS
        ):

            raise ValueError(

                f"{ticker} "
                f"{row['Model']}: "
                f"expected "
                f"{EXPECTED_VALIDATION_FORECASTS} "
                f"validation forecasts."
            )


# Positive variance forecasts

if (
    combined_test_forecasts[
        "GARCH_Forecast"
    ] <= 0
).any():

    raise ValueError(
        "Non-positive final "
        "GARCH forecast detected."
    )


# Missing final forecasts

if combined_test_forecasts[
    "GARCH_Forecast"
].isna().any():

    raise ValueError(
        "Missing final "
        "GARCH forecast detected."
    )


# Target-date chronology

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
        "in final GARCH forecasts."
    )


## Display results

print("\n")
print("=" * 100)
print("SELECTED GARCH MODELS")
print("=" * 100)

print(
    selected_models_df.to_string(
        index=False
    )
)


print("\n")
print("=" * 100)
print("FINAL TEST PERFORMANCE")
print("=" * 100)

print(
    test_metrics_df[
        [
            "Ticker",
            "Model",
            "Observations",
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
print("FINAL MODEL DIAGNOSTICS")
print("=" * 100)

print(
    diagnostics_df.to_string(
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


print("\n")
print("=" * 100)
print("GARCH MODELLING COMPLETE")
print("=" * 100)