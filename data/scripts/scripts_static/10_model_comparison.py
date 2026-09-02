import pandas as pd
import numpy as np

from pathlib import Path
from itertools import combinations

from scipy.stats import t as student_t


## Paths

EXPERIMENT_PATH = Path(
    "data/experiment"
)

BASELINE_PATH = Path(
    "outputs/baseline"
)

GARCH_PATH = Path(
    "outputs/garch"
)

RF_PATH = Path(
    "outputs/random_forest"
)

LSTM_PATH = Path(
    "outputs/lstm"
)

OUTPUT_PATH = Path(
    "outputs/comparison"
)

OUTPUT_PATH.mkdir(
    parents=True,
    exist_ok=True
)


## Input files

EXPERIMENT_FILE = (
    EXPERIMENT_PATH /
    "combined_experiment_data.csv"
)

BASELINE_FILE = (
    BASELINE_PATH /
    "combined_historical_variance_forecasts.csv"
)

GARCH_FILE = (
    GARCH_PATH /
    "combined_garch_test_forecasts.csv"
)

RF_FILE = (
    RF_PATH /
    "combined_rf_test_forecasts.csv"
)

LSTM_FILE = (
    LSTM_PATH /
    "combined_lstm_test_forecasts.csv"
)


## Settings

EPSILON = 1e-8

EXPECTED_ASSETS = 7

EXPECTED_TEST_ROWS_PER_ASSET = 647

EXPECTED_TOTAL_TEST_ROWS = (
    EXPECTED_ASSETS
    *
    EXPECTED_TEST_ROWS_PER_ASSET
)

FORECAST_HORIZON = 1

SIGNIFICANCE_LEVEL = 0.05


## Model forecast columns

FORECAST_COLUMNS = {

    "Historical Variance":
        "Historical_Variance_Forecast",

    "GARCH":
        "GARCH_Forecast",

    "Random Forest":
        "RF_Forecast",

    "LSTM":
        "LSTM_Forecast"
}


MODEL_NAMES = list(
    FORECAST_COLUMNS.keys()
)


## Common key columns

KEY_COLUMNS = [
    "Date",
    "Target_Date",
    "Ticker",
    "Asset",
    "Role"
]


## Load function

def load_file(
    path
):

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


## Load data

experiment_df = load_file(
    EXPERIMENT_FILE
)

baseline_df = load_file(
    BASELINE_FILE
)

garch_df = load_file(
    GARCH_FILE
)

rf_df = load_file(
    RF_FILE
)

lstm_df = load_file(
    LSTM_FILE
)


## Display raw input counts

print("\n")
print("=" * 100)
print("RAW INPUT COUNTS")
print("=" * 100)

print(
    "Historical variance raw rows:",
    len(
        baseline_df
    )
)

print(
    "GARCH raw rows:",
    len(
        garch_df
    )
)

print(
    "Random Forest raw rows:",
    len(
        rf_df
    )
)

print(
    "LSTM raw rows:",
    len(
        lstm_df
    )
)


## Canonical test target

# The experiment dataset is the single canonical
# source for actual squared returns.
# This avoids tiny floating-point differences in
# model-specific output files.

canonical_test = (
    experiment_df[
        experiment_df[
            "Split"
        ] == "Test"
    ][
        KEY_COLUMNS
        +
        [
            "Target_Squared_Return"
        ]
    ]
    .copy()
)


canonical_test = (
    canonical_test
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


## Canonical test checks

if (
    len(
        canonical_test
    )
    != EXPECTED_TOTAL_TEST_ROWS
):

    raise ValueError(

        "Canonical test dataset expected "
        f"{EXPECTED_TOTAL_TEST_ROWS} rows "
        f"but found "
        f"{len(canonical_test)}."
    )


if canonical_test.duplicated(
    KEY_COLUMNS
).any():

    raise ValueError(
        "Duplicate canonical test keys detected."
    )


if (
    canonical_test[
        "Target_Squared_Return"
    ] < 0
).any():

    raise ValueError(
        "Negative canonical variance target detected."
    )


if not (
    canonical_test[
        "Target_Date"
    ]
    >
    canonical_test[
        "Date"
    ]
).all():

    raise ValueError(
        "Target chronology error "
        "in canonical test data."
    )


## Forecast preparation function

def prepare_forecasts(
    data,
    forecast_column,
    model_name
):

    prepared = (
        data.copy()
    )


    ## Test-period filtering
    # Historical variance includes validation and test forecasts; Step 10 compares test forecasts only.

    if (
        "Split"
        in prepared.columns
    ):

        prepared = (
            prepared[
                prepared[
                    "Split"
                ] == "Test"
            ]
            .copy()
        )


    required_columns = (
        KEY_COLUMNS
        +
        [
            forecast_column
        ]
    )


    missing_columns = [

        column

        for column
        in required_columns

        if column
        not in prepared.columns
    ]


    if (
        len(
            missing_columns
        )
        > 0
    ):

        raise ValueError(

            f"{model_name}: missing columns "
            f"{missing_columns}"
        )


    prepared = (
        prepared[
            required_columns
        ]
        .copy()
    )


    prepared = (
        prepared
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


    return prepared


## Prepare all four forecast datasets

baseline_forecasts = (
    prepare_forecasts(

        baseline_df,

        "Historical_Variance_Forecast",

        "Historical Variance"
    )
)


garch_forecasts = (
    prepare_forecasts(

        garch_df,

        "GARCH_Forecast",

        "GARCH"
    )
)


rf_forecasts = (
    prepare_forecasts(

        rf_df,

        "RF_Forecast",

        "Random Forest"
    )
)


lstm_forecasts = (
    prepare_forecasts(

        lstm_df,

        "LSTM_Forecast",

        "LSTM"
    )
)


forecast_datasets = {

    "Historical Variance":
        baseline_forecasts,

    "GARCH":
        garch_forecasts,

    "Random Forest":
        rf_forecasts,

    "LSTM":
        lstm_forecasts
}


## Display test-only counts

print("\n")
print("=" * 100)
print("TEST-ONLY FORECAST COUNTS")
print("=" * 100)


for (
    model_name,
    model_df
) in forecast_datasets.items():

    print(
        f"{model_name}: "
        f"{len(model_df)}"
    )


## Forecast dataset quality checks

for (
    model_name,
    model_df
) in forecast_datasets.items():


    # Correct number of rows

    if (
        len(
            model_df
        )
        != EXPECTED_TOTAL_TEST_ROWS
    ):

        raise ValueError(

            f"{model_name}: expected "
            f"{EXPECTED_TOTAL_TEST_ROWS} "
            f"test forecasts but found "
            f"{len(model_df)}."
        )


    # Duplicate keys

    if model_df.duplicated(
        KEY_COLUMNS
    ).any():

        raise ValueError(

            f"{model_name}: duplicate "
            "forecast keys detected."
        )


    # Exact key coverage

    coverage_check = (

        canonical_test[
            KEY_COLUMNS
        ]

        .merge(

            model_df[
                KEY_COLUMNS
            ],

            on=
                KEY_COLUMNS,

            how=
                "left",

            indicator=
                True,

            validate=
                "one_to_one"
        )
    )


    if not (
        coverage_check[
            "_merge"
        ]
        == "both"
    ).all():

        raise ValueError(

            f"{model_name}: forecast dates "
            "do not exactly match the "
            "canonical test dates."
        )


## Merge all four models

combined = (
    canonical_test

    .merge(

        baseline_forecasts,

        on=
            KEY_COLUMNS,

        how=
            "left",

        validate=
            "one_to_one"
    )

    .merge(

        garch_forecasts,

        on=
            KEY_COLUMNS,

        how=
            "left",

        validate=
            "one_to_one"
    )

    .merge(

        rf_forecasts,

        on=
            KEY_COLUMNS,

        how=
            "left",

        validate=
            "one_to_one"
    )

    .merge(

        lstm_forecasts,

        on=
            KEY_COLUMNS,

        how=
            "left",

        validate=
            "one_to_one"
    )
)


combined = (
    combined
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


## Merged data quality checks

if (
    len(
        combined
    )
    != EXPECTED_TOTAL_TEST_ROWS
):

    raise ValueError(
        "Incorrect combined test forecast count."
    )


for (
    model_name,
    column
) in FORECAST_COLUMNS.items():


    if combined[
        column
    ].isna().any():

        raise ValueError(

            f"{model_name}: missing "
            "forecast detected after merge."
        )


    if not np.isfinite(
        combined[
            column
        ]
    ).all():

        raise ValueError(

            f"{model_name}: non-finite "
            "forecast detected."
        )


    if (
        combined[
            column
        ] <= 0
    ).any():

        raise ValueError(

            f"{model_name}: non-positive "
            "variance forecast detected."
        )


# 647 observations per asset

for ticker in (
    combined[
        "Ticker"
    ].unique()
):

    ticker_count = len(

        combined[
            combined[
                "Ticker"
            ] == ticker
        ]
    )


    if (
        ticker_count
        != EXPECTED_TEST_ROWS_PER_ASSET
    ):

        raise ValueError(

            f"{ticker}: expected "
            f"{EXPECTED_TEST_ROWS_PER_ASSET} "
            f"test observations but found "
            f"{ticker_count}."
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


    forecast = np.maximum(
        forecast,
        EPSILON
    )


    error = (
        forecast
        -
        actual
    )


    # RMSE

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )


    # MAE

    mae = np.mean(
        np.abs(
            error
        )
    )


    # QLIKE

    qlike = np.mean(

        np.log(
            forecast
        )

        +

        actual
        /
        forecast
    )


    return {

        "RMSE":
            rmse,

        "MAE":
            mae,

        "QLIKE":
            qlike
    }


## Metrics by asset + model

metric_results = []


for ticker in (
    combined[
        "Ticker"
    ].unique()
):

    ticker_df = (
        combined[
            combined[
                "Ticker"
            ] == ticker
        ]
        .copy()
    )


    asset_name = (
        ticker_df[
            "Asset"
        ].iloc[0]
    )


    role = (
        ticker_df[
            "Role"
        ].iloc[0]
    )


    actual = (
        ticker_df[
            "Target_Squared_Return"
        ]
        .to_numpy()
    )


    for (
        model_name,
        forecast_column
    ) in FORECAST_COLUMNS.items():


        forecast = (
            ticker_df[
                forecast_column
            ]
            .to_numpy()
        )


        metrics = calculate_metrics(

            actual,
            forecast
        )


        metric_results.append(
            {
                "Ticker":
                    ticker,

                "Asset":
                    asset_name,

                "Role":
                    role,

                "Model":
                    model_name,

                "Observations":
                    len(
                        ticker_df
                    ),

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
                    ],

                "Mean_Actual_Variance":
                    np.mean(
                        actual
                    ),

                "Mean_Forecast_Variance":
                    np.mean(
                        forecast
                    )
            }
        )


metrics_df = pd.DataFrame(
    metric_results
)


## Stock-only average performance

stock_metrics = (
    metrics_df[
        metrics_df[
            "Role"
        ] == "Stock"
    ]
    .copy()
)


stock_average = (

    stock_metrics

    .groupby(
        "Model",
        as_index=False
    )

    .agg(

        Mean_RMSE=(
            "RMSE",
            "mean"
        ),

        Mean_MAE=(
            "MAE",
            "mean"
        ),

        Mean_QLIKE=(
            "QLIKE",
            "mean"
        ),

        Number_of_Stocks=(
            "Ticker",
            "nunique"
        )
    )
)


# Ranks: lower = better

stock_average[
    "RMSE_Rank"
] = (
    stock_average[
        "Mean_RMSE"
    ]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)


stock_average[
    "MAE_Rank"
] = (
    stock_average[
        "Mean_MAE"
    ]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)


stock_average[
    "QLIKE_Rank"
] = (
    stock_average[
        "Mean_QLIKE"
    ]
    .rank(
        method="min",
        ascending=True
    )
    .astype(int)
)


stock_average = (
    stock_average
    .sort_values(
        "Mean_QLIKE"
    )
    .reset_index(
        drop=True
    )
)


## Asset-by-asset rankings

ranking_results = []


for ticker in (
    metrics_df[
        "Ticker"
    ].unique()
):

    ticker_metrics = (
        metrics_df[
            metrics_df[
                "Ticker"
            ] == ticker
        ]
        .copy()
    )


    asset_name = (
        ticker_metrics[
            "Asset"
        ].iloc[0]
    )


    role = (
        ticker_metrics[
            "Role"
        ].iloc[0]
    )


    for metric in [
        "RMSE",
        "MAE",
        "QLIKE"
    ]:


        ranked = (
            ticker_metrics[
                [
                    "Model",
                    metric
                ]
            ]
            .sort_values(
                metric
            )
            .reset_index(
                drop=True
            )
        )


        ranked[
            "Rank"
        ] = np.arange(
            1,
            len(
                ranked
            )
            + 1
        )


        for _, row in (
            ranked.iterrows()
        ):

            ranking_results.append(
                {
                    "Ticker":
                        ticker,

                    "Asset":
                        asset_name,

                    "Role":
                        role,

                    "Metric":
                        metric,

                    "Model":
                        row[
                            "Model"
                        ],

                    "Value":
                        row[
                            metric
                        ],

                    "Rank":
                        int(
                            row[
                                "Rank"
                            ]
                        )
                }
            )


rankings_df = pd.DataFrame(
    ranking_results
)


## Model win counts

stock_winners = (
    rankings_df[
        (
            rankings_df[
                "Role"
            ] == "Stock"
        )
        &
        (
            rankings_df[
                "Rank"
            ] == 1
        )
    ]
    .copy()
)


win_counts = (

    stock_winners

    .groupby(
        [
            "Metric",
            "Model"
        ],
        as_index=False
    )

    .size()

    .rename(
        columns={
            "size":
                "Number_of_Stock_Wins"
        }
    )
)


## Improvement vs historical baseline

improvement_results = []


for ticker in (
    metrics_df[
        "Ticker"
    ].unique()
):

    ticker_metrics = (
        metrics_df[
            metrics_df[
                "Ticker"
            ] == ticker
        ]
    )


    baseline_row = (
        ticker_metrics[
            ticker_metrics[
                "Model"
            ]
            == "Historical Variance"
        ]
        .iloc[0]
    )


    for model_name in [
        "GARCH",
        "Random Forest",
        "LSTM"
    ]:


        model_row = (
            ticker_metrics[
                ticker_metrics[
                    "Model"
                ]
                == model_name
            ]
            .iloc[0]
        )


        improvement_results.append(
            {
                "Ticker":
                    ticker,

                "Asset":
                    model_row[
                        "Asset"
                    ],

                "Role":
                    model_row[
                        "Role"
                    ],

                "Model":
                    model_name,

                "RMSE_Improvement_Percent":

                    (
                        (
                            baseline_row[
                                "RMSE"
                            ]
                            -
                            model_row[
                                "RMSE"
                            ]
                        )
                        /
                        baseline_row[
                            "RMSE"
                        ]
                        *
                        100
                    ),

                "MAE_Improvement_Percent":

                    (
                        (
                            baseline_row[
                                "MAE"
                            ]
                            -
                            model_row[
                                "MAE"
                            ]
                        )
                        /
                        baseline_row[
                            "MAE"
                        ]
                        *
                        100
                    ),

                "QLIKE_Improvement_Percent":

                    (
                        (
                            baseline_row[
                                "QLIKE"
                            ]
                            -
                            model_row[
                                "QLIKE"
                            ]
                        )
                        /
                        baseline_row[
                            "QLIKE"
                        ]
                        *
                        100
                    )
            }
        )


improvement_df = pd.DataFrame(
    improvement_results
)


## QLIKE comparison by asset

qlike_wide = (

    metrics_df

    .pivot(

        index=[
            "Ticker",
            "Asset",
            "Role"
        ],

        columns=
            "Model",

        values=
            "QLIKE"
    )

    .reset_index()
)


qlike_wide.columns.name = None


## Loss function

def calculate_loss(
    actual,
    forecast,
    loss_name
):

    actual = np.asarray(
        actual,
        dtype=float
    )


    forecast = np.asarray(
        forecast,
        dtype=float
    )


    forecast = np.maximum(
        forecast,
        EPSILON
    )


    # QLIKE

    if (
        loss_name
        == "QLIKE"
    ):

        return (

            np.log(
                forecast
            )

            +

            actual
            /
            forecast
        )


    # Squared error

    elif (
        loss_name
        == "Squared_Error"
    ):

        return (

            forecast
            -
            actual

        ) ** 2


    # Absolute error

    elif (
        loss_name
        == "Absolute_Error"
    ):

        return np.abs(

            forecast
            -
            actual
        )


    else:

        raise ValueError(

            f"Unknown loss function: "
            f"{loss_name}"
        )


## Diebold-mariano test

def diebold_mariano_test(
    loss_model_1,
    loss_model_2,
    horizon=1
):

    loss_model_1 = np.asarray(
        loss_model_1,
        dtype=float
    )


    loss_model_2 = np.asarray(
        loss_model_2,
        dtype=float
    )


    if (
        len(
            loss_model_1
        )
        !=
        len(
            loss_model_2
        )
    ):

        raise ValueError(

            "DM loss arrays must "
            "have equal length."
        )


    # Loss differential:
    # d_t = Loss(Model 1) - Loss(Model 2)
    # Negative:
    # Model 1 lower loss
    # Positive:
    # Model 2 lower loss

    d = (
        loss_model_1
        -
        loss_model_2
    )


    n = len(
        d
    )


    mean_d = np.mean(
        d
    )


    centered = (
        d
        -
        mean_d
    )


    # Long-run variance
    # For one-step-ahead forecasts,
    # h - 1 = 0.

    long_run_variance = np.mean(

        centered
        *
        centered
    )


    for lag in range(
        1,
        horizon
    ):


        covariance = np.mean(

            centered[
                lag:
            ]

            *

            centered[
                :-lag
            ]
        )


        long_run_variance += (

            2
            *
            covariance
        )


    if (
        long_run_variance
        <= 0
        or
        not np.isfinite(
            long_run_variance
        )
    ):

        return {

            "DM_Statistic":
                np.nan,

            "Raw_p_value":
                np.nan,

            "Mean_Loss_Difference":
                mean_d
        }


    # Standard DM statistic

    dm_statistic = (

        mean_d

        /

        np.sqrt(

            long_run_variance
            /
            n
        )
    )


    # Harvey-Leybourne-Newbold correction

    correction = np.sqrt(

        (
            n
            +
            1
            -
            2
            *
            horizon
            +
            (
                horizon
                *
                (
                    horizon
                    -
                    1
                )
                /
                n
            )
        )

        /

        n
    )


    modified_dm = (

        dm_statistic
        *
        correction
    )


    # Two-sided p-value

    p_value = (

        2

        *

        student_t.sf(

            np.abs(
                modified_dm
            ),

            df=
                n - 1
        )
    )


    return {

        "DM_Statistic":
            modified_dm,

        "Raw_p_value":
            p_value,

        "Mean_Loss_Difference":
            mean_d
    }


## Holm multiple-comparison adjustment

def holm_adjust(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )


    m = len(
        p_values
    )


    order = np.argsort(
        p_values
    )


    sorted_p = (
        p_values[
            order
        ]
    )


    adjusted_sorted = np.empty(
        m,
        dtype=float
    )


    previous = 0.0


    for i in range(
        m
    ):


        adjusted_value = (

            (
                m - i
            )

            *

            sorted_p[
                i
            ]
        )


        adjusted_value = min(
            adjusted_value,
            1.0
        )


        adjusted_value = max(
            adjusted_value,
            previous
        )


        adjusted_sorted[
            i
        ] = adjusted_value


        previous = (
            adjusted_value
        )


    adjusted = np.empty(
        m,
        dtype=float
    )


    adjusted[
        order
    ] = adjusted_sorted


    return adjusted


## Run pairwise dm tests

LOSS_FUNCTIONS = [
    "QLIKE",
    "Squared_Error",
    "Absolute_Error"
]


MODEL_PAIRS = list(

    combinations(
        MODEL_NAMES,
        2
    )
)


dm_results = []


for ticker in (
    combined[
        "Ticker"
    ].unique()
):


    ticker_df = (
        combined[
            combined[
                "Ticker"
            ] == ticker
        ]
        .copy()
    )


    actual = (
        ticker_df[
            "Target_Squared_Return"
        ]
        .to_numpy()
    )


    role = (
        ticker_df[
            "Role"
        ].iloc[0]
    )


    asset_name = (
        ticker_df[
            "Asset"
        ].iloc[0]
    )


    for loss_name in (
        LOSS_FUNCTIONS
    ):


        for (
            model_1,
            model_2
        ) in MODEL_PAIRS:


            forecast_1 = (
                ticker_df[
                    FORECAST_COLUMNS[
                        model_1
                    ]
                ]
                .to_numpy()
            )


            forecast_2 = (
                ticker_df[
                    FORECAST_COLUMNS[
                        model_2
                    ]
                ]
                .to_numpy()
            )


            loss_1 = calculate_loss(

                actual,
                forecast_1,
                loss_name
            )


            loss_2 = calculate_loss(

                actual,
                forecast_2,
                loss_name
            )


            result = (
                diebold_mariano_test(

                    loss_1,
                    loss_2,

                    horizon=
                        FORECAST_HORIZON
                )
            )


            mean_loss_1 = np.mean(
                loss_1
            )


            mean_loss_2 = np.mean(
                loss_2
            )


            if (
                mean_loss_1
                <
                mean_loss_2
            ):

                lower_loss_model = (
                    model_1
                )


            elif (
                mean_loss_2
                <
                mean_loss_1
            ):

                lower_loss_model = (
                    model_2
                )


            else:

                lower_loss_model = (
                    "Equal"
                )


            dm_results.append(
                {
                    "Ticker":
                        ticker,

                    "Asset":
                        asset_name,

                    "Role":
                        role,

                    "Loss_Function":
                        loss_name,

                    "Model_1":
                        model_1,

                    "Model_2":
                        model_2,

                    "Observations":
                        len(
                            ticker_df
                        ),

                    "Mean_Loss_Model_1":
                        mean_loss_1,

                    "Mean_Loss_Model_2":
                        mean_loss_2,

                    "Mean_Loss_Difference":
                        result[
                            "Mean_Loss_Difference"
                        ],

                    "DM_Statistic":
                        result[
                            "DM_Statistic"
                        ],

                    "Raw_p_value":
                        result[
                            "Raw_p_value"
                        ],

                    "Lower_Mean_Loss_Model":
                        lower_loss_model
                }
            )


dm_df = pd.DataFrame(
    dm_results
)


## Holm-adjust dm results

dm_df[
    "Holm_Adjusted_p"
] = np.nan


for (
    ticker,
    loss_name
), group in dm_df.groupby(
    [
        "Ticker",
        "Loss_Function"
    ]
):


    adjusted = holm_adjust(

        group[
            "Raw_p_value"
        ]
        .to_numpy()
    )


    dm_df.loc[
        group.index,
        "Holm_Adjusted_p"
    ] = adjusted


dm_df[
    "Significant_Raw_5pct"
] = (

    dm_df[
        "Raw_p_value"
    ]
    <
    SIGNIFICANCE_LEVEL
)


dm_df[
    "Significant_Holm_5pct"
] = (

    dm_df[
        "Holm_Adjusted_p"
    ]
    <
    SIGNIFICANCE_LEVEL
)


dm_df[
    "Preferred_Model_if_Holm_Significant"
] = np.where(

    dm_df[
        "Significant_Holm_5pct"
    ],

    dm_df[
        "Lower_Mean_Loss_Model"
    ],

    "No significant difference"
)


## Primary QLIKE dm results

dm_qlike = (
    dm_df[
        dm_df[
            "Loss_Function"
        ] == "QLIKE"
    ]
    .copy()
)


## Save outputs

combined.to_csv(

    OUTPUT_PATH /
    "combined_test_forecasts_all_models.csv",

    index=False
)


metrics_df.to_csv(

    OUTPUT_PATH /
    "all_model_test_metrics.csv",

    index=False
)


stock_average.to_csv(

    OUTPUT_PATH /
    "stock_average_model_comparison.csv",

    index=False
)


rankings_df.to_csv(

    OUTPUT_PATH /
    "asset_metric_rankings.csv",

    index=False
)


win_counts.to_csv(

    OUTPUT_PATH /
    "model_win_counts.csv",

    index=False
)


improvement_df.to_csv(

    OUTPUT_PATH /
    "improvement_vs_historical_variance.csv",

    index=False
)


qlike_wide.to_csv(

    OUTPUT_PATH /
    "qlike_comparison_by_asset.csv",

    index=False
)


dm_df.to_csv(

    OUTPUT_PATH /
    "diebold_mariano_all_losses.csv",

    index=False
)


dm_qlike.to_csv(

    OUTPUT_PATH /
    "diebold_mariano_qlike.csv",

    index=False
)


## Display stock averages

print("\n")
print("=" * 110)
print("STOCK-ONLY MODEL COMPARISON")
print("=" * 110)


print(

    stock_average[
        [
            "Model",
            "Mean_RMSE",
            "Mean_MAE",
            "Mean_QLIKE",
            "RMSE_Rank",
            "MAE_Rank",
            "QLIKE_Rank"
        ]
    ]

    .to_string(
        index=False
    )
)


## Display QLIKE winners

print("\n")
print("=" * 110)
print("QLIKE WINNER BY ASSET")
print("=" * 110)


qlike_winners = (
    rankings_df[
        (
            rankings_df[
                "Metric"
            ] == "QLIKE"
        )
        &
        (
            rankings_df[
                "Rank"
            ] == 1
        )
    ]
)


print(

    qlike_winners[
        [
            "Ticker",
            "Asset",
            "Role",
            "Model",
            "Value"
        ]
    ]

    .to_string(
        index=False
    )
)


## Display primary dm results

print("\n")
print("=" * 110)
print("QLIKE DIEBOLD-MARIANO RESULTS")
print("=" * 110)


print(

    dm_qlike[
        [
            "Ticker",
            "Model_1",
            "Model_2",
            "Mean_Loss_Difference",
            "DM_Statistic",
            "Raw_p_value",
            "Holm_Adjusted_p",
            "Lower_Mean_Loss_Model",
            "Significant_Holm_5pct"
        ]
    ]

    .to_string(
        index=False
    )
)


## Complete

print("\n")
print("=" * 110)
print("MODEL COMPARISON COMPLETE")
print("=" * 110)

print(
    "\nExpected combined test rows:",
    EXPECTED_TOTAL_TEST_ROWS
)

print(
    "Actual combined test rows:",
    len(
        combined
    )
)