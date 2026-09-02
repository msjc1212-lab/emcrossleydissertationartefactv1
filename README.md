# Financial Volatility Forecasting

This is my MSc Data Science dissertation project comparing traditional and machine-learning approaches to one-step-ahead financial volatility forecasting.

The frozen research experiment compares:

- Historical Variance
- GARCH
- Random Forest
- LSTM

The deployment artefact extends the research into a live forecasting system without changing the frozen empirical results.

## Research design

The research experiment uses a fixed chronological split:

- Training: 2010–2021
- Validation: 2022–2023
- Test: 2024–31 July 2026
- Forecast horizon: one trading session ahead
- Target: next-day squared percentage log return

Model specifications are selected using the validation period and evaluated on the fixed test period. Research outputs are not overwritten by later deployment refits.

## Repository structure

```text
artefact/                         Streamlit application and live deployment code
data/processed/                   Clean return data used by the research pipeline
data/experiment/                  Frozen train/validation/test experiment data
data/features/                    Model-specific research features
data/scripts/scripts_static/      Research pipeline, Steps 1–10
data/scripts/11_prepare_deployment_models.py
                                  Deployment model preparation
models/deployment/                Fitted models, scalers, metadata and GARCH parameters
outputs/                          Frozen research results and live forecast snapshots
```

This submission includes raw downloads, live market data, fitted Random Forest and LSTM models, scalers, and saved forecast and refit histories.

## Installation

Create and activate a Python environment, then install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

The recorded deployment environment is stored in:

```text
models/deployment/metadata/deployment_environment.json
```

## Reproducing the research pipeline

The research scripts are numbered and should be run in order:

```text
01_download_data.py
02_clean_and_returns.py
03_eda_diagnostics.py
04_define_experiment.py
05_feature_engineering.py
06_historical_variance_baseline.py
07_garch_models.py
08_random_forest.py
09_lstm_models.py
10_model_comparison.py
```

The repository contains the frozen processed data, features and research outputs used for the dissertation.

## Preparing deployment models

Fitted deployment models and scalers are included, so regeneration is not required to launch the submitted artefact. To reproduce their initial preparation from the frozen research outputs, run the following in a separate working copy:

```bash
python data/scripts/11_prepare_deployment_models.py
```

This reuses the specifications selected during the research validation stage and creates deployment-only models. It does not overwrite the frozen research comparison.

## Running the live artefact

Launch the Streamlit application from the repository root:

```bash
python -m streamlit run artefact/app.py
```

The application is organised into six tabs:

- **Live Predictor** — recent live market history, current next-session variance and volatility forecasts, and comparison of GARCH, Random Forest, LSTM and Historical Variance forecasts
- **Asset Analysis** — asset-level research metrics, out-of-sample forecast history and Diebold–Mariano QLIKE comparisons
- **Forecast Check** — resolves earlier live forecasts against the next completed trading session when a later observation becomes available
- **Refit History** — deployment refit records and maintenance history
- **Methodology** — explanation of the frozen research design, live deployment and refit process
- **Research Results** — frozen out-of-sample dissertation results and statistical comparison

The sidebar provides asset and forecast-model selection, selected-asset market-data refresh, deployment status, scheduled refit eligibility after 63 new trading observations and manually forced deployment refits.

The submission includes the accumulated live forecast history and refit history. Subsequent forecasts and refits update this runtime state locally; they do not alter the frozen research evaluation.

Deployment LSTM refits run in an isolated subprocess using eager TensorFlow execution for reliability. The selected LSTM architecture and research hyperparameters remain unchanged.

## Research and deployment separation

The live layer is deliberately separate from the dissertation experiment.

Research results remain fixed after the 31 July 2026 test cutoff. Live observations can update deployment forecasts and can later be used to refit the already-selected deployment specifications. These refits do not alter the original test forecasts, model rankings or Diebold–Mariano tests.

## Notes

The system forecasts volatility magnitude rather than return direction. It is a research and demonstration system and should not be interpreted as investment advice.
