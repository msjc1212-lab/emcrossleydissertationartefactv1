from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = PROJECT_ROOT / "models/deployment/metadata/deployment_manifest.csv"
MANIFEST_OUTPUT_FILE = PROJECT_ROOT / "outputs/deployment/deployment_manifest.csv"
GARCH_PARAMETER_FILE = PROJECT_ROOT / "models/deployment/garch/garch_deployment_parameters.csv"


def _normalise_column(path, column):
    if not path.exists():
        return

    data = pd.read_csv(path)
    if column not in data.columns:
        return

    parsed = pd.to_datetime(data[column], format="mixed", errors="raise")
    data[column] = parsed.dt.strftime("%Y-%m-%d")
    data.to_csv(path, index=False)


def normalize_deployment_dates():
    _normalise_column(MANIFEST_FILE, "GARCH_Training_Through")
    _normalise_column(MANIFEST_FILE, "RF_Target_Through")
    _normalise_column(MANIFEST_FILE, "LSTM_Target_Through")

    _normalise_column(MANIFEST_OUTPUT_FILE, "GARCH_Training_Through")
    _normalise_column(MANIFEST_OUTPUT_FILE, "RF_Target_Through")
    _normalise_column(MANIFEST_OUTPUT_FILE, "LSTM_Target_Through")

    _normalise_column(GARCH_PARAMETER_FILE, "Training_Through")


if __name__ == "__main__":
    normalize_deployment_dates()
    print("Deployment date columns normalised successfully.")
