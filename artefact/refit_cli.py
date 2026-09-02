import argparse
import sys
import time
import traceback

from tensorflow import keras

import artefact.deployment_refit as deployment_refit
from artefact.normalize_deployment_dates import normalize_deployment_dates


class EpochProgress(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print("      LSTM training started", flush=True)

    def on_epoch_begin(self, epoch, logs=None):
        self._start = time.time()
        print(
            f"      epoch {epoch + 1}/{self.params['epochs']} starting...",
            flush=True,
        )

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self._start
        loss = None if logs is None else logs.get("loss")
        message = (
            f"      epoch {epoch + 1}/{self.params['epochs']} "
            f"finished in {elapsed:.1f}s"
        )
        if loss is not None:
            message += f" | loss={loss:.6f}"
        print(message, flush=True)


def build_eager_lstm_model(
    lookback,
    number_features,
    units,
    dropout,
    learning_rate
):
    model = keras.Sequential([
        keras.Input(shape=(lookback, number_features)),
        keras.layers.LSTM(
            units=units,
            dropout=dropout,
            recurrent_dropout=0.0
        ),
        keras.layers.Dense(
            1,
            activation="softplus"
        )
    ])

    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0
    )

    model.compile(
        optimizer=optimizer,
        loss="mse",
        run_eagerly=True
    )

    return model


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a deployment-model refit outside the Streamlit process. "
            "The LSTM refit uses eager execution because the compiled training "
            "path stalls on this local macOS/TensorFlow environment."
        )
    )
    parser.add_argument(
        "ticker",
        help="Ticker to refit, e.g. AAPL or ^GSPC"
    )
    parser.add_argument(
        "--trigger",
        choices=["Forced", "Scheduled"],
        default="Forced",
        help="Refit trigger type. Default: Forced"
    )
    args = parser.parse_args()

    deployment_refit.build_lstm_model = build_eager_lstm_model

    print("=" * 100, flush=True)
    print(
        f"STARTING {args.trigger.upper()} DEPLOYMENT REFIT FOR {args.ticker}",
        flush=True
    )
    print(
        "Running outside Streamlit so the dashboard process remains independent.",
        flush=True
    )
    print(
        "LSTM training mode: eager execution (run_eagerly=True)",
        flush=True
    )
    print("=" * 100, flush=True)

    try:
        result = deployment_refit.refit_selected_asset(
            ticker=args.ticker,
            trigger=args.trigger,
            lstm_callbacks=[EpochProgress()],
        )

        normalize_deployment_dates()
        print("Deployment date columns normalised.", flush=True)

        print("\n" + "=" * 100, flush=True)
        print("REFIT COMPLETED SUCCESSFULLY", flush=True)
        print("=" * 100, flush=True)

        for key, value in result.items():
            print(f"{key}: {value}", flush=True)

    except KeyboardInterrupt:
        print(
            "\nRefit interrupted by user. Existing deployment files should remain "
            "unchanged unless replacement had already completed.",
            flush=True
        )
        sys.exit(130)

    except Exception:
        print("\nREFIT FAILED", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
