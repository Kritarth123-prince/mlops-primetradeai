import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


# ---------------------------------------------------------------------------
# Config handling
# ---------------------------------------------------------------------------

REQUIRED_CONFIG_KEYS = {"seed", "window", "version"}


def load_config(config_path: str, logger: logging.Logger) -> dict:
    logger.info("Loading config from: %s", config_path)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in config: {exc}") from exc

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping (key-value pairs).")

    missing = REQUIRED_CONFIG_KEYS - config.keys()
    if missing:
        raise ValueError(f"Config missing required fields: {sorted(missing)}")

    if not isinstance(config["seed"], int):
        raise ValueError(f"'seed' must be an integer, got: {type(config['seed']).__name__}")
    if not isinstance(config["window"], int) or config["window"] < 1:
        raise ValueError(f"'window' must be a positive integer, got: {config['window']}")
    if not isinstance(config["version"], str) or not config["version"].strip():
        raise ValueError("'version' must be a non-empty string.")

    logger.info(
        "Config validated — seed=%s, window=%s, version=%s",
        config["seed"], config["window"], config["version"],
    )
    return config


# ---------------------------------------------------------------------------
# Data loading + validation
# ---------------------------------------------------------------------------

def load_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    logger.info("Loading input data from: %s", input_path)

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError("Input CSV is empty.")
    except Exception as exc:
        raise ValueError(f"Failed to parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Input CSV has no data rows.")

    if "close" not in df.columns:
        raise ValueError(
            f"Missing required column 'close'. Found columns: {list(df.columns)}"
        )

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    n_invalid = df["close"].isna().sum()
    if n_invalid == len(df):
        raise ValueError("All values in 'close' column are non-numeric.")
    if n_invalid > 0:
        logger.warning("Dropping %d rows with non-numeric 'close' values.", n_invalid)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

    logger.info("Rows loaded after validation: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Processing steps
# ---------------------------------------------------------------------------

def compute_rolling_mean(series: pd.Series, window: int, logger: logging.Logger) -> pd.Series:
    logger.info("Computing rolling mean with window=%d (first %d rows will be NaN).", window, window - 1)
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    valid_count = rolling_mean.notna().sum()
    logger.debug("Rolling mean computed — valid (non-NaN) values: %d / %d", valid_count, len(series))
    return rolling_mean


def generate_signals(close: pd.Series, rolling_mean: pd.Series, logger: logging.Logger) -> pd.Series:
    logger.info("Generating signals (1 = close > rolling_mean, 0 otherwise).")
    signal = (close > rolling_mean).astype(int)
    signal[rolling_mean.isna()] = 0
    logger.debug("Signal distribution — 1s: %d, 0s: %d", signal.sum(), (signal == 0).sum())
    return signal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    df: pd.DataFrame,
    signal: pd.Series,
    config: dict,
    latency_ms: float,
    logger: logging.Logger,
) -> dict:
    rows_processed = len(df)
    signal_rate = round(float(signal.mean()), 6)

    metrics = {
        "version": config["version"],
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": signal_rate,
        "latency_ms": round(latency_ms, 2),
        "seed": config["seed"],
        "status": "success",
    }

    logger.info(
        "Metrics — rows_processed=%d, signal_rate=%.6f, latency_ms=%.2f",
        rows_processed, signal_rate, latency_ms,
    )
    return metrics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_metrics(metrics: dict, output_path: str, logger: logging.Logger) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics written to: %s", output_path)


def write_error_metrics(
    error_message: str,
    output_path: str,
    version: str = "unknown",
) -> None:
    error_payload = {
        "version": version,
        "status": "error",
        "error_message": error_message,
    }
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(error_payload, f, indent=2)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps batch signal pipeline.")
    parser.add_argument("--input",   required=True, help="Path to input CSV file.")
    parser.add_argument("--config",  required=True, help="Path to YAML config file.")
    parser.add_argument("--output",  required=True, help="Path to output metrics JSON.")
    parser.add_argument("--log-file", required=True, dest="log_file", help="Path to log file.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    start_time = time.perf_counter()

    logger = setup_logging(args.log_file)
    logger.info("=" * 60)
    logger.info("MLOps Pipeline — Job started")
    logger.info("  input   : %s", args.input)
    logger.info("  config  : %s", args.config)
    logger.info("  output  : %s", args.output)
    logger.info("  log-file: %s", args.log_file)
    logger.info("=" * 60)

    version = "unknown"

    try:
        # 1. Config
        config = load_config(args.config, logger)
        version = config["version"]

        # 2. Reproducibility seed
        np.random.seed(config["seed"])
        logger.info("NumPy random seed set to %d.", config["seed"])

        # 3. Data
        df = load_data(args.input, logger)

        # 4. Rolling mean
        rolling_mean = compute_rolling_mean(df["close"], config["window"], logger)

        # 5. Signals
        signal = generate_signals(df["close"], rolling_mean, logger)

        # 6. Latency
        latency_ms = (time.perf_counter() - start_time) * 1000

        # 7. Metrics
        metrics = compute_metrics(df, signal, config, latency_ms, logger)

        # 8. Write output
        write_metrics(metrics, args.output, logger)

        # Print to stdout for Docker visibility
        print(json.dumps(metrics, indent=2))

        logger.info("=" * 60)
        logger.info("MLOps Pipeline — Job completed successfully.")
        logger.info("=" * 60)

    except Exception as exc:
        error_message = str(exc)
        logger.error("Pipeline FAILED: %s", error_message)
        logger.debug("Full traceback:\n%s", traceback.format_exc())

        write_error_metrics(error_message, args.output, version=version)
        print(json.dumps({"version": version, "status": "error", "error_message": error_message}, indent=2))

        logger.info("=" * 60)
        logger.info("MLOps Pipeline — Job ended with errors.")
        logger.info("=" * 60)

        sys.exit(1)

if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)