# MLOps Batch Signal Pipeline

A minimal, production-ready MLOps batch pipeline that:
- Loads OHLCV CSV data
- Computes a rolling mean on the `close` price
- Generates a binary trading signal (`1` if `close > rolling_mean`, else `0`)
- Outputs structured metrics and a detailed log

---

## Project Structure
```
.
├── run.py           # Main pipeline script
├── config.yaml      # Pipeline configuration
├── data.csv         # Input OHLCV data
├── requirements.txt # Python dependencies
├── Dockerfile       # Docker build definition
└── README.md        # This file
```

---

## Local Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the pipeline
```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

### 3. Inspect outputs
```bash
cat metrics.json
cat run.log
```

---

## Docker Run

### Build
```bash
docker build -t mlops-task .
```

### Run
```bash
docker run --rm mlops-task
```

The container will:
1. Execute the full pipeline
2. Print `metrics.json` to stdout
3. Print `run.log` to stdout
4. Exit `0` on success, non-zero on failure

### Extract output files (optional)
```bash
# Run with a named container (don't auto-remove)
docker run --name mlops-run mlops-task

# Copy files out
docker cp mlops-run:/app/metrics.json ./metrics.json
docker cp mlops-run:/app/run.log ./run.log

# Clean up
docker rm mlops-run
```

---

## Configuration Reference (`config.yaml`)

| Key       | Type    | Description                          |
|-----------|---------|--------------------------------------|
| `seed`    | integer | NumPy random seed for reproducibility |
| `window`  | integer | Rolling mean window size (rows)       |
| `version` | string  | Pipeline version tag in output JSON  |

---

## Signal Logic

| Condition                        | Signal |
|----------------------------------|--------|
| `close > rolling_mean`           | `1`    |
| `close <= rolling_mean`          | `0`    |
| Warm-up rows (first `window-1`)  | `0`    |

The first `window - 1` rows produce `NaN` rolling means (not enough history). These rows receive `signal = 0` — no trade during the warm-up period.

---

## Example Output

### `metrics.json` (success)
```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 70.16,
  "seed": 42,
  "status": "success"
}
```

### `metrics.json` (error)
```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Missing required column 'close'. Found columns: ['timestamp,open,high,low,close,volume_btc,volume_usd']"
}
```

---

## Error Handling

The pipeline handles and reports the following failures gracefully:

| Scenario                    | Behaviour                          |
|-----------------------------|------------------------------------|
| Missing input CSV           | Error metrics written, exit 1      |
| Invalid CSV format          | Error metrics written, exit 1      |
| Empty CSV                   | Error metrics written, exit 1      |
| Missing `close` column      | Error metrics written, exit 1      |
| Missing config file         | Error metrics written, exit 1      |
| Invalid YAML                | Error metrics written, exit 1      |
| Missing config keys         | Error metrics written, exit 1      |
| Non-numeric close values    | Rows dropped with warning          |

`metrics.json` is **always** written — even on failure.