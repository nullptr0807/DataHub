# DataHub

## Setup
- Install dependencies with `python -m pip install -r requirements.txt`.
- (Optional) export `BINANCE_API_KEY` and `BINANCE_API_SECRET` if you have elevated Binance limits; the downloader works without keys but is slower.

## BTC Price Data Downloader
Populate the `Collector/data` directory using the defaults defined in `config.json`.

Download the kline datasets:

```powershell
python Collector/BinanceDownloader.py --target 1m=Collector/data/btc_1m.csv --target 3m=Collector/data/btc_3m.csv --target 5m=Collector/data/btc_5m.csv --target 1h=Collector/data/btc_1h.csv
```

Download the 1-second aggregate dataset:

```powershell
python Collector/BinanceDownloader.py --mode second --target 1s=Collector/data/btc_1s.csv
```

Both commands read the symbol and start timestamps from `config.json`. Run them once on a fresh clone to seed the CSVs. Afterwards you can append new rows automatically:

```powershell
python Collector/BinanceDownloader.py --scan-dir Collector/data --append
```

The downloader inspects filenames like `btc_1m.csv`, infers the interval, and appends only unseen rows.

## Grid Strategy Backtest Framework
`Backtest/strategies/grid_basic.py` simulates a grid strategy on the downloaded data. Example:

```powershell
python Backtest/strategies/grid_basic.py --data Collector/data/btc_1m.csv --lower 90000 --upper 110000 --levels 21 --days-back 90
```

Key options:
- `--data`: path to the source CSV (defaults to `Collector/data/btc_1m.csv`).
- `--lower` / `--upper`: grid bounds (required).
- `--levels`: number of grid steps (default 21).
- `--fee`: taker/maker fee rate, e.g. `0.0004`.
- `--quote-balance` / `--base-balance`: starting balances for USDT and BTC.
- `--days-back`: filter to the most recent N days before running the grid.
- `--limit`: cap the number of rows processed after filtering.
- `--risk-free`: annual risk-free rate (decimal) used when reporting the Sharpe ratio (default 0.02).

Run `python Backtest/strategies/grid_basic.py --help` for detailed usage and defaults.

The backtest summary prints return metrics along with the annualised Sharpe ratio; `--risk-free` defaults to `0.02` but can be overridden to match your benchmark.

## Web UI (Go)

The `web` directory contains a lightweight Go application that wraps the downloader/backtester and exposes a browser UI for experimenting with different parameters.

1. Install Go (e.g. `brew install go` on macOS) if it is not already available.
2. Ensure the Python virtual environment is set up (`python -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`).
3. Start the server: `cd web && go run ./src` (or from the repo root `go run ./web/src`).
4. Visit `http://localhost:8080` to launch the dashboard, tweak grid settings, and run backtests interactively.

The UI calls the existing Python scripts under the hood, so any CSV files produced in `Collector/data` are immediately available for selection.
