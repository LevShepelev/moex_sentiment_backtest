# MOEX Sentiment Backtest

Event-driven backtesting for MOEX equities based on **news sentiment** (e.g., RuAdaptQwen2.5-32B markup).

This project is intentionally built for:
- **large** news datasets (your 900k-row table),
- **intraday** MOEX candles (1-minute),
- realistic execution toggles: **bid/ask spread**, **commissions**, **slippage**,
- optional **short-selling** with simplified **borrow + margin** modeling,
- multiple exit rules (TP/SL, next day, after N days).

> You will typically do a one-time conversion of the Excel news file to a compact parquet event table,
> and a one-time download of minute candles to parquet. After that, backtests run fast.

## Quick start

### 1) Install

```bash
poetry install
poetry run pre-commit install
```

### 2) Convert your Excel news table to an *events* parquet (only non-zero sentiments)

```bash
poetry run msbt prepare-news \
  --news-path path/to/news.xlsx \
  --out data/news/events.parquet
```

### 3) Download MOEX 1-minute candles for all tickers present in the news table

```bash
poetry run msbt download-prices \
  --events data/news/events.parquet \
  --start 2017-01-01 \
  --end 2026-02-08 \
  --out-dir data/prices \
  --interval 1 \
  --chunk-days 5
```

### 4) Run the included strategy suite & build plots

```bash
poetry run msbt run \
  --events data/news/events_full.parquet \
  --prices-dir data/prices \
  --config configs/default.yaml \
  --out-dir reports/run_01
```

This will produce:
- `reports/run_01/metrics.csv`
- `reports/run_01/trades.parquet`
- `reports/run_01/equity_curves.png`
- `reports/run_01/drawdowns.png`
- `reports/run_01/trade_return_hist.png`

## Data formats

### Events (parquet)

Produced by `msbt prepare-news`. Columns:

- `event_id` (int)
- `ts` (datetime) - timestamp of the news/post
- `ticker` (str) - MOEX ticker, e.g. `GAZP`
- `sentiment` (i8) - in {-2,-1,1,2}; 0 rows are removed
- optional metadata columns: `message_id`, `channel_link`, `post_link` (if present)

### Prices (parquet per ticker)

Produced by `msbt download-prices` into `data/prices/TICKER.parquet`:

- `ts` (datetime) - candle begin time
- `open`, `high`, `low`, `close` (float)
- `volume`, `value` (float, optional)

## Notes / assumptions

- Timestamp alignment: entry uses the **first candle whose `ts >= event.ts`** (i.e., next minute bar).
- Exits:
  - TP/SL are checked using candle OHLC with a configurable **intrabar priority** (`stop_first` vs `tp_first`).
  - If not hit, a time exit occurs at the configured **open** or **close** of the exit day.
- Shorting model (simplified but realistic enough for research):
  - bid/ask & commissions apply on both entry/exit,
  - borrow fee accrues linearly by holding time,
  - Reg-T style short margin is approximated via `short_margin_pct` (default 0.5 => 150% initial margin).

## Development

```bash
poetry run ruff check .
poetry run ruff format .
poetry run mypy src
poetry run pytest -q
```
