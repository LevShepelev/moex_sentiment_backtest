from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from time import sleep
from typing import Iterable, Optional

import apimoex
import numpy as np
import polars as pl
import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _iter_date_chunks(start: dt.date, end: dt.date, chunk_days: int) -> Iterable[tuple[dt.date, dt.date]]:
    cur = start
    while cur <= end:
        chunk_end = min(cur + dt.timedelta(days=chunk_days - 1), end)
        yield cur, chunk_end
        cur = chunk_end + dt.timedelta(days=1)


def download_ticker_candles(
    ticker: str,
    start: str,
    end: str,
    out_path: Path,
    interval: int = 1,
    chunk_days: int = 5,
    sleep_s: float = 0.2,
) -> None:
    """Download MOEX candles for a ticker into a parquet file.

    We chunk by days because minute-level downloads over many years exceed ISS limits.
    """
    start_d = dt.datetime.strptime(start, "%Y-%m-%d").date()
    end_d = dt.datetime.strptime(end, "%Y-%m-%d").date()

    all_rows: list[dict] = []
    with requests.Session() as session:
        for s, e in _iter_date_chunks(start_d, end_d, chunk_days=chunk_days):
            data = apimoex.get_market_candles(
                session,
                ticker,
                interval=interval,
                start=str(s),
                end=str(e),
            )
            if data:
                all_rows.extend(data)
            sleep(sleep_s)

    if not all_rows:
        raise ValueError(f"No candles returned for {ticker} in {start}..{end}.")

    df = pl.from_dicts(all_rows)
    # Normalize schema
    df = df.with_columns(
        pl.col("begin").str.strptime(pl.Datetime, strict=False).alias("ts"),
    ).drop("begin")
    # Keep only standard columns if present
    keep = [c for c in ["ts", "open", "high", "low", "close", "volume", "value"] if c in df.columns]
    df = df.select(keep).sort("ts").unique(subset=["ts"], keep="last")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path, compression="zstd")
    logger.info("Saved %s candles for %s -> %s", df.height, ticker, out_path)


def list_event_tickers(events_path: Path) -> list[str]:
    events = pl.read_parquet(events_path, columns=["ticker"])
    return sorted(events.get_column("ticker").unique().to_list())


def download_prices_for_events(
    events_path: Path,
    start: str,
    end: str,
    out_dir: Path,
    interval: int = 1,
    chunk_days: int = 5,
    sleep_s: float = 0.2,
    n_tickers: Optional[int] = None,
) -> None:
    tickers = list_event_tickers(events_path)
    if n_tickers is not None:
        tickers = tickers[:n_tickers]

    out_dir.mkdir(parents=True, exist_ok=True)
    for t in tqdm(tickers, desc="Downloading MOEX candles"):
        out_path = out_dir / f"{t}.parquet"
        try:
            download_ticker_candles(
                ticker=t,
                start=start,
                end=end,
                out_path=out_path,
                interval=interval,
                chunk_days=chunk_days,
                sleep_s=sleep_s,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed %s: %s", t, e)


def load_prices_as_numpy(prices_path: Path) -> dict[str, np.ndarray]:
    """Load a ticker parquet into numpy arrays (fast for numba simulation)."""
    df = pl.read_parquet(prices_path)
    # Ensure order
    df = df.sort("ts")
    return {
        "ts": df.get_column("ts").to_numpy(),
        "open": df.get_column("open").to_numpy(),
        "high": df.get_column("high").to_numpy(),
        "low": df.get_column("low").to_numpy(),
        "close": df.get_column("close").to_numpy(),
    }
