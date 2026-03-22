from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests


ISS = "https://iss.moex.com/iss"


@dataclass(frozen=True)
class BenchmarkSpec:
    name: str
    secid: str
    engine: str = "stock"
    market: str = "index"
    board: str | None = None  # usually None works for indices


def _fetch_candles(
    secid: str,
    start: str,
    end: str,
    interval: int = 24,
    engine: str = "stock",
    market: str = "index",
    board: str | None = None,
) -> pd.DataFrame:
    """
    Fetch candles from MOEX ISS. For indices, engine=stock, market=index typically works.
    Returns a pandas DataFrame with at least: begin, open, high, low, close.
    """
    if board:
        url = f"{ISS}/engines/{engine}/markets/{market}/boards/{board}/securities/{secid}/candles.json"
    else:
        url = f"{ISS}/engines/{engine}/markets/{market}/securities/{secid}/candles.json"

    rows: list[list] = []
    columns: list[str] | None = None

    with requests.Session() as s:
        start_row = 0
        while True:
            params = {
                "from": start,
                "till": end,
                "interval": interval,
                "start": start_row,
            }
            j = s.get(url, params=params, timeout=60).json()
            block = j.get("candles", {})
            data = block.get("data", [])
            cols = block.get("columns", [])

            if columns is None:
                columns = cols

            if not data:
                break

            rows.extend(data)
            start_row += len(data)

    if not rows or not columns:
        return pd.DataFrame(columns=["begin", "open", "high", "low", "close"])

    df = pd.DataFrame(rows, columns=columns)
    df["begin"] = pd.to_datetime(df["begin"])
    df = df.sort_values("begin")
    return df


def download_benchmarks(
    out_dir: Path,
    start: str,
    end: str,
    specs: list[BenchmarkSpec],
    interval: int = 24,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for sp in specs:
        df = _fetch_candles(
            secid=sp.secid,
            start=start,
            end=end,
            interval=interval,
            engine=sp.engine,
            market=sp.market,
            board=sp.board,
        )
        path = out_dir / f"{sp.secid}.parquet"
        df.to_parquet(path, index=False)
        print(f"[bench] {sp.secid}: {len(df):,} rows -> {path}")


def load_benchmark_equity(
    bench_path: Path,
    initial_capital: float,
) -> pd.Series:
    """
    Read benchmark parquet and convert index level into 'equity' scaled to initial_capital.
    Uses close price.
    """
    df = pd.read_parquet(bench_path)
    if df.empty:
        return pd.Series(dtype=float)

    df = df.sort_values("begin")
    close = pd.to_numeric(df["close"], errors="coerce")
    close.index = pd.to_datetime(df["begin"])
    close = close.dropna()
    if close.empty:
        return pd.Series(dtype=float)

    eq = initial_capital * (close / close.iloc[0])
    eq.name = bench_path.stem
    return eq
