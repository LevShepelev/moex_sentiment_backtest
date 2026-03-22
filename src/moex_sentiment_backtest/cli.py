from __future__ import annotations

from pathlib import Path
from typing import Optional

import fire

from moex_sentiment_backtest.logging_config import setup_logging


def prepare_news(
    news_path: str,
    out: str,
    keep_text: bool = False,
    date_column: str | None = None,
    ticker_columns: str | None = None,
    sheet_name: str | int | None = None,
    all_sheets: bool = False,
    log_level: str = "INFO",
) -> None:
    setup_logging(log_level)
    from moex_sentiment_backtest.data.news import prepare_events

    tickers = None
    if ticker_columns:
        tickers = [x.strip() for x in ticker_columns.split(",") if x.strip()]

    prepare_events(
        news_path=Path(news_path),
        out_path=Path(out),
        keep_text=keep_text,
        date_column=date_column,
        ticker_columns=tickers,
        sheet_name=sheet_name,
        all_sheets=all_sheets,
    )
    print(f"Saved events -> {out}")

def download_prices(
    events: str,
    start: str,
    end: str,
    out_dir: str,
    interval: int = 1,
    chunk_days: int = 5,
    sleep_s: float = 0.2,
    n_tickers: Optional[int] = None,
    log_level: str = "INFO",
) -> None:
    setup_logging(log_level)
    from moex_sentiment_backtest.data.moex import download_prices_for_events

    download_prices_for_events(
        events_path=Path(events),
        start=start,
        end=end,
        out_dir=Path(out_dir),
        interval=interval,
        chunk_days=chunk_days,
        sleep_s=sleep_s,
        n_tickers=n_tickers,
    )
    print(f"Saved prices -> {out_dir}")
def download_benchmarks(
    start: str,
    end: str,
    out_dir: str = "data/benchmarks",
    secids: Union[str, Sequence[str]] = ("IMOEX", "RGBITR"),
    log_level: str = "INFO",
) -> None:
    setup_logging(log_level)
    from moex_sentiment_backtest.data.benchmarks import BenchmarkSpec, download_benchmarks as dl

    if isinstance(secids, str):
        ids = [x.strip() for x in secids.split(",") if x.strip()]
    else:
        # Fire passes multiple args as tuple -> handle it
        ids = [str(x).strip() for x in secids if str(x).strip()]

    specs = [BenchmarkSpec(name=i, secid=i) for i in ids]
    dl(out_dir=Path(out_dir), start=start, end=end, specs=specs, interval=24)

def run(
    events: str,
    prices_dir: str,
    config: str,
    out_dir: str,
    n_jobs: int = 1,
    log_level: str = "INFO",
) -> None:
    setup_logging(log_level)
    from moex_sentiment_backtest.backtest.report import run_suite

    run_suite(
        events_path=Path(events),
        prices_dir=Path(prices_dir),
        config_path=Path(config),
        out_dir=Path(out_dir),
        n_jobs=n_jobs,
    )
    print(f"Report saved -> {out_dir}")


def main() -> None:
    fire.Fire(
    {
        "prepare-news": prepare_news,
        "download-prices": download_prices,
        "download-benchmarks": download_benchmarks,  # <-- add
        "run": run,
    }
    )

