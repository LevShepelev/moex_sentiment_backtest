from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from moex_sentiment_backtest.backtest.calendar import TradingCalendarIndex
from moex_sentiment_backtest.backtest.engine import ExecutionCosts, simulate_trades
from moex_sentiment_backtest.data.moex import load_prices_as_numpy

logger = logging.getLogger(__name__)

# Canonical candidate schema used downstream (portfolio/metrics/report).
CANDIDATE_SCHEMA: list[tuple[str, pl.DataType]] = [
    ("event_id", pl.Int64),
    ("ticker", pl.Utf8),
    ("strategy", pl.Utf8),
    ("sentiment", pl.Int8),
    ("weight", pl.Float64),
    ("direction", pl.Int8),
    ("signal_ts", pl.Datetime),

    ("entry_ts", pl.Datetime),
    ("exit_ts", pl.Datetime),
    ("entry_px", pl.Float64),
    ("exit_px", pl.Float64),

    ("exit_reason", pl.Utf8),
    ("gross_ret", pl.Float64),
    ("gross_pnl", pl.Float64),
    ("hold_minutes", pl.Int32),
]


def _empty_candidates_df() -> pl.DataFrame:
    """Create an empty DataFrame with canonical schema."""
    return pl.DataFrame({name: pl.Series(name, [], dtype=dtype) for name, dtype in CANDIDATE_SCHEMA})


def _ensure_candidate_schema(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure dataframe has all required columns with correct dtypes.
    Missing columns are added as nulls; then columns are ordered canonically.
    """
    if df.height == 0:
        # return true empty with schema
        return _empty_candidates_df()

    for name, dtype in CANDIDATE_SCHEMA:
        if name not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(name))
        else:
            # try to cast to canonical dtype
            try:
                df = df.with_columns(pl.col(name).cast(dtype).alias(name))
            except Exception:
                pass

    return df.select([n for n, _ in CANDIDATE_SCHEMA])


@dataclass(frozen=True)
class CandidateTradeTable:
    """A compact trade table (one row per signal) with entry/exit and gross returns."""

    df: pl.DataFrame

    def write_parquet(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.write_parquet(path, compression="zstd")


def _simulate_one_ticker(
    ticker: str,
    sig: pl.DataFrame,
    prices_path: Path,
    costs: ExecutionCosts,
    intrabar_priority: str,
    missing_bars: str,
) -> pl.DataFrame:
    """
    Simulate trades for one ticker based on signal rows.
    Expects signals to contain columns at least:
      ts, event_id, sentiment, weight, direction, strategy,
      take_profit_pct, stop_loss_pct, exit_mode, exit_n_days, exit_price
    """
    if sig.height == 0:
        return _empty_candidates_df()

    arr = load_prices_as_numpy(prices_path)
    ts = arr["ts"]  # numpy datetime64 array
    ts_i64 = ts.astype(np.int64)

    open_ = arr["open"]
    high = arr["high"]
    low = arr["low"]

    cal = TradingCalendarIndex.build(ts)

    # Per-signal arrays
    signal_ts = sig.get_column("ts").to_numpy()
    signal_ts_i64 = signal_ts.astype(np.int64)

    direction = sig.get_column("direction").to_numpy().astype(np.int64)
    tp = sig.get_column("take_profit_pct").to_numpy().astype(np.float64)
    sl = sig.get_column("stop_loss_pct").to_numpy().astype(np.float64)

    exit_mode = sig.get_column("exit_mode").to_list()
    exit_n_days = sig.get_column("exit_n_days").to_numpy().astype(np.int64)
    exit_price = sig.get_column("exit_price").to_list()

    # Entry index = first bar with ts >= signal_ts
    entry_idx = np.searchsorted(ts_i64, signal_ts_i64)

    valid = entry_idx < ts.shape[0]
    if not valid.all():
        if missing_bars == "raise":
            raise ValueError(f"{ticker}: some signals occur after last available bar.")
        # filter invalid signals
        mask = valid.tolist()
        sig = sig.filter(pl.Series("valid", mask))
        entry_idx = entry_idx[valid]
        direction = direction[valid]
        tp = tp[valid]
        sl = sl[valid]
        exit_n_days = exit_n_days[valid]
        exit_mode = [exit_mode[i] for i, ok in enumerate(mask) if ok]
        exit_price = [exit_price[i] for i, ok in enumerate(mask) if ok]

        if sig.height == 0:
            return _empty_candidates_df()

    # Time-exit indices using calendar logic
    time_exit_idx = np.empty_like(entry_idx)
    for k in range(entry_idx.shape[0]):
        mode = str(exit_mode[k])
        n = int(exit_n_days[k]) if int(exit_n_days[k]) > 0 else 1
        price = str(exit_price[k])
        ei = int(entry_idx[k])
        time_exit_idx[k] = cal.time_exit_index(entry_idx=ei, mode=mode, n_days=n, price=price)

    time_exit_idx = np.maximum(time_exit_idx, entry_idx)

    intrabar_stop_first = intrabar_priority == "stop_first"
    half_spread = costs.half_spread_rate() if costs.enable else 0.0
    slip = costs.slippage_rate() if costs.enable else 0.0

    exit_idx, entry_fill, exit_fill, reason = simulate_trades(
        open_=open_,
        high=high,
        low=low,
        entry_idx=entry_idx.astype(np.int64),
        time_exit_idx=time_exit_idx.astype(np.int64),
        direction=direction,
        take_profit_pct=tp,
        stop_loss_pct=sl,
        intrabar_stop_first=intrabar_stop_first,
        half_spread=half_spread,
        slip=slip,
    )

    entry_ts = ts[entry_idx]
    exit_ts = ts[exit_idx]

    gross_ret = direction * (exit_fill - entry_fill) / entry_fill
    gross_pnl = direction * (exit_fill - entry_fill)  # per-share/unit

    reason_map = {1: "TP", 2: "SL", 3: "TIME"}
    reason_str = [reason_map.get(int(r), "TIME") for r in reason.tolist()]

    # Build output with required columns
    base = sig.select(
        [
            "event_id",
            "ts",
            "strategy",
            "sentiment",
            "weight",
            "direction",
        ]
    ).rename({"ts": "signal_ts"})

    # build first (no referencing newly created cols)
    out = base.with_columns(
        pl.lit(str(ticker)).alias("ticker"),
        pl.Series(name="entry_ts", values=entry_ts),
        pl.Series(name="exit_ts", values=exit_ts),
        pl.Series(name="entry_px", values=entry_fill),
        pl.Series(name="exit_px", values=exit_fill),
        pl.Series(name="exit_reason", values=reason_str),
        pl.Series(name="gross_ret", values=gross_ret.astype(np.float64)),
        pl.Series(name="gross_pnl", values=gross_pnl.astype(np.float64)),
    )

    # compute hold_minutes safely (numpy)
    hold_minutes = ((exit_ts - entry_ts) / np.timedelta64(1, "m")).astype(np.int32)
    out = out.with_columns(pl.Series(name="hold_minutes", values=hold_minutes))

    return _ensure_candidate_schema(out)



def simulate_candidates(
    signals: pl.DataFrame,
    prices_dir: Path,
    costs: ExecutionCosts,
    intrabar_priority: str,
    n_jobs: int = 1,
    missing_prices: str = "skip",
    missing_bars: str = "skip",
) -> CandidateTradeTable:
    if signals.height == 0:
        return CandidateTradeTable(df=_empty_candidates_df())

    tickers = [str(t) for t in signals.get_column("ticker").unique().to_list()]

    tasks: list[tuple[str, pl.DataFrame, Path]] = []
    for t in tickers:
        p = prices_dir / f"{t}.parquet"
        if not p.exists():
            if missing_prices == "raise":
                raise FileNotFoundError(p)
            logger.warning("Missing prices for %s -> skipping", t)
            continue
        sig_t = signals.filter(pl.col("ticker") == t)
        if sig_t.height == 0:
            continue
        tasks.append((t, sig_t, p))

    if not tasks:
        return CandidateTradeTable(df=_empty_candidates_df())

    results: list[pl.DataFrame] = []
    if n_jobs <= 1:
        for t, sig_t, p in tasks:
            r = _simulate_one_ticker(
                ticker=t,
                sig=sig_t,
                prices_path=p,
                costs=costs,
                intrabar_priority=intrabar_priority,
                missing_bars=missing_bars,
            )
            if r.height:
                results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [
                ex.submit(_simulate_one_ticker, t, sig_t, p, costs, intrabar_priority, missing_bars)
                for (t, sig_t, p) in tasks
            ]
            for f in futs:
                r = f.result()
                if r.height:
                    results.append(r)

    if not results:
        return CandidateTradeTable(df=_empty_candidates_df())

    all_trades = pl.concat(results, how="vertical").sort(["entry_ts", "ticker"])
    all_trades = _ensure_candidate_schema(all_trades)
    return CandidateTradeTable(df=all_trades)
