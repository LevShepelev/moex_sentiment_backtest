from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Union

import polars as pl


@dataclass(frozen=True)
class SignalSpec:
    name: str
    long_threshold: int
    short_threshold: int
    take_profit_pct: float
    stop_loss_pct: float
    size_by_sentiment: bool

    # time exit
    time_exit_mode: Literal["next_day", "n_days"]
    time_exit_n_days: Optional[int]
    time_exit_price: Literal["open", "close"]

    # NEW knobs (implemented)
    entry_delay_minutes: int = 0              # shift entry timestamp forward
    cooldown_minutes: int = 0                 # per-ticker cooldown between accepted signals
    per_ticker_daily_limit: int = 0           # 0 = unlimited, else max signals per (ticker, day)
    daily_top_k: int = 0                      # 0 = disabled, else keep top K signals per day by score

    def allows_short(self) -> bool:
        return self.short_threshold < 0


SpecLike = Union[SignalSpec, Mapping[str, Any]]


def _schema_names(frame: pl.DataFrame | pl.LazyFrame) -> list[str]:
    if isinstance(frame, pl.DataFrame):
        return frame.columns
    try:
        return frame.collect_schema().names()
    except Exception:
        return frame.columns


def _ensure_event_id(events: pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    lf = events.lazy() if isinstance(events, pl.DataFrame) else events
    cols = _schema_names(lf)
    if "event_id" in cols:
        return lf
    if hasattr(lf, "with_row_index"):
        return lf.with_row_index("event_id")
    return lf.with_row_count("event_id")


def _as_spec(s: SpecLike) -> SignalSpec:
    if isinstance(s, SignalSpec):
        return s

    def _int(x, d=0):
        return d if x in (None, "", "null") else int(x)

    def _float(x, d=0.0):
        return d if x in (None, "", "null") else float(x)

    return SignalSpec(
        name=str(s.get("name", "strategy")),
        long_threshold=int(s.get("long_threshold", 1)),
        short_threshold=int(s.get("short_threshold", -1)),
        take_profit_pct=_float(s.get("take_profit_pct", 0.03)),
        stop_loss_pct=_float(s.get("stop_loss_pct", 0.01)),
        size_by_sentiment=bool(s.get("size_by_sentiment", False)),
        time_exit_mode=str(s.get("time_exit_mode", "next_day")),   # type: ignore[arg-type]
        time_exit_n_days=None if s.get("time_exit_n_days") in (None, "", "null") else int(s["time_exit_n_days"]),
        time_exit_price=str(s.get("time_exit_price", "open")),     # type: ignore[arg-type]

        entry_delay_minutes=_int(s.get("entry_delay_minutes", 0), 0),
        cooldown_minutes=_int(s.get("cooldown_minutes", 0), 0),
        per_ticker_daily_limit=_int(s.get("per_ticker_daily_limit", 0), 0),
        daily_top_k=_int(s.get("daily_top_k", 0), 0),
    )


def make_signals(events: pl.DataFrame | pl.LazyFrame, sigspec: SpecLike) -> pl.DataFrame:
    spec = _as_spec(sigspec)

    lf = _ensure_event_id(events).select(["event_id", "ts", "ticker", "sentiment"])

    # direction
    lf = lf.with_columns(
        pl.when(pl.col("sentiment") >= spec.long_threshold)
        .then(pl.lit(1))
        .when(pl.col("sentiment") <= spec.short_threshold)
        .then(pl.lit(-1))
        .otherwise(pl.lit(0))
        .alias("direction")
    ).filter(pl.col("direction") != 0)

    # weight / signal_strength
    if spec.size_by_sentiment:
        lf = lf.with_columns(pl.col("sentiment").abs().cast(pl.Float64).alias("signal_strength"))
    else:
        lf = lf.with_columns(pl.lit(1.0).alias("signal_strength"))
    lf = lf.with_columns(pl.col("signal_strength").alias("weight"))

    # entry delay: shift ts forward
    if spec.entry_delay_minutes > 0:
        lf = lf.with_columns(
            (pl.col("ts") + pl.duration(minutes=spec.entry_delay_minutes)).alias("ts")
        )

    # per-ticker cooldown
    if spec.cooldown_minutes > 0:
        lf = lf.sort(["ticker", "ts"]).with_columns(
            (pl.col("ts") - pl.col("ts").shift(1)).dt.total_minutes().over("ticker").alias("_dt_min")
        ).filter(
            pl.col("_dt_min").is_null() | (pl.col("_dt_min") >= spec.cooldown_minutes)
        ).drop("_dt_min")

    # per-ticker daily limit
    if spec.per_ticker_daily_limit and spec.per_ticker_daily_limit > 0:
        lf = lf.with_columns(pl.col("ts").dt.date().alias("_day")).sort(["ticker", "_day", "ts"]).with_columns(
            pl.int_range(0, pl.len()).over(["ticker", "_day"]).alias("_k")
        ).filter(pl.col("_k") < spec.per_ticker_daily_limit).drop(["_day", "_k"])

    # daily top-k by score (abs(sentiment)*weight)
    if spec.daily_top_k and spec.daily_top_k > 0:
        lf = lf.with_columns(
            pl.col("ts").dt.date().alias("_day"),
            (pl.col("sentiment").abs().cast(pl.Float64) * pl.col("weight")).alias("_score"),
        ).with_columns(
            pl.col("_score").rank(method="dense", descending=True).over("_day").alias("_rank")
        ).filter(pl.col("_rank") <= spec.daily_top_k).drop(["_day", "_score", "_rank"])

    # attach simulator-required params and compatibility names
    exit_n = -1 if spec.time_exit_n_days is None else int(spec.time_exit_n_days)

    lf = lf.with_columns(
        pl.lit(float(spec.take_profit_pct)).alias("take_profit_pct"),
        pl.lit(float(spec.stop_loss_pct)).alias("stop_loss_pct"),

        pl.lit(str(spec.time_exit_mode)).alias("time_exit_mode"),
        pl.lit(exit_n).alias("time_exit_n_days"),
        pl.lit(str(spec.time_exit_price)).alias("time_exit_price"),

        pl.lit(str(spec.time_exit_mode)).alias("exit_mode"),
        pl.lit(exit_n).alias("exit_n_days"),
        pl.lit(str(spec.time_exit_price)).alias("exit_price"),

        pl.lit(str(spec.name)).alias("strategy"),
    )

    return lf.select(
        [
            "event_id",
            "ts",
            "ticker",
            "sentiment",
            "direction",
            "weight",
            "signal_strength",
            "take_profit_pct",
            "stop_loss_pct",
            "exit_mode",
            "exit_n_days",
            "exit_price",
            "strategy",
        ]
    ).collect()
