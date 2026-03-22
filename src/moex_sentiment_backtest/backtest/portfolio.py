from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
import datetime as dt


@dataclass(frozen=True)
class PortfolioConfig:
    initial_capital: float
    max_positions: int
    sizing_mode: Literal["fixed_cash", "fraction_of_equity"]
    fixed_cash: float
    fraction: float
    allow_multiple_positions_per_ticker: bool


@dataclass(frozen=True)
class ShortingConfig:
    enable_shorts: bool
    borrow_rate_annual: float
    rebate_rate_annual: float
    margin_pct: float          # extra margin as fraction of short notional (0.5 => 150% initial margin)
    collateralize_proceeds: bool


@dataclass(frozen=True)
class CostsConfig:
    enable: bool
    commission_bps: float


def _commission_cost(entry_value: float, exit_value: float, commission_bps: float) -> float:
    return (entry_value + exit_value) * (commission_bps / 1e4)


def _short_financing_cost(
    notional: float,
    hold_days: float,
    borrow_rate_annual: float,
    rebate_rate_annual: float,
) -> float:
    net = borrow_rate_annual - rebate_rate_annual
    if net <= 0:
        return 0.0
    return notional * net * (hold_days / 365.0)

def _to_dt64_ns(x) -> np.datetime64:
    """Normalize various datetime-like objects to numpy datetime64[ns]."""
    if x is None:
        return np.datetime64("NaT")
    if isinstance(x, np.datetime64):
        return x.astype("datetime64[ns]")
    if isinstance(x, dt.datetime):
        return np.datetime64(x, "ns")
    if isinstance(x, dt.date):
        # date -> midnight datetime
        return np.datetime64(dt.datetime.combine(x, dt.time.min), "ns")
    # fallback: let numpy try
    return np.datetime64(x).astype("datetime64[ns]")

def build_portfolio_equity(
    candidates: pl.DataFrame,
    portfolio: PortfolioConfig,
    shorting: ShortingConfig,
    costs: CostsConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Select candidate trades under cash/margin constraints and build an equity curve.

    Equity definition here (conservative but avoids fake drawdowns from cash flows):
      equity = free_cash + long_book_value_at_entry + reserved_short_margin

    We do *not* mark-to-market open positions; equity changes only when a trade is closed
    (realized P&L). This is often OK for news-driven event studies and keeps runtime small.

    Position sizing supports per-signal `weight` (e.g., abs(sentiment)) when present.
    """
    if candidates.is_empty():
        return candidates, pl.DataFrame()

    c = candidates.sort(["entry_ts", "ticker"]).with_row_index(name="row_id")
    rows = c.to_dicts()

    free_cash = float(portfolio.initial_capital)
    long_book = 0.0
    reserved_margin = 0.0

    open_positions: list[dict] = []

    executed_rows: list[int] = []
    executed_notional: list[float] = []
    executed_pnl: list[float] = []
    executed_commission: list[float] = []
    executed_financing: list[float] = []

    eq_ts: list[np.datetime64] = []
    eq_val: list[float] = []

    def current_equity() -> float:
        return free_cash + long_book + reserved_margin

    def close_until(ts: np.datetime64) -> None:
        nonlocal free_cash, long_book, reserved_margin, open_positions
        still_open: list[dict] = []
        for pos in open_positions:
            ts64 = _to_dt64_ns(ts)
            if _to_dt64_ns(pos["exit_ts"]) <= ts64:
                free_cash += pos["cash_delta"]
                long_book += pos["long_book_delta"]
                reserved_margin += pos["margin_delta"]
                eq_ts.append(pos["exit_ts"])
                eq_val.append(current_equity())
            else:
                still_open.append(pos)
        open_positions = still_open

    for row in rows:
        entry_ts = row["entry_ts"]
        exit_ts = row["exit_ts"]
        ticker = str(row["ticker"])
        direction = int(row["direction"])
        weight = float(row.get("weight", 1.0))

        close_until(entry_ts)

        if len(open_positions) >= portfolio.max_positions:
            continue
        if (not portfolio.allow_multiple_positions_per_ticker) and any(p["ticker"] == ticker for p in open_positions):
            continue
        if direction == -1 and (not shorting.enable_shorts):
            continue

        equity = current_equity()
        if portfolio.sizing_mode == "fixed_cash":
            base_notional = portfolio.fixed_cash
        else:
            base_notional = max(0.0, equity * portfolio.fraction)

        notional = base_notional * max(0.0, weight)
        if notional <= 0:
            continue

        entry_px = float(row["entry_px"])
        exit_px = float(row["exit_px"])
        shares = notional / entry_px
        entry_value = shares * entry_px
        exit_value = shares * exit_px

        hold_minutes = int(row.get("hold_minutes", row.get("duration_minutes", 0)))

        hold_days = hold_minutes / (60.0 * 24.0)

        commission = _commission_cost(entry_value, exit_value, costs.commission_bps if costs.enable else 0.0)
        financing = 0.0

        if direction == 1:
            if free_cash < entry_value:
                continue

            free_cash -= entry_value
            long_book += entry_value
            eq_ts.append(entry_ts)
            eq_val.append(current_equity())

            pnl = exit_value - entry_value
            pnl_net = pnl - commission

            cash_delta = exit_value - commission
            long_book_delta = -entry_value
            margin_delta = 0.0
        else:
            margin_deposit = entry_value * shorting.margin_pct
            if free_cash < margin_deposit:
                continue

            free_cash -= margin_deposit
            reserved_margin += margin_deposit
            eq_ts.append(entry_ts)
            eq_val.append(current_equity())

            financing = _short_financing_cost(
                notional=entry_value,
                hold_days=hold_days,
                borrow_rate_annual=shorting.borrow_rate_annual,
                rebate_rate_annual=shorting.rebate_rate_annual,
            )

            pnl = entry_value - exit_value
            pnl_net = pnl - commission - financing

            cash_delta = margin_deposit + pnl - commission - financing
            long_book_delta = 0.0
            margin_delta = -margin_deposit

        open_positions.append(
            {
                "ticker": ticker,
                "exit_ts": exit_ts,
                "cash_delta": cash_delta,
                "long_book_delta": long_book_delta,
                "margin_delta": margin_delta,
            }
        )

        executed_rows.append(int(row["row_id"]))
        executed_notional.append(float(entry_value))
        executed_pnl.append(float(pnl_net))
        executed_commission.append(float(commission))
        executed_financing.append(float(financing))

    close_until(np.datetime64("3000-01-01"))

    executed = c.filter(pl.col("row_id").is_in(executed_rows)).sort(["entry_ts", "ticker"])
    if executed.is_empty():
        return executed, pl.DataFrame()

    executed = executed.with_columns(
        pl.Series(name="notional", values=executed_notional),
        pl.Series(name="pnl_net", values=executed_pnl),
        pl.Series(name="commission", values=executed_commission),
        pl.Series(name="financing", values=executed_financing),
    )

    equity_curve = pl.DataFrame({"ts": eq_ts, "equity": eq_val}).sort("ts")
    return executed, equity_curve
