from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    n_trades: int
    win_rate: float
    avg_trade_return: float
    median_trade_return: float
    profit_factor: float
    expectancy: float
    max_drawdown: float
    cagr: float
    vol_annual: float
    sharpe: float
    sortino: float
    calmar: float


def _drawdown(equity: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd


def _annualize_return(total_return: float, days: float) -> float:
    if days <= 0:
        return 0.0
    years = days / 365.0
    if years <= 0:
        return 0.0
    return (1.0 + total_return) ** (1.0 / years) - 1.0


def compute_metrics(
    trade_returns: np.ndarray,
    equity_curve: Optional[pd.Series],
    risk_free_rate_annual: float = 0.0,
    trading_days_per_year: int = 252,
) -> PerformanceMetrics:
    trade_returns = trade_returns[~np.isnan(trade_returns)]
    n = int(trade_returns.shape[0])
    if n == 0:
        return PerformanceMetrics(
            n_trades=0,
            win_rate=0.0,
            avg_trade_return=0.0,
            median_trade_return=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            max_drawdown=0.0,
            cagr=0.0,
            vol_annual=0.0,
            sharpe=0.0,
            sortino=0.0,
            calmar=0.0,
        )

    win_rate = float((trade_returns > 0).mean())
    avg_r = float(trade_returns.mean())
    med_r = float(np.median(trade_returns))

    gains = trade_returns[trade_returns > 0].sum()
    losses = -trade_returns[trade_returns < 0].sum()
    profit_factor = float(gains / losses) if losses > 0 else float("inf")
    expectancy = avg_r

    max_dd = 0.0
    cagr = 0.0
    vol_ann = 0.0
    sharpe = 0.0
    sortino = 0.0
    calmar = 0.0

    if equity_curve is not None and len(equity_curve) > 2:
        eq = equity_curve.astype(float).values
        dd = _drawdown(eq)
        max_dd = float(dd.min())
        # ---- FIX: resample requires unique, monotonic datetime index ----
        equity_curve = equity_curve.sort_index()

        # If duplicates exist (multiple updates at same timestamp), keep the last one
        if equity_curve.index.has_duplicates:
            equity_curve = equity_curve[~equity_curve.index.duplicated(keep="last")]

        # Ensure DatetimeIndex (sometimes numpy datetime64 can come through as object)
        equity_curve.index = pd.to_datetime(equity_curve.index)

        # daily returns for Sharpe/Sortino: resample to daily, forward-fill equity
        daily = equity_curve.resample("1D").ffill().dropna()
        daily_ret = daily.pct_change().dropna()

        if len(daily_ret) > 5:
            rf_daily = (1.0 + risk_free_rate_annual) ** (1.0 / trading_days_per_year) - 1.0
            excess = daily_ret - rf_daily
            vol_ann = float(daily_ret.std() * np.sqrt(trading_days_per_year))
            if daily_ret.std() > 0:
                sharpe = float(excess.mean() / daily_ret.std() * np.sqrt(trading_days_per_year))

            downside = daily_ret[daily_ret < 0]
            if downside.std() > 0:
                sortino = float(excess.mean() / downside.std() * np.sqrt(trading_days_per_year))

            total_ret = float(daily.iloc[-1] / daily.iloc[0] - 1.0)
            days = float((daily.index[-1] - daily.index[0]).days)
            cagr = _annualize_return(total_ret, days)
            if max_dd < 0:
                calmar = float(cagr / abs(max_dd))

    return PerformanceMetrics(
        n_trades=n,
        win_rate=win_rate,
        avg_trade_return=avg_r,
        median_trade_return=med_r,
        profit_factor=profit_factor,
        expectancy=expectancy,
        max_drawdown=max_dd,
        cagr=cagr,
        vol_annual=vol_ann,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
    )
