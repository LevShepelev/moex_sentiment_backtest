from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from moex_sentiment_backtest.backtest.engine import ExecutionCosts
from moex_sentiment_backtest.backtest.metrics import compute_metrics
from moex_sentiment_backtest.backtest.portfolio import (
    CostsConfig,
    PortfolioConfig,
    ShortingConfig,
    build_portfolio_equity,
)
from moex_sentiment_backtest.backtest.signals import SignalSpec, make_signals
from moex_sentiment_backtest.backtest.simulate import simulate_candidates
from moex_sentiment_backtest.data.benchmarks import load_benchmark_equity
from moex_sentiment_backtest.settings import AppConfig
from moex_sentiment_backtest.utils import read_yaml
from moex_sentiment_backtest.viz.plots import (
    plot_drawdowns,
    plot_equity_curves,
    plot_trade_return_hist,
)

logger = logging.getLogger(__name__)


def _rel_stats(strategy_eq: pd.Series, bench_eq: pd.Series) -> dict[str, float]:
    """
    Compute beta/alpha/info ratio and CAGR excess vs benchmark using daily resampled equity.
    """
    s = strategy_eq.sort_index()
    b = bench_eq.sort_index()

    s = s[~s.index.duplicated(keep="last")]
    b = b[~b.index.duplicated(keep="last")]

    s = s.resample("1D").ffill().dropna()
    b = b.resample("1D").ffill().dropna()

    s, b = s.align(b, join="inner")
    if len(s) < 10:
        return {}

    rs = s.pct_change().dropna()
    rb = b.pct_change().dropna()
    rs, rb = rs.align(rb, join="inner")
    if len(rs) < 10:
        return {}

    varb = float(np.var(rb))
    beta = float(np.cov(rs, rb)[0, 1] / varb) if varb > 0 else 0.0
    alpha_daily = float(rs.mean() - beta * rb.mean())
    alpha_ann = alpha_daily * 252.0

    ex = rs - rb
    ex_std = float(ex.std(ddof=1))
    ir = float(ex.mean() * 252.0 / (ex_std * np.sqrt(252.0))) if ex_std > 0 else 0.0

    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr_s = float((s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    cagr_b = float((b.iloc[-1] / b.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0

    return {
        "beta": beta,
        "alpha_ann": alpha_ann,
        "info_ratio": ir,
        "bench_cagr": cagr_b,
        "cagr_excess": cagr_s - cagr_b,
    }


def run_suite(
    events_path: Path,
    prices_dir: Path,
    config_path: Path,
    out_dir: Path,
    n_jobs: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_cfg = read_yaml(config_path)
    cfg = AppConfig.model_validate(raw_cfg)

    events = pl.read_parquet(events_path)

    metrics_rows: list[dict] = []
    equity_curves: dict[str, pd.Series] = {}
    trade_returns: dict[str, list[float]] = {}

    # Load benchmarks if present
    bench_dir = Path("data/benchmarks")
    benchmarks: dict[str, pd.Series] = {}
    for secid in ["IMOEX", "RGBITR"]:
        p = bench_dir / f"{secid}.parquet"
        if p.exists():
            benchmarks[secid] = load_benchmark_equity(
                p,
                initial_capital=float(cfg.portfolio.initial_capital),
            )

    for spec in cfg.strategies:
        logger.info("Running strategy: %s", spec.name)

        sigspec = SignalSpec(
            name=spec.name,
            long_threshold=spec.long_threshold,
            short_threshold=spec.short_threshold,
            take_profit_pct=spec.take_profit_pct,
            stop_loss_pct=spec.stop_loss_pct,
            size_by_sentiment=spec.size_by_sentiment,
            time_exit_mode=spec.time_exit.mode,
            time_exit_n_days=spec.time_exit.n_days,
            time_exit_price=spec.time_exit.price,
            entry_delay_minutes=spec.entry_delay_minutes,
            cooldown_minutes=spec.cooldown_minutes,
            per_ticker_daily_limit=spec.per_ticker_daily_limit,
            daily_top_k=spec.daily_top_k,
        )

        signals = make_signals(events, sigspec)

        costs = ExecutionCosts(
            enable=cfg.execution.enable_costs,
            spread_bps=cfg.execution.spread_bps,
            commission_bps=cfg.execution.commission_bps,
            slippage_bps=cfg.execution.slippage_bps,
        )

        candidates_tbl = simulate_candidates(
            signals=signals,
            prices_dir=prices_dir,
            costs=costs,
            intrabar_priority=cfg.execution.intrabar_priority,
            n_jobs=n_jobs,
            missing_prices=cfg.data.missing_prices,
            missing_bars=cfg.data.missing_bars,
        )

        candidates = candidates_tbl.df
        if candidates.is_empty():
            logger.warning("No candidate trades for strategy %s", spec.name)
            continue

        port_cfg = PortfolioConfig(
            initial_capital=float(cfg.portfolio.initial_capital),
            max_positions=int(cfg.portfolio.max_positions),
            sizing_mode=cfg.portfolio.position_sizing.mode,
            fixed_cash=float(cfg.portfolio.position_sizing.fixed_cash),
            fraction=float(cfg.portfolio.position_sizing.fraction),
            allow_multiple_positions_per_ticker=bool(cfg.portfolio.allow_multiple_positions_per_ticker),
        )

        short_cfg = ShortingConfig(
            enable_shorts=bool(cfg.shorting.enable_shorts),
            borrow_rate_annual=float(cfg.shorting.short_borrow_rate_annual),
            rebate_rate_annual=float(cfg.shorting.short_rebate_rate_annual),
            margin_pct=float(cfg.shorting.short_margin_pct),
            collateralize_proceeds=bool(cfg.shorting.collateralize_proceeds),
        )

        costs_cfg = CostsConfig(
            enable=bool(cfg.execution.enable_costs),
            commission_bps=float(cfg.execution.commission_bps),
        )

        executed, equity_curve_pl = build_portfolio_equity(
            candidates=candidates,
            portfolio=port_cfg,
            shorting=short_cfg,
            costs=costs_cfg,
        )

        executed.write_parquet(out_dir / f"trades_{spec.name}.parquet", compression="zstd")

        equity_curve = pd.Series(dtype=float)
        if not equity_curve_pl.is_empty():
            ec = equity_curve_pl.to_pandas()
            ec["ts"] = pd.to_datetime(ec["ts"])
            equity_curve = pd.Series(ec["equity"].values, index=ec["ts"], name=spec.name).sort_index()
            equity_curves[spec.name] = equity_curve

        # trade returns for histogram + metrics
        tr = executed.get_column("pnl_net").to_numpy() / executed.get_column("notional").to_numpy()
        trade_returns[spec.name] = tr.tolist()

        m = compute_metrics(
            trade_returns=tr,
            equity_curve=equity_curve if not equity_curve.empty else None,
            risk_free_rate_annual=float(cfg.metrics.risk_free_rate_annual),
            trading_days_per_year=int(cfg.metrics.trading_days_per_year),
            hold_minutes=executed.get_column("hold_minutes").to_numpy(),
        )

        row = {
            "strategy": spec.name,
            "n_trades": m.n_trades,
            "win_rate": m.win_rate,
            "avg_trade_return": m.avg_trade_return,
            "median_trade_return": m.median_trade_return,
            "median_hold_minutes": m.median_hold_minutes,
            "profit_factor": m.profit_factor,
            "expectancy": m.expectancy,
            "max_drawdown": m.max_drawdown,
            "cagr": m.cagr,
            "vol_annual": m.vol_annual,
            "sharpe": m.sharpe,
            "sortino": m.sortino,
            "calmar": m.calmar,
        }

        # add benchmark-relative stats if we have equity curve
        if not equity_curve.empty and benchmarks:
            for bname, beq in benchmarks.items():
                st = _rel_stats(equity_curve, beq)
                for k, v in st.items():
                    row[f"{bname}_{k}"] = v

        metrics_rows.append(row)

    # Save metrics
    metrics_df = pd.DataFrame(metrics_rows)
    if not metrics_df.empty and "sharpe" in metrics_df.columns:
        metrics_df = metrics_df.sort_values(by=["sharpe"], ascending=False)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)

    # Plots (overlay benchmarks on equity curves)
    if equity_curves:
        plot_equity_curves(equity_curves, out_dir / "equity_curves.png", benchmarks=benchmarks if benchmarks else None)
        plot_drawdowns(equity_curves, out_dir / "drawdowns.png")

    if trade_returns:
        plot_trade_return_hist(trade_returns, out_dir / "trade_return_hist.png")
