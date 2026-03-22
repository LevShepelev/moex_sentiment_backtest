from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, PositiveFloat


class DataSettings(BaseModel):
    missing_prices: Literal["skip", "raise"] = "skip"
    missing_bars: Literal["skip", "raise"] = "skip"


class ExecutionSettings(BaseModel):
    enable_costs: bool = True
    spread_bps: float = 0.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    intrabar_priority: Literal["stop_first", "tp_first"] = "stop_first"


class ShortingSettings(BaseModel):
    enable_shorts: bool = True
    short_borrow_rate_annual: float = 0.0
    short_rebate_rate_annual: float = 0.0
    short_margin_pct: float = 0.5
    collateralize_proceeds: bool = True


class PositionSizing(BaseModel):
    mode: Literal["fixed_cash", "fraction_of_equity"] = "fraction_of_equity"
    fixed_cash: PositiveFloat = 100000.0
    fraction: float = Field(0.05, ge=0.0, le=1.0)


class PortfolioSettings(BaseModel):
    initial_capital: PositiveFloat = 1_000_000.0
    max_positions: int = Field(10, ge=1)
    position_sizing: PositionSizing = PositionSizing()
    allow_multiple_positions_per_ticker: bool = False


class MetricsSettings(BaseModel):
    risk_free_rate_annual: float = 0.0
    trading_days_per_year: int = 252


class TimeExit(BaseModel):
    mode: Literal["next_day", "n_days"] = "next_day"
    n_days: Optional[int] = Field(default=None, ge=1)
    price: Literal["open", "close"] = "open"


class StrategySpec(BaseModel):
    name: str
    long_threshold: int = 1
    short_threshold: int = -1
    take_profit_pct: float = Field(0.03, ge=0.0, le=1.0)
    stop_loss_pct: float = Field(0.01, ge=0.0, le=1.0)
    size_by_sentiment: bool = False
    time_exit: TimeExit = TimeExit()


class AppConfig(BaseModel):
    data: DataSettings = DataSettings()
    execution: ExecutionSettings = ExecutionSettings()
    shorting: ShortingSettings = ShortingSettings()
    portfolio: PortfolioSettings = PortfolioSettings()
    metrics: MetricsSettings = MetricsSettings()
    strategies: list[StrategySpec] = []
