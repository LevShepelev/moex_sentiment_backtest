from __future__ import annotations

from moex_sentiment_backtest.settings import StrategySpec


def test_strategy_spec_entry_delay_from_yaml_dict() -> None:
    s = StrategySpec.model_validate(
        {
            "name": "S",
            "entry_delay_minutes": 2,
            "time_exit": {"mode": "next_day", "price": "open"},
        }
    )
    assert s.entry_delay_minutes == 2


def test_strategy_spec_merge_flat_time_exit() -> None:
    s = StrategySpec.model_validate(
        {
            "name": "S",
            "entry_delay_minutes": 1,
            "time_exit_mode": "n_days",
            "time_exit_n_days": 5,
            "time_exit_price": "close",
        }
    )
    assert s.time_exit.mode == "n_days"
    assert s.time_exit.n_days == 5
    assert s.time_exit.price == "close"


def test_strategy_spec_nested_time_exit_wins_over_flat() -> None:
    s = StrategySpec.model_validate(
        {
            "name": "S",
            "time_exit": {"mode": "next_day", "price": "open"},
            "time_exit_mode": "n_days",
            "time_exit_n_days": 99,
            "time_exit_price": "close",
        }
    )
    assert s.time_exit.mode == "next_day"
    assert s.time_exit.price == "open"
