from __future__ import annotations

import numpy as np

from moex_sentiment_backtest.backtest.calendar import TradingCalendarIndex


def test_calendar_exit_index() -> None:
    ts = np.array(
        [
            np.datetime64("2024-01-01T10:00"), np.datetime64("2024-01-01T10:01"),
            np.datetime64("2024-01-02T10:00"), np.datetime64("2024-01-02T10:01"),
            np.datetime64("2024-01-05T10:00"), np.datetime64("2024-01-05T10:01"),
        ],
        dtype="datetime64[ns]"
    )
    cal = TradingCalendarIndex.build(ts)
    entry_idx = 1
    # next_day open => first bar of Jan 2
    assert cal.time_exit_index(entry_idx, mode="next_day", n_days=1, price="open") == 2
    # n_days=3 from Jan 1 => Jan 4 target, should fall forward to Jan 5 open
    assert cal.time_exit_index(entry_idx, mode="n_days", n_days=3, price="open") == 4
