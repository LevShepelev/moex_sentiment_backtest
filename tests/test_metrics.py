from __future__ import annotations

import numpy as np
import pandas as pd

from moex_sentiment_backtest.backtest.metrics import compute_metrics


def test_compute_metrics_smoke() -> None:
    r = np.array([0.01, -0.005, 0.02, -0.01, 0.0], dtype=float)
    idx = pd.date_range("2024-01-01", periods=6, freq="D")
    eq = pd.Series([100, 101, 100.5, 102.5, 101.5, 103], index=idx)

    m = compute_metrics(r, eq, risk_free_rate_annual=0.0, trading_days_per_year=252)
    assert m.n_trades == 5
    assert 0.0 <= m.win_rate <= 1.0
