from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np


@dataclass(frozen=True)
class ExecutionCosts:
    """Execution friction model applied at the trade fill level.

    - spread_bps: quoted *full* spread in bps. We apply half on each side.
    - slippage_bps: additional slippage per side in bps.
    - commission_bps: used later in portfolio layer (value-based); we store it here for convenience.
    """

    enable: bool = True
    spread_bps: float = 0.0
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    def half_spread_rate(self) -> float:
        return (self.spread_bps / 1e4) / 2.0

    def slippage_rate(self) -> float:
        return self.slippage_bps / 1e4

    def commission_rate(self) -> float:
        return self.commission_bps / 1e4


@nb.njit(cache=True)
def _apply_fill_price(px: float, direction: int, half_spread: float, slip: float, is_entry: bool) -> float:
    """Convert a mid-like price to a filled bid/ask price with optional slippage.

    - Long:
        entry = ask => +half_spread
        exit  = bid => -half_spread
    - Short:
        entry = sell at bid => -half_spread
        exit  = cover at ask => +half_spread

    Slippage is applied against the trader (always worse).
    """
    if direction == 1:
        # long
        if is_entry:
            return px * (1.0 + half_spread + slip)
        return px * (1.0 - half_spread - slip)

    # short
    if is_entry:
        return px * (1.0 - half_spread - slip)
    return px * (1.0 + half_spread + slip)


@nb.njit(cache=True)
def _scan_exit(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_idx: int,
    time_exit_idx: int,
    direction: int,
    tp_pct: float,
    sl_pct: float,
    intrabar_stop_first: bool,
) -> tuple[int, float, int]:
    """Scan forward minute bars and return (exit_idx, exit_mid_price, reason_code).

    reason_code:
      1 = take_profit
      2 = stop_loss
      3 = time_exit

    Prices returned are *mid-like* exit prices (before spreads/slippage).
    """
    entry_px = open_[entry_idx]

    # tp_pct / sl_pct are positive numbers (e.g., 0.03 and 0.01)
    if direction == 1:
        tp_px = entry_px * (1.0 + tp_pct)
        sl_px = entry_px * (1.0 - sl_pct)
    else:
        tp_px = entry_px * (1.0 - tp_pct)
        sl_px = entry_px * (1.0 + sl_pct)

    for i in range(entry_idx + 1, time_exit_idx + 1):
        o = open_[i]
        h = high[i]
        l = low[i]

        if direction == 1:
            # gap checks
            if o <= sl_px:
                return i, o, 2
            if o >= tp_px:
                return i, o, 1

            if intrabar_stop_first:
                if l <= sl_px:
                    return i, sl_px, 2
                if h >= tp_px:
                    return i, tp_px, 1
            else:
                if h >= tp_px:
                    return i, tp_px, 1
                if l <= sl_px:
                    return i, sl_px, 2
        else:
            # short: gap checks
            if o >= sl_px:
                return i, o, 2
            if o <= tp_px:
                return i, o, 1

            if intrabar_stop_first:
                if h >= sl_px:
                    return i, sl_px, 2
                if l <= tp_px:
                    return i, tp_px, 1
            else:
                if l <= tp_px:
                    return i, tp_px, 1
                if h >= sl_px:
                    return i, sl_px, 2

    return time_exit_idx, open_[time_exit_idx], 3


@nb.njit(cache=True)
def simulate_trades(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_idx: np.ndarray,
    time_exit_idx: np.ndarray,
    direction: np.ndarray,
    take_profit_pct: np.ndarray,
    stop_loss_pct: np.ndarray,
    intrabar_stop_first: bool,
    half_spread: float,
    slip: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate many trades on one ticker (arrays already aligned and validated).

    Returns:
      exit_idx, entry_fill, exit_fill, reason_code
    """
    n = entry_idx.shape[0]
    exit_idx = np.empty(n, dtype=np.int64)
    entry_fill = np.empty(n, dtype=np.float64)
    exit_fill = np.empty(n, dtype=np.float64)
    reason = np.empty(n, dtype=np.int64)

    for k in range(n):
        ei = int(entry_idx[k])
        tei = int(time_exit_idx[k])

        entry_mid = open_[ei]
        entry_px = _apply_fill_price(entry_mid, int(direction[k]), half_spread, slip, True)

        ex_i, exit_mid, rc = _scan_exit(
            open_=open_,
            high=high,
            low=low,
            entry_idx=ei,
            time_exit_idx=tei,
            direction=int(direction[k]),
            tp_pct=float(take_profit_pct[k]),
            sl_pct=float(stop_loss_pct[k]),
            intrabar_stop_first=intrabar_stop_first,
        )
        exit_idx[k] = ex_i
        exit_px = _apply_fill_price(exit_mid, int(direction[k]), half_spread, slip, False)

        entry_fill[k] = entry_px
        exit_fill[k] = exit_px
        reason[k] = rc

    return exit_idx, entry_fill, exit_fill, reason
