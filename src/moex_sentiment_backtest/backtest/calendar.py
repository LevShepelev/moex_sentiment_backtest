from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TradingCalendarIndex:
    """Index helpers computed from a ticker's minute timestamps."""

    ts: np.ndarray              # datetime64[ns] array (sorted)
    days: np.ndarray            # datetime64[D] array per bar (same length as ts)
    unique_days: np.ndarray     # unique trading days (sorted)
    first_idx: np.ndarray       # first bar index for each unique day
    last_idx: np.ndarray        # last bar index for each unique day

    @staticmethod
    def build(ts: np.ndarray) -> "TradingCalendarIndex":
        days = ts.astype("datetime64[D]")
        unique_days, first_idx = np.unique(days, return_index=True)
        # last index per day: next_first - 1, last day ends at len-1
        last_idx = np.empty_like(first_idx)
        last_idx[:-1] = first_idx[1:] - 1
        last_idx[-1] = ts.shape[0] - 1
        return TradingCalendarIndex(ts=ts, days=days, unique_days=unique_days, first_idx=first_idx, last_idx=last_idx)

    def time_exit_index(self, entry_idx: int, mode: str, n_days: int, price: str) -> int:
        """Compute the index to use if TP/SL are not hit.

        - mode='next_day': uses next trading day.
        - mode='n_days': uses (entry_day + n_days), falling forward to next available trading day if needed.
        - price: 'open' => first bar of the exit day; 'close' => last bar of the exit day.
        """
        entry_day = self.days[entry_idx]
        if mode == "next_day":
            target_day = entry_day + np.timedelta64(1, "D")
        elif mode == "n_days":
            target_day = entry_day + np.timedelta64(n_days, "D")
        else:
            raise ValueError(f"Unknown time exit mode: {mode}")

        # find insertion point in unique_days (next trading day >= target_day)
        j = int(np.searchsorted(self.unique_days, target_day))
        if j >= self.unique_days.shape[0]:
            # No data far enough; exit at last bar available
            return int(self.last_idx[-1])

        if price == "open":
            return int(self.first_idx[j])
        if price == "close":
            return int(self.last_idx[j])
        raise ValueError(f"Unknown exit price: {price}")
