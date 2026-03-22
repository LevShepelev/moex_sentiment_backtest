from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


ExitReason = Literal["TP", "SL", "TIME"]


@dataclass(frozen=True)
class Trade:
    event_id: int
    ticker: str
    direction: int  # +1 long, -1 short
    sentiment: int

    entry_ts: np.datetime64
    exit_ts: np.datetime64

    entry_px: float  # filled (bid/ask + slippage)
    exit_px: float   # filled (bid/ask + slippage)

    exit_reason: ExitReason
    duration_minutes: int

    # Return on *notional* (entry trade value)
    gross_return: float
