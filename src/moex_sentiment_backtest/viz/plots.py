from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

def plot_equity_curves(equity_by_name: dict[str, pd.Series], out_path: Path, benchmarks: dict[str, pd.Series] | None = None) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    for name, s in equity_by_name.items():
        plt.plot(s.index, s.values, label=name)

    if benchmarks:
        for name, b in benchmarks.items():
            plt.plot(b.index, b.values, label=f"BENCH:{name}", linewidth=2)

    plt.title("Equity curves")
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_drawdowns(curves: dict[str, pd.Series], out_path: Path) -> None:
    plt.figure(figsize=(14, 6))
    for name, s in curves.items():
        if s.empty:
            continue
        eq = s.values.astype(float)
        peak = (pd.Series(eq).cummax()).values
        dd = eq / peak - 1.0
        plt.plot(s.index, dd, label=name)
    plt.title("Drawdowns")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def plot_trade_return_hist(returns: dict[str, Iterable[float]], out_path: Path, bins: int = 80) -> None:
    plt.figure(figsize=(14, 7))
    for name, r in returns.items():
        r_list = list(r)
        if not r_list:
            continue
        plt.hist(r_list, bins=bins, alpha=0.35, label=name, density=True)
    plt.title("Distribution of trade returns")
    plt.xlabel("Trade return")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()
