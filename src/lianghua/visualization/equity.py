from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from lianghua.engine.backtester import RunResult


def plot_equity_curve(result: RunResult, path: str | None) -> None:
    if not path or not result.snapshots:
        return
    chart_path = Path(path)
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    timestamps: List = [snap.timestamp for snap in result.snapshots]
    equity: List[float] = [snap.equity for snap in result.snapshots]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(timestamps, equity, label="Equity", color="#1f77b4", linewidth=2)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (CNY)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
