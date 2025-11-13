from __future__ import annotations

from typing import Protocol

from lianghua.data.feed import Bar
from lianghua.portfolio.account import PortfolioSnapshot, TradeResult


class EngineObserver(Protocol):
    def on_bar(self, bar: Bar, snapshot: PortfolioSnapshot) -> None:  # pragma: no cover - interface
        ...

    def on_trade(self, trade: TradeResult) -> None:  # pragma: no cover - interface
        ...

    def on_complete(self) -> None:  # pragma: no cover - interface
        ...
