from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from lianghua.data.feed import Bar, DataFeed
from lianghua.engine.observer import EngineObserver
from lianghua.portfolio.account import Account, PortfolioSnapshot, TradeResult
from lianghua.strategy.base import Strategy


@dataclass(slots=True)
class RunResult:
    trades: List[TradeResult] = field(default_factory=list)
    snapshots: List[PortfolioSnapshot] = field(default_factory=list)

    def latest_snapshot(self) -> Optional[PortfolioSnapshot]:
        return self.snapshots[-1] if self.snapshots else None


class BacktestEngine:
    def __init__(
        self,
        feed: DataFeed,
        strategy: Strategy,
        account: Account,
        observer: EngineObserver | None = None,
    ) -> None:
        self.feed = feed
        self.strategy = strategy
        self.account = account
        self.observer = observer

    def run(self) -> RunResult:
        result = RunResult()
        for bar in self.feed.stream():
            signal = self.strategy.on_bar(bar)
            if signal:
                trade = self.account.execute_target_weight(bar, signal.target_weight)
                if trade:
                    result.trades.append(trade)
                    if self.observer:
                        self.observer.on_trade(trade)
            snapshot = self.account.snapshot(
                timestamp=bar.timestamp,
                prices={bar.symbol: bar.close},
            )
            result.snapshots.append(snapshot)
            if self.observer:
                self.observer.on_bar(bar, snapshot)
        if self.observer:
            self.observer.on_complete()
        return result
