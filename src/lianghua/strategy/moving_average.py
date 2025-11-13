from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np

from lianghua.data.feed import Bar
from lianghua.strategy.base import Signal, Strategy


class MovingAverageStrategy(Strategy):
    def __init__(self, symbol: str, short_window: int = 5, long_window: int = 20) -> None:
        if short_window >= long_window:
            msg = "short_window must be smaller than long_window"
            raise ValueError(msg)
        self.symbol = symbol
        self.short_window = short_window
        self.long_window = long_window
        self._history: Deque[float] = deque(maxlen=long_window)

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        if bar.symbol != self.symbol:
            return None
        self._history.append(bar.close)
        if len(self._history) < self.long_window:
            return None
        prices = np.array(self._history, dtype=float)
        short_ma = np.mean(prices[-self.short_window :])
        long_ma = np.mean(prices)
        target_weight = 1.0 if short_ma > long_ma else 0.0
        return Signal(symbol=self.symbol, target_weight=target_weight)
