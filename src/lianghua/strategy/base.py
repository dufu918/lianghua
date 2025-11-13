from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from lianghua.data.feed import Bar


@dataclass(slots=True)
class Signal:
    symbol: str
    target_weight: float


class Strategy(Protocol):
    def on_bar(self, bar: Bar) -> Optional[Signal]:
        ...
