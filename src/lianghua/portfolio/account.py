from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd

from lianghua.data.feed import Bar


@dataclass(slots=True)
class Position:
    symbol: str
    quantity: int = 0
    avg_price: float = 0.0

    def market_value(self, price: float) -> float:
        return self.quantity * price


@dataclass(slots=True)
class TradeResult:
    timestamp: pd.Timestamp
    symbol: str
    quantity: int
    price: float
    action: str
    commission: float
    cash_after: float
    position_after: int


@dataclass(slots=True)
class PortfolioSnapshot:
    timestamp: pd.Timestamp
    cash: float
    positions: Dict[str, int]
    equity: float


class Account:
    def __init__(
        self,
        cash: float,
        commission_bps: float = 1.5,
        slippage_bps: float = 1.0,
        lot_size: int = 100,
    ) -> None:
        self.cash = cash
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.lot_size = lot_size
        self.positions: Dict[str, Position] = {}

    def _get_position(self, symbol: str) -> Position:
        return self.positions.setdefault(symbol, Position(symbol=symbol))

    def total_equity(self, last_prices: Dict[str, float]) -> float:
        equity = self.cash
        for symbol, pos in self.positions.items():
            price = last_prices.get(symbol, pos.avg_price)
            equity += pos.market_value(price)
        return equity

    def _apply_slippage(self, price: float, direction: str) -> float:
        slippage = price * self.slippage_bps / 10_000
        return price + slippage if direction == "BUY" else price - slippage

    def execute_target_weight(
        self,
        bar: Bar,
        target_weight: float,
    ) -> Optional[TradeResult]:
        price = self._apply_slippage(bar.close, "BUY" if target_weight > 0 else "SELL")
        last_prices = {bar.symbol: price}
        total_equity = self.total_equity(last_prices)
        position = self._get_position(bar.symbol)
        current_value = position.market_value(price)
        target_value = total_equity * target_weight
        delta_value = target_value - current_value
        qty = int(delta_value / price / self.lot_size) * self.lot_size
        if qty == 0:
            return None
        if qty > 0:
            cost = qty * price
            commission = cost * self.commission_bps / 10_000
            if self.cash < cost + commission:
                return None
            self.cash -= cost + commission
            new_qty = position.quantity + qty
            position.avg_price = (
                (position.avg_price * position.quantity + cost) / new_qty if new_qty else 0
            )
            position.quantity = new_qty
            action = "BUY"
        else:
            qty_abs = abs(qty)
            if qty_abs > position.quantity:
                qty_abs = position.quantity
                qty = -qty_abs
            proceeds = qty_abs * price
            commission = proceeds * self.commission_bps / 10_000
            self.cash += proceeds - commission
            position.quantity -= qty_abs
            if position.quantity == 0:
                position.avg_price = 0.0
            action = "SELL"
        return TradeResult(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            quantity=qty,
            price=price,
            action=action,
            commission=commission,
            cash_after=self.cash,
            position_after=position.quantity,
        )

    def snapshot(self, timestamp: pd.Timestamp, prices: Dict[str, float]) -> PortfolioSnapshot:
        equity = self.total_equity(prices)
        positions = {sym: pos.quantity for sym, pos in self.positions.items()}
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions=positions,
            equity=equity,
        )
