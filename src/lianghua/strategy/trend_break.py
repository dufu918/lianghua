from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np

from lianghua.data.feed import Bar
from lianghua.strategy.base import Signal, Strategy


@dataclass
class TrendBreakConfig:
    symbol: str
    breakout_window: int = 60
    confirm_window: int = 20
    atr_window: int = 14
    ema_fast: int = 9
    ema_slow: int = 30
    volume_window: int = 30
    volume_multiplier: float = 1.8
    risk_per_trade: float = 0.01
    max_position: float = 0.5
    stop_atr: float = 1.8
    take_profit_atr: float = 3.0
    trail_atr: float = 1.2
    pullback_atr: float = 0.8
    pyramid_steps: int = 2
    allow_short: bool = False
    cooldown_bars: int = 5


class TrendBreakStrategy(Strategy):
    def __init__(self, **params) -> None:
        cfg = TrendBreakConfig(**params)
        self.symbol = cfg.symbol
        self.breakout_window = cfg.breakout_window
        self.confirm_window = cfg.confirm_window
        self.atr_window = cfg.atr_window
        self.ema_fast_span = cfg.ema_fast
        self.ema_slow_span = cfg.ema_slow
        self.volume_window = cfg.volume_window
        self.volume_multiplier = cfg.volume_multiplier
        self.risk_per_trade = cfg.risk_per_trade
        self.max_position = cfg.max_position
        self.stop_atr = cfg.stop_atr
        self.take_profit_atr = cfg.take_profit_atr
        self.trail_atr = cfg.trail_atr
        self.pullback_atr = cfg.pullback_atr
        self.pyramid_steps = max(1, cfg.pyramid_steps)
        self.allow_short = cfg.allow_short
        self.cooldown_bars = cfg.cooldown_bars

        max_buffer = max(
            self.breakout_window,
            self.confirm_window,
            self.volume_window,
            self.ema_slow_span * 2,
        )
        self.high_buffer: Deque[float] = deque(maxlen=max_buffer + 2)
        self.low_buffer: Deque[float] = deque(maxlen=max_buffer + 2)
        self.close_buffer: Deque[float] = deque(maxlen=max_buffer + 2)
        self.volume_buffer: Deque[float] = deque(maxlen=self.volume_window)
        self.tr_buffer: Deque[float] = deque(maxlen=self.atr_window)
        self.prev_close: Optional[float] = None
        self.ema_fast_val: Optional[float] = None
        self.ema_slow_val: Optional[float] = None
        self.current_weight: float = 0.0
        self.entry_price: Optional[float] = None
        self.highest_close: Optional[float] = None
        self.lowest_close: Optional[float] = None
        self.current_layers: int = 0
        self.pullback_ready = False
        self.cooldown_counter = 0

    def _ema(self, prev: Optional[float], value: float, span: int) -> float:
        alpha = 2.0 / (span + 1)
        if prev is None:
            return value
        return prev + alpha * (value - prev)

    def _atr(self) -> float:
        if len(self.tr_buffer) < self.atr_window:
            return 0.0
        return float(np.mean(self.tr_buffer))

    def _avg_volume(self) -> float:
        if len(self.volume_buffer) < self.volume_window:
            return 0.0
        return float(np.mean(self.volume_buffer))

    def _position_size(self, price: float, atr: float) -> float:
        if atr <= 0 or price <= 0:
            return min(self.max_position, self.risk_per_trade)
        atr_frac = atr / price
        if atr_frac <= 0:
            return min(self.max_position, self.risk_per_trade)
        weight = self.risk_per_trade / atr_frac
        return float(np.clip(weight, 0.0, self.max_position))

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        if bar.symbol != self.symbol:
            return None
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        self.high_buffer.append(bar.high)
        self.low_buffer.append(bar.low)
        self.close_buffer.append(bar.close)
        self.volume_buffer.append(bar.volume)

        if self.prev_close is not None:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self.prev_close),
                abs(bar.low - self.prev_close),
            )
            self.tr_buffer.append(tr)
        self.prev_close = bar.close

        self.ema_fast_val = self._ema(self.ema_fast_val, bar.close, self.ema_fast_span)
        self.ema_slow_val = self._ema(self.ema_slow_val, bar.close, self.ema_slow_span)

        if len(self.high_buffer) <= self.breakout_window or self.ema_slow_val is None:
            return None

        recent_highs = list(self.high_buffer)[:-1]
        recent_lows = list(self.low_buffer)[:-1]
        if len(recent_highs) < self.breakout_window or len(recent_lows) < self.breakout_window:
            return None
        breakout_high = max(recent_highs[-self.breakout_window :])
        breakout_low = min(recent_lows[-self.breakout_window :])
        confirm_low = min(list(self.low_buffer)[-self.confirm_window :])
        confirm_high = max(list(self.high_buffer)[-self.confirm_window :])

        atr = self._atr()
        avg_vol = self._avg_volume()
        volume_ok = avg_vol == 0.0 or bar.volume >= avg_vol * self.volume_multiplier

        ema_trend_up = self.ema_fast_val >= self.ema_slow_val
        ema_trend_down = self.ema_fast_val <= self.ema_slow_val

        # manage existing position exits
        if self.current_weight > 0 and self.entry_price is not None:
            self.highest_close = max(self.highest_close or bar.close, bar.close)
            stop_price = self.entry_price - self.stop_atr * atr
            trail_price = (self.highest_close or bar.close) - self.trail_atr * atr
            take_price = self.entry_price + self.take_profit_atr * atr
            if bar.close <= max(stop_price, trail_price, confirm_low) or bar.close <= self.ema_slow_val:
                return self._exit()
            if atr > 0 and bar.close >= take_price:
                return self._exit()
        elif self.current_weight < 0 and self.entry_price is not None:
            self.lowest_close = min(self.lowest_close or bar.close, bar.close)
            stop_price = self.entry_price + self.stop_atr * atr
            trail_price = (self.lowest_close or bar.close) + self.trail_atr * atr
            take_price = self.entry_price - self.take_profit_atr * atr
            if bar.close >= min(stop_price, trail_price, confirm_high) or bar.close >= self.ema_slow_val:
                return self._exit()
            if atr > 0 and bar.close <= take_price:
                return self._exit()

        # avoid frequent flips
        if self.cooldown_counter > 0:
            return None

        if (
            bar.close > breakout_high
            and ema_trend_up
            and volume_ok
            and self.current_weight <= 0
        ):
            target = self._position_size(bar.close, atr)
            return self._enter(target, bar.close)

        if (
            self.current_weight > 0
            and self.current_layers < self.pyramid_steps
            and atr > 0
        ):
            if bar.low <= (self.entry_price or bar.close) - self.pullback_atr * atr:
                self.pullback_ready = True
            if (
                self.pullback_ready
                and bar.close > breakout_high
                and ema_trend_up
                and volume_ok
            ):
                add_weight = self._position_size(bar.close, atr)
                new_weight = np.clip(self.current_weight + add_weight, 0.0, self.max_position)
                return self._add_layer(new_weight, bar.close)

        if (
            self.allow_short
            and bar.close < breakout_low
            and ema_trend_down
            and volume_ok
            and self.current_weight >= 0
        ):
            target = -self._position_size(bar.close, atr)
            return self._enter(target, bar.close)

        return None

    def _enter(self, weight: float, price: float) -> Signal:
        if weight == 0:
            return Signal(symbol=self.symbol, target_weight=0.0)
        self.current_weight = weight
        self.entry_price = price
        self.highest_close = price
        self.lowest_close = price
        self.current_layers = 1
        self.pullback_ready = False
        self.cooldown_counter = self.cooldown_bars
        return Signal(symbol=self.symbol, target_weight=weight)

    def _exit(self) -> Signal:
        self.current_weight = 0.0
        self.entry_price = None
        self.highest_close = None
        self.lowest_close = None
        self.current_layers = 0
        self.pullback_ready = False
        self.cooldown_counter = self.cooldown_bars
        return Signal(symbol=self.symbol, target_weight=0.0)

    def _add_layer(self, target_weight: float, price: float) -> Signal:
        self.current_weight = np.clip(target_weight, -self.max_position, self.max_position)
        self.entry_price = price if self.entry_price is None else (self.entry_price * (self.current_layers) + price) / (self.current_layers + 1)
        self.current_layers = min(self.pyramid_steps, self.current_layers + 1)
        self.highest_close = max(self.highest_close or price, price)
        self.cooldown_counter = self.cooldown_bars
        return Signal(symbol=self.symbol, target_weight=self.current_weight)
