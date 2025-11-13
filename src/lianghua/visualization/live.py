from __future__ import annotations

from collections import deque
import time
from typing import Deque, List, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

from lianghua.data.feed import Bar
from lianghua.engine.observer import EngineObserver
from lianghua.portfolio.account import PortfolioSnapshot, TradeResult


def _ts_to_float(ts) -> float:
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Shanghai")
    else:
        ts = ts.tz_convert("Asia/Shanghai")
    ts = ts.tz_localize(None)
    return mdates.date2num(ts)


class LivePricePlotter(EngineObserver):
    def __init__(
        self,
        symbol: str,
        title: Optional[str] = None,
        window_minutes: int = 20,
        start_equity: float = 0.0,
        ) -> None:
        if not symbol:
            msg = "LivePricePlotter requires a symbol to focus on."
            raise ValueError(msg)
        self.symbol = symbol
        self.title = title or f"{symbol} Live Simulation"
        self.window = pd.Timedelta(minutes=window_minutes)
        self.start_equity = start_equity
        plt.ion()
        self.fig = plt.figure(figsize=(12, 4))
        gs = self.fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
        self.ax = self.fig.add_subplot(gs[0, 0])
        manager = getattr(self.fig.canvas, "manager", None)
        if manager is not None:
            manager.set_window_title(self.title)
        self.ax.set_title(self.title)
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Price")
        self.ax.grid(True, linestyle="--", alpha=0.4)
        self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        self.price_times: Deque[pd.Timestamp] = deque()
        self.price_values: Deque[float] = deque()
        (self.price_line,) = self.ax.plot([], [], color="#1f77b4", linewidth=2, label="Close")
        self.buy_times: Deque[pd.Timestamp] = deque()
        self.buy_prices: Deque[float] = deque()
        self.sell_times: Deque[pd.Timestamp] = deque()
        self.sell_prices: Deque[float] = deque()
        (self.buy_scatter,) = self.ax.plot([], [], "^", color="#2ca02c", markersize=10, label="Buy")
        (self.sell_scatter,) = self.ax.plot([], [], "v", color="#d62728", markersize=10, label="Sell")
        self.ax.legend(loc="upper left")
        self.sidebar = self.fig.add_subplot(gs[0, 1])
        self.sidebar.axis("off")
        self.sidebar.set_title("Trades", loc="left", fontsize=10)
        self.trade_display = 12
        self.trade_history: List[str] = []
        self.trade_offset = 0
        self.trade_text = self.sidebar.text(0.02, 0.9, "No trades yet", va="top", ha="left", fontsize=9, family="monospace")
        btn_up_ax = self.fig.add_axes([0.92, 0.88, 0.05, 0.08])
        btn_down_ax = self.fig.add_axes([0.92, 0.12, 0.05, 0.08])
        self.btn_up = Button(btn_up_ax, "↑")
        self.btn_down = Button(btn_down_ax, "↓")
        self.btn_up.on_clicked(self._scroll_older)
        self.btn_down.on_clicked(self._scroll_newer)
        self.info_text = self.ax.text(
            0.98,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="right",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
        )
        self._last_redraw = 0.0
        self._redraw_interval = 0.25  # seconds
        self._pending_snapshot: Optional[PortfolioSnapshot] = None
        self._pending_price: float = 0.0

    def on_bar(self, bar: Bar, snapshot: PortfolioSnapshot) -> None:
        if bar.symbol != self.symbol:
            return
        self._append_and_trim(bar.timestamp, bar.close)
        self._pending_snapshot = snapshot
        self._pending_price = bar.close
        self._refresh()

    def on_trade(self, trade: TradeResult) -> None:
        if trade.symbol != self.symbol:
            return
        ts = pd.to_datetime(trade.timestamp)
        if trade.quantity > 0:
            self.buy_times.append(ts)
            self.buy_prices.append(trade.price)
            action = "BUY"
        else:
            self.sell_times.append(ts)
            self.sell_prices.append(trade.price)
            action = "SELL"
        entry = f"{ts.strftime('%m-%d %H:%M')} {action} {trade.quantity:+} @ {trade.price:.5f}"
        self.trade_history.insert(0, entry)
        self._render_trade_log()

    def on_complete(self) -> None:
        plt.ioff()
        plt.show(block=True)

    def _append_and_trim(self, timestamp: pd.Timestamp, price: float) -> None:
        self.price_times.append(timestamp)
        self.price_values.append(price)
        cutoff = timestamp - self.window
        while self.price_times and self.price_times[0] < cutoff:
            self.price_times.popleft()
            self.price_values.popleft()
        while self.buy_times and self.buy_times[0] < cutoff:
            self.buy_times.popleft()
            self.buy_prices.popleft()
        while self.sell_times and self.sell_times[0] < cutoff:
            self.sell_times.popleft()
            self.sell_prices.popleft()

    def _update_info(self, snapshot: PortfolioSnapshot, last_price: float) -> None:
        position_qty = snapshot.positions.get(self.symbol, 0)
        pnl = snapshot.equity - self.start_equity
        info = (
            f"Cash: {snapshot.cash:,.2f}\n"
            f"Position: {position_qty} @ {last_price:.4f}\n"
            f"Equity: {snapshot.equity:,.2f}\n"
            f"PnL: {pnl:+,.2f}"
        )
        self.info_text.set_text(info)

    def _refresh(self) -> None:
        now = time.monotonic()
        if now - self._last_redraw < self._redraw_interval:
            return
        self._last_redraw = now
        if not self.price_times:
            return
        times_float = [_ts_to_float(ts) for ts in self.price_times]
        self.price_line.set_data(times_float, list(self.price_values))
        self.buy_scatter.set_data(
            [_ts_to_float(ts) for ts in self.buy_times],
            list(self.buy_prices),
        )
        self.sell_scatter.set_data(
            [_ts_to_float(ts) for ts in self.sell_times],
            list(self.sell_prices),
        )
        x_start = times_float[0]
        x_end = times_float[-1]
        if x_start == x_end:
            x_end += 1e-6
        self.ax.set_xlim(x_start, x_end)
        min_price = min(self.price_values)
        max_price = max(self.price_values)
        if min_price == max_price:
            pad = max(1e-6, abs(min_price) * 0.001 + 1e-6)
            min_price -= pad
            max_price += pad
        self.ax.set_ylim(min_price, max_price)
        if self._pending_snapshot:
            self._update_info(self._pending_snapshot, self._pending_price)
            self._pending_snapshot = None
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _render_trade_log(self) -> None:
        total = len(self.trade_history)
        if total == 0:
            self.trade_text.set_text("No trades yet")
            return
        max_offset = max(0, total - self.trade_display)
        self.trade_offset = min(self.trade_offset, max_offset)
        start = self.trade_offset
        end = min(total, start + self.trade_display)
        subset = self.trade_history[start:end]
        text = "\n".join(subset)
        self.trade_text.set_text(text)

    def _scroll_older(self, _event) -> None:
        total = len(self.trade_history)
        if total <= self.trade_display:
            return
        max_offset = total - self.trade_display
        self.trade_offset = min(max_offset, self.trade_offset + self.trade_display)
        self._render_trade_log()

    def _scroll_newer(self, _event) -> None:
        if self.trade_offset == 0:
            return
        self.trade_offset = max(0, self.trade_offset - self.trade_display)
        self._render_trade_log()
