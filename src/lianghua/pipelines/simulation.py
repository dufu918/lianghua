from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Type

import numpy as np

from lianghua.config import AppConfig
from lianghua.data.feed import build_data_feed
from lianghua.engine.backtester import BacktestEngine, RunResult
from lianghua.portfolio.account import Account
from lianghua.strategy.base import Strategy
from lianghua.engine.observer import EngineObserver
from lianghua.strategy.moving_average import MovingAverageStrategy
from lianghua.strategy.deep_learning import DeepLearningStrategy
from lianghua.strategy.trend_break import TrendBreakStrategy
from lianghua.visualization.equity import plot_equity_curve
from lianghua.visualization.live import LivePricePlotter

StrategyFactory = Callable[[Dict[str, float]], Strategy]


STRATEGY_REGISTRY: Dict[str, Callable[..., Strategy]] = {
    "moving_average": MovingAverageStrategy,
    "deep_learning": DeepLearningStrategy,
    "trend_break": TrendBreakStrategy,
}


def create_strategy(name: str, params: Dict[str, float]) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        msg = f"Unknown strategy: {name}"
        raise ValueError(msg)
    return STRATEGY_REGISTRY[name](**params)


def _summaries(result: RunResult) -> Dict[str, float]:
    if not result.snapshots:
        return {}
    equity = np.array([snap.equity for snap in result.snapshots])
    start_equity = equity[0]
    end_equity = equity[-1]
    total_return = (end_equity / start_equity) - 1
    rolling_max = np.maximum.accumulate(equity)
    drawdowns = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())
    return {
        "start_equity": float(start_equity),
        "end_equity": float(end_equity),
        "total_return": float(total_return),
        "max_drawdown": float(max_drawdown),
        "num_trades": len(result.trades),
    }


def _trade_to_dict(trade) -> Dict[str, float]:
    payload = asdict(trade)
    payload["timestamp"] = trade.timestamp.isoformat()
    return payload


def _dump_report(result: RunResult, path: str | None) -> None:
    if not path:
        return
    report_path = Path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": _summaries(result),
        "trades": [_trade_to_dict(trade) for trade in result.trades],
        "equity_curve": [
            {"timestamp": snap.timestamp.isoformat(), "equity": snap.equity}
            for snap in result.snapshots
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_observer(config: AppConfig) -> EngineObserver | None:
    if not config.visualization.live:
        return None
    symbol = config.visualization.symbol or config.strategy.params.get("symbol")
    if not symbol:
        msg = "Live visualization requires a symbol (set visualization.symbol or strategy params)."
        raise ValueError(msg)
    return LivePricePlotter(
        symbol=symbol,
        window_minutes=config.visualization.window_minutes,
        start_equity=config.simulation.cash,
    )


def run_simulation(config: AppConfig) -> RunResult:
    feed = build_data_feed(config.data, simulation=config.simulation)
    strategy = create_strategy(config.strategy.name, config.strategy.params)
    account = Account(
        cash=config.simulation.cash,
        commission_bps=config.simulation.commission_bps,
        slippage_bps=config.simulation.slippage_bps,
    )
    observer = _build_observer(config)
    engine = BacktestEngine(feed=feed, strategy=strategy, account=account, observer=observer)
    result = engine.run()
    _dump_report(result, config.report.output)
    plot_equity_curve(result, config.report.chart)
    return result
