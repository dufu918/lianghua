# lianghua

Windows-first, pure-Python quant framework for A-share and crypto simulation/backtesting.

## Highlights

- **Event-driven pipeline**: `DataFeed → Strategy → Account` works for CSV, Tushare, Finnhub, Binance REST/WS.
- **Plugin strategies**: implement `Strategy` and register; sample moving-average and deep-learning strategies included.
- **Account management**: handles positions, cash, commissions/slippage, and equity snapshots.
- **Live GUI**: matplotlib window shows recent 20 min price, buy/sell markers, and live `Cash / Position / Equity / PnL`.

## Quick Start

```powershell
cd g:\gupiao\lianghua
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m lianghua.main --config configs/sample.yaml
```

For GUI-based parameter input (symbol/date/strategy/agents/proxy), append `--interactive`.

## Directory Structure

```
├─configs/           # YAML configs (CSV, Tushare, Finnhub, Binance, DL)
├─data/              # Sample data + generated features
├─scripts/           # Utility scripts (feature builder, model training)
├─src/lianghua/
│  ├─config.py       # Pydantic configs
│  ├─data/feed.py    # Datafeed adapters
│  ├─engine/         # Backtester + observer hooks
│  ├─features/       # Feature engineering utilities
│  ├─portfolio/      # Account/positions
│  ├─strategy/       # Strategy implementations
│  ├─visualization/  # Equity report + live GUI
│  └─pipelines/      # High-level runners
└─README.md
```

## Data Sources

- **Finnhub**: set `FINNHUB_API_KEY`, use `configs/finnhub.yaml`.
- **Tushare**: set `TUSHARE_TOKEN`, enable via `setx TUSHARE_ENABLED true`, use `configs/tushare.yaml`.
- **Binance (DOGE/BTC/…)**: `configs/binance_doge.yaml` covers:
  - `mode: rest`: use REST klines; `data.params.latest=true` grabs latest `limit` bars, otherwise uses `start/end`.
  - `mode: websocket`: subscribe to `wss://stream.binance.com:9443/ws/<symbol>@trade` and aggregate every `aggregate_seconds`.
  - Proxy fields `proxy` / `ws_proxy` feed through to requests/websocket (e.g., `http://127.0.0.1:7897`).

## Deep Learning Workflow

1. **Build features**
   ```powershell
   python scripts/build_features.py --symbol DOGEUSDT --interval 1m ^
     --start "2024-05-01 00:00" --end "2024-05-20 00:00" ^
     --output data/doge_features.parquet --stats-output data/doge_stats.json
   ```
   Generates technical/microstructure features (RSI/MACD/bollinger width/ATR/volume ratio/realized vol) plus triple-barrier labels.

2. **Train LSTM (auto-detects GPU)**
   ```powershell
   python scripts/train_lstm.py --dataset data/doge_features.parquet ^
     --stats data/doge_stats.json --model-output models/dl_model.ts ^
     --seq-len 60 --model gru --label-mode binary --use-sampler ^
     --focal-loss --device cuda
   ```
   训练脚本现支持 LSTM/GRU/TCN、三分类/二分类标签、类别权重与可选 WeightedRandomSampler；输出包括分类报告、TorchScript 模型和 metadata（包含 label_mode、映射、标准化参数）。

3. **Run DL strategy**
   ```powershell
   python -m lianghua.main --config configs/binance_doge_dl.yaml --interactive
   ```
   `deep_learning` 策略会：
   - 在线复用 FeatureEngineer，以保持与训练特征一致；
   - 检查连续 `confirm_bars` 的预测一致性；
   - Indicator filter（EMA/RSI/Regime）与 `cooldown_bars` 避免频繁交易，可通过配置开关；
   - 根据置信度/波动率动态调仓，可配置 `allow_short`；
   - 置信度不足或判断中性时自动回退至均线策略。

## Live Visualization

- Enable with `visualization.live=true` and set `visualization.symbol`.
- Rolling window length controlled via `visualization.window_minutes`.
- Info box显示 `Cash / Position / Equity / PnL`，右侧面板可滚动回看近期成交列表；GUI 表单可调整 symbol/日期/频率/手续费/代理/策略参数/窗口，并支持 `confirm_bars`、`cooldown_bars`、`health_window/min_health`、`require_meta_label`、`risk_budget`、`max_trades_per_window` 等节奏控制。

## Scripts

- `scripts/build_features.py`: REST fetch → feature engineering → dataset + stats JSON.
- `scripts/train_lstm.py`: train LSTM classifier on engineered features, output TorchScript + metadata.

## Advanced Notes

- Strategies can register via `STRATEGY_REGISTRY` in `pipelines/simulation.py`.
- Risk controls (max position, fallback weight, volatility window) configurable per-strategy.
- For real trading, ensure:
  - Proper commission/slippage modeling.
  - Strict time splits (no forward-looking leakage).
  - Latency compensation & fallback strategies (implemented in DL strategy skeleton).
  - Monitoring of model confidence vs realized performance; trigger re-training when divergence enlarges.

Feel free to request more connectors, strategies, or risk modules—the framework is designed to extend quickly.
