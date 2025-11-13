# Lianghua Quant Framework

> Windows-first、纯 Python 的量化研究/实盘模拟框架，现已适配多场景深度学习策略、BTC 参考特征以及风控网格执行。

## 核心能力

- **多源数据**：CSV / Tushare / Finnhub / Binance REST&WS；支持统一代理、WS 自动重连、实时 GUI 可视化。
- **场景感知策略**：`deep_learning` 策略可加载多场景（平盘/趋势）模型，按置信度 + 波动自适应调仓，内置均值回归 + 网格 + 连败冷静逻辑。
- **特征与训练流水线**：`build_features.py`/`augment_reference_features.py`/`train_lstm.py` 配套脚本可构建 DOGE + BTC 多资产特征、三分类标签、walk-forward、温度校准等。
- **回测 / 实盘统一入口**：`python -m lianghua.main --config ... [--interactive]` 既能跑历史，也可实时订阅 Binance WS。

## 快速上手
```powershell
cd g:\gupiao\lianghua
python -m venv .venv
.venv\Scripts\activate
pip install -e .
python -m lianghua.main --config configs/sample.yaml --interactive
```
交互模式可现场填写 symbol / 时间段 / 策略参数；非交互模式直接读取 YAML。

## 目录
```
configs/            # 各类数据/策略配置（binance_doge_dl.yaml 为主策略）
data/               # 特征/统计文件 (DOGE、BTC 等)
scripts/            # build_features、augment_reference_features、train_lstm 等工具
src/lianghua/
  ├─config.py       # Pydantic 配置
  ├─data/feed.py    # 数据源适配（含 Binance REST/WS 自动重连）
  ├─engine/         # 回测引擎 + 观察者
  ├─features/       # FeatureEngineer + 线上特征维护
  ├─strategy/       # MA、DL 等策略实现
  └─pipelines/      # `run_cli()` 入口
```

## 数据与特征
1. **构建 DOGE/BTC 特征（近一年样例）**
   ```powershell
   python scripts/build_features.py --symbol DOGEUSDT --interval 1m --start 2024-11-10 --end 2025-11-10 --limit 0 --output data/doge_features_1y.parquet --stats-output data/doge_stats_1y.json --label-mode triple --label-horizon 5 --label-threshold 0.002 --extra-intervals "5min,15min" --meta-label --meta-horizon 10 --meta-up 0.003 --meta-down 0.003 --triple-barrier --tb-horizon 12 --tb-up 0.003 --tb-down 0.003
   python scripts/build_features.py --symbol BTCUSDT  --interval 1m --start 2024-11-10 --end 2025-11-10 --limit 0 --output data/btc_features_1y.parquet  --stats-output data/btc_stats_1y.json  --label-mode triple --label-horizon 5 --label-threshold 0.002 --extra-intervals "5min,15min"
   ```
2. **合并参考资产特征**
   ```powershell
   python scripts/augment_reference_features.py --primary data/doge_features_1y.parquet --primary-stats data/doge_stats_1y.json --reference data/btc_features_1y.parquet --prefix btc --output data/doge_features_1y_with_btc.parquet --stats-output data/doge_stats_1y_with_btc.json
   ```
   默认会拉入 `btc_close/btc_ret_1/.../btc_vwap_delta` 等列，可通过 `--columns` 自定义。

## 模型训练
- **抽样调参（快速）**
  ```powershell
  python scripts/train_lstm.py --dataset data/doge_features_1y_with_btc.parquet --stats data/doge_stats_1y_with_btc.json --model-output models/dl_model_transformer.ts --models gru,transformer --seq-len 60 --label-mode triple --use-sampler --oversample-positive 8 --focal-loss --device cuda --holdout-start 2025-08-01 --evaluate-holdout --sample-rate 0.5 --max-train-samples 150000 --disable-walk-forward --scenario-column scenario_label --scenario-include flat,trend_up,trend_down --min-epochs 8 --patience 6 --scheduler cosine --hidden-dim 160 --transformer-heads 8 --transformer-ffn 256
  ```
  日志会输出 `[Scenario Summary] ...`，可快速评估各场景表现。
- **全量定稿（上线前必跑）**
  ```powershell
  python scripts/train_lstm.py --dataset data/doge_features_1y_with_btc.parquet --stats data/doge_stats_1y_with_btc.json --model-output models/dl_model_transformer.ts --models gru,transformer --seq-len 60 --label-mode triple --use-sampler --oversample-positive 8 --focal-loss --device cuda --holdout-start 2025-08-01 --evaluate-holdout --sample-rate 1.0 --max-train-samples 0 --scenario-column scenario_label --min-epochs 8 --patience 6 --scheduler cosine --hidden-dim 160 --transformer-heads 8 --transformer-ffn 256
  ```
  输出包含 TorchScript 模型与 `dl_model_transformer.meta.json`（特征列表、标准化参数、场景模型路径、温度等）。

## 深度学习策略 (`configs/binance_doge_dl.yaml`)
- **数据段**：可配置 `mode: rest` (历史) / `websocket` (实盘)。WS 版带自动重连并支持代理。
- **策略参数重点**：
  - `scenario_overrides.flat`：`mean_reversion` + `grid` 控制震荡网格间距/仓位；`loss_streak_limit` 冷静多长时间。
  - `scenario_overrides.trend_up`：高仓位 + 连败冷静，配合新模型提升顺势持仓。
  - 全局 `loss_streak_limit`/`loss_cooldown_bars`：任何场景连续亏损都会触发冷却。
  - `max_trades_per_window`、`trade_window_minutes`：限制单位时间交易次数。
- **执行流程**：
  1. FeatureEngineer 生成实时特征（含 BTC 列）。
  2. 按场景选择模型 → 温度校准 → 置信度/波动调仓。
  3. 若置信度不足，先尝试均值回归/网格；再 fallback 到均线策略或 `fallback_weight`。
  4. 记录每次交易所属场景，更新健康度与连败计数。

## 运行回测/实盘
```powershell
# 批量回测 (默认 REST)
python -m lianghua.main --config configs/binance_doge_dl.yaml
# 交互模式（在命令行填写 symbol/日期/参数)
python -m lianghua.main --config configs/binance_doge_dl.yaml --interactive
```
实盘使用 `mode: websocket` 时，可设置 `data.params.ws_proxy`；握手失败会自动指数回退重连。

## 风险与优化建议
- **特征多样性**：建议定期补充 BTC/ETH 等跨资产特征，或引入资金费率/盘口数据，减少单资产过拟合。
- **训练流程**：调参→抽样验证→全量重训是推荐流程，切勿直接微调旧模型。
- **风控**：合理设置 `loss_streak_limit`、`grid entry/exit`、`max_trades_per_window`；实盘可再叠加净值回撤暂停、限价成交等机制。
- **监控**：利用 `[Scenario Summary] ... | holdout` 观察各场景泛化，发现某场景漂移时及时重训和调仓。

如需新增数据源/策略/风险模块，欢迎继续扩展 `scripts/` 与 `src/lianghua/strategy/`，整个项目已为快速定制预留接口。祝研究顺利。
