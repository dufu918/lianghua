from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from lianghua.features import compute_feature_frame, compute_labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Binance feature dataset.")
    parser.add_argument("--symbol", default="DOGEUSDT")
    parser.add_argument("--interval", default="1m")
    parser.add_argument("--start", help="YYYY-MM-DD HH:MM", required=False)
    parser.add_argument("--end", help="YYYY-MM-DD HH:MM", required=False)
    parser.add_argument("--latest", action="store_true", help="Ignore start/end, pull latest limit bars.")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("data/binance_features.parquet"))
    parser.add_argument("--stats-output", type=Path, default=Path("data/binance_feature_stats.json"))
    parser.add_argument("--label-horizon", type=int, default=5)
    parser.add_argument("--label-threshold", type=float, default=0.002)
    parser.add_argument("--label-mode", choices=["triple", "binary"], default="triple")
    parser.add_argument("--up-threshold", type=float, default=0.002)
    parser.add_argument("--down-threshold", type=float, default=0.002)
    parser.add_argument("--regime-fast", type=int, default=12)
    parser.add_argument("--regime-slow", type=int, default=48)
    parser.add_argument("--regime-threshold", type=float, default=0.0)
    parser.add_argument("--extra-intervals", type=str, default="", help="Comma-separated list such as 5min,15min")
    parser.add_argument("--meta-label", action="store_true")
    parser.add_argument("--meta-horizon", type=int, default=10)
    parser.add_argument("--meta-up", type=float, default=0.003)
    parser.add_argument("--meta-down", type=float, default=0.003)
    parser.add_argument("--triple-barrier", action="store_true", help="Use triple-barrier style labels.")
    parser.add_argument("--tb-horizon", type=int, default=12)
    parser.add_argument("--tb-up", type=float, default=0.003)
    parser.add_argument("--tb-down", type=float, default=0.003)
    return parser.parse_args()


def _ts_to_ms(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _fetch_klines(
    symbol: str,
    interval: str,
    start_ms: Optional[int],
    end_ms: Optional[int],
    max_rows: Optional[int],
    chunk_size: int,
) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    rows = []
    current = start_ms
    total = 0
    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": min(chunk_size, 1000),
        }
        if current is not None:
            params["startTime"] = current
        if end_ms is not None:
            params["endTime"] = end_ms
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = batch[-1][0]
        current = last_open + 1
        total += len(batch)
        print(f"[fetch] downloaded {total} rows (last={pd.to_datetime(last_open, unit='ms')})", flush=True)
        if max_rows and total >= max_rows:
            break
        if end_ms is not None and current >= end_ms:
            break
        if len(batch) < params["limit"]:
            break
        if start_ms is None:
            # latest mode, only one batch
            break
    if not rows:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    df = pd.DataFrame(
        rows,
        columns=[*cols, "quote_asset_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"],
    )
    df["open_time"] = (
        pd.to_datetime(df["open_time"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df[["open", "high", "low", "close", "volume"]]


def _append_multi_interval_features(base: pd.DataFrame, raw_df: pd.DataFrame, intervals: List[str]) -> None:
    for interval in intervals:
        try:
            resampled = raw_df.resample(interval).agg(
                {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            )
        except ValueError:
            continue
        resampled = resampled.reindex(base.index).ffill()
        prefix = interval.replace(" ", "").replace("min", "m")
        base[f"close_{prefix}"] = resampled["close"]
        base[f"ret_{prefix}"] = resampled["close"].pct_change().fillna(0)
        base[f"volume_{prefix}"] = resampled["volume"]


def _compute_meta_labels(close: pd.Series, horizon: int, up: float, down: float) -> pd.Series:
    future = close.shift(-horizon)
    returns = (future - close) / close
    meta = np.zeros(len(close), dtype=int)
    meta[(returns > up).values] = 1
    meta[(returns < -down).values] = -1
    return pd.Series(meta, index=close.index, name="meta_label")


def _triple_barrier(close: pd.Series, horizon: int, up: float, down: float) -> pd.Series:
    prices = close.to_numpy()
    n = len(prices)
    labels = np.zeros(n, dtype=np.int8)
    if horizon <= 0:
        return pd.Series(labels, index=close.index, name="label")
    for idx in range(n):
        start_price = prices[idx]
        upper = start_price * (1.0 + up)
        lower = start_price * (1.0 - down)
        end = min(n - 1, idx + horizon)
        future = prices[idx + 1 : end + 1]
        if future.size == 0:
            break
        up_hits = np.where(future >= upper)[0]
        down_hits = np.where(future <= lower)[0]
        first_up = up_hits[0] if up_hits.size > 0 else None
        first_down = down_hits[0] if down_hits.size > 0 else None
        if first_up is None and first_down is None:
            continue
        if first_down is None or (first_up is not None and first_up < first_down):
            labels[idx] = 1
        elif first_up is None or (first_down is not None and first_down < first_up):
            labels[idx] = -1
    return pd.Series(labels, index=close.index, name="label")


def main() -> None:
    args = _parse_args()
    start_ms = None if args.latest else _ts_to_ms(args.start)
    end_ms = None if args.latest else _ts_to_ms(args.end)
    df = _fetch_klines(
        args.symbol,
        args.interval,
        start_ms,
        end_ms,
        args.limit if args.limit > 0 else None,
        chunk_size=1000,
    )
    if df.empty:
        raise SystemExit("No data fetched.")
    features = compute_feature_frame(
        df,
        regime_fast=args.regime_fast,
        regime_slow=args.regime_slow,
        regime_threshold=args.regime_threshold,
    )
    extra_intervals = [item.strip() for item in args.extra_intervals.split(",") if item.strip()]
    if extra_intervals:
        _append_multi_interval_features(features, df, extra_intervals)
    if args.triple_barrier:
        tb_labels = _triple_barrier(features["close"], args.tb_horizon, args.tb_up, args.tb_down)
        if args.label_mode == "binary":
            features["label"] = (tb_labels == 1).astype(int)
        else:
            features["label"] = tb_labels
    else:
        future_returns = features["close"].pct_change(args.label_horizon).shift(-args.label_horizon)
        if args.label_mode == "binary":
            features["label"] = (future_returns > args.up_threshold).astype(int)
        else:
            labels = np.zeros(len(features), dtype=int)
            labels[future_returns > args.up_threshold] = 1
            labels[future_returns < -args.down_threshold] = -1
            features["label"] = labels
    if args.meta_label:
        meta = _compute_meta_labels(features["close"], args.meta_horizon, args.meta_up, args.meta_down)
        features["meta_label"] = meta
    features = features.dropna()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(args.output)
    if args.label_mode == "binary":
        label_mapping = {"0": 0, "1": 1}
    else:
        label_mapping = {"-1": 0, "0": 1, "1": 2}
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = {"label", "mid"}
    feature_columns = [col for col in numeric_cols if col not in exclude_cols]
    stats = {
        "feature_columns": feature_columns,
        "mean": features.mean(numeric_only=True).to_dict(),
        "std": features.std(numeric_only=True).replace(0, 1).to_dict(),
        "label_counts": features["label"].value_counts().to_dict(),
        "label_mode": args.label_mode,
        "label_mapping": label_mapping,
    }
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved {len(features)} rows to {args.output}")


if __name__ == "__main__":
    main()
