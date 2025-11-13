from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import ta


def _infer_scenario_labels(
    data: pd.DataFrame,
    *,
    trend_threshold: float = 0.0012,
    neutral_band: float = 0.00035,
    vol_threshold: float = 1.8,
) -> pd.Series:
    """
    Assign simple regime labels (trend_up/down/flat) using EMA slope and realized volatility.
    """
    price_ref = data["close"].rolling(3, min_periods=1).mean().replace(0, np.nan)
    ema_spread = (data["ema_fast"] - data["ema_slow"]) / price_ref
    ema_spread = ema_spread.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    realized_vol = data["realized_vol"].replace(0, np.nan)
    vol_anchor = realized_vol.rolling(60, min_periods=1).mean()
    vol_ratio = (realized_vol / (vol_anchor + 1e-9)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    scenario = np.full(len(data), "flat", dtype=object)
    strong_up = ema_spread >= trend_threshold
    strong_down = ema_spread <= -trend_threshold
    calm = np.abs(ema_spread) <= neutral_band
    choppy = vol_ratio >= vol_threshold

    scenario[strong_up & ~choppy] = "trend_up"
    scenario[strong_down & ~choppy] = "trend_down"
    scenario[choppy | calm] = "flat"
    return pd.Series(scenario, index=data.index, name="scenario_label")


def compute_feature_frame(
    df: pd.DataFrame,
    *,
    use_volume: bool = True,
    return_windows: Sequence[int] = (1, 3, 5, 10, 20),
    vol_window: int = 20,
    regime_fast: int = 12,
    regime_slow: int = 48,
    regime_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Given an OHLCV dataframe sorted by time, append a standard set of features.
    """
    data = df.copy()
    data["mid"] = (data["high"] + data["low"]) / 2
    for w in return_windows:
        data[f"ret_{w}"] = data["close"].pct_change(w)
        data[f"ret_mean_{w}"] = data["ret_1"].rolling(w, min_periods=1).mean()
        data[f"ret_std_{w}"] = (
            data["ret_1"].rolling(w, min_periods=1).std(ddof=0).fillna(0)
        )
    data["ret_ema_fast"] = data["ret_1"].ewm(span=5, adjust=False).mean()
    data["ret_ema_slow"] = data["ret_1"].ewm(span=20, adjust=False).mean()
    data["log_ret"] = np.log(data["close"]).diff()
    data["realized_vol"] = data["log_ret"].rolling(vol_window).std().fillna(0)
    data["vol_of_vol"] = data["realized_vol"].rolling(vol_window).std().fillna(0)
    data["log_volatility_ratio"] = (
        data["realized_vol"] / (data["realized_vol"].rolling(vol_window).mean() + 1e-9)
    ).fillna(1)
    data["abs_ret"] = data["ret_1"].abs()
    data["volume_ratio"] = (
        data["volume"] / data["volume"].rolling(vol_window).mean()
    ).fillna(1)
    data["price_zscore"] = (
        (data["close"] - data["close"].rolling(vol_window).mean())
        / (data["close"].rolling(vol_window).std() + 1e-9)
    ).fillna(0)
    data["hl_range_pct"] = (data["high"] - data["low"]) / data["close"]
    data["body_pct"] = (data["close"] - data["open"]) / data["open"]
    body_top = np.maximum(data["open"], data["close"])
    body_bottom = np.minimum(data["open"], data["close"])
    data["upper_shadow"] = (data["high"] - body_top) / data["close"]
    data["lower_shadow"] = (body_bottom - data["low"]) / data["close"]
    # Technical indicators
    data["rsi_14"] = ta.momentum.RSIIndicator(close=data["close"], window=14).rsi()
    macd = ta.trend.MACD(close=data["close"], window_slow=26, window_fast=12, window_sign=9)
    data["macd"] = macd.macd()
    data["macd_signal"] = macd.macd_signal()
    data["macd_hist"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(close=data["close"], window=20, window_dev=2)
    data["bb_width"] = (
        (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
    ).replace([np.inf, -np.inf], np.nan)
    atr = ta.volatility.AverageTrueRange(high=data["high"], low=data["low"], close=data["close"], window=14)
    data["atr_14"] = atr.average_true_range()
    ema_fast = ta.trend.EMAIndicator(close=data["close"], window=9)
    ema_slow = ta.trend.EMAIndicator(close=data["close"], window=26)
    data["ema_fast"] = ema_fast.ema_indicator()
    data["ema_slow"] = ema_slow.ema_indicator()
    data["ema_diff"] = data["ema_fast"] - data["ema_slow"]
    price_min = data["close"].rolling(120, min_periods=1).min()
    price_max = data["close"].rolling(120, min_periods=1).max()
    data["price_position_120"] = (data["close"] - price_min) / (price_max - price_min + 1e-9)
    data["range_ratio_120"] = ((price_max - price_min) / (price_min + 1e-9)).fillna(0)
    for window in (3, 6, 12, 24, 60):
        data[f"momentum_{window}"] = data["close"].pct_change(window)
    data["momentum_combo"] = (
        0.5 * data["momentum_3"].fillna(0) + 0.3 * data["momentum_6"].fillna(0) + 0.2 * data["momentum_12"].fillna(0)
    )
    # VWAP style features
    pv = data["close"] * data["volume"]
    vwap_window = max(30, vol_window)
    rolling_vol = data["volume"].rolling(vwap_window).sum()
    data["vwap"] = (pv.rolling(vwap_window).sum() / (rolling_vol + 1e-9)).bfill()
    data["vwap_delta"] = (data["close"] - data["vwap"]) / (data["vwap"] + 1e-9)
    kc = ta.volatility.KeltnerChannel(high=data["high"], low=data["low"], close=data["close"], window=20)
    data["keltner_bandwidth"] = (
        (kc.keltner_channel_hband() - kc.keltner_channel_lband()) / kc.keltner_channel_mband()
    ).replace([np.inf, -np.inf], np.nan)
    mfi = ta.volume.MFIIndicator(high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], window=14)
    data["mfi_14"] = mfi.money_flow_index()
    stoch = ta.momentum.StochasticOscillator(high=data["high"], low=data["low"], close=data["close"], window=14)
    data["stoch_k"] = stoch.stoch()
    data["stoch_d"] = stoch.stoch_signal()
    data["jump_flag"] = (data["abs_ret"] > data["abs_ret"].rolling(vol_window).mean() * 3).astype(int)
    ema_fast_regime = ta.trend.EMAIndicator(close=data["close"], window=regime_fast).ema_indicator()
    ema_slow_regime = ta.trend.EMAIndicator(close=data["close"], window=regime_slow).ema_indicator()
    regime_signal = ema_fast_regime - ema_slow_regime
    regime = np.zeros(len(data))
    regime[regime_signal > regime_threshold] = 1
    regime[regime_signal < -regime_threshold] = -1
    data["regime_state"] = regime
    if use_volume:
        data["money_flow"] = data["close"] * data["volume"]
        data["money_flow_ratio"] = data["money_flow"].rolling(vol_window).sum() / (
            data["money_flow"].rolling(vol_window * 2).sum() + 1e-9
        )
        data["volume_delta"] = data["volume"].diff().fillna(0)
        data["up_volume"] = np.where(data["close"] >= data["open"], data["volume"], 0)
        data["down_volume"] = np.where(data["close"] < data["open"], data["volume"], 0)
        total_vol = data["up_volume"] + data["down_volume"]
        data["volume_imbalance"] = np.where(total_vol == 0, 0, (data["up_volume"] - data["down_volume"]) / total_vol)
        data["volume_zscore"] = (
            (data["volume"] - data["volume"].rolling(vol_window).mean())
            / (data["volume"].rolling(vol_window).std() + 1e-9)
        ).fillna(0)
        obv = ta.volume.OnBalanceVolumeIndicator(close=data["close"], volume=data["volume"])
        data["obv"] = obv.on_balance_volume()
        chaikin = ta.volume.ChaikinMoneyFlowIndicator(
            high=data["high"], low=data["low"], close=data["close"], volume=data["volume"], window=20
        )
        data["chaikin_mf"] = chaikin.chaikin_money_flow()
    if "scenario_label" not in data.columns:
        data["scenario_label"] = _infer_scenario_labels(data)
    return data


def triple_barrier_labels(
    future_return: pd.Series,
    threshold: float = 0.002,
) -> pd.Series:
    labels = np.zeros(len(future_return), dtype=np.int8)
    labels[future_return > threshold] = 1
    labels[future_return < -threshold] = -1
    return pd.Series(labels, index=future_return.index, name="label")


def compute_labels(
    closes: pd.Series,
    horizon: int = 5,
    threshold: float = 0.002,
) -> pd.Series:
    future_returns = closes.pct_change(periods=horizon).shift(-horizon)
    return triple_barrier_labels(future_returns, threshold=threshold)


@dataclass
class FeatureEngineer:
    """
    Incremental feature helper for live usage.
    Keeps a rolling dataframe to recompute indicators when new bars arrive.
    """

    min_rows: int = 60
    window: int = 120
    extra_intervals: Sequence[str] = ()

    def __post_init__(self) -> None:
        self.buffer = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    def update(self, bar) -> Optional[pd.Series]:
        entry = {
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        row = pd.DataFrame([entry], index=[bar.timestamp])
        if self.buffer.empty:
            self.buffer = row
        else:
            self.buffer = pd.concat([self.buffer, row], axis=0)
        self.buffer = self.buffer.iloc[-self.window :]
        if len(self.buffer) < self.min_rows:
            return None
        feats = compute_feature_frame(self.buffer)
        if self.extra_intervals:
            feats = self._append_multi_interval(feats)
        feats["meta_label"] = 1
        series = feats.iloc[-1]
        return series

    def _append_multi_interval(self, base: pd.DataFrame) -> pd.DataFrame:
        """
        Append multi-interval aggregates (close/ret/volume) similar to offline builder.
        """
        if self.buffer.index.tzinfo is None:
            raw = self.buffer.copy()
            raw.index = pd.to_datetime(raw.index)
        else:
            raw = self.buffer
        for interval in self.extra_intervals:
            try:
                resampled = raw.resample(interval).agg(
                    {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
                )
            except ValueError:
                continue
            resampled = resampled.reindex(base.index).ffill()
            prefix = interval.replace(" ", "").replace("min", "m")
            base[f"close_{prefix}"] = resampled["close"]
            base[f"ret_{prefix}"] = resampled["close"].pct_change(fill_method=None).fillna(0)
            base[f"volume_{prefix}"] = resampled["volume"]
        return base
