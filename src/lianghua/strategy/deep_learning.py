from __future__ import annotations

import json
from collections import defaultdict, deque
from datetime import timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import pandas as pd
import torch

from lianghua.data.feed import Bar
from lianghua.features import FeatureEngineer
from lianghua.strategy.base import Signal, Strategy
from lianghua.strategy.moving_average import MovingAverageStrategy


def _normalize_scenario_key(value):
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        return text or None
    text = str(value).strip().lower()
    return text or None


@dataclass
class DLStrategyConfig:
    model_path: Path
    metadata_path: Path
    seq_len: int = 60
    min_confidence: float = 0.55
    max_position: float = 1.0
    volatility_window: int = 30
    fallback_weight: float = 0.0


class DeepLearningStrategy(Strategy):
    def __init__(self, **params) -> None:
        if "model_path" not in params:
            raise ValueError("model_path is required for DL strategy")
        model_path = Path(params["model_path"])
        meta_path = Path(params.get("metadata_path", model_path.with_suffix(".meta.json")))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.feature_cols = meta["feature_columns"]
        self.seq_len = meta.get("seq_len", params.get("seq_len", 60))
        norm = meta["normalization"]
        self.mean = np.array(norm["mean"], dtype=np.float32)
        self.std = np.array(norm["std"], dtype=np.float32)
        self.std[self.std == 0] = 1.0
        mapping_raw = meta.get("label_mapping")
        if mapping_raw:
            self.label_mapping = {int(k): v for k, v in mapping_raw.items()}
        else:
            self.label_mapping = {0: -1, 1: 0, 2: 1}
        self.label_mode = meta.get("label_mode", "triple")
        self.scenario_column = meta.get("scenario_column") or params.get("scenario_column")
        model_cache: dict[str, torch.jit.ScriptModule] = {}

        def _load_module(path_str: str) -> torch.jit.ScriptModule:
            path = Path(path_str)
            key = str(path.resolve())
            if key not in model_cache:
                model_cache[key] = torch.jit.load(str(path), map_location=self.device).eval()
            return model_cache[key]

        self.scenario_models: dict[Optional[str], list[tuple[torch.jit.ScriptModule, float]]] = {}
        scenarios_meta = meta.get("scenarios") or []
        for scenario_entry in scenarios_meta:
            key = _normalize_scenario_key(scenario_entry.get("value"))
            specs = []
            for item in scenario_entry.get("models", []):
                module = _load_module(item["path"])
                specs.append((module, float(item.get("temperature", 1.0))))
            if specs:
                self.scenario_models[key] = specs

        self.default_models: list[tuple[torch.jit.ScriptModule, float]] = []
        models_meta = meta.get("models") or []
        for item in models_meta:
            module = _load_module(item["path"])
            spec = (module, float(item.get("temperature", 1.0)))
            scen_key = _normalize_scenario_key(item.get("scenario"))
            if scen_key is None:
                self.default_models.append(spec)
            else:
                self.scenario_models.setdefault(scen_key, []).append(spec)
        if not self.default_models:
            self.default_models = list(self.scenario_models.get(None, []))
        if not self.default_models and self.scenario_models:
            first_specs = next(iter(self.scenario_models.values()))
            self.default_models = list(first_specs)
        if not self.default_models:
            fallback_module = torch.jit.load(str(model_path), map_location=self.device).eval()
            self.default_models = [(fallback_module, 1.0)]
        thresholds = params.get("thresholds", {"flat": 0.4})
        self.min_conf = params.get("min_confidence", 0.55)
        self.flat_th = thresholds.get("flat", 0.4)
        self.min_conf_low_vol = params.get("min_confidence_low_vol", max(0.4, self.min_conf - 0.05))
        self.vol_adapt_window = params.get("vol_adapt_window", 120)
        self.vol_adapt_factor = params.get("vol_adapt_factor", 0.5)
        self.max_position = params.get("max_position", 1.0)
        interval_suffixes = set()
        for col in self.feature_cols:
            for prefix in ("close_", "ret_", "volume_"):
                if col.startswith(prefix):
                    suffix = col[len(prefix) :]
                    if suffix.endswith("m") and suffix[:-1].isdigit():
                        minutes = suffix[:-1]
                        interval_suffixes.add(f"{minutes}min")
        self.fe = FeatureEngineer(extra_intervals=sorted(interval_suffixes))
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.symbol = params.get("symbol")
        if not self.symbol:
            raise ValueError("symbol must be provided in strategy params")
        self.vol_window = params.get("volatility_window", 30)
        self.recent_returns: Deque[float] = deque(maxlen=self.vol_window)
        self.recent_returns_long: Deque[float] = deque(maxlen=self.vol_adapt_window)
        self.fallback_weight = params.get("fallback_weight", 0.0)
        self.allow_short = bool(params.get("allow_short", False))
        self.confirm_bars = int(params.get("confirm_bars", 2))
        self.pred_history: Deque[int] = deque(maxlen=self.confirm_bars)
        self.use_indicator_filter = bool(params.get("use_indicator_filter", False))
        self.ema_confirm_fast = int(params.get("ema_confirm_fast", 9))
        self.ema_confirm_slow = int(params.get("ema_confirm_slow", 21))
        self.rsi_confirm = float(params.get("rsi_confirm", 55.0))
        self.regime_required = params.get("regime_required")
        self.cooldown_bars = int(params.get("cooldown_bars", 0))
        self.cooldown_counter = 0
        self.health_window = int(params.get("health_window", 50))
        self.min_health = float(params.get("min_health", 0.55))
        self.health_scores: Deque[int] = deque(maxlen=self.health_window)
        self.pending_direction: Optional[int] = None
        self.require_meta_label = bool(params.get("require_meta_label", False))
        self.risk_budget = float(params.get("risk_budget", 0.1))
        self.max_trades_per_window = int(params.get("max_trades_per_window", 5))
        self.trade_window_minutes = int(params.get("trade_window_minutes", 60))
        self.loss_streak_limit = int(params.get("loss_streak_limit", 4))
        self.loss_cooldown_bars = int(params.get("loss_cooldown_bars", 30))
        self.scenario_loss_streak: defaultdict[Optional[str], int] = defaultdict(int)
        self.last_trade_scenario: Optional[str] = None
        self.trade_timestamps: Deque[pd.Timestamp] = deque()
        self.fallback_strategy = None
        if params.get("fallback_strategy", True):
            self.fallback_strategy = MovingAverageStrategy(
                symbol=self.symbol,
                short_window=params.get("fallback_short_window", 5),
                long_window=params.get("fallback_long_window", 20),
            )
        overrides = params.get("scenario_overrides") or {}
        self.scenario_overrides: dict[Optional[str], dict] = {}
        for key, cfg in overrides.items():
            norm_key = _normalize_scenario_key(key)
            self.scenario_overrides[norm_key] = dict(cfg)
        self.grid_states: dict[Optional[str], dict] = {}

    def _standardize(self, row: pd.Series) -> np.ndarray:
        aligned = row.reindex(self.feature_cols)
        aligned = aligned.infer_objects(copy=False).fillna(0.0)
        vec = aligned.values.astype(np.float32, copy=False)
        return (vec - self.mean) / self.std

    def _volatility(self) -> float:
        if len(self.recent_returns) < 2:
            return 0.01
        arr = np.array(self.recent_returns)
        return float(arr.std() + 1e-4)

    def _adaptive_confidence_threshold(self, min_conf: float, min_conf_low_vol: float) -> float:
        if len(self.recent_returns_long) < max(10, self.vol_adapt_window // 2):
            return min_conf
        long_vol = float(np.std(list(self.recent_returns_long)) + 1e-5)
        if long_vol < 0.0008:
            return max(min_conf_low_vol, min_conf - self.vol_adapt_factor * 0.1)
        return min_conf

    def _adaptive_risk_budget(self, vol: float, base_budget: float) -> float:
        if vol <= 0:
            return base_budget
        target_vol = 0.002
        scale = np.clip(target_vol / vol, 0.5, 1.5)
        return float(base_budget * scale)

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        if bar.symbol != self.symbol:
            return None
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return None
        features = self.fe.update(bar)
        if features is None:
            return None
        feat_vec = self._standardize(features)
        if np.isnan(feat_vec).any():
            return None
        self.buffer.append(feat_vec)
        ret = features.get("ret_1", 0.0)
        self.recent_returns.append(float(ret))
        self.recent_returns_long.append(float(ret))
        self._update_health(ret)
        if self._current_health() < self.min_health:
            return self._fallback_or_overlay(bar, features, {})
        if len(self.buffer) < self.seq_len:
            return None
        scenario_value = features.get(self.scenario_column) if self.scenario_column else None
        scenario_cfg = self._scenario_config(scenario_value)
        min_conf = scenario_cfg.get("min_confidence", self.min_conf)
        min_conf_low_vol = scenario_cfg.get("min_confidence_low_vol", self.min_conf_low_vol)
        flat_th = scenario_cfg.get("flat_threshold", self.flat_th)
        max_position = scenario_cfg.get("max_position", self.max_position)
        risk_budget = scenario_cfg.get("risk_budget", self.risk_budget)
        allow_short = scenario_cfg.get("allow_short", self.allow_short)
        require_meta = scenario_cfg.get("require_meta_label", self.require_meta_label)
        cooldown_len = int(scenario_cfg.get("cooldown_bars", self.cooldown_bars))
        if require_meta and features.get("meta_label", 1) <= 0:
            return self._fallback_or_overlay(bar, features, scenario_cfg)
        model_specs = self._select_models_for_scenario(scenario_value)
        if not model_specs:
            return self._fallback_or_overlay(bar, features, scenario_cfg)
        with torch.no_grad():
            tensor = torch.from_numpy(np.stack(self.buffer)).unsqueeze(0).to(self.device)
            logits_list = []
            for module, temperature in model_specs:
                logits = module(tensor)
                if temperature != 1.0:
                    logits = logits / temperature
                logits_list.append(logits)
            logits = torch.stack(logits_list).mean(dim=0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        cls = int(np.argmax(probs))
        conf = float(probs[cls])
        self.pred_history.append(cls)
        if self.confirm_bars > 1 and len(self.pred_history) == self.confirm_bars:
            recent = list(self.pred_history)
            if len(set(recent)) > 1:
                return self._fallback_or_overlay(bar, features, scenario_cfg)
        if conf < self._adaptive_confidence_threshold(min_conf, min_conf_low_vol):
            return self._fallback_or_overlay(bar, features, scenario_cfg)
        raw_label = self.label_mapping.get(cls, 0)
        if raw_label > 0:
            target = 1.0
        elif raw_label < 0 and allow_short:
            target = -1.0
        else:
            target = 0.0
        if target == 0.0 or conf < flat_th:
            return self._fallback_or_overlay(bar, features, scenario_cfg)
        if self.use_indicator_filter and not self._passes_indicator_filter(features, target):
            return self._fallback_or_overlay(bar, features, scenario_cfg)
        if not self._can_trade(bar.timestamp):
            return self._fallback_or_overlay(bar, features, scenario_cfg)
        vol = self._volatility()
        magnitude = (conf - 0.5) * 2.0  # [-1,1]
        risk = self._adaptive_risk_budget(vol, risk_budget)
        scaled = np.clip(target * magnitude * (risk / max(vol, 1e-4)), -max_position, max_position)
        scenario_key = _normalize_scenario_key(scenario_value)
        if abs(scaled) > 0:
            self.cooldown_counter = cooldown_len
            self.pending_direction = int(np.sign(target)) if target != 0 else None
            self._record_trade_time(bar.timestamp)
            self.last_trade_scenario = scenario_key
        else:
            self.last_trade_scenario = None
        return Signal(symbol=self.symbol, target_weight=scaled)

    def _select_models_for_scenario(self, scenario_value) -> list[tuple[torch.jit.ScriptModule, float]]:
        key = _normalize_scenario_key(scenario_value)
        if key in self.scenario_models:
            return self.scenario_models[key]
        return self.default_models

    def _scenario_config(self, scenario_value) -> dict:
        key = _normalize_scenario_key(scenario_value)
        return self.scenario_overrides.get(key, {})
    
    def _scenario_loss_params(self, scenario_value) -> tuple[int, int]:
        cfg = self._scenario_config(scenario_value)
        limit = int(cfg.get("loss_streak_limit", self.loss_streak_limit))
        cooldown = int(cfg.get("loss_cooldown_bars", self.loss_cooldown_bars))
        return limit, cooldown

    def _mean_reversion_signal(self, features: pd.Series, scenario_cfg: dict) -> Optional[Signal]:
        cfg = scenario_cfg.get("mean_reversion")
        if not cfg or not cfg.get("enabled"):
            return None
        feature_name = cfg.get("feature", "price_zscore")
        zscore = features.get(feature_name)
        if zscore is None or np.isnan(zscore):
            return None
        lower = cfg.get("lower_z", -1.15)
        upper = cfg.get("upper_z", 1.15)
        target_weight = float(cfg.get("target_weight", scenario_cfg.get("max_position", self.max_position)))
        max_cap = scenario_cfg.get("max_position", self.max_position)
        target_weight = np.clip(target_weight, 0.0, max_cap)
        if zscore <= lower:
            return Signal(symbol=self.symbol, target_weight=target_weight)
        if zscore >= upper:
            return Signal(symbol=self.symbol, target_weight=0.0)
        return None

    def _grid_signal(self, features: pd.Series, scenario_cfg: dict) -> Optional[Signal]:
        cfg = scenario_cfg.get("grid")
        if not cfg or not cfg.get("enabled"):
            return None
        price = features.get("close")
        if price is None or np.isnan(price) or price <= 0:
            return None
        key = _normalize_scenario_key(features.get(self.scenario_column)) if self.scenario_column else None
        state = self.grid_states.setdefault(key, {"center": float(price)})
        alpha = float(cfg.get("center_ema", 0.08))
        state["center"] = (1 - alpha) * state["center"] + alpha * float(price)
        center = state["center"]
        entry_pct = float(cfg.get("entry_pct", 0.001))
        exit_pct = float(cfg.get("exit_pct", entry_pct * 1.5))
        target_weight = float(cfg.get("target_weight", scenario_cfg.get("max_position", self.max_position) * 0.5))
        max_cap = scenario_cfg.get("max_position", self.max_position)
        target_weight = np.clip(target_weight, 0.0, max_cap)
        band = cfg.get("band_pct")
        if band:
            entry_level = center * (1 - band)
            exit_level = center * (1 + band)
        else:
            entry_level = center * (1 - entry_pct)
            exit_level = center * (1 + exit_pct)
        if price <= entry_level:
            return Signal(symbol=self.symbol, target_weight=target_weight)
        if price >= exit_level:
            return Signal(symbol=self.symbol, target_weight=0.0)
        return None

    def _fallback_or_overlay(self, bar: Bar, features: pd.Series, scenario_cfg: dict) -> Optional[Signal]:
        overlay = self._mean_reversion_signal(features, scenario_cfg)
        if overlay is not None:
            return overlay
        grid_sig = self._grid_signal(features, scenario_cfg)
        if grid_sig is not None:
            return grid_sig
        return self._fallback_signal(bar)

    def _fallback_signal(self, bar: Bar) -> Optional[Signal]:
        if self.fallback_strategy:
            return self.fallback_strategy.on_bar(bar)
        return Signal(symbol=self.symbol, target_weight=self.fallback_weight)

    def _passes_indicator_filter(self, features: pd.Series, target: float) -> bool:
        fast_ema = features.get("ema_fast")
        slow_ema = features.get("ema_slow")
        rsi = features.get("rsi_14")
        regime = features.get("regime_state")
        if fast_ema is None or slow_ema is None or rsi is None:
            return False
        if target > 0:
            if fast_ema <= slow_ema or rsi < self.rsi_confirm:
                return False
        elif target < 0:
            if fast_ema >= slow_ema or rsi > (100 - self.rsi_confirm):
                return False
        if self.regime_required is not None and regime is not None:
            if int(regime) != int(self.regime_required):
                return False
        return True

    def _update_health(self, ret: Optional[float]) -> None:
        if self.pending_direction is None or ret is None:
            return
        direction = int(np.sign(ret)) if ret != 0 else 0
        if direction == self.pending_direction:
            self.health_scores.append(1)
            if self.last_trade_scenario is not None:
                self.scenario_loss_streak[self.last_trade_scenario] = 0
        else:
            self.health_scores.append(0)
            scenario_key = self.last_trade_scenario
            if scenario_key is not None:
                self.scenario_loss_streak[scenario_key] += 1
                limit, cooldown = self._scenario_loss_params(scenario_key)
                if self.scenario_loss_streak[scenario_key] >= limit:
                    self.cooldown_counter = max(self.cooldown_counter, cooldown)
                    self.scenario_loss_streak[scenario_key] = 0
        self.pending_direction = None
        self.last_trade_scenario = None

    def _current_health(self) -> float:
        if not self.health_scores:
            return 1.0
        return sum(self.health_scores) / len(self.health_scores)

    def _record_trade_time(self, timestamp: pd.Timestamp) -> None:
        self.trade_timestamps.append(timestamp)
        window_start = timestamp - pd.Timedelta(minutes=self.trade_window_minutes)
        while self.trade_timestamps and self.trade_timestamps[0] < window_start:
            self.trade_timestamps.popleft()

    def _can_trade(self, timestamp: pd.Timestamp) -> bool:
        window_start = timestamp - pd.Timedelta(minutes=self.trade_window_minutes)
        while self.trade_timestamps and self.trade_timestamps[0] < window_start:
            self.trade_timestamps.popleft()
        return len(self.trade_timestamps) < self.max_trades_per_window
