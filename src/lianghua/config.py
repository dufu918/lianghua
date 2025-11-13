from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, PositiveFloat, PositiveInt, field_validator


class SimulationConfig(BaseModel):
    start: str
    end: str
    cash: PositiveFloat
    commission_bps: float = 1.5
    slippage_bps: float = 1.0
    benchmark: Optional[str] = None


class DataParams(BaseModel):
    path: Optional[str] = None
    symbol: Optional[str] = None
    resolution: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    api_key: Optional[str] = None
    token: Optional[str] = None
    frequency: Optional[str] = None
    enabled: Optional[bool] = None
    interval: Optional[str] = None
    mode: Optional[str] = None
    aggregate_seconds: Optional[int] = None
    proxy: Optional[str] = None
    ws_proxy: Optional[str] = None
    latest: Optional[bool] = None
    limit: Optional[int] = None


class DataConfig(BaseModel):
    feed: str
    params: DataParams = DataParams()


class StrategyConfig(BaseModel):
    name: str
    params: Dict[str, Any] = {}


class ReportConfig(BaseModel):
    output: Optional[str] = None
    chart: Optional[str] = None


class VisualizationConfig(BaseModel):
    live: bool = False
    symbol: Optional[str] = None
    window_minutes: int = 20


class AppConfig(BaseModel):
    simulation: SimulationConfig
    data: DataConfig
    strategy: StrategyConfig
    report: ReportConfig = ReportConfig()
    visualization: VisualizationConfig = VisualizationConfig()

    @field_validator("data")
    @classmethod
    def validate_data(cls, value: DataConfig) -> DataConfig:
        if value.feed == "csv" and not value.params.path:
            msg = "CSV data feed requires 'path' in params"
            raise ValueError(msg)
        if value.feed in {"finnhub", "tushare"} and not value.params.symbol:
            msg = f"{value.feed} feed requires 'symbol' in params"
            raise ValueError(msg)
        return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return AppConfig(**payload)
