from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os
import time
from typing import Iterator, Optional, Protocol
from urllib.parse import urlparse

import pandas as pd
import requests
import websocket

try:
    import tushare as ts  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    ts = None

from lianghua.config import DataConfig, SimulationConfig


@dataclass(slots=True)
class Bar:
    timestamp: pd.Timestamp
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class DataFeed(Protocol):
    def stream(self) -> Iterator[Bar]:  # pragma: no cover - interface
        ...


class CSVDataFeed:
    def __init__(self, path: str | Path, symbol: str | None = None) -> None:
        self.path = Path(path)
        self.symbol = symbol

    def stream(self) -> Iterator[Bar]:
        df = pd.read_csv(self.path, parse_dates=["date"])
        if self.symbol:
            df = df[df["symbol"] == self.symbol]
        df = df.sort_values("date")
        for _, row in df.iterrows():
            yield Bar(
                timestamp=row["date"],
                symbol=row["symbol"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["volume"]),
            )


class FinnhubDataFeed:
    BASE_URL = "https://finnhub.io/api/v1/stock/candle"

    def __init__(
        self,
        symbol: str,
        api_key: str,
        resolution: str = "D",
        start: str | None = None,
        end: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Finnhub API key is required")
        if not start or not end:
            raise ValueError("Finnhub feed requires both 'start' and 'end' timestamps")
        self.symbol = symbol
        self.api_key = api_key
        self.resolution = resolution or "D"
        self.start = start
        self.end = end

    def _to_epoch(self, value: str) -> int:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())

    def _fetch(self) -> dict:
        params = {
            "symbol": self.symbol,
            "resolution": self.resolution,
            "from": self._to_epoch(self.start),
            "to": self._to_epoch(self.end),
            "token": self.api_key,
        }
        resp = requests.get(self.BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("s")
        if status != "ok":
            raise RuntimeError(f"Finnhub API error: {status} ({payload})")
        return payload

    def stream(self) -> Iterator[Bar]:
        payload = self._fetch()
        for o, h, l, c, v, ts in zip(
            payload["o"], payload["h"], payload["l"], payload["c"], payload["v"], payload["t"]
        ):
            ts_pd = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Shanghai")
            yield Bar(
                timestamp=ts_pd,
                symbol=self.symbol,
                open=float(o),
                high=float(h),
                low=float(l),
                close=float(c),
                volume=float(v),
            )


class TushareDataFeed:
    def __init__(
        self,
        symbol: str,
        token: str,
        start: str | None,
        end: str | None,
        frequency: str = "D",
    ) -> None:
        if ts is None:
            raise ImportError("tushare package is not installed; run pip install tushare")
        if not token:
            raise ValueError("Tushare token is required")
        if not start or not end:
            raise ValueError("Tushare feed requires 'start' and 'end' dates")
        self.symbol = symbol
        self.start = self._format_date(start)
        self.end = self._format_date(end)
        self.frequency = frequency.upper() if frequency else "D"
        ts.set_token(token)
        self.api = ts.pro_api()

    def _format_date(self, value: str) -> str:
        return pd.Timestamp(value).strftime("%Y%m%d")

    def _load_dataframe(self) -> pd.DataFrame:
        if self.frequency in {"D", "W", "M"}:
            return self.api.daily(ts_code=self.symbol, start_date=self.start, end_date=self.end)
        return ts.pro_bar(  # pragma: no cover - heavy external call
            ts_code=self.symbol,
            start_date=self.start,
            end_date=self.end,
            freq=self.frequency.lower(),
        )

    def stream(self) -> Iterator[Bar]:
        df = self._load_dataframe()
        if df is None or df.empty:
            msg = (
                f"Tushare returned empty data for {self.symbol} between "
                f"{self.start} and {self.end}. "
                "Check token权限、调用额度或日期/频率是否有效。"
            )
            raise RuntimeError(msg)
        df = df.sort_values("trade_date")
        for _, row in df.iterrows():
            timestamp = pd.to_datetime(row["trade_date"]).tz_localize("Asia/Shanghai")
            volume = float(row.get("vol", row.get("volume", 0)))
            if "vol" in row:
                volume *= 100.0
            yield Bar(
                timestamp=timestamp,
                symbol=self.symbol,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=volume,
            )


class BinanceKlineFeed:
    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(
        self,
        symbol: str,
        interval: str = "1m",
        start: str | None = None,
        end: str | None = None,
        limit: int = 1000,
        proxy: Optional[str] = None,
        latest: bool = False,
    ) -> None:
        self.symbol = symbol.upper()
        self.interval = interval
        self.start = start
        self.end = end
        self.limit = limit
        self.proxy = proxy
        self._proxies = {"http": proxy, "https": proxy} if proxy else None
        self.latest = latest

    def _to_millis(self, value: str | None) -> Optional[int]:
        if not value:
            return None
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp() * 1000)

    def _fetch(self, start_ms: Optional[int], end_ms: Optional[int]) -> list:
        params = {"symbol": self.symbol, "interval": self.interval, "limit": self.limit}
        if start_ms:
            params["startTime"] = start_ms
        if end_ms:
            params["endTime"] = end_ms
        resp = requests.get(self.BASE_URL, params=params, timeout=10, proxies=self._proxies)
        resp.raise_for_status()
        return resp.json()

    def stream(self) -> Iterator[Bar]:
        start_ms = None if self.latest else self._to_millis(self.start)
        end_ms = None if self.latest else self._to_millis(self.end)
        while True:
            batch = self._fetch(start_ms, end_ms)
            if not batch:
                break
            for entry in batch:
                open_time = pd.to_datetime(entry[0], unit="ms", utc=True).tz_convert(
                    "Asia/Shanghai"
                )
                if end_ms and entry[0] >= end_ms:
                    return
                yield Bar(
                    timestamp=open_time,
                    symbol=self.symbol,
                    open=float(entry[1]),
                    high=float(entry[2]),
                    low=float(entry[3]),
                    close=float(entry[4]),
                    volume=float(entry[5]),
                )
            last_open = batch[-1][0]
            next_start = last_open + 1
            if end_ms and next_start >= end_ms:
                break
            start_ms = next_start


class BinanceTradeWebSocketFeed:
    STREAM_URL = "wss://stream.binance.com:9443/ws/{stream}"

    def __init__(
        self,
        symbol: str,
        aggregate_seconds: int = 1,
        proxy: Optional[str] = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.aggregate_seconds = max(1, aggregate_seconds)
        self.stream_name = f"{self.symbol.lower()}@trade"
        self.url = self.STREAM_URL.format(stream=self.stream_name)
        self.proxy = proxy
        self._ws = None

    def _bar_from_state(self, bucket, o, h, l, c, vol) -> Bar:
        return Bar(
            timestamp=bucket,
            symbol=self.symbol,
            open=float(o),
            high=float(h),
            low=float(l),
            close=float(c),
            volume=float(vol),
        )

    def stream(self) -> Iterator[Bar]:
        current_bucket: Optional[pd.Timestamp] = None
        open_price = high_price = low_price = close_price = None
        volume = 0.0
        try:
            while True:
                ws = self._ensure_connection()
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                except websocket.WebSocketConnectionClosedException:
                    self._reset_connection()
                    continue
                except Exception:
                    self._reset_connection()
                    time.sleep(1)
                    continue
                data = json.loads(raw)
                if data.get("e") != "trade":
                    continue
                ts = pd.to_datetime(data["T"], unit="ms", utc=True).tz_convert("Asia/Shanghai")
                bucket = ts.floor(f"{self.aggregate_seconds}s")
                price = float(data["p"])
                qty = float(data["q"])
                if current_bucket is None:
                    current_bucket = bucket
                    open_price = high_price = low_price = close_price = price
                    volume = qty
                    continue
                if bucket == current_bucket:
                    high_price = max(high_price, price)
                    low_price = min(low_price, price)
                    close_price = price
                    volume += qty
                else:
                    yield self._bar_from_state(
                        current_bucket, open_price, high_price, low_price, close_price, volume
                    )
                    current_bucket = bucket
                    open_price = high_price = low_price = close_price = price
                    volume = qty
        except KeyboardInterrupt:  # pragma: no cover - manual stop
            pass
        finally:
            if current_bucket is not None:
                yield self._bar_from_state(
                    current_bucket, open_price, high_price, low_price, close_price, volume
                )
            self._reset_connection()

    def _build_proxy_args(self) -> dict:
        if not self.proxy:
            return {}
        parsed = urlparse(self.proxy)
        kwargs = {
            "http_proxy_host": parsed.hostname,
            "http_proxy_port": parsed.port,
            "proxy_type": parsed.scheme or "http",
        }
        if parsed.username:
            kwargs["http_proxy_auth"] = (parsed.username, parsed.password or "")
        return kwargs

    def _ensure_connection(self):
        if self._ws is not None:
            return self._ws
        proxy_args = self._build_proxy_args()
        attempt = 0
        delay = 2.0
        while self._ws is None:
            attempt += 1
            try:
                self._ws = websocket.create_connection(self.url, timeout=10, **proxy_args)
            except Exception as exc:  # pragma: no cover - network
                print(f"[Binance WS] connect attempt {attempt} failed: {exc}")
                time.sleep(delay)
                delay = min(delay * 1.5, 30.0)
                continue
        return self._ws

    def _reset_connection(self) -> None:
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws = None


def _resolve_series_day(value: str | None, fallback: str | None) -> str | None:
    return value or fallback


def _env_is_disabled(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"0", "false", "no", "off"}


def build_data_feed(config: DataConfig, simulation: SimulationConfig | None = None) -> DataFeed:
    if config.feed == "csv":
        return CSVDataFeed(path=config.params.path, symbol=config.params.symbol)

    if config.feed == "finnhub":
        api_key = config.params.api_key or os.getenv("FINNHUB_API_KEY")
        start = _resolve_series_day(config.params.start, getattr(simulation, "start", None))
        end = _resolve_series_day(config.params.end, getattr(simulation, "end", None))
        return FinnhubDataFeed(
            symbol=config.params.symbol,
            api_key=api_key,
            resolution=config.params.resolution or "D",
            start=start,
            end=end,
        )

    if config.feed == "tushare":
        if _env_is_disabled(os.getenv("TUSHARE_ENABLED")):
            raise RuntimeError("Tushare feed disabled by TUSHARE_ENABLED environment variable")
        token = config.params.token or os.getenv("TUSHARE_TOKEN")
        start = _resolve_series_day(config.params.start, getattr(simulation, "start", None))
        end = _resolve_series_day(config.params.end, getattr(simulation, "end", None))
        return TushareDataFeed(
            symbol=config.params.symbol,
            token=token,
            start=start,
            end=end,
            frequency=config.params.frequency or config.params.resolution or "D",
        )

    if config.feed == "binance":
        mode = (config.params.mode or "rest").lower()
        symbol = config.params.symbol
        if not symbol:
            raise ValueError("Binance feed requires 'symbol'")
        proxy_url = (
            config.params.proxy
            or os.getenv("BINANCE_PROXY")
            or os.getenv("HTTPS_PROXY")
        )
        ws_proxy_url = (
            config.params.ws_proxy
            or config.params.proxy
            or os.getenv("BINANCE_WS_PROXY")
            or os.getenv("HTTPS_PROXY")
        )
        if mode == "rest":
            start = config.params.start or getattr(simulation, "start", None)
            end = config.params.end or getattr(simulation, "end", None)
            interval = config.params.interval or config.params.resolution or "1m"
            return BinanceKlineFeed(
                symbol=symbol,
                interval=interval,
                start=None if config.params.latest else start,
                end=None if config.params.latest else end,
                latest=bool(config.params.latest),
                limit=config.params.limit or 1000,
                proxy=proxy_url,
            )
        if mode == "websocket":
            agg = config.params.aggregate_seconds or 1
            return BinanceTradeWebSocketFeed(
                symbol=symbol,
                aggregate_seconds=agg,
                proxy=ws_proxy_url,
            )
        raise ValueError(f"Unsupported Binance mode: {mode}")

    raise ValueError(f"Unsupported data feed: {config.feed}")
