from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict

from lianghua.config import AppConfig


def _build_label_entry(
    parent: tk.Widget,
    text: str,
    row: int,
    textvariable: tk.StringVar,
    width: int = 25,
) -> ttk.Entry:
    ttk.Label(parent, text=text).grid(row=row, column=0, sticky="e", padx=6, pady=4)
    entry = ttk.Entry(parent, textvariable=textvariable, width=width)
    entry.grid(row=row, column=1, sticky="w", padx=6, pady=4)
    return entry


def prompt_simulation_inputs(config: AppConfig) -> AppConfig:
    root = tk.Tk()
    root.title("Lianghua Interactive Setup")
    root.resizable(False, False)
    container = ttk.Frame(root, padding=12)
    container.grid(row=0, column=0)

    feed_var = tk.StringVar(value=config.data.feed)
    symbol_var = tk.StringVar(
        value=config.data.params.symbol
        or config.strategy.params.get("symbol", "")
        or (config.visualization.symbol or "")
    )
    start_var = tk.StringVar(value=config.data.params.start or config.simulation.start)
    end_var = tk.StringVar(value=config.data.params.end or config.simulation.end)
    freq_var = tk.StringVar(
        value=config.data.params.frequency
        or config.data.params.resolution
        or "D"
    )
    path_var = tk.StringVar(value=config.data.params.path or "")
    cash_var = tk.StringVar(value=f"{config.simulation.cash}")
    commission_var = tk.StringVar(value=f"{config.simulation.commission_bps}")
    slippage_var = tk.StringVar(value=f"{config.simulation.slippage_bps}")
    short_var = tk.StringVar(value=str(config.strategy.params.get("short_window", "")))
    long_var = tk.StringVar(value=str(config.strategy.params.get("long_window", "")))
    finnhub_var = tk.StringVar(value=config.data.params.api_key or "")
    tushare_var = tk.StringVar(value=config.data.params.token or "")
    proxy_var = tk.StringVar(value=config.data.params.proxy or "")
    ws_proxy_var = tk.StringVar(value=config.data.params.ws_proxy or "")
    live_var = tk.BooleanVar(value=config.visualization.live)
    window_var = tk.StringVar(value=str(config.visualization.window_minutes))
    latest_var = tk.BooleanVar(value=bool(config.data.params.latest))
    limit_var = tk.StringVar(value=str(config.data.params.limit or 1000))

    ttk.Label(container, text="数据源").grid(row=0, column=0, sticky="e", padx=6, pady=4)
    feed_combo = ttk.Combobox(
        container,
        textvariable=feed_var,
        values=("csv", "tushare", "finnhub", "binance"),
        state="readonly",
        width=22,
    )
    feed_combo.grid(row=0, column=1, sticky="w", padx=6, pady=4)

    _build_label_entry(container, "股票代码（ts_code / ticker）", 1, symbol_var)
    _build_label_entry(container, "开始日期 (YYYY-MM-DD)", 2, start_var)
    _build_label_entry(container, "结束日期 (YYYY-MM-DD)", 3, end_var)
    _build_label_entry(container, "频率/分辨率 / 聚合秒数", 4, freq_var)
    _build_label_entry(container, "CSV 路径（仅 CSV 时）", 5, path_var, width=40)
    ttk.Checkbutton(
        container,
        text="Binance REST 使用最新数据（忽略起止时间）",
        variable=latest_var,
    ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(2, 2))
    _build_label_entry(container, "Binance REST Limit", 7, limit_var)

    ttk.Separator(container, orient="horizontal").grid(
        row=8, column=0, columnspan=2, sticky="ew", pady=(6, 6)
    )

    _build_label_entry(container, "初始资金 (CNY)", 9, cash_var)
    _build_label_entry(container, "佣金 (bps)", 10, commission_var)
    _build_label_entry(container, "滑点 (bps)", 11, slippage_var)

    ttk.Separator(container, orient="horizontal").grid(
        row=12, column=0, columnspan=2, sticky="ew", pady=(6, 6)
    )

    _build_label_entry(container, "短期窗口", 13, short_var)
    _build_label_entry(container, "长期窗口", 14, long_var)

    ttk.Separator(container, orient="horizontal").grid(
        row=15, column=0, columnspan=2, sticky="ew", pady=(6, 6)
    )

    _build_label_entry(container, "Finnhub API Key", 16, finnhub_var, width=40)
    _build_label_entry(container, "Tushare Token", 17, tushare_var, width=40)
    _build_label_entry(container, "HTTP Proxy (如 http://127.0.0.1:7897)", 18, proxy_var, width=40)
    _build_label_entry(container, "WS Proxy (可选)", 19, ws_proxy_var, width=40)

    live_check = ttk.Checkbutton(
        container,
        text="启用实时可视化",
        variable=live_var,
    )
    live_check.grid(row=20, column=0, columnspan=2, pady=(4, 4))

    _build_label_entry(container, "可视化窗口（分钟）", 21, window_var)

    status_var = tk.StringVar(value="")
    ttk.Label(container, textvariable=status_var, foreground="#555").grid(
        row=22, column=0, columnspan=2, pady=(8, 4)
    )

    state: Dict[str, Any] = {"submitted": False}

    def on_submit() -> None:
        symbol = symbol_var.get().strip()
        start = start_var.get().strip()
        end = end_var.get().strip()
        if not symbol:
            messagebox.showerror("缺少字段", "请填写股票代码（symbol / ts_code）")
            return
        if not start or not end:
            messagebox.showerror("缺少字段", "请填写开始/结束日期")
            return
        if feed_var.get() == "csv" and not path_var.get().strip():
            messagebox.showerror("缺少路径", "CSV 源需要提供文件路径")
            return
        try:
            float(cash_var.get())
            float(commission_var.get())
            float(slippage_var.get())
        except ValueError:
            messagebox.showerror("数值错误", "请检查资金、佣金或滑点字段是否为数字")
            return
        if short_var.get().strip() and long_var.get().strip():
            try:
                short_v = int(short_var.get())
                long_v = int(long_var.get())
                if short_v >= long_v:
                    messagebox.showerror("窗口设置无效", "短期窗口必须小于长期窗口")
                    return
            except ValueError:
                messagebox.showerror("窗口设置无效", "均线窗口必须为整数")
                return
        if window_var.get().strip():
            try:
                int(window_var.get())
            except ValueError:
                messagebox.showerror("数值错误", "可视化窗口必须为整数")
                return
        if latest_var.get():
            try:
                int(limit_var.get())
            except ValueError:
                messagebox.showerror("数值错误", "Binance REST limit 必须为整数")
                return
        state["submitted"] = True
        root.destroy()

    def on_cancel() -> None:
        state["submitted"] = False
        root.destroy()

    btn_frame = ttk.Frame(container)
    btn_frame.grid(row=23, column=0, columnspan=2, pady=(8, 0))
    ttk.Button(btn_frame, text="取消", command=on_cancel).grid(row=0, column=0, padx=6)
    ttk.Button(btn_frame, text="开始模拟", command=on_submit).grid(row=0, column=1, padx=6)

    status_var.set("修改字段后点击“开始模拟”以使用新参数。")
    root.mainloop()

    if not state["submitted"]:
        return config

    updated = config.model_copy(deep=True)
    symbol = symbol_var.get().strip()
    start = start_var.get().strip()
    end = end_var.get().strip()
    freq = freq_var.get().strip()
    path = path_var.get().strip()
    feed = feed_var.get().strip()

    updated.data.feed = feed or updated.data.feed
    updated.data.params.symbol = symbol
    updated.strategy.params["symbol"] = symbol
    updated.visualization.symbol = symbol
    updated.data.params.start = start
    updated.data.params.end = end
    updated.simulation.start = start
    updated.simulation.end = end
    if freq:
        updated.data.params.frequency = freq
        updated.data.params.resolution = freq
        if updated.data.feed == "binance":
            try:
                updated.data.params.aggregate_seconds = int(float(freq))
            except ValueError:
                pass
    if path:
        updated.data.params.path = path

    updated.simulation.cash = float(cash_var.get())
    updated.simulation.commission_bps = float(commission_var.get())
    updated.simulation.slippage_bps = float(slippage_var.get())

    if short_var.get().strip():
        updated.strategy.params["short_window"] = int(short_var.get())
    if long_var.get().strip():
        updated.strategy.params["long_window"] = int(long_var.get())

    if finnhub_var.get().strip():
        updated.data.params.api_key = finnhub_var.get().strip()
    if tushare_var.get().strip():
        updated.data.params.token = tushare_var.get().strip()
    if proxy_var.get().strip():
        updated.data.params.proxy = proxy_var.get().strip()
    if ws_proxy_var.get().strip():
        updated.data.params.ws_proxy = ws_proxy_var.get().strip()

    updated.visualization.live = bool(live_var.get())
    if window_var.get().strip():
        updated.visualization.window_minutes = int(window_var.get().strip())
    updated.data.params.latest = bool(latest_var.get())
    if limit_var.get().strip():
        updated.data.params.limit = int(limit_var.get().strip())

    return updated
