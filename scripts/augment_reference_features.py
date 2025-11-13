from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge reference asset features (e.g., BTC) into the primary dataset (e.g., DOGE)."
    )
    parser.add_argument("--primary", type=Path, required=True, help="Primary parquet (e.g., DOGE).")
    parser.add_argument("--primary-stats", type=Path, required=True, help="Stats json for primary dataset.")
    parser.add_argument("--reference", type=Path, required=True, help="Reference parquet (e.g., BTC).")
    parser.add_argument("--prefix", type=str, default="btc", help="Column prefix for reference features.")
    parser.add_argument(
        "--columns",
        type=str,
        default="close,ret_1,ret_mean_1,ret_std_1,ret_ema_fast,ret_ema_slow,log_ret,realized_vol,"
        "price_position_120,momentum_3,momentum_6,momentum_12,vwap,vwap_delta",
        help="Comma list of reference columns to merge; defaults cover momentum/volatility/VWAP.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Augmented parquet output path.")
    parser.add_argument("--stats-output", type=Path, required=True, help="Augmented stats json output path.")
    return parser.parse_args()


def _load_dataframe(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ("timestamp", "open_time", "date"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col)
                break
    return df.sort_index()


def _select_reference_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    available = [col for col in columns if col in df.columns]
    if not available:
        raise SystemExit("No requested reference columns were found in the reference dataset.")
    return df[available]


def _compute_stats(df: pd.DataFrame, label_mode: str, label_mapping: dict) -> dict:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"label", "mid"}
    feature_columns = [col for col in numeric_cols if col not in exclude]
    stats = {
        "feature_columns": feature_columns,
        "mean": df[feature_columns].mean().to_dict(),
        "std": df[feature_columns].std().replace(0, 1).to_dict(),
        "label_counts": df["label"].value_counts().to_dict() if "label" in df else {},
        "label_mode": label_mode,
        "label_mapping": label_mapping,
    }
    return stats


def main() -> None:
    args = _parse_args()
    primary_df = _load_dataframe(args.primary)
    ref_df = _load_dataframe(args.reference)

    # align reference to primary timeline
    ref_cols = [col.strip() for col in args.columns.split(",") if col.strip()]
    ref_selected = _select_reference_columns(ref_df, ref_cols)
    aligned = ref_selected.reindex(primary_df.index).ffill().bfill()
    aligned.columns = [f"{args.prefix}_{col}" for col in aligned.columns]

    augmented = primary_df.copy()
    for col in aligned.columns:
        augmented[col] = aligned[col]

    primary_stats = json.loads(args.primary_stats.read_text(encoding="utf-8"))
    label_mode = primary_stats.get("label_mode", "triple")
    label_mapping = primary_stats.get("label_mapping", {"-1": 0, "0": 1, "1": 2})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_parquet(args.output)

    stats = _compute_stats(augmented, label_mode, label_mapping)
    args.stats_output.parent.mkdir(parents=True, exist_ok=True)
    args.stats_output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"Saved augmented features to {args.output} with prefix '{args.prefix}'.")


if __name__ == "__main__":
    main()
