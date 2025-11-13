from __future__ import annotations

import argparse
from pathlib import Path

from lianghua.config import load_config
from lianghua.pipelines.simulation import run_simulation, _summaries
from lianghua.ui.forms import prompt_simulation_inputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A-share simulation runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sample.yaml"),
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch GUI to override symbol/date/feed before running.",
    )
    return parser


def run_cli() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    if args.interactive:
        config = prompt_simulation_inputs(config)
    result = run_simulation(config)
    summary = _summaries(result)
    print("Simulation summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    run_cli()
