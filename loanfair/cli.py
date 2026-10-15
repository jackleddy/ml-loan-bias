"""Shared argument handling for the small executable scripts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import DEFAULT_CONFIG, read_config, write_json


def main(command):
    parser = argparse.ArgumentParser(description=f"Virginia mortgage pricing: {command.replace('_', ' ')}")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    if command in {"fetch_references", "run_pipeline"}:
        parser.add_argument("--offline", action="store_true", help="Use cached raw reference files only.")
        parser.add_argument("--refresh", action="store_true", help="Replace cached references with official downloads.")
    if command in {"train_models", "run_pipeline"}:
        from .train import SCENARIOS
        parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=SCENARIOS, help="Must include primary; default runs every sensitivity.")
    if command == "compare_loan":
        parser.add_argument("record", type=Path, help="Borrower JSON input.")
        parser.add_argument("--output", type=Path, help="Also save the result as JSON.")
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        if command in {"fetch_references", "run_pipeline"}:
            from .references import fetch_references
            if args.offline and args.refresh:
                raise ValueError("--offline and --refresh cannot be combined.")
            fetch_references(config, args.offline, args.refresh)
        if command in {"prepare_data", "run_pipeline"}:
            from .prepare import prepare
            prepare(config)
        if command in {"train_models", "run_pipeline"}:
            from .train import train
            train(config, args.scenarios)
        if command in {"analyze_fairness", "run_pipeline"}:
            from .analysis import analyze
            analyze(config)
        if command == "compare_loan":
            from .compare import compare
            result = compare(json.loads(args.record.read_text(encoding="utf-8")), config)
            if args.output:
                write_json(args.output, result)
            print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, FileNotFoundError, RuntimeError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
