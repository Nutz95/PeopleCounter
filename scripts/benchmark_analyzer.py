#!/usr/bin/env python3
"""Benchmark CSV analyzer for PeopleCounter captures.

Usage:
    python scripts/benchmark_analyzer.py /path/to/benchmark_capture.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import fmean
from typing import Iterable

METRICS = [
    "e2e_ms",
    "src_wait_ms",
    "preproc_ms",
    "infer_critical_ms",
    "postdecode_ms",
    "other_ms",
    "src_copy_sync_ms",
    "trt_prepare_ms",
    "trt_sync_ms",
    "decode_ms",
    "publish_total_ms",
]

SPIKE_COMPONENTS = [
    "src_wait_ms",
    "preproc_ms",
    "infer_critical_ms",
    "postdecode_ms",
    "other_ms",
]


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    w = pos - lo
    return sorted_values[lo] * (1.0 - w) + sorted_values[hi] * w


def ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}ms"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_metric_stats(rows: list[dict[str, str]], metric: str) -> dict[str, float] | None:
    values = [v for v in (to_float(r.get(metric)) for r in rows) if v is not None]
    values.sort()
    if not values:
        return None
    return {
        "mean": fmean(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": values[-1],
    }


def dominant_component(row: dict[str, str]) -> tuple[str, float | None]:
    best_name = "n/a"
    best_value = None
    for name in SPIKE_COMPONENTS:
        value = to_float(row.get(name))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_name = name
            best_value = value
    return best_name, best_value


def top_spikes(rows: list[dict[str, str]], k: int = 10) -> list[dict[str, str | float | None]]:
    scored: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        e2e = to_float(row.get("e2e_ms"))
        if e2e is None:
            continue
        scored.append((e2e, row))
    scored.sort(key=lambda x: x[0], reverse=True)

    out: list[dict[str, str | float | None]] = []
    for e2e, row in scored[:k]:
        comp_name, comp_value = dominant_component(row)
        out.append(
            {
                "frame_id": row.get("frame_id", "?"),
                "e2e_ms": e2e,
                "dominant_component": comp_name,
                "dominant_ms": comp_value,
                "mode": row.get("mode", ""),
                "sync_mode": row.get("sync_mode", ""),
            }
        )
    return out


def print_summary(rows: list[dict[str, str]]) -> None:
    print(f"Rows: {len(rows)}")
    print()
    print("Metric              mean     p50      p95      p99      max")
    print("---------------------------------------------------------------")
    for metric in METRICS:
        stats = compute_metric_stats(rows, metric)
        if stats is None:
            print(f"{metric:<18} n/a      n/a      n/a      n/a      n/a")
            continue
        print(
            f"{metric:<18} "
            f"{ms(stats['mean']):<8} "
            f"{ms(stats['p50']):<8} "
            f"{ms(stats['p95']):<8} "
            f"{ms(stats['p99']):<8} "
            f"{ms(stats['max']):<8}"
        )

    print()
    print("Top e2e spikes (dominant component):")
    for i, spike in enumerate(top_spikes(rows), start=1):
        print(
            f"{i:2d}. frame {str(spike['frame_id']):>5} | "
            f"e2e={ms(spike['e2e_ms'])} | "
            f"{spike['dominant_component']}={ms(spike['dominant_ms'])} | "
            f"mode={spike['mode']} sync={spike['sync_mode']}"
        )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze PeopleCounter benchmark CSV captures")
    parser.add_argument("csv_path", type=Path, help="Path to benchmark_capture_*.csv")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    if not args.csv_path.exists():
        print(f"ERROR: file not found: {args.csv_path}")
        return 2

    rows = read_rows(args.csv_path)
    if not rows:
        print("ERROR: no rows found in CSV")
        return 3

    print_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
