from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from app_v2.enums import FusionStrategyType


DEFAULT_STAGE_BUDGET_MS_BY_STRATEGY: dict[str, dict[str, float]] = {
    FusionStrategyType.STRICT_SYNC.value: {
        "nvdec_ms": 35.0,
        "preprocess_nv12_bridge_ms": 15.0,
        "preprocess_ms": 32.0,
        "preprocess_model_yolo_global_ms": 6.0,
        "preprocess_model_yolo_tiles_ms": 8.0,
        "preprocess_model_max_ms": 12.0,
        "preprocess_model_sum_ms": 20.0,
        "preprocess_critical_path_ms": 27.0,
        "preprocess_serial_overhead_ms": 8.0,
        "fusion_wait_ms": 20.0,
        "overlay_lag_ms": 10.0,
        "end_to_end_ms": 130.0,
        "tensor_pool_wait_ms": 2.0,
    },
    FusionStrategyType.ASYNC_OVERLAY.value: {
        "nvdec_ms": 42.0,
        "preprocess_nv12_bridge_ms": 16.0,
        "preprocess_ms": 30.0,
        "preprocess_model_yolo_global_ms": 6.0,
        "preprocess_model_yolo_tiles_ms": 9.0,
        "preprocess_model_max_ms": 12.0,
        "preprocess_model_sum_ms": 20.0,
        "preprocess_critical_path_ms": 27.0,
        "preprocess_serial_overhead_ms": 10.0,
        "fusion_wait_ms": 8.0,
        "overlay_lag_ms": 25.0,
        "end_to_end_ms": 110.0,
        "tensor_pool_wait_ms": 2.0,
    },
    FusionStrategyType.RAW_STREAM_WITH_METADATA.value: {
        "nvdec_ms": 35.0,
        "preprocess_nv12_bridge_ms": 15.0,
        "preprocess_ms": 28.0,
        "preprocess_model_yolo_global_ms": 6.0,
        "preprocess_model_yolo_tiles_ms": 9.0,
        "preprocess_model_max_ms": 11.0,
        "preprocess_model_sum_ms": 18.0,
        "preprocess_critical_path_ms": 25.0,
        "preprocess_serial_overhead_ms": 8.0,
        "fusion_wait_ms": 5.0,
        "overlay_lag_ms": 40.0,
        "end_to_end_ms": 95.0,
        "tensor_pool_wait_ms": 2.0,
    },
}


def _normalize_fusion_strategy(value: str | None) -> str:
    candidate = str(value or "").strip().upper()
    allowed = {member.value for member in FusionStrategyType}
    if candidate in allowed:
        return candidate
    return FusionStrategyType.ASYNC_OVERLAY.value


@dataclass(frozen=True)
class PerfBudgetReport:
    mode: str
    fusion_strategy: str
    checked: dict[str, tuple[float, float]]
    violations: dict[str, tuple[float, float]]
    summary: dict[str, float | str]

    @property
    def should_fail(self) -> bool:
        return self.mode == "fail" and bool(self.violations)


def _status_badge(value: float, limit: float) -> str:
    if limit <= 0.0:
        return "⚪"
    ratio = value / limit
    if ratio <= 1.0:
        return "🟢"
    if ratio <= 1.10:
        return "🟠"
    return "🔴"


def render_perf_budget_table(report: PerfBudgetReport) -> str:
    """Render a markdown table with color-coded budget health for quick perf review."""
    rows = [
        "| Statut | Métrique | Valeur (ms) | Budget (ms) | Utilisation | Impact 4K@30 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    target_period_ms = float(report.summary.get("target_period_ms", 33.3333333333))
    for metric_key in sorted(report.checked.keys()):
        value, limit = report.checked[metric_key]
        utilization_pct = (value / limit * 100.0) if limit > 0.0 else 0.0
        impact_pct = (value / target_period_ms * 100.0) if target_period_ms > 0.0 else 0.0
        rows.append(
            "| "
            f"{_status_badge(value, limit)}"
            f" | `{metric_key}` | {value:.3f} | {limit:.3f} | {utilization_pct:.1f}% | {impact_pct:.1f}% |"
        )
    return "\n".join(rows)


def perf_budget_mode() -> str:
    value = os.environ.get("PEOPLE_COUNTER_PERF_BUDGET_MODE", "warn").strip().lower()
    if value in {"off", "warn", "fail"}:
        return value
    return "warn"


def evaluate_perf_budget(
    snapshot: Mapping[str, float | int],
    stage_budget_ms: Mapping[str, float] | None = None,
    fusion_strategy: str | None = None,
) -> PerfBudgetReport:
    mode = perf_budget_mode()
    normalized_strategy = _normalize_fusion_strategy(fusion_strategy)
    strategy_budget = DEFAULT_STAGE_BUDGET_MS_BY_STRATEGY.get(
        normalized_strategy,
        DEFAULT_STAGE_BUDGET_MS_BY_STRATEGY[FusionStrategyType.ASYNC_OVERLAY.value],
    )
    budget = dict(stage_budget_ms or strategy_budget)

    checked: dict[str, tuple[float, float]] = {}
    violations: dict[str, tuple[float, float]] = {}

    for metric_key, limit_ms in budget.items():
        if metric_key not in snapshot:
            continue
        value = float(snapshot[metric_key])
        checked[metric_key] = (value, float(limit_ms))
        if value > float(limit_ms):
            violations[metric_key] = (value, float(limit_ms))

    target_fps = 30.0
    target_period_ms = 1000.0 / target_fps
    critical_path_ms = float(snapshot.get("preprocess_critical_path_ms", snapshot.get("preprocess_ms", 0.0)))
    model_sum_ms = float(snapshot.get("preprocess_model_sum_ms", 0.0))
    model_max_ms = float(snapshot.get("preprocess_model_max_ms", 0.0))
    summary: dict[str, float | str] = {
        "target_fps": target_fps,
        "target_period_ms": target_period_ms,
        "critical_path_ms": critical_path_ms,
        "fps_margin_ms": target_period_ms - critical_path_ms,
        "parallel_efficiency_ratio": model_sum_ms / model_max_ms if model_max_ms > 0.0 else 0.0,
    }

    return PerfBudgetReport(
        mode=mode,
        fusion_strategy=normalized_strategy,
        checked=checked,
        violations=violations,
        summary=summary,
    )
