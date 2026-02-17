from __future__ import annotations

import os
from pathlib import Path
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
        "inference_model_yolo_global_ms": 14.0,
        "inference_model_yolo_tiles_ms": 18.0,
        "inference_model_sum_ms": 26.0,
        "inference_model_max_ms": 18.0,
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
        "inference_model_yolo_global_ms": 15.0,
        "inference_model_yolo_tiles_ms": 20.0,
        "inference_model_sum_ms": 30.0,
        "inference_model_max_ms": 20.0,
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
        "inference_model_yolo_global_ms": 12.0,
        "inference_model_yolo_tiles_ms": 16.0,
        "inference_model_sum_ms": 24.0,
        "inference_model_max_ms": 16.0,
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


def _status_css(value: float, limit: float) -> str:
    badge = _status_badge(value, limit)
    if badge == "🟢":
        return "ok"
    if badge == "🟠":
        return "warn"
    return "bad"


def _recommendations(report: PerfBudgetReport) -> list[str]:
    if not report.violations:
        return ["Aucune violation: la chaîne respecte le budget configuré."]

    ranked = sorted(
        report.violations.items(),
        key=lambda item: (item[1][0] / item[1][1]) if item[1][1] > 0.0 else 0.0,
        reverse=True,
    )
    tips: list[str] = []
    for metric_key, _ in ranked[:4]:
        if metric_key == "nvdec_ms":
            tips.append("NVDEC est dominant: valider codec/preset caméra et stabilité réseau.")
        elif metric_key == "preprocess_nv12_bridge_ms":
            tips.append("Pont NV12->Tensor coûteux: réduire copies/convertions et vérifier format source.")
        elif metric_key.startswith("inference_model_"):
            tips.append("Inference dépasse le budget: ajuster batch/engine TRT, précision, ou fréquences de branche.")
        elif metric_key == "end_to_end_ms":
            tips.append("Latence bout-en-bout élevée: inspecter files d’attente et pression en publication.")
        else:
            tips.append(f"Optimiser la métrique {metric_key} (au-dessus du budget).")
    return tips


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


def render_perf_budget_html(report: PerfBudgetReport) -> str:
    target_period_ms = float(report.summary.get("target_period_ms", 33.3333333333))
    rows: list[str] = []
    bar_rows: list[str] = []
    spark_values: list[float] = []
    for metric_key in sorted(report.checked.keys()):
        value, limit = report.checked[metric_key]
        utilization_pct = (value / limit * 100.0) if limit > 0.0 else 0.0
        spark_values.append(utilization_pct)
        impact_pct = (value / target_period_ms * 100.0) if target_period_ms > 0.0 else 0.0
        badge = _status_badge(value, limit)
        css_class = _status_css(value, limit)
        rows.append(
            "<tr>"
            f"<td class='{css_class}'>{badge}</td>"
            f"<td><code>{metric_key}</code></td>"
            f"<td>{value:.3f}</td>"
            f"<td>{limit:.3f}</td>"
            f"<td>{utilization_pct:.1f}%</td>"
            f"<td>{impact_pct:.1f}%</td>"
            "</tr>"
        )
        width_pct = min(max(utilization_pct, 0.0), 250.0)
        bar_rows.append(
            "<div class='bar-line'>"
            f"<span><code>{metric_key}</code></span>"
            f"<div class='bar-track'><div class='bar-fill {css_class}' style='width:{width_pct:.1f}%'></div></div>"
            f"<span>{utilization_pct:.1f}%</span>"
            "</div>"
        )

    spark_points: list[str] = []
    if spark_values:
        max_value = max(max(spark_values), 1.0)
        for idx, value in enumerate(spark_values):
            x = 10 + idx * (280 / max(1, len(spark_values) - 1))
            y = 90 - (min(value, 250.0) / max_value) * 70
            spark_points.append(f"{x:.1f},{y:.1f}")
    sparkline = " ".join(spark_points)
    reco_items = "".join(f"<li>{tip}</li>" for tip in _recommendations(report))

    return f"""<!doctype html>
<html lang="fr">
<head>
    <meta charset="utf-8" />
    <title>PeopleCounter - Performance Budget Report</title>
    <style>
        body {{ font-family: Inter, Segoe UI, Arial, sans-serif; margin: 24px; color: #1f2937; }}
        h1 {{ margin-bottom: 6px; }}
        .meta {{ color: #6b7280; margin-bottom: 20px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; }}
        th:nth-child(1), td:nth-child(1), th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
        th {{ background: #f9fafb; }}
        .ok {{ color: #047857; font-weight: 600; }}
        .warn {{ color: #b45309; font-weight: 600; }}
        .bad {{ color: #b91c1c; font-weight: 600; }}
        code {{ background: #f3f4f6; padding: 1px 5px; border-radius: 4px; }}
        .summary {{ margin-top: 16px; display: grid; grid-template-columns: repeat(3, minmax(160px, 1fr)); gap: 10px; }}
        .card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #ffffff; }}
        .label {{ color: #6b7280; font-size: 12px; }}
        .value {{ font-size: 18px; font-weight: 700; }}
        .bars {{ margin-top: 18px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
        .bar-line {{ display: grid; grid-template-columns: 280px 1fr 70px; gap: 10px; align-items: center; margin: 6px 0; }}
        .bar-track {{ background: #f3f4f6; height: 10px; border-radius: 999px; overflow: hidden; }}
        .bar-fill {{ height: 100%; border-radius: 999px; }}
        .bar-fill.ok {{ background: #10b981; }}
        .bar-fill.warn {{ background: #f59e0b; }}
        .bar-fill.bad {{ background: #ef4444; }}
        .spark {{ margin-top: 16px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; }}
        .reco {{ margin-top: 16px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #fff; }}
    </style>
</head>
<body>
    <h1>Performance Budget Report</h1>
    <div class="meta">Strategy: <b>{report.fusion_strategy}</b> | Mode: <b>{report.mode}</b></div>
    <table>
        <thead>
            <tr>
                <th>Statut</th>
                <th>Métrique</th>
                <th>Valeur (ms)</th>
                <th>Budget (ms)</th>
                <th>Utilisation</th>
                <th>Impact 4K@30</th>
            </tr>
        </thead>
        <tbody>
            {"\n      ".join(rows)}
        </tbody>
    </table>
    <div class="summary">
        <div class="card"><div class="label">Target FPS</div><div class="value">{float(report.summary.get("target_fps", 30.0)):.1f}</div></div>
        <div class="card"><div class="label">Critical path (ms)</div><div class="value">{float(report.summary.get("critical_path_ms", 0.0)):.3f}</div></div>
        <div class="card"><div class="label">FPS margin (ms)</div><div class="value">{float(report.summary.get("fps_margin_ms", 0.0)):.3f}</div></div>
    </div>
    <div class="spark">
        <div class="label">Sparkline utilisation budgets (%)</div>
        <svg viewBox="0 0 300 100" width="100%" height="120" role="img" aria-label="Sparkline utilisation">
            <polyline fill="none" stroke="#2563eb" stroke-width="2" points="{sparkline}" />
        </svg>
    </div>
    <div class="bars">
        <div class="label">Utilisation par métrique</div>
        {"".join(bar_rows)}
    </div>
    <div class="reco">
        <div class="label">Recommandations automatiques</div>
        <ul>{reco_items}</ul>
    </div>
</body>
</html>
"""


def write_perf_budget_html_report(
    report: PerfBudgetReport,
    output_dir: Path | None = None,
    report_name: str = "perf_budget_report",
) -> Path:
    target_dir = output_dir or (Path(__file__).resolve().parent / "artifacts")
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = report_name.strip().replace(" ", "_") or "perf_budget_report"
    output_file = target_dir / f"{safe_name}.html"
    output_file.write_text(render_perf_budget_html(report), encoding="utf-8")
    return output_file


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
    critical_path_ms = float(
        snapshot.get(
            "end_to_end_ms",
            snapshot.get("preprocess_critical_path_ms", snapshot.get("preprocess_ms", 0.0)),
        )
    )
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
