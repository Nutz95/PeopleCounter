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
        "preprocess_model_yolo_global_ms": 1.5,
        "preprocess_model_yolo_tiles_ms": 5.0,
        "preprocess_model_max_ms": 12.0,
        "preprocess_model_sum_ms": 20.0,
        "preprocess_critical_path_ms": 27.0,
        "preprocess_serial_overhead_ms": 2.0,
        "inference_model_yolo_global_ms": 14.0,
        "inference_model_yolo_tiles_ms": 18.0,
        # density runs on an independent async stream — does not block YOLO pipeline
        # measured: ~92ms for both configs (FP16) — proportional to total pixels processed
        # 2×2 @ 1920×1088 (default): 92ms, no rescaling, better accuracy
        # 6×3 @  640×720 (legacy) : 92ms, same latency
        # for < 30ms: FP8 quantization required (provide --calib-dir UCF-QNRF)
        "inference_model_density_ms": 100.0,
        "preprocess_model_density_ms": 8.0,
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
        "preprocess_model_yolo_global_ms": 2.0,
        "preprocess_model_yolo_tiles_ms": 5.0,
        "preprocess_model_max_ms": 12.0,
        "preprocess_model_sum_ms": 20.0,
        "preprocess_critical_path_ms": 27.0,
        # Serial overhead = preprocess_ms - (bridge + model_max).
        # With persistent executor the dispatch overhead is < 0.5 ms; the dominant
        # component is preprocess_sync_ms (CPU blocking on stream.synchronize()).
        # That sync time equals the true GPU execution time of the tiles preprocess.
        # Target 6 ms is aspirational (requires removing host-sync from hot path).
        "preprocess_serial_overhead_ms": 6.0,
        # dispatch = time to submit both worker threads + collect results (Python overhead only)
        "preprocess_dispatch_ms": 4.0,
        # sync = time CPU blocks on stream.synchronize() = actual GPU tiles preprocess time
        "preprocess_sync_ms": 13.0,
        "inference_model_yolo_global_ms": 15.0,
        "inference_model_yolo_tiles_ms": 20.0,
        # density runs on an independent async stream — does not block YOLO pipeline
        # measured: ~92ms for 18-tile FP16 batch on RTX 5060 Ti
        "inference_model_density_ms": 100.0,
        "preprocess_model_density_ms": 8.0,
        "inference_model_sum_ms": 30.0,
        "inference_model_max_ms": 20.0,
        # fusion_wait ≈ tiles_inference_ms − global_inference_ms in ASYNC_OVERLAY.
        # With yolo_tiles at ~20 ms and yolo_global at ~4 ms, the inherent gap
        # is ~16-18 ms; 21 ms gives a small margin.
        "fusion_wait_ms": 21.0,
        "overlay_lag_ms": 25.0,
        # 33 ms = 1 frame at 30 fps.  Achievable with optimised preprocess (~5 ms)
        # + sequential global (4 ms) + tiles (20 ms) + overhead (~4 ms) ≈ 33 ms.
        "end_to_end_ms": 33.0,
        "tensor_pool_wait_ms": 2.0,
    },
    FusionStrategyType.RAW_STREAM_WITH_METADATA.value: {
        "nvdec_ms": 35.0,
        "preprocess_nv12_bridge_ms": 15.0,
        "preprocess_ms": 28.0,
        "preprocess_model_yolo_global_ms": 1.5,
        "preprocess_model_yolo_tiles_ms": 5.0,
        "preprocess_model_max_ms": 11.0,
        "preprocess_model_sum_ms": 18.0,
        "preprocess_critical_path_ms": 25.0,
        "preprocess_serial_overhead_ms": 2.0,
        "inference_model_yolo_global_ms": 12.0,
        "inference_model_yolo_tiles_ms": 16.0,
        # density runs on an independent async stream — does not block YOLO pipeline
        "inference_model_density_ms": 100.0,
        "preprocess_model_density_ms": 8.0,
        "inference_model_sum_ms": 24.0,
        "inference_model_max_ms": 16.0,
        "fusion_wait_ms": 5.0,
        "overlay_lag_ms": 40.0,
        "end_to_end_ms": 33.0,
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


def _percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = int(round((len(ordered) - 1) * ratio))
    position = max(0, min(position, len(ordered) - 1))
    return float(ordered[position])


def _build_histogram_svg(values: list[float], title: str, width: int = 360, height: int = 120) -> str:
    if not values:
        return f"<div class='hist-card'><div class='label'>{title}</div><div class='label'>No data</div></div>"
    bins = 8
    min_v = min(values)
    max_v = max(values)
    span = max(max_v - min_v, 1e-6)
    counts = [0 for _ in range(bins)]
    for value in values:
        index = int((value - min_v) / span * (bins - 1))
        counts[max(0, min(index, bins - 1))] += 1
    max_count = max(counts) if counts else 1
    bar_w = (width - 20) / bins
    rects: list[str] = []
    for idx, count in enumerate(counts):
        h = ((height - 30) * (count / max_count)) if max_count > 0 else 0
        x = 10 + idx * bar_w
        y = height - 20 - h
        rects.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w - 3:.1f}' height='{h:.1f}' fill='#2563eb' opacity='0.85'/>")
    return (
        "<div class='hist-card'>"
        f"<div class='label'>{title}</div>"
        f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' role='img' aria-label='Histogram {title}'>"
        + "".join(rects)
        + "</svg></div>"
    )


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


def render_perf_budget_html(
    report: PerfBudgetReport,
    history: Mapping[str, list[float]] | None = None,
    warmup_frames: int = 0,
) -> str:
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

    preprocess_total = float(report.checked.get("preprocess_ms", (0.0, 0.0))[0])
    bridge = float(report.checked.get("preprocess_nv12_bridge_ms", (0.0, 0.0))[0])
    critical_model = float(report.checked.get("preprocess_model_max_ms", (0.0, 0.0))[0])
    serial_overhead = float(report.checked.get("preprocess_serial_overhead_ms", (0.0, 0.0))[0])
    decomposition_gap = preprocess_total - (bridge + critical_model + serial_overhead)

    # Sub-breakdown of serial_overhead (dispatch + sync) — from new telemetry
    dispatch_ms = float(report.summary.get("preprocess_dispatch_ms", 0.0))
    sync_ms = float(report.summary.get("preprocess_sync_ms", 0.0))

    inf_global = float(report.checked.get("inference_model_yolo_global_ms", (0.0, 0.0))[0])
    inf_tiles = float(report.checked.get("inference_model_yolo_tiles_ms", (0.0, 0.0))[0])
    inf_density = float(report.checked.get("inference_model_density_ms", (0.0, 0.0))[0])
    e2e_ms = float(report.checked.get("end_to_end_ms", (0.0, 0.0))[0])
    nvdec_ms = float(report.checked.get("nvdec_ms", (0.0, 0.0))[0])
    fusion_wait_ms = float(report.checked.get("fusion_wait_ms", (0.0, 0.0))[0])
    # Critical inference path = sequential global + tiles (current architecture)
    inf_sequential = inf_global + inf_tiles
    known_sub_total = nvdec_ms + preprocess_total + inf_sequential
    python_overhead_ms = max(0.0, e2e_ms - known_sub_total)

    hist = history or {}
    stat_keys = [
        "nvdec_ms",
        "preprocess_ms",
        "preprocess_nv12_bridge_ms",
        "preprocess_dispatch_ms",
        "preprocess_sync_ms",
        "preprocess_serial_overhead_ms",
        "inference_model_sum_ms",
        "end_to_end_ms",
    ]
    stat_rows: list[str] = []
    for key in stat_keys:
        values = [float(v) for v in hist.get(key, [])]
        if not values:
            continue
        stat_rows.append(
            "<tr>"
            f"<td><code>{key}</code></td>"
            f"<td>{sum(values) / len(values):.3f}</td>"
            f"<td>{_percentile(values, 0.50):.3f}</td>"
            f"<td>{_percentile(values, 0.90):.3f}</td>"
            f"<td>{max(values):.3f}</td>"
            f"<td>{len(values)}</td>"
            "</tr>"
        )

    histograms = "".join(
        [
            _build_histogram_svg([float(v) for v in hist.get("nvdec_ms", [])], "NVDEC"),
            _build_histogram_svg([float(v) for v in hist.get("preprocess_ms", [])], "Preprocess"),
            _build_histogram_svg([float(v) for v in hist.get("inference_model_sum_ms", [])], "Inference sum"),
            _build_histogram_svg([float(v) for v in hist.get("end_to_end_ms", [])], "End-to-end"),
        ]
    )

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
        .group {{ margin-top: 16px; border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px; background: #fff; }}
        .group ul {{ margin: 8px 0 0 18px; }}
        .hist-grid {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 12px; margin-top: 10px; }}
        .hist-card {{ border: 1px solid #e5e7eb; border-radius: 8px; padding: 10px; background: #fff; }}
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
    <div class="group">
        <div class="label">Décomposition preprocess</div>
        <ul>
            <li><b>preprocess_ms</b> = {preprocess_total:.3f} ms</li>
            <li>├─ <b>preprocess_nv12_bridge_ms</b> = {bridge:.3f} ms</li>
            <li>├─ <b>preprocess_model_max_ms</b> (branche critique parallèle) = {critical_model:.3f} ms</li>
            <li>└─ <b>preprocess_serial_overhead_ms</b> = {serial_overhead:.3f} ms</li>
            <li>&nbsp;&nbsp;&nbsp;├─ <b>preprocess_dispatch_ms</b> (soumission threads + collect résultats) = {dispatch_ms:.3f} ms</li>
            <li>&nbsp;&nbsp;&nbsp;└─ <b>preprocess_sync_ms</b> (stream.synchronize() = temps GPU réel tiles) = {sync_ms:.3f} ms</li>
        </ul>
        <div class="label">Écart de décomposition (doit rester proche de 0): {decomposition_gap:.3f} ms</div>
        <div class="label">Note: <b>preprocess_sync_ms</b> = temps réel d'exécution GPU du preprocess tiles (stream.synchronize() bloque le CPU). Pour le réduire: supprimer le host-sync et utiliser des CUDA events (inference_stream.wait_event) pour ordonnancer sans bloquer le CPU.</div>
    </div>
    <div class="group">
        <div class="label">Comparaison inférence YOLO + Density</div>
        <ul>
            <li><b>inference_model_yolo_global_ms</b> = {inf_global:.3f} ms</li>
            <li><b>inference_model_yolo_tiles_ms</b> = {inf_tiles:.3f} ms</li>
            <li><b>inference_model_density_ms</b> = {inf_density:.3f} ms (stream indépendant, ne bloque pas YOLO)</li>
        </ul>
    </div>
    <div class="group">
        <div class="label">Décomposition end-to-end (waterfall)</div>
        <ul>
            <li><b>end_to_end_ms</b> = {e2e_ms:.3f} ms</li>
            <li>├─ <b>nvdec_ms</b> (décodage NVDEC) = {nvdec_ms:.3f} ms</li>
            <li>├─ <b>preprocess_ms</b> (dispatch CPU, GPU async via CUDA events) = {preprocess_total:.3f} ms</li>
            <li>├─ <b>inference_yolo_global_ms</b> = {inf_global:.3f} ms</li>
            <li>├─ <b>inference_yolo_tiles_ms</b> (chemin critique) = {inf_tiles:.3f} ms</li>
            <li>├─ (sous-total connu = {known_sub_total:.3f} ms)</li>
            <li>└─ <b>python_overhead_ms</b> (GIL, scheduling, agrégation, tensor pool) = {python_overhead_ms:.3f} ms</li>
        </ul>
        <div class="label">Note: python_overhead_ms inclut la latence d'ordonnancement Python entre les étapes (GIL, ThreadPoolExecutor, agrégateur). Pour réduire davantage: paralléliser global+tiles, C++ wrapper, ou asyncio.</div>
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
        <div class="group">
                <div class="label">Statistiques multi-frames (post-warmup: {warmup_frames} frame(s))</div>
                <table>
                    <thead>
                        <tr>
                            <th>Métrique</th>
                            <th>Moyenne (ms)</th>
                            <th>P50 (ms)</th>
                            <th>P90 (ms)</th>
                            <th>Max (ms)</th>
                            <th>N</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(stat_rows) if stat_rows else "<tr><td colspan='6'>No history</td></tr>"}</tbody>
                </table>
                <div class="hist-grid">{histograms}</div>
        </div>
</body>
</html>
"""


def write_perf_budget_html_report(
    report: PerfBudgetReport,
    output_dir: Path | None = None,
    report_name: str = "perf_budget_report",
    history: Mapping[str, list[float]] | None = None,
    warmup_frames: int = 0,
) -> Path:
    target_dir = output_dir or (Path(__file__).resolve().parent / "artifacts")
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = report_name.strip().replace(" ", "_") or "perf_budget_report"
    output_file = target_dir / f"{safe_name}.html"
    output_file.write_text(render_perf_budget_html(report, history=history, warmup_frames=warmup_frames), encoding="utf-8")
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
        # Sub-breakdown of preprocess serial overhead
        "preprocess_dispatch_ms": float(snapshot.get("preprocess_dispatch_ms", 0.0)),
        "preprocess_sync_ms": float(snapshot.get("preprocess_sync_ms", 0.0)),
    }

    return PerfBudgetReport(
        mode=mode,
        fusion_strategy=normalized_strategy,
        checked=checked,
        violations=violations,
        summary=summary,
    )
