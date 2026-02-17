from __future__ import annotations

from app_v2.enums import FusionStrategyType
from app_v2.tests.integration.pipeline.perf_budget import evaluate_perf_budget, perf_budget_mode


def test_perf_budget_mode_falls_back_to_warn_for_invalid_value(monkeypatch) -> None:
    monkeypatch.setenv("PEOPLE_COUNTER_PERF_BUDGET_MODE", "not-a-mode")
    assert perf_budget_mode() == "warn"


def test_evaluate_perf_budget_uses_strategy_specific_budget() -> None:
    snapshot = {
        "preprocess_critical_path_ms": 26.0,
        "preprocess_model_sum_ms": 14.0,
        "preprocess_model_max_ms": 8.0,
        "preprocess_ms": 24.0,
        "nvdec_ms": 33.0,
        "preprocess_nv12_bridge_ms": 12.0,
        "tensor_pool_wait_ms": 0.3,
    }

    strict_report = evaluate_perf_budget(snapshot, fusion_strategy=FusionStrategyType.STRICT_SYNC.value)
    async_report = evaluate_perf_budget(snapshot, fusion_strategy=FusionStrategyType.ASYNC_OVERLAY.value)

    assert strict_report.fusion_strategy == FusionStrategyType.STRICT_SYNC.value
    assert async_report.fusion_strategy == FusionStrategyType.ASYNC_OVERLAY.value
    assert "preprocess_critical_path_ms" in strict_report.checked
    assert "parallel_efficiency_ratio" in strict_report.summary


def test_evaluate_perf_budget_unknown_strategy_defaults_to_async_overlay() -> None:
    snapshot = {"preprocess_ms": 10.0, "preprocess_model_sum_ms": 5.0, "preprocess_model_max_ms": 5.0}

    report = evaluate_perf_budget(snapshot, fusion_strategy="UNKNOWN")

    assert report.fusion_strategy == FusionStrategyType.ASYNC_OVERLAY.value
