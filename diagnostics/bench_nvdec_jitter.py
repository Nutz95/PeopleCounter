#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from app_v2.infrastructure.rtsp_frame_source import RTSPFrameSource


@dataclass
class ScenarioSummary:
    scenario: str
    samples: int
    elapsed_s: float
    fps: float
    inter_frame_mean_ms: float
    inter_frame_median_ms: float
    inter_frame_p95_ms: float
    inter_frame_std_ms: float
    jitter_std_ms: float
    jitter_abs_mean_ms: float
    wait_mean_ms: float
    wait_p95_ms: float
    nvdec_mean_ms: float
    nvdec_p95_ms: float
    copy_sync_mean_ms: float
    copy_sync_p95_ms: float
    infer_mean_ms: float
    infer_p95_ms: float


class _P2PLikeInference:
    """Synthetic GPU workload to simulate inference contention with NVDEC."""

    def __init__(self, *, width: int, height: int, device: torch.device) -> None:
        self._conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        self._conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False).to(device)
        self._head_logits = torch.nn.Conv2d(64, 2, kernel_size=1, bias=True).to(device)
        self._head_pts = torch.nn.Conv2d(64, 2, kernel_size=1, bias=True).to(device)
        self._input = torch.rand((1, 3, height, width), dtype=torch.float32, device=device)

    def run(self) -> None:
        x = torch.relu(self._conv1(self._input))
        x = torch.relu(self._conv2(x))
        logits = self._head_logits(x)
        pts = self._head_pts(x)
        logits_flat = logits.flatten(2).transpose(1, 2).contiguous()
        pts_flat = pts.flatten(2).transpose(1, 2).contiguous()
        scores = torch.softmax(logits_flat[0], dim=-1)[:, 1]
        _ = pts_flat[0][scores > 0.5]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    rank = (len(vals) - 1) * (p / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(vals) - 1)
    frac = rank - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


def _safe_mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _safe_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _safe_p95(values: list[float]) -> float:
    return float(_percentile(values, 95.0)) if values else 0.0


def _safe_std(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _summarize(
    *,
    scenario: str,
    elapsed_s: float,
    inter_frame_ms: list[float],
    wait_ms: list[float],
    nvdec_ms: list[float],
    copy_sync_ms: list[float],
    infer_ms: list[float],
    fps_target: float,
) -> ScenarioSummary:
    samples = len(wait_ms)
    fps = (samples / elapsed_s) if elapsed_s > 0 else 0.0
    target_period = 1000.0 / max(1e-6, fps_target)
    jitters = [v - target_period for v in inter_frame_ms] if inter_frame_ms else []

    return ScenarioSummary(
        scenario=scenario,
        samples=samples,
        elapsed_s=elapsed_s,
        fps=fps,
        inter_frame_mean_ms=_safe_mean(inter_frame_ms),
        inter_frame_median_ms=_safe_median(inter_frame_ms),
        inter_frame_p95_ms=_safe_p95(inter_frame_ms),
        inter_frame_std_ms=_safe_std(inter_frame_ms),
        jitter_std_ms=_safe_std(jitters),
        jitter_abs_mean_ms=_safe_mean([abs(x) for x in jitters]) if jitters else 0.0,
        wait_mean_ms=_safe_mean(wait_ms),
        wait_p95_ms=_safe_p95(wait_ms),
        nvdec_mean_ms=_safe_mean(nvdec_ms),
        nvdec_p95_ms=_safe_p95(nvdec_ms),
        copy_sync_mean_ms=_safe_mean(copy_sync_ms),
        copy_sync_p95_ms=_safe_p95(copy_sync_ms),
        infer_mean_ms=_safe_mean(infer_ms),
        infer_p95_ms=_safe_p95(infer_ms),
    )


def _run_scenario(
    *,
    stream_url: str,
    duration_s: int,
    fps_target: float,
    with_inference: bool,
    infer_width: int,
    infer_height: int,
) -> tuple[list[dict[str, float]], ScenarioSummary]:
    scenario = "decode_plus_inference" if with_inference else "decode_only"

    source = RTSPFrameSource(stream_url)
    source.connect()

    infer_stream = torch.cuda.Stream() if with_inference else None
    infer_model = (
        _P2PLikeInference(width=infer_width, height=infer_height, device=torch.device("cuda"))
        if with_inference
        else None
    )

    rows: list[dict[str, float]] = []
    inter_frame_ms: list[float] = []
    wait_ms: list[float] = []
    nvdec_ms: list[float] = []
    copy_sync_ms: list[float] = []
    infer_ms: list[float] = []

    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns + int(duration_s * 1e9)
    prev_recv_ns: int | None = None
    frame_id = 0

    try:
        while time.perf_counter_ns() < deadline_ns:
            frame_id += 1
            recv_start_ns = time.perf_counter_ns()
            frame = source.next_frame(frame_id)
            recv_end_ns = time.perf_counter_ns()

            tele = getattr(frame, "telemetry", None)
            snap = tele.snapshot() if tele is not None else {}

            wait = float(snap.get("frame_source_wait_latest_ms", 0.0) or 0.0)
            nvdec = float(snap.get("nvdec_ms", 0.0) or 0.0)
            copy_sync = float(snap.get("frame_source_copy_sync_ms", 0.0) or 0.0)

            infer_t = 0.0
            if infer_model is not None and infer_stream is not None:
                infer_start = torch.cuda.Event(enable_timing=True)
                infer_end = torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(infer_stream):
                    infer_start.record(infer_stream)
                    infer_model.run()
                    infer_end.record(infer_stream)
                infer_end.synchronize()
                infer_t = float(infer_start.elapsed_time(infer_end))

            now_ns = recv_end_ns
            inter = 0.0
            if prev_recv_ns is not None:
                inter = (now_ns - prev_recv_ns) / 1_000_000.0
                inter_frame_ms.append(inter)
            prev_recv_ns = now_ns

            elapsed_ms = (recv_end_ns - start_ns) / 1_000_000.0
            loop_ms = (recv_end_ns - recv_start_ns) / 1_000_000.0

            rows.append(
                {
                    "frame_idx": float(frame_id),
                    "elapsed_ms": float(elapsed_ms),
                    "inter_frame_ms": float(inter),
                    "next_frame_loop_ms": float(loop_ms),
                    "frame_source_wait_latest_ms": float(wait),
                    "nvdec_ms": float(nvdec),
                    "frame_source_copy_sync_ms": float(copy_sync),
                    "infer_ms": float(infer_t),
                }
            )

            wait_ms.append(wait)
            nvdec_ms.append(nvdec)
            copy_sync_ms.append(copy_sync)
            if with_inference:
                infer_ms.append(infer_t)
    finally:
        try:
            source.disconnect()
        except Exception:
            pass

    elapsed_s = (time.perf_counter_ns() - start_ns) / 1e9
    summary = _summarize(
        scenario=scenario,
        elapsed_s=elapsed_s,
        inter_frame_ms=inter_frame_ms,
        wait_ms=wait_ms,
        nvdec_ms=nvdec_ms,
        copy_sync_ms=copy_sync_ms,
        infer_ms=infer_ms,
        fps_target=fps_target,
    )
    return rows, summary


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _try_plot(output_dir: Path, decode_rows: list[dict[str, float]], infer_rows: list[dict[str, float]]) -> tuple[bool, str | None]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    def arr(rows: list[dict[str, float]], key: str) -> list[float]:
        return [float(r.get(key, 0.0)) for r in rows]

    x0 = arr(decode_rows, "elapsed_ms")
    x1 = arr(infer_rows, "elapsed_ms")

    # 1) inter-frame
    plt.figure(figsize=(12, 4))
    plt.plot(x0, arr(decode_rows, "inter_frame_ms"), label="decode-only", linewidth=1.0)
    plt.plot(x1, arr(infer_rows, "inter_frame_ms"), label="decode+infer", linewidth=1.0, alpha=0.8)
    plt.title("Inter-frame delta over time")
    plt.xlabel("elapsed ms")
    plt.ylabel("ms")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "inter_frame_ms.png", dpi=140)
    plt.close()

    # 2) wait latest
    plt.figure(figsize=(12, 4))
    plt.plot(x0, arr(decode_rows, "frame_source_wait_latest_ms"), label="decode-only", linewidth=1.0)
    plt.plot(x1, arr(infer_rows, "frame_source_wait_latest_ms"), label="decode+infer", linewidth=1.0, alpha=0.8)
    plt.title("frame_source_wait_latest_ms")
    plt.xlabel("elapsed ms")
    plt.ylabel("ms")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "frame_wait_ms.png", dpi=140)
    plt.close()

    # 3) nvdec + copy sync
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(x0, arr(decode_rows, "nvdec_ms"), label="decode-only", linewidth=1.0)
    ax[0].plot(x1, arr(infer_rows, "nvdec_ms"), label="decode+infer", linewidth=1.0, alpha=0.8)
    ax[0].set_ylabel("nvdec_ms")
    ax[0].grid(alpha=0.25)
    ax[0].legend()
    ax[1].plot(x0, arr(decode_rows, "frame_source_copy_sync_ms"), label="decode-only", linewidth=1.0)
    ax[1].plot(x1, arr(infer_rows, "frame_source_copy_sync_ms"), label="decode+infer", linewidth=1.0, alpha=0.8)
    ax[1].set_ylabel("copy_sync_ms")
    ax[1].set_xlabel("elapsed ms")
    ax[1].grid(alpha=0.25)
    fig.suptitle("NVDEC and copy sync timing")
    fig.tight_layout()
    fig.savefig(output_dir / "nvdec_copy_sync_ms.png", dpi=140)
    plt.close(fig)

    # 4) jitter histogram
    target = 1000.0 / 30.0
    d_jitter = [float(v) - target for v in arr(decode_rows, "inter_frame_ms") if v > 0.0]
    i_jitter = [float(v) - target for v in arr(infer_rows, "inter_frame_ms") if v > 0.0]

    plt.figure(figsize=(10, 5))
    plt.hist(d_jitter, bins=80, alpha=0.6, label="decode-only")
    plt.hist(i_jitter, bins=80, alpha=0.6, label="decode+infer")
    plt.title("Inter-frame jitter distribution (vs 33.33 ms)")
    plt.xlabel("jitter ms")
    plt.ylabel("count")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "jitter_hist_ms.png", dpi=140)
    plt.close()

    return True, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark NVDEC decode jitter/inter-frame/wait on RTSP stream (decode-only vs decode+inference)."
    )
    parser.add_argument("--stream-url", type=str, required=True)
    parser.add_argument("--duration-s", type=int, default=60)
    parser.add_argument("--fps-target", type=float, default=30.0)
    parser.add_argument("--infer-width", type=int, default=1920)
    parser.add_argument("--infer-height", type=int, default=1080)
    parser.add_argument("--output-dir", type=str, default="diagnostics/artifacts")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for NVDEC benchmark.")

    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"nvdec_bench_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[nvdec-bench] device={torch.cuda.get_device_name(0)} url={args.stream_url} "
        f"duration={args.duration_s}s fps_target={args.fps_target} out={out_dir}"
    )

    decode_rows, decode_summary = _run_scenario(
        stream_url=args.stream_url,
        duration_s=args.duration_s,
        fps_target=args.fps_target,
        with_inference=False,
        infer_width=args.infer_width,
        infer_height=args.infer_height,
    )
    infer_rows, infer_summary = _run_scenario(
        stream_url=args.stream_url,
        duration_s=args.duration_s,
        fps_target=args.fps_target,
        with_inference=True,
        infer_width=args.infer_width,
        infer_height=args.infer_height,
    )

    decode_csv = out_dir / "decode_only.csv"
    infer_csv = out_dir / "decode_plus_inference.csv"
    _write_csv(decode_csv, decode_rows)
    _write_csv(infer_csv, infer_rows)

    plots_ok, plot_msg = _try_plot(out_dir, decode_rows, infer_rows)

    summary = {
        "device": torch.cuda.get_device_name(0),
        "stream_url": args.stream_url,
        "duration_s": args.duration_s,
        "fps_target": args.fps_target,
        "decode_only": asdict(decode_summary),
        "decode_plus_inference": asdict(infer_summary),
        "plots_generated": plots_ok,
        "plot_note": plot_msg,
        "artifacts": {
            "decode_csv": str(decode_csv),
            "decode_plus_inference_csv": str(infer_csv),
            "inter_frame_plot": str(out_dir / "inter_frame_ms.png"),
            "frame_wait_plot": str(out_dir / "frame_wait_ms.png"),
            "nvdec_copy_sync_plot": str(out_dir / "nvdec_copy_sync_ms.png"),
            "jitter_hist_plot": str(out_dir / "jitter_hist_ms.png"),
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("scenario,samples,elapsed_s,fps,inter_mean_ms,inter_p95_ms,jitter_std_ms,wait_mean_ms,wait_p95_ms,nvdec_mean_ms,nvdec_p95_ms,copy_sync_mean_ms,copy_sync_p95_ms,infer_mean_ms,infer_p95_ms")
    for s in (decode_summary, infer_summary):
        print(
            f"{s.scenario},{s.samples},{s.elapsed_s:.3f},{s.fps:.3f},"
            f"{s.inter_frame_mean_ms:.3f},{s.inter_frame_p95_ms:.3f},{s.jitter_std_ms:.3f},"
            f"{s.wait_mean_ms:.3f},{s.wait_p95_ms:.3f},{s.nvdec_mean_ms:.3f},{s.nvdec_p95_ms:.3f},"
            f"{s.copy_sync_mean_ms:.3f},{s.copy_sync_p95_ms:.3f},{s.infer_mean_ms:.3f},{s.infer_p95_ms:.3f}"
        )

    print(f"[nvdec-bench] summary: {summary_path}")
    if not plots_ok and plot_msg:
        print(f"[nvdec-bench] plots skipped: {plot_msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
