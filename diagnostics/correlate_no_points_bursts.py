#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


@dataclass
class CorrelationSummary:
    frames: int
    burst_frames: int
    no_points_frames: int
    both_frames: int
    burst_rate: float
    no_points_rate: float
    both_rate: float
    p_no_points_given_burst: float
    p_no_points_given_no_burst: float
    relative_risk: float
    phi: float
    burst_threshold_ms: float


def _read_sse_json_messages(url: str, duration_s: int):
    req = Request(url, headers={"Accept": "text/event-stream"})
    start = time.perf_counter()
    with urlopen(req, timeout=duration_s + 15) as resp:
        event_data_lines: list[str] = []
        while time.perf_counter() - start < duration_s:
            line = resp.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if text.startswith(":"):
                continue
            if text.startswith("data:"):
                event_data_lines.append(text[5:].lstrip())
                continue
            if text == "":
                if event_data_lines:
                    joined = "\n".join(event_data_lines)
                    event_data_lines.clear()
                    try:
                        yield json.loads(joined)
                    except Exception:
                        continue
                continue


def _safe_float(d: dict[str, Any], key: str) -> float:
    try:
        return float(d.get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _extract_count(payload: list[dict[str, Any]]) -> int:
    count = 0
    for p in payload:
        if not isinstance(p, dict):
            continue
        for key in ("detection_count", "hotspot_count", "count"):
            v = p.get(key)
            if isinstance(v, (int, float)):
                count = max(count, int(v))
        dens = p.get("density_count")
        if isinstance(dens, (int, float)):
            count = max(count, int(round(float(dens))))
        dets = p.get("detections")
        if isinstance(dets, list):
            count = max(count, len(dets))
        hots = p.get("hotspots")
        if isinstance(hots, list):
            count = max(count, len(hots))
    return count


def _phi(a: int, b: int, c: int, d: int) -> float:
    # contingency:
    # a both, b burst only, c no_points only, d neither
    num = (a * d) - (b * c)
    den = math.sqrt(max(1e-12, (a + b) * (c + d) * (a + c) * (b + d)))
    return float(num / den)


def main() -> int:
    parser = argparse.ArgumentParser(description="Correlate no-points events with latency bursts from SSE telemetry.")
    parser.add_argument("--sse-url", type=str, default="http://127.0.0.1:5000/api/stream")
    parser.add_argument("--duration-s", type=int, default=60)
    parser.add_argument("--burst-ms", type=float, default=40.0, help="Primary burst threshold on src_wait_ms")
    parser.add_argument("--output-dir", type=str, default="diagnostics/artifacts")
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    tag = args.tag.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"no_points_corr_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    last_miss_counter = 0.0

    for msg in _read_sse_json_messages(args.sse_url, args.duration_s):
        if not isinstance(msg, dict):
            continue
        if msg.get("passthrough"):
            continue
        payload = msg.get("payload")
        if not isinstance(payload, list):
            payload = []

        telemetry: dict[str, Any] = {}
        for p in payload:
            if isinstance(p, dict) and isinstance(p.get("telemetry"), dict):
                telemetry = p.get("telemetry")
                break

        frame_id = int(msg.get("frame_id", 0) or 0)
        src_wait_ms = _safe_float(telemetry, "frame_source_wait_latest_ms")
        nvdec_ms = _safe_float(telemetry, "nvdec_ms")
        e2e_ms = _safe_float(telemetry, "end_to_end_ms")

        hs_lookup_mode = _safe_float(telemetry, "video_hotspot_lookup_mode_code")
        hs_miss_counter = _safe_float(telemetry, "video_hotspot_lookup_miss")
        hs_fallback_counter = _safe_float(telemetry, "video_hotspot_lookup_fallback")
        hs_lookup_ms = _safe_float(telemetry, "video_hotspot_lookup_ms")

        miss_delta = hs_miss_counter - last_miss_counter
        if miss_delta < 0:
            miss_delta = 0.0
        last_miss_counter = hs_miss_counter

        count = _extract_count(payload)

        no_points_event = (hs_lookup_mode == 3.0) or (miss_delta > 0.0)
        burst = (src_wait_ms >= args.burst_ms) or (nvdec_ms >= args.burst_ms) or (e2e_ms >= args.burst_ms)

        rows.append(
            {
                "frame_id": frame_id,
                "src_wait_ms": src_wait_ms,
                "nvdec_ms": nvdec_ms,
                "e2e_ms": e2e_ms,
                "hotspot_lookup_mode_code": hs_lookup_mode,
                "hotspot_lookup_ms": hs_lookup_ms,
                "hotspot_lookup_miss_counter": hs_miss_counter,
                "hotspot_lookup_fallback_counter": hs_fallback_counter,
                "count": count,
                "no_points_event": int(no_points_event),
                "burst_event": int(burst),
            }
        )

    csv_path = out_dir / "correlation_rows.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    frames = len(rows)
    burst_frames = sum(int(r["burst_event"]) for r in rows)
    no_points_frames = sum(int(r["no_points_event"]) for r in rows)
    both_frames = sum(int(r["burst_event"]) and int(r["no_points_event"]) for r in rows)

    burst_only = burst_frames - both_frames
    no_points_only = no_points_frames - both_frames
    neither = frames - both_frames - burst_only - no_points_only

    p_no_points_given_burst = (both_frames / burst_frames) if burst_frames else 0.0
    p_no_points_given_no_burst = (no_points_only / (frames - burst_frames)) if (frames - burst_frames) > 0 else 0.0
    rr = (p_no_points_given_burst / p_no_points_given_no_burst) if p_no_points_given_no_burst > 0 else 0.0

    summary = CorrelationSummary(
        frames=frames,
        burst_frames=burst_frames,
        no_points_frames=no_points_frames,
        both_frames=both_frames,
        burst_rate=(burst_frames / frames) if frames else 0.0,
        no_points_rate=(no_points_frames / frames) if frames else 0.0,
        both_rate=(both_frames / frames) if frames else 0.0,
        p_no_points_given_burst=p_no_points_given_burst,
        p_no_points_given_no_burst=p_no_points_given_no_burst,
        relative_risk=rr,
        phi=_phi(both_frames, burst_only, no_points_only, neither) if frames else 0.0,
        burst_threshold_ms=float(args.burst_ms),
    )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")

    # Optional plots
    plot_note = None
    try:
        import matplotlib.pyplot as plt  # type: ignore

        x = [int(r["frame_id"]) for r in rows]
        src_wait = [float(r["src_wait_ms"]) for r in rows]
        nvdec = [float(r["nvdec_ms"]) for r in rows]
        no_points = [int(r["no_points_event"]) for r in rows]

        plt.figure(figsize=(13, 4))
        plt.plot(x, src_wait, label="src_wait_ms", linewidth=1.0)
        plt.plot(x, nvdec, label="nvdec_ms", linewidth=1.0, alpha=0.8)
        for i, v in enumerate(no_points):
            if v:
                plt.axvline(x=x[i], color="#ef4444", alpha=0.12, linewidth=1)
        plt.axhline(args.burst_ms, color="#f59e0b", linestyle="--", label=f"burst={args.burst_ms}ms")
        plt.title("Bursts vs no-points events (red vertical markers)")
        plt.xlabel("frame_id")
        plt.ylabel("ms")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / "burst_vs_no_points.png", dpi=140)
        plt.close()

        # Contingency bar
        labels = ["burst", "no_points", "both"]
        vals = [summary.burst_frames, summary.no_points_frames, summary.both_frames]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, vals)
        plt.title("Event counts over capture window")
        plt.tight_layout()
        plt.savefig(out_dir / "event_counts.png", dpi=140)
        plt.close()
    except Exception as exc:
        plot_note = str(exc)

    print("frames,burst_frames,no_points_frames,both_frames,p(no_points|burst),p(no_points|no_burst),rr,phi")
    print(
        f"{summary.frames},{summary.burst_frames},{summary.no_points_frames},{summary.both_frames},"
        f"{summary.p_no_points_given_burst:.4f},{summary.p_no_points_given_no_burst:.4f},"
        f"{summary.relative_risk:.4f},{summary.phi:.4f}"
    )
    print(f"[corr] rows={csv_path}")
    print(f"[corr] summary={summary_path}")
    if plot_note:
        print(f"[corr] plots skipped: {plot_note}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
