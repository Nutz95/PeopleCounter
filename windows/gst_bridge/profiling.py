from __future__ import annotations

import csv
import time
from pathlib import Path

from .models import ProfileSample, RuntimeMetrics


class ProfileRecorder:
    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._start = time.perf_counter()
        self._samples: list[ProfileSample] = []

    def add_sample(self, metrics: RuntimeMetrics) -> None:
        self._samples.append(
            ProfileSample(
                t_s=time.perf_counter() - self._start,
                decoder_fps=metrics.decoder_fps,
                output_fps=metrics.output_fps,
                dropped_fps=metrics.dropped_fps,
                warnings=metrics.warnings,
                errors=metrics.errors,
            )
        )

    def finalize(self) -> tuple[Path, Path | None]:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self._output_dir / "metrics.csv"
        png_path = self._output_dir / "metrics.png"
        self._write_csv(csv_path)
        rendered = self._render_plot(png_path)
        return csv_path, rendered

    def _write_csv(self, csv_path: Path) -> None:
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t_s", "decoder_fps", "output_fps", "dropped_fps", "warnings", "errors"])
            for sample in self._samples:
                writer.writerow([
                    sample.t_s,
                    sample.decoder_fps,
                    sample.output_fps,
                    sample.dropped_fps,
                    sample.warnings,
                    sample.errors,
                ])

    def _render_plot(self, png_path: Path) -> Path | None:
        if not self._samples:
            return None
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return None

        t = [sample.t_s for sample in self._samples]
        decode = [sample.decoder_fps for sample in self._samples]
        output = [sample.output_fps for sample in self._samples]
        dropped = [sample.dropped_fps for sample in self._samples]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle("GStreamer Bridge Metrics")
        axes[0].plot(t, decode, label="decode fps", color="#2ca02c")
        axes[0].plot(t, output, label="output fps", color="#1f77b4")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.25)
        axes[0].set_ylabel("fps")

        axes[1].plot(t, dropped, label="dropped fps", color="#d62728")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.25)
        axes[1].set_ylabel("fps")
        axes[1].set_xlabel("time (s)")
        fig.tight_layout()
        fig.savefig(png_path, dpi=140)
        plt.close(fig)
        return png_path
