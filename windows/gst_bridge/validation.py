from __future__ import annotations

import os
import subprocess
from pathlib import Path

from .config import REQUIRED_GST_PLUGINS
from .models import ValidationResult


PLUGIN_FALLBACKS: dict[str, tuple[str, str]] = {
    "qsvh264enc": ("qsv", "qsvh264enc"),
    "qsvh265enc": ("qsv", "qsvh265enc"),
    "qsvh264dec": ("qsv", "qsvh264dec"),
    "qsvh265dec": ("qsv", "qsvh265dec"),
}


def build_runtime_env(gstreamer_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "GST_PLUGIN_PATH",
        "GST_PLUGIN_PATH_1_0",
        "GST_PLUGIN_SYSTEM_PATH",
        "GST_PLUGIN_SYSTEM_PATH_1_0",
        "GST_REGISTRY",
        "GST_REGISTRY_FORK",
        "GST_REGISTRY_REUSE_PLUGIN_SCANNER",
        "GST_PLUGIN_SCANNER",
        "GST_PLUGIN_SCANNER_1_0",
    ):
        env.pop(key, None)
    bin_dir = gstreamer_root / "bin"
    libexec_dir = gstreamer_root / "libexec" / "gstreamer-1.0"
    plugin_dir = gstreamer_root / "lib" / "gstreamer-1.0"
    env["PATH"] = os.pathsep.join([str(bin_dir), str(libexec_dir), env.get("PATH", "")])
    env["GST_PLUGIN_SYSTEM_PATH_1_0"] = str(plugin_dir)
    env["GST_PLUGIN_PATH_1_0"] = str(plugin_dir)
    env["GSTREAMER_1_0_ROOT_X86_64"] = str(gstreamer_root)
    env["GSTREAMER_1_0_ROOT_MSVC_X86_64"] = str(gstreamer_root)
    return env


def validate_runtime(gstreamer_root: Path, mediamtx_path: Path) -> ValidationResult:
    gst_launch = gstreamer_root / "bin" / "gst-launch-1.0.exe"
    gst_inspect = gstreamer_root / "bin" / "gst-inspect-1.0.exe"
    if not gst_launch.exists() or not gst_inspect.exists():
        raise RuntimeError(
            f"GStreamer local install is incomplete under {gstreamer_root}. "
            "Expected gst-launch-1.0.exe and gst-inspect-1.0.exe."
        )
    if not mediamtx_path.exists():
        raise RuntimeError(f"MediaMTX binary missing: {mediamtx_path}")

    env = build_runtime_env(gstreamer_root)
    gst_version = _read_gst_version(gst_launch, env)
    missing_plugins = [name for name in REQUIRED_GST_PLUGINS if not _plugin_exists(gst_inspect, name, env)]
    if missing_plugins:
        diagnostics = _build_missing_plugin_diagnostics(gst_inspect, env, missing_plugins)
        raise RuntimeError(
            "Missing required GStreamer plugins for the local bridge: "
            + ", ".join(missing_plugins)
            + ".\n"
            + diagnostics
        )
    return ValidationResult(
        gstreamer_root=gstreamer_root,
        gst_launch_path=gst_launch,
        gst_inspect_path=gst_inspect,
        mediamtx_path=mediamtx_path,
        gst_version=gst_version,
        plugins=list(REQUIRED_GST_PLUGINS),
    )


def _read_gst_version(gst_launch: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        [str(gst_launch), "--version"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    text = (result.stdout or result.stderr).strip().splitlines()
    return text[0] if text else "unknown"


def _plugin_exists(gst_inspect: Path, plugin_name: str, env: dict[str, str]) -> bool:
    result = subprocess.run(
        [str(gst_inspect), plugin_name],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode == 0:
        return True
    if "Factory Details:" in output or "Plugin Details:" in output:
        return True

    fallback = PLUGIN_FALLBACKS.get(plugin_name)
    if fallback is None:
        return False

    fallback_plugin, fallback_feature = fallback
    fallback_result = subprocess.run(
        [str(gst_inspect), fallback_plugin],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    fallback_output = "\n".join(part for part in (fallback_result.stdout, fallback_result.stderr) if part)
    return fallback_result.returncode == 0 and f"{fallback_feature}:" in fallback_output


def _build_missing_plugin_diagnostics(
    gst_inspect: Path,
    env: dict[str, str],
    missing_plugins: list[str],
) -> str:
    plugin_dir = env.get("GST_PLUGIN_SYSTEM_PATH_1_0", "unknown")
    lines = [
        f"GStreamer root: {env.get('GSTREAMER_1_0_ROOT_MSVC_X86_64') or env.get('GSTREAMER_1_0_ROOT_X86_64', 'unknown')}",
        f"Plugin directory: {plugin_dir}",
        f"Verify with: {gst_inspect} <plugin-name>",
    ]
    for plugin_name in missing_plugins:
        result = subprocess.run(
            [str(gst_inspect), plugin_name],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        detail = (result.stderr or result.stdout).strip().splitlines()
        first_line = detail[0] if detail else f"exit={result.returncode}"
        lines.append(f"- {plugin_name}: {first_line}")
    return "\n".join(lines)
