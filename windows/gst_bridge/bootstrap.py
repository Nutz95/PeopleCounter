from __future__ import annotations

import json
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

from .config import (
    DEFAULT_GSTREAMER_INSTALLER_URL,
    DEFAULT_GSTREAMER_VERSION,
    DEFAULT_MEDIAMTX_DOWNLOAD_URL,
    DEFAULT_MEDIAMTX_VERSION,
)


def ensure_third_party_layout(third_party_dir: Path) -> None:
    (third_party_dir / "downloads").mkdir(parents=True, exist_ok=True)
    (third_party_dir / "gstreamer").mkdir(parents=True, exist_ok=True)
    (third_party_dir / "mediamtx").mkdir(parents=True, exist_ok=True)


def ensure_gstreamer(third_party_dir: Path) -> Path:
    ensure_third_party_layout(third_party_dir)
    install_root = third_party_dir / "gstreamer" / DEFAULT_GSTREAMER_VERSION
    marker = install_root / "bin" / "gst-launch-1.0.exe"
    if marker.exists():
        _write_manifest(third_party_dir, {"gstreamer": DEFAULT_GSTREAMER_VERSION})
        return install_root

    downloads_dir = third_party_dir / "downloads"
    installer_path = downloads_dir / f"gstreamer-{DEFAULT_GSTREAMER_VERSION}.exe"
    if not installer_path.exists():
        urllib.request.urlretrieve(DEFAULT_GSTREAMER_INSTALLER_URL, installer_path)

    install_root.mkdir(parents=True, exist_ok=True)
    _run_installer_with_known_silent_modes(installer_path, install_root, marker)
    _write_manifest(third_party_dir, {"gstreamer": DEFAULT_GSTREAMER_VERSION})
    return install_root


def ensure_mediamtx(third_party_dir: Path) -> Path:
    ensure_third_party_layout(third_party_dir)
    mediamtx_dir = third_party_dir / "mediamtx" / DEFAULT_MEDIAMTX_VERSION
    executable = mediamtx_dir / "mediamtx.exe"
    if executable.exists():
        _write_manifest(third_party_dir, {"mediamtx": DEFAULT_MEDIAMTX_VERSION})
        return executable

    downloads_dir = third_party_dir / "downloads"
    archive_path = downloads_dir / f"mediamtx-{DEFAULT_MEDIAMTX_VERSION}.zip"
    if not archive_path.exists():
        urllib.request.urlretrieve(DEFAULT_MEDIAMTX_DOWNLOAD_URL, archive_path)

    mediamtx_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(mediamtx_dir)

    discovered = next(mediamtx_dir.rglob("mediamtx.exe"), None)
    if discovered is None:
        raise RuntimeError("mediamtx.exe not found after extraction")
    if discovered != executable:
        shutil.copy2(discovered, executable)
    _write_manifest(third_party_dir, {"mediamtx": DEFAULT_MEDIAMTX_VERSION})
    return executable


def _run_installer_with_known_silent_modes(installer_path: Path, install_root: Path, marker: Path) -> None:
    attempts = [
        ["/S", f"/D={install_root}"],
        ["/SILENT", f"/DIR={install_root}"],
        ["/VERYSILENT", f"/DIR={install_root}", "/SUPPRESSMSGBOXES", "/NORESTART"],
    ]
    failures: list[str] = []
    for args in attempts:
        process = subprocess.run([str(installer_path), *args], check=False)
        if process.returncode == 0 and marker.exists():
            return
        failures.append(f"{' '.join(args)} -> exit={process.returncode}")
    raise RuntimeError(
        "Unable to install GStreamer silently into the project-local directory. "
        f"Tried installer strategies against {installer_path}. "
        f"Attempts: {'; '.join(failures)}. "
        "If the downloaded installer is corrupted or refuses silent mode, re-download the official MSVC x86_64 installer manually into "
        f"{installer_path.parent} and run it manually with destination {install_root}."
    )


def _write_manifest(third_party_dir: Path, updates: dict[str, str]) -> None:
    manifest_path = third_party_dir / "manifest.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {}
    data.update(updates)
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
