# Windows GStreamer Bridge

This document covers the new Windows streaming bridge based on **Python + GStreamer + MediaMTX**.

It is designed to keep Python in the **UI/control plane** while GStreamer owns decoding, timing, buffering, hardware encoding, and publication.

## What this bridge does

- Publishes a local RTSP stream through MediaMTX
- Streams images and video files today
- Keeps a Python UI for source selection and runtime metrics
- Requires a **real Intel QSV encoder** (`qsvh264enc`)
- Does **not** silently fall back to software encoding

Launcher:

- `run_gst_stream.bat`

Python runtime:

- Dedicated `windows\venv_bridge`
- Defaults to **Python 3.12**
- Rebuilds the venv automatically if it was created with another Python version

## First start

From the repository root, run:

- `run_gst_stream.bat`

The launcher will:

1. Create or rebuild the dedicated Python 3.12 virtual environment
2. Install Python dependencies from `windows\requirements-gst-bridge.txt`
3. Ensure local dependencies under `windows\third_party\`
4. Validate the local GStreamer runtime and required plugins
5. Start MediaMTX and the Python UI

## Local third-party layout

The bridge expects this project-local layout:

- `windows\third_party\downloads\`
- `windows\third_party\gstreamer\1.28.2\`
- `windows\third_party\mediamtx\1.11.3\`

## If the downloaded GStreamer installer is corrupted

This can happen with large downloads on Windows.

Expected installer file:

- `windows\third_party\downloads\gstreamer-1.28.2.exe`

Recovery steps:

1. Re-download the official **MSVC x86_64** GStreamer installer manually
2. Replace the corrupted file in `windows\third_party\downloads\`
3. Re-run `run_gst_stream.bat`

If silent install still fails, run the installer manually and choose this destination:

- `windows\third_party\gstreamer\1.28.2`

After manual installation, restart the terminal and launch the bridge again.

## Environment variables

Depending on how GStreamer was installed, Windows may expose a root variable such as:

- `GSTREAMER_1_0_ROOT_MSVC_X86_64`

The bridge also sets the local runtime environment explicitly before launching GStreamer, including:

- `GSTREAMER_1_0_ROOT_X86_64`
- `GSTREAMER_1_0_ROOT_MSVC_X86_64`
- `GST_PLUGIN_SYSTEM_PATH_1_0`
- `GST_PLUGIN_PATH_1_0`

That keeps the runtime focused on the project-local install under `windows\third_party\gstreamer\1.28.2`.

## How to verify the required plugins

The most useful binary is:

- `windows\third_party\gstreamer\1.28.2\bin\gst-inspect-1.0.exe`

Check the required encoder:

- `windows\third_party\gstreamer\1.28.2\bin\gst-inspect-1.0.exe qsvh264enc`

Also verify the publication and metrics elements:

- `windows\third_party\gstreamer\1.28.2\bin\gst-inspect-1.0.exe rtspclientsink`
- `windows\third_party\gstreamer\1.28.2\bin\gst-inspect-1.0.exe fpsdisplaysink`

Check the installed runtime version:

- `windows\third_party\gstreamer\1.28.2\bin\gst-launch-1.0.exe --version`

If `qsvh264enc` is available, you should see output similar to:

- plugin name: `qsv`
- element: `qsvh264enc`
- encoder description mentioning **Intel Quick Sync Video**

## What was verified on this machine

The local project install under:

- `windows\third_party\gstreamer\1.28.2`

was checked successfully for:

- `gst-launch-1.0.exe`
- `gst-inspect-1.0.exe`
- `qsvh264enc`
- `rtspclientsink`
- `fpsdisplaysink`

`gst-inspect-1.0.exe qsvh264enc` reported an Intel Arc H.264 hardware encoder from `gstqsv.dll`.

## Troubleshooting

### Error: missing `qsvh264enc`

Use:

- `windows\third_party\gstreamer\1.28.2\bin\gst-inspect-1.0.exe qsvh264enc`

If it fails:

- verify you installed the **MSVC x86_64** package, not another variant
- verify the plugin file exists under `windows\third_party\gstreamer\1.28.2\lib\gstreamer-1.0\gstqsv.dll`
- restart the terminal after a manual installation
- re-run `run_gst_stream.bat`

### Error: silent installer failed

The launcher tries several silent modes. If all fail:

- keep the installer in `windows\third_party\downloads\`
- run it manually
- install to `windows\third_party\gstreamer\1.28.2`

### Error: bridge still reports a missing plugin even though `gst-inspect` works

Re-run the bridge from a fresh terminal. The bridge now prints a more detailed diagnostic including:

- selected GStreamer root
- plugin directory
- exact `gst-inspect` verification command
- first diagnostic line returned for each missing plugin

## Notes

- The new bridge is intended for **strict hardware validation**, so hidden software fallbacks are intentionally avoided.
- The legacy FFmpeg bridge remains available separately for comparison and fallback testing during development.
