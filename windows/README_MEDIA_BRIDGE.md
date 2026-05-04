# Media Bridge Windows

> This document covers the **legacy FFmpeg HTTP bridge**.
> For the new Windows streaming path based on **GStreamer + MediaMTX**, see [`README_GST_BRIDGE.md`](README_GST_BRIDGE.md).

This bridge publishes a VLC- and NVDEC-friendly H.264 MPEG-TS stream over HTTP at `/video_feed`.

## Features

- Camera mode via DirectShow, with selectable capture modes
- Media mode for a single video or image source, looped through FFmpeg
- Continuous HTTP stream exposed on `http://<windows-ip>:5002/video_feed`
- H.264 encoding with repeated headers and periodic keyframes for fast client startup
- Letterbox scaling so every source is normalized to the configured output size and frame rate

## Quick start

Run `setup_and_run_ref_video.bat` from the `windows/` folder.

## Notes

- VLC can open the stream directly via `http://<windows-ip>:5002/video_feed`.
- The same URL can be used by the NVDEC-based `app_v2` pipeline through the `RTSP_URL` environment variable or the run command shown in the console.
- The bridge intentionally stays on HTTP MPEG-TS instead of RTSP so it remains easy to consume from both desktop players and GPU decoders.
