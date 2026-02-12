from __future__ import annotations

import importlib


def test_pynvcodec_import_and_basic_instantiation() -> None:
    nvc = importlib.import_module("PyNvCodec")

    packet = nvc.PacketData()
    assert packet is not None

    color_ctx = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_709, nvc.ColorRange.MPEG)
    assert color_ctx is not None

    surface = nvc.Surface.Make(nvc.PixelFormat.NV12, 640, 480, 0)
    assert surface is not None
