from __future__ import annotations

from app_v2.application.processing_graph import ProcessingGraph


def test_processing_graph_registers_nodes_for_tracing() -> None:
    graph = ProcessingGraph()
    graph.register("nvdec", {"type": "decoder"})

    trace = graph.trace(42)
    assert trace["frame_id"] == 42
    assert "nvdec" in trace["nodes"]
