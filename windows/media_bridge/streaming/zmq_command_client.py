from __future__ import annotations

import zmq


class ZmqCommandClient:
    def __init__(self, endpoint: str) -> None:
        self._context = zmq.Context.instance()
        self._endpoint = endpoint

    def send(self, target: str, command: str, argument: str, timeout_ms: int = 4000) -> str:
        payload = f"{target} {command} {argument}".strip()
        socket = self._context.socket(zmq.REQ)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.connect(self._endpoint)
        try:
            socket.send_string(payload)
            return socket.recv_string()
        finally:
            socket.close(linger=0)

    def close(self) -> None:
        return None