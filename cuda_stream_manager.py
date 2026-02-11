try:
    import torch
except Exception:
    torch = None

try:
    from cuda.bindings import runtime as cudart
except Exception:
    cudart = None


class CudaStreamManager:
    """Centralizes CUDA stream creation and safe reuse for TensorRT execution."""

    def __init__(self, name="cuda"):
        self.name = name
        self._cudart = cudart
        self._torch_stream = None
        self._handle = 0
        self.reset()

    def reset(self):
        self.destroy()
        if torch is not None and torch.cuda.is_available():
            self._torch_stream = torch.cuda.Stream()
            self._handle = self._torch_stream.cuda_stream
        elif self._cudart is not None:
            err, stream = self._cudart.cudaStreamCreate()
            if int(err) != 0:
                raise RuntimeError(f"CUDA Stream creation failed ({err})")
            self._handle = stream
        else:
            self._handle = 0

    def destroy(self):
        if self._torch_stream is not None:
            self._torch_stream = None
        elif self._cudart is not None and self._handle:
            self._cudart.cudaStreamDestroy(self._handle)
        self._handle = 0

    def handle(self):
        return int(self._handle)

    def synchronize(self):
        if self._torch_stream is not None:
            self._torch_stream.synchronize()
        elif self._cudart is not None and self._handle:
            self._cudart.cudaStreamSynchronize(self._handle)

    def is_valid(self):
        return bool(self._handle)

    def ensure_valid(self):
        if not self.is_valid():
            self.reset()

    def is_capturing(self):
        if self._cudart is None or not self._handle:
            return False
        err, status = self._cudart.cudaStreamIsCapturing(self._handle)
        if int(err) != 0:
            return False
        try:
            capturing_enum = self._cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusNone
        except AttributeError:
            return False
        return status != capturing_enum
