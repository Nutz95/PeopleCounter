#!/usr/bin/env bash
set -euo pipefail

# Diagnostics run before starting the main application to ensure cuda-python
# and libcudart are visible inside the container when profiling.

printf '[diag] LD_LIBRARY_PATH="%s"\n' "$LD_LIBRARY_PATH"
ldconfig -p | grep libcudart || true
python3 - <<'PY'
import importlib
import traceback

try:
    runtime = importlib.import_module('cuda.bindings.runtime')
    print('[diag] cuda.bindings runtime path:', runtime.__file__)
    rc = runtime.cudaProfilerStart()
    print('[diag] cudaProfilerStart dry run ->', rc)
    if rc == 0:
        rc_stop = runtime.cudaProfilerStop()
        print('[diag] cudaProfilerStop dry run ->', rc_stop)
    else:
        print('[diag] cudaProfilerStart returned non-zero, skipping stop')
except Exception as exc:
    print('[diag] cuda runtime import failed:', exc)
    traceback.print_exc()
finally:
    print('[diag] cuda diagnostics script exiting')
PY
