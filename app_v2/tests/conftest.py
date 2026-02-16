from __future__ import annotations

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*pynvml package is deprecated.*",
    category=FutureWarning,
)
