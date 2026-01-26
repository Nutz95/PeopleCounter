#!/usr/bin/env python3
"""Download LWCC pretrained weights into ~/.lwcc/weights using lwcc.util.functions.weights_check
"""
from pathlib import Path
import os
import sys

# Add lwcc to path
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root, "external", "lwcc"))

try:
    from lwcc.util.functions import weights_check
except Exception as e:
    print(f"Failed to import lwcc. Error: {e}")
    # Try another common location just in case
    sys.path.append(os.path.join(root, "lwcc"))
    from lwcc.util.functions import weights_check

PAIRS = [
    ("CSRNet", "SHA"),
    ("CSRNet", "SHB"),
    ("Bay", "QNRF"),
    ("Bay", "SHA"),
    ("Bay", "SHB"),
    ("DM-Count", "QNRF"),
    ("DM-Count", "SHA"),
    ("DM-Count", "SHB"),
    ("SFANet", "SHB"),
]

def main():
    home = str(Path.home())
    weights_dir = os.path.join(home, ".lwcc", "weights")
    os.makedirs(weights_dir, exist_ok=True)
    for model, weights in PAIRS:
        try:
            print(f"Ensuring LWCC weight: {model}_{weights}")
            out = weights_check(model, weights)
            print("->", out)
        except Exception as e:
            print(f"Failed to download {model}_{weights}: {e}")

if __name__ == '__main__':
    main()
