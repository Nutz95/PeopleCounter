#!/usr/bin/env python3
"""Download LWCC pretrained weights into ~/.lwcc/weights using lwcc.util.functions.weights_check
"""
from pathlib import Path
import os
import sys

# Add lwcc to path
root = os.path.dirname(os.path.abspath(__file__))
# Check various common locations for lwcc
for p in [os.path.join(root, "vendors", "lwcc"), os.path.join(root, "external", "lwcc"), root]:
    if p not in sys.path:
        sys.path.append(p)

try:
    from lwcc.util.functions import weights_check
except Exception as e:
    print(f"Failed to import lwcc. Error: {e}")
    sys.exit(1)

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
            print(f"weights_check failed for {model}_{weights}: {e}")
            # Fallback: download prebuilt weights from Nutz95 release v0.1 on GitHub
            try:
                filename = f"{model}_{weights}.pth"
                url = f"https://github.com/Nutz95/lwcc_weights/releases/download/v0.1/{filename}"
                dest = os.path.join(weights_dir, filename)
                if os.path.exists(dest):
                    print(f"Already present: {dest}")
                else:
                    print(f"Attempting fallback download from {url}")
                    try:
                        # Use urllib to avoid extra dependencies
                        from urllib.request import urlopen
                        with urlopen(url) as resp, open(dest, 'wb') as out_f:
                            out_f.write(resp.read())
                        print(f"Downloaded fallback weight to {dest}")
                    except Exception as de:
                        print(f"Fallback download failed for {filename}: {de}")
            except Exception as fe:
                print(f"Fallback handling failed for {model}_{weights}: {fe}")

if __name__ == '__main__':
    main()
