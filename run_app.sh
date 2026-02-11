#!/usr/bin/env bash
set -euo pipefail

# Legacy entry point preserved for compatibility. Prefer invoking 0_build_image.sh and 1_prepare.sh before
# running 3_run_app.sh when switching between versions.
exec "$(dirname "$0")/3_run_app.sh" "$@"
