#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLICER_EXECUTABLE="${SLICER_EXECUTABLE:-/home/zhuojian/Applications/Slicer-5.10.0-linux-amd64/Slicer}"

exec "$SLICER_EXECUTABLE" \
  --no-main-window \
  --python-script "$SCRIPT_DIR/seeded_grow_from_seeds.py" \
  "$@"
