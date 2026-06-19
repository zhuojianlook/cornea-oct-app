#!/bin/bash
# Single-port launcher for the Cornea OCT app.
#
# Builds the React UI into dist/ (only when missing or stale) and starts ONLY
# the FastAPI sidecar, which then serves both the API and the built UI on :8765
# (see the StaticFiles mount in python-sidecar/api_server.py). One process, one
# port — use this where a separate Vite dev server can't be kept alive.
#
# For live-reload development use dev-launch.sh instead (sidecar + Vite on 1420).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="/tmp/cornea-dev"
mkdir -p "$LOGDIR"
: > "$LOGDIR/backend.log"

cd "$SCRIPT_DIR"

# Rebuild if dist/ is missing, or any source file is newer than the built bundle.
NEEDS_BUILD=0
if [ ! -f dist/index.html ]; then
  NEEDS_BUILD=1
elif [ -n "$(find src index.html package.json vite.config.ts -type f -newer dist/index.html 2>/dev/null | head -1)" ]; then
  NEEDS_BUILD=1
fi

if [ "$NEEDS_BUILD" = "1" ]; then
  echo "$(date): building frontend (npm run build)..."
  npm run build
else
  echo "$(date): dist/ is up to date — skipping build."
fi

echo "$(date): cleaning old processes..."
pkill -f "api_server.py --port 8765" 2>/dev/null || true
lsof -ti:8765 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

echo "$(date): starting Python sidecar on 8765 (serving API + UI)..."
cd "$SCRIPT_DIR/python-sidecar"
python3 api_server.py --port 8765 >> "$LOGDIR/backend.log" 2>&1 &
BACKEND_PID=$!
for _ in $(seq 1 30); do
  grep -q "READY:8765" "$LOGDIR/backend.log" 2>/dev/null && break
  sleep 0.5
done
echo "$(date): sidecar PID=$BACKEND_PID"

echo ""
echo "=== Cornea OCT (single-port) ==="
echo "  App + API : http://127.0.0.1:8765            (log: $LOGDIR/backend.log)"
echo ""
echo "Open http://127.0.0.1:8765 in a browser. Ctrl-C to stop."
wait $BACKEND_PID
