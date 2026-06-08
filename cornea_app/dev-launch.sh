#!/bin/bash
# Dev launcher for the Cornea OCT app.
#
# Browser-dev-first: starts the FastAPI sidecar (8765) + Vite (1420) and opens
# in a browser. The native Tauri window is DEFERRED — it needs
# `sudo apt install libwebkit2gtk-4.1-dev` (this machine only has 4.0). Pass
# --native to additionally launch `tauri dev` once that lib is installed.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGDIR="/tmp/cornea-dev"
mkdir -p "$LOGDIR"
: > "$LOGDIR/backend.log"
: > "$LOGDIR/vite.log"

NATIVE=0
[ "$1" = "--native" ] && NATIVE=1

echo "$(date): cleaning old processes..."
pkill -f "api_server.py --port 8765" 2>/dev/null || true
pkill -f "vite.*1420" 2>/dev/null || true
lsof -ti:8765 2>/dev/null | xargs -r kill -9 2>/dev/null || true
lsof -ti:1420 2>/dev/null | xargs -r kill -9 2>/dev/null || true
sleep 1

echo "$(date): starting Python sidecar on 8765..."
cd "$SCRIPT_DIR/python-sidecar"
python3 api_server.py --port 8765 >> "$LOGDIR/backend.log" 2>&1 &
BACKEND_PID=$!
for _ in $(seq 1 30); do
  grep -q "READY:8765" "$LOGDIR/backend.log" 2>/dev/null && break
  sleep 0.5
done
echo "$(date): sidecar PID=$BACKEND_PID"

echo "$(date): starting Vite on 1420..."
cd "$SCRIPT_DIR"
npx vite --port 1420 >> "$LOGDIR/vite.log" 2>&1 &
VITE_PID=$!
for _ in $(seq 1 30); do
  grep -q "ready in" "$LOGDIR/vite.log" 2>/dev/null && break
  sleep 0.5
done
echo "$(date): vite PID=$VITE_PID"

echo ""
echo "=== Cornea OCT dev environment ==="
echo "  Sidecar : http://127.0.0.1:8765/api/health  (log: $LOGDIR/backend.log)"
echo "  App     : http://localhost:1420             (log: $LOGDIR/vite.log)"
echo ""

if [ "$NATIVE" = "1" ]; then
  echo "$(date): launching native Tauri window..."
  cd "$SCRIPT_DIR"
  npx tauri dev
else
  echo "Open http://localhost:1420 in a browser. Ctrl-C to stop."
  wait $BACKEND_PID $VITE_PID
fi
