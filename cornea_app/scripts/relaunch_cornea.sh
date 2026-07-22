#!/usr/bin/env bash
# Relaunch the Cornea OCT app as EXACTLY ONE instance.
#
# Always stop-then-start: two instances both bind port 8765, so the second one's sidecar dies and its
# window silently talks to the FIRST instance's backend. That presents as "my change didn't apply" on a
# build that is actually correct.
#
# THE TRAP THAT BIT US: the AppImage is only a launcher. It FUSE-mounts itself at /tmp/.mount_CorneaXXXXXX
# and execs the real GUI binary, which is called `app` — its argv contains NOTHING matching "Cornea". So
# `pkill -f 'Cornea.OCT_*.AppImage'` kills the wrapper and leaves the actual window, its two WebKit helper
# processes and its sidecar running. The reliable identifier is the EXE PATH: every process belonging to an
# instance has /proc/<pid>/exe under /tmp/.mount_Cornea*. Kill by that, not by argv.
#
# Usage: relaunch_cornea.sh [/path/to/Cornea.OCT_X.Y.Z_amd64.AppImage]   (default: newest in this dir)
set -uo pipefail

BIN_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${CORNEA_DATA_DIR:-/home/zhuojian/Desktop/Integration/review_cases}"
APP="${1:-$(ls -t "$BIN_DIR"/Cornea.OCT_*_amd64.AppImage 2>/dev/null | head -1)}"
LOG="/tmp/cornea-dev/app.log"
mkdir -p /tmp/cornea-dev

[ -x "$APP" ] || { echo "ERROR: no executable AppImage at '$APP'"; exit 1; }

# every pid whose executable lives inside a mounted Cornea AppImage (the window, its WebKit helpers,
# the bundled sidecar) — regardless of what its command line says
mount_pids() {
  local p pid exe
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = "$$" ] && continue
    exe=$(readlink -f "$p/exe" 2>/dev/null) || continue
    case "$exe" in */.mount_Cornea*) echo "$pid";; esac
  done
}

echo "== stopping any running instance =="
for pid in $(mount_pids); do
  exe=$(readlink -f "/proc/$pid/exe" 2>/dev/null)
  kill -9 "$pid" 2>/dev/null && echo "  killed $pid  (${exe##*/})"
done
# the outer AppImage launcher, if it outlived its child
for pid in $(pgrep -f 'Cornea\.OCT_[0-9.]+_amd64\.AppImage' 2>/dev/null); do
  [ "$pid" != "$$" ] && kill -9 "$pid" 2>/dev/null && echo "  killed launcher $pid"
done
# anything still on the port, whatever it is
for pid in $(lsof -ti:8765 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null && echo "  killed port-8765 holder $pid"
done

for _ in $(seq 1 20); do
  [ -z "$(mount_pids)" ] && [ -z "$(lsof -ti:8765 2>/dev/null)" ] && break
  sleep 0.5
done
if [ -n "$(mount_pids)" ] || [ -n "$(lsof -ti:8765 2>/dev/null)" ]; then
  echo "ERROR: processes survived; refusing to start a second instance"
  for pid in $(mount_pids); do echo "  still alive: $pid $(readlink -f /proc/$pid/exe 2>/dev/null)"; done
  exit 1
fi
# stale FUSE mounts leak when a process is SIGKILLed; harmless but they accumulate under /tmp
for m in /tmp/.mount_Cornea*; do
  [ -d "$m" ] && fusermount -u "$m" 2>/dev/null && echo "  unmounted stale $m"
done
echo "  clean: no instance processes, port 8765 free"

echo "== starting $(basename "$APP") =="
DISPLAY="${DISPLAY:-:0}" CORNEA_DATA_DIR="$DATA_DIR" \
  setsid nohup "$APP" > "$LOG" 2>&1 < /dev/null &

for _ in $(seq 1 40); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8765/api/health 2>/dev/null)" = "200" ] && break
  sleep 1
done

ver=$(curl -s http://localhost:8765/api/health 2>/dev/null \
      | python3 -c 'import sys,json;print(json.load(sys.stdin).get("shell_version","?"))' 2>/dev/null)
# count INSTANCES by distinct mount dir, not by process (one instance = window + 2 WebKit helpers + sidecar)
n_inst=$(for pid in $(mount_pids); do readlink -f "/proc/$pid/exe" 2>/dev/null \
         | sed -E 's#(/tmp/\.mount_Cornea[^/]*)/.*#\1#'; done | sort -u | wc -l)
n_port=$(lsof -ti:8765 2>/dev/null | wc -l)

echo "== running =="
echo "  shell_version : ${ver:-<no response>}"
echo "  instances     : $n_inst   port-8765 listeners: $n_port   (both must be 1)"
[ -n "${ver:-}" ] && [ "$n_inst" = "1" ] && [ "$n_port" = "1" ] \
  || { echo "  *** NOT healthy — check $LOG ***"; exit 1; }
