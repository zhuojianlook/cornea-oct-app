"""Sidecar settings: workspace paths + mutable runtime config.

Non-secret settings (Slicer path, open case) persist across sidecar restarts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Optional data-dir override: a packaged/installed app sets CORNEA_DATA_DIR (the OS app-data dir,
# passed by the Tauri shell) so cases + state are written somewhere user-writable instead of the
# read-only app bundle. Unset (dev / run-from-source) keeps the original in-repo paths exactly.
_DATA_DIR = os.environ.get("CORNEA_DATA_DIR")
_STATE_FILE = (Path(_DATA_DIR) / ".sidecar_state.json") if _DATA_DIR else (Path(__file__).parent / ".sidecar_state.json")
_PERSIST_KEYS = ("slicer_executable", "default_case_id")

# Code location of this sidecar: dev = <repo>/cornea_app/python-sidecar; bundle = read-only resource dir.
_SIDECAR_DIR = Path(__file__).resolve().parent
# python-sidecar/ -> cornea_app/ -> Integration/ (dev); or CORNEA_DATA_DIR when packaged.
WORKSPACE_ROOT = Path(_DATA_DIR) if _DATA_DIR else _SIDECAR_DIR.parents[1]
CASES_ROOT = WORKSPACE_ROOT / "cases"
if _DATA_DIR:
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    CASES_ROOT.mkdir(parents=True, exist_ok=True)


def _resolve_slicer_bridge() -> Path:
    """slicer_bridge (preview_io.py etc.) is shipped CODE, not data: it sits at the repo root in
    dev and is bundled NEXT TO python-sidecar in the packaged app — NOT in CORNEA_DATA_DIR, which
    holds only cases/output. Resolve to wherever preview_io.py actually exists."""
    for cand in (
        _SIDECAR_DIR.parents[1] / "slicer_bridge",   # dev: <repo>/slicer_bridge
        _SIDECAR_DIR.parent / "slicer_bridge",        # bundle: <resource_dir>/slicer_bridge (sibling)
        _SIDECAR_DIR / "slicer_bridge",               # nested fallback
    ):
        if (cand / "preview_io.py").exists():
            return cand
    return _SIDECAR_DIR.parents[1] / "slicer_bridge"


SLICER_BRIDGE_DIR = _resolve_slicer_bridge()

DEFAULT_SLICER_EXECUTABLE = os.environ.get(
    "SLICER_EXECUTABLE",
    "/home/zhuojian/Applications/Slicer-5.10.0-linux-amd64/Slicer",
)

# Mutable runtime settings, seeded from env (overridable via PUT /api/config).
_settings = {
    "slicer_executable": DEFAULT_SLICER_EXECUTABLE,
    "default_case_id": "case_oct_real",
}


def _load_state() -> None:
    try:
        data = json.loads(_STATE_FILE.read_text())
        for key in _PERSIST_KEYS:
            if key in data and data[key]:
                _settings[key] = data[key]
    except Exception:
        pass


def _save_state() -> None:
    try:
        _STATE_FILE.write_text(json.dumps({k: _settings[k] for k in _PERSIST_KEYS}, indent=2))
    except Exception:
        pass


def get_settings() -> dict:
    return dict(_settings)


def update_settings(updates: dict) -> dict:
    for key in ("slicer_executable", "default_case_id"):
        if key in updates and updates[key] is not None:
            _settings[key] = updates[key]
    _save_state()
    return get_settings()


# Restore persisted settings at import.
_load_state()


def public_config() -> dict:
    s = _settings
    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "cases_root": str(CASES_ROOT),
        "slicer_executable": s["slicer_executable"],
        "default_case_id": s["default_case_id"],
    }
