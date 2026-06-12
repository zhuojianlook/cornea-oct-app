"""Sidecar settings: workspace paths + mutable runtime config.

Non-secret settings (Slicer path, open case) persist across sidecar restarts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_STATE_FILE = Path(__file__).parent / ".sidecar_state.json"
_PERSIST_KEYS = ("slicer_executable", "default_case_id")

# python-sidecar/ -> cornea_app/ -> Integration/
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = WORKSPACE_ROOT / "cases"
SLICER_BRIDGE_DIR = WORKSPACE_ROOT / "slicer_bridge"

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
