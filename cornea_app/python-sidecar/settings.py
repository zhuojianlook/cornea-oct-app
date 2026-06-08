"""Sidecar settings: workspace paths + mutable vision-provider config.

Mirrors the fields the old Rust `app_config` (tauri_pipeline/src-tauri/src/main.rs)
exposed, so the frontend config shape is unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

# python-sidecar/ -> cornea_app/ -> Integration/
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
CASES_ROOT = WORKSPACE_ROOT / "cases"
SLICER_BRIDGE_DIR = WORKSPACE_ROOT / "slicer_bridge"
LOCAL_VISION_DIR = WORKSPACE_ROOT / "local_vision"

DEFAULT_SLICER_EXECUTABLE = os.environ.get(
    "SLICER_EXECUTABLE",
    "/home/zhuojian/Applications/Slicer-5.10.0-linux-amd64/Slicer",
)

# Mutable runtime settings, seeded from env (overridable via PUT /api/config).
_settings = {
    "slicer_executable": DEFAULT_SLICER_EXECUTABLE,
    "default_case_id": "case_001",
    "vision_provider": os.environ.get("VISION_PROVIDER", "local"),
    "openai_model": os.environ.get("OPENAI_MODEL", "local-vision-model"),
    "local_vision_base_url": os.environ.get(
        "LOCAL_VISION_BASE_URL",
        os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://127.0.0.1:1234/v1"),
    ),
    # Held in-process only; never written to disk or returned to the client.
    "openai_api_key": os.environ.get("OPENAI_API_KEY", os.environ.get("CODEX_API_KEY", "")),
}


def get_settings() -> dict:
    return dict(_settings)


def update_settings(updates: dict) -> dict:
    for key in (
        "slicer_executable",
        "default_case_id",
        "vision_provider",
        "openai_model",
        "local_vision_base_url",
        "openai_api_key",
    ):
        if key in updates and updates[key] is not None:
            _settings[key] = updates[key]
    return get_settings()


def public_config() -> dict:
    """Config safe to send to the frontend (no secrets)."""
    s = _settings
    return {
        "workspace_root": str(WORKSPACE_ROOT),
        "cases_root": str(CASES_ROOT),
        "slicer_executable": s["slicer_executable"],
        "default_case_id": s["default_case_id"],
        "vision_provider": s["vision_provider"],
        "openai_model": s["openai_model"],
        "local_vision_base_url": s["local_vision_base_url"],
        "has_openai_api_key": bool(s["openai_api_key"].strip()),
    }
