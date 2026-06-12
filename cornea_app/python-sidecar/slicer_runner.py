"""Subprocess wrapper around the 3D Slicer bridge scripts.

The slicer_bridge/*.py scripts `import slicer`, so they only run under the
Slicer executable, never the sidecar's Python. We invoke them exactly like the
old Rust orchestrator did:

    Slicer --no-main-window --python-script <script> <flags...>

with cwd + PWD set to the workspace root (the scripts resolve relative paths
from PWD). Runs are slow and serialised, so callers should dispatch them off
the event loop (run_in_executor).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Sequence

import settings


def run_slicer(script_name: str, args: Sequence[str], timeout: int = 1200) -> dict:
    """Run a slicer_bridge script and capture {status, stdout, stderr}."""
    script = settings.SLICER_BRIDGE_DIR / script_name
    if not script.exists():
        return {"status": -1, "stdout": "", "stderr": f"Missing script: {script}"}

    slicer_exe = settings.get_settings()["slicer_executable"]
    cmd = [
        slicer_exe,
        "--no-main-window",
        "--python-script",
        str(script),
        *[str(a) for a in args],
    ]
    env = dict(os.environ)
    env["PWD"] = str(settings.WORKSPACE_ROOT)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(settings.WORKSPACE_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {
            "status": -1,
            "stdout": "",
            "stderr": f"Slicer executable not found: {slicer_exe}. Set it in Settings.",
        }
    except subprocess.TimeoutExpired as exc:
        return {"status": -1, "stdout": exc.stdout or "", "stderr": f"Slicer timed out after {timeout}s"}
    return {
        "status": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def convert_to_nifti(input_volume: str, output: str) -> dict:
    """The only Slicer dependency in the focused app: DICOM → NIfTI."""
    return run_slicer(
        "convert_to_nifti.py",
        ["--input-volume", input_volume, "--output", output],
        timeout=600,
    )
