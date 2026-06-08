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


def render_context_previews(input_volume: str, preview_dir: Path, max_slices: int = 9) -> dict:
    return run_slicer(
        "render_volume_context.py",
        ["--input-volume", input_volume, "--preview-dir", str(preview_dir),
         "--max-slices-per-orientation", str(max_slices)],
    )


def render_seed_previews(input_volume: str, seed_json: str, preview_dir: Path) -> dict:
    return run_slicer(
        "render_seed_previews.py",
        ["--input-volume", input_volume, "--seed-json", seed_json, "--preview-dir", str(preview_dir)],
    )


def heuristic_seeds(input_volume: str, output_seed_json: str, qa_json: str,
                    preview_dir: Path, feedback_json: str) -> dict:
    return run_slicer(
        "agent_refine_paint.py",
        ["--input-volume", input_volume, "--output-seed-json", output_seed_json,
         "--qa-json", qa_json, "--preview-dir", str(preview_dir),
         "--feedback-json", feedback_json],
    )


def grow_from_seeds(input_volume: str, seed_json: str, output_seg: str, qa_json: str,
                    scene: str, preview_dir: Path, seed_locality_factor: float = 0.0) -> dict:
    return run_slicer(
        "seeded_grow_from_seeds.py",
        ["--input-volume", input_volume, "--seed-json", seed_json,
         "--output-seg", output_seg, "--qa-json", qa_json, "--scene", scene,
         "--preview-dir", str(preview_dir),
         "--seed-locality-factor", str(seed_locality_factor)],
    )
