"""Case lifecycle, manifest, and preview helpers.

Per-case path layout under cases/<id>/, manifest read/merge/write, and the
base64 preview listing used by the 2D slice gallery.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import settings

SUBDIRS = ["input", "segmentation", "previews"]


def safe_case_id(value: str) -> str:
    out = []
    for ch in (value or "").strip():
        out.append(ch if (ch.isalnum() or ch in "_-.") else "_")
    cleaned = "".join(out).strip("._-")
    return cleaned or "case_001"


def case_root(case_id: str) -> Path:
    return settings.CASES_ROOT / safe_case_id(case_id)


def case_qa_json(case_id: str) -> Path:
    cid = safe_case_id(case_id)
    return case_root(cid) / "segmentation" / f"{cid}_qa.json"


def manifest_path(case_id: str) -> Path:
    return case_root(case_id) / "manifest.json"


def context_preview_dir(case_id: str) -> Path:
    return case_root(case_id) / "previews" / "context"


def segmentation_preview_dir(case_id: str) -> Path:
    return case_root(case_id) / "previews" / "segmentation"


def ensure_case_dirs(case_id: str) -> None:
    root = case_root(case_id)
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def read_manifest(case_id: str) -> dict:
    data = read_json(manifest_path(case_id))
    return data if isinstance(data, dict) else {}


def write_manifest_value(case_id: str, updates: dict) -> dict:
    current = read_manifest(case_id)
    current["case_id"] = safe_case_id(case_id)
    current.update(updates)
    path = manifest_path(case_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2))
    return current


# ── Preview listing for the 2D slice gallery ───────────────────────────────
def preview_metadata_by_file(directory: Path) -> dict:
    manifest = read_json(directory / "preview_manifest.json")
    by_file: dict[str, dict] = {}
    if isinstance(manifest, dict):
        for image in manifest.get("images", []) or []:
            fn = image.get("file_name")
            if fn:
                by_file[fn] = image
    return by_file


def png_paths_in_dir(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() == ".png")


def preview_images_from_dir(group: str, directory: Path) -> list[dict]:
    """List PNG previews as base64 data-URLs plus their manifest metadata."""
    if not directory.exists():
        return []
    meta = preview_metadata_by_file(directory)
    images = []
    for path in png_paths_in_dir(directory):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        m = meta.get(path.name, {})
        images.append({
            "label": f"{group} / {path.name}",
            "path": str(path),
            "group": group,
            "file_name": path.name,
            "data_url": f"data:image/png;base64,{encoded}",
            "orientation": m.get("orientation"),
            "slice_index": m.get("slice_index"),
            "source_width": m.get("source_width"),
            "source_height": m.get("source_height"),
            "image_width": m.get("image_width"),
            "image_height": m.get("image_height"),
        })
    return images


def current_case_info(case_id: str) -> dict:
    cid = safe_case_id(case_id)
    root = case_root(cid)
    return {
        "case_id": cid,
        "root": str(root),
        "input_dir": str(root / "input"),
        "segmentation_dir": str(root / "segmentation"),
        "qa_json": str(case_qa_json(cid)),
        "manifest": read_manifest(cid),
    }
