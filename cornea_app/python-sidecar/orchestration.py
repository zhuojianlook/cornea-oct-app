"""Case lifecycle, manifest, and preview helpers.

Per-case path layout under cases/<id>/, manifest read/merge/write, and the
base64 preview listing used by the 2D slice gallery.
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import settings

# Serializes manifest read-modify-write within this process. The sidecar is a single
# process, so an in-process lock is sufficient to make concurrent endpoint handlers
# (run in a threadpool by Starlette) safe; combined with the atomic replace below a
# reader never observes a half-written file. RLock so a future nested write is safe.
_MANIFEST_LOCK = threading.RLock()

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


def _atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: write a sibling temp file, fsync, then
    os.replace (atomic on POSIX and Windows). A crash mid-write leaves the previous
    file intact instead of a truncated/empty one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_manifest_value(case_id: str, updates: dict) -> dict:
    path = manifest_path(case_id)
    with _MANIFEST_LOCK:
        # Guard against silently nuking a populated manifest: if the file exists and is
        # non-empty but fails to parse (corruption / partial earlier write), back up the
        # bytes rather than overwriting an empty {} merged with the update — that would
        # destroy oct_source and every prior flag (HIGH-severity data-loss path).
        if path.exists():
            try:
                raw = path.read_text()
            except Exception:
                raw = ""
            if raw.strip():
                try:
                    parsed = json.loads(raw)
                except Exception:
                    backup = path.with_suffix(path.suffix + ".corrupt")
                    try:
                        os.replace(path, backup)
                    except OSError:
                        pass
                    parsed = {}
                current = parsed if isinstance(parsed, dict) else {}
            else:
                current = {}
        else:
            current = {}
        current["case_id"] = safe_case_id(case_id)
        current.update(updates)
        _atomic_write_text(path, json.dumps(current, indent=2))
        return current


def filter_scheduled(case_ids: list[str]) -> list[str]:
    """Honor the timeline's "Schedule for training" gate: if ANY of these cases is flagged
    training_scheduled, keep ONLY the scheduled ones; if none are flagged, return them all
    (backward-compatible — scheduling nothing means train/export everything). Order preserved."""
    scheduled = [c for c in case_ids if read_manifest(c).get("training_scheduled")]
    return scheduled if scheduled else list(case_ids)


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


def preview_listing_from_dir(group: str, directory: Path, src_base: str) -> list[dict]:
    """Like preview_images_from_dir but returns a lazy `src` URL (``src_base/<file>``) for
    each PNG instead of an inline base64 data-URL. The client then loads only the slice on
    screen, so a DENSE group (every slice of a scrub) is cheap to list and to view — the
    base64 form would inline every slice (tens of MB) into one response."""
    if not directory.exists():
        return []
    meta = preview_metadata_by_file(directory)
    images = []
    for path in png_paths_in_dir(directory):
        m = meta.get(path.name, {})
        try:
            ver = int(path.stat().st_mtime)   # cache-bust: re-rendered PNGs get a new URL
        except OSError:
            ver = 0
        images.append({
            "label": f"{group} / {path.name}",
            "path": str(path),
            "group": group,
            "file_name": path.name,
            "data_url": "",
            "src": f"{src_base}/{path.name}?v={ver}",
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
