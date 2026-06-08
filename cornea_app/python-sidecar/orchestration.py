"""Case lifecycle, manifest, and seed-spec helpers.

Ported from the old Rust orchestrator (tauri_pipeline/src-tauri/src/main.rs):
case-dir scaffolding, manifest read/merge/write, seed-template defaulting, and
the per-case path layout under cases/<id>/.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import settings

SUBDIRS = ["input", "filtering", "correction", "segmentation", "review", "previews"]


def safe_case_id(value: str) -> str:
    out = []
    for ch in (value or "").strip():
        out.append(ch if (ch.isalnum() or ch in "_-.") else "_")
    cleaned = "".join(out).strip("._-")
    return cleaned or "case_001"


def case_root(case_id: str) -> Path:
    return settings.CASES_ROOT / safe_case_id(case_id)


def case_seed_json(case_id: str) -> Path:
    return case_root(case_id) / "segmentation" / "seeds.json"


def case_output_seg(case_id: str) -> Path:
    cid = safe_case_id(case_id)
    return case_root(cid) / "segmentation" / f"{cid}.seg.nrrd"


def case_qa_json(case_id: str) -> Path:
    cid = safe_case_id(case_id)
    return case_root(cid) / "segmentation" / f"{cid}_qa.json"


def case_scene(case_id: str) -> Path:
    cid = safe_case_id(case_id)
    return case_root(cid) / "segmentation" / f"{cid}.mrb"


def manifest_path(case_id: str) -> Path:
    return case_root(case_id) / "manifest.json"


def context_preview_dir(case_id: str) -> Path:
    return case_root(case_id) / "previews" / "context"


def seed_preview_dir(case_id: str) -> Path:
    return case_root(case_id) / "previews" / "seeds"


def segmentation_preview_dir(case_id: str) -> Path:
    return case_root(case_id) / "previews" / "segmentation"


def feedback_json(case_id: str) -> Path:
    return case_root(case_id) / "review" / "feedback.json"


def _default_seed_json() -> str:
    template = settings.SLICER_BRIDGE_DIR / "seed_template.json"
    if template.exists():
        return template.read_text()
    return json.dumps(
        {
            "segments": [
                {"name": "background", "color": [0.05, 0.05, 0.05],
                 "seeds": [{"ijk": [20, 20, 20], "radius_voxels": [6, 6, 3]}]},
                {"name": "cornea", "color": [0.1, 0.7, 1.0],
                 "seeds": [{"ijk": [48, 48, 40], "radius_voxels": [8, 8, 3]}]},
            ]
        },
        indent=2,
    )


def ensure_seed_file(case_id: str) -> None:
    path = case_seed_json(case_id)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_default_seed_json())


def ensure_case_dirs(case_id: str) -> None:
    root = case_root(case_id)
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    ensure_seed_file(case_id)


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


# ── Preview metadata + sampling (ported from old Rust) ─────────────────────
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


def _preview_orientation(path: Path) -> str:
    name = path.name
    for o in ("axial", "coronal", "sagittal"):
        if f"_{o}_" in name:
            return o
    return "other"


def _sample_three(paths: list[Path]) -> list[Path]:
    if len(paths) <= 3:
        return list(paths)
    last = len(paths) - 1
    return [paths[last // 4], paths[last // 2], paths[(last * 3) // 4]]


def selected_preview_png_paths(all_paths: list[Path], provider_name: str) -> list[Path]:
    if provider_name == "openai":
        return list(all_paths)
    selected: list[Path] = []
    for orientation in ("axial", "coronal", "sagittal"):
        group = sorted(p for p in all_paths if _preview_orientation(p) == orientation)
        selected.extend(_sample_three(group))
    return selected if selected else list(all_paths)[:9]


# ── Coordinate mapping (PNG pixel → IJK). Validated against the old Rust. ────
def _value_f64(v) -> float | None:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def _point_pair(value) -> tuple[float, float] | None:
    if isinstance(value, list) and len(value) >= 2:
        a, b = _value_f64(value[0]), _value_f64(value[1])
        return (a, b) if a is not None and b is not None else None
    if isinstance(value, dict):
        x = _value_f64(value.get("x", value.get("px_x")))
        y = _value_f64(value.get("y", value.get("px_y")))
        return (x, y) if x is not None and y is not None else None
    return None


def _point_ijk(value) -> list[int] | None:
    if isinstance(value, list) and len(value) >= 3:
        vals = [_value_f64(value[i]) for i in range(3)]
        if all(v is not None for v in vals):
            return [round(v) for v in vals]  # type: ignore[arg-type]
    return None


def radius_voxels(value, segment: str) -> list[int]:
    fallback = [6, 6, 2] if segment == "background" else [5, 5, 2]
    if value is None:
        return fallback
    num = _value_f64(value)
    if num is not None and not isinstance(value, list):
        r = max(1, round(num))
        return [r, r, r]
    if isinstance(value, list) and len(value) == 3:
        out = []
        for item in value:
            n = _value_f64(item)
            if n is None:
                return fallback
            out.append(max(1, round(n)))
        return out
    return fallback


def px_to_ijk(metadata: dict, x: float, y: float) -> list[int]:
    orientation = metadata["orientation"]
    slice_index = int(metadata["slice_index"])
    sw = max(1, int(metadata["source_width"]))
    sh = max(1, int(metadata["source_height"]))
    iw = max(1, int(metadata["image_width"]))
    ih = max(1, int(metadata["image_height"]))
    x = min(max(x, 0.0), iw - 1.0)
    y = min(max(y, 0.0), ih - 1.0)
    src_col = 0.0 if iw <= 1 else x * (sw - 1.0) / (iw - 1.0)
    unflipped_row = ih - 1.0 - y
    src_row = 0.0 if ih <= 1 else unflipped_row * (sh - 1.0) / (ih - 1.0)
    col = round(src_col)
    row = round(src_row)
    if orientation == "axial":
        return [col, row, slice_index]
    if orientation == "coronal":
        return [col, slice_index, row]
    if orientation == "sagittal":
        return [slice_index, col, row]
    raise ValueError(f"Unknown preview orientation: {orientation}")


# ── Model-output parsing → seed spec (ported from old Rust) ─────────────────
def parse_json_from_model_text(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except Exception:
        return {}


def _stroke_items(parsed: dict) -> list[dict]:
    for key in ("strokes", "paint_strokes", "agent_strokes"):
        items = parsed.get(key)
        if isinstance(items, list):
            return list(items)
    out: list[dict] = []
    for segment in parsed.get("segments", []) or []:
        name = segment.get("name")
        if not name:
            continue
        for stroke in segment.get("strokes", []) or []:
            item = dict(stroke)
            item["segment"] = name
            out.append(item)
    return out


def seed_spec_from_agent_json(parsed: dict, metadata_by_file: dict) -> tuple[dict, dict]:
    cornea: list[dict] = []
    background: list[dict] = []
    for stroke in _stroke_items(parsed):
        segment = str(stroke.get("segment", stroke.get("label", ""))).strip().lower()
        if segment not in ("cornea", "background"):
            continue
        radius = radius_voxels(stroke.get("radius_voxels", stroke.get("radius")), segment)
        points: list[list[int]] = []
        ijk_points = stroke.get("points_ijk", stroke.get("ijk_points"))
        if isinstance(ijk_points, list):
            for p in ijk_points:
                ijk = _point_ijk(p)
                if ijk:
                    points.append(ijk)
        else:
            file_name = (
                stroke.get("image_file") or stroke.get("file_name")
                or stroke.get("preview_file") or stroke.get("image")
            )
            if not file_name:
                raise ValueError("Agent stroke is missing image_file")
            metadata = metadata_by_file.get(file_name)
            if metadata is None:
                raise ValueError(f"Agent referenced unknown preview file: {file_name}")
            pts = stroke.get("points_px", stroke.get("pixel_points", stroke.get("points")))
            if not isinstance(pts, list):
                raise ValueError(f"Agent stroke for {file_name} is missing points_px")
            for p in pts:
                pair = _point_pair(p)
                if pair:
                    points.append(px_to_ijk(metadata, pair[0], pair[1]))
        if not points:
            continue
        normalized = {"points_ijk": points, "radius_voxels": radius}
        (background if segment == "background" else cornea).append(normalized)

    if not cornea or not background:
        raise ValueError(
            f"Agent paint must include both cornea and background strokes. "
            f"Got cornea={len(cornea)}, background={len(background)}."
        )
    stats = {"cornea_stroke_count": len(cornea), "background_stroke_count": len(background)}
    seed_spec = {
        "segments": [
            {"name": "background", "color": [0.05, 0.05, 0.05], "seeds": [], "strokes": background},
            {"name": "cornea", "color": [0.1, 0.7, 1.0], "seeds": [], "strokes": cornea},
        ]
    }
    return seed_spec, stats


def current_case_info(case_id: str) -> dict:
    cid = safe_case_id(case_id)
    root = case_root(cid)
    return {
        "case_id": cid,
        "root": str(root),
        "input_dir": str(root / "input"),
        "segmentation_dir": str(root / "segmentation"),
        "review_dir": str(root / "review"),
        "seed_json": str(case_seed_json(cid)),
        "output_seg": str(case_output_seg(cid)),
        "qa_json": str(case_qa_json(cid)),
        "scene": str(case_scene(cid)),
        "manifest": read_manifest(cid),
    }
