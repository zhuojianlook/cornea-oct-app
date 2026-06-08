"""Stage 4 — scar detection as an optional 3rd class inside cornea.

Scar is a sub-region of cornea (may be absent). We add a `scar` segment to
seeds.json *inside* the current cornea segmentation, re-grow into a 3-class
labelmap (background / cornea / scar), and report metrics. A deterministic
heuristic is provided for testing without a vision model.

Label convention in the grown .seg.nrrd follows seeds.json segment order:
label = index + 1. The exported nnUNet labelmap remaps to 0/1/2 (see export.py).
"""
from __future__ import annotations

import numpy as np

import orchestration as orch


def cornea_label(seed_spec: dict) -> int | None:
    for idx, seg in enumerate(seed_spec.get("segments", [])):
        if seg.get("name") == "cornea":
            return idx + 1
    return None


def _read_seg_labelmap(case_id: str) -> np.ndarray | None:
    import nrrd

    seg_path = orch.case_output_seg(case_id)
    if not seg_path.exists():
        return None
    data, _ = nrrd.read(str(seg_path))
    data = np.asarray(data)
    if data.ndim == 4:
        data = data.max(axis=int(np.argmin(data.shape)))
    return data  # (i, j, k)


def heuristic_scar_seed(case_id: str, fraction: float = 0.35) -> dict:
    """Place a scar seed blob in the centre of the cornea segmentation.

    Returns the scar segment dict to merge into seeds.json. Raises if there is
    no cornea segmentation to anchor on.
    """
    seed_spec = orch.read_json(orch.case_seed_json(case_id))
    label = cornea_label(seed_spec)
    seg = _read_seg_labelmap(case_id)
    if label is None or seg is None:
        raise ValueError("No cornea segmentation found. Run Grow from Seeds first.")
    cornea_vox = np.argwhere(seg == label)
    if len(cornea_vox) == 0:
        raise ValueError("Cornea segmentation is empty.")
    centroid = cornea_vox.mean(axis=0)
    extent = cornea_vox.max(axis=0) - cornea_vox.min(axis=0) + 1
    radius = np.maximum((extent * fraction / 2).astype(int), 2)
    return {
        "name": "scar",
        "color": [1.0, 0.55, 0.1],
        "seeds": [{"ijk": [int(centroid[0]), int(centroid[1]), int(centroid[2])],
                   "radius_voxels": [int(radius[0]), int(radius[1]), int(radius[2])]}],
        "strokes": [],
    }


def merge_scar_segment(seed_spec: dict, scar_segment: dict) -> dict:
    """Return seed_spec with the scar segment added/replaced (bg+cornea kept)."""
    segments = [s for s in seed_spec.get("segments", []) if s.get("name") != "scar"]
    segments.append(scar_segment)
    return {"segments": segments}


def scar_segment_from_agent_json(parsed: dict, metadata_by_file: dict) -> dict | None:
    """Build a scar segment from a vision model's strokes. None if no scar."""
    if isinstance(parsed, dict) and parsed.get("scar_present") is False:
        return None
    strokes = []
    for stroke in orch._stroke_items(parsed):
        seg = str(stroke.get("segment", stroke.get("label", ""))).strip().lower()
        if seg != "scar":
            continue
        radius = orch.radius_voxels(stroke.get("radius_voxels", stroke.get("radius")), "scar")
        points: list[list[int]] = []
        ijk_points = stroke.get("points_ijk") or stroke.get("ijk_points")
        if isinstance(ijk_points, list):
            for p in ijk_points:
                ijk = orch._point_ijk(p)
                if ijk:
                    points.append(ijk)
        else:
            fn = (stroke.get("image_file") or stroke.get("file_name")
                  or stroke.get("preview_file") or stroke.get("image"))
            meta = metadata_by_file.get(fn) if fn else None
            pts = stroke.get("points_px") or stroke.get("pixel_points") or stroke.get("points")
            if meta and isinstance(pts, list):
                for p in pts:
                    pair = orch._point_pair(p)
                    if pair:
                        points.append(orch.px_to_ijk(meta, pair[0], pair[1]))
        if points:
            strokes.append({"points_ijk": points, "radius_voxels": radius})
    if not strokes:
        return None
    return {"name": "scar", "color": [1.0, 0.55, 0.1], "seeds": [], "strokes": strokes}


def scar_metrics(case_id: str, seed_spec: dict) -> dict:
    """Compute scar metrics from the (re-grown) labelmap."""
    seg = _read_seg_labelmap(case_id)
    if seg is None:
        return {"scar_present": False}
    label_of = {s.get("name"): i + 1 for i, s in enumerate(seed_spec.get("segments", []))}
    scar_l = label_of.get("scar")
    cornea_l = label_of.get("cornea")
    scar_vox = int((seg == scar_l).sum()) if scar_l else 0
    cornea_vox = int((seg == cornea_l).sum()) if cornea_l else 0
    corneal_tissue = scar_vox + cornea_vox
    present = scar_vox > 0
    metrics = {
        "scar_present": present,
        "scar_voxels": scar_vox,
        "cornea_voxels": cornea_vox,
        "scar_fraction_of_cornea": round(scar_vox / corneal_tissue, 4) if corneal_tissue else 0.0,
    }
    if present:
        coords = np.argwhere(seg == scar_l)
        metrics["scar_bounds_ijk"] = {
            "min": coords.min(0).tolist(),
            "max": coords.max(0).tolist(),
        }
    return metrics
