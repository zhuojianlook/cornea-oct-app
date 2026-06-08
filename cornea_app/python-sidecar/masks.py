"""Seed ↔ niivue drawing-layer conversion.

The interactive editor uses niivue's voxel "drawing" bitmap. We round-trip it:
  seeds.json  →  label NIfTI (same grid as volume.nii.gz)  →  niivue pen edits
              →  edited label NIfTI  →  seeds.json

Pen label convention (drawing voxels):  1 = cornea, 2 = background, 3 = scar.
preview_io is pure numpy, so we import it directly from slicer_bridge/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib

import settings

sys.path.insert(0, str(settings.SLICER_BRIDGE_DIR))
import preview_io  # noqa: E402  (pure-numpy; safe outside Slicer)

# Pen label ↔ segment name. Order matters for overlap: later wins on the canvas.
PEN_BY_NAME = {"background": 2, "cornea": 1, "scar": 3}
NAME_BY_PEN = {1: "cornea", 2: "background", 3: "scar"}
PAINT_ORDER = ["background", "cornea", "scar"]  # scar overrides cornea overrides bg

SEGMENT_COLORS = {
    "background": [0.05, 0.05, 0.05],
    "cornea": [0.1, 0.7, 1.0],
    "scar": [1.0, 0.55, 0.1],
}
MAX_SEEDS_PER_CLASS = 6000  # cap seed JSON size when converting a dense drawing


def build_seed_drawing(volume_nifti: Path, seed_spec: dict, dst: Path) -> Path:
    """Render seeds.json into a label NIfTI on the base volume's exact grid."""
    base = nib.load(str(volume_nifti))
    shape_ijk = base.shape[:3]
    shape_kji = (shape_ijk[2], shape_ijk[1], shape_ijk[0])
    masks = preview_io.seed_masks_from_spec(shape_kji, seed_spec)  # name -> mask_kji
    label_kji = np.zeros(shape_kji, dtype=np.uint8)
    for name in PAINT_ORDER:
        mask = masks.get(name)
        if mask is not None:
            label_kji[mask > 0] = PEN_BY_NAME[name]
    label_ijk = np.ascontiguousarray(label_kji.transpose(2, 1, 0))
    out = nib.Nifti1Image(label_ijk, base.affine)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return dst


def seeds_from_drawing(drawing_nifti: Path) -> tuple[dict, dict]:
    """Convert an edited drawing label NIfTI back into a seed spec.

    Each labelled voxel becomes a unit seed; classes are subsampled to keep the
    JSON bounded. Returns (seed_spec, counts).
    """
    img = nib.load(str(drawing_nifti))
    data = np.asarray(img.dataobj)
    data = np.rint(data).astype(np.int32)  # (i, j, k)

    segments = []
    counts: dict[str, int] = {}
    for pen, name in NAME_BY_PEN.items():
        voxels = np.argwhere(data == pen)
        total = int(len(voxels))
        counts[name] = total
        if total == 0:
            continue
        if total > MAX_SEEDS_PER_CLASS:
            stride = int(np.ceil(total / MAX_SEEDS_PER_CLASS))
            voxels = voxels[::stride]
        seeds = [{"ijk": [int(i), int(j), int(k)], "radius_voxels": [1, 1, 1]} for i, j, k in voxels]
        segments.append({
            "name": name,
            "color": SEGMENT_COLORS.get(name, [0.5, 0.5, 0.5]),
            "seeds": seeds,
            "strokes": [],
        })
    # Keep a deterministic order (background, cornea, scar) for downstream tools.
    segments.sort(key=lambda s: PAINT_ORDER.index(s["name"]) if s["name"] in PAINT_ORDER else 99)
    return {"segments": segments}, counts
