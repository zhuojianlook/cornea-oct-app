"""Render segmentation + context slice previews in-process (pure numpy).

Uses slicer_bridge/preview_io (pure numpy, no Slicer) so the 2D gallery works
without WebGL and without a Slicer pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib

import settings

sys.path.insert(0, str(settings.SLICER_BRIDGE_DIR))
import preview_io  # noqa: E402  (pure-numpy; safe outside Slicer)

# nnUNet labels in the canonical labelmap.
BG, CORNEA, SCAR = 0, 1, 2


def _spacing(affine: np.ndarray) -> list[float]:
    return [float(np.linalg.norm(affine[:3, i])) for i in range(3)]


def render_seg_previews(volume_nifti: Path, arr_ijk: np.ndarray, out_dir: Path,
                        density_vol: np.ndarray | None = None) -> int:
    """Render segmentation overlay PNGs from a labelmap, in-process (numpy).

    When `density_vol` is given, the scar is shown in 3 reflectivity tiers
    (diffuse → dense) instead of flat red, so the mix of opacities is visible."""
    img = nib.load(str(volume_nifti))
    vol_kji = np.ascontiguousarray(np.asarray(img.dataobj).transpose(2, 1, 0))
    scar = arr_ijk == SCAR
    masks_by_name = {
        "cornea": np.ascontiguousarray((arr_ijk == CORNEA).transpose(2, 1, 0)),
        "background": np.ascontiguousarray((arr_ijk == BG).transpose(2, 1, 0)),
    }
    if density_vol is not None and scar.any():
        vals = density_vol[scar]
        lo, hi = np.quantile(vals, 1 / 3), np.quantile(vals, 2 / 3)
        masks_by_name["scar_diffuse"] = np.ascontiguousarray((scar & (density_vol < lo)).transpose(2, 1, 0))
        masks_by_name["scar_mod"] = np.ascontiguousarray((scar & (density_vol >= lo) & (density_vol < hi)).transpose(2, 1, 0))
        masks_by_name["scar"] = np.ascontiguousarray((scar & (density_vol >= hi)).transpose(2, 1, 0))
    else:
        masks_by_name["scar"] = np.ascontiguousarray(scar.transpose(2, 1, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_io.save_previews(vol_kji, masks_by_name, str(out_dir), "segmentation",
                             spacing_ijk=_spacing(img.affine), max_slices_per_orientation=9)
    return int(masks_by_name["cornea"].sum())


def render_context_previews(volume_nifti: Path, out_dir: Path) -> int:
    """Render plain grayscale slice PNGs (no overlay) of the raw volume so the 2D
    gallery can show the OCT before segmentation."""
    img = nib.load(str(volume_nifti))
    vol_kji = np.ascontiguousarray(np.asarray(img.dataobj).transpose(2, 1, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    # Dense AXIAL (= the OCT B-scan frames) so the user can actually scrub every frame;
    # coronal/sagittal stay sampled. Size-capped so the (now ~100-slice) payload stays small.
    preview_io.save_previews(vol_kji, {}, str(out_dir), "context",
                             spacing_ijk=_spacing(img.affine),
                             max_slices_per_orientation={"axial": 100000, "coronal": 16, "sagittal": 16},
                             max_dim=512)
    return int(vol_kji.size)
