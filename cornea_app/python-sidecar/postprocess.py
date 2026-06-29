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
import scar as scar_mod  # absolute density-tier helper (shared with the 3D display labelmap)

# nnUNet labels in the canonical labelmap.
BG, CORNEA, SCAR = 0, 1, 2


def _spacing(affine: np.ndarray) -> list[float]:
    return [float(np.linalg.norm(affine[:3, i])) for i in range(3)]


def render_seg_previews(volume_nifti: Path, arr_ijk: np.ndarray, out_dir: Path,
                        density_vol: np.ndarray | None = None,
                        dense_rotated: bool = False, density_from_self: bool = False) -> int:
    """Render segmentation overlay PNGs from a labelmap, in-process (numpy).

    When `density_vol` is given, the scar is shown in 3 reflectivity tiers
    (diffuse → dense) instead of flat red, so the mix of opacities is visible.
    `density_from_self=True` derives the tiers from THIS volume's own reflectivity (used for the
    consensus/step-9 previews, which have no separate raw volume to pass).

    `dense_rotated=True` renders EVERY slice with the same rotation/size as the grayscale
    context previews — used for the gallery's 3rd before/after panel so the overlay scrubs
    in lock-step with raw/corrected (display-only; the clickable segmentation group stays
    sparse + unrotated, so scar-edit/hint coordinates are unaffected)."""
    img = nib.load(str(volume_nifti))
    if density_vol is None and density_from_self:
        density_vol = np.asarray(img.dataobj)   # IJK reflectivity (same grid as arr_ijk) → density tiers
    vol_kji = np.ascontiguousarray(np.asarray(img.dataobj).transpose(2, 1, 0))
    scar = arr_ijk == SCAR
    masks_by_name = {
        "cornea": np.ascontiguousarray((arr_ijk == CORNEA).transpose(2, 1, 0)),
        "background": np.ascontiguousarray((arr_ijk == BG).transpose(2, 1, 0)),
    }
    if density_vol is not None and scar.any():
        # Absolute, cross-eye-comparable tiers (scar reflectivity vs this eye's normal-cornea median).
        tier, _ = scar_mod.density_tiers_absolute(scar, density_vol, arr_ijk == CORNEA)
        masks_by_name["scar_diffuse"] = np.ascontiguousarray((tier == 1).transpose(2, 1, 0))
        masks_by_name["scar_mod"] = np.ascontiguousarray((tier == 2).transpose(2, 1, 0))
        masks_by_name["scar"] = np.ascontiguousarray((tier >= 3).transpose(2, 1, 0))
    else:
        masks_by_name["scar"] = np.ascontiguousarray(scar.transpose(2, 1, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    if dense_rotated:
        preview_io.save_previews(
            vol_kji, masks_by_name, str(out_dir), "segmentation",
            spacing_ijk=_spacing(img.affine),
            max_slices_per_orientation={"axial": 100000, "coronal": 100000, "sagittal": 100000},
            max_dim=512, rotate={"sagittal": -1, "axial": 2}, compress_level=1)
    else:
        # Rotate + size-cap IDENTICALLY to the grayscale context previews so the segmentation
        # overlay is geometrically the SAME as the before/after slices (clicks are made rotation-
        # aware via the manifest's rotate_k, so scar edit/hint coordinates still map correctly).
        preview_io.save_previews(vol_kji, masks_by_name, str(out_dir), "segmentation",
                                 spacing_ijk=_spacing(img.affine), max_slices_per_orientation=9,
                                 max_dim=512, rotate={"sagittal": -1, "axial": 2})
    return int(masks_by_name["cornea"].sum())


def render_context_previews(volume_nifti: Path, out_dir: Path) -> int:
    """Render plain grayscale slice PNGs (no overlay) of the raw volume so the 2D
    gallery can show the OCT before segmentation."""
    img = nib.load(str(volume_nifti))
    vol_kji = np.ascontiguousarray(np.asarray(img.dataobj).transpose(2, 1, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    # DENSE in ALL three orientations so scrubbing the raw volume never skips a slice
    # (sagittal/coronal used to be sampled to 16, so dragging the slider jumped ~30 slices
    # at a time — see the OCT smoothness review, which scrubs every sagittal slice). The
    # PNGs are served lazily as URLs (one per request), so a dense group stays cheap on the
    # client. Size-capped per slice so each PNG is small.
    # Rotate for review so the cornea surface sits on top: sagittal 90° clockwise (k=-1),
    # axial 180° (k=2). Display-only — segmentation previews are NOT rotated, so the scar
    # edit / hint coordinate mapping is untouched.
    preview_io.save_previews(vol_kji, {}, str(out_dir), "context",
                             spacing_ijk=_spacing(img.affine),
                             max_slices_per_orientation={"axial": 100000, "coronal": 100000, "sagittal": 100000},
                             max_dim=512, rotate={"sagittal": -1, "axial": 2})
    return int(vol_kji.size)
