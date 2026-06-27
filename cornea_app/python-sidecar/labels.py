"""Canonical segmentation labelmap — single source of truth for display + export.

Final label convention (nnU-Net target): 0=background, 1=cornea, 2=scar.

A case's labelmap is the corrected labelmap (<case>_corrected.nii.gz, already
0/1/2) written by SAM2 and refined by the expert correction round-trip. The
niivue overlay, the nnU-Net export, and the correction drawing all go through
best_labelmap_nnunet so they can never disagree.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib

import orchestration as orch

NNUNET_LABELS = {"background": 0, "cornea": 1, "scar": 2}


def corrected_path(case_id: str) -> Path:
    cid = orch.safe_case_id(case_id)
    return orch.case_root(cid) / "segmentation" / f"{cid}_corrected.nii.gz"


def best_labelmap_nnunet(case_id: str) -> tuple[np.ndarray | None, str | None]:
    """Return (labelmap_ijk in {0,1,2}, source) — the corrected labelmap, or None."""
    cp = corrected_path(case_id)
    if cp.exists():
        arr = np.rint(np.asarray(nib.load(str(cp)).dataobj)).astype(np.uint8)
        return arr, "corrected"
    return None, None


def write_label_nifti(arr_ijk: np.ndarray, base_nifti: Path, dst: Path) -> Path:
    """Write the labelmap, stamped with the base volume's affine. Atomic (write to a
    temp then os.replace) so a crash mid-write can't corrupt the canonical labelmap, and
    shape-guarded so a stale label from a different capture can't be stamped with the
    wrong geometry."""
    import os
    base = nib.load(str(base_nifti))
    arr = np.ascontiguousarray(arr_ijk.astype(np.uint8))
    if tuple(base.shape[:3]) != tuple(arr.shape[:3]):
        raise ValueError(
            f"Label shape {tuple(arr.shape[:3])} != base volume shape {tuple(base.shape[:3])}; "
            f"refusing to stamp a mismatched affine onto {dst.name}.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name("_tmp_" + dst.name)             # keeps .nii.gz so nibabel gzips
    nib.save(nib.Nifti1Image(arr, base.affine), str(tmp))
    os.replace(str(tmp), str(dst))                      # atomic rename on the same filesystem
    return dst


def write_display_labelmap(arr_ijk: np.ndarray, density_vol_ijk, base_nifti: Path, dst: Path) -> Path:
    """DISPLAY-ONLY labelmap for the 3D viewer: 0=bg, 1=cornea, and scar split into reflectivity
    tiers 2 (diffuse) / 3 (moderate) / 4 (dense) so the overlay shows density instead of one flat red.
    The CANONICAL training label stays 0/1/2 (write_label_nifti) — this is a separate file the niivue
    overlay loads. Falls back to plain scar=2 when there is no scar or no density volume."""
    import scar as scar_mod
    arr = np.rint(np.asarray(arr_ijk)).astype(np.uint8)
    out = np.where(arr == 1, 1, 0).astype(np.uint8)         # cornea
    scar_mask = arr == 2
    if scar_mask.any():
        if density_vol_ijk is not None:
            tier, _ = scar_mod.density_tiers_absolute(scar_mask, np.asarray(density_vol_ijk), arr == 1)
            out[scar_mask] = (tier[scar_mask] + 1).astype(np.uint8)   # tiers 1/2/3 → labels 2/3/4
        else:
            out[scar_mask] = 4   # no density → render as solid (dense) red, not the faint diffuse tier
    return write_label_nifti(out, base_nifti, dst)


def labelmap_counts(arr_ijk: np.ndarray, spacing_mm3: float | None = None) -> dict:
    """Per-class voxel counts (and optional volume) for QA, keyed by class name."""
    out: dict[str, dict] = {}
    for name, value in NNUNET_LABELS.items():
        n = int((arr_ijk == value).sum())
        entry: dict = {"voxel_count": n}
        if spacing_mm3 is not None:
            entry["volume_mm3"] = round(n * spacing_mm3, 4)
        out[name] = entry
    return out
