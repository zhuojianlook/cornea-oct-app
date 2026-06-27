"""Segmentation ↔ niivue drawing-layer conversion for interactive correction.

Round-trip: a 0/1/2 labelmap → niivue pen bitmap → expert edits → 0/1/2 labelmap.
Pen label convention (drawing voxels):  1 = cornea, 2 = background, 3 = scar.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib

PEN_BY_NAME = {"background": 2, "cornea": 1, "scar": 3}


def build_correction_drawing(base_nifti: Path, labelmap_ijk, dst: Path) -> Path:
    """Build a niivue drawing (pen labels) from a 0/1/2 labelmap for editing.

    cornea → pen 1, scar → pen 3; background/empty stays 0 so the editor adjusts
    the foreground classes (paint pen 2 to erase a region back to background).
    """
    base = nib.load(str(base_nifti))
    arr = np.asarray(labelmap_ijk)
    if tuple(arr.shape[:3]) != tuple(base.shape[:3]):
        raise ValueError(
            f"Label shape {tuple(arr.shape[:3])} != base volume shape {tuple(base.shape[:3])}; "
            f"refusing to stamp a mismatched affine onto {dst.name}.")
    pen = np.zeros(arr.shape, dtype=np.uint8)
    pen[arr == 1] = PEN_BY_NAME["cornea"]  # 1
    pen[arr == 2] = PEN_BY_NAME["scar"]    # 3
    out = nib.Nifti1Image(np.ascontiguousarray(pen), base.affine)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(dst))
    return dst


def corrected_labelmap_from_drawing(drawing_nifti: Path, base_nifti: Path, dst: Path):
    """Parse an edited niivue drawing into a canonical 0/1/2 labelmap.

    pen 1 (cornea) → 1, pen 3 (scar) → 2, everything else (pen 2 background and
    erase) → 0. Writes the labelmap and returns the array.
    """
    import labels  # local import avoids any import-order coupling at module load
    img = nib.load(str(drawing_nifti))
    pen = np.rint(np.asarray(img.dataobj)).astype(np.int32)
    out = np.zeros(pen.shape, dtype=np.uint8)
    out[pen == PEN_BY_NAME["cornea"]] = 1
    out[pen == PEN_BY_NAME["scar"]] = 2
    labels.write_label_nifti(out, base_nifti, dst)  # shape-guarded + atomic + base affine
    return out
