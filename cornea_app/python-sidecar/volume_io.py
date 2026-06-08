"""Volume conversion for the niivue viewer.

niivue reliably reads NIfTI. We convert the input OCT volume (NRRD/NIfTI) to a
RAS NIfTI, and later the Slicer `.seg.nrrd` labelmap to a matching-affine label
NIfTI, so the grayscale and segmentation overlays share one coordinate frame
(plan Risk B).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib

# NRRD "space" axis tokens → sign to convert that world axis to RAS (R+,A+,S+).
_AXIS_SIGN = {
    "right": ("x", +1), "left": ("x", -1),
    "anterior": ("y", +1), "posterior": ("y", -1),
    "superior": ("z", +1), "inferior": ("z", -1),
}
_RAS_INDEX = {"x": 0, "y": 1, "z": 2}


def _lps_to_ras_diag(space: str | None) -> np.ndarray:
    """Diagonal sign matrix mapping the NRRD world frame to RAS.

    Defaults to LPS→RAS = diag(-1,-1,1) when the space string is missing/odd.
    """
    signs = [-1.0, -1.0, 1.0]
    if space:
        tokens = space.replace("-", " ").lower().split()
        if len(tokens) == 3 and all(t in _AXIS_SIGN for t in tokens):
            tmp = [1.0, 1.0, 1.0]
            for world_pos, tok in enumerate(tokens):
                axis, sign = _AXIS_SIGN[tok]
                # world axis `world_pos` corresponds to RAS axis `axis`; its sign
                # tells whether +voxel-direction already points the RAS-positive way.
                tmp[world_pos] = float(sign)
            signs = tmp
    return np.diag(signs + [1.0])


def _nrrd_affine(header: dict) -> np.ndarray:
    sd = header.get("space directions")
    origin = np.array(header.get("space origin", [0.0, 0.0, 0.0]), dtype=float)
    if sd is None:
        aff = np.eye(4)
        aff[:3, 3] = origin
        return aff
    sd = np.array(sd, dtype=float)  # rows = per-voxel-axis world vectors
    aff = np.eye(4)
    aff[:3, :3] = sd.T  # columns = axis vectors
    aff[:3, 3] = origin
    return _lps_to_ras_diag(header.get("space")) @ aff


def nrrd_to_nifti(src: Path, dst: Path) -> Path:
    import nrrd  # local import; only needed for NRRD inputs

    data, header = nrrd.read(str(src))
    affine = _nrrd_affine(header)
    # niivue/nibabel handle float32 fine; keep dtype but ensure C-order array.
    img = nib.Nifti1Image(np.ascontiguousarray(data), affine)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(dst))
    return dst


def seg_to_label_nifti(seg_src: Path, base_nifti: Path, dst: Path) -> Path:
    """Convert a Slicer .seg.nrrd labelmap to an integer-label NIfTI on the base
    volume's grid (so it overlays the grayscale exactly).

    Slicer writes the segmentation as an NRRD whose voxels already hold integer
    segment label values; we just re-grid them onto the base volume's affine.
    Handles a possible leading 'list' axis (4D) by collapsing to the max label.
    """
    import nrrd

    data, _header = nrrd.read(str(seg_src))
    data = np.asarray(data)
    if data.ndim == 4:
        # Slicer may prepend a layer axis (L, i, j, k) — collapse to one labelmap.
        axis = int(np.argmin(data.shape))
        data = data.max(axis=axis)
    base = nib.load(str(base_nifti))
    if data.shape != tuple(base.shape[:3]):
        # Seg grid should match the input volume; if a bounding-box crop shifted
        # it, fail loudly rather than mis-align the overlay.
        raise ValueError(f"Segmentation shape {data.shape} != volume shape {base.shape[:3]}")
    img = nib.Nifti1Image(np.ascontiguousarray(data.astype(np.uint8)), base.affine)
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(dst))
    return dst


def ensure_nifti(src: Path, dst: Path) -> Path:
    """Produce a niivue-ready NIfTI at `dst` from a volume at `src`.

    NRRD → convert; NIfTI → load+resave (canonicalises); other → error (DICOM is
    handled upstream via a Slicer subprocess that emits a .nii.gz first).
    """
    suffix = "".join(src.suffixes).lower()
    if suffix.endswith(".nrrd") or src.suffix.lower() == ".nrrd":
        return nrrd_to_nifti(src, dst)
    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        img = nib.load(str(src))
        dst.parent.mkdir(parents=True, exist_ok=True)
        nib.save(img, str(dst))
        return dst
    raise ValueError(f"Unsupported volume format for niivue: {src.name}")
