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
# NRRD also allows the abbreviated 3-letter anatomical space codes (e.g. "LPS", "RAS", "RAI"),
# which pynrrd returns verbatim — expand each letter to its full token before parsing.
_ABBREV = {"r": "right", "l": "left", "a": "anterior", "p": "posterior", "s": "superior", "i": "inferior"}


def _lps_to_ras_world(space: str | None) -> np.ndarray:
    """World-frame transform mapping the NRRD space frame to RAS (R+,A+,S+).

    Builds a real permutation+sign matrix: each NRRD world axis is placed into
    its correct RAS row (right/left->x, anterior/posterior->y, superior/inferior
    ->z) with the sign that points it the RAS-positive way. This is correct even
    when the space axes are permuted (e.g. "posterior-left-superior"); a plain
    diagonal could not swap axes. Defaults to LPS→RAS = diag(-1,-1,1) when the
    space string is missing; an unrecognised string is rejected rather than
    silently treated as LPS.
    """
    if not space:
        return np.diag([-1.0, -1.0, 1.0, 1.0])
    tokens = space.replace("-", " ").lower().split()
    if len(tokens) == 1 and len(tokens[0]) == 3 and all(ch in _ABBREV for ch in tokens[0]):
        tokens = [_ABBREV[ch] for ch in tokens[0]]   # abbreviated code, e.g. "LPS" -> [left, posterior, superior]
    if len(tokens) != 3 or not all(t in _AXIS_SIGN for t in tokens):
        raise ValueError(f"Unrecognised NRRD space {space!r}; expected 3 "
                         "right/left/anterior/posterior/superior/inferior axes")
    if len({_AXIS_SIGN[t][0] for t in tokens}) != 3:
        raise ValueError(f"Degenerate NRRD space {space!r}; axes must be distinct")
    m = np.zeros((4, 4))
    m[3, 3] = 1.0
    for world_pos, tok in enumerate(tokens):
        axis, sign = _AXIS_SIGN[tok]
        # world axis `world_pos` maps to RAS row `axis`; `sign` tells whether its
        # +direction already points the RAS-positive way.
        m[_RAS_INDEX[axis], world_pos] = float(sign)
    return m


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
    return _lps_to_ras_world(header.get("space")) @ aff


def nrrd_to_nifti(src: Path, dst: Path) -> Path:
    import nrrd  # local import; only needed for NRRD inputs

    data, header = nrrd.read(str(src))
    affine = _nrrd_affine(header)
    # niivue/nibabel handle float32 fine; keep dtype but ensure C-order array.
    img = nib.Nifti1Image(np.ascontiguousarray(data), affine)
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
