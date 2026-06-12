"""Volume preprocessing to make the cornea easier to segment.

OCT B-scans are speckly and low-contrast. We:
  1. Gaussian-blur within each B-scan plane (denoise speckle) — light blur along
     the scan axis so we don't smear across frames.
  2. Per-B-scan contrast stretch: clip to robust percentiles and rescale to
     0–255, which lifts the corneal band away from air/anterior-chamber noise.

The geometry is unchanged (same affine/shape), so seeds and Grow-from-Seeds stay
aligned. The whole pipeline (previews, paint, grow, viewer) runs on this output.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter


def preprocess_volume(
    src_nifti: Path,
    dst_nifti: Path,
    sigma=(2.0, 2.0, 0.8),
    clip_pct=(45.0, 99.5),
    gamma=1.4,
) -> Path:
    img = nib.load(str(src_nifti))
    data = np.asarray(img.dataobj).astype(np.float32)  # (i, j, k); B-scan = i-j plane

    blurred = gaussian_filter(data, sigma=sigma)

    # Per-frame (per-k B-scan) contrast mapping. The low clip sits near the
    # median: background (the majority of each B-scan) is crushed to black, so
    # speckle isn't lifted into the cornea's intensity range. Gamma>1 darkens
    # mid-tones further, leaving the bright corneal band well separated.
    out = np.empty_like(blurred)
    for k in range(blurred.shape[2]):
        sl = blurred[:, :, k]
        lo, hi = np.percentile(sl, clip_pct)
        if hi - lo < 1e-6:
            out[:, :, k] = 0.0
        else:
            norm = np.clip((sl - lo) / (hi - lo), 0.0, 1.0)
            out[:, :, k] = (norm ** gamma) * 255.0

    dst_nifti.parent.mkdir(parents=True, exist_ok=True)
    # uint8 (0–255): a quarter the size of float32, and grow/Slicer read it fine.
    nib.save(nib.Nifti1Image(np.rint(out).astype(np.uint8), img.affine), str(dst_nifti))
    return dst_nifti
