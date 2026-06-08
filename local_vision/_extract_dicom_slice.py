#!/usr/bin/env python3
"""Pull a raw 2D slice from the 3D OCT DICOM volume (no overlays)."""
import sys
import cv2
import numpy as np
import pydicom

dcm_path = sys.argv[1]
ds = pydicom.dcmread(dcm_path)
vol = ds.pixel_array  # (frames, H, W) or similar
print("volume shape:", vol.shape, "dtype:", vol.dtype)

vol = np.asarray(vol)
if vol.ndim == 2:
    vol = vol[None, ...]

# Take a middle sagittal-ish slice through the cornea.
mid = vol.shape[0] // 2
sl = vol[mid].astype(np.float32)

# Normalize to 8-bit for the segmenter.
lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
sl = np.clip((sl - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)

out = "/home/zhuojian/Desktop/Integration/local_vision/real_slice.png"
cv2.imwrite(out, sl)
print("wrote", out, "slice index", mid, "shape", sl.shape)
