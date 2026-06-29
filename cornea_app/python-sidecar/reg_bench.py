"""Shared benchmark harness for repeat-scan registration experiments (CS001-OD).

Every strategy is scored identically here so results are comparable. A strategy is
a function `register(fvol, flab, mvol, mlab) -> sitk.Transform` that aligns the MOVING
scan onto the FIXED (reference) grid — i.e. the transform you pass to sitk.Resample to
pull `moving` into `fixed` space (SimpleITK fixed→moving point mapping convention).

Usage in a one-off script:

    import reg_bench as rb
    def register(fvol, flab, mvol, mlab):
        return rb.identity()          # ← your strategy here
    print(rb.report_json(register, name="my_strategy"))

Metrics (mean over the 4 moving scans v2..v5 aligned to v1, plus volume CV over all 5):
  cornea_dice      Dice of cornea (label>=1) after alignment — did the eye align?
  scar_dice        Dice of scar (==2) after alignment — did the scars overlap?
  ref_scar_covered fraction of REF scar voxels covered by the warped moving scar
  moving_scar_kept fraction of warped moving scar landing inside REF scar
  cornea_vox_ratio warped-moving cornea voxels / ref cornea voxels (≈1 = no collapse;
                   a degenerate transform that shrinks/explodes the volume is flagged)
  scar_volume_cv%  CV of scar volume across ref + 4 warped scans (volume reproducibility)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

CASES_ROOT = Path("/home/zhuojian/Desktop/Integration/cases")
REF = "case_cs001_od_v1"
MOVING = ["case_cs001_od_v2", "case_cs001_od_v3", "case_cs001_od_v4", "case_cs001_od_v5"]
ALL = [REF] + MOVING
SCAR, CORNEA_MIN = 2, 1


def vol_path(cid: str) -> Path:
    return CASES_ROOT / cid / "previews" / "volume.nii.gz"


def lab_path(cid: str) -> Path:
    return CASES_ROOT / cid / "segmentation" / f"{cid}_corrected.nii.gz"


def canon(img: sitk.Image) -> sitk.Image:
    """Reset origin to 0 and direction to identity so masks/volumes share a frame."""
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img


_cache: dict[str, tuple[sitk.Image, sitk.Image]] = {}


def load(cid: str) -> tuple[sitk.Image, sitk.Image]:
    """Return (volume float32, label uint8), both canonicalised. Cached."""
    if cid not in _cache:
        vol = canon(sitk.ReadImage(str(vol_path(cid)), sitk.sitkFloat32))
        lab = canon(sitk.ReadImage(str(lab_path(cid)), sitk.sitkUInt8))
        _cache[cid] = (vol, lab)
    return _cache[cid]


def identity() -> sitk.Transform:
    return sitk.Euler3DTransform()


def smoothed_mask(lab_img: sitk.Image, scar_only: bool, sigma: float = 1.5) -> sitk.Image:
    """Gaussian-smoothed float mask of scar (==2) or cornea (>=1) — for mask registration."""
    if scar_only:
        m = sitk.BinaryThreshold(lab_img, SCAR, SCAR, 1, 0)
    else:
        m = sitk.BinaryThreshold(lab_img, CORNEA_MIN, 255, 1, 0)
    return sitk.SmoothingRecursiveGaussian(sitk.Cast(m, sitk.sitkFloat32), sigma)


def warp_label(moving_lab: sitk.Image, fixed_lab: sitk.Image, tx: sitk.Transform) -> np.ndarray:
    out = sitk.Resample(moving_lab, fixed_lab, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)  # (z,y,x)


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum()); s = int(a.sum()) + int(b.sum())
    return float(2 * inter / s) if s else 0.0


def _frac(part: np.ndarray, whole: np.ndarray) -> float:
    w = int(whole.sum())
    return float(np.logical_and(part, whole).sum()) / w if w else 0.0


def evaluate(register_fn, name: str = "strategy") -> dict:
    """Align each moving scan to REF with register_fn, measure overlap + reproducibility."""
    fvol, flab = load(REF)
    flab_arr = sitk.GetArrayFromImage(flab)
    ref_cornea = flab_arr >= CORNEA_MIN
    ref_scar = flab_arr == SCAR
    sp = flab.GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    ref_cornea_vox = int(ref_cornea.sum())
    ref_scar_vol = float(ref_scar.sum() * vmm3)

    per = []
    scar_vols = [round(ref_scar_vol, 4)]
    t0 = time.time()
    for cid in MOVING:
        mvol, mlab = load(cid)
        row = {"case": cid}
        try:
            tx = register_fn(fvol, flab, mvol, mlab)
            warped = warp_label(mlab, flab, tx)
            wcornea = warped >= CORNEA_MIN
            wscar = warped == SCAR
            row["cornea_dice"] = round(_dice(ref_cornea, wcornea), 3)
            row["scar_dice"] = round(_dice(ref_scar, wscar), 3)
            row["ref_scar_covered"] = round(_frac(ref_scar, wscar), 3)
            row["moving_scar_kept"] = round(_frac(wscar, ref_scar), 3)
            row["cornea_vox_ratio"] = round(int(wcornea.sum()) / ref_cornea_vox, 3) if ref_cornea_vox else 0.0
            scar_vols.append(round(float(wscar.sum() * vmm3), 4))
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        per.append(row)
    elapsed = round(time.time() - t0, 1)

    ok = [r for r in per if "error" not in r]
    mean = lambda k: round(float(np.mean([r[k] for r in ok])), 3) if ok else None  # noqa: E731
    vols = np.array(scar_vols, dtype=float)
    # Sample SD (ddof=1) — the correct test-retest dispersion estimator, matching every production path
    # (consensus.py, scar_bench.py, observer_analysis.py). ddof=1 is undefined for n=1, so guard it.
    cv = round(float(vols.std(ddof=1) / vols.mean() * 100), 2) if (len(vols) > 1 and vols.mean()) else 0.0
    return {
        "name": name,
        "n_ok": len(ok),
        "mean_cornea_dice": mean("cornea_dice"),
        "mean_scar_dice": mean("scar_dice"),
        "mean_ref_scar_covered": mean("ref_scar_covered"),
        "mean_moving_scar_kept": mean("moving_scar_kept"),
        "mean_cornea_vox_ratio": mean("cornea_vox_ratio"),
        "scar_volume_cv_percent": cv,
        "scar_volumes_mm3": scar_vols,
        "per_scan": per,
        "seconds": elapsed,
    }


def report_json(register_fn, name: str = "strategy") -> str:
    return json.dumps(evaluate(register_fn, name), indent=2)


if __name__ == "__main__":
    # Baseline sanity check: identity (no registration at all).
    print(report_json(lambda fvol, flab, mvol, mlab: identity(), "identity"))
