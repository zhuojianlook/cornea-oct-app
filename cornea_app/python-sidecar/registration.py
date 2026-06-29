"""Registration of repeat OCT scans (SimpleITK) for consensus.

Repeat acquisitions image slightly different, optically-warped patches of the same
eye, and the patient's eye shifts/rotates between scans. Aligning the *masks* alone
(scar or cornea) cannot localise the gross lateral offset on badly-shifted repeats —
the raw image content can. So we align the RAW VOLUMES with a guarded cascade:

  1. Isotropic-resample both volumes (the raw OCT grid is ~0.008x0.006x0.04 mm,
     highly anisotropic and sub-mm, which makes the optimizer's first physical step
     fling the moving image off the buffer). Isotropic spacing fixes this; the
     resulting transform lives in physical space and applies to the original grid.
  2. Multi-resolution rigid (Euler3D) by Mattes mutual information, kept only if it
     improves cornea overlap over identity (best-of-identity guard) — reliable on the
     grossly-misaligned scans, never worse than doing nothing on the aligned ones.
  3. A minimal, heavily-regularised BSpline (coarse 4^3 mesh, LBFGSB capped) masked to
     the cornea, to recover residual optical warp — kept only if it does not degrade
     cornea overlap (anti-overfit guard).

Benchmarked on CS001-OD's 5 repeats: this lifts mean scar Dice 0.369 -> 0.602 and
cornea 0.646 -> 0.804, rescuing the badly-shifted v4/v5 (scar 0.20/0.28 -> 0.58/0.59)
without degenerate volume collapse. A genuine partial-FOV floor (~0.58-0.65 on v4/v5)
remains — surfaced downstream as matched_fraction / low_correspondence, not forced away.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk

SCAR, CORNEA_MIN = 2, 1
ISO = 0.02    # isotropic spacing (mm) for the rigid intensity optimisation grid
ISO_B = 0.05  # coarser iso grid for the BSpline (keeps LBFGSB tractable on CPU)


def _canon(img: sitk.Image) -> sitk.Image:
    img.SetOrigin((0.0, 0.0, 0.0))
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img


def _read_vol(path: Path) -> sitk.Image:
    return _canon(sitk.ReadImage(str(path), sitk.sitkFloat32))


def _read_label(path: Path) -> sitk.Image:
    return _canon(sitk.ReadImage(str(path), sitk.sitkUInt8))


def identity() -> sitk.Transform:
    return sitk.Euler3DTransform()


# ── isotropic resampling (the load-bearing fix for the anisotropic OCT grid) ────
def _iso(img: sitk.Image, interp=sitk.sitkLinear, iso: float = ISO) -> sitk.Image:
    sz, sp = img.GetSize(), img.GetSpacing()
    nsz = [int(round(sz[i] * sp[i] / iso)) for i in range(3)]
    return sitk.Resample(img, nsz, sitk.Transform(), interp,
                         img.GetOrigin(), [iso] * 3, img.GetDirection(),
                         0.0, img.GetPixelID())


# ── cascade stages ──────────────────────────────────────────────────────────
def _rigid_intensity(fi: sitk.Image, mi: sitk.Image,
                     fixed_mask: sitk.Image | None = None) -> sitk.Transform:
    """Multi-resolution rigid (Euler3D = translation + rotation) via Mattes MI on raw iso
    intensities. When fixed_mask is given the metric is restricted to it (the cornea), so the
    alignment is driven by the cornea and the dark background is ignored."""
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    if fixed_mask is not None:
        R.SetMetricFixedMask(fixed_mask)   # cornea-only metric → align the cornea, not the background
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.8, minStep=1e-4, numberOfIterations=80,
        relaxationFactor=0.7, gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([4, 2, 1])
    R.SetSmoothingSigmasPerLevel([2.0, 1.0, 0.0])
    init = sitk.CenteredTransformInitializer(
        fi, mi, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R.SetInitialTransform(init, inPlace=False)
    return R.Execute(fi, mi)


def _minimal_bspline(fb: sitk.Image, mb: sitk.Image, rigid: sitk.Transform,
                     cornea_mask_b: sitk.Image) -> sitk.Transform:
    """Minimal, heavily-regularised BSpline refinement, masked to the cornea.

    Coarse 4^3 control mesh (stiff, few DOF), LBFGSB with a tight iteration/eval cap
    (no run-away), metric restricted to the dilated cornea. The rigid is the moving-
    initial transform so the BSpline optimises only the residual. Returns
    CompositeTransform([rigid, bspline])."""
    bspline = sitk.BSplineTransformInitializer(fb, [4, 4, 4], order=3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.10, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricFixedMask(cornea_mask_b)
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5, numberOfIterations=20,
        maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=60,
        costFunctionConvergenceFactor=1e9)
    R.SetShrinkFactorsPerLevel([1])
    R.SetSmoothingSigmasPerLevel([0.0])
    R.SetMovingInitialTransform(rigid)
    R.SetInitialTransform(bspline, inPlace=True)
    R.Execute(fb, mb)
    return sitk.CompositeTransform([rigid, bspline])


def _cornea_dice_iso(mlab: sitk.Image, flab_iso: sitk.Image,
                     ref_cm_iso: np.ndarray, tx: sitk.Transform) -> float:
    """Cornea Dice of the moving label warped by tx onto the iso reference grid."""
    w = sitk.Resample(_iso(mlab, interp=sitk.sitkNearestNeighbor), flab_iso, tx,
                      sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    a = sitk.GetArrayFromImage(w) >= CORNEA_MIN
    inter = int(np.logical_and(a, ref_cm_iso).sum())
    s = int(a.sum()) + int(ref_cm_iso.sum())
    return 2 * inter / s if s else 0.0


def align_transform(fixed_vol_path: Path, fixed_label_path: Path,
                    moving_vol_path: Path, moving_label_path: Path) -> tuple[sitk.Transform, str]:
    """Align the moving scan onto the fixed by a RIGID-ONLY, CORNEA-MASKED registration.

    Translation + rotation ONLY — the volumes are never WARPED/deformed, so the cornea (and the
    scar inside it) is repositioned rigidly and any residual scar-overlap gap reflects true
    test–retest variability rather than a deformation forced to inflate overlap. The Mattes-MI
    metric is restricted to the (dilated) reference cornea, so the dark background is ignored.
    Kept only if it beats identity on cornea overlap (best-of-identity guard), else identity.
    Returns (transform, mode∈{"rigid","identity"}); the transform pulls the moving image/label
    into the fixed grid (sitk fixed→moving convention)."""
    fvol, mvol = _read_vol(fixed_vol_path), _read_vol(moving_vol_path)
    flab, mlab = _read_label(fixed_label_path), _read_label(moving_label_path)

    fi, mi = _iso(fvol), _iso(mvol)
    flab_iso = _iso(flab, interp=sitk.sitkNearestNeighbor)
    ref_cm_iso = sitk.GetArrayFromImage(flab_iso) >= CORNEA_MIN
    # cornea-only metric mask on the iso fixed grid (dilated ~0.2 mm for context; dark bg excluded).
    cornea_mask_iso = sitk.Cast(
        sitk.BinaryDilate(sitk.BinaryThreshold(flab_iso, CORNEA_MIN, 255, 1, 0), [10, 10, 10]),
        sitk.sitkUInt8)

    ident = identity()
    d_id = _cornea_dice_iso(mlab, flab_iso, ref_cm_iso, ident)
    try:
        rigid = _rigid_intensity(fi, mi, fixed_mask=cornea_mask_iso)
        d_rig = _cornea_dice_iso(mlab, flab_iso, ref_cm_iso, rigid)
    except Exception:  # noqa: BLE001 — diverged optimiser → fall back to identity
        rigid, d_rig = ident, -1.0
    if d_rig > d_id:
        return rigid, "rigid"
    return ident, "identity"


# ── Stronger VOLUMETRIC registration (rigid → affine → denser cornea-driven BSpline) ──
# Additive: the production cascade above is untouched. This recovers scale/shear (affine) + more
# optical warp (denser BSpline) on the CORRECTED volumes, to tighten replicate alignment so the
# post-segmentation scar overlaps better. Every stage is GUARDED on cornea Dice (best-of-previous),
# and the BSpline is cornea-MASKED + regularised, so it aligns the cornea (carrying the scar inside
# it) rather than deforming the scar to fake overlap. Volume CV is unaffected (measured native).
def _affine_intensity(fi: sitk.Image, mi: sitk.Image, base: sitk.Transform) -> sitk.Transform:
    """12-DOF affine (Mattes MI) starting from the rigid base — recovers scale/shear between repeats."""
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.05, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetOptimizerAsRegularStepGradientDescent(
        learningRate=0.5, minStep=1e-4, numberOfIterations=80,
        relaxationFactor=0.7, gradientMagnitudeTolerance=1e-6)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel([2, 1])
    R.SetSmoothingSigmasPerLevel([1.0, 0.0])
    R.SetMovingInitialTransform(base)               # rigid as the starting point
    R.SetInitialTransform(sitk.AffineTransform(3), inPlace=False)
    aff = R.Execute(fi, mi)
    return sitk.CompositeTransform([base, aff])


def _strong_bspline(fb: sitk.Image, mb: sitk.Image, base: sitk.Transform,
                    cornea_mask_b: sitk.Image, mesh=(8, 8, 6)) -> sitk.Transform:
    """Denser cornea-masked BSpline (8x8x6 mesh vs the production 4^3) for residual optical warp."""
    bspline = sitk.BSplineTransformInitializer(fb, list(mesh), order=3)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    R.SetMetricSamplingStrategy(R.RANDOM)
    R.SetMetricSamplingPercentage(0.12, seed=1)
    R.SetInterpolator(sitk.sitkLinear)
    R.SetMetricFixedMask(cornea_mask_b)
    R.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5, numberOfIterations=60,
        maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=200,
        costFunctionConvergenceFactor=1e7)
    R.SetShrinkFactorsPerLevel([1])
    R.SetSmoothingSigmasPerLevel([0.0])
    R.SetMovingInitialTransform(base)
    R.SetInitialTransform(bspline, inPlace=True)
    R.Execute(fb, mb)
    return sitk.CompositeTransform([base, bspline])


def align_transform_v2(fixed_vol_path: Path, fixed_label_path: Path,
                       moving_vol_path: Path, moving_label_path: Path) -> tuple[sitk.Transform, str]:
    """Stronger volumetric alignment: rigid → affine → denser cornea-driven BSpline, each kept only
    if it improves (rigid/affine) or doesn't degrade (BSpline) cornea overlap. Same return contract
    as align_transform. Use for tighter replicate alignment / scar-overlap reproducibility."""
    fvol, mvol = _read_vol(fixed_vol_path), _read_vol(moving_vol_path)
    flab, mlab = _read_label(fixed_label_path), _read_label(moving_label_path)
    fi, mi = _iso(fvol), _iso(mvol)
    flab_iso = _iso(flab, interp=sitk.sitkNearestNeighbor)
    ref_cm_iso = sitk.GetArrayFromImage(flab_iso) >= CORNEA_MIN

    ident = identity()
    d_id = _cornea_dice_iso(mlab, flab_iso, ref_cm_iso, ident)
    # Stage 1: rigid (best-of-identity)
    try:
        rigid = _rigid_intensity(fi, mi)
        d_rig = _cornea_dice_iso(mlab, flab_iso, ref_cm_iso, rigid)
    except Exception:  # noqa: BLE001
        rigid, d_rig = ident, -1.0
    base, base_dice, mode = (rigid, d_rig, "rigid") if d_rig > d_id else (ident, d_id, "identity")
    # Stage 2: affine (best-of-rigid)
    try:
        aff = _affine_intensity(fi, mi, base)
        d_aff = _cornea_dice_iso(mlab, flab_iso, ref_cm_iso, aff)
        if d_aff > base_dice:
            base, base_dice, mode = aff, d_aff, ("rigid+affine" if base is rigid else "affine")
    except Exception:  # noqa: BLE001
        pass
    # Stage 3: denser cornea-masked BSpline (anti-overfit guard on cornea Dice)
    fb, mb = _iso(fvol, iso=ISO_B), _iso(mvol, iso=ISO_B)
    flab_b = _iso(flab, interp=sitk.sitkNearestNeighbor, iso=ISO_B)
    cornea_mask = sitk.Cast(sitk.BinaryDilate(sitk.BinaryThreshold(flab_b, CORNEA_MIN, 255, 1, 0), [4, 4, 2]), sitk.sitkUInt8)
    try:
        comp = _strong_bspline(fb, mb, base, cornea_mask)
        d_bsp = _cornea_dice_iso(mlab, flab_iso, ref_cm_iso, comp)
        if d_bsp >= base_dice - 0.002:
            return comp, mode + "+bspline"
    except Exception:  # noqa: BLE001
        pass
    return base, mode


def resample_label(label_path: Path, fixed_path: Path, tx: sitk.Transform) -> np.ndarray:
    fixed = _read_vol(fixed_path)
    out = sitk.Resample(_read_label(label_path), fixed, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    return sitk.GetArrayFromImage(out)


def resample_volume(vol_path: Path, fixed_path: Path, tx: sitk.Transform) -> sitk.Image:
    fixed = _read_vol(fixed_path)
    return sitk.Resample(_read_vol(vol_path), fixed, tx, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
