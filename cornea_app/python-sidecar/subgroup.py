"""Automatic SUBGROUP assignment by PURE bright-spot (hysteresis scar) alignment.

A subgroup is a replicate SET of one eye that depicts the SAME lesion and must be aligned/voted together
(distinct lesions of the same eye are voted separately, never merged). We cluster an eye's cornea-segmented
replicates by rigidly fitting their hysteresis bright-spot constellations onto each other — IGNORING the
cornea (the user's chosen 'pure bright-spot fit'), robust to a scan cutting off part of the scar because the
fit is driven by the moments + overlap of the VISIBLE bright spots.

Key: a pure rigid fit can always slide two single blobs onto each other, so raw post-fit overlap alone would
merge everything. We therefore GATE similarity by the fit MAGNITUDE — replicates of the same lesion already
sit in nearly the same place (a small shift makes them coincide), whereas a lesion in a different location
needs a large shift to force overlap → low similarity → its own subgroup.
"""
from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy import ndimage

import orchestration as orch
import labels as label_mod
import scar as scar_mod
import registration as reg

# similarity = post-fit bright-spot Dice × magnitude gate. The gate decays with the translation needed to
# achieve the overlap: ~1 for an acquisition-jitter shift, →0 once the fit teleports the lesion across the
# cornea. Calibrated to the observed replicate jitter (~0.2 mm) vs a different-lesion separation (mm-scale).
DEFAULT = {
    "phi_percentile": 92.0,
    "min_blob_mm3": 0.02,     # drop hysteresis noise specks (real scar dominant blob is ~1-2 mm³)
    "centroid_tol_mm": 0.5,   # bright-spot centroid distance (mm) at which the magnitude gate = 0.5 (replicate
                              # jitter ~0.2 mm << this << a different-lesion separation of mm-scale)
    "link_threshold": 0.45,   # two scans are same-subgroup when gated similarity ≥ this (single-link cluster)
    "smooth_mm": 0.12,        # gaussian on the mask for a continuous registration metric
}


def _vol(cid: str) -> Path:
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _scar_mask_np(cid: str, phi: float, min_mm3: float):
    """Hysteresis bright-spot mask (nibabel i,j,k order) with noise specks removed, + (n_blobs, total_mm3)."""
    lab = np.rint(np.asarray(nib.load(str(label_mod.corrected_path(cid))).dataobj)).astype(np.uint8)
    vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
    scar = np.asarray(scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=phi)) & ((lab == 1) | (lab == 2))
    img = sitk.ReadImage(str(_vol(cid))); sp = img.GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    lbl, n = ndimage.label(scar)
    if n:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s * vmm3 >= min_mm3]
        scar = np.isin(lbl, keep) if keep else np.zeros_like(scar)
        n_keep = len(keep)
    else:
        n_keep = 0
    return scar.astype(np.uint8), n_keep, float(scar.sum()) * vmm3


def _to_sitk_mask(mask_np: np.ndarray, cid: str, tmpd: Path) -> sitk.Image:
    """Write a numpy (i,j,k) mask with the scan's geometry, read back as an sitk image (correct physical
    space) — the same write_label_nifti→ReadImage round-trip the consensus/benchmark code uses."""
    p = tmpd / f"{cid}_bs.nii.gz"
    label_mod.write_label_nifti(mask_np.astype(np.uint8), _vol(cid), p)
    return sitk.Cast(sitk.ReadImage(str(p)) > 0, sitk.sitkUInt8)


def _dice(a: sitk.Image, b: sitk.Image) -> float:
    aa = sitk.GetArrayViewFromImage(a).astype(bool); bb = sitk.GetArrayViewFromImage(b).astype(bool)
    s = int(aa.sum()) + int(bb.sum())
    return 2.0 * int((aa & bb).sum()) / s if s else 0.0


def _centroid_mm(mask: sitk.Image):
    """Physical-space (mm) centroid of a bright-spot mask, or None if empty. Robust to partial cutoff (a cut
    blob shifts the centroid only slightly) — the basis of the same-vs-different-lesion magnitude gate."""
    arr = sitk.GetArrayViewFromImage(mask)            # (z,y,x)
    idx = np.argwhere(arr > 0)
    if not len(idx):
        return None
    cz, cy, cx = idx.mean(axis=0)
    return np.array(mask.TransformContinuousIndexToPhysicalPoint((float(cx), float(cy), float(cz))), dtype=float)


def fit_bright_spots(fixed: sitk.Image, moving: sitk.Image, smooth_mm: float = 0.12):
    """PURE bright-spot rigid fit of `moving` onto `fixed` (both binary masks in physical space). Moments
    initialiser aligns the centroid + principal axes (the 'dimensions of their spatial relationships'); a
    short rigid refine on the gaussian-smoothed masks tightens it, kept only if it improves overlap. Returns
    (transform, warped_moving_mask, dice) — the post-fit overlap = how well the bright-spot CONFIGURATIONS match
    once aligned (position-independent); the same-vs-different-lesion gate is the native centroid distance,
    scored separately in auto_subgroups."""
    if int(sitk.GetArrayViewFromImage(fixed).sum()) == 0 or int(sitk.GetArrayViewFromImage(moving).sum()) == 0:
        return sitk.Euler3DTransform(), moving, 0.0
    ff = sitk.SmoothingRecursiveGaussian(sitk.Cast(fixed, sitk.sitkFloat32), smooth_mm)
    mf = sitk.SmoothingRecursiveGaussian(sitk.Cast(moving, sitk.sitkFloat32), smooth_mm)
    try:
        init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                 sitk.CenteredTransformInitializerFilter.MOMENTS)
    except Exception:  # noqa: BLE001 — degenerate moments → centroid-only (GEOMETRY)
        init = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler3DTransform(),
                                                 sitk.CenteredTransformInitializerFilter.GEOMETRY)

    def _apply(tx):
        w = sitk.Resample(moving, fixed, tx, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return w, _dice(fixed, w)

    w0, d0 = _apply(init)
    best_tx, best_w, best_d = init, w0, d0
    try:
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, seed=1)
        R.SetInterpolator(sitk.sitkLinear)
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=80,
                                                   relaxationFactor=0.6, gradientMagnitudeTolerance=1e-6)
        R.SetOptimizerScalesFromPhysicalShift()
        R.SetInitialTransform(sitk.Euler3DTransform(init), inPlace=False)
        ref = R.Execute(ff, mf)
        wr, dr = _apply(ref)
        if dr >= d0:                                # best-of: keep the refine only if it improves overlap
            best_tx, best_w, best_d = ref, wr, dr
    except Exception:  # noqa: BLE001 — optimiser diverged → keep the moments init
        pass
    return best_tx, best_w, best_d


def _gate(centroid_dist_mm: float, p: dict) -> float:
    """Magnitude gate ∈ (0,1]: ~1 when the bright-spot centroids nearly coincide (replicate jitter), 0.5 at
    centroid_tol_mm, →0 once they sit a lesion-apart. Transform-free → robust to partial cutoff."""
    d = float(centroid_dist_mm) / max(1e-6, float(p["centroid_tol_mm"]))
    return float(1.0 / (1.0 + d * d))


def _cluster(sim: np.ndarray, thr: float) -> list[int]:
    """Single-link agglomerative clustering: connect scans whose gated similarity ≥ thr, label components."""
    n = sim.shape[0]
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= thr:
                parent[find(i)] = find(j)
    roots = {}
    out = []
    for i in range(n):
        r = find(i)
        out.append(roots.setdefault(r, len(roots) + 1))   # subgroup labels 1..k
    return out


_SUBGROUP_RGB = [(255, 80, 80), (90, 200, 110), (90, 150, 255), (235, 200, 70), (210, 110, 235), (90, 220, 220)]


def overlay_png(member_ids: list[str], subgroups: dict, params: dict | None = None) -> str:
    """En-face OVERLAY (base64 PNG data-URL) of the eye's bright spots in a common ANATOMICAL frame, coloured
    by assigned subgroup, so the user can VERIFY the auto-assignment: same-subgroup scars pile into one bright
    blob; a scan placed in a different subgroup (a displaced lesion) shows its own colour off to the side.

    The CLUSTERING uses the pure bright-spot fit (auto_subgroups); this overlay instead warps each scan into
    the reference's CORNEA frame (so lesions appear in their true relative positions — a fit-to-ref would hide
    the very displacement we want to show). Reference = first member."""
    import base64
    import oct_preprocess as _oct
    p = {**DEFAULT, **(params or {})}
    members = [c for c in member_ids if label_mod.corrected_path(c).exists() and _vol(c).exists()]
    if not members:
        return ""
    ref = members[0]; ref_vol = _vol(ref)
    tmpd = Path(tempfile.mkdtemp(prefix="subovl_"))
    try:
        # depth axis on the WARPED (sitk z,y,x) ref cornea — the SAME array order the warped masks below use
        # (computing it on the nibabel i,j,k mask collapses the wrong axis → a B-scan sliver, not en-face).
        ref_cornea_w = reg.resample_label(label_mod.corrected_path(ref), ref_vol, reg.identity()) >= 1  # (z,y,x)
        dax = scar_mod._depth_axis(ref_cornea_w)
        acc = None  # (H,W,3) float accumulation
        for c in members:
            mnp, _nb, _mm3 = _scar_mask_np(c, float(p["phi_percentile"]), float(p["min_blob_mm3"]))
            pth = tmpd / f"{c}.nii.gz"; label_mod.write_label_nifti(mnp.astype(np.uint8), _vol(c), pth)
            tx = reg.identity() if c == ref else reg.align_transform(
                ref_vol, label_mod.corrected_path(ref), _vol(c), label_mod.corrected_path(c))[0]
            warped = reg.resample_label(pth, ref_vol, tx) >= 1                 # (z,y,x) in ref frame
            enf = warped.max(axis=dax).astype(np.float32)                      # en-face footprint (correct axis)
            if acc is None:
                acc = np.zeros((*enf.shape, 3), np.float32)
            col = _SUBGROUP_RGB[(int(subgroups.get(c, 1)) - 1) % len(_SUBGROUP_RGB)]
            for ch in range(3):
                acc[..., ch] += enf * col[ch]
        if acc is None:
            return ""
        rgb = np.clip(acc, 0, 255).astype(np.uint8)
        H, W = rgb.shape[:2]
        scl = max(1, int(480 / max(H, W)))
        rgb = np.repeat(np.repeat(rgb, scl, 0), scl, 1)
        png = _oct._png_bytes(rgb)
        return "data:image/png;base64," + base64.b64encode(png).decode()
    finally:
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)


def auto_subgroups(member_ids: list[str], params: dict | None = None) -> dict:
    """Cluster an eye's cornea-segmented replicates into subgroups by pure bright-spot fit. Returns
    {members, subgroups:{cid:label}, similarity:[[..]], pairs:[{a,b,dice,trans_mm,rot_deg,sim}], blobs:{cid:..},
    n_subgroups}. READ-ONLY (masks built in a tempdir, cleaned up)."""
    p = {**DEFAULT, **(params or {})}
    members = [c for c in dict.fromkeys(member_ids)
               if label_mod.corrected_path(c).exists() and _vol(c).exists()]
    if len(members) < 2:
        raise ValueError("Need ≥2 cornea-segmented replicate scans to auto-assign subgroups.")
    tmpd = Path(tempfile.mkdtemp(prefix="subgrp_"))
    try:
        masks, blobs, cents = {}, {}, {}
        for c in members:
            mnp, nb, mm3 = _scar_mask_np(c, float(p["phi_percentile"]), float(p["min_blob_mm3"]))
            masks[c] = _to_sitk_mask(mnp, c, tmpd)
            cents[c] = _centroid_mm(masks[c])
            blobs[c] = {"n_blobs": nb, "scar_mm3": round(mm3, 3), "empty": cents[c] is None}
        n = len(members)
        sim = np.eye(n, dtype=float)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = members[i], members[j]
                _tx, _w, d = fit_bright_spots(masks[a], masks[b], float(p["smooth_mm"]))
                cd = (float(np.linalg.norm(cents[a] - cents[b]))
                      if (cents[a] is not None and cents[b] is not None) else float("inf"))
                s = d * _gate(cd, p)
                sim[i, j] = sim[j, i] = s
                pairs.append({"a": a, "b": b, "dice": round(d, 3),
                              "centroid_dist_mm": round(cd, 3) if math.isfinite(cd) else None, "sim": round(s, 3)})
        labels_out = _cluster(sim, float(p["link_threshold"]))
        subgroups = {members[i]: labels_out[i] for i in range(n)}
        return {"members": members, "subgroups": subgroups, "n_subgroups": len(set(labels_out)),
                "similarity": [[round(float(sim[i, j]), 3) for j in range(n)] for i in range(n)],
                "pairs": pairs, "blobs": blobs, "params": {k: p[k] for k in DEFAULT}}
    finally:
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)
