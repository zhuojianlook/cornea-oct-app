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
    "centroid_tol_mm": 0.4,   # relative-centroid distance (mm) at which the gate = 0.5 (replicate jitter ~0.1-
                              # 0.36 mm validated << this << a different-lesion mm-scale separation). 0.4 keeps a
                              # clear margin from link_threshold so a high single-blob Dice can't merge at the boundary.
    "centroid_hardsplit_mm": 0.8,  # beyond this relative-centroid distance → DIFFERENT lesion, sim forced 0 (the
                              # cornea-ignored config Dice can't override a clearly-displaced lesion)
    "link_threshold": 0.45,   # two scans are same-subgroup when gated similarity ≥ this (single-link cluster)
    "smooth_mm": 0.12,        # gaussian on the mask for a continuous registration metric
    "gate_cornea_frame": True,  # gate on the scar centroid RELATIVE TO the scan's own CORNEA centroid (scar
                              # position within the cornea) — removes between-scan EYE MOTION (cornea+scar shift
                              # together), registration-free. False = raw native scar-centroid distance.
}


def _vol(cid: str) -> Path:
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


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


def _scan_masks(cid: str, p: dict):
    """Per-scan (nibabel i,j,k) masks + geometry for subgrouping: (scar_mask, cornea_mask, zooms_mm, n_blobs,
    scar_mm3). Hysteresis bright spots with noise specks dropped; cornea = labels 1|2."""
    lab = np.rint(np.asarray(nib.load(str(label_mod.corrected_path(cid))).dataobj)).astype(np.uint8)
    vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
    zooms = np.array(nib.load(str(_vol(cid))).header.get_zooms()[:3], dtype=float)
    vmm3 = float(np.prod(zooms))
    scar = np.asarray(scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=float(p["phi_percentile"]))) & ((lab == 1) | (lab == 2))
    lbl, nlb = ndimage.label(scar); nb = 0
    if nlb:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, nlb + 1))
        keep = [k + 1 for k, s in enumerate(sizes) if s * vmm3 >= float(p["min_blob_mm3"])]
        scar = np.isin(lbl, keep) if keep else np.zeros_like(scar); nb = len(keep)
    return scar.astype(np.uint8), (lab >= 1), zooms, nb, float(scar.sum()) * vmm3


def _int_shift(mask: np.ndarray, shift) -> np.ndarray:
    """Translate a 3-D mask by an integer (i,j,k) voxel offset, zero-padded (no wrap) — the overlay's
    cornea-centroid alignment (a pure translation, registration-free)."""
    out = np.zeros_like(mask)
    src, dst = [slice(None)] * 3, [slice(None)] * 3
    for ax in range(3):
        s = int(round(shift[ax])); n = mask.shape[ax]
        if s >= 0:
            dst[ax] = slice(min(s, n), n); src[ax] = slice(0, max(0, n - s))
        else:
            dst[ax] = slice(0, max(0, n + s)); src[ax] = slice(min(-s, n), n)
    out[tuple(dst)] = mask[tuple(src)]
    return out


def auto_subgroups(member_ids: list[str], params: dict | None = None) -> dict:
    """Cluster an eye's cornea-segmented SCAR scans into subgroups (same lesion → together).

    The bright-spot CONFIGURATION match (post-fit Dice) is a PURE bright-spot fit (cornea-ignored, robust to
    partial cutoff). The same-vs-different-lesion GATE is the distance between each scan's scar centroid measured
    RELATIVE TO ITS OWN CORNEA centroid (scar position WITHIN the cornea): between-scan EYE MOTION shifts cornea
    and scar together so the relative vector is unchanged (no false split), while a lesion in a different place
    changes it. Registration-free + robust to cutoff (centroids of the VISIBLE cornea/scar) — gate_cornea_frame
    =False reverts to the raw scar-centroid distance. sim = post-fit Dice × gate(dist); single-link cluster ≥
    link_threshold. The overlay translates each scan's footprint so its cornea centroid coincides with the
    reference's (the SAME alignment the gate uses → overlay and clustering are consistent). Returns {members,
    subgroups, n_subgroups, similarity, pairs[{a,b,dice,centroid_dist_mm,centroid_dist_native_mm,sim}], blobs,
    overlay, gate_frame, params}. READ-ONLY (temp masks cleaned up); CPU."""
    import base64
    import oct_preprocess as _oct
    p = {**DEFAULT, **(params or {})}
    members = [c for c in dict.fromkeys(member_ids)
               if label_mod.corrected_path(c).exists() and _vol(c).exists()]
    if len(members) < 2:
        raise ValueError("Need ≥2 cornea-segmented replicate scans to auto-assign subgroups.")
    cornea_gate = bool(p.get("gate_cornea_frame", True))
    ref = members[0]
    tmpd = Path(tempfile.mkdtemp(prefix="subgrp_"))
    try:
        masks, blobs, scar_np, corn_idx, scar_mm, relvec = {}, {}, {}, {}, {}, {}
        dax = 0
        for c in members:
            scar, cornea, zooms, nb, mm3 = _scan_masks(c, p)
            scar_np[c] = scar
            masks[c] = _to_sitk_mask(scar, c, tmpd)                  # native mask → the pure-fit Dice
            cidx = np.argwhere(cornea); corn_idx[c] = cidx.mean(0) if len(cidx) else None
            sidx = np.argwhere(scar); scent = sidx.mean(0) if len(sidx) else None
            scar_mm[c] = (scent * zooms) if scent is not None else None
            relvec[c] = ((scent - corn_idx[c]) * zooms) if (scent is not None and corn_idx[c] is not None) else None
            blobs[c] = {"n_blobs": nb, "scar_mm3": round(mm3, 3), "empty": scent is None}
            if c == ref:
                dax = scar_mod._depth_axis(cornea)

        def _dist(a, b, table):
            return (float(np.linalg.norm(table[a] - table[b])) if (table[a] is not None and table[b] is not None) else float("inf"))

        n = len(members)
        sim = np.eye(n, dtype=float)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = members[i], members[j]
                _tx, _w, d = fit_bright_spots(masks[a], masks[b], float(p["smooth_mm"]))
                cd_nat = _dist(a, b, scar_mm); cd_rel = _dist(a, b, relvec)
                cd = cd_rel if cornea_gate else cd_nat
                # HARD SPLIT: beyond this separation it is a different lesion regardless of the (cornea-ignored)
                # configuration Dice — stops a high single-blob Dice from merging a clearly-displaced lesion.
                s = 0.0 if cd > float(p["centroid_hardsplit_mm"]) else d * _gate(cd, p)
                sim[i, j] = sim[j, i] = s
                pairs.append({"a": a, "b": b, "dice": round(d, 3),
                              "centroid_dist_mm": round(cd, 3) if math.isfinite(cd) else None,
                              "centroid_dist_native_mm": round(cd_nat, 3) if math.isfinite(cd_nat) else None,
                              "sim": round(s, 3)})
        labels_out = _cluster(sim, float(p["link_threshold"]))
        subgroups = {members[i]: labels_out[i] for i in range(n)}

        # OVERLAY: translate each footprint so its cornea centroid lands on the reference's (cornea-centroid
        # alignment — the same eye-motion removal the gate does), en-face MIP, coloured by subgroup. Each MIP is
        # padded/cropped into ONE reference-sized canvas (replicates may differ in frame count / lateral extent
        # — never broadcast directly), and the whole render is best-effort: a hiccup yields an empty overlay,
        # NEVER a failed proposal (the cluster decision is already made above).
        overlay = ""
        try:
            enfs = {}
            for c in members:
                sh = (corn_idx[ref] - corn_idx[c]) if (corn_idx[ref] is not None and corn_idx[c] is not None) else np.zeros(3)
                enfs[c] = _int_shift(scar_np[c], sh).max(axis=dax).astype(np.float32)
            Hc = max(e.shape[0] for e in enfs.values()); Wc = max(e.shape[1] for e in enfs.values())
            acc = np.zeros((Hc, Wc, 3), np.float32)
            for c in members:
                e = enfs[c]; h, w = min(e.shape[0], Hc), min(e.shape[1], Wc)
                col = _SUBGROUP_RGB[(int(subgroups.get(c, 1)) - 1) % len(_SUBGROUP_RGB)]
                for ch in range(3):
                    acc[:h, :w, ch] += e[:h, :w] * col[ch]
            rgb = np.clip(acc, 0, 255).astype(np.uint8)
            scl = max(1, int(480 / max(Hc, Wc)))
            rgb = np.repeat(np.repeat(rgb, scl, 0), scl, 1)
            overlay = "data:image/png;base64," + base64.b64encode(_oct._png_bytes(rgb)).decode()
        except Exception:  # noqa: BLE001 — the overlay is a verification aid; never sink the proposal
            overlay = ""

        return {"members": members, "subgroups": subgroups, "n_subgroups": len(set(labels_out)),
                "similarity": [[round(float(sim[i, j]), 3) for j in range(n)] for i in range(n)],
                "pairs": pairs, "blobs": blobs, "overlay": overlay,
                "gate_frame": "cornea-relative" if cornea_gate else "native", "params": {k: p[k] for k in DEFAULT}}
    finally:
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)
