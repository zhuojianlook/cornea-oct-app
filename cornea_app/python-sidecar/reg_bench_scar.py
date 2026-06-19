"""Registration config SWEEP on replicate scans (segmentation held fixed = on-disk labels).

Warps each replicate into the reference (v1) frame under several registration configs and reports
pairwise CORNEA Dice (did alignment improve?) and pairwise SCAR Dice (the target), plus a dilation-
slack diagnostic (how much scar Dice rises when both masks dilate by r voxels — large rise ⇒ the gap
is alignment/boundary slack, small ⇒ genuine shape disagreement). READ-ONLY (no case writes).

Configs span the DOF spectrum so we can see whether more registration freedom helps or just adds noise:
  identity · rigid · affine · current(rigid+minimal-BSpline) · bspline_fine(rigid+denser-BSpline) ·
  v2(rigid+affine+denser-BSpline) · demons(rigid+diffeomorphic-Demons, cornea-region).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as label_mod
import registration as reg

REPLICATES = ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"]


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return round(2.0 * (a & b).sum() / s, 3) if s else float("nan")


def _pairwise(masks):
    cids = list(masks)
    return [_dice(masks[cids[i]], masks[cids[j]]) for i in range(len(cids)) for j in range(i + 1, len(cids))]


def _dilate_dice(masks, r):
    d = {c: (ndimage.binary_dilation(masks[c], iterations=r) if r else masks[c]) for c in masks}
    return round(float(np.mean(_pairwise(d))), 3)


# ── registration variants (reuse registration.py primitives) ──
def _base_rigid(fv, fl, mv, ml):
    fvol, mvol = reg._read_vol(fv), reg._read_vol(mv)
    flab, mlab = reg._read_label(fl), reg._read_label(ml)
    fi, mi = reg._iso(fvol), reg._iso(mvol)
    flab_iso = reg._iso(flab, interp=sitk.sitkNearestNeighbor)
    ref_cm = sitk.GetArrayFromImage(flab_iso) >= reg.CORNEA_MIN
    ident = reg.identity()
    d_id = reg._cornea_dice_iso(mlab, flab_iso, ref_cm, ident)
    try:
        rigid = reg._rigid_intensity(fi, mi)
        d_rig = reg._cornea_dice_iso(mlab, flab_iso, ref_cm, rigid)
    except Exception:  # noqa: BLE001
        rigid, d_rig = ident, -1.0
    base, bd, mode = (rigid, d_rig, "rigid") if d_rig > d_id else (ident, d_id, "identity")
    return dict(fvol=fvol, mvol=mvol, fl=fl, mlab=mlab, fi=fi, mi=mi, flab_iso=flab_iso,
                ref_cm=ref_cm, base=base, base_dice=bd, mode=mode)


def align_identity(fv, fl, mv, ml):
    return reg.identity(), "identity"


def align_rigid(fv, fl, mv, ml):
    c = _base_rigid(fv, fl, mv, ml)
    return c["base"], c["mode"]


def align_affine(fv, fl, mv, ml):
    c = _base_rigid(fv, fl, mv, ml)
    try:
        aff = reg._affine_intensity(c["fi"], c["mi"], c["base"])
        if reg._cornea_dice_iso(c["mlab"], c["flab_iso"], c["ref_cm"], aff) > c["base_dice"]:
            return aff, c["mode"] + "+affine"
    except Exception:  # noqa: BLE001
        pass
    return c["base"], c["mode"]


def _cornea_mask_b(fl):
    flab_b = reg._iso(reg._read_label(fl), interp=sitk.sitkNearestNeighbor, iso=reg.ISO_B)
    return sitk.Cast(sitk.BinaryDilate(sitk.BinaryThreshold(flab_b, reg.CORNEA_MIN, 255, 1, 0), [4, 4, 2]), sitk.sitkUInt8)


def align_bspline_fine(fv, fl, mv, ml):
    c = _base_rigid(fv, fl, mv, ml)
    fb, mb = reg._iso(c["fvol"], iso=reg.ISO_B), reg._iso(c["mvol"], iso=reg.ISO_B)
    try:
        comp = reg._strong_bspline(fb, mb, c["base"], _cornea_mask_b(fl), mesh=(8, 8, 6))
        if reg._cornea_dice_iso(c["mlab"], c["flab_iso"], c["ref_cm"], comp) >= c["base_dice"] - 0.002:
            return comp, c["mode"] + "+bsplineF"
    except Exception:  # noqa: BLE001
        pass
    return c["base"], c["mode"]


def align_demons(fv, fl, mv, ml):
    """rigid → diffeomorphic Demons on the cornea-region iso intensities (histogram-matched)."""
    c = _base_rigid(fv, fl, mv, ml)
    try:
        # bring moving onto the fixed iso grid via the rigid base, then deformable-refine
        mov_on_fix = sitk.Resample(c["mi"], c["fi"], c["base"], sitk.sitkLinear, 0.0, sitk.sitkFloat32)
        hm = sitk.HistogramMatchingImageFilter(); hm.SetNumberOfHistogramLevels(128); hm.SetNumberOfMatchPoints(10)
        mov_hm = hm.Execute(mov_on_fix, c["fi"])
        demons = sitk.DiffeomorphicDemonsRegistrationFilter()
        demons.SetNumberOfIterations(40); demons.SetStandardDeviations(1.5)
        field = demons.Execute(c["fi"], mov_hm)
        disp = sitk.DisplacementFieldTransform(field)
        comp = sitk.CompositeTransform([c["base"], disp])
        if reg._cornea_dice_iso(c["mlab"], c["flab_iso"], c["ref_cm"], comp) >= c["base_dice"] - 0.005:
            return comp, c["mode"] + "+demons"
    except Exception as exc:  # noqa: BLE001
        return c["base"], c["mode"] + f" (demons failed: {str(exc)[:40]})"
    return c["base"], c["mode"]


CONFIGS = [
    ("identity", align_identity),
    ("rigid", align_rigid),
    ("affine", align_affine),
    ("current(rigid+minBSpline)", reg.align_transform),
    ("bspline_fine", align_bspline_fine),
    ("v2(rigid+affine+denseBSpline)", reg.align_transform_v2),
    ("demons", align_demons),
]


def run(align_fn, name):
    ref = REPLICATES[0]
    ref_vol, ref_lab_path = _vol(ref), label_mod.corrected_path(ref)
    ref_lab = reg.resample_label(ref_lab_path, ref_vol, reg.identity())
    scar = {ref: ref_lab == 2}
    cornea = {ref: ref_lab >= 1}
    modes = {}
    for mov in REPLICATES[1:]:
        tx, mode = align_fn(ref_vol, ref_lab_path, _vol(mov), label_mod.corrected_path(mov))
        w = reg.resample_label(label_mod.corrected_path(mov), ref_vol, tx)
        scar[mov], cornea[mov] = w == 2, w >= 1
        modes[mov] = mode
    cor = round(float(np.mean(_pairwise(cornea))), 3)
    sca = round(float(np.mean(_pairwise(scar))), 3)
    slack = f"{_dilate_dice(scar,0)}/{_dilate_dice(scar,1)}/{_dilate_dice(scar,2)}/{_dilate_dice(scar,3)}"
    print(f"  {name:32} cornea {cor:<6} scar {sca:<6} scar@dil0/1/2/3 {slack:18} modes={list(modes.values())}", flush=True)
    return name, cor, sca


if __name__ == "__main__":
    print("Registration config sweep — CS001 OS replicates (segmentation fixed = on-disk labels)\n")
    rows = []
    for name, fn in CONFIGS:
        try:
            rows.append(run(fn, name))
        except Exception as exc:  # noqa: BLE001
            print(f"  {name:32} ERROR: {str(exc)[:80]}")
    print("\n================ SUMMARY (higher cornea+scar Dice = better alignment) ================")
    base = next((s for n, c, s in rows if n.startswith("current")), None)
    for n, c, s in sorted(rows, key=lambda r: -r[2]):
        delta = f"(Δscar {round(s-base,3):+})" if base is not None else ""
        print(f"  {n:32} cornea {c:<6} scar {s:<6} {delta}")
