"""Per-eye multi-scan consensus (partial-overlap, probabilistic).

Repeat scans image slightly different optically-warped patches of the same eye, so
their scars only PARTIALLY correspond. We anchor each scan to a reference on the
scar shape (registration.py), warp it into the reference frame, then build a
probabilistic agreement map and a majority consensus. Per scan we report the
matched fraction (how much of its scar falls in the consensus) — alongside the
reproducible volume (CV%). Per-scan warped volume + mask are written for the tabs.
"""
from __future__ import annotations

import math
import sys

import numpy as np
import SimpleITK as sitk

import orchestration as orch
import labels
import registration as reg

REF_SCAR, REF_CORNEA = 2, 1


def _inverse_rigid(tx: sitk.Transform) -> sitk.Transform:
    """An invertible approximation of a scan→reference transform, for pulling the voted
    consensus back into a scan's NATIVE frame: the rigid part's exact inverse (the optional
    BSpline residual is small and not cheaply invertible). Identity → identity."""
    try:
        if isinstance(tx, sitk.CompositeTransform):
            return tx.GetNthTransform(0).GetInverse()
        return tx.GetInverse()
    except Exception:  # noqa: BLE001
        return sitk.Euler3DTransform()


def _vol_path(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum()); s = int(a.sum()) + int(b.sum())
    return float(2 * inter / s) if s else 0.0


def _frac(part: np.ndarray, whole: np.ndarray) -> float:
    w = int(whole.sum())
    return round(float(np.logical_and(part, whole).sum()) / w, 3) if w else 0.0


def _native_scar_mm3(cid) -> float:
    """Scar volume in the scan's OWN space — the reproducibility biomarker. Measured
    before any registration so the non-rigid warp (which deforms the scar to align
    shapes) cannot distort the reported volume."""
    img = sitk.ReadImage(str(labels.corrected_path(cid)))
    sp = img.GetSpacing()
    arr = sitk.GetArrayFromImage(img)
    return round(float(int((arr == REF_SCAR).sum()) * sp[0] * sp[1] * sp[2]), 4)


def _write(img_arr_zyx: np.ndarray, ref_img: sitk.Image, dst, dtype=sitk.sitkUInt8):
    out = sitk.GetImageFromArray(img_arr_zyx.astype(np.uint8) if dtype == sitk.sitkUInt8 else img_arr_zyx)
    out.CopyInformation(ref_img)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(out, str(dst))


# ── boundary-tolerant agreement (3D-overlap tolerance slider) ──────────────────
# Re-score the warped per-scan scars allowing a slack of `tol_mm`: a scan "agrees" at a voxel if it has
# scar within tol_mm of it. tol=0 reproduces the strict agreement map. Distance transforms are cached
# (bbox-cropped) so the slider is interactive after the first call.
_TOL_CACHE: dict = {}


def _tol_state(consensus_case_id):
    import nibabel as nib
    from scipy import ndimage

    scans_dir = orch.case_root(consensus_case_id) / "scans"
    members = orch.read_manifest(consensus_case_id).get("consensus_cases") or []
    paths = [(c, scans_dir / c / "label.nii.gz") for c in members]
    paths = [(c, p) for c, p in paths if p.exists()]
    if len(paths) < 2:
        raise ValueError("No warped per-scan labels for this case — rebuild the consensus first.")
    sig = tuple((c, p.stat().st_mtime_ns) for c, p in paths)
    cached = _TOL_CACHE.get(consensus_case_id)
    if cached and cached["sig"] == sig:
        return cached

    scars_full = [np.rint(np.asarray(nib.load(str(p)).dataobj)).astype(np.uint8) == REF_SCAR for _, p in paths]
    vol_img = sitk.ReadImage(str(orch.case_root(consensus_case_id) / "previews" / "volume.nii.gz"))
    sp = vol_img.GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    # `scars_full` is loaded via nibabel, whose array axes (i,j,k) match the file's voxel axes — the
    # SAME order as SimpleITK's GetSpacing(). So sampling is (sp[0], sp[1], sp[2]); reversing it (the
    # old code) only made sense for a sitk GetArrayFromImage (z,y,x) array, which this is NOT — the
    # reversal applied the depth spacing to the lateral axis and vice-versa, so every tol_mm distance
    # (and the tolerant Dice) was wrong on the anisotropic OCT grid.
    sampling = (sp[0], sp[1], sp[2])
    shape = scars_full[0].shape
    union = np.zeros(shape, bool)
    for s in scars_full:
        union |= s
    if not union.any():
        raise ValueError("No scar voxels in the warped labels.")
    # crop to the scar bbox + a margin (≥ max usable tolerance) so the distance transforms stay cheap
    margin = [int(np.ceil(0.4 / sampling[a])) for a in range(3)]
    idx = np.where(union)
    slc = tuple(slice(max(0, idx[a].min() - margin[a]), min(shape[a], idx[a].max() + margin[a] + 1)) for a in range(3))
    scars = [s[slc] for s in scars_full]
    edts = [ndimage.distance_transform_edt(~s, sampling=sampling) for s in scars]
    # Reference cornea, ANDed into the vote tiers below so the tolerant core/consensus/union
    # volumes are clipped to the same region build_consensus persists. Resample the reference
    # label onto the consensus grid (z,y,x, like build_consensus), then reorder to nibabel
    # (x,y,z) to match scars_full and crop to the same bbox slice.
    ref = orch.read_manifest(consensus_case_id).get("reference")
    cornea = np.ones(scars[0].shape, bool)
    try:
        if ref and labels.corrected_path(ref).exists():
            ref_lab_zyx = reg.resample_label(
                labels.corrected_path(ref),
                orch.case_root(consensus_case_id) / "previews" / "volume.nii.gz",
                reg.identity())
            cornea = (np.transpose(ref_lab_zyx, (2, 1, 0)) >= REF_CORNEA)[slc]
    except Exception as exc:  # noqa: BLE001 — fall back to no clip (current behavior) if unavailable
        print(f"[consensus] tolerant cornea clip unavailable for {consensus_case_id}: {exc}", file=sys.stderr)
    state = {"sig": sig, "n": len(scars), "vmm3": vmm3, "shape": shape, "slc": slc,
             "scars": scars, "edts": edts, "cornea": cornea}
    _TOL_CACHE[consensus_case_id] = state
    return state


def tolerant_agreement(consensus_case_id, tol_mm: float = 0.0):
    """Return (agreement_map_uint8 in nibabel x,y,z, stats dict) at boundary tolerance `tol_mm` (mm)."""
    st = _tol_state(consensus_case_id)
    n, edts, scars, vmm3 = st["n"], st["edts"], st["scars"], st["vmm3"]
    cornea = st["cornea"]
    tol = max(0.0, float(tol_mm))
    within = [e <= tol for e in edts]               # within tol of each scan's scar
    votes = np.zeros(scars[0].shape, np.uint8)
    for w in within:
        votes += w.astype(np.uint8)
    agr = np.zeros(st["shape"], np.uint8)
    agr[st["slc"]] = np.rint(votes.astype(np.float32) * (100.0 / n)).astype(np.uint8)
    thr = math.floor(n / 2) + 1
    dices = []
    for i in range(n):
        for j in range(i + 1, n):
            den = int(scars[i].sum()) + int(scars[j].sum())
            if den:
                dices.append((int((within[j] & scars[i]).sum()) + int((within[i] & scars[j]).sum())) / den)
    stats = {"tol_mm": round(tol, 4), "n": n,
             "core_mm3": round(int(((votes >= n) & cornea).sum()) * vmm3, 4),
             "consensus_mm3": round(int(((votes >= thr) & cornea).sum()) * vmm3, 4),
             "union_mm3": round(int(((votes >= 1) & cornea).sum()) * vmm3, 4),
             "mean_pairwise_dice": round(float(np.mean(dices)), 3) if dices else None}
    return agr, stats


def build_consensus(case_ids, consensus_case_id, reference=None) -> dict:
    """Align all scans to a reference (scar-anchored), warp them into the reference
    frame, vote a probabilistic consensus, and write per-scan + consensus artifacts.
    Returns the reproducibility report."""
    cids = [c for c in dict.fromkeys(case_ids)              # de-dupe so a repeated id
            if labels.corrected_path(c).exists() and _vol_path(c).exists()]  # can't double-count
    if len(cids) < 2:
        raise ValueError("Need at least 2 scans with a segmentation for consensus.")
    ref = reference if reference in cids else cids[0]
    ref_overridden = bool(reference) and reference != ref

    ref_vol_path = _vol_path(ref)
    ref_img = sitk.ReadImage(str(ref_vol_path))
    reg._canon(ref_img)
    sp = ref_img.GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    ref_lab = reg.resample_label(labels.corrected_path(ref), ref_vol_path, reg.identity())  # (z,y,x)
    ref_cornea = ref_lab >= REF_CORNEA
    ref_scar = ref_lab == REF_SCAR

    scans_dir = orch.case_root(consensus_case_id) / "scans"
    warped_scars = {}
    data_masks = {}                                    # scan → where it has image data (post-warp FOV)
    transforms = {}                                    # scan → the chosen scan→ref transform
    per_scan = []
    for c in cids:
        if c == ref:
            wvol = reg.resample_volume(ref_vol_path, ref_vol_path, reg.identity())
            wlab = ref_lab
            mode, sd = "reference", 1.0
            chosen = reg.identity()
        else:
            # Align by the guarded intensity+BSpline cascade on the RAW volumes
            # (masks alone can't localise the gross inter-scan shift; see registration.py).
            tx, mode = reg.align_transform(ref_vol_path, labels.corrected_path(ref),
                                           _vol_path(c), labels.corrected_path(c))
            lab_reg = reg.resample_label(labels.corrected_path(c), ref_vol_path, tx)
            lab_id = reg.resample_label(labels.corrected_path(c), ref_vol_path, reg.identity())
            # best-of guard (final safety net): never accept an alignment whose scar
            # overlaps the reference worse than no registration at all.
            if _dice(ref_scar, lab_reg == REF_SCAR) >= _dice(ref_scar, lab_id == REF_SCAR):
                wlab, chosen = lab_reg, tx
            else:
                wlab, chosen, mode = lab_id, reg.identity(), "identity"
            wvol = reg.resample_volume(_vol_path(c), ref_vol_path, chosen)
            sd = round(_dice(ref_scar, wlab == REF_SCAR), 3)
        warped_scars[c] = wlab == REF_SCAR
        transforms[c] = chosen
        wvol_arr = sitk.GetArrayFromImage(wvol)
        data_masks[c] = wvol_arr > 0                    # this scan's field-of-view in the ref frame
        _write(wvol_arr, ref_img, scans_dir / c / "volume.nii.gz", dtype=sitk.sitkFloat32)
        _write(wlab, ref_img, scans_dir / c / "label.nii.gz")
        # Volume = NATIVE (reproducibility biomarker); shape metrics = post-alignment.
        per_scan.append({"case": c, "role": mode,
                         "scar_volume_mm3": _native_scar_mm3(c),
                         "scar_dice_to_ref": sd})

    # probabilistic agreement + majority consensus (within the reference cornea)
    stack = np.stack([warped_scars[c] for c in cids]).astype(np.uint8)
    votes = stack.sum(axis=0)
    n = len(cids)
    prob = (votes / n).astype(np.float32)
    consensus = (votes >= math.floor(n / 2) + 1) & ref_cornea

    # Per-scan NATIVE-frame consensus, for the gallery's 3rd before/after panel: pull the voted
    # consensus scar from the reference frame back into each scan's own grid (inverse of the
    # scan→ref transform), keep it within that scan's own cornea, and write it beside the scan's
    # warped artifacts. The reference scan is exact (identity); others use the rigid inverse.
    cons_scar_ref = sitk.GetImageFromArray(consensus.astype(np.uint8))
    cons_scar_ref.CopyInformation(ref_img)   # ref_img is CANON (origin 0, identity direction) — see reg._canon above
    for c in cids:
        try:
            native_img = sitk.ReadImage(str(_vol_path(c)))
            # GEOMETRY FIX: transforms[c] + cons_scar_ref both live in CANON space (reg._canon zeroes the
            # origin + identity direction). The raw native_img carries the OCT's rotated/flipped direction
            # (e.g. [[1,0,0],[0,0,1],[0,-1,0]]), so resampling the canon consensus straight onto it through a
            # canon-space transform sent every output point to the wrong physical location → the consensus
            # scar sampled background everywhere and cons_native came out with ZERO scar (silently). _canon
            # only rewrites the origin/direction metadata (NOT the voxel grid), so a canon'd copy has the
            # identical sampling grid — resample onto THAT, then write the result back with the ORIGINAL
            # native geometry so it overlays the member's own files.
            native_canon = reg._canon(sitk.ReadImage(str(_vol_path(c))))
            cs = sitk.Resample(cons_scar_ref, native_canon, _inverse_rigid(transforms[c]),
                               sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            cs_arr = sitk.GetArrayFromImage(cs) > 0
            # TRUNCATE to this scan's actual data FOV (not just its cornea): a partial-FOV replicate only gets
            # the part of the consensus scar that falls within its own imaged data — important for training.
            cs_arr = cs_arr & (sitk.GetArrayFromImage(native_img) > 0)
            own_cornea = reg.resample_label(labels.corrected_path(c), _vol_path(c), reg.identity()) >= REF_CORNEA
            nat = np.where(own_cornea, REF_CORNEA, 0).astype(np.uint8)
            nat[cs_arr & own_cornea] = REF_SCAR
            out = sitk.GetImageFromArray(nat)
            out.CopyInformation(native_img)
            (scans_dir / c).mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(out, str(scans_dir / c / "cons_native.nii.gz"))
        except Exception as exc:  # noqa: BLE001 — panel is convenience; never fail the build
            print(f"[consensus] native consensus map skipped for {c}: {exc}", file=sys.stderr)

    # FOV-restricted agreement: Dice measured ONLY where BOTH scans have image data, so a scar
    # region one scan captured but that lies outside another's field-of-view (a partial cut) is
    # not counted as a disagreement. A high FOV-Dice alongside a lower full Dice proves the gap is
    # partial coverage, not mis-segmentation — exactly the "only partial matching is possible" case.
    ref_mask = data_masks[ref]
    for p in per_scan:
        c = p["case"]
        common = data_masks[c] & ref_mask
        # matched_fraction = fraction of THIS scan's own scar that falls in the consensus (its scar is the
        # denominator) — matches the docstring + types.ts and makes low_correspondence flag a scan whose scar
        # mostly DISAGREES with the consensus. (Was _frac(warped, consensus) = recall-of-consensus, which
        # inverted the flag: a small scar fully inside a big consensus was wrongly flagged, and a big scar
        # barely overlapping a small consensus looked perfect.)
        p["matched_fraction"] = _frac(consensus, warped_scars[c])
        p["low_correspondence"] = p["matched_fraction"] < 0.3 and p["role"] != "reference"
        p["scar_dice_to_ref_fov"] = round(_dice(ref_scar & common, warped_scars[c] & common), 3)
        union = data_masks[c] | ref_mask
        p["fov_overlap_fraction"] = round(float(common.sum()) / float(union.sum() or 1), 3)

    vols = [p["scar_volume_mm3"] for p in per_scan]
    # Sample std (ddof=1) is the correct test-retest dispersion estimator for a small
    # set of repeat acquisitions; population std (ddof=0) understates reproducibility CV.
    mean = float(np.mean(vols)); std = float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0
    pair = [round(_dice(warped_scars[cids[i]], warped_scars[cids[j]]), 3)
            for i in range(n) for j in range(i + 1, n)]
    pair_fov = []
    for i in range(n):
        for j in range(i + 1, n):
            common = data_masks[cids[i]] & data_masks[cids[j]]
            pair_fov.append(round(_dice(warped_scars[cids[i]] & common, warped_scars[cids[j]] & common), 3))

    # write consensus labelmap (cornea=1, consensus scar=2) + agreement (prob*100) map
    cons_label = np.where(ref_cornea, REF_CORNEA, 0).astype(np.uint8)
    cons_label[consensus] = REF_SCAR
    _write(cons_label, ref_img, labels.corrected_path(consensus_case_id))
    _write((prob * 100).astype(np.uint8), ref_img,
           orch.case_root(consensus_case_id) / "previews" / "agreement.nii.gz")
    # the consensus case's display volume = the reference volume (shared frame)
    _write(sitk.GetArrayFromImage(reg.resample_volume(ref_vol_path, ref_vol_path, reg.identity())),
           ref_img, orch.case_root(consensus_case_id) / "previews" / "volume.nii.gz", dtype=sitk.sitkFloat32)

    return {
        "n_scans": n, "reference": ref, "reference_overridden": ref_overridden,
        "reference_requested": reference,
        "agreement_threshold": math.floor(n / 2) + 1,
        "scar_volume_mm3": {"mean": round(mean, 4), "std": round(std, 4),
                            "cv_percent": round(std / mean * 100, 2) if mean else 0.0,
                            "per_scan": vols},
        "consensus_scar_mm3": round(float(consensus.sum() * vmm3), 4),
        "core_full_agreement_mm3": round(float(((votes >= n) & ref_cornea).sum() * vmm3), 4),
        "union_mm3": round(float(((votes >= 1) & ref_cornea).sum() * vmm3), 4),
        "mean_pairwise_scar_dice": round(float(np.mean(pair)), 3) if pair else None,
        "mean_pairwise_scar_dice_fov": round(float(np.mean(pair_fov)), 3) if pair_fov else None,
        "per_scan": per_scan,
        "scans": cids,
    }
