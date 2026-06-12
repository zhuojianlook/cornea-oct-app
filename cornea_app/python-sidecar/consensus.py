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

import numpy as np
import SimpleITK as sitk

import orchestration as orch
import labels
import registration as reg

REF_SCAR, REF_CORNEA = 2, 1


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
    per_scan = []
    for c in cids:
        if c == ref:
            wvol = reg.resample_volume(ref_vol_path, ref_vol_path, reg.identity())
            wlab = ref_lab
            mode, sd = "reference", 1.0
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
        _write(sitk.GetArrayFromImage(wvol), ref_img, scans_dir / c / "volume.nii.gz", dtype=sitk.sitkFloat32)
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

    for p in per_scan:
        p["matched_fraction"] = _frac(warped_scars[p["case"]], consensus)
        p["low_correspondence"] = p["matched_fraction"] < 0.3 and p["role"] != "reference"

    vols = [p["scar_volume_mm3"] for p in per_scan]
    # Sample std (ddof=1) is the correct test-retest dispersion estimator for a small
    # set of repeat acquisitions; population std (ddof=0) understates reproducibility CV.
    mean = float(np.mean(vols)); std = float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0
    pair = [round(_dice(warped_scars[cids[i]], warped_scars[cids[j]]), 3)
            for i in range(n) for j in range(i + 1, n)]

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
        "agreement_threshold": math.floor(n / 2) + 1,
        "scar_volume_mm3": {"mean": round(mean, 4), "std": round(std, 4),
                            "cv_percent": round(std / mean * 100, 2) if mean else 0.0,
                            "per_scan": vols},
        "consensus_scar_mm3": round(float(consensus.sum() * vmm3), 4),
        "core_full_agreement_mm3": round(float(((votes >= n) & ref_cornea).sum() * vmm3), 4),
        "union_mm3": round(float(((votes >= 1) & ref_cornea).sum() * vmm3), 4),
        "mean_pairwise_scar_dice": round(float(np.mean(pair)), 3) if pair else None,
        "per_scan": per_scan,
        "scans": cids,
    }
