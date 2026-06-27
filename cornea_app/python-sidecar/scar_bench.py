"""Comprehensive scar-strategy reproducibility benchmark on REPLICATE scans (CS001 OS v1/v2/v3).

Replicates of one eye should give near-identical scar → high pairwise 3D Dice, low boundary HD95, low
volume CV. Aligns the 3 to the reference ONCE (config-independent), then:
  (A) PER-SCAN detectors — pairwise Dice + pairwise HD95 (mm, boundary) + volume CV% + repeatability
      coefficient (RC = 2.77·SD_within), to see which detector reproduces best (shape AND volume).
  (B) REPLICATE-FUSION methods (majority ≥2/3 · soft-vote ≥0.5 · STAPLE) on the best detector's aligned
      masks — mean consensus↔scan Dice + leave-one-out stability, the replicate-leveraging fusion.
READ-ONLY on the real cases (masks in memory; temp labelmaps in /tmp). CPU only (no SAM2). ~2-4 min.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as label_mod
import scar as scar_mod
import registration as reg

REPLICATES = ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"]
TMP = Path(tempfile.mkdtemp(prefix="scar_bench_"))

DETECTORS = {
    "brightness": lambda cid, lab, vol, sp: scar_mod.detect_scar_in_cornea(vol, lab, percentile=92),
    "hysteresis": lambda cid, lab, vol, sp: scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=92),
    "hysteresis_TTA": lambda cid, lab, vol, sp: scar_mod.detect_scar_hysteresis_tta(vol, lab, phi_percentile=92),
    "depthnorm_self": lambda cid, lab, vol, sp: scar_mod.detect_scar_depthnorm(vol, lab),
    "normal_anchor": lambda cid, lab, vol, sp: scar_mod.detect_scar_normal_anchor(vol, lab),
    "robust_mad": lambda cid, lab, vol, sp: scar_mod.detect_scar_robust_mad(vol, lab),
    "morph_lcc": lambda cid, lab, vol, sp: scar_mod.detect_scar_morph_lcc(vol, lab, percentile=92),
}


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else float("nan")


def _hd95(a, b, sampling):
    """Symmetric 95th-percentile Hausdorff (mm) between two boolean masks on the same grid."""
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    asurf = a & ~ndimage.binary_erosion(a)
    bsurf = b & ~ndimage.binary_erosion(b)
    dt_b = ndimage.distance_transform_edt(~bsurf, sampling=sampling)
    dt_a = ndimage.distance_transform_edt(~asurf, sampling=sampling)
    d = np.concatenate([dt_b[asurf], dt_a[bsurf]])
    return float(np.percentile(d, 95)) if d.size else float("nan")


def _pairwise(masks, fn):
    cids = list(masks)
    return [fn(masks[cids[i]], masks[cids[j]]) for i in range(len(cids)) for j in range(i + 1, len(cids))]


# Production scar strategies, keyed by the names the UI/method-dropdown uses. Each → a mask in cornea.
def _strategy_detectors(phi: float):
    return {
        "hysteresis": lambda lab, vol: scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=phi),
        "depthnorm": lambda lab, vol: scar_mod.detect_scar_depthnorm(vol, lab, phi_percentile=phi),
        "normal_anchor": lambda lab, vol: scar_mod.detect_scar_normal_anchor(vol, lab),
        "robust_mad": lambda lab, vol: scar_mod.detect_scar_robust_mad(vol, lab),
        "morph_lcc": lambda lab, vol: scar_mod.detect_scar_morph_lcc(vol, lab, percentile=phi),
        "brightness": lambda lab, vol: scar_mod.detect_scar_in_cornea(vol, lab, percentile=phi),
    }


def compare_strategies(member_ids, strategies=None, phi_percentile: float = 92.0) -> dict:
    """READ-ONLY test–retest reproducibility of each scar strategy on a set of REPLICATE scans (same eye+
    subgroup, already cornea-segmented). Aligns the replicates to the reference ONCE (volume-intensity
    driven, detector-independent), then for each strategy computes the scar mask per replicate IN MEMORY
    (never persisted — the canonical labelmaps are untouched), warps them into the reference frame, and
    reports pairwise 3D Dice, pairwise HD95 (mm, boundary), native scar-volume mean / CV% / repeatability
    coefficient (RC = 2.77·SD). Returns {rows, members, n, phi_percentile, reference}.

    NOTE for interpretation: Dice rises with mask size, so a detector that flags MORE scar can look more
    reproducible — read pairwise Dice ALONGSIDE the volume + CV. Reproducibility, not accuracy (no GT)."""
    import tempfile
    members = [c for c in dict.fromkeys(member_ids)
               if label_mod.corrected_path(c).exists() and _vol(c).exists()]
    if len(members) < 2:
        raise ValueError("Need ≥2 segmented replicate scans to compare reproducibility.")
    dets = _strategy_detectors(float(phi_percentile))
    chosen = [s for s in (strategies or list(dets)) if s in dets] or list(dets)

    ref = members[0]; ref_vol = _vol(ref)
    sp = reg._read_vol(ref_vol).GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    samp = (sp[2], sp[1], sp[0])   # warped masks are sitk (z,y,x) → reversed spacing for HD95
    tx = {ref: reg.identity()}
    for mov in members[1:]:
        tx[mov] = reg.align_transform(ref_vol, label_mod.corrected_path(ref), _vol(mov), label_mod.corrected_path(mov))[0]
    data = {c: (np.rint(np.asarray(nib.load(str(label_mod.corrected_path(c))).dataobj)).astype(np.uint8),
                np.asarray(nib.load(str(_vol(c))).dataobj).astype(np.float32)) for c in members}

    tmpd = Path(tempfile.mkdtemp(prefix="cmp_strat_"))

    def warp(mask, cid):
        tmp = tmpd / f"{cid}.nii.gz"
        label_mod.write_label_nifti(mask.astype(np.uint8), _vol(cid), tmp)
        return reg.resample_label(tmp, ref_vol, tx[cid]) >= 1

    rows = []
    try:
        for name in chosen:
            try:
                warped, vols = {}, []
                for cid in members:
                    lab, vol = data[cid]
                    m = np.asarray(dets[name](lab, vol)) & ((lab == 1) | (lab == 2))
                    vols.append(float(m.sum()) * vmm3)
                    warped[cid] = warp(m, cid)
                mean = float(np.mean(vols)); sd = float(np.std(vols, ddof=1)) if len(vols) > 1 else 0.0
                pd = _pairwise(warped, _dice)
                ph = _pairwise(warped, lambda a, b: _hd95(a, b, samp))
                rows.append({
                    "strategy": name,
                    "mean_volume_mm3": round(mean, 3),
                    "cv_percent": round(sd / mean * 100, 2) if mean else 0.0,
                    "rc_mm3": round(2.77 * sd, 3),
                    "mean_pairwise_dice": round(float(np.mean(pd)), 3) if pd else None,
                    "mean_pairwise_hd95_mm": round(float(np.nanmean(ph)), 3) if ph and not np.all(np.isnan(ph)) else None,
                    "n": len(members),
                })
            except Exception as exc:  # noqa: BLE001 — one bad strategy shouldn't kill the table
                rows.append({"strategy": name, "error": str(exc)[:120], "n": len(members)})
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    return {"rows": rows, "members": members, "n": len(members),
            "phi_percentile": float(phi_percentile), "reference": ref}


def main():
    ref = REPLICATES[0]
    ref_vol = _vol(ref)
    ref_img = reg._read_vol(ref_vol)
    sp = ref_img.GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    samp = (sp[2], sp[1], sp[0])  # array axis (z,y,x) ↔ sitk spacing (x,y,z) reversed, for HD95
    # align each replicate → ref once (volume-intensity driven; detector-independent)
    tx = {ref: reg.identity()}
    for mov in REPLICATES[1:]:
        tx[mov] = reg.align_transform(ref_vol, label_mod.corrected_path(ref), _vol(mov), label_mod.corrected_path(mov))[0]
        print(f"  aligned {mov}", flush=True)

    data = {}
    for cid in REPLICATES:
        lab = np.rint(np.asarray(nib.load(str(label_mod.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        data[cid] = (lab, vol)

    def warp(mask, cid):
        tmp = TMP / f"{cid}.nii.gz"
        label_mod.write_label_nifti(mask.astype(np.uint8), _vol(cid), tmp)
        return reg.resample_label(tmp, ref_vol, tx[cid]) >= 1

    print("\nNOTE: on-disk scar labels are themselves hysteresis(phi=92) output (no independent manual GT),")
    print("so this is TEST-RETEST REPRODUCIBILITY only — not accuracy. Dice rises with mask size, so read")
    print("pairwise Dice ALONGSIDE volume (see scar_sweep_bench.py for the size-matched comparison).")
    print("\n========== (A) PER-SCAN scar detectors — replicate reproducibility ==========")
    print(f"{'detector':16}{'mean mm³':>9}{'CV%':>7}{'RC mm³':>9}{'pairDice':>9}{'pairHD95mm':>11}")
    warped_by_det = {}
    for name, fn in DETECTORS.items():
        try:
            warped, vols = {}, []
            for cid in REPLICATES:
                lab, vol = data[cid]
                m = fn(cid, lab, vol, sp) & ((lab == 1) | (lab == 2))
                vols.append(float(m.sum()) * vmm3)
                warped[cid] = warp(m, cid)
            mean = float(np.mean(vols)); sd = float(np.std(vols, ddof=1))
            cv = round(sd / mean * 100, 2) if mean else 0.0
            rc = round(2.77 * sd, 4)   # repeatability coeff = smallest detectable change, ABSOLUTE mm³
            pd = round(float(np.mean(_pairwise(warped, _dice))), 3)
            ph = round(float(np.nanmean(_pairwise(warped, lambda a, b: _hd95(a, b, samp)))), 3)
            warped_by_det[name] = warped
            print(f"{name:16}{mean:>9.2f}{cv:>7.2f}{rc:>9.4f}{pd:>9}{ph:>11}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"{name:16} ERROR {str(exc)[:50]}")

    # (B) Fusion on FIXED valid-volume detectors (NOT the size-confounded "best pairwise Dice").
    # At n=3, majority(≥2/3) ≡ soft(mean≥0.5) ≡ STAPLE(≥0.5) are the SAME set by algebra — so we report
    # majority only and label this CONSENSUS REPRESENTATIVENESS, not consensus test-retest (needs ≥2 eyes).
    print("\n========== (B) REPLICATE FUSION (majority ≥2/3 — ≡ soft ≡ STAPLE at n=3) ==========")
    print(f"{'detector':16}{'cons mm³':>9}{'meanDice→scans':>16}{'LOO Dice':>10}")
    for det in ("hysteresis", "robust_mad", "normal_anchor"):
        wm = warped_by_det.get(det)
        if wm is None:
            continue
        cids = list(wm)
        stack = np.stack([wm[c] for c in cids]).astype(np.uint8)
        full = list(range(len(cids)))
        def maj(idx):
            return stack[idx].sum(0) >= (len(idx) // 2 + 1)
        cons = maj(full)
        mean_to = round(float(np.mean([_dice(cons, wm[c]) for c in cids])), 3)
        loo = [_dice(maj([i for i in full if i != h]), wm[cids[h]]) for h in range(len(cids))]
        print(f"{det:16}{float(cons.sum())*vmm3:>9.2f}{mean_to:>16}{round(float(np.mean(loo)),3):>10}", flush=True)
    print("(meanDice→scans = how representative the consensus is of each input; LOO is degenerate at n=2 —")
    print(" reflects replicate near-identity of one eye, not generalization. Validate fusion on ≥2 eyes.)")

    print("\nGoal: HIGH pairwise Dice + LOW HD95 + LOW CV/RC, at a VALID (non-inflated) scar volume.")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
