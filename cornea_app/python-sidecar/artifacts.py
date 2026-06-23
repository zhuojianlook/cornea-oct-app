"""Severe-artifact QC: flag OCT scans whose artifacts preprocessing can't fix.

Volume-only metrics (no segmentation needed):
  blank_frac  — fraction of near-black B-scan frames (blink / signal dropout)
  motion      — 1 − median adjacent-frame correlation (motion / registration jumps between B-scans)
  sat_frac    — fraction of saturated voxels (clipped highlights)
  contrast    — (p99−p50)/p99 of foreground (low = washed-out / poor signal)
A scan is flagged if any metric is a strong cohort-relative outlier (robust z on the bad side) or
breaches a hard limit. Designed to catch e.g. CS007OD (user-flagged severe artifacts).
"""
from __future__ import annotations

import numpy as np
import nibabel as nib

import orchestration as orch

HARD = {"blank_frac": 0.20, "motion": 0.55, "sat_frac": 0.05}   # absolute "obviously bad" limits
Z = 3.5                                                          # robust-z outlier cutoff


def _vol_path(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def scan_metrics(cid) -> dict | None:
    p = _vol_path(cid)
    if not p.exists():
        return None
    img = nib.load(str(p))
    zooms = img.header.get_zooms()[:3]
    frame_ax = int(np.argmax(zooms))                 # B-scan / slice axis = coarsest spacing
    v = np.asarray(img.dataobj).astype(np.float32)
    vmax = float(v.max()) or 1.0
    vn = v / vmax
    fm = vn.mean(axis=tuple(a for a in range(3) if a != frame_ax))   # per-frame mean intensity
    med = float(np.median(fm)) or 1e-6
    blank_frac = float((fm < 0.25 * med).mean())
    # adjacent-frame correlation along the frame axis (motion / dropout → low corr)
    fr = np.moveaxis(vn, frame_ax, 0).reshape(vn.shape[frame_ax], -1)
    corrs = []
    step = max(1, fr.shape[1] // 20000)              # subsample columns for speed
    fr = fr[:, ::step]
    for i in range(fr.shape[0] - 1):
        a, b = fr[i], fr[i + 1]
        sa, sb = a.std(), b.std()
        if sa > 1e-6 and sb > 1e-6:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    motion = float(1.0 - np.median(corrs)) if corrs else 1.0
    sat_frac = float((vn >= 0.99).mean())
    p50, p99 = np.percentile(vn[vn > 0.02], [50, 99]) if (vn > 0.02).any() else (0.0, 1.0)
    contrast = float((p99 - p50) / (p99 + 1e-6))
    return {"case": cid, "blank_frac": round(blank_frac, 4), "motion": round(motion, 4),
            "sat_frac": round(sat_frac, 4), "contrast": round(contrast, 4)}


def _rz(x, med, mad):
    return (x - med) / mad if mad > 1e-9 else 0.0


def flag_cohort(metrics: list[dict]) -> list[dict]:
    """Add `artifact` (bool) + `reasons` to each metrics row using hard limits + robust-z outliers."""
    if not metrics:
        return metrics
    arr = {k: np.array([m[k] for m in metrics], float) for k in ("blank_frac", "motion", "sat_frac", "contrast")}
    stat = {k: (float(np.median(v)), 1.4826 * float(np.median(np.abs(v - np.median(v))))) for k, v in arr.items()}
    for m in metrics:
        reasons = []
        for k in ("blank_frac", "motion", "sat_frac"):
            if m[k] >= HARD[k]:
                reasons.append(f"{k}={m[k]} (hard≥{HARD[k]})")
            elif _rz(m[k], *stat[k]) >= Z:
                reasons.append(f"{k}={m[k]} (z{_rz(m[k], *stat[k]):.1f})")
        # low contrast = bad on the LOW side
        if _rz(m["contrast"], *stat["contrast"]) <= -Z:
            reasons.append(f"contrast={m['contrast']} (low z{_rz(m['contrast'], *stat['contrast']):.1f})")
        m["artifact"] = bool(reasons)
        m["reasons"] = reasons
    return metrics
