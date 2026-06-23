"""Normal corneal reflectivity baseline from CONTROL scans (no scar).

A normal cornea is intrinsically hyper-reflective at certain relative depths (anterior epithelium/
Bowman's, posterior Descemet's/endothelium) and near the specular apex — which an absolute brightness
threshold over-flags as scar. This module pools gain-normalised reflectivity by RELATIVE CORNEAL DEPTH
across control scans into a normal profile (mean[],sd[]); `scar.detect_scar_depthnorm` then flags scar
only as EXCESS over that normal profile, removing the Bowman's-region over-detection.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib

import orchestration as orch
import settings
import labels as L
import scar as scar_mod

PROFILE_PATH = settings.WORKSPACE_ROOT / "output" / "normal_profile.json"
CONTROL_CLASSES = {"control", "normal"}
NR, NRHO, NTH, MINCELL = 24, 6, 8, 50   # 3-D normal atlas: depth × en-face radius × meridian


def _vol_path(cid: str) -> Path:
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _robust(s):
    m = float(np.median(s)); mad = 1.4826 * float(np.median(np.abs(s - m)))
    return m, (mad if mad > 1e-6 else (float(s.std()) or 1.0))


def _scan_coords(cid):
    """(vn[roi], r[roi], rho[roi], theta[roi]) gain-normalised in-cornea samples for the atlas."""
    lab = np.rint(np.asarray(nib.load(str(L.corrected_path(cid))).dataobj)).astype(np.uint8)
    vol = np.asarray(nib.load(str(_vol_path(cid))).dataobj).astype(np.float32)
    sp = nib.load(str(_vol_path(cid))).header.get_zooms()[:3]
    cornea, roi, v = scar_mod.cornea_roi_smoothed(vol, lab)
    if not roi.any():
        return None
    depth = scar_mod._depth_axis(cornea)
    rd = scar_mod.relative_corneal_depth(cornea, depth)
    rho, theta = scar_mod.enface_coords(cornea, depth, sp)
    ref = scar_mod._gain_ref(v, roi, rd)
    return (v[roi] / ref).astype(np.float32), np.nan_to_num(rd[roi]), rho[roi], theta[roi]


def _build_atlas(pool):
    """Robust 3-D normal atlas (mean,sd over depth×radius×meridian) from control samples, with empty
    meridian cells filled from the (depth,radius) marginal and circular θ-smoothing (avoids over-fitting
    a finite control set)."""
    vn = np.concatenate([p[0] for p in pool]); r = np.concatenate([p[1] for p in pool])
    rho = np.concatenate([p[2] for p in pool]); th = np.concatenate([p[3] for p in pool])
    br = np.clip((r * NR).astype(int), 0, NR - 1)
    bp = np.clip((rho * NRHO).astype(int), 0, NRHO - 1)
    bt = np.clip((th / (2 * np.pi) * NTH).astype(int), 0, NTH - 1)
    g = float(np.median(vn))
    m2 = np.full((NR, NRHO), g); s2 = np.ones((NR, NRHO))
    for i in range(NR):
        for j in range(NRHO):
            s = vn[(br == i) & (bp == j)]
            if s.size >= MINCELL:
                m2[i, j], s2[i, j] = _robust(s)
    m3 = np.empty((NR, NRHO, NTH)); s3 = np.empty((NR, NRHO, NTH))
    for i in range(NR):
        for j in range(NRHO):
            for k in range(NTH):
                s = vn[(br == i) & (bp == j) & (bt == k)]
                if s.size >= MINCELL:
                    m3[i, j, k], s3[i, j, k] = _robust(s)
                else:
                    m3[i, j, k], s3[i, j, k] = m2[i, j], s2[i, j]
    m3 = (np.roll(m3, 1, 2) + 2 * m3 + np.roll(m3, -1, 2)) / 4.0
    s3 = (np.roll(s3, 1, 2) + 2 * s3 + np.roll(s3, -1, 2)) / 4.0
    return m3, s3


def control_cases() -> list[str]:
    """Labelled control cases: scar_classification ∈ {control,normal}, has a cornea labelmap + volume,
    and is not a consensus case."""
    out: list[str] = []
    if not settings.CASES_ROOT.exists():
        return out
    for d in sorted(settings.CASES_ROOT.iterdir()):
        if not d.is_dir():
            continue
        cid = d.name
        m = orch.read_manifest(cid)
        if m.get("consensus_cases"):
            continue
        if str(m.get("scar_classification") or "").lower() in CONTROL_CLASSES \
                and L.corrected_path(cid).exists() and _vol_path(cid).exists():
            out.append(cid)
    return out


def build_profile(case_ids: list[str] | None = None) -> dict:
    """Build the 3-D normal atlas (depth × en-face radius × meridian) of gain-normalised reflectivity
    from the control scans, calibrate the absolute threshold k_abs (controls→~0 scar), and persist.
    Benchmarked: control-vs-scar AUC 0.805 (depth) → 0.846 (+radius) → 0.862 (+meridian)."""
    cids = list(case_ids) if case_ids is not None else control_cases()
    if not cids:
        raise ValueError("No labelled control scans. Tag scans 'control', segment the cornea, then rebuild.")
    pool, used = [], []
    for cid in cids:
        try:
            c = _scan_coords(cid)
            if c is not None:
                pool.append(c); used.append(cid)
        except Exception as exc:  # noqa: BLE001
            print(f"[normal_baseline] skipped {cid}: {exc}")
    if not pool:
        raise ValueError("Could not read any control scans (need a cornea labelmap + volume).")
    mean, sd = _build_atlas(pool)
    # Calibrate k_abs = high percentile of the controls' own atlas z (normal almost never exceeds it).
    zc = []
    for vn, r, rho, th in pool:
        br = np.clip((r * NR).astype(int), 0, NR - 1)
        bp = np.clip((rho * NRHO).astype(int), 0, NRHO - 1)
        bt = np.clip((th / (2 * np.pi) * NTH).astype(int), 0, NTH - 1)
        zc.append((vn - mean[br, bp, bt]) / np.maximum(sd[br, bp, bt], 1e-6))
    k_abs = float(np.percentile(np.concatenate(zc), 99.7))
    save_profile(mean, sd, used, k_abs)
    return {"controls": used, "n_controls": len(used), "dims": [NR, NRHO, NTH], "k_abs": round(k_abs, 3)}


def save_profile(mean: np.ndarray, sd: np.ndarray, controls: list[str], k_abs: float = 4.0) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps({
        "dims": [int(NR), int(NRHO), int(NTH)],
        "mean": [float(x) for x in np.asarray(mean).ravel()],
        "sd": [float(x) for x in np.asarray(sd).ravel()],
        "k_abs": float(k_abs),
        "controls": controls,
    }))


def load_profile():
    """3-D atlas dict {mean,sd shaped (NR,NRHO,NTH)} for atlas_z, or None if no baseline built yet."""
    if not PROFILE_PATH.exists():
        return None
    d = json.loads(PROFILE_PATH.read_text())
    dims = tuple(d.get("dims", [NR, NRHO, NTH]))
    return {"dims": dims,
            "mean": np.asarray(d["mean"], np.float32).reshape(dims),
            "sd": np.asarray(d["sd"], np.float32).reshape(dims)}


def load_kabs():
    """Control-calibrated absolute z threshold (controls→~0 scar), or None if no baseline."""
    if not PROFILE_PATH.exists():
        return None
    return float(json.loads(PROFILE_PATH.read_text()).get("k_abs", 4.0))


def atlas_z(vol_ijk, lab_ijk, spacing, atlas=None):
    """Per-voxel z = (v_norm − μ_atlas(r,ρ,θ)) / σ_atlas(r,ρ,θ). Returns (z, cornea, roi) or None."""
    atlas = atlas or load_profile()
    if atlas is None:
        return None
    cornea, roi, v = scar_mod.cornea_roi_smoothed(vol_ijk, lab_ijk)
    if not roi.any():
        return None
    depth = scar_mod._depth_axis(cornea)
    rd = scar_mod.relative_corneal_depth(cornea, depth)
    rho, theta = scar_mod.enface_coords(cornea, depth, spacing)
    ref = scar_mod._gain_ref(v, roi, rd)
    vn = (v / ref).astype(np.float32)
    nr, nrho, nth = atlas["dims"]; mean, sd = atlas["mean"], atlas["sd"]
    br = np.clip((np.nan_to_num(rd) * nr).astype(int), 0, nr - 1)
    bp = np.clip((rho * nrho).astype(int), 0, nrho - 1)
    bt = np.clip((theta / (2 * np.pi) * nth).astype(int), 0, nth - 1)
    z = np.zeros(lab_ijk.shape, np.float32)
    z[roi] = (vn[roi] - mean[br[roi], bp[roi], bt[roi]]) / np.maximum(sd[br[roi], bp[roi], bt[roi]], 1e-6)
    return z, cornea, roi


def profile_info() -> dict:
    if not PROFILE_PATH.exists():
        return {"exists": False, "controls": [], "available_controls": control_cases()}
    d = json.loads(PROFILE_PATH.read_text())
    return {"exists": True, "dims": d.get("dims"), "controls": d.get("controls", []),
            "available_controls": control_cases()}
