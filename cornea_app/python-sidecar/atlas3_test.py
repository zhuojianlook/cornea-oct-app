"""Benchmark normal-model granularity: depth → depth×radius → depth×radius×meridian.

Builds robust control atlases at three granularities and compares control-vs-scar separation
(burden A = Σ max(0,z)·vmm³, AUC) AND scar-eye replicate reproducibility (CV across subgroup-1 scans).
Meridian cells with too few samples fall back to the (r,ρ) marginal; θ is circularly smoothed to avoid
over-fitting 21 controls. READ-ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import settings
import labels as L
import scar as scar_mod

NR, NRHO, NTH = 24, 6, 8
MIN = 50


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def coords(cid):
    lab = np.rint(np.asarray(nib.load(str(L.corrected_path(cid))).dataobj)).astype(np.uint8)
    vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
    sp = nib.load(str(_vol(cid))).header.get_zooms()[:3]
    cornea, roi, v = scar_mod.cornea_roi_smoothed(vol, lab)
    if not roi.any():
        return None
    depth = scar_mod._depth_axis(cornea)
    rd = scar_mod.relative_corneal_depth(cornea, depth)
    ref = scar_mod._gain_ref(v, roi, rd)
    vn = (v / ref).astype(np.float32)
    ax = [a for a in range(3) if a != depth]
    idx = np.indices(cornea.shape)
    ci = idx[ax[0]][cornea].mean(); cj = idx[ax[1]][cornea].mean()
    dx = (idx[ax[0]] - ci) * sp[ax[0]]; dy = (idx[ax[1]] - cj) * sp[ax[1]]
    rho = np.sqrt(dx ** 2 + dy ** 2).astype(np.float32)
    rmax = float(np.percentile(rho[cornea], 95)) or 1.0
    th = (np.arctan2(dy, dx).astype(np.float32) + 2 * np.pi) % (2 * np.pi)
    return (vn[roi], np.nan_to_num(rd[roi]), np.clip(rho[roi] / rmax, 0, 1), th[roi],
            float(sp[0] * sp[1] * sp[2]))


def robust(s):
    m = float(np.median(s)); mad = 1.4826 * float(np.median(np.abs(s - m)))
    return m, (mad if mad > 1e-6 else (float(s.std()) or 1.0))


def build(pool, dims):
    """dims: 1=(r,), 2=(r,rho), 3=(r,rho,th). Returns (mean,sd, marg2_mean,marg2_sd) for fallback."""
    vn = np.concatenate([p[0] for p in pool]); r = np.concatenate([p[1] for p in pool])
    rho = np.concatenate([p[2] for p in pool]); th = np.concatenate([p[3] for p in pool])
    br = np.clip((r * NR).astype(int), 0, NR - 1)
    bp = np.clip((rho * NRHO).astype(int), 0, NRHO - 1)
    bt = np.clip((th / (2 * np.pi) * NTH).astype(int), 0, NTH - 1)
    g = float(np.median(vn))
    if dims == 1:
        mean = np.full(NR, g); sd = np.ones(NR)
        for i in range(NR):
            s = vn[br == i]
            if s.size >= MIN: mean[i], sd[i] = robust(s)
        return ("1", mean, sd)
    m2 = np.full((NR, NRHO), g); s2 = np.ones((NR, NRHO))
    for i in range(NR):
        for j in range(NRHO):
            s = vn[(br == i) & (bp == j)]
            if s.size >= MIN: m2[i, j], s2[i, j] = robust(s)
    if dims == 2:
        return ("2", m2, s2)
    m3 = np.zeros((NR, NRHO, NTH)); s3 = np.zeros((NR, NRHO, NTH)); have = np.zeros((NR, NRHO, NTH), bool)
    for i in range(NR):
        for j in range(NRHO):
            for k in range(NTH):
                s = vn[(br == i) & (bp == j) & (bt == k)]
                if s.size >= MIN: m3[i, j, k], s3[i, j, k] = robust(s); have[i, j, k] = True
    # circular smooth across theta (1,2,1), fall back to (r,rho) marginal where missing
    for i in range(NR):
        for j in range(NRHO):
            for k in range(NTH):
                if not have[i, j, k]:
                    m3[i, j, k], s3[i, j, k] = m2[i, j], s2[i, j]
    m3 = (np.roll(m3, 1, 2) + 2 * m3 + np.roll(m3, -1, 2)) / 4.0
    s3 = (np.roll(s3, 1, 2) + 2 * s3 + np.roll(s3, -1, 2)) / 4.0
    return ("3", m3, s3)


def burden(p, mdl):
    vn, r, rho, th, vmm3 = p; dims, mean, sd = mdl
    br = np.clip((r * NR).astype(int), 0, NR - 1)
    if dims == "1":
        z = (vn - mean[br]) / np.maximum(sd[br], 1e-6)
    else:
        bp = np.clip((rho * NRHO).astype(int), 0, NRHO - 1)
        if dims == "2":
            z = (vn - mean[br, bp]) / np.maximum(sd[br, bp], 1e-6)
        else:
            bt = np.clip((th / (2 * np.pi) * NTH).astype(int), 0, NTH - 1)
            z = (vn - mean[br, bp, bt]) / np.maximum(sd[br, bp, bt], 1e-6)
    return float(np.maximum(0, z).sum()) * vmm3


def auc(neg, pos):
    a = np.array(neg); return sum((x > a).sum() + 0.5 * (x == a).sum() for x in pos) / (len(a) * len(pos))


def main():
    ctl, sca = [], []
    eye_sub = defaultdict(list)
    for d in sorted(settings.CASES_ROOT.glob("case_*")):
        cid = d.name; m = orch.read_manifest(cid)
        if m.get("consensus_cases") or not L.corrected_path(cid).exists() or not _vol(cid).exists():
            continue
        c = str(m.get("scar_classification"))
        if c == "control": ctl.append(cid)
        elif c == "scar": sca.append(cid)
        else: continue
        p = cid.split("_"); eye_sub[(f"{p[1]}_{p[2]}", str(m.get("scar_subgroup") or "1"), c)].append(cid)
    print(f"controls {len(ctl)} scars {len(sca)} — loading coords...", flush=True)
    data = {}
    for cid in ctl + sca:
        cc = coords(cid)
        if cc is not None: data[cid] = cc
    pool = [data[c] for c in ctl if c in data]
    for dims in (1, 2, 3):
        mdl = build(pool, dims)
        cb = [burden(data[c], mdl) for c in ctl if c in data]
        sb = [burden(data[c], mdl) for c in sca if c in data]
        cvs = []
        for (eye, sg, cls), cids in eye_sub.items():
            if cls != "scar" or sg != "1": continue
            vals = [burden(data[c], mdl) for c in cids if c in data]
            if len(vals) >= 2 and np.mean(vals) > 1e-9:
                cvs.append(np.std(vals, ddof=1) / np.mean(vals) * 100)
        lbl = {1: "depth", 2: "depth×radius", 3: "depth×radius×meridian"}[dims]
        print(f"  {lbl:24} AUC {auc(cb, sb):.3f}   ctl {np.mean(cb):.2f}  scar {np.mean(sb):.2f}   "
              f"scar-eye CV median {np.median(cvs):.1f}%", flush=True)


if __name__ == "__main__":
    main()
