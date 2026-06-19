"""Does a richer normal model (depth × en-face radius) beat the 1-D depth profile?

Builds two control normal models and compares control-vs-scar separation (burden A = Σ max(0,z)·vmm³,
AUC) + control suppression:
  1D:  μ(r),σ(r)              — current (relative depth only)
  2D:  μ(r,ρ),σ(r,ρ)          — adds normalized en-face radius ρ from the cornea's en-face centroid
                                (captures apex-specular vs peripheral falloff that 1-D ignores)
If 2D raises AUC / cleans controls better, the 3-D-atlas direction is worth building; if not, the
~0.80 ceiling is data-limited. READ-ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import settings
import labels as L
import scar as scar_mod

NR, NRHO = 24, 6


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def coords(cid):
    """Return (in-cornea) gain-normalised intensity, relative depth r, en-face radius ρ∈[0,1], vmm³."""
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
    rho = np.sqrt(((idx[ax[0]] - ci) * sp[ax[0]]) ** 2 + ((idx[ax[1]] - cj) * sp[ax[1]]) ** 2).astype(np.float32)
    rmax = float(np.percentile(rho[cornea], 95)) or 1.0
    rho_n = np.clip(rho / rmax, 0, 1)
    return vn[roi], np.nan_to_num(rd[roi]), rho_n[roi], float(sp[0] * sp[1] * sp[2])


def robust(vals):
    med = float(np.median(vals)); mad = 1.4826 * float(np.median(np.abs(vals - med)))
    return med, (mad if mad > 1e-6 else (float(vals.std()) or 1.0))


def build_1d(pool):
    mean = np.zeros(NR); sd = np.ones(NR)
    vn = np.concatenate([p[0] for p in pool]); r = np.concatenate([p[1] for p in pool])
    b = np.clip((r * NR).astype(int), 0, NR - 1)
    for i in range(NR):
        s = vn[b == i]
        if s.size >= 50:
            mean[i], sd[i] = robust(s)
    return mean, sd


def build_2d(pool):
    mean = np.zeros((NR, NRHO)); sd = np.ones((NR, NRHO))
    vn = np.concatenate([p[0] for p in pool]); r = np.concatenate([p[1] for p in pool]); rho = np.concatenate([p[2] for p in pool])
    br = np.clip((r * NR).astype(int), 0, NR - 1); bp = np.clip((rho * NRHO).astype(int), 0, NRHO - 1)
    for i in range(NR):
        for j in range(NRHO):
            s = vn[(br == i) & (bp == j)]
            if s.size >= 50:
                mean[i, j], sd[i, j] = robust(s)
    return mean, sd


def burden_1d(p, mdl):
    vn, r, rho, vmm3 = p; mean, sd = mdl
    b = np.clip((r * NR).astype(int), 0, NR - 1)
    z = (vn - mean[b]) / np.maximum(sd[b], 1e-6)
    return float(np.maximum(0, z).sum()) * vmm3


def burden_2d(p, mdl):
    vn, r, rho, vmm3 = p; mean, sd = mdl
    br = np.clip((r * NR).astype(int), 0, NR - 1); bp = np.clip((rho * NRHO).astype(int), 0, NRHO - 1)
    z = (vn - mean[br, bp]) / np.maximum(sd[br, bp], 1e-6)
    return float(np.maximum(0, z).sum()) * vmm3


def auc(neg, pos):
    a = np.array(neg); g = sum((x > a).sum() + 0.5 * (x == a).sum() for x in pos)
    return g / (len(a) * len(pos))


def main():
    ctl_cids, sca_cids = [], []
    for d in sorted(settings.CASES_ROOT.glob("case_*")):
        cid = d.name; m = orch.read_manifest(cid)
        if m.get("consensus_cases") or not L.corrected_path(cid).exists() or not _vol(cid).exists():
            continue
        c = str(m.get("scar_classification"))
        (ctl_cids if c == "control" else sca_cids if c == "scar" else []).append(cid)
    print(f"controls {len(ctl_cids)}  scars {len(sca_cids)}  — loading coords...", flush=True)
    data = {}
    for cid in ctl_cids + sca_cids:
        c = coords(cid)
        if c is not None:
            data[cid] = c
    ctl = [c for c in ctl_cids if c in data]
    pool = [data[c] for c in ctl]
    m1, m2 = build_1d(pool), build_2d(pool)
    for name, bfn, mdl in [("1D depth", burden_1d, m1), ("2D depth×radius", burden_2d, m2)]:
        cb = [bfn(data[c], mdl) for c in ctl]
        sb = [bfn(data[c], mdl) for c in sca_cids if c in data]
        print(f"  {name:16}: control burden {np.mean(cb):.2f}  scar burden {np.mean(sb):.2f}  "
              f"AUC {auc(cb, sb):.3f}", flush=True)
    print("\nIf 2D AUC >> 1D AUC, en-face conditioning helps → build the full normal atlas; else data-limited.")


if __name__ == "__main__":
    main()
