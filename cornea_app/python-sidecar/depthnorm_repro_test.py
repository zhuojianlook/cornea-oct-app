"""Can depth-normalisation keep hysteresis-level reproducibility?

The fixed-k z-cut in detect_scar_depthnorm reproduces badly (CV ~9-13%). Hypothesis: use PERCENTILE
hysteresis on the depth-conditional z-map (seed at percentile φ of in-stroma z, grow to φ-gap) — the
adaptive, connectivity-grown boundary that made plain hysteresis reproducible — so we get normal-
subtraction specificity AND reproducibility. Compares, on CS001 OS v1/v2/v3 (self-profile):
  fixed-k (k=3)  vs  percentile-hysteresis-on-z (φ=92, gap=12)
by pairwise Dice + native CV + anterior-quarter (Bowman's) share. READ-ONLY.
"""
from __future__ import annotations

import shutil, sys, tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as L
import registration as reg
import scar as scar_mod

CASES = ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"]
TMP = Path(tempfile.mkdtemp(prefix="dnrepro_"))


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _dice(a, b):
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else float("nan")


def _zmap(vol, lab):
    cornea, roi, v = scar_mod.cornea_roi_smoothed(vol, lab)
    depth = scar_mod._depth_axis(cornea)
    rd = scar_mod.relative_corneal_depth(cornea, depth)
    ref = scar_mod._gain_ref(v, roi, rd)
    vn = (v / ref).astype(np.float32)
    mean, sd = scar_mod.depth_profile_stats(vn[roi], rd[roi], scar_mod.NPROF_BINS)
    bins = np.clip((np.nan_to_num(rd) * scar_mod.NPROF_BINS).astype(int), 0, scar_mod.NPROF_BINS - 1)
    z = np.zeros(lab.shape, np.float32)
    z[roi] = (vn[roi] - mean[bins[roi]]) / np.maximum(sd[bins[roi]], 1e-6)
    return cornea, roi, z, rd


def _fixed_k(cornea, roi, z, k=3.0, gap_z=1.5):
    seed = roi & (z >= k); grow = roi & (z >= k - gap_z)
    lbl, _ = ndimage.label(grow); keep = set(np.unique(lbl[seed])) - {0}
    return scar_mod._morph_clean(np.isin(lbl, list(keep)), cornea)


def _pct_hyst(cornea, roi, z, phi=92.0, gap=12.0):
    zr = z[roi]
    thi = float(np.percentile(zr, phi)); tlo = float(np.percentile(zr, max(0.0, phi - gap)))
    seed = roi & (z >= thi); grow = roi & (z >= tlo)
    lbl, _ = ndimage.label(grow); keep = set(np.unique(lbl[seed])) - {0}
    return scar_mod._morph_clean(np.isin(lbl, list(keep)), cornea)


def main():
    ref = CASES[0]; ref_vol = _vol(ref)
    sp = reg._read_vol(ref_vol).GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    tx = {ref: reg.identity()}
    for mov in CASES[1:]:
        tx[mov] = reg.align_transform(ref_vol, L.corrected_path(ref), _vol(mov), L.corrected_path(mov))[0]

    cache = {}
    for cid in CASES:
        lab = np.rint(np.asarray(nib.load(str(L.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        cache[cid] = (lab, *(_zmap(vol, lab)))

    def score(fn, name):
        warped, vols, ant = {}, [], []
        for cid in CASES:
            lab, cornea, roi, z, rd = cache[cid]
            m = fn(cornea, roi, z) & ((lab == 1) | (lab == 2))
            vols.append(float(m.sum()) * vmm3)
            rr = rd[m]; rr = rr[~np.isnan(rr)]
            ant.append(float((rr < 0.25).mean()) if rr.size else 0.0)
            tmp = TMP / f"{cid}.nii.gz"; L.write_label_nifti(m.astype(np.uint8), _vol(cid), tmp)
            warped[cid] = reg.resample_label(tmp, ref_vol, tx[cid]) >= 1
        pair = [_dice(warped[CASES[i]], warped[CASES[j]]) for i in range(3) for j in range(i + 1, 3)]
        mean = float(np.mean(vols)); sd = float(np.std(vols, ddof=1))
        print(f"{name:26} mean {mean:.2f}mm³  CV {sd/mean*100:5.2f}%  pairDice {np.mean(pair):.3f}  "
              f"anterior {np.mean(ant)*100:4.1f}%", flush=True)

    print("depth-normalised scar — thresholding on the z-map (self-profile):")
    score(_fixed_k, "fixed-k (current)")
    score(_pct_hyst, "percentile-hysteresis")
    print("baseline: hysteresis CV 0.60% / Dice 0.79 / anterior ~51% ; depthnorm should cut anterior + keep CV low")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
