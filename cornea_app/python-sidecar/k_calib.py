"""Calibrate an ABSOLUTE depth-conditional threshold k from controls so normal eyes → ~0 scar.

Percentile hysteresis always flags ~the brightest 8% (controls can't reach 0). With a control baseline
we can threshold at z = (v_norm − μ_ctrl(r))/σ_ctrl(r) ≥ k ABSOLUTELY: normal tissue rarely exceeds the
control mean by k·σ, scar does. Sweep k on sample control vs scar scans → pick the k where controls
collapse to ~0 while scars retain a real volume. (seed z≥k, grow connected z≥k−1.0). READ-ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as L
import scar as scar_mod
import normal_baseline as nb

CONTROLS = ["case_cs004_os_v1", "case_cs008_os_v1", "case_cs009_os_v1", "case_cs011_od_v1",
            "case_cs017_os_v1", "case_cs021_os_v1"]
SCARS = ["case_cs001_os_v1", "case_cs010_od_v1", "case_cs015_od_v1", "case_cs003_od_v1", "case_cs009_od_v1"]
KS = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5]


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _zmap(vol, lab, prof):
    cornea, roi, v = scar_mod.cornea_roi_smoothed(vol, lab)
    rd = scar_mod.relative_corneal_depth(cornea, scar_mod._depth_axis(cornea))
    ref = scar_mod._gain_ref(v, roi, rd)
    vn = (v / ref).astype(np.float32)
    mean = np.asarray(prof[0], np.float32); sd = np.asarray(prof[1], np.float32); nb_ = mean.shape[0]
    bins = np.clip((np.nan_to_num(rd) * nb_).astype(int), 0, nb_ - 1)
    z = np.zeros(lab.shape, np.float32)
    z[roi] = (vn[roi] - mean[bins[roi]]) / np.maximum(sd[bins[roi]], 1e-6)
    return cornea, roi, z


def _scar_k(cornea, roi, z, k, vmm3, margin=1.0):
    seed = roi & (z >= k)
    if not seed.any():
        return 0.0
    grow = roi & (z >= k - margin)
    lbl, _ = ndimage.label(grow); keep = set(np.unique(lbl[seed])) - {0}
    m = scar_mod._morph_clean(np.isin(lbl, list(keep)), cornea)
    return float(m.sum()) * vmm3


def main():
    prof = nb.load_profile()
    print(f"{'k':>5} | " + "  ".join(f"{'ctl':>5}" for _ in CONTROLS) + " | " + "  ".join(f"{'scar':>5}" for _ in SCARS))
    cache = {}
    for cid in CONTROLS + SCARS:
        lab = np.rint(np.asarray(nib.load(str(L.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        sp = nib.load(str(_vol(cid))).header.get_zooms()[:3]
        cache[cid] = (*_zmap(vol, lab, prof), float(sp[0] * sp[1] * sp[2]))
    print("        " + "controls (want →0)".center(6 * len(CONTROLS)) + "   " + "scars (want retained)".center(6 * len(SCARS)))
    for k in KS:
        cvals = [_scar_k(*cache[c][:3], k, cache[c][3]) for c in CONTROLS]
        svals = [_scar_k(*cache[c][:3], k, cache[c][3]) for c in SCARS]
        print(f"{k:>5} | " + "  ".join(f"{x:5.2f}" for x in cvals) + " | " + "  ".join(f"{x:5.2f}" for x in svals)
              + f"   ||  ctl mean {np.mean(cvals):.2f}  scar mean {np.mean(svals):.2f}", flush=True)
    print("\nPick the smallest k where control mean ≈ 0 while scar mean stays well above 0.")


if __name__ == "__main__":
    main()
