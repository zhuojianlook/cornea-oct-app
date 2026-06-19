"""How much of the replicate-scar disagreement is a SMALL RESIDUAL SHIFT vs genuine boundary variability?

Aligns CS001 OS v1/v2/v3 (same as the consensus), warps the hysteresis scars into the ref frame, then
re-scores pairwise overlap with a BOUNDARY TOLERANCE d (mm): a scar voxel "agrees" if the other scan
has scar within d of it. Strict Dice = d=0. If overlap jumps to ~1 at a tiny d, the gap is a small
shift (registration/sampling); if it stays low, it is genuine shape disagreement. READ-ONLY.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as label_mod
import scar as scar_mod
import registration as reg

REPLICATES = ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"]
TMP = Path(tempfile.mkdtemp(prefix="tol_diag_"))
TOLS_MM = [0.0, 0.02, 0.05, 0.10, 0.15]


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _tol_dice(a, b, sampling, d):
    """Boundary-tolerant Dice: a voxel counts as matched if the other mask is within d mm."""
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    if d == 0:
        inter = (a & b).sum()
        return 2.0 * inter / (a.sum() + b.sum())
    db = ndimage.distance_transform_edt(~b, sampling=sampling)
    da = ndimage.distance_transform_edt(~a, sampling=sampling)
    a_in = (db[a] <= d).sum()   # A voxels within d of B
    b_in = (da[b] <= d).sum()   # B voxels within d of A
    return float(a_in + b_in) / float(a.sum() + b.sum())


def main():
    ref = REPLICATES[0]
    ref_vol = _vol(ref)
    ref_img = reg._read_vol(ref_vol)
    sp = ref_img.GetSpacing()
    samp = (sp[2], sp[1], sp[0])
    print(f"voxel spacing (mm): lateral≈{sp[0]:.4f}  depth≈{sp[1]:.5f}  frame≈{sp[2]:.3f}", flush=True)
    tx = {ref: reg.identity()}
    for mov in REPLICATES[1:]:
        tx[mov] = reg.align_transform(ref_vol, label_mod.corrected_path(ref), _vol(mov), label_mod.corrected_path(mov))[0]

    warped = {}
    for cid in REPLICATES:
        lab = np.rint(np.asarray(nib.load(str(label_mod.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        m = scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=92) & ((lab == 1) | (lab == 2))
        tmp = TMP / f"{cid}.nii.gz"
        label_mod.write_label_nifti(m.astype(np.uint8), _vol(cid), tmp)
        warped[cid] = reg.resample_label(tmp, ref_vol, tx[cid]) >= 1

    pairs = [(REPLICATES[i], REPLICATES[j]) for i in range(3) for j in range(i + 1, 3)]
    print("\nBoundary-tolerant pairwise scar Dice (d=0 is the strict Dice you already saw):")
    print(f"{'tolerance d (mm)':>16}" + "".join(f"{p[0].split('_')[-1]+'/'+p[1].split('_')[-1]:>10}" for p in pairs) + f"{'mean':>9}")
    for d in TOLS_MM:
        vals = [_tol_dice(warped[x], warped[y], samp, d) for x, y in pairs]
        print(f"{d:>16.2f}" + "".join(f"{v:>10.3f}" for v in vals) + f"{np.mean(vals):>9.3f}", flush=True)

    # How the agreement TIERS shift under a small tolerance: fraction of the "fringe" (scar in only
    # 1 or 2 scans) that lies within d of ALL three scans — i.e. would become core if d-aligned.
    stack = np.stack([warped[c] for c in REPLICATES])
    votes = stack.sum(0)
    fringe = (votes >= 1) & (votes < 3)
    print(f"\nStrict tiers (voxels): 1-scan={int((votes==1).sum())}  2-scan={int((votes==2).sum())}  "
          f"3-scan(core)={int((votes==3).sum())}", flush=True)
    dists = [ndimage.distance_transform_edt(~warped[c], sampling=samp) for c in REPLICATES]
    print("Fraction of FRINGE voxels within d of ALL 3 scans' scar (→ would be core under that tolerance):")
    for d in TOLS_MM[1:]:
        within_all = np.ones(votes.shape, bool)
        for dt in dists:
            within_all &= (dt <= d)
        frac = float((fringe & within_all).sum()) / float(fringe.sum() or 1)
        print(f"   d={d:.2f}mm: {frac*100:5.1f}% of fringe", flush=True)

    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
