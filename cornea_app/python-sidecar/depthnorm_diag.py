"""Does depth-normalised scar detection reduce anterior/Bowman's over-detection vs hysteresis?

On CS001 OS v1/v2/v3 (self-normalised, no controls needed yet): compare hysteresis vs depthnorm scar by
total volume + how the scar voxels distribute across relative corneal depth (anterior r<0.25 ≈ Bowman's
region · mid · posterior). A drop in the anterior fraction = less Bowman's false scar. READ-ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as L
import scar as scar_mod

CASES = ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"]


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def depth_bands(mask, rdepth):
    rr = rdepth[mask]
    rr = rr[~np.isnan(rr)]
    if rr.size == 0:
        return (0.0, 0.0, 0.0)
    return (float((rr < 0.25).mean()), float(((rr >= 0.25) & (rr <= 0.75)).mean()), float((rr > 0.75).mean()))


def main():
    print(f"{'case':16}{'method':12}{'mm3':>8}{'ant%':>7}{'mid%':>7}{'post%':>7}")
    for cid in CASES:
        lab = np.rint(np.asarray(nib.load(str(L.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        sp = nib.load(str(_vol(cid))).header.get_zooms()
        vmm3 = float(sp[0] * sp[1] * sp[2])
        cornea = (lab == 1) | (lab == 2)
        rdepth = scar_mod.relative_corneal_depth(cornea, scar_mod._depth_axis(cornea))
        for name, fn in (("hysteresis", lambda: scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=92)),
                         ("depthnorm", lambda: scar_mod.detect_scar_depthnorm(vol, lab))):
            m = fn() & cornea
            a, mid, p = depth_bands(m, rdepth)
            print(f"{cid.split('_')[-1]:16}{name:12}{m.sum()*vmm3:>8.2f}{a*100:>7.1f}{mid*100:>7.1f}{p*100:>7.1f}",
                  flush=True)
        print()
    print("ant% = fraction of scar voxels in the anterior quarter (Bowman's region); lower for depthnorm = "
          "less normal-Bowman's flagged.")


if __name__ == "__main__":
    main()
