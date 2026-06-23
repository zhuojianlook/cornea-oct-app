"""Render a visual comparison of scar detectors + replicate reproducibility (CS001 OS) to PNG.

Row 1: the SAME B-scan of v1 with each detector's scar contour overlaid (hysteresis / normal_anchor /
       robust_mad) — shows how the boundary/extent differs between strategies on one scan.
Row 2: the matched B-scan of v1 / v2 / v3 with the hysteresis scar overlaid — shows replicate
       reproducibility (the boundary should land in nearly the same place across the three scans).
Saved to /tmp/scar_compare.png for inspection. READ-ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as label_mod
import scar as scar_mod

REPLICATES = ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"]


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _load(cid):
    lab = np.rint(np.asarray(nib.load(str(label_mod.corrected_path(cid))).dataobj)).astype(np.uint8)
    vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
    return lab, vol


def _contour(ax, sl, scar2d, color):
    ax.imshow(sl.T, cmap="gray", origin="lower", aspect="auto")
    if scar2d.any():
        ax.contour(scar2d.T, levels=[0.5], colors=[color], linewidths=0.9)


DETS = {
    "hysteresis": lambda lab, vol: scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=92),
    "normal_anchor": lambda lab, vol: scar_mod.detect_scar_normal_anchor(vol, lab, k=2.0),
    "robust_mad": lambda lab, vol: scar_mod.detect_scar_robust_mad(vol, lab, k=0.6),
}


def main():
    lab1, vol1 = _load(REPLICATES[0])
    masks1 = {n: fn(lab1, vol1) & ((lab1 == 1) | (lab1 == 2)) for n, fn in DETS.items()}
    # choose the frame (axis 0) with the most hysteresis scar on v1
    f = int(np.argmax(masks1["hysteresis"].sum(axis=(1, 2))))
    print(f"frame={f}  vol1 shape={vol1.shape}", flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (name, m) in zip(axes[0], masks1.items()):
        _contour(ax, vol1[f], m[f], "#ff3b3b")
        ax.set_title(f"v1 · {name}  ({int(m.sum())} vox)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # row 2: hysteresis scar on each replicate at its own max-scar frame
    for ax, cid in zip(axes[1], REPLICATES):
        lab, vol = _load(cid)
        m = scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=92) & ((lab == 1) | (lab == 2))
        ff = int(np.argmax(m.sum(axis=(1, 2))))
        _contour(ax, vol[ff], m[ff], "#33ff66")
        ax.set_title(f"{cid.split('_')[-1]} · hysteresis (f={ff}, {int(m.sum())} vox)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Scar detectors (row 1, v1) + hysteresis replicate reproducibility (row 2)", fontsize=12)
    fig.tight_layout()
    out = "/tmp/scar_compare.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
