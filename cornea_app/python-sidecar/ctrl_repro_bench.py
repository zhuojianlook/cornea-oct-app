"""Optimize control integration: control-corrected + ALIGNED replicate scars should MATCH.

Reuses each eye's consensus warped artifacts (scans/<cid>/volume.nii.gz + label.nii.gz, already in the
reference frame), computes the control-atlas z once per scan, then grid-searches the operating point
(absolute k, hysteresis margin, z-smoothing) for the best replicate agreement:
  scar eyes  → pairwise scar Dice ↑ + volume CV ↓  (replicates should match)
  control eyes → mean scar volume ≈ 0             (specificity preserved)
Picks the config with the best scar replicate-match subject to controls staying ~0. READ-ONLY.
"""
from __future__ import annotations

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import settings
import scar as scar_mod
import normal_baseline as nb

# (k_abs, margin, z_smooth_sigma) — refine around the specific seed (k=4.82), widen the hysteresis grow
CONFIGS = [
    (4.82, 1.0, 0.0),   # current default (ref)
    (4.82, 2.5, 0.0),
    (4.82, 3.0, 0.0),
    (4.82, 3.5, 0.0),
    (4.82, 3.0, 1.0),
    (4.82, 3.5, 1.0),
    (5.20, 3.5, 1.0),   # even more specific seed + wide grow + smoothing
]


def _dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else float("nan")


def main():
    atlas = nb.load_profile()
    if atlas is None:
        print("No atlas."); return
    # per consensus case: cache (z, cornea, roi, vmm3, cls) for each member, in the ref frame
    eyes = {}   # eye -> dict(cls, members=[(z,cornea,roi)], vmm3)
    for d in sorted(settings.CASES_ROOT.glob("case_*_consensus")):
        ccid = d.name
        members = orch.read_manifest(ccid).get("consensus_cases") or []
        cls = "control" if any(str(orch.read_manifest(c).get("scar_classification")) == "control" for c in members) else "scar"
        vol_img = nib.load(str(d / "previews" / "volume.nii.gz"))
        sp = vol_img.header.get_zooms()[:3]; vmm3 = float(sp[0] * sp[1] * sp[2])
        recs = []
        for c in members:
            vp = d / "scans" / c / "volume.nii.gz"; lp = d / "scans" / c / "label.nii.gz"
            if not (vp.exists() and lp.exists()):
                continue
            wv = np.asarray(nib.load(str(vp)).dataobj).astype(np.float32)
            wl = np.rint(np.asarray(nib.load(str(lp)).dataobj)).astype(np.uint8)
            zres = nb.atlas_z(wv, wl, sp, atlas)
            if zres is not None:
                recs.append(zres)   # (z, cornea, roi)
        if len(recs) >= 2:
            eyes[ccid.replace("case_", "").replace("_consensus", "")] = {"cls": cls, "recs": recs, "vmm3": vmm3}
        print(f"  cached {ccid} ({cls}, {len(recs)} scans)", flush=True)

    print(f"\n{'config (k,margin,zσ)':22}{'scarDice':>9}{'scarCV%':>9}{'ctlVol mm³':>11}{'ctl<0.2':>9}")
    for (k, margin, zs) in CONFIGS:
        scar_dices, scar_cvs, ctl_vols = [], [], []
        for eye, e in eyes.items():
            masks, vols = [], []
            for (z, cornea, roi) in e["recs"]:
                zz = ndimage.gaussian_filter(z, zs) if zs > 0 else z
                m = scar_mod.scar_from_z(zz, cornea, roi, k_abs=k, margin=margin)
                masks.append(m); vols.append(float(m.sum()) * e["vmm3"])
            if e["cls"] == "scar":
                pair = [_dice(masks[i], masks[j]) for i in range(len(masks)) for j in range(i + 1, len(masks))]
                pair = [p for p in pair if p == p]
                if pair:
                    scar_dices.append(np.mean(pair))
                mu = np.mean(vols)
                if mu > 1e-9:
                    scar_cvs.append(np.std(vols, ddof=1) / mu * 100)
            else:
                ctl_vols.extend(vols)
        print(f"({k},{margin},{zs})".ljust(22)
              + f"{np.median(scar_dices):>9.3f}{np.median(scar_cvs):>9.1f}"
              + f"{np.mean(ctl_vols):>11.3f}{sum(1 for v in ctl_vols if v < 0.2)}/{len(ctl_vols)}".rjust(9), flush=True)
    print("\nWant: scarDice↑ scarCV↓ with ctlVol≈0. (replicate scars match after control-correction+alignment)")


if __name__ == "__main__":
    main()
