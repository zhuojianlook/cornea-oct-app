"""Do control scans improve scar segmentation? Compare default (hysteresis, on-disk label) vs
depth-normalised-with-CONTROL-baseline, per scan, across the cohort.

Controls are KNOWN normal (no scar) → their default "scar" is false-positive (normal Bowman's/anterior
hyper-reflectivity). If the control baseline works, depthnorm should drop control scar toward ~0 while
RETAINING scar on the scar eyes. Reports per-class means + per-control-eye, and the anterior(Bowman's)
share before/after on scar eyes. READ-ONLY.
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
import normal_baseline as nb


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def main():
    prof = nb.load_profile()
    kabs = nb.load_kabs()
    if prof is None:
        print("No control baseline built."); return
    print(f"control-anchored absolute mode: k_abs={kabs:.2f}\n")
    rows = []
    for d in sorted(settings.CASES_ROOT.glob("case_*")):
        cid = d.name
        m = orch.read_manifest(cid)
        if m.get("consensus_cases") or not L.corrected_path(cid).exists() or not _vol(cid).exists():
            continue
        cls = str(m.get("scar_classification"))
        if cls not in ("scar", "control"):
            continue
        lab = np.rint(np.asarray(nib.load(str(L.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        sp = nib.load(str(_vol(cid))).header.get_zooms()[:3]
        vmm3 = float(sp[0] * sp[1] * sp[2])
        cornea = (lab == 1) | (lab == 2)
        default = (lab == 2)
        zres = nb.atlas_z(vol, lab, sp, prof)
        dn = (scar_mod.scar_from_z(zres[0], zres[1], zres[2], k_abs=kabs) if zres is not None
              else np.zeros(lab.shape, bool)) & cornea
        # anterior (Bowman's) share
        rd = scar_mod.relative_corneal_depth(cornea, scar_mod._depth_axis(cornea))
        def ant(mask):
            r = rd[mask]; r = r[~np.isnan(r)]
            return float((r < 0.25).mean()) if r.size else 0.0
        rows.append({"cid": cid, "cls": cls,
                     "default_mm3": round(default.sum() * vmm3, 3),
                     "dn_mm3": round(dn.sum() * vmm3, 3),
                     "ant_default": round(ant(default) * 100, 1),
                     "ant_dn": round(ant(dn) * 100, 1)})
        print(f"  {cid:24}{cls:8} default {rows[-1]['default_mm3']:>7} -> depthnorm {rows[-1]['dn_mm3']:>7} mm³"
              f"   anterior {rows[-1]['ant_default']:>5}% -> {rows[-1]['ant_dn']:>5}%", flush=True)

    for cls in ("control", "scar"):
        sub = [r for r in rows if r["cls"] == cls]
        if not sub:
            continue
        dmean = np.mean([r["default_mm3"] for r in sub]); nmean = np.mean([r["dn_mm3"] for r in sub])
        ad = np.mean([r["ant_default"] for r in sub]); an = np.mean([r["ant_dn"] for r in sub])
        print(f"\n=== {cls} (n={len(sub)}) ===")
        print(f"  scar volume  default {dmean:.2f}  ->  depthnorm {nmean:.2f} mm³   ({100*(1-nmean/dmean):.0f}% change)")
        print(f"  anterior share  default {ad:.0f}%  ->  depthnorm {an:.0f}%")
        if cls == "control":
            near0 = sum(1 for r in sub if r["dn_mm3"] < 0.5)
            print(f"  controls with depthnorm scar < 0.5 mm³ (≈ none): {near0}/{len(sub)}  "
                  f"(default had {sum(1 for r in sub if r['default_mm3'] < 0.5)}/{len(sub)})")
    print("\nGoal: controls → ~0 (false scar removed), scar eyes RETAIN scar with lower anterior/Bowman's share.")


if __name__ == "__main__":
    main()
