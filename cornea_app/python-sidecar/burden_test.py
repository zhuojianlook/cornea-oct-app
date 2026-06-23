"""Prototype: continuous scar BURDEN = integrated reflectivity excess over the control-normal profile,
with NO hard threshold (sidesteps the volume-threshold specificity/sensitivity/reproducibility trilemma).

Per in-stroma voxel z = (v_norm − μ_ctrl(r))/σ_ctrl(r). Candidate burdens (integrated over cornea):
  A relu_z   = Σ max(0, z) · vmm³            (all positive excess; has a normal baseline)
  B floor2_z = Σ max(0, z−2) · vmm³          (only clear excess; controls→low)
  C exc2_int = Σ max(0, v_norm−μ−2σ) · vmm³  (excess reflectivity in normalised-intensity units beyond 2σ)
Evaluates: control vs scar separation (Mann–Whitney AUC) and REPLICATE reproducibility (CV across each
eye's subgroup-1 scans) — vs the thresholded depthnorm volume's reproducibility. READ-ONLY.
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


def burdens(vol, lab, prof, vmm3, floor=2.0):
    cornea, roi, v = scar_mod.cornea_roi_smoothed(vol, lab)
    if not roi.any():
        return {"A": 0.0, "B": 0.0, "C": 0.0}
    rd = scar_mod.relative_corneal_depth(cornea, scar_mod._depth_axis(cornea))
    ref = scar_mod._gain_ref(v, roi, rd)
    vn = (v / ref).astype(np.float32)
    mean = np.asarray(prof[0], np.float32); sd = np.asarray(prof[1], np.float32); nbn = mean.shape[0]
    bins = np.clip((np.nan_to_num(rd) * nbn).astype(int), 0, nbn - 1)
    mu = mean[bins[roi]]; sg = np.maximum(sd[bins[roi]], 1e-6)
    z = (vn[roi] - mu) / sg
    exc = vn[roi] - mu - floor * sg
    return {"A": float(np.maximum(0, z).sum()) * vmm3,
            "B": float(np.maximum(0, z - floor).sum()) * vmm3,
            "C": float(np.maximum(0, exc).sum()) * vmm3}


def auc(neg, pos):
    if not neg or not pos:
        return float("nan")
    a = np.array(neg); b = np.array(pos); g = 0.0
    for x in b:
        g += (x > a).sum() + 0.5 * (x == a).sum()
    return g / (len(a) * len(b))


def main():
    prof = nb.load_profile()
    by = {}
    eye_sub = defaultdict(list)   # (eye,subgroup) -> [cid]
    for d in sorted(settings.CASES_ROOT.glob("case_*")):
        cid = d.name; m = orch.read_manifest(cid)
        if m.get("consensus_cases") or not L.corrected_path(cid).exists() or not _vol(cid).exists():
            continue
        cls = str(m.get("scar_classification"))
        if cls not in ("scar", "control"):
            continue
        lab = np.rint(np.asarray(nib.load(str(L.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        sp = nib.load(str(_vol(cid))).header.get_zooms()[:3]
        by[cid] = (cls, burdens(vol, lab, prof, float(sp[0] * sp[1] * sp[2])))
        parts = cid.split("_"); eye = f"{parts[1]}_{parts[2]}"
        eye_sub[(eye, str(m.get("scar_subgroup") or "1"))].append(cid)
        print(f"  {cid:24}{cls:8} A={by[cid][1]['A']:.3f} B={by[cid][1]['B']:.3f} C={by[cid][1]['C']:.4f}", flush=True)

    for key in ("A", "B", "C"):
        ctl = [v[1][key] for v in by.values() if v[0] == "control"]
        sca = [v[1][key] for v in by.values() if v[0] == "scar"]
        print(f"\n=== burden {key} ===  control mean {np.mean(ctl):.3f}  scar mean {np.mean(sca):.3f}  "
              f"AUC(control<scar) {auc(ctl, sca):.3f}")
        # reproducibility: CV across each eye's subgroup-1 replicates (>=2)
        cvs = []
        for (eye, sg), cids in eye_sub.items():
            if sg != "1" or len(cids) < 2:
                continue
            vals = [by[c][1][key] for c in cids if c in by]
            if len(vals) >= 2 and np.mean(vals) > 1e-9:
                cvs.append(np.std(vals, ddof=1) / np.mean(vals) * 100)
        if cvs:
            print(f"    replicate CV%: median {np.median(cvs):.1f}  [{np.min(cvs):.1f}, {np.max(cvs):.1f}]  (n={len(cvs)} eyes)")
    print("\nWant: AUC→1 (controls separate from scars) + LOW replicate CV (reproducible burden).")


if __name__ == "__main__":
    main()
