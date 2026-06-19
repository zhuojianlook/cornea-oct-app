"""SAM2 cornea-segmentation reproducibility benchmark on replicate scans (CS001 OS v1/v2/v3).

Replicates should yield near-identical cornea → high pairwise 3D cornea Dice + low cornea-volume CV%.
`segment_volume` runs SAM2 per plane then majority-votes, so we segment each plane ONCE per scan, then
sweep the fusion knobs (vote threshold × which planes) for free and score each config's reproducibility
(align the 3 to the reference with the SAME transform, then pairwise cornea Dice; volume CV from native
masks). READ-ONLY on the real cases (cornea masks in memory; temp labelmaps in /tmp). Run with the
SIDECAR python (has SAM2), GPU.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import labels as label_mod
import registration as reg
import sam2_segment

REPLICATES = ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"]
PLANES = ("axial", "coronal", "sagittal")
WORK = Path("/tmp/cornea_bench_work")
TMP = Path(tempfile.mkdtemp(prefix="cornea_bench_lab_"))

CONFIGS = [  # (name, planes, vote)
    ("all3 · vote1 (union)", PLANES, 1),
    ("all3 · vote2 (current)", PLANES, 2),
    ("all3 · vote3 (unanimous)", PLANES, 3),
    ("axial only", ("axial",), 1),
    ("coronal only", ("coronal",), 1),
    ("sagittal only", ("sagittal",), 1),
    ("axial+coronal · v2", ("axial", "coronal"), 2),
    ("axial+sagittal · v2", ("axial", "sagittal"), 2),
]


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else float("nan")


def segment_all_planes(cid):
    """The 3 per-plane SAM2 cornea masks (mirrors segment_volume's preprocessing)."""
    raw = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
    vol = gaussian_filter(raw, sigma=(1.0, 1.0, 0.4))
    out = {}
    for pl in PLANES:
        m, _ = sam2_segment.segment_plane(vol, pl, WORK / cid)
        out[pl] = m.astype(np.uint8)
        sam2_segment._free_gpu()
    return out


def fuse(masks, planes, vote):
    """Majority-vote the chosen planes → largest 3D component + fill holes (as segment_volume)."""
    votes = np.zeros(next(iter(masks.values())).shape, np.uint8)
    for pl in planes:
        votes += masks[pl]
    fused = votes >= vote
    lbl, n = ndimage.label(fused)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        fused = lbl == int(np.argmax(sizes)) + 1
    return ndimage.binary_fill_holes(fused)


def main():
    ref = REPLICATES[0]
    ref_vol = _vol(ref)
    ref_img = reg._read_vol(ref_vol)
    sp = ref_img.GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    # transforms (volume-intensity driven; config-independent) — compute once
    tx = {ref: reg.identity()}
    for mov in REPLICATES[1:]:
        tx[mov] = reg.align_transform(ref_vol, label_mod.corrected_path(ref), _vol(mov), label_mod.corrected_path(mov))[0]
        print(f"  aligned {mov}", flush=True)

    print("Segmenting each plane (SAM2) per scan — GPU, ~1-2 min/scan...", flush=True)
    planes_cache = {}
    for cid in REPLICATES:
        t = time.time()
        planes_cache[cid] = segment_all_planes(cid)
        print(f"  {cid}: planes done in {time.time()-t:.0f}s "
              f"(voxels {{ {', '.join(f'{p}:{int(planes_cache[cid][p].sum())}' for p in PLANES)} }})", flush=True)

    def warp(mask, cid):
        tmp = TMP / f"{cid}.nii.gz"
        label_mod.write_label_nifti(mask.astype(np.uint8), _vol(cid), tmp)
        return reg.resample_label(tmp, ref_vol, tx[cid]) >= 1

    print("\n================ SAM2 CORNEA REPRODUCIBILITY (CS001 OS v1/v2/v3) ================")
    print(f"{'config':28} {'mean mm³':>9} {'CV%':>7} {'pairwise cornea Dice':>22}")
    results = []
    for name, planes, vote in CONFIGS:
        try:
            warped, vols = {}, []
            for cid in REPLICATES:
                cornea = fuse(planes_cache[cid], planes, vote)
                vols.append(float(cornea.sum()) * vmm3)
                warped[cid] = warp(cornea, cid)
            cids = REPLICATES
            pair = [round(_dice(warped[cids[i]], warped[cids[j]]), 3) for i in range(3) for j in range(i + 1, 3)]
            mean = float(np.mean(vols)); std = float(np.std(vols, ddof=1))
            cv = round(std / mean * 100, 2) if mean else 0.0
            md = round(float(np.mean(pair)), 3)
            results.append((name, round(mean, 2), cv, md, pair))
            print(f"{name:28} {mean:>9.1f} {cv:>7.2f} {md:>10}  {pair}", flush=True)
        except Exception as exc:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"{name:28} ERROR {str(exc)[:50]}")

    print("\n--- ranked by reproducibility (cornea Dice ↓, then CV ↑) ---")
    for name, mean, cv, md, pair in sorted(results, key=lambda r: (-r[3], r[2])):
        print(f"  {name:28} cornea Dice {md}  CV {cv}%  vol {mean} mm³")
    print("\nGoal: HIGH cornea Dice + LOW CV% (replicates → same cornea).")
    shutil.rmtree(WORK, ignore_errors=True)
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
