"""Size-matched threshold sweep for the scar detectors on CS001 OS replicates.

The plain benchmark (scar_bench.py) found normal_anchor has the highest pairwise Dice (~0.88) but also
~2-4x the volume of hysteresis — and Dice rises mechanically with mask size. This sweep varies each
detector's threshold across a common VOLUME range and reports pairwise Dice + HD95 + CV at each point,
so Dice can be compared AT MATCHED VOLUME (the size confound removed). Whichever family sits highest on
the Dice-vs-volume curve (and lowest on HD95/CV) is genuinely the more reproducible boundary, not just
the more inclusive threshold. READ-ONLY; reuses align-once + warp infra. CPU, ~3 min.
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
TMP = Path(tempfile.mkdtemp(prefix="scar_sweep_"))

# (family, param-label, callable(lab, vol)) swept across a threshold range
SWEEPS = {
    "brightness":    [("pct", p, (lambda lab, vol, p=p: scar_mod.detect_scar_in_cornea(vol, lab, percentile=p)))
                      for p in (84, 87, 90, 92, 94, 96)],
    "hysteresis":    [("phi", p, (lambda lab, vol, p=p: scar_mod.detect_scar_hysteresis(vol, lab, phi_percentile=p)))
                      for p in (84, 87, 90, 92, 94)],
    "normal_anchor": [("k", k, (lambda lab, vol, k=k: scar_mod.detect_scar_normal_anchor(vol, lab, k=k)))
                      for k in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5)],
    "robust_mad":    [("k", k, (lambda lab, vol, k=k: scar_mod.detect_scar_robust_mad(vol, lab, k=k)))
                      for k in (0.2, 0.4, 0.6, 0.9, 1.2, 1.6)],
}


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    return 2.0 * (a & b).sum() / s if s else float("nan")


def _hd95(a, b, sampling):
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    asurf = a & ~ndimage.binary_erosion(a)
    bsurf = b & ~ndimage.binary_erosion(b)
    dt_b = ndimage.distance_transform_edt(~bsurf, sampling=sampling)
    dt_a = ndimage.distance_transform_edt(~asurf, sampling=sampling)
    d = np.concatenate([dt_b[asurf], dt_a[bsurf]])
    return float(np.percentile(d, 95)) if d.size else float("nan")


def _pair(masks, fn):
    c = list(masks)
    return [fn(masks[c[i]], masks[c[j]]) for i in range(len(c)) for j in range(i + 1, len(c))]


def main():
    ref = REPLICATES[0]
    ref_vol = _vol(ref)
    ref_img = reg._read_vol(ref_vol)
    sp = ref_img.GetSpacing(); vmm3 = sp[0] * sp[1] * sp[2]
    samp = (sp[2], sp[1], sp[0])
    tx = {ref: reg.identity()}
    for mov in REPLICATES[1:]:
        tx[mov] = reg.align_transform(ref_vol, label_mod.corrected_path(ref), _vol(mov), label_mod.corrected_path(mov))[0]
        print(f"  aligned {mov}", flush=True)

    data = {}
    for cid in REPLICATES:
        lab = np.rint(np.asarray(nib.load(str(label_mod.corrected_path(cid))).dataobj)).astype(np.uint8)
        vol = np.asarray(nib.load(str(_vol(cid))).dataobj).astype(np.float32)
        data[cid] = (lab, vol)

    def warp(mask, cid):
        tmp = TMP / f"{cid}.nii.gz"
        label_mod.write_label_nifti(mask.astype(np.uint8), _vol(cid), tmp)
        return reg.resample_label(tmp, ref_vol, tx[cid]) >= 1

    print("\n===== SIZE-MATCHED SWEEP: pairwise Dice / HD95 / CV vs scar VOLUME =====")
    print(f"{'family':14}{'param':>8}{'mean mm³':>10}{'CV%':>7}{'pairDice':>9}{'HD95mm':>8}")
    rows = []
    for fam, points in SWEEPS.items():
        for plabel, pval, fn in points:
            try:
                warped, vols = {}, []
                for cid in REPLICATES:
                    lab, vol = data[cid]
                    m = fn(lab, vol) & ((lab == 1) | (lab == 2))
                    vols.append(float(m.sum()) * vmm3)
                    warped[cid] = warp(m, cid)
                mean = float(np.mean(vols)); sd = float(np.std(vols, ddof=1))
                cv = round(sd / mean * 100, 2) if mean else 0.0
                pd = round(float(np.mean(_pair(warped, _dice))), 3)
                ph = round(float(np.nanmean(_pair(warped, lambda a, b: _hd95(a, b, samp)))), 3)
                rows.append((fam, mean, cv, pd, ph))
                print(f"{fam:14}{f'{plabel}={pval}':>8}{mean:>10.2f}{cv:>7.2f}{pd:>9}{ph:>8}", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"{fam:14}{f'{plabel}={pval}':>8}  ERROR {str(exc)[:40]}")

    # Dice at MATCHED volume: for each family, interpolate Dice & HD95 at common target volumes
    print("\n===== Dice @ matched volume (linear interp on each family's volume→Dice curve) =====")
    targets = [1.0, 1.5, 2.0, 2.7, 3.5]
    fams = list(SWEEPS)
    print(f"{'target mm³':>11}" + "".join(f"{f:>15}" for f in fams))
    for t in targets:
        cells = []
        for f in fams:
            fr = sorted([(m, d, h) for (ff, m, c, d, h) in rows if ff == f], key=lambda x: x[0])
            vs = [r[0] for r in fr]
            if not fr or t < vs[0] - 1e-6 or t > vs[-1] + 1e-6:
                cells.append(f"{'—':>15}")
            else:
                d = float(np.interp(t, vs, [r[1] for r in fr]))
                h = float(np.interp(t, vs, [r[2] for r in fr]))
                cells.append(f"{f'{d:.3f}/{h:.3f}':>15}")
        print(f"{t:>11.1f}" + "".join(cells), flush=True)
    print("(cell = pairwise Dice / HD95 mm at that volume; '—' = volume outside the family's swept range)")
    print("\nHigher Dice + lower HD95 at the SAME volume ⇒ genuinely more reproducible boundary, not just bigger.")
    shutil.rmtree(TMP, ignore_errors=True)


if __name__ == "__main__":
    main()
