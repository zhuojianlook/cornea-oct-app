"""Cohort-wide robustness check: is the volume-alignment + scar-registration methodology robust?

For every per-eye consensus case, read the warped per-scan labels (scans/<cid>/label.nii.gz, already
in the reference frame) and compute pairwise CORNEA Dice (did the volumes align?) + pairwise SCAR Dice
(FOV-restricted; did the scar register?) + scar-volume CV (from the report). Aggregate across the cohort,
split scar vs control. High cornea Dice everywhere ⇒ alignment robust; the scar-Dice distribution shows
where scar registration holds vs where eyes genuinely disagree (candidate subgroups / artifacts).
READ-ONLY.
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


def _dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else float("nan")


def _fov_dice(sa, da, sb, db):
    c = da & db
    a, b = sa & c, sb & c
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else float("nan")


def main():
    root = settings.CASES_ROOT
    rows = []
    for d in sorted(root.glob("case_*_consensus")):
        ccid = d.name
        m = orch.read_manifest(ccid)
        members = m.get("consensus_cases") or []
        report = m.get("consensus_report") or {}
        scans_dir = d / "scans"
        labs = {}
        for c in members:
            p = scans_dir / c / "label.nii.gz"
            if p.exists():
                labs[c] = np.rint(np.asarray(nib.load(str(p)).dataobj)).astype(np.uint8)
        if len(labs) < 2:
            continue
        cids = list(labs)
        cornea = {c: labs[c] >= 1 for c in cids}
        scar = {c: labs[c] == 2 for c in cids}
        # cornea data/FOV ≈ cornea mask region; use cornea for FOV-restriction of scar
        cor_pairs, scar_pairs = [], []
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                cor_pairs.append(_dice(cornea[a], cornea[b]))
                scar_pairs.append(_fov_dice(scar[a], cornea[a], scar[b], cornea[b]))
        cls = "control" if any("control" == str(orch.read_manifest(c).get("scar_classification")) for c in members) else "scar"
        eye = ccid.replace("case_", "").replace("_consensus", "")
        rows.append({
            "eye": eye, "cls": cls, "n": len(cids),
            "cornea_dice": round(float(np.nanmean(cor_pairs)), 3),
            "scar_dice_fov": round(float(np.nanmean(scar_pairs)), 3) if not all(np.isnan(scar_pairs)) else float("nan"),
            "scar_cv": report.get("scar_volume_mm3", {}).get("cv_percent"),
            "scar_mm3": report.get("scar_volume_mm3", {}).get("mean"),
        })

    rows.sort(key=lambda r: (r["cls"], r["cornea_dice"]))
    print(f"{'eye':12}{'cls':8}{'n':>3}{'corneaDice':>11}{'scarDiceFOV':>12}{'CV%':>7}{'scar mm³':>9}")
    for r in rows:
        sd = "—" if (r["scar_dice_fov"] != r["scar_dice_fov"]) else f'{r["scar_dice_fov"]}'
        cv = "" if r["scar_cv"] is None else f'{r["scar_cv"]}'
        print(f"{r['eye']:12}{r['cls']:8}{r['n']:>3}{r['cornea_dice']:>11}{sd:>12}{cv:>7}{str(r['scar_mm3']):>9}")

    def agg(sel, key):
        vals = [r[key] for r in rows if r["cls"] == sel and isinstance(r[key], (int, float)) and r[key] == r[key]]
        return (round(float(np.median(vals)), 3), round(float(np.min(vals)), 3), round(float(np.max(vals)), 3)) if vals else None
    print("\n=== cornea alignment (Dice) — median[min,max] ===")
    for s in ("scar", "control"):
        a = agg(s, "cornea_dice"); print(f"  {s}: {a[0]} [{a[1]}, {a[2]}]" if a else f"  {s}: -")
    print("=== scar registration (FOV Dice) — scar eyes ===")
    a = agg("scar", "scar_dice_fov"); print(f"  scar: {a[0]} [{a[1]}, {a[2]}]" if a else "  -")
    print("=== scar volume CV% ===")
    for s in ("scar", "control"):
        a = agg(s, "scar_cv"); print(f"  {s}: {a[0]} [{a[1]}, {a[2]}]" if a else f"  {s}: -")
    # flag eyes where alignment is weak (candidate subgroup-split / artifact)
    weak = [r["eye"] for r in rows if r["cornea_dice"] < 0.85]
    print(f"\nlow cornea-Dice eyes (<0.85 → check subgroup/artifact): {weak}")


if __name__ == "__main__":
    main()
