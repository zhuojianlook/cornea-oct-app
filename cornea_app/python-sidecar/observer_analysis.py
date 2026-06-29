"""Inter-/intra-observer reproducibility from the annotator's ground-truth output folder (#4).

The companion annotator writes, per saved ground truth, a labelmap
  <root>/<volume_stem>/<username>__rep<replicate>__<session>.nii.gz   (voxels 0=bg, 1=cornea, 2=scar)
and appends a row to <root>/manifest.json / manifest.csv with username, volume_stem, replicate,
blind_label, session_id, saved_at, voxel counts, scar_mm3, spacing.

This module pairs those up and computes pairwise Dice (scar + cornea) and scar-volume agreement:
  • INTRA-observer = SAME user, SAME scan, different REPLICATES (test–retest of one rater).
  • INTER-observer = SAME scan, DIFFERENT users (one representative per user).
Lower-deviation / higher-Dice = more reproducible. Pure stdlib + numpy + SimpleITK; no torch.
"""
from __future__ import annotations
import csv as _csv
import itertools
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def _load_label(path: Path) -> np.ndarray | None:
    try:
        return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))  # (z,y,x) ints 0/1/2
    except Exception:
        return None


def _dice(a: np.ndarray, b: np.ndarray, label: int) -> float | None:
    A = a == label
    B = b == label
    denom = int(A.sum()) + int(B.sum())
    if denom == 0:
        return None  # neither annotation has this label → undefined (skip, don't count as 1.0 or 0.0)
    return float(2.0 * int(np.logical_and(A, B).sum()) / denom)


def _gt_path(root: Path, row: dict) -> Path:
    rep = int(row.get("replicate", 1) or 1)
    return root / str(row["volume_stem"]) / f"{row['username']}__rep{rep}__{row['session_id']}.nii.gz"


def _voxvol_mm3(spacing: str | None) -> float:
    """mm³ per voxel from a 'a×b×c' spacing string (annotator manifest); 1.0 if unparseable."""
    try:
        return float(np.prod([float(x) for x in str(spacing).replace("x", "×").split("×")]))
    except Exception:
        return 1.0


def _scar_vol_mm3(lab: np.ndarray, vox_mm3: float) -> float:
    return float((lab == 2).sum()) * vox_mm3


def analyze(root: str | Path) -> dict:
    """Compute inter/intra-observer reproducibility for an annotator output folder. Returns a dict with
    `intra` rows, `inter` rows, `volume` per-scan volume reproducibility, and a `summary`."""
    root = Path(root)
    mpath = root / "manifest.json"
    if not mpath.exists():
        return {"ok": False, "error": f"No manifest.json in {root}"}
    try:
        manifest = json.loads(mpath.read_text())
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"Bad manifest.json: {e}"}
    if not isinstance(manifest, list) or not manifest:
        return {"ok": False, "error": "manifest.json is empty"}

    # Keep the LATEST save per (stem, user, replicate) — a re-save overwrites the same labelmap file.
    latest: dict[tuple, dict] = {}
    for r in manifest:
        k = (r.get("volume_stem"), r.get("username"), int(r.get("replicate", 1) or 1))
        if k[0] is None or k[1] is None:
            continue
        if k not in latest or str(r.get("saved_at", "")) >= str(latest[k].get("saved_at", "")):
            latest[k] = r
    rows = list(latest.values())

    by_stem: dict[str, list[dict]] = {}
    for r in rows:
        by_stem.setdefault(str(r["volume_stem"]), []).append(r)

    intra, inter, volume = [], [], []
    cache: dict[str, np.ndarray | None] = {}

    def lab_of(r: dict) -> np.ndarray | None:
        p = _gt_path(root, r)
        if str(p) not in cache:
            cache[str(p)] = _load_label(p)
        return cache[str(p)]

    for stem, rs in sorted(by_stem.items()):
        by_user: dict[str, list[dict]] = {}
        for r in rs:
            by_user.setdefault(str(r["username"]), []).append(r)

        # ── INTRA: per user, every pair of replicates of THIS scan ──
        for user, urs in sorted(by_user.items()):
            urs = sorted(urs, key=lambda r: int(r.get("replicate", 1) or 1))
            for a, b in itertools.combinations(urs, 2):
                A, B = lab_of(a), lab_of(b)
                if A is None or B is None or A.shape != B.shape:
                    continue
                intra.append({
                    "volume_stem": stem, "username": user,
                    "rep_a": int(a.get("replicate", 1)), "rep_b": int(b.get("replicate", 1)),
                    "dice_scar": _dice(A, B, 2), "dice_cornea": _dice(A, B, 1),
                })

        # ── INTER: one representative per user (lowest replicate), all user pairs ──
        reps = {user: min(urs, key=lambda r: int(r.get("replicate", 1) or 1)) for user, urs in by_user.items()}
        for (ua, ra), (ub, rb) in itertools.combinations(sorted(reps.items()), 2):
            A, B = lab_of(ra), lab_of(rb)
            if A is None or B is None or A.shape != B.shape:
                continue
            inter.append({
                "volume_stem": stem, "user_a": ua, "user_b": ub,
                "dice_scar": _dice(A, B, 2), "dice_cornea": _dice(A, B, 1),
            })

        # ── VOLUME reproducibility: split INTRA (test-retest, per rater across replicates) vs INTER (between
        #    raters), mirroring the Dice split, so a single CV doesn't conflate the two regimes. A pooled CV
        #    (every annotation) is kept too for backward compatibility. ──
        per_user_vols: dict[str, list[float]] = {}
        all_vols: list[float] = []
        for user, urs in by_user.items():
            uvs = []
            for r in sorted(urs, key=lambda r: int(r.get("replicate", 1) or 1)):
                L = lab_of(r)
                if L is None:
                    continue
                uvs.append(_scar_vol_mm3(L, _voxvol_mm3(r.get("spacing"))))
            if uvs:
                per_user_vols[user] = uvs       # replicate-sorted (lowest first)
                all_vols.extend(uvs)

        def _cv(vs: list[float]) -> float | None:
            if len(vs) < 2:
                return None
            a = np.array(vs, float); m = float(a.mean())
            return round(float(a.std(ddof=1) / m), 4) if m > 0 else None

        # INTRA = mean of each rater's OWN across-replicate CV (raters with >=2 replicates of this scan).
        intra_cvs = [c for c in (_cv(vs) for vs in per_user_vols.values()) if c is not None]
        cv_intra = round(float(np.mean(intra_cvs)), 4) if intra_cvs else None
        # INTER = CV across ONE representative volume per rater (lowest replicate, matching the Dice `reps`).
        rep_vols = [vs[0] for vs in per_user_vols.values() if vs]
        cv_inter = _cv(rep_vols)
        if len(all_vols) >= 2:
            v = np.array(all_vols, float)
            mean = float(v.mean())
            volume.append({"volume_stem": stem, "n": len(all_vols), "scar_mm3_mean": round(mean, 4),
                           "scar_mm3_cv": round(float(v.std(ddof=1) / mean), 4) if mean > 0 else None,  # pooled (legacy)
                           "scar_mm3_cv_intra": cv_intra, "scar_mm3_cv_intra_n_users": len(intra_cvs),
                           "scar_mm3_cv_inter": cv_inter, "scar_mm3_cv_inter_n_users": len(rep_vols),
                           "scar_mm3_min": round(float(v.min()), 4), "scar_mm3_max": round(float(v.max()), 4)})

    def _mean(rows_: list[dict], key: str) -> float | None:
        xs = [r[key] for r in rows_ if r.get(key) is not None]
        return round(float(np.mean(xs)), 4) if xs else None

    summary = {
        "scans": len(by_stem), "users": sorted({str(r["username"]) for r in rows}),
        "n_annotations": len(rows), "n_intra_pairs": len(intra), "n_inter_pairs": len(inter),
        "intra_dice_scar_mean": _mean(intra, "dice_scar"), "intra_dice_cornea_mean": _mean(intra, "dice_cornea"),
        "inter_dice_scar_mean": _mean(inter, "dice_scar"), "inter_dice_cornea_mean": _mean(inter, "dice_cornea"),
        "scar_volume_cv_mean": _mean(volume, "scar_mm3_cv"),                       # pooled (legacy)
        "intra_scar_volume_cv_mean": _mean(volume, "scar_mm3_cv_intra"),          # test-retest, per rater
        "inter_scar_volume_cv_mean": _mean(volume, "scar_mm3_cv_inter"),          # between raters
    }
    return {"ok": True, "root": str(root), "summary": summary, "intra": intra, "inter": inter, "volume": volume}


def write_csvs(result: dict, out_dir: str | Path) -> list[str]:
    """Write intra/inter/volume tables as CSVs alongside a JSON summary; returns the paths written."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("intra", "inter", "volume"):
        rows = result.get(name) or []
        p = out_dir / f"observer_{name}.csv"
        cols = list(rows[0].keys()) if rows else []
        with open(p, "w", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=cols)
            if cols:
                w.writeheader(); w.writerows(rows)
        written.append(str(p))
    sp = out_dir / "observer_summary.json"
    sp.write_text(json.dumps(result.get("summary", {}), indent=2))
    written.append(str(sp))
    return written


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Inter/intra-observer reproducibility from an annotator output folder")
    ap.add_argument("root", help="annotator output dir (contains manifest.json + <stem>/ labelmaps)")
    ap.add_argument("--out", default="", help="write CSVs + summary here (default: <root>)")
    a = ap.parse_args()
    res = analyze(a.root)
    print(json.dumps(res.get("summary", res), indent=2))
    if res.get("ok"):
        for p in write_csvs(res, a.out or a.root):
            print("wrote", p)
