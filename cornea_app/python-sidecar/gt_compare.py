"""Import + compare MANUAL ground-truth labelmaps against the app's auto segmentation.

A manual GT is a 0/1/2 labelmap (0=background, 1=cornea, 2=scar — the same
``labels.NNUNET_LABELS`` convention) produced by the companion annotator app on the
EXACT working volume this app exported (``GET /api/case/{id}/preprocessed.nii.gz``).
Because it was painted on those exact bytes it is already voxel-aligned to the case's
working volume AND the app's own segmentation — no registration needed — so the import
just validates geometry and stores it, and the comparison is a direct voxel/boundary
overlap of two arrays.

Metrics are self-contained (scipy only) so this stays importable in the sidecar — it does
NOT pull in nnunet_report (which lives in the nnU-Net venv with matplotlib/pandas).
Per-class quantification reuses ``scar.quantify`` so volumes/areas/density match the rest
of the app exactly.
"""
from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

import orchestration as orch
import scar as scar_mod

# Cornea = the whole cornea (label 1 ∪ scar), scar = label 2 — matches scar.quantify and the
# nnU-Net report (gt_cornea = gt>0, gt_scar = gt==2). Scar is a hyper-reflective sub-region of cornea.
CLASS_MASKS = {
    "cornea": lambda a: a > 0,
    "scar": lambda a: a == 2,
}
AGREEMENT_LABELS = {"tp": 1, "auto_only": 2, "gt_only": 3}  # both, auto-not-gt (FP), gt-not-auto (FN)


# ── per-case storage layout ───────────────────────────────────────────────────
def manual_gt_dir(case_id: str) -> Path:
    return orch.case_root(case_id) / "manual_gt"


def agreement_dir(case_id: str) -> Path:
    return manual_gt_dir(case_id) / "_agreement"


def safe_name(filename: str) -> str:
    """A filesystem/URL-safe GT name from an uploaded filename (drop .nii/.nii.gz, sanitize)."""
    base = re.sub(r"\.nii(\.gz)?$", "", str(filename), flags=re.IGNORECASE)
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return base or "gt"


def manual_gt_path(case_id: str, name: str) -> Path:
    return manual_gt_dir(case_id) / f"{safe_name(name)}.nii.gz"


def load_labelmap(path: Path) -> np.ndarray:
    """Load a .nii.gz labelmap as a 0/1/2 uint8 array (so callers needn't import nibabel)."""
    return np.rint(np.asarray(nib.load(str(path)).dataobj)).astype(np.uint8)


def list_gts(case_id: str) -> list[dict]:
    """Imported GTs for a case (top-level .nii.gz only — skips the _agreement cache)."""
    d = manual_gt_dir(case_id)
    out: list[dict] = []
    if not d.exists():
        return out
    for p in sorted(d.glob("*.nii.gz")):
        try:
            arr = np.rint(np.asarray(nib.load(str(p)).dataobj)).astype(np.uint8)
            out.append({
                "name": p.name[:-7],  # strip ".nii.gz"
                "cornea_voxels": int((arr > 0).sum()),
                "scar_voxels": int((arr == 2).sum()),
                "imported_at": round(p.stat().st_mtime, 0),
            })
        except Exception:  # noqa: BLE001 — a corrupt file shouldn't break the whole listing
            out.append({"name": p.name[:-7], "cornea_voxels": -1, "scar_voxels": -1, "error": "unreadable"})
    return out


# ── import: validate an uploaded labelmap against the case's working volume ─────
def validate_and_store(data: bytes, filename: str, base_nifti: Path, dst: Path) -> dict:
    """Validate uploaded labelmap bytes against ``base_nifti`` (the working volume) and, if it
    matches, store it (re-stamped with the base affine, atomic). Raises ValueError on any mismatch
    so the caller can surface a clear per-file error. Returns QA counts on success."""
    base = nib.load(str(base_nifti))
    base_shape = tuple(int(s) for s in base.shape[:3])
    is_gz = len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B
    # Unique temp name so concurrent imports to the same case can't clobber each other's upload
    # (and distinct from write_label_nifti's own "_tmp_<dst>"); suffix matches the data so nibabel reads it.
    tmp = dst.parent / (f"_tmpup_{uuid.uuid4().hex}{'.nii.gz' if is_gz else '.nii'}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_bytes(data)
    try:
        img = nib.load(str(tmp))
        arr = np.rint(np.asarray(img.dataobj)).astype(np.uint8)
        if tuple(int(s) for s in arr.shape[:3]) != base_shape:
            raise ValueError(
                f"labelmap shape {tuple(int(s) for s in arr.shape[:3])} != this case's volume "
                f"{base_shape} — did you pick a labelmap from a different scan?")
        if not np.allclose(np.asarray(img.affine, dtype=float), np.asarray(base.affine, dtype=float), atol=1e-3):
            raise ValueError(
                "labelmap geometry (affine) does not match this case's working volume — it was "
                "annotated on a different scan. Open the matching case before importing.")
        vals = set(int(v) for v in np.unique(arr).tolist())
        if not vals.issubset({0, 1, 2}):
            raise ValueError(
                f"unexpected label values {sorted(vals)} — a manual GT must be 0=background / "
                f"1=cornea / 2=scar.")
        # Atomic, affine-stamped write (also re-runs the shape guard).
        from labels import write_label_nifti  # local import avoids a cycle at module load
        write_label_nifti(arr, base_nifti, dst)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass
    return {"name": dst.name[:-7], "cornea_voxels": int((arr > 0).sum()), "scar_voxels": int((arr == 2).sum())}


# ── overlap metrics (self-contained) ──────────────────────────────────────────
def _dice(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.astype(bool); b = b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return None  # undefined: neither has this class (e.g. no scar in either)
    return round(2.0 * int(np.logical_and(a, b).sum()) / denom, 4)


def _jaccard(a: np.ndarray, b: np.ndarray) -> float | None:
    a = a.astype(bool); b = b.astype(bool)
    union = int(np.logical_or(a, b).sum())
    if union == 0:
        return None
    return round(int(np.logical_and(a, b).sum()) / union, 4)


def _surface_distances(a: np.ndarray, b: np.ndarray, spacing) -> np.ndarray | None:
    """Symmetric surface-to-surface distances (mm) between two masks. Cropped to the union bbox
    (+1 vox pad so erosion isn't fooled by the crop edge) for speed without changing the result."""
    if not a.any() or not b.any():
        return None
    union = np.logical_or(a, b)
    coords = np.argwhere(union)
    mn = np.maximum(coords.min(0) - 1, 0)
    mx = np.minimum(coords.max(0) + 2, np.array(a.shape))
    sl = tuple(slice(int(lo), int(hi)) for lo, hi in zip(mn, mx))
    a = a[sl]; b = b[sl]
    a_surf = np.logical_and(a, ~ndimage.binary_erosion(a))
    b_surf = np.logical_and(b, ~ndimage.binary_erosion(b))
    if not a_surf.any():
        a_surf = a
    if not b_surf.any():
        b_surf = b
    dt_to_b = ndimage.distance_transform_edt(~b_surf, sampling=spacing)
    dt_to_a = ndimage.distance_transform_edt(~a_surf, sampling=spacing)
    return np.concatenate([dt_to_b[a_surf], dt_to_a[b_surf]])


def _boundary_metrics(a: np.ndarray, b: np.ndarray, spacing) -> tuple[float | None, float | None]:
    d = _surface_distances(a, b, spacing)
    if d is None or d.size == 0:
        return None, None
    return round(float(np.percentile(d, 95)), 4), round(float(d.mean()), 4)  # (hd95_mm, assd_mm)


def compare(gt_path: Path, auto_ijk: np.ndarray, base_nifti: Path, name: str = "", auto_source: str = "") -> dict:
    """Compare a stored manual GT (gt_path) against the app's auto labelmap (auto_ijk), both 0/1/2 on
    the same grid. Returns per-class Dice/Jaccard/HD95/ASSD/volumes(+diff)/voxel-overlap plus the full
    ``scar.quantify`` for each side.

    base_nifti MUST be the RAW volume (api_server._ensure_volume_nifti): spacing AND density are taken
    from it so quantify matches what /scar/auto persists (it quantifies on raw reflectivity + raw geometry
    — the cross-scan-comparable biomarker). The GT/auto arrays are the same index grid, so this is exact."""
    base = nib.load(str(base_nifti))
    spacing = tuple(float(z) for z in base.header.get_zooms()[:3])  # (sp_i, sp_j, sp_k), array-axis order
    gt = np.rint(np.asarray(nib.load(str(gt_path)).dataobj)).astype(np.uint8)
    if tuple(int(s) for s in gt.shape[:3]) != tuple(int(s) for s in auto_ijk.shape[:3]):
        raise ValueError(f"GT shape {gt.shape[:3]} != segmentation shape {auto_ijk.shape[:3]}.")
    density = np.asarray(base.dataobj).astype(np.float32)
    gt_q = scar_mod.quantify(gt, spacing, density)
    auto_q = scar_mod.quantify(auto_ijk, spacing, density)

    classes: dict[str, dict] = {}
    for cname, mask_fn in CLASS_MASKS.items():
        g = mask_fn(gt); a = mask_fn(auto_ijk)
        hd95, assd = _boundary_metrics(g, a, spacing)
        gv = gt_q["cornea_volume_mm3"] if cname == "cornea" else gt_q["scar_volume_mm3"]
        av = auto_q["cornea_volume_mm3"] if cname == "cornea" else auto_q["scar_volume_mm3"]
        entry = {
            "dice": _dice(g, a),
            "jaccard": _jaccard(g, a),
            "hd95_mm": hd95,
            "assd_mm": assd,
            "gt_voxels": int(g.sum()),
            "auto_voxels": int(a.sum()),
            "tp": int(np.logical_and(g, a).sum()),
            "fp": int(np.logical_and(a, ~g).sum()),   # auto-only (over-segmentation)
            "fn": int(np.logical_and(g, ~a).sum()),   # gt-only (missed)
            "gt_volume_mm3": gv,
            "auto_volume_mm3": av,
            "volume_signed_diff_mm3": round(av - gv, 6),     # auto − manual
            "volume_abs_diff_mm3": round(abs(av - gv), 6),
            "volume_rel_diff_pct": round(100.0 * (av - gv) / gv, 2) if gv > 0 else None,
        }
        if cname == "scar":
            entry["gt_area_mm2"] = gt_q.get("scar_area_mm2")
            entry["auto_area_mm2"] = auto_q.get("scar_area_mm2")
        classes[cname] = entry

    return {
        "name": name,
        "auto_source": auto_source,
        "spacing_mm": [round(s, 6) for s in spacing],
        "classes": classes,
        "gt_quant": gt_q,
        "auto_quant": auto_q,
    }


def agreement_map(gt_ijk: np.ndarray, auto_ijk: np.ndarray, klass: str = "scar") -> np.ndarray:
    """Per-voxel comparison map for one class: 1=agree (TP), 2=auto-only (FP), 3=GT-only (FN)."""
    mask_fn = CLASS_MASKS.get(klass, CLASS_MASKS["scar"])
    g = mask_fn(gt_ijk); a = mask_fn(auto_ijk)
    out = np.zeros(gt_ijk.shape[:3], dtype=np.uint8)
    out[np.logical_and(g, a)] = AGREEMENT_LABELS["tp"]
    out[np.logical_and(a, ~g)] = AGREEMENT_LABELS["auto_only"]
    out[np.logical_and(g, ~a)] = AGREEMENT_LABELS["gt_only"]
    return out
