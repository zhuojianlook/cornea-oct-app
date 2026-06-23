"""Cross-case scar quantification export — the research deliverable.

Recomputes scar/cornea volume (mm³) and en-face scar area (mm²) from each case's
*current* corrected labelmap (so the table always reflects the latest expert GT,
not a stale manifest), tags each row with patient/eye/date parsed from the source
filename, and writes `scar_summary.csv` + `.json` ready to merge with outcomes.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import nibabel as nib

import settings
import orchestration as orch
import labels
import scar as scar_mod

# preprocessed_CS001_14145_3D Cornea_OD_2024-07-11 (2)_0.dcm
_NAME_RE = re.compile(
    r"(?P<pid>[A-Za-z]+\d+)_(?P<dev>\d+)_.*?_(?P<eye>O[DS])_(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:[ _]*\((?P<variant>\d+)\))?",
    re.IGNORECASE,
)

_COLUMNS = ["case", "patient_id", "eye", "date", "variant", "scar_present",
            "scar_volume_mm3", "scar_area_mm2", "cornea_volume_mm3",
            "scar_fraction_of_cornea", "scar_density_mean", "scar_density_weighted_mm3u",
            "label_source"]


def parse_case_meta(input_volume: str | None) -> dict:
    """Pull patient_id / eye / date / variant from the source filename."""
    meta = {"patient_id": "", "eye": "", "date": "", "variant": ""}
    if not input_volume:
        return meta
    m = _NAME_RE.search(Path(input_volume).name)
    if m:
        meta.update(patient_id=m.group("pid").upper(), eye=m.group("eye").upper(),
                    date=m.group("date"), variant=m.group("variant") or "")
    return meta


def resolve_case_meta(case_id: str) -> dict:
    """Best-effort patient/eye/date/variant for a case, so EVERY row (including the
    consensus row, the deliverable biomarker) carries a mergeable key:
      1. consensus cases inherit the reference member's identity (variant='consensus');
      2. OCT cases parse the ORIGINAL oct_source filename — it preserves the '(N)'
         replicate suffix that the safe-id'd working filename collapses to '_N';
      3. legacy cases fall back to the input_volume name."""
    m = orch.read_manifest(case_id) or {}
    rep = m.get("consensus_report") or {}
    ref = rep.get("reference")
    if ref and ref != case_id:
        return {**resolve_case_meta(ref), "variant": "consensus"}
    meta = parse_case_meta(None)
    for key in ("oct_source", "input_volume"):
        parsed = parse_case_meta(m.get(key))
        if parsed.get("patient_id"):
            meta = parsed
            break
    # A user-corrected patient/eye persisted on the case (group-header edit) wins over the
    # filename parse; date/variant still come from the filename.
    if m.get("patient_id"):
        meta = {**meta, "patient_id": str(m["patient_id"]).upper()}
    if m.get("eye"):
        meta = {**meta, "eye": str(m["eye"]).upper()}
    return meta


def _cases_with_labelmap() -> list[str]:
    if not settings.CASES_ROOT.exists():
        return []
    out = []
    for d in sorted(settings.CASES_ROOT.iterdir()):
        if not d.is_dir():
            continue
        cid = d.name
        if labels.corrected_path(cid).exists():
            out.append(cid)
    return out


def _base_volume(case_id: str) -> Path | None:
    p = orch.case_root(case_id) / "previews" / "volume.nii.gz"
    return p if p.exists() else None


def build_row(case_id: str) -> dict | None:
    import numpy as np
    arr, source = labels.best_labelmap_nnunet(case_id)
    base = _base_volume(case_id)
    if arr is None or base is None:
        return None
    img = nib.load(str(base))
    spacing = img.header.get_zooms()[:3]
    raw = np.asarray(img.dataobj).astype(np.float32)   # comparable reflectivity for density
    m = scar_mod.quantify(arr, spacing, density_vol_ijk=raw)
    dens = m.get("scar_density", {})
    meta = resolve_case_meta(case_id)
    return {
        "case": case_id, **meta,
        "scar_present": m["scar_present"],
        "scar_volume_mm3": m["scar_volume_mm3"],
        "scar_area_mm2": m["scar_area_mm2"],
        "cornea_volume_mm3": m["cornea_volume_mm3"],
        "scar_fraction_of_cornea": m["scar_fraction_of_cornea"],
        "scar_density_mean": dens.get("mean", ""),
        "scar_density_weighted_mm3u": dens.get("weighted_volume_mm3u", ""),
        "label_source": source,
    }


def build_summary(case_ids: list[str] | None = None) -> list[dict]:
    ids = case_ids if case_ids else _cases_with_labelmap()
    rows = [r for r in (build_row(c) for c in ids) if r is not None]
    return rows


def write_summary(rows: list[dict], out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (settings.WORKSPACE_ROOT / "output")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "scar_summary.csv"
    json_path = out_dir / "scar_summary.json"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in _COLUMNS})
    json_path.write_text(json.dumps(rows, indent=2))
    return {"csv": str(csv_path), "json": str(json_path), "n_cases": len(rows)}
