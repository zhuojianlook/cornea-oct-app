"""Cohort grouping for batch .OCT processing.

Discovers .OCT scans under a directory tree, pairs each with its companion .txt, and
groups repeat scans by (patient, eye) — each group becomes one consensus label. The
batch runner (api_server) then preprocesses → SAM2-segments → builds consensus per
group, mass-producing the labeled training set. Grouping is pure (no heavy deps)."""
from __future__ import annotations

from pathlib import Path

import oct_preprocess as oct_mod


def discover(root: str | Path) -> list[dict]:
    """Every .OCT under `root` (recursive), with parsed patient/eye/series + companion .txt.
    '3D Cornea' scans are kept first (the volumetric acquisitions); others still listed."""
    root = Path(root).expanduser()
    octs = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".oct")
    items = []
    for p in octs:
        fm = oct_mod.parse_oct_filename(p.name)
        txt = p.with_suffix(".txt")
        if not txt.exists():
            txt = p.with_suffix(".TXT")
        items.append({
            "path": str(p),
            "filename": p.name,
            "patient": fm.get("patient_name", ""),
            "eye": (fm.get("laterality", "") or "").upper(),
            "series": fm.get("series_number", 1),
            "companion": str(txt) if txt.exists() else None,
            "is_3d_cornea": "3d cornea" in p.name.lower(),
        })
    return items


def group_by_eye(items: list[dict], only_3d_cornea: bool = True) -> list[dict]:
    """Group scans into per-(patient, eye) sets of repeat scans (sorted by series)."""
    groups: dict[tuple, list[dict]] = {}
    for it in items:
        if only_3d_cornea and not it["is_3d_cornea"]:
            continue
        key = (it["patient"] or Path(it["path"]).stem, it["eye"] or "?")
        groups.setdefault(key, []).append(it)
    out = []
    for (patient, eye), scans in sorted(groups.items()):
        out.append({
            "patient": patient,
            "eye": eye,
            "scans": sorted(scans, key=lambda s: s["series"]),
        })
    return out
