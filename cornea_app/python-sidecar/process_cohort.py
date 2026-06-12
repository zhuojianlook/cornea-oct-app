"""Cohort driver — ingest every preprocessed OCT volume and run the scar pipeline.

For each DICOM in the source dir: create a case, register the volume (Slicer
DICOM→NIfTI), segment cornea with SAM2, pre-annotate scar inside the cornea, then
write the cross-case scar_summary. Drives the *running* sidecar over HTTP so it
reuses the exact tested endpoints. The expert still corrects each case afterwards
in the app; re-running /metrics/summary refreshes the table from the corrected GT.

Usage:
    python process_cohort.py [--src DIR] [--limit N] [--base URL]
                             [--skip-sam2] [--sigma S] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

DEFAULT_SRC = Path("/home/zhuojian/Desktop/PROJECT: 3D OCT/PROCESSED DICOMS")
_RE = re.compile(r"(?P<pid>[A-Za-z]+\d+)_(?P<dev>\d+)_.*?_(?P<eye>O[DS])_"
                 r"(?P<date>\d{4}-\d{2}-\d{2})(?:[ _]*\((?P<v>\d+)\))?", re.IGNORECASE)


def case_id_for(path: Path) -> str:
    m = _RE.search(path.name)
    if not m:
        return "case_" + re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")[:40]
    v = m.group("v") or "1"
    return f"case_{m.group('pid').lower()}_{m.group('eye').lower()}_v{v}"


def post(base: str, path: str, body: dict, timeout: int):
    r = requests.post(f"{base}{path}", json=body, timeout=timeout)
    r.raise_for_status()
    return r.json()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--skip-sam2", action="store_true", help="ingest only, no segmentation")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    dicoms = sorted(p for p in args.src.glob("*.dcm"))
    if args.limit:
        dicoms = dicoms[:args.limit]
    if not dicoms:
        print(f"No .dcm under {args.src}", file=sys.stderr)
        return 1
    print(f"{len(dicoms)} volume(s) to process from {args.src}")
    for i, dcm in enumerate(dicoms, 1):
        cid = case_id_for(dcm)
        print(f"\n[{i}/{len(dicoms)}] {cid}  ←  {dcm.name}")
        if args.dry_run:
            continue
        t0 = time.time()
        try:
            post(args.base, "/api/case", {"case_id": cid}, 30)
            post(args.base, f"/api/case/{cid}/volume/register", {"volume_path": str(dcm)}, 600)
            if not args.skip_sam2:
                seg = post(args.base, f"/api/case/{cid}/segment/sam2", {"vote": 2}, 900)
                print(f"   SAM2 cornea voxels={seg['qa']['sam2']['cornea_voxels']}")
                scar = post(args.base, f"/api/case/{cid}/scar/auto", {"sigma": args.sigma}, 300)
                m = scar["metrics"]
                print(f"   scar_present={m['scar_present']} vol_mm3={m['scar_volume_mm3']} "
                      f"area_mm2={m['scar_area_mm2']}")
        except requests.HTTPError as e:
            print(f"   ! failed: {e} :: {e.response.text[:200]}", file=sys.stderr)
            continue
        print(f"   done in {time.time()-t0:.0f}s")

    summ = post(args.base, "/api/metrics/summary", {}, 300)
    print(f"\nscar_summary → {summ['csv']}  ({summ['n_cases']} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
