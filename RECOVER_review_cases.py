#!/usr/bin/env python3
"""RECOVERY for review_cases/ — two DISTINCT problems caused by agent sidecars on 2026-07-17.

Run:   python3 RECOVER_review_cases.py --dry-run     # show what it would do (default)
       python3 RECOVER_review_cases.py --apply       # actually fix

────────────────────────────────────────────────────────────────────────────────────────
PROBLEM 1 (all 308 manifests, written 19:26) — URGENT, it is a time bomb.
  input_volume / corrected_volume in EVERY manifest were rewritten to point at
      /tmp/claude-1000/.../scratchpad/ws/cases/<case>/input/<file>.nii.gz
  `ws/cases` is a SYMLINK to the real case store, so those paths still resolve TODAY and the
  app looks fine. When /tmp is cleared (reboot / session cleanup) the symlink dies and all 308
  cases lose their volume -> "Registered volume is missing" for every case.
  FIX: rewrite the path prefix back to the real in-case path. Pure string fix, no data touched.

PROBLEM 2 (47 previews, written 20:06-20:11) — silent, and it does NOT self-heal.
  previews/volume.nii.gz for 47 cases was overwritten with a RAW (uncorrected) re-conversion,
  replacing the preprocessed/vetted volume. Verified: on an untouched case
  NCC(input/<work>.nii.gz, previews/volume.nii.gz) == 1.0 (the preview IS the corrected volume);
  on a damaged case it is 0.699 (it is the raw one).
  It will NOT self-heal because _ensure_volume_nifti only regenerates when
  dst.mtime < src.mtime, and the damaged previews are NEWER than the Jul-12 working volumes.
  FIX: re-run the app's OWN volume_io.ensure_nifti(work -> preview). Reconstruction was
  validated: it reproduces the original byte sizes exactly (48,351,130 / 48,120,635).

NOT AFFECTED (verified): input/, passes/, segmentation/ — 0 files touched. No .nii.gz added or
removed (1204 before and after). All 308 manifests parse.
"""
import argparse, glob, json, os, shutil, sys
from pathlib import Path

REAL = Path("/home/zhuojian/Desktop/Integration/review_cases/cases")
BAD_PREFIX = "/tmp/claude-1000/-home-zhuojian-Desktop-Integration/2b81569c-111a-46a7-9e12-79ea6ef2f4cf/scratchpad/ws/cases"
SIDECAR = "/home/zhuojian/Desktop/ctwt/cornea_app/python-sidecar"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true", help="show what would change (this is the default)")
ap.add_argument("--apply", action="store_true", help="write the fixes (default: dry run)")
ap.add_argument("--skip-previews", action="store_true", help="only fix manifest paths")
args = ap.parse_args()
DRY = not args.apply
tag = "DRY-RUN" if DRY else "APPLY"

# ── Problem 1: manifest paths ────────────────────────────────────────────────────────────
print(f"[{tag}] 1/2  manifest input_volume/corrected_volume -> real in-case paths")
n_fixed = 0
for f in sorted(glob.glob(str(REAL / "*" / "manifest.json"))):
    txt = open(f).read()
    if BAD_PREFIX not in txt:
        continue
    new = txt.replace(BAD_PREFIX, str(REAL))
    m = json.loads(new)                                     # must still parse
    for k in ("input_volume", "corrected_volume"):
        p = m.get(k)
        if p and not os.path.exists(p):
            print(f"    !! target missing, skipping {f}: {p}");  break
    else:
        if not DRY:
            bak = f + ".bak_recover"
            if not os.path.exists(bak):
                shutil.copy2(f, bak)
            with open(f, "w") as fh:
                fh.write(new)
        n_fixed += 1
print(f"      manifests {'would be ' if DRY else ''}fixed: {n_fixed}")

# ── Problem 2: raw-clobbered previews ────────────────────────────────────────────────────
if args.skip_previews:
    sys.exit(0)
print(f"\n[{tag}] 2/2  regenerate previews/volume.nii.gz from the INTACT corrected working volume")
sys.path.insert(0, SIDECAR)
import numpy as np, nibabel as nib
import volume_io                                            # the app's own converter

def ncc(a, b):
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    return float((a * b).mean())

n_ok = n_skip = 0
for case_dir in sorted(REAL.glob("case_*")):
    prev = case_dir / "previews" / "volume.nii.gz"
    if not prev.exists():
        continue
    m = json.load(open(case_dir / "manifest.json"))
    src = m.get("corrected_volume") or m.get("input_volume")
    if not src:
        continue
    work = Path(os.path.realpath(src))
    if not work.exists() or not m.get("oct_preprocessed"):
        continue
    try:
        a = np.asanyarray(nib.load(str(work)).dataobj).astype(np.float32)
        b = np.asanyarray(nib.load(str(prev)).dataobj).astype(np.float32)
    except Exception as e:
        print(f"    !! unreadable {case_dir.name}: {e}");  continue
    if a.shape != b.shape or ncc(a, b) < 0.999:             # preview != corrected volume => damaged
        print(f"    {case_dir.name}: NCC={ncc(a,b) if a.shape==b.shape else float('nan'):.4f} -> regenerate")
        if not DRY:
            volume_io.ensure_nifti(work, prev)
            c = np.asanyarray(nib.load(str(prev)).dataobj).astype(np.float32)
            got = ncc(a, c)
            print(f"        verified NCC(work, preview) = {got:.4f}" + ("  OK" if got > 0.999 else "  *** STILL WRONG ***"))
        n_ok += 1
    else:
        n_skip += 1
print(f"\n      previews {'would be ' if DRY else ''}regenerated: {n_ok}   |   already correct, untouched: {n_skip}")
print("\nDone." + ("  Re-run with --apply to write." if DRY else "  Manifest backups: <case>/manifest.json.bak_recover"))
