#!/usr/bin/env python3
"""Deterministic synthetic fixtures for the Playwright E2E suite.

Builds a set of small, FAST, GPU-free cases (one per lifecycle step) + a per-eye replicate
CONSENSUS, directly via the sidecar's own orchestration/labels/postprocess/consensus code (no SAM2,
no real .OCT), into CORNEA_DATA_DIR. Idempotent: wipes + rebuilds the zz* test cases each run.

Run by tests/e2e/global-setup.ts before the suite. NEVER point CORNEA_DATA_DIR at real cases/.

Cases produced (id -> lifecycle step asserted by the suite):
  case_zz_raw         1  Raw (input only)
  case_zz_auto        2  Preprocessed/auto (before-after + fix-columns surfaces)
  case_zz_vetted      3  Vetted (awaiting the group alignment) -> Align group button
  case_zz_classified  4  Group-aligned (+ classified) -> Run SAM2 button
  case_zz_cornea      5  Cornea segmented (awaiting vet) -> paint/vet
  case_zz_subgroup    8  Subgroup assigned -> scar detect
  case_zz_scar        9  Scar segmented -> align/correct
  case_zz_od_v1/v2/v3 9  consensus members (scar, subgroup 1)
  case_zz_od_consensus 10 Scar replicates aligned (the step-10 consensus surface)
"""
import os
import shutil
import sys
from pathlib import Path

SIDE = Path(__file__).resolve().parents[2] / "python-sidecar"
sys.path.insert(0, str(SIDE))
os.environ.setdefault("CORNEA_DATA_DIR", "/tmp/cornea_pw_e2e")
os.environ.setdefault("CORNEA_API_TOKEN", "")
os.chdir(str(SIDE))

import numpy as np
import nibabel as nib

import settings
import orchestration as orch
import labels
import postprocess
import api_server

# The sidecar stamps every preprocessing run with oct_preprocess.PIPELINE_VERSION and reports a run as stale
# (case-open response `pipeline_current: false`) when the stamp differs. The frontend then AUTO re-runs an
# unapproved stale scan on open. Step-2 fixtures that must open QUIETLY are stamped current here; the one
# fixture that must trigger the auto re-run (case_zz_stale) is left unstamped. None (older sidecar without
# the constant) leaves the fixtures unstamped, which such a sidecar does not flag either.
try:
    from oct_preprocess import PIPELINE_VERSION
except ImportError:  # pragma: no cover — sidecar predates pipeline stamping
    PIPELINE_VERSION = None


def _run(**extra):
    """An oct_iter run record stamped as CURRENT (when the sidecar stamps runs at all)."""
    rec = {"passes": 1}
    if PIPELINE_VERSION is not None:
        rec["pipeline_version"] = PIPELINE_VERSION
    rec.update(extra)
    return rec

# Small anisotropic OCT-ish grid (frames, depth, lateral). Diagonal RAS affine — the consensus
# cons_native geometry fix (v0.0.92+) makes the native-frame consensus survive regardless.
NF, ND, NL = 24, 64, 48
AFF = np.diag([0.012, 0.006, 0.04, 1.0]).astype(float)
CORNEA_D = (26, 36)            # bright stromal band in depth
SCAR_BOX = (8, 18, 28, 33, 16, 32)   # f0,f1,d0,d1,l0,l1


def _vol(scar=True, shift=0, bright_scar=True):
    vol = np.full((NF, ND, NL), 22, np.uint16)
    vol[:, CORNEA_D[0]:CORNEA_D[1], :] = 210
    if scar and bright_scar:
        f0, f1, d0, d1, l0, l1 = SCAR_BOX
        vol[f0:f1, d0:d1, l0:l1] = 380
    if shift:
        vol = np.roll(vol, shift, axis=0)
    return vol


def _label(scar=True, shift=0):
    lab = np.zeros((NF, ND, NL), np.uint8)
    lab[:, CORNEA_D[0]:CORNEA_D[1], :] = 1
    if scar:
        f0, f1, d0, d1, l0, l1 = SCAR_BOX
        lab[f0:f1, d0:d1, l0:l1] = 2
    if shift:
        lab = np.roll(lab, shift, axis=0)
    return lab


def _write_vol(cid, vol):
    pv = orch.case_root(cid) / "previews" / "volume.nii.gz"
    pv.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(vol, AFF), str(pv))
    return pv


def _ctx(cid, vol):
    """Grayscale context previews for raw + corrected (before/after + fix-columns + scrub)."""
    pv = orch.case_root(cid) / "previews" / "_ctxsrc.nii.gz"
    nib.save(nib.Nifti1Image(vol, AFF), str(pv))
    postprocess.render_context_previews(pv, orch.case_root(cid) / "previews" / "context_raw")
    postprocess.render_context_previews(pv, orch.context_preview_dir(cid))


def make(cid, manifest, vol=None, lab=None, seg=False, ctx=False):
    orch.ensure_case_dirs(cid)
    if vol is None:
        vol = _vol(scar=("scar" in str(manifest.get("scar_classification", ""))))
    pv = _write_vol(cid, vol)
    base = {"input_volume": str(pv), "corrected_volume": str(pv), "patient_id": "ZZ", "eye": "OS"}
    base.update(manifest)
    if lab is not None:
        labels.write_label_nifti(lab, pv, labels.corrected_path(cid))
    if seg and lab is not None:
        postprocess.render_seg_previews(pv, lab, orch.segmentation_preview_dir(cid), density_vol=vol)
    if ctx:
        _ctx(cid, vol)
    orch.write_manifest_value(cid, base)
    return cid


def main():
    root = settings.CASES_ROOT
    print(f"[seed] CORNEA_DATA_DIR={settings.WORKSPACE_ROOT}  cases={root}", flush=True)
    # wipe only OUR zz* test cases (never touch anything else); case_zz* also catches the
    # derived consensus id (case_zzod_od_consensus has no underscore after zz).
    for p in sorted(root.glob("case_zz*")) if root.exists() else []:
        shutil.rmtree(p, ignore_errors=True)

    # 1) Raw
    make("case_zz_raw", {"oct_preprocessed": None}, vol=_vol(scar=False))
    # 2) Preprocessed (auto, unvetted) — before/after + fix-columns need context previews + a pass count
    make("case_zz_auto", {"oct_preprocessed": True, "oct_iter": _run()}, vol=_vol(scar=False), ctx=True)
    # 3) Vetted — awaiting the group-wise 3D alignment (step 4): the action bar offers "⧉ Align group"
    make("case_zz_vetted", {"oct_preprocessed": True, "preproc_vetted": True}, ctx=True)
    # 4) Group-aligned (+ a scar tag) — awaiting SAM2: the action bar offers "▶ Run SAM2 (cornea)"
    make("case_zz_classified", {"oct_preprocessed": True, "preproc_vetted": True,
                                "group_aligned": {"ts": "2026-09-10T00:00:00", "group": "zz_os", "source": "seed"},
                                "scar_classification": "scar"}, ctx=True)
    # 5) Cornea segmented (awaiting vet) — cornea-only labelmap + seg previews
    make("case_zz_cornea", {"oct_preprocessed": True, "preproc_vetted": True, "scar_classification": "scar",
                            "sam2_meta": {"ok": True}}, lab=_label(scar=False), seg=True, ctx=True)
    # 6) Cornea/background vetted (scar) — awaiting SUBGROUP assignment (Confirm subgroup / Auto subgroups)
    make("case_zz_corneavet", {"oct_preprocessed": True, "preproc_vetted": True, "scar_classification": "scar",
                               "sam2_meta": {"ok": True}, "cornea_vetted": True}, lab=_label(scar=False), seg=True, ctx=True)
    # 7) Subgroup assigned (scar) — cornea vetted + subgroup confirmed, scar not yet done
    make("case_zz_subgroup", {"oct_preprocessed": True, "preproc_vetted": True, "scar_classification": "scar",
                              "sam2_meta": {"ok": True}, "cornea_vetted": True, "subgroup_confirmed": True,
                              "scar_subgroup": "1"}, lab=_label(scar=False), seg=True, ctx=True)
    # 8) Scar segmented
    make("case_zz_scar", {"oct_preprocessed": True, "preproc_vetted": True, "scar_classification": "scar",
                          "sam2_meta": {"ok": True}, "cornea_vetted": True, "subgroup_confirmed": True,
                          "scar_subgroup": "1", "scar_done": True}, lab=_label(scar=True), seg=True, ctx=True)

    # 7c) CONTROL (no scar), cornea vetted — steps 8-12 are N/A; it goes Classified (7) -> Scheduled (13).
    make("case_zz_control", {"oct_preprocessed": True, "preproc_vetted": True, "scar_classification": "control",
                             "sam2_meta": {"ok": True}, "cornea_vetted": True}, lab=_label(scar=False), seg=True, ctx=True)

    # Dedicated MUTABLE cases for the progression spec (so mutation tests don't couple to read-only ones):
    #   case_zz_vet       step 2 — Approve preprocessing -> classify
    #   case_zz_corrected step 12 — Schedule / Unschedule
    make("case_zz_vet", {"oct_preprocessed": True, "oct_iter": _run()}, vol=_vol(scar=False), ctx=True)
    # STALE step-2 case: preprocessed + unapproved, but its run record carries NO pipeline_version → the
    # sidecar reports pipeline_current:false and the app auto re-runs it on open (pipeline-stale.spec.ts).
    make("case_zz_stale", {"oct_preprocessed": True, "oct_iter": {"passes": 1}}, vol=_vol(scar=False), ctx=True)
    make("case_zz_corrected", {"oct_preprocessed": True, "preproc_vetted": True, "scar_classification": "scar",
                               "sam2_meta": {"ok": True}, "cornea_vetted": True, "subgroup_confirmed": True,
                               "scar_subgroup": "1", "scar_done": True, "corrected_labelmap": True},
         lab=_label(scar=True), seg=True, ctx=True)

    # 10) Consensus from 3 replicate members (scar, subgroup 1), via the real build (no SAM2)
    members = []
    for i, sh in enumerate((0, 2, -2), start=1):
        cid = f"case_zz_od_v{i}"
        make(cid, {"oct_preprocessed": True, "preproc_vetted": True, "scar_classification": "scar",
                   "sam2_meta": {"ok": True}, "cornea_vetted": True, "subgroup_confirmed": True,
                   "scar_subgroup": "1", "scar_done": True, "patient_id": "ZZOD", "eye": "OD"},
             vol=_vol(scar=True, shift=sh), lab=_label(scar=True, shift=sh), seg=True, ctx=True)
        members.append(cid)
    api_server._ensure_segmented = lambda cid: None     # members are already segmented
    ccid, report = api_server._build_consensus_case(members, subgroup="1", ensure=False)
    print(f"[seed] consensus={ccid} scans={report.get('scans')} ref={report.get('reference')}", flush=True)

    n = len(list(root.glob('case_zz*')))
    print(f"[seed] OK — {n} zz* cases built (consensus={ccid}).", flush=True)


if __name__ == "__main__":
    main()
