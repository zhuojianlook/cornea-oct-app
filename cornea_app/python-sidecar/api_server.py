#!/usr/bin/env python3
"""
FastAPI server for the Cornea OCT Segmentation app.

Launched as a Tauri sidecar (or, in browser-dev, started directly by
dev-launch.sh). Communicates with the frontend over HTTP on 127.0.0.1:8765,
either directly (browser fetch) or proxied through the Rust shell.

Focused pipeline: load 3D OCT → SAM2 segments cornea → expert corrects →
detect scar (hyper-reflective) → expert corrects → quantify (volume / en-face
area / density) → cross-case scar_summary.csv (+ nnU-Net export). The only 3D
Slicer dependency is DICOM→NIfTI conversion; everything else is in-sidecar numpy.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel

import settings
import orchestration as orch
import volume_io
import labels
import gt_compare
import slicer_runner
import masks
import scar as scar_mod
import export as export_mod
import nnunet_train as nntrain
import preprocess
import postprocess
import metrics_export
import consensus as consensus_mod
import normal_baseline
import oct_preprocess as oct_mod
import oct_motion as oct_motion_mod
import cohort as cohort_mod

app = FastAPI(title="Cornea OCT Segmentation Sidecar")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    # shell_version echoes the env the Tauri shell set when it spawned this sidecar, so the app can
    # confirm it's talking to the sidecar IT launched (not a stale/foreign one). Empty in dev.
    return {"status": "ok", "shell_version": os.environ.get("CORNEA_SHELL_VERSION", "")}


@app.get("/api/config")
def get_config() -> dict:
    return settings.public_config()


class ConfigUpdate(BaseModel):
    slicer_executable: str | None = None
    default_case_id: str | None = None


@app.put("/api/config")
def put_config(update: ConfigUpdate) -> dict:
    settings.update_settings(update.model_dump(exclude_unset=True))
    return settings.public_config()


# ── Case lifecycle ─────────────────────────────────────────────────────────
class CasePayload(BaseModel):
    case_id: str


@app.post("/api/case")
def create_case(payload: CasePayload) -> dict:
    orch.ensure_case_dirs(payload.case_id)
    return orch.current_case_info(payload.case_id)


@app.get("/api/case/{case_id}")
def get_case(case_id: str) -> dict:
    return orch.current_case_info(case_id)


# ── Volume registration / upload / conversion ──────────────────────────────
class RegisterVolume(BaseModel):
    volume_path: str


def _registered_volume(case_id: str) -> Path:
    manifest = orch.read_manifest(case_id)
    path = manifest.get("corrected_volume") or manifest.get("input_volume")
    if not path:
        raise HTTPException(404, "No volume registered for this case.")
    return Path(path)


@app.post("/api/case/{case_id}/volume/register")
def register_volume(case_id: str, payload: RegisterVolume) -> dict:
    orch.ensure_case_dirs(case_id)
    volume = Path(payload.volume_path)
    if not volume.exists():
        raise HTTPException(400, f"Volume does not exist: {payload.volume_path}")
    orch.write_manifest_value(
        case_id, {"input_volume": str(volume), "corrected_volume": str(volume)})
    return orch.current_case_info(case_id)


@app.post("/api/case/{case_id}/volume/upload")
async def upload_volume(case_id: str, files: List[UploadFile] = File(...)) -> dict:
    orch.ensure_case_dirs(case_id)
    if not files:
        raise HTTPException(400, "No file uploaded.")
    upload = files[0]
    dest = orch.case_root(case_id) / "input" / Path(upload.filename or "volume").name
    dest.write_bytes(await upload.read())
    orch.write_manifest_value(
        case_id, {"input_volume": str(dest), "corrected_volume": str(dest)})
    return orch.current_case_info(case_id)


def _ensure_volume_nifti(case_id: str) -> Path:
    src = _registered_volume(case_id)
    if not src.exists():
        raise HTTPException(404, f"Registered volume is missing: {src}")
    dst = orch.case_root(case_id) / "previews" / "volume.nii.gz"
    if (not dst.exists()) or dst.stat().st_mtime < src.stat().st_mtime:
        suffix = "".join(src.suffixes).lower()
        is_dicom = suffix.endswith(".dcm") or suffix.endswith(".dicom") or src.suffix.lower() in (".dcm", ".dicom")
        if is_dicom:
            # niivue/nibabel can't read DICOM — convert through Slicer.
            proc = slicer_runner.convert_to_nifti(str(src), str(dst))
            if proc["status"] != 0 or not dst.exists():
                raise HTTPException(500, f"DICOM → NIfTI conversion failed:\n{proc['stderr']}")
        else:
            try:
                volume_io.ensure_nifti(src, dst)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(500, f"Volume conversion failed: {exc}")
    return dst


def _preprocessed_path(case_id: str) -> Path:
    return orch.case_root(case_id) / "previews" / "preprocessed.nii.gz"


def _working_volume(case_id: str) -> Path:
    """The volume the pipeline operates on: the preprocessed (denoised+contrast)
    NIfTI if present and current, else the plain converted NIfTI. Unifying on the
    NIfTI keeps previews, segmentation and the viewer in one coordinate space."""
    base = _ensure_volume_nifti(case_id)
    pre = _preprocessed_path(case_id)
    if pre.exists() and pre.stat().st_mtime >= base.stat().st_mtime:
        return pre
    return base


def _pass_volume_path(case_id: str, pass_num: int | None) -> Path:
    """Resolve the NIfTI to download for a specific iterative-refinement pass (1-based), or the
    working/best volume when pass_num is None. Each pass Vk is persisted at passes/pass_{k}.nii.gz by
    oct_preprocess_case; falls back to the working (best) volume if that pass wasn't persisted (e.g. a
    single-pass scan, or pass_num out of range)."""
    if pass_num is None:
        return _working_volume(case_id)
    p = orch.case_root(case_id) / "passes" / f"pass_{int(pass_num)}.nii.gz"
    return p if p.exists() else _working_volume(case_id)


@app.get("/api/case/{case_id}/volume.nii.gz")
def get_volume_nifti(case_id: str) -> FileResponse:
    dst = _working_volume(case_id)
    return FileResponse(str(dst), media_type="application/gzip", filename="volume.nii.gz")


def _scan_filename_stem(case_id: str) -> str:
    """A human-recognizable download stem: the ORIGINAL source scan filename (what the user sees in
    the loader, e.g. 'CS001_14145_3D Cornea_OD_2024-07-11'), minus its extension. Falls back to the
    case_id when no source is recorded. So a downloaded file matches the scan it came from."""
    cid = orch.safe_case_id(case_id)
    try:
        m = orch.read_manifest(cid)
        src = m.get("oct_source") or m.get("companion_txt") or ""
        if src:
            base = os.path.basename(str(src)).strip()
            base = re.sub(r"\.(oct|txt|nii\.gz|nii|nrrd|dcm)$", "", base, flags=re.IGNORECASE).strip()
            if base:
                return base
    except Exception:  # noqa: BLE001 — naming is best-effort; never block a download
        pass
    return cid


@app.get("/api/case/{case_id}/preprocessed.nii.gz")
def download_preprocessed_nifti(case_id: str, pass_num: int | None = None) -> FileResponse:
    """Download ONE preprocessed (corrected) scan as a NIfTI, named ``<case_id>.nii.gz``.

    Same bytes as the working volume the viewer/pipeline use, but with a per-scan
    filename so a folder of these drops straight into the ground-truth annotator app
    (each file's stem becomes the scan id → clean inter-/intra-observer grouping).
    404 until the scan has actually been preprocessed."""
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, f"No such case: {case_id}")
    try:
        dst = _pass_volume_path(cid, pass_num)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"No preprocessed volume for {case_id}: {exc}")
    if not Path(dst).exists():
        raise HTTPException(404, f"No preprocessed volume for {case_id}. Preprocess the scan first.")
    # Only tag the filename with the pass when that pass actually exists (else _pass_volume_path fell
    # back to the working/best volume — don't mislabel it as the requested pass).
    pass_exists = bool(pass_num) and (orch.case_root(cid) / "passes" / f"pass_{int(pass_num)}.nii.gz").exists()
    suffix = f"_pass{int(pass_num)}" if pass_exists else ""
    return FileResponse(str(dst), media_type="application/gzip", filename=f"{_scan_filename_stem(cid)}{suffix}.nii.gz")


@app.get("/api/preprocessed-zip")
def download_preprocessed_zip(cases: str = "", pass_num: int | None = None) -> FileResponse:
    """Bundle several preprocessed scans into one ``.zip`` — a folder-ready SET for
    manual ground-truth segmentation. Each entry is ``<case_id>.nii.gz`` (the working
    volume), so unzipping gives a directory the annotator app can open directly.

    ``cases`` is a comma-separated list of case ids. Ids are normalized with
    ``safe_case_id`` (so two inputs that normalize to the same id collapse to one
    entry); missing/un-preprocessed ids are skipped. The zip contains whatever
    resolved. 404 only if none resolved."""
    ids = [c.strip() for c in cases.split(",") if c.strip()]
    if not ids:
        raise HTTPException(400, "No cases specified.")
    tmp = tempfile.NamedTemporaryFile(prefix="preprocessed_", suffix=".zip", delete=False)
    included: list[str] = []
    missing: list[str] = []
    try:
        # .nii.gz is already gzip-compressed → ZIP_STORED avoids pointless re-compression.
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
            seen: set[str] = set()
            used_names: set[str] = set()
            for raw in ids:
                cid = orch.safe_case_id(raw)
                if cid in seen:
                    continue
                seen.add(cid)
                if not orch.case_root(cid).exists():
                    missing.append(raw)
                    continue
                try:
                    src = _pass_volume_path(cid, pass_num)
                except Exception:  # noqa: BLE001 — skip a bad scan, keep the rest of the set
                    missing.append(raw)
                    continue
                if src and Path(src).exists():
                    # Name each entry after the source scan; disambiguate rare collisions with the case id.
                    stem = _scan_filename_stem(cid)
                    arc = f"{stem}.nii.gz"
                    if arc in used_names:
                        arc = f"{stem}__{cid}.nii.gz"
                    used_names.add(arc)
                    zf.write(str(src), arcname=arc)
                    included.append(cid)
                else:
                    missing.append(raw)
        tmp.close()
    except Exception as exc:  # noqa: BLE001
        tmp.close()
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise HTTPException(500, f"Zip build failed: {exc}")
    if not included:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise HTTPException(404, f"No preprocessed volumes found for: {', '.join(missing) or cases}")
    return FileResponse(
        tmp.name,
        media_type="application/zip",
        filename="preprocessed_scans.zip",
        background=BackgroundTask(os.unlink, tmp.name),  # delete the temp zip after it streams
    )


class SavePreprocessedRequest(BaseModel):
    dest: str
    pass_num: int | None = None   # 1-based iterative pass to export; None = working/best volume


@app.post("/api/case/{case_id}/save-preprocessed")
def save_preprocessed(case_id: str, req: SavePreprocessedRequest) -> dict:
    """Native-save (Tauri shell): copy a scan's preprocessed/working volume to a user-chosen path
    (picked via the desktop Save dialog), so the user controls the destination."""
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, f"No such case: {case_id}")
    try:
        src = _pass_volume_path(cid, req.pass_num)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"No preprocessed volume for {case_id}: {exc}")
    if not Path(src).exists():
        raise HTTPException(404, f"No preprocessed volume for {case_id}. Preprocess the scan first.")
    dest = Path(req.dest).expanduser()
    try:
        if dest.parent:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(src), str(dest))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Save failed: {exc}")
    return {"ok": True, "dest": str(dest)}


class SaveZipRequest(BaseModel):
    cases: List[str]
    dest: str
    pass_num: int | None = None   # 1-based iterative pass to export for every scan; None = working/best


@app.post("/api/preprocessed-zip-save")
def save_preprocessed_zip(req: SaveZipRequest) -> dict:
    """Native-save (Tauri shell): write a folder-ready .zip of several preprocessed scans to a
    user-chosen path. Entries are named after the source scans, like /api/preprocessed-zip."""
    ids = [c.strip() for c in req.cases if c and c.strip()]
    if not ids:
        raise HTTPException(400, "No cases specified.")
    dest = Path(req.dest).expanduser()
    included: list[str] = []
    missing: list[str] = []
    try:
        if dest.parent:
            dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(dest), "w", compression=zipfile.ZIP_STORED) as zf:
            seen: set[str] = set()
            used_names: set[str] = set()
            for raw in ids:
                cid = orch.safe_case_id(raw)
                if cid in seen:
                    continue
                seen.add(cid)
                if not orch.case_root(cid).exists():
                    missing.append(raw)
                    continue
                try:
                    src = _pass_volume_path(cid, req.pass_num)
                except Exception:  # noqa: BLE001
                    missing.append(raw)
                    continue
                if src and Path(src).exists():
                    stem = _scan_filename_stem(cid)
                    arc = f"{stem}.nii.gz"
                    if arc in used_names:
                        arc = f"{stem}__{cid}.nii.gz"
                    used_names.add(arc)
                    zf.write(str(src), arcname=arc)
                    included.append(cid)
                else:
                    missing.append(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Zip save failed: {exc}")
    if not included:
        try:
            os.unlink(str(dest))
        except OSError:
            pass
        raise HTTPException(404, f"No preprocessed volumes found for: {', '.join(missing) or req.cases}")
    return {"ok": True, "dest": str(dest), "n": len(included)}


class PreprocessRequest(BaseModel):
    enabled: bool = True
    sigma: float | None = None        # in-plane gaussian sigma (voxels)
    clip_low: float | None = None     # contrast clip low percentile (crush background)
    clip_high: float | None = None    # contrast clip high percentile
    gamma: float | None = None        # >1 darkens mid-tone speckle


@app.post("/api/case/{case_id}/preprocess")
def preprocess_case(case_id: str, req: PreprocessRequest) -> dict:
    """Create (or remove) a denoised + contrast-stretched working volume.
    When enabled, all previews/segmentation and the viewer use it."""
    orch.ensure_case_dirs(case_id)
    pre = _preprocessed_path(case_id)
    if not req.enabled:
        if pre.exists():
            pre.unlink()
        return {"case_info": orch.current_case_info(case_id), "preprocessed": False}
    base = _ensure_volume_nifti(case_id)
    sigma = req.sigma if req.sigma is not None else 2.0
    clip = (req.clip_low if req.clip_low is not None else 45.0,
            req.clip_high if req.clip_high is not None else 99.5)
    gamma = req.gamma if req.gamma is not None else 1.4
    try:
        preprocess.preprocess_volume(
            base, pre, sigma=(sigma, sigma, max(0.4, sigma * 0.4)), clip_pct=clip, gamma=gamma)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Preprocessing failed: {exc}")
    return {"case_info": orch.current_case_info(case_id), "preprocessed": True,
            "sigma": sigma, "clip_pct": clip, "gamma": gamma}


# ── Slice previews (2D gallery: works without WebGL) ────────────────────────
def _preview_group_dir(case_id: str, group: str) -> Path:
    """Previews live under previews/<group>/ (context, segmentation, or per-tab
    consensus groups like scan_<cid>_self / scan_<cid>_cons)."""
    return orch.case_root(case_id) / "previews" / orch.safe_case_id(group)


def _clear_iter_preview_groups(case_id: str) -> None:
    """Remove the per-pass iterative-refinement artifacts: the preview groups (previews/context_iter*)
    AND the persisted per-pass NIfTIs (passes/). Re-created by each preprocess; stale after a re-run
    or a raw re-scrub."""
    import shutil as _sh
    previews = orch.case_root(case_id) / "previews"
    if previews.exists():
        for d in previews.glob("context_iter*"):
            _sh.rmtree(d, ignore_errors=True)
    _sh.rmtree(orch.case_root(case_id) / "passes", ignore_errors=True)


def _parse_iter_info(worker_stdout: str) -> dict:
    """Parse the `ITER {json}` line the oct_preprocess worker prints (per-pass convergence)."""
    for line in (worker_stdout or "").splitlines():
        if line.startswith("ITER "):
            try:
                return json.loads(line[5:])
            except Exception:  # noqa: BLE001
                break
    return {"passes": 1, "metrics": [], "applied": [True], "stopped": "single"}


@app.get("/api/case/{case_id}/previews/{group}")
def list_previews(case_id: str, group: str) -> dict:
    # Lazy `src` URLs (not inline base64): the gallery loads only the slice on screen, so a
    # DENSE context group (every slice, for skip-free scrubbing) lists cheaply. The src_base
    # repeats the raw `group` string the client asked for; the file route re-resolves it the
    # same way (_preview_group_dir applies safe_case_id), so they land on the same folder.
    src_base = f"/api/case/{case_id}/preview-file/{group}"
    images = orch.preview_listing_from_dir(group, _preview_group_dir(case_id, group), src_base)
    return {"group": group, "images": images}


@app.get("/api/case/{case_id}/preview-file/{group}/{name}")
def get_preview_file(case_id: str, group: str, name: str) -> FileResponse:
    """Serve one preview PNG (referenced lazily by list_previews) — keeps a dense scrub
    group off the JSON payload. Path-traversal-guarded: a bare *.png basename only."""
    safe_name = Path(name).name
    if safe_name != name or not safe_name.lower().endswith(".png"):
        raise HTTPException(400, "Invalid preview file name.")
    p = _preview_group_dir(case_id, group) / safe_name
    if not p.exists():
        raise HTTPException(404, "Preview not found.")
    return FileResponse(str(p), media_type="image/png")


@app.post("/api/case/{case_id}/context-previews")
def context_previews(case_id: str) -> dict:
    """Render plain grayscale slice PNGs of the working volume (in-sidecar, numpy)
    so the 2D gallery can show the raw OCT before any segmentation."""
    orch.ensure_case_dirs(case_id)
    src = _working_volume(case_id)
    ctx = orch.context_preview_dir(case_id)
    try:
        postprocess.render_context_previews(src, ctx)
        (ctx / ".rev3").write_text("")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Context preview render failed: {exc}")
    # Lazy listing (not base64): a dense context group is large; the gallery loads slices on
    # demand via /preview-file. (Callers that just trigger a render ignore this anyway.)
    src_base = f"/api/case/{case_id}/preview-file/context"
    return {"images": orch.preview_listing_from_dir("Context", ctx, src_base)}


@app.post("/api/case/{case_id}/refresh-panel")
def refresh_panel(case_id: str) -> dict:
    """Re-render this scan's dense+rotated own-segmentation overlay (context_seg) from its
    CURRENT labelmap, so the subgroup grid's "per scan" scar reflects a correction made in the
    focused single-scan view. (context_cons is the vote — it only changes on a consensus rebuild.)"""
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        return {"ok": False, "reason": "no segmentation"}
    base = _ensure_volume_nifti(case_id)
    try:
        postprocess.render_seg_previews(base, arr, _preview_group_dir(case_id, "context_seg"), dense_rotated=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Panel refresh failed: {exc}")
    return {"ok": True}


# ── Stage 1: SAM2 cornea segmentation ──────────────────────────────────────
class Sam2Request(BaseModel):
    vote: int = 2                                   # planes that must agree (1–3)
    planes: List[str] = ["axial", "coronal", "sagittal"]


@app.post("/api/case/{case_id}/segment/sam2")
def segment_sam2(case_id: str, req: Sam2Request) -> dict:
    """SAM2 segments the cornea in each plane treated as a movie, then
    majority-votes the planes into one 3D cornea labelmap. The result is written
    as the canonical corrected labelmap (cornea=1)."""
    import sam2_segment  # lazy: only pull in torch/CUDA when actually segmenting
    import nibabel as nib
    orch.ensure_case_dirs(case_id)
    if not req.planes:
        raise HTTPException(400, "Request at least one plane.")
    base = _ensure_volume_nifti(case_id)            # SAM2 likes natural raw contrast
    work = orch.case_root(case_id) / "sam2_work"
    vote = max(1, min(req.vote, len(req.planes)))   # vote can't exceed #planes (else always empty)
    with _GPU_LOCK:                                  # one SAM2/CUDA inference at a time
        label, meta = sam2_segment.segment_volume(
            base, work, planes=tuple(req.planes), vote=vote)
    if label.sum() == 0:
        raise HTTPException(500, f"SAM2 produced an empty mask: {meta}")
    # Persist as the canonical labelmap so the overlay and nnU-Net export use it.
    backdrop = _working_volume(case_id)
    labels.write_label_nifti(label, base, labels.corrected_path(case_id))
    postprocess.render_seg_previews(backdrop, label, orch.segmentation_preview_dir(case_id))
    sp = nib.load(str(base)).header.get_zooms()[:3]
    counts = labels.labelmap_counts(label, spacing_mm3=float(sp[0] * sp[1] * sp[2]))
    qa = {"source": "sam2", "segments": counts, "sam2": meta}
    orch.case_qa_json(case_id).write_text(json.dumps(qa, indent=2))
    orch.write_manifest_value(case_id, {
        "qa_json": str(orch.case_qa_json(case_id)),
        "segmentation_preview_dir": str(orch.segmentation_preview_dir(case_id)),
        "sam2_meta": meta,
    })
    return {"case_info": orch.current_case_info(case_id), "qa": qa,
            "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id))}


# ── Stage 2: interactive correction (niivue drawing round-trip) ─────────────
@app.get("/api/case/{case_id}/segmentation-drawing.nii.gz")
def get_segmentation_drawing(case_id: str) -> FileResponse:
    """Current segmentation as an editable niivue drawing (cornea=1, bg=2, scar=3)."""
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        raise HTTPException(404, "No segmentation yet. Run SAM2 first.")
    base = _ensure_volume_nifti(case_id)
    dst = orch.case_root(case_id) / "previews" / "segmentation-drawing.nii.gz"
    try:
        masks.build_correction_drawing(base, arr, dst)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Correction drawing build failed: {exc}")
    return FileResponse(str(dst), media_type="application/gzip", filename="segmentation-drawing.nii.gz")


@app.post("/api/case/{case_id}/segmentation/from-drawing")
async def segmentation_from_drawing(case_id: str, files: List[UploadFile] = File(...)) -> dict:
    """Save an edited segmentation drawing as the canonical corrected labelmap,
    then re-render the overlay so the gallery reflects the correction."""
    orch.ensure_case_dirs(case_id)
    if not files:
        raise HTTPException(400, "No drawing uploaded.")
    data = await files[0].read()
    is_gz = len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B
    tmp = orch.case_root(case_id) / "previews" / ("edited-seg.nii.gz" if is_gz else "edited-seg.nii")
    tmp.write_bytes(data)
    base = _ensure_volume_nifti(case_id)
    try:
        arr = masks.corrected_labelmap_from_drawing(tmp, base, labels.corrected_path(case_id))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not parse corrected drawing: {exc}")
    postprocess.render_seg_previews(_working_volume(case_id), arr, orch.segmentation_preview_dir(case_id))
    qa = {"segments": labels.labelmap_counts(arr), "source": "corrected"}
    orch.write_manifest_value(case_id, {"corrected_labelmap": str(labels.corrected_path(case_id))})
    return {"case_info": orch.current_case_info(case_id), "qa": qa,
            "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id))}


@app.get("/api/case/{case_id}/segmentation.nii.gz")
def get_segmentation_nifti(case_id: str) -> FileResponse:
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        raise HTTPException(404, "No segmentation yet. Run SAM2 first.")
    base = _ensure_volume_nifti(case_id)
    dst = orch.case_root(case_id) / "previews" / "segmentation.nii.gz"
    try:
        labels.write_label_nifti(arr, base, dst)  # 0=bg, 1=cornea, 2=scar
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Segmentation conversion failed: {exc}")
    return FileResponse(str(dst), media_type="application/gzip", filename="segmentation.nii.gz")


# ── manual ground-truth import + comparison vs the auto segmentation ───────────
@app.post("/api/case/{case_id}/manual-gt")
async def import_manual_gt(case_id: str, files: List[UploadFile] = File(...)) -> dict:
    """Import one or more MANUAL ground-truth labelmaps (0/1/2) made in the annotator app on this
    case's exported working volume. Each file is validated (shape + affine + label values) against the
    working volume, then stored under manual_gt/<name>.nii.gz. Per-file errors don't abort the batch."""
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, f"No such case: {case_id}")
    if not files:
        raise HTTPException(400, "No file uploaded.")
    try:
        base = _working_volume(cid)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"This case has no working volume to align against: {exc}")
    imported: list[dict] = []
    errors: list[dict] = []
    for f in files:
        data = await f.read()
        try:
            dst = gt_compare.manual_gt_path(cid, Path(f.filename or "gt").name)
            imported.append(gt_compare.validate_and_store(data, f.filename or "gt", base, dst))
        except Exception as exc:  # noqa: BLE001 — surface a per-file error, keep importing the rest
            errors.append({"file": f.filename or "gt", "error": str(exc)})
    if not imported and errors:
        raise HTTPException(400, "; ".join(e["error"] for e in errors))
    return {"imported": imported, "errors": errors, "gts": gt_compare.list_gts(cid)}


@app.get("/api/case/{case_id}/manual-gt")
def list_manual_gt(case_id: str) -> dict:
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, f"No such case: {case_id}")
    auto, src = labels.best_labelmap_nnunet(cid)
    return {"gts": gt_compare.list_gts(cid), "has_segmentation": auto is not None, "auto_source": src}


@app.get("/api/case/{case_id}/manual-gt/{name}/labelmap.nii.gz")
def get_manual_gt_nifti(case_id: str, name: str) -> FileResponse:
    cid = orch.safe_case_id(case_id)
    p = gt_compare.manual_gt_path(cid, name)
    if not p.exists():
        raise HTTPException(404, f"No imported GT named {name}.")
    return FileResponse(str(p), media_type="application/gzip", filename=f"{gt_compare.safe_name(name)}.nii.gz")


@app.get("/api/case/{case_id}/manual-gt/{name}/compare")
def compare_manual_gt(case_id: str, name: str) -> dict:
    """Per-class (cornea, scar) Dice / Jaccard / HD95 / ASSD / volume(+diff) / voxel-overlap of the
    named manual GT vs the app's auto labelmap, plus full scar.quantify for each side."""
    cid = orch.safe_case_id(case_id)
    p = gt_compare.manual_gt_path(cid, name)
    if not p.exists():
        raise HTTPException(404, f"No imported GT named {name}.")
    auto, src = labels.best_labelmap_nnunet(cid)
    if auto is None:
        raise HTTPException(400, "No auto segmentation yet — run SAM2 / scar detection first, then compare.")
    # Quantify on the RAW volume (spacing + reflectivity) so the numbers match what /scar/auto persists
    # (raw reflectivity is the cross-scan biomarker). GT and auto share this index grid, so it's exact.
    base = _ensure_volume_nifti(cid)
    try:
        return gt_compare.compare(p, auto, base, name=gt_compare.safe_name(name), auto_source=src or "")
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Comparison failed: {exc}")


@app.get("/api/case/{case_id}/manual-gt/{name}/agreement.nii.gz")
def get_manual_gt_agreement(case_id: str, name: str, klass: str = "scar") -> FileResponse:
    """Agreement overlay for one class (scar|cornea): 1=agree (TP), 2=auto-only (FP), 3=GT-only (FN).
    Stamped with the working-volume affine so it aligns with /volume.nii.gz in the compare viewer."""
    cid = orch.safe_case_id(case_id)
    p = gt_compare.manual_gt_path(cid, name)
    if not p.exists():
        raise HTTPException(404, f"No imported GT named {name}.")
    auto, _ = labels.best_labelmap_nnunet(cid)
    if auto is None:
        raise HTTPException(400, "No auto segmentation yet.")
    klass = "cornea" if klass == "cornea" else "scar"
    base = _working_volume(cid)
    gt = gt_compare.load_labelmap(p)
    amap = gt_compare.agreement_map(gt, auto, klass)
    dst = gt_compare.agreement_dir(cid) / f"{gt_compare.safe_name(name)}__{klass}.nii.gz"
    try:
        labels.write_label_nifti(amap, base, dst)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Agreement map failed: {exc}")
    return FileResponse(str(dst), media_type="application/gzip", filename=f"agreement_{klass}.nii.gz")


@app.delete("/api/case/{case_id}/manual-gt/{name}")
def delete_manual_gt(case_id: str, name: str) -> dict:
    cid = orch.safe_case_id(case_id)
    p = gt_compare.manual_gt_path(cid, name)
    if p.exists():
        try:
            p.unlink()
        except OSError as exc:
            raise HTTPException(500, f"Could not delete: {exc}")
    ad = gt_compare.agreement_dir(cid)
    if ad.exists():
        for f in ad.glob(f"{gt_compare.safe_name(name)}__*.nii.gz"):
            try:
                f.unlink()
            except OSError:
                pass
    return {"gts": gt_compare.list_gts(cid)}


@app.get("/api/case/{case_id}/agreement.nii.gz")
def get_agreement_nifti(case_id: str, tol_mm: float = 0.0) -> FileResponse:
    """The replicate-agreement map written by consensus.build_consensus: per-voxel % of member
    scans whose scar covers it (0 / 33 / 66 / 100 for 3 scans). Powers the 3D overlap viewer.
    With `tol_mm` > 0, re-scores allowing that boundary slack (mm) — small residual shifts no longer
    read as disagreement, so the fringe collapses into the core."""
    strict = orch.case_root(case_id) / "previews" / "agreement.nii.gz"
    if not strict.exists():
        raise HTTPException(404, "No agreement map — build a consensus over the replicate scans first.")
    if tol_mm <= 0:
        return FileResponse(str(strict), media_type="application/gzip", filename="agreement.nii.gz")
    try:
        agr, _ = consensus_mod.tolerant_agreement(case_id, tol_mm)
        base = orch.case_root(case_id) / "previews" / "volume.nii.gz"
        dst = orch.case_root(case_id) / "previews" / "agreement_tol.nii.gz"
        labels.write_label_nifti(agr, base, dst)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Tolerant agreement failed: {exc}")
    return FileResponse(str(dst), media_type="application/gzip", filename="agreement_tol.nii.gz")


@app.get("/api/case/{case_id}/scan/{member}/{kind}.nii.gz")
def get_warped_scan(case_id: str, member: str, kind: str) -> FileResponse:
    """A consensus member's volume (or label) WARPED into the reference frame — written by
    build_consensus to scans/<member>/. `kind` ∈ {volume,label}. Powers the volume-alignment viewer
    (overlay the registered replicate volumes to see whether the scans actually align)."""
    if kind not in ("volume", "label"):
        raise HTTPException(400, "kind must be 'volume' or 'label'.")
    p = orch.case_root(case_id) / "scans" / orch.safe_case_id(member) / f"{kind}.nii.gz"
    if not p.exists():
        raise HTTPException(404, f"No warped {kind} for {member} — rebuild the consensus.")
    return FileResponse(str(p), media_type="application/gzip", filename=f"{kind}.nii.gz")


@app.get("/api/case/{case_id}/agreement-stats")
def get_agreement_stats(case_id: str, tol_mm: float = 0.0) -> dict:
    """Reproducibility readout for the overlap viewer at boundary tolerance `tol_mm`: tier volumes +
    mean pairwise tolerant Dice, plus the NATIVE per-scan scar biomarker (mean ± CV) from the report."""
    try:
        _, stats = consensus_mod.tolerant_agreement(case_id, tol_mm)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(404, f"No tolerant agreement available: {exc}")
    report = orch.read_manifest(case_id).get("consensus_report") or {}
    vol = report.get("scar_volume_mm3") or {}
    stats["native_scar_mm3"] = vol.get("mean")
    stats["native_scar_cv_percent"] = vol.get("cv_percent")
    stats["strict_pairwise_dice"] = report.get("mean_pairwise_scar_dice")
    return stats


# ── Normal reflectivity baseline (from control scans) ──────────────────────
class NormalProfileRequest(BaseModel):
    case_ids: List[str] | None = None   # default: all labelled control-tagged cases


@app.get("/api/normal-profile")
def normal_profile_status() -> dict:
    """Whether a control-derived normal reflectivity baseline exists + which controls are available."""
    return normal_baseline.profile_info()


@app.post("/api/normal-profile/build")
def normal_profile_build(req: NormalProfileRequest) -> dict:
    """Build the normal reflectivity profile (vs relative corneal depth) from the labelled control
    scans, so depth-normalised scar detection flags only EXCESS over normal (no Bowman's false scar)."""
    try:
        return normal_baseline.build_profile(req.case_ids)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


# ── Stage 3: scar detection + quantification ───────────────────────────────
class ScarAutoRequest(BaseModel):
    percentile: float = 88.0     # sensitivity: flag the brightest (100−percentile)% of cornea
    min_voxels: int = 500        # continuity: drop connected components smaller than this
    erode_surface: int = 6       # drop the epithelium/Bowman's/endothelium reflective rind
    replace: bool = False        # False: merge candidates with existing scar (keep manual edits)
    method: str = "hysteresis"   # strategy: hysteresis | normal_anchor | robust_mad | morph_lcc | brightness


@app.post("/api/case/{case_id}/scar/auto")
def scar_auto(case_id: str, req: ScarAutoRequest) -> dict:
    """Scar pre-annotation: inside the cornea mask, flag the brightest
    (hyper-reflective) voxels as scar *candidates* (label 2) on the contrast-enhanced
    volume the user sees, write back to the canonical labelmap, render the overlay
    (density-tiered), and quantify (volume mm³, en-face area mm², density).
    `percentile` is the sensitivity knob. The expert then prunes/extends the scar
    in the drawing layer. Requires a cornea labelmap (run SAM2 first)."""
    import nibabel as nib
    orch.ensure_case_dirs(case_id)
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None or not ((arr == 1) | (arr == 2)).any():
        raise HTTPException(400, "No cornea segmentation yet. Run SAM2 first.")
    base = _ensure_volume_nifti(case_id)            # raw volume: geometry + comparable reflectivity
    work = _working_volume(case_id)                 # contrast-enhanced volume the user sees
    vol = np.asarray(nib.load(str(work)).dataobj).astype(np.float32)
    raw = np.asarray(nib.load(str(base)).dataobj).astype(np.float32)
    had_scar = bool((arr == 2).any())
    # Run the selected scar STRATEGY (default hysteresis — benchmarked most reproducible) on the RAW
    # reflectivity volume, NOT the per-scan contrast-normalised working volume: the same physical
    # reflectivity then reads as scar in every replicate (comparable across scans/eyes, like the density
    # metric; live CS001-OS Dice 0.745→0.79 switching work→raw). `method` lets the strategies be
    # A/B-compared in the viewer. The expert still prunes/extends.
    method = (req.method or "hysteresis").lower()
    profile_note = ""
    if method in ("depthnorm", "control", "normal_profile"):
        # Depth-normalised: flag scar as EXCESS over the NORMAL corneal reflectivity profile (per
        # relative depth), so normal Bowman's/anterior brightness isn't mistaken for scar. Use the
        # CONTROL-derived profile when one has been built, else self-normalise from this scan.
        atlas = normal_baseline.load_profile()        # 3-D control atlas (depth×radius×meridian) or None
        zres = normal_baseline.atlas_z(raw, arr, nib.load(str(base)).header.get_zooms()[:3], atlas) if atlas else None
        if zres is not None:
            z, cornea_m, roi_m = zres
            kabs = max(2.0, normal_baseline.load_kabs() + (req.percentile - 92.0) * 0.05)  # sensitivity nudge
            scar_mask = scar_mod.scar_from_z(z, cornea_m, roi_m, k_abs=kabs)
            profile_note = f" (control atlas k={kabs:.1f})"
        else:
            scar_mask = scar_mod.detect_scar_depthnorm(raw, arr, phi_percentile=req.percentile)
            profile_note = " (self)"
    else:
        scar_mask = scar_mod.scar_detector(req.method)(raw, arr, req.percentile)
    new_label = scar_mod.apply_scar_to_labelmap(arr, scar_mask, replace=req.replace)
    labels.write_label_nifti(new_label, base, labels.corrected_path(case_id))
    postprocess.render_seg_previews(work, new_label, orch.segmentation_preview_dir(case_id), density_vol=vol)
    metrics = scar_mod.quantify(new_label, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
    metrics["scar_method"] = (req.method or "hysteresis") + profile_note
    orch.write_manifest_value(case_id, {"scar_metrics": metrics,
                                        "segmentation_preview_dir": str(orch.segmentation_preview_dir(case_id))})
    return {"case_info": orch.current_case_info(case_id), "metrics": metrics,
            "merged_with_existing": had_scar and not req.replace,
            "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id))}


class ScarEditRequest(BaseModel):
    voxels: List[List[int]]          # [[i,j,k], …] brush footprint on one slice
    mode: str = "paint"              # "paint" (cornea→scar) | "erase" (scar→cornea)


@app.post("/api/case/{case_id}/scar/edit")
def scar_edit(case_id: str, req: ScarEditRequest) -> dict:
    """Manual 2D scar edit: paint (cornea→scar) or erase (scar→cornea) the listed voxels
    in the canonical labelmap, then re-render the overlay + re-quantify (correct geometry).
    Hand-fixes the voted consensus scar (or any case's scar) before it becomes ground truth.
    Paint only promotes cornea→scar and erase only demotes scar→cornea, so scar ⊆ cornea
    is preserved and the cornea boundary is never touched."""
    import nibabel as nib
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        raise HTTPException(400, "No segmentation to edit — segment the cornea first.")
    v = np.asarray(req.voxels or [], dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) == 0:
        raise HTTPException(400, "voxels must be a non-empty list of [i, j, k].")
    s = arr.shape
    inb = (v[:, 0] >= 0) & (v[:, 0] < s[0]) & (v[:, 1] >= 0) & (v[:, 1] < s[1]) & (v[:, 2] >= 0) & (v[:, 2] < s[2])
    v = v[inb]
    if len(v) == 0:
        raise HTTPException(400, "All edit voxels were out of bounds.")
    ii, jj, kk = v[:, 0], v[:, 1], v[:, 2]
    cur = arr[ii, jj, kk]
    cur = np.where(cur == 2, 1, cur) if req.mode == "erase" else np.where(cur == 1, 2, cur)
    arr[ii, jj, kk] = cur

    base = _ensure_volume_nifti(case_id)
    work = _working_volume(case_id)
    vol = np.asarray(nib.load(str(work)).dataobj).astype(np.float32)
    raw = np.asarray(nib.load(str(base)).dataobj).astype(np.float32)
    labels.write_label_nifti(arr, base, labels.corrected_path(case_id))
    postprocess.render_seg_previews(work, arr, orch.segmentation_preview_dir(case_id), density_vol=vol)
    metrics = scar_mod.quantify(arr, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
    orch.write_manifest_value(case_id, {"scar_metrics": metrics})
    return {"metrics": metrics,
            "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id))}


class ScarClick(BaseModel):
    ijk: List[int]
    orientation: str                 # axial | coronal | sagittal
    positive: bool = True            # True = this is scar, False = not scar


class ScarHintRequest(BaseModel):
    points: List[ScarClick]
    replace: bool = False            # False: add SAM2 scar to existing; True: replace scar
    percentile: float = 80.0         # brightness cut that delineates scar within the click region


@app.post("/api/case/{case_id}/scar/sam2-hint")
def scar_sam2_hint(case_id: str, req: ScarHintRequest) -> dict:
    """Guide scar with SAM2: the user's clicked points (positive = scar, negative =
    not) prompt SAM2 to segment scar within the cornea; the result is merged into
    (or replaces) the scar in the canonical labelmap, then re-rendered + quantified."""
    import sam2_segment
    import nibabel as nib
    orch.ensure_case_dirs(case_id)
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None or not ((arr == 1) | (arr == 2)).any():
        raise HTTPException(400, "No cornea segmentation yet. Run SAM2 first.")
    points = [p.model_dump() for p in req.points]
    if not any(p["positive"] for p in points):
        raise HTTPException(400, "Add at least one positive (scar) click.")
    s = arr.shape
    for p in points:                                # reject OOB clicks before the GPU lock
        ijk = p.get("ijk") or []
        if len(ijk) != 3 or not all(0 <= ijk[d] < s[d] for d in range(3)):
            raise HTTPException(400, f"Click {ijk} is outside the volume {tuple(s)}.")
    base = _ensure_volume_nifti(case_id)
    work_vol = _working_volume(case_id)
    vol = np.asarray(nib.load(str(work_vol)).dataobj).astype(np.float32)
    raw = np.asarray(nib.load(str(base)).dataobj).astype(np.float32)
    work = orch.case_root(case_id) / "sam2_work"
    with _GPU_LOCK:                                  # one SAM2/CUDA inference at a time
        scar_sam, meta = sam2_segment.segment_scar_from_clicks(base, arr, points, work)
    # SAM2 localizes *where* you clicked; constrain it to the hyper-reflective tissue
    # so it keeps only the scar within that region (a raw point grabs the whole band).
    from scipy import ndimage
    bright = scar_mod.hyper_reflective_mask(vol, arr, percentile=req.percentile)
    scar_click = scar_sam & bright
    lbl, n = ndimage.label(scar_click)
    if n:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        scar_click = np.isin(lbl, [i + 1 for i, s in enumerate(sizes) if s >= 200])
    meta["constrained_voxels"] = int(scar_click.sum())
    if scar_click.sum() == 0:
        raise HTTPException(422, "No hyper-reflective scar in the clicked region — "
                                 "click on brighter tissue or lower the brightness cut.")
    existing = arr == 2
    cornea = (arr == 1) | (arr == 2)
    new_scar = (scar_click if req.replace else (existing | scar_click)) & cornea
    new_label = np.where(cornea, 1, 0).astype(np.uint8)
    new_label[new_scar] = 2
    labels.write_label_nifti(new_label, base, labels.corrected_path(case_id))
    postprocess.render_seg_previews(work_vol, new_label, orch.segmentation_preview_dir(case_id), density_vol=vol)
    metrics = scar_mod.quantify(new_label, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
    metrics["sam2_hint"] = meta
    orch.write_manifest_value(case_id, {"scar_metrics": metrics})
    return {"case_info": orch.current_case_info(case_id), "metrics": metrics,
            "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id))}


class ScarAutoSam2Request(BaseModel):
    percentile: float = 88.0       # brightness cut for the candidate + final hyper-reflective constraint
    erode_surface: int = 6         # drop epithelium/Bowman's/endothelium rind before seeding
    smooth: float = 2.5            # in-plane smoothing for the brightness candidate
    vote: int = 2                  # consensus: keep voxels ≥ this many of the 3 views agree
    min_voxels: int = 200          # drop connected components smaller than this
    max_seeds: int = 5             # how many bright components to seed SAM2 from
    replace: bool = False          # False: merge with existing scar (keep manual edits)
    use_scar_range: bool = True    # confine to the frames marked as containing scar (if any)


@app.post("/api/case/{case_id}/scar/auto-sam2")
def scar_auto_sam2(case_id: str, req: ScarAutoSam2Request) -> dict:
    """Automatic scar via the cornea-style strategy: auto-seed from the brightest in-cornea region
    (optionally within the marked scar frame-range), run SAM2 on axial+coronal+sagittal as videos,
    keep the ≥`vote`-of-3 CONSENSUS, then constrain to hyper-reflective tissue. No clicks needed."""
    import nibabel as nib
    orch.ensure_case_dirs(case_id)
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None or not ((arr == 1) | (arr == 2)).any():
        raise HTTPException(400, "No cornea segmentation yet. Run SAM2 first.")
    base = _ensure_volume_nifti(case_id)            # raw: geometry + comparable reflectivity (density)
    work_vol = _working_volume(case_id)             # contrast-enhanced volume the user sees
    vol = np.asarray(nib.load(str(work_vol)).dataobj).astype(np.float32)
    raw = np.asarray(nib.load(str(base)).dataobj).astype(np.float32)
    spacing = nib.load(str(base)).header.get_zooms()
    # Confine to the marked scar frames (the user knows where scar is) — removes out-of-range
    # false positives and focuses the seeds. No-op if no range was marked.
    m = orch.read_manifest(case_id)
    frame_mask = scar_mod.frame_range_mask(arr.shape, spacing, m.get("scar_range")) if req.use_scar_range else None
    seeds, bright = scar_mod.auto_scar_seeds(vol, arr, percentile=req.percentile, erode_surface=req.erode_surface,
                                             smooth=req.smooth, frame_mask=frame_mask, max_seeds=req.max_seeds)
    if not seeds:
        raise HTTPException(422, "No hyper-reflective scar candidate found in the cornea"
                                 + (" (within the marked frames)." if frame_mask is not None else "."))
    work = orch.case_root(case_id) / "sam2_work"
    with _GPU_LOCK:                                  # one SAM2/CUDA inference at a time
        fused, meta = sam2_segment.segment_scar_consensus(base, arr, seeds, work, vote=req.vote)
    # Constrain the 3-view consensus to hyper-reflective tissue + coherent components (as the click path does).
    from scipy import ndimage
    scar_c = fused & bright
    lbl, n = ndimage.label(scar_c)
    if n:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        scar_c = np.isin(lbl, [i + 1 for i, s in enumerate(sizes) if s >= req.min_voxels])
    meta["seeds"] = seeds
    meta["constrained_voxels"] = int(scar_c.sum())
    if scar_c.sum() == 0:
        raise HTTPException(422, "3-view SAM2 consensus found no hyper-reflective scar — try a lower "
                                 "percentile or vote=1.")
    cornea = (arr == 1) | (arr == 2)
    existing = arr == 2
    new_scar = (scar_c if req.replace else (existing | scar_c)) & cornea
    new_label = np.where(cornea, 1, 0).astype(np.uint8)
    new_label[new_scar] = 2
    labels.write_label_nifti(new_label, base, labels.corrected_path(case_id))
    postprocess.render_seg_previews(work_vol, new_label, orch.segmentation_preview_dir(case_id), density_vol=vol)
    metrics = scar_mod.quantify(new_label, spacing, density_vol_ijk=raw)
    metrics["scar_auto_sam2"] = meta
    orch.write_manifest_value(case_id, {"scar_metrics": metrics,
                                        "segmentation_preview_dir": str(orch.segmentation_preview_dir(case_id))})
    return {"case_info": orch.current_case_info(case_id), "metrics": metrics,
            "merged_with_existing": bool((arr == 2).any()) and not req.replace,
            "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id))}


class MetricsSummaryRequest(BaseModel):
    cases: List[str] | None = None   # default: all cases with a labelmap


@app.post("/api/metrics/summary")
def metrics_summary(req: MetricsSummaryRequest) -> dict:
    """Recompute scar volume (mm³) + en-face area (mm²) + density for every case
    from its current corrected labelmap and write scar_summary.csv/.json."""
    rows = metrics_export.build_summary(req.cases)
    paths = metrics_export.write_summary(rows)
    return {"rows": rows, **paths}


# ── Multi-scan consensus (repeat acquisitions of one eye) ──────────────────
@app.post("/api/consensus/upload")
async def consensus_upload(files: List[UploadFile] = File(...)) -> dict:
    """Upload several volumes (repeat scans of an eye) → one case per file."""
    if not files:
        raise HTTPException(400, "No files uploaded.")
    cases = []
    for idx, up in enumerate(files):
        name = up.filename or f"scan_{idx + 1}"
        meta = metrics_export.parse_case_meta(name)
        if meta["patient_id"]:
            cid = orch.safe_case_id(f"case_{meta['patient_id'].lower()}_{meta['eye'].lower()}_v{meta['variant'] or (idx + 1)}")
        else:
            cid = orch.safe_case_id(f"scan_{Path(name).stem}")
        orch.ensure_case_dirs(cid)
        dest = orch.case_root(cid) / "input" / Path(name).name
        dest.write_bytes(await up.read())
        orch.write_manifest_value(cid, {"input_volume": str(dest), "corrected_volume": str(dest)})
        seg = labels.best_labelmap_nnunet(cid)[0]
        has_scar = bool(seg is not None and (seg == 2).any())
        cases.append({"case_id": cid, "filename": name, "segmented": has_scar, **meta})
    return {"cases": cases}


def _ensure_segmented(case_id: str) -> None:
    """Make sure a case has a cornea+scar labelmap (preprocess → SAM2 → scar/auto)."""
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        if not _preprocessed_path(case_id).exists():
            preprocess_case(case_id, PreprocessRequest(enabled=True))
        segment_sam2(case_id, Sam2Request())
        scar_auto(case_id, ScarAutoRequest())
    elif not (arr == 2).any():
        if not _preprocessed_path(case_id).exists():
            preprocess_case(case_id, PreprocessRequest(enabled=True))
        scar_auto(case_id, ScarAutoRequest())


@app.post("/api/case/{case_id}/consensus-segment")
def consensus_segment_case(case_id: str) -> dict:
    """Segment one consensus scan (preprocess → SAM2 → scar/auto). Driven per-scan by
    the frontend so the panel can show live per-scan progress."""
    import nibabel as nib
    _ensure_segmented(case_id)
    arr, _ = labels.best_labelmap_nnunet(case_id)
    base = _ensure_volume_nifti(case_id)
    # Render this scan's own cornea+scar as a dense+rotated overlay for the gallery's 3rd
    # before/after panel ("This scan") — available even for a single-scan subgroup (no consensus).
    try:
        postprocess.render_seg_previews(base, arr, _preview_group_dir(case_id, "context_seg"), dense_rotated=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[consensus-segment] context_seg render skipped for {case_id}: {exc}", file=sys.stderr)
    m = scar_mod.quantify(arr, nib.load(str(base)).header.get_zooms())
    return {"case_id": case_id, "scar_present": m["scar_present"],
            "scar_volume_mm3": m["scar_volume_mm3"]}


class ConsensusBuildRequest(BaseModel):
    cases: List[str]
    reference: str | None = None
    group: str | None = None
    subgroup: str | None = None   # replicate set WITHIN the eye (e.g. "posterior"); "1"/blank = default


def _read_label_ijk(path: Path) -> np.ndarray:
    import nibabel as nib
    return np.rint(np.asarray(nib.load(str(path)).dataobj)).astype(np.uint8)


def _subgroup_slug(subgroup: str | None) -> str:
    """Normalise a subgroup label for a case-id segment. The default subgroup ("1"/blank)
    yields "" so the id stays the back-compatible `case_<pid>_<eye>_consensus`; a real
    subgroup ("posterior") becomes a slug inserted before `_consensus`."""
    s = (subgroup or "").strip().lower()
    if s in ("", "1"):
        return ""
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9_-]+", "-", s)).strip("-")


def _consensus_case_id(cases: List[str], group: str | None = None, subgroup: str | None = None) -> str:
    """Deterministic consensus case id. An explicit `group` (the cohort path) wins; otherwise
    derive a stable EYE×SUBGROUP id from the members' shared identity
    (`case_<patient>_<eye>[_<subgroup>]_consensus`, lowercased to match the cohort) so the
    endpoint and the cohort converge on ONE id, and two subgroups of the SAME eye don't collide.
    Falls back to an order-independent id if unparseable."""
    if group:
        return orch.safe_case_id(group)
    sub = _subgroup_slug(subgroup)
    seg = f"_{sub}" if sub else ""
    m0 = orch.read_manifest(cases[0]) if cases else {}
    # Prefer a user-corrected patient/eye persisted on the case (group-header edit); fall back
    # to parsing the original filename.
    pid = (m0.get("patient_id") or "").strip()
    eye = (m0.get("eye") or "").strip()
    if not (pid and eye):
        meta = metrics_export.parse_case_meta(m0.get("oct_source") or m0.get("input_volume"))
        pid = pid or meta.get("patient_id", "")
        eye = eye or meta.get("eye", "")
    if pid and eye:
        return orch.safe_case_id(f"case_{pid.lower()}_{eye.lower()}{seg}_consensus")
    return orch.safe_case_id("_".join(sorted(cases)) + seg + "_consensus")


def _build_consensus_case(cases: List[str], group: str | None = None,
                          reference: str | None = None, ensure: bool = True,
                          subgroup: str | None = None) -> tuple[str, dict]:
    """Segment each scan (if needed), register + vote a partial-overlap consensus, render
    the per-tab previews, and persist. Shared by the /consensus/build endpoint and the
    cohort batch. `subgroup` keeps replicate sets of the SAME eye (e.g. posterior vs inferior)
    in separate consensus cases. Returns (consensus_case_id, report)."""
    import nibabel as nib
    cases = list(dict.fromkeys(cases))      # de-dupe members (order-preserving) so a
    if len(cases) < 2:                      # repeated id can't double-count in CV%
        raise ValueError("Need at least 2 scans of the same eye for consensus.")
    seg_errors: dict = {}
    if ensure:
        for cid in cases:
            try:
                _ensure_segmented(cid)
            except HTTPException as exc:
                seg_errors[cid] = str(exc.detail)
            except Exception as exc:  # noqa: BLE001
                seg_errors[cid] = str(exc)

    ccid = _consensus_case_id(cases, group, subgroup)
    orch.ensure_case_dirs(ccid)
    report = consensus_mod.build_consensus(cases, ccid, reference)
    report["segmentation_errors"] = seg_errors
    sub_label = (subgroup or "1").strip() or "1"
    report["subgroup"] = sub_label

    cons_vol = orch.case_root(ccid) / "previews" / "volume.nii.gz"
    cons_lab = _read_label_ijk(labels.corrected_path(ccid))
    postprocess.render_seg_previews(cons_vol, cons_lab, _preview_group_dir(ccid, "segmentation"))
    # Per-scan tabs: each scan's warped image with its own scar, and with the consensus
    # scar clipped to that scan's FOV (so it isn't painted over empty background).
    scans_dir = orch.case_root(ccid) / "scans"
    for cid in report["scans"]:
        svol = scans_dir / cid / "volume.nii.gz"
        slab = _read_label_ijk(scans_dir / cid / "label.nii.gz")
        data_mask = np.asarray(nib.load(str(svol)).dataobj) > 0
        cons_clipped = np.where(data_mask, cons_lab, 0).astype(np.uint8)
        postprocess.render_seg_previews(svol, slab, _preview_group_dir(ccid, f"scan_{cid}_self"))
        postprocess.render_seg_previews(svol, cons_clipped, _preview_group_dir(ccid, f"scan_{cid}_cons"))
        # Dense+rotated overlays in the SCAN's NATIVE frame for the gallery's 3rd before/after
        # panel (aligns slice-for-slice with raw/corrected): own cornea+scar (context_seg) and
        # the subgroup consensus mapped to native (context_cons). Convenience — never fail build.
        try:
            nat_vol = orch.case_root(cid) / "previews" / "volume.nii.gz"
            own = _read_label_ijk(labels.corrected_path(cid))
            postprocess.render_seg_previews(nat_vol, own, _preview_group_dir(cid, "context_seg"), dense_rotated=True)
            cons_native = scans_dir / cid / "cons_native.nii.gz"
            if cons_native.exists():
                postprocess.render_seg_previews(nat_vol, _read_label_ijk(cons_native),
                                                _preview_group_dir(cid, "context_cons"), dense_rotated=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[consensus] native before/after panel skipped for {cid}: {exc}", file=sys.stderr)
        # Link the scan back to its consensus + subgroup (frontend panel + metrics attribution).
        orch.write_manifest_value(cid, {"consensus_case": ccid, "scar_subgroup": sub_label})

    orch.write_manifest_value(ccid, {
        "input_volume": str(cons_vol), "corrected_volume": str(cons_vol),
        "consensus_report": report, "consensus_cases": report["scans"], "reference": report["reference"],
        "scar_subgroup": sub_label,
    })
    return ccid, report


@app.post("/api/consensus/build")
def consensus_build(req: ConsensusBuildRequest) -> dict:
    """Segment each scan (if needed), scar-anchor-register the repeats, build a
    probabilistic partial-overlap consensus, and render per-tab previews."""
    if len(req.cases) < 2:
        raise HTTPException(400, "Upload at least 2 scans of the same eye for consensus.")
    try:
        ccid, report = _build_consensus_case(req.cases, req.group, req.reference, subgroup=req.subgroup)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"consensus_case": ccid, "report": report,
            "images": orch.preview_images_from_dir("Segmentation", _preview_group_dir(ccid, "segmentation"))}


# ── .OCT preprocessing (Optovue Avanti): inspect → correct → register case ──
# Pipeline (oct_preprocess.py, ported from the user's OCT_Extraction scripts):
#   upload .OCT (+ companion .txt) → raw z-stack NIfTI for scrubbing → on Run, the
#   corneal-edge + column + 3D-active correction → corrected NIfTI (correct Avanti
#   geometry) which becomes the case's working volume for SAM2/consensus.
class OctPreprocessRequest(BaseModel):
    params: dict | None = None
    volume_index: int | None = None
    classification: str | None = None   # "scar" | "control" (no scar) | None
    scar_range: List[int] | None = None  # [start_frame, end_frame], 1-based
    patient: str | None = None  # user-corrected identity (group header edit); overrides filename parse
    eye: str | None = None      # "OD"/"OS"; "?" / blank are ignored
    force_columns: List[int] | None = None  # BAD frame indices to re-correct ("re-run preprocessing")
    good_columns: List[int] | None = None    # GOOD/anchor frame indices guiding the re-correction
                                             # (all reuse the scan's persisted settings)
    max_iterations: int | None = None        # >1 = iterative refinement (auto-converge); 1 = single faithful pass
    inject_pass: int | None = None           # re-run iteration applying force_columns at ONLY this pass (1-based)
    manual_shifts: dict | None = None        # #2 drag-to-correct: {frame_index: depth_px} manual per-frame
                                             # depth nudges (positive = DOWN), applied LAST as manual ground truth
    slice_index: int | None = None           # steps viewer: which sagittal slice to render the border+fit on
    border_pass: int | None = None            # border-curve: which pass to fix — detect on its INPUT (raw for
                                              # pass 1, the prior pass's output for pass>1), never the result
    border_anchors: dict | None = None        # fix-columns "Confirm": {str(slice_index): {str(frame): true_depth}}
                                              # corrected ABSOLUTE surface depths (depth 0 = TOP). The server MARCHES
                                              # a tilt-aware re-detection of the whole RAW volume seeded by these.
    use_redetect: bool | None = None          # oct-preprocess: flatten to the confirmed re-detected surface
                                              # (provided_edges) instead of auto-detecting — the fix-columns "Run".
    parabola: bool | None = None              # fix-columns "Confirm" parabola mode: the anchors are a DENSE fitted
                                              # quadratic → use it EXACTLY (seed window 0), don't re-snap per frame.
    ascan_rate_hz: float | None = None        # eye-motion tab: A-scan (line) rate → frame rate → Hz axis (Avanti ~70000)
    detrend_order: int | None = None          # eye-motion tab: per-A-line shape-removal polynomial order (default 2)
    sinc_correct: bool | None = None          # eye-motion tab: divide out the intra-frame motion-blur boxcar


def _oct_working_path(case_id: str, src: str) -> Path:
    return orch.case_root(case_id) / "input" / f"{orch.safe_case_id(Path(src).stem)}.nii.gz"


def _oct_case_taken(cid: str, name: str) -> bool:
    """True if a case with this id already holds a DIFFERENT .OCT (don't reuse/overwrite it)."""
    src = orch.read_manifest(cid).get("oct_source")
    return bool(src) and Path(src).name != name


def _nifti_frames(path: Path) -> int:
    """Frame count (z dim) of a working NIfTI — drives the scar frame-range slider."""
    import nibabel as nib
    try:
        return int(nib.load(str(path)).shape[2])
    except Exception:  # noqa: BLE001
        return 0


def _run_oct_worker(mode: str, src: str, out: Path, params: dict, vi: int,
                    companion: str | None = None, extra: list | None = None) -> str:
    """Run the oct_preprocess CLI in an isolated subprocess (keeps its fork-based
    parallelism away from the sidecar's CUDA/torch state). New session so a timeout can
    reap the whole fork-pool process group. `companion` = the .txt filespec whose
    per-scan geometry (XY Scan Size1 etc.) is baked into the NIfTI spacing. `extra` =
    mode-specific flags (e.g. --bad-cols for the steps filmstrip)."""
    import os
    import signal
    cmd = [sys.executable, str(Path(oct_mod.__file__)), mode, str(src), str(out),
           "--params", json.dumps(params or {}), "--volume-index", str(vi)]
    if companion:
        cmd += ["--companion-txt", str(companion)]
    if extra:
        cmd += [str(x) for x in extra]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout_text, err = proc.communicate(timeout=1200)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise HTTPException(504, f"OCT {mode} timed out (>1200s).")
    if proc.returncode != 0 or not Path(out).exists():
        raise HTTPException(500, f"OCT {mode} failed: {(err or '')[-800:]}")
    return stdout_text or ""


def _previews_fresh(out_dir: Path, base: Path) -> bool:
    """True if the slice PNGs in out_dir are present, render-current (.rev3 marker), and
    newer than the volume — so we can skip the expensive re-render on a repeat scrub."""
    manifest = out_dir / "preview_manifest.json"
    marker = out_dir / ".rev3"   # bump when the render changes (rotation / dense slices) to invalidate old PNGs
    if not (manifest.exists() and marker.exists()):
        return False
    try:
        return manifest.stat().st_mtime >= base.stat().st_mtime
    except OSError:
        return False


def _ensure_raw_snapshot(case_id: str, raw_dir: Path) -> bool:
    """Render the pre-correction ("before") slices into raw_dir from a FRESH conversion of
    the original .OCT — never from the working volume (which is the CORRECTED one once a scan
    has been preprocessed). Copying the working context/ was the "before == after" bug: that
    directory already held the corrected slices, so both panels showed corrected. Idempotent:
    a no-op once a CURRENT (.rev3) snapshot exists — an older one (the buggy corrected-as-raw
    snapshot, or a sparse render) is regenerated. Returns True if present afterwards."""
    if (raw_dir / "preview_manifest.json").exists() and (raw_dir / ".rev3").exists():
        return True
    m = orch.read_manifest(case_id)
    src = m.get("oct_source")
    if not src or not Path(src).exists():
        return False
    vi = int(m.get("oct_volume_index", 0))
    tmp = orch.case_root(case_id) / "input" / "_raw_snapshot.nii.gz"
    try:
        oct_mod.raw_oct_to_nifti(src, tmp, volume_index=vi, companion_txt=m.get("companion_txt"))
        postprocess.render_context_previews(tmp, raw_dir)
        (raw_dir / ".rev3").write_text("")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[oct] raw before/after snapshot failed: {exc}", file=sys.stderr)
        return False
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass


def _oct_render_volume(case_id: str, work: Path, preprocessed: bool, extra: dict,
                       render_previews: bool = True) -> dict:
    """Point the case at `work` as its volume, drop stale segmentation, render grayscale.
    Records the resolved per-scan voxel spacing (from the companion .txt) and warns if it
    falls outside the plausible Avanti range — a wrong-geometry volume silently corrupts
    every downstream scar mm³/mm² metric."""
    import nibabel as nib
    spacing = None
    geom_warnings: list = []
    try:
        spacing = [float(z) for z in nib.load(str(work)).header.get_zooms()[:3]]
        geom_warnings = oct_mod.validate_spacing(spacing)
    except Exception:  # noqa: BLE001
        pass
    orch.write_manifest_value(case_id, {
        "input_volume": str(work), "corrected_volume": str(work),
        "oct_preprocessed": preprocessed, "oct_spacing": spacing, **extra,
    })
    if not preprocessed:
        # Showing a RAW capture (fresh scrub or a switched volume_index): drop any stale
        # segmentation from a prior capture so it can't be applied to the new volume (and
        # write_label_nifti's shape guard can't later reject a mismatched leftover).
        import shutil
        seg_dir = orch.segmentation_preview_dir(case_id)
        if seg_dir.exists():
            shutil.rmtree(seg_dir, ignore_errors=True)
        # Drop the before/after 3rd-panel overlays too — they belong to the dropped segmentation.
        for grp in ("context_seg", "context_cons"):
            shutil.rmtree(_preview_group_dir(case_id, grp), ignore_errors=True)
        _clear_iter_preview_groups(case_id)   # stale per-pass refinement previews
        labels.corrected_path(case_id).unlink(missing_ok=True)
        orch.case_qa_json(case_id).unlink(missing_ok=True)
        orch.write_manifest_value(case_id, {"scar_metrics": None})
    base = _ensure_volume_nifti(case_id)
    if render_previews:
        # context/ holds the CURRENT working slices (raw while scrubbing, corrected after
        # preprocessing) so the single "Slices" view always matches the working volume. For a
        # CORRECTED scan, also ensure the "before" snapshot exists — rendered from the original
        # .OCT (NEVER copied from context/, which now holds corrected slices). Cache: skip the
        # (expensive) re-render when the slices are already up to date.
        ctx = orch.context_preview_dir(case_id)
        if preprocessed:
            _ensure_raw_snapshot(case_id, _preview_group_dir(case_id, "context_raw"))
        if not _previews_fresh(ctx, base):
            postprocess.render_context_previews(base, ctx)
            (ctx / ".rev3").write_text("")
    # The gallery pulls slices lazily via /previews + /preview-file, so don't base64 the (now
    # DENSE) context group into this response — it would inline tens of MB the frontend ignores.
    return {"case_info": orch.current_case_info(case_id), "spacing": spacing,
            "geometry_warnings": geom_warnings, "images": []}


@app.post("/api/oct/upload")
async def oct_upload(files: List[UploadFile] = File(...)) -> dict:
    """Upload .OCT files (+ optional companion .txt). One case per .OCT; metadata is
    parsed from the filename + companion. No conversion yet — fast for whole directories."""
    if not files:
        raise HTTPException(400, "No files uploaded.")
    blobs = [(up.filename or "", await up.read()) for up in files]
    octs = [(n, b) for n, b in blobs if n.lower().endswith(".oct")]
    txts = {Path(n).stem.lower(): (n, b) for n, b in blobs if n.lower().endswith(".txt")}
    if not octs:
        raise HTTPException(400, "No .OCT files found (also drop the companion .txt files).")
    used: set = set()
    cases = []
    for name, data in octs:
        fm = oct_mod.parse_oct_filename(name)
        if fm.get("patient_id"):
            base = orch.safe_case_id(f"case_{fm['patient_name'].lower()}_{fm['laterality'].lower()}_v{fm.get('series_number', 1)}")
        else:
            base = orch.safe_case_id(f"oct_{Path(name).stem}")
        # Unique per distinct .OCT: reuse iff the same file is already there, else suffix —
        # otherwise repeat scans of one eye (the consensus case!) would overwrite each other.
        cid, k = base, 2
        while cid in used or _oct_case_taken(cid, name):
            cid = f"{base}_{k}"
            k += 1
        used.add(cid)
        orch.ensure_case_dirs(cid)
        oct_dst = orch.case_root(cid) / "input" / Path(name).name
        oct_dst.write_bytes(data)
        comp = txts.get(Path(name).stem.lower())
        txt_dst = None
        if comp:
            txt_dst = orch.case_root(cid) / "input" / Path(comp[0]).name
            txt_dst.write_bytes(comp[1])
        meta = oct_mod.metadata_for(name, str(txt_dst) if txt_dst else None)
        orch.write_manifest_value(cid, {
            "oct_source": str(oct_dst), "companion_txt": str(txt_dst) if txt_dst else None,
            "oct_volume_index": 0, "oct_preprocessed": False,
        })
        cases.append({"case_id": cid, "filename": name, "patient": meta["patient_name"],
                      "eye": fm.get("laterality", ""), "preprocessed": False})
    return {"cases": cases}


@app.post("/api/case/{case_id}/oct-volume")
def oct_volume(case_id: str, req: OctPreprocessRequest) -> dict:
    """Materialise the RAW .OCT z-stack as the working NIfTI + grayscale previews so the
    user can scrub/inspect before correcting. Lazy — only the previewed scan is read."""
    m = orch.read_manifest(case_id)
    src = m.get("oct_source")
    if not src or not Path(src).exists():
        raise HTTPException(400, f"Case {case_id} has no .OCT source.")
    vi = req.volume_index if req.volume_index is not None else int(m.get("oct_volume_index", 0))
    work = _oct_working_path(case_id, src)
    changed_index = req.volume_index is not None and req.volume_index != int(m.get("oct_volume_index", 0))
    # If the case is already corrected (and we're not switching capture), RE-SHOW the
    # corrected volume rather than reverting it to raw — re-inspecting must not clobber it.
    show_corrected = bool(m.get("oct_preprocessed")) and work.exists() and not changed_index
    # Cache the raw conversion: a raw z-stack already materialised for this same capture can
    # be reused as-is, so re-clicking a scan is instant instead of re-reading + reconverting
    # the .OCT every time (the main cause of slow scan-to-scan scrubbing). Validate the cached
    # file (>1 frame) so a truncated leftover from a killed preprocess can't be served — fall
    # through to a fresh conversion, which self-heals it.
    reuse_raw = (work.exists() and not changed_index and not m.get("oct_preprocessed")
                 and _nifti_frames(work) > 1)
    if not show_corrected and not reuse_raw:
        try:
            oct_mod.raw_oct_to_nifti(src, work, volume_index=vi, companion_txt=m.get("companion_txt"))
        except oct_mod.MissingCompanionError as exc:
            raise HTTPException(400, str(exc))           # actionable: user forgot the .txt
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Reading .OCT failed: {exc}")
    # Render the slice PNGs the 2D gallery shows — but _oct_render_volume now CACHES them
    # (skips the render when up to date), so a repeat scrub is cheap while a first view still
    # gets its slices immediately (no extra on-demand round-trip).
    out = _oct_render_volume(case_id, work, preprocessed=show_corrected, extra={"oct_volume_index": vi})
    out["n_frames"] = _nifti_frames(work)
    out["preprocessed"] = show_corrected
    return out


@app.post("/api/case/{case_id}/oct-preprocess")
def oct_preprocess_case(case_id: str, req: OctPreprocessRequest) -> dict:
    """Run the corneal-edge + column + 3D-active correction on the case's .OCT and make
    the corrected volume (correct Avanti geometry) the working volume for SAM2/consensus.
    Persists the scar/control classification + scar frame range for the later Scar stage."""
    m = orch.read_manifest(case_id)
    src = m.get("oct_source")
    if not src or not Path(src).exists():
        raise HTTPException(400, f"Case {case_id} has no .OCT source.")
    vi = req.volume_index if req.volume_index is not None else int(m.get("oct_volume_index", 0))
    work = _oct_working_path(case_id, src)
    # Snapshot the pre-correction ("before") slices for the before/after view. If the scan was
    # already scrubbed, context/ holds the genuine RAW slices (not yet corrected) — copy them
    # cheaply BEFORE the correction overwrites the working volume, so we don't re-decode the
    # .OCT. Otherwise _oct_render_volume renders the "before" from a fresh .OCT conversion (it
    # must never copy the post-correction context/). Convenience only — never block preprocess.
    raw_dir = _preview_group_dir(case_id, "context_raw")
    ctx = orch.context_preview_dir(case_id)
    if (not (raw_dir / "preview_manifest.json").exists()
            and not m.get("oct_preprocessed")
            and (ctx / "preview_manifest.json").exists()):
        try:
            import shutil
            shutil.copytree(ctx, raw_dir, dirs_exist_ok=True)   # context/ is genuine RAW here
            (raw_dir / ".rev3").write_text("")
        except Exception as exc:  # noqa: BLE001
            print(f"[oct-preprocess] raw snapshot copy skipped: {exc}", file=sys.stderr)
    # Merge persisted settings so a viewer "re-run on bad columns" reuses the scan's original
    # params / classification / scar range and just ADDS the force_columns override (which then
    # sticks, so the user's column fix survives later re-runs). A normal preprocess (full params
    # from the loader, no force_columns) keeps its prior behaviour.
    eff_params = {**(m.get("oct_params") or {}), **(req.params or {})}

    def _int_list(xs):
        out = []
        for c in xs or []:
            try:
                out.append(int(c))
            except (ValueError, TypeError):
                pass
        return out

    # Per-pass column fix (inject_pass set): the marked frames apply to ONLY that pass of the
    # iteration — they must NOT become global eff_params (which would force them at every pass). They
    # ride along as --inject-* instead. Otherwise (legacy single-pass Fix-columns re-run, no
    # inject_pass) force_columns stays a global param as before.
    inject_pass = req.inject_pass if (req.inject_pass and int(req.inject_pass) > 0) else None
    if inject_pass is None:
        if req.force_columns is not None:
            eff_params["force_columns"] = _int_list(req.force_columns)
        if req.good_columns is not None:
            eff_params["good_columns"] = _int_list(req.good_columns)
    else:
        eff_params.pop("force_columns", None)
        eff_params.pop("good_columns", None)
    eff_params.pop("coronal_check", None)    # removed feature — strip any stale persisted flag
    eff_params.pop("manual_columns", None)   # removed feature — strip any stale persisted nudges
    # #2 drag-to-correct: explicit per-frame manual depth nudges. When provided (non-None), REPLACE the
    # persisted set (an empty {} clears them); when omitted, the persisted nudges carry through so manual
    # ground truth stays applied on every later re-run. Flows to the worker inside eff_params (--params).
    if req.manual_shifts is not None:
        eff_params["manual_shifts"] = req.manual_shifts
    # Sanitize the EFFECTIVE set (request-provided OR carried-through from persisted oct_params): drop any
    # zero / NaN / Infinity / malformed entry so the manifest never accumulates no-op garbage and always
    # matches the frontend's zero-free view (a zero shift is a no-op the frontend already removes).
    if eff_params.get("manual_shifts"):
        clean: dict = {}
        for k, v in dict(eff_params["manual_shifts"]).items():
            try:
                fv = float(v)
                if math.isfinite(fv) and int(round(fv)) != 0:
                    clean[str(int(k))] = int(round(fv))
            except (TypeError, ValueError, OverflowError):
                continue
        eff_params["manual_shifts"] = clean
    # Fix-columns "Run" (use_redetect): flatten the volume to the CONFIRMED tilt-aware re-detected surface
    # (the cached marched result) instead of auto-detecting — the same surface the scrub preview drew, so
    # preview == result. A SINGLE warp pass (no iteration / no axial-refine, see preprocess_oct_to_nifti).
    redetect_npz: Path | None = None
    if req.use_redetect:
        anchors = (m.get("oct_params") or {}).get("border_anchors") or {}
        if not anchors:
            raise HTTPException(400, "No confirmed border anchors to apply — drag the border and Confirm first.")
        # ensure a FRESH cache for the persisted anchors (recompute if missing/stale), then feed it to the worker
        if _redetect_surface_fresh(case_id, anchors) is None:
            _compute_redetect_cache(case_id, m, anchors)
        redetect_npz = _redetect_cache_path(case_id)
        eff_params["border_anchors"] = anchors        # keep them persisted on the case
        # the re-detect warp flattens to EXACTLY the previewed surface — legacy per-frame manual_shifts (which
        # the scrub preview does NOT show) would break preview==result, so they're superseded here.
        eff_params.pop("manual_shifts", None)
    else:
        # a NORMAL auto preprocess SUPERSEDES any prior manual re-detection: drop the persisted anchors +
        # the cached surface so a later fix-columns scrub/Run can't show/apply a stale re-detected border.
        eff_params.pop("border_anchors", None)
        eff_params.pop("detect_lo", None); eff_params.pop("detect_hi", None)   # legacy band keys, if any
        import shutil as _sh0
        _sh0.rmtree(orch.case_root(case_id) / "border_cache", ignore_errors=True)
    cls = req.classification or m.get("scar_classification")
    sr = req.scar_range or m.get("scar_range")
    # Iterative refinement: auto-converge by default (cap 8). Persisted as oct_max_iterations so a
    # later viewer re-run reuses the user's setting. Each pass re-flattens the boundary toward its
    # fit; the worker auto-stops when the correction stops shrinking (see iterate_smooth_volume).
    max_it = req.max_iterations if req.max_iterations is not None else int(m.get("oct_max_iterations", 5))
    max_it = max(1, min(8, int(max_it)))
    if redetect_npz is not None:
        max_it = 1                                  # the re-detect warp is a single, deliberate pass
    import shutil as _sh
    iter_dir = orch.case_root(case_id) / "input" / "_iter"
    _sh.rmtree(iter_dir, ignore_errors=True)        # clear stale intermediate pass NIfTIs
    _clear_iter_preview_groups(case_id)             # clear stale per-pass preview groups
    extra = ["--max-iter", str(max_it), "--iter-dir", str(iter_dir)]
    if redetect_npz is not None:
        extra += ["--provided-edges", str(redetect_npz)]   # flatten to the confirmed re-detected surface
    elif inject_pass is not None:
        extra += ["--inject-pass", str(int(inject_pass)),
                  "--inject-force", json.dumps(_int_list(req.force_columns)),
                  "--inject-good", json.dumps(_int_list(req.good_columns))]
    worker_out = _run_oct_worker("preprocess", src, work, eff_params, vi,
                                 companion=m.get("companion_txt"), extra=extra)
    iter_info = _parse_iter_info(worker_out)
    # The corrected volume just changed → drop any segmentation built on the OLD correction so a
    # stale overlay can't show on the re-corrected volume (the user re-runs SAM2 next).
    seg_dir = orch.segmentation_preview_dir(case_id)
    if seg_dir.exists():
        _sh.rmtree(seg_dir, ignore_errors=True)
    for grp in ("context_seg", "context_cons"):
        _sh.rmtree(_preview_group_dir(case_id, grp), ignore_errors=True)
    labels.corrected_path(case_id).unlink(missing_ok=True)
    orch.case_qa_json(case_id).unlink(missing_ok=True)
    extra = {"oct_volume_index": vi, "oct_params": eff_params, "scar_metrics": None,
             "oct_max_iterations": max_it, "oct_iter": iter_info,
             # a fresh preprocessing (auto OR a Fix-columns re-run) invalidates the manual-vetting and
             # training-schedule flags → the per-scan timeline drops back to "Preprocessed [Auto]" (red)
             # and the user re-approves. scar_classification is kept (it's scan content, not geometry).
             "preproc_vetted": False, "training_scheduled": False}
    if cls:
        extra["scar_classification"] = cls
    if sr:
        extra["scar_range"] = sr
    # A patient/eye corrected in the group header overrides the filename-parsed identity for
    # the later consensus naming + export — persist it so the correction isn't lost. Normalize
    # to the SAME space the filename parser uses (UPPER patient; eye constrained to OD/OS with
    # common synonyms mapped), so an override-named case still groups/merges with parsed ones.
    # An unrecognized/"?"/blank eye is ignored so it never clobbers a good filename parse.
    if req.patient and req.patient.strip():
        extra["patient_id"] = req.patient.strip().upper()
    if req.eye and req.eye.strip():
        eye = req.eye.strip().upper()
        eye = {"R": "OD", "RIGHT": "OD", "L": "OS", "LEFT": "OS"}.get(eye, eye)
        if eye in ("OD", "OS"):
            extra["eye"] = eye
    out = _oct_render_volume(case_id, work, preprocessed=True, extra=extra)
    # Render EVERY corrected pass (V1..Vm) so the user can step through all of them in the before/
    # after viewer and SEE which is best: pass 0 = context_raw, pass k = context_iter{k}; the chosen
    # best (oct_iter.best_pass) is the working "context"/volume. Best-effort — a render failure never
    # fails the preprocess (the final result is already in).
    try:
        passes = int(iter_info.get("passes", 1))
        for k in range(1, passes + 1):
            pv = iter_dir / f"pass_{k}.nii.gz"
            if pv.exists():
                grp_dir = _preview_group_dir(case_id, f"context_iter{k}")
                postprocess.render_context_previews(pv, grp_dir)
                (grp_dir / ".rev3").write_text("")
    except Exception as exc:  # noqa: BLE001
        print(f"[oct-preprocess] per-pass preview render skipped: {exc}", file=sys.stderr)
    finally:
        # Persist the per-pass NIfTIs (passes/pass_{k}.nii.gz) so the user can DOWNLOAD a specific
        # pass, not just the best. Replace any stale set; if there are no intermediates, just clean up.
        passes_dir = orch.case_root(case_id) / "passes"
        _sh.rmtree(passes_dir, ignore_errors=True)
        if iter_dir.exists() and any(iter_dir.iterdir()):
            try:
                _sh.move(str(iter_dir), str(passes_dir))
            except Exception:  # noqa: BLE001
                _sh.rmtree(iter_dir, ignore_errors=True)
        else:
            _sh.rmtree(iter_dir, ignore_errors=True)
    out["preprocessed"] = True
    out["n_frames"] = _nifti_frames(work)
    out["oct_iter"] = iter_info
    return out


@app.post("/api/case/{case_id}/keep-raw")
def keep_raw_case(case_id: str) -> dict:
    """Before/after "Use original (raw)": make the RAW (un-corrected) .OCT conversion the working volume
    — for scans where the original is already good enough and the edge/column correction would only add
    warp. Re-converts raw → working path, drops any segmentation / per-pass previews / corrected label
    built on the corrected volume, clears persisted warps (raw means no corrections), and marks the scan
    preprocessed + manually VETTED (timeline → orange) so it advances straight to classification. SAM2
    must be (re-)run afterwards. Mirrors the re-run's stale-artifact cleanup."""
    import shutil as _sh
    m = orch.read_manifest(case_id)
    src = m.get("oct_source")
    if not src or not Path(src).exists():
        raise HTTPException(400, f"Case {case_id} has no .OCT source.")
    vi = int(m.get("oct_volume_index", 0))
    work = _oct_working_path(case_id, src)
    # Raw means NO warps: strip persisted column / manual-shift corrections so neither this conversion
    # nor a later re-preprocess re-applies them on top of the (intentionally raw) volume.
    eff_params = {k: v for k, v in (m.get("oct_params") or {}).items()
                  if k not in ("force_columns", "good_columns", "manual_shifts", "manual_columns", "coronal_check",
                               "detect_lo", "detect_hi", "border_anchors")}
    _sh.rmtree(orch.case_root(case_id) / "border_cache", ignore_errors=True)   # raw = no re-detected surface
    # Convert the ORIGINAL .OCT to NIfTI with NO correction → the working volume.
    oct_mod.raw_oct_to_nifti(src, work, volume_index=vi, params=eff_params, companion_txt=m.get("companion_txt"))
    # The working volume changed → drop segmentation, per-pass previews/NIfTIs, corrected label, QA + metrics.
    seg_dir = orch.segmentation_preview_dir(case_id)
    if seg_dir.exists():
        _sh.rmtree(seg_dir, ignore_errors=True)
    for grp in ("context_seg", "context_cons"):
        _sh.rmtree(_preview_group_dir(case_id, grp), ignore_errors=True)
    _clear_iter_preview_groups(case_id)
    _sh.rmtree(orch.case_root(case_id) / "passes", ignore_errors=True)
    labels.corrected_path(case_id).unlink(missing_ok=True)
    orch.case_qa_json(case_id).unlink(missing_ok=True)
    extra = {"oct_volume_index": vi, "oct_params": eff_params, "scar_metrics": None,
             # 0 passes / best_pass 0 = raw kept (BeforeAfterViewer reads this; passCount is Math.max(1,…)-guarded).
             "oct_iter": {"passes": 0, "best_pass": 0, "metrics": [], "stopped": "kept_raw"},
             "oct_kept_raw": True,
             # the user explicitly approved the raw as the final preprocessing → vet it (timeline → orange);
             # a later auto re-preprocess clears these as usual.
             "preproc_vetted": True, "training_scheduled": False}
    if m.get("scar_classification"):
        extra["scar_classification"] = m.get("scar_classification")
    if m.get("scar_range"):
        extra["scar_range"] = m.get("scar_range")
    out = _oct_render_volume(case_id, work, preprocessed=True, extra=extra)
    out["preprocessed"] = True
    out["kept_raw"] = True
    out["n_frames"] = _nifti_frames(work)
    return out


class ClassificationRequest(BaseModel):
    classification: str | None = None   # "scar" | "control" | null (clear)
    scar_range: list[int] | None = None # optional [start,end] frame range (1-based)


@app.post("/api/case/{case_id}/classification")
def set_case_classification(case_id: str, req: ClassificationRequest) -> dict:
    """Set the scar / not-scar (control) decision AFTER preprocessing (#4) — manifest metadata only, no
    re-correction (the geometric OCT correction never used it). Mirrors the keys oct-preprocess writes so
    downstream consensus / control-baseline / nnUNet tooling keeps working, and lets the user defer the
    choice until the corrected volume exists instead of declaring it up front."""
    cls = (req.classification or "").strip().lower() or None
    if cls is not None and cls not in ("scar", "control"):
        raise HTTPException(status_code=400, detail="classification must be 'scar', 'control', or null")
    updates: dict = {"scar_classification": cls}     # None clears it
    if req.scar_range is not None:
        updates["scar_range"] = [int(x) for x in req.scar_range] or None
    m = orch.write_manifest_value(case_id, updates)
    return {"ok": True, "scar_classification": m.get("scar_classification"),
            "scar_range": m.get("scar_range")}


@app.post("/api/case/{case_id}/vet-preprocessing")
def vet_preprocessing(case_id: str) -> dict:
    """Timeline step 3: mark the preprocessing as MANUALLY VETTED (the user reviewed before/after +
    Fix-columns and approves it). Manifest metadata only — turns the scan entry orange and is the gate
    before scar/control classification. A later auto/Fix-columns re-run clears this (see oct-preprocess)."""
    m = orch.write_manifest_value(case_id, {"preproc_vetted": True})
    return {"ok": True, "preproc_vetted": bool(m.get("preproc_vetted"))}


class TrainingScheduleRequest(BaseModel):
    scheduled: bool = True


@app.post("/api/case/{case_id}/training/schedule")
def schedule_training(case_id: str, req: TrainingScheduleRequest) -> dict:
    """Timeline final step: schedule (or unschedule) this scan for nnU-Net training (turns the entry
    green). Manifest flag only; nnunet_train restricts to scheduled scans when any scan is scheduled."""
    m = orch.write_manifest_value(case_id, {"training_scheduled": bool(req.scheduled)})
    return {"ok": True, "training_scheduled": bool(m.get("training_scheduled"))}


class ObserverAnalysisRequest(BaseModel):
    root: str   # the annotator's ground-truth OUTPUT folder (contains manifest.json + <stem>/ labelmaps)


@app.post("/api/observer-analysis")
def observer_analysis(req: ObserverAnalysisRequest) -> dict:
    """#4: derive INTER-/INTRA-observer reproducibility from a folder of companion-annotator ground
    truth. Computes pairwise scar/cornea Dice (intra = same user across replicates; inter = same scan
    across users) + scar-volume CV, writes observer_{intra,inter,volume}.csv + observer_summary.json
    into the folder, and returns the summary + tables."""
    root = Path(req.root).expanduser()
    if not root.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {root}")
    import observer_analysis as _oa
    res = _oa.analyze(root)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("error", "analysis failed"))
    try:
        res["written"] = _oa.write_csvs(res, root)
    except Exception:  # noqa: BLE001
        res["written"] = []
    return res


@app.post("/api/case/{case_id}/oct-preprocess-steps")
def oct_preprocess_steps(case_id: str, req: OctPreprocessRequest) -> dict:
    """Render EVERY preprocessing step for the central sagittal slice (original → hist-eq →
    bilateral → edge → side-correct → quadratic fit → 3D active → final warp). Diagnostic
    filmstrip — does NOT touch the working volume. Reuses the scan's persisted params, plus the
    current bad-column selection (or the persisted one on a plain double-click), so the steps
    reflect exactly what a re-run would do. Returns base64 PNGs (small, one-shot)."""
    import base64
    m = orch.read_manifest(case_id)
    src = m.get("oct_source")
    if not src or not Path(src).exists():
        raise HTTPException(400, f"Case {case_id} has no .OCT source.")
    vi = req.volume_index if req.volume_index is not None else int(m.get("oct_volume_index", 0))
    eff_params = {**(m.get("oct_params") or {}), **(req.params or {})}
    # Honor explicit bad columns if the caller sent them (Fix-columns), else fall back to the
    # persisted set (a plain double-click), so the filmstrip's final warp matches a real re-run.
    persisted = m.get("oct_params") or {}
    bad = [int(c) for c in (req.force_columns if req.force_columns is not None else persisted.get("force_columns") or [])]
    out_dir = _preview_group_dir(case_id, "oct_steps")
    extra = ["--bad-cols", json.dumps(bad)]
    if req.slice_index is not None:
        extra += ["--slice-index", str(int(req.slice_index))]
    _run_oct_worker("steps", src, out_dir, eff_params, vi, companion=m.get("companion_txt"), extra=extra)
    try:
        raw = json.loads((out_dir / "labels.json").read_text())
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT steps produced no output: {exc}")
    # New worker shape is {slices, index, steps:[…]}; tolerate the legacy bare list too.
    entries = raw.get("steps", raw) if isinstance(raw, dict) else raw
    n_slices = int(raw.get("slices", 0)) if isinstance(raw, dict) else 0
    cur_index = int(raw.get("index", 0)) if isinstance(raw, dict) else 0
    steps = []
    for it in entries:
        fp = out_dir / it["file"]
        if fp.exists():
            b64 = base64.b64encode(fp.read_bytes()).decode("ascii")
            steps.append({"label": it["label"], "data_url": f"data:image/png;base64,{b64}",
                          "kind": it.get("kind", "stage"), "branch": it.get("branch", ""),
                          "group": "per-slice"})
    # ── VOLUME-LEVEL decisions (the newer steps): note nodes carrying the REAL numbers from the last
    # preprocess. These are whole-volume decisions (keep-best iteration, axial ping-pong refine,
    # inter-slice smoothing) that can't be faithfully shown from one slice, so they're reported as
    # the decision + its outcome. (Images for these need an orientation check on a real scan first.)
    oct_iter = m.get("oct_iter") or {}
    passes = int(oct_iter.get("passes", 0) or 0)
    if passes and passes > 0:
        metrics = oct_iter.get("metrics") or []
        best = oct_iter.get("best_pass")
        stopped = oct_iter.get("stopped", "")
        dev = ", ".join(f"{float(x):.2f}" for x in metrics) if metrics else "—"
        steps.append({"label": f"9. Keep-best iteration — {passes} pass(es), kept pass {best}",
                      "kind": "decision", "group": "volume",
                      "branch": f"boundary deviation per pass: [{dev}] px · argmin kept · stop: {stopped}"})
    ism = float((eff_params.get("interslice_smooth") or 0) or 0)
    steps.append({"label": "10. Inter-slice smoothing",
                  "kind": "decision", "group": "volume",
                  "branch": (f"displacement field smoothed across slices (σ={ism:.1f})" if ism > 0
                             else "off (interslice_smooth = 0)")})
    axial_on = bool(eff_params.get("axial_refine", True))
    ax = oct_iter.get("axial") or oct_iter.get("axial_refine") or {}
    if isinstance(ax, dict) and ax:
        ax_note = ", ".join(f"{k}={v}" for k, v in ax.items())
    else:
        ax_note = "ping-pong axial pass; per-frame kept only where it lowers lateral roughness (global never-worse guard)"
    steps.append({"label": "11. Axial ping-pong refine — " + ("applied" if axial_on else "off"),
                  "kind": "decision", "group": "volume", "branch": ax_note})
    steps.append({"label": "12. Manual depth nudges (Fix-columns)",
                  "kind": "decision", "group": "volume",
                  "branch": (f"{len(eff_params.get('manual_shifts') or {})} frame(s) nudged — applied LAST as ground truth"
                             if eff_params.get("manual_shifts") else "none — applied LAST, after all fitting/guards")})
    return {"steps": steps, "slices": n_slices, "index": cur_index}


_BORDER_VOL_CACHE: dict = {}  # path -> (mtime, ndarray) — last border-input volume, so SCRUBBING a pass's
                              # slices doesn't re-decompress the .nii.gz every request (smooth scrolling).
def _load_border_vol(path: Path):
    import os
    import numpy as np
    import nibabel as nib
    key = str(path)
    mt = os.path.getmtime(path)
    cached = _BORDER_VOL_CACHE.get(key)
    if cached and cached[0] == mt:
        return cached[1]
    arr = np.ascontiguousarray(np.asanyarray(nib.load(key).dataobj))
    _BORDER_VOL_CACHE.clear()                       # keep only the most-recent input (bound memory)
    _BORDER_VOL_CACHE[key] = (mt, arr)
    return arr


def _ensure_raw_border_nifti(case_id: str) -> Path:
    """A persistent raw (un-corrected) NIfTI for the Fix-columns border (pass-1 input). Created once from
    the .OCT; kept (own path, so a re-preprocess's tmp raw snapshot never clobbers it) so border-curve
    requests load a single slice fast instead of re-reading the .OCT on every scrub."""
    raw = orch.case_root(case_id) / "input" / "_raw_border.nii.gz"
    if raw.exists():
        return raw
    m = orch.read_manifest(case_id)
    src = m.get("oct_source")
    if not src or not Path(src).exists():
        raise HTTPException(400, f"Case {case_id} has no .OCT source.")
    raw.parent.mkdir(parents=True, exist_ok=True)
    oct_mod.raw_oct_to_nifti(src, raw, volume_index=int(m.get("oct_volume_index", 0)), companion_txt=m.get("companion_txt"))
    return raw


@app.post("/api/case/{case_id}/oct-border-curves-all")
def oct_border_curves_all(case_id: str, req: OctPreprocessRequest) -> dict:
    """ALL per-slice detected borders for a pass in ONE call, computed with a FAST detector (gradient
    argmax + outlier/median cleanup — no bilateral / hist-eq / RANSAC), so the frontend can cache the whole
    set and scrubbing the fix-columns border is INSTANT (no per-slice round-trip; the per-slice detector is
    ~258ms, the whole-volume fast pass is ~0.5s). The slower, more robust per-slice detector
    (oct-border-curve) then refines just the slice the user settles on. x=frame, y=depth (depth 0 = TOP)."""
    import numpy as np
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    pass_n = max(1, int(req.border_pass or 1))
    if pass_n <= 1:
        inp = _ensure_raw_border_nifti(case_id)
    else:
        pv = orch.case_root(case_id) / "passes" / f"pass_{pass_n - 1}.nii.gz"
        inp = pv if pv.exists() else _ensure_raw_border_nifti(case_id)
    try:
        arr = _load_border_vol(inp)                                # (lateral, depth, frames), cached
        n = int(arr.shape[0]); depth_vox = int(arr.shape[1]); n_frames = int(arr.shape[2])
        p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        sigma = float(p["sigma"]); max_jump = float(p["max_jump"]); mfs = int(p["median_filter_size"])
        xs = np.arange(n_frames, dtype=np.float64)
        # If the user has CONFIRMED a re-detection (pass 1), serve the cached tilt-aware surface for EVERY
        # slice (so scrubbing shows the confirmed border, matching oct-border-curve's per-slice cache read);
        # otherwise the fast auto detector.
        anc = (m.get("oct_params") or {}).get("border_anchors") or {}
        surf = _redetect_surface_fresh(case_id, anc) if (pass_n <= 1 and anc) else None
        use_surf = surf is not None and surf.shape[0] == n and surf.shape[1] == n_frames
        edges: list = []; fits: list = []
        for i in range(n):
            if use_surf:
                e = np.asarray(surf[i], dtype=np.float64)
            else:
                sl = np.ascontiguousarray(arr[i]).astype(np.float32)
                raw = oct_mod._detect_surface_gradient(sl, sigma)  # fast, no prior, no bilateral
                e = oct_mod._smooth_median(oct_mod._correct_surface(raw, max_jump), mfs).astype(np.float64)
            try:
                f = np.polyval(np.polyfit(xs, e, 2), xs)           # quick quadratic fit (cosmetic blue line)
            except Exception:  # noqa: BLE001
                f = e
            edges.append([round(float(v), 1) for v in e])
            fits.append([round(float(v), 1) for v in f])
        return {"slices": n, "n_frames": n_frames, "depth_vox": depth_vox, "pass": pass_n,
                "edges": edges, "fits": fits}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT border curves(all) failed: {exc}")


@app.post("/api/case/{case_id}/oct-motion")
def oct_motion_analyze(case_id: str, req: OctPreprocessRequest) -> dict:
    """EYE-MOTION analysis from the detected corneal surface. The 3-D scan's SLOW (frame) axis is a TIME axis
    (~136 Hz on the Avanti), so the per-frame surface depth — once the smooth corneal shape is removed — is the
    patient's eye/head motion during the ~0.74 s scan. Returns the motion(t) trace (µm), its power spectrum +
    labelled dominant-frequency peaks, candidate saccade/microsaccade spikes, a dominant motion direction
    (axial vs in-plane), and an SNR gate. Reuses the cached raw-border volume → fast on a scrubbed case.
    Frequencies derive from the A-scan rate (Avanti ~70 kHz, editable) since the .OCT carries no timing."""
    import numpy as np
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    try:
        arr = _load_border_vol(_ensure_raw_border_nifti(case_id))    # (lateral, depth, frames), cached
        eff = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        persisted = (m.get("oct_params") or {}).get("ascan_rate_hz")
        rate = float(req.ascan_rate_hz) if req.ascan_rate_hz else float(persisted or oct_motion_mod.DEFAULT_ASCAN_RATE_HZ)
        sp = oct_mod._resolve_spacing(eff, m.get("companion_txt"), n_frames=int(arr.shape[2]))  # (lateral, depth, slice)
        res = oct_motion_mod.analyze_motion(
            np.ascontiguousarray(arr), ascan_rate_hz=rate, ascans_per_frame=int(arr.shape[0]),
            depth_spacing_mm=float(sp[1]), lateral_spacing_mm=float(sp[0]),
            detrend_order=int(req.detrend_order or 2), sinc_correct=bool(req.sinc_correct), params=eff)
        if req.ascan_rate_hz:                                        # remember a user-chosen rate on the case
            op = dict(m.get("oct_params") or {}); op["ascan_rate_hz"] = rate
            orch.write_manifest_value(case_id, {"oct_params": op})
        return res
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT motion analysis failed: {exc}")


@app.post("/api/case/{case_id}/oct-border-curve")
def oct_border_curve(case_id: str, req: OctPreprocessRequest) -> dict:
    """Per-frame DETECTED corneal surface + RANSAC best-fit for ONE sagittal slice of the selected pass's
    INPUT volume — pass 1's input is the RAW original, pass k's input is pass (k-1)'s output — because
    correcting the detection on a pass's input is what improves that pass's result (editing the border on
    the downstream/corrected result is meaningless). Returns coordinate arrays so the UI draws + drags the
    border. Loads only the requested slice (fast scrubbing). Orientation matches the sagittal preview
    (arr[idx] = (depth, frames); depth 0 = TOP), so x=frame/n_frames, y=depth/depth_vox align."""
    import numpy as np
    import nibabel as nib
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    pass_n = max(1, int(req.border_pass or 1))
    # INPUT of pass `pass_n`: raw for pass 1, else the prior pass's saved output (fallback to raw).
    if pass_n <= 1:
        inp = _ensure_raw_border_nifti(case_id)
    else:
        pv = orch.case_root(case_id) / "passes" / f"pass_{pass_n - 1}.nii.gz"
        inp = pv if pv.exists() else _ensure_raw_border_nifti(case_id)
    try:
        arr = _load_border_vol(inp)                                # (lateral, depth, frames), cached
        n = int(arr.shape[0])                                      # lateral = sagittal slice count
        idx = n // 2 if req.slice_index is None else max(0, min(n - 1, int(req.slice_index)))
        sl = np.ascontiguousarray(arr[idx]).astype(np.float32)     # (depth, frames)
        p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {}), **(req.params or {})}
        # If the user has CONFIRMED a fix-columns re-detection (pass 1 / raw), show that cached tilt-aware
        # surface as the border instead of the live auto detection — so scrubbing reveals the new detected
        # border the warp will use (preview == result). Falls back to live auto if no/stale cache.
        anc = (m.get("oct_params") or {}).get("border_anchors") or {}
        surf = _redetect_surface_fresh(case_id, anc) if (pass_n <= 1 and anc) else None
        if surf is not None and 0 <= idx < surf.shape[0] and surf.shape[1] == sl.shape[1]:
            edge = np.asarray(surf[idx], dtype=np.float32)
        else:
            edge = oct_mod._merged_side_edge(sl, p)
        fit = oct_mod._fit_quadratic_ransac(edge, float(p["residual_threshold"]))
        return {"slices": n, "index": int(idx), "n_frames": int(sl.shape[1]), "depth_vox": int(sl.shape[0]),
                "pass": pass_n, "edge": [float(v) for v in edge], "fit": [float(v) for v in fit]}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT border curve failed: {exc}")


def _border_anchors_sig(anchors: dict) -> str:
    """Canonical signature of the anchor set (for cache freshness)."""
    if not isinstance(anchors, dict):
        return ""
    parts = []
    for s in sorted(anchors.keys(), key=lambda x: int(x)):
        fr = anchors[s]
        if not isinstance(fr, dict):
            continue
        inner = ",".join(f"{int(f)}={int(round(float(fr[f])))}"
                         for f in sorted(fr.keys(), key=lambda x: int(x)))
        if inner:
            parts.append(f"{int(s)}:{inner}")
    return ";".join(parts)


# Detection-relevant params: a baseline/redetect surface cache must invalidate if ANY of these change.
# Today every param change already rmtrees border_cache, so this is defence-in-depth against a future writer.
_DETECT_PARAM_KEYS = ("sigma", "max_jump", "median_filter_size", "d", "sigmaColor", "sigmaSpace",
                      "side_window", "side_threshold_factor", "residual_threshold", "active_threshold",
                      "detect_window", "detect_seed_window", "redetect_frame_margin", "redetect_slice_band",
                      "redetect_seed_window")


def _detect_params_sig(p: dict) -> str:
    """Canonical signature of the detection-relevant params (for surface-cache freshness)."""
    return ";".join(f"{k}={p.get(k)}" for k in _DETECT_PARAM_KEYS)


def _redetect_cache_path(case_id: str) -> Path:
    # NOT under passes/ or input/_iter (both rmtree'd on every preprocess) — its own dir, keyed to the RAW.
    return orch.case_root(case_id) / "border_cache" / "redetect.npz"


def _baseline_cache_path(case_id: str) -> Path:
    return orch.case_root(case_id) / "border_cache" / "baseline.npz"


def _baseline_surface(case_id: str, arr, p: dict):
    """The robust auto-detected surface for EVERY slice (the 'satisfactory rest' the local-band re-detection
    keeps untouched), cached per-case keyed to the raw-border volume so repeat Confirms are fast."""
    import os
    import numpy as np
    raw = _ensure_raw_border_nifti(case_id)
    n, W = int(arr.shape[0]), int(arr.shape[2])
    cp = _baseline_cache_path(case_id)
    psig = _detect_params_sig(p)
    if cp.exists():
        try:
            z = np.load(cp, allow_pickle=False)
            if (abs(float(z["raw_mtime"]) - float(os.path.getmtime(raw))) <= 1e-6
                    and str(z["params_sig"]) == psig):
                s = np.asarray(z["surface"], dtype=np.float32)
                if s.shape == (n, W):
                    return s
        except Exception:  # noqa: BLE001 — a corrupt/old cache just forces a recompute
            pass
    surface = oct_mod.detect_surface_all(arr, p)
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_name("baseline.tmp.npz")   # MUST end .npz (np.savez_compressed appends it otherwise)
    np.savez_compressed(tmp, surface=surface.astype(np.float32), raw_mtime=float(os.path.getmtime(raw)),
                        params_sig=psig)
    os.replace(tmp, cp)
    return surface


def _redetect_surface_fresh(case_id: str, anchors: dict):
    """The cached tilt-aware re-detected surface (lateral, frames) iff it is FRESH for `anchors` + the
    current raw-border volume; else None."""
    import os
    import numpy as np
    cp = _redetect_cache_path(case_id)
    if not cp.exists():
        return None
    try:
        raw = _ensure_raw_border_nifti(case_id)
        z = np.load(cp, allow_pickle=False)
        if str(z["anchors_sig"]) != _border_anchors_sig(anchors):
            return None
        if abs(float(z["raw_mtime"]) - float(os.path.getmtime(raw))) > 1e-6:
            return None
        p = {**oct_mod.DEFAULT_PARAMS, **(orch.read_manifest(case_id).get("oct_params") or {})}
        if str(z["params_sig"]) != _detect_params_sig(p):    # detection params changed → stale
            return None
        return np.asarray(z["surface"], dtype=np.float32)
    except Exception:  # noqa: BLE001 — a corrupt/old cache just forces a recompute
        return None


def _compute_redetect_cache(case_id: str, m: dict, anchors: dict):
    """MARCH the tilt-aware re-detection on the RAW volume seeded by `anchors`, cache it (+ anchors sig +
    raw mtime), and return the surface (lateral, frames). Shared by Confirm and Run so both use the SAME
    surface (preview == result). The RAW sagittal arr == reformat_to_sagittal(.OCT read) (SITK↔nibabel
    axis reversal), so this surface aligns with the warp's sagittal volume."""
    import os
    import numpy as np
    raw = _ensure_raw_border_nifti(case_id)
    arr = _load_border_vol(raw)                              # (lateral, depth, frames)
    p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
    baseline = _baseline_surface(case_id, arr, p)           # cached auto surface (the satisfactory rest)
    surface = oct_mod.redetect_surface(arr, anchors, p, baseline=baseline)   # local-band correction (lateral, frames)
    cp = _redetect_cache_path(case_id)
    cp.parent.mkdir(parents=True, exist_ok=True)
    # tmp MUST end in .npz — np.savez_compressed appends '.npz' to any path that doesn't, which would make
    # os.replace move a nonexistent file. Write tmp then atomically replace so a crash can't leave a partial.
    tmp = cp.with_name("redetect.tmp.npz")
    np.savez_compressed(tmp, surface=surface.astype(np.float32),
                        anchors_sig=_border_anchors_sig(anchors),
                        raw_mtime=float(os.path.getmtime(raw)),
                        params_sig=_detect_params_sig(p))
    os.replace(tmp, cp)
    return surface


@app.post("/api/case/{case_id}/oct-border-redetect")
def oct_border_redetect(case_id: str, req: OctPreprocessRequest) -> dict:
    """Fix-columns "Confirm": LOCAL-BAND re-detection seeded by the user's anchors (true surface points).
    The auto-detected surface is kept everywhere EXCEPT a local band around the corrected ("pink line")
    region — the corrected frames plus the neighbouring slices around them, marched out until the
    re-detection re-converges to the auto edge — so the rest of the satisfactory border is left untouched
    (replaces the previous whole-volume march, which often replaced a good surface with a worse one). The
    spliced surface is cached per-case and the anchors persisted in oct_params, so the scrub preview
    (oct-border-curve) shows the corrected border and a later Run flattens the volume to exactly that surface
    (preview == result). Empty anchors clear it (revert to auto)."""
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    anchors = req.border_anchors if isinstance(req.border_anchors, dict) else {}
    try:
        op = dict(m.get("oct_params") or {})
        op.pop("detect_lo", None); op.pop("detect_hi", None)   # legacy global band — removed
        op["border_anchors"] = anchors
        # Parabola mode: the anchors are a DENSE fitted quadratic → use it EXACTLY (seed window 0). Edge mode:
        # the default tight window. Persisted so the cache write/read/run all derive the SAME seed window (and
        # so params_sig matches — _redetect_surface_fresh reads this back).
        op["redetect_seed_window"] = 0.0 if req.parabola else float(oct_mod.DEFAULT_PARAMS.get("redetect_seed_window", 2.0))
        orch.write_manifest_value(case_id, {"oct_params": op})
        if anchors:
            _compute_redetect_cache(case_id, {**m, "oct_params": op}, anchors)
            n_anchors = sum(len(v) for v in anchors.values() if isinstance(v, dict))
        else:
            _redetect_cache_path(case_id).unlink(missing_ok=True)   # cleared → auto on scrub + run
            n_anchors = 0
        return {"ok": True, "n_anchors": int(n_anchors)}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT border re-detect failed: {exc}")


class OctLoadDirRequest(BaseModel):
    directory: str


@app.post("/api/oct/load-dir")
def oct_load_dir(req: OctLoadDirRequest) -> dict:
    """Load every .OCT in a SERVER-SIDE directory as cases (referenced in place, with their
    .txt companions auto-paired). For local data this beats the browser folder picker:
    no re-upload, and the companion .txt is always found next to the .OCT."""
    d = Path(req.directory).expanduser()
    if not d.is_dir():
        raise HTTPException(400, f"Not a directory: {req.directory}")
    items = cohort_mod.discover(req.directory)
    cornea = [it for it in items if it["is_3d_cornea"]]
    items = cornea if cornea else items
    if not items:
        raise HTTPException(400, "No .OCT scans found under that directory.")
    used: set = set()
    cases = []
    for it in items:
        cid = _cohort_make_case(it, used)   # references in place + pairs companion
        # Report whether this scan was ALREADY corrected in a prior session (loaded in place,
        # so its manifest survives) — the loader colours those scans as done.
        cases.append({"case_id": cid, "filename": it["filename"], "patient": it["patient"],
                      "eye": it["eye"], "has_companion": bool(it["companion"]),
                      "preprocessed": bool(orch.read_manifest(cid).get("oct_preprocessed"))})
    return {"cases": cases}


@app.post("/api/oct/pick-dir")
def oct_pick_dir() -> dict:
    """Open a NATIVE folder picker on the sidecar host and return the chosen absolute
    path, so a local user can select a folder with one click instead of typing it. This
    only makes sense for the normal local-app case (browser + sidecar share a desktop);
    on a headless/remote host there's no display, so we fail with a clear message and the
    user falls back to typing the path or to "Pick files". The folder is still loaded in
    place via /api/oct/load-dir afterwards — nothing is uploaded."""
    import os
    import shutil

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        raise HTTPException(400, "No desktop on the sidecar host — type the folder path instead.")
    zenity = shutil.which("zenity")
    if not zenity:
        raise HTTPException(400, "Native folder picker (zenity) isn't installed — type the folder path instead.")
    try:
        proc = subprocess.run(
            [zenity, "--file-selection", "--directory", "--title=Select the OCT scans folder"],
            capture_output=True, text=True, timeout=600,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(408, "Folder picker timed out — try again.")
    except OSError as e:
        raise HTTPException(500, f"Couldn't open the folder picker: {e}")
    # returncode 0 with a path = a folder was chosen. Otherwise zenity's exit code 1 is
    # ambiguous — it covers BOTH a genuine user cancel/close AND a launch failure (DISPLAY
    # is set but the X authority/cookie denies access, no desktop portal, GTK init failed).
    # zenity also prints harmless GTK/accessibility warnings to stderr even on a clean
    # cancel, so we can't treat *any* stderr as an error; match only the fatal display
    # signatures and surface those, and log-but-ignore the rest (a true cancel is a no-op).
    if proc.returncode == 0 and proc.stdout.strip():
        return {"directory": proc.stdout.strip()}
    err = (proc.stderr or "").strip()
    fatal = ("cannot open display", "unable to init server", "could not open display",
             "failed to parse", "authorization required")
    if err and any(sig in err.lower() for sig in fatal):
        raise HTTPException(500, f"Folder picker couldn't open a window: {err[:300]}")
    if err:
        print(f"[pick-dir] zenity exited {proc.returncode} (treated as cancel); stderr: {err[:300]}", file=sys.stderr)
    return {"directory": None}


def _dir_size(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


@app.get("/api/cases/stat")
def cases_stat() -> dict:
    """How many cases are persisted on disk (drives the Wipe button's count). Cheap — counts
    directories only, no size walk."""
    root = settings.CASES_ROOT
    n = sum(1 for c in root.iterdir() if c.is_dir()) if root.exists() else 0
    return {"count": n, "cases_root": str(root)}


@app.get("/api/cases/list")
def cases_list() -> dict:
    """Enumerate persisted OCT cases so the loader can re-hydrate them on startup WITHOUT a folder
    reload. Returns the LoadedCase shape (case_id, filename, patient, eye, n_volumes, preprocessed),
    matching /api/oct/load-dir. Skips synthetic consensus cases (those open via the consensus viewer)
    and non-OCT cases (e.g. directly-registered volumes)."""
    root = settings.CASES_ROOT
    out: list[dict] = []
    if not root.exists():
        return {"cases": []}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        cid = child.name
        try:
            m = orch.read_manifest(cid)
        except Exception:  # noqa: BLE001 — skip an unreadable case, keep the rest
            continue
        if not m or m.get("consensus_cases"):
            continue
        src = m.get("oct_source") or m.get("companion_txt")
        if not src:
            continue  # not an OCT-loader case
        meta = {}
        try:
            meta = metrics_export.parse_case_meta(src)
        except Exception:  # noqa: BLE001
            pass
        out.append({
            "case_id": cid,
            "filename": os.path.basename(str(src)),
            "patient": ((m.get("patient_id") or meta.get("patient_id") or "").strip() or None),
            "eye": ((m.get("eye") or meta.get("eye") or "").strip() or None),
            "n_volumes": m.get("n_frames") or m.get("n_volumes"),
            "preprocessed": bool(m.get("oct_preprocessed")),
            "passes": int((m.get("oct_iter") or {}).get("passes", 1)) if m.get("oct_iter") else 1,
            # Per-scan lifecycle flags so the loader can colour each entry by its timeline step
            # (mirrors api/lifecycle.ts scanStep). All booleans except scar_classification.
            "life": {
                "input_volume": bool(m.get("input_volume") or m.get("corrected_volume")),
                "oct_preprocessed": bool(m.get("oct_preprocessed")),
                "preproc_vetted": bool(m.get("preproc_vetted")),
                "scar_classification": m.get("scar_classification") or None,
                "sam2_meta": bool(m.get("sam2_meta")),
                "corrected_labelmap": bool(m.get("corrected_labelmap")),
                "training_scheduled": bool(m.get("training_scheduled")),
            },
        })
    return {"cases": out}


@app.post("/api/cases/wipe")
def cases_wipe() -> dict:
    """DESTRUCTIVE: delete every persisted case under CASES_ROOT (corrected volumes,
    segmentations, labels, previews, manifests). Used to get a clean slate so a re-upload of
    the same scans starts fresh instead of reusing the deterministic case folder + its old
    output. Removes the case folders but keeps CASES_ROOT itself for new uploads."""
    import shutil
    root = settings.CASES_ROOT
    # Guard: only ever operate on a real, expected "cases" directory — never a parent/other path.
    if not root.exists() or not root.is_dir() or root.name != "cases":
        raise HTTPException(400, f"Refusing to wipe — unexpected cases root: {root}")
    with _COHORT_LOCK:
        if _COHORT.get("running"):
            raise HTTPException(409, "A cohort batch is running — stop it before wiping cases.")
    removed, freed = 0, 0
    for child in list(root.iterdir()):
        try:
            if child.is_dir():
                freed += _dir_size(child)
                shutil.rmtree(child, ignore_errors=True)
            else:
                freed += child.stat().st_size
                child.unlink()
            removed += 1
        except OSError as e:
            print(f"[wipe] could not remove {child}: {e}", file=sys.stderr)
    return {"removed": removed, "freed_bytes": freed}


# ── Cohort batch: mass-produce the labeled training set ─────────────────────
# Point at a directory of .OCT scans → group repeat scans by (patient, eye) → per
# scan preprocess + SAM2 + scar → per group build the consensus label. Runs in a
# background thread; resumable (skips already-corrected/segmented scans).
_COHORT: dict = {"running": False, "done": False, "error": None, "groups": []}
_COHORT_LOCK = threading.Lock()
# Serialises all SAM2/CUDA inference (cohort worker thread + user-triggered endpoints
# run on separate threads and share one predictor + CUDA context).
_GPU_LOCK = threading.Lock()


def _cohort_case_conflict(cid: str, full_path: str) -> bool:
    """True if a case with this id already references a DIFFERENT scan. Compared by
    BASENAME (matching _oct_case_taken), so the SAME .OCT re-loaded from a different
    location reuses its case instead of forking a "_2" duplicate. The cid already encodes
    patient/eye/series, so a basename match within one cid is necessarily the same scan."""
    src = orch.read_manifest(cid).get("oct_source")
    return bool(src) and Path(src).name != Path(full_path).name


def _cohort_make_case(scan: dict, used: set) -> str:
    """Create/reuse a case for a disk .OCT scan (references it in place; no copy)."""
    fm = oct_mod.parse_oct_filename(scan["filename"])
    if fm.get("patient_id"):
        base = orch.safe_case_id(f"case_{fm['patient_name'].lower()}_{fm['laterality'].lower()}_v{fm.get('series_number', 1)}")
    else:
        base = orch.safe_case_id(f"oct_{Path(scan['path']).stem}")
    cid, k = base, 2
    while cid in used or _cohort_case_conflict(cid, scan["path"]):
        cid = f"{base}_{k}"
        k += 1
    used.add(cid)
    orch.ensure_case_dirs(cid)
    if not orch.read_manifest(cid).get("oct_source"):
        orch.write_manifest_value(cid, {"oct_source": scan["path"], "companion_txt": scan.get("companion"),
                                        "oct_volume_index": 0, "oct_preprocessed": False})
    return cid


def _cohort_worker(params: dict, do_preprocess: bool) -> None:
    try:
        import nibabel as nib
        used: set = set()
        for g in _COHORT["groups"]:
            g["status"] = "running"
            cids = []
            for sc in g["scans"]:
                try:
                    cid = _cohort_make_case(sc["_scan"], used)
                    sc["case_id"] = cid
                    work = _oct_working_path(cid, sc["_scan"]["path"])
                    m = orch.read_manifest(cid)
                    companion = sc["_scan"].get("companion")
                    if do_preprocess and not m.get("oct_preprocessed"):
                        sc["status"] = "preprocessing"
                        _run_oct_worker("preprocess", sc["_scan"]["path"], work, params, 0, companion=companion)
                        orch.write_manifest_value(cid, {"input_volume": str(work), "corrected_volume": str(work),
                                                        "oct_preprocessed": True, "oct_params": params})
                    elif not work.exists():
                        _run_oct_worker("raw" if not do_preprocess else "preprocess", sc["_scan"]["path"], work, params, 0, companion=companion)
                        orch.write_manifest_value(cid, {"input_volume": str(work), "corrected_volume": str(work),
                                                        "oct_preprocessed": do_preprocess})
                    sc["status"] = "segmenting"
                    _ensure_segmented(cid)
                    arr, _ = labels.best_labelmap_nnunet(cid)
                    mm = scar_mod.quantify(arr, nib.load(str(_ensure_volume_nifti(cid))).header.get_zooms())
                    sc["scar_mm3"] = mm["scar_volume_mm3"]
                    sc["status"] = "done"
                    cids.append(cid)
                except Exception as exc:  # noqa: BLE001
                    sc["status"] = "error"
                    sc["error"] = str(exc)[:300]
            ok = [c for c in cids if labels.corrected_path(c).exists()]
            if len(ok) > 1:
                g["status"] = "consensus"
                try:
                    ccid, report = _build_consensus_case(
                        ok, orch.safe_case_id(f"case_{(g['patient'] or 'x').lower()}_{(g['eye'] or 'x').lower()}_consensus"),
                        ensure=False)
                    g["consensus_case"] = ccid
                    g["scar_volume_mm3"] = report["scar_volume_mm3"]["mean"]
                    g["cv_percent"] = report["scar_volume_mm3"]["cv_percent"]
                except Exception as exc:  # noqa: BLE001
                    g["error"] = str(exc)[:300]
            elif len(ok) == 1:
                g["single_case"] = ok[0]
            # Don't paint a failed group green: surface consensus/segmentation failures.
            if g.get("error"):
                g["status"] = "error"
            elif not ok:
                g["status"] = "error"
                g["error"] = "all scans in this group failed to preprocess/segment"
            else:
                g["status"] = "done"
        _COHORT["done"] = True
    except Exception as exc:  # noqa: BLE001
        _COHORT["error"] = str(exc)[:500]
    finally:
        _COHORT["running"] = False


class CohortScanRequest(BaseModel):
    directory: str


class CohortRunRequest(BaseModel):
    directory: str
    params: dict | None = None
    preprocess: bool = True


@app.post("/api/cohort/scan")
def cohort_scan(req: CohortScanRequest) -> dict:
    """Discover + group the .OCT scans under a directory (the run plan). Fast, no decode."""
    if not Path(req.directory).expanduser().is_dir():
        raise HTTPException(400, f"Not a directory: {req.directory}")
    groups = cohort_mod.group_by_eye(cohort_mod.discover(req.directory))
    return {"n_groups": len(groups), "n_scans": sum(len(g["scans"]) for g in groups),
            "groups": [{"patient": g["patient"], "eye": g["eye"],
                        "scans": [s["filename"] for s in g["scans"]]} for g in groups]}


@app.post("/api/cohort/run")
def cohort_run(req: CohortRunRequest) -> dict:
    """Start the batch: preprocess → SAM2 → scar per scan, consensus per (patient, eye).
    Runs in the background; poll /api/cohort/status."""
    with _COHORT_LOCK:
        if _COHORT["running"]:
            raise HTTPException(409, "A cohort run is already in progress.")
        groups = cohort_mod.group_by_eye(cohort_mod.discover(req.directory))
        if not groups:
            raise HTTPException(400, "No 3D Cornea .OCT scans found under that directory.")
        _COHORT.update({
            "running": True, "done": False, "error": None,
            "groups": [{"patient": g["patient"], "eye": g["eye"], "status": "queued",
                        "scans": [{"filename": s["filename"], "status": "queued", "_scan": s} for s in g["scans"]]}
                       for g in groups],
        })
        threading.Thread(target=_cohort_worker, args=(req.params or {}, req.preprocess), daemon=True).start()
    return {"started": True, "n_groups": len(groups),
            "n_scans": sum(len(g["scans"]) for g in groups)}


@app.get("/api/cohort/status")
def cohort_status() -> dict:
    """Live progress of the running/last cohort batch. Snapshots keys (not live .items())
    so it can't crash with 'dict changed size' while the worker thread inserts keys."""
    def view(d: dict, skip: str) -> dict:
        return {k: d[k] for k in list(d.keys()) if k != skip}
    groups = []
    for g in list(_COHORT["groups"]):
        gv = view(g, "scans")
        gv["scans"] = [view(s, "_scan") for s in list(g["scans"])]
        groups.append(gv)
    return {"running": _COHORT["running"], "done": _COHORT["done"], "error": _COHORT["error"], "groups": groups}


# ── nnU-Net export (the corrected labels become the training set) ──────────
class ExportRequest(BaseModel):
    dataset_name: str = "Dataset501_CorneaOCT"
    cases: List[str] | None = None  # default: all cases with a corrected labelmap


@app.post("/api/export/nnunet")
def export_nnunet(req: ExportRequest) -> dict:
    cases = req.cases if req.cases else export_mod.cases_with_segmentation()
    if not cases:
        raise HTTPException(400, "No cases with a segmentation to export. Run SAM2 first.")
    dataset_dir = export_mod.DATASET_ROOT / req.dataset_name
    export_mod.clean_dataset(dataset_dir)           # drop orphans from a prior export
    # Leakage guard: warn if both a consensus case and its own member repeats are in the
    # set — training on correlated repeats of one eye (and across train/val) inflates
    # apparent accuracy. Caller can pass an explicit `cases` subset to avoid it.
    case_set = set(cases)
    leakage = []
    for cid in cases:
        members = set(orch.read_manifest(cid).get("consensus_cases") or [])
        overlap = members & case_set
        if overlap:
            leakage.append({"consensus_case": cid, "member_repeats_also_exported": sorted(overlap)})
    results = []
    for cid in cases:
        base = orch.case_root(cid) / "previews" / "volume.nii.gz"
        if not base.exists():
            try:
                base = _ensure_volume_nifti(cid)
            except HTTPException:
                results.append({"case_id": cid, "exported": False, "reason": "no volume"})
                continue
        try:
            results.append(export_mod.export_case(cid, dataset_dir, base))
        except Exception as exc:  # noqa: BLE001
            results.append({"case_id": cid, "exported": False, "reason": str(exc)})
    num = sum(1 for r in results if r.get("exported"))
    export_mod.write_dataset_json(dataset_dir, num)
    return {"dataset_dir": str(dataset_dir), "num_training": num, "results": results,
            "leakage_warning": leakage}


# ── nnU-Net training proof-of-concept (per-scan labels, isolated venv) ──────
class TrainRequest(BaseModel):
    mode: str = "single3"      # "single3" (bg/cornea/scar) | "cascade" (cornea, then scar-in-cornea)
    config: str = "2d"         # "2d" | "3d_fullres"
    length: str = "short"      # "short" (~10 epochs) | "full" (1000 epochs)
    cases: list[str] | None = None   # optional subset of candidate cases to train on (None = all)


@app.get("/api/train/nnunet/status")
def train_status() -> dict:
    """Live training status + the per-scan cases that WOULD be used (consensus excluded)."""
    st = nntrain.status()
    st["candidate_cases"] = nntrain.per_scan_segmented_cases()
    return st


@app.post("/api/train/nnunet/setup")
def train_setup() -> dict:
    """Create the isolated nnU-Net venv if absent (reuses system torch). Runs in the background;
    poll /status for venv_ready."""
    if nntrain.venv_ready():
        return {"venv_ready": True, "already": True}

    def _bg():
        try:
            nntrain.ensure_venv()
        except Exception as exc:  # noqa: BLE001
            print(f"[nnunet] venv setup failed: {exc}", file=sys.stderr)

    threading.Thread(target=_bg, daemon=True).start()
    return {"venv_ready": False, "started": True}


@app.post("/api/train/nnunet/start")
def train_start(req: TrainRequest) -> dict:
    """Build the per-scan dataset(s) across all subgroups and run the standard nnU-Net workflow
    (plan_and_preprocess → train) in the isolated venv. Returns immediately; poll /status."""
    try:
        return nntrain.start_training(req.mode, req.config, req.length, _ensure_volume_nifti,
                                      subset=req.cases)
    except RuntimeError as exc:
        raise HTTPException(400, str(exc))


@app.get("/api/train/nnunet/runs")
def train_runs() -> dict:
    """List the saved First-Run Folders (previous training runs), newest first."""
    return {"runs": nntrain.list_runs()}


@app.delete("/api/train/nnunet/runs/{name}")
def train_run_delete(name: str) -> dict:
    """Delete one previous training run (its First-Run Folder) by name."""
    try:
        ok = nntrain.delete_run(name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"ok": ok, "runs": nntrain.list_runs()}


# ── Serve the built frontend (single-port mode) ────────────────────────────
# When cornea_app/dist exists (after `npm run build`), the sidecar also serves the
# React UI, so the whole app runs as ONE process on :8765 — handy where a separate
# Vite dev server can't be kept alive. Mounted LAST so all /api/* routes win first.
_DIST = Path(__file__).resolve().parents[1] / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles

    # Serve index.html with no-cache so a fresh `npm run build` shows up on a plain reload (the
    # hashed asset filenames bust their own cache; only the entry HTML must always be revalidated —
    # otherwise the browser keeps a stale index pointing at the old JS bundle). Registered BEFORE
    # the catch-all mount so these exact paths win.
    @app.get("/", include_in_schema=False)
    @app.get("/index.html", include_in_schema=False)
    def _spa_index() -> FileResponse:
        return FileResponse(str(_DIST / "index.html"), media_type="text/html",
                            headers={"Cache-Control": "no-cache, no-store, must-revalidate"})

    app.mount("/", StaticFiles(directory=str(_DIST), html=True), name="spa")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cornea OCT sidecar")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    # Signal readiness for dev-launch.sh (greps for "READY:{port}").
    print(f"READY:{args.port}", flush=True)
    uvicorn.run(
        "api_server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
