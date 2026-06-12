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
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import settings
import orchestration as orch
import volume_io
import labels
import slicer_runner
import masks
import scar as scar_mod
import export as export_mod
import preprocess
import postprocess
import metrics_export
import consensus as consensus_mod
import oct_preprocess as oct_mod
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
    return {"status": "ok"}


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


@app.get("/api/case/{case_id}/volume.nii.gz")
def get_volume_nifti(case_id: str) -> FileResponse:
    dst = _working_volume(case_id)
    return FileResponse(str(dst), media_type="application/gzip", filename="volume.nii.gz")


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


@app.get("/api/case/{case_id}/previews/{group}")
def list_previews(case_id: str, group: str) -> dict:
    images = orch.preview_images_from_dir(group, _preview_group_dir(case_id, group))
    return {"group": group, "images": images}


@app.post("/api/case/{case_id}/context-previews")
def context_previews(case_id: str) -> dict:
    """Render plain grayscale slice PNGs of the working volume (in-sidecar, numpy)
    so the 2D gallery can show the raw OCT before any segmentation."""
    orch.ensure_case_dirs(case_id)
    src = _working_volume(case_id)
    try:
        postprocess.render_context_previews(src, orch.context_preview_dir(case_id))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Context preview render failed: {exc}")
    return {"images": orch.preview_images_from_dir("Context", orch.context_preview_dir(case_id))}


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


# ── Stage 3: scar detection + quantification ───────────────────────────────
class ScarAutoRequest(BaseModel):
    percentile: float = 88.0     # sensitivity: flag the brightest (100−percentile)% of cornea
    min_voxels: int = 500        # continuity: drop connected components smaller than this
    erode_surface: int = 6       # drop the epithelium/Bowman's/endothelium reflective rind
    replace: bool = False        # False: merge candidates with existing scar (keep manual edits)


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
    scar_mask = scar_mod.detect_scar_in_cornea(vol, arr, percentile=req.percentile,
                                               min_voxels=req.min_voxels, erode_surface=req.erode_surface)
    new_label = scar_mod.apply_scar_to_labelmap(arr, scar_mask, replace=req.replace)
    labels.write_label_nifti(new_label, base, labels.corrected_path(case_id))
    postprocess.render_seg_previews(work, new_label, orch.segmentation_preview_dir(case_id), density_vol=vol)
    metrics = scar_mod.quantify(new_label, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
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
    m = scar_mod.quantify(arr, nib.load(str(base)).header.get_zooms())
    return {"case_id": case_id, "scar_present": m["scar_present"],
            "scar_volume_mm3": m["scar_volume_mm3"]}


class ConsensusBuildRequest(BaseModel):
    cases: List[str]
    reference: str | None = None
    group: str | None = None


def _read_label_ijk(path: Path) -> np.ndarray:
    import nibabel as nib
    return np.rint(np.asarray(nib.load(str(path)).dataobj)).astype(np.uint8)


def _consensus_case_id(cases: List[str], group: str | None = None) -> str:
    """Deterministic consensus case id. An explicit `group` (the cohort path) wins;
    otherwise derive a stable EYE-LEVEL id from the members' shared identity
    (`case_<patient>_<eye>_consensus`, lowercased to match the cohort exactly) so the
    endpoint and the cohort converge on ONE id instead of the old order-dependent
    `{cases[0]}_consensus`. Falls back to an order-independent id if unparseable."""
    if group:
        return orch.safe_case_id(group)
    m0 = orch.read_manifest(cases[0]) if cases else {}
    meta = metrics_export.parse_case_meta(m0.get("oct_source") or m0.get("input_volume"))
    if meta.get("patient_id") and meta.get("eye"):
        return orch.safe_case_id(f"case_{meta['patient_id'].lower()}_{meta['eye'].lower()}_consensus")
    return orch.safe_case_id("_".join(sorted(cases)) + "_consensus")


def _build_consensus_case(cases: List[str], group: str | None = None,
                          reference: str | None = None, ensure: bool = True) -> tuple[str, dict]:
    """Segment each scan (if needed), register + vote a partial-overlap consensus, render
    the per-tab previews, and persist. Shared by the /consensus/build endpoint and the
    cohort batch. Returns (consensus_case_id, report)."""
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

    ccid = _consensus_case_id(cases, group)
    orch.ensure_case_dirs(ccid)
    report = consensus_mod.build_consensus(cases, ccid, reference)
    report["segmentation_errors"] = seg_errors

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

    orch.write_manifest_value(ccid, {
        "input_volume": str(cons_vol), "corrected_volume": str(cons_vol),
        "consensus_report": report, "consensus_cases": report["scans"], "reference": report["reference"],
    })
    return ccid, report


@app.post("/api/consensus/build")
def consensus_build(req: ConsensusBuildRequest) -> dict:
    """Segment each scan (if needed), scar-anchor-register the repeats, build a
    probabilistic partial-overlap consensus, and render per-tab previews."""
    if len(req.cases) < 2:
        raise HTTPException(400, "Upload at least 2 scans of the same eye for consensus.")
    try:
        ccid, report = _build_consensus_case(req.cases, req.group, req.reference)
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
                    companion: str | None = None) -> None:
    """Run the oct_preprocess CLI in an isolated subprocess (keeps its fork-based
    parallelism away from the sidecar's CUDA/torch state). New session so a timeout can
    reap the whole fork-pool process group. `companion` = the .txt filespec whose
    per-scan geometry (XY Scan Size1 etc.) is baked into the NIfTI spacing."""
    import os
    import signal
    cmd = [sys.executable, str(Path(oct_mod.__file__)), mode, str(src), str(out),
           "--params", json.dumps(params or {}), "--volume-index", str(vi)]
    if companion:
        cmd += ["--companion-txt", str(companion)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        _, err = proc.communicate(timeout=1200)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.communicate()
        raise HTTPException(504, f"OCT {mode} timed out (>1200s).")
    if proc.returncode != 0 or not Path(out).exists():
        raise HTTPException(500, f"OCT {mode} failed: {(err or '')[-800:]}")


def _oct_render_volume(case_id: str, work: Path, preprocessed: bool, extra: dict) -> dict:
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
        labels.corrected_path(case_id).unlink(missing_ok=True)
        orch.case_qa_json(case_id).unlink(missing_ok=True)
        orch.write_manifest_value(case_id, {"scar_metrics": None})
    base = _ensure_volume_nifti(case_id)
    postprocess.render_context_previews(base, orch.context_preview_dir(case_id))
    return {"case_info": orch.current_case_info(case_id), "spacing": spacing,
            "geometry_warnings": geom_warnings,
            "images": orch.preview_images_from_dir("Context", orch.context_preview_dir(case_id))}


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
    if not show_corrected:
        try:
            oct_mod.raw_oct_to_nifti(src, work, volume_index=vi, companion_txt=m.get("companion_txt"))
        except oct_mod.MissingCompanionError as exc:
            raise HTTPException(400, str(exc))           # actionable: user forgot the .txt
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Reading .OCT failed: {exc}")
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
    _run_oct_worker("preprocess", src, work, req.params or {}, vi, companion=m.get("companion_txt"))
    extra = {"oct_volume_index": vi, "oct_params": req.params or {}}
    if req.classification:
        extra["scar_classification"] = req.classification
    if req.scar_range:
        extra["scar_range"] = req.scar_range
    out = _oct_render_volume(case_id, work, preprocessed=True, extra=extra)
    out["preprocessed"] = True
    out["n_frames"] = _nifti_frames(work)
    return out


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
        cases.append({"case_id": cid, "filename": it["filename"], "patient": it["patient"],
                      "eye": it["eye"], "has_companion": bool(it["companion"]), "preprocessed": False})
    return {"cases": cases}


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
