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
    base = _ensure_volume_nifti(case_id)            # SAM2 likes natural raw contrast
    work = orch.case_root(case_id) / "sam2_work"
    label, meta = sam2_segment.segment_volume(
        base, work, planes=tuple(req.planes), vote=req.vote)
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
    scar_mask = scar_mod.detect_scar_in_cornea(vol, arr, percentile=req.percentile,
                                               min_voxels=req.min_voxels, erode_surface=req.erode_surface)
    new_label = scar_mod.apply_scar_to_labelmap(arr, scar_mask)
    labels.write_label_nifti(new_label, base, labels.corrected_path(case_id))
    postprocess.render_seg_previews(work, new_label, orch.segmentation_preview_dir(case_id), density_vol=vol)
    metrics = scar_mod.quantify(new_label, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
    orch.write_manifest_value(case_id, {"scar_metrics": metrics,
                                        "segmentation_preview_dir": str(orch.segmentation_preview_dir(case_id))})
    return {"case_info": orch.current_case_info(case_id), "metrics": metrics,
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
    base = _ensure_volume_nifti(case_id)
    work_vol = _working_volume(case_id)
    vol = np.asarray(nib.load(str(work_vol)).dataobj).astype(np.float32)
    raw = np.asarray(nib.load(str(base)).dataobj).astype(np.float32)
    work = orch.case_root(case_id) / "sam2_work"
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


@app.post("/api/consensus/build")
def consensus_build(req: ConsensusBuildRequest) -> dict:
    """Segment each scan (if needed), scar-anchor-register the repeats, build a
    probabilistic partial-overlap consensus, and render per-tab previews (each scan's
    own image with its mask, and with the consensus mask, all in the reference frame)."""
    if len(req.cases) < 2:
        raise HTTPException(400, "Upload at least 2 scans of the same eye for consensus.")
    seg_errors: dict = {}
    for cid in req.cases:
        try:
            _ensure_segmented(cid)
        except HTTPException as exc:
            seg_errors[cid] = str(exc.detail)
        except Exception as exc:  # noqa: BLE001
            seg_errors[cid] = str(exc)

    ccid = orch.safe_case_id(req.group or f"{req.cases[0]}_consensus")
    orch.ensure_case_dirs(ccid)
    try:
        report = consensus_mod.build_consensus(req.cases, ccid, req.reference)
    except ValueError as exc:
        detail = str(exc) + (f" — failed scans: {seg_errors}" if seg_errors else "")
        raise HTTPException(400, detail)
    report["segmentation_errors"] = seg_errors

    import nibabel as nib
    cons_vol = orch.case_root(ccid) / "previews" / "volume.nii.gz"
    cons_lab = _read_label_ijk(labels.corrected_path(ccid))
    # Consensus tab: reference image + consensus scar
    postprocess.render_seg_previews(cons_vol, cons_lab, _preview_group_dir(ccid, "segmentation"))
    # Per-scan tabs: each scan's warped image with (a) its own scar, (b) the consensus scar.
    # For (b) the consensus mask is clipped to where THIS scan actually has data (its FOV),
    # so the comparison stays on the scan's image instead of painting over empty background.
    scans_dir = orch.case_root(ccid) / "scans"
    for cid in report["scans"]:
        svol = scans_dir / cid / "volume.nii.gz"
        slab = _read_label_ijk(scans_dir / cid / "label.nii.gz")
        data_mask = np.asarray(nib.load(str(svol)).dataobj) > 0  # warped FOV (fill is exact 0)
        cons_clipped = np.where(data_mask, cons_lab, 0).astype(np.uint8)
        postprocess.render_seg_previews(svol, slab, _preview_group_dir(ccid, f"scan_{cid}_self"))
        postprocess.render_seg_previews(svol, cons_clipped, _preview_group_dir(ccid, f"scan_{cid}_cons"))

    orch.write_manifest_value(ccid, {
        "input_volume": str(cons_vol), "corrected_volume": str(cons_vol),
        "consensus_report": report, "consensus_cases": report["scans"], "reference": report["reference"],
    })
    return {"consensus_case": ccid, "report": report,
            "images": orch.preview_images_from_dir("Segmentation", _preview_group_dir(ccid, "segmentation"))}


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
    return {"dataset_dir": str(dataset_dir), "num_training": num, "results": results}


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
