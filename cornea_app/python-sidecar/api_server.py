#!/usr/bin/env python3
"""
FastAPI server for the Cornea OCT Segmentation app.

Launched as a Tauri sidecar (or, in browser-dev, started directly by
dev-launch.sh). Communicates with the frontend over HTTP on 127.0.0.1:8765,
either directly (browser fetch) or proxied through the Rust shell.

The heavy lifting (3D Slicer subprocess calls, vision-model orchestration)
lives in the sibling modules; this file is the HTTP surface.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import List

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

import settings
import orchestration as orch
import volume_io
import slicer_runner
import vision
import masks
import scar as scar_mod
import export as export_mod

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
    vision_provider: str | None = None
    openai_model: str | None = None
    local_vision_base_url: str | None = None
    openai_api_key: str | None = None


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
    orch.write_manifest_value(
        payload.case_id, {"seed_json": str(orch.case_seed_json(payload.case_id))}
    )
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
        case_id,
        {
            "input_volume": str(volume),
            "corrected_volume": str(volume),
            "seed_json": str(orch.case_seed_json(case_id)),
        },
    )
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
        case_id,
        {
            "input_volume": str(dest),
            "corrected_volume": str(dest),
            "seed_json": str(orch.case_seed_json(case_id)),
        },
    )
    return orch.current_case_info(case_id)


def _ensure_volume_nifti(case_id: str) -> Path:
    src = _registered_volume(case_id)
    if not src.exists():
        raise HTTPException(404, f"Registered volume is missing: {src}")
    dst = orch.case_root(case_id) / "previews" / "volume.nii.gz"
    if (not dst.exists()) or dst.stat().st_mtime < src.stat().st_mtime:
        try:
            volume_io.ensure_nifti(src, dst)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Volume conversion failed: {exc}")
    return dst


@app.get("/api/case/{case_id}/volume.nii.gz")
def get_volume_nifti(case_id: str) -> FileResponse:
    dst = _ensure_volume_nifti(case_id)
    return FileResponse(str(dst), media_type="application/gzip", filename="volume.nii.gz")


# ── Previews + seeds + AI paint (Stage 1). Slicer calls are sync → FastAPI ──
# runs these handlers in its threadpool, keeping the event loop responsive.
_GROUP_DIRS = {
    "context": orch.context_preview_dir,
    "seeds": orch.seed_preview_dir,
    "segmentation": orch.segmentation_preview_dir,
}


@app.get("/api/case/{case_id}/previews/{group}")
def list_previews(case_id: str, group: str) -> dict:
    if group not in _GROUP_DIRS:
        raise HTTPException(400, f"Unknown preview group: {group}")
    images = orch.preview_images_from_dir(group.capitalize(), _GROUP_DIRS[group](case_id))
    return {"group": group, "images": images}


@app.get("/api/case/{case_id}/seeds")
def get_seeds(case_id: str) -> dict:
    return orch.read_json(orch.case_seed_json(case_id))


@app.put("/api/case/{case_id}/seeds")
def put_seeds(case_id: str, seed_spec: dict) -> dict:
    orch.ensure_case_dirs(case_id)
    orch.case_seed_json(case_id).write_text(json.dumps(seed_spec, indent=2))
    return {"ok": True, "seed_json": str(orch.case_seed_json(case_id))}


@app.post("/api/case/{case_id}/context-previews")
def context_previews(case_id: str) -> dict:
    orch.ensure_case_dirs(case_id)
    src = _registered_volume(case_id)
    proc = slicer_runner.render_context_previews(str(src), orch.context_preview_dir(case_id))
    if proc["status"] != 0:
        raise HTTPException(500, f"Context preview render failed:\n{proc['stderr']}")
    return {"process": proc, "images": orch.preview_images_from_dir(
        "Context", orch.context_preview_dir(case_id))}


@app.post("/api/case/{case_id}/seed-previews")
def seed_previews(case_id: str) -> dict:
    orch.ensure_case_dirs(case_id)
    src = _registered_volume(case_id)
    proc = slicer_runner.render_seed_previews(
        str(src), str(orch.case_seed_json(case_id)), orch.seed_preview_dir(case_id))
    if proc["status"] != 0:
        raise HTTPException(500, f"Seed preview render failed:\n{proc['stderr']}")
    return {"process": proc, "images": orch.preview_images_from_dir(
        "Seeds", orch.seed_preview_dir(case_id))}


@app.get("/api/case/{case_id}/seed-drawing.nii.gz")
def get_seed_drawing(case_id: str) -> FileResponse:
    """Label NIfTI (1=cornea,2=background,3=scar) on the base volume grid,
    loaded by niivue as the editable drawing layer."""
    base = _ensure_volume_nifti(case_id)
    seed_spec = orch.read_json(orch.case_seed_json(case_id))
    dst = orch.case_root(case_id) / "previews" / "seed-drawing.nii.gz"
    try:
        masks.build_seed_drawing(base, seed_spec if isinstance(seed_spec, dict) else {}, dst)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Seed drawing build failed: {exc}")
    return FileResponse(str(dst), media_type="application/gzip", filename="seed-drawing.nii.gz")


@app.post("/api/case/{case_id}/seeds/from-drawing")
async def seeds_from_drawing(case_id: str, files: List[UploadFile] = File(...)) -> dict:
    """Accept an edited niivue drawing NIfTI and convert it back to seeds.json."""
    orch.ensure_case_dirs(case_id)
    if not files:
        raise HTTPException(400, "No drawing uploaded.")
    data = await files[0].read()
    # niivue exports uncompressed NIfTI; uploads may be gzipped. Pick extension
    # by the gzip magic so nibabel reads it correctly.
    is_gz = len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B
    tmp = orch.case_root(case_id) / "previews" / ("edited-drawing.nii.gz" if is_gz else "edited-drawing.nii")
    tmp.write_bytes(data)
    try:
        seed_spec, counts = masks.seeds_from_drawing(tmp)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Could not parse drawing: {exc}")
    orch.case_seed_json(case_id).write_text(json.dumps(seed_spec, indent=2))

    # Re-render seed previews so the panel reflects the edits.
    src = str(_registered_volume(case_id))
    render = slicer_runner.render_seed_previews(
        src, str(orch.case_seed_json(case_id)), orch.seed_preview_dir(case_id))
    orch.write_manifest_value(case_id, {"seed_json": str(orch.case_seed_json(case_id))})
    return {
        "case_info": orch.current_case_info(case_id),
        "seed_spec": seed_spec,
        "counts": counts,
        "render_status": render["status"],
    }


class AiPaintRequest(BaseModel):
    provider: str | None = None
    model: str | None = None
    api_key: str | None = None
    local_base_url: str | None = None
    reviewer_prompt: str = ""


@app.post("/api/case/{case_id}/ai-paint/heuristic")
def ai_paint_heuristic(case_id: str) -> dict:
    orch.ensure_case_dirs(case_id)
    src = _registered_volume(case_id)
    qa_json = orch.case_root(case_id) / "segmentation" / "auto_seed_qa.json"
    proc = slicer_runner.heuristic_seeds(
        str(src), str(orch.case_seed_json(case_id)), str(qa_json),
        orch.seed_preview_dir(case_id), str(orch.feedback_json(case_id)))
    if proc["status"] != 0:
        raise HTTPException(500, f"Heuristic seed generation failed:\n{proc['stderr']}")
    seed_spec = orch.read_json(orch.case_seed_json(case_id))
    qa = orch.read_json(qa_json) if qa_json.exists() else None
    orch.write_manifest_value(case_id, {
        "seed_json": str(orch.case_seed_json(case_id)),
        "auto_seed_qa_json": str(qa_json),
        "seed_preview_dir": str(orch.seed_preview_dir(case_id)),
    })
    return {"case_info": orch.current_case_info(case_id), "seed_spec": seed_spec,
            "qa": qa, "mode": "heuristic"}


@app.post("/api/case/{case_id}/ai-paint")
def ai_paint(case_id: str, req: AiPaintRequest) -> dict:
    orch.ensure_case_dirs(case_id)
    src = str(_registered_volume(case_id))
    cfg = settings.get_settings()
    provider = (req.provider or cfg["vision_provider"] or "local").strip().lower()
    model = (req.model or cfg["openai_model"] or "local-vision-model").strip()
    base_url = req.local_base_url or cfg["local_vision_base_url"]

    # 1. Render unpainted context previews for the agent.
    ctx_dir = orch.context_preview_dir(case_id)
    proc = slicer_runner.render_context_previews(src, ctx_dir)
    if proc["status"] != 0:
        raise HTTPException(500, f"Could not render context previews:\n{proc['stderr']}")

    all_paths = orch.png_paths_in_dir(ctx_dir)
    paths = orch.selected_preview_png_paths(all_paths, provider)
    if not paths:
        raise HTTPException(500, "No context preview PNGs were generated.")
    meta_by_file = orch.preview_metadata_by_file(ctx_dir)
    ctx_meta = []
    for p in paths:
        m = meta_by_file.get(p.name)
        if m is None:
            raise HTTPException(500, f"No preview metadata for context image: {p.name}")
        ctx_meta.append(m)

    # 2. Call the vision model with prior feedback in the prompt.
    feedback = orch.read_json(orch.feedback_json(case_id))
    prompt = vision.vision_paint_prompt(ctx_meta, feedback, req.reviewer_prompt)
    try:
        call = vision.call_vision_model(provider, base_url, model, req.api_key, prompt, paths, 1600)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Vision model call failed: {exc}")

    parsed = orch.parse_json_from_model_text(call["output_text"])
    if not isinstance(parsed, dict) or not parsed:
        raise HTTPException(422, f"Paint agent did not return a JSON object.\nRaw output:\n{call['output_text']}")
    try:
        seed_spec, marking = orch.seed_spec_from_agent_json(parsed, meta_by_file)
    except ValueError as exc:
        raise HTTPException(422, f"{exc}\nRaw agent output:\n{call['output_text']}")

    # 3. Persist seeds and render seed previews.
    orch.case_seed_json(case_id).write_text(json.dumps(seed_spec, indent=2))
    render = slicer_runner.render_seed_previews(
        src, str(orch.case_seed_json(case_id)), orch.seed_preview_dir(case_id))
    if render["status"] != 0:
        raise HTTPException(500, f"Seed preview render failed:\n{render['stderr']}")

    qa = {
        "agent_mode": "vision_generated_paint",
        "agent_marking": marking,
        "confidence": parsed.get("confidence"),
        "issues": parsed.get("issues", []),
        "paint_agent": {
            "provider": provider, "endpoint": call["endpoint"], "model": model,
            "context_files": call["image_files"], "output_text": call["output_text"],
            "parsed": parsed,
        },
    }
    qa_json = orch.case_root(case_id) / "segmentation" / "auto_seed_qa.json"
    qa_json.write_text(json.dumps(qa, indent=2))
    orch.write_manifest_value(case_id, {
        "seed_json": str(orch.case_seed_json(case_id)),
        "auto_seed_qa_json": str(qa_json),
        "seed_preview_dir": str(orch.seed_preview_dir(case_id)),
        "latest_agent_paint": qa["paint_agent"],
    })
    return {"case_info": orch.current_case_info(case_id), "seed_spec": seed_spec, "qa": qa}


# ── Grow from Seeds (Stage 2) ──────────────────────────────────────────────
class GrowRequest(BaseModel):
    seed_locality_factor: float = 0.0


@app.post("/api/case/{case_id}/grow")
def grow(case_id: str, req: GrowRequest) -> dict:
    orch.ensure_case_dirs(case_id)
    src = str(_registered_volume(case_id))
    output_seg = orch.case_output_seg(case_id)
    qa_json = orch.case_qa_json(case_id)
    scene = orch.case_scene(case_id)
    proc = slicer_runner.grow_from_seeds(
        src, str(orch.case_seed_json(case_id)), str(output_seg), str(qa_json),
        str(scene), orch.segmentation_preview_dir(case_id), req.seed_locality_factor)
    if proc["status"] != 0 or not output_seg.exists():
        raise HTTPException(500, f"Grow from Seeds failed:\n{proc['stderr']}")
    orch.write_manifest_value(case_id, {
        "segmentation": str(output_seg),
        "qa_json": str(qa_json),
        "scene": str(scene),
        "segmentation_preview_dir": str(orch.segmentation_preview_dir(case_id)),
    })
    qa = orch.read_json(qa_json) if qa_json.exists() else None
    return {
        "case_info": orch.current_case_info(case_id),
        "qa": qa,
        "process": {"status": proc["status"]},
        "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id)),
    }


# ── Active-learning feedback ───────────────────────────────────────────────
class FeedbackRequest(BaseModel):
    decision: str
    notes: str = ""


@app.post("/api/case/{case_id}/feedback")
def save_feedback(case_id: str, req: FeedbackRequest) -> dict:
    orch.ensure_case_dirs(case_id)
    path = orch.feedback_json(case_id)
    feedback = orch.read_json(path)
    if not isinstance(feedback, list):
        feedback = []
    decision = "Accepted" if req.decision.strip().lower() in ("accept", "accepted", "correct") else "Rejected"
    feedback.append({
        "decision": decision,
        "notes": req.notes,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(feedback, indent=2))
    orch.write_manifest_value(case_id, {"feedback_json": str(path)})
    return {"case_info": orch.current_case_info(case_id), "feedback": feedback, "decision": decision}


@app.get("/api/case/{case_id}/segmentation/qa")
def segmentation_qa(case_id: str) -> dict:
    qa_json = orch.case_qa_json(case_id)
    if not qa_json.exists():
        raise HTTPException(404, "No segmentation QA yet. Run Grow from Seeds first.")
    return orch.read_json(qa_json)


# ── Stage 4: scar detection (optional 3rd class within cornea) ──────────────
def _regrow(case_id: str) -> dict:
    src = str(_registered_volume(case_id))
    proc = slicer_runner.grow_from_seeds(
        src, str(orch.case_seed_json(case_id)), str(orch.case_output_seg(case_id)),
        str(orch.case_qa_json(case_id)), str(orch.case_scene(case_id)),
        orch.segmentation_preview_dir(case_id), 0.0)
    if proc["status"] != 0 or not orch.case_output_seg(case_id).exists():
        raise HTTPException(500, f"Grow from Seeds failed:\n{proc['stderr']}")
    return proc


@app.post("/api/case/{case_id}/scar/heuristic")
def scar_heuristic(case_id: str) -> dict:
    """Add a deterministic scar seed inside cornea, re-grow, report metrics."""
    orch.ensure_case_dirs(case_id)
    try:
        scar_segment = scar_mod.heuristic_scar_seed(case_id)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    seed_spec = orch.read_json(orch.case_seed_json(case_id))
    merged = scar_mod.merge_scar_segment(seed_spec, scar_segment)
    orch.case_seed_json(case_id).write_text(json.dumps(merged, indent=2))
    _regrow(case_id)
    metrics = scar_mod.scar_metrics(case_id, merged)
    orch.write_manifest_value(case_id, {"scar_metrics": metrics})
    return {"case_info": orch.current_case_info(case_id), "metrics": metrics,
            "qa": orch.read_json(orch.case_qa_json(case_id))}


@app.post("/api/case/{case_id}/scar")
def scar_vision(case_id: str, req: AiPaintRequest) -> dict:
    """Ask a vision model to outline scar inside cornea, re-grow, report metrics."""
    orch.ensure_case_dirs(case_id)
    src = str(_registered_volume(case_id))
    cfg = settings.get_settings()
    provider = (req.provider or cfg["vision_provider"] or "local").strip().lower()
    model = (req.model or cfg["openai_model"] or "local-vision-model").strip()
    base_url = req.local_base_url or cfg["local_vision_base_url"]

    ctx_dir = orch.context_preview_dir(case_id)
    proc = slicer_runner.render_context_previews(src, ctx_dir)
    if proc["status"] != 0:
        raise HTTPException(500, f"Could not render context previews:\n{proc['stderr']}")
    paths = orch.selected_preview_png_paths(orch.png_paths_in_dir(ctx_dir), provider)
    meta_by_file = orch.preview_metadata_by_file(ctx_dir)
    ctx_meta = [meta_by_file[p.name] for p in paths if p.name in meta_by_file]

    prompt = vision.vision_scar_prompt(ctx_meta, req.reviewer_prompt)
    try:
        call = vision.call_vision_model(provider, base_url, model, req.api_key, prompt, paths, 1600)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"Vision model call failed: {exc}")
    parsed = orch.parse_json_from_model_text(call["output_text"])
    scar_segment = scar_mod.scar_segment_from_agent_json(parsed, meta_by_file)

    seed_spec = orch.read_json(orch.case_seed_json(case_id))
    if scar_segment is None:
        # Model reported no scar — valid (scar may be absent).
        metrics = {"scar_present": False, "scar_voxels": 0,
                   "note": "Vision model found no scar region."}
        orch.write_manifest_value(case_id, {"scar_metrics": metrics})
        return {"case_info": orch.current_case_info(case_id), "metrics": metrics,
                "qa": orch.read_json(orch.case_qa_json(case_id))}

    merged = scar_mod.merge_scar_segment(seed_spec, scar_segment)
    orch.case_seed_json(case_id).write_text(json.dumps(merged, indent=2))
    _regrow(case_id)
    metrics = scar_mod.scar_metrics(case_id, merged)
    orch.write_manifest_value(case_id, {"scar_metrics": metrics})
    return {"case_info": orch.current_case_info(case_id), "metrics": metrics,
            "qa": orch.read_json(orch.case_qa_json(case_id))}


@app.get("/api/case/{case_id}/segmentation.nii.gz")
def get_segmentation_nifti(case_id: str) -> FileResponse:
    seg = orch.case_output_seg(case_id)
    if not seg.exists():
        raise HTTPException(404, "No segmentation yet. Run Grow from Seeds first.")
    base = _ensure_volume_nifti(case_id)
    dst = orch.case_root(case_id) / "previews" / "segmentation.nii.gz"
    if (not dst.exists()) or dst.stat().st_mtime < seg.stat().st_mtime:
        try:
            volume_io.seg_to_label_nifti(seg, base, dst)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Segmentation conversion failed: {exc}")
    return FileResponse(str(dst), media_type="application/gzip", filename="segmentation.nii.gz")


# ── nnU-Net export (capstone) ──────────────────────────────────────────────
class ExportRequest(BaseModel):
    dataset_name: str = "Dataset501_CorneaOCT"
    cases: List[str] | None = None  # default: all cases with a segmentation


@app.post("/api/export/nnunet")
def export_nnunet(req: ExportRequest) -> dict:
    cases = req.cases if req.cases else export_mod.cases_with_segmentation()
    if not cases:
        raise HTTPException(400, "No cases with a segmentation to export. Run Grow from Seeds first.")
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
