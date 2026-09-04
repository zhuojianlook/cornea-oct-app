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
from fastapi.responses import FileResponse, JSONResponse, Response
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
import detector_tune
import oct_motion as oct_motion_mod
import cohort as cohort_mod
import debug_align

app = FastAPI(title="Cornea OCT Segmentation Sidecar")

# Only OUR OWN frontends may use this sidecar: the Tauri webview (tauri://localhost /
# https://tauri.localhost), the loopback host on any port (single-port serve.sh mode + direct niivue
# resource loads), and the Vite dev server. NOT "*", which let any website the user had open issue
# cross-origin calls to the localhost sidecar and READ the responses (wipe cases / write files /
# exfiltrate paths). Loopback binding alone does not stop other ORIGINS on the same machine.
_CORS_ORIGIN_REGEX = r"^(tauri://localhost|https://tauri\.localhost|https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?)$"
_ORIGIN_RE = re.compile(_CORS_ORIGIN_REGEX)
# Optional per-launch shared secret. When the Tauri shell injects CORNEA_API_TOKEN at spawn, every
# state-changing /api call must carry it (the Rust IPC proxy adds the header; a foreign page can never
# read it). Empty (dev / serve.sh) disables the check so those flows keep working.
_API_TOKEN = os.environ.get("CORNEA_API_TOKEN", "")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_CORS_ORIGIN_REGEX,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _origin_and_token_guard(request, call_next):
    """Defence-in-depth over CORS (which only governs response READABILITY): refuse a request whose
    Origin is present and not one of ours, and — when a token is configured — require it on mutating
    /api routes. Requests with NO Origin (same-origin GETs, the Rust proxy's server-to-server calls)
    are allowed; GET/HEAD are token-exempt so direct niivue resource fetches keep working."""
    origin = request.headers.get("origin")
    if origin and not _ORIGIN_RE.match(origin):
        return JSONResponse({"detail": "Forbidden origin."}, status_code=403)
    if _API_TOKEN and request.url.path.startswith("/api/") and request.method not in ("GET", "HEAD", "OPTIONS"):
        if request.headers.get("x-cornea-token", "") != _API_TOKEN:
            return JSONResponse({"detail": "Unauthorized."}, status_code=401)
    return await call_next(request)


@app.get("/api/health")
def health() -> dict:
    # shell_version echoes the env the Tauri shell set when it spawned this sidecar, so the app can
    # confirm it's talking to the sidecar IT launched (not a stale/foreign one). Empty in dev.
    return {"status": "ok", "shell_version": os.environ.get("CORNEA_SHELL_VERSION", "")}


def _require_case(case_id: str) -> str:
    """Resolve + sanitize a case id, 404 if its directory doesn't exist. write_manifest_value mkdirs the
    case dir, so a flag-only endpoint posting to a typo'd/unknown id would otherwise silently materialize a
    ghost case under CASES_ROOT. Mirrors the guard reset_step / vet_cornea already use."""
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, f"No such case: {case_id}")
    return cid


# ── Upload size limits (DoS guard) ─────────────────────────────────────────
# The sidecar listens on loopback and is reachable by any allowed-origin page, so an upload
# handler that reads the whole body into memory in one shot can be made to exhaust RAM/disk.
# Stream uploads to disk (or a bounded buffer) in chunks and reject anything over budget with 413.
# Generous defaults so legitimate OCT volumes/cohorts never trip them; env-overridable.
_UPLOAD_CHUNK = 1 << 20  # 1 MiB read granularity
_MAX_UPLOAD_BYTES = int(os.environ.get("CORNEA_MAX_UPLOAD_BYTES", str(2 * 1024 ** 3)))      # 2 GiB per file
_MAX_UPLOAD_FILES = int(os.environ.get("CORNEA_MAX_UPLOAD_FILES", "512"))                   # files per request
_MAX_REQUEST_BYTES = int(os.environ.get("CORNEA_MAX_REQUEST_BYTES", str(16 * 1024 ** 3)))   # total per request


def _check_upload_count(files: List[UploadFile]) -> None:
    """Cap the number of files accepted in one multi-file upload request."""
    if len(files) > _MAX_UPLOAD_FILES:
        raise HTTPException(413, f"Too many files in one request (max {_MAX_UPLOAD_FILES}).")


async def _read_upload_bytes(up: UploadFile, max_bytes: int = _MAX_UPLOAD_BYTES) -> bytes:
    """Read an UploadFile fully into memory in bounded chunks, aborting with 413 once max_bytes
    is exceeded (so an oversized upload can't be buffered in one unbounded read())."""
    buf = bytearray()
    while True:
        chunk = await up.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        buf += chunk
        if len(buf) > max_bytes:
            raise HTTPException(413, f"Upload exceeds the maximum allowed size ({max_bytes} bytes).")
    return bytes(buf)


async def _stream_upload_to(up: UploadFile, dest: Path, max_bytes: int = _MAX_UPLOAD_BYTES) -> int:
    """Stream an UploadFile to dest in bounded chunks, aborting with 413 (and removing the partial
    file) once max_bytes is exceeded. Returns the number of bytes written."""
    written = 0
    with open(dest, "wb") as fh:
        while True:
            chunk = await up.read(_UPLOAD_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > max_bytes:
                fh.close()
                try:
                    dest.unlink()
                except OSError:
                    pass
                raise HTTPException(413, f"Upload exceeds the maximum allowed size ({max_bytes} bytes).")
            fh.write(chunk)
    return written


def _total_ram_gb() -> float:
    """Total system RAM in GiB (best-effort, cross-platform). Used to size batch concurrency so the app uses
    the machine it runs on without oversubscribing memory."""
    try:
        # POSIX (Linux): sysconf is exact and dependency-free.
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024.0 ** 3)
    except (ValueError, AttributeError, OSError):
        pass
    try:
        import psutil  # optional
        return float(psutil.virtual_memory().total) / (1024.0 ** 3)
    except Exception:  # noqa: BLE001
        return 8.0  # conservative fallback


def _gpu_info() -> dict:
    """CUDA GPU name + VRAM via the nvidia-smi SUBPROCESS — deliberately NOT `import torch; torch.cuda...`,
    because this long-lived sidecar later runs detect_surface_all through an in-process mp 'fork' pool, and
    initialising a CUDA context here would fork a CUDA-bearing process (the no-CUDA-before-fork invariant the
    codebase relies on). Returns cuda=False if nvidia-smi is absent or fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=4)
        line = (out.stdout or "").strip().splitlines()
        if out.returncode == 0 and line:
            parts = [x.strip() for x in line[0].split(",")]
            name = parts[0] if parts else None
            vram = round(float(parts[1]) / 1024.0, 1) if len(parts) > 1 and parts[1] else 0.0  # MiB → GiB
            return {"cuda": True, "name": name, "vram_gb": vram}
    except Exception:  # noqa: BLE001 — no GPU / no driver / unparseable → report no CUDA
        pass
    return {"cuda": False, "name": None, "vram_gb": 0.0}


def _cpu_budget() -> int:
    """Threads to spread CPU work across — all cores but one (left for the main/IO thread)."""
    return max(2, (os.cpu_count() or 2) - 1)


def _recommend_max_concurrency(ram_gb: float | None = None) -> int:
    """How many scans to PREPROCESS at once on this machine. Bound by CPU (each concurrent scan still wants a
    few worker threads for its per-slice parallel phases) AND by RAM (~3 GiB peak per concurrent scan: the
    sagittal volume + per-pass copies + worker processes), so a big box runs many in parallel and a small one
    stays safe. The per-scan worker count is then cpu_budget // concurrency (see oct-preprocess)."""
    cpu = _cpu_budget()
    ram = _total_ram_gb() if ram_gb is None else float(ram_gb)
    by_cpu = max(1, cpu // 3)                       # keep >=3 worker threads per concurrent scan
    by_ram = max(1, int((ram - 4.0) // 3.0))       # ~3 GiB per scan AFTER a 4 GiB OS/app/browser reserve
    return max(1, min(by_cpu, by_ram, 16))


@app.get("/api/system/capabilities")
def system_capabilities() -> dict:
    """System resources so the frontend can size batch preprocessing to THIS machine (CPU cores, RAM, GPU).
    The app aims to use whatever it runs on: max_concurrency scans preprocess at once, each getting
    cpu_budget // concurrency CPU workers; SAM2/nnU-Net use the GPU (serialised by a lock to fit VRAM)."""
    ram = round(_total_ram_gb(), 1)
    return {
        "cpu_count": os.cpu_count() or 2,
        "cpu_budget": _cpu_budget(),
        "ram_gb": ram,
        "gpu": _gpu_info(),
        "max_concurrency": _recommend_max_concurrency(ram),
    }


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


def _invalidate_derived_volume(case_id: str) -> None:
    """Remove the derived NIfTI (previews/volume.nii.gz) and its dependent
    preprocessed preview so they are rebuilt from the (new) registered source by
    _ensure_volume_nifti / _working_volume. Needed because those rebuild on an
    mtime '<' comparison that can wrongly serve a stale conversion when the source
    is re-pointed at a different (or in-place replaced) file."""
    previews = orch.case_root(case_id) / "previews"
    for name in ("volume.nii.gz", "preprocessed.nii.gz", "volume_display.nii.gz"):
        try:
            (previews / name).unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _invalidate_derived_volume_if_source_changed(case_id: str, new_src: Path) -> None:
    """Invalidate the derived volume only when the registered source path actually
    changes, so re-registering the same path is a no-op (keeps any preprocessing)."""
    try:
        prev = orch.read_manifest(case_id).get("input_volume")
    except Exception:  # noqa: BLE001 — best-effort; on any read failure, rebuild to be safe
        prev = None
    if prev is None or str(Path(prev)) != str(new_src):
        _invalidate_derived_volume(case_id)


@app.post("/api/case/{case_id}/volume/register")
def register_volume(case_id: str, payload: RegisterVolume) -> dict:
    orch.ensure_case_dirs(case_id)
    volume = Path(payload.volume_path)
    if not volume.exists():
        raise HTTPException(400, f"Volume does not exist: {payload.volume_path}")
    _invalidate_derived_volume_if_source_changed(case_id, volume)
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
    await _stream_upload_to(upload, dest)
    # The bytes may have changed even when the path is reused; always rebuild the derived
    # NIfTI rather than trusting the mtime '<' check in _ensure_volume_nifti.
    _invalidate_derived_volume(case_id)
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
    base = _working_volume(case_id)
    # CROP-SHADE: serve the dim-crop DISPLAY volume to the viewer when present + current — it shows the
    # cropped-out (uncorrected) tissue at reduced intensity so the user can see what was removed. The pipeline
    # / SAM2 read _working_volume (the FILE, crop zeroed) directly, so segmentation is unaffected; the display
    # volume has the SAME shape/affine (only the crop-region intensity differs) so any overlay still aligns.
    disp = orch.case_root(case_id) / "previews" / "volume_display.nii.gz"
    dst = disp if (disp.exists() and disp.stat().st_mtime >= base.stat().st_mtime) else base
    return FileResponse(str(dst), media_type="application/gzip", filename="volume.nii.gz")


@app.get("/api/case/{case_id}/crop-mask.nii.gz")
def get_crop_mask(case_id: str) -> FileResponse:
    """CROP-SHADE (red highlight): a binary mask = 1 over the cropped (blink / off-cornea) region, 0 elsewhere,
    same shape/affine as the working volume. The viewer overlays it in RED so the cropped region is highlighted
    (over the dim cropped tissue served by /volume.nii.gz). 404 when nothing was cropped."""
    import nibabel as nib
    m = orch.read_manifest(case_id)
    cr = (m.get("oct_params") or {}).get("crop_region") or m.get("auto_crop_region")
    cl = (m.get("oct_params") or {}).get("crop_lateral")
    base = _working_volume(case_id)
    if not base.exists():
        raise HTTPException(404, "no volume")
    img = nib.load(str(base))
    shp = img.shape                                              # (lateral, depth, frame)
    mask = np.zeros(shp, dtype=np.uint8)
    frames = [int(f) for f in ((cr or {}).get("frames") or [])]
    lat = (cr or {}).get("lateral") or [0, shp[0] - 1]
    lo = max(0, int(lat[0])); hi = min(shp[0] - 1, int(lat[1]) if int(lat[1]) >= 0 else shp[0] - 1)
    for f in frames:
        if 0 <= f < shp[2]:
            mask[lo:hi + 1, :, f] = 1
    for l in (cl or []):
        if 0 <= int(l) < shp[0]:
            mask[int(l), :, :] = 1
    if int(mask.sum()) == 0:
        raise HTTPException(404, "no crop")
    out = orch.case_root(case_id) / "previews" / "crop_mask.nii.gz"
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out))
    return FileResponse(str(out), media_type="application/gzip", filename="crop-mask.nii.gz")


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


def _reject_protected_dest(dest: Path) -> None:
    """Native-save destinations are user-chosen (desktop Save dialog), but because CORS is open a
    request could aim `dest` at the app's own data — refuse to write inside the case store /
    workspace so these endpoints can never clobber managed case files or the sidecar state."""
    try:
        resolved = dest.expanduser().resolve()
    except Exception:  # noqa: BLE001
        return
    for guarded in (settings.CASES_ROOT, settings.WORKSPACE_ROOT):
        try:
            groot = Path(guarded).resolve()
        except Exception:  # noqa: BLE001
            continue
        if resolved == groot or groot in resolved.parents:
            raise HTTPException(400, "Destination is inside the app data directory; choose a path outside it.")


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
    _reject_protected_dest(dest)
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
    _reject_protected_dest(dest)
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
    # De-dupe (order-preserving) + validate plane names: each LIST ENTRY casts one vote in segment_volume, so a
    # duplicated plane (e.g. ['axial','axial','coronal']) would let a SINGLE physical plane reach the 2-of-3
    # threshold on its own — defeating the cross-plane consensus. Clamp the vote to the DISTINCT plane count.
    planes = list(dict.fromkeys(req.planes))
    bad = [p for p in planes if p not in ("axial", "coronal", "sagittal")]
    if bad:
        raise HTTPException(400, f"Unknown plane(s): {bad}. Use axial/coronal/sagittal.")
    base = _ensure_volume_nifti(case_id)            # SAM2 likes natural raw contrast
    work = orch.case_root(case_id) / "sam2_work"
    vote = max(1, min(req.vote, len(planes)))       # vote can't exceed #DISTINCT planes (else always empty)
    n_planes = len(planes)

    def _progress(phase, index, total):
        if phase == "fuse":
            _sam2_progress_set(case_id, "fuse", "Fusing planes in 3D", total, total)
        else:
            _sam2_progress_set(case_id, phase, f"Tracking cornea — {phase} ({index + 1}/{total})", index, total)

    _sam2_progress_set(case_id, "start", "Starting SAM2…", 0, n_planes)
    try:
        with _GPU_LOCK:                              # one SAM2/CUDA inference at a time
            label, meta = sam2_segment.segment_volume(
                base, work, planes=tuple(planes), vote=vote, progress=_progress)
        # Regularize the cornea to a SMOOTH shell — drops protrusion spikes where SAM2 follows the central
        # specular/saturation streak past the true posterior surface (the user shouldn't have to paint these
        # away slice-by-slice). Conservative + scar-safe; no-op on an already-smooth cornea.
        label = scar_mod.regularize_cornea(label)
        # ANTERIOR-SURFACE CLIP: SAM2 auto-prompts on the bright band, so its anterior is a rough band-top, not the
        # epithelium. Force the cornea's ANTERIOR onto the reviewer-accurate CORRECTED-RESULT reconstruction (drawn
        # edges + across-lateral interpolation, DP fallback) — the same surface the corrected pane serves. So the
        # reviewer's pixel-accurate edge flows straight into the training label, especially on steep limbus
        # descents where the raw DP fails. base == the corrected volume (byte-identical), so label/surface align.
        _m_seg = orch.read_manifest(case_id)
        if bool((_m_seg.get("oct_params") or {}).get("seg_clip_anterior", oct_mod.DEFAULT_PARAMS.get("seg_clip_anterior", True))):
            try:
                _pp = {**oct_mod.DEFAULT_PARAMS, **(_m_seg.get("oct_params") or {})}
                _bvol = np.asarray(nib.load(str(base)).dataobj).astype(np.float32)   # (lateral, depth, frame)
                _surf = oct_mod.detect_surface_all(_bvol, _pp)                        # (lateral, frames)
                _cea = _pp.get("corrected_edge_anchors")
                if _cea:
                    _surf = oct_mod.generalize_corrected_surface(_surf, _cea, _pp)
                label = postprocess.clip_labelmap_anterior_to_surface(label, _surf)
            except Exception as _clip_exc:  # noqa: BLE001 — best-effort; never fail segmentation over the clip
                print(f"[segment] anterior-surface clip skipped: {_clip_exc}", file=sys.stderr)
    except Exception:
        _sam2_progress_set(case_id, "error", "SAM2 failed", n_planes, n_planes)
        raise
    if label.sum() == 0:
        _sam2_progress_set(case_id, "error", "SAM2 produced an empty mask", n_planes, n_planes)
        raise HTTPException(500, f"SAM2 produced an empty mask: {meta}")
    _sam2_progress_set(case_id, "done", "Cornea segmented", n_planes, n_planes)
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


@app.get("/api/case/{case_id}/segment/sam2/status")
def segment_sam2_status(case_id: str) -> dict:
    """Live SAM2 progress for the front-end poll (served on a separate thread while the POST holds the
    GPU lock). Returns {phase, index, total, message}; phase 'idle' when nothing is/was running."""
    return _sam2_progress_get(case_id)


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
async def segmentation_from_drawing(case_id: str, files: List[UploadFile] = File(...),
                                    cornea_vet: bool = False) -> dict:
    """Save an edited segmentation drawing as the canonical corrected labelmap, then re-render the overlay
    so the gallery reflects the correction. #11 cornea_vet=true → this is the CORNEA/BACKGROUND vet step
    (paint cornea/background only, before scar): the labelmap is saved the same way, but we set the
    `cornea_vetted` flag (which gates the Scar step) INSTEAD of `corrected_labelmap` (the final manual-
    correction flag), so the timeline advances Cornea → Cornea/bg-vetted, not straight to Corrected."""
    orch.ensure_case_dirs(case_id)
    if not files:
        raise HTTPException(400, "No drawing uploaded.")
    data = await _read_upload_bytes(files[0])
    is_gz = len(data) >= 2 and data[0] == 0x1F and data[1] == 0x8B
    tmp = orch.case_root(case_id) / "previews" / ("edited-seg.nii.gz" if is_gz else "edited-seg.nii")
    tmp.write_bytes(data)
    base = _ensure_volume_nifti(case_id)
    with _labelmap_lock(case_id):
        try:
            arr = masks.corrected_labelmap_from_drawing(tmp, base, labels.corrected_path(case_id))
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Could not parse corrected drawing: {exc}")
        postprocess.render_seg_previews(_working_volume(case_id), arr, orch.segmentation_preview_dir(case_id))
        qa = {"segments": labels.labelmap_counts(arr), "source": "cornea_vet" if cornea_vet else "corrected"}
        orch.write_manifest_value(case_id, {"cornea_vetted": True} if cornea_vet
                                  else {"corrected_labelmap": str(labels.corrected_path(case_id))})
        return {"case_info": orch.current_case_info(case_id), "qa": qa,
                "images": orch.preview_images_from_dir("Segmentation", orch.segmentation_preview_dir(case_id))}


@app.post("/api/case/{case_id}/segmentation/from-drawing-cornea-vet")
async def segmentation_from_drawing_cornea_vet(case_id: str, files: List[UploadFile] = File(...)) -> dict:
    """#11 cornea/background VET confirm — same as segmentation/from-drawing but cornea_vet hardcoded true
    (sets `cornea_vetted`, not `corrected_labelmap`). A DEDICATED endpoint so the flag can't be lost to a
    dropped `?cornea_vet=true` query string through the upload proxy (which was making Confirm a no-op)."""
    return await segmentation_from_drawing(case_id, files, cornea_vet=True)


@app.post("/api/case/{case_id}/vet-cornea")
def vet_cornea(case_id: str) -> dict:
    """#11 — confirm the cornea/background segmentation is correct WITHOUT painting (the SAM2 result was
    already good). Sets `cornea_vetted`, which unlocks the Scar step. (Painting + confirm goes through
    segmentation/from-drawing?cornea_vet=true instead.)"""
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, "Unknown case.")
    m = orch.write_manifest_value(cid, {"cornea_vetted": True})
    return {"ok": True, "cornea_vetted": bool(m.get("cornea_vetted"))}


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


@app.get("/api/case/{case_id}/segmentation-display.nii.gz")
def get_segmentation_display_nifti(case_id: str) -> FileResponse:
    """DISPLAY overlay for the 3D viewer: cornea=1, scar split into density tiers 2/3/4 (diffuse→dense)
    so reflectivity is visible instead of one flat red. The canonical 0/1/2 training label is untouched
    (see segmentation.nii.gz). Density = the raw reflectivity volume, normalised per-eye to the cornea."""
    import nibabel as nib
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        raise HTTPException(404, "No segmentation yet. Run SAM2 first.")
    base = _ensure_volume_nifti(case_id)
    dst = orch.case_root(case_id) / "previews" / "segmentation-display.nii.gz"
    with _labelmap_lock(case_id):          # serialise the regenerate (fixed tmp name in write_label_nifti)
        try:
            density = np.asarray(nib.load(str(base)).dataobj).astype(np.float32)
            if density.shape[:3] != arr.shape[:3]:
                density = None             # geometry mismatch → skip tiering rather than raise
            labels.write_display_labelmap(arr, density, base, dst)
        except Exception as exc:  # noqa: BLE001 — never block the viewer; fall back to the plain overlay
            print(f"[display-labelmap] tiered overlay failed for {case_id}: {exc}", file=sys.stderr)
            try:
                labels.write_display_labelmap(arr, None, base, dst)   # no density → cornea=1, scar=4 (solid red)
            except Exception as exc2:  # noqa: BLE001 — last resort
                raise HTTPException(500, f"Segmentation display conversion failed: {exc2}")
    return FileResponse(str(dst), media_type="application/gzip", filename="segmentation-display.nii.gz")


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
    _check_upload_count(files)
    try:
        base = _working_volume(cid)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"This case has no working volume to align against: {exc}")
    imported: list[dict] = []
    errors: list[dict] = []
    for f in files:
        try:
            data = await _read_upload_bytes(f)
        except HTTPException as exc:
            errors.append({"file": f.filename or "gt", "error": exc.detail})
            continue
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


@app.get("/api/case/{case_id}/overlap/{a}/{b}.nii.gz")
def get_overlap_nifti(case_id: str, a: str, b: str, label: str = "cornea") -> FileResponse:
    """3-label OVERLAP map (1 = A only, 2 = B only, 3 = both) of the `label` region (cornea|scar) between two
    sources, BOTH already in the reference frame: a consensus member id, or the literal "consensus" for the
    voted result. Powers the Volume-align view (each scan its own colour, overlap red, cornea only — no
    background) and the Both view (per-scan scar vs the consensus scar, distinct colours)."""
    ccid = orch.safe_case_id(case_id)
    if not orch.read_manifest(ccid).get("consensus_cases"):
        raise HTTPException(400, "Not a consensus case — align the replicates first.")
    want_scar = (label == "scar")

    def _mask(src: str) -> np.ndarray:
        if src == "consensus":
            p = labels.corrected_path(ccid)
        else:
            p = orch.case_root(ccid) / "scans" / orch.safe_case_id(src) / "label.nii.gz"
        if not Path(str(p)).exists():
            raise HTTPException(404, f"No warped label for '{src}' — rebuild the consensus.")
        arr = _read_label_ijk(p)
        return (arr == 2) if want_scar else (arr >= 1)   # scar = label 2; cornea = cornea+scar (≥1)

    ma, mb = _mask(a), _mask(b)
    if ma.shape != mb.shape:
        raise HTTPException(400, "Overlap operands differ in shape — rebuild the consensus.")
    ov = np.zeros(ma.shape, dtype=np.uint8)
    ov[ma & ~mb] = 1            # A only
    ov[mb & ~ma] = 2            # B only
    ov[ma & mb] = 3            # both → red
    base = orch.case_root(ccid) / "previews" / "volume.nii.gz"
    safe = lambda s: "cons" if s == "consensus" else orch.safe_case_id(s)
    dst = orch.case_root(ccid) / "previews" / f"overlap_{'scar' if want_scar else 'cornea'}_{safe(a)}_{safe(b)}.nii.gz"
    labels.write_label_nifti(ov, base, dst)
    return FileResponse(str(dst), media_type="application/gzip", filename="overlap.nii.gz")


@app.get("/api/case/{case_id}/cornea.nii.gz")
def get_cornea_nifti(case_id: str) -> FileResponse:
    """The case's CORNEA mask (1 where cornea or scar) — a faint anatomical CONTEXT layer for the scar-overlap
    view, so the scar agreement isn't floating in empty space. Derived from the corrected labelmap."""
    cid = orch.safe_case_id(case_id)
    lp = labels.corrected_path(cid)
    if not Path(str(lp)).exists():
        raise HTTPException(404, "No segmentation for this case.")
    arr = (_read_label_ijk(lp) >= 1).astype(np.uint8)
    base = orch.case_root(cid) / "previews" / "volume.nii.gz"
    base = base if base.exists() else lp
    dst = orch.case_root(cid) / "previews" / "cornea_mask.nii.gz"
    labels.write_label_nifti(arr, base, dst)
    return FileResponse(str(dst), media_type="application/gzip", filename="cornea.nii.gz")


@app.get("/api/case/{case_id}/align-region/{a}/{b}/{region}.nii.gz")
def get_align_region_nifti(case_id: str, a: str, b: str, region: str) -> FileResponse:
    """For Volume-align: a per-region map (1 = cornea, 2 = scar) of ONE of the three regions of the A/B overlap
    (both in the reference frame). region ∈ {a, b, both}: 'a' = in A but not B, 'b' = in B but not A,
    'both' = the intersection. The viewer loads all three with distinct colours + independent opacity (cornea
    faint, scar opaque), so 'what aligns' is clear: red (both) = aligned; a coloured ghost = residual offset."""
    ccid = orch.safe_case_id(case_id)
    if not orch.read_manifest(ccid).get("consensus_cases"):
        raise HTTPException(400, "Not a consensus case — align the replicates first.")
    if region not in ("a", "b", "both"):
        raise HTTPException(400, "region must be a|b|both.")

    def _lab(src: str) -> np.ndarray:
        p = labels.corrected_path(ccid) if src == "consensus" else orch.case_root(ccid) / "scans" / orch.safe_case_id(src) / "label.nii.gz"
        if not Path(str(p)).exists():
            raise HTTPException(404, f"No warped label for '{src}' — rebuild the consensus.")
        return _read_label_ijk(p)

    la, lb = _lab(a), _lab(b)
    if la.shape != lb.shape:
        raise HTTPException(400, "Operands differ in shape — rebuild the consensus.")
    ac, bc, asc, bsc = la >= 1, lb >= 1, la == 2, lb == 2
    if region == "a":
        cm, sm = ac & ~bc, asc & ~bsc
    elif region == "b":
        cm, sm = bc & ~ac, bsc & ~asc
    else:
        cm, sm = ac & bc, asc & bsc
    out = np.zeros(la.shape, dtype=np.uint8)
    out[cm] = 1          # region cornea
    out[sm] = 2          # region scar (overrides cornea where both)
    base = orch.case_root(ccid) / "previews" / "volume.nii.gz"
    dst = orch.case_root(ccid) / "previews" / f"align_{orch.safe_case_id(a)}_{orch.safe_case_id(b)}_{region}.nii.gz"
    labels.write_label_nifti(out, base, dst)
    return FileResponse(str(dst), media_type="application/gzip", filename="region.nii.gz")


class ConsensusScarRequest(BaseModel):
    mode: str = "own"   # "consensus" → push the voted consensus scar to every member; "own" → keep each member's


@app.post("/api/case/{case_id}/consensus-scar")
def consensus_scar_choice(case_id: str, req: ConsensusScarRequest | None = None) -> dict:
    """STEP 9 scar-source decision for an aligned consensus. mode='consensus' → set each member's
    corrected_labelmap = its own cornea + the VOTED CONSENSUS scar mapped into that member's native frame
    (cons_native.nii.gz, already truncated to the member's cornea + data FOV, so a partial-FOV scan only gets
    the part of the consensus scar within its own data). mode='own' → keep each member's own scar (no-op).
    Records consensus_scar_source on the consensus + each member."""
    mode = (req.mode if req else "own").strip().lower()
    if mode not in ("consensus", "own"):
        raise HTTPException(400, "mode must be 'consensus' or 'own'.")
    cid = orch.safe_case_id(case_id)
    m = orch.read_manifest(cid)
    ccid = cid if m.get("consensus_cases") else (m.get("consensus_case") or "")
    if not ccid or not orch.read_manifest(ccid).get("consensus_cases"):
        raise HTTPException(400, "No aligned consensus for this scan yet — align the replicates first.")
    members = list(orch.read_manifest(ccid).get("consensus_cases") or [])
    applied, skipped = [], []
    if mode == "consensus":
        scans_dir = orch.case_root(ccid) / "scans"
        for mc in members:
            cn = scans_dir / orch.safe_case_id(mc) / "cons_native.nii.gz"
            nat_vol = orch.case_root(mc) / "previews" / "volume.nii.gz"
            if not cn.exists() or not nat_vol.exists():
                print(f"[consensus-scar] no native consensus map for {mc} — skipped", file=sys.stderr)
                skipped.append(mc)
                continue
            try:
                cons_arr = _read_label_ijk(cn)
                # SAFETY: never wipe a member's own scar with an empty consensus. If the voted consensus has no
                # scar in THIS scan's FOV (cons_native is cornea-only), keep the member's own scar untouched.
                if not (cons_arr == 2).any():
                    print(f"[consensus-scar] {mc}: consensus has no scar in this scan's FOV — kept its own scar", file=sys.stderr)
                    skipped.append(mc)
                    continue
                # Serialize against scar/edit + scar/auto on the SAME member (they do a locked
                # read-modify-write of this labelmap) — without the lock a concurrent brush/auto run
                # could clobber, or be clobbered by, this consensus write (last-writer-wins).
                with _labelmap_lock(mc):
                    labels.write_label_nifti(cons_arr, nat_vol, labels.corrected_path(mc))
                    # the member's OWN scar is now the consensus scar — refresh its context_seg preview so the
                    # Scans-grid "Per scan" column matches the new labelmap (else it shows the pre-apply scar).
                    try:
                        postprocess.render_seg_previews(nat_vol, _read_label_ijk(labels.corrected_path(mc)),
                                                        _preview_group_dir(mc, "context_seg"), dense_rotated=True, density_from_self=True)
                    except Exception:  # noqa: BLE001 — preview refresh is best-effort
                        pass
                    orch.case_qa_json(mc).unlink(missing_ok=True)   # stale QA; recomputed on next view
                    orch.write_manifest_value(mc, {"corrected_labelmap": True, "consensus_scar_source": "consensus"})
                applied.append(mc)
            except Exception as exc:  # noqa: BLE001
                print(f"[consensus-scar] apply failed for {mc}: {exc}", file=sys.stderr)
    else:
        for mc in members:
            orch.write_manifest_value(mc, {"consensus_scar_source": "own"})
    orch.write_manifest_value(ccid, {"consensus_scar_source": mode})
    return {"ok": True, "mode": mode, "members": members, "applied": applied, "skipped": skipped}


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
    percentile: float = 92.0     # sensitivity: flag the brightest (100−percentile)% of cornea; default = validated hysteresis phi
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
    with _labelmap_lock(case_id):
        return _scar_auto_locked(case_id, req, nib)


def _scar_auto_locked(case_id: str, req: "ScarAutoRequest", nib) -> dict:
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
    postprocess.render_seg_previews(work, new_label, orch.segmentation_preview_dir(case_id), density_vol=raw)
    metrics = scar_mod.quantify(new_label, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
    metrics["scar_method"] = (req.method or "hysteresis") + profile_note
    orch.write_manifest_value(case_id, {"scar_metrics": metrics, "scar_done": True,
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
    with _labelmap_lock(case_id):
        return _scar_edit_locked(case_id, req, nib)


def _scar_edit_locked(case_id: str, req: "ScarEditRequest", nib) -> dict:
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
    postprocess.render_seg_previews(work, arr, orch.segmentation_preview_dir(case_id), density_vol=raw)
    metrics = scar_mod.quantify(arr, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
    orch.write_manifest_value(case_id, {"scar_metrics": metrics, "scar_done": True})
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
    with _labelmap_lock(case_id):
        return _scar_sam2_hint_locked(case_id, req, sam2_segment, nib)


def _scar_sam2_hint_locked(case_id: str, req: "ScarHintRequest", sam2_segment, nib) -> dict:
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
    postprocess.render_seg_previews(work_vol, new_label, orch.segmentation_preview_dir(case_id), density_vol=raw)
    metrics = scar_mod.quantify(new_label, nib.load(str(base)).header.get_zooms(), density_vol_ijk=raw)
    metrics["sam2_hint"] = meta
    orch.write_manifest_value(case_id, {"scar_metrics": metrics, "scar_done": True})
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
    import sam2_segment
    import nibabel as nib
    orch.ensure_case_dirs(case_id)
    with _labelmap_lock(case_id):
        return _scar_auto_sam2_locked(case_id, req, sam2_segment, nib)


def _scar_auto_sam2_locked(case_id: str, req: "ScarAutoSam2Request", sam2_segment, nib) -> dict:
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
    postprocess.render_seg_previews(work_vol, new_label, orch.segmentation_preview_dir(case_id), density_vol=raw)
    metrics = scar_mod.quantify(new_label, spacing, density_vol_ijk=raw)
    metrics["scar_auto_sam2"] = meta
    orch.write_manifest_value(case_id, {"scar_metrics": metrics, "scar_done": True,
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
def _scar_request() -> "ScarAutoRequest":
    """Default scar request, CONTROL-NORMALISED when a control baseline exists: with controls tagged + a normal
    profile built, scar is flagged as EXCESS over the normal corneal reflectivity ("depthnorm"), which is more
    reproducible than the absolute-brightness hysteresis fallback. No baseline → hysteresis (the default)."""
    try:
        if normal_baseline.load_profile() is not None:
            return ScarAutoRequest(method="depthnorm")
    except Exception:  # noqa: BLE001
        pass
    return ScarAutoRequest()


def _ensure_segmented(case_id: str) -> None:
    """Make sure a case has a cornea+scar labelmap (preprocess → SAM2 → scar/auto). Scar uses the
    control-normalised method when a control baseline has been built (see _scar_request)."""
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        if not _preprocessed_path(case_id).exists():
            preprocess_case(case_id, PreprocessRequest(enabled=True))
        segment_sam2(case_id, Sam2Request())
        scar_auto(case_id, _scar_request())
    elif not (arr == 2).any():
        if not _preprocessed_path(case_id).exists():
            preprocess_case(case_id, PreprocessRequest(enabled=True))
        scar_auto(case_id, _scar_request())


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
    postprocess.render_seg_previews(cons_vol, cons_lab, _preview_group_dir(ccid, "segmentation"), density_from_self=True)
    # Per-scan tabs: each scan's warped image with its own scar, and with the consensus
    # scar clipped to that scan's FOV (so it isn't painted over empty background).
    scans_dir = orch.case_root(ccid) / "scans"
    for cid in report["scans"]:
        svol = scans_dir / cid / "volume.nii.gz"
        slab = _read_label_ijk(scans_dir / cid / "label.nii.gz")
        data_mask = np.asarray(nib.load(str(svol)).dataobj) > 0
        cons_clipped = np.where(data_mask, cons_lab, 0).astype(np.uint8)
        postprocess.render_seg_previews(svol, slab, _preview_group_dir(ccid, f"scan_{cid}_self"), density_from_self=True)
        postprocess.render_seg_previews(svol, cons_clipped, _preview_group_dir(ccid, f"scan_{cid}_cons"), density_from_self=True)
        # Dense+rotated overlays in the SCAN's NATIVE frame for the gallery's 3rd before/after panel (aligns
        # slice-for-slice with raw/corrected). context_seg (own cornea+scar) and context_cons (the subgroup
        # consensus scar mapped to this scan's native frame) are rendered in SEPARATE try blocks so a failure
        # of one never silently drops the other — the Scans-grid "Per scan ↔ Consensus" toggle needs BOTH.
        nat_vol = orch.case_root(cid) / "previews" / "volume.nii.gz"
        try:
            postprocess.render_seg_previews(nat_vol, _read_label_ijk(labels.corrected_path(cid)),
                                            _preview_group_dir(cid, "context_seg"), dense_rotated=True, density_from_self=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[consensus] context_seg panel skipped for {cid}: {exc}", file=sys.stderr)
        try:
            cons_native = scans_dir / cid / "cons_native.nii.gz"
            # Fall back to the scan's own CORNEA (no scar) if the native consensus map is missing, so the
            # Consensus toggle ALWAYS shows a distinct (scar-free) result instead of reusing the per-scan image.
            cons_lab_native = (_read_label_ijk(cons_native) if cons_native.exists()
                               else np.where(_read_label_ijk(labels.corrected_path(cid)) >= 1, 1, 0).astype(np.uint8))
            postprocess.render_seg_previews(nat_vol, cons_lab_native,
                                            _preview_group_dir(cid, "context_cons"), dense_rotated=True, density_from_self=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[consensus] context_cons panel skipped for {cid}: {exc}", file=sys.stderr)
        # Link the scan back to its consensus + subgroup (frontend panel + metrics attribution).
        orch.write_manifest_value(cid, {"consensus_case": ccid, "scar_subgroup": sub_label})

    orch.write_manifest_value(ccid, {
        "input_volume": str(cons_vol), "corrected_volume": str(cons_vol),
        "consensus_report": report, "consensus_cases": report["scans"], "reference": report["reference"],
        "scar_subgroup": sub_label,
    })
    return ccid, report


def _case_identity(cid: str) -> tuple[str, str, str]:
    """(patient, eye, subgroup) for a case, lowercased — manifest first, then filename parse; subgroup
    defaults to '1'. Used to group replicates of the SAME eye."""
    m = orch.read_manifest(cid)
    pid = str(m.get("patient_id") or "").strip().lower()
    eye = str(m.get("eye") or "").strip().lower()
    sub = (str(m.get("scar_subgroup") or "1").strip() or "1").lower()
    if not (pid and eye):
        meta = metrics_export.parse_case_meta(m.get("oct_source") or m.get("input_volume"))
        pid = pid or str(meta.get("patient_id", "")).strip().lower()
        eye = eye or str(meta.get("eye", "")).strip().lower()
    return pid, eye, sub


def _case_crop_lateral(case_id: str) -> list[int]:
    """LEGACY #9 v1 — the persisted full-slice lateral-crop columns (oct_params.crop_lateral), or []."""
    raw = (orch.read_manifest(case_id).get("oct_params") or {}).get("crop_lateral") or []
    out = []
    for c in raw:
        try:
            out.append(int(c))
        except (ValueError, TypeError):
            pass
    return sorted(set(out))


def _case_crop_region(case_id: str):
    """#9 v2 — the persisted BOX crop (oct_params.crop_region = {'lateral':[lo,hi], 'frames':[…]}), or None."""
    r = (orch.read_manifest(case_id).get("oct_params") or {}).get("crop_region")
    if not isinstance(r, dict):
        return None
    lat = r.get("lateral") or []
    frames = r.get("frames") or []
    if len(lat) != 2 or not frames:
        return None
    try:
        lo, hi = sorted((int(lat[0]), int(lat[1])))
        fs = sorted({int(f) for f in frames})
    except (ValueError, TypeError):
        return None
    return {"lateral": [lo, hi], "frames": fs}


def _case_valid_mask(case_id: str):
    """#9 crop-aware analytics — a bool validity volume in the case's labelmap grid: True everywhere except
    the cropped region. The saved labelmap axis order is (lateral=axis0, depth=axis1, frames=axis2), so a BOX
    crop {lateral:[lo,hi], frames:[…]} zeros valid[lo:hi+1, :, f]; the LEGACY full-slice crop zeros
    valid[lat, :, :]. None if nothing cropped (callers treat None as all-valid). Used by compare-strategies."""
    import numpy as np
    region = _case_crop_region(case_id)
    legacy = _case_crop_lateral(case_id)
    if region is None and not legacy:
        return None
    arr, _ = labels.best_labelmap_nnunet(case_id)
    if arr is None:
        return None
    n_lat, _depth, n_fr = arr.shape          # saved labelmap = (lateral, depth, frames)
    valid = np.ones(arr.shape, bool)
    if region is not None:
        lo = max(0, min(n_lat - 1, region["lateral"][0]))
        hi = max(0, min(n_lat - 1, region["lateral"][1]))
        for f in region["frames"]:
            if 0 <= f < n_fr:
                valid[lo:hi + 1, :, f] = False
    if legacy:
        idx = [i for i in legacy if 0 <= i < n_lat]
        if idx:
            valid[idx, :, :] = False
    return valid


def _eye_replicates(case_id: str) -> tuple[list[str], dict]:
    """SEGMENTED replicates of this scan's eye + scar-subgroup (same patient_id + eye + scar_subgroup, a cornea
    labelmap present, not a consensus case). Returns (member_ids incl. case_id first, {patient,eye,subgroup})."""
    pid, eye, sub = _case_identity(case_id)
    members: list[str] = []
    if pid and eye and settings.CASES_ROOT.exists():
        for d in sorted(settings.CASES_ROOT.iterdir()):
            if not d.is_dir():
                continue
            cid = d.name
            if orch.read_manifest(cid).get("consensus_cases"):
                continue
            if _case_identity(cid) != (pid, eye, sub):
                continue
            arr, _ = labels.best_labelmap_nnunet(cid)
            if arr is not None:
                members.append(cid)
    if case_id not in members:
        members.insert(0, case_id)
    else:                                            # keep the active scan first (preferred consensus reference)
        members = [case_id] + [c for c in members if c != case_id]
    return members, {"patient": pid, "eye": eye, "subgroup": sub}


def _eye_all_segmented(case_id: str) -> tuple[list[str], dict]:
    """ALL cornea-segmented SCAR scans of this scan's eye (same patient + eye, ANY subgroup, not a consensus
    case, NOT a control) — the candidate pool for AUTOMATIC subgroup assignment, which DECIDES the subgroups
    and so must not pre-filter by the current subgroup. CONTROLS are excluded: they carry no scar, so they'd
    cluster as meaningless empty singletons (subgrouping is about lesions). Active scan first (overlay ref)."""
    pid, eye, _sub = _case_identity(case_id)
    members: list[str] = []
    if pid and eye and settings.CASES_ROOT.exists():
        for d in sorted(settings.CASES_ROOT.iterdir()):
            if not d.is_dir():
                continue
            cid = d.name
            mm = orch.read_manifest(cid)
            if mm.get("consensus_cases"):
                continue
            if str(mm.get("scar_classification") or "").strip().lower() == "control":
                continue
            p2, e2, _ = _case_identity(cid)
            if (p2, e2) != (pid, eye):
                continue
            arr, _ = labels.best_labelmap_nnunet(cid)
            if arr is not None:
                members.append(cid)
    if case_id not in members:
        members.insert(0, case_id)
    else:
        members = [case_id] + [c for c in members if c != case_id]
    return members, {"patient": pid, "eye": eye}


@app.post("/api/case/{case_id}/subgroup/auto")
def subgroup_auto(case_id: str, req: SubgroupAutoRequest | None = None) -> dict:
    """AUTO-ASSIGN subgroups for this scan's eye by PURE bright-spot (hysteresis scar) alignment: cluster the
    eye's cornea-segmented scans so the SAME lesion's replicates group together and a different/displaced lesion
    splits off (subgroup.auto_subgroups), plus an en-face OVERLAY (coloured by proposed subgroup) to verify.
    READ-ONLY — proposes only; the user applies via /subgroup/auto/apply. CPU (SimpleITK), no GPU."""
    import subgroup as sg
    members, key = _eye_all_segmented(case_id)
    if len(members) < 2:
        raise HTTPException(400, f"Need ≥2 cornea-segmented scans of this eye to auto-assign subgroups (found "
                                 f"{len(members)}). Run SAM2 cornea on the eye's other repeats first.")
    try:
        res = sg.auto_subgroups(members, req.params if req else None)   # includes the cornea-aligned en-face overlay
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    res["patient"] = key["patient"]; res["eye"] = key["eye"]
    return res


@app.post("/api/case/{case_id}/subgroup/auto/apply")
def subgroup_auto_apply(case_id: str, req: SubgroupApplyRequest) -> dict:
    """Persist the user-verified auto subgroup labels: write scar_subgroup for each scan in `assignments`
    ({case_id: label}). Optionally confirm the ACTIVE scan's subgroup so the timeline advances. The members
    are all the same eye (from /subgroup/auto), so this only relabels that eye's scans."""
    # SAFETY: only relabel scans that ARE this eye's auto-subgroup members — never write to arbitrary cases an
    # untrusted/stale `assignments` map might name (the only mutating path in the feature).
    allowed, _key = _eye_all_segmented(case_id)
    allowed_set = set(allowed)
    written, rejected = {}, []
    for cid, lab in (req.assignments or {}).items():
        c = orch.safe_case_id(cid)
        if c not in allowed_set or not orch.case_root(c).exists():
            rejected.append(cid)
            continue
        sub = str(lab).strip() or "1"
        upd = {"scar_subgroup": sub}
        # The user verified the WHOLE grouping, so confirm each member. Subgroup is now assigned BEFORE scar
        # (cornea✓ → subgroup → scar → align), so confirming does NOT need scar_done — it advances each member
        # to the Scar step. (Align stays after scar and re-segments any member missing scar via _ensure_segmented.)
        if req.confirm:
            upd["subgroup_confirmed"] = True
        orch.write_manifest_value(c, upd)
        written[c] = sub
    return {"ok": True, "written": written, "rejected": rejected, "confirmed": bool(req.confirm)}


@app.post("/api/case/{case_id}/align-replicates")
def align_replicates(case_id: str) -> dict:
    """STEP 7 — ALIGN this eye+subgroup's segmented replicates into one consensus using their scar AS-IS
    (no control-normalisation here). Register + vote the repeats; the per-scan members are linked to the
    consensus. Control-normalisation is a SEPARATE later step (normalize-consensus), run once enough
    control scans exist. Returns the consensus case."""
    members, key = _eye_replicates(case_id)
    if len(members) < 2:
        raise HTTPException(400, f"Need ≥2 segmented replicate scans of this eye+subgroup to align (found {len(members)}). "
                                 "Run SAM2 on the eye's other repeat scans (same subgroup) first.")
    try:
        ccid, report = _build_consensus_case(members, subgroup=key["subgroup"])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"consensus_case": ccid, "replicates": members, "n_replicates": len(members),
            "subgroup": key["subgroup"], "report": report,
            "images": orch.preview_images_from_dir("Segmentation", _preview_group_dir(ccid, "segmentation"))}


@app.post("/api/case/{case_id}/normalize-consensus")
def normalize_consensus(case_id: str) -> dict:
    """STEP 8 — NORMALISE an aligned consensus against the control (no-scar) baseline: build the control
    reflectivity atlas, re-derive each member's scar as EXCESS over the normal profile (depthnorm,
    reproducible) replacing the absolute-threshold scar, then REBUILD the consensus and mark it normalised.
    `case_id` is the consensus case (or any member — we resolve its consensus). Needs control scans."""
    cid = orch.safe_case_id(case_id)
    m = orch.read_manifest(cid)
    # Resolve the consensus case: this IS one (consensus_cases), or a member linking to one.
    ccid = cid if m.get("consensus_cases") else (m.get("consensus_case") or "")
    if not ccid or not orch.read_manifest(ccid).get("consensus_cases"):
        raise HTTPException(400, "No aligned consensus for this scan yet — align the replicates first.")
    members = list(orch.read_manifest(ccid).get("consensus_cases") or [])
    if not normal_baseline.control_cases():
        raise HTTPException(400, "No control (no-scar) scans tagged yet — tag + segment some controls, then normalise.")
    try:
        n_controls = int(normal_baseline.build_profile().get("n_controls", 0))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Could not build the control baseline: {exc}")
    for mc in members:                                 # re-derive each member's scar control-normalised
        try:
            scar_auto(mc, ScarAutoRequest(method="depthnorm", replace=True))
            # If the member's labelmap had been overwritten by an earlier "Use consensus (all)" apply,
            # scar_auto has now REPLACED it with the depthnorm scar — so clear the now-stale flags
            # (corrected_labelmap made scanStep read it as step 11 "Manually corrected"; consensus_scar_source
            # no longer reflects the on-disk labelmap). Without this a normalized member falsely shows as
            # corrected with a consensus source it no longer carries.
            orch.write_manifest_value(mc, {"corrected_labelmap": None, "consensus_scar_source": None})
        except Exception as exc:  # noqa: BLE001
            print(f"[normalize] depthnorm scar skipped for {mc}: {exc}", file=sys.stderr)
    sub = orch.read_manifest(ccid).get("scar_subgroup")
    try:
        ccid2, report = _build_consensus_case(members, subgroup=sub)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    orch.write_manifest_value(ccid2, {"normalized": True, "n_controls": n_controls})
    return {"consensus_case": ccid2, "normalized": True, "n_controls": n_controls,
            "n_replicates": len(members), "report": report,
            "images": orch.preview_images_from_dir("Segmentation", _preview_group_dir(ccid2, "segmentation"))}


@app.post("/api/case/{case_id}/skip-normalization")
def skip_normalization(case_id: str) -> dict:
    """STEP 9 — skip control-normalisation: keep the aligned consensus AS-IS (no depthnorm re-derivation) and
    advance the timeline so it can be corrected / scheduled. Records normalization_skipped so the export can
    note this consensus was NOT control-normalised."""
    cid = orch.safe_case_id(case_id)
    m = orch.read_manifest(cid)
    ccid = cid if m.get("consensus_cases") else (m.get("consensus_case") or "")
    if not ccid or not orch.read_manifest(ccid).get("consensus_cases"):
        raise HTTPException(400, "No aligned consensus for this scan yet — align the replicates first.")
    mm = orch.write_manifest_value(ccid, {"normalized": True, "normalization_skipped": True})
    return {"ok": True, "consensus_case": ccid, "normalized": bool(mm.get("normalized")), "normalization_skipped": True}


class CompareStrategiesRequest(BaseModel):
    strategies: List[str] | None = None    # None = all production strategies
    phi_percentile: float = 92.0           # benchmark-validated operating point


class SubgroupAutoRequest(BaseModel):
    params: dict | None = None             # optional subgroup.DEFAULT overrides (tolerances/threshold)


class SubgroupApplyRequest(BaseModel):
    assignments: dict                      # {case_id: subgroup_label} to persist (the user-verified grouping)
    confirm: bool | None = None            # also confirm the active scan's subgroup (advance the timeline)


@app.post("/api/case/{case_id}/compare-strategies")
def compare_strategies(case_id: str, req: CompareStrategiesRequest) -> dict:
    """PUBLICATION: test–retest reproducibility of each scar strategy on this eye+subgroup's segmented
    replicates — pairwise 3D Dice, pairwise HD95 (mm), native scar-volume mean / CV% / repeatability
    coefficient. READ-ONLY: scar masks are computed in memory and the canonical labelmaps are untouched.
    Resolve the replicate set (a member or the consensus case), then run scar_bench.compare_strategies."""
    import scar_bench
    cid = orch.safe_case_id(case_id)
    m = orch.read_manifest(cid)
    if m.get("consensus_cases"):                       # a consensus case → use its members directly
        members = list(m.get("consensus_cases") or [])
        key = {"subgroup": str(m.get("scar_subgroup") or "1")}
    else:
        members, key = _eye_replicates(cid)
    if len(members) < 2:
        raise HTTPException(400, f"Need ≥2 segmented replicate scans of this eye+subgroup to compare "
                                 f"reproducibility (found {len(members)}).")

    # Injected SAM2-scar (deep-learning) mask per replicate so the comparison includes SAM2 too — computed
    # READ-ONLY (mirrors _scar_auto_sam2_locked WITHOUT writing the canonical labelmap): auto-seed the
    # brightest in-cornea tissue, run the 3-view SAM2 consensus under the GPU lock, constrain to bright +
    # coherent components. Returns a native scar mask in the scan's grid; raises → that strategy row errors.
    import nibabel as nib
    from scipy import ndimage as _ndi

    def _sam2_scar_mask(mc: str):
        import sam2_segment
        arr, _ = labels.best_labelmap_nnunet(mc)
        if arr is None:
            raise ValueError("no cornea segmentation")
        base = _ensure_volume_nifti(mc)
        vol = np.asarray(nib.load(str(_working_volume(mc))).dataobj).astype(np.float32)
        seeds, bright = scar_mod.auto_scar_seeds(vol, arr, percentile=float(req.phi_percentile),
                                                 erode_surface=6, smooth=2.5, max_seeds=5)
        if not seeds:
            return np.zeros(arr.shape, bool)
        with _GPU_LOCK:
            fused, _meta = sam2_segment.segment_scar_consensus(base, arr, seeds, orch.case_root(mc) / "sam2_work", vote=2)
        scar_c = fused & bright
        lbl, n = _ndi.label(scar_c)
        if n:
            sizes = _ndi.sum(np.ones_like(lbl), lbl, range(1, n + 1))
            scar_c = np.isin(lbl, [i + 1 for i, s in enumerate(sizes) if s >= 200])
        return scar_c & ((arr == 1) | (arr == 2))

    # #15 cooperative cancel: a concurrent /compare-strategies/cancel sets the flag for this case; the
    # bench loop checks it between strategies AND replicates, so Cancel actually stops the (slow, SAM2)
    # run rather than letting it grind on in the background. Clear any stale flag before starting.
    _COMPARE_CANCEL.discard(cid)
    # #9 crop-aware: pass each replicate's validity mask so cropped lateral bands are excluded from the
    # common comparison region (a cropped replicate has no data there — comparing the full volume would bias
    # the metric). Cheap to build; None per case when nothing was cropped.
    valid_masks = {mc: _case_valid_mask(mc) for mc in members}
    try:
        result = scar_bench.compare_strategies(members, req.strategies, req.phi_percentile,
                                               sam2_scar_fn=_sam2_scar_mask,
                                               should_cancel=lambda: cid in _COMPARE_CANCEL,
                                               valid_masks=valid_masks)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    finally:
        _COMPARE_CANCEL.discard(cid)
    result["subgroup"] = key.get("subgroup")
    return result


@app.post("/api/case/{case_id}/compare-strategies/cancel")
def compare_strategies_cancel(case_id: str) -> dict:
    """#15 — request the in-flight compare-strategies run for this case to STOP. The running endpoint
    (a separate threadpool thread) polls this flag between strategies/replicates and returns early. A
    SAM2 step already in progress finishes (no mid-kernel interrupt), then no further work is done."""
    _COMPARE_CANCEL.add(orch.safe_case_id(case_id))
    return {"ok": True}


@app.post("/api/case/{case_id}/subgroup/confirm")
def confirm_subgroup(case_id: str) -> dict:
    """Confirm this scan's scar-subgroup (already set via /subgroup): which lesion set it belongs to, so the
    right repeats align together. Sets subgroup_confirmed so the timeline advances Cornea✓ → Subgroup → Scar
    (subgroup is assigned BEFORE scar so the strategy comparison at the Scar step is per-subgroup)."""
    cid = _require_case(case_id)
    sub = str(orch.read_manifest(cid).get("scar_subgroup") or "1").strip() or "1"
    m = orch.write_manifest_value(cid, {"scar_subgroup": sub, "subgroup_confirmed": True})
    return {"ok": True, "scar_subgroup": m.get("scar_subgroup"), "subgroup_confirmed": True}


@app.post("/api/case/{case_id}/scar/skip")
def skip_scar(case_id: str) -> dict:
    """For a CONTROL (no-scar) scan: mark the scar step done WITHOUT running a detector (there is no scar to
    segment). Controls are an eye-wide normal baseline with no lesion subgroup, so they skip the Subgroup step
    and go Cornea✓ → (no scar) → align/correct."""
    m = orch.write_manifest_value(_require_case(case_id), {"scar_done": True})
    return {"ok": True, "scar_done": bool(m.get("scar_done"))}


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
    skip_previews: bool | None = None   # batch/populate: skip the eager DENSE context-slice render (they
                                        #   render lazily via /context-previews on open) — big populate speedup
    classification: str | None = None   # "scar" | "control" (no scar) | None
    scar_range: List[int] | None = None  # [start_frame, end_frame], 1-based
    patient: str | None = None  # user-corrected identity (group header edit); overrides filename parse
    eye: str | None = None      # "OD"/"OS"; "?" / blank are ignored
    force_columns: List[int] | None = None  # BAD frame indices to re-correct ("re-run preprocessing")
    good_columns: List[int] | None = None    # GOOD/anchor frame indices guiding the re-correction
                                             # (all reuse the scan's persisted settings)
    # Reviewer's manual POSTERIOR (bottom) edge points, {slice: {frame: depth}} — the same shape as
    # border_anchors. Supplied live while dragging so the preview reflects the line being drawn; persisted via
    # oct-marks and thereafter read from oct_params.
    crop_post_anchors: dict | None = None
    surface_crop_frames: List[int] | None = None  # "Detect surface crop" Confirm: B-scan columns whose apex is
                                             # cropped (no top surface). A STICKY oct_param; on re-run those
                                             # frames are reconstructed by posterior continuity (bottom-edge
                                             # guidance). None = carry persisted set; [] = clear the crop.
    surface_crop_mode: str | None = None     # USER CONTROL over the surface-crop correction (v0.0.211):
                                             #   "auto"   — let the detector decide (the historical default)
                                             #   "manual" — use EXACTLY surface_crop_frames, detector disabled
                                             #   "off"    — never surface-crop this scan, even if auto-detected
                                             # None = leave the persisted mode alone. Needed because clearing the
                                             # frame set used to just POP the param, so the next run re-detected
                                             # and re-applied the crop — there was no way to say "no".
    crop_lateral: List[int] | None = None    # LEGACY #9 v1 full-slice crop (kept for old cases).
    crop_region: dict | None = None          # #9 v2 "Crop": a BOX = {'lateral':[lo,hi], 'frames':[…]} — remove
                                             # certain FRAME columns over a RANGE of LATERAL slices (zeroed before
                                             # SAM2). A STICKY oct_param recorded so scar-alignment excludes the
                                             # lost box. None = carry persisted; {} or empty frames = clear the crop.
    crop_bands: dict | None = None           # #9 v3 per-lateral ARTIFACT band {'str(lateral)':[lo,hi]} — a
                                             # time-domain artifact whose frame extent VARIES per slice; marked on
                                             # several slices, interpolated across laterals, EXCLUDED from the fit +
                                             # zeroed before SAM2. Sticky. None = carry persisted; {} = clear.
    max_iterations: int | None = None        # >1 = iterative refinement (auto-converge); 1 = single faithful pass
    inject_pass: int | None = None           # re-run iteration applying force_columns at ONLY this pass (1-based)
    manual_patch: dict | None = None         # reviewer-accepted RIGID patch: {frame_index: [depth_px, tilt_px]}.
                                             # Sticky like manual_shifts and applied after it. Carries a ROTATION
                                             # term, which manual_shifts cannot express and which is the defect
                                             # class review most often finds; stored as params so a re-preprocess
                                             # re-applies it instead of erasing it, and so the accumulated set
                                             # doubles as the corpus the auto passes must learn to reproduce.
    manual_shifts: dict | None = None        # #2 drag-to-correct: {frame_index: depth_px} manual per-frame
                                             # depth nudges (positive = DOWN), applied LAST as manual ground truth
    slice_index: int | None = None           # steps viewer: which sagittal slice to render the border+fit on
    border_pass: int | None = None            # border-curve: which pass to fix — detect on its INPUT (raw for
                                              # pass 1, the prior pass's output for pass>1), never the result
    border_anchors: dict | None = None        # fix-columns "Confirm": {str(slice_index): {str(frame): true_depth}}
                                              # corrected ABSOLUTE surface depths (depth 0 = TOP). The server MARCHES
                                              # a tilt-aware re-detection of the whole RAW volume seeded by these.
    axial_anchors: dict | None = None         # AXIAL fix-tool "Confirm": {str(frame): {str(lateral): true_depth}}
    corrected_edge_anchors: dict | None = None  # CORRECTED-result sagittal fix-tool: {str(lateral): {str(frame): corrected_depth}}
                                              # anterior-surface depths (CORRECTED-output depth space, 0 = TOP) drawn on
                                              # an axial B-scan across laterals. STICKY like manual_shifts: applied as a
                                              # post-hoc additive per-frame warp (apply_axial_surface_gt); {} clears.
    corrected_smooth_align: bool = False      # corrected pane "use the detected edge": smooth-fit the trusted detected
                                              # surface across frames and rigidly align each B-scan onto it. ONE-SHOT
                                              # (never persisted to oct_params), like apply_proposals — the reviewer
                                              # re-invokes it per re-run; not a sticky GT (they chose align, not lock).
    corrected_edit_feedback: bool = False     # CORRECTED-pane edits are the more accurate observation (reviewer
                                              # directive 2026-09-01: "corrections on the corrected image are always
                                              # to be treated as more accurate"), so fold them BACK into the ORIGINAL
                                              # scan's border_anchors and re-run from the improved GT, instead of the
                                              # post-hoc warp on the finished volume. ONE-SHOT like corrected_smooth_align
                                              # — never persisted, because it REWRITES the reviewer's raw GT and must be
                                              # an explicit act each time. A backup is written before it fires.
    corrected_trusted_laterals: list | None = None  # smooth-align: ARRAY laterals (== oct-corrected-curve slice_index)
                                              # whose detected border the reviewer marked GOOD; the smooth-fit reference
                                              # is built from ONLY these. Empty/None → central-lateral fit. One-shot.
    use_redetect: bool | None = None          # oct-preprocess: flatten to the confirmed re-detected surface
                                              # (provided_edges) instead of auto-detecting — the fix-columns "Run".
    # WHICH slices are an exact fitted curve, when the payload mixes both kinds. `parabola` alone could only
    # say "all or nothing", so a reviewer who shaped a curve on one slice and dragged edge points on another
    # had to lose one of the two — the client resolved that by sending only whatever tool was selected.
    parabola_slices: list | None = None
    parabola: bool | None = None              # fix-columns "Confirm" parabola mode: the anchors are a DENSE fitted
                                              # quadratic → use it EXACTLY (seed window 0), don't re-snap per frame.
    concurrency: int | None = None            # batch preprocess: how many scans the caller runs AT ONCE → each
                                              # scan uses (cpu-2)//concurrency workers (avoids oversubscription).
    ascan_rate_hz: float | None = None        # eye-motion tab: A-scan (line) rate → frame rate → Hz axis (Avanti ~70000)
    detrend_order: int | None = None          # eye-motion tab: per-A-line shape-removal polynomial order (default 2)
    sinc_correct: bool | None = None          # eye-motion tab: divide out the intra-frame motion-blur boxcar
    want_conf: bool | None = None             # oct-border-curves-all: also return a per-(slice,frame) confidence
                                              # map (surface_confidence_map) over the SERVED edge, so the fix-columns
                                              # editor can flag the LOW-confidence stretches that still need marking.
    clear_all_corrections: bool | None = None  # "Clear all corrections & re-preprocess": drop EVERY sticky manual
                                              # correction (border/corrected-edge/axial anchors, crop bands, surface
                                              # crop, marks, force/good columns, manual patch/shifts) from oct_params
                                              # before this AUTO run, so the scan returns to a pure fresh-detect state.
                                              # Detection config (dp_*), classification, review flags + training GT are
                                              # kept. See _CORRECTION_PARAM_FIELDS.


# EVERY per-scan oct_params field that stores a MANUAL reviewer correction/mark. "Clear all corrections" pops all of
# these so the scan is re-processed as pure AUTO. Deliberately EXCLUDES: dp_*/detect config, oct_max_iterations,
# auto_tune (tuning, not a correction); scar_classification/scar_range/subgroup (not corrections); and the training
# corpus GT (manifest.border_gt) + review metadata (review_flags/difficult_scan), which are not this scan's live edits.
_CORRECTION_PARAM_FIELDS = (
    "border_anchors", "border_generalize", "border_guided", "parabola", "parabola_slices",  # anterior edge / quadratic
    "manual_shifts", "manual_patch",                                                          # legacy per-frame nudges
    "corrected_edge_anchors", "corrected_trusted_laterals", "edit_transform",                 # corrected-result edge / trusted / transform
    "axial_anchors",                                                                          # axial fix-tool
    "crop_bands", "crop_region", "crop_lateral", "crop_post_anchors",                          # artifact / box / bottom-line crops
    "surface_crop_frames", "surface_crop_mode", "auto_surface_crop",                          # surface crop (manual + auto flag)
    "surface_cut", "zero_cols",                                                                # cut / zeroed frames
    "force_columns", "good_columns",                                                           # fix-columns force / good
    "detilt",                                                                                  # applied de-tilt
    "redetect_seed_window", "redetect_seed_window_slices",                                     # seed windows tied to anchors
)


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
        "oct_spacing": spacing, **extra,
    })  # NB: oct_preprocessed is set LAST (below), only after volume.nii.gz is fully written — see there.
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
    # Set the pipeline-gate flag LAST — only now that previews/volume.nii.gz is fully written (base, above).
    # The interactive app is sequential so this ordering is invisible to it, but the streaming populate polls
    # oct_preprocessed from a separate process to decide a scan is ready for SAM2; setting it before the volume
    # write finished let a SAM2 worker read a half-written/empty volume.nii.gz ("Compressed file ended").
    orch.write_manifest_value(case_id, {"oct_preprocessed": preprocessed})
    # The gallery pulls slices lazily via /previews + /preview-file, so don't base64 the (now
    # DENSE) context group into this response — it would inline tens of MB the frontend ignores.
    return {"case_info": orch.current_case_info(case_id), "spacing": spacing,
            "geometry_warnings": geom_warnings, "images": []}


def _write_crop_shade_display(case_id: str, oct_src: str, vi: int,
                              crop_region: dict | None, crop_lateral: list | None,
                              dim: float = 0.35) -> None:
    """CROP-SHADE: write previews/volume_display.nii.gz = the corrected volume with the CROPPED region filled by
    the RAW (uncorrected) tissue at `dim`× intensity, so the viewer (get_volume_nifti) shows what was cropped in
    a distinct dim shade. The pipeline/SAM2 volume (previews/volume.nii.gz) is left untouched. No-op (display
    removed) when there's no crop or the canvas was extended (surface-crop) so the shapes can't align."""
    import nibabel as nib
    disp_path = orch.case_root(case_id) / "previews" / "volume_display.nii.gz"
    frames = [int(f) for f in ((crop_region or {}).get("frames") or [])]
    lat = (crop_region or {}).get("lateral") or [0, -1]
    legacy = [int(l) for l in (crop_lateral or [])]
    if not frames and not legacy:
        disp_path.unlink(missing_ok=True)
        return
    try:
        cimg = nib.load(str(_working_volume(case_id)))
        corr = np.asanyarray(cimg.dataobj)                       # (lateral, depth, frame)
        raw = np.transpose(oct_mod.read_oct_zstack(oct_src, vi), (2, 1, 0)).astype(np.float32)  # → (lat, depth, frame)
        if corr.shape != raw.shape:                              # extended canvas → can't align; no crop-shade
            disp_path.unlink(missing_ok=True)
            return
        L = corr.shape[0]
        out = corr.astype(np.float32).copy()
        lo = max(0, int(lat[0])); hi = min(L - 1, int(lat[1]) if int(lat[1]) >= 0 else L - 1)
        for f in frames:
            if 0 <= f < corr.shape[2]:
                out[lo:hi + 1, :, f] = raw[lo:hi + 1, :, f] * dim
        for l in legacy:
            if 0 <= l < L:
                out[l, :, :] = raw[l, :, :] * dim
        if np.issubdtype(corr.dtype, np.integer):
            out = np.rint(out).astype(corr.dtype)
        else:
            out = out.astype(corr.dtype)
        nib.save(nib.Nifti1Image(out, cimg.affine, cimg.header), str(disp_path))
    except Exception:  # noqa: BLE001 — display convenience; never block preprocessing
        disp_path.unlink(missing_ok=True)


@app.post("/api/oct/upload")
async def oct_upload(files: List[UploadFile] = File(...)) -> dict:
    """Upload .OCT files (+ optional companion .txt). One case per .OCT; metadata is
    parsed from the filename + companion. No conversion yet — fast for whole directories."""
    if not files:
        raise HTTPException(400, "No files uploaded.")
    _check_upload_count(files)
    blobs = []
    request_total = 0
    for up in files:
        data = await _read_upload_bytes(up)
        request_total += len(data)
        if request_total > _MAX_REQUEST_BYTES:
            raise HTTPException(413, f"Upload request exceeds the maximum total size ({_MAX_REQUEST_BYTES} bytes).")
        blobs.append((up.filename or "", data))
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
    out = _oct_render_volume(case_id, work, preprocessed=show_corrected, extra={"oct_volume_index": vi},
                             render_previews=not bool(req.skip_previews))
    out["n_frames"] = _nifti_frames(work)
    out["preprocessed"] = show_corrected
    return out


# ── ONE PREPROCESSING RUN PER SCAN AT A TIME ──────────────────────────────────────────────────────────────
# The re-run button's only guard was React state (`busy`), which lives in one page. Reload the page mid-run and
# the UI forgets a run is going; open a second client and it never knew. Either way a second POST starts while
# the first is still writing, and BOTH runs write previews/volume.nii.gz, the border caches and manifest.json
# for the same case — a data race on the reviewer's real data whose loser is silently interleaved, not
# detected. Observed on case_cs048_od_v1: two full runs back to back, outputs 14:38:33 and 14:40:27.
#
# The lock is per case (two different scans may legitimately run at once) and non-blocking: a second request is
# REFUSED with 409 rather than queued, because queueing would silently do the work twice — the caller wants to
# know its click did nothing. Same idiom the cohort endpoints already use.
_CASE_RUN_LOCKS: dict = {}
_CASE_RUN_GUARD = threading.Lock()


def _case_run_lock(case_id: str) -> threading.Lock:
    with _CASE_RUN_GUARD:
        lk = _CASE_RUN_LOCKS.get(case_id)
        if lk is None:
            lk = _CASE_RUN_LOCKS[case_id] = threading.Lock()
        return lk


def case_run_in_progress(case_id: str) -> bool:
    return _case_run_lock(case_id).locked()


@app.get("/api/case/{case_id}/oct-preprocess-running")
def oct_preprocess_running(case_id: str) -> dict:
    """Is a preprocessing run in flight for this scan? Lets a page that was reloaded mid-run restore its
    busy state instead of offering a button that would be refused (or, before the lock, would double-run)."""
    return {"running": case_run_in_progress(case_id)}


@app.post("/api/case/{case_id}/oct-preprocess")
def oct_preprocess_case(case_id: str, req: OctPreprocessRequest) -> dict:
    """Preprocess this scan, refusing to start a second concurrent run on the same case (409)."""
    lk = _case_run_lock(case_id)
    if not lk.acquire(blocking=False):
        raise HTTPException(409, "A preprocessing run is already in progress for this scan.")
    try:
        return _oct_preprocess_case_impl(case_id, req)
    finally:
        lk.release()


def _oct_preprocess_case_impl(case_id: str, req: OctPreprocessRequest) -> dict:
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
    # CLEAR ALL CORRECTIONS: drop every sticky manual correction so this becomes a pure fresh AUTO run. Do it FIRST,
    # before any correction is read back below (the plain-auto path already pops border_anchors + rmtrees border_cache,
    # but it KEEPS corrected_edge_anchors / crop_bands / axial_anchors / surface-crop / force-columns sticky — so those
    # must be popped explicitly here). Also force the auto branch (no re-detect / no smooth-align) and drop the derived
    # caches so nothing stale is re-served. defect_marks (top-level) is already wiped by every preprocess run.
    if req.clear_all_corrections:
        for _cf in _CORRECTION_PARAM_FIELDS:
            eff_params.pop(_cf, None)
        req.use_redetect = False
        req.corrected_smooth_align = False
        req.corrected_edge_anchors = None
        req.corrected_trusted_laterals = None
        req.axial_anchors = None
        req.crop_bands = None
        _AXIAL_SURF_CACHE.pop(case_id, None)
        try:
            import shutil as _sh_clr
            _sh_clr.rmtree(orch.case_root(case_id) / "border_cache", ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass
        # surface_crop_manual is a TOP-LEVEL review mark (not oct_params, steers nothing) — clear it too so the
        # scan reads as never-reviewed. Not in `extra` below, so this write survives the end-of-run manifest merge.
        try:
            orch.write_manifest_value(case_id, {"surface_crop_manual": None})
        except Exception:  # noqa: BLE001
            pass
    # apply_proposals is a ONE-SHOT approve action (bake in the detected de-tilt/crop/surface-crop), NOT a sticky
    # oct_param — pop it so it isn't persisted (a later fresh auto-preprocess proposes again for re-review); it's
    # re-injected only into the worker params for this run.
    _apply_prop = bool(eff_params.pop("apply_proposals", False))
    # Auto-tune OFF must mean the FIXED DEFAULTS: drop any dp_* the auto-tuner persisted on a prior run, so the
    # warp falls back to DEFAULT_PARAMS instead of silently freezing the last auto-tuned surface (review MEDIUM).
    if not eff_params.get("auto_tune", True):
        for _k in ("dp_sigma_depth", "dp_sigma_frame", "dp_below", "dp_max_jump"):
            eff_params.pop(_k, None)

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
    # "Detect surface crop" Confirm: the user-verified set of surface-cropped frames is a STICKY oct_param
    # (carried through the eff_params merge above on later re-runs, so the crop correction persists like a
    # geometric property of the scan). When THIS request supplies it (non-None), REPLACE the persisted set;
    # an empty list clears the crop. The worker (preprocess_oct_to_nifti) reconstructs these frames by
    # posterior continuity via the provided_edges path.
    if req.surface_crop_frames is not None:
        scf = sorted(set(_int_list(req.surface_crop_frames)))
        if scf:
            eff_params["surface_crop_frames"] = scf
        else:
            eff_params.pop("surface_crop_frames", None)
    # SURFACE-CROP MODE (v0.0.211) — the user's explicit decision, sticky like the frame set itself.
    # Previously the only controls were "here is a frame set" or "clear it", and clearing merely POPPED the
    # param, so the very next run re-ran the detector and could re-apply the same crop the user had just
    # removed. The A/B showed why that matters: the correction genuinely rescues some scans (case_cs024_od_v3
    # delivered dev 7.52 -> 0.89, i.e. from worst-in-store to the accepted median) and genuinely harms others
    # (case_cs008_od_v1 4.05 -> 4.96), so neither always-on nor always-off is right — it has to be per scan.
    #   auto   = detector decides (historical behaviour)
    #   manual = use exactly surface_crop_frames; the detector must not add to or override the user's set
    #   off    = never surface-crop this scan
    # Encoding note: the worker reads oct_params.auto_surface_crop and oct_params.surface_crop_frames.
    # `_crop_frames = params.get("surface_crop_frames")` is checked for `is None` BEFORE the auto detector
    # runs, so an explicit EMPTY LIST both suppresses auto-detection and (being falsy) applies no repair —
    # that is what makes "off" durable rather than advisory.
    apply_surface_crop_mode(eff_params, req.surface_crop_mode)
    # #9 "Crop" — STICKY box crop (carried through the eff_params merge on later re-runs, like a geometric
    # property of the scan). crop_region = {'lateral':[lo,hi], 'frames':[…]} removes those frame columns over
    # the lateral-slice range (zeroed before SAM2); recorded so the compare/subgroup analytics exclude the box
    # (crop-aware). When THIS request supplies crop_region (non-None), REPLACE it; empty/no frames clears it AND
    # the legacy full-slice crop. The legacy crop_lateral path is kept only for old cases.
    if req.crop_region is not None:
        cr = req.crop_region if isinstance(req.crop_region, dict) else {}
        lat = cr.get("lateral") or []
        frames = sorted({int(f) for f in (cr.get("frames") or [])})
        if len(lat) == 2 and frames:
            eff_params["crop_region"] = {"lateral": [int(lat[0]), int(lat[1])], "frames": frames}
            eff_params.pop("crop_lateral", None)   # a VALID box crop supersedes the legacy full-slice crop
        else:
            eff_params.pop("crop_region", None)    # explicit clear (empty frames) — don't touch legacy here
    elif req.crop_lateral is not None:         # legacy full-slice crop (old clients)
        cl = sorted(set(_int_list(req.crop_lateral)))
        if cl:
            eff_params["crop_lateral"] = cl
        else:
            eff_params.pop("crop_lateral", None)
    # #9 v3 ARTIFACT BANDS — STICKY per-lateral artifact crop, its OWN independent `if` (NOT part of the
    # crop_region/crop_lateral if/elif chain above — placed after it so the legacy elif still binds to crop_region).
    # REPLACE when this request supplies crop_bands (non-None); empty dict clears. Interpolated across laterals +
    # excluded from the fit at run time (oct_preprocess._artifact_bands).
    if req.crop_bands is not None:
        cb: dict = {}
        for lk, band in (req.crop_bands if isinstance(req.crop_bands, dict) else {}).items():
            if not isinstance(band, (list, tuple)) or len(band) != 2:
                continue
            try:
                cb[str(int(lk))] = [int(band[0]), int(band[1])]
            except (TypeError, ValueError):
                continue
        if cb:
            eff_params["crop_bands"] = cb
        else:
            eff_params.pop("crop_bands", None)
    eff_params.pop("coronal_check", None)    # removed feature — strip any stale persisted flag
    eff_params.pop("manual_columns", None)   # removed feature — strip any stale persisted nudges
    # surface_cut (fix-columns "Re-run with cuts") is a PER-RUN override like force_columns, NOT a sticky
    # param: strip any persisted one unless THIS request supplies it, so a normal auto preprocess can't
    # silently re-apply a stale cut (which would exclude columns + leave them unwarped → degraded labels).
    if not (req.params and "surface_cut" in req.params):
        eff_params.pop("surface_cut", None)
    # #2 drag-to-correct: explicit per-frame manual depth nudges. When provided (non-None), REPLACE the
    # persisted set (an empty {} clears them); when omitted, the persisted nudges carry through so manual
    # ground truth stays applied on every later re-run. Flows to the worker inside eff_params (--params).
    if req.manual_shifts is not None:
        eff_params["manual_shifts"] = req.manual_shifts
    if req.manual_patch is not None:                       # a request set REPLACES; omitted carries the persisted set
        eff_params["manual_patch"] = req.manual_patch
    # AXIAL fix-tool GT (sticky, like manual_shifts): a request set REPLACES; omitted carries the persisted set
    # through (merged from oct_params above) so the axial correction re-applies on every re-run. Never popped by the
    # normal-auto supersede / use_redetect blocks below → it composes with (survives) a sagittal fix-columns Run.
    if req.axial_anchors is not None:
        eff_params["axial_anchors"] = req.axial_anchors
    # CORRECTED-result sagittal fix-tool GT (sticky, same rules as axial_anchors): a post-hoc per-frame rigid
    # warp applied to the finished corrected volume, so it survives use_redetect (composes with the sagittal
    # raw-GT Run rather than being superseded). Request set REPLACES; omitted carries the persisted set through.
    if req.corrected_edge_anchors is not None:
        eff_params["corrected_edge_anchors"] = req.corrected_edge_anchors
    # CORRECTED-RESULT EDIT FEEDBACK (reviewer directive): fold edits/approvals made on the CORRECTED result BACK
    # into the ORIGINAL scan's border_anchors GT, then re-run the warp from the improved GT — instead of the old
    # post-hoc rigid warp on the already-corrected volume (align_corrected_to_smooth / apply_sagittal_surface_gt).
    # "alter the corrections to the original scan such that a better corrected result is formed, not correcting the
    # corrected scan itself." Folds only on a corrections re-run; a strict no-op if there are no corrected edits.
    # NOTE: pure GT-fold-back is OFF by default (corrected_edit_feedback). Verified it does NOT work on its own:
    # a corrected-result edit folds into border_anchors correctly, but the RIGID per-frame warp (which fits ONE
    # motion across ALL laterals) averages a local few-lateral GT edit away (measured: -30px edit → ~0.5px in the
    # result). Residual corrected-result error is inter-frame MOTION, not a surface-position error, so it needs a
    # per-frame rigid transform fit from the trusted laterals (align_corrected_to_smooth), not a GT edit. Kept
    # gated for the record; the default path runs align_corrected_to_smooth (which now composes with the fixed
    # rotation warp). See notes to the reviewer.
    _folded_corrected = False
    if ((req.use_redetect or req.corrected_smooth_align)
            and (req.corrected_edit_feedback or bool(eff_params.get("corrected_edit_feedback")))):
        try:
            # corrected_edit_mode="transform" (default, reviewer 2026-09-04): the corrected-pane edits MODIFY THE
            # TRANSFORM — one rigid shift+tilt per frame fitted to the edit deltas, accumulated on the case and
            # applied to the original as the last rigid move of every run. "fold" = the 2026-09-02 surface-GT
            # mechanism (write the depths into border_anchors), measured to change the per-frame move by ~0 px at
            # 8 edited slices because they are ~6% of the flatten's joint fit.
            _mode = str(eff_params.get("corrected_edit_mode",
                                       oct_mod.DEFAULT_PARAMS.get("corrected_edit_mode", "transform"))).lower()
            if _mode == "fold":
                _folded_corrected = _fold_corrected_edits_into_border_anchors(
                    case_id, m, eff_params, req.corrected_trusted_laterals)
            else:
                # transform mode: the TRANSFORM carries the correction; the fold still records the accurate lines
                # into the raw GT so the served surface (and the pane line drawn from it) is right where they drew.
                # The transform is fitted from the pane edits BEFORE the fold clears them.
                _cea_keep = dict(eff_params.get("corrected_edge_anchors") or {})
                _tr = _fit_corrected_edits_into_transform(case_id, m, eff_params, req.corrected_trusted_laterals)
                if _tr:
                    eff_params["corrected_edge_anchors"] = _cea_keep
                    try:
                        _fd = _fold_corrected_edits_into_border_anchors(case_id, m, eff_params, req.corrected_trusted_laterals)
                    except Exception as _fe:  # noqa: BLE001
                        print(f"[corrected-edit] fold after transform failed: {type(_fe).__name__}", file=sys.stderr); _fd = None
                    eff_params.pop("corrected_edge_anchors", None); eff_params.pop("corrected_accurate", None)
                    eff_params["edit_transform"] = (m.get("oct_params") or {}).get("edit_transform") or eff_params.get("edit_transform")
                    if isinstance(_fd, dict):
                        _tr["fold"] = {k: _fd.get(k) for k in ("anchors_before", "anchors_after", "generalized", "backup")}
                _folded_corrected = _tr
        except Exception as _fexc:  # noqa: BLE001 — never fail the re-run over the fold-back; fall through to a plain re-run
            print(f"[corrected-edit] fold/transform failed for {case_id}: {type(_fexc).__name__}: {_fexc}", file=sys.stderr)
            _folded_corrected = False
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
    _det_rep: dict | None = None      # determinism report; stays None on an AUTO run (no anchors → no prompt)
    # Smooth-align ("use the detected edge") re-runs to reproduce the corrected result and iron out its
    # undulation. It does NOT depend on a re-detected surface: on an AUTO-only scan there are no border anchors
    # to flatten to, so fall back to a normal auto preprocess (+ the align at the end) instead of erroring.
    _want_redetect = bool(req.use_redetect) and bool((m.get("oct_params") or {}).get("border_anchors"))
    # use_redetect with NO confirmed border anchors: there is no re-detected surface to flatten to. Rather than
    # hard-erroring, fall back to a normal AUTO preprocess whenever there is still work it WILL do — a smooth-align,
    # or a STICKY CROP to apply (surface_crop / crop_region / crop_bands / legacy crop_lateral). The auto pass
    # re-detects and applies those (e.g. crop_bands excludes the artifact from the fit + zeroes it before SAM2), so
    # "Re-run with corrections" works after marking a crop even with no border edit. Only error when there is
    # genuinely nothing to do.
    _has_sticky_crop = bool(eff_params.get("crop_bands") or eff_params.get("crop_region")
                            or eff_params.get("surface_crop_frames") or eff_params.get("crop_lateral"))
    if req.use_redetect and not _want_redetect and not req.corrected_smooth_align and not _has_sticky_crop:
        raise HTTPException(400, "No confirmed border anchors to apply — drag the border and Confirm first.")
    if _want_redetect:
        anchors = (m.get("oct_params") or {}).get("border_anchors") or {}
        # ensure a FRESH cache for the persisted anchors (recompute if missing/stale incl. an algorithm
        # upgrade), then feed it to the worker — same surface the scrub display uses (preview == result).
        # _redetect_surface_cached returns WHICHEVER surface the scan carries (guided > generalize > local
        # redetect) — the exact one the scrub display draws. The warp must flatten to THAT, not always to
        # redetect.npz: on a border_guided scan those differ, so the old code showed one edge and warped to
        # another. Persist the returned surface to its own npz so the worker flattens to what the reviewer saw.
        surf_for_warp = _redetect_surface_cached(case_id, m, anchors)
        if surf_for_warp is None:
            raise HTTPException(400, "No re-detected surface to apply — drag the border and Confirm first.")
        # LIMBUS/edge-band smoothing of the warp TARGET so the corrected tissue ascends smoothly onto the faint
        # limbus (the reviewer's "not smooth at the right end" — the flatten reproduces the target's edge roughness).
        # Gentle-average (no anchor re-pin); gated by edge_band_smooth_frames; corrections path only.
        surf_for_warp = oct_mod.smooth_surface_edge_band(surf_for_warp, {**oct_mod.DEFAULT_PARAMS, **eff_params})
        pe_path = orch.case_root(case_id) / "border_cache" / "provided_edges.npz"
        pe_path.parent.mkdir(parents=True, exist_ok=True)
        _pe_tmp = pe_path.with_name("provided_edges.tmp.npz")   # MUST end .npz (savez appends it otherwise)
        np.savez_compressed(_pe_tmp, surface=np.asarray(surf_for_warp, dtype=np.float32))
        os.replace(_pe_tmp, pe_path)
        redetect_npz = pe_path
        # DETERMINISM REPORT (reviewer request 2026-08-25: "point the user towards" the edges still worth
        # drawing "after running"). Measured on surf_for_warp — literally the surface this run warps to — so
        # it describes the DELIVERED result rather than whatever is live in the editor. Advisory telemetry:
        # it rides in oct_iter beside final_qa, gates nothing and never touches a voxel. Swallowed on failure.
        try:
            # SAME params object _compute_redetect_cache / _compute_generalize_cache just used, so the
            # baseline is a warm cache hit (a params_sig mismatch here would silently cost a ~25 s full
            # re-detect for a purely advisory number).
            _det_p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
            _det_base = _baseline_surface(case_id, _load_border_vol(_ensure_raw_border_nifti(case_id)), _det_p)
            _det_rep = oct_mod.determinism_report(anchors, _det_base, surf_for_warp,
                                                  {**_det_p, **{k: v for k, v in eff_params.items()
                                                                if str(k).startswith("determinism_")}})
        except Exception:  # noqa: BLE001 — advisory; never fail a preprocessing run over it
            _det_rep = None
        eff_params["border_anchors"] = anchors        # keep them persisted on the case
        # the re-detect warp flattens to EXACTLY the previewed surface — legacy per-frame manual_shifts (which
        # the scrub preview does NOT show) would break preview==result, so they're superseded here.
        eff_params.pop("manual_shifts", None)
    else:
        # a NORMAL auto preprocess SUPERSEDES any prior manual re-detection: drop the persisted anchors +
        # the cached surface so a later fix-columns scrub/Run can't show/apply a stale re-detected border.
        eff_params.pop("border_anchors", None)
        eff_params.pop("border_generalize", None)   # whole-volume-generalize flag (its cache is in border_cache)
        # ...and the per-lateral artifact bands. "↻ Re-preprocess" means "start this scan again from the raw
        # .OCT", and the reviewer reasonably expects their ⊟ Crop artifact marks to go with the rest of the
        # manual state (2026-09-02). They were being kept as "sticky geometry", which left a re-preprocessed
        # scan still carrying bands the reviewer thought they had cleared — and crop_bands interpolate across
        # the FULL width, so a stale one is not a local leftover.
        eff_params.pop("crop_bands", None)
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
    # Scan-level concurrency: when the caller runs K scans at once (req.concurrency), give each scan
    # (cpu-2)//K workers so K scans × that ≈ all cores (no oversubscription) and the serial phases of one
    # scan overlap the parallel phases of another → fuller CPU use than one-scan-at-a-time. K<=1 → auto (full).
    _k = max(1, int(req.concurrency or 1))
    if _k > 1:
        _w = max(2, _cpu_budget() // _k)   # K scans × _w ≈ all cores → full CPU, no oversubscription
        extra += ["--workers", str(_w)]
    if redetect_npz is not None:
        extra += ["--provided-edges", str(redetect_npz)]   # flatten to the confirmed re-detected surface
    elif inject_pass is not None:
        extra += ["--inject-pass", str(int(inject_pass)),
                  "--inject-force", json.dumps(_int_list(req.force_columns)),
                  "--inject-good", json.dumps(_int_list(req.good_columns))]
    # ONE-SHOT worker flags — passed to THIS run only, never merged into eff_params (which is what gets
    # persisted to oct_params), so they don't stick like a sticky correction. apply_proposals bakes the auto
    # crop; corrected_smooth_align is the corrected pane's "use the detected edge" smooth-fit rigid align.
    _wparams = dict(eff_params)
    if _apply_prop:
        _wparams["apply_proposals"] = True
    if req.corrected_smooth_align and not _folded_corrected:
        # Only the OLD post-hoc smooth-align path (warp the corrected volume). Skipped when the edits were folded
        # back into the raw GT above — that re-runs the warp from the improved GT instead (reviewer's spec).
        _wparams["corrected_smooth_align"] = True
        if req.corrected_trusted_laterals:
            _wparams["corrected_trusted_laterals"] = [int(v) for v in req.corrected_trusted_laterals
                                                      if isinstance(v, (int, float))]
    worker_out = _run_oct_worker("preprocess", src, work, _wparams, vi,
                                 companion=m.get("companion_txt"), extra=extra)
    iter_info = _parse_iter_info(worker_out)
    # Ride the determinism report (measured above, on the surface this run warps to) into oct_iter, so it is
    # persisted by the manifest write below and survives a reload / sidecar restart. AUTO runs never compute
    # it (there are no anchors), so the key is simply absent there — the prompt is structurally off.
    if isinstance(iter_info, dict) and isinstance(_det_rep, dict) and _det_rep:
        iter_info["determinism"] = _det_rep
    # Same for the corrected-edit FOLD: it rewrote the raw GT before the worker ran, so record what it did
    # (points folded, laterals, backup file) where a reload can still read it.
    if isinstance(iter_info, dict) and isinstance(_folded_corrected, dict):
        iter_info["corrected_fold"] = _folded_corrected
    # NATIVE AUTO-TUNE: the worker tuned the DP detector to this scan; persist the chosen dp_* into the case's
    # oct_params so the fix-columns baseline + steps re-detect with the SAME params the warp used (preview ==
    # result). The cache params_sig includes the dp_* keys, so the surface caches recompute accordingly.
    _tuned = (iter_info.get("auto_tune") or {}).get("params") if isinstance(iter_info, dict) else None
    if isinstance(_tuned, dict) and _tuned:
        eff_params.update({k: v for k, v in _tuned.items() if k in
                           ("dp_sigma_depth", "dp_sigma_frame", "dp_below", "dp_max_jump")})
    # AUTO crop-region: the worker auto-detected + zeroed off-cornea NOISE frames. PERSIST the box into
    # oct_params so it is STICKY — re-applied on every later run (incl. a fix-columns redetect, which would
    # otherwise resurrect the noise) and visible to scar-exclusion / consensus / the crop UI. A manual crop in
    # THIS request still wins (the worker only auto-detects when no manual crop is set).
    _acr = iter_info.get("auto_crop_region") if isinstance(iter_info, dict) else None
    if isinstance(_acr, dict) and _acr.get("frames") and not eff_params.get("crop_region") \
            and not eff_params.get("crop_lateral"):
        eff_params["crop_region"] = {"lateral": [int(_acr["lateral"][0]), int(_acr["lateral"][1])],
                                     "frames": sorted(int(f) for f in _acr["frames"]), "auto": True}
    # STALE-CROP RECONCILE: the worker's crop guard refused to destructively zero CLIPPED-APEX frames (routing
    # them to surface-crop reconstruction instead). Mirror that in the PERSISTED crop_region so a stale/
    # mis-classified crop_region doesn't linger in the manifest — otherwise it is re-applied AND painted red by
    # the crop-shade on every later run. Drop exactly the frames the guard reported removing; clear the box if
    # it becomes empty (an entirely-clipped-apex "crop" that is really a surface-crop).
    _guard_removed = set(int(f) for f in (iter_info.get("crop_guard_removed_frames") or [])) \
        if isinstance(iter_info, dict) else set()
    if _guard_removed and isinstance(eff_params.get("crop_region"), dict):
        _cr0 = eff_params["crop_region"]
        _kept0 = [int(f) for f in (_cr0.get("frames") or []) if int(f) not in _guard_removed]
        if _kept0:
            eff_params["crop_region"] = {**_cr0, "frames": _kept0}
        else:
            eff_params.pop("crop_region", None)
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
             # Crop-approval workflow: the auto de-tilt / crop-region / surface-crop the preprocessing DETECTED
             # but did NOT apply (unless apply_proposals). The UI shows these (pink + glowing fix-cols/crop
             # buttons) so the user reviews/approves them at the vetting step. null/empty → nothing proposed.
             "oct_proposals": (iter_info.get("proposals") if isinstance(iter_info, dict) else None),
             # a fresh preprocessing (auto OR a Fix-columns re-run) invalidates the manual-vetting and
             # training-schedule flags → the per-scan timeline drops back to "Preprocessed [Auto]" (red)
             # and the user re-approves. scar_classification is kept (it's scan content, not geometry).
             "preproc_vetted": False, "training_scheduled": False,
             # GT confirmation follows preproc_vetted: a fresh output must be re-approved before its (sticky)
             # corrected border re-enters the GT corpus. The anchor points persist in oct_params; re-approval
             # re-confirms them (with the current anchors signature).
             "border_gt": None,
             # CYBERNETIC-LOOP step-3 RESET: the just-produced volume is a NEW algorithm's output, so the user's
             # defect_marks (which point at the OLD output's wrong columns) are stale → clear them; the user
             # re-inspects the fresh output and re-marks anything still wrong (step 4b). difficult_scan is NOT
             # cleared here — a "too damaged for auto" judgement is stable across algorithm iterations (persists;
             # the batch can still grow, and the user un-flags it manually if a later algorithm fixes it).
             "defect_marks": [],
             # The segmentation files were just deleted above; CLEAR their manifest flags too, else
             # scanStep (which keys off sam2_meta/corrected_labelmap/consensus_case BEFORE preproc_vetted)
             # would keep reporting the scan as segmented while its overlay 404s. (Mirrors _STEP_RESET_FLAGS.)
             # subgroup_confirmed is cleared as well: a leftover would make the re-segmented scan jump straight
             # to the Subgroup step (subgroup is now before scar), skipping the cornea/background vet step.
             "sam2_meta": None, "corrected_labelmap": None, "consensus_case": None, "scar_done": None,
             "cornea_vetted": None, "subgroup_confirmed": None,
             "qa_json": None, "segmentation_preview_dir": None}
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
    # CROP-SHADE: build the dim-crop DISPLAY volume so the viewer shows the cropped-out (uncorrected) tissue in
    # a distinct shade instead of black (best-effort; never blocks preprocess). Uses the manual crop or the
    # auto-detected off-cornea/blink crop; a no-op (display cleared) when nothing was cropped.
    _write_crop_shade_display(
        case_id, str(src), vi,
        (eff_params.get("crop_region") or (iter_info.get("auto_crop_region") if isinstance(iter_info, dict) else None)),
        eff_params.get("crop_lateral"))
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
             # NO LONGER AUTO-VETTED (reviewer: "Use original still requires a manual approval"). Choosing to
             # keep the raw volume says the CORRECTION was wrong, which is not the same as saying the raw one
             # is good — and this silently marked the scan approved, so a scan could reach the training set
             # without anyone judging it. It now lands at Preprocessed and waits for Approve like everything
             # else. preproc_vetted is explicitly cleared rather than left alone, so a previously-vetted scan
             # does not keep an approval that referred to the discarded correction.
             "preproc_vetted": False, "training_scheduled": False,
             # seg files were deleted above → clear their flags so the timeline drops to Vetted (not SAM2).
             "sam2_meta": None, "corrected_labelmap": None, "consensus_case": None, "scar_done": None, "cornea_vetted": None,
             "qa_json": None, "segmentation_preview_dir": None}
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
    if cls != "scar":
        # A scar frame-range is meaningless once the scan is a control (or untagged) — clear it so a
        # stale range left from an earlier "scar" tag can't confine a later detection. (Mirrors the
        # frontend's intent of sending scar_range:null on demotion, which the conditional below would
        # otherwise ignore.)
        updates["scar_range"] = None
    elif req.scar_range is not None:
        updates["scar_range"] = [int(x) for x in req.scar_range] or None
    m = orch.write_manifest_value(_require_case(case_id), updates)
    return {"ok": True, "scar_classification": m.get("scar_classification"),
            "scar_range": m.get("scar_range")}


def _border_anchor_stats(m: dict) -> tuple[int, int, str]:
    """(n_slices, n_points, signature) of the manual border anchors persisted in this case's oct_params.
    The signature is a stable hash of the {slice:{frame:depth}} anchor set, so the GT confirmation can be
    detected as STALE the moment the user re-corrects (new anchors → new sig → needs re-approval)."""
    anc = ((m.get("oct_params") or {}).get("border_anchors")) or {}
    ns = 0; npts = 0; parts: list[str] = []
    for sl in sorted(anc.keys(), key=lambda k: str(k)):
        fr = anc.get(sl)
        if isinstance(fr, dict) and fr:
            ns += 1; npts += len(fr)
            inner = ",".join(f"{f}:{round(float(fr[f]), 1)}" for f in sorted(fr.keys(), key=lambda k: str(k)))
            parts.append(f"{sl}={inner}")
    import hashlib
    sig = hashlib.sha1(";".join(parts).encode()).hexdigest()[:16] if parts else ""
    return ns, npts, sig


class VetRequest(BaseModel):
    corpus_eligible: bool = True   # include this scan's corrected borders in the GT corpus that tunes the detector


@app.post("/api/case/{case_id}/vet-preprocessing")
def vet_preprocessing(case_id: str, req: VetRequest | None = None) -> dict:
    """Timeline step 3: mark the preprocessing as MANUALLY VETTED (the user reviewed before/after +
    Fix-columns and approves it). Manifest metadata only — turns the scan entry orange and is the gate
    before scar/control classification. A later auto/Fix-columns re-run clears this (see oct-preprocess).

    GT CAPTURE: if the user manually corrected the border (oct_params.border_anchors present), approving also
    records those anchor points as CONFIRMED GROUND TRUTH (manifest.border_gt), so they (a) keep being applied
    and (b) feed the GT-vs-auto corpus that measures + improves the auto-detector (see /api/gt-corpus). The
    anchor points ARE the user's true depths — pure ground truth, independent of the auto-detector. Pass
    corpus_eligible=False to keep a scan's correction applied but EXCLUDE an idealised case (e.g. a
    motion-corrupted scan whose hand-drawn border is not real geometry) from the global-tuning corpus."""
    cid = _require_case(case_id)
    corpus_eligible = bool(req.corpus_eligible) if req is not None else True
    ns, npts, sig = _border_anchor_stats(orch.read_manifest(cid))
    updates: dict = {"preproc_vetted": True}
    updates["border_gt"] = ({"confirmed": True, "corpus_eligible": corpus_eligible,
                             "n_slices": ns, "n_points": npts, "anchors_sig": sig,
                             "ts": round(time.time(), 1)} if npts > 0 else None)
    m = orch.write_manifest_value(cid, updates)
    return {"ok": True, "preproc_vetted": bool(m.get("preproc_vetted")), "border_gt": m.get("border_gt")}


class ReviewFlagRequest(BaseModel):
    flags: list[str] = []


# Review-flag vocabulary: an OPEN set of slugs, not a closed allowlist — a new flag needs no backend
# change, only an entry in the frontend's table (src/api/reviewFlags.ts) for its label/colour.
_REVIEW_FLAG_RE = re.compile(r"^[a-z][a-z0-9-]{1,31}$")
_MAX_REVIEW_FLAGS = 12


@app.post("/api/case/{case_id}/review-flag")
def set_review_flags(case_id: str, req: ReviewFlagRequest) -> dict:
    """Reviewer ISSUE FLAGS marking a scan for attention — verbose slugs naming the actual defect
    ("zeroed-frames", "still-clipped", "crop-clamped", "residual-motion", "jagged-surface", "weak-edge").
    Manifest-only metadata (review_flags = deduplicated, sorted list of slugs matching
    ^[a-z][a-z0-9-]{1,31}$, capped at 12/scan); anything else is silently dropped. The assistant and the
    sidebar filter find flagged scans by reading manifest.review_flags; the frontend renders an unknown
    slug verbatim, so the vocabulary can grow here first. NOTE the pattern rejects the retired
    single-letter cybernetic-loop flags (A/B/C), so a scan still carrying them cannot be re-written
    without being renamed to the new vocabulary. Does not touch the volume, segmentation or lifecycle."""
    slugs = {str(f).strip().lower() for f in (req.flags or [])}
    flags = sorted(s for s in slugs if _REVIEW_FLAG_RE.match(s))[:_MAX_REVIEW_FLAGS]
    m = orch.write_manifest_value(_require_case(case_id), {"review_flags": flags})
    return {"ok": True, "review_flags": m.get("review_flags", [])}


class DefectMark(BaseModel):
    orient: str                       # "sagittal" | "axial" (the shown single-plane view)
    slice: int                        # slice index within that view
    cols: list[int] = []              # column voxel indices (non-depth in-plane axis) that are WRONG
    tag: str | None = None            # defect TYPE: "edge_detection" | "curvature" | "surface_roughness" | free-text


class DefectMarksRequest(BaseModel):
    marks: list[DefectMark] = []


@app.post("/api/case/{case_id}/defect-marks")
def set_defect_marks(case_id: str, req: DefectMarksRequest) -> dict:
    """Precise DEFECT LOCATIONS the user marked live in the viewer: for a sagittal/axial slice, which COLUMNS
    (frame indices for sagittal, lateral indices for axial) are wrong. Persisted to manifest.defect_marks (a
    list of {orient, slice, cols}) so the assistant reads exactly which frames/columns to fix. Manifest-only
    metadata — does not touch the volume, segmentation or lifecycle. Mirrors the review-flag pattern."""
    marks: list[dict] = []
    for mk in (req.marks or []):
        orient = str(mk.orient).strip().lower()
        if orient not in ("sagittal", "axial"):
            continue
        cols = sorted({int(c) for c in (mk.cols or []) if int(c) >= 0})
        if not cols:
            continue
        entry = {"orient": orient, "slice": int(mk.slice), "cols": cols}
        tag = (mk.tag or "").strip()
        if tag:
            entry["tag"] = tag[:60]                       # defect TYPE the user tagged (bounded free-text)
        marks.append(entry)
    m = orch.write_manifest_value(_require_case(case_id), {"defect_marks": marks})
    return {"ok": True, "defect_marks": m.get("defect_marks", [])}


class DifficultRequest(BaseModel):
    difficult: bool = True
    # WHY the scan was rejected, in the reviewer's own words ("the places I marked are where the surface is
    # too wavy"). The flag alone says a scan is unusable but not what to fix, and by the time the algorithm
    # work happens the reason has been lost — so it is captured at the moment of rejection, next to the marks
    # that motivated it. Cleared whenever the flag is cleared, so a stale reason can never outlive its flag.
    reason: str | None = None


@app.post("/api/case/{case_id}/difficult")
def set_difficult_scan(case_id: str, req: DifficultRequest) -> dict:
    """Flag this scan as a DIFFICULT SCAN needing manual help (persisted to manifest.difficult_scan). Toggle
    the user sets in the viewer; the assistant reads it to know which scans to hand-correct. Manifest-only.
    Difficult scans are EXCLUDED from nnU-Net training candidate selection (per_scan_segmented_cases).

    An optional free-text `reason` is persisted alongside as manifest.difficult_reason (with the time it was
    given), and is cleared when the flag is cleared.

    GT CAPTURE ON REJECT (mirrors vet_preprocessing). A border correction is ground truth about WHERE THE
    CORNEA IS, and that is true whether or not the resulting volume was good enough to accept. Capturing it
    only on Approve — as this used to — dropped exactly the most informative corrections, because a scan you
    are rejecting is one the detector got wrong, and you would not approve it. The reviewer's own framing for
    this workflow is "correct the detected edge ... this will inform detector", which cannot happen if the
    correction dies with the rejection.
    Recorded with confirmed=False, so the corpus can tell a signed-off scan from one that was corrected and
    still rejected. corpus_eligible is False as well: the anchor POINTS are real geometry, but a scan whose
    output the reviewer refused should not silently become a global-tuning target — /api/gt-corpus can opt
    them in deliberately. An existing CONFIRMED border_gt is never downgraded."""
    difficult = bool(req.difficult)
    values: dict = {"difficult_scan": difficult}
    if not difficult:
        values["difficult_reason"] = None
        values["reviewer_rejected"] = None
        # …and the rejection RECORD with it. border_gt written by a rejection carries rejected=True; leaving it
        # behind means a scan that has been un-rejected (typically because it was reprocessed with its
        # corrections and is going back for a fresh look) still counts in the rejected-GT pool and still reads
        # as judged. A CONFIRMED record is never touched: that one came from an approval, not a rejection, and
        # is the corpus's own provenance.
        _prev_gt = orch.read_manifest(_require_case(case_id)).get("border_gt")
        if isinstance(_prev_gt, dict) and _prev_gt.get("rejected") and not _prev_gt.get("confirmed"):
            values["border_gt"] = None
    else:
        # A REVIEWER REJECTION, stamped explicitly. difficult_scan cannot answer "how many have I rejected":
        # bulk preprocessing sets it too, so every scan awaiting approval already carries it. border_gt only
        # appears when the rejection came with anchors, and difficult_reason only when a note was typed or
        # derived — so a bare "this one is wrong, next" left no trace at all. This is the one marker every
        # rejection writes, which is what a counter has to be built on.
        values["reviewer_rejected"] = {"ts": round(time.time(), 1)}
        _m0 = orch.read_manifest(_require_case(case_id))
        _ns, _npts, _sig = _border_anchor_stats(_m0)
        _prev = _m0.get("border_gt") or {}
        if _npts > 0 and not (isinstance(_prev, dict) and _prev.get("confirmed")):
            values["border_gt"] = {"confirmed": False, "corpus_eligible": False,
                                   "n_slices": _ns, "n_points": _npts, "anchors_sig": _sig,
                                   "rejected": True, "ts": round(time.time(), 1)}
        text = (req.reason or "").strip()
        if text:
            # `ts` as an epoch float, matching the defect-marks record just above — the module has no
            # datetime import and one timestamp idiom per file is worth more than a prettier string.
            values["difficult_reason"] = {"text": text[:2000], "ts": round(time.time(), 1)}
    m = orch.write_manifest_value(_require_case(case_id), values)
    return {"ok": True, "difficult_scan": bool(m.get("difficult_scan")),
            "difficult_reason": m.get("difficult_reason"),
            "border_gt": m.get("border_gt")}


def apply_surface_crop_mode(eff_params: dict, mode: str | None) -> dict:
    """Apply the user's surface-crop DECISION to a case's effective oct_params, in place.

    "auto"   — hand the decision back to the detector (clear both overrides)
    "manual" — use EXACTLY eff_params["surface_crop_frames"]; the detector must not add to it
    "off"    — never surface-crop this scan
    An unrecognised / None mode leaves the persisted decision untouched.

    Why a mode and not just a frame set: clearing the set used to POP the param, so the next run
    re-ran the detector and could re-apply the very crop the user had just removed — there was no way
    to say "no". The A/B that motivated this showed the correction genuinely rescues some scans
    (case_cs024_od_v3 delivered dev 7.52 -> 0.89, worst-in-store to the accepted median) and genuinely
    harms others (case_cs008_od_v1 4.05 -> 4.96), so it has to be decidable per scan.

    ENCODING (relied on by oct_preprocess): the worker reads `surface_crop_frames` and checks it for
    `is None` BEFORE running the auto detector, then applies the repair only `if _crop_frames:`. So an
    explicit EMPTY LIST both suppresses auto-detection and applies nothing — that is what makes "off"
    durable rather than advisory. Do not "tidy" the empty list away."""
    m = (mode or "").strip().lower()
    if m not in ("auto", "manual", "off"):
        return eff_params
    eff_params["surface_crop_mode"] = m          # recorded so the UI can show the effective state
    if m == "auto":
        eff_params.pop("auto_surface_crop", None)
        eff_params.pop("surface_crop_frames", None)
    elif m == "off":
        eff_params["auto_surface_crop"] = False
        eff_params["surface_crop_frames"] = []
    else:                                         # manual — the user's set is the whole truth
        eff_params["auto_surface_crop"] = False
        if not eff_params.get("surface_crop_frames"):
            # "manual" with no frames would otherwise hand control back to the detector; that is "off".
            eff_params["surface_crop_frames"] = []
    return eff_params


class SurfaceCropRequest(BaseModel):
    surface_crop: bool | None = True   # True = confirm/mark surface-crop, False = mark NOT, None = clear the review


@app.post("/api/case/{case_id}/surface-crop")
def set_surface_crop_manual(case_id: str, req: SurfaceCropRequest) -> dict:
    """MANUALLY mark this scan's surface-crop (clipped-cornea) classification — human review of the pipeline's
    AUTO-detected set (manifest.oct_iter.stopped == "surface_crop"). True = confirm / mark it surface-cropped,
    False = it is NOT (override a false positive), None = clear back to auto-only. Persisted to
    manifest.surface_crop_manual. Manifest-only (does not itself re-preprocess)."""
    m = orch.write_manifest_value(_require_case(case_id), {"surface_crop_manual": req.surface_crop})
    return {"ok": True, "surface_crop_manual": m.get("surface_crop_manual")}


# ── GUARDED REVIEW RE-RUN ────────────────────────────────────────────────────────────────────────────────
# The reviewer works in batches: correct ~10 scans, then re-run preprocessing with those corrections. The
# re-run must be SAFE to fire repeatedly, which means it can only ever improve the store — a scan is
# re-written solely when it measures better, otherwise it keeps exactly the bytes it has. That guard is what
# lets the loop run many times without anyone auditing every pass.
#
# TWO criteria, because one is not enough: across-frame ROUGHNESS is what the rigid passes target and what
# "the surface is not smooth" refers to, while per-frame UNDULATION cannot change under translation+rotation
# at all — so if it moves, something is wrong regardless of how good the roughness looks. A change that
# improves roughness while worsening undulation is refused (that combination is exactly what the
# surface_crop_derotate investigation turned up).
_REPROC_JOB: dict = {"running": False, "done": 0, "total": 0, "written": 0, "kept": 0, "failed": 0,
                     "started": None, "cases": []}
_REPROC_LOCK = threading.Lock()

# ── GLOBAL DETECTOR TUNING ────────────────────────────────────────────────────────────────────────────────
# The step that makes the review loop converge. Re-running corrected scans fixes those scans; this changes the
# DETECTOR, so the 300-odd scans nobody has corrected get better too — which is the only way the queue can
# ever empty. See detector_tune for the objective (hinge on a tolerance band) and the no-regression guard.
_TUNE_LOCK = threading.Lock()
# The tuning pass runs as a SUBPROCESS, never a thread here. detect_surface_all forks a worker pool per
# detection, and _map_slices states the rule plainly: that is safe only outside this process. Run in a sidecar
# thread it deadlocked — children inherit locks held by threads that do not exist in the fork, so they wedge,
# the parent waits forever, and cancellation cannot land because nothing raises. Files carry state across the
# boundary: a status file for progress, a sentinel file for cancel, and the process group for a hard stop.
_TUNE_PROC: "subprocess.Popen | None" = None
_TUNE_PATHS: dict = {}          # {dir, job, status, cancel}
_TUNE_LAST: dict = {"running": False, "phase": "idle", "note": "", "adopted": None}
# Detector settings live PER SCAN in oct_params as well as globally, and the per-scan copy wins. 306 of 308
# scans carry these four, so adopting a global change without clearing them would be a no-op everywhere —
# the tuner would report success and nothing would behave differently.
_TUNED_KEYS = tuple(k for k, _v in detector_tune.SEARCH_SPACE)


def _param_overrides_path() -> Path:
    """Where the globally tuned detector lives. Beside the case store, not in the app bundle, so an app
    update cannot silently revert the detector the reviewer tuned."""
    return Path(settings.CASES_ROOT).parent / "detector_overrides.json"


def _apply_param_overrides_at_startup() -> dict:
    """Publish the override path into the environment and fold the file into DEFAULT_PARAMS.

    The environment part is what reaches the CLI SUBPROCESS path (_run_oct_worker inherits os.environ), so
    both ways of running the pipeline read the same detector. Without it, an in-process re-run and a
    subprocess re-run of the same scan would use different parameters."""
    os.environ["CORNEA_PARAM_OVERRIDES"] = str(_param_overrides_path())
    return oct_mod.load_param_overrides()


def _tune_corpus_and_guard(guard_limit: int = 8) -> tuple[list[dict], list[dict]]:
    """Split the store into what the tuner LEARNS from and what it is FORBIDDEN to disturb.

    corpus = every scan carrying corrections, approved or rejected. Pressing the button is the reviewer's
    deliberate opt-in, which is exactly the consent that kept rejected corrections out of the passive corpus.
    guard  = approved scans, sampled evenly across the store so the check is not all one patient."""
    corpus: list[dict] = []
    approved: list[dict] = []
    root = settings.CASES_ROOT
    for child in sorted(root.iterdir()) if root.exists() else []:
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        try:
            m = orch.read_manifest(child.name)
        except Exception:  # noqa: BLE001
            continue
        src = m.get("oct_source")
        if not src or not Path(str(src)).exists():
            continue
        # EYE identity, so the held-out split can keep every replicate of one eye on the SAME side. A scan
        # is not "unseen" if four other scans of the same eye were trained on — the search would have fitted
        # that cornea already, and the held-out number would report generalisation it has not demonstrated.
        # patient_id/eye are USUALLY ABSENT from the manifest — the sidebar fills them by parsing the source
        # filename — so reading them directly gave every scan the same key "none|none", collapsing the corpus
        # to a single eye and silently disabling the split for good. Parse first, then fall back to the case
        # id with its replicate suffix stripped (case_cs002_os_v3 -> case_cs002_os), which encodes the same
        # identity and cannot be missing.
        _pid, _eye = m.get("patient_id"), m.get("eye")
        if not (_pid and _eye):
            try:
                _meta = metrics_export.parse_case_meta(str(src))
                _pid = _pid or _meta.get("patient_id"); _eye = _eye or _meta.get("eye")
            except Exception:  # noqa: BLE001
                pass
        _grp = (f"{_pid}|{_eye}".lower() if (_pid and _eye)
                else re.sub(r"_v\d+(?:_\d+)*$", "", child.name).lower())
        entry = {"case_id": child.name, "src": str(src), "group": _grp,
                 "volume_index": int(m.get("oct_volume_index", 0) or 0),
                 "params": dict(m.get("oct_params") or {})}
        pts = detector_tune._anchor_points(m)
        if pts:
            corpus.append({**entry, "pts": pts})
        elif m.get("preproc_vetted"):
            approved.append(entry)
    if len(approved) > guard_limit:                      # even spread, not the first N alphabetically
        step = len(approved) / float(guard_limit)
        approved = [approved[int(i * step)] for i in range(guard_limit)]
    return corpus, approved


def _tune_adopt(params: dict) -> dict:
    """Persist the winning detector and make it actually take effect.

    Three writes, and all three are needed: the file (so subprocesses and restarts see it), the in-memory
    DEFAULT_PARAMS (so the running sidecar does), and the removal of the per-scan copies that would otherwise
    shadow it. UNAPPROVED scans only — an approved scan keeps the exact settings its signed-off output was
    produced under, so re-running it still reproduces what the reviewer accepted."""
    path = _param_overrides_path()
    prev = {}
    try:
        prev = (json.loads(path.read_text(encoding="utf-8")) or {}).get("params") or {}
    except (OSError, ValueError):
        pass
    merged = {**prev, **params}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"params": merged, "ts": round(time.time(), 1)}, indent=2), encoding="utf-8")
    oct_mod.DEFAULT_PARAMS.update(merged)
    cleared = 0
    root = settings.CASES_ROOT
    for child in sorted(root.iterdir()) if root.exists() else []:
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        # READ-MODIFY-WRITE UNDER THE LOCK. oct_params is written back whole, so without this the reviewer
        # committing a border correction in the gap between this read and this write would have it discarded
        # — silently, and on the one scan they are actively working on. This sweep touches every unapproved
        # scan, so it is exactly the background writer that gap exists for.
        with orch.manifest_lock():
            try:
                m = orch.read_manifest(child.name)
            except Exception:  # noqa: BLE001
                continue
            if m.get("preproc_vetted"):
                continue
            op = dict(m.get("oct_params") or {})
            if not any(k in op for k in _TUNED_KEYS):
                continue
            for k in _TUNED_KEYS:
                op.pop(k, None)
            orch.write_manifest_value(child.name, {"oct_params": op})
        cleared += 1
    return {"params": merged, "cleared_scans": cleared}


def _tune_read_status() -> dict:
    """Whatever the child last published. Missing/half-written reads as 'no news', never as failure."""
    path = _TUNE_PATHS.get("status")
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh) or {}
    except (OSError, ValueError):
        return {}


def _tune_watch(proc: "subprocess.Popen") -> None:
    """Wait for the child, then ADOPT IN THIS PROCESS. The adopt writes manifests, and doing that here rather
    than in the child keeps every manifest write behind the sidecar's own lock — the child has no access to
    it, so a child-side write could land on top of a correction the reviewer committed meanwhile."""
    global _TUNE_LAST
    try:
        proc.wait()
    except Exception:  # noqa: BLE001
        pass
    st = _tune_read_status()
    st["running"] = False
    if proc.returncode not in (0, None) and st.get("phase") not in ("done", "cancelled"):
        st.setdefault("phase", "failed")
        st.setdefault("note", f"Tuning process exited with code {proc.returncode}.")
    if st.get("adopted"):
        try:
            st["adopt"] = _tune_adopt(st["adopted"])
        except Exception as exc:  # noqa: BLE001
            st["note"] = f"Tuned, but adopting failed: {exc}"[:300]
            st["adopted"] = None
    _TUNE_LAST = dict(st)
    with _TUNE_LOCK:
        globals()["_TUNE_PROC"] = None


def _reproc_measure(path: str) -> dict:
    """Roughness of the across-frame profile + per-frame undulation, both from the delivered volume."""
    import numpy as np
    import nibabel as nib          # module-local, matching the rest of this file (nibabel is not a top-level import)
    sag = np.ascontiguousarray(np.asanyarray(nib.load(path).dataobj)).astype(np.float32)
    sag = oct_mod._fill_pad_background(sag, 0, 24)     # measurement only; padding must not be scored
    L, _D, F = sag.shape
    S = np.asarray(oct_mod.detect_surface_all(sag, {}, workers=2), float)
    lat = np.arange(L, dtype=float)
    und = []
    for f in range(F):
        y = S[:, f]; g = np.isfinite(y)
        if int(g.sum()) < 200:
            continue
        c0 = np.polyfit(lat[g], y[g], 2); r0 = y[g] - np.polyval(c0, lat[g])
        k = np.abs(r0) <= np.percentile(np.abs(r0), 90)
        und.append(float(np.sqrt(np.mean((y[g] - np.polyval(np.polyfit(lat[g][k], y[g][k], 2), lat[g])) ** 2))))
    prof = np.nanmedian(S[int(0.2 * L):int(0.8 * L)], axis=0)
    gp = np.isfinite(prof)
    return {"rough": float(np.mean(np.abs(np.diff(prof[gp], 2)))) if gp.sum() > 3 else float("nan"),
            "undul": float(np.median(und)) if und else float("nan")}


def _reproc_worker(case_ids: list[str]) -> None:
    import numpy as np
    import shutil as _sh2
    import tempfile
    for cid in case_ids:
        tmp = tempfile.mkdtemp(prefix=f"reproc_{cid}_")
        try:
            m = orch.read_manifest(cid)
            src = m.get("oct_source"); cur = m.get("input_volume")
            if not (src and Path(src).exists() and cur and Path(cur).exists()):
                with _REPROC_LOCK:
                    _REPROC_JOB["failed"] += 1; _REPROC_JOB["done"] += 1
                continue
            params = dict(m.get("oct_params") or {})
            params.pop("apply_proposals", None)
            dst = str(Path(tmp) / "new.nii.gz")
            oct_mod.preprocess_oct_to_nifti(src, dst, params=params,
                                            volume_index=int(m.get("oct_volume_index", 0) or 0),
                                            companion_txt=m.get("companion_txt"), workers=2)
            old, new = _reproc_measure(cur), _reproc_measure(dst)
            rel = (old["rough"] - new["rough"]) / old["rough"] if old["rough"] > 0 else 0.0
            d_und = new["undul"] - old["undul"]
            better = rel >= 0.02 and d_und <= 1.0        # must gain roughness AND not lose per-frame shape
            if better:
                os.replace(dst, cur)                      # same filesystem → atomic
                orch.write_manifest_value(cid, {"review_flags": ["changed"]})
                with _REPROC_LOCK:
                    _REPROC_JOB["written"] += 1
            else:
                with _REPROC_LOCK:
                    _REPROC_JOB["kept"] += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[reprocess-batch] {cid}: {exc}", file=sys.stderr)
            with _REPROC_LOCK:
                _REPROC_JOB["failed"] += 1
        finally:
            _sh2.rmtree(tmp, ignore_errors=True)
            with _REPROC_LOCK:
                _REPROC_JOB["done"] += 1
    with _REPROC_LOCK:
        _REPROC_JOB["running"] = False


@app.post("/api/review/reprocess-batch")
def start_reprocess_batch() -> dict:
    """Re-run preprocessing on the scans the reviewer has CORRECTED but not yet approved, guarded so the
    store can only improve. Returns immediately; poll /api/review/reprocess-status."""
    with _REPROC_LOCK:
        if _REPROC_JOB["running"]:
            return {"ok": True, "already_running": True, **_REPROC_JOB}
    # Candidates: carrying a border correction or crop marks, and NOT already approved. An approved scan is
    # left alone entirely — the reviewer signed off on the bytes it has, and re-running it would put a scan
    # they already judged back into the queue for no reason.
    cands: list[str] = []
    for child in sorted(settings.CASES_ROOT.iterdir()) if settings.CASES_ROOT.exists() else []:
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        try:
            m = orch.read_manifest(child.name)
        except Exception:  # noqa: BLE001
            continue
        if m.get("preproc_vetted"):
            continue
        op = m.get("oct_params") or {}
        if (op.get("border_anchors") or op.get("crop_post_anchors")
                or op.get("surface_crop_frames") or op.get("crop_region") or m.get("border_gt")):
            cands.append(child.name)
    with _REPROC_LOCK:
        _REPROC_JOB.update({"running": bool(cands), "done": 0, "total": len(cands), "written": 0,
                            "kept": 0, "failed": 0, "started": round(time.time(), 1), "cases": cands})
    if cands:
        threading.Thread(target=_reproc_worker, args=(cands,), daemon=True).start()
    return {"ok": True, "started": len(cands), "cases": cands}


@app.post("/api/review/tune-detector")
def start_tune_detector() -> dict:
    """Search the detector's parameters for a setting that reproduces the reviewer's corrections better, and
    adopt it globally if it does — without disturbing the scans they have already approved. Returns at once;
    poll /api/review/tune-status. This is the step that makes the queue converge: it changes the ALGORITHM,
    not just the scans that were corrected."""
    global _TUNE_PROC, _TUNE_PATHS
    with _TUNE_LOCK:
        if _TUNE_PROC is not None and _TUNE_PROC.poll() is None:
            return {"ok": True, "already_running": True, **_tune_read_status()}
        corpus, guard = _tune_corpus_and_guard()
        if not corpus:
            return {"ok": True, "started": 0, "note": "No corrections to learn from yet."}
        d = Path(tempfile.mkdtemp(prefix="cornea_tune_"))
        paths = {"dir": str(d), "job": str(d / "job.json"), "status": str(d / "status.json"),
                 "cancel": str(d / "cancel")}
        Path(paths["job"]).write_text(json.dumps({
            "corpus": corpus, "guard": guard, "workers": max(2, oct_mod.auto_workers() // 2)}), encoding="utf-8")
        Path(paths["status"]).write_text(json.dumps(
            {"running": True, "phase": "starting", "done": 0, "total": 0,
             "n_corpus": len(corpus), "n_guard": len(guard)}), encoding="utf-8")
        # start_new_session so a cancel can reap the whole fork pool, exactly as _run_oct_worker does.
        proc = subprocess.Popen(
            [sys.executable, str(Path(detector_tune.__file__)),
             "--job", paths["job"], "--status", paths["status"], "--cancel", paths["cancel"]],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, start_new_session=True)
        _TUNE_PROC, _TUNE_PATHS = proc, paths
    threading.Thread(target=_tune_watch, args=(proc,), daemon=True).start()
    return {"ok": True, "started": len(corpus), "n_corpus": len(corpus), "n_guard": len(guard),
            "n_points": sum(len(c["pts"]) for c in corpus)}


@app.get("/api/review/tune-status")
def tune_status() -> dict:
    with _TUNE_LOCK:
        live = _TUNE_PROC is not None and _TUNE_PROC.poll() is None
    if live:
        st = _tune_read_status()
        st["running"] = True                 # the file may predate the child's first publish
        return {"ok": True, **st}
    return {"ok": True, **_TUNE_LAST}


@app.post("/api/review/tune-cancel")
def tune_cancel(hard: bool = False) -> dict:
    """Sentinel first, then the process group if it does not take.

    The graceful path only lands at a checkpoint, and a checkpoint can be a whole volume detection away — so
    `hard` exists to kill the group outright. Nothing is written until the very end of a successful run, so a
    hard stop cannot leave the store half-updated."""
    import signal
    with _TUNE_LOCK:
        proc, paths = _TUNE_PROC, dict(_TUNE_PATHS)
    if proc is None or proc.poll() is not None:
        return {"ok": True, "cancelling": False}
    try:
        Path(paths["cancel"]).write_text("1", encoding="utf-8")
    except OSError:
        pass
    if hard:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return {"ok": True, "cancelling": True, "hard": bool(hard)}


@app.get("/api/review/detector-overrides")
def detector_overrides() -> dict:
    """What the detector has been tuned to, if anything — so the UI can show that the algorithm has moved."""
    path = _param_overrides_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {"ok": True, "path": str(path), "params": (data or {}).get("params") or {}, "ts": (data or {}).get("ts")}


@app.get("/api/review/reprocess-status")
def reprocess_status() -> dict:
    with _REPROC_LOCK:
        return {"ok": True, **_REPROC_JOB}


class OctMarksRequest(BaseModel):
    surface_crop_frames: list[int] | None = None      # frames whose apex is above the window
    crop_region: dict | None = None                   # {"lateral":[lo,hi], "frames":[...]}
    crop_post_anchors: dict | None = None             # {slice: {frame: depth}} manual posterior edge
    crop_bands: dict | None = None                    # #9 v3 per-lateral artifact band: {str(lateral):[lo,hi]}
                                                       #   marked on several slices, interpolated across laterals.
                                                       #   REPLACE semantics (full map each commit); {} clears.


@app.post("/api/case/{case_id}/oct-marks")
def set_oct_marks(case_id: str, req: OctMarksRequest) -> dict:
    """Persist the reviewer's CROP MARKS to oct_params WITHOUT re-running the pipeline.

    WHY THIS EXISTS. Marking surface-cropped frames or a crop region used to reach disk only via
    "Confirm & re-run", which re-processes the whole scan (~2 min). That is the right cost when the reviewer
    wants the corrected volume NOW, and entirely the wrong one during a review pass, where the marks are being
    recorded as ground truth and the volume is regenerated later in bulk. Without a cheap path the marks would
    simply be lost when the reviewer moves on, which is worse than the extra click it replaced.

    Manifest-only, and sticky exactly like the same keys set by oct-preprocess: the next re-run picks them up.
    An explicit empty list CLEARS a set (matching apply_surface_crop_mode's "off" semantics, where an empty
    list both suppresses auto-detection and applies no repair); None leaves that key untouched."""
    cid = _require_case(case_id)
    m = orch.read_manifest(cid)
    op = dict(m.get("oct_params") or {})
    changed: dict = {}
    if req.surface_crop_frames is not None:
        scf = sorted({int(f) for f in req.surface_crop_frames})
        if scf:
            op["surface_crop_frames"] = scf
            op["surface_crop_mode"] = "manual"        # the user's set wins over the detector's
        else:
            op.pop("surface_crop_frames", None)
            op["surface_crop_mode"] = "off"
        changed["surface_crop_frames"] = scf
    if req.crop_region is not None:
        cr = req.crop_region or {}
        lat = cr.get("lateral") or []
        frames = sorted({int(f) for f in (cr.get("frames") or [])})
        if len(lat) == 2 and frames:
            op["crop_region"] = {"lateral": [int(lat[0]), int(lat[1])], "frames": frames}
            op.pop("crop_lateral", None)             # a valid box supersedes the legacy full-slice crop
        else:
            op.pop("crop_region", None)
        changed["crop_region"] = op.get("crop_region")
    if req.crop_post_anchors is not None:
        # MERGED per slice, not replaced: the reviewer corrects one slice at a time and a whole-object write
        # would silently drop every other slice's posterior corrections. An empty row clears that slice.
        cur = dict(op.get("crop_post_anchors") or {})
        for sk, row in (req.crop_post_anchors or {}).items():
            if isinstance(row, dict) and row:
                cur[str(sk)] = {str(f): float(d) for f, d in row.items()}
            else:
                cur.pop(str(sk), None)
        if cur:
            op["crop_post_anchors"] = cur
        else:
            op.pop("crop_post_anchors", None)
        changed["crop_post_anchors"] = {k: len(v) for k, v in cur.items()}
    if req.crop_bands is not None:
        # #9 v3 per-lateral artifact band. REPLACE semantics (the frontend holds the full {lateral:[lo,hi]} map):
        # this request fully defines the set; an empty dict CLEARS. Interpolated across laterals at run time.
        cb: dict = {}
        for lk, band in (req.crop_bands or {}).items():
            if not isinstance(band, (list, tuple)) or len(band) != 2:
                continue
            try:
                lat = int(lk); lo, hi = sorted((int(band[0]), int(band[1])))
            except (TypeError, ValueError):
                continue
            cb[str(lat)] = [lo, hi]
        if cb:
            op["crop_bands"] = cb
        else:
            op.pop("crop_bands", None)
        changed["crop_bands"] = op.get("crop_bands")
    if not changed:
        return {"ok": True, "changed": {}}
    orch.write_manifest_value(cid, {"oct_params": op})
    return {"ok": True, "changed": changed}


class SubgroupRequest(BaseModel):
    subgroup: str | None = None   # e.g. "1" (default), "posterior", "inferior"


@app.post("/api/case/{case_id}/subgroup")
def set_case_subgroup(case_id: str, req: SubgroupRequest) -> dict:
    """Persist a scan's scar-subgroup label (a replicate SET within one eye — distinct lesions of the
    same eye that must be voted SEPARATELY, never merged). Without this the loader's per-scan subgroup
    is client-only and lost on reload, silently collapsing distinct lesions into one consensus."""
    sub = (req.subgroup or "1").strip() or "1"
    m = orch.write_manifest_value(_require_case(case_id), {"scar_subgroup": sub})
    return {"ok": True, "scar_subgroup": m.get("scar_subgroup")}


# ── GROUND-TRUTH CORPUS (v0.0.159) ── the user's manually-corrected borders, once Approved, become a growing
# ground-truth corpus. /api/gt-corpus lists it (cheap); /api/gt-corpus/evaluate runs the AUTO detector against
# every corrected point to show WHERE the detector systematically fails, score any proposed change, and track
# convergence (how many past corrections the algorithm now gets right on its own).
def _gt_corpus_cases() -> list[tuple[str, dict]]:
    """(case_id, manifest) for every case whose corrected border is CONFIRMED + corpus-eligible AND whose
    anchors still match the confirmation signature (a later re-correction without re-approval → stale → skip)."""
    out: list[tuple[str, dict]] = []
    root = settings.CASES_ROOT
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        try:
            m = orch.read_manifest(child.name)
        except Exception:  # noqa: BLE001
            continue
        gt = m.get("border_gt") if isinstance(m, dict) else None
        if not (isinstance(gt, dict) and gt.get("confirmed") and gt.get("corpus_eligible")):
            continue
        _ns, npts, sig = _border_anchor_stats(m)
        if npts <= 0 or sig != gt.get("anchors_sig"):
            continue
        out.append((child.name, m))
    return out


def _rejected_gt_cases() -> list[tuple[str, dict]]:
    """(case_id, manifest) for cases CORRECTED but then REJECTED — border_gt present with confirmed=False.

    These are captured by set_difficult_scan and are deliberately NOT corpus-eligible, but they must still be
    VISIBLE: a correction that is recorded and then never surfaced is write-only, and the reviewer has no way
    to know it accumulated or to opt it in. They are also the corrections most likely to matter — a rejected
    scan is one the auto-detector got wrong."""
    out: list[tuple[str, dict]] = []
    root = settings.CASES_ROOT
    if not root.exists():
        return out
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        try:
            m = orch.read_manifest(child.name)
        except Exception:  # noqa: BLE001
            continue
        gt = m.get("border_gt") if isinstance(m, dict) else None
        if not (isinstance(gt, dict) and not gt.get("confirmed") and gt.get("n_points")):
            continue
        _ns, npts, sig = _border_anchor_stats(m)
        if npts <= 0 or sig != gt.get("anchors_sig"):     # re-corrected since → stale, same rule as confirmed
            continue
        out.append((child.name, m))
    return out


@app.get("/api/gt-corpus")
def gt_corpus() -> dict:
    """Cheap summary of the confirmed, corpus-eligible corrected-border cases (the ground-truth corpus) — no
    detector run. Used by the UI badge + the assistant to see how much GT has accumulated.

    Also reports the CORRECTED-BUT-REJECTED pool separately: real anchor geometry the reviewer drew on scans
    they then rejected. Not part of the tuning corpus (a scan whose output was refused should not silently
    become a global target), but counted here so it is discoverable and can be opted in deliberately."""
    cases = _gt_corpus_cases()
    total = sum(_border_anchor_stats(m)[1] for _, m in cases)
    rej = _rejected_gt_cases()
    rej_pts = sum(_border_anchor_stats(m)[1] for _, m in rej)
    return {"ok": True, "n_cases": len(cases), "n_points": total,
            "cases": [{"case_id": cid,
                       "n_slices": (m.get("border_gt") or {}).get("n_slices"),
                       "n_points": (m.get("border_gt") or {}).get("n_points")} for cid, m in cases],
            "n_rejected_cases": len(rej), "n_rejected_points": rej_pts,
            "rejected_cases": [{"case_id": cid,
                                "n_slices": (m.get("border_gt") or {}).get("n_slices"),
                                "n_points": (m.get("border_gt") or {}).get("n_points")} for cid, m in rej]}


class GtCorpusEvalRequest(BaseModel):
    params: dict | None = None    # detector param overrides → test a PROPOSED algorithm change against the corpus
    limit: int | None = None
    tol: float = 2.0              # px: an anchor point counts as "already correct" if the auto detector is within tol


@app.post("/api/gt-corpus/evaluate")
def evaluate_gt_corpus(req: GtCorpusEvalRequest | None = None) -> dict:
    """GT-vs-AUTO harness. For every confirmed, corpus-eligible corrected case, run the AUTO anterior detector on
    the RAW volume (fix-columns anchors are in raw / border_pass=1 coordinates) and measure its error at the
    user's true-depth anchor points. Returns: WHERE the detector fails (edge vs interior + a per-frame
    histogram of corrections), a single auto-vs-GT score, and a CONVERGENCE metric (fraction of past corrections
    the detector now gets right on its own, within tol). Pass `params` to score a proposed detector change
    against the whole corpus in one call. May be slow (~detector time per case); use `limit` to sample."""
    import numpy as _np
    req = req or GtCorpusEvalRequest()
    tol = float(req.tol)
    cases = _gt_corpus_cases()
    if req.limit:
        cases = cases[: int(req.limit)]
    all_err: list[float] = []; edge_err: list[float] = []; inter_err: list[float] = []
    frame_hist: dict[int, int] = {}; per_case: list[dict] = []; F_EDGE = 10
    for cid, m in cases:
        src = m.get("oct_source"); vi = int(m.get("oct_volume_index", 0) or 0)
        anc = ((m.get("oct_params") or {}).get("border_anchors")) or {}
        if not src or not anc:
            continue
        try:
            params = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
            if req.params:
                params.update(req.params)
            sag = oct_mod.reformat_to_sagittal(oct_mod.read_oct_zstack(src, vi))
            surf = oct_mod.detect_surface_all(sag.astype("float32"), params, workers=4)  # (lateral, frames)
        except Exception as e:  # noqa: BLE001
            per_case.append({"case_id": cid, "error": str(e)[:140]}); continue
        L, Fn = surf.shape; cerr: list[float] = []
        for sl, fr_map in anc.items():
            try:
                lat = int(sl)
            except Exception:  # noqa: BLE001
                continue
            if not (0 <= lat < L) or not isinstance(fr_map, dict):
                continue
            for fk, depth in fr_map.items():
                try:
                    fr = int(fk); gt = float(depth)
                except Exception:  # noqa: BLE001
                    continue
                if not (0 <= fr < Fn):
                    continue
                a = float(surf[lat, fr])
                if a <= 1:
                    continue
                e = abs(a - gt)
                all_err.append(e); cerr.append(e)
                (edge_err if (fr < F_EDGE or fr >= Fn - F_EDGE) else inter_err).append(e)
                frame_hist[fr] = frame_hist.get(fr, 0) + 1
        if cerr:
            ca = _np.array(cerr)
            per_case.append({"case_id": cid, "n_points": len(cerr), "mean_err": round(float(ca.mean()), 2),
                             "p90_err": round(float(_np.percentile(ca, 90)), 2),
                             "already_correct_frac": round(float((ca <= tol).mean()), 2)})

    def _stats(arr: list[float]) -> dict:
        if not arr:
            return {"n": 0}
        a = _np.array(arr)
        return {"n": len(arr), "mean": round(float(a.mean()), 2), "median": round(float(_np.median(a)), 2),
                "p90": round(float(_np.percentile(a, 90)), 2),
                "already_correct_frac": round(float((a <= tol).mean()), 2)}

    return {"ok": True, "n_cases": len([p for p in per_case if "n_points" in p]), "tol_px": tol,
            "overall": _stats(all_err), "edge_frames": _stats(edge_err), "interior_frames": _stats(inter_err),
            "corrections_by_frame": {int(k): int(v) for k, v in sorted(frame_hist.items())},
            "per_case": per_case,
            "note": ("auto detector run on RAW (anchors are border_pass=1); error = |auto − user true depth| at "
                     "each corrected point; already_correct_frac = convergence (detector now matches GT unaided).")}


class TrainingScheduleRequest(BaseModel):
    scheduled: bool = True


@app.post("/api/case/{case_id}/training/schedule")
def schedule_training(case_id: str, req: TrainingScheduleRequest) -> dict:
    """Timeline final step: schedule (or unschedule) this scan for nnU-Net training (turns the entry
    green). Manifest flag only; nnunet_train restricts to scheduled scans when any scan is scheduled."""
    m = orch.write_manifest_value(_require_case(case_id), {"training_scheduled": bool(req.scheduled)})
    return {"ok": True, "training_scheduled": bool(m.get("training_scheduled"))}


# Per-step manifest flags, in lifecycle order (mirrors api/lifecycle.ts scanStep). Resetting TO step N
# clears the flags of every step AFTER N, so the scan drops back to N and the user can redo from there.
# Files on disk are left intact (re-running a step overwrites its artifact) — this is flag-only + reversible.
_STEP_RESET_FLAGS: dict[int, list[str]] = {
    2: ["oct_preprocessed", "oct_iter"],          # Preprocessed (auto)
    3: ["preproc_vetted"],                          # Vetted
    4: ["scar_classification", "scar_range"],       # Classified (scar/control)
    5: ["sam2_meta", "qa_json", "segmentation_preview_dir"],  # Cornea (SAM2)
    6: ["cornea_vetted"],                           # Cornea/background paint-vetted
    7: ["subgroup_confirmed"],                      # Subgroup assigned (now BEFORE scar)
    8: ["scar_done", "scar_metrics"],               # Scar segmented (now AFTER subgroup)
    9: ["consensus_case", "consensus_scar_source"],            # Aligned (link + the scar-source choice)
    10: ["normalized", "normalization_skipped"],               # Normalised against controls (or skipped)
    11: ["corrected_labelmap"],                     # Manually corrected
    12: ["training_scheduled"],                     # Scheduled for training
}


class ResetStepRequest(BaseModel):
    step: int   # target step to return to (1-12); everything AFTER it is cleared


@app.post("/api/case/{case_id}/reset-step")
def reset_step(case_id: str, req: ResetStepRequest) -> dict:
    """Step regression: roll a scan back to `step` by clearing the manifest flags of all later steps
    (flag-only, non-destructive — re-running a step overwrites its artifact). Refuses on a consensus
    case (its consensus_cases/report define its identity; rebuild it instead)."""
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, f"No such case: {case_id}")
    if orch.read_manifest(cid).get("consensus_cases"):
        raise HTTPException(400, "This is a built consensus case — rebuild it rather than resetting a step.")
    target = int(req.step)
    if target < 1 or target > 12:
        raise HTTPException(400, "step must be 1-12.")
    updates: dict = {}
    cleared: list[str] = []
    for s, keys in _STEP_RESET_FLAGS.items():
        if s > target:
            for k in keys:
                updates[k] = None
                cleared.append(k)
    # Rolling back BELOW SAM2 (target < 5) must also remove the on-disk labelmap + QA + previews:
    # nnU-Net training/export, the metrics summary, and the served overlays all gate on FILE existence
    # (labels.best_labelmap_nnunet), not the manifest flags. Leaving the file would silently keep a
    # rolled-back scan in the training cohort and serve a stale overlay (review HIGH #2/#3, MED #16).
    if target < 5:
        updates["scar_metrics"] = None
        labels.corrected_path(cid).unlink(missing_ok=True)
        orch.case_qa_json(cid).unlink(missing_ok=True)
        seg_dir = orch.segmentation_preview_dir(cid)
        if seg_dir.exists():
            shutil.rmtree(seg_dir, ignore_errors=True)
        for grp in ("context_seg", "context_cons"):
            shutil.rmtree(_preview_group_dir(cid, grp), ignore_errors=True)
    if updates:
        orch.write_manifest_value(cid, updates)
    return {"ok": True, "step": target, "cleared": cleared,
            "case_info": orch.current_case_info(cid)}


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
                          "group": "per-slice", "lane": it.get("lane", "full")})
    # ── VOLUME-LEVEL decisions: note nodes carrying the REAL numbers from the last preprocess. These are
    # whole-volume decisions (keep-best iteration, de-tilt, detector auto-tune, surface-crop, minimal crop)
    # that can't be read off one slice, so they're reported as the decision + its outcome. The rigid
    # inter-frame STAGES (motion-correct → dewarp → flatten → height-refine → derotate) are shown visually
    # per-slice above. Number the volume-level nodes CONTIGUOUSLY after the per-slice filmstrip so there is
    # no gap in the sequence.
    _vn = [len(steps)]
    def _vlabel(text: str) -> str:
        _vn[0] += 1
        return f"{_vn[0]}. {text}"
    oct_iter = m.get("oct_iter") or {}
    passes = int(oct_iter.get("passes", 0) or 0)
    if passes and passes > 0:
        metrics = oct_iter.get("metrics") or []
        best = oct_iter.get("best_pass")
        stopped = oct_iter.get("stopped", "")
        dev = ", ".join(f"{float(x):.2f}" for x in metrics) if metrics else "—"
        steps.append({"label": _vlabel(f"Keep-best iteration — {passes} pass(es), kept pass {best}"),
                      "kind": "decision", "group": "volume",
                      "branch": f"boundary deviation per pass: [{dev}] px · argmin kept · stop: {stopped}"})
    # Volume-level decisions from the REAL last run (oct_iter) that complement the per-slice rigid filmstrip
    # (de-tilt / auto-tune / surface-crop / minimal crop can't be read off one slice). The RIGID stages
    # themselves (motion-correct → dewarp → flatten → height-refine → derotate) are shown per-slice above.
    detilt = oct_iter.get("detilt") if isinstance(oct_iter.get("detilt"), dict) else None
    prop = m.get("oct_proposals") or {}
    if detilt:
        steps.append({"label": _vlabel("Global de-tilt — applied"), "kind": "decision", "group": "volume",
                      "branch": f"whole-volume tilt removed: total {detilt.get('total_tilt', '?')}px, slope {detilt.get('slope_per_frame', '?')}px/frame (the constant real decentration is kept)"})
    elif isinstance(prop.get("detilt"), dict):
        pdt = prop["detilt"]
        steps.append({"label": _vlabel("Global de-tilt — PROPOSED (not applied)"), "kind": "decision", "group": "volume",
                      "branch": f"detected total tilt {pdt.get('total_tilt', '?')}px, slope {pdt.get('slope_per_frame', '?')}px/frame — below the auto-apply gate → left for review"})
    at_info = oct_iter.get("auto_tune") if isinstance(oct_iter.get("auto_tune"), dict) else {}
    if at_info.get("params"):
        prm = at_info["params"]
        steps.append({"label": _vlabel("Detector auto-tune"), "kind": "decision", "group": "volume",
                      "branch": "DP surface detector tuned to THIS scan → " + ", ".join(f"{k}={v}" for k, v in prm.items())
                                + (f" · score {at_info.get('score')}" if at_info.get("score") is not None else "")})
    scv = oct_iter.get("surface_crop") if isinstance(oct_iter.get("surface_crop"), dict) else None
    if scv:
        steps.append({"label": _vlabel("Surface-crop extend — clipped apex reconstructed"), "kind": "decision", "group": "volume",
                      "branch": f"posterior parabola fit + canvas extended upward: {scv.get('n_frames', '?')} frame(s), pad {scv.get('pad', '?')}px" + (" (clamped)" if scv.get("clamped") else "")})
    cic = oct_iter.get("crop_incomplete_cornea") if isinstance(oct_iter.get("crop_incomplete_cornea"), dict) else None
    if cic:
        steps.append({"label": _vlabel("Minimal lateral crop (rotation-incomplete edges)"), "kind": "decision", "group": "volume",
                      "branch": f"trimmed left {cic.get('left', 0)} / right {cic.get('right', 0)} lateral slice(s) → {cic.get('new_lateral', '?')} wide, so SAM2 sees a full cornea in every sagittal slice"})
    steps.append({"label": _vlabel("Manual depth nudges (Fix-columns)"),
                  "kind": "decision", "group": "volume",
                  "branch": (f"{len(eff_params.get('manual_shifts') or {})} frame(s) nudged — applied LAST as ground truth"
                             if eff_params.get("manual_shifts") else "none — applied LAST, after all fitting/guards")})
    # #9 Custom crop — the user-removed BOX (frame columns × lateral-slice range, zeroed before SAM2). Shown
    # as a volume node; scar-alignment analytics exclude this box so a partial crop doesn't bias metrics.
    def _runs(xs):
        xs = sorted(set(int(x) for x in xs)); out, s0, prev = [], None, None
        for c in xs + [None]:
            if c is None or (prev is not None and c != prev + 1):
                out.append(f"{s0}–{prev}" if prev > s0 else f"{s0}"); s0 = c
            else:
                s0 = s0 if s0 is not None else c
            prev = c if c is not None else prev
        return out
    region = eff_params.get("crop_region") if isinstance(eff_params.get("crop_region"), dict) else None
    if region and region.get("frames") and (region.get("lateral") or []):
        lo, hi = int(region["lateral"][0]), int(region["lateral"][1])
        fr = sorted(int(f) for f in region["frames"])
        steps.append({"label": _vlabel(f"Custom crop — {len(fr)} frame-column(s) over lateral {min(lo,hi)}–{max(lo,hi)}"),
                      "kind": "decision", "group": "volume",
                      "branch": f"frames {', '.join(_runs(fr))} zeroed across depth over lateral slices {min(lo,hi)}–{max(lo,hi)} of 513, before SAM2 — excluded from scar-alignment (crop-aware)"})
    crop_lat = sorted(int(c) for c in (eff_params.get("crop_lateral") or []))   # legacy full-slice crop
    if crop_lat:
        steps.append({"label": _vlabel(f"Custom column crop (legacy) — {len(crop_lat)} lateral slice(s) removed"),
                      "kind": "decision", "group": "volume",
                      "branch": f"lateral {', '.join(_runs(crop_lat))} of 513 fully zeroed before SAM2 — excluded from scar-alignment"})
    return {"steps": steps, "slices": n_slices, "index": cur_index}


@app.post("/api/case/{case_id}/export-correction-mp4")
def export_correction_mp4_endpoint(case_id: str) -> dict:
    """#10 — render the scan's preprocessing correction as an MP4 grid (rows = axial/coronal/sagittal,
    columns = after(final) → passes → before(raw); each frame scrubs a slice). Saved under the case's
    exports/ folder; returns its path + a download URL. Read-only on the case data."""
    import correction_video
    cid = orch.safe_case_id(case_id)
    if not orch.case_root(cid).exists():
        raise HTTPException(404, "Unknown case.")
    pid, eye, _ = _case_identity(cid)
    stem = f"{(pid or 'scan').upper()}_{(eye or '').upper()}_{cid}_correction".replace(" ", "_").replace("/", "-")
    out = orch.case_root(cid) / "exports" / f"{stem}.mp4"
    # clear any prior export so the download route always serves THIS render
    if out.parent.exists():
        for old in out.parent.glob("*_correction.mp4"):
            old.unlink(missing_ok=True)
    try:
        info = correction_video.export_correction_mp4(cid, out)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    info["download_url"] = f"/api/case/{cid}/correction.mp4"
    return info


@app.get("/api/case/{case_id}/correction.mp4")
def download_correction_mp4(case_id: str) -> FileResponse:
    """Serve the most-recent exported correction MP4 (see export-correction-mp4)."""
    cid = orch.safe_case_id(case_id)
    exp = orch.case_root(cid) / "exports"
    files = sorted(exp.glob("*_correction.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if exp.exists() else []
    if not files:
        raise HTTPException(404, "No exported correction video — export it first.")
    return FileResponse(str(files[0]), media_type="video/mp4", filename=files[0].name)


_BORDER_VOL_CACHE: dict = {}  # path -> (mtime, ndarray) — last border-input volume, so SCRUBBING a pass's
                              # slices doesn't re-decompress the .nii.gz every request (smooth scrolling).
_BORDER_VOL_CACHE_LOCK = threading.Lock()  # concurrent scrub requests run on FastAPI's threadpool — guard get/clear/set
def _load_border_vol(path: Path):
    import os
    import numpy as np
    import nibabel as nib
    key = str(path)
    mt = os.path.getmtime(path)
    with _BORDER_VOL_CACHE_LOCK:
        cached = _BORDER_VOL_CACHE.get(key)
        if cached and cached[0] == mt:
            return cached[1]
    arr = np.ascontiguousarray(np.asanyarray(nib.load(key).dataobj))
    with _BORDER_VOL_CACHE_LOCK:
        _BORDER_VOL_CACHE.clear()                   # keep only the most-recent input (bound memory)
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
        # If the user has CONFIRMED a re-detection (pass 1), serve the cached re-detected surface for EVERY
        # slice (so scrubbing shows the confirmed border). Otherwise, on pass 1, serve the cached ROBUST
        # BASELINE (the SAME _merged_side_edge surface Confirm uses) — NOT a separate fast detector. This is
        # what makes the scrub preview == the Confirm result: with two different detectors, Confirm replaced
        # the whole surface with the robust one and un-edited slices visibly changed. First call computes the
        # baseline (~6s, cached as baseline.npz); later scrubs load it instantly. (pass>1 keeps the fast
        # detector — no per-pass baseline cache.)
        anc = (m.get("oct_params") or {}).get("border_anchors") or {}
        surf = _redetect_surface_cached(case_id, m, anc) if (pass_n <= 1 and anc) else None
        use_surf = surf is not None and surf.shape[0] == n and surf.shape[1] == n_frames
        base_surf = None
        if not use_surf and pass_n <= 1:
            try:
                base_surf = _baseline_surface(case_id, arr, p)         # cached robust = Confirm's baseline
                if base_surf is None or base_surf.shape[0] != n or base_surf.shape[1] != n_frames:
                    base_surf = None
            except Exception:  # noqa: BLE001 — fall back to the fast detector if the baseline can't be built
                base_surf = None
        # #9 crop_region: make the previewed edge/curve reflect the TRUNCATED volume — for slices INSIDE the
        # cropped lateral range, exclude the cropped frame-columns from the quadratic fit and interpolate the
        # edge across them (matches the re-detected corrected volume; without this the overlay never changes).
        _crop_box = oct_mod._crop_region_box(p, n_frames, n)
        _crop_lo, _crop_hi, _crop_fs = _crop_box if _crop_box else (0, -1, [])
        _crop_keep = (np.array([j not in set(int(f) for f in _crop_fs) for j in range(n_frames)])
                      if _crop_fs else None)
        _art_bands = oct_mod._artifact_bands(p, n_frames, n)   # #9 v3 per-lateral artifact band → exclude from the fit
        edges: list = []; fits: list = []
        for i in range(n):
            if use_surf:
                e = np.asarray(surf[i], dtype=np.float64)
            elif base_surf is not None:
                e = np.asarray(base_surf[i], dtype=np.float64)
            else:
                sl = np.ascontiguousarray(arr[i]).astype(np.float32)
                raw = oct_mod._detect_surface_gradient(sl, sigma)  # fast, no prior, no bilateral (pass>1 only)
                e = oct_mod._smooth_median(oct_mod._correct_surface(raw, max_jump), mfs).astype(np.float64)
            in_crop = (_crop_box is not None and _crop_lo <= i <= _crop_hi
                       and _crop_keep is not None and int(_crop_keep.sum()) >= 3)
            # frames EXCLUDED from the cyan fit for THIS lateral: crop_region box cols (when in range) + this
            # lateral's artifact band — so the flat-held band can't drag the parabola off the cornea (#9 v3).
            _fitkeep = (_crop_keep.copy() if in_crop else np.ones(n_frames, dtype=bool))
            _ab = _art_bands[i] if (_art_bands is not None and i < len(_art_bands)) else None
            if _ab is not None and _ab.size:
                _fitkeep[_ab[(_ab >= 0) & (_ab < n_frames)]] = False
            try:
                if int(_fitkeep.sum()) >= 3 and not bool(_fitkeep.all()):
                    f = np.polyval(np.polyfit(xs[_fitkeep], e[_fitkeep], 2), xs)   # fit cornea frames, extrapolate
                    if in_crop:
                        e = e.copy(); e[~_crop_keep] = f[~_crop_keep]              # crop_region interpolates the EDGE too
                else:
                    f = np.polyval(np.polyfit(xs, e, 2), xs)           # quick quadratic fit (cosmetic blue line)
            except Exception:  # noqa: BLE001
                f = e
            edges.append([round(float(v), 1) for v in e])
            fits.append([round(float(v), 1) for v in f])
        out = {"slices": n, "n_frames": n_frames, "depth_vox": depth_vox, "pass": pass_n,
               "edges": edges, "fits": fits}
        # Per-(slice,frame) confidence over the SERVED edge (the array actually displayed) — the fix-columns
        # editor uses it to flag the low-confidence stretches that still need edge corrections. Measured on the
        # same `edges` we return, so preview == what the banner reasons about. ~0.4s over the whole volume; only
        # computed when the editor asks (want_conf), so the non-assist scrub path is unchanged.
        if bool(req.want_conf):
            try:
                conf = oct_mod.surface_confidence_map(arr, np.asarray(edges, dtype=np.float64), p)
                out["conf"] = [[round(float(v), 3) for v in row] for row in conf]
            except Exception:  # noqa: BLE001 — confidence is advisory; never fail the curve fetch over it
                pass
            # UNDER-DETERMINATION report from the LAST corrections run (manifest oct_iter.determinism —
            # READ, never recomputed here: it must describe the surface that was DELIVERED, not whatever is
            # live and unconfirmed in the editor). Rides this existing round-trip, gated on the same
            # want_conf flag, so the editor gets both prompts in one fetch. Absent on auto-only scans.
            _dt = (m.get("oct_iter") or {}).get("determinism")
            if isinstance(_dt, dict) and _dt.get("per_frame"):
                out["determinism"] = _dt
        return out
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT border curves(all) failed: {exc}")


@app.post("/api/case/{case_id}/oct-surface-crop/detect")
def oct_surface_crop_detect(case_id: str, req: OctPreprocessRequest) -> dict:
    """AUTO-DETECT surface-cropped frames (B-scan columns whose corneal apex is above the acquisition window,
    so they have no anterior surface) for the user to VERIFY/EDIT before a re-run. Read-only: runs the validated
    per-slice clip detector across the RAW volume and returns {frames, counts, n_slices, n_frames, depth_vox,
    selected} — `frames` = the auto-suggested set, `selected` = the currently persisted confirmed set (so the
    UI restores prior edits). The confirmed set is applied (posterior-continuity reconstruction) by the next
    oct-preprocess with surface_crop_frames."""
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    try:
        arr = _load_border_vol(_ensure_raw_border_nifti(case_id))   # (lateral, depth, frames) = sagittal, cached
        p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        # Feed the SAME evidence the preprocessing pipeline feeds the rule, so the editor suggests what a
        # re-run would actually do. Without this the endpoint stays on the legacy count rule and keeps
        # re-suggesting its false positives (e.g. the CS010 peripheral-limbus graze) the moment a scan has no
        # persisted set and no run snapshot to fall back on. S_raw is the cached baseline surface, so a warm
        # call is CHEAPER than before (the old code ran an uncached detect pass on every request).
        S_raw = _baseline_surface(case_id, arr, p)
        S_mc, M = _sc_evidence(case_id, arr, p)
        res = oct_mod.detect_surface_crop_frames(arr, p, detect=S_raw,
                                                sc_s_mc=S_mc, sc_s_raw=S_raw, sc_shift=M)
        res["selected"] = sorted(int(f) for f in ((m.get("oct_params") or {}).get("surface_crop_frames") or []))
        res["algo"] = getattr(oct_mod, "_SC_ALGO_VERSION", None)
        # Was the persisted run snapshot produced by an OLDER rule? The mark editors PREFER that snapshot over
        # this live suggestion (so a user's reviewed set is never silently overwritten), which also means an
        # improved detector would stay invisible on every already-processed scan. Expose staleness and let the
        # UI decide — it demotes the snapshot only when there is no confirmed set to protect.
        _snap = ((m.get("oct_iter") or {}).get("surface_crop") or {})
        res["stale_snapshot"] = bool(_snap) and _snap.get("algo") != res["algo"]
        return res
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT surface-crop detect failed: {exc}")


@app.post("/api/case/{case_id}/oct-surface-crop/preview")
def oct_surface_crop_preview(case_id: str, req: OctPreprocessRequest) -> dict:
    """PER-SLICE preview for the surface-crop tool so the user SEES what the correction is based on: the
    detected BOTTOM (posterior) edge — the guidance for cropped frames — and the reconstructed anterior surface
    (posterior continuity), which can extend ABOVE the frame (negative depth) where the apex is cropped. Body:
    {slice_index, surface_crop_frames}. Returns {top, bottom, recon, adopted, n_frames, depth_vox} (x=frame,
    y=depth, depth 0 = TOP). Read-only; a GUIDANCE view of the bottom-edge match (the extend warp flattens to a
    posterior parabola and extends the canvas upward, so the corrected volume is taller and not pixel-identical
    to this reconstructed anterior)."""
    import numpy as np
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    try:
        arr = _load_border_vol(_ensure_raw_border_nifti(case_id))   # (lateral, depth, frames) = sagittal, cached
        n, depth_vox, n_frames = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
        si = max(0, min(n - 1, int(req.slice_index or 0)))
        p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        frames = req.surface_crop_frames
        if frames is None:
            frames = (m.get("oct_params") or {}).get("surface_crop_frames") or []
        sl = np.ascontiguousarray(arr[si]).astype(np.float32)
        top = oct_mod._merged_side_edge(sl, p)                       # detected anterior (DP + scar-guard)
        # The reviewer's manual posterior points for THIS slice, so the preview shows the line they dragged
        # rather than the detector's version of it — preview == what the re-run will use. Request-supplied
        # anchors take precedence over the persisted ones (live drag before it is committed).
        _pa = req.crop_post_anchors if getattr(req, "crop_post_anchors", None) is not None \
            else (m.get("oct_params") or {}).get("crop_post_anchors")
        _row = None
        if isinstance(_pa, dict):
            _row = _pa.get(str(si)) or _pa.get(si) or None
        # GUIDED POSTERIOR: the reviewer's bottom-edge corrections used as a search PRIOR, so the detected
        # bottom edge improves on slices they did NOT draw on. Without this, crop_post_anchors was a per-frame
        # override on its own slice and nothing else — re-running could never improve the bottom edge anywhere,
        # which is the reviewer's "bottom edge detection does not seem to improve after iterations".
        # Held-out measurement (prior from slice 195, scored on slice 214's own points): median error
        # 14.6 -> 11.5 px, within-5px 25.5% -> 31.9%. Best-effort — any failure falls back to plain detection.
        _post_row = None
        if isinstance(_pa, dict) and _pa and bool(p.get("post_guided", True)):
            try:
                _post_row = oct_mod.guided_posterior_row(arr, si, _pa, p)
            except Exception:  # noqa: BLE001
                _post_row = None
        recon, bottom, adopted = oct_mod._crop_reconstruct_slice(
            sl, top, frames, {**p, "_post_anchor_row": _row, "_post_row_detected": _post_row})
        r1 = lambda a: [round(float(v), 1) for v in np.asarray(a)]
        return {"slice_index": si, "n_frames": n_frames, "depth_vox": depth_vox,
                "top": r1(top), "bottom": r1(bottom), "recon": r1(recon),
                "adopted": [int(f) for f in np.where(np.asarray(adopted))[0]]}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT surface-crop preview failed: {exc}")


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
        surf = _redetect_surface_cached(case_id, m, anc) if (pass_n <= 1 and anc) else None
        if surf is not None and 0 <= idx < surf.shape[0] and surf.shape[1] == sl.shape[1]:
            edge = np.asarray(surf[idx], dtype=np.float32)
        elif pass_n <= 1:
            # No confirmed anchors: the scrub preview (oct-border-curves-all) ALREADY serves the robust
            # whole-volume baseline (detect_surface_all, cached in baseline.npz) for every slice, and
            # Confirm/Run flatten to THAT surface. Re-running the single-slice _merged_side_edge here (~360ms)
            # only recomputes a near-identical edge (measured median ~0.4px vs the baseline) but WITHOUT the
            # cross-slice cascade — so the "settle refine" cost the reviewer ~450ms per slice while sometimes
            # nudging the line OFF the surface the warp actually uses. Serve the SAME cached baseline slice
            # instead (a warm .npz read): scrubbing settles instantly AND the displayed edge == the flatten's
            # edge (no on-settle line jump). Fall back to the live detector only if the baseline is absent.
            edge = None
            try:
                _bs = _baseline_surface(case_id, arr, p)
                if _bs is not None and 0 <= idx < _bs.shape[0] and _bs.shape[1] == sl.shape[1]:
                    edge = np.asarray(_bs[idx], dtype=np.float32)
            except Exception:  # noqa: BLE001 — no/stale baseline → live detect below
                edge = None
            if edge is None:
                edge = oct_mod._merged_side_edge(sl, p)
        else:
            edge = oct_mod._merged_side_edge(sl, p)
        # cyan fit EXCLUDES this lateral's artifact band (else the flat-held band drags the parabola off the cornea).
        fit = oct_mod.fit_quadratic_excluding_bands(edge, p, idx, n)
        return {"slices": n, "index": int(idx), "n_frames": int(sl.shape[1]), "depth_vox": int(sl.shape[0]),
                "pass": pass_n, "edge": [float(v) for v in edge], "fit": [float(v) for v in fit]}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT border curve failed: {exc}")


@app.get("/api/case/{case_id}/oct-border-slice")
def oct_border_slice_png(case_id: str, slice_index: int = 0, border_pass: int = 1) -> Response:
    """The fix-columns editor's B-scan at NATIVE voxel resolution (depth rows x frame cols, depth 0 = TOP) as
    a grayscale PNG — NO physical-aspect upscaling. The normal context preview upscales the 101-frame axis to a
    physically-correct aspect with nearest-neighbour at a non-integer ratio, which bakes in UNEVEN frame-column
    widths; the editor instead displays THIS native image at an integer pixels-per-frame so every column is the
    same width AND pixel-sharp. Coordinates match the border curves (arr[idx] = (depth, frames)) so the SVG
    overlay aligns exactly. Pass 1 = the raw volume; pass k = pass (k-1)'s output."""
    import io
    import numpy as np
    from PIL import Image
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    pass_n = max(1, int(border_pass))
    if pass_n <= 1:
        inp = _ensure_raw_border_nifti(case_id)
    else:
        pv = orch.case_root(case_id) / "passes" / f"pass_{pass_n - 1}.nii.gz"
        inp = pv if pv.exists() else _ensure_raw_border_nifti(case_id)
    try:
        arr = _load_border_vol(inp)                              # (lateral, depth, frames), cached
        n = int(arr.shape[0])
        idx = max(0, min(n - 1, int(slice_index)))
        sl = np.ascontiguousarray(arr[idx]).astype(np.float32)  # (depth, frames), depth 0 = TOP
        # percentile 1-99 contrast stretch → uint8 (same look as the context previews' normalize_gray)
        finite = sl[np.isfinite(sl)]
        if finite.size:
            lo = float(np.percentile(finite, 1)); hi = float(np.percentile(finite, 99))
            if hi <= lo:
                hi = lo + 1.0
            gray = (np.clip((sl - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros(sl.shape, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(gray, mode="L").save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT border slice failed: {exc}")


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
                      "redetect_seed_window", "redetect_seed_window_slices", "redetect_interp_window",
                      # native DP detector selection + tuning — a change must invalidate the surface caches
                      "detector", "dp_sigma_depth", "dp_sigma_frame", "dp_below", "dp_max_jump",
                      # DP scar-guard (cross-checks DP vs legacy, pulls DP off a bright internal scar) — its
                      # params change the detected surface, so they must invalidate the surface caches too
                      "dp_scar_guard", "dp_scar_tol", "dp_scar_window", "dp_scar_min_run", "dp_scar_darker_margin",
                      # generalize_surface (propagate learned correction to the whole volume) — its params change
                      # the generalized surface, so they must invalidate the generalize.npz cache
                      "gen_min_slices", "gen_min_resid", "gen_sign_frac", "gen_resid_cap", "gen_taper_slices",
                      "gen_frame_margin", "gen_slice_sigma", "gen_frame_sigma",
                      # frame-edge epithelium snap (frame-axis sibling of edge_follow) — toggling it changes the
                      # detected surface at the left/right frame edges, so it must invalidate the surface caches
                      "frame_edge_snap",
                      # PURE interpolation between dense manual corrections — its params change the served/warp
                      # surface on the dense-anchor path, so they must invalidate the redetect.npz cache
                      "dense_pure_interp", "interp_min_slices", "interp_frame_taper")

# Bumped whenever redetect_surface()'s region/march LOGIC changes (not just its params), so an APP UPDATE
# invalidates surfaces written by the old algorithm. "per-slice-v2" = the per-slice frame-region fix (a
# redetect.npz from the prior global-union code would otherwise be served unchanged after an update — the
# detection params are identical — silently keeping the old buggy surface on already-confirmed cases).
_REDETECT_ALGO_VERSION = "dp-v7-pure-interp"   # dense-anchor path now PURE-interpolates the drawn edge between slices (interpolate_anchors_surface) instead of re-detecting; + dp-v6 corner re-trace


def _detect_params_sig(p: dict) -> str:
    """Canonical signature of the detection-relevant params + algorithm version (for surface-cache freshness)."""
    sig = f"algo={_REDETECT_ALGO_VERSION};" + ";".join(f"{k}={p.get(k)}" for k in _DETECT_PARAM_KEYS)
    # #9 v3: crop_bands now ALTERS detect_surface_all's output (the band-reconstruction that makes the DISPLAYED
    # surface ignore the artifact), so it MUST invalidate the surface caches — otherwise the cheap mark-then-scrub
    # path (set_oct_marks, no re-run) serves a stale baseline that still dives into the artifact. Sorted JSON so
    # the same bands hash identically regardless of key order. (This is the first sticky crop that touches the
    # detector; the "detection is independent of crop_*" assumption elsewhere no longer holds for crop_bands.)
    _cb = p.get("crop_bands")
    if _cb:
        import json as _j
        try:
            sig += ";crop_bands=" + _j.dumps({str(k): [int(v[0]), int(v[1])] for k, v in _cb.items()}, sort_keys=True)
        except (TypeError, ValueError, IndexError):
            sig += f";crop_bands={_cb}"
    return sig


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


def _sc_evidence_cache_path(case_id: str) -> Path:
    return orch.case_root(case_id) / "border_cache" / "sc_evidence.npz"


def _sc_evidence(case_id: str, arr, p: dict):
    """(S_mc, M) for the surface-crop "geom" rule: the anterior surface AFTER axial motion correction, plus
    that correction's per-frame shift. Returns (None, None) if the correction is a no-op or fails, which makes
    detect_surface_crop_frames fall back to the legacy count rule.

    `arr` is the cached raw-border volume in SAGITTAL order (lateral, depth, frames). The float32 cast before
    the correction is DELIBERATE: it reproduces the geometry the rule was validated on
    (.work/sc_dump_surfaces.py) bit-exactly, and it is harmless here because this corrected volume is used
    only to detect a surface and then discarded. Cached per-case beside the baseline surface — border_cache is
    already rmtree'd whenever preprocessing params change, so invalidation is free."""
    import os
    import numpy as np
    raw = _ensure_raw_border_nifti(case_id)
    F = int(arr.shape[2])
    # _DETECT_PARAM_KEYS covers the surface DETECTOR only — no crop_*/clip_* key — so bake the rule's own
    # version in as well, otherwise a change to the crop rule would silently reuse stale evidence.
    psig = _detect_params_sig(p) + ";sc=" + str(getattr(oct_mod, "_SC_ALGO_VERSION", "?"))
    cp = _sc_evidence_cache_path(case_id)
    if cp.exists():
        try:
            z = np.load(cp, allow_pickle=False)
            if (abs(float(z["raw_mtime"]) - float(os.path.getmtime(raw))) <= 1e-6
                    and str(z["params_sig"]) == psig):
                s = np.asarray(z["S_mc"], dtype=np.float32); mm = np.asarray(z["M"], dtype=np.float64)
                if s.shape == (int(arr.shape[0]), F) and mm.size == F:
                    return s, mm
        except Exception:  # noqa: BLE001 — a corrupt/old cache just forces a recompute
            pass
    try:
        S_raw = _baseline_surface(case_id, arr, p)        # already cached; also AMC's own detection
        vol = np.ascontiguousarray(arr.transpose(2, 1, 0)).astype(np.float32)   # → (frames, depth, lateral)
        out, info = oct_mod.axial_motion_correct(vol, p, detect=S_raw)
        if not (info.get("applied") and info.get("shift") is not None):
            return None, None                            # no-op correction ⇒ unvalidated configuration
        S_mc = oct_mod.detect_surface_all(oct_mod.reformat_to_sagittal(out), p)
        M = np.asarray(info["shift"], dtype=np.float64)
    except Exception:  # noqa: BLE001 — evidence is best-effort; the count rule still works
        return None, None
    try:
        cp.parent.mkdir(parents=True, exist_ok=True)
        tmp = cp.with_name("sc_evidence.tmp.npz")   # MUST end .npz (np.savez_compressed appends it otherwise)
        np.savez_compressed(tmp, S_mc=S_mc.astype(np.float32), M=M,
                            raw_mtime=float(os.path.getmtime(raw)), params_sig=psig)
        os.replace(tmp, cp)
    except Exception:  # noqa: BLE001 — a cache write failure must not fail the request
        pass
    return S_mc, M


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
        try:
            _nlat = int(_load_border_vol(raw).shape[0])   # cached; needed for the dense-anchor window override
        except Exception:  # noqa: BLE001
            _nlat = None
        p = _effective_redetect_params(anchors, _nlat, orch.read_manifest(case_id))  # dense → window=0 (match compute)
        if str(z["params_sig"]) != _detect_params_sig(p):    # detection params (incl. dense window) changed → stale
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
    # DENSE anchors → pure connect-the-dots interpolation (redetect_interp_window=0): the per-slice re-detect
    # snap adds detection bumps to the warp target without improving tightness. Baseline stays on the original
    # params (window unused by detect_surface_all); only the redetect surface + its cache sig use p_eff.
    p_eff = _effective_redetect_params(anchors, int(arr.shape[0]), m)
    surface = oct_mod.redetect_surface(arr, anchors, p_eff, baseline=baseline)   # local-band correction (lateral, frames)
    # PIN the reviewer's drawn frames to their exact value: the local-band march snaps to within a couple of
    # px of the drawn line (redetect_seed_window), which drifts a correction off where it was drawn. Ground
    # truth wins at the frames the reviewer actually touched; the march still governs the propagated band.
    oct_mod.pin_anchors(surface, anchors, int(arr.shape[1]), float(p_eff.get("crop_max_pad", 120)))
    # DENSE (only): light across-FRAME smoothing to remove the pinned hand-drawn jitter that would otherwise
    # STEP the rigid warp (bumpy, un-curved B-scan). Confined to cornea frames (below the earliest crop-band lo),
    # so the artifact-band reconstruction can't be blurred into the cornea. Applied AFTER pin so it smooths the
    # drawn line itself — a reviewer's line is approximate; the low-frequency correction survives, only jitter goes.
    _fs = float(p_eff.get("redetect_frame_smooth", 0.0) or 0.0)
    if _fs > 0.0:
        from scipy import ndimage as _ndi
        _depth = int(arr.shape[1])
        _lo = int(arr.shape[2])
        _ab = oct_mod._artifact_bands(p_eff, int(arr.shape[2]), int(arr.shape[0]))
        if _ab is not None:
            _los = [int(b.min()) for b in _ab if b is not None and getattr(b, "size", 0)]
            if _los:
                _lo = max(1, min(_los))
        surface[:, :_lo] = _ndi.gaussian_filter1d(surface[:, :_lo], sigma=_fs, axis=1, mode="nearest")
        # CLAMP the smooth at the reviewer's DRAWN frames (fidelity option 2). The across-frame blur above
        # removes hand-drawn frame-to-frame jitter (which would step the rigid warp) — but on a large correction
        # it pulls the pinned point back toward its un-corrected neighbours by several px, silently UNDOING the
        # correction (measured mean 3.3px, up to 16px off the drawn line). Re-cap each drawn point to within
        # ±clamp px of EXACTLY what was drawn: the un-drawn frames stay smoothed, but the drawn line is honoured
        # to ~subpixel. Uses the same anchor normalization as pin_anchors (ABSENT sentinel skipped, above-canvas
        # negatives clamped to [-crop_max_pad, depth-1]). clamp<0 disables (falls back to the pure blur).
        _clamp = float(p_eff.get("redetect_frame_smooth_anchor_clamp", 1.0))
        _pad = float(p_eff.get("crop_max_pad", 120))
        if _clamp >= 0.0:
            for _sk, _frames in (anchors or {}).items():
                try:
                    _s = int(_sk)
                except (TypeError, ValueError):
                    continue
                if not (0 <= _s < surface.shape[0]) or not isinstance(_frames, dict):
                    continue
                for _fk, _d in _frames.items():
                    try:
                        _f = int(_fk); _dv = float(_d)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= _f < _lo and np.isfinite(_dv) and _dv < _depth - 1:
                        _tgt = float(np.clip(_dv, -_pad, _depth - 1))
                        surface[_s, _f] = float(np.clip(surface[_s, _f], _tgt - _clamp, _tgt + _clamp))
    # PURE INTERPOLATION between manually corrected slices (reviewer directive, cs048). The local-band re-detect
    # above interpolates each slice's WHOLE border (drawn frames mixed with auto for the undrawn ones) then
    # re-detects within a window, so the auto detector's faint-limbus jitter leaks in across slices (measured
    # 5-12px) even though the drawn limbus values are smooth — the "bumpy right end". Replace it with a per-frame
    # linear interp of ONLY the drawn values across slices: smooth AND exact at every drawn edge. Frames drawn on
    # < interp_min_slices keep the auto baseline (the mid-dome the reviewer left alone). Only for DENSE anchors
    # (gap-bounded, so interpolation never spans a huge un-marked gap). See interpolate_anchors_surface.
    if p.get("dense_pure_interp", True) and _anchors_are_dense(anchors, int(arr.shape[0]), p):
        surface = oct_mod.interpolate_anchors_surface(anchors, baseline, p)
        oct_mod.pin_anchors(surface, anchors, int(arr.shape[1]), float(p_eff.get("crop_max_pad", 120)))
    cp = _redetect_cache_path(case_id)
    cp.parent.mkdir(parents=True, exist_ok=True)
    # tmp MUST end in .npz — np.savez_compressed appends '.npz' to any path that doesn't, which would make
    # os.replace move a nonexistent file. Write tmp then atomically replace so a crash can't leave a partial.
    tmp = cp.with_name("redetect.tmp.npz")
    np.savez_compressed(tmp, surface=surface.astype(np.float32),
                        anchors_sig=_border_anchors_sig(anchors),
                        raw_mtime=float(os.path.getmtime(raw)),
                        params_sig=_detect_params_sig(p_eff))
    os.replace(tmp, cp)
    return surface


# ── GENERALIZE (propagate the learned correction to the WHOLE volume) — a sibling cache of redetect.npz.
#    generalize.npz holds the surface from oct_mod.generalize_surface (the interpolated residual field). It is
#    keyed like redetect.npz (anchors_sig + raw_mtime + params_sig, which now includes the gen_* keys). When the
#    manifest flag oct_params['border_generalize'] is set, _redetect_surface_cached returns THIS surface instead
#    of the local-band redetect — so BOTH the scrub preview and the Run/warp use the generalized surface with no
#    change at those call sites. A "generalize" endpoint sets the flag + computes the cache; "discard" clears it.
def _generalize_cache_path(case_id: str) -> Path:
    return orch.case_root(case_id) / "border_cache" / "generalize.npz"


def _generalize_surface_fresh(case_id: str, anchors: dict):
    import os
    import numpy as np
    cp = _generalize_cache_path(case_id)
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
        if str(z["params_sig"]) != _detect_params_sig(p):
            return None
        return np.asarray(z["surface"], dtype=np.float32)
    except Exception:  # noqa: BLE001
        return None


def _compute_generalize_cache(case_id: str, m: dict, anchors: dict):
    """Compute the whole-volume generalized surface (interpolated residual field) from `anchors` and cache it."""
    import os
    import numpy as np
    raw = _ensure_raw_border_nifti(case_id)
    arr = _load_border_vol(raw)                              # (lateral, depth, frames)
    p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
    baseline = _baseline_surface(case_id, arr, p)           # cached auto surface
    surface = oct_mod.generalize_surface(arr, anchors, p, baseline=baseline)
    # PIN drawn frames exact: the residual field is interpolated across the volume, so at the drawn frames it
    # should reproduce the drawn line exactly, not a smoothed approximation of it.
    oct_mod.pin_anchors(surface, anchors, int(arr.shape[1]), float(p.get("crop_max_pad", 120)))
    cp = _generalize_cache_path(case_id)
    cp.parent.mkdir(parents=True, exist_ok=True)
    tmp = cp.with_name("generalize.tmp.npz")
    np.savez_compressed(tmp, surface=surface.astype(np.float32),
                        anchors_sig=_border_anchors_sig(anchors),
                        raw_mtime=float(os.path.getmtime(raw)),
                        params_sig=_detect_params_sig(p))
    os.replace(tmp, cp)
    return surface


def _generalize_surface_cached(case_id: str, m: dict, anchors: dict):
    if not anchors:
        return None
    surf = _generalize_surface_fresh(case_id, anchors)
    if surf is None:
        try:
            surf = _compute_generalize_cache(case_id, m, anchors)
        except Exception:  # noqa: BLE001
            return None
    return surf


def _anchors_are_dense(anchors: dict, n_lateral, p: dict) -> bool:
    """True when the fix-columns anchors span the volume with no large gap — so the LOCAL-redetect surface
    (redetect_surface step-2: linear connect-the-dots interpolation of the residual between adjacent anchored
    slices) already covers the whole volume, and we can serve that TIGHT surface instead of the smoothed/gated
    whole-volume generalize field. Generalize is tuned to spread a FEW corrections robustly across laterals and
    so attenuates ~40% of the drawn correction between dense anchors; redetect follows the drawn points ~3×
    tighter and still drives the rigid flatten with volume-spanning anchors. Sparse anchors → False → keep
    generalize. Gated by dense_anchor_redetect so the behaviour can be disabled."""
    if not p.get("dense_anchor_redetect", True):
        return False
    try:
        sl = sorted(int(k) for k in (anchors or {}).keys())
    except (TypeError, ValueError):
        return False
    if len(sl) < 4:                                    # too few to call "dense" — let generalize generalize
        return False
    band = max(1, int(p.get("redetect_slice_band", 20)))
    max_gap = max(sl[i + 1] - sl[i] for i in range(len(sl) - 1))
    # dense_max_gap (user directive 2026-08-24 "allow interpolation even at 40 slices apart"): the widest gap between
    # drawn slices that still counts DENSE → connect-the-dots INTERPOLATED (not re-detected) between them. Was 2*band=40.
    if max_gap > int(p.get("dense_max_gap", 2 * band)):
        return False
    if n_lateral and int(n_lateral) > 1:               # the anchored span must reach both ends, else the ends taper
        if sl[0] > band or sl[-1] < int(n_lateral) - 1 - band:   # to auto and generalize reaches more of those laterals
            return False
    return True


def _effective_redetect_params(anchors: dict, n_lateral, m: dict) -> dict:
    """Detection params for the LOCAL-redetect surface, with the DENSE-anchor override: dense anchors use
    redetect_interp_window=0 (PURE connect-the-dots interpolation between drawn slices) instead of the per-slice
    re-detect snap. The snap re-detects every in-between slice from the image within ±window px of the interp
    prior, which adds per-slice detection BUMPS to the warp target without improving tightness to the drawn
    points (measured cs007: across-slice roughness 1.00→0.54px = as smooth as generalize, tightness unchanged
    ~1.9px, dome preserved). redetect_interp_window is in _DETECT_PARAM_KEYS, so keying it here makes a stale
    window=3 (bumpy) redetect.npz auto-invalidate on the next access for a dense scan."""
    p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
    if _anchors_are_dense(anchors, n_lateral, p):
        # window=0 → pure connect-the-dots interpolation across slices; frame_smooth → remove the hand-drawn
        # anchors' frame-to-frame JITTER, which otherwise steps the rigid warp into a bumpy, "un-curved" B-scan.
        # σ≈1.5 matches generalize's across-frame smoothness (frame-roughness 1.94→0.99px) while KEEPING the tight
        # across-slice interpolation (the smoothing is a different axis). Corrections are approximate by design —
        # BUT the σ=1.5 blur alone dragged a big drawn correction (a floating edge snapped down onto the
        # epithelium) back toward its un-corrected neighbours by up to 16px (mean 3.3px off the drawn line),
        # silently undoing it. redetect_frame_smooth_anchor_clamp caps how far a DRAWN point may move from
        # exactly what the reviewer placed (±1px): jitter is still smoothed on the un-drawn frames, but the drawn
        # line is honoured to ~subpixel. See _compute_redetect_cache for the clamp application.
        p = {**p, "redetect_interp_window": 0.0, "redetect_frame_smooth": 1.5,
             "redetect_frame_smooth_anchor_clamp": 1.0}
    return p


def _redetect_surface_cached(case_id: str, m: dict, anchors: dict):
    """The re-detected surface for `anchors`: the fresh cache if valid, else recompute+cache. This makes an
    ALGORITHM upgrade (or a param change) transparently refresh the surface for BOTH the scrub display and the
    warp — so a case the user confirmed under the OLD algorithm shows/uses the NEW corrected surface without a
    manual re-Confirm. Returns None when there are no anchors (→ caller shows the plain auto baseline).

    If oct_params['border_generalize'] is set, returns the WHOLE-VOLUME generalized surface (generalize.npz)
    instead of the local-band redetect — rerouting scrub + Run to the generalization in one place."""
    # DENSE anchors FIRST — the reviewer wants their marks INTERPOLATED between the drawn slices, NOT re-detected.
    # Serve the local-redetect PURE connect-the-dots surface (redetect_interp_window=0 via
    # _effective_redetect_params) and SKIP guided. Guided is a from-image re-detection of EVERY slice — i.e.
    # "automatic edge detection between the manually marked slices", the OPPOSITE of interpolation (reviewer
    # directive 2026-08-17: "not interpolating between the manually drawn slices, instead reverting to automatic
    # edge detection between them"). guided/generalize stay for SPARSE marks, where interpolating across large
    # gaps is unreliable and from-image detection genuinely helps.
    if anchors:
        _pden = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        _nlden = None
        try:
            _nlden = int(_load_border_vol(_ensure_raw_border_nifti(case_id)).shape[0])
        except Exception:  # noqa: BLE001
            _nlden = None
        if _anchors_are_dense(anchors, _nlden, _pden):
            _surf = _redetect_surface_fresh(case_id, anchors)
            if _surf is None:
                try:
                    _surf = _compute_redetect_cache(case_id, m, anchors)
                except Exception:  # noqa: BLE001
                    _surf = None
            if _surf is not None:
                return _surf
    # GUIDED (now only for SPARSE marks), when it won its guard. It is a detection of every slice from the image
    # (seeded by the corrections), rather than the corrections interpolated across slices, and it is only ever
    # written when it measured better than auto. Falls through if the cache is missing or stale, rather than
    # serving a surface that no longer matches the anchors it was judged on.
    if (m.get("oct_params") or {}).get("border_guided"):
        gp = _guided_cache_path(case_id)
        try:
            if gp.exists():
                z = np.load(gp, allow_pickle=False)
                if str(z["anchors_sig"]) == _border_anchors_sig(anchors or {}):
                    return np.asarray(z["surface"], dtype=np.float32)
        except Exception:  # noqa: BLE001
            pass
    if not anchors:
        return None
    if (m.get("oct_params") or {}).get("border_generalize"):
        # DENSE anchors (spanning the volume, no gap > 2×redetect_slice_band) → skip the smoothed/gated
        # generalize field and serve the TIGHT local-redetect connect-the-dots surface instead: it follows the
        # drawn points ~3× tighter (≈1.4px vs ≈4px from the neighbour blend) and still drives the rigid flatten
        # (measured 4.2px median shift / 74% laterals on cs007). Both the scrub display and the warp call this,
        # so preview == result is preserved. Sparse corrections keep generalize (its robust wide-lateral spread
        # is why it exists). n_lateral from the cached raw-border volume (a warm dict lookup on every caller).
        _p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        _nlat = None
        try:
            _nlat = int(_load_border_vol(_ensure_raw_border_nifti(case_id)).shape[0])
        except Exception:  # noqa: BLE001 — density falls back to gap-only (no span check) if the shape is unknown
            _nlat = None
        if not _anchors_are_dense(anchors, _nlat, _p):
            gs = _generalize_surface_cached(case_id, m, anchors)
            if gs is not None:
                return gs
            # generalize failed → fall through to the local redetect rather than showing bare auto
    surf = _redetect_surface_fresh(case_id, anchors)
    if surf is None:
        try:
            surf = _compute_redetect_cache(case_id, m, anchors)
        except Exception:  # noqa: BLE001 — fall back to the baseline display if the recompute fails
            return None
    return surf


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
        op.pop("border_generalize", None)   # a fresh local Confirm exits whole-volume-generalize mode
        op.pop("border_guided", None)       # ...and guided mode: its cache was judged against the OLD anchors
        # MERGE OR REPLACE, decided by how complete the payload is — not by taste.
        #   edge mode      the client sends EVERY anchored slice (its editable set is seeded from what is
        #                  persisted), so the payload is authoritative: replace. This is also the only way to
        #                  clear a slice, since a slice emptied by dragging back onto the detected edge simply
        #                  drops out of the payload.
        #   quadratic mode the client sends ONLY the slices whose curve was shaped. Replacing on that payload
        #                  DELETED every edge correction on every other slice — a reviewer who fixed the border
        #                  on one slice and later shaped a curve on another silently lost the first.
        prev_anc = dict(op.get("border_anchors") or {})
        prev_win = dict(op.get("redetect_seed_window_slices") or {})
        # Which slices in THIS payload are an exact fitted curve. Explicit list when the client sends one;
        # otherwise fall back to the old all-or-nothing flag so an older client still behaves as before.
        _exact = ({str(x) for x in (req.parabola_slices or [])} if req.parabola_slices is not None
                  else (set(anchors) if req.parabola else set()))
        if req.parabola_slices is not None or req.parabola:
            # MERGE, because a payload that names exact slices is describing part of the picture, not all of
            # it — replacing on it would drop edge corrections made on slices this commit does not mention.
            merged = {**prev_anc, **anchors} if anchors else {}
            # A shaped quadratic IS the surface → follow it exactly (window 0) on ITS slices only. Slices
            # carrying hand-drawn edge anchors keep the default window: those are approximate by the
            # reviewer's own account, so honouring them to the pixel would be reading in more than they said.
            # A slice carrying BOTH is NOT exact: it contains approximate points, and treating the whole
            # slice as exact would pin those to the pixel.
            # Per-slice seed window. A slice this commit RE-SENDS is redefined by this payload: exact (0.0)
            # if named in _exact, else approximate (dropped → read with the global default). A slice this
            # commit does NOT mention keeps its PRIOR window — so an exact quadratic shaped on an earlier
            # commit stays exact instead of silently reverting to the approximate global window (the bug that
            # made a multi-slice correction lose the first slice's exactness on the next commit). A slice
            # carrying BOTH shaped and hand-drawn anchors is approximate: _exact lists only purely-shaped slices.
            _this = {str(k) for k in (anchors or {})}
            win = {}
            for _k in merged:
                _ks = str(_k)
                if _ks in _this:
                    if _ks in _exact:
                        win[_ks] = 0.0
                elif _ks in prev_win:
                    win[_ks] = prev_win[_ks]
        else:
            merged = anchors
            win = {k: v for k, v in prev_win.items() if k in merged}
        op["border_anchors"] = merged
        # sorted: this dict is stringified into the detector cache signature, so a stable key order keeps an
        # unchanged set from reading as a change and needlessly discarding the cached surface
        op["redetect_seed_window_slices"] = {k: win[k] for k in sorted(win, key=lambda x: int(x))}
        # The GLOBAL window stays at the default and is what any slice not named above is read with. It used to
        # be flipped to 0 for the whole scan by a single quadratic commit, which re-interpreted every other
        # slice's approximate anchors as exact.
        op["redetect_seed_window"] = float(oct_mod.DEFAULT_PARAMS.get("redetect_seed_window", 2.0))
        anchors = merged
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


# ── AXIAL fix-tool: correct the anterior surface in the AXIAL (B-scan) plane (fixed FRAME, drag across laterals),
#    which the sagittal fix-columns tool structurally cannot reach. Serves the CORRECTED preprocessed-output volume
#    (what the user sees) + its detected surface; Confirm persists sticky axial_anchors; Run re-applies them via the
#    post-hoc apply_axial_surface_gt warp. See oct_preprocess.apply_axial_surface_gt. ─────────────────────────────
def _oct_corrected_vol_path(case_id: str, m: dict) -> Path:
    """The corrected (preprocessed-output) working NIfTI — the volume the axial fix-tool draws on + warps."""
    src = m.get("oct_source")
    if not src:
        raise HTTPException(400, f"Case {case_id} has no .OCT source.")
    return _oct_working_path(case_id, src)


_AXIAL_SURF_CACHE: dict = {}   # case_id -> (work_mtime, params_sig, surf(lateral,frames), (L, D, nF))


def _corrected_prior_surface(case_id: str, work: Path, p: dict, vol_shape):
    """The reviewer's CORRECTION CURVE, carried into CORRECTED depth space. Returns (prior(lateral,frames), meta)
    or (None, reason).

    "The curve that was used for correction of the original edge" is border_cache/provided_edges.npz — the exact
    surface the corrections warp flattened to, written at api_server.py:2727 from whichever served surface the
    reviewer was looking at, and pinned to their drawn anchors (measured on cs048: RMS 0.000 px at all 1746 drawn
    points over 80 slices). It lives in RAW depth space, so it has to be moved by whatever the pipeline moved.

    HOW THE MOVE IS OBTAINED — two options were considered:
      (a) MEASURE it here, detector-free, by whole-A-scan cross-correlation of input/_raw_border.nii.gz against
          the delivered volume (oct_preprocess.measure_applied_move), cached to border_cache/applied_move.npz.
      (b) PERSIST the total applied per-frame move during preprocess_oct_to_nifti's corrections branch and read
          it back from the manifest / a npz.
    (b) is exact and free at serve time, but the move is not currently recorded anywhere — manifest oct_iter
    carries only per-stage SUMMARIES (frames_adjusted, max_deg, roughness), not the composed field — so it would
    exist only for scans preprocessed AFTER the change: all 308 already-processed scans, including the one the
    reviewer is looking at, would still need (a) as the fallback. That makes (b) a second code path that buys
    nothing today, so this implements (a) only. (b) remains the right follow-up if the measurement ever proves
    too slow or too fragile; the cache file is the natural place to write it into.

    The measured move is fitted per frame as a RIGID shift + tilt in lateral — the same class of transform the
    pipeline is allowed to apply. On cs048 the residual about that line is 0.01-0.12 px MAD (so the model is not
    a simplification), and the tilt term is NOT negligible: its half-span reaches 5.7 px at frame 25, which a
    per-frame scalar would smear straight through a +-1 px window.

    Everything degrades to (None, reason) → the caller keeps today's free detection."""
    import numpy as np
    bc = orch.case_root(case_id) / "border_cache"
    pe = bc / "provided_edges.npz"
    raw = orch.case_root(case_id) / "input" / "_raw_border.nii.gz"
    if not pe.exists():
        return None, "no provided_edges.npz (scan has no correction curve)"
    if not raw.exists():
        return None, "no input/_raw_border.nii.gz"          # deliberately NOT regenerated: that re-reads the .OCT
    L, D, F = int(vol_shape[0]), int(vol_shape[1]), int(vol_shape[2])
    try:
        surf0 = np.asarray(np.load(pe)["surface"], dtype=np.float64)
    except Exception as exc:  # noqa: BLE001
        return None, f"provided_edges unreadable: {exc}"
    if surf0.shape != (L, F):
        return None, f"provided_edges shape {surf0.shape} != corrected {(L, F)}"
    # CANVAS PAD (bug found 2026-09-04 on cs048_od_v1_3). provided_edges.npz is in RAW-canvas rows, but when the
    # pipeline extends the canvas for an above-window apex it pads the volume at the TOP and shifts the provided
    # edges by the same amount (preprocess_oct_to_nifti: `provided_edges = _pe + float(_ext_pad)`). The move below
    # is measured against the raw padded the same way, so it is in PADDED rows too — carrying an UNPADDED surface
    # by it left the corrected-pane line exactly `pad` rows above the tissue (measured: the carried line sat 6 px
    # ABOVE the dark->bright crossing where the reviewer's own convention is 5 px below it; +9 brought it to -3),
    # and the fold-back of corrected-pane edits (new_GT = served + edit - carried) then wrote the reviewer's redrawn
    # depths `pad` rows too DEEP into border_anchors (measured -13 px vs -5 px for their own raw-pane anchors).
    # Offset the surface into the corrected canvas here, once, for both the cached and the measured branch.
    try:
        import nibabel as _nib
        _raw_depth = int(_nib.load(str(raw)).shape[1])
    except Exception:  # noqa: BLE001
        _raw_depth = D
    _canvas_pad = int(D - _raw_depth) if D > _raw_depth else 0
    if _canvas_pad > 0:
        surf0 = surf0 + float(_canvas_pad)
    key = f"{raw.stat().st_mtime_ns}:{work.stat().st_mtime_ns}:{p.get('corrected_prior_max_lag')}:{p.get('corrected_prior_min_cols')}"
    mv_path = bc / "applied_move.npz"
    move = None
    meta: dict = {}
    if mv_path.exists():
        try:
            z = np.load(mv_path, allow_pickle=False)
            if str(z["key"]) == key and z["move"].shape == (L, F):
                move = np.asarray(z["move"], dtype=np.float64)
                meta = {"n_extrapolated": int(z["n_extrapolated"]), "cached": True}
        except Exception:  # noqa: BLE001
            move = None
    if move is None:
        import nibabel as nib
        try:
            rv = np.asarray(nib.load(str(raw)).dataobj).astype(np.float32)
            cv = np.asarray(nib.load(str(work)).dataobj).astype(np.float32)
            # A CANVAS PAD IS NOT A REASON TO GIVE UP. When the corrected apex would sit above the window the
            # pipeline extends the canvas upward (oct_iter.canvas_extend) — 9 px on cs048_od_v1_3 — so the
            # corrected volume is DEEPER than the raw one and the content sits that many rows lower. Bailing
            # here turned the corrected pane into a free re-detection: it stopped following the reviewer's own
            # line, and the two surfaces drifted 36 px apart. The move is still perfectly rigid; only the frame
            # of reference moved. Pad the raw volume by the same amount, at the same end, and measure normally.
            if rv.shape != cv.shape:
                if rv.shape[0] == cv.shape[0] and rv.shape[2] == cv.shape[2] and cv.shape[1] > rv.shape[1]:
                    _pad = int(cv.shape[1] - rv.shape[1])
                    rv = np.concatenate([np.zeros((rv.shape[0], _pad, rv.shape[2]), dtype=rv.dtype), rv], axis=1)
                else:
                    return None, (f"raw {rv.shape} != corrected {cv.shape} "
                                  f"(canvas changed in lateral/frame — move is not rigid)")
            r = oct_mod.measure_applied_move(rv, cv, p)
        except Exception as exc:  # noqa: BLE001
            return None, f"applied-move measurement failed: {exc}"
        move = np.asarray(r["move"], dtype=np.float64)
        meta = {"n_extrapolated": len(r["extrapolated_frames"]), "cached": False,
                "extrapolated_frames": r["extrapolated_frames"]}
        try:                                                 # best-effort cache; a read-only store just re-measures
            bc.mkdir(parents=True, exist_ok=True)
            tmp = mv_path.with_name("applied_move.tmp.npz")
            np.savez_compressed(tmp, key=np.array(key), move=move.astype(np.float32),
                                n_extrapolated=np.array(meta["n_extrapolated"]))
            os.replace(tmp, mv_path)
        except Exception:  # noqa: BLE001
            pass
    # The carry itself lives in oct_preprocess so the PIPELINE can build the identical prior at its own warp
    # sites (oct_preprocess.carry_correction_curve). Everything above is case-store I/O the worker has no access
    # to; the arithmetic is the part that has to be shared.
    meta["canvas_pad"] = _canvas_pad
    return oct_mod.carry_correction_curve(surf0, move, D), meta


def _corrected_detected_surface(case_id: str, work: Path, p: dict, vol, return_prior: bool = False):
    """The DETECTED anterior surface of the CORRECTED volume (lateral, frames) — free DP, or, when the scan
    carries a correction curve, a re-detection constrained to +-corrected_prior_window px of that curve.

    ONE function so the corrected pane and the fold-back of corrected edits into border_anchors diff against the
    SAME baseline. They must: the fold-back computes new_GT = served + (reviewer_edit - corrected_detected), so if
    the pane drew the constrained line while the fold-back subtracted a free detection, every folded anchor would
    be wrong by the difference between them — up to 240 px on cs048 (see _corrected_prior_surface).

    The +-window bound is enforced by an explicit CLIP, because guided_redetect_all does NOT guarantee it. Two
    places inside it leave the band: (1) _redetect_one_slice's tail — _correct_surface's cubic outlier fill and the
    median filter across frames — measured on cs048 puts 44.1% of picks further than 1 px from the prior, max 20.4;
    (2) _surface_post_passes' whole-volume smoothers (robust dome, edge-dome follow, corner re-trace), which take
    38.8% of points out of the band, max 66.5 px. The ADAPTIVE window is the one part that is already safe: it
    interpolates between guided_window_min and the window, so it can only shrink — provided guided_window_min is
    not larger than the window, which is why it is clamped here."""
    import numpy as np
    cpw = float(p.get("corrected_prior_window", oct_mod.DEFAULT_PARAMS.get("corrected_prior_window", 0.0)) or 0.0)
    if cpw > 0:
        prior, meta = _corrected_prior_surface(case_id, work, p, vol.shape)
        if prior is None:
            print(f"[corrected-prior] OFF for {case_id}: {meta}", file=sys.stderr, flush=True)
        else:
            # The recipe lives in oct_preprocess (constrained_corrected_surface) so the corrections-path warps
            # can build the SAME line at their own call sites instead of re-running a free detect_surface_all.
            s = oct_mod.constrained_corrected_surface(vol, prior, p, window=cpw)
            print(f"[corrected-prior] ON for {case_id}: window={cpw:.2f}px {meta}", file=sys.stderr, flush=True)
            return (s, prior) if return_prior else s
    free = oct_mod.detect_surface_all(vol, p)                          # (lateral, frames) — the free detection
    return (free, None) if return_prior else free


def _axial_surface_cached(case_id: str, work: Path, p: dict):
    """detect_surface_all on the CORRECTED volume (lateral,depth,frame) → (lateral, frames), cached per case by the
    output file mtime + a detection-params signature so scrubbing frames is instant (first call detects the whole
    volume, ~seconds). This is the SAME detector apply_axial_surface_gt diffs against, so the drawn line the user
    corrects is the one the warp re-diffs (preview ≈ result).

    CORRECTED-PRIOR RE-DETECTION (reviewer, 2026-08-25): the detection itself is now _corrected_detected_surface,
    which on a scan carrying a correction curve re-detects within +-corrected_prior_window px of that curve
    instead of detecting freely. The cache signature therefore also keys on the two files that prior is built
    from (border_cache/provided_edges.npz, input/_raw_border.nii.gz) and on the window."""
    import json as _json
    import numpy as np
    import nibabel as nib
    mt = work.stat().st_mtime
    # CORRECTED-GENERALIZE: the reviewer's drawn edges reconstruct the served surface where the DP fails, so they
    # must be part of the cache key (drawing a slice must re-serve). Full anchors dict → any change invalidates.
    _cea = p.get("corrected_edge_anchors") or {}
    # The corrected-prior path depends on border_cache/provided_edges.npz and on input/_raw_border.nii.gz, so both
    # of their mtimes join the signature — re-confirming the border rewrites provided_edges and MUST re-serve.
    _bc = orch.case_root(case_id) / "border_cache"
    _pe_mt = _bc.joinpath("provided_edges.npz").stat().st_mtime_ns if _bc.joinpath("provided_edges.npz").exists() else 0
    _raw_p = orch.case_root(case_id) / "input" / "_raw_border.nii.gz"
    _raw_mt = _raw_p.stat().st_mtime_ns if _raw_p.exists() else 0
    _cpw = float(p.get("corrected_prior_window", oct_mod.DEFAULT_PARAMS.get("corrected_prior_window", 0.0)) or 0.0)
    # The constrained path ALSO depends on guided_redetect_all's own knobs and on the two params that key the
    # applied-move cache — none of which are in _DETECT_PARAM_KEYS (that list covers detect_surface_all only).
    # Without them a change to e.g. guided_conf_lo would keep serving the previous surface until the volume mtime
    # moved or the process restarted: a silent staleness hole that did NOT exist before the corrected-prior change.
    _CPRIOR_SIG_KEYS = ("guided_adaptive", "guided_window_min", "guided_conf_lo", "guided_conf_hi",
                        "guided_conf_pctl", "corrected_prior_max_lag", "corrected_prior_min_cols")
    sig = (";".join(f"{k}={p.get(k)}" for k in _DETECT_PARAM_KEYS if k in p) + ";cea=" + _json.dumps(_cea, sort_keys=True)
           + ";" + ";".join(f"{k}={p.get(k)}" for k in _CPRIOR_SIG_KEYS if k in p)
           + f";cpw={_cpw};pe={_pe_mt};raw={_raw_mt}")
    c = _AXIAL_SURF_CACHE.get(case_id)
    if c and c[0] == mt and c[1] == sig:
        return c[2], c[3]
    vol = np.asarray(nib.load(str(work)).dataobj).astype(np.float32)   # (lateral, depth, frame) = sagittal layout
    surf, _prior = _corrected_detected_surface(case_id, work, p, vol, return_prior=True)
    _pre_gen = np.array(surf, dtype=np.float32)   # the line the WARP stages use — see _warp_gap below
    if _cea:
        # Bypass the DP where it fails on steep limbus descents: reconstruct from the drawn edges + across-lateral
        # interpolation (pixel-exact where drawn). See oct_preprocess.generalize_corrected_surface.
        surf = oct_mod.generalize_corrected_surface(surf, _cea, p)
    # Match the warp TARGET's limbus/edge-band smoothing so the SERVED surface (the corrected-pane red line + the
    # segmentation clip) is as smooth as the delivered tissue — the re-detection re-jitters ~3px at the faint edge
    # on top of the now-smooth tissue, and left raw it would read as "still bumpy". DEFAULT_PARAMS carries the gate.
    surf = oct_mod.smooth_surface_edge_band(surf, {**oct_mod.DEFAULT_PARAMS, **p})
    # RE-ASSERT THE ±window BOUND, because the two transforms above are UNBOUNDED. The clip inside
    # _corrected_detected_surface only bounds the detection; generalize_corrected_surface spreads the reviewer's
    # corrected-edge anchors across laterals and smooth_surface_edge_band re-smooths the limbus band, and measured
    # on cs048 those took 3.7-10.5% of served points back outside ±1px (max 7.9px). A line described to the reviewer
    # as "within 1px of the correction curve" must actually be that, so the bound is applied LAST.
    # The ONE sanctioned exception: a point the reviewer DREW on the corrected result (corrected_edge_anchors) is
    # ground truth and outranks the band — those exact points are restored after the clip. Their across-lateral
    # INTERPOLATION is not drawn, only derived, so it stays inside the band.
    if _prior is not None and _cpw > 0:
        _drawn = None
        if _cea:
            _drawn = [(int(_l), int(_f), float(_d))
                      for _l, _fr in _cea.items() if isinstance(_fr, dict)
                      for _f, _d in _fr.items()]
        surf = np.clip(np.asarray(surf, dtype=np.float64), _prior - _cpw, _prior + _cpw).astype(np.float32)
        for _l, _f, _d in (_drawn or []):
            if 0 <= _l < surf.shape[0] and 0 <= _f < surf.shape[1]:
                surf[_l, _f] = _d
    # WARP GAP, recorded so it can never be silently hidden again. The three corrections-path warp stages diff the
    # reviewer's edits against `_pre_gen` — the constrained detection BEFORE generalize/anchor-pin — because a
    # generalized, anchor-pinned line leaves them zero residual and makes them decline (the correction collapses).
    # The pane, however, shows the generalized+pinned line, because the reviewer's drawn corrected-edge anchors are
    # GT and they expect to see them honoured. Both are defensible, but they are NOT the same line: measured on
    # cs020_od_v2 (235 drawn anchors on 20 laterals) they differ by up to 20.5 px across 497 of 513 laterals.
    # So "the line you approved is the line the Run uses" holds ONLY where this gap is ~0. Surfaced via
    # /oct-corrected-curve so the reviewer sees the number for the slice in front of them instead of trusting a
    # claim. No corrected_edge_anchors → generalize is a no-op → the gap is 0 and nothing is reported.
    with np.errstate(all="ignore"):
        _gap = np.abs(np.asarray(surf, np.float64) - np.asarray(_pre_gen, np.float64))
    _gap = np.where(np.isfinite(_gap), _gap, 0.0)
    if float(_gap.max()) > 1e-6:
        print(f"[corrected-prior] warp gap for {case_id}: median {np.median(_gap):.3f} max {_gap.max():.3f} px "
              f"on {int((_gap > 1.0).any(axis=1).sum())}/{_gap.shape[0]} laterals "
              f"(pane shows generalize+anchor-pin; the warp stages do not)", file=sys.stderr, flush=True)
    shape = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))
    # Cache BOTH lines. surf (generalize + anchor-pin) stays the primary for every existing consumer — the
    # segmentation clip, the fix-axial pane — so nothing else changes. _pre_gen is the constrained DETECTION,
    # which is what the three corrections-path warp stages diff against, and it is what the corrected pane now
    # draws in red so "the line you approve" and "the line Run uses" are the same object. The reviewer's target
    # (surf) is drawn alongside it in its own colour instead of silently replacing it.
    _AXIAL_SURF_CACHE[case_id] = (mt, sig, surf, shape, _gap.astype(np.float32), _pre_gen)
    return surf, shape


def _fit_corrected_edits_into_transform(case_id: str, m: dict, eff_params: dict, trusted_laterals=None):
    """The reviewer's CORRECTED-pane edits → the case's sticky per-frame TRANSFORM (reviewer, 2026-09-04: the
    corrections to the corrected edge inform the axial shifts/tilts applied to the original; the regenerate button
    modifies the transform; iterate until the corrected slice is right).

    delta[l, f] = depth the reviewer DREW − where the tissue edge sits on the current corrected result (the pane's
    own constrained line, i.e. the served surface carried by the measured move, canvas pad included). A slice
    MARKED accurate contributes delta 0 at every frame (the current edge is right there — it anchors the fit so
    the tilt does not swing those slices). fit_edit_transform turns the deltas into one (shift, tilt) per frame;
    the result is ADDED to oct_params.edit_transform (rounds accumulate), the pane verifications are cleared (they
    described the previous result), and the run applies the accumulated transform as its last rigid move. Returns
    the record for oct_iter.corrected_fold (folded=True so the UI flow is unchanged) or False if nothing to do."""
    import numpy as np
    import nibabel as nib
    cea = eff_params.get("corrected_edge_anchors") or {}
    trusted = {int(x) for x in (trusted_laterals or []) if isinstance(x, (int, float))}
    for _k in ((m.get("oct_params") or {}).get("corrected_accurate") or {}):
        try:
            trusted.add(int(_k))
        except (TypeError, ValueError):
            continue
    if not cea and not trusted:
        return False
    op = dict(m.get("oct_params") or {})
    p = {**oct_mod.DEFAULT_PARAMS, **op}
    try:
        _work = _oct_corrected_vol_path(case_id, m)
        _cvol = np.asarray(nib.load(str(_work)).dataobj)               # (lateral, depth, frame)
        L, depth, F = int(_cvol.shape[0]), int(_cvol.shape[1]), int(_cvol.shape[2])
        cdet = np.asarray(_corrected_detected_surface(case_id, _work, p, _cvol.astype(np.float32)),
                          dtype=np.float64)                            # the pane's line on THIS result
    except Exception as exc:  # noqa: BLE001
        print(f"[corrected-edit] no current corrected line for {case_id}: {type(exc).__name__}", file=sys.stderr)
        return False
    if cdet.shape != (L, F):
        return False
    # THE REVIEWER'S SEMANTICS (2026-09-03, verbatim): "A slice marked as accurate ... merely means that the edge
    # that is detected/edited is CORRECT and this edge can then be used to be PULLED TOWARDS A BETTER FIT TOWARDS A
    # QUADRATIC." So a verified edge (drawn, or marked = the pane's own line) is WHERE THE EDGE IS, and the move it
    # asks for is the one that makes it quadratic: delta[l,f] = (its own deg-2 across frames) - edge[l,f]. NOT
    # "drawn minus the pane line": that mistakes a wrong pane line for a requested move (measured on cs048: it
    # moved the last frames 40 px, stepped 116/128 laterals at 96->97 and raised waviness 0.89 -> 1.24 px).
    deltas: dict = {}
    n_edit = 0
    def _line_to_quad(fr_list, dep_list):
        fr = np.asarray(fr_list, dtype=np.float64); dep = np.asarray(dep_list, dtype=np.float64)
        if fr.size < 8:
            return None
        c = np.polyfit(fr, dep, 2)
        return {int(f): float(qv - dv) for f, qv, dv in zip(fr, np.polyval(c, fr), dep)}
    for l_str, fm in cea.items():
        try:
            l = int(l_str)
        except (TypeError, ValueError):
            continue
        if not (0 <= l < L) or not isinstance(fm, dict):
            continue
        pts = []
        for f_str, d_new in fm.items():
            try:
                f = int(f_str); dn = float(d_new)
            except (TypeError, ValueError):
                continue
            if 0 <= f < F and np.isfinite(dn):
                pts.append((f, dn))
        pts.sort()
        dl = _line_to_quad([q[0] for q in pts], [q[1] for q in pts])
        if dl:
            deltas[l] = dl; n_edit += len(dl)
    for l in sorted(trusted):                                          # marked accurate = the pane's own line is the edge
        if 0 <= l < L and l not in deltas:
            fr = [f for f in range(F) if np.isfinite(cdet[l, f])]
            dl = _line_to_quad(fr, [float(cdet[l, f]) for f in fr])
            if dl:
                deltas[l] = dl
    if not deltas:
        return False
    fit = oct_mod.fit_edit_transform(deltas, L, F, p)
    prev = op.get("edit_transform") if isinstance(op.get("edit_transform"), dict) else None
    shift = np.asarray(fit["shift"], dtype=np.float64); tilt = np.asarray(fit["tilt"], dtype=np.float64)
    rounds = 1
    if prev and isinstance(prev.get("shift"), list) and len(prev["shift"]) == F:
        shift = shift + np.asarray(prev["shift"], dtype=np.float64)
        tilt = tilt + np.asarray(prev.get("tilt") or [0.0] * F, dtype=np.float64)
        rounds = int(prev.get("rounds", 1) or 1) + 1
    hist = list((prev or {}).get("history") or [])[-9:]
    hist.append({"at": int(time.time()), "laterals": sorted(int(k) for k in cea), "marked": sorted(trusted),
                 "n_points": int(n_edit), "fit": {k: fit[k] for k in ("fitted_frames", "resid_px", "shift_range", "tilt_max_px")}})
    et = {"shift": [round(float(v), 3) for v in shift], "tilt": [round(float(v), 3) for v in tilt],
          "rounds": rounds, "history": hist}
    # backup of the previous state (same place the fold keeps its snapshots), then persist
    _bak_rel = None
    try:
        _bdir = orch.case_root(case_id) / "fold_backup"
        _bdir.mkdir(parents=True, exist_ok=True)
        _bak = _bdir / f"pretransform_{int(time.time())}.json"
        _bak.write_text(json.dumps({"saved_at": int(time.time()), "edit_transform": prev,
                                    "corrected_edge_anchors": cea, "trusted_laterals": sorted(trusted)}), encoding="utf-8")
        _bak_rel = _bak.name
        for _o in sorted(_bdir.glob("pretransform_*.json"))[:-10]:
            try:
                _o.unlink()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        return False
    _verified_cleared = dict(op.get("corrected_accurate") or {})
    op["edit_transform"] = et
    op.pop("corrected_edge_anchors", None)
    op.pop("corrected_accurate", None)
    orch.write_manifest_value(case_id, {"oct_params": op})
    m["oct_params"] = op
    eff_params["edit_transform"] = et
    eff_params.pop("corrected_edge_anchors", None)
    eff_params.pop("corrected_accurate", None)
    return {"folded": True, "mode": "transform", "n_points": int(n_edit),
            "laterals": sorted(int(k) for k in cea), "pinned_laterals": sorted(trusted),
            "verified_cleared": {k: {"baseline_px": (v or {}).get("baseline_px"),
                                     "current_px": (v or {}).get("current_px")}
                                 for k, v in _verified_cleared.items()},
            "transform": {"rounds": rounds, "fitted_frames": fit["fitted_frames"], "resid_px": fit["resid_px"],
                          "shift_range": fit["shift_range"], "tilt_max_px": fit["tilt_max_px"],
                          "total_shift_range": [round(float(shift.min()), 1), round(float(shift.max()), 1)]},
            "backup": _bak_rel}


def _fold_corrected_edits_into_border_anchors(case_id: str, m: dict, eff_params: dict,
                                              trusted_laterals=None) -> bool:
    """Reviewer edits/approves the edge on the CORRECTED result → fold them BACK into the ORIGINAL scan's
    border_anchors (the raw GT the warp flattens to), so the NEXT re-run produces a better corrected result.
    This REPLACES the old post-hoc rigid warp on the already-corrected volume (align_corrected_to_smooth /
    apply_sagittal_surface_gt), which corrected the corrected scan itself — the reviewer's explicit spec is to
    alter the ORIGINAL scan's corrections instead.

    The warp is a per-column DEPTH SHIFT, so a depth DELTA the reviewer makes in corrected-output space maps 1:1
    to raw/GT space:   new_GT[l,f] = served[l,f] + (corrected_edit[l,f] - corrected_detected[l,f]).
    REPLACE where a border mark already exists at (l,f), ADD where it doesn't. Approved laterals are PINNED to
    their current served surface (delta 0) so the re-run reproduces them. Updates m['oct_params'] + eff_params in
    place, persists the manifest, invalidates the served-surface caches, and drops the post-hoc corrected-edit
    flags. Returns True if it folded anything (→ caller skips the post-hoc warp and re-runs from the new GT)."""
    import numpy as np
    import nibabel as nib
    cea = eff_params.get("corrected_edge_anchors") or {}
    # TRUSTED = the request's set (what the editor holds this session) UNION the PERSISTED accurate marks
    # (oct_params.corrected_accurate). The marks outlive a reload while the session set does not, so reading
    # only the request meant a reviewer who reloaded saw "5 marked" in the banner and got NONE of them pinned —
    # the protection silently disappeared exactly when they thought it was recorded. Reviewer, 2026-09-02.
    trusted = {int(x) for x in (trusted_laterals or []) if isinstance(x, (int, float))}
    for _k in ((m.get("oct_params") or {}).get("corrected_accurate") or {}):
        try:
            trusted.add(int(_k))
        except (TypeError, ValueError):
            continue
    trusted = sorted(trusted)
    if not cea and not trusted:
        return False
    op = dict(m.get("oct_params") or {})
    p = {**oct_mod.DEFAULT_PARAMS, **op}
    anchors = dict(op.get("border_anchors") or {})
    # served = current GT surface (RAW depth) the warp flattened to — add the reviewer's delta to THIS
    served = _redetect_surface_cached(case_id, m, anchors)
    if served is None:
        try:
            _arr0 = _load_border_vol(_ensure_raw_border_nifti(case_id))
            served = _baseline_surface(case_id, _arr0, p)
        except Exception:  # noqa: BLE001
            return False
    if served is None:
        return False
    served = np.asarray(served, dtype=np.float64)
    L, F = served.shape
    # corrected DETECTED surface (corrected-output depth) — the surface the reviewer edited FROM. NOT generalized
    # (the delta is drawn-line minus the actual tissue edge on the delivered volume), but it MUST be the same
    # baseline the corrected pane drew, so it goes through _corrected_detected_surface: on a scan with a correction
    # curve that is the +-corrected_prior_window re-detection, not the free DP the pane no longer shows.
    try:
        _work = _oct_corrected_vol_path(case_id, m)
        _cvol = np.asarray(nib.load(str(_work)).dataobj)               # (lateral, depth, frame)
        depth = int(_cvol.shape[1])
        cdet = np.asarray(_corrected_detected_surface(case_id, _work, p, _cvol.astype(np.float32)),
                          dtype=np.float64)                            # (lateral, frames)
    except Exception:  # noqa: BLE001
        return False
    if cdet.shape != served.shape:
        return False

    def _clamp(x: float) -> float:
        return float(max(0.0, min(depth - 1.0, x)))

    ba = {str(int(k)): {str(int(ff)): float(dd) for ff, dd in v.items()}
          for k, v in anchors.items() if isinstance(v, dict)}
    n_edit = 0
    for l_str, fm in cea.items():
        try:
            l = int(l_str)
        except (TypeError, ValueError):
            continue
        if not (0 <= l < L) or not isinstance(fm, dict):
            continue
        for f_str, d_new in fm.items():
            try:
                f = int(f_str); dn = float(d_new)
            except (TypeError, ValueError):
                continue
            if not (0 <= f < F) or not np.isfinite(cdet[l, f]) or not np.isfinite(served[l, f]):
                continue
            delta = dn - float(cdet[l, f])                             # correction (corrected space == raw shift)
            ba.setdefault(str(l), {})[str(f)] = int(round(_clamp(served[l, f] + delta)))
            n_edit += 1
    # MARKED-ACCURATE LATERALS ARE EVIDENCE, NOT A FREEZE (reviewer, 2026-09-03, correcting an earlier
    # misreading of the tag): "A slice marked as accurate does not mean that it is a pinned real cornea edge,
    # it merely means that the edge that is detected/edited is correct and this edge can then be used to be
    # pulled towards a better fit towards a quadratic. It is functionally the same as correcting an edge on
    # the corrected slice, just a faster way to do it."
    #
    # So marking certifies the SHAPE of the detected edge, not its POSITION. Writing served[l] back as raw GT
    # for every frame (the old "pin", delta 0) certified the position instead: it made the lateral hand-drawn
    # ground truth the warp must reproduce, which is exactly what stops the rigid move from pulling it onto
    # the quadratic. Measured on cs048 with 8 marks + 1 drawing: off-quadratic at the marked laterals went
    # 9.55 -> 9.52 px mean, i.e. nothing moved, because every marked lateral had been frozen where it was.
    #
    # A marked lateral therefore contributes NO anchor row. What it contributes is trust, carried to the fit
    # through corrected_accurate (see _frame_rigid_minimax's frq_verified_weight), which is why the pops below
    # no longer strip it from this run's params.
    if n_edit == 0 and not trusted:
        return False
    # REVERSIBILITY. This is the one operation that REWRITES the reviewer's own raw GT (1877 hand-drawn
    # points on cs048) and deletes four surface caches, so snapshot what it is about to replace BEFORE
    # touching anything. Written outside border_cache/ because a plain auto re-run rmtree's that directory.
    _bak_rel = None
    try:
        _bdir = orch.case_root(case_id) / "fold_backup"
        _bdir.mkdir(parents=True, exist_ok=True)
        _bak = _bdir / f"prefold_{int(time.time())}.json"
        _bak.write_text(json.dumps({
            "saved_at": int(time.time()),
            "border_anchors": anchors,                      # the pre-fold raw GT, verbatim
            "corrected_edge_anchors": cea,                  # what was folded in
            "trusted_laterals": trusted,
            "border_generalize": op.get("border_generalize"),
            "border_guided": op.get("border_guided"),
        }), encoding="utf-8")
        _bak_rel = _bak.name
        _olds = sorted(_bdir.glob("prefold_*.json"))        # keep the last 10 folds, drop older snapshots
        for _o in _olds[:-10]:
            try:
                _o.unlink()
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001 — a fold with no backup is not worth the risk
        return False
    _n_lat_before = len(anchors)
    _n_pts_before = sum(len(v) for v in anchors.values() if isinstance(v, dict))
    op["border_anchors"] = ba
    op.pop("corrected_edge_anchors", None)                            # folded into the raw GT → no post-hoc warp
    # CLEAR THE CORRECTED-SCAN STATE. The reviewer's loop (2026-09-02) is: verify edges on the corrected scan
    # (by DRAWING them or by MARKING them accurate — both are verifications), regenerate the corrected scan from
    # them, then start again on the scan just produced. Both kinds of verification refer to the OLD corrected
    # surface, so carrying them forward would have the reviewer looking at stale judgements about a volume that
    # no longer exists. They are absorbed here (drawn depths into the GT, marked slices pinned to the surface
    # they vouched for) and their readings are kept in the run record rather than in live state.
    _verified_cleared = dict(op.get("corrected_accurate") or {})
    _verified_for_fit = {str(int(_l)): (_verified_cleared.get(str(int(_l))) or {}) for _l in trusted}
    op.pop("corrected_accurate", None)
    # GENERALIZE ACROSS LATERALS (reviewer, 2026-09-01: "make the fold generalize across laterals").
    # Why this is not optional: the flatten applies ONE rigid depth shift per frame, and that shift is the
    # MEDIAN across the ~360 laterals of the central band (_frame_common_shift). A fold writes anchors only
    # at the laterals the reviewer drew — measured on cs048, 10 folded laterals moved that median by max
    # 0.37 px / mean 0.04 px while the drawn correction was median 12.5 px, i.e. ~3% of it reached the
    # result and the corrected image did not visibly change. Ten samples cannot move a 360-sample median.
    # So the fold now LEARNS the per-frame residual field from the new GT and interpolates it across ALL
    # laterals (the same generalize_surface machinery the "⤢ Generalize to whole volume" button uses,
    # held-out 1.3 px), which is what makes a few corrected slices reach the delivered volume.
    # pin_anchors inside _compute_generalize_cache keeps the drawn frames EXACT, so generalizing never
    # softens the reviewer's own line — it only fills in the laterals they did not draw.
    op["border_generalize"] = True
    op.pop("border_guided", None)          # guided RE-DETECTION is a separate guarded thing; serve the GT
    orch.write_manifest_value(case_id, {"oct_params": op})
    m["oct_params"] = op
    eff_params["border_anchors"] = ba
    # KEEP the verified sets on THIS run's params. They are what tells the rigid fit which laterals' edges the
    # reviewer has certified (frq_verified_weight), and the fold is precisely the run that should use them —
    # popping them here meant the one run that was supposed to act on the reviewer's verification was the one
    # run that could not see it. They are still cleared from the PERSISTED oct_params above, so the next round
    # starts from the scan just produced (reviewer's step 3).
    # corrected_edge_anchors stays POPPED: those edits are now raw GT anchors, and leaving them in params would
    # also arm the post-hoc apply_sagittal_surface_gt warp, applying the same correction twice.
    eff_params.pop("corrected_edge_anchors", None)
    if _verified_for_fit:
        eff_params["corrected_accurate"] = _verified_for_fit
    eff_params["border_generalize"] = True
    eff_params.pop("border_guided", None)
    _bc = orch.case_root(case_id) / "border_cache"                    # served surface changed → drop its caches
    for _nm in ("redetect.npz", "generalize.npz", "guided.npz", "provided_edges.npz"):
        try:
            (_bc / _nm).unlink()
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass
    # Rebuild generalize.npz from the NEW anchors, in this request, so the re-run that follows serves the
    # generalized surface rather than recomputing it lazily from a half-updated cache. If it fails the fold
    # still stands: the flag is cleared so the scan falls back to the dense re-detect of the new GT (the old
    # local behaviour) instead of a stale or missing generalized surface.
    _gen_ok = False
    try:
        _gs = _compute_generalize_cache(case_id, {**m, "oct_params": op}, ba)
        _gen_ok = getattr(_gs, "shape", None) is not None
    except Exception:  # noqa: BLE001
        _gen_ok = False
    if not _gen_ok:
        op.pop("border_generalize", None)
        eff_params.pop("border_generalize", None)
        orch.write_manifest_value(case_id, {"oct_params": op})
        m["oct_params"] = op
    return {"folded": True, "generalized": bool(_gen_ok), "n_points": n_edit,
            "laterals": sorted(int(k) for k in cea),
            "pinned_laterals": sorted(trusted),
            "verified_cleared": {k: {"baseline_px": (v or {}).get("baseline_px"),
                                     "current_px": (v or {}).get("current_px")}
                                 for k, v in _verified_cleared.items()},
            "anchors_before": {"laterals": _n_lat_before, "points": _n_pts_before},
            "anchors_after": {"laterals": len(ba), "points": sum(len(v) for v in ba.values())},
            "backup": _bak_rel}


@app.get("/api/case/{case_id}/oct-axial-slice")
def oct_axial_slice_png(case_id: str, frame: int = 0) -> Response:
    """The Fix-axial editor's B-scan (a FIXED FRAME) at NATIVE voxel resolution (depth rows × lateral cols,
    depth 0 = TOP) as a grayscale PNG, from the CORRECTED preprocessed-output volume (what the user is fixing).
    Mirrors oct-border-slice's 1-99 percentile stretch + no-store. array lateral 0 = column 0; the frontend
    scaleX(-1)-flips it so lateral 0 renders on the VISUAL RIGHT (matches niivue's L/R flip)."""
    import io
    import numpy as np
    import nibabel as nib
    from PIL import Image
    m = orch.read_manifest(case_id)
    work = _oct_corrected_vol_path(case_id, m)
    if not work.exists():
        raise HTTPException(400, f"Case {case_id} is not preprocessed yet.")
    try:
        vol = np.asarray(nib.load(str(work)).dataobj)            # (lateral, depth, frame)
        nF = int(vol.shape[2])
        f = max(0, min(nF - 1, int(frame)))
        sl = np.ascontiguousarray(vol[:, :, f].T).astype(np.float32)   # (depth, lateral), depth 0 = TOP
        finite = sl[np.isfinite(sl)]
        if finite.size:
            lo = float(np.percentile(finite, 1)); hi = float(np.percentile(finite, 99))
            if hi <= lo:
                hi = lo + 1.0
            gray = (np.clip((sl - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros(sl.shape, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(gray, mode="L").save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT axial slice failed: {exc}")


@app.post("/api/case/{case_id}/oct-axial-curve")
def oct_axial_curve(case_id: str, req: OctPreprocessRequest) -> dict:
    """The detected anterior corneal surface across LATERALS for ONE frame of the CORRECTED volume (+ a robust
    quadratic fit) so the Fix-axial UI draws + drags the border. req.slice_index = the FRAME index (central if
    None). Coordinates: edge[lateral] = depth (0 = TOP), aligning with oct-axial-slice's (depth, lateral) image."""
    import nibabel as nib  # noqa: F401 — used by _axial_surface_cached
    import numpy as np
    m = orch.read_manifest(case_id)
    work = _oct_corrected_vol_path(case_id, m)
    if not work.exists():
        raise HTTPException(400, f"Case {case_id} is not preprocessed yet.")
    try:
        p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        surf, (L, D, nF) = _axial_surface_cached(case_id, work, p)   # (lateral, frames)
        f = nF // 2 if req.slice_index is None else max(0, min(nF - 1, int(req.slice_index)))
        edge = np.asarray(surf[:, f], dtype=np.float32)
        fit = oct_mod._fit_quadratic_ransac(edge, float(p["residual_threshold"]))
        return {"n_lateral": int(L), "depth_vox": int(D), "n_frames": int(nF), "frame": int(f),
                "edge": [float(v) for v in edge], "fit": [float(v) for v in fit]}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT axial curve failed: {exc}")


@app.post("/api/case/{case_id}/oct-axial-redetect")
def oct_axial_redetect(case_id: str, req: OctPreprocessRequest) -> dict:
    """Fix-axial "Confirm": persist the user's axial-surface anchors as STICKY GT (oct_params.axial_anchors) so a
    later Run re-applies them (and the correction is recorded for detector-learning). NO warp + NO cache here — the
    warp is the post-hoc apply_axial_surface_gt pass inside preprocess_oct_to_nifti. Empty anchors clear it (revert
    to auto). {str(frame): {str(lateral): true_depth}} in corrected-output depth space."""
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    anchors = req.axial_anchors if isinstance(req.axial_anchors, dict) else {}
    try:
        op = dict(m.get("oct_params") or {})
        if anchors:
            op["axial_anchors"] = anchors
        else:
            op.pop("axial_anchors", None)
        orch.write_manifest_value(case_id, {"oct_params": op})
        n_anchors = sum(len(v) for v in anchors.values() if isinstance(v, dict))
        return {"ok": True, "n_anchors": int(n_anchors)}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT axial re-detect failed: {exc}")


# ── CORRECTED-RESULT sagittal fix-tool: edit the anterior surface on the CORRECTED output in the SAGITTAL plane
#    (fixed LATERAL, across FRAMES) — the before/after "corrected" pane. Edge detection is cleaner on the flattened
#    result, and this reaches residual inter-frame drift that editing the raw GT cannot (measured gain 0.055). Confirm
#    persists sticky corrected_edge_anchors; Run re-applies them via the post-hoc apply_sagittal_surface_gt warp. The
#    trio mirrors the axial fix-tool but reslices sagittally (surf[idx,:] across frames vs surf[:,f] across laterals).
@app.get("/api/case/{case_id}/oct-corrected-slice")
def oct_corrected_slice_png(case_id: str, slice_index: int = 0) -> Response:
    """The corrected-result B-scan for ONE sagittal slice (a FIXED LATERAL) at NATIVE voxel resolution
    (depth rows × frame cols, depth 0 = TOP) as a grayscale PNG, from the CORRECTED preprocessed-output volume.
    Mirrors oct-border-slice (same 1-99 stretch, no-store) but sourced from the corrected output, not the raw
    input — so the editable line drawn over it lands on the SAME grid the corrected surface was detected on."""
    import io
    import numpy as np
    import nibabel as nib
    from PIL import Image
    m = orch.read_manifest(case_id)
    work = _oct_corrected_vol_path(case_id, m)
    if not work.exists():
        raise HTTPException(400, f"Case {case_id} is not preprocessed yet.")
    try:
        vol = np.asarray(nib.load(str(work)).dataobj)            # (lateral, depth, frame)
        n = int(vol.shape[0])
        idx = max(0, min(n - 1, int(slice_index)))
        sl = np.ascontiguousarray(vol[idx]).astype(np.float32)   # (depth, frames), depth 0 = TOP
        finite = sl[np.isfinite(sl)]
        if finite.size:
            lo = float(np.percentile(finite, 1)); hi = float(np.percentile(finite, 99))
            if hi <= lo:
                hi = lo + 1.0
            gray = (np.clip((sl - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            gray = np.zeros(sl.shape, dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(gray, mode="L").save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png", headers={"Cache-Control": "no-store"})
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT corrected slice failed: {exc}")


@app.post("/api/case/{case_id}/oct-corrected-curve")
def oct_corrected_curve(case_id: str, req: OctPreprocessRequest) -> dict:
    """The detected anterior surface across FRAMES for ONE sagittal slice (a FIXED LATERAL) of the CORRECTED
    volume (+ a robust quadratic fit) so the before/after corrected pane draws + drags the border. req.slice_index
    = the LATERAL index (central if None). Coordinates: edge[frame] = depth (0 = TOP), aligning with the
    oct-corrected-slice (depth, frames) image and the fix-columns sagittal editor. Reuses the SAME cached corrected
    surface (surf(lateral,frames)) the warp's apply_sagittal_surface_gt re-diffs against, so preview ≈ result."""
    import nibabel as nib  # noqa: F401 — used by _axial_surface_cached
    import numpy as np
    m = orch.read_manifest(case_id)
    work = _oct_corrected_vol_path(case_id, m)
    if not work.exists():
        raise HTTPException(400, f"Case {case_id} is not preprocessed yet.")
    try:
        p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
        surf, (L, D, nF) = _axial_surface_cached(case_id, work, p)   # (lateral, frames)
        idx = L // 2 if req.slice_index is None else max(0, min(L - 1, int(req.slice_index)))
        edge = np.asarray(surf[idx, :], dtype=np.float32)            # across FRAMES for this lateral slice
        fit = oct_mod._fit_quadratic_ransac(edge, float(p["residual_threshold"]))
        # NOTE: no pin. The corrected surface shown is the REAL re-detected surface of the (guarded) rigid axial
        # correction applied inside preprocess (apply_sagittal_surface_gt). The reviewer sees the true result —
        # fixed where a rigid move helped, honestly unchanged where it was declined — never a painted line.
        # TWO LINES (reviewer, 2026-08-25: "draw both lines, red detection and my target in another colour").
        #   edge   = the constrained DETECTION, un-generalized — where the tissue reads inside ±window of the
        #            correction curve. This is EXACTLY the array the three corrections-path warp stages diff
        #            against, so what the reviewer approves here is what a Run consumes.
        #   target = the same line after generalize_corrected_surface + the drawn-anchor re-pin, i.e. where the
        #            reviewer's own corrected-edge drawing says the surface should be. Drawn in its own colour.
        # They coincide on any scan with no corrected_edge_anchors (target omitted then, so the pane draws one
        # line). warp_gap quantifies the difference per slice rather than leaving it to be discovered.
        _c = _AXIAL_SURF_CACHE.get(case_id)
        _gp = _c[4][idx] if (_c and len(_c) > 4 and _c[4] is not None and idx < len(_c[4])) else None
        _det = _c[5][idx] if (_c and len(_c) > 5 and _c[5] is not None and idx < len(_c[5])) else None
        _res = {"slices": int(L), "index": int(idx), "depth_vox": int(D), "n_frames": int(nF),
                "edge": [float(v) for v in (edge if _det is None else _det)],
                "fit": [float(v) for v in fit],
                "warp_gap": (None if _gp is None else
                             {"median": round(float(np.median(_gp)), 3), "max": round(float(np.max(_gp)), 3)})}
        if _det is not None and _gp is not None and float(np.max(_gp)) > 1e-6:
            _res["target"] = [float(v) for v in edge]     # the generalize + anchor-pinned line
        return _res
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT corrected curve failed: {exc}")


def _corrected_offquad_rms(case_id: str, m: dict):
    """Per-lateral RMS distance from the CORRECTED surface to its OWN deg-2 least-squares fit across frames.

    This is exactly the red-vs-blue gap the corrected pane draws (reviewer, 2026-09-01: "the closeness of the
    red line and the blue curve on the corrected slice can be used as a measure of how good the correction"),
    computed on the same cached corrected surface the pane reads, so the number and the picture agree.

    Honest limit: where the reviewer has NOT drawn, the red line is a detection, so this scores the detector
    as much as the correction. It is a tracking signal for improvement between runs, never an adjudication of
    where the true edge is — only a drawn line does that. Returns (rms[L], L, n_frames)."""
    import nibabel as nib  # noqa: F401 — used by _axial_surface_cached
    import numpy as np
    work = _oct_corrected_vol_path(case_id, m)
    if not work.exists():
        raise HTTPException(400, f"Case {case_id} is not preprocessed yet.")
    p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
    surf, (L, D, nF) = _axial_surface_cached(case_id, work, p)      # (lateral, frames)
    S = np.asarray(surf, dtype=np.float64)
    f = np.arange(nF, dtype=np.float64)
    rms = np.full(L, np.nan)
    for l in range(L):
        y = S[l, :]
        ok = np.isfinite(y) & (y > 0) & (y < D - 1)
        if int(ok.sum()) < 8:
            continue
        try:
            cf = np.polyfit(f[ok], y[ok], 2)
        except Exception:  # noqa: BLE001 — a degenerate lateral simply gets no score
            continue
        rms[l] = float(np.sqrt(np.mean((y[ok] - np.polyval(cf, f[ok])) ** 2)))
    return rms, int(L), int(nF)


@app.post("/api/case/{case_id}/oct-corrected-marks")
def oct_corrected_marks(case_id: str, req: OctPreprocessRequest) -> dict:
    """DEFECT MARKS drawn on the CORRECTED result: {"<lateral>": [[frame_lo, frame_hi], ...]}.

    The existing ⚑ Mark tool paints on the ORIGINAL scan, in raw geometry. This is its corrected-space sibling:
    the reviewer marks a region of the CORRECTED image that they judge could still improve — a place to look,
    expressed on the picture they are actually judging, in frames of that lateral.

    Deliberately separate from manifest.defect_marks (raw/niivue-canonical indices, wiped by a re-preprocess).
    These are stored in oct_params.corrected_defect_marks, in CORRECTED frame coordinates, and they gate nothing:
    no voxel changes, no fit excludes them. They exist so a human can point at something and be understood."""
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    raw = ((req.params or {}) if isinstance(req.params, dict) else {}).get("marks")
    out: dict = {}
    if isinstance(raw, dict):
        for k, bands in raw.items():
            try:
                lat = int(k)
            except (TypeError, ValueError):
                continue
            keep = []
            for b in (bands or []):
                try:
                    a, z = int(b[0]), int(b[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if z < a:
                    a, z = z, a
                keep.append([a, z])
            if keep:
                out[str(lat)] = sorted(keep)
    try:
        op = dict(m.get("oct_params") or {})
        if out:
            op["corrected_defect_marks"] = out
        else:
            op.pop("corrected_defect_marks", None)
        orch.write_manifest_value(case_id, {"oct_params": op})
        return {"ok": True, "marks": out,
                "n_bands": sum(len(v) for v in out.values()), "n_slices": len(out)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT corrected marks failed: {exc}")


@app.post("/api/case/{case_id}/oct-corrected-accurate")
def oct_corrected_accurate(case_id: str, req: OctPreprocessRequest) -> dict:
    """Mark sagittal slices whose CORRECTED edge the reviewer judges ACCURATE, and track them across re-runs.

    The request carries the FULL marked set (corrected_trusted_laterals); it replaces what is stored. For each
    slice the red-vs-blue gap is measured now: the FIRST time a slice is marked that value is kept as its
    BASELINE, and every later read reports baseline vs current — so after a re-run the reviewer can see whether
    the slices they vouched for got better or worse, on the same measure they were judging by eye.

    Persisted in oct_params.corrected_accurate so it survives a reload (the old session-only trusted-slice set
    did not). Marking is a JUDGEMENT, not a correction: nothing here changes a voxel or a surface."""
    import math
    m = orch.read_manifest(case_id)
    want = sorted({int(x) for x in (req.corrected_trusted_laterals or []) if isinstance(x, (int, float))})
    try:
        rms, L, _nF = _corrected_offquad_rms(case_id, m)
        op = dict(m.get("oct_params") or {})
        prev = dict(op.get("corrected_accurate") or {})
        out: dict = {}
        for l in want:
            if not (0 <= l < L):
                continue
            cur = float(rms[l]) if math.isfinite(float(rms[l])) else None
            was = prev.get(str(l)) or {}
            out[str(l)] = {"baseline_px": was.get("baseline_px", cur),   # first mark wins → the comparison point
                           "current_px": cur,
                           "at": int(was.get("at") or time.time())}
        if out:
            op["corrected_accurate"] = out
        else:
            op.pop("corrected_accurate", None)
        orch.write_manifest_value(case_id, {"oct_params": op})
        return {"ok": True, "accurate": out}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT corrected accurate-mark failed: {exc}")


@app.post("/api/case/{case_id}/oct-corrected-suggest")
def oct_corrected_suggest(case_id: str, req: OctPreprocessRequest) -> dict:
    """Slices worth CORRECTING on the CORRECTED result — the corrected-mode counterpart of the determinism
    prompt (which is about the ORIGINAL line and is hidden in this mode).

    Score per lateral = RMS distance from that lateral's corrected surface to its OWN deg-2 least-squares fit
    across frames. That is the reviewer's stated quality criterion (a corrected anterior surface should be a
    smooth quadratic), computed on the same cached corrected surface the pane draws — no extra detection pass.

    It is a WHERE-TO-LOOK ranking, not a claim the edge is wrong: the surface it scores is a detection, and this
    project's rule is that only the reviewer's drawn line adjudicates the true edge. The wording in the UI says
    "check", never "wrong".

    Picks are STRATIFIED across the central band rather than taken greedily: adjacent laterals score almost
    identically, so a global top-N degenerates into one contiguous block (measured: median |delta| 0.083 px
    between neighbours), and picks outside the band extrapolate badly. Laterals already drawn — or within
    `near` of one — are skipped so the queue advances instead of re-suggesting the same place."""
    import nibabel as nib  # noqa: F401 — used by _axial_surface_cached
    import numpy as np
    m = orch.read_manifest(case_id)
    work = _oct_corrected_vol_path(case_id, m)
    if not work.exists():
        raise HTTPException(400, f"Case {case_id} is not preprocessed yet.")
    _rp = (req.params or {}) if isinstance(req.params, dict) else {}
    n_pick = max(1, min(24, int(_rp.get("n", 8))))
    near = max(0, int(_rp.get("near", 20)))
    try:
        rms, L, nF = _corrected_offquad_rms(case_id, m)   # the red-vs-blue gap, per lateral
        drawn = sorted(int(k) for k in ((m.get("oct_params") or {}).get("corrected_edge_anchors") or {}))
        # VERIFIED-REAL DEVIATION, learned from the slices the reviewer marked ACCURATE (reviewer, 2026-09-02:
        # "marking as accurate should inform the next round of corrections because you have an accurate edge to
        # compare against the curve").
        # WHAT THE MARK MEANS (reviewer, 2026-09-02, and this is the load-bearing definition): "the DETECTED edge
        # on that slice is accurate, EVEN IF NOT SMOOTH". So on a marked slice the gap to the quadratic is REAL
        # surface shape, not detector error — it is the deviation that SHOULD be there and must not be flattened.
        # That makes the marks a position-dependent reference for what real looks like: on cs048 they span 2.45 to
        # 10.03 px, because a quadratic fits the steep peripheral limbus less well than the centre. The level is
        # INTERPOLATED across laterals between the marks (held flat beyond the outermost) and a slice is only
        # suggested when it exceeds the local verified-real deviation by `margin` — i.e. when it deviates by more
        # than the reviewer has certified as genuine at that position. With no marks this is inert and the raw
        # ranking applies. It is NOT an "acceptable smoothness" threshold; nothing here judges smoothness.
        _acc_lats = []
        for _k in ((m.get("oct_params") or {}).get("corrected_accurate") or {}):
            try:
                _l = int(_k)
            except (TypeError, ValueError):
                continue
            if 0 <= _l < L and np.isfinite(rms[_l]):
                _acc_lats.append(_l)
        _acc_lats = sorted(set(_acc_lats))
        margin = float(_rp.get("margin", 1.0))          # px a slice must EXCEED the accepted level by
        if _acc_lats:
            ref = np.interp(np.arange(L, dtype=float), np.array(_acc_lats, dtype=float),
                            np.array([rms[l] for l in _acc_lats], dtype=float))
            score = rms - ref                           # deviation BEYOND what they verified as real here
        else:
            ref = np.full(L, np.nan)
            score = rms.copy()                          # no reference yet → rank on the raw gap
        lo, hi = int(0.2 * L), int(0.8 * L)                             # central band: the edges extrapolate badly
        elig = np.zeros(L, bool)
        elig[lo:hi] = True
        elig &= np.isfinite(score)
        if _acc_lats:
            elig &= (score > margin)      # nothing deviates beyond the verified-real level → no pick
        for d in drawn:
            elig[max(0, d - near):min(L, d + near + 1)] = False
        for a in _acc_lats:
            elig[a] = False               # a slice they called accurate is not work
        picks: list[dict] = []
        if elig.any():
            edges = np.linspace(lo, hi, n_pick + 1).astype(int)
            for a, b in zip(edges[:-1], edges[1:]):
                if b <= a:
                    continue
                seg = np.arange(a, b)[elig[a:b]]
                if seg.size == 0:
                    continue
                best = int(seg[int(np.argmax(score[seg]))])
                picks.append({"lateral": best, "rms_px": round(float(rms[best]), 2),
                              "over_px": (round(float(score[best]), 2) if _acc_lats else None),
                              "ref_px": (round(float(ref[best]), 2) if _acc_lats else None)})
        # ALREADY-DRAWN laterals ride back with their scores so the queue can show them as DONE instead of
        # silently dropping them. Without this the reviewer loses the count across a reload: a slice they
        # corrected simply disappears from the row, which reads as "it was never suggested".
        done = [{"lateral": int(d), "rms_px": (round(float(rms[d]), 2) if (0 <= d < L and np.isfinite(rms[d])) else None)}
                for d in drawn[:12]]
        # ACCURACY MARKS ride back with a FRESH current reading, so the banner can show baseline -> now for
        # every slice the reviewer vouched for. That is the "did the re-run improve it?" signal.
        _acc_stored = ((m.get("oct_params") or {}).get("corrected_accurate") or {})
        accurate = {}
        for _k, _v in _acc_stored.items():
            try:
                _l = int(_k)
            except (TypeError, ValueError):
                continue
            _now = (round(float(rms[_l]), 2) if (0 <= _l < L and np.isfinite(rms[_l])) else None)
            accurate[str(_l)] = {"baseline_px": (_v or {}).get("baseline_px"), "current_px": _now}
        _fin = rms[np.isfinite(rms)]
        return {"slices": int(L), "band": [lo, hi], "n_frames": int(nF),
                "ref_from_marks": bool(_acc_lats), "n_marks": len(_acc_lats), "margin_px": margin,
                "all_within_accurate": bool(_acc_lats) and not picks,
                "drawn": drawn, "picks": picks, "done": done, "accurate": accurate,
                "median_rms_px": (round(float(np.median(_fin)), 2) if _fin.size else None),
                "p90_rms_px": (round(float(np.percentile(_fin, 90)), 2) if _fin.size else None)}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT corrected suggest failed: {exc}")


@app.post("/api/case/{case_id}/oct-corrected-redetect")
def oct_corrected_redetect(case_id: str, req: OctPreprocessRequest) -> dict:
    """CORRECTED-result sagittal fix-tool "Confirm": persist the user's corrected-surface anchors as STICKY GT
    (oct_params.corrected_edge_anchors) so a later Run re-applies them. NO warp + NO cache here — the warp is the
    post-hoc apply_sagittal_surface_gt pass inside preprocess_oct_to_nifti. Empty anchors clear it (revert to
    auto). {str(lateral): {str(frame): true_depth}} in CORRECTED-output depth space (0 = TOP)."""
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    anchors = req.corrected_edge_anchors if isinstance(req.corrected_edge_anchors, dict) else {}
    try:
        op = dict(m.get("oct_params") or {})
        if anchors:
            op["corrected_edge_anchors"] = anchors
        else:
            op.pop("corrected_edge_anchors", None)
        # DRAWING A CORRECTED EDGE IS ALSO VOUCHING FOR IT (reviewer, 2026-09-03: "corrected edge corrections
        # should automatically be considered as marked accurate"). A slice the reviewer has just drawn is, by
        # definition, one whose surface they have decided — so it joins corrected_accurate without them having
        # to press the second button. That makes the two paths identical everywhere the mark is read: the
        # queue's reference level, the tracking readout, the fold's pinning, and the fit's trust weight.
        # baseline_px is left null here — this endpoint is the ~900 ms autosave and must not trigger a cold
        # surface detection; /oct-corrected-accurate and /oct-corrected-suggest fill the reading in later.
        _acc = dict(op.get("corrected_accurate") or {})
        for _k in anchors:
            if str(_k) not in _acc:
                _acc[str(_k)] = {"baseline_px": None, "current_px": None, "at": int(time.time()),
                                 "from": "drawn"}
        for _k in [k for k in _acc if (_acc[k] or {}).get("from") == "drawn" and str(k) not in anchors]:
            _acc.pop(_k, None)          # a cleared drawing withdraws the mark it implied
        if _acc:
            op["corrected_accurate"] = _acc
        else:
            op.pop("corrected_accurate", None)
        orch.write_manifest_value(case_id, {"oct_params": op})
        n_anchors = sum(len(v) for v in anchors.values() if isinstance(v, dict))
        return {"ok": True, "n_anchors": int(n_anchors), "accurate": sorted(int(k) for k in _acc)}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT corrected re-detect failed: {exc}")


@app.post("/api/case/{case_id}/oct-border-generalize")
def oct_border_generalize(case_id: str, req: OctPreprocessRequest) -> dict:
    """GENERALIZE the confirmed fix-columns corrections to the WHOLE volume: learn the systematic per-frame
    correction (residual vs auto) from the anchored slices and interpolate it across ALL slices
    (oct_mod.generalize_surface). Sets oct_params['border_generalize'] + computes generalize.npz, so BOTH the
    scrub preview (oct-border-curves-all) AND a subsequent fix-columns Run/Approve use the generalized surface.
    Reversible via oct-border-generalize/discard. Requires anchors already Confirmed."""
    m = orch.read_manifest(case_id)
    if not (m.get("input_volume") or m.get("corrected_volume")):
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    anchors = (m.get("oct_params") or {}).get("border_anchors") or {}
    if not anchors:
        raise HTTPException(400, "No confirmed border corrections to generalize — Confirm some anchors first.")
    try:
        op = dict(m.get("oct_params") or {})
        op["border_generalize"] = True
        orch.write_manifest_value(case_id, {"oct_params": op})
        surf = _compute_generalize_cache(case_id, {**m, "oct_params": op}, anchors)
        n_anchors = sum(len(v) for v in anchors.values() if isinstance(v, dict))
        return {"ok": True, "n_anchors": int(n_anchors), "n_slices": int(getattr(surf, "shape", [0])[0])}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT generalize failed: {exc}")


def _guided_cache_path(case_id: str) -> Path:
    return orch.case_root(case_id) / "border_cache" / "guided.npz"


@app.post("/api/case/{case_id}/oct-border-guided")
def oct_border_guided(case_id: str) -> dict:
    """GUARDED GUIDED RE-DETECTION — use the reviewer's corrections to improve DETECTION on this scan.

    The correction supplies a PRIOR (where to look); the detector still decides where the edge is, on every
    slice, from the image. That is the difference between improving detection and spreading a drawn line: on a
    held-out corrected slice, interpolation returned a surface identical to auto, while this cut the median
    error from 11.1 px to 3.0 px.

    GUARDED, because unconditionally it is a coin flip. Across four approved scans it improved the delivered
    volume on two (-20.4%, -1.8% shift-roughness) and worsened it on two (+21.1%, +11.0%) — a +2.5% mean that
    hides both. So both surfaces are measured and the better one is kept:

      * WITH corrections, the reviewer's anchors decide it. That is real ground truth and beats any proxy.
      * WITHOUT, gradient@surface must improve AND the delivered per-frame shift must not get rougher. Two
        signals, because a wandering search raises the first on its own by finding other structures.

    Losing leaves the scan exactly as it was, and says so."""
    m = orch.read_manifest(_require_case(case_id))
    work = m.get("input_volume") or m.get("corrected_volume")
    if not work or not Path(str(work)).exists():
        raise HTTPException(400, f"Case {case_id} has no working volume.")
    src = m.get("oct_source")
    if not src or not Path(str(src)).exists():
        raise HTTPException(400, "Raw .OCT not available — guided re-detection reads the raw volume.")
    p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
    anchors = (m.get("oct_params") or {}).get("border_anchors") or {}
    try:
        sag = oct_mod.reformat_to_sagittal(
            oct_mod.read_oct_zstack(str(src), int(m.get("oct_volume_index", 0) or 0))).astype("float32")
        depth = int(sag.shape[1])
        auto = oct_mod.detect_surface_all(sag, p, workers=max(2, oct_mod.auto_workers() // 2))
        # The PRIOR: the corrections carried across the volume where they exist, else plain auto. Even a weak
        # prior is useful — the gain above was measured with the prior equal to auto.
        prior = auto
        if anchors:
            try:
                prior = oct_mod.generalize_surface(sag, anchors, p, baseline=auto)
            except Exception:  # noqa: BLE001 — a failed generalize just means a weaker prior, not a failure
                prior = auto
        guided = oct_mod.guided_redetect_all(sag, prior, p)

        ga, gg = (oct_mod.surface_gradient_score(sag, auto), oct_mod.surface_gradient_score(sag, guided))
        sa, sg = (oct_mod.shift_roughness(auto), oct_mod.shift_roughness(guided))
        ea, na = oct_mod.anchor_error(auto, anchors, depth)
        eg, _n = oct_mod.anchor_error(guided, anchors, depth)

        if na >= 8 and math.isfinite(ea) and math.isfinite(eg):
            accept = eg < ea
            why = (f"reviewer's anchors: median error {ea:.1f} -> {eg:.1f} px on {na} point(s)")
        else:
            accept = (gg > ga) and (sg <= sa * 1.02)
            why = (f"gradient {ga:.0f} -> {gg:.0f} ({100*(gg/ga-1):+.1f}%), "
                   f"delivered shift-roughness {sa:.3f} -> {sg:.3f} ({100*(sg/sa-1):+.1f}%)")

        op = dict(m.get("oct_params") or {})
        if accept:
            # PIN drawn frames to the reviewer's exact line BEFORE caching. The guard judged the UN-pinned
            # guided (a fair test of whether the detector's own search found the drawn edge); but what gets
            # delivered must honor the anchors exactly at the frames drawn — guided's 40 px search otherwise
            # lands well off a manifestly-correct line. Non-drawn frames keep guided's from-image detection.
            oct_mod.pin_anchors(guided, anchors, depth, float(p.get("crop_max_pad", 120)))
            cp = _guided_cache_path(case_id)
            cp.parent.mkdir(parents=True, exist_ok=True)
            tmp = cp.with_name("guided.tmp.npz")
            np.savez_compressed(tmp, surface=guided.astype(np.float32),
                                anchors_sig=_border_anchors_sig(anchors),
                                raw_mtime=str(Path(str(work)).stat().st_mtime),
                                params_sig=_detect_params_sig(p))
            os.replace(tmp, cp)
            op["border_guided"] = True
        else:
            op.pop("border_guided", None)
            _guided_cache_path(case_id).unlink(missing_ok=True)
        orch.write_manifest_value(case_id, {"oct_params": op})
        return {"ok": True, "accepted": bool(accept), "why": why, "judged_on": ("anchors" if na >= 8 else "proxies"),
                "gradient": {"auto": round(ga, 1), "guided": round(gg, 1)},
                "shift_roughness": {"auto": round(sa, 4), "guided": round(sg, 4)},
                "anchor_error": {"auto": (round(ea, 2) if math.isfinite(ea) else None),
                                 "guided": (round(eg, 2) if math.isfinite(eg) else None), "n": na}}
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Guided re-detection failed: {exc}")


@app.post("/api/case/{case_id}/oct-border-generalize/discard")
def oct_border_generalize_discard(case_id: str) -> dict:
    """Revert the whole-volume generalization: clear the border_generalize flag + delete generalize.npz, so
    scrub + Run go back to the local-band redetect (the user's per-slice anchors are untouched)."""
    m = orch.read_manifest(case_id)
    op = dict(m.get("oct_params") or {})
    op.pop("border_generalize", None)
    orch.write_manifest_value(case_id, {"oct_params": op})
    _generalize_cache_path(case_id).unlink(missing_ok=True)
    return {"ok": True}


@app.post("/api/case/{case_id}/oct-smooth-corrected")
def oct_smooth_corrected_case(case_id: str, req: OctPreprocessRequest) -> dict:
    """SMOOTH the already manually-corrected volume. The fix-columns Run warps to the corrected surface via
    provided_edges with inter-slice smoothing DISABLED (oct_preprocess.py:2934) so the exact drag is honoured —
    which leaves residual slice-to-slice jitter. This applies the guarded post-hoc smoothing passes to the
    CORRECTED output (axial_consistency → [surface_refine_2d if enabled] → frame_boundary_lat_smooth), in the
    same order the auto pipeline uses, MINUS axial_refine (which re-flattens whole frames to a fresh auto
    quadratic → would undo the manual correction). The included passes are HARD-GATED local nudges: each
    re-detects the CURRENT surface, builds a locally-smoothed target of it, and moves only columns whose
    deviation exceeds a small threshold, capped small — a strict no-op on already-smooth regions, so the manual
    corrections are preserved within those caps. Runs on the already-written corrected nifti ONLY (never
    re-reads the raw .OCT). Drops segmentation (geometry shifted); keeps the anchors + vetting + GT."""
    import numpy as _np
    import shutil as _sh2
    m = orch.read_manifest(case_id)
    src = m.get("oct_source")
    if not m.get("oct_preprocessed") or not src or not Path(src).exists():
        raise HTTPException(400, "Smoothing needs a preprocessed scan.")
    if not ((m.get("oct_params") or {}).get("border_anchors")):
        raise HTTPException(400, "Smoothing applies to a manually-corrected volume — Confirm border corrections first.")
    try:
        work = _oct_working_path(case_id, src)
        params = dict(m.get("oct_params") or {})
        sag = _np.asarray(_load_border_vol(work))              # (lateral, depth, frames) = sagittal
        corrected = oct_mod.revert_sagittal(sag)               # → (frames, depth, lateral) for the passes
        # PRIMARY: re-detect the (reliable) corrected surface + smooth ACROSS SLICES + re-warp — removes the
        # slice-to-slice jitter the provided_edges warp left, preserving frame-localized corrections exactly.
        corrected, _sm = oct_mod.smooth_corrected_volume(corrected, params, workers=None)
        # extra cleanup of the low-signal first/last acquisition-edge frames (gated; interior untouched)
        corrected, _fbs = oct_mod.frame_boundary_lat_smooth(corrected, params, workers=None)
        # re-assert manual overrides LAST (mirror the auto chain) so they stay final, never smoothed away
        ms = params.get("manual_shifts")
        if ms:
            corrected, _ = oct_mod.apply_manual_shifts(corrected, ms)
        corrected, _ = oct_mod._apply_crop(corrected, params)
        sp = oct_mod._resolve_spacing(params, m.get("companion_txt"), n_frames=int(corrected.shape[0]))
        oct_mod.write_volume_nifti(corrected, work, sp)        # atomic; same NIFTI_DIRECTION/spacing/origin
        # geometry shifted → drop segmentation (user re-runs SAM2); KEEP anchors + preproc_vetted + border_gt
        seg_dir = orch.segmentation_preview_dir(case_id)
        if seg_dir.exists():
            _sh2.rmtree(seg_dir, ignore_errors=True)
        for grp in ("context_seg", "context_cons"):
            _sh2.rmtree(_preview_group_dir(case_id, grp), ignore_errors=True)
        labels.corrected_path(case_id).unlink(missing_ok=True)
        orch.case_qa_json(case_id).unlink(missing_ok=True)
        extra = {"oct_volume_index": int(m.get("oct_volume_index", 0)), "oct_params": params, "scar_metrics": None,
                 "sam2_meta": None, "corrected_labelmap": None, "consensus_case": None, "scar_done": None,
                 "cornea_vetted": None, "subgroup_confirmed": None,
                 "qa_json": None, "segmentation_preview_dir": None}
        return _oct_render_volume(case_id, work, preprocessed=True, extra=extra)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"OCT smooth-corrected failed: {exc}")


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
                "scar_range": (list(m.get("scar_range")) if m.get("scar_range") else None),
                "scar_subgroup": (str(m.get("scar_subgroup")).strip() if m.get("scar_subgroup") else None),
                "sam2_meta": bool(m.get("sam2_meta")),
                "scar_done": bool(m.get("scar_done")),
                "subgroup_confirmed": bool(m.get("subgroup_confirmed")),
                "consensus_case": bool(m.get("consensus_case")),   # so an ALIGNED member colours as step 7
                "normalized": bool(m.get("normalized")),
                "corrected_labelmap": bool(m.get("corrected_labelmap")),
                "training_scheduled": bool(m.get("training_scheduled")),
                "review_flags": (list(m.get("review_flags")) if isinstance(m.get("review_flags"), list) else []),
                # Defect-marking feature: precise wrong-column marks + a "needs manual help" flag, surfaced so
                # both the sidebar and the assistant can see them (count only for the marks, to keep the list light).
                "defect_marks": (len(m.get("defect_marks")) if isinstance(m.get("defect_marks"), list) else 0),
                "difficult_scan": bool(m.get("difficult_scan")),
                # Rejected BY THE REVIEWER, as distinct from difficult_scan (which bulk preprocessing also
                # sets). This is what the review counter is built on.
                "reviewer_rejected": bool(m.get("reviewer_rejected")),
                # The rejection reason travels with the flag so the sidebar can show WHY a scan was rejected
                # on hover, without opening it. Text only — the timestamp stays in the manifest.
                "difficult_reason": (((m.get("difficult_reason") or {}).get("text"))
                                     if isinstance(m.get("difficult_reason"), dict) else None),
                # Surface-crop (clipped cornea): AUTO = the pipeline took the surface-crop path
                # (oct_iter.stopped == "surface_crop"); MANUAL = the human's review override (True/False/None).
                # The loader badges both so every auto-detected clip is easy to find + verify.
                "surface_crop_auto": ((m.get("oct_iter") or {}).get("stopped") == "surface_crop"),
                "surface_crop_manual": m.get("surface_crop_manual"),
                # The user's explicit surface-crop DECISION (auto | manual | off) and the frame set it acts on.
                # Distinct from surface_crop_manual, which is only a review mark and steers nothing.
                "surface_crop_mode": ((m.get("oct_params") or {}).get("surface_crop_mode") or "auto"),
                "surface_crop_n_frames": len(((m.get("oct_params") or {}).get("surface_crop_frames") or [])),
                # DELIVERED-VOLUME QA (oct_iter.final_qa) — measured on the volume actually written, unlike
                # oct_iter.metrics which describes a pre-rigid intermediate that is never delivered. ADVISORY
                # ONLY: nothing gates on it; it exists so the review queue can be ORDERED worst-first instead
                # of alphabetically. Present only on scans preprocessed since this was added, so the loader
                # must treat null as "not measured", NOT as "clean".
                # `coverage` and `path` are NOT optional extras: dev is deflated by destroyed volume, so
                # two scans are only comparable at similar coverage and within the same path stratum.
                # `score` (= dev + 0.5*axial) is the exact quantity the pipeline itself minimises.
                "final_qa": (lambda q: {
                    "dev": q.get("dev"), "axial": q.get("axial"), "score": q.get("score"),
                    "coverage": q.get("coverage"), "path": q.get("path"),
                    "reported_dev": q.get("reported_dev"), "tilt_total": q.get("tilt_total"),
                    "max_jitter": q.get("max_jitter"), "needs_review": bool(q.get("needs_review")),
                    "review_reasons": list(q.get("review_reasons") or []),
                    "qa_errors": list(q.get("qa_errors") or []),
                } if isinstance(q, dict) and q else None)((m.get("oct_iter") or {}).get("final_qa")),
                # UNDER-DETERMINATION (oct_iter.determinism) — how many slices the last CORRECTIONS run says
                # are still worth drawing, and the largest px of the reviewer's own drawn edge the delivered
                # surface currently discards. ADVISORY, like final_qa: it orders the queue, gates nothing.
                # Absent (null) on every auto-only scan, which is 300 of 308 in the store.
                "determinism": (lambda d: {
                    "n_suggest": d.get("n_suggest"), "n_findings": d.get("n_findings"),
                    "worst_px": (max([f.get("E_px") or 0.0 for f in (d.get("findings") or [])])
                                 if d.get("findings") else 0.0),
                    "kinds": sorted({str(f.get("kind")) for f in (d.get("findings") or [])}),
                    "T_px": d.get("T_px"), "route": d.get("route"),
                    "floor_px": ((d.get("floor") or {}).get("rms_px")
                                 if (d.get("floor") or {}).get("quotable") else None),
                } if isinstance(d, dict) and d.get("per_frame") else None)(
                    (m.get("oct_iter") or {}).get("determinism")),
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

# #15 — case_ids whose in-flight compare-strategies run was asked to stop. The (slow) compare endpoint
# polls membership between strategies/replicates and returns early; the cancel endpoint just adds the id.
# A plain set is fine: add/discard/membership on str keys are atomic under CPython's GIL.
_COMPARE_CANCEL: set[str] = set()

# Live SAM2 progress, keyed by safe_case_id, so the UI can poll a meaningful phase ("axial 1/3",
# "fusing", "scar") instead of an opaque spinner. Written by segment_sam2's callback (under the GPU
# lock) and read by the status GET (served on a separate threadpool thread, no GPU lock needed).
_SAM2_PROGRESS: dict[str, dict] = {}
_SAM2_PROGRESS_GUARD = threading.Lock()
_PLANE_LABEL = {"axial": "axial", "coronal": "coronal", "sagittal": "sagittal", "fuse": "fusing planes in 3D"}


def _sam2_progress_set(case_id: str, phase: str, message: str, index: int = 0, total: int = 3) -> None:
    with _SAM2_PROGRESS_GUARD:
        _SAM2_PROGRESS[orch.safe_case_id(case_id)] = {
            "phase": phase, "index": int(index), "total": int(total), "message": message}


def _sam2_progress_get(case_id: str) -> dict:
    with _SAM2_PROGRESS_GUARD:
        return dict(_SAM2_PROGRESS.get(orch.safe_case_id(case_id)) or {"phase": "idle", "message": ""})


# Per-case lock for the canonical labelmap read-modify-write. scar_auto/scar_edit/
# scar_sam2_hint/scar_auto_sam2/segmentation_from_drawing each load the corrected
# labelmap, mutate it, and write it back; without this two concurrent corrections
# (e.g. a brush edit racing an auto run, or worker threads) would clobber each
# other's voxels. RLock so a future nested labelmap op on the same case is safe.
_LABELMAP_LOCKS: dict[str, threading.RLock] = {}
_LABELMAP_LOCKS_GUARD = threading.Lock()


def _labelmap_lock(case_id: str) -> threading.RLock:
    """Return the (lazily-created) RLock guarding this case's canonical labelmap,
    keyed on safe_case_id so two inputs that normalise to the same case share one lock."""
    cid = orch.safe_case_id(case_id)
    with _LABELMAP_LOCKS_GUARD:
        lk = _LABELMAP_LOCKS.get(cid)
        if lk is None:
            lk = threading.RLock()
            _LABELMAP_LOCKS[cid] = lk
        return lk


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
    # De-duplicate (keyed on the normalized id) so numTraining == distinct pairs written: export_case is an
    # idempotent overwrite, so a duplicated id in an explicit req.cases list would otherwise be counted twice
    # while only one image/label pair exists on disk — overstating dataset.json's numTraining.
    seen: set = set()
    cases = [c for c in cases if not (orch.safe_case_id(c) in seen or seen.add(orch.safe_case_id(c)))]
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


# ── Debug tab: replicate-alignment comparison ──────────────────────────────
# Pick two repeat scans of one eye, run several alignment methods, and SEE the overlap as a
# magenta(fixed)/green(moving) composite. Developer/adjudication tooling — it never writes to a
# case; renders go to a temp cache dir (debug_align._RENDER_ROOT). See debug_align.py's docstring
# for the methods, the metric's two load-bearing properties, and why TEASER++ is absent.
class DebugAlignRequest(BaseModel):
    fixed_case: str
    moving_case: str
    methods: List[str] | None = None
    # Opt-in 3-D interactive volumes. When false/absent the 2-D path is unchanged; when true the job
    # writes a shared `fixed_iso.nii.gz` (job-level `fixed3d`/`iso_mm`) plus per-method aligned-moving
    # + disagreement .nii.gz (result `volumes3d`), which the Debug tab renders LIVE in niivue.
    render_3d: bool = False


@app.get("/api/debug/align/groups")
def debug_align_groups() -> dict:
    """Eyes with >=2 replicate scans. Handles BOTH case-naming schemes (see debug_align.parse_case:
    scheme B `case_cs030_od_v1_2` MUST be matched before scheme A `case_cs001_os_v2`)."""
    return {"groups": debug_align.groups()}


@app.post("/api/debug/align/compare")
def debug_align_compare(req: DebugAlignRequest) -> dict:
    """Start an alignment comparison. Returns a job_id; poll /api/debug/align/job/{job_id}.

    Runs in a background thread (the cohort pattern): a full run is tens of seconds — several
    SimpleITK registrations plus a 33M-voxel FFT — and must not hold a threadpool worker that long.
    Concurrent runs are serialised inside the worker, so a second job queues rather than thrashing
    every core; it reports status "running" while it waits."""
    try:
        job_id = debug_align.start_compare(req.fixed_case, req.moving_case, req.methods,
                                           render_3d=req.render_3d)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Alignment comparison failed to start: {exc}")
    return {"job_id": job_id}


@app.get("/api/debug/align/job/{job_id}")
def debug_align_job(job_id: str) -> dict:
    """Live status/progress/results of a comparison job. `views` hold GET-able PNG URLs."""
    view = debug_align.job_view(job_id)
    if view is None:
        raise HTTPException(404, "No such alignment job (it may have been pruned).")
    return view


@app.get("/api/debug/align/view/{job_id}/{name}")
def debug_align_view(job_id: str, name: str) -> FileResponse:
    """One rendered composite PNG (2-D views) or interactive 3-D volume (.nii.gz). GET => token-exempt,
    so a plain <img src> or niivue's own fetch loads it directly. Path-traversal-guarded exactly like
    get_preview_file: a bare *.png / *.nii.gz basename only, and the job_id is resolved through the
    same sanitiser rather than pasted into a path.

    The .nii.gz is served with media_type application/gzip and NO Content-Encoding header: niivue picks
    its gunzip decoder from the .nii.gz URL extension, so a Content-Encoding: gzip would make the
    browser double-decompress and break niivue's own gunzip."""
    safe_name = Path(name).name
    low = safe_name.lower()
    is_png, is_nii = low.endswith(".png"), low.endswith(".nii.gz")
    if safe_name != name or not (is_png or is_nii):
        raise HTTPException(400, "Invalid view file name.")
    safe_job = orch.safe_case_id(job_id)
    if safe_job != job_id:
        raise HTTPException(400, "Invalid job id.")
    p = debug_align.job_dir(safe_job) / safe_name
    if not p.exists():
        raise HTTPException(404, "View not found.")
    return FileResponse(str(p), media_type="image/png" if is_png else "application/gzip")


# ── Debug tab: N-replicate consensus (all replicates of one eye, in one 3-D volume) ──
# The N-way generalisation of the pairwise magenta/green/white overlay: ALL replicates of the eye are
# aligned to the first (reference) by ONE chosen method and composited into a single min/excess RGBA
# volume — agreement=white, each replicate its own hue. niivue renders the RGBA .nii.gz directly in
# 3-D (colours baked backend-side, so there is no client windowing to mis-scale). Same long-job
# pattern as compare; the RGBA volume + 2-D composite PNGs are served by the token-exempt view route.
class ConsensusRequest(BaseModel):
    eye: str
    method: str = "fixed"
    space: str = "intensity"    # "intensity" (whole cornea) | "scar" (binary scar min/excess)


@app.post("/api/debug/consensus")
def debug_consensus(req: ConsensusRequest) -> dict:
    """Start an N-replicate consensus render for one eye, in one SPACE. Returns a job_id; poll
    /api/debug/consensus/job/{job_id}. Runs in a background thread serialised with the pairwise
    compare job (a second click queues rather than thrashing every core). space="scar" segments each
    replicate (SAM2 + hysteresis, cached) and composites the binary scar masks."""
    try:
        job_id = debug_align.start_consensus(req.eye, req.method, req.space)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Consensus render failed to start: {exc}")
    return {"job_id": job_id}


@app.get("/api/debug/consensus/job/{job_id}")
def debug_consensus_job(job_id: str) -> dict:
    """Live status/progress/result of a consensus job. `volume` is the RGBA .nii.gz URL (min/excess
    composite of every replicate), `replicates` the per-replicate colour legend, `slices` the matching
    2-D composite PNGs — all served via the existing token-exempt /api/debug/align/view route."""
    view = debug_align.consensus_job_view(job_id)
    if view is None:
        raise HTTPException(404, "No such consensus job (it may have been pruned).")
    return view


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

    # Fold any globally tuned detector settings in BEFORE serving, and publish their path so the CLI
    # subprocess path picks up the same ones. Done here rather than at import so it also covers --reload.
    _tuned = _apply_param_overrides_at_startup()
    if _tuned:
        print(f"detector overrides applied: {_tuned}", flush=True)

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
