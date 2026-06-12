"""Headless port of the OCT_Extraction preprocessing pipeline (from the user's
Streamlit scripts) used to produce the corrected volumes the cornea app consumes:

  1. read .OCT  (oct_converter POCT)            → raw B-scan z-stack
  2. oct_to_dicom (DICOMGeneratorlossless.py)   → uint16 multi-frame DICOM + geometry
  3. smooth_volume (DICOMSmootherSteps.py)      → corneal-edge + column correction,
                                                  3D active correction across slices

Streamlit/matplotlib UI and all visualization were dropped; only the numeric pipeline
remains, with the smoother parameters exposed via a params dict. Faithful to the
originals except: (a) the read contract is fixed to `read_oct_volume()[0].volume`
(the installed oct_converter returns volume objects, so step 2's `np.stack(frames)`
was a version bug); (b) the per-slice 3D-active correction is computed in O(N) by
caching each slice's edge once instead of reprocessing neighbours; (c) the previously
unused `corr_factor` now scales the column displacement (default 1.0 = unchanged).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import cv2
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Defaults mirror DICOMSmootherSteps.py's sidebar defaults + the lossless converter.
DEFAULT_PARAMS: dict = {
    "sigma": 2.0,                 # gaussian sigma for the column gradient
    "max_jump": 10.0,             # outlier clamp between adjacent columns
    "median_filter_size": 5,      # boundary median filter
    "d": 9,                       # bilateral filter diameter
    "sigmaColor": 75,
    "sigmaSpace": 75,
    "side_window": 10,            # intelligent side-correction window
    "side_threshold_factor": 2.0,
    "residual_threshold": 5.0,    # RANSAC quadratic residual
    "active_threshold": 5.0,      # 3D active correction across neighbouring slices
    "corr_factor": 1.0,           # scales the column-correction displacement (0..1)
}
# Optovue Angiovue XR Avanti "3D Cornea" geometry (corrected from the companion .txt;
# the conversion script's hardcoded 0.00625/0.0078 implied a 4x4x4mm cube — wrong, the
# real volume is 6.00mm lateral x 4.04mm x 2.006mm depth). Array is (frames,rows,cols)
# = (101 slices, 640 depth, 513 lateral). All exposed/overridable via params.
DEPTH_SPACING = round(2.006 / 640, 7)     # rows  (axial / Scan Depth / OCT Window Height)
LATERAL_SPACING = round(6.00 / 513, 7)    # cols  (fast B-scan line / XY Scan Size1 / Length)
SLICE_SPACING = 0.040                      # frames(slow axis / XY Scan Interval1)
DEFAULT_SLICE_THICKNESS = SLICE_SPACING
DEFAULT_PIXEL_SPACING = (DEPTH_SPACING, LATERAL_SPACING)   # DICOM [row, col]
# NIfTI geometry to match the app's existing volumes: sitk spacing (x,y,z)=(lateral,depth,slice),
# direction as Slicer produced for these OPT volumes, origin 0.
NIFTI_SPACING = (LATERAL_SPACING, DEPTH_SPACING, SLICE_SPACING)
NIFTI_DIRECTION = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)


# ── 1) read .OCT ───────────────────────────────────────────────────────────
class MissingCompanionError(ValueError):
    """The .OCT's companion .txt filespec isn't next to it (POCT can't read without it)."""


def read_oct_zstack(oct_path: str | Path, volume_index: int = 0) -> np.ndarray:
    """Read one volume's B-scan stack from an .OCT file → (frames, H, W) float32.

    An .OCT may hold several captures; the original pipeline uses volume 0. The
    Optovue .OCT stores its dimensions in a companion .txt that MUST sit next to it —
    POCT fails without it, so we check up front and raise an actionable error."""
    from oct_converter.readers import POCT
    p = Path(oct_path)
    if not (p.with_suffix(".txt").exists() or p.with_suffix(".TXT").exists()):
        raise MissingCompanionError(
            f"'{p.name}' has no companion .txt next to it — an Optovue .OCT can't be read "
            "without it. Upload the .OCT together with its .txt (or load the whole folder).")
    vols = POCT(str(oct_path)).read_oct_volume()
    if not vols:
        raise ValueError(f"No OCT volumes found in {oct_path}")
    vi = volume_index if 0 <= volume_index < len(vols) else 0
    arr = np.asarray(vols[vi].volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected OCT volume shape {arr.shape} in {oct_path}")
    return arr


def oct_num_volumes(oct_path: str | Path) -> int:
    from oct_converter.readers import POCT
    return len(POCT(str(oct_path)).read_oct_volume())


# ── 2) OCT → DICOM (metadata from filename + companion .txt) ────────────────
def parse_oct_filename(filename: str) -> dict:
    base = os.path.splitext(os.path.basename(filename))[0]
    toks = base.split("_")
    if len(toks) < 5:
        return {}
    m = re.match(r"(\d{4}-\d{2}-\d{2})\s*\((\d+)\)", toks[4])
    return {
        "patient_name": toks[0],
        "patient_id": toks[1],
        "study_description": toks[2],
        "laterality": toks[3],
        "study_date": m.group(1) if m else "",
        "series_number": int(m.group(2)) if m else 1,
    }


def parse_companion_file(txt_path: str | Path) -> dict:
    data: dict = {}
    with open(txt_path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            if "=" in line:
                key, val = [x.strip() for x in line.split("=", 1)]
                k = key.lower()
                if k == "eye scanned":
                    data["eye_scanned"] = val
                elif k == "scan depth":
                    data["scan_depth"] = _to_float(val)
                elif k == "physical video width":
                    data["physical_video_width"] = _to_float(val)
                elif k == "physical video height":
                    data["physical_video_height"] = _to_float(val)
    return data


def _to_float(val: str) -> float | None:
    try:
        return float(re.sub(r"[^0-9.\-]", "", val))
    except ValueError:
        return None


def oct_to_dicom(oct_path: str | Path, output_path: str | Path,
                 patient_name: str = "", patient_id: str = "", study_desc: str = "",
                 series_num: int = 1, orient_vec=None,
                 slice_thickness: float = DEFAULT_SLICE_THICKNESS,
                 pixel_spacing=DEFAULT_PIXEL_SPACING,
                 volume_index: int = 0) -> str:
    """Lossless OCT → uint16 multi-frame DICOM (DICOMGeneratorlossless.oct_to_dicom),
    with the read contract fixed to volume[volume_index].volume."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    num_frames, rows, cols = vol.shape

    # Multi-frame Grayscale Word Secondary Capture (valid, widely readable by Slicer/ITK).
    sop_class = "1.2.840.10008.5.1.4.1.1.7.3"
    sop_instance = generate_uid()
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_instance
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OPT"
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyDescription = study_desc
    ds.SeriesDescription = f"{patient_name} Series {series_num}".strip()
    ds.SeriesNumber = series_num
    ds.NumberOfFrames = num_frames
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SliceThickness = str(slice_thickness)
    ds.SpacingBetweenSlices = str(slice_thickness)
    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
    if orient_vec and len(orient_vec) == 6:
        ds.ImageOrientationPatient = [float(x) for x in orient_vec]
    ds.PixelData = vol.tobytes()

    os.makedirs(os.path.dirname(str(output_path)) or os.getcwd(), exist_ok=True)
    pydicom.dcmwrite(str(output_path), ds, write_like_original=False)
    return str(output_path)


def metadata_for(oct_filename: str, companion_txt: str | Path | None = None) -> dict:
    """Combine filename + companion-.txt metadata into oct_to_dicom kwargs."""
    fm = parse_oct_filename(oct_filename)
    comp = parse_companion_file(companion_txt) if companion_txt and Path(companion_txt).exists() else {}
    desc = (fm.get("study_description", "") + " " + comp.get("eye_scanned", fm.get("laterality", ""))).strip()
    return {
        "patient_name": fm.get("patient_name", ""),
        "patient_id": fm.get("patient_id", ""),
        "study_desc": desc,
        "series_num": fm.get("series_number", 1),
    }


# ── 3) smoother: corneal-edge + column correction (DICOMSmootherSteps.py) ───
def _histeq(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        lo, hi = img.min(), img.max()
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(img, np.uint8)
    return cv2.equalizeHist(img)


def reformat_to_sagittal(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume, (2, 1, 0))


def revert_sagittal(volume_sag: np.ndarray) -> np.ndarray:
    return np.transpose(volume_sag, (2, 1, 0))


def _detect_surface_gradient(img: np.ndarray, sigma: float) -> np.ndarray:
    # Vectorized over columns: smooth each column along depth, take the gradient, and
    # the brightest rising edge → corneal surface row. (Same result as the per-column
    # loop in the original, but ~order-of-magnitude faster.)
    sm = ndimage.gaussian_filter1d(img.astype(np.float32), sigma=sigma, axis=0)
    return np.argmax(np.gradient(sm, axis=0), axis=0)


def _correct_surface(surface_y: np.ndarray, max_jump: float) -> np.ndarray:
    surface_y = surface_y.astype(float)
    n = surface_y.size
    if n < 2:
        return surface_y
    outlier = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if abs(surface_y[i] - surface_y[i - 1]) > max_jump:
            outlier[i] = True
    valid = np.where(~outlier)[0]
    if len(valid) < 2:
        return surface_y
    f = interp1d(valid, surface_y[valid], kind="cubic", fill_value="extrapolate")
    out = surface_y.copy()
    out[outlier] = f(np.where(outlier)[0])
    return out


def _smooth_median(surface_y: np.ndarray, size: int) -> np.ndarray:
    return ndimage.median_filter(surface_y, size=size)


def _advanced_edge(img: np.ndarray, p: dict) -> np.ndarray:
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filt = cv2.bilateralFilter(img, d=int(p["d"]), sigmaColor=int(p["sigmaColor"]), sigmaSpace=int(p["sigmaSpace"]))
    raw = _detect_surface_gradient(filt, sigma=p["sigma"])
    corrected = _correct_surface(raw, max_jump=p["max_jump"])
    return _smooth_median(corrected, size=int(p["median_filter_size"]))


def _intelligent_side_correction(boundary: np.ndarray, window: int, thresh: float, side_fraction: float = 0.05) -> np.ndarray:
    corrected = boundary.copy().astype(float)
    W = len(boundary)
    for x in range(int(W * side_fraction)):
        s, e = x + 1, min(W, x + window)
        if s >= e:
            continue
        med = np.median(boundary[s:e])
        mad = np.median(np.abs(boundary[s:e] - med))
        if corrected[x] < med - thresh * mad:
            corrected[x] = med
    for x in range(int(W * (1 - side_fraction)), W):
        s, e = max(0, x - window), x
        if s >= e:
            continue
        med = np.median(boundary[s:e])
        mad = np.median(np.abs(boundary[s:e] - med))
        if corrected[x] < med - thresh * mad:
            corrected[x] = med
    return corrected.astype(int)


def _side_correction_quadratic_bias(boundary: np.ndarray, quadratic: np.ndarray, window: int, thresh: float,
                                    side_fraction: float = 0.05, bias_weight: float = 0.7) -> np.ndarray:
    corrected = boundary.copy().astype(float)
    W = len(boundary)
    for x in range(int(W * side_fraction)):
        s, e = x + 1, min(W, x + window)
        if s >= e:
            continue
        cand = bias_weight * quadratic[x] + (1 - bias_weight) * np.median(boundary[s:e])
        if abs(boundary[x] - quadratic[x]) > thresh:
            corrected[x] = cand
    for x in range(int(W * (1 - side_fraction)), W):
        s, e = max(0, x - window), x
        if s >= e:
            continue
        cand = bias_weight * quadratic[x] + (1 - bias_weight) * np.median(boundary[s:e])
        if abs(boundary[x] - quadratic[x]) > thresh:
            corrected[x] = cand
    return corrected.astype(int)


def _fit_quadratic_ransac(edge: np.ndarray, residual_threshold: float) -> np.ndarray:
    x = np.arange(len(edge)).reshape(-1, 1)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    ransac = RANSACRegressor(estimator=model, min_samples=0.3,
                             residual_threshold=residual_threshold, random_state=42)
    ransac.fit(x, edge)
    return ransac.predict(x)


def _warp_by_displacement(img: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    H, W = img.shape
    warped = np.zeros_like(img)
    for x in range(W):
        shift = int(round(displacement[x]))
        if shift > 0:
            nh = H - shift
            if nh > 0:
                warped[shift:, x] = img[:nh, x]
        elif shift < 0:
            nh = H + shift
            if nh > 0:
                warped[:nh, x] = img[-shift:, x]
        else:
            warped[:, x] = img[:, x]
    return warped


def _merged_side_edge(slice_img: np.ndarray, p: dict) -> np.ndarray:
    """The per-slice corrected boundary (process_slice_single_stage → 'Merged Side Edge'),
    choosing the lower-error edge of {hist-eq, raw} advanced-filtered detections."""
    edge_h = _advanced_edge(_histeq(slice_img), p)
    edge_r = _advanced_edge(slice_img, p)
    q_h = _fit_quadratic_ransac(edge_h, p["residual_threshold"])
    q_r = _fit_quadratic_ransac(edge_r, p["residual_threshold"])
    chosen = edge_h if np.sum((edge_h - q_h) ** 2) <= np.sum((edge_r - q_r) ** 2) else edge_r
    quad_prelim = _fit_quadratic_ransac(chosen, p["residual_threshold"])
    return _side_correction_quadratic_bias(chosen, quad_prelim,
                                           window=int(p["side_window"]), thresh=p["side_threshold_factor"])


def _edge_worker(packed):
    sl, p = packed
    return _merged_side_edge(sl, p)


def _warp_worker(packed):
    sl, active_edge, residual, corr_factor = packed
    quad = _fit_quadratic_ransac(active_edge, residual)
    return _warp_by_displacement(sl, (quad - active_edge) * corr_factor)


def _map_slices(worker, items, progress, lo, hi, workers):
    """Map a per-slice worker across slices on a spawn pool (no CUDA-fork issues),
    falling back to serial on any failure. Reports progress in [lo, hi]."""
    n = len(items)
    out = [None] * n
    try:
        import concurrent.futures
        import multiprocessing as mp
        # fork: children inherit this (clean, torch-free) process's memory — fast, no
        # re-import, no recursion. Safe because the heavy smoother runs in an isolated
        # subprocess (oct_preprocess CLI), never directly inside the CUDA-bearing sidecar.
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for i, r in enumerate(ex.map(worker, items, chunksize=8)):
                out[i] = r
                if progress:
                    progress(lo + (hi - lo) * (i + 1) / n)
        return out
    except Exception:
        for i, it in enumerate(items):
            out[i] = worker(it)
            if progress:
                progress(lo + (hi - lo) * (i + 1) / n)
        return out


def smooth_volume(volume: np.ndarray, params: dict | None = None, progress=None,
                  workers: int | None = None) -> np.ndarray:
    """Apply the corneal-edge + column correction with 3D active correction to a
    (frames, H, W) volume; returns the corrected volume (same shape/dtype).

    Equivalent to DICOMSmootherSteps' process_slice_with_3d_active over every sagittal
    slice, but each slice's edge is computed once (O(N), not O(3N)) and the two
    independent per-slice phases are parallelised across CPU cores."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    sag = reformat_to_sagittal(volume)             # (sag_slices, H, W)
    n = sag.shape[0]
    corr_factor = float(p.get("corr_factor", 1.0))
    active_threshold = float(p.get("active_threshold", 5.0))
    if workers is None:
        workers = max(1, min(16, (os.cpu_count() or 2) - 2))

    # 1) per-slice corrected boundary (the expensive bilateral+edge+RANSAC) — parallel.
    edges = np.array(_map_slices(_edge_worker, [(sag[i], p) for i in range(n)], progress, 0.0, 0.5, workers))

    # 2) 3D active correction: snap each slice's edge toward its neighbours' median.
    active = edges.copy()
    for i in range(1, n - 1):
        med = np.median(np.stack([edges[i - 1], edges[i + 1]]), axis=0)
        dev = np.abs(edges[i] - med)
        active[i][dev > active_threshold] = med[dev > active_threshold]

    # 3) flatten each slice to its quadratic via column warp — parallel — then revert.
    res = float(p["residual_threshold"])
    warped = _map_slices(_warp_worker, [(sag[i], active[i], res, corr_factor) for i in range(n)], progress, 0.5, 1.0, workers)
    return revert_sagittal(np.array(warped))


# ── NIfTI output (correct Avanti geometry, matching the app's existing volumes) ──
def write_volume_nifti(vol_zyx: np.ndarray, out_path: str | Path,
                       spacing_xyz=NIFTI_SPACING, direction=NIFTI_DIRECTION) -> str:
    """Write a (frames, rows, cols) = (z, y, x) array as a NIfTI with explicit spacing
    (mm) and direction — bypassing the multi-frame-DICOM spacing loss so the geometry
    that drives scar mm³ is exactly right."""
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol_zyx))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetDirection(tuple(float(d) for d in direction))
    img.SetOrigin((0.0, 0.0, 0.0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path))
    return str(out_path)


def _spacing_from_params(params: dict | None):
    p = params or {}
    return (float(p.get("lateral_spacing", LATERAL_SPACING)),
            float(p.get("depth_spacing", DEPTH_SPACING)),
            float(p.get("slice_spacing", SLICE_SPACING)))


def raw_oct_to_nifti(oct_path: str | Path, out_nifti: str | Path,
                     volume_index: int = 0, params: dict | None = None) -> str:
    """Raw .OCT z-stack → NIfTI (no corrections) for inspection/scrubbing."""
    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    return write_volume_nifti(vol, out_nifti, _spacing_from_params(params))


def preprocess_oct_to_nifti(oct_path: str | Path, out_nifti: str | Path,
                            params: dict | None = None, volume_index: int = 0,
                            progress=None) -> str:
    """Full pipeline: read .OCT → smoother corrections → NIfTI with correct geometry."""
    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    corrected = smooth_volume(vol, params, progress=progress)
    return write_volume_nifti(corrected, out_nifti, _spacing_from_params(params))


# ── CLI: run the heavy pipeline in an isolated subprocess (called by the sidecar,
#    so the fork-based parallelism never touches the sidecar's CUDA/torch state) ──
if __name__ == "__main__":
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description="OCT preprocessing worker")
    ap.add_argument("mode", choices=["raw", "preprocess"])
    ap.add_argument("oct_path")
    ap.add_argument("out_nifti")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--volume-index", type=int, default=0)
    a = ap.parse_args()
    _p = _json.loads(a.params)
    if a.mode == "raw":
        raw_oct_to_nifti(a.oct_path, a.out_nifti, volume_index=a.volume_index, params=_p)
    else:
        preprocess_oct_to_nifti(a.oct_path, a.out_nifti, params=_p, volume_index=a.volume_index)
    print("OK " + str(a.out_nifti))
