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
    # ── over-correction guard (#2) ── A low-signal lateral column gets a garbage edge whose deviation
    # from the dome's quadratic is huge, so (quad-edge) demands a 100-360px shift that bends the edge
    # and (re-detected on the warped output) compounds every pass. Any per-column displacement beyond
    # max_displacement (px) is therefore NOT trusted: that column is treated as bad and its shift is
    # interpolated from its good neighbours, then hard-clamped. Real corrections are a few px (a raw
    # boundary deviates < ~17px from its fit), so this is a no-op on well-detected columns — clean scans
    # are unchanged; only the pathological lateral runaway is tamed.
    "max_displacement": 40.0,
    # ── ping-pong axial refine (#2) ── After the sagittal correction, run the SAME correction in the
    # axial domain (flatten along lateral, per frame) and keep it per-frame where it makes the en-face
    # boundary smoother — cleans the 'hairy' axial boundary the sagittal pass leaves at noisy slice ends.
    # Confirmed on real scans to give the smoothest 3-D surface; a global guard makes it never worse.
    "axial_refine": True,
    # ── axial consistency (#3) ── Sagittal slices are corrected independently, so neighbouring slices
    # can shift inconsistently → the en-face/axial corneal boundary turns jagged ("hairier"). Smoothing
    # the per-column displacement FIELD across the slice (lateral) axis with this Gaussian sigma (px,
    # 0 = off) makes neighbours shift consistently → a smoother axial boundary, while the per-slice
    # quadratic still carries the real lateral curvature. Small sigma stays close to the per-slice fit.
    "interslice_smooth": 1.0,
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
    # The date token is "YYYY-MM-DD" optionally followed by a replicate suffix "(N)".
    # Parse the date even when there's no "(N)" so the FIRST scan isn't left date-less.
    m = re.match(r"(\d{4}-\d{2}-\d{2})(?:\s*\((\d+)\))?", toks[4])
    return {
        "patient_name": toks[0],
        "patient_id": toks[1],
        "study_description": toks[2],
        "laterality": toks[3],
        "study_date": m.group(1) if m else "",
        "series_number": int(m.group(2)) if (m and m.group(2)) else 1,
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


# ── per-scan voxel geometry from the companion .txt (the source of truth) ────
# The .OCT's companion .txt records the TRUE acquisition geometry. It varies per
# scan (e.g. XY Scan Size1 = 4.60mm for CS019, 6.00mm for CS015), so the geometry
# must be read per-scan, not hardcoded. The file lists several "[CL - 3D Cornea
# Step N]" blocks; only ONE is the active 3D acquisition — the Step whose
# "XY Scan Usage" equals the slice/frame count (the others are Usage=1 placeholders).
def _parse_companion_full(txt_path: str | Path):
    """Parse the companion .txt into (top-level dict, {step_num: detail dict}).

    Top-level: oct_window_height, scan_depth, eye_scanned.
    Per step: length (XY Scan Length), usage (XY Scan Usage), size1 (XY Scan
    Size1, mm), interval1 (XY Scan Interval1, mm)."""
    top: dict = {}
    steps: dict = {}
    cur_step, in_detail = None, False
    with open(txt_path, "r", encoding="utf8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            sm = re.match(r"\[CL - 3D Cornea Step (\d+)(\s+Detail)?\]", line)
            if sm:
                cur_step, in_detail = int(sm.group(1)), bool(sm.group(2))
                steps.setdefault(cur_step, {})
                continue
            if line.startswith("["):                 # a non-step section resets context
                cur_step, in_detail = None, False
            if "=" not in line:
                continue
            key, val = [x.strip() for x in line.split("=", 1)]
            k = key.lower()
            if cur_step is None:
                if k == "oct window height":
                    top["oct_window_height"] = _to_float(val)
                elif k == "scan depth":
                    top["scan_depth"] = _to_float(val)
                elif k == "eye scanned":
                    top["eye_scanned"] = val
            else:
                s = steps[cur_step]
                if not in_detail:
                    if k == "xy scan length":
                        s["length"] = _to_float(val)
                    elif k == "xy scan usage":
                        s["usage"] = _to_float(val)
                else:
                    if k == "xy scan size1":
                        s["size1"] = _to_float(val)
                    elif k == "xy scan interval1":
                        s["interval1"] = _to_float(val)
                    elif k == "xy scan usage1" and s.get("usage") is None:
                        s["usage"] = _to_float(val)
    return top, steps


def companion_geometry(txt_path: str | Path, n_frames: int | None = None) -> dict:
    """Derive per-scan voxel spacing (mm) from the companion .txt. Returns a dict
    with any of lateral_spacing / depth_spacing / slice_spacing that could be
    resolved (empty if the file is unreadable/unrecognised — caller falls back to
    the Avanti constants). Picks the active acquisition Step by frame count."""
    try:
        top, steps = _parse_companion_full(txt_path)
    except Exception:  # noqa: BLE001
        return {}
    if not steps:
        return {}

    def usage(s: dict) -> float:
        return s.get("usage") or 0.0

    active = None
    if n_frames:
        active = next((s for s in steps.values() if usage(s) == n_frames), None)
    if active is None:                                # else the most-acquired step
        active = max(steps.values(), key=usage, default=None)
    if not active:
        return {}
    geom: dict = {}
    size1, length = active.get("size1"), active.get("length")
    depth, win_h = top.get("scan_depth"), top.get("oct_window_height")
    interval1 = active.get("interval1")
    if size1 and length:
        geom["lateral_spacing"] = size1 / length
    if depth and win_h:
        geom["depth_spacing"] = depth / win_h
    if interval1:
        geom["slice_spacing"] = interval1
    return geom


# Plausible Avanti 3D-Cornea voxel-spacing ranges (mm); outside these we warn so a
# wrong-geometry volume can't silently corrupt the scar metric.
SPACING_BOUNDS = {"lateral": (0.0050, 0.0140), "depth": (0.0025, 0.0040), "slice": (0.020, 0.060)}


def validate_spacing(spacing_xyz) -> list:
    """Return human-readable warnings for any (lateral, depth, slice) spacing that
    falls outside the plausible Avanti range — purely advisory, never raises."""
    sp = [float(s) for s in spacing_xyz[:3]]
    names = ("lateral", "depth", "slice")
    warns = []
    for val, name in zip(sp, names):
        lo, hi = SPACING_BOUNDS[name]
        if not (lo <= val <= hi):
            warns.append(f"{name} spacing {val:.5f}mm outside Avanti range [{lo}, {hi}]")
    return warns


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
    """Faithful to DICOMSmootherSteps.fit_quadratic_ransac: sklearn RANSAC quadratic fit of the
    corneal boundary (degree-2 polynomial, min_samples=0.3, fixed seed)."""
    x = np.arange(len(edge)).reshape(-1, 1)
    try:
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        ransac = RANSACRegressor(estimator=model, min_samples=0.3,
                                 residual_threshold=residual_threshold, random_state=42)
        ransac.fit(x, edge)
        return ransac.predict(x)
    except Exception:  # noqa: BLE001
        # RANSAC found no valid consensus (degenerate/noisy edge, e.g. an artifacted scan) → plain
        # degree-2 least squares so the scan still preprocesses instead of crashing the whole run.
        xv = np.arange(len(edge))
        if len(edge) >= 3:
            return np.polyval(np.polyfit(xv, np.asarray(edge, float), 2), xv)
        return np.asarray(edge, float)


def _warp_by_displacement(img: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    H, W = img.shape
    warped = np.zeros_like(img)
    for x in range(W):
        shift = int(displacement[x])   # truncate toward zero (faithful to warp_image_by_edge)
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


def _fill_cols_along_rows(img: np.ndarray) -> np.ndarray:
    """In a sagittal slice (rows=depth, cols=frames), replace each column's LEADING/TRAILING zero run
    (the black padding a prior column-warp left) with the nearest real pixel. Used between iterative
    passes so the edge detector can't lock onto the black-band→tissue edge (which caused 100–360px
    runaway shifts on pass 2+). Pure edge-replication; only touches padding, never real tissue."""
    H, W = img.shape
    out = img.copy()
    nz = img != 0
    has = nz.any(axis=0)
    firstnz = np.argmax(nz, axis=0)
    lastnz = H - 1 - np.argmax(nz[::-1], axis=0)
    for x in range(W):
        if not has[x]:
            continue
        f, l = int(firstnz[x]), int(lastnz[x])
        if f > 0:
            out[:f, x] = img[f, x]
        if l < H - 1:
            out[l + 1:, x] = img[l, x]
    return out


def _fill_black_bands(volume: np.ndarray) -> np.ndarray:
    """Fill the warp's black padding throughout a (frames, depth, lateral) volume, in the SAME
    sagittal domain the warp operates on, so a re-fed (already-corrected) volume detects cleanly.
    Operates on a COPY — reformat_to_sagittal is a transpose VIEW, so writing through it would mutate
    the caller's stored chain volume (corrupting the kept pass + its previews)."""
    sag = reformat_to_sagittal(volume).copy()
    for i in range(sag.shape[0]):
        sag[i] = _fill_cols_along_rows(sag[i])
    return np.ascontiguousarray(revert_sagittal(sag))


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


def _interp_bad_displacement(disp: np.ndarray, bad_cols, good_cols) -> np.ndarray:
    """Replace the DISPLACEMENT (not the edge) at bad columns with a smooth interpolation from the
    GOOD anchor columns, so a bad column gets a correction consistent with its good neighbours.
    Interpolating the displacement (the correction field) rather than the detected edge avoids the
    overshoot that enlarged real curvature, and preserves the underlying tissue shape."""
    if not bad_cols:
        return disp
    W = len(disp)
    bad = [c for c in bad_cols if 0 <= c < W]
    if good_cols:
        anchors = sorted({c for c in good_cols if 0 <= c < W} - set(bad))
    else:
        bad_set = set(bad)
        anchors = [c for c in range(W) if c not in bad_set]
    if bad and len(anchors) >= 2:
        anchors = np.array(anchors)
        disp[bad] = np.interp(np.array(bad), anchors, disp[anchors])
    return disp


def _slice_displacement(active_edge, residual, corr_factor, bad_cols, good_cols, max_disp):
    """The per-column shift that flattens one sagittal slice's boundary to its quadratic, WITH the
    over-correction guard (#2): a column whose demanded shift |quad-edge| exceeds max_disp is a runaway
    (a garbage low-signal edge the quadratic can't trust), so it is treated as bad and its shift is
    interpolated from the good (well-detected) columns, then hard-clamped — a runaway can no longer bend
    the slice by 100-360px or compound across passes. With NO runaway column (the normal case — a raw
    boundary deviates < ~17px from its fit) this is exactly the faithful (quad-edge)*corr_factor field,
    so well-detected scans/columns are unchanged. max_disp<=0 disables the guard (legacy)."""
    quad = _fit_quadratic_ransac(active_edge, residual)
    disp = (quad - active_edge) * corr_factor
    bad = set(int(c) for c in bad_cols)
    if max_disp and max_disp > 0:
        bad |= {int(c) for c in np.where(np.abs(disp) > max_disp)[0]}
    disp = _interp_bad_displacement(disp, sorted(bad), good_cols)  # runaway cols → good-neighbour shift
    if max_disp and max_disp > 0:
        np.clip(disp, -max_disp, max_disp, out=disp)              # backstop (e.g. an all-bad slice)
    return disp


def _disp_worker(packed):
    sl, active_edge, residual, corr_factor, bad_cols, good_cols, max_disp = packed
    return _slice_displacement(active_edge, residual, corr_factor, bad_cols, good_cols, max_disp)


def _axial_roughness(edges: np.ndarray) -> float:
    """Mean |first-difference of the detected corneal boundary ACROSS sagittal slices| (axis 0) — i.e.
    how jagged the en-face / AXIAL boundary is. Per-slice correction is independent, so inconsistent
    inter-slice shifts make this grow ('hairier' axial view, #3); lower = smoother axial boundary."""
    e = np.asarray(edges, dtype=float)
    if e.ndim != 2 or e.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(e, axis=0))))


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
                  workers: int | None = None, return_metric: bool = False,
                  detect_volume: np.ndarray | None = None):
    """Apply the corneal-edge + column correction with 3D active correction to a
    (frames, H, W) volume; returns the corrected volume (same shape/dtype).

    Equivalent to DICOMSmootherSteps' process_slice_with_3d_active over every sagittal
    slice, but each slice's edge is computed once (O(N), not O(3N)) and the two
    independent per-slice phases are parallelised across CPU cores.

    return_metric=True → also return (mean per-column correction magnitude px, axial roughness px):
    the iterative-refinement convergence signal and the en-face boundary jaggedness (#3). The corrected
    array is identical either way.

    NOTE: the correction is no longer byte-identical to DICOMSmootherSteps — by design (the user asked
    to fix two failure modes): the OVER-CORRECTION GUARD (#2, max_displacement) interpolates+clamps a
    runaway lateral shift, and INTER-SLICE SMOOTHING (#3, interslice_smooth) smooths the displacement
    field across slices for a consistent axial boundary. Both are no-ops at their off values
    (max_displacement<=0, interslice_smooth=0) and the guard is a no-op on well-detected columns, so a
    clean scan is essentially unchanged; only the pathological lateral runaway/hairiness is tamed.

    detect_volume: if given, the corneal edge is DETECTED on this volume (e.g. a black-band-filled
    copy, so re-detection on a warped input isn't fooled by the warp's zero padding) while the warp is
    applied to `volume` itself — so the OUTPUT never contains the filled (fake-tissue) pixels, only the
    real data + honest zero padding. The cornea sits at the same row in both (filling only touches
    padding), so the detected displacement aligns `volume`'s cornea correctly."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    sag = reformat_to_sagittal(volume)             # the volume to WARP (real data, never filled)
    det = reformat_to_sagittal(detect_volume) if detect_volume is not None else sag  # detect on this
    n = sag.shape[0]
    corr_factor = float(p.get("corr_factor", 1.0))
    active_threshold = float(p.get("active_threshold", 5.0))
    if workers is None:
        workers = max(1, min(16, (os.cpu_count() or 2) - 2))

    # 1) per-slice corrected boundary (the expensive bilateral+edge+RANSAC) — parallel. Detected on
    #    `det` (the filled copy when iterating) so the warp's black padding can't fool the detector.
    edges = np.array(_map_slices(_edge_worker, [(det[i], p) for i in range(n)], progress, 0.0, 0.5, workers))

    # 2) 3D active correction — faithful to DICOMSmootherSteps.process_slice_with_3d_active: snap
    #    each slice's edge toward the median of ITSELF + its available neighbours (boundaries
    #    included), where the deviation exceeds the threshold.
    active = edges.copy()
    for i in range(n):
        stack = [edges[i]]
        if i > 0:
            stack.append(edges[i - 1])
        if i < n - 1:
            stack.append(edges[i + 1])
        med = np.median(np.stack(stack), axis=0)
        dev = np.abs(edges[i] - med)
        active[i][dev > active_threshold] = med[dev > active_threshold]

    # 3) per-slice displacement that flattens the boundary to its quadratic — parallel — WITH the
    #    over-correction guard (#2): a runaway shift (garbage low-signal edge) is interpolated from good
    #    neighbours + clamped, so it can't bend the edge or compound across passes.
    res = float(p["residual_threshold"])
    max_disp = float(p.get("max_displacement", 0.0) or 0.0)
    bad_cols = [int(c) for c in (p.get("force_columns") or [])]
    good_cols = [int(c) for c in (p.get("good_columns") or [])]
    items = [(sag[i], active[i], res, corr_factor, bad_cols, good_cols, max_disp) for i in range(n)]
    disp_field = np.array(_map_slices(_disp_worker, items, progress, 0.5, 0.9, workers))  # (n_slices, n_frames)
    # The per-pass metric is the mean per-column deviation of the boundary from its quadratic fit (the
    # iterative-refinement convergence signal + abs_floor calibration) — measured on the PRE-smoothing
    # field so its meaning is unchanged by #3's inter-slice smoothing (which only affects the warp).
    disp_mean = float(np.mean(np.abs(disp_field))) if disp_field.size else 0.0

    # 3b) axial consistency (#3): smooth the displacement FIELD across the slice (lateral) axis so
    #     neighbouring sagittal slices shift consistently → a smoother en-face/axial boundary. The
    #     depth/frame axis is untouched (the per-slice quadratic governs it); sigma=0 → per-slice field.
    ism = float(p.get("interslice_smooth", 0.0) or 0.0)
    if ism > 0 and n > 2:
        disp_field = ndimage.gaussian_filter1d(disp_field.astype(np.float64), sigma=ism, axis=0)

    # 4) warp each slice by its guarded+smoothed displacement, then revert.
    warped = np.array([_warp_by_displacement(sag[i], disp_field[i]) for i in range(n)])
    if progress:
        progress(1.0)
    corrected = revert_sagittal(warped)
    if return_metric:
        # disp_mean (deviation from fit, pre-smoothing) + axial roughness of the DETECTED boundary (the
        # en-face jaggedness the keep-best selection should also minimise, #3).
        return corrected, disp_mean, _axial_roughness(edges)
    return corrected


def _boundary_deviation(volume: np.ndarray, params: dict | None = None,
                        workers: int | None = None, detect_volume: np.ndarray | None = None):
    """Score a candidate volume's boundary quality on its own terms (no warp kept). Returns
    (in_plane_deviation, axial_roughness): the mean per-column deviation of the DETECTED boundary from
    its quadratic fit (how jagged WITHIN each sagittal slice), and the mean inter-slice first-difference
    (how jagged ACROSS slices = the en-face/axial 'hairiness', #3). Both in pixels; lower = better."""
    _, m, ax = smooth_volume(volume, params, workers=workers, return_metric=True, detect_volume=detect_volume)
    return float(m), float(ax)


def iterate_smooth_volume(volume: np.ndarray, params: dict | None = None,
                          max_iter: int = 5, min_improvement: float = 0.15,
                          abs_floor: float = 0.3, progress=None, workers: int | None = None,
                          inject_pass: int | None = None, inject_force=None, inject_good=None,
                          axial_weight: float = 0.5):
    """Iteratively re-apply smooth_volume to its own output, then KEEP THE BEST pass — the one whose
    detected corneal boundary deviates LEAST from a smooth fit (lowest "boundary deviation", px).

    Why keep-the-best rather than keep-the-last: each pass warps the boundary toward its quadratic
    fit, so the deviation usually SHRINKS pass over pass — but a pass can OVERSHOOT and produce a
    MORE deviant (worse) boundary than an earlier pass or even than the raw original (re-detection on
    an over-warped volume picks up a jagged edge). So we score EVERY candidate volume's deviation and
    select the minimum: a worse pass is never kept, and the result can never be more deviant than the
    raw input (raw is in the candidate set). This is the user's "compare so the subsequent border is
    not a more extreme deviation than the original".

    The search stops early (no more passes) once the deviation stops improving — it GREW vs the prior
    pass (overshoot), improved by < min_improvement (diminishing), fell below abs_floor (converged),
    or hit max_iter. But the FINAL choice is always argmin over all measured candidates.

    Returns (chain, best_idx, info): chain = [V0(raw), V1, …, Vm] every measured volume (for the UI
    pass-stepper); best_idx = index of the kept volume; info = {passes (corrected passes produced =
    len(chain)-1), best_pass, metrics (deviation px of each chain volume), stopped}."""
    max_iter = max(1, int(max_iter))
    # The iteration applies a manual column fix PER-PASS only (the user's "fix columns for a particular
    # iteration"): force_columns/good_columns are NOT global params here — they're injected at exactly
    # inject_pass (1-based) and absent on every other pass.
    base = dict(params or {})
    base.pop("force_columns", None)
    base.pop("good_columns", None)
    chain: list = [volume]       # V0 = raw, then each accepted pass
    rough: list = []             # rough[i] = in-plane boundary deviation of chain[i] (convergence signal)
    axial: list = []             # axial[i] = en-face/axial roughness of chain[i] (#3, folded into select)
    stopped = "max_iter"
    for k in range(max_iter):
        lo = k / max_iter
        hi = (k + 1) / max_iter
        pp = dict(base)
        if inject_pass is not None and (k + 1) == int(inject_pass):
            pp["force_columns"] = [int(c) for c in (inject_force or [])]
            pp["good_columns"] = [int(c) for c in (inject_good or [])]
        # A re-fed pass (k>=1) runs on the PREVIOUS pass's warped output, whose black padding would
        # fool the edge detector into 100-360px runaway shifts — DETECT on a filled copy. But WARP the
        # real (unfilled) chain[k], so the output never carries the fill's fake pixels (only honest
        # zero padding). Pass 1 runs on raw with no fill → byte-identical to the faithful single pass.
        det = None if k == 0 else _fill_black_bands(chain[k])
        nxt, r, ax = smooth_volume(chain[k], pp, progress=(
            (lambda f, lo=lo, hi=hi: progress(lo + (hi - lo) * f)) if progress else None),
            workers=workers, return_metric=True, detect_volume=det)   # r/ax = in-plane/axial of chain[k]
        rough.append(float(r)); axial.append(float(ax))
        # Force the iteration to REACH (and keep) the injected pass — never early-stop before it, or
        # the user's per-pass column fix would be silently discarded. Past the inject pass, the normal
        # keep-best stop logic resumes.
        force_reach = inject_pass is not None and (k + 1) <= int(inject_pass)
        # Stop producing more passes once the boundary stops getting smoother (but we've still
        # MEASURED chain[k], so it stays a candidate for the argmin below).
        if not force_reach and k >= 1:
            if r >= rough[k - 1]:
                stopped = "grew"; break          # chain[k] is MORE deviant than chain[k-1]
            if (rough[k - 1] - r) / max(rough[k - 1], 1e-9) < min_improvement:
                stopped = "diminishing"; break
        if not force_reach and r < abs_floor:
            stopped = "converged"
            chain.append(nxt)                    # a final tiny refinement is safe; keep + measure it
            break
        chain.append(nxt)                        # accept the next pass into the chain
    # Make sure EVERY chain volume has a measured deviation so it can compete in the argmin (the last
    # accepted pass is otherwise unmeasured when we stop by max_iter / converged).
    while len(rough) < len(chain):
        idx = len(rough)
        det = None if idx == 0 else _fill_black_bands(chain[idx])
        dev, ax = _boundary_deviation(chain[idx], base, workers=workers, detect_volume=det)
        rough.append(dev); axial.append(ax)
    # KEEP-THE-BEST by a COMBINED score: in-plane deviation + axial_weight × en-face/axial roughness
    # (#3). A pass that flattens each sagittal slice but leaves a HAIRIER axial boundary now loses to a
    # more axially-consistent pass — the old pure-in-plane argmin even preferred the hairiest pass.
    score = [rough[i] + axial_weight * axial[i] for i in range(len(chain))]
    best_idx = min(range(len(chain)), key=lambda i: score[i])
    info = {"passes": len(chain) - 1, "best_pass": best_idx,
            "metrics": [float(x) for x in rough], "axial_metrics": [float(x) for x in axial],
            "scores": [float(x) for x in score], "stopped": stopped}
    return chain, best_idx, info


# ── Ping-pong: axial correction after sagittal, for the hairy frames only (#2) ──────────────────────
# The sagittal correction flattens the boundary ALONG FRAMES (independently per lateral slice), so it
# leaves roughness ACROSS LATERAL — the en-face/"axial" boundary can look hairy where the sagittal slice
# was noisy at its ends. Running the SAME correction in the axial domain (flatten ALONG LATERAL, per
# frame) cleans those up. Empirically (real Avanti scans) a SINGLE axial pass after the sagittal one is
# the smoothest 3D surface; more ping-pong passes over-correct. Applying the axial result PER FRAME only
# where it actually reduces that frame's lateral roughness ("hairy frames only") is best + can't regress.
_FRAME_LATERAL_SWAP = (2, 1, 0)  # frames<->lateral (depth stays axis 1); makes axial slices the warp slices


def _axial_smooth_volume(volume: np.ndarray, params: dict | None, workers: int | None) -> np.ndarray:
    """Run smooth_volume in the AXIAL domain (flatten the boundary along the LATERAL axis, per frame) by
    swapping frames<->lateral, correcting, swapping back. Detects on a black-band-filled copy so the
    prior sagittal warp's padding can't fool the detector."""
    vt = np.ascontiguousarray(volume.transpose(*_FRAME_LATERAL_SWAP))
    out = smooth_volume(vt, params, workers=workers, detect_volume=_fill_black_bands(vt))
    return np.ascontiguousarray(out.transpose(*_FRAME_LATERAL_SWAP))


def _frame_boundary_surface(volume: np.ndarray, params: dict, workers: int | None) -> np.ndarray:
    """The corneal boundary B(frame, lateral) detected per FRAME (axial B-scan = depth×lateral) on a
    black-band-filled copy (so the warp padding can't fool detection). Shape (n_frames, n_lateral)."""
    vf = _fill_black_bands(volume)
    res = _map_slices(_edge_worker, [(vf[f], params) for f in range(vf.shape[0])], None, 0.0, 1.0, workers)
    return np.array([(r[0] if isinstance(r, tuple) else r) for r in res])


def _surface_rms(B: np.ndarray) -> float:
    """RMS deviation of the boundary surface from a smooth 2-D quadratic fit (3-D smoothness; lower=better)."""
    if B.ndim != 2 or B.size < 6:
        return 0.0
    ff, ll = np.mgrid[0:B.shape[0], 0:B.shape[1]].astype(float)
    A = np.c_[np.ones(B.size), ff.ravel(), ll.ravel(), ff.ravel() ** 2, ll.ravel() ** 2, (ff * ll).ravel()]
    coef, *_ = np.linalg.lstsq(A, B.ravel(), rcond=None)
    return float(np.sqrt(np.mean((B.ravel() - A @ coef) ** 2)))


def axial_refine_volume(v_sag: np.ndarray, params: dict | None = None, workers: int | None = None):
    """#2 ping-pong refine: after the sagittal correction, run an axial pass and KEEP it PER FRAME only
    where it reduces that frame's lateral boundary roughness (the user's 'axial correction for hairy
    axial slices'). A global guard then accepts the blend only if the whole 3-D surface got smoother — so
    this can never produce a worse surface than sagittal-only. Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = max(1, min(16, (os.cpu_count() or 2) - 2))
    v_ax = _axial_smooth_volume(v_sag, p, workers)
    B_sag = _frame_boundary_surface(v_sag, p, workers)
    B_ax = _frame_boundary_surface(v_ax, p, workers)
    tvl_sag = np.mean(np.abs(np.diff(B_sag, axis=1)), axis=1)   # per-frame lateral roughness (sagittal)
    tvl_ax = np.mean(np.abs(np.diff(B_ax, axis=1)), axis=1)     # per-frame lateral roughness (axial)
    use = tvl_ax < tvl_sag                                       # frames the axial pass actually improved
    out = v_sag.copy()
    out[use] = v_ax[use]
    B_out = np.where(use[:, None], B_ax, B_sag)                  # blended surface (no re-detect needed)
    rms_before, rms_after = _surface_rms(B_sag), _surface_rms(B_out)
    if use.any() and rms_after <= rms_before:                    # global guard: only accept a smoother surface
        return out, {"frames_refined": int(use.sum()), "n_frames": int(B_sag.shape[0]),
                     "surf_rms_before": rms_before, "surf_rms_after": rms_after, "applied": True}
    return v_sag, {"frames_refined": 0, "n_frames": int(B_sag.shape[0]),
                   "surf_rms_before": rms_before, "surf_rms_after": rms_before, "applied": False}


def apply_manual_shifts(volume: np.ndarray, shifts) -> tuple[np.ndarray, int]:
    """#2 fix-columns drag-to-correct: shift a specific frame (B-scan) UP/DOWN in DEPTH by an explicit
    pixel offset the annotator dragged in the fix-columns view — a per-frame manual ground-truth nudge
    applied ON TOP of the automatic boundary correction (so the user can fix any frame the auto-detect
    still placed wrong, especially the last few sagittal slices). `shifts` maps frame_index ->
    depth_pixels (positive = DOWN / deeper, matching the on-screen drag down); accepts a dict
    {frame: px} or a list of [frame, px] pairs. Vacated rows are zero-filled. Returns (volume,
    n_frames_shifted)."""
    pairs = []
    if isinstance(shifts, dict):
        pairs = list(shifts.items())
    elif isinstance(shifts, (list, tuple)):
        for item in shifts:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((item[0], item[1]))
    nz, depth = volume.shape[0], volume.shape[1]
    out = volume.copy()
    n = 0
    for f, px in pairs:
        try:
            fi, s = int(f), int(round(float(px)))
        except (TypeError, ValueError):
            continue
        if not (0 <= fi < nz) or s == 0:
            continue
        b = out[fi]                      # (depth, lateral)
        shifted = np.zeros_like(b)       # vacated rows stay 0 (background)
        if s > 0 and s < depth:          # move pixels DOWN (toward larger depth index)
            shifted[s:, :] = b[:depth - s, :]
        elif s < 0 and -s < depth:       # move pixels UP
            shifted[:depth + s, :] = b[-s:, :]
        out[fi] = shifted
        n += 1
    return out, n


# ── NIfTI output (correct Avanti geometry, matching the app's existing volumes) ──
def write_volume_nifti(vol_zyx: np.ndarray, out_path: str | Path,
                       spacing_xyz=NIFTI_SPACING, direction=NIFTI_DIRECTION) -> str:
    """Write a (frames, rows, cols) = (z, y, x) array as a NIfTI with explicit spacing
    (mm) and direction — bypassing the multi-frame-DICOM spacing loss so the geometry
    that drives scar mm³ is exactly right."""
    import os
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol_zyx))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetDirection(tuple(float(d) for d in direction))
    img.SetOrigin((0.0, 0.0, 0.0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Write atomically (tmp + replace) so a killed/crashed worker can never leave a truncated
    # NIfTI at the real path — a later reader (e.g. the raw-scrub cache) must never see a
    # half-written volume. The temp keeps the .nii.gz suffix so SimpleITK still gzips it.
    tmp = f"{out_path}.tmp.nii.gz"
    sitk.WriteImage(img, tmp)
    os.replace(tmp, str(out_path))
    return str(out_path)


def _resolve_spacing(params: dict | None, companion_txt: str | Path | None = None,
                     n_frames: int | None = None):
    """Resolve (lateral, depth, slice) spacing with precedence: explicit params >
    companion-.txt-derived per-scan geometry > Avanti constants. The companion is
    the per-scan source of truth (XY Scan Size1 varies 4–6mm between scans)."""
    geom = {}
    if companion_txt and Path(companion_txt).exists():
        geom = companion_geometry(companion_txt, n_frames)
    p = params or {}

    def pick(key: str, default: float) -> float:
        if p.get(key) is not None:
            return float(p[key])
        if geom.get(key) is not None:
            return float(geom[key])
        return default

    return (pick("lateral_spacing", LATERAL_SPACING),
            pick("depth_spacing", DEPTH_SPACING),
            pick("slice_spacing", SLICE_SPACING))


def raw_oct_to_nifti(oct_path: str | Path, out_nifti: str | Path,
                     volume_index: int = 0, params: dict | None = None,
                     companion_txt: str | Path | None = None) -> str:
    """Raw .OCT z-stack → NIfTI (no corrections) for inspection/scrubbing."""
    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    sp = _resolve_spacing(params, companion_txt, n_frames=vol.shape[0])
    return write_volume_nifti(vol, out_nifti, sp)


def preprocess_oct_to_nifti(oct_path: str | Path, out_nifti: str | Path,
                            params: dict | None = None, volume_index: int = 0,
                            progress=None, companion_txt: str | Path | None = None,
                            max_iterations: int = 1, min_improvement: float = 0.15,
                            abs_floor: float = 0.3, iter_dir: str | Path | None = None,
                            inject_pass: int | None = None, inject_force=None, inject_good=None) -> dict:
    """Full pipeline: read .OCT → smoother corrections → NIfTI with correct geometry.

    max_iterations<=1 → single pass. max_iterations>1 → iterative refinement (iterate_smooth_volume),
    auto-stopping when the boundary correction stops shrinking, then keeping the BEST pass (lowest
    in-plane deviation + axial roughness, #3). Both paths apply the over-correction guard (#2) +
    inter-slice smoothing (#3) — see smooth_volume (no longer byte-identical to DICOMSmootherSteps by
    design). FINALLY (#2 ping-pong) the chosen volume is AXIAL-refined: an axial correction pass kept
    per-frame where it makes the en-face boundary smoother (axial_refine param, default on; a global
    guard makes it never worse). When iter_dir is given, each INTERMEDIATE sagittal pass volume
    (V1..V(n-1)) is written there as pass_{k}.nii.gz so the UI can step through them; out_nifti is the
    axial-refined best (so the delivered volume can be slightly smoother than the last stepped pass).
    Returns {out, passes, metrics, applied, stopped}."""
    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    sp = _resolve_spacing(params, companion_txt, n_frames=vol.shape[0])
    if max_iterations and int(max_iterations) > 1:
        chain, best_idx, info = iterate_smooth_volume(
            vol, params, max_iter=int(max_iterations),
            min_improvement=min_improvement, abs_floor=abs_floor, progress=progress,
            inject_pass=inject_pass, inject_force=inject_force, inject_good=inject_good)
        corrected = chain[best_idx]                 # the BEST pass (least-deviant boundary)
        # Write EVERY corrected pass (V1..Vm) so the UI can step through them all and SEE why the
        # best was chosen (a worse pass is visibly more deviant). chain[0] = raw = context_raw.
        if iter_dir is not None and len(chain) > 1:
            idir = Path(iter_dir)
            idir.mkdir(parents=True, exist_ok=True)
            for k, pv in enumerate(chain[1:], start=1):
                write_volume_nifti(pv, idir / f"pass_{k}.nii.gz", sp)
    else:
        corrected, m, ax = smooth_volume(vol, params, progress=progress, return_metric=True)
        info = {"passes": 1, "best_pass": 1, "metrics": [float(m)], "axial_metrics": [float(ax)], "stopped": "single"}
    # #2 ping-pong: refine the sagittally-corrected volume with an AXIAL pass, kept per-frame only where
    # it makes the en-face boundary smoother (and only if the whole 3-D surface improves). Confirmed on
    # real scans to give the smoothest 3-D corneal surface; never worse than sagittal-only.
    p_all = {**DEFAULT_PARAMS, **(params or {})}
    if p_all.get("axial_refine", True):
        corrected, ref = axial_refine_volume(corrected, params)
        info["axial_refine"] = ref
    # #2 fix-columns drag-to-correct: apply the annotator's explicit per-frame manual depth nudges LAST,
    # so they override whatever the auto-correction left for those frames (manual ground truth wins).
    ms = p_all.get("manual_shifts")
    if ms:
        corrected, n_ms = apply_manual_shifts(corrected, ms)
        info["manual_shifts"] = {"n_frames": int(n_ms)}
    write_volume_nifti(corrected, out_nifti, sp)
    info["out"] = str(out_nifti)
    return info


# ── Diagnostic: render EVERY processing step for the central sagittal slice ──
# (mirrors the Streamlit generate_visualization_steps filmstrip; adds coronal steps on request).
_C_RED, _C_GREEN, _C_BLUE, _C_MAGENTA = (255, 64, 64), (64, 220, 96), (90, 150, 255), (235, 90, 235)


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to PNG with only stdlib (no preview_io dependency)."""
    import struct
    import zlib
    rgb = np.ascontiguousarray(np.asarray(rgb, np.uint8))
    H, W, _ = rgb.shape
    sl = np.empty((H, 1 + W * 3), np.uint8)
    sl[:, 0] = 0
    sl[:, 1:] = rgb.reshape(H, W * 3)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(sl.tobytes(), 6)) + chunk(b"IEND", b""))


def _gray_rgb(img2d: np.ndarray) -> np.ndarray:
    g = np.asarray(img2d, np.float32)
    f = g[np.isfinite(g)]
    lo, hi = (float(np.percentile(f, 1)), float(np.percentile(f, 99))) if f.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    u = (np.clip((g - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([u, u, u], -1)


def _draw_curve(rgb: np.ndarray, y_per_x: np.ndarray, color, dashed: bool = False) -> np.ndarray:
    H, W = rgb.shape[:2]
    for x in range(min(W, len(y_per_x))):
        if dashed and (x // 5) % 2:
            continue
        yy = int(round(float(y_per_x[x])))
        for dy in (-1, 0, 1):
            if 0 <= yy + dy < H:
                rgb[yy + dy, x] = color
    return rgb


def _disp_resize(rgb: np.ndarray, out_h: int = 320, out_w: int = 460) -> np.ndarray:
    H, W = rgb.shape[:2]
    if H == out_h and W == out_w:
        return rgb
    ri = np.linspace(0, H - 1, out_h).round().astype(int)
    ci = np.linspace(0, W - 1, out_w).round().astype(int)
    return np.ascontiguousarray(rgb[ri][:, ci])


def preprocess_steps(oct_path, params=None, volume_index=0, companion_txt=None,
                     bad_cols=None, workers=None):
    """Return [(label, rgb_uint8)] for every preprocessing step on the CENTRAL sagittal slice.
    Faithful to the per-slice pipeline; the final warp reflects the current bad-column selection
    so the filmstrip shows exactly what a re-run would produce."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    res = float(p["residual_threshold"]); cf = float(p.get("corr_factor", 1.0)); at = float(p.get("active_threshold", 5.0))
    vol = read_oct_zstack(oct_path, volume_index)
    sag = reformat_to_sagittal(vol)                 # (lateral, depth, frames)
    n = sag.shape[0]; idx = n // 2
    sl = sag[idx].astype(np.float32)
    steps = []

    def add(label, rgb):
        steps.append((label, _disp_resize(rgb)))

    add(f"1. Original — central sagittal slice ({idx}/{n})", _gray_rgb(sl))
    heq = _histeq(sl)
    add("2. Histogram equalized", _gray_rgb(heq))
    filt = cv2.bilateralFilter(cv2.normalize(heq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                               int(p["d"]), int(p["sigmaColor"]), int(p["sigmaSpace"]))
    add("3. Bilateral filtered", _gray_rgb(filt))
    raw_edge = _detect_surface_gradient(filt, p["sigma"])
    add("4. Surface edge detected (red)", _draw_curve(_gray_rgb(heq), raw_edge, _C_RED))
    merged = _merged_side_edge(sl, p)
    add("5. Side-corrected boundary (green)", _draw_curve(_gray_rgb(sl), merged, _C_GREEN))
    quad = _fit_quadratic_ransac(merged, res)
    im6 = _draw_curve(_gray_rgb(sl), merged, _C_GREEN)
    add("6. Quadratic fit — green=edge, blue=fit", _draw_curve(im6, quad, _C_BLUE, dashed=True))
    nb = [merged]
    if idx > 0:
        nb.append(_merged_side_edge(sag[idx - 1].astype(np.float32), p))
    if idx < n - 1:
        nb.append(_merged_side_edge(sag[idx + 1].astype(np.float32), p))
    med = np.median(np.stack(nb), axis=0)
    active_e = merged.copy(); dvv = np.abs(merged - med); active_e[dvv > at] = med[dvv > at]
    quad_a = _fit_quadratic_ransac(active_e, res)
    im7 = _draw_curve(_gray_rgb(sl), active_e, _C_MAGENTA)
    add("7. 3D active correction — magenta=corrected, blue=fit", _draw_curve(im7, quad_a, _C_BLUE, dashed=True))
    # Final warp: same logic as smooth_volume — with the over-correction guard (#2) so the filmstrip
    # matches a real re-run (runaway shift interpolated from good neighbours + clamped).
    disp = _slice_displacement(active_e, res, cf, [int(c) for c in (bad_cols or [])],
                               [int(c) for c in (p.get("good_columns") or [])],
                               float(p.get("max_displacement", 0.0) or 0.0))
    warped = _warp_by_displacement(sag[idx], disp)
    add("8. Final corrected — column warp", _gray_rgb(warped.astype(np.float32)))
    return steps


# ── CLI: run the heavy pipeline in an isolated subprocess (called by the sidecar,
#    so the fork-based parallelism never touches the sidecar's CUDA/torch state) ──
if __name__ == "__main__":
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description="OCT preprocessing worker")
    ap.add_argument("mode", choices=["raw", "preprocess", "steps"])
    ap.add_argument("oct_path")
    ap.add_argument("out_nifti")   # for mode=steps this is the OUTPUT DIRECTORY for the step PNGs
    ap.add_argument("--params", default="{}")
    ap.add_argument("--volume-index", type=int, default=0)
    ap.add_argument("--companion-txt", default="")
    ap.add_argument("--bad-cols", default="[]")
    ap.add_argument("--max-iter", type=int, default=1)        # >1 = iterative refinement
    ap.add_argument("--min-improvement", type=float, default=0.15)
    ap.add_argument("--abs-floor", type=float, default=0.3)
    ap.add_argument("--iter-dir", default="")                 # where to write intermediate pass NIfTIs
    ap.add_argument("--inject-pass", type=int, default=0)     # apply the column fix at ONLY this pass (1-based; 0=none)
    ap.add_argument("--inject-force", default="[]")           # bad frame indices for the injected pass
    ap.add_argument("--inject-good", default="[]")            # good/anchor frame indices for the injected pass
    a = ap.parse_args()
    _p = _json.loads(a.params)
    _comp = a.companion_txt or None
    if a.mode == "raw":
        raw_oct_to_nifti(a.oct_path, a.out_nifti, volume_index=a.volume_index, params=_p, companion_txt=_comp)
    elif a.mode == "steps":
        _steps = preprocess_steps(a.oct_path, params=_p, volume_index=a.volume_index, companion_txt=_comp,
                                  bad_cols=_json.loads(a.bad_cols or "[]"))
        _outdir = Path(a.out_nifti)
        _outdir.mkdir(parents=True, exist_ok=True)
        for old in _outdir.glob("step_*.png"):   # clear stale steps from a prior run
            old.unlink()
        _labels = []
        for _i, (_label, _rgb) in enumerate(_steps):
            _fn = f"step_{_i:02d}.png"
            (_outdir / _fn).write_bytes(_png_bytes(_rgb))
            _labels.append({"label": _label, "file": _fn})
        (_outdir / "labels.json").write_text(_json.dumps(_labels))
    else:
        _info = preprocess_oct_to_nifti(
            a.oct_path, a.out_nifti, params=_p, volume_index=a.volume_index, companion_txt=_comp,
            max_iterations=a.max_iter, min_improvement=a.min_improvement, abs_floor=a.abs_floor,
            iter_dir=(a.iter_dir or None),
            inject_pass=(a.inject_pass or None), inject_force=_json.loads(a.inject_force or "[]"),
            inject_good=_json.loads(a.inject_good or "[]"))
        # Single machine-readable line the sidecar parses for the per-pass convergence report.
        print("ITER " + _json.dumps(_info))
    print("OK " + str(a.out_nifti))
