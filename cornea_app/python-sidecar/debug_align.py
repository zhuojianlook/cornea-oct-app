"""debug_align.py — replicate-alignment comparison for the Debug tab.

Pick two repeat scans of ONE eye, run several alignment methods, and SEE the overlap as a
MAGENTA (fixed) / GREEN (moving) composite: where they align it reads WHITE/GREY, where they
do not you get magenta and green fringes. This is a VISUAL ADJUDICATION tool — the numbers are
here to support the picture, not replace it.

METHODS
  identity   — no alignment. The essential reference: without it no other number means anything.
  asis       — registration._rigid_intensity EXACTLY as shipped (lr 0.8, sigmas [2.0,1.0,0.0]).
               Measured over 178 replicate pairs it raises ITK's "All samples map outside moving
               image buffer" on 82/178 (46%) and only beats identity on 30/178 (17%).
               IMPORTANT — THIS IS NOT A PRODUCTION BUG. align_transform wraps this call in a
               best-of-identity cornea-Dice guard and the `except` sets d_rig = -1.0, a deliberate
               sentinel that forces identity. In production the method is INERT, not harmful. This
               tab exists to show that inertness, not to imply the shipped app is broken.
               NOT RUN-TO-RUN REPRODUCIBLE, despite the hardcoded Mattes seed=1: measured here on
               cs001_os v2->v3, three consecutive in-process runs gave rot 4.59 / 4.56 / 4.41 deg.
               Pinning ITK to a single thread makes it deterministic (4.894), so the variance is
               multithreaded floating-point reduction order in the metric, amplified by an optimiser
               that is in a chaotic regime at lr 0.8 with 2.0 mm sigmas. `fixed` is bit-identical at
               1 and 24 threads. We do NOT pin threads here: production runs multithreaded, and this
               tab must show production as it actually behaves. So expect asis's numbers/picture to
               wobble slightly between runs — that instability is itself part of the finding.
  fixed      — the SAME function with two constants changed: sigmas [2.0,1.0,0.0] -> [0.04,0.02,0.0]
               mm and lr 0.8 -> 0.03, mask=None. 0 raises, 163/178 (91.6%) beat identity, median
               delta +0.2607. (The sigmas are PHYSICAL mm on a 0.02 mm iso grid, so the shipped
               2.0 mm blurs by 100 voxels and erases the cornea at the coarse levels.)
  bruteforce — exhaustive full-res 3-DOF translation by FFT cross-correlation. Deterministic; no
               optimiser, no seed, no local minimum. NEVER subsample OCT speckle without
               anti-aliasing in a coarse-to-fine search — it aliases and traps the fine pass in a
               local optimum. This searches EXHAUSTIVELY at full resolution for that reason.

TEASER++ is deliberately absent: it needs a hand-built C++ .so ABI-pinned to Python 3.10/x86_64,
is not on PyPI, and this sidecar ships bundled through CI. It was also the worst working method.

SCORING — ported from /home/zhuojian/Desktop/teaser_bench/evalmetric.py (a bench directory; NOT
imported at runtime, the sidecar must be self-contained). Two properties MUST survive the port or
every number below is meaningless:
  (a) A FIXED, TRANSFORM-INDEPENDENT eval mask derived from the FIXED volume ONLY (Otsu + 0.2 mm
      dilation), so no transform can win by changing what gets scored. A tissue-ONLY mask is
      provably WRONG here: inside the cornea the only variation is speckle, an independent
      realisation per acquisition, which correlates -0.11 between replicates. The alignment signal
      is the bright-cornea/dark-background EDGE, so the mask must SPAN it.
  (b) BLUR-MATCHING on a common isotropic 0.02 mm OFF-GRID lattice, with the FIXED resampled
      through it too. Otherwise identity is never interpolated while every other transform is, and
      trilinear interpolation smooths independent speckle for a FREE +0.03-0.05 NCC (measured:
      a physically meaningless HALF-VOXEL shift bought +0.0424/+0.0495).
Out-of-FOV voxels are PENALISED, never dropped: they stay in the score valued 0, so evicting bright
tissue costs correlation. Differences in 'primary' below ~0.005 (BLUR_FLOOR) are noise.

DO NOT RANK ON 'primary' — rank on resid_um. NCC scored the 2-constant fix (0.8547) and brute-force
translation (0.8432) as a TIE while their surface residuals differ 6x (1.6 vs 9.6 vox), because the
v2->v3 offset is TILTED and NCC over a dilated mask is dominated by bulk overlap. 'primary' is an
intensity proxy; the surface residual is the geometry, and the geometry is what propagating a scar
label onto a replicate actually depends on. See the surface_residual block below.

RENDER SCRATCH: PNGs go to a temp cache dir, NEVER under the case store. The app's own convention
is WORKSPACE_ROOT/"output", but WORKSPACE_ROOT follows CORNEA_DATA_DIR — which for the review
workflow points AT the read-only case store — so this uses the system temp dir instead. These PNGs
are disposable debug output; nothing here is user data worth persisting.

ANCHOR (held-out, reproduced by this module's own test): cs001_os v2 (fixed) vs v3 (moving) —
identity primary 0.3165, 2-constant fix 0.8547, rot 1.438 deg, t_eff [-0.015, -0.213, -0.074] mm.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi

import orchestration as orch
import registration as reg
import settings

# ── metric constants (evalmetric.py, verbatim) ───────────────────────────────
G = 0.02                 # isotropic evaluation grid spacing, mm
DILATE_MM = 0.2          # tissue dilation for the eval mask, mm
FRAC_OUT_REJECT = 0.25   # frac_out above this sets reject=True
BLUR_FLOOR = 0.005       # measured residual free-blur gain; deltas below this are noise

# ── the 2-constant fix ───────────────────────────────────────────────────────
FIX_LR = 0.03
FIX_SIGMAS = (0.04, 0.02, 0.0)

# ── brute-force search half-window, mm (per axis: lat, depth, frames) ────────
# Matches the reference's verified window on the canonical grid (~0.62/0.47/0.80 mm) but is
# expressed in mm so it survives the real spacing heterogeneity across this cohort.
BF_RADIUS_MM = (0.6, 0.5, 0.8)

# ── render ───────────────────────────────────────────────────────────────────
RENDER_WIDTH = 700
RENDER_MAX_H = 900
MARGIN_MM = 0.15         # tissue-bbox margin
ZOOM_LAT_MM = 2.0        # zoom crop width around the apex
# Zoom crop above/below the apex surface. The window is anchored on the FIXED volume's surface, so
# the headroom must EXCEED the misalignment it exists to reveal — the moving scan's surface sits
# above it by exactly the offset under inspection (0.179 mm on the canonical cs001_os v2/v3 pair).
# At 0.15 mm the moving surface fell outside the crop, hiding the very thing being adjudicated.
ZOOM_UP_MM, ZOOM_DOWN_MM = 0.30, 0.55

# ── 3-D interactive replicate-agreement volumes (Debug tab "see the consensus in 3D") ────
# NO backend turntable any more. The Debug tab now renders the volumes LIVE on the GPU in a dedicated
# niivue instance (the 3-D SLICE_TYPE.RENDER path, which works on this WebKitGTK stack). The backend's
# only job is to WRITE the isotropic cropped volumes as .nii.gz and serve them token-exempt; niivue
# rotates/pans/zooms them at full resolution. This is both CHEAPER (no 24-angle CPU rotate+MIP) and
# far CRISPER (a fine iso grid the GPU renders directly, vs pre-baked downsampled MIP PNGs).
# Two content modes, SAME window/crop/grid across every method (all per-job constants):
#   OVERLAP       fixed + aligned-moving as two volumes (magenta / green in the client).
#   DISAGREEMENT  |fixed - aligned_moving| gated to the tissue edge, one HOT scalar volume.
DBG_ISO_MM = 0.02         # interactive render grid (mm) = reg.ISO. FINE: ~5x the linear detail of the
                          # old 0.035 mm turntable grid, and the GPU renders it live so cost is trivial.
DBG_DILATE_MM = 0.30      # disagreement gate: dilate the FIXED tissue mask to SPAN the surface edge, so a
                          # surface OFFSET (tissue in one scan, background in the other) is not masked out.
DBG_HOT_PCT = 99.0        # disagreement normaliser: percentile of the IDENTITY diff, SHARED across methods.

METHODS: dict[str, str] = {
    "identity": "identity (no alignment)",
    "asis": "production as-is",
    "fixed": "2-constant fix",
    "bruteforce": "brute-force translation",
}
VIEWS = ("bscan", "sagittal", "zoom")

_RENDER_ROOT = Path(tempfile.gettempdir()) / "cornea_debug_align"
_MAX_KEPT_JOBS = 8
# Age after which a job dir from a PREVIOUS sidecar process is swept. _MAX_KEPT_JOBS bounds the dirs
# of THIS process; only age can bound the ones left by a process that is gone. See sweep_render_root.
_JOB_TTL_S = 6 * 3600.0

# ═════════════════════════════════════════════════════════════════════════════
# Per-replicate LABEL cache (the SCAR consensus space)
# ═════════════════════════════════════════════════════════════════════════════
# The scar consensus needs each replicate's cornea + scar labelmap. NONE exist on disk, and SAM2 costs
# ~100 s/vol, so the labelmaps are cached PERSISTENTLY, keyed by case_id + preview-volume mtime + a
# params hash, in a dedicated /tmp dir. A data change (new mtime) or a params change (new hash) misses
# cleanly. NEVER written inside the read-only case store — mirrors _RENDER_ROOT's system-temp policy.
_LABEL_CACHE_ROOT = Path(tempfile.gettempdir()) / "cornea_debug_labels"
# Best-effort WARM source: the reproducibility study already computed these EXACT labelmaps (same SAM2
# vote=2 + regularize_cornea + hysteresis phi=92 recipe) for several eyes. If a mask there matches the
# current preview volume's SHAPE it is copied into the cache instead of re-running SAM2 (~15 min for a
# 9-rep eye). Read-only; skipped silently if absent or shape-mismatched. Overridable for tests/deploys.
_DENOISE_WARM_DIR = Path(
    os.environ.get("CORNEA_DEBUG_LABEL_WARM_DIR", "/home/zhuojian/Desktop/teaser_bench/denoise/masks")
)
# Label-generation parameters, baked into the cache key so any change invalidates cleanly. These are the
# PRODUCTION defaults (the expC_build_masks recipe): SAM2 3-plane vote=2 -> regularize_cornea, then
# detect_scar_hysteresis on the RAW volume.
LABEL_PARAMS: dict = {
    "sam2_planes": ("axial", "coronal", "sagittal"),
    "sam2_vote": 2,
    "regularize": True,
    "phi_percentile": 92.0,
    "gap": 12.0,
    "erode_surface": 6,
    "smooth": 2.5,
    "min_voxels": 500,
    "open_iter": 1,
    "close_iter": 4,
}


def _label_params_hash() -> str:
    return hashlib.md5(repr(sorted(LABEL_PARAMS.items())).encode()).hexdigest()[:12]


def label_cache_paths(case_id: str) -> tuple[Path, Path]:
    """(cornea_path, scar_path) for a case's cached labelmaps, keyed by case + preview mtime + params
    hash so a data or params change misses cleanly. Paths live under the /tmp label cache and may not
    exist yet — NEVER under the case store."""
    case_id = orch.safe_case_id(case_id)
    try:
        mt = int(volume_path(case_id).stat().st_mtime)
    except OSError:
        mt = 0
    key = f"{case_id}__{mt}__{_label_params_hash()}"
    return (_LABEL_CACHE_ROOT / f"cornea__{key}.nii.gz",
            _LABEL_CACHE_ROOT / f"scar__{key}.nii.gz")


def _save_label_cache(path: Path, arr: np.ndarray) -> None:
    """Write a labelmap to the cache as a gzip NIfTI, via a temp file + atomic rename so a crash mid-write
    can never leave a half file the next run trusts. Affine is identity: only the array (native
    [lat,depth,frames] order) is ever read back."""
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex[:8]}.tmp.nii.gz"
    nib.save(nib.Nifti1Image(np.ascontiguousarray(arr), np.eye(4)), str(tmp))
    tmp.replace(path)


def _warm_labels_from_denoise(case_id: str, ref_shape, cornea_p: Path, scar_p: Path) -> bool:
    """Copy the reproducibility-study cornea+scar masks into the cache when they match this case's
    volume SHAPE (skips a ~100 s/vol SAM2 re-run). Best-effort + read-only on the warm source: returns
    True on a verified hit, False on absent/shape-mismatch/any error."""
    if not _DENOISE_WARM_DIR.exists():
        return False
    rep = case_id[len("case_"):] if case_id.startswith("case_") else case_id   # case_cs001_od_v1 -> cs001_od_v1
    src_cornea = _DENOISE_WARM_DIR / f"cornea_{rep}.nii.gz"
    src_scar = _DENOISE_WARM_DIR / f"scar_{rep}.nii.gz"
    if not (src_cornea.exists() and src_scar.exists()):
        return False
    try:
        import nibabel as nib
        if tuple(nib.load(str(src_cornea)).shape) != tuple(ref_shape):
            return False
        if tuple(nib.load(str(src_scar)).shape) != tuple(ref_shape):
            return False
        _LABEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_cornea, cornea_p)
        shutil.copyfile(src_scar, scar_p)
        return True
    except Exception:  # noqa: BLE001 — warming is best-effort; fall back to a cold SAM2 build
        return False


def load_or_build_labels(case_id: str, progress=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-replicate cornea ({0,1} uint8) + scar (bool) labelmaps on the case's NATIVE grid.

    Cache-first: a hit returns instantly; a miss warms from the study masks if shape-matched, else runs
    the PRODUCTION chain — SAM2 3-plane vote=2 -> regularize_cornea, then detect_scar_hysteresis on the
    RAW volume (~100 s/vol on the 3090) — and writes the cache. `progress(phase, index, total)` is
    forwarded to SAM2 so a caller can surface "segmenting … · axial". Returns (cornea, scar, meta) where
    meta = {source, degraded, cornea_voxels}; a DEGRADED replicate (SAM2 plane failure or empty cornea)
    returns an all-zero scar and is NOT cached (a later good run must be free to replace it). NEVER
    writes into the case store — the cache lives under _LABEL_CACHE_ROOT (/tmp)."""
    import nibabel as nib
    case_id = orch.safe_case_id(case_id)
    volp = volume_path(case_id)
    if not volp.exists():
        raise FileNotFoundError(f"No preview volume for {case_id}")
    cornea_p, scar_p = label_cache_paths(case_id)

    if cornea_p.exists() and scar_p.exists():
        cornea = np.rint(np.asarray(nib.load(str(cornea_p)).dataobj)).astype(np.uint8)
        scar = np.asarray(nib.load(str(scar_p)).dataobj) > 0
        return cornea, scar, {"source": "cache", "degraded": False,
                              "cornea_voxels": int((cornea >= 1).sum())}

    base = nib.load(str(volp))
    raw = np.asarray(base.dataobj).astype(np.float32)

    if _warm_labels_from_denoise(case_id, raw.shape, cornea_p, scar_p):
        cornea = np.rint(np.asarray(nib.load(str(cornea_p)).dataobj)).astype(np.uint8)
        scar = np.asarray(nib.load(str(scar_p)).dataobj) > 0
        return cornea, scar, {"source": "warm", "degraded": False,
                              "cornea_voxels": int((cornea >= 1).sum())}

    # Cold build (GPU). Lazy imports so a plain `import debug_align` (tests, api_server startup) never
    # pulls in torch/SAM2.
    import sam2_segment
    import scar as scar_mod
    _LABEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sam2_", dir=str(_LABEL_CACHE_ROOT)) as work:
        label, meta = sam2_segment.segment_volume(
            volp, Path(work), planes=tuple(LABEL_PARAMS["sam2_planes"]),
            vote=int(LABEL_PARAMS["sam2_vote"]), progress=progress)
    cornea = scar_mod.regularize_cornea(label).astype(np.uint8)
    cvox = int((cornea >= 1).sum())
    if bool(meta.get("degraded")) or cvox == 0:
        return cornea, np.zeros(cornea.shape, bool), {
            "source": "sam2", "degraded": True, "cornea_voxels": cvox}
    scar = scar_mod.detect_scar_hysteresis(
        raw, cornea, phi_percentile=float(LABEL_PARAMS["phi_percentile"]),
        gap=float(LABEL_PARAMS["gap"]), erode_surface=int(LABEL_PARAMS["erode_surface"]),
        smooth=float(LABEL_PARAMS["smooth"]), min_voxels=int(LABEL_PARAMS["min_voxels"]),
        open_iter=int(LABEL_PARAMS["open_iter"]), close_iter=int(LABEL_PARAMS["close_iter"])).astype(bool)
    _save_label_cache(cornea_p, cornea.astype(np.uint8))
    _save_label_cache(scar_p, scar.astype(np.uint8))
    return cornea, scar, {"source": "sam2", "degraded": False, "cornea_voxels": cvox}


# ═════════════════════════════════════════════════════════════════════════════
# Replicate enumeration
# ═════════════════════════════════════════════════════════════════════════════
# TWO naming schemes exist and BOTH are genuine repeats of the same eye:
#   A: case_cs001_os_v1, _v2, _v3          B: case_cs030_od_v1, _v1_2, _v1_3
# Scheme B MUST be matched FIRST. A naive /_v(\d+)$/ is greedy: it silently drops the 97 scheme-B
# cases (211/308 captured) AND mis-parses case_cs030_od_v1_2 as eye "cs030_od_v1" replicate 2,
# splitting one eye into phantom singletons. Verified on the real store: scheme-B-first yields
# 88 eyes / 86 with >=2 replicates; naive yields 88 eyes from only 211 cases.
_RE_B = re.compile(r"^case_(?P<eye>.+?)_v(?P<v>\d+)_(?P<r>\d+)$")
_RE_A = re.compile(r"^case_(?P<eye>.+?)_v(?P<v>\d+)$")


def parse_case(name: str) -> tuple[str | None, tuple[int, int] | None]:
    """case dir name -> (eye, sort key). Scheme B is tried FIRST — see the note above."""
    m = _RE_B.match(name)
    if m:
        return m.group("eye"), (int(m.group("v")), int(m.group("r")))
    m = _RE_A.match(name)
    if m:
        return m.group("eye"), (int(m.group("v")), 0)
    return None, None


def volume_path(case_id: str) -> Path:
    return settings.CASES_ROOT / orch.safe_case_id(case_id) / "previews" / "volume.nii.gz"


def groups() -> list[dict]:
    """Eyes with >=2 replicate scans that actually have a readable preview volume."""
    root = settings.CASES_ROOT
    if not root.exists():
        return []
    by_eye: dict[str, list[tuple[tuple[int, int], str]]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        eye, key = parse_case(child.name)
        if eye is None or not (child / "previews" / "volume.nii.gz").exists():
            continue
        by_eye.setdefault(eye, []).append((key, child.name))
    out = []
    for eye in sorted(by_eye):
        cases = [n for _, n in sorted(by_eye[eye])]
        if len(cases) >= 2:
            out.append({"eye": eye, "cases": cases})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Volume I/O  (evalmetric.load / to_sitk, ported)
# ═════════════════════════════════════════════════════════════════════════════
def load_volume(case_id: str) -> tuple[np.ndarray, list[float]]:
    """-> (float32 [lat, depth, frames], [mm, mm, mm]). Axis order per evalmetric.load."""
    import nibabel as nib
    p = volume_path(case_id)
    if not p.exists():
        raise FileNotFoundError(f"No preview volume for {case_id}")
    img = nib.load(str(p))
    return img.get_fdata().astype(np.float32), [float(z) for z in img.header.get_zooms()]


def to_sitk(vol: np.ndarray, spacing) -> sitk.Image:
    """numpy [lat, depth, frames] -> sitk.Image with x=lat, y=depth, z=frames, origin 0, identity
    direction — the same convention as registration._canon, so transforms are interchangeable."""
    im = sitk.GetImageFromArray(np.transpose(np.asarray(vol, np.float32), (2, 1, 0)))
    im.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    im.SetOrigin((0.0, 0.0, 0.0))
    im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return im


def _from_sitk(img: sitk.Image) -> np.ndarray:
    """sitk (z=frames, y=depth, x=lat) -> numpy [lat, depth, frames]."""
    return np.transpose(sitk.GetArrayFromImage(img), (2, 1, 0))


# ═════════════════════════════════════════════════════════════════════════════
# Metric  (ported from teaser_bench/evalmetric.py — see the module docstring)
# ═════════════════════════════════════════════════════════════════════════════
_fixed_cache: dict = {}
_fixed_cache_order: list = []
_MAX_FIXED_CACHE = 3


def _fingerprint(img: sitk.Image):
    a = sitk.GetArrayViewFromImage(img)
    h = hashlib.md5(np.ascontiguousarray(a[::16, ::16, ::16]).tobytes()).hexdigest()
    return (h, img.GetSize(), img.GetSpacing(), img.GetOrigin(), float(a.sum()))


def _sigmas(img: sitk.Image) -> list[float]:
    """Anti-alias sigma per axis (mm): smooth to the coarser of (eval grid, source grid). Applied
    IDENTICALLY to both images — this is what kills the free-interpolation-blur bonus."""
    return [max(G, s) / 2.0 for s in img.GetSpacing()]


def _smooth(img: sitk.Image, sig) -> sitk.Image:
    return sitk.SmoothingRecursiveGaussian(img, [float(s) for s in sig])


def _eval_grid(ref: sitk.Image) -> sitk.Image:
    """Isotropic G-mm lattice strictly inside ref's interpolation domain, offset off-grid by G/2 so
    the FIXED is genuinely interpolated too (no identity-only sharpness)."""
    ext = [(ref.GetSize()[i] - 1) * ref.GetSpacing()[i] for i in range(3)]
    org = [G / 2.0] * 3
    size = [int(np.floor((ext[i] - org[i]) / G)) + 1 for i in range(3)]
    im = sitk.Image(size, sitk.sitkFloat32)
    im.SetSpacing((G, G, G))
    im.SetOrigin(tuple(org))
    im.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return im


def _tx(R=None, t=None) -> sitk.Transform:
    """R (3x3) / t (mm) in the ORIGIN-CENTERED convention: maps FIXED physical points -> MOVING
    physical points, rotating about (0,0,0) — a CORNER of the volume, not its centre."""
    tx = sitk.Euler3DTransform()
    if R is not None:
        tx.SetMatrix(tuple(np.asarray(R, float).ravel()))
        tx.SetTranslation(tuple(np.asarray(t if t is not None else [0, 0, 0], float).ravel()))
    return tx


def _onto(grid: sitk.Image, img: sitk.Image, tx: sitk.Transform, default: float = 0.0) -> sitk.Image:
    return sitk.Resample(img, grid, tx, sitk.sitkLinear, default, sitk.sitkFloat32)


def _fixed_side(fixed_img: sitk.Image):
    """Everything depending ONLY on the fixed image. Cached: the eval region must be byte-identical
    across every transform scored against this fixed image, or the comparison is rigged."""
    key = _fingerprint(fixed_img)
    if key in _fixed_cache:
        return _fixed_cache[key]
    grid = _eval_grid(fixed_img)
    F = sitk.GetArrayFromImage(_onto(grid, _smooth(fixed_img, _sigmas(fixed_img)), _tx()))
    ot = sitk.OtsuThresholdImageFilter()
    ot.SetInsideValue(0)
    ot.SetOutsideValue(1)
    tis = sitk.GetArrayFromImage(ot.Execute(sitk.GetImageFromArray(F))).astype(bool)
    st = ndi.generate_binary_structure(3, 1)
    masks = {
        "tissue": tis,
        "dil0.2": ndi.binary_dilation(tis, st, iterations=max(1, int(round(0.2 / G)))),
        "dil0.4": ndi.binary_dilation(tis, st, iterations=max(1, int(round(0.4 / G)))),
        "fov": np.ones_like(tis, bool),
    }
    out = (grid, F, masks, float(ot.GetThreshold()))
    _fixed_cache[key] = out
    _fixed_cache_order.append(key)
    while len(_fixed_cache_order) > _MAX_FIXED_CACHE:
        _fixed_cache.pop(_fixed_cache_order.pop(0), None)
    return out


def _ncc(x, y) -> float:
    x = x.astype(np.float64) - x.mean()
    y = y.astype(np.float64) - y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float("nan")


def _nmi(x, y, bins: int = 64) -> float:
    h, _, _ = np.histogram2d(x, y, bins=bins)
    p = h / h.sum()
    px, py = p.sum(1), p.sum(0)

    def H(q):
        q = q[q > 0]
        return float(-(q * np.log(q)).sum())

    hxy = H(p.ravel())
    return (H(px) + H(py)) / hxy if hxy > 0 else float("nan")


def score(fixed_img: sitk.Image, moving_img: sitk.Image, R=None, t=None) -> dict:
    """Score a rigid transform. Higher 'primary' = better. The eval region comes from fixed_img
    ALONE and is identical for every transform; out-of-FOV voxels stay in the score valued 0."""
    grid, F, masks, otsu = _fixed_side(fixed_img)
    tx = _tx(R, t)

    M = _smooth(moving_img, _sigmas(moving_img))
    B = sitk.GetArrayFromImage(_onto(grid, M, tx, 0.0))

    # Geometric coverage: resample a ones-image, so "outside the FOV" is never confused with
    # "a voxel whose value happens to be 0".
    ones = sitk.Image(moving_img.GetSize(), sitk.sitkFloat32)
    ones.CopyInformation(moving_img)
    ones = sitk.Add(ones, 1.0)
    cov = sitk.GetArrayFromImage(_onto(grid, ones, tx, 0.0)) > 0.5

    m = masks["dil0.2"]
    inf = cov[m]
    frac_out = float(1.0 - inf.mean())

    fx, bx = F[m], B[m]
    bx = np.where(inf, bx, 0.0)   # explicit: evicted voxels stay in, valued 0
    res = {
        "primary": _ncc(fx, bx),
        "n_voxels": int(m.sum()),
        "frac_out": frac_out,
        "reject": bool(frac_out > FRAC_OUT_REJECT),
        "n_out": int((~inf).sum()),
        "ncc_in": _ncc(fx[inf], bx[inf]) if inf.sum() > 1000 else float("nan"),
        "nmi": _nmi(fx, bx),
        "otsu": otsu,
    }
    for nm in ("dil0.4", "fov"):
        mm = masks[nm]
        i2 = cov[mm]
        res["ncc_" + nm] = _ncc(F[mm], np.where(i2, B[mm], 0.0))
    return res


# ═════════════════════════════════════════════════════════════════════════════
# Transform helpers  (verify_incumbent.extract_rigid / bench2.ang, ported)
# ═════════════════════════════════════════════════════════════════════════════
def extract_rigid(tx: sitk.Transform) -> tuple[np.ndarray, np.ndarray]:
    """Extract (R, t_eff) in the ORIGIN-CENTERED convention from a sitk rigid transform.

    NOT GetMatrix() on the CompositeTransform (a real bug the bench hit): flatten, require a single
    Euler3D, then y = R(x-c)+c+t = Rx + (c+t-Rc)."""
    t = tx if isinstance(tx, sitk.CompositeTransform) else sitk.CompositeTransform(tx)
    n = t.GetNumberOfTransforms()
    if n != 1:
        raise ValueError(f"expected a single rigid transform, got a composite of {n}")
    e = sitk.Euler3DTransform(t.GetNthTransform(0))   # downcast; raises if not Euler3D
    R = np.array(e.GetMatrix(), float).reshape(3, 3)
    c = np.array(e.GetCenter(), float)
    tr = np.array(e.GetTranslation(), float)
    return R, c + tr - R @ c


def ang(R) -> float:
    return float(np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def centre_translation(tx: sitk.Transform) -> list[float]:
    """The Euler3D's own GetTranslation() = the physical displacement of the volume CENTRE. t_eff is
    origin-centered and absorbs an (R-I)@c term, so it is NOT the displacement a clinician expects."""
    t = tx if isinstance(tx, sitk.CompositeTransform) else sitk.CompositeTransform(tx)
    e = sitk.Euler3DTransform(t.GetNthTransform(0))
    return [float(x) for x in e.GetTranslation()]


# ═════════════════════════════════════════════════════════════════════════════
# Brute-force exhaustive translation (FFT cross-correlation)
# ═════════════════════════════════════════════════════════════════════════════
# Ported from teaser_bench/viz/bf_translation.py, whose FFT+integral-image NCC was verified against
# direct computation to 2e-7. THAT reference runs on torch/CUDA; torch is NOT a declared sidecar
# dependency, so this is a scipy.fft port. Two further deviations, both deliberate:
#   - the reference ASSERTS equal shape+spacing and would CRASH on the real heterogeneity in this
#     cohort (cs005_od v1 0.0078 vs v9 0.01111 mm); here the moving is resampled onto the fixed grid
#     first, which is exact in physical space (both are origin-0 / identity-direction).
#   - no coordinate-descent refinement ON the eval metric: that would be tuning the method against
#     the very number used to rank it. The exhaustive argmax + a 3-point parabola is deterministic
#     and objective-independent.
def _integral(v: np.ndarray) -> np.ndarray:
    # float64 is not optional: these are running sums over ~33M voxels of squared intensities
    # (~1e13); float32 would lose the low-order bits the NCC denominator is made of.
    a = np.cumsum(v, axis=0, dtype=np.float64)
    np.cumsum(a, axis=1, out=a)
    np.cumsum(a, axis=2, out=a)
    return np.pad(a, ((1, 0), (1, 0), (1, 0)))


def _boxsum(I, a0, a1, b0, b1, c0, c1):
    A0, B0, C0 = a0[:, None, None], b0[None, :, None], c0[None, None, :]
    A1, B1, C1 = a1[:, None, None], b1[None, :, None], c1[None, None, :]
    return (I[A1, B1, C1] - I[A0, B1, C1] - I[A1, B0, C1] - I[A1, B1, C0]
            + I[A0, B0, C1] + I[A0, B1, C0] + I[A1, B0, C0] - I[A0, B0, C0])


def _para(vals) -> float:
    """3-point parabola vertex offset, clipped to +-0.5 voxel."""
    a, b, c = [float(x) for x in vals]
    d = a - 2 * b + c
    return 0.0 if abs(d) < 1e-12 else float(np.clip(-0.5 * (c - a) / d, -0.5, 0.5))


def bruteforce_translation(vf: np.ndarray, vm: np.ndarray, sp) -> tuple[np.ndarray, dict]:
    """Exhaustive full-res NCC over every integer lag in a physical window, + sub-voxel parabola.

    vf/vm share a grid with spacing sp. Sign convention (derived + numerically verified in the
    reference): C[d] = sum_p fixed[p]*moving[p+d], so a peak at lag d means fixed[p] ~ moving[p+d];
    with the corner-origin Euler3D (p_mov = p_fix + t) that gives t_mm = lag * spacing.
    Returns (t_mm, info).
    """
    from scipy import fft as sfft

    shape = vf.shape
    sp = np.asarray(sp, float)
    # Physical half-window -> voxels, clamped so the search box cannot leave the volume.
    rad = []
    for i in range(3):
        r = int(round(BF_RADIUS_MM[i] / max(sp[i], 1e-9)))
        rad.append(int(max(1, min(r, shape[i] // 2 - 4))))
    DL, DD, DF = rad
    box = (DL, shape[0] - DL, DD, shape[1] - DD, DF, shape[2] - DF)
    L0, L1, D0, D1, F0, F1 = box
    if L1 - L0 < 8 or D1 - D0 < 8 or F1 - F0 < 4:
        raise ValueError(
            f"volume too small for an exhaustive search (shape {shape}, window {(DL, DD, DF)})")

    N = (L1 - L0) * (D1 - D0) * (F1 - F0)
    X = np.zeros(shape, np.float64)
    b = vf[L0:L1, D0:D1, F0:F1].astype(np.float64)
    xh = b - b.mean()
    X[L0:L1, D0:D1, F0:F1] = xh
    xn = float(np.linalg.norm(xh))

    # One FFT cross-correlation gives the numerator sum(x_hat[p]*m[p+d]) for EVERY lag at once.
    # Freed eagerly step by step: naively chained, the float64 spectra of a 33M-voxel volume peak
    # over 2 GB.
    m64 = vm.astype(np.float64)
    Fx = sfft.rfftn(X, workers=-1)
    del X
    Fm = sfft.rfftn(m64, workers=-1)
    np.conj(Fx, out=Fx)
    Fx *= Fm
    del Fm
    C = sfft.irfftn(Fx, s=shape, workers=-1)
    del Fx
    # Integral images of m and m^2 give the EXACT per-lag windowed mean/variance, so the NCC
    # denominator is exact at every lag — no windowing bias, no approximation.
    I1 = _integral(m64)
    sq = m64 * m64
    del m64
    I2 = _integral(sq)
    del sq

    dls, dds, dfs = (np.arange(-DL, DL + 1), np.arange(-DD, DD + 1), np.arange(-DF, DF + 1))
    S1 = _boxsum(I1, L0 + dls, L1 + dls, D0 + dds, D1 + dds, F0 + dfs, F1 + dfs)
    S2 = _boxsum(I2, L0 + dls, L1 + dls, D0 + dds, D1 + dds, F0 + dfs, F1 + dfs)
    del I1, I2
    num = C[(dls % shape[0])[:, None, None], (dds % shape[1])[None, :, None],
            (dfs % shape[2])[None, None, :]]
    denom = xn * np.sqrt(np.clip(S2 - S1 * S1 / N, 1e-9, None))
    NCC = num / denom
    del C, S1, S2, num, denom

    i, j, k = np.unravel_index(np.nanargmax(NCC), NCC.shape)
    lag = np.array([i - DL, j - DD, k - DF])
    peak = float(NCC[i, j, k])
    ident = float(NCC[DL, DD, DF])
    edge = bool(abs(lag[0]) == DL or abs(lag[1]) == DD or abs(lag[2]) == DF)

    sub = np.array([
        _para(NCC[i - 1:i + 2, j, k]) if 0 < i < NCC.shape[0] - 1 else 0.0,
        _para(NCC[i, j - 1:j + 2, k]) if 0 < j < NCC.shape[1] - 1 else 0.0,
        _para(NCC[i, j, k - 1:k + 2]) if 0 < k < NCC.shape[2] - 1 else 0.0,
    ])
    t_mm = (lag + sub) * sp
    return t_mm, {
        "lag_vox": [int(x) for x in lag],
        "subvox": [float(x) for x in sub],
        "box_ncc_peak": peak,
        "box_ncc_identity": ident,
        "window_vox": [DL, DD, DF],
        "n_lags": int((2 * DL + 1) * (2 * DD + 1) * (2 * DF + 1)),
        "on_window_edge": edge,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Rendering — magenta (fixed) / green (moving) composites
# ═════════════════════════════════════════════════════════════════════════════
SURF_RUN_MM = 0.05   # a real surface starts a SUSTAINED tissue run, not one bright speckle


def _tissue_mask(vf: np.ndarray) -> np.ndarray:
    img = sitk.GetImageFromArray(np.transpose(vf, (2, 1, 0)))
    ot = sitk.OtsuThresholdImageFilter()
    ot.SetInsideValue(0)
    ot.SetOutsideValue(1)
    return _from_sitk(ot.Execute(img)).astype(bool)


def _clean_tissue(tis: np.ndarray, sp) -> np.ndarray:
    """Binary-open the Otsu mask along DEPTH only: drop any tissue run shorter than SURF_RUN_MM.

    Load-bearing, not cosmetic. Raw Otsu passes speckle in the dark background, and it comes in
    CONTIGUOUS PATCHES, not isolated voxels — measured on cs001_os_v2, `argmax(tis, depth)` put the
    surface at depth 0-6 for the extreme columns while the true surface sits at ~124, and the
    resulting min/argmin apex landed on pure noise (lat 71, depth 0) with a size-5 median filter
    powerless against it. A cornea is a thick continuous band (~0.5 mm); speckle is not. Requiring a
    sustained run is what separates them.

    Done with cumsum windows rather than ndi.binary_opening: O(n) and a couple of hundred ms on a
    33M-voxel volume, where a (1,k,1) structuring element is far slower.
    """
    k = int(max(2, round(SURF_RUN_MM / max(float(sp[1]), 1e-9))))
    d = tis.shape[1]
    if k >= d:
        return tis
    c = np.cumsum(tis, axis=1, dtype=np.int32)
    c = np.pad(c, ((0, 0), (1, 0), (0, 0)))
    run = (c[:, k:, :] - c[:, :-k, :]) == k        # [lat, d-k+1, frames]: a FULL run starts here
    # Dilate the run starts back over the k voxels each one covers = the opened mask.
    r = np.cumsum(run, axis=1, dtype=np.int32)
    r = np.pad(r, ((0, 0), (1, 0), (0, 0)))
    n = r.shape[1] - 1
    out = np.zeros_like(tis)
    hi = np.minimum(np.arange(d) + 1, n)
    lo = np.maximum(np.arange(d) + 1 - k, 0)
    out[:, :, :] = (r[:, hi, :] - r[:, lo, :]) > 0
    return out


def _surface(tis_clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First tissue depth per (lat, frame) = the anterior surface; inf where a column has none."""
    valid = tis_clean.any(axis=1)
    surf = np.argmax(tis_clean, axis=1).astype(np.float32)
    surf[~valid] = np.inf
    return surf, valid


def _bbox_axis(tis: np.ndarray, axis: int, sp_a: float) -> tuple[int, int]:
    """Robust extent along `axis`: keep where the projected tissue count exceeds 2% of its max, so
    stray speckle above Otsu in the background cannot inflate the crop. + MARGIN_MM."""
    other = tuple(a for a in range(3) if a != axis)
    proj = tis.sum(axis=other)
    if proj.max() <= 0:
        return 0, tis.shape[axis]
    idx = np.flatnonzero(proj > 0.02 * proj.max())
    mg = int(round(MARGIN_MM / max(sp_a, 1e-9)))
    return int(max(0, idx[0] - mg)), int(min(tis.shape[axis], idx[-1] + 1 + mg))


def _fixed_masks(vf: np.ndarray, sp) -> dict:
    """The FIXED volume's tissue mask + anterior surface + the UNCLIPPED-column mask.

    Split out of view_geometry because the surface residual needs the SAME `usable` mask to decide
    which frames carry a real, unclipped surface, and _clean_tissue costs ~1.3 s on a 33M-voxel
    volume — computed once per job and passed to both, never twice.

    CLIPPED columns are excluded. In this cohort the dome apex is often cut off by the top of the
    acquisition frame (a documented condition here — cases even carry a surface_crop_manual flag),
    and there the tissue genuinely starts at depth 0. Measured on cs001_os_v2: 342 columns over
    lat 56-230 / frames 25-51 have real, sustained signal (~1650 vs an Otsu threshold of 885)
    running to the frame edge. So the "shallowest surface" is truthfully a clipped one — and useless
    both to zoom on (the band is truncated) and to measure a residual against (there is no edge to
    detect, so the steepest-rise lands on noise).
    """
    sp = np.asarray(sp, float)
    tis = _clean_tissue(_tissue_mask(vf), sp)
    surf, _valid = _surface(tis)
    clip_guard = int(max(2, round(0.02 / max(float(sp[1]), 1e-9))))
    return {"tissue": tis, "surface": surf, "usable": np.isfinite(surf) & (surf > clip_guard)}


def view_geometry(vf: np.ndarray, sp, masks: dict | None = None) -> dict:
    """Slice indices + crops, computed ONCE from the FIXED volume so EVERY method is rendered at
    exactly the same place — otherwise the pictures are not comparable."""
    sp = np.asarray(sp, float)
    m = masks if masks is not None else _fixed_masks(vf, sp)
    tis, surf, usable = m["tissue"], m["surface"], m["usable"]
    l0, l1 = _bbox_axis(tis, 0, sp[0])
    d0, d1 = _bbox_axis(tis, 1, sp[1])
    f0, f1 = _bbox_axis(tis, 2, sp[2])

    finite = np.isfinite(surf)
    # The apex hunt runs on UNCLIPPED columns only — see _fixed_masks.
    if usable.any():
        sm = surf.copy()
        sm[~usable] = float(np.max(surf[usable]))
        sm = ndi.median_filter(sm, size=5)          # kill residual single-column specular spikes
        sm[~usable] = np.inf
        _, a_frame = np.unravel_index(np.argmin(sm), sm.shape)
        apex_frame = int(a_frame)
    else:
        sm = np.where(finite, surf, np.inf)
        usable = finite
        apex_frame = vf.shape[2] // 2

    fr = int((f0 + f1) // 2)                          # central B-scan (centre of the tissue bbox)
    fr = int(np.clip(fr, 0, vf.shape[2] - 1))

    # The lateral anchor for the sagittal + zoom: the shallowest UNCLIPPED position ON THE DISPLAYED
    # B-SCAN whose whole zoom window is also unclipped. Two things are load-bearing:
    #  - it is computed at frame `fr`, the frame actually being shown. A global argmin over all
    #    frames anchored the zoom to a different B-scan's apex, so the depth window did not match
    #    the local surface and the crop landed on band interior instead of the air/tissue edge.
    #  - the ERODE step forces the entire window to be unclipped. Merely excluding clipped columns
    #    parks the anchor hard against the clip boundary (measured: lat 55, right beside the clipped
    #    patch at lat 56-230), so the crop is still mostly truncated tissue with no visible surface.
    half = int(round(0.5 * ZOOM_LAT_MM / max(sp[0], 1e-9)))
    half = int(min(half, max(2, vf.shape[0] // 2 - 1)))
    col_ok = usable[:, fr]
    win_ok = ndi.binary_erosion(col_ok, structure=np.ones(2 * half + 1, bool))
    prof = np.where(np.isfinite(sm[:, fr]), sm[:, fr], np.inf)
    if win_ok.any():
        anchor = int(np.argmin(np.where(win_ok, prof, np.inf)))
    elif col_ok.any():
        anchor = int(np.argmin(np.where(col_ok, prof, np.inf)))
    else:
        anchor = (l0 + l1) // 2
    zl0 = int(np.clip(anchor - half, 0, max(0, vf.shape[0] - 2 * half)))
    zl1 = int(min(vf.shape[0], zl0 + 2 * half))

    # Depth window from the surface WITHIN the zoom's own lateral window — a low percentile, not
    # min(): min() over the window is a worst-case statistic one surviving speckle column hijacks.
    band = prof[zl0:zl1]
    band = band[np.isfinite(band)]
    d_apex = float(np.percentile(band, 1)) if band.size else float(d0)
    zd0 = int(max(0, round(d_apex - ZOOM_UP_MM / sp[1])))
    zd1 = int(min(vf.shape[1], round(d_apex + ZOOM_DOWN_MM / sp[1])))
    if zd1 - zd0 < 4:
        zd0, zd1 = d0, d1
    if zl1 - zl0 < 4:
        zl0, zl1 = l0, l1
    return {
        "lat": [l0, l1], "depth": [d0, d1], "frames": [f0, f1],
        "frame": fr, "apex_lat": anchor, "apex_frame": apex_frame,
        "zoom_depth": [zd0, zd1], "zoom_lat": [zl0, zl1],
    }


def _win(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0).astype(np.uint8)


def _aspect_resize(rgb: np.ndarray, row_mm: float, col_mm: float) -> np.ndarray:
    """Rescale to a TRUE mm aspect ratio. The grid is ~13x anisotropic; drawn 1 voxel = 1 pixel the
    picture LIES about the cornea's shape. Nearest-neighbour: no invented intensities."""
    h, w = rgb.shape[:2]
    ph, pw = h * row_mm, w * col_mm
    tw = RENDER_WIDTH
    th = max(1, int(round(tw * ph / max(pw, 1e-9))))
    if th > RENDER_MAX_H:
        tw = max(1, int(round(tw * RENDER_MAX_H / th)))
        th = RENDER_MAX_H
    r = np.linspace(0, h - 1, th).round().astype(int)
    c = np.linspace(0, w - 1, tw).round().astype(int)
    return rgb[r][:, c]


def _composite(f2d: np.ndarray, m2d: np.ndarray, lo: float, hi: float,
               row_mm: float, col_mm: float) -> np.ndarray:
    """R=fixed, G=moving, B=fixed -> magenta = fixed only, green = moving only, WHITE/GREY = both.
    BOTH windowed identically from the FIXED volume: a per-image window would let a brightness
    difference masquerade as a misalignment."""
    f, m = _win(f2d, lo, hi), _win(m2d, lo, hi)
    return _aspect_resize(np.dstack([f, m, f]), row_mm, col_mm)


def render_views(vf: np.ndarray, vm: np.ndarray, sp, geom: dict, lo: float, hi: float,
                 out_dir: Path, method: str) -> dict:
    """Three composites (central B-scan, sagittal through the apex, apex zoom) at the SAME indices
    for every method. vm is the moving ALREADY resampled onto the fixed grid."""
    sp = np.asarray(sp, float)
    l0, l1 = geom["lat"]
    d0, d1 = geom["depth"]
    f0, f1 = geom["frames"]
    fr, lat = geom["frame"], geom["apex_lat"]
    zd0, zd1 = geom["zoom_depth"]
    zl0, zl1 = geom["zoom_lat"]

    def bscan(v):      # (depth, lat): rows = depth, cols = lat
        return v[l0:l1, d0:d1, fr].T

    def sagittal(v):   # (depth, frames): rows = depth, cols = frames
        return v[lat, d0:d1, f0:f1]

    def zoom(v):
        return v[zl0:zl1, zd0:zd1, fr].T

    specs = (
        ("bscan", bscan, sp[1], sp[0]),
        ("sagittal", sagittal, sp[1], sp[2]),
        ("zoom", zoom, sp[1], sp[0]),
    )
    out = {}
    for name, fn, row_mm, col_mm in specs:
        rgb = _composite(fn(vf), fn(vm), lo, hi, float(row_mm), float(col_mm))
        p = out_dir / f"{method}_{name}.png"
        _write_png(p, rgb)
        out[name] = p.name
    return out


def _write_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB").save(str(path))


# ═════════════════════════════════════════════════════════════════════════════
# 3-D interactive replicate-agreement volumes — written as .nii.gz, rendered LIVE by niivue
# ═════════════════════════════════════════════════════════════════════════════
# The Debug tab renders these on the GPU (a dedicated niivue instance, SLICE_TYPE.RENDER). The backend
# only CROPS + RESAMPLES to a fine isotropic grid and writes NIfTI; niivue rotates/pans/zooms at full
# resolution. Two content modes, the 3-D twin of the 2-D overlay:
#   OVERLAP       fixed + aligned-moving as two volumes (the client tints them magenta / green).
#   DISAGREEMENT  |fixed - aligned_moving| gated to the tissue edge — one HOT scalar volume. A residual
#                 TILT reads as a glowing edge band; a genuine per-replicate SCAR difference reads as a
#                 localised blob — both invisible on a single slice, obvious when rotated. THIS is the
#                 novel, high-value view.
# Correctness that is NOT optional:
#   * ANISOTROPY ~13x. The tissue-bbox crop is resampled to an ISOTROPIC mm grid (a rendered volume with
#     anisotropic voxels is geometrically wrong), anti-aliased along the downsampled axes — which also
#     suppresses the independent per-scan speckle that would otherwise fill the disagreement with noise.
#   * ALL THREE VOLUMES SHARE ONE CROP + GRID + AFFINE, so niivue overlays them voxel-perfect with no
#     client-side alignment. fixed_iso is written once; only moving/disagree are per method.
#   * ONE window (lo,hi from the FIXED volume) reported so the client sets cal_min/cal_max identically
#     for every method — brightness cannot masquerade as disagreement.
#   * The disagreement is normalised by a percentile of the IDENTITY diff, SHARED across methods, so a
#     good aligner reads visibly COOLER than identity — that comparison is the whole point.


def _dbg_affine(iso: float) -> np.ndarray:
    """Voxel->world affine for an iso-mm debug volume, oriented like the app's MAIN viewer.

    The debug array is [lat, depth, frames]; the app's preview volumes map lat->X, depth->Z,
    frames->Y (axcodes L,I,P). This reproduces that mapping isotropically, so the cornea shows the
    right way up and the initial camera pose matches the main canvas. Origin 0 is correct for BOTH
    volumes: fixed_iso and moving_iso are cropped from the identical region and resampled to the same
    lattice, so one shared affine overlays them voxel-perfect."""
    return np.array([
        [-iso, 0.0,  0.0, 0.0],
        [0.0,  0.0, -iso, 0.0],   # frames (axis 2) -> world Y
        [0.0, -iso,  0.0, 0.0],   # depth  (axis 1) -> world Z
        [0.0,  0.0,  0.0, 1.0],
    ], dtype=float)


def _write_nifti(path: Path, arr: np.ndarray, iso: float, dtype) -> None:
    """Write a [lat, depth, frames] array as a .nii.gz on the shared iso grid + affine. uint16 arrays
    are clipped to the type range first (intensities are non-negative)."""
    import nibabel as nib
    a = np.asarray(arr)
    if np.dtype(dtype) == np.uint16:
        a = np.clip(a, 0, 65535)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(a.astype(dtype)), _dbg_affine(iso)), str(path))


def build_disagree(f_iso: np.ndarray, m_iso: np.ndarray, tissue_iso: np.ndarray,
                   cov_iso: np.ndarray, iso: float) -> tuple[np.ndarray, np.ndarray]:
    """|fixed - aligned_moving| gated to the tissue EDGE, INSIDE the shared FOV. Returns (diff, gate).

    The gate is load-bearing and mirrors the turntable's logic exactly (only the rotate/MIP is gone):
      * DILATE the fixed tissue mask: a surface offset is bright in one scan and dark (background) in
        the other, so a tissue-ONLY gate would zero out the very band that IS the disagreement.
      * AND with coverage: an alignment that moves tissue out of the FOV leaves MISSING data, not a
        disagreement — that eviction shows up as fixed-only in the OVERLAP view, and glowing it hot
        here would penalise a good aligner. cov marks where the aligned moving is inside its FOV."""
    it = max(1, int(round(DBG_DILATE_MM / iso)))
    gate = ndi.binary_dilation(tissue_iso, ndi.generate_binary_structure(3, 1), iterations=it) & cov_iso
    diff = np.abs(np.asarray(f_iso, np.float32) - np.asarray(m_iso, np.float32))
    diff[~gate] = 0.0
    return diff, gate


def _iso_crop(v: np.ndarray, spf, geom: dict, iso: float, *, order: int = 1,
              antialias: bool = True) -> np.ndarray:
    """Crop to the FIXED tissue bbox and resample to an `iso`-mm ISOTROPIC grid. Anti-aliased along any
    axis being downsampled (Nyquist sigma matched to the zoom) — that also suppresses the independent
    per-scan speckle. Crop indices come from `geom` and are identical for the fixed and the moving
    (same grid), so both land on the SAME iso lattice."""
    l0, l1 = geom["lat"]
    d0, d1 = geom["depth"]
    f0, f1 = geom["frames"]
    sub = np.asarray(v[l0:l1, d0:d1, f0:f1], np.float32)
    fac = [float(spf[i]) / iso for i in range(3)]
    if antialias:
        sig = [0.5 / f if f < 1.0 else 0.0 for f in fac]   # smooth only the DOWNSAMPLED axes
        if any(s > 0 for s in sig):
            sub = ndi.gaussian_filter(sub, sig)
    return ndi.zoom(sub, fac, order=order)


def build_method_volumes(f_iso: np.ndarray, tissue_iso: np.ndarray, m_res: np.ndarray,
                         spf, geom: dict, iso: float, out_dir: Path, job_id: str, method: str,
                         *, scale: float | None) -> dict:
    """Write one method's iso-cropped ALIGNED-MOVING volume + its DISAGREEMENT volume as .nii.gz, and
    return the URLs + the shared-scale disagreement summary.

    `f_iso` (the fixed volume already iso-cropped, shared across methods) and `tissue_iso` (the fixed
    cleaned tissue mask on the same iso grid) are computed once per job. `m_res` is the moving ALREADY
    resampled onto the fixed grid — the same array the 2-D renderer and the surface residual consume.

    `scale` (the identity disagreement percentile) is REUSED across methods when passed; when None it
    is computed here from THIS method's diff (identity is always first) so a good aligner reads
    visibly cooler at the same cal_max. Returns {"volumes3d": {...}, "scale": s}."""
    m_iso = _iso_crop(m_res, spf, geom, iso, order=1)
    # Coverage = where the aligned moving is inside its FOV (exactly 0 marks the out-of-FOV region an
    # alignment evicts). Nearest-neighbour from the UNSMOOTHED m_res so the boundary is crisp.
    cov = _iso_crop((np.asarray(m_res) > 0).astype(np.float32), spf, geom, iso,
                    order=0, antialias=False) > 0.5
    diff, gate = build_disagree(f_iso, m_iso, tissue_iso, cov, iso)

    if scale is None:
        pos = diff[gate]
        pos = pos[pos > 0]
        scale = float(np.percentile(pos, DBG_HOT_PCT)) if pos.size else 0.0
        if not (scale > 0):
            scale = 1e-6
    # Mean normalised disagreement inside the gate, at the SHARED scale: identity (unaligned) reads
    # hottest, a good aligner coolest — the numeric twin of the picture.
    gv = diff[gate]
    disagree_mean = float(np.clip(gv / scale, 0.0, 1.0).mean()) if gv.size else 0.0

    mv_name = f"{method}_moving_iso.nii.gz"
    dg_name = f"{method}_disagree_iso.nii.gz"
    _write_nifti(out_dir / mv_name, m_iso, iso, np.uint16)
    _write_nifti(out_dir / dg_name, diff, iso, np.float32)
    return {
        "volumes3d": {
            "moving": f"/api/debug/align/view/{job_id}/{mv_name}",
            "disagree": f"/api/debug/align/view/{job_id}/{dg_name}",
            "disagree_mean": disagree_mean,
            # RAW disagreement intensity scale (the SHARED identity percentile). The client sets niivue
            # cal_max = disagree_max on the hot volume so every method is windowed identically and a
            # good aligner reads visibly cooler; disagree_mean above is its normalised [0,1] summary.
            "disagree_max": float(scale),
        },
        "scale": float(scale),
    }


# ═════════════════════════════════════════════════════════════════════════════
# Surface residual — the GEOMETRIC truth, and the number that should be ranked on
# ═════════════════════════════════════════════════════════════════════════════
# WHY THIS EXISTS. 'primary' (NCC over a dilated mask) scored the 2-constant fix (0.8547) and the
# brute-force translation (0.8432) as a statistical TIE: +0.0127, under the ~0.02 run-to-run floor
# for rotation-bearing methods. THE PICTURES DISAGREE. The v2->v3 offset is not a constant shift, it
# is TILTED — 67 voxels at frame 5 falling to 26 at frame 95, a 41-voxel tilt riding on the mean
# shift. A pure translation STRUCTURALLY CANNOT remove a tilt, and you can see it fail: brute-force's
# sagittal carries a green fringe at the left flipping to magenta at the right, while the rigid fix
# is clean white across the whole span.
#
# The surface residual is what agrees with the eye: ~1.6 vox (~5 um) for the rigid methods vs ~9.6
# (~30 um) for brute force — a 6x geometric difference that NCC barely registers, because NCC over a
# dilated mask is dominated by bulk tissue overlap and a small tilt hardly moves it.
#
# It also matters for the pipeline: the point of aligning replicates is to propagate ONE scar label
# onto them. A 30 um boundary error lands straight in scar Dice — and scar Dice ~0.79 IS the
# reproducibility problem this project exists to solve. So NCC is the wrong yardstick for choosing an
# aligner; this is the right one. Rank on resid_um; read primary as an intensity proxy.
#
# PORTED from teaser_bench/viz/wedge2.py (steepest-rise on a smoothed sagittal slab, search
# CONSTRAINED to the fixed surface +-90 vox). Two properties of that port are load-bearing:
#   - the CONSTRAINT is not optional. Unconstrained, the detector locks onto the out-of-FOV zero
#     step at the top of the resampled moving volume (this is exactly what ruined wedge.py).
#   - a FRAME-VALIDITY GATE that wedge2.py does NOT have. wedge2 hardcoded lat 128 and got clean
#     numbers because that lateral happens to be unclipped on this one pair; at the apex lateral
#     this tool actually renders (136), frame 98's fixed surface is CLIPPED (depth 4) and the
#     "residual" there is garbage: -43 vox, inflating |resid|max from 7 to 43. Gating on the same
#     `usable` mask view_geometry uses drops it, and lat 128 vs 136 then agree to 0.1 vox. A
#     measurement that depends on a hand-picked lateral is not a measurement.
#
# NOT oct_preprocess.detect_surface_all (the DP detector): it is a whole-volume multiprocess DP with
# dome/lateral regularisation tuned for RAW acquisition geometry. On a resampled moving volume with
# hard FOV edges it has no equivalent of the +-90 constraint, its dome smoothing would fight the very
# tilt being measured, and it costs seconds per volume x4 methods. This costs ~40 ms per method.
RESID_LAT_HALF = 8        # average +-8 laterals: kills speckle, still local to the rendered sagittal
RESID_SEARCH_VOX = 90     # search half-window around the FIXED surface, depth voxels
RESID_EDGE_FRAMES = 2     # the outermost frames are never trustworthy
RESID_MIN_FRAMES = 8      # below this a tilt fit is meaningless
RESID_TILT_LO, RESID_TILT_HI = 0.05, 0.95    # tilt is reported across this fraction of the frames


def _sag_slab_grad(vol: np.ndarray, lat: int) -> np.ndarray:
    """Depth-gradient of a smoothed sagittal slab centred on `lat` -> (depth, frames). The anterior
    surface is its steepest rise."""
    l0 = max(0, lat - RESID_LAT_HALF)
    l1 = min(vol.shape[0], lat + RESID_LAT_HALF + 1)
    s = ndi.gaussian_filter(vol[l0:l1, :, :].mean(axis=0).astype(np.float32), sigma=[4.0, 1.0])
    return np.gradient(s, axis=0)


def surface_reference(vf: np.ndarray, sp, geom: dict, masks: dict | None = None) -> dict:
    """The FIXED side of the residual — computed ONCE per job, shared by every method.

    Anchored at geom["apex_lat"], the SAME lateral the sagittal panel is rendered at, so the number
    explains the picture the expert is looking at (the panel is one column; this averages a 17-column
    slab around it to suppress speckle)."""
    sp = np.asarray(sp, float)
    m = masks if masks is not None else _fixed_masks(vf, sp)
    lat = int(geom["apex_lat"])
    l0 = max(0, lat - RESID_LAT_HALF)
    l1 = min(vf.shape[0], lat + RESID_LAT_HALF + 1)
    # A frame counts only if EVERY column of the slab has a real, unclipped surface — the residual
    # reads the slab MEAN, so one clipped column contaminates it.
    ok = m["usable"][l0:l1, :].all(axis=0)
    ok[:RESID_EDGE_FRAMES] = False
    if RESID_EDGE_FRAMES:
        ok[ok.size - RESID_EDGE_FRAMES:] = False
    return {"lat": lat, "surface": _sag_slab_grad(vf, lat).argmax(axis=0).astype(np.float64),
            "frame_ok": ok, "n_frames": int(ok.sum()), "vox_um": float(sp[1]) * 1000.0}


def surface_residual(ref: dict, vm: np.ndarray) -> dict:
    """Per-frame anterior-surface residual (fixed - moving) at the reference lateral, in DEPTH
    VOXELS. `vm` is the moving ALREADY resampled onto the fixed grid.

    SIGN: positive = the moving surface still sits SHALLOWER than the fixed = a GREEN fringe above
    the white surface in the sagittal panel; negative = overshoot = MAGENTA above.

    resid_saturated marks a residual CLAMPED by the search window (see below): read the value as a
    lower bound. It flags the at-or-just-beyond-the-bound regime that real pairs land in; it is not a
    certificate that an unflagged number is sound.
    """
    s2, ok = ref["surface"], ref["frame_ok"]
    out = {"resid_vox": None, "resid_um": None, "tilt_vox": None, "resid_max_vox": None,
           "resid_frames": int(ok.sum()), "resid_lat": int(ref["lat"]), "resid_saturated": False}
    if int(ok.sum()) < RESID_MIN_FRAMES:
        return out
    g3 = _sag_slab_grad(vm, ref["lat"])
    nd = vm.shape[1]
    d = np.full(s2.size, np.nan)
    # SATURATION. The constraint that saves the detector from the out-of-FOV step can also CLAMP a
    # genuinely huge offset: an argmax pinned to either END of the window means the true surface is
    # at or beyond the bound, so that frame's residual is a LOWER BOUND, not a measurement. Observed
    # on cs005_od v1 vs v9 (different lateral spacing, different FOV): every method pegged at exactly
    # 90.0. Averaging clamped values silently would report a plausible-looking number that is wrong.
    sat = np.zeros(s2.size, bool)
    for f in np.flatnonzero(ok):
        c = int(s2[f])
        a, b = max(0, c - RESID_SEARCH_VOX), min(nd, c + RESID_SEARCH_VOX + 1)
        i = int(g3[a:b, f].argmax())
        d[f] = s2[f] - (a + i)
        sat[f] = i == 0 or i == b - a - 1
    fr = np.flatnonzero(ok & np.isfinite(d))
    if fr.size < RESID_MIN_FRAMES:
        return out
    out["resid_saturated"] = bool(sat[fr].any())
    ad = np.abs(d[fr])
    # TILT by least-squares slope over EVERY valid frame, evaluated at the 5%/95% frame positions —
    # not wedge2's d[5]-d[95]. Two hand-picked frames are undefined the moment either is gated out
    # (frame 95 IS gated out on this very pair) and carry each frame's full detector noise; the fit
    # uses the whole span and is stable across laterals (38.7 at lat 128 vs 39.3 at lat 136).
    slope = float(np.polyfit(fr.astype(np.float64), d[fr], 1)[0])
    span = (RESID_TILT_HI - RESID_TILT_LO) * (s2.size - 1)
    out.update({
        "resid_vox": float(ad.mean()),
        "resid_um": float(ad.mean() * ref["vox_um"]),
        "tilt_vox": float(-slope * span),
        "resid_max_vox": float(ad.max()),
        "resid_frames": int(fr.size),
    })
    return out


def window_from_fixed(vf: np.ndarray) -> tuple[float, float]:
    """The ONE intensity window, from the FIXED volume's nonzero voxels, used for every method."""
    nz = vf[vf > 0]
    if nz.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(nz, 1))
    hi = float(np.percentile(nz, 99.5))
    return lo, (hi if hi > lo else lo + 1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Job runner
# ═════════════════════════════════════════════════════════════════════════════
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
# SimpleITK registration + a 33M-voxel FFT are CPU-hungry; serialise runs so two impatient clicks
# queue instead of thrashing every core. Jobs report status "running" while they wait.
_RUN_LOCK = threading.Lock()
# Orphan sweep: once per process, at the first job.
_SWEPT = False
_SWEEP_LOCK = threading.Lock()


def _set(job_id: str, **kw) -> None:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if j is not None:
            j.update(kw)


def _append_result(job_id: str, r: dict) -> None:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if j is not None:
            j["results"].append(r)


def job_view(job_id: str) -> dict | None:
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if j is None:
            return None
        return {
            "status": j["status"], "progress": round(float(j["progress"]), 3),
            "error": j["error"], "results": [dict(r) for r in j["results"]],
            "fixed_case": j["fixed_case"], "moving_case": j["moving_case"],
            "geometry": j.get("geometry"), "note": j.get("note"),
            # Shared interactive-3-D volume (fixed) + its grid + the shared disagreement cal_max,
            # when render_3d was requested.
            "fixed3d": j.get("fixed3d"), "iso_mm": j.get("iso_mm"),
            "disagree_max": j.get("disagree_max"),
        }


def job_dir(job_id: str) -> Path:
    return _RENDER_ROOT / job_id


def _prune_jobs() -> None:
    with _JOBS_LOCK:
        ids = [k for k, v in sorted(_JOBS.items(), key=lambda kv: kv[1]["started"]) if not v["running"]]
    for jid in ids[:max(0, len(ids) - _MAX_KEPT_JOBS)]:
        with _JOBS_LOCK:
            _JOBS.pop(jid, None)
        shutil.rmtree(job_dir(jid), ignore_errors=True)


def sweep_render_root(ttl_s: float = _JOB_TTL_S) -> int:
    """Delete job dirs left behind by PREVIOUS sidecar processes. Returns the number removed.

    _prune_jobs only knows this process's in-memory _JOBS, so _MAX_KEPT_JOBS never applies across a
    restart: every dir from a prior run is orphaned the moment the sidecar exits, and at ~2 MB/job
    /tmp grows without bound (observed: 17 dirs / 34 MB). Age-based, because a dir's job_id is the
    only thing tying it to a process and that process is gone.

    Two guards: never touch a dir belonging to a LIVE job in this process, and never touch one
    younger than the TTL — a second sidecar could be mid-render in it right now.
    """
    if not _RENDER_ROOT.exists():
        return 0
    with _JOBS_LOCK:
        live = set(_JOBS)
    now, n = time.time(), 0
    try:
        children = list(_RENDER_ROOT.iterdir())
    except OSError:
        return 0
    for child in children:
        if child.name in live or not child.is_dir():
            continue
        try:
            if now - child.stat().st_mtime <= ttl_s:
                continue
        except OSError:
            continue
        shutil.rmtree(child, ignore_errors=True)
        n += 1
    return n


def _sweep_once() -> None:
    """Once per process, before the first job — not at import: importing debug_align must not do
    filesystem work (api_server imports it at startup, and the tests import it constantly)."""
    global _SWEPT
    with _SWEEP_LOCK:
        if _SWEPT:
            return
        _SWEPT = True
    try:
        sweep_render_root()
    except Exception:  # noqa: BLE001 — housekeeping must never fail a job
        pass


def start_compare(fixed_case: str, moving_case: str, methods: list[str] | None,
                  render_3d: bool = False) -> str:
    _sweep_once()
    fixed_case, moving_case = orch.safe_case_id(fixed_case), orch.safe_case_id(moving_case)
    for c in (fixed_case, moving_case):
        if not volume_path(c).exists():
            raise FileNotFoundError(f"No preview volume for {c}")
    if fixed_case == moving_case:
        raise ValueError("Pick two DIFFERENT replicate scans.")
    ms = [m for m in (methods or list(METHODS)) if m in METHODS]
    if not ms:
        raise ValueError(f"No known methods requested. Known: {', '.join(METHODS)}")
    if "identity" not in ms:
        ms = ["identity"] + ms     # the reference is never optional
    job_id = uuid.uuid4().hex[:16]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "running", "progress": 0.0, "error": None, "results": [],
            "fixed_case": fixed_case, "moving_case": moving_case, "methods": ms,
            "running": True, "started": time.time(), "geometry": None, "note": None,
            "render_3d": bool(render_3d), "fixed3d": None, "iso_mm": None,
            "disagree_max": None,
        }
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, args=(job_id, fixed_case, moving_case, ms, bool(render_3d)),
                     daemon=True).start()
    return job_id


def _worker(job_id: str, fixed_case: str, moving_case: str, ms: list[str],
            render_3d: bool = False) -> None:
    try:
        with _RUN_LOCK:
            _run(job_id, fixed_case, moving_case, ms, render_3d)
    except Exception as exc:  # noqa: BLE001 — a job must never take the sidecar down
        _set(job_id, status="error", error=f"{type(exc).__name__}: {exc}", progress=1.0, running=False)
    finally:
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j is not None:
                j["running"] = False
                if j["status"] == "running":
                    j["status"] = "done"
                    j["progress"] = 1.0
        _prune_jobs()


def _run(job_id: str, fixed_case: str, moving_case: str, ms: list[str],
         render_3d: bool = False) -> None:
    out_dir = job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    vf, spf = load_volume(fixed_case)
    vm, spm = load_volume(moving_case)
    fi, mi = to_sitk(vf, spf), to_sitk(vm, spm)
    fi_iso, mi_iso = reg._iso(fi), reg._iso(mi)
    _set(job_id, progress=0.08)

    masks = _fixed_masks(vf, spf)          # ~1.3 s, once — shared by the geometry and the residual
    geom = view_geometry(vf, spf, masks)
    ref = surface_reference(vf, spf, geom, masks)
    # The 3-D disagreement gate needs the FIXED tissue mask; keep it (only when 3-D is on) rather
    # than recomputing the ~1.3 s _clean_tissue per method.
    tissue = masks["tissue"] if render_3d else None
    del masks
    lo, hi = window_from_fixed(vf)

    note = None
    if vf.shape != vm.shape or not np.allclose(spf, spm, rtol=1e-3, atol=1e-6):
        # Real and expected: shapes and even LATERAL SPACING differ within an eye
        # (cs005_od v1 0.0078 vs v9 0.01111 mm). Physical-space registration handles it; only the
        # brute force needs a common lattice, and it resamples for itself.
        note = (f"Geometry differs: fixed {vf.shape} @ {np.round(spf, 5).tolist()} mm vs "
                f"moving {vm.shape} @ {np.round(spm, 5).tolist()} mm. Alignment is computed in "
                f"physical space, so this is handled — but the two scans do not cover the same FOV.")
    _set(job_id, geometry={**geom, "fixed_shape": list(vf.shape), "moving_shape": list(vm.shape),
                           "fixed_spacing_mm": [float(x) for x in spf],
                           "moving_spacing_mm": [float(x) for x in spm],
                           "window": [lo, hi],
                           "resid_lat": int(ref["lat"]), "resid_frames": int(ref["n_frames"]),
                           "resid_vox_um": round(float(ref["vox_um"]), 4)},
         note=note, progress=0.12)

    # 3-D (opt-in): crop the FIXED volume + its tissue mask to the shared FINE iso grid ONCE, write
    # fixed_iso.nii.gz (referenced by every method), and remember them for each method's diff. All
    # three per-method volumes land on this same lattice, so niivue overlays them voxel-perfect.
    iso3d = DBG_ISO_MM
    f_iso3d = t_iso3d = None
    fixed3d_url = None
    if render_3d and tissue is not None:
        try:
            f_iso3d = _iso_crop(vf, spf, geom, iso3d, order=1)
            t_iso3d = _iso_crop(np.asarray(tissue, np.float32), spf, geom, iso3d,
                                order=0, antialias=False) > 0.5
            _write_nifti(out_dir / "fixed_iso.nii.gz", f_iso3d, iso3d, np.uint16)
            fixed3d_url = f"/api/debug/align/view/{job_id}/fixed_iso.nii.gz"
        except Exception as exc:  # noqa: BLE001 — a 3-D failure must not lose the 2-D job
            f_iso3d = t_iso3d = None
            _set(job_id, note="3-D volume build failed (2-D results are unaffected): "
                              f"{type(exc).__name__}: {exc}")
    _set(job_id, fixed3d=fixed3d_url, iso_mm=(iso3d if fixed3d_url else None))

    # Identity FIRST: every other method's delta is measured against it.
    id_primary: float | None = None
    # The 3-D disagreement normaliser: computed from IDENTITY's diff (the first method) and reused for
    # every method, so a good aligner's disagreement volume is directly comparable and reads cooler.
    tt_scale: float | None = None
    step = 0.86 / max(len(ms), 1)

    for n, method in enumerate(ms):
        t0 = time.time()
        base = 0.12 + n * step
        _set(job_id, progress=base)
        raised, err = False, None
        R, t = np.eye(3), np.zeros(3)
        extra: dict = {}
        try:
            if method == "identity":
                pass
            elif method == "asis":
                # UNMODIFIED production settings. This is the one that raises ~46% of the time; the
                # raise IS the finding, so catch it and fall back to identity's transform rather
                # than failing the method — exactly what align_transform's guard does in production.
                try:
                    tx = reg._rigid_intensity(fi_iso, mi_iso, fixed_mask=None)
                    R, t = extract_rigid(tx)
                    extra["centre_t_mm"] = centre_translation(tx)
                except Exception as exc:  # noqa: BLE001
                    raised, err = True, f"{type(exc).__name__}: {exc}"
                    R, t = np.eye(3), np.zeros(3)
            elif method == "fixed":
                tx = reg._rigid_intensity(fi_iso, mi_iso, fixed_mask=None,
                                          learning_rate=FIX_LR, smoothing_sigmas=FIX_SIGMAS)
                R, t = extract_rigid(tx)
                extra["centre_t_mm"] = centre_translation(tx)
            elif method == "bruteforce":
                if vf.shape == vm.shape and np.allclose(spf, spm, rtol=1e-3, atol=1e-6):
                    bf_m = vm
                else:
                    bf_m = _from_sitk(sitk.Resample(mi, fi, sitk.Transform(), sitk.sitkLinear,
                                                    0.0, sitk.sitkFloat32))
                t, info = bruteforce_translation(vf, bf_m, spf)
                R = np.eye(3)
                extra.update(info)

            s = score(fi, mi, R=R, t=t)
            if method == "identity":
                id_primary = float(s["primary"])
            m_res = _from_sitk(sitk.Resample(mi, fi, _tx(R, t), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
            views = render_views(vf, m_res, spf, geom, lo, hi, out_dir, method)
            extra.update(surface_residual(ref, m_res))   # ~40 ms, reuses the render's resampling

            # 3-D interactive volumes (opt-in): write this method's aligned-moving + disagreement
            # .nii.gz on the SHARED fixed iso grid. In its OWN try/except so a 3-D failure degrades to
            # null volumes3d instead of losing this method's 2-D result. `f_iso3d`/`t_iso3d` are the
            # fixed volume + tissue mask already iso-cropped once for the whole job.
            vol3d = None
            if render_3d and f_iso3d is not None and t_iso3d is not None:
                try:
                    mv = build_method_volumes(f_iso3d, t_iso3d, m_res, spf, geom, iso3d,
                                              out_dir, job_id, method, scale=tt_scale)
                    if tt_scale is None:
                        tt_scale = float(mv["scale"])
                        # Job-level too (shared, identity-derived): the client sets cal_max from it.
                        _set(job_id, disagree_max=tt_scale)
                    vol3d = mv["volumes3d"]
                    extra["disagree_mean"] = float(vol3d["disagree_mean"])
                except Exception as exc:  # noqa: BLE001 — a 3-D failure must not lose the 2-D result
                    vol3d = None
                    extra["vol3d_error"] = f"{type(exc).__name__}: {exc}"
            del m_res

            _append_result(job_id, {
                "method": method, "label": METHODS[method], "ok": True,
                "raised": raised, "error": err,
                "rot_deg": ang(R), "t_mm": [float(x) for x in t],
                "primary": float(s["primary"]),
                "identity_primary": id_primary,
                "delta": (float(s["primary"]) - id_primary) if id_primary is not None else None,
                "frac_out": float(s["frac_out"]), "reject": bool(s["reject"]),
                "ncc_in": float(s["ncc_in"]), "nmi": float(s["nmi"]),
                "runtime_s": round(time.time() - t0, 2),
                "views": {k: f"/api/debug/align/view/{job_id}/{v}" for k, v in views.items()},
                "volumes3d": vol3d,
                **extra,
            })
        except Exception as exc:  # noqa: BLE001 — one bad method must not kill the job
            _append_result(job_id, {
                "method": method, "label": METHODS[method], "ok": False, "raised": raised,
                "error": f"{type(exc).__name__}: {exc}", "rot_deg": None, "t_mm": None,
                "primary": None, "identity_primary": id_primary, "delta": None,
                "frac_out": None, "runtime_s": round(time.time() - t0, 2), "views": {},
                "volumes3d": None,
                # Explicit nulls: the UI renders a residual column for every row.
                "resid_vox": None, "resid_um": None, "tilt_vox": None, "resid_max_vox": None,
                "resid_saturated": False,
            })
        _set(job_id, progress=min(0.98, base + step))

    # Backfill identity_primary/delta: methods that ran before identity finished (they cannot, since
    # identity is forced first — but a caller-supplied order could still surprise us).
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if j and id_primary is not None:
            for r in j["results"]:
                if r.get("identity_primary") is None:
                    r["identity_primary"] = id_primary
                    if r.get("primary") is not None:
                        r["delta"] = float(r["primary"]) - id_primary
    _set(job_id, status="done", progress=1.0)


# ═════════════════════════════════════════════════════════════════════════════
# Consensus — ALL replicates of one eye, aligned to a reference, in ONE 3-D volume
# ═════════════════════════════════════════════════════════════════════════════
# This is the N-way generalisation of the pairwise magenta/green/white overlay: instead of a fixed
# vs a single moving, EVERY replicate of the eye is aligned to the FIRST (reference) by ONE chosen
# method and composited into a single RGBA volume via a MIN/EXCESS decomposition —
#   shared   = min_i(I_i)                 -> the AGREEMENT colour (white): where all replicates coincide
#   excess_i = max(0, I_i - shared)       -> replicate i's OWN hue: where it sticks out (misalignment
#                                            or a unique feature)
#   RGB      = shared*WHITE + Σ excess_i*COLOR_i   (clipped to [0,1])
#   alpha    = max_i(I_i)                 -> background (all ~0) is transparent
# Where every replicate agrees the excess is 0 and the voxel is pure white (the "consensus glow");
# where one diverges it fringes in THAT replicate's hue. niivue 0.68.2 renders the RGBA NIfTI
# (DT_RGBA32 = 2304) directly as a 3-D volume, so the colours are baked here and the client just
# loads the volume — there is NO client windowing, so the pairwise disagree_mean-as-cal_max mis-scale
# class cannot recur on this path. Every replicate lands on the SAME iso lattice (reference tissue
# bbox, DBG_ISO_MM isotropic), so the composite is voxel-perfect with no client-side alignment.

# Qualitative, colour-blind-aware replicate palette (Okabe-Ito extended). Saturated + visually
# distinct, and every hue is kept clearly distinct from the WHITE agreement colour. Scales to the
# 9-replicate eye (cs005_od); cycles if an eye ever has more than len(PALETTE) replicates.
CONSENSUS_PALETTE: list[tuple[int, int, int]] = [
    (230, 159,   0),   # orange
    ( 86, 180, 233),   # sky blue
    (  0, 158, 115),   # bluish green
    (213,  94,   0),   # vermillion
    (204, 121, 167),   # reddish purple
    (  0, 114, 178),   # blue
    (240, 228,  66),   # yellow
    (170,  68, 255),   # violet
    (255,  99, 146),   # rose
    ( 90, 200,  90),   # green
]
AGREE_COLOR: tuple[int, int, int] = (255, 255, 255)   # unmistakable, distinct from every hue
# Cap so a pathological eye cannot spawn a runaway job; the cohort maxes at 9 replicates.
MAX_CONSENSUS_REPS = len(CONSENSUS_PALETTE)

# ── scar-consensus space ──────────────────────────────────────────────────────
CONSENSUS_SPACES = ("intensity", "scar")
# Faint translucent grey reference-cornea shell baked into the SCAR RGBA, so the sparse scar overlay has
# anatomical context. Kept subtle (low alpha, neutral grey) and painted ONLY where NO replicate has scar,
# so it never dims the scar overlay nor reintroduces the background/bulk-tissue problem the scar space
# exists to avoid. It is the CORNEA — not raw intensity — so background stays fully transparent.
SCAR_CTX_ENABLED = True
SCAR_CTX_GREY = 0.42
SCAR_CTX_ALPHA = 0.12


def eye_cases(eye: str) -> list[str]:
    """The ordered replicate case list for one eye — the SAME enumeration groups() offers, so both
    naming schemes (A: _v1/_v2/_v3, B: _v1/_v1_2/_v1_3) are handled identically. [] if unknown."""
    for g in groups():
        if g["eye"] == eye:
            return list(g["cases"])
    return []


def _rep_label(eye: str, case: str) -> str:
    """`case_cs001_os_v2` -> `v2`; `case_cs030_od_v1_2` -> `v1_2` (matches the client's repLabel)."""
    prefix = f"case_{eye}_"
    return case[len(prefix):] if case.startswith(prefix) else case


def consensus_colors(eye: str, cases: list[str]) -> list[dict]:
    """Per-replicate legend: {case, label, color=[R,G,B], is_ref}. First replicate = reference."""
    out = []
    for i, c in enumerate(cases):
        col = CONSENSUS_PALETTE[i % len(CONSENSUS_PALETTE)]
        out.append({"case": c, "label": _rep_label(eye, c),
                    "color": [int(col[0]), int(col[1]), int(col[2])], "is_ref": (i == 0)})
    return out


def consensus_decompose(I_list: list[np.ndarray], colors01: list) -> tuple[np.ndarray, np.ndarray]:
    """The MIN/EXCESS decomposition. `I_list` = N intensity arrays windowed to [0,1] on the shared
    iso lattice; `colors01` = N per-replicate hues in [0,1]. Returns (rgb[...,3] float[0,1], alpha).

    Running min/max (no giant N-deep stack) so the 9-replicate eye stays well under memory budget."""
    if not I_list:
        raise ValueError("consensus_decompose needs at least one replicate")
    shared = np.asarray(I_list[0], np.float32).copy()
    alpha = shared.copy()
    for I in I_list[1:]:
        Ia = np.asarray(I, np.float32)
        np.minimum(shared, Ia, out=shared)
        np.maximum(alpha, Ia, out=alpha)
    rgb = np.repeat(shared[..., None], 3, axis=-1).astype(np.float32)   # shared * WHITE
    for I, col in zip(I_list, colors01):
        excess = np.asarray(I, np.float32) - shared
        np.maximum(excess, 0.0, out=excess)
        rgb += excess[..., None] * np.asarray(col, np.float32)
    np.clip(rgb, 0.0, 1.0, out=rgb)
    return rgb, alpha


def _write_rgba_nifti(path: Path, rgb: np.ndarray, alpha: np.ndarray, iso: float) -> None:
    """Write an RGBA volume as DT_RGBA32 (.nii.gz) on the shared iso grid + debug affine. rgb is
    [lat,depth,frames,3] float[0,1]; alpha is [lat,depth,frames] float[0,1]. niivue renders the RGB
    per-voxel directly and uses A as opacity (colormap/cal_min/cal_max are ignored on this path)."""
    import nibabel as nib
    dt = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1"), ("A", "u1")])
    out = np.zeros(rgb.shape[:3], dtype=dt)
    r8 = np.clip(np.asarray(rgb, np.float32) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    for i, c in enumerate("RGB"):
        out[c] = r8[..., i]
    out["A"] = np.clip(np.asarray(alpha, np.float32) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(out), _dbg_affine(iso)), str(path))


def _consensus_transform(method: str, fi: sitk.Image, mi: sitk.Image, fi_iso: sitk.Image,
                         vf: np.ndarray, vm: np.ndarray, spf, spm) -> tuple[np.ndarray, np.ndarray]:
    """(R, t) aligning ONE `moving` (mi/vm) to the reference (fi/vf) by `method`. This is the exact
    per-method dispatch of _run's method loop, reduced to a single pair — no aligner is reimplemented
    (registration.py is untouched; the shipped `asis` raise-and-fall-back-to-identity behaviour and
    the `fixed` 2-constant kwargs are reused verbatim)."""
    if method == "identity":
        return np.eye(3), np.zeros(3)
    if method == "asis":
        try:
            tx = reg._rigid_intensity(fi_iso, reg._iso(mi), fixed_mask=None)
            return extract_rigid(tx)
        except Exception:  # noqa: BLE001 — the ~46% raise IS the finding; fall back to identity
            return np.eye(3), np.zeros(3)
    if method == "fixed":
        tx = reg._rigid_intensity(fi_iso, reg._iso(mi), fixed_mask=None,
                                  learning_rate=FIX_LR, smoothing_sigmas=FIX_SIGMAS)
        return extract_rigid(tx)
    if method == "bruteforce":
        if vf.shape == vm.shape and np.allclose(spf, spm, rtol=1e-3, atol=1e-6):
            bf_m = vm
        else:
            bf_m = _from_sitk(sitk.Resample(mi, fi, sitk.Transform(), sitk.sitkLinear,
                                            0.0, sitk.sitkFloat32))
        t, _info = bruteforce_translation(vf, bf_m, spf)
        return np.eye(3), t
    raise ValueError(f"Unknown method '{method}'")


def _write_consensus_slices(rgb: np.ndarray, geom: dict, spf, iso: float,
                            out_dir: Path, job_id: str) -> dict:
    """The SAME min/excess composite as the 3-D volume, sliced for the 2-D B-scan + sagittal views so
    2-D and 3-D match exactly. rgb is [lat,depth,frames,3] float[0,1] on the iso lattice; the geom
    slice indices (native grid) are mapped into the iso-cropped lattice."""
    spf = np.asarray(spf, float)
    L, _D, Fr = rgb.shape[:3]
    frame_iso = int(np.clip(round((geom["frame"] - geom["frames"][0]) * spf[2] / iso), 0, Fr - 1))
    lat_iso = int(np.clip(round((geom["apex_lat"] - geom["lat"][0]) * spf[0] / iso), 0, L - 1))
    bscan = np.transpose(rgb[:, :, frame_iso, :], (1, 0, 2))   # (depth, lat, 3)
    sag = rgb[lat_iso, :, :, :]                                # (depth, frames, 3)
    out = {}
    for name, sl in (("bscan", bscan), ("sagittal", sag)):
        u8 = np.clip(np.asarray(sl, np.float32) * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
        u8 = _aspect_resize(u8, iso, iso)   # isotropic -> aspect is 1:1, just scale to RENDER_WIDTH
        _write_png(out_dir / f"consensus_{name}.png", u8)
        out[name] = f"/api/debug/align/view/{job_id}/consensus_{name}.png"
    return out


def consensus_job_view(job_id: str) -> dict | None:
    """Live status/result of a consensus job (None for a missing id OR a non-consensus job id)."""
    with _JOBS_LOCK:
        j = _JOBS.get(job_id)
        if j is None or j.get("kind") != "consensus":
            return None
        return {
            "status": j["status"], "progress": round(float(j["progress"]), 3), "error": j["error"],
            "eye": j["eye"], "method": j["method"], "space": j.get("space", "intensity"),
            "reference": j["reference"],
            "replicates": [dict(r) for r in j["replicates"]],
            "agree_color": list(j["agree_color"]),
            "volume": j.get("volume"), "iso_mm": j.get("iso_mm"),
            "slices": dict(j.get("slices") or {}),
            "geometry": j.get("geometry"), "note": j.get("note"),
            "skipped": list(j.get("skipped") or []),
        }


def start_consensus(eye: str, method: str = "fixed", space: str = "intensity") -> str:
    """Start a consensus render for one eye by one method + one SPACE. Returns a job_id; poll
    consensus_job_view. Aligns every replicate to the first (reference) and composites the min/excess
    RGBA volume — over windowed INTENSITY (space="intensity", the whole cornea) or over binary SCAR
    masks (space="scar", a white agreement core + a coloured disagreement halo at the unstable
    boundary; SAM2 + hysteresis per replicate, cached)."""
    _sweep_once()
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Known: {', '.join(METHODS)}")
    if space not in CONSENSUS_SPACES:
        raise ValueError(f"Unknown space '{space}'. Known: {', '.join(CONSENSUS_SPACES)}")
    cases = eye_cases(eye)
    if len(cases) < 2:
        raise ValueError(f"Eye '{eye}' has fewer than 2 replicate scans to compare.")
    note = None
    if len(cases) > MAX_CONSENSUS_REPS:
        note = f"Eye has {len(cases)} replicates; consensus shows the first {MAX_CONSENSUS_REPS}."
        cases = cases[:MAX_CONSENSUS_REPS]
    reps = consensus_colors(eye, cases)
    job_id = uuid.uuid4().hex[:16]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "kind": "consensus",
            "status": "running", "progress": 0.0, "error": None,
            "eye": eye, "method": method, "space": space, "reference": cases[0], "cases": cases,
            "replicates": reps, "agree_color": list(AGREE_COLOR), "skipped": [],
            "volume": None, "iso_mm": None, "slices": {}, "geometry": None,
            "running": True, "started": time.time(), "note": note,
            # Benign pairwise-view keys so a job_view() call with this id cannot KeyError.
            "results": [], "fixed_case": None, "moving_case": None, "fixed3d": None,
            "disagree_max": None,
        }
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_consensus_worker, args=(job_id, cases, method, space), daemon=True).start()
    return job_id


def _consensus_worker(job_id: str, cases: list[str], method: str, space: str = "intensity") -> None:
    try:
        with _RUN_LOCK:                      # shared with the pairwise job — a second click queues
            _consensus_run(job_id, cases, method, space)
    except Exception as exc:  # noqa: BLE001 — a job must never take the sidecar down
        _set(job_id, status="error", error=f"{type(exc).__name__}: {exc}", progress=1.0, running=False)
    finally:
        with _JOBS_LOCK:
            j = _JOBS.get(job_id)
            if j is not None:
                j["running"] = False
                if j["status"] == "running":
                    j["status"] = "done"
                    j["progress"] = 1.0
        _prune_jobs()


def _bake_cornea_context(rgb: np.ndarray, alpha: np.ndarray, cornea_iso: np.ndarray) -> None:
    """Overlay a very translucent grey reference-cornea shell into the SCAR RGBA IN PLACE, ONLY where no
    replicate has scar (alpha==0) — so the scar overlay is never dimmed and the background stays fully
    transparent. Gives the sparse scar overlay anatomical context without reintroducing the background/
    bulk-tissue problem the scar space exists to avoid."""
    ctx = np.asarray(cornea_iso, bool) & (alpha <= 1e-6)
    if not ctx.any():
        return
    rgb[ctx] = SCAR_CTX_GREY
    alpha[ctx] = SCAR_CTX_ALPHA


def _consensus_scar_masks(job_id: str, cases: list[str], method: str, fi: sitk.Image,
                          fi_iso: sitk.Image, vf: np.ndarray, spf, geom: dict,
                          iso: float) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray | None, list[str]]:
    """Build every replicate's binary SCAR mask on the shared reference iso grid, for the min/excess
    decomposition. Returns (I_list, colors01, cornea_ctx_iso, skipped).

    Per replicate: fetch its cached cornea+scar labelmap (SAM2 + regularize + hysteresis; see
    load_or_build_labels), SKIP it if SAM2 is degraded (recorded), align it to the reference VOLUME with
    the SAME transform the intensity consensus uses, warp the scar mask NEAREST-NEIGHBOUR onto the
    reference grid, and iso-crop it (order=0) to the shared lattice as a {0,1} array. The reference
    (cases[0]) is identity when usable. A usable replicate with ZERO scar is kept (a legitimate
    "no scar here" vote that dissolves the white core), only SAM2 failures are skipped. cornea_ctx_iso is
    the reference cornea on the iso grid (the faint anatomy shell) or None if the reference is degraded."""
    n = len(cases)
    with _JOBS_LOCK:
        reps_legend = _JOBS[job_id]["replicates"]      # list of dicts, index-aligned to `cases`
    I_list: list[np.ndarray] = []
    colors01: list[np.ndarray] = []
    cornea_ctx_iso: np.ndarray | None = None
    skipped: list[str] = []
    for i, c in enumerate(cases):
        label = reps_legend[i]["label"]
        _set(job_id, progress=0.05 + 0.80 * (i / n), note=f"segmenting {label} ({i + 1}/{n})")

        def _prog(phase, idx, total, _lbl=label, _i=i):
            _set(job_id, note=f"segmenting {_lbl} ({_i + 1}/{n}) · {phase}")

        cornea, scar, meta = load_or_build_labels(c, progress=_prog)
        col = np.asarray(reps_legend[i]["color"], np.float32) / 255.0
        if meta.get("degraded"):
            skipped.append(label)
            with _JOBS_LOCK:
                reps_legend[i]["skipped"] = True
                reps_legend[i]["scar_volume_mm3"] = None
            del cornea, scar
            continue

        scar_vox = int(scar.sum())
        _set(job_id, note=f"aligning {label} ({i + 1}/{n})")
        if i == 0:                                       # reference = identity (aligns to itself)
            spm = spf
            iso_mask = (_iso_crop(scar.astype(np.float32), spf, geom, iso,
                                  order=0, antialias=False) > 0.5).astype(np.float32)
            if SCAR_CTX_ENABLED and cornea is not None and (cornea >= 1).any():
                cornea_ctx_iso = (_iso_crop((cornea >= 1).astype(np.float32), spf, geom, iso,
                                            order=0, antialias=False) > 0.5)
        else:
            vm, spm = load_volume(c)
            mi = to_sitk(vm, spm)
            R, t = _consensus_transform(method, fi, mi, fi_iso, vf, vm, spf, spm)
            sm_sitk = to_sitk(scar.astype(np.float32), spm)   # warp the SCAR mask (NN) onto the ref grid
            warped = _from_sitk(sitk.Resample(sm_sitk, fi, _tx(R, t),
                                              sitk.sitkNearestNeighbor, 0.0, sitk.sitkFloat32))
            iso_mask = (_iso_crop(warped, spf, geom, iso, order=0, antialias=False) > 0.5).astype(np.float32)
            del vm, mi, sm_sitk, warped

        scar_mm3 = scar_vox * float(spm[0]) * float(spm[1]) * float(spm[2])
        with _JOBS_LOCK:
            reps_legend[i]["skipped"] = False
            reps_legend[i]["scar_volume_mm3"] = round(scar_mm3, 6)
        I_list.append(iso_mask)
        colors01.append(col)
        del cornea, scar

    _set(job_id, skipped=skipped)
    if len(I_list) < 2:
        raise ValueError(
            "Scar consensus needs at least 2 replicates with a valid cornea/scar segmentation; only "
            f"{len(I_list)} of {n} produced one"
            + (f" (skipped: {', '.join(skipped)})" if skipped else "") + ".")
    return I_list, colors01, cornea_ctx_iso, skipped


def _consensus_run(job_id: str, cases: list[str], method: str, space: str = "intensity") -> None:
    out_dir = job_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    reference, others = cases[0], cases[1:]
    with _JOBS_LOCK:
        colors01 = [np.asarray(r["color"], np.float32) / 255.0 for r in _JOBS[job_id]["replicates"]]

    vf, spf = load_volume(reference)
    fi = to_sitk(vf, spf)
    fi_iso = reg._iso(fi)
    _set(job_id, progress=0.05)

    masks = _fixed_masks(vf, spf)            # ~1.3 s, once — defines the shared crop
    geom = view_geometry(vf, spf, masks)
    del masks
    lo, hi = window_from_fixed(vf)
    iso = DBG_ISO_MM

    cornea_ctx_iso = None
    skipped: list[str] = []
    if space == "scar":
        # SCAR space: min/excess over binary SCAR masks (white agreement core + coloured disagreement
        # halo). Reuses the SAME alignment + the SAME min/excess builder — only the per-replicate input
        # array changes (scar mask instead of windowed intensity), and I_list/colors01 span only the
        # replicates that produced a valid segmentation.
        I_list, colors01, cornea_ctx_iso, skipped = _consensus_scar_masks(
            job_id, cases, method, fi, fi_iso, vf, spf, geom, iso)
    else:
        def to_I(arr: np.ndarray) -> np.ndarray:
            return np.clip((np.asarray(arr, np.float32) - lo) / max(hi - lo, 1e-6),
                           0.0, 1.0).astype(np.float32)

        # Reference = identity (aligns to itself), iso-cropped once.
        I_list = [to_I(_iso_crop(vf, spf, geom, iso, order=1))]

        n_steps = max(len(others), 1)
        step = 0.80 / n_steps
        for n, other in enumerate(others):
            _set(job_id, progress=0.10 + n * step)
            vm, spm = load_volume(other)
            mi = to_sitk(vm, spm)
            R, t = _consensus_transform(method, fi, mi, fi_iso, vf, vm, spf, spm)
            m_res = _from_sitk(sitk.Resample(mi, fi, _tx(R, t), sitk.sitkLinear, 0.0, sitk.sitkFloat32))
            I_list.append(to_I(_iso_crop(m_res, spf, geom, iso, order=1)))
            del vm, mi, m_res

    # note is only touched for the scar space; the intensity path leaves the MAX_CONSENSUS_REPS note
    # (set in start_consensus) exactly as it was, so space="intensity" stays byte-identical.
    if space == "scar":
        _set(job_id, progress=0.92, note="compositing")
    else:
        _set(job_id, progress=0.92)
    rgb, alpha = consensus_decompose(I_list, colors01)
    if space == "scar" and cornea_ctx_iso is not None:
        _bake_cornea_context(rgb, alpha, cornea_ctx_iso)
    vol_name = "consensus_rgba.nii.gz"
    _write_rgba_nifti(out_dir / vol_name, rgb, alpha, iso)
    slices = _write_consensus_slices(rgb, geom, spf, iso, out_dir, job_id)

    done_kwargs = dict(
        volume=f"/api/debug/align/view/{job_id}/{vol_name}", iso_mm=iso, slices=slices,
        geometry={**geom, "fixed_shape": list(vf.shape),
                  "fixed_spacing_mm": [float(x) for x in spf], "window": [lo, hi],
                  "iso_shape": [int(x) for x in rgb.shape[:3]], "space": space},
        status="done", progress=1.0)
    if space == "scar":
        done_kwargs["note"] = f"skipped {', '.join(skipped)}" if skipped else None
    _set(job_id, **done_kwargs)
