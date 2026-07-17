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


def start_compare(fixed_case: str, moving_case: str, methods: list[str] | None) -> str:
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
        }
    job_dir(job_id).mkdir(parents=True, exist_ok=True)
    threading.Thread(target=_worker, args=(job_id, fixed_case, moving_case, ms), daemon=True).start()
    return job_id


def _worker(job_id: str, fixed_case: str, moving_case: str, ms: list[str]) -> None:
    try:
        with _RUN_LOCK:
            _run(job_id, fixed_case, moving_case, ms)
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


def _run(job_id: str, fixed_case: str, moving_case: str, ms: list[str]) -> None:
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

    # Identity FIRST: every other method's delta is measured against it.
    id_primary: float | None = None
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
                **extra,
            })
        except Exception as exc:  # noqa: BLE001 — one bad method must not kill the job
            _append_result(job_id, {
                "method": method, "label": METHODS[method], "ok": False, "raised": raised,
                "error": f"{type(exc).__name__}: {exc}", "rot_deg": None, "t_mm": None,
                "primary": None, "identity_primary": id_primary, "delta": None,
                "frac_out": None, "runtime_s": round(time.time() - t0, 2), "views": {},
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
