"""Strategy 2 — SAM2 video segmentation of cornea, per plane, fused in 3D.

Each orthogonal plane is treated as a *movie*: the slices along that axis are a
frame sequence.  We auto-prompt SAM2 on the middle frame (positive points on the
bright corneal band, negative points in the air), let it propagate the mask
across the whole sequence, and reassemble a full-volume cornea mask for that
plane.  Running all three planes and majority-voting the three volumes yields a
3D cornea mask that is more coherent than any single 2D pass.

The natural movie is the axial sweep (the real B-scan acquisition); coronal and
sagittal are reslices and add cross-plane consistency.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# Quiet and friendly to a shared GPU. NOTE: this path is NOT deterministic/bit-exact:
# per-frame SAM2 inputs are written as lossy JPEG (quality=95, below) — re-encoded the
# same way each run but not pixel-exact — and inference uses bf16 autocast on CUDA with
# no torch/cudnn deterministic seeding, so results may vary slightly run-to-run on GPU.
# (For exact input reproducibility, switch the frame writes to lossless PNG.)
os.environ.setdefault("HYDRA_FULL_ERROR", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
_CKPT_NAME = "sam2.1_hiera_small.pt"


def _ckpt_candidates() -> list[Path]:
    """Where to look for the SAM2 checkpoint, in priority order. The packaged app's resource dir
    (parents[1]) is READ-ONLY and does NOT bundle the checkpoint, so it must also be found in a writable
    user location: CORNEA_SAM2_CKPT (explicit file) → CORNEA_DATA_DIR/sam2_ckpt → the dev/source repo
    (parents[1]/sam2_ckpt) → the default app data dir. First existing one wins."""
    c: list[Path] = []
    env = os.environ.get("CORNEA_SAM2_CKPT")
    if env:
        c.append(Path(env).expanduser())
    dd = os.environ.get("CORNEA_DATA_DIR")
    if dd:
        c.append(Path(dd).expanduser() / "sam2_ckpt" / _CKPT_NAME)
    c.append(Path(__file__).resolve().parents[1] / "sam2_ckpt" / _CKPT_NAME)   # dev/source (and bundle, if ever shipped)
    c.append(Path.home() / ".local" / "share" / "com.cornea.oct" / "sam2_ckpt" / _CKPT_NAME)  # default app data dir
    return c


def _resolve_ckpt() -> Path | None:
    for c in _ckpt_candidates():
        if c.exists():
            return c
    return None


_PREDICTOR = None
_CUDA_TUNED = False


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _tune_cuda():
    """#5 — squeeze the installed GPU (e.g. an Ampere RTX 3090): enable TF32 matmul/cuDNN and cuDNN
    autotuning so SAM2's convolutions/matmuls run at full speed. Idempotent + best-effort (never fail
    inference over a tuning flag). nnU-Net runs in its own subprocess and self-tunes, so this is SAM2-only."""
    global _CUDA_TUNED
    if _CUDA_TUNED:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True     # Ampere TF32 matmul (≈free speedup)
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True            # autotune conv algos for our fixed slice sizes
            try:
                torch.set_float32_matmul_precision("high")   # newer torch — prefer TF32 paths
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    _CUDA_TUNED = True


def _predictor():
    global _PREDICTOR
    _tune_cuda()
    if _PREDICTOR is None:
        ckpt = _resolve_ckpt()
        if ckpt is None:
            searched = "\n  ".join(str(c) for c in _ckpt_candidates())
            raise FileNotFoundError(
                f"SAM2 checkpoint '{_CKPT_NAME}' not found. The installed app does not bundle it. "
                f"Place it at ~/.local/share/com.cornea.oct/sam2_ckpt/{_CKPT_NAME} (the app data dir) or set "
                f"CORNEA_SAM2_CKPT to its path. Searched:\n  {searched}")
        from sam2.build_sam import build_sam2_video_predictor
        _PREDICTOR = build_sam2_video_predictor(_CFG, str(ckpt), device=_device())
    return _PREDICTOR


# ---- per-slice helpers (shared idea with paint_strategy, kept self-contained) ----

def _otsu(arr: np.ndarray, nbins: int = 256) -> float:
    v = arr[np.isfinite(arr)]
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return lo
    hist, edges = np.histogram(v, bins=nbins, range=(lo, hi))
    p = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    muT = mu[-1]
    valid = (omega > 1e-6) & (omega < 1 - 1e-6)
    sb = np.zeros_like(omega)
    sb[valid] = (muT * omega[valid] - mu[valid]) ** 2 / (omega[valid] * (1 - omega[valid]))
    return float(centers[int(np.argmax(sb))])


def _norm8(sl: np.ndarray) -> np.ndarray:
    """Per-slice percentile stretch to uint8 (SAM2 wants natural-image contrast)."""
    v = sl[np.isfinite(sl)]
    if v.size == 0:
        return np.zeros(sl.shape, np.uint8)
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    if hi <= lo:
        hi = lo + 1
    return np.clip((sl - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _auto_prompt(mid: np.ndarray):
    """Positive points on the bright band core, negative points in clear air.

    Returns (points Nx2 as (x=col, y=row), labels N) or None if no band found.
    """
    t = _otsu(mid)
    m = mid >= t
    frac = float(m.mean())
    if frac < 0.01 or frac > 0.85 or m.sum() < 50:
        return None
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == int(np.argmax(sizes)) + 1
    m = ndimage.binary_fill_holes(m)
    core = ndimage.binary_erosion(m, iterations=2)
    core = core if core.any() else m
    # positives: brightest interior, spread out
    interior_vals = np.where(core, mid, -np.inf)
    cut = np.percentile(mid[core], 40)
    pos_mask = core & (mid >= cut)
    pos_mask = pos_mask if pos_mask.any() else core
    pos = _spread(np.argwhere(pos_mask), 8)
    # negatives: clearly outside the band
    far = ~ndimage.binary_dilation(m, iterations=12)
    neg = _spread(np.argwhere(far), 6)
    if len(pos) == 0:
        return None
    pts_rc = np.vstack([pos, neg]) if len(neg) else pos
    labels = np.array([1] * len(pos) + [0] * len(neg), np.int32)
    pts_xy = pts_rc[:, ::-1].astype(np.float32)  # (row,col) -> (x,y)
    return pts_xy, labels


def _spread(coords: np.ndarray, target: int) -> np.ndarray:
    if len(coords) == 0:
        return coords
    idx = np.linspace(0, len(coords) - 1, min(target, len(coords))).astype(int)
    return coords[idx]


# ---- one plane as a movie ----

def _frames_for_plane(vol: np.ndarray, plane: str):
    """Yield (frame_index, 2D slice) and the mapping back to (i,j,k).

    plane 'axial'    : frame=k, slice=(i,j)
    plane 'coronal'  : frame=j, slice=(i,k)
    plane 'sagittal' : frame=i, slice=(j,k)
    """
    ni, nj, nk = vol.shape
    if plane == "axial":
        return nk, (lambda f: vol[:, :, f])
    if plane == "coronal":
        return nj, (lambda f: vol[:, f, :])
    if plane == "sagittal":
        return ni, (lambda f: vol[f, :, :])
    raise ValueError(plane)


def _scatter_mask(out: np.ndarray, plane: str, frame: int, mask2d: np.ndarray):
    if plane == "axial":
        out[:, :, frame] |= mask2d
    elif plane == "coronal":
        out[:, frame, :] |= mask2d
    else:
        out[frame, :, :] |= mask2d


def segment_plane(vol: np.ndarray, plane: str, work: Path) -> tuple[np.ndarray, int]:
    """Run SAM2 over one plane's frame sequence; return (3D bool mask, prompt_frame)."""
    import torch
    from PIL import Image

    nframes, get = _frames_for_plane(vol, plane)
    # Frames scratch on RAM (tmpfs /dev/shm) when available: SAM2 reads them back each propagate step, so
    # RAM-backing removes the JPEG disk round-trip. PID-scoped name so concurrent processes (parallel sweep)
    # never collide. Falls back to `work` if /dev/shm isn't usable. Cleaned up in the finally below.
    _shm = Path("/dev/shm")
    _froot = _shm if (_shm.is_dir() and os.access(_shm, os.W_OK)) else work
    fdir = _froot / f"sam2frames_{plane}_{os.getpid()}"
    if fdir.exists():
        shutil.rmtree(fdir, ignore_errors=True)
    fdir.mkdir(parents=True)
    for f in range(nframes):
        Image.fromarray(np.repeat(_norm8(get(f))[:, :, None], 3, axis=2)).save(
            fdir / f"{f:05d}.jpg", quality=95)

    # Prompt on the middle frame (most reliable corneal band). If the centre
    # frame is unusable, search the immediately adjacent frames first with a
    # fine step before widening to a coarse fallback, so one bad central frame
    # doesn't force the prompt tens of slices away.
    mid = nframes // 2
    half = nframes // 2
    fine = min(5, half)
    coarse = max(1, nframes // 20)
    offsets = list(range(0, fine + 1))                       # 0,1,2,... fine (step 1)
    offsets += [o for o in range(fine + coarse, half, coarse) if o > fine]
    prm = None
    for off in offsets:
        for cand in (mid + off, mid - off):
            if 0 <= cand < nframes:
                prm = _auto_prompt(get(cand))
                if prm is not None:
                    mid = cand
                    break
        if prm is not None:
            break
    out = np.zeros(vol.shape, bool)
    if prm is None:
        shutil.rmtree(fdir, ignore_errors=True)     # no band found: still clean up frames
        return out, mid
    pts_xy, labels = prm

    predictor = _predictor()
    big = nframes > 200
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if _device() == "cuda" else _nullctx()
    state = None
    try:
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(fdir),
                                         offload_video_to_cpu=big, offload_state_to_cpu=big)
            predictor.add_new_points_or_box(state, frame_idx=mid, obj_id=1,
                                            points=pts_xy, labels=labels)
            # propagate forward then backward from the prompt frame
            for rev in (False, True):
                for fidx, _ids, logits in predictor.propagate_in_video(state, reverse=rev):
                    msk = (logits[0] > 0.0).squeeze(0).cpu().numpy()   # drop ONLY the channel axis (1,H,W)->(H,W); a bare squeeze() would collapse a degenerate size-1 spatial dim
                    _scatter_mask(out, plane, fidx, msk)
    finally:
        # Always release SAM2 state + frames, even on a CUDA OOM mid-propagate, so the
        # failure doesn't leak GPU memory or a frames dir into the next plane/scan.
        if state is not None:
            try:
                predictor.reset_state(state)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(fdir, ignore_errors=True)
    return out, mid


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ── Click-guided scar: SAM2 prompted by the user's positive/negative points ──

def _export_frames(vol: np.ndarray, plane: str, work: Path):
    from PIL import Image
    nframes, get = _frames_for_plane(vol, plane)
    fdir = work / f"frames_{plane}"
    if fdir.exists():
        shutil.rmtree(fdir)
    fdir.mkdir(parents=True)
    for f in range(nframes):
        Image.fromarray(np.repeat(_norm8(get(f))[:, :, None], 3, axis=2)).save(
            fdir / f"{f:05d}.jpg", quality=95)
    return nframes, fdir


def _ijk_to_prompt(plane: str, ijk):
    """Map a voxel (i,j,k) to (frame, (x,y)) in that plane's SAM2 frame.

    Frames are vol[:, :, k] / vol[:, j, :] / vol[i, :, :]; PIL images are (H,W) of
    that 2D slice, and SAM2 points are (x=col along width, y=row along height)."""
    i, j, k = int(ijk[0]), int(ijk[1]), int(ijk[2])
    if plane == "axial":     # slice (i,j) → H=i, W=j
        return k, (j, i)
    if plane == "coronal":   # slice (i,k) → H=i, W=k
        return j, (k, i)
    return i, (k, j)         # sagittal: slice (j,k) → H=j, W=k


def segment_plane_prompted(vol: np.ndarray, plane: str, work: Path, frame_points: dict) -> np.ndarray:
    """SAM2 over one plane, prompted by user points. frame_points: {frame: [(x,y,label),...]}
    label 1 = positive (scar), 0 = negative. Returns a 3D bool mask."""
    import torch
    nframes, fdir = _export_frames(vol, plane, work)
    out = np.zeros(vol.shape, bool)
    predictor = _predictor()
    big = nframes > 200
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if _device() == "cuda" else _nullctx()
    state = None
    try:
        with torch.inference_mode(), autocast:
            state = predictor.init_state(video_path=str(fdir),
                                         offload_video_to_cpu=big, offload_state_to_cpu=big)
            for frame, pts in frame_points.items():
                arr = np.array(pts, dtype=np.float32)
                predictor.add_new_points_or_box(state, frame_idx=int(frame), obj_id=1,
                                                points=arr[:, :2], labels=arr[:, 2].astype(np.int32))
            for rev in (False, True):
                for fidx, _ids, logits in predictor.propagate_in_video(state, reverse=rev):
                    out_msk = (logits[0] > 0.0).squeeze(0).cpu().numpy()   # drop ONLY the channel axis (see above)
                    _scatter_mask(out, plane, fidx, out_msk)
    finally:
        if state is not None:
            try:
                predictor.reset_state(state)
            except Exception:  # noqa: BLE001
                pass
        shutil.rmtree(fdir, ignore_errors=True)
    return out


def segment_scar_from_clicks(base_nifti: Path, labelmap_ijk: np.ndarray, clicks, work: Path):
    """Run SAM2 prompted by the user's scar clicks, restricted to the cornea.

    clicks: list of {ijk:[i,j,k], orientation:'axial'|'coronal'|'sagittal', positive:bool}.
    Returns (scar_mask ⊆ cornea, meta)."""
    raw = np.asarray(nib.load(str(base_nifti)).dataobj).astype(np.float32)
    vol = gaussian_filter(raw, sigma=(1.0, 1.0, 0.4))
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    cornea = (labelmap_ijk == 1) | (labelmap_ijk == 2)

    by_plane: dict = {}
    for c in clicks:
        pl = c["orientation"]
        frame, (x, y) = _ijk_to_prompt(pl, c["ijk"])
        lab = 1 if c.get("positive", True) else 0
        by_plane.setdefault(pl, {}).setdefault(frame, []).append((x, y, lab))

    scar = np.zeros(vol.shape, bool)
    per_plane = {}
    for pl, frame_points in by_plane.items():
        m = segment_plane_prompted(vol, pl, work, frame_points) & cornea
        scar |= m
        per_plane[pl] = {"voxels": int(m.sum()), "frames": sorted(frame_points.keys())}
    _free_gpu()
    return scar, {"per_plane": per_plane, "model": "sam2.1_hiera_small"}


def _plane_2d(arr3d: np.ndarray, plane: str, frame: int) -> np.ndarray:
    """The 2-D slice SAM2 sees for (plane, frame). Indexed [row, col]; a SAM2 point is (x=col, y=row)
    — matching _ijk_to_prompt (axial vol[:,:,k], coronal vol[:,j,:], sagittal vol[i,:,:])."""
    if plane == "axial":
        return arr3d[:, :, frame]
    if plane == "coronal":
        return arr3d[:, frame, :]
    return arr3d[frame, :, :]


def _dim_negatives(vol2d: np.ndarray, cornea2d: np.ndarray, pos_xy, k: int = 2, min_dist: int = 12):
    """Negative SAM2 points = the dimmest in-cornea pixels of this frame, spaced apart and away from
    the positive seeds — telling SAM2 'scar is the bright spot, NOT the normal stroma', so it carves
    the lesion instead of grabbing the whole reflective band."""
    ys, xs = np.where(cornea2d)
    if ys.size == 0:
        return []
    order = np.argsort(vol2d[ys, xs])               # dimmest first
    chosen = []
    for oi in order[:4000]:
        x, y = int(xs[oi]), int(ys[oi])
        if all(abs(x - px) + abs(y - py) > min_dist for px, py in pos_xy) and \
           all(abs(x - cx) + abs(y - cy) > min_dist for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) >= k:
            break
    return chosen


def segment_scar_consensus(base_nifti: Path, labelmap_ijk: np.ndarray, seed_ijks, work: Path,
                           vote: int = 2, planes=("axial", "coronal", "sagittal"), neg_per_frame: int = 2):
    """AUTO scar via the same 3-views-as-videos + consensus strategy used for cornea: prompt SAM2 on
    EACH plane (as a video) at the brightness seed points — PLUS dim-stroma negative points on each
    prompted frame so SAM2 carves the scar rather than the whole reflective band — propagate in 3D
    (fwd+back), and keep the CONSENSUS (≥`vote` of 3 views) ∩ cornea. Returns (mask, meta)."""
    planes = tuple(dict.fromkeys(planes))   # defence-in-depth: a duplicated plane would double-count its vote
    raw = np.asarray(nib.load(str(base_nifti)).dataobj).astype(np.float32)
    vol = gaussian_filter(raw, sigma=(1.0, 1.0, 0.4))
    work = Path(work); work.mkdir(parents=True, exist_ok=True)
    cornea = (labelmap_ijk == 1) | (labelmap_ijk == 2)
    if not seed_ijks:
        return np.zeros(vol.shape, bool), {"per_plane": {}, "vote": vote, "n_seeds": 0,
                                           "model": "sam2.1_hiera_small", "reason": "no seeds"}
    votes = np.zeros(vol.shape, np.uint8)
    per_plane = {}
    for pl in planes:
        frame_points: dict = {}
        for ijk in seed_ijks:
            frame, (x, y) = _ijk_to_prompt(pl, ijk)
            frame_points.setdefault(int(frame), []).append((x, y, 1))   # positive (scar)
        if neg_per_frame > 0:                                            # add dim-stroma negatives per frame
            for frame in list(frame_points):
                pos_xy = [(p[0], p[1]) for p in frame_points[frame] if p[2] == 1]
                for nx, ny in _dim_negatives(_plane_2d(vol, pl, frame), _plane_2d(cornea, pl, frame),
                                             pos_xy, k=neg_per_frame):
                    frame_points[frame].append((nx, ny, 0))
        try:
            m = segment_plane_prompted(vol, pl, work, frame_points) & cornea
        except Exception as exc:  # noqa: BLE001 — keep other planes if one OOMs
            per_plane[pl] = {"voxels": 0, "error": str(exc)[:200]}
            _free_gpu()
            continue
        votes += m.astype(np.uint8)
        per_plane[pl] = {"voxels": int(m.sum()), "seed_frames": sorted(frame_points)}
    fused = (votes >= vote) & cornea
    _free_gpu()
    return fused, {"per_plane": per_plane, "vote": vote, "n_seeds": len(seed_ijks),
                   "neg_per_frame": neg_per_frame, "model": "sam2.1_hiera_small"}


def segment_volume(volume_nifti: Path, work: Path,
                   planes=("axial", "coronal", "sagittal"),
                   vote: int = 2, progress=None) -> tuple[np.ndarray, dict]:
    """SAM2 each plane, majority-vote into a single 3D cornea labelmap (0/1).

    `progress(phase, index, total)` is an optional callback invoked at the start of each plane
    (phase=the plane name) and before the 3D fuse (phase="fuse"), so a caller can surface live
    progress. Defaults to None so the standalone/__main__ and other callers are unaffected."""
    planes = tuple(dict.fromkeys(planes))   # defence-in-depth: a duplicated plane would double-count its vote
    raw = np.asarray(nib.load(str(volume_nifti)).dataobj).astype(np.float32)
    vol = gaussian_filter(raw, sigma=(1.0, 1.0, 0.4))
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)

    votes = np.zeros(vol.shape, np.uint8)
    per_plane = {}
    planes_failed = {}
    for idx, pl in enumerate(planes):
        if progress is not None:
            try:
                progress(pl, idx, len(planes))
            except Exception:  # noqa: BLE001 — progress is best-effort, never fail the segmentation
                pass
        try:
            m, prm = segment_plane(vol, pl, work)
        except Exception as exc:  # noqa: BLE001  (e.g. CUDA OOM): record + keep other planes
            planes_failed[pl] = str(exc)[:200]
            per_plane[pl] = {"voxels": 0, "error": str(exc)[:200]}
            _free_gpu()
            continue
        nvox = int(m.sum())
        if nvox == 0:                               # SAM2 found no corneal band on this plane
            planes_failed[pl] = "no cornea band found (auto-prompt failed)"
        votes += m.astype(np.uint8)
        per_plane[pl] = {"voxels": nvox, "prompt_frame": prm}

    if progress is not None:
        try:
            progress("fuse", len(planes), len(planes))
        except Exception:  # noqa: BLE001
            pass
    label, core_grow, thickness_filled = _fuse_votes(votes, vote)
    meta = {"per_plane": per_plane, "vote_threshold": vote,
            "cornea_voxels": int(label.sum()), "model": "sam2.1_hiera_small",
            "planes_failed": planes_failed,
            "thickness_filled_voxels": thickness_filled,   # #13 — how many in-thickness gaps were closed
            "core_grow": core_grow,                        # peripheral-run recovery (core→contiguous union)
            "degraded": bool(planes_failed)}        # surfaced so a silent under-segment is visible
    _free_gpu()
    return label, meta


def _finalize_vote_mask(m: np.ndarray):
    """Largest connected component → fill enclosed holes → solid-fill each A-scan (depth) column between its
    first/last cornea voxel (#13: the cornea is one contiguous band along depth, so an in-column gap is a SAM2
    vote dropout, not anatomy; downstream scar code assumes a solid cornea). Returns (mask, n_thickness_filled)."""
    lbl, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
        m = lbl == int(np.argmax(sizes)) + 1
    m = ndimage.binary_fill_holes(m)
    before = int(m.sum())
    m = _fill_cornea_thickness(m)
    return m, int(m.sum()) - before


def _fuse_votes(votes: np.ndarray, vote: int):
    """Fuse the per-plane vote count into the final cornea label. Returns (label uint8, core_grow dict,
    thickness_filled int).

    CORE-GROW recovery: SAM2 tracks each plane as a video, so a plane can lose the cornea on a run of
    PERIPHERAL frames and never recover — then the 2-of-3 vote collapses to empty there even though another
    plane (which cuts ACROSS frames) still sees it (observed CS020 OS(2): frames 73-100 present only in the
    axial plane → dropped). Recover by growing the high-confidence CORE (votes≥vote) through spatially-
    CONTIGUOUS single-plane (union, votes≥1) evidence. The acceptance test is PER union-component and is judged
    on the FINAL (thickness-filled) mass — because thickness-fill amplifies a deep/contiguous flood: a
    component is kept only if, once finalized, it stays ≤2.5× the CORE MASS IT CONTAINS (a genuine peripheral
    run runs ~1.5-2×; a single plane leaking into sclera, or a depth-leak that thickness-fill balloons, blows
    past it). Per-component (not one global cap) so an unrelated over-cap blob can't veto a legitimate
    recovery; an isolated single-plane blob never touches the core so is never grown. (surface-crop floods are
    also gated upstream in oct_preprocess.)"""
    core = votes >= vote
    core_final, thickness_filled = _finalize_vote_mask(core)
    fused = core_final
    core_grow = {"applied": False, "core_vox": int(core.sum()), "grown_vox": int(core_final.sum())}
    if vote > 1 and core.any():
        union = votes >= 1
        lbl_u, _n = ndimage.label(union)
        # Per-A-scan-column thickness of the CORE (voxels along the depth axis). A genuine peripheral-run
        # recovery EXTENDS the band into more frames at ~the cornea's own thickness; a single-plane sclera/
        # depth leak THICKENS columns (and thickness-fill amplifies it into a fat slab). So besides the mass
        # cap, reject a component whose FINALIZED per-column thickness (95th pct) exceeds ~1.8× the core's
        # median — a total-voxel ratio alone can't tell a lateral extension from a depth over-segmentation
        # (both ~1.5-1.8× mass), but the thickness does. (Regression guard: CS001 OS(2) slab, mass 1.66×.)
        depth = _depth_axis(core)
        core_ct = core.sum(axis=depth)
        core_med = float(np.median(core_ct[core_ct > 0])) if (core_ct > 0).any() else 0.0
        recovered = core.copy()
        for lab in set(int(v) for v in np.unique(lbl_u[core]) if v):
            comp = lbl_u == lab
            cc = int((comp & core).sum())
            if not cc:
                continue
            cand = _finalize_vote_mask(comp)[0]
            cand_ct = cand.sum(axis=depth)
            cand_p95 = float(np.percentile(cand_ct[cand_ct > 0], 95)) if (cand_ct > 0).any() else 0.0
            if int(cand.sum()) <= 2.5 * cc and (core_med <= 0 or cand_p95 <= 1.8 * core_med):
                recovered |= comp
        if int(recovered.sum()) > int(core.sum()):
            grown_final, tf_grown = _finalize_vote_mask(recovered)
            fused, thickness_filled = grown_final, tf_grown
            core_grow["applied"] = True
            core_grow["grown_vox"] = int(grown_final.sum())
    return fused.astype(np.uint8), core_grow, thickness_filled


def _depth_axis(mask: np.ndarray) -> int:
    """The A-scan/depth axis = the one whose face-on projection fills the cornea into the densest disc
    (the cornea is a thin curved shell; collapsing its thin direction yields the largest footprint).
    Mirrors scar._depth_axis so the thickness-fill agrees with downstream scar detection."""
    shape = mask.shape
    best_axis, best_score = 0, -1.0
    for a in range(3):
        footprint = int(mask.any(axis=a).sum())
        plane_area = shape[(a + 1) % 3] * shape[(a + 2) % 3]
        score = footprint / max(plane_area, 1)
        if score > best_score:
            best_axis, best_score = a, score
    return best_axis


def _fill_cornea_thickness(mask: np.ndarray) -> np.ndarray:
    """Make the cornea a SOLID band along the A-scan/depth axis: for each (lateral, frame) column, fill
    every voxel between the first and last cornea voxel. Closes in-thickness vote-dropout holes that
    binary_fill_holes can't (those that leak to an edge). A column with no cornea stays empty. Vectorised."""
    if not mask.any():
        return mask
    depth = _depth_axis(mask)
    m = np.moveaxis(mask, depth, -1)                 # (A, B, D) — depth last
    present = m.any(axis=-1)                          # columns that contain any cornea
    if not present.any():
        return mask
    D = m.shape[-1]
    idx = np.arange(D)
    first = m.argmax(axis=-1)                         # first True along depth (0 where empty — masked below)
    last = D - 1 - m[..., ::-1].argmax(axis=-1)       # last True along depth
    band = (idx[None, None, :] >= first[..., None]) & (idx[None, None, :] <= last[..., None])
    band &= present[..., None]                        # don't fabricate a band in empty columns
    return np.moveaxis(band, -1, depth)


def _free_gpu():
    """Release cached GPU memory so back-to-back scans (consensus) don't accumulate."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    import sys
    import json
    vn = Path(sys.argv[1])
    pl = sys.argv[2] if len(sys.argv) > 2 else "axial"
    raw = np.asarray(nib.load(str(vn)).dataobj).astype(np.float32)
    vol = gaussian_filter(raw, sigma=(1.0, 1.0, 0.4))
    m, prm = segment_plane(vol, pl, Path("/tmp/sam2_work"))
    print(json.dumps({"plane": pl, "voxels": int(m.sum()),
                      "prompt_frame": prm, "shape": list(m.shape)}))
