"""Scar detection + quantification — the optional 3rd class inside cornea.

Scar is a hyper-reflective sub-region of the corneal stroma (may be absent). We
flag bright candidates inside the corrected cornea mask for expert correction,
then quantify the corrected scar: volume (mm³), en-face area (mm²), and
densitometry (reflectivity + density tiers). Label convention: 0/1/2.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

BG, CORNEA, SCAR = 0, 1, 2


# ── Strategy-2 PoC: direct scar mask inside the cornea ROI (no Slicer grow) ──

def cornea_roi_smoothed(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray,
                        erode_surface: int = 6, smooth: float = 2.5):
    """The deep-stroma ROI (epithelium/Bowman's/endothelium rind trimmed ALONG DEPTH only — Bowman's
    and the endothelium are offset in depth, so eroding across slices would needlessly drop edge
    slices) + the smoothed volume. Shared by the hyper-reflective threshold and the reproducibility-
    tuned scar strategies. Returns (cornea, roi, smoothed_vol)."""
    cornea = (labelmap_ijk == CORNEA) | (labelmap_ijk == SCAR)
    if not cornea.any():
        return cornea, cornea, vol_ijk
    if erode_surface > 0:
        depth = _depth_axis(cornea)
        st = np.zeros((3, 3, 3), bool); st[1, 1, 1] = True
        for off in (0, 2):
            nb = [1, 1, 1]; nb[depth] = off; st[tuple(nb)] = True
        roi = ndimage.binary_erosion(cornea, structure=st, iterations=erode_surface)
    else:
        roi = cornea
    if not roi.any():
        roi = cornea
    v = ndimage.gaussian_filter(vol_ijk, sigma=(smooth, smooth, max(0.6, smooth * 0.4))) \
        if smooth > 0 else vol_ijk
    return cornea, roi, v


def hyper_reflective_mask(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray,
                          percentile: float = 88.0, erode_surface: int = 6,
                          smooth: float = 2.5) -> np.ndarray:
    """The bright (hyper-reflective) voxels inside the cornea — scar candidates
    before morphology. Erodes the epithelium/Bowman's/endothelium rind along the
    depth axis, smooths, and keeps the brightest `100−percentile`% of the stroma.
    Used both by the auto detector and to constrain a SAM2 click region to scar."""
    cornea, roi, v = cornea_roi_smoothed(vol_ijk, labelmap_ijk, erode_surface, smooth)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    thresh = float(np.percentile(v[roi].astype(np.float32), percentile))
    return roi & (v >= thresh)


def detect_scar_in_cornea(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray,
                          percentile: float = 88.0, min_voxels: int = 500,
                          erode_surface: int = 6, smooth: float = 2.5,
                          open_iter: int = 1, close_iter: int = 4) -> np.ndarray:
    """ROI-restricted scar *candidate* highlighter for expert correction.

    Corneal scar reads as hyper-reflective (bright/white) tissue *within the deep
    stroma*. Run this on the **contrast-enhanced working volume** (what the user
    sees). Steps:
      1. Erode the cornea by `erode_surface` voxels to drop the reflective rind —
         epithelium + **Bowman's layer** (anterior) and endothelium (posterior),
         which are normally bright and must not be mistaken for scar.
      2. Gaussian-smooth (`smooth` in-plane, less through-plane) so the bright scar
         reads as a contiguous region, not speckle, then flag the brightest
         `100−percentile`% of the remaining deep-stroma voxels.
      3. Make a WELL-DEFINED, CONTINUOUS area: open (despeckle) → close (bridge gaps)
         → fill holes → keep only 3D connected components ≥ `min_voxels`. A real scar
         is a coherent 3D volume, so this also guarantees the axial/coronal/sagittal
         views show the SAME object (cross-plane sanity check by construction).

    `percentile` is the sensitivity knob (lower → more highlighted). Whether a bright
    region is truly scar vs normal reflection is a clinical judgement, so the expert
    prunes/extends this in the drawing layer; presence is decided by the corrected
    labelmap. Returns a boolean scar mask ⊆ cornea.
    """
    cornea = (labelmap_ijk == CORNEA) | (labelmap_ijk == SCAR)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    scar = hyper_reflective_mask(vol_ijk, labelmap_ijk, percentile=percentile,
                                 erode_surface=erode_surface, smooth=smooth)
    if open_iter > 0:
        scar = ndimage.binary_opening(scar, iterations=open_iter)
    if close_iter > 0:
        # Edge-preserving closing: a plain binary_closing erodes the volume's end
        # faces (border treated as background), which would zero the scar on the
        # first/last slices. Dilate, then erode with border_value=1 so the volume
        # boundary is kept — the scar can reach the first and last slices.
        scar = ndimage.binary_dilation(scar, iterations=close_iter)
        scar = ndimage.binary_erosion(scar, iterations=close_iter, border_value=1)
    scar = ndimage.binary_fill_holes(scar)          # solid area, no internal gaps
    lbl, n = ndimage.label(scar)                    # 3D connectivity → coherent volumes
    if n == 0:
        return np.zeros(labelmap_ijk.shape, bool)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
    if not keep:
        return np.zeros(labelmap_ijk.shape, bool)
    return np.isin(lbl, keep) & cornea


def _morph_clean(scar: np.ndarray, cornea: np.ndarray, open_iter: int = 1,
                 close_iter: int = 4, min_voxels: int = 500, largest_only: bool = False) -> np.ndarray:
    """Despeckle (open) → edge-preserving close → fill holes → keep 3D components ≥ min_voxels ∩
    cornea (or only the single largest, if `largest_only`). The close dilates then erodes with
    border_value=1 so the scar can reach the first/last slice (a plain closing zeroes the end faces)."""
    if open_iter > 0:
        scar = ndimage.binary_opening(scar, iterations=open_iter)
    if close_iter > 0:
        scar = ndimage.binary_dilation(scar, iterations=close_iter)
        scar = ndimage.binary_erosion(scar, iterations=close_iter, border_value=1)
    scar = ndimage.binary_fill_holes(scar)
    lbl, n = ndimage.label(scar)
    if n == 0:
        return np.zeros(cornea.shape, bool)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    if largest_only:
        return (lbl == int(np.argmax(sizes)) + 1) & cornea if sizes.max() >= min_voxels else np.zeros(cornea.shape, bool)
    keep = [i + 1 for i, s in enumerate(sizes) if s >= min_voxels]
    return (np.isin(lbl, keep) & cornea) if keep else np.zeros(cornea.shape, bool)


def detect_scar_hysteresis(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray, phi_percentile: float = 92.0,
                           gap: float = 12.0, erode_surface: int = 6, smooth: float = 2.5,
                           min_voxels: int = 500, open_iter: int = 1, close_iter: int = 4) -> np.ndarray:
    """Hysteresis scar candidate (more REPRODUCIBLE across repeat scans than a single percentile cut).

    Seed at the high percentile `phi_percentile`, then grow to all in-stroma voxels connected to a
    seed that exceed the lower cut (`phi_percentile - gap`). The boundary is decided by CONNECTIVITY
    to a stable bright core, not by where intensity crosses one threshold on the histogram's steep
    flank — so the per-scan gain/brightness drift that flips a marginal rind in/out (the main source
    of replicate disagreement) no longer moves the boundary. Benchmarked on CS001-OS v1/v2/v3: pairwise
    3D scar Dice 0.66→0.79 AND volume CV 0.86%→0.60% vs the percentile detector, at the same volume
    scale (validated on one eye — confirm on more). Returns a boolean scar mask ⊆ cornea."""
    cornea, roi, v = cornea_roi_smoothed(vol_ijk, labelmap_ijk, erode_surface, smooth)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    vs = v[roi].astype(np.float32)
    thi = float(np.percentile(vs, phi_percentile))
    tlo = float(np.percentile(vs, max(0.0, phi_percentile - gap)))
    lbl, _ = ndimage.label(roi & (v >= tlo))            # low-threshold connected regions
    keep = set(np.unique(lbl[roi & (v >= thi)])) - {0}  # ...that contain a high-threshold seed
    return _morph_clean(np.isin(lbl, list(keep)), cornea, open_iter, close_iter, min_voxels)


def detect_scar_normal_anchor(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray, k: float = 2.0,
                              erode_surface: int = 6, smooth: float = 2.5, min_voxels: int = 500,
                              open_iter: int = 1, close_iter: int = 4) -> np.ndarray:
    """Threshold anchored to NORMAL stroma: μ,σ of the dim half (≤ median) of in-cornea voxels; scar =
    voxels brighter than μ+k·σ. A per-scan z-score → flags the same RELATIVE excess reflectivity in
    every replicate. Highest overlap in benchmarks (Dice ~0.88) but ~3–4× the volume of hysteresis (a
    more inclusive 'hyper-reflective burden', not a like-for-like scar)."""
    cornea, roi, v = cornea_roi_smoothed(vol_ijk, labelmap_ijk, erode_surface, smooth)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    vs = v[roi].astype(np.float32)
    dim = vs[vs <= np.median(vs)]
    thr = float(dim.mean() + k * dim.std())
    return _morph_clean(roi & (v >= thr), cornea, open_iter, close_iter, min_voxels)


def detect_scar_robust_mad(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray, k: float = 0.6,
                           erode_surface: int = 6, smooth: float = 2.5, min_voxels: int = 500,
                           open_iter: int = 1, close_iter: int = 4) -> np.ndarray:
    """median + k·1.4826·MAD over in-cornea voxels — MAD ignores the scar's own bright tail, so the
    per-scan threshold is reproducible. Middle ground: Dice ~0.84, ~2.5× hysteresis volume."""
    cornea, roi, v = cornea_roi_smoothed(vol_ijk, labelmap_ijk, erode_surface, smooth)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    vs = v[roi].astype(np.float32)
    med = float(np.median(vs))
    mad = 1.4826 * float(np.median(np.abs(vs - med)))
    return _morph_clean(roi & (v >= med + k * mad), cornea, open_iter, close_iter, min_voxels)


def detect_scar_morph_lcc(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray, percentile: float = 88.0,
                          erode_surface: int = 6, smooth: float = 2.5, min_voxels: int = 500,
                          open_iter: int = 2, close_iter: int = 8) -> np.ndarray:
    """Percentile candidate + stronger morphology, keeping ONLY the largest 3D component (drops the
    satellite blobs that flicker between replicates). Compact, lowest volume of the set."""
    cornea, roi, v = cornea_roi_smoothed(vol_ijk, labelmap_ijk, erode_surface, smooth)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    thresh = float(np.percentile(v[roi].astype(np.float32), percentile))
    return _morph_clean(roi & (v >= thresh), cornea, open_iter, close_iter, min_voxels, largest_only=True)


def detect_scar_hysteresis_tta(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray, phi_percentile: float = 92.0,
                               gap: float = 12.0, jitters=(-3, -2, -1, 0, 1, 2, 3), erode_surface: int = 6,
                               smooth: float = 2.5, min_voxels: int = 500, open_iter: int = 1, close_iter: int = 4):
    """Threshold-MARGINALISED (soft) hysteresis: run the seed/grow over a BAND of seed percentiles
    (phi + each jitter), average the resulting masks into a per-voxel scar probability, and keep
    voxels that are scar for the MAJORITY of thresholds (prob ≥ 0.5). This is a training-free, soft-
    boundary / test-time-augmentation idea: it averages out the boundary's sensitivity to the exact
    cut (the main source of cross-replicate jitter) instead of committing to one threshold. The
    expensive smoothing is done once; only the percentile varies."""
    cornea, roi, v = cornea_roi_smoothed(vol_ijk, labelmap_ijk, erode_surface, smooth)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    vs = v[roi].astype(np.float32)
    acc = np.zeros(labelmap_ijk.shape, np.float32)
    for dj in jitters:
        phi = float(min(99.0, max(50.0, phi_percentile + dj)))
        thi = float(np.percentile(vs, phi))
        tlo = float(np.percentile(vs, max(0.0, phi - gap)))
        lbl, _ = ndimage.label(roi & (v >= tlo))
        keep = set(np.unique(lbl[roi & (v >= thi)])) - {0}
        acc += np.isin(lbl, list(keep)).astype(np.float32)
    prob = acc / len(jitters)                          # per-voxel scar probability over the threshold band
    return _morph_clean(prob >= 0.5, cornea, open_iter, close_iter, min_voxels)


# ── Depth-normalised scar: flag EXCESS over the NORMAL corneal reflectivity profile ────────────────
# Normal cornea is intrinsically hyper-reflective at certain RELATIVE depths — the anterior
# epithelium/Bowman's layer and the posterior Descemet's/endothelium — and brightest where reflectance
# peaks (e.g. the specular apex). An absolute brightness threshold therefore over-flags these normal
# bright bands as "scar" (the Bowman's over-sensitivity). The fix: model normal reflectivity as a
# function of relative corneal depth (0=anterior surface → 1=posterior surface) and flag scar only
# where a voxel exceeds the NORMAL mean+k·sd FOR ITS DEPTH. The normal profile is learned from CONTROL
# scans when available (normal_baseline.py), else estimated robustly from the scan itself.
NPROF_BINS = 40


def relative_corneal_depth(cornea: np.ndarray, depth_axis: int) -> np.ndarray:
    """Per-voxel relative depth through the cornea along `depth_axis`: 0 at the anterior surface, 1 at
    the posterior surface of that A-scan; NaN outside the cornea. Normalising by each A-scan's own
    thickness makes Bowman's (just below the anterior surface) land at a consistent relative depth
    across the curved, variable-thickness cornea."""
    c = np.moveaxis(cornea, depth_axis, -1)
    D = c.shape[-1]
    anyc = c.any(axis=-1)
    first = np.argmax(c, axis=-1)
    last = D - 1 - np.argmax(c[..., ::-1], axis=-1)
    span = np.maximum(last - first, 1).astype(np.float32)
    idx = np.arange(D, dtype=np.float32)
    r = (idx[None, ...] - first[..., None].astype(np.float32)) / span[..., None]
    r = np.where(c, np.clip(r, 0.0, 1.0), np.nan).astype(np.float32)
    r[~anyc] = np.nan
    return np.moveaxis(r, -1, depth_axis)


def _gain_ref(v: np.ndarray, roi: np.ndarray, rdepth: np.ndarray) -> float:
    """Per-scan gain reference = robust mid-stroma level (relative depth 0.4–0.6), so absolute
    reflectivity/gain differences between scans cancel before comparing to a normal profile."""
    mid = roi & (rdepth >= 0.4) & (rdepth <= 0.6)
    sel = v[mid] if mid.any() else v[roi]
    ref = float(np.median(sel)) if sel.size else 1.0
    return ref if ref > 1e-6 else 1.0


def depth_profile_stats(vn_roi: np.ndarray, r_roi: np.ndarray, nbins: int = NPROF_BINS):
    """Robust normal reflectivity per relative-depth bin: (median, 1.4826·MAD) of gain-normalised
    intensity `vn_roi` at relative depths `r_roi`. MAD/median ignore the scar's own bright tail, so a
    profile estimated from a scan that CONTAINS scar still reflects the normal tissue. Empty bins are
    filled by interpolation. Returns (mean[nbins], sd[nbins])."""
    bins = np.clip((r_roi * nbins).astype(int), 0, nbins - 1)
    mean = np.full(nbins, np.nan, np.float32)
    sd = np.full(nbins, np.nan, np.float32)
    for b in range(nbins):
        sel = vn_roi[bins == b]
        if sel.size >= 20:
            med = float(np.median(sel))
            mad = 1.4826 * float(np.median(np.abs(sel - med)))
            mean[b] = med
            sd[b] = mad if mad > 1e-6 else (float(sel.std()) or 1.0)
    ok = ~np.isnan(mean)
    if ok.any():
        xs = np.arange(nbins)
        mean = np.interp(xs, xs[ok], mean[ok])
        sd = np.interp(xs, xs[ok], sd[ok])
        sd[sd <= 1e-6] = float(np.nanmedian(sd[sd > 1e-6])) if (sd > 1e-6).any() else 1.0
    else:
        mean[:] = float(np.median(vn_roi)) if vn_roi.size else 0.0
        sd[:] = 1.0
    return mean, sd


def enface_coords(cornea: np.ndarray, depth_axis: int, spacing):
    """Per-voxel normalized en-face radius ρ∈[0,1] (from the cornea's en-face centroid) and meridian
    θ∈[0,2π), in the plane orthogonal to `depth_axis`. Feeds the 3-D normal atlas (depth×radius×meridian),
    which captures the bright specular apex + peripheral signal falloff that a depth-only profile blurs."""
    ax = [a for a in range(3) if a != depth_axis]
    idx = np.indices(cornea.shape)
    ci = float(idx[ax[0]][cornea].mean()); cj = float(idx[ax[1]][cornea].mean())
    dx = (idx[ax[0]] - ci) * spacing[ax[0]]; dy = (idx[ax[1]] - cj) * spacing[ax[1]]
    rho = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    rmax = float(np.percentile(rho[cornea], 95)) or 1.0
    theta = ((np.arctan2(dy, dx).astype(np.float32)) + 2.0 * np.pi) % (2.0 * np.pi)
    return np.clip(rho / rmax, 0.0, 1.0).astype(np.float32), theta


def scar_from_z(z: np.ndarray, cornea: np.ndarray, roi: np.ndarray, k_abs=None, margin: float = 2.5,
                phi_percentile: float = 92.0, gap: float = 12.0, min_voxels: int = 500,
                open_iter: int = 1, close_iter: int = 4) -> np.ndarray:
    """Hysteresis + morphology on a depth/atlas-conditional z-map → scar mask. ABSOLUTE (k_abs given:
    seed z≥k_abs, grow≥k_abs−margin — controls→~0) or PERCENTILE (seed φ-th pct, grow φ−gap)."""
    if k_abs is not None:
        thi, tlo = float(k_abs), float(k_abs) - float(margin)
    else:
        zr = z[roi]
        thi = float(np.percentile(zr, phi_percentile))
        tlo = float(np.percentile(zr, max(0.0, phi_percentile - gap)))
    seed = roi & (z >= thi)
    if not seed.any():
        return np.zeros(cornea.shape, bool)
    grow = roi & (z >= tlo)
    lbl, _ = ndimage.label(grow)
    keep = set(np.unique(lbl[seed])) - {0}
    return _morph_clean(np.isin(lbl, list(keep)), cornea, open_iter, close_iter, min_voxels)


def detect_scar_depthnorm(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray, profile=None, k_abs=None,
                          margin: float = 2.5, phi_percentile: float = 92.0, gap: float = 12.0,
                          erode_surface: int = 6, smooth: float = 2.5, min_voxels: int = 500,
                          open_iter: int = 1, close_iter: int = 4) -> np.ndarray:
    """Scar = reflectivity EXCESS over the normal corneal depth profile, by HYSTERESIS on a depth-
    conditional z. At each in-stroma voxel: z = (v_norm − μ(r)) / σ(r), where r is relative corneal
    depth and μ,σ are normal reflectivity at that depth — from `profile` (a CONTROL-derived (mean[],sd[])
    tuple) when given, else estimated from THIS scan. Normal Bowman's/anterior brightness is folded into
    μ(r) so it is NOT flagged; only excess over normal is.

    Two thresholding modes:
      • CONTROL-ANCHORED ABSOLUTE (k_abs set, needs a control profile): seed where z ≥ k_abs, grow to
        connected z ≥ k_abs−margin. k_abs is calibrated from controls so a NORMAL cornea yields ~0 scar
        (true specificity: controls→~0); scar = genuine σ-excess over normal. This is the mode that
        makes control scans actually improve segmentation.
      • PERCENTILE hysteresis (no k_abs, e.g. self-mode): seed at the φ-th percentile of in-stroma z,
        grow to ≥ φ−gap. Reproducible but ALWAYS flags ~(100−φ)% — cannot reach 0 on a normal cornea.
    Returns a boolean scar mask ⊆ cornea."""
    cornea, roi, v = cornea_roi_smoothed(vol_ijk, labelmap_ijk, erode_surface, smooth)
    if not cornea.any() or not roi.any():
        return np.zeros(labelmap_ijk.shape, bool)
    depth = _depth_axis(cornea)
    rdepth = relative_corneal_depth(cornea, depth)
    ref = _gain_ref(v, roi, rdepth)
    vn = (v / ref).astype(np.float32)
    if profile is not None:
        mean = np.asarray(profile[0], np.float32)
        sd = np.asarray(profile[1], np.float32)
        nbins = mean.shape[0]
    else:
        nbins = NPROF_BINS
        mean, sd = depth_profile_stats(vn[roi], rdepth[roi], nbins)
    bins_full = np.clip((np.nan_to_num(rdepth, nan=0.0) * nbins).astype(int), 0, nbins - 1)
    z = np.zeros(labelmap_ijk.shape, np.float32)
    z[roi] = (vn[roi] - mean[bins_full[roi]]) / np.maximum(sd[bins_full[roi]], 1e-6)
    return scar_from_z(z, cornea, roi, k_abs=k_abs, margin=margin, phi_percentile=phi_percentile,
                       gap=gap, min_voxels=min_voxels, open_iter=open_iter, close_iter=close_iter)


# The module-default percentile (== hyper_reflective_mask / detect_scar_in_cornea default). Used as the
# anchor point where the percentile→k map reproduces a detector's benchmarked default k, so threading the
# slider through normal_anchor/robust_mad doesn't shift their tuned default behaviour.
_DEFAULT_PCT = 88.0


def _k_from_percentile(default_k: float, pct: float) -> float:
    """Map the sensitivity slider's percentile to a sigma-multiplier `k` for the threshold detectors
    (normal_anchor μ+k·σ, robust_mad median+k·MAD). The slider sends percentile = 100 − sensitivity, so a
    HIGHER sensitivity → LOWER percentile → LOWER k (a more inclusive, lower threshold). Anchored so pct ==
    _DEFAULT_PCT (88) reproduces the detector's tuned default k; scaled linearly off the population midpoint
    (50) and floored so k stays positive. Monotone in pct."""
    k = default_k * (float(pct) - 50.0) / (_DEFAULT_PCT - 50.0)
    return max(0.1, k)


# Strategy registry — name → detector(vol, lab, percentile). Used by /scar/auto's `method` selector
# so the strategies can be A/B-compared in the viewer.
def scar_detector(method: str):
    m = (method or "hysteresis").lower()
    if m in ("depthnorm", "control", "normal_profile"):
        # self-normalised depth-conditional (the api passes a control profile when one is built)
        return lambda vol, lab, pct: detect_scar_depthnorm(vol, lab, phi_percentile=pct)
    if m in ("tta", "hysteresis_tta"):
        return lambda vol, lab, pct: detect_scar_hysteresis_tta(vol, lab, phi_percentile=pct)
    if m == "normal_anchor":
        return lambda vol, lab, pct: detect_scar_normal_anchor(vol, lab, k=_k_from_percentile(2.0, pct))
    if m == "robust_mad":
        return lambda vol, lab, pct: detect_scar_robust_mad(vol, lab, k=_k_from_percentile(0.6, pct))
    if m == "morph_lcc":
        return lambda vol, lab, pct: detect_scar_morph_lcc(vol, lab, percentile=pct)
    if m == "brightness":
        return lambda vol, lab, pct: detect_scar_in_cornea(vol, lab, percentile=pct)
    return lambda vol, lab, pct: detect_scar_hysteresis(vol, lab, phi_percentile=pct)  # default


def auto_scar_seeds(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray, percentile: float = 88.0,
                    erode_surface: int = 6, smooth: float = 2.5, frame_mask: np.ndarray | None = None,
                    max_seeds: int = 5, min_seed_voxels: int = 80):
    """Turn the hyper-reflective scar candidate into a few SAM2 SEED POINTS — the brightest voxel of
    each of the largest bright components inside cornea (optionally restricted to the marked scar
    frames via `frame_mask`). Returns (seed_ijks, bright_mask): seeds prompt the 3-view SAM2
    consensus, and bright_mask constrains its output to truly hyper-reflective scar."""
    bright = hyper_reflective_mask(vol_ijk, labelmap_ijk, percentile=percentile,
                                   erode_surface=erode_surface, smooth=smooth)
    if frame_mask is not None:
        bright = bright & frame_mask
    if not bright.any():
        return [], bright
    lbl, n = ndimage.label(bright)
    if n == 0:
        return [], bright
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    seeds = []
    for idx in np.argsort(sizes)[::-1]:             # largest bright components first
        if sizes[idx] < min_seed_voxels:
            break
        comp = lbl == (idx + 1)
        vals = np.where(comp, vol_ijk, -np.inf)     # brightest voxel of the component = most scar-like
        s = np.unravel_index(int(np.argmax(vals)), vals.shape)
        seeds.append([int(s[0]), int(s[1]), int(s[2])])
        if len(seeds) >= max_seeds:
            break
    return seeds, bright


def frame_range_mask(shape, spacing_xyz, scar_range):
    """Boolean volume True only on the frames in `scar_range` ([start,end], 1-based inclusive). The
    frame/B-scan axis is the COARSEST-spacing axis (slice spacing ≫ depth/lateral). None if invalid."""
    if not scar_range or len(scar_range) != 2:
        return None                                          # genuinely no range supplied → whole-volume
    fax = int(np.argmax([float(s) for s in spacing_xyz[:3]]))
    lo, hi = sorted((int(scar_range[0]), int(scar_range[1])))  # normalize inverted ranges (e.g. [300,200])
    lo -= 1                                                  # 1-based inclusive → 0-based [lo, hi)
    lo, hi = max(0, lo), min(shape[fax], hi)
    fm = np.zeros(shape, bool)
    if lo >= hi:
        return fm                                            # marked range fell outside the volume → empty
                                                             # mask (caller errors distinctly, NOT whole-volume)
    sl = [slice(None)] * 3
    sl[fax] = slice(lo, hi)
    fm[tuple(sl)] = True
    return fm


def apply_scar_to_labelmap(labelmap_ijk: np.ndarray, scar_mask: np.ndarray,
                           replace: bool = False) -> np.ndarray:
    """Return a 0/1/2 labelmap with scar overlaid on cornea (scar overrides cornea).

    replace=False (default): MERGE — union the new candidates with existing scar so a
    re-run never silently erases the expert's manual scar (or finds nothing and wipes
    it). replace=True: discard prior scar first (a clean re-detection)."""
    out = labelmap_ijk.copy()
    if replace:
        out[(out == SCAR)] = CORNEA       # clean re-detection: reset prior scar to cornea
    out[scar_mask & (out == CORNEA)] = SCAR
    return out


def _depth_axis(cornea_mask: np.ndarray) -> int:
    """The A-scan/depth axis = the one whose face-on projection fills the cornea
    into the densest disc (the cornea is a thin curved shell — collapsing its thin
    direction yields the largest footprint). Returns 0, 1, or 2."""
    shape = cornea_mask.shape
    best_axis, best_score = 0, -1.0
    for a in range(3):
        footprint = int(cornea_mask.any(axis=a).sum())
        plane_area = shape[(a + 1) % 3] * shape[(a + 2) % 3]
        score = footprint / max(plane_area, 1)      # how fully the en-face disc fills
        if score > best_score:
            best_axis, best_score = a, score
    return best_axis


def density_tiers(scar_mask: np.ndarray, density_vol_ijk: np.ndarray, n_tiers: int = 3):
    """Split a scar mask into `n_tiers` reflectivity tiers (diffuse→dense) by
    intra-scar intensity quantiles. Returns (tier_index_volume, edges) where tier
    index is 1..n_tiers inside the scar and 0 elsewhere."""
    out = np.zeros(scar_mask.shape, np.uint8)
    vals = density_vol_ijk[scar_mask].astype(np.float32)
    if vals.size == 0:
        return out, []
    qs = [float(np.quantile(vals, i / n_tiers)) for i in range(1, n_tiers)]
    tier = np.digitize(density_vol_ijk, qs).astype(np.uint8) + 1   # 1..n_tiers
    out[scar_mask] = tier[scar_mask]
    return out, qs


def quantify(labelmap_ijk: np.ndarray, spacing_xyz, density_vol_ijk=None) -> dict:
    """Scar/cornea volume (mm³), en-face scar area (mm²), fraction, presence, and —
    when `density_vol_ijk` (raw reflectivity, comparable across eyes) is given —
    scar densitometry: mean/median/spread of reflectivity, a density-weighted volume,
    and per-density-tier volumes (the "mix of opacities").

    spacing_xyz: (sp_i, sp_j, sp_k) in mm for the (i,j,k) axes.
    """
    sp = [float(s) for s in spacing_xyz[:3]]
    voxel_mm3 = sp[0] * sp[1] * sp[2]
    scar = labelmap_ijk == SCAR
    cornea_tissue = (labelmap_ijk == CORNEA) | scar
    scar_vox = int(scar.sum())
    cornea_vox = int(cornea_tissue.sum())
    present = scar_vox > 0
    # en-face area: project scar along the depth axis; each footprint pixel spans
    # the in-plane area of the OTHER two axes.
    depth = _depth_axis(cornea_tissue)
    plane_axes = [a for a in range(3) if a != depth]
    pixel_mm2 = sp[plane_axes[0]] * sp[plane_axes[1]]
    scar_area_mm2 = round(float(scar.any(axis=depth).sum()) * pixel_mm2, 5) if present else 0.0
    metrics = {
        "scar_present": present,
        "scar_voxels": scar_vox,
        "scar_volume_mm3": round(scar_vox * voxel_mm3, 6),
        "scar_area_mm2": scar_area_mm2,
        "cornea_voxels": cornea_vox,
        "cornea_volume_mm3": round(cornea_vox * voxel_mm3, 6),
        "scar_fraction_of_cornea": round(scar_vox / cornea_vox, 4) if cornea_vox else 0.0,
        "depth_axis": depth,
        "spacing_mm": sp,
    }
    if present:
        coords = np.argwhere(scar)
        metrics["scar_bounds_ijk"] = {"min": coords.min(0).tolist(), "max": coords.max(0).tolist()}
        # continuity / cross-plane sanity: a coherent scar is few 3D components,
        # most of its volume in the largest one (so all three views show one object).
        lbl, n = ndimage.label(scar)
        sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1)) if n else []
        metrics["scar_components"] = int(n)
        metrics["largest_component_fraction"] = round(float(max(sizes)) / scar_vox, 3) if n else 0.0
        if density_vol_ijk is not None:
            d = density_vol_ijk[scar].astype(np.float32)
            tiers, _ = density_tiers(scar, density_vol_ijk, n_tiers=3)
            metrics["scar_density"] = {
                "mean": round(float(d.mean()), 2), "median": round(float(np.median(d)), 2),
                "std": round(float(d.std()), 2),
                "p10": round(float(np.percentile(d, 10)), 2),
                "p90": round(float(np.percentile(d, 90)), 2),
                # density-weighted volume = ∫ reflectivity dV (opacity burden, mm³·units)
                "weighted_volume_mm3u": round(float(d.sum()) * voxel_mm3, 4),
                "tier_volume_mm3": [round(int((tiers == t).sum()) * voxel_mm3, 5) for t in (1, 2, 3)],
            }
    return metrics
