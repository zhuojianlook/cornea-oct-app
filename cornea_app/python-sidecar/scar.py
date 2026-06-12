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

def hyper_reflective_mask(vol_ijk: np.ndarray, labelmap_ijk: np.ndarray,
                          percentile: float = 88.0, erode_surface: int = 6,
                          smooth: float = 2.5) -> np.ndarray:
    """The bright (hyper-reflective) voxels inside the cornea — scar candidates
    before morphology. Erodes the epithelium/Bowman's/endothelium rind along the
    depth axis, smooths, and keeps the brightest `100−percentile`% of the stroma.
    Used both by the auto detector and to constrain a SAM2 click region to scar."""
    cornea = (labelmap_ijk == CORNEA) | (labelmap_ijk == SCAR)
    if not cornea.any():
        return np.zeros(labelmap_ijk.shape, bool)
    # Trim the reflective rind ALONG THE DEPTH (A-scan) axis only — Bowman's and the
    # endothelium are offset in depth, so eroding across slices would needlessly drop
    # edge slices. Anisotropic erosion keeps every slice's lateral extent.
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
