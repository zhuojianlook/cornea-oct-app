"""Unit tests for scar.py — the density-tier API and the hysteresis scar detector.

Focus (per the module's documented invariants):
  * density_tiers_absolute: cornea-median-anchored tiers, ascending by intensity,
    0 outside the scar, and the relative-quantile fallback when no cornea reference.
  * detect_scar_hysteresis: on a synthetic bright blob inside a stromal band, the
    detected mask covers the blob, is a single connected component, excludes
    background, and grows monotonically as the lower cut drops (larger gap).

Everything uses tiny synthetic uint16 arrays (<=32 per axis) and the detector
default morphology (erode/smooth/min_voxels) is dialed down so a small volume is
not eroded away. No SAM2 / torch / network / real data.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

import scar


# ── local helpers (kept out of conftest per task constraints) ───────────────

def _blob_volume(F=16, D=24, L=16, band=(8, 18), blob_intensity=320,
                 stroma_intensity=80, bg=20,
                 blob=(5, 11, 11, 15, 5, 11)):
    """Synthetic OCT-like volume: a dim uniform stromal band along depth with one
    clearly-brightest scar blob inside it, plus a cornea labelmap (0=bg,1=cornea).
    Returns (vol_uint16, labelmap_uint8, blob_bool_mask)."""
    vol = np.full((F, D, L), bg, np.uint16)
    vol[:, band[0]:band[1], :] = stroma_intensity
    f0, f1, d0, d1, l0, l1 = blob
    vol[f0:f1, d0:d1, l0:l1] = blob_intensity
    lab = np.zeros((F, D, L), np.uint8)
    lab[:, band[0]:band[1], :] = scar.CORNEA
    bmask = np.zeros((F, D, L), bool)
    bmask[f0:f1, d0:d1, l0:l1] = True
    return vol, lab, bmask


def _run_hyst(vol, lab, gap=12.0, phi=92.0):
    """detect_scar_hysteresis with small-volume-friendly morphology params."""
    return scar.detect_scar_hysteresis(
        vol, lab, phi_percentile=phi, gap=gap,
        erode_surface=1, smooth=0.6, min_voxels=5, open_iter=0, close_iter=0)


# ── density_tiers_absolute ──────────────────────────────────────────────────

def test_density_tiers_absolute_three_ascending_tiers():
    """3 clear brightness levels inside the scar map to ascending tiers 1/2/3,
    cut at the cornea-median × (1.6, 2.4) defaults."""
    F, D, L = 8, 12, 12
    v = np.zeros((F, D, L), np.uint16)
    cornea = np.zeros((F, D, L), bool)
    cornea[:, 2:8, :] = True
    v[cornea] = 100  # normal cornea reflectivity → reference median = 100

    scar_mask = np.zeros((F, D, L), bool)
    lo = np.s_[:, 3:4, 0:4]
    mid = np.s_[:, 3:4, 4:8]
    hi = np.s_[:, 3:4, 8:12]
    for s in (lo, mid, hi):
        scar_mask[s] = True
    # 1.2× / 2.0× / 3.0× of the ref → tier 1 / 2 / 3
    v[lo] = 120
    v[mid] = 200
    v[hi] = 300

    cornea_full = cornea | scar_mask
    tiers, cutoffs = scar.density_tiers_absolute(scar_mask, v, cornea_full)

    # cutoffs are ref × ratios (ref median = 100, ratios = (1.6, 2.4))
    assert cutoffs == pytest.approx([160.0, 240.0])
    # exactly the three tiers appear inside the scar, ascending by intensity
    assert sorted(np.unique(tiers[scar_mask]).tolist()) == [1, 2, 3]
    assert np.unique(tiers[lo]).tolist() == [1]
    assert np.unique(tiers[mid]).tolist() == [2]
    assert np.unique(tiers[hi]).tolist() == [3]
    # nothing tiered outside the scar
    assert (tiers[~scar_mask] == 0).all()
    assert tiers.dtype == np.uint8


def test_density_tiers_absolute_monotone_in_intensity():
    """Tier assignment is monotone non-decreasing in voxel intensity: a brighter
    scar voxel never lands in a lower tier than a dimmer one."""
    F, D, L = 4, 8, 8
    v = np.zeros((F, D, L), np.uint16)
    cornea = np.zeros((F, D, L), bool)
    cornea[:, 1:7, :] = True
    v[cornea] = 50  # ref median = 50 → cutoffs 80, 120

    scar_mask = np.zeros((F, D, L), bool)
    scar_mask[:, 3:5, :] = True
    # ramp of intensities across the lateral axis inside the scar slab
    ramp = np.linspace(40, 300, L).astype(np.uint16)
    v[:, 3:5, :] = ramp[None, None, :]

    tiers, cutoffs = scar.density_tiers_absolute(scar_mask, v, cornea | scar_mask)
    assert cutoffs == pytest.approx([80.0, 120.0])
    # within the slab, intensity and tier rise together
    lat_intensity = v[0, 3, :].astype(float)
    lat_tier = tiers[0, 3, :].astype(int)
    order = np.argsort(lat_intensity)
    assert np.all(np.diff(lat_tier[order]) >= 0)


def test_density_tiers_absolute_uses_cornea_only_for_reference():
    """The reference median ignores scar voxels (cornea_only = cornea & ~scar): a
    very bright scar must NOT pull the reference up and dilute its own tiers."""
    F, D, L = 4, 8, 8
    v = np.zeros((F, D, L), np.uint16)
    cornea = np.zeros((F, D, L), bool)
    cornea[:, 1:7, :] = True
    v[cornea] = 100  # normal cornea
    scar_mask = np.zeros((F, D, L), bool)
    scar_mask[:, 3:5, :] = True
    v[scar_mask] = 5000  # extreme bright scar; would wreck a naive whole-cornea ref

    _, cutoffs = scar.density_tiers_absolute(scar_mask, v, cornea | scar_mask)
    # ref stays 100 (scar excluded) → cutoffs 160, 240, NOT scaled by the 5000s
    assert cutoffs == pytest.approx([160.0, 240.0])


def test_density_tiers_absolute_custom_ratios_change_cutoffs():
    """Passing custom ratios scales the cutoffs by ref × ratio and yields
    len(ratios)+1 tiers."""
    F, D, L = 3, 6, 6
    v = np.zeros((F, D, L), np.uint16)
    cornea = np.zeros((F, D, L), bool)
    cornea[:, 1:5, :] = True
    v[cornea] = 100
    scar_mask = np.zeros((F, D, L), bool)
    scar_mask[:, 2:4, :] = True
    v[:, 2:4, 0:2] = 150
    v[:, 2:4, 2:4] = 250
    v[:, 2:4, 4:6] = 450

    tiers, cutoffs = scar.density_tiers_absolute(
        scar_mask, v, cornea | scar_mask, ratios=(2.0, 4.0))
    assert cutoffs == pytest.approx([200.0, 400.0])
    # 150<200 →1, 250 in [200,400) →2, 450>=400 →3
    assert np.unique(tiers[:, 2:4, 0:2]).tolist() == [1]
    assert np.unique(tiers[:, 2:4, 2:4]).tolist() == [2]
    assert np.unique(tiers[:, 2:4, 4:6]).tolist() == [3]


def test_density_tiers_absolute_empty_scar_returns_empty():
    """No scar → all-zero tier volume and empty cutoffs."""
    F, D, L = 4, 6, 6
    v = np.full((F, D, L), 100, np.uint16)
    cornea = np.ones((F, D, L), bool)
    empty = np.zeros((F, D, L), bool)
    tiers, cutoffs = scar.density_tiers_absolute(empty, v, cornea)
    assert tiers.sum() == 0
    assert cutoffs == []


def test_density_tiers_absolute_fallback_to_quantiles_without_reference():
    """When there is no cornea tissue outside the scar (ref unusable), it falls
    back to intra-scar quantile tiers (still 1..3, 0 elsewhere)."""
    F, D, L = 4, 6, 6
    scar_mask = np.zeros((F, D, L), bool)
    scar_mask[:, 2:4, :] = True
    v = np.zeros((F, D, L), np.uint16)
    # spread of intensities so quantile splitting yields 3 distinct tiers
    v[scar_mask] = (np.arange(scar_mask.sum()) % 3) * 100 + 50
    cornea_only_empty = scar_mask.copy()  # cornea_mask == scar → cornea_only empty

    tiers, cutoffs = scar.density_tiers_absolute(scar_mask, v, cornea_only_empty)
    assert sorted(np.unique(tiers[scar_mask]).tolist()) == [1, 2, 3]
    assert (tiers[~scar_mask] == 0).all()
    # fallback cutoffs are intra-scar quantiles, NOT the fixed ref×ratio values
    assert len(cutoffs) == 2
    assert cutoffs[0] < cutoffs[1]


# ── detect_scar_hysteresis ──────────────────────────────────────────────────

def test_hysteresis_covers_blob_connected_and_excludes_background():
    """On a bright blob inside a stromal band: the mask fully covers the blob,
    is a single connected component, lies inside the cornea, and touches no bg."""
    vol, lab, blob = _blob_volume()
    mask = _run_hyst(vol, lab, gap=12.0)

    assert mask.shape == vol.shape
    assert mask.dtype == bool
    assert mask.any()
    # covers the bright blob entirely
    assert (blob & ~mask).sum() == 0
    # single coherent 3D object (cross-plane sanity by construction)
    _, n = ndimage.label(mask)
    assert n == 1
    # entirely inside the cornea, never in background
    cornea = lab == scar.CORNEA
    assert (mask & ~cornea).sum() == 0
    assert (mask & (lab == scar.BG)).sum() == 0


def test_hysteresis_monotone_lower_cut_grows_mask():
    """Growing to a LOWER cut (larger gap → lower tlo) yields a SUPERSET mask:
    monotone non-decreasing, never flips voxels off."""
    vol, lab, _ = _blob_volume()
    gaps = [4.0, 8.0, 12.0, 20.0]
    masks = [_run_hyst(vol, lab, gap=g) for g in gaps]
    sizes = [int(m.sum()) for m in masks]
    # non-decreasing in size
    assert sizes == sorted(sizes)
    # and actually nested: each larger-gap mask is a superset of the smaller
    for prev, cur in zip(masks, masks[1:]):
        assert (prev & ~cur).sum() == 0


def test_hysteresis_seed_must_exist_higher_percentile_smaller_or_equal():
    """Raising phi_percentile (a stricter seed) cannot grow the mask: the high-
    percentile result is a subset of (or equal to) the lower-percentile result."""
    vol, lab, blob = _blob_volume()
    loose = _run_hyst(vol, lab, gap=12.0, phi=85.0)
    strict = _run_hyst(vol, lab, gap=12.0, phi=95.0)
    assert (strict & ~loose).sum() == 0
    # the strict seed still catches the unambiguously-bright blob
    assert (blob & ~strict).sum() == 0


def test_hysteresis_no_cornea_returns_empty():
    """No cornea label → empty mask of the right shape (no crash, no false scar)."""
    vol, _, _ = _blob_volume()
    empty_lab = np.zeros(vol.shape, np.uint8)
    mask = scar.detect_scar_hysteresis(
        vol, empty_lab, erode_surface=1, smooth=0.6, min_voxels=5)
    assert mask.shape == vol.shape
    assert not mask.any()


def test_hysteresis_no_scar_when_stroma_is_flat():
    """A uniform stromal band with NO bright blob: min_voxels pruning + the lack
    of a distinct bright core means the detector does not invent a coherent scar
    blob covering the whole flat stroma when the seed/grow span the whole ROI."""
    F, D, L = 16, 24, 16
    vol = np.full((F, D, L), 20, np.uint16)
    vol[:, 8:18, :] = 100  # perfectly flat stroma, no scar
    lab = np.zeros((F, D, L), np.uint8)
    lab[:, 8:18, :] = scar.CORNEA
    # high min_voxels relative to volume: a flat percentile cut grabs a thin shell
    # that should be pruned, so no spurious scar survives.
    mask = scar.detect_scar_hysteresis(
        vol, lab, phi_percentile=99.0, gap=1.0,
        erode_surface=1, smooth=0.6, min_voxels=100000,
        open_iter=0, close_iter=0)
    assert not mask.any()


def test_hysteresis_min_voxels_prunes_tiny_components():
    """A scar smaller than min_voxels is dropped (the candidate must be a coherent
    3D volume of sufficient size)."""
    vol, lab, blob = _blob_volume()
    big_min = int(blob.sum()) * 1000
    mask = scar.detect_scar_hysteresis(
        vol, lab, phi_percentile=92.0, gap=12.0,
        erode_surface=1, smooth=0.6, min_voxels=big_min,
        open_iter=0, close_iter=0)
    assert not mask.any()


# ── regularize_cornea: reconstruct a SMOOTH cornea band (scar-safe) ──
def test_regularize_cornea_clips_spike_keeps_band_and_scar():
    import scar as S
    nl, nd, nf = 40, 64, 30          # (lateral, depth, frame); depth = axis 1
    L = np.zeros((nl, nd, nf), np.uint8)
    L[:, 24:34, :] = S.CORNEA        # smooth cornea band, depth 24..33
    L[10:14, 24:34, 12:18] = S.SCAR  # a scar blob INSIDE the band
    L[20, 34:52, 15] = S.CORNEA      # a thin downward SPIKE at one A-line (the artifact)
    out = S.regularize_cornea(L)
    # the spike beyond the smoothed posterior is removed entirely (snapped to the shell, no margin)
    assert (out[20, 34:52, 15] == 0).all()
    # the smooth band is preserved/filled
    assert (out[:, 24:34, :] >= S.CORNEA).all()
    # scar is never altered
    assert int((out == S.SCAR).sum()) == int((L == S.SCAR).sum())
    assert (out[10:14, 24:34, 12:18] == S.SCAR).all()


def test_regularize_cornea_fills_posterior_notch():
    import scar as S
    nl, nd, nf = 40, 64, 30
    L = np.zeros((nl, nd, nf), np.uint8)
    L[:, 24:34, :] = S.CORNEA        # smooth band depth 24..33
    L[:, 30:34, 14:17] = 0           # carve a posterior NOTCH (under-segmentation) at a few frames
    out = S.regularize_cornea(L)
    # the notch is filled back up to the smoothed posterior surface (trim-only could not do this)
    assert (out[:, 24:34, 14:17] == S.CORNEA).all()


def test_regularize_cornea_noop_on_smooth_band():
    import scar as S
    L = np.zeros((30, 50, 20), np.uint8)
    L[:, 20:30, :] = S.CORNEA        # already smooth → reconstruct returns the same band
    out = S.regularize_cornea(L)
    assert int((out == S.CORNEA).sum()) == int((L == S.CORNEA).sum())


def test_regularize_cornea_preserves_edge_trend():
    # The cornea deepens toward the peripheral frames; the surface smoothing must PRESERVE that trend at the
    # first/last frames (odd-reflect/linear-extrapolation pad), not flatten it — a plain reflect boundary
    # pulled the band UP ~10-15 vox at the start/end frames (the axial-edge displacement). Regression for that.
    import scar as S
    nl, nd, nf = 60, 220, 60
    L = np.zeros((nl, nd, nf), np.uint8)
    f = np.arange(nf, dtype=float)
    centre = (90.0 + 0.045 * (f - nf / 2) ** 2).astype(int)   # cornea centre deepens toward both frame edges
    th = 30
    for fr in range(nf):
        L[:, centre[fr] - th // 2: centre[fr] + th // 2, fr] = S.CORNEA
    out = S.regularize_cornea(L)

    def post(b, fr):
        col = np.where(b[30, :, fr] >= 1)[0]
        return int(col.max()) if col.size else -1
    # the first/last frames' posterior must follow the (deeper) input, not be flattened up toward the centre
    for fr in (0, nf - 1):
        assert abs(post(out, fr) - post(L, fr)) <= 4


def test_regularize_cornea_never_orphans_scar():
    # A deep spike whose tip is SCAR: despiking trims the cornea spike, but the band must be
    # clamped to enclose the scar so scar stays connected to cornea (not floating in background).
    import scar as S
    from scipy import ndimage
    nl, nd, nf = 60, 200, 60
    L = np.zeros((nl, nd, nf), np.uint8)
    L[:, 90:106, :] = S.CORNEA            # smooth band depth 90..105
    L[30, 106:140, 30] = S.CORNEA         # a deep cornea spike at one A-line
    L[30, 135:140, 30] = S.SCAR           # scar at the spike tip (deep excursion)
    out = S.regularize_cornea(L)
    assert int((out == S.SCAR).sum()) == int((L == S.SCAR).sum())   # scar count untouched
    lbl, _ = ndimage.label(out >= S.CORNEA)                          # cornea ∪ scar connectivity
    scar_comps = set(np.unique(lbl[out == S.SCAR])) - {0}
    main_comp = int(np.argmax(np.bincount(lbl[lbl > 0].ravel()))) if (lbl > 0).any() else 0
    assert scar_comps == {main_comp}     # scar shares the single main component → never orphaned


def test_regularize_cornea_keeps_genuine_thick_minority():
    # A genuinely THICK region that is the spatial MINORITY must NOT be flattened to the thinner majority — a
    # SYMMETRIC outlier-reject would collapse it, deleting real cornea. The posterior model is ASYMMETRIC:
    # only too-THIN (under-segmented) A-lines are overridden; a reliable thick posterior is preserved.
    import scar as S
    nl, nd, nf = 120, 240, 40
    L = np.zeros((nl, nd, nf), np.uint8)
    L[:, 40:100, :] = S.CORNEA            # baseline thickness 60 (depth 40..99)
    L[50:70, 40:160, :] = S.CORNEA        # a thick block (thickness 120, width 20 > despike) — minority in win
    out = S.regularize_cornea(L)
    col = np.where(out[60, :, nf // 2] >= 1)[0]    # posterior at the thick block
    assert col.max() >= 150                # retained near its true depth (~159), NOT pulled up to ~99


def test_regularize_cornea_maintains_faint_lower_border():
    # Where the posterior edge is faint/absent SAM2 under-segments (a NOTCH). The lower border must be
    # MAINTAINED by carrying the corneal thickness in from the neighbouring good A-lines, not bitten into.
    import scar as S
    nl, nd, nf = 60, 240, 40
    L = np.zeros((nl, nd, nf), np.uint8)
    L[:, 40:120, :] = S.CORNEA            # good band, thickness 80 (depth 40..119)
    L[20:32, 100:120, :] = 0              # a faint-edge notch (thickness only 60) — the spatial minority
    out = S.regularize_cornea(L)
    col = np.where(out[26, :, nf // 2] >= 1)[0]    # posterior in the notch region
    assert col.max() >= 114                # filled back down to ~the local thickness (~119), not left short


def test_regularize_cornea_removes_anterior_streak_bump():
    # Specular-streak bump: the anterior is labelled ~20 voxels too SHALLOW over a compact ~13-wide
    # central region (cornea marked ABOVE the true epithelium). The despike window must EXCEED the
    # bump's en-face width to reject it, without flattening the broad dome (regression for v0.0.103).
    import scar as S
    nl, nd, nf = 60, 120, 60
    L = np.zeros((nl, nd, nf), np.uint8)
    L[:, 50:66, :] = S.CORNEA              # smooth band, true anterior = 50
    L[24:37, 30:50, 24:37] = S.CORNEA      # 13×13 anterior bump up to depth 30 at the centre
    out = S.regularize_cornea(L)           # default despike=21 > 13 → bump rejected
    centre_ant = int(np.where(out[30, :, 30] == S.CORNEA)[0].min())
    assert centre_ant >= 46                # anterior no longer pulled up to ~30 (dome ~50 preserved)
    # away from the streak the surface is unchanged
    assert int(np.where(out[5, :, 5] == S.CORNEA)[0].min()) == 50


def test_regularize_cornea_leaves_wide_clear_cavity_empty():
    # A genuine WIDE enclosed clear region (e.g. a central full-thickness defect) must NOT be
    # fabricated into solid cornea — only thin streak-like gaps are bridged.
    import scar as S
    nl, nd, nf = 60, 80, 60
    L = np.zeros((nl, nd, nf), np.uint8)
    L[:, 30:46, :] = S.CORNEA            # solid band
    L[24:36, :, 24:36] = 0              # carve a 12×12 through-depth clear cavity (en-face)
    out = S.regularize_cornea(L)
    centre = out[28:32, :, 28:32]        # cavity core
    assert int((centre == S.CORNEA).sum()) == 0   # core stays background, not fabricated cornea
