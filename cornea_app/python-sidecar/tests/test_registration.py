"""Unit tests for registration.py (SimpleITK repeat-scan alignment for consensus).

Covers the public API:
  * align_transform   — rigid-only cornea-masked, best-of-identity guard, mode∈{rigid,identity}
  * align_transform_v2 — rigid→affine→denser BSpline cascade, each stage guarded on cornea Dice
  * _cornea_dice_iso  — the cornea Dice helper (unit-tested on known masks)
  * identity / _iso / _canon / resample_label / resample_volume — supporting helpers

All volumes are tiny (<=18^3) synthetic labelmaps so SimpleITK stays fast and deterministic.
No network, GPU, SAM2/torch, or real .OCT data is touched.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import SimpleITK as sitk

import registration as reg

# Isotropic OCT-like affine matching conftest's geometry (anisotropic in z).
_AFFINE = np.diag([0.02, 0.02, 0.04, 1.0]).astype(float)


# ── local helpers (not in conftest) ─────────────────────────────────────────
def _band_volume(shape=(18, 18, 18), band=(7, 12), scar=(5, 13, 8, 11, 7, 13),
                 shift=(0, 0, 0)):
    """A tiny OCT-like (volume, label) pair: a bright cornea band (label 1) with a
    brighter scar box (label 2) inside it, optionally rolled by `shift` voxels.

    Returns (vol uint16, lab uint8) both of identical shape. Rolling moves the same
    content, so the moving scan is a pure translation of the fixed one — exactly the
    test-retest offset registration is meant to recover."""
    vol = np.full(shape, 20, np.uint16)
    lab = np.zeros(shape, np.uint8)
    d0, d1 = band
    vol[:, d0:d1, :] = 200
    lab[:, d0:d1, :] = reg.CORNEA_MIN
    f0, f1, sd0, sd1, l0, l1 = scar
    vol[f0:f1, sd0:sd1, l0:l1] = 360
    lab[f0:f1, sd0:sd1, l0:l1] = reg.SCAR
    sf, sd, sl = shift
    if shift != (0, 0, 0):
        vol = np.roll(vol, (sf, sd, sl), axis=(0, 1, 2))
        lab = np.roll(lab, (sf, sd, sl), axis=(0, 1, 2))
    return vol, lab


@pytest.fixture
def reg_pair(tmp_path, write_nifti):
    """Factory: write a (vol, lab) pair to disk as NIfTI, returns (vol_path, lab_path)."""
    n = {"i": 0}

    def _make(vol, lab):
        n["i"] += 1
        i = n["i"]
        vp = write_nifti(vol, tmp_path / f"vol{i}.nii.gz", _AFFINE)
        lp = write_nifti(lab, tmp_path / f"lab{i}.nii.gz", _AFFINE)
        return vp, lp

    return _make


def _dice_of(tx, mvp_lab_path, flab_path):
    """Cornea Dice on the iso reference grid for moving-label `mvp_lab_path` warped by `tx`."""
    flab = reg._read_label(str(flab_path))
    mlab = reg._read_label(str(mvp_lab_path))
    flab_iso = reg._iso(flab, interp=sitk.sitkNearestNeighbor)
    ref = sitk.GetArrayFromImage(flab_iso) >= reg.CORNEA_MIN
    return reg._cornea_dice_iso(mlab, flab_iso, ref, tx)


# ── _cornea_dice_iso: the overlap helper, on known masks ────────────────────
def test_dice_identical_masks_is_one(reg_pair):
    """A label registered against itself with identity → Dice exactly 1.0."""
    vol, lab = _band_volume()
    vp, lp = reg_pair(vol, lab)
    assert _dice_of(reg.identity(), lp, lp) == pytest.approx(1.0)


def test_dice_disjoint_masks_is_zero(reg_pair):
    """Two cornea bands (each with its own enclosed scar) at non-overlapping depths →
    Dice 0.0 under identity."""
    fvol, flab = _band_volume(band=(1, 4), scar=(5, 13, 2, 3, 7, 13))
    mvol, mlab = _band_volume(band=(14, 17), scar=(5, 13, 15, 16, 7, 13))
    _, flp = reg_pair(fvol, flab)
    _, mlp = reg_pair(mvol, mlab)
    assert _dice_of(reg.identity(), mlp, flp) == pytest.approx(0.0)


def test_dice_partial_overlap_in_unit_interval(reg_pair):
    """A cornea band shifted in DEPTH partially overlaps the fixed band → Dice strictly
    between 0 and 1 (a depth shift truly moves the band; a lateral shift would wrap the
    full-width band onto itself)."""
    fvol, flab = _band_volume()
    mvol, mlab = _band_volume(shift=(0, 3, 0))
    _, flp = reg_pair(fvol, flab)
    _, mlp = reg_pair(mvol, mlab)
    d = _dice_of(reg.identity(), mlp, flp)
    assert 0.0 < d < 1.0


def test_dice_empty_moving_label_is_zero(reg_pair):
    """No cornea in the moving label → denominator nonzero (fixed has cornea) → Dice 0.0,
    and an all-empty pair (both denominators 0) → Dice 0.0 (the `if s else 0.0` branch)."""
    fvol, flab = _band_volume()
    _, flp = reg_pair(fvol, flab)
    empty = np.zeros(fvol.shape, np.uint8)
    _, mlp_empty = reg_pair(np.full(fvol.shape, 20, np.uint16), empty)
    assert _dice_of(reg.identity(), mlp_empty, flp) == pytest.approx(0.0)
    # both empty: fixed cornea also gone → s==0 → guarded 0.0
    fempty = np.zeros(fvol.shape, np.uint8)
    _, flp_empty = reg_pair(np.full(fvol.shape, 20, np.uint16), fempty)
    assert _dice_of(reg.identity(), mlp_empty, flp_empty) == pytest.approx(0.0)


# ── identity() ──────────────────────────────────────────────────────────────
def test_identity_is_zero_euler3d():
    """identity() is an Euler3DTransform with all-zero parameters."""
    t = reg.identity()
    assert isinstance(t, sitk.Euler3DTransform)
    assert all(p == 0.0 for p in t.GetParameters())


# ── _canon / _read_vol ──────────────────────────────────────────────────────
def test_read_vol_is_canonicalised(reg_pair):
    """_read_vol zeroes origin and forces an identity direction cosine matrix."""
    vol, lab = _band_volume()
    vp, _ = reg_pair(vol, lab)
    img = reg._read_vol(str(vp))
    assert img.GetOrigin() == (0.0, 0.0, 0.0)
    assert img.GetDirection() == (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)


# ── _iso ────────────────────────────────────────────────────────────────────
def test_iso_makes_spacing_isotropic(reg_pair):
    """_iso resamples the anisotropic OCT grid to isotropic ISO spacing on every axis."""
    vol, lab = _band_volume()
    vp, _ = reg_pair(vol, lab)
    iso = reg._iso(reg._read_vol(str(vp)))
    assert iso.GetSpacing() == pytest.approx((reg.ISO, reg.ISO, reg.ISO))
    # axis that was coarser (0.04) gets MORE samples than the finer (0.02) axes.
    nsz = iso.GetSize()
    assert nsz[2] > nsz[0]  # z had 0.04 spacing -> doubled sample count vs x


# ── align_transform: identical volumes → ~identity, no worsening ─────────────
def test_align_identical_returns_identity_no_worsening(reg_pair):
    """Registering a scan against ITSELF must return identity (cannot beat a perfect
    overlap) and must not worsen the perfect cornea Dice."""
    vol, lab = _band_volume()
    vp, lp = reg_pair(vol, lab)
    d_id = _dice_of(reg.identity(), lp, lp)
    assert d_id == pytest.approx(1.0)

    tx, mode = reg.align_transform(vp, lp, vp, lp)
    assert mode == "identity"
    d_tx = _dice_of(tx, lp, lp)
    # identity transform → params all zero, and Dice not reduced.
    assert all(p == 0.0 for p in tx.GetParameters())
    assert d_tx >= d_id - 1e-9


# ── align_transform: shifted volume → guarded, never worse than identity ─────
def test_align_shifted_mode_is_rigid_or_identity(reg_pair):
    """A translated repeat must be tagged 'rigid' (if it beat identity) or 'identity'
    (if the optimiser could not improve) — never any other mode."""
    fvol, flab = _band_volume()
    mvol, mlab = _band_volume(shift=(0, 2, 1))
    fvp, flp = reg_pair(fvol, flab)
    mvp, mlp = reg_pair(mvol, mlab)
    tx, mode = reg.align_transform(fvp, flp, mvp, mlp)
    assert mode in ("rigid", "identity")


def test_align_shifted_does_not_reduce_cornea_dice(reg_pair):
    """THE core invariant: the returned transform, when applied, never reduces cornea
    Dice below the identity baseline (best-of-identity guard)."""
    fvol, flab = _band_volume()
    mvol, mlab = _band_volume(shift=(0, 2, 2))
    fvp, flp = reg_pair(fvol, flab)
    mvp, mlp = reg_pair(mvol, mlab)

    d_id = _dice_of(reg.identity(), mlp, flp)
    tx, mode = reg.align_transform(fvp, flp, mvp, mlp)
    d_tx = _dice_of(tx, mlp, flp)
    assert d_tx >= d_id - 1e-9
    # If it returned 'rigid' it MUST have strictly beaten identity (the guard's contract).
    if mode == "rigid":
        assert d_tx > d_id


def test_align_runs_fast(reg_pair):
    """align_transform on tiny volumes completes well under the 2s budget."""
    fvol, flab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10))
    mvol, mlab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10), shift=(0, 1, 1))
    fvp, flp = reg_pair(fvol, flab)
    mvp, mlp = reg_pair(mvol, mlab)
    t0 = time.time()
    reg.align_transform(fvp, flp, mvp, mlp)
    assert time.time() - t0 < 2.0


# ── align_transform_v2: cascade, mode contract, anti-degrade guard ──────────
def test_align_v2_identical_returns_identity(reg_pair):
    """v2 against itself: the base stage is identity (rigid cannot beat a perfect overlap);
    a '+bspline' refinement may be appended but cornea Dice must not drop below 1.0-eps."""
    vol, lab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                            scar=(4, 10, 6, 8, 5, 10))
    vp, lp = reg_pair(vol, lab)
    tx, mode = reg.align_transform_v2(vp, lp, vp, lp)
    assert mode.split("+")[0] == "identity"
    d_tx = _dice_of(tx, lp, lp)
    # BSpline is kept only if it does not degrade beyond the -0.002 tolerance.
    assert d_tx >= 1.0 - 0.002 - 1e-6


def test_align_v2_mode_contract(reg_pair):
    """v2 mode is a '+'-joined chain whose head is a valid base and whose tail tokens are
    only the documented refinement stages."""
    fvol, flab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10))
    mvol, mlab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10), shift=(0, 2, 1))
    fvp, flp = reg_pair(fvol, flab)
    mvp, mlp = reg_pair(mvol, mlab)
    tx, mode = reg.align_transform_v2(fvp, flp, mvp, mlp)
    tokens = mode.split("+")
    assert tokens[0] in ("rigid", "affine", "identity")
    assert all(t in ("rigid", "affine", "bspline", "identity") for t in tokens)


def test_align_v2_does_not_reduce_cornea_dice(reg_pair):
    """v2's guarded cascade must not push cornea Dice below identity (each stage is kept
    only on improvement, BSpline only within a -0.002 tolerance, so net >= identity-tol)."""
    fvol, flab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10))
    mvol, mlab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10), shift=(0, 1, 1))
    fvp, flp = reg_pair(fvol, flab)
    mvp, mlp = reg_pair(mvol, mlab)
    d_id = _dice_of(reg.identity(), mlp, flp)
    tx, _mode = reg.align_transform_v2(fvp, flp, mvp, mlp)
    d_tx = _dice_of(tx, mlp, flp)
    assert d_tx >= d_id - 0.002 - 1e-6


def test_align_v2_runs_fast(reg_pair):
    """The full v2 cascade on tiny volumes stays under the 2s budget."""
    fvol, flab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10))
    mvol, mlab = _band_volume(shape=(14, 14, 14), band=(5, 9),
                              scar=(4, 10, 6, 8, 5, 10), shift=(0, 1, 1))
    fvp, flp = reg_pair(fvol, flab)
    mvp, mlp = reg_pair(mvol, mlab)
    t0 = time.time()
    reg.align_transform_v2(fvp, flp, mvp, mlp)
    assert time.time() - t0 < 2.0


# ── resample_label / resample_volume ────────────────────────────────────────
def test_resample_label_identity_preserves_labels(reg_pair):
    """resample_label with identity onto the same grid returns the same label array
    (nearest-neighbour preserves the discrete 0/1/2 classes, shape unchanged)."""
    vol, lab = _band_volume()
    vp, lp = reg_pair(vol, lab)
    out = reg.resample_label(lp, vp, reg.identity())
    assert out.shape == lab.shape
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) == {0, reg.CORNEA_MIN, reg.SCAR}
    # identity resample onto its own grid is exact for these axis-aligned masks.
    assert np.array_equal(out, sitk.GetArrayFromImage(reg._read_label(str(lp))))


def test_resample_volume_identity_returns_image(reg_pair):
    """resample_volume returns a sitk.Image on the fixed grid with the fixed's size."""
    vol, lab = _band_volume()
    vp, lp = reg_pair(vol, lab)
    fixed = reg._read_vol(str(vp))
    out = reg.resample_volume(vp, vp, reg.identity())
    assert isinstance(out, sitk.Image)
    assert out.GetSize() == fixed.GetSize()
    assert out.GetSpacing() == pytest.approx(fixed.GetSpacing())


def test_resample_label_into_different_fixed_grid(reg_pair):
    """A moving label resampled into a SMALLER fixed grid takes the fixed grid's shape
    (the transform pulls moving→fixed; output lives on the fixed grid)."""
    fvol, flab = _band_volume(shape=(12, 12, 12), band=(4, 8),
                              scar=(3, 9, 5, 7, 4, 8))
    mvol, mlab = _band_volume(shape=(18, 18, 18))
    fvp, _ = reg_pair(fvol, flab)
    _, mlp = reg_pair(mvol, mlab)
    out = reg.resample_label(mlp, fvp, reg.identity())
    assert out.shape == (12, 12, 12)
