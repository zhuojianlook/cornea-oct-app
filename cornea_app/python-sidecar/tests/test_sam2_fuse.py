"""Unit tests for sam2_segment._fuse_votes — the per-plane vote fusion + CORE-GROW peripheral recovery.

Geometry: arrays are (lateral, depth, frame); the cornea is a thin band along DEPTH. `votes` is the per-voxel
count of planes (0..n) that segmented cornea. These tests exercise the fusion logic WITHOUT SAM2/torch (the
heavy imports in sam2_segment are lazy, inside functions)."""
import numpy as np
import sam2_segment as S

NL, ND, NF = 40, 80, 40
BAND = slice(30, 40)          # cornea depth band (thin → depth = the fill axis)


def _band_vox(label):
    return int(label[:, BAND, :].sum())


def test_fuse_clean_scan_unchanged():
    # All 3 planes agree on the band → no growth needed, no spurious voxels outside the band.
    votes = np.zeros((NL, ND, NF), np.uint8)
    votes[:, BAND, :] = 3
    label, cg, _tf = S._fuse_votes(votes, vote=2)
    assert label[:, BAND, :].all()                       # full band kept
    assert int(label[:, :25, :].sum()) == 0             # nothing above
    assert int(label[:, 45:, :].sum()) == 0             # nothing below


def test_fuse_recovers_dropped_peripheral_run():
    # 2 planes cover frames 0-30; only 1 plane covers frames 31-39 (a dropped peripheral run, CS020-style).
    votes = np.zeros((NL, ND, NF), np.uint8)
    votes[:, BAND, :31] = 2
    votes[:, BAND, 31:] = 1
    core_only, _, _ = S._fuse_votes(votes, vote=99)      # impossible vote → pure core proxy (none)
    label, cg, _tf = S._fuse_votes(votes, vote=2)
    assert cg["applied"]
    assert int(label[:, BAND, 35].sum()) > 0            # frame 35 recovered
    assert int(label[:, BAND, 39].sum()) > 0            # last frame recovered


def test_fuse_rejects_deep_single_plane_leak():
    # Core band + one plane that leaks a DEEP sheet connected to the band by a thin bridge. thickness-fill
    # would balloon it (fill every column top→deep), so the finalized-mass cap must reject the whole component.
    votes = np.zeros((NL, ND, NF), np.uint8)
    votes[:, BAND, :] = 2                                # core band
    votes[:, 60:65, :] = 1                              # deep sheet (1 plane)
    votes[20, 40:60, 20] = 1                            # thin bridge band→sheet at one column
    label, cg, _tf = S._fuse_votes(votes, vote=2)
    # the deep sheet must NOT be admitted as cornea
    assert int(label[:, 60:65, :].sum()) < 0.2 * _band_vox(label)
    assert int(label[:, 50:58, :].sum()) == 0          # no thickness-fill bridge into the gap


def test_fuse_rejects_contiguous_thick_flood():
    # One plane over-segments a THICK slab contiguously below the band (doubling thickness). Must be rejected.
    votes = np.zeros((NL, ND, NF), np.uint8)
    votes[:, BAND, :] = 2                                # core band (10 deep)
    votes[:, 40:62, :] = 1                              # contiguous thick slab (22 deep) from one plane
    label, cg, _tf = S._fuse_votes(votes, vote=2)
    assert int(label[:, 45:62, :].sum()) < 0.3 * _band_vox(label)   # slab not adopted


def test_fuse_unrelated_blob_does_not_veto_recovery():
    # Main band needs a peripheral recovery (component A); a SEPARATE spurious 2-vote blob with a big 1-vote
    # halo (component B) sits elsewhere. A global cap would let B's halo veto the whole recovery; the
    # per-component test must still recover A.
    votes = np.zeros((NL, ND, NF), np.uint8)
    votes[:, BAND, :31] = 2                              # main band core, frames 0-30
    votes[:, BAND, 31:] = 1                              # peripheral run (component A)
    votes[0:5, 5:8, :] = 2                              # small spurious 2-vote blob (component B core)
    votes[0:18, 4:26, :] = np.maximum(votes[0:18, 4:26, :], 1)   # big 1-vote halo around B
    label, cg, _tf = S._fuse_votes(votes, vote=2)
    assert int(label[:, BAND, 35].sum()) > 0            # recovery survived the unrelated blob


def test_fuse_vote1_no_grow():
    # vote=1 (union segmentation): no core-grow attempted, just finalize.
    votes = np.zeros((NL, ND, NF), np.uint8)
    votes[:, BAND, :] = 1
    label, cg, _tf = S._fuse_votes(votes, vote=1)
    assert not cg["applied"]
    assert label[:, BAND, :].all()
