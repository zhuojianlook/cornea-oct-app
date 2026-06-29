"""Unit tests for postprocess.py — the in-process (pure-numpy) preview renderer.

Targets:
  * postprocess._spacing            (pure: affine -> per-axis voxel size)
  * postprocess.render_seg_previews (headless PNG + manifest render; density_from_self flag)
  * postprocess.render_context_previews
  * scar.density_tiers_absolute     (the colour-tier helper render_seg_previews relies on)

All tests run headless on tiny synthetic volumes: no GPU, no matplotlib, no SAM2/torch,
no network, no real .OCT data. PNGs are written into tmp dirs only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

import postprocess
import scar as scar_mod

BG, CORNEA, SCAR = 0, 1, 2


# --------------------------------------------------------------------------- #
# local helpers
# --------------------------------------------------------------------------- #
def _write_vol(arr, path, affine=None):
    """Save an IJK volume as a NIfTI and return the Path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.diag([0.02, 0.02, 0.04, 1.0]).astype(float)
    nib.save(nib.Nifti1Image(np.asarray(arr), affine), str(path))
    return path


def _read_manifest(out_dir: Path) -> dict:
    mpath = out_dir / "preview_manifest.json"
    assert mpath.exists(), "preview_manifest.json was not written"
    return json.loads(mpath.read_text())


def _labelmap(shape=(8, 24, 20)):
    """3-class labelmap: a cornea band + a scar blob inside it (matches conftest layout)."""
    lab = np.zeros(shape, np.uint8)
    lab[:, 10:14, :] = CORNEA
    lab[2:6, 11:13, 6:12] = SCAR
    return lab


def _density_vol(shape=(8, 24, 20)):
    """Reflectivity volume: dim background, mid cornea, and a scar blob whose voxels
    span a range of brightness so the absolute tiers split into >1 tier."""
    vol = np.full(shape, 20, np.uint16)
    vol[:, 10:14, :] = 100                     # normal cornea reference ~100
    # scar blob (matches _labelmap's scar box) graded so it covers diffuse..dense tiers
    vol[2:6, 11:13, 6:9] = 120                 # ~1.2x  -> diffuse
    vol[2:6, 11:13, 9:11] = 200                # ~2.0x  -> moderate
    vol[2:6, 11:13, 11:12] = 300               # ~3.0x  -> dense
    return vol


# --------------------------------------------------------------------------- #
# _spacing
# --------------------------------------------------------------------------- #
def test_spacing_from_diagonal_affine():
    aff = np.diag([0.02, 0.03, 0.04, 1.0]).astype(float)
    sp = postprocess._spacing(aff)
    assert sp == pytest.approx([0.02, 0.03, 0.04])
    assert all(isinstance(v, float) for v in sp)


def test_spacing_is_column_norm_not_just_diagonal():
    # A rotated (non-diagonal) affine: spacing is the L2 norm of each direction column,
    # so a pure rotation of an isotropic 0.5mm grid must still report 0.5 on every axis.
    theta = np.deg2rad(30.0)
    rot = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                    [np.sin(theta),  np.cos(theta), 0.0],
                    [0.0,            0.0,           1.0]])
    aff = np.eye(4)
    aff[:3, :3] = rot * 0.5
    sp = postprocess._spacing(aff)
    assert sp == pytest.approx([0.5, 0.5, 0.5])


# --------------------------------------------------------------------------- #
# render_seg_previews — basic headless render
# --------------------------------------------------------------------------- #
def test_render_seg_previews_writes_pngs_and_manifest(tmp_path):
    vol = _density_vol()
    lab = _labelmap()
    nifti = _write_vol(vol, tmp_path / "vol.nii.gz")
    out = tmp_path / "out"

    cornea_vox = postprocess.render_seg_previews(nifti, lab, out)

    # return value is the cornea voxel count
    assert cornea_vox == int((lab == CORNEA).sum())
    assert cornea_vox > 0

    pngs = sorted(out.glob("segmentation_*.png"))
    assert pngs, "no segmentation PNGs were written"
    # filenames follow segmentation_{orientation}_{index:04d}.png in all 3 orientations
    orientations = {p.name.split("_")[1] for p in pngs}
    assert orientations == {"axial", "coronal", "sagittal"}
    for p in pngs:
        assert p.stat().st_size > 0
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"  # valid PNG signature

    manifest = _read_manifest(out)
    assert isinstance(manifest.get("images"), list)
    # one manifest entry per PNG
    assert len(manifest["images"]) == len(pngs)
    files_in_manifest = {item["file_name"] for item in manifest["images"]}
    assert files_in_manifest == {p.name for p in pngs}


def test_render_seg_previews_manifest_records_rotation(tmp_path):
    # render_seg_previews bakes rotate={"sagittal": -1, "axial": 2} into the PNGs and
    # records rotate_k in the manifest so click coords can be un-rotated.
    nifti = _write_vol(_density_vol(), tmp_path / "vol.nii.gz")
    out = tmp_path / "out"
    postprocess.render_seg_previews(nifti, _labelmap(), out)
    rot_by_orientation = {}
    for item in _read_manifest(out)["images"]:
        rot_by_orientation.setdefault(item["orientation"], item["rotate_k"])
    assert rot_by_orientation.get("sagittal") == -1
    assert rot_by_orientation.get("axial") == 2
    assert rot_by_orientation.get("coronal") == 0  # coronal is not rotated


# --------------------------------------------------------------------------- #
# render_seg_previews — density tiers vs flat scar
# --------------------------------------------------------------------------- #
def test_density_from_self_splits_scar_into_tiers(tmp_path):
    # density_from_self=True derives the tiers from the volume's own reflectivity:
    # the manifest's paint_pixels then carries the per-tier scar masks (diffuse/mod/dense).
    nifti = _write_vol(_density_vol(), tmp_path / "vol.nii.gz")
    out = tmp_path / "out"
    postprocess.render_seg_previews(nifti, _labelmap(), out, density_from_self=True)

    tier_names = set()
    for item in _read_manifest(out)["images"]:
        tier_names.update(k for k in item["paint_pixels"] if k.startswith("scar"))
    # graded scar -> all three reflectivity tiers must appear as separate masks
    assert {"scar_diffuse", "scar_mod", "scar"} <= tier_names


def test_no_density_scar_is_flat(tmp_path):
    # Without density info the scar is a single flat mask: no tier sub-masks are emitted.
    nifti = _write_vol(_density_vol(), tmp_path / "vol.nii.gz")
    out = tmp_path / "out"
    postprocess.render_seg_previews(nifti, _labelmap(), out)  # density_from_self defaults False

    seen = set()
    for item in _read_manifest(out)["images"]:
        seen.update(item["paint_pixels"].keys())
    assert "scar" in seen
    assert "scar_diffuse" not in seen
    assert "scar_mod" not in seen


def test_explicit_density_vol_used_for_tiers(tmp_path):
    # Passing density_vol explicitly (a separate raw volume) also enables the tiers,
    # even when the labelmap-bearing NIfTI itself is uniform.
    plain = np.full((8, 24, 20), 50, np.uint16)  # the "volume" passed as nifti (uniform)
    nifti = _write_vol(plain, tmp_path / "vol.nii.gz")
    out = tmp_path / "out"
    postprocess.render_seg_previews(nifti, _labelmap(), out, density_vol=_density_vol())
    tier_names = set()
    for item in _read_manifest(out)["images"]:
        tier_names.update(k for k in item["paint_pixels"] if k.startswith("scar"))
    assert {"scar_diffuse", "scar_mod", "scar"} <= tier_names


def test_no_scar_labelmap_renders_only_cornea_bg(tmp_path):
    # A labelmap with no scar: density tiering is skipped (scar.any() is False),
    # only background/cornea/(empty)scar masks exist and cornea count is returned.
    lab = np.zeros((8, 24, 20), np.uint8)
    lab[:, 10:14, :] = CORNEA
    nifti = _write_vol(_density_vol(), tmp_path / "vol.nii.gz")
    out = tmp_path / "out"
    cornea_vox = postprocess.render_seg_previews(nifti, lab, out, density_from_self=True)
    assert cornea_vox == int((lab == CORNEA).sum())
    seen = set()
    for item in _read_manifest(out)["images"]:
        seen.update(k for k in item["paint_pixels"] if k.startswith("scar"))
    assert "scar_diffuse" not in seen and "scar_mod" not in seen


# --------------------------------------------------------------------------- #
# render_seg_previews — dense_rotated panel
# --------------------------------------------------------------------------- #
def test_dense_rotated_yields_more_slices(tmp_path):
    # dense_rotated=True renders EVERY slice (cap 100000) instead of the sparse 9-per-axis
    # default, so the overlay scrubs in lock-step with the raw/corrected context previews.
    nifti = _write_vol(_density_vol(), tmp_path / "vol.nii.gz")
    sparse_out = tmp_path / "sparse"
    dense_out = tmp_path / "dense"
    postprocess.render_seg_previews(nifti, _labelmap(), sparse_out, dense_rotated=False)
    postprocess.render_seg_previews(nifti, _labelmap(), dense_out, dense_rotated=True)
    n_sparse = len(list(sparse_out.glob("segmentation_*.png")))
    n_dense = len(list(dense_out.glob("segmentation_*.png")))
    assert n_dense > n_sparse


def test_render_seg_previews_creates_missing_out_dir(tmp_path):
    nifti = _write_vol(_density_vol(), tmp_path / "vol.nii.gz")
    out = tmp_path / "a" / "b" / "out"   # nested, does not exist yet
    assert not out.exists()
    postprocess.render_seg_previews(nifti, _labelmap(), out)
    assert out.is_dir()
    assert (out / "preview_manifest.json").exists()


# --------------------------------------------------------------------------- #
# render_context_previews
# --------------------------------------------------------------------------- #
def test_render_context_previews(tmp_path):
    vol = _density_vol()
    nifti = _write_vol(vol, tmp_path / "vol.nii.gz")
    out = tmp_path / "ctx"
    total = postprocess.render_context_previews(nifti, out)
    assert total == int(vol.size)
    pngs = sorted(out.glob("context_*.png"))
    assert pngs, "no context PNGs written"
    # context previews carry no overlay masks at all
    for item in _read_manifest(out)["images"]:
        assert item["paint_pixels"] == {}
        assert item["prefix"] == "context"


# --------------------------------------------------------------------------- #
# scar.density_tiers_absolute — the colour-tier helper
# --------------------------------------------------------------------------- #
def test_density_tiers_absolute_empty_scar():
    shape = (4, 4, 4)
    out, cutoffs = scar_mod.density_tiers_absolute(
        np.zeros(shape, bool), np.full(shape, 100, np.float32), np.ones(shape, bool))
    assert out.shape == shape
    assert out.dtype == np.uint8
    assert not out.any()
    assert cutoffs == []


def test_density_tiers_absolute_tiers_and_cutoffs():
    shape = (1, 1, 6)
    cornea = np.zeros(shape, bool)
    scar = np.zeros(shape, bool)
    dens = np.zeros(shape, np.float32)
    # normal cornea reference voxels (median = 100)
    cornea[0, 0, 0:2] = True
    dens[0, 0, 0:2] = 100.0
    # three scar voxels: diffuse (<1.6x), moderate (1.6-2.4x), dense (>=2.4x)
    scar[0, 0, 2] = True; dens[0, 0, 2] = 120.0   # 1.2x  -> tier 1
    scar[0, 0, 3] = True; dens[0, 0, 3] = 200.0   # 2.0x  -> tier 2
    scar[0, 0, 4] = True; dens[0, 0, 4] = 300.0   # 3.0x  -> tier 3

    tier, cutoffs = scar_mod.density_tiers_absolute(scar, dens, cornea)
    # cutoffs = ref(=100) * ratios(1.6, 2.4)
    assert cutoffs == pytest.approx([160.0, 240.0])
    # tiers are 1..len(ratios)+1 only inside the scar, 0 elsewhere
    assert tier[0, 0, 2] == 1
    assert tier[0, 0, 3] == 2
    assert tier[0, 0, 4] == 3
    assert tier[0, 0, 0] == 0 and tier[0, 0, 1] == 0   # cornea ref voxels untouched
    assert tier[0, 0, 5] == 0                           # background untouched
    # tier is strictly confined to the scar mask
    assert np.array_equal(tier > 0, scar)
    # tier values span the full 1..3 range
    assert set(np.unique(tier[scar]).tolist()) == {1, 2, 3}


def test_density_tiers_absolute_falls_back_without_cornea_reference():
    # No usable cornea reference (no cornea-only voxels) -> falls back to intra-scar
    # quantiles via density_tiers (still 1..n_tiers inside scar, 0 elsewhere).
    shape = (1, 1, 4)
    scar = np.zeros(shape, bool)
    dens = np.zeros(shape, np.float32)
    scar[0, 0, 1:4] = True
    dens[0, 0, 1:4] = [50.0, 150.0, 250.0]
    cornea = np.zeros(shape, bool)   # no cornea reference at all

    tier, cutoffs = scar_mod.density_tiers_absolute(scar, dens, cornea)
    expected, _ = scar_mod.density_tiers(scar, dens, n_tiers=len(scar_mod.DENSITY_TIER_RATIOS) + 1)
    assert np.array_equal(tier, expected)
    # tiering confined to scar; background stays 0
    assert tier[0, 0, 0] == 0
    assert (tier[scar] >= 1).all()


def test_density_tiers_absolute_all_voxels_diffuse_when_dim():
    # Scar dimmer than 1.6x the cornea reference -> every scar voxel is tier 1 (diffuse).
    shape = (1, 1, 4)
    cornea = np.zeros(shape, bool); cornea[0, 0, 0] = True
    dens = np.zeros(shape, np.float32); dens[0, 0, 0] = 100.0
    scar = np.zeros(shape, bool); scar[0, 0, 1:4] = True
    dens[0, 0, 1:4] = [90.0, 110.0, 130.0]   # all < 160
    tier, cutoffs = scar_mod.density_tiers_absolute(scar, dens, cornea)
    assert cutoffs == pytest.approx([160.0, 240.0])
    assert set(np.unique(tier[scar]).tolist()) == {1}
