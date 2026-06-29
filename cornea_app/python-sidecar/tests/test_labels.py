"""Unit tests for labels.py — the canonical segmentation labelmap module.

Covers the documented invariants:
  * write_label_nifti: shape guard, array+affine round-trip, atomic (no _tmp_ leftover),
    concurrent-style writes to the same dst both succeed (uuid temp).
  * write_display_labelmap: density -> reflectivity tiers 2/3/4, cornea stays 1;
    density_vol=None -> scar becomes 4; no scar -> stays cornea/bg.
  * labelmap_counts: per-class voxel counts + volume_mm3.
  * best_labelmap_nnunet: corrected present -> (arr in {0,1,2}, "corrected"); absent -> (None, None).

Pure-numpy / nibabel / on-disk; no GPU, no SAM2, no torch, no network, no real data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib
import pytest

import labels
import orchestration as orch
import scar as scar_mod


# ── local helpers (fixtures live in conftest; only thin helpers here) ─────────

def _base_nifti(write_nifti, tmp_path, shape=(4, 6, 5), affine=None):
    """A tiny base volume on disk whose affine the labels get stamped with."""
    vol = np.zeros(shape, np.uint16)
    if affine is None:
        affine = np.diag([0.013, 0.013, 0.041, 1.0]).astype(float)
    return write_nifti(vol, tmp_path / "base.nii.gz", affine), affine


# ───────────────────────── write_label_nifti ────────────────────────────────

def test_write_label_nifti_shape_mismatch_raises(write_nifti, tmp_path):
    base, _ = _base_nifti(write_nifti, tmp_path, shape=(4, 6, 5))
    wrong = np.zeros((4, 6, 4), np.uint8)          # last axis differs
    with pytest.raises(ValueError):
        labels.write_label_nifti(wrong, base, tmp_path / "out.nii.gz")


def test_write_label_nifti_roundtrip_array_and_affine(write_nifti, tmp_path):
    base, affine = _base_nifti(write_nifti, tmp_path, shape=(4, 6, 5))
    arr = np.zeros((4, 6, 5), np.uint8)
    arr[:, 2:4, :] = 1
    arr[1:3, 2:3, 1:3] = 2
    dst = tmp_path / "seg" / "labelmap.nii.gz"

    ret = labels.write_label_nifti(arr, base, dst)
    assert ret == dst
    assert dst.exists()

    img = nib.load(str(dst))
    out = np.asarray(img.dataobj)
    # array equals input
    np.testing.assert_array_equal(out.astype(np.uint8), arr)
    assert out.dtype == np.uint8
    # affine equals the base volume's affine
    np.testing.assert_allclose(img.affine, affine)


def test_write_label_nifti_is_gzipped(write_nifti, tmp_path):
    base, _ = _base_nifti(write_nifti, tmp_path, shape=(4, 6, 5))
    arr = np.zeros((4, 6, 5), np.uint8)
    dst = tmp_path / "labelmap.nii.gz"
    labels.write_label_nifti(arr, base, dst)
    # gzip magic bytes
    with open(dst, "rb") as fh:
        magic = fh.read(2)
    assert magic == b"\x1f\x8b"


def test_write_label_nifti_leaves_no_tmp_file(write_nifti, tmp_path):
    base, _ = _base_nifti(write_nifti, tmp_path, shape=(4, 6, 5))
    arr = np.zeros((4, 6, 5), np.uint8)
    dst = tmp_path / "out" / "labelmap.nii.gz"
    labels.write_label_nifti(arr, base, dst)
    leftovers = list(dst.parent.glob("_tmp_*"))
    assert leftovers == [], f"temp file(s) left behind: {leftovers}"
    # exactly one output file
    assert [p.name for p in dst.parent.iterdir()] == [dst.name]


def test_write_label_nifti_concurrent_style_writes_same_dst(write_nifti, tmp_path, monkeypatch):
    """Two interleaved writes to the SAME dst must both succeed because each uses a
    UNIQUE uuid temp. We genuinely interleave: spy on os.replace so that while the
    FIRST write is poised to rename, a SECOND full write to the same dst runs to
    completion first; both temps must coexist (no collision) and both renames succeed.
    `os` is imported locally inside write_label_nifti, so patching the real os module
    object reaches it."""
    import os as _os

    base, _ = _base_nifti(write_nifti, tmp_path, shape=(4, 6, 5))
    dst = tmp_path / "shared" / "labelmap.nii.gz"

    a = np.zeros((4, 6, 5), np.uint8)
    a[0, 0, 0] = 1
    b = np.zeros((4, 6, 5), np.uint8)
    b[1, 1, 1] = 2

    seen_tmp = []
    real_replace = _os.replace
    reentered = {"done": False}

    def spy_replace(src, dst_):
        seen_tmp.append(Path(src).name)
        assert Path(src).exists(), "temp must exist at replace time"
        if not reentered["done"]:
            # While write #1 is paused mid-rename, run write #2 to completion. Its
            # temp must NOT collide with write #1's still-present temp, and both
            # temps coexist on disk at this instant -> proves uuid uniqueness.
            reentered["done"] = True
            existing_tmps = set(p.name for p in Path(src).parent.glob("_tmp_*"))
            assert Path(src).name in existing_tmps
            labels.write_label_nifti(b, base, dst)        # nested, same dst
            assert Path(src).exists(), "write #1's temp survived write #2 (no clobber)"
        return real_replace(src, dst_)

    monkeypatch.setattr(_os, "replace", spy_replace)
    labels.write_label_nifti(a, base, dst)                 # write #1 (triggers nested #2)
    monkeypatch.undo()

    # write #1 wrote 'a' second (its replace ran after the nested 'b'), so final == a
    assert len(seen_tmp) == 2
    assert seen_tmp[0] != seen_tmp[1], "uuid temp names must be unique per write"
    assert all(t.startswith("_tmp_") and t.endswith(dst.name) for t in seen_tmp)
    out = np.asarray(nib.load(str(dst)).dataobj).astype(np.uint8)
    np.testing.assert_array_equal(out, a)
    assert list(dst.parent.glob("_tmp_*")) == [], "no temp files left behind"


# ───────────────────────── write_display_labelmap ───────────────────────────

def test_display_labelmap_density_tiers_2_3_4(write_nifti, tmp_path):
    """With a density volume, scar voxels split into tiers 2/3/4 and cornea stays 1.

    density_tiers_absolute: ref = median cornea-only reflectivity; cutoffs = ref*(1.6, 2.4);
    tier = digitize(density, cutoffs)+1 -> 1/2/3; write_display adds 1 -> labels 2/3/4.
    """
    shape = (4, 6, 6)
    base, _ = _base_nifti(write_nifti, tmp_path, shape=shape)

    labelmap = np.zeros(shape, np.uint8)
    labelmap[:, 2:4, :] = 1                    # cornea band
    # three scar voxels we will control independently
    diffuse = (0, 2, 0)
    moderate = (0, 2, 1)
    dense = (0, 2, 2)
    for v in (diffuse, moderate, dense):
        labelmap[v] = 2

    density = np.zeros(shape, np.float32)
    # cornea-only (label==1 and not scar) reflectivity -> ref median = 100
    cornea_only = (labelmap == 1)
    density[cornea_only] = 100.0
    # cutoffs = [160, 240]; pick scar reflectivities straddling each band
    density[diffuse] = 100.0     # < 160  -> tier1 -> label 2
    density[moderate] = 200.0    # 160..240 -> tier2 -> label 3
    density[dense] = 500.0       # >= 240 -> tier3 -> label 4

    dst = tmp_path / "display.nii.gz"
    labels.write_display_labelmap(labelmap, density, base, dst)
    out = np.asarray(nib.load(str(dst)).dataobj).astype(np.uint8)

    # cornea preserved as 1
    cornea_disp = (out == 1)
    np.testing.assert_array_equal(cornea_disp, cornea_only & (labelmap != 2))
    # tiers
    assert out[diffuse] == 2
    assert out[moderate] == 3
    assert out[dense] == 4
    # background stays 0
    assert out[3, 5, 5] == 0
    # only the expected labels present
    assert set(np.unique(out).tolist()) <= {0, 1, 2, 3, 4}


def test_display_labelmap_none_density_scar_becomes_4(write_nifti, tmp_path):
    shape = (4, 6, 5)
    base, _ = _base_nifti(write_nifti, tmp_path, shape=shape)
    labelmap = np.zeros(shape, np.uint8)
    labelmap[:, 2:4, :] = 1
    labelmap[1:3, 2:3, 1:3] = 2
    scar_mask = labelmap == 2

    dst = tmp_path / "display_nodens.nii.gz"
    labels.write_display_labelmap(labelmap, None, base, dst)
    out = np.asarray(nib.load(str(dst)).dataobj).astype(np.uint8)

    # every scar voxel -> 4 (solid dense)
    assert np.all(out[scar_mask] == 4)
    # cornea (non-scar label-1) -> 1
    assert np.all(out[(labelmap == 1)] == 1)
    assert set(np.unique(out).tolist()) <= {0, 1, 4}


def test_display_labelmap_no_scar_stays_cornea_bg(write_nifti, tmp_path):
    shape = (4, 6, 5)
    base, _ = _base_nifti(write_nifti, tmp_path, shape=shape)
    labelmap = np.zeros(shape, np.uint8)
    labelmap[:, 2:4, :] = 1                    # cornea only, no scar
    density = np.full(shape, 50.0, np.float32)

    dst = tmp_path / "display_noscar.nii.gz"
    labels.write_display_labelmap(labelmap, density, base, dst)
    out = np.asarray(nib.load(str(dst)).dataobj).astype(np.uint8)

    np.testing.assert_array_equal(out, (labelmap == 1).astype(np.uint8))
    assert set(np.unique(out).tolist()) <= {0, 1}


# ───────────────────────────── labelmap_counts ──────────────────────────────

def test_labelmap_counts_voxel_counts_only():
    arr = np.zeros((4, 6, 5), np.uint8)
    arr[:, 2:4, :] = 1                          # cornea voxels
    arr[1:3, 2:3, 1:3] = 2                      # scar voxels (carved out of cornea)
    n_cornea_total = int((arr == 1).sum())
    n_scar = int((arr == 2).sum())
    n_bg = int((arr == 0).sum())

    out = labels.labelmap_counts(arr)
    assert set(out.keys()) == {"background", "cornea", "scar"}
    assert out["background"]["voxel_count"] == n_bg
    assert out["cornea"]["voxel_count"] == n_cornea_total
    assert out["scar"]["voxel_count"] == n_scar
    # without spacing there is no volume_mm3
    assert "volume_mm3" not in out["cornea"]
    # counts partition the volume
    assert n_bg + n_cornea_total + n_scar == arr.size


def test_labelmap_counts_with_spacing_volume_mm3():
    arr = np.zeros((3, 3, 3), np.uint8)
    arr[0, 0, 0] = 2                            # 1 scar voxel
    arr[:, 1, :] = 1                            # 9 cornea voxels
    spacing = 0.5                               # mm^3 per voxel
    out = labels.labelmap_counts(arr, spacing_mm3=spacing)

    assert out["scar"]["voxel_count"] == 1
    assert out["scar"]["volume_mm3"] == round(1 * spacing, 4)
    assert out["cornea"]["voxel_count"] == 9
    assert out["cornea"]["volume_mm3"] == round(9 * spacing, 4)
    assert out["background"]["volume_mm3"] == round((arr.size - 10) * spacing, 4)


def test_labelmap_counts_empty_classes_zero():
    arr = np.full((2, 2, 2), 1, np.uint8)      # all cornea
    out = labels.labelmap_counts(arr, spacing_mm3=2.0)
    assert out["background"]["voxel_count"] == 0
    assert out["scar"]["voxel_count"] == 0
    assert out["background"]["volume_mm3"] == 0.0
    assert out["cornea"]["voxel_count"] == 8
    assert out["cornea"]["volume_mm3"] == 16.0


# ─────────────────────────── best_labelmap_nnunet ───────────────────────────

def test_best_labelmap_nnunet_present(make_case):
    cid = make_case("case_best_present")
    arr, src = labels.best_labelmap_nnunet(cid)
    assert src == "corrected"
    assert arr is not None
    # canonical training labels are exactly {0,1,2}
    assert set(np.unique(arr).tolist()) <= {0, 1, 2}
    assert arr.dtype == np.uint8
    # the corrected labelmap built by make_case has both cornea and scar
    assert (arr == 1).any()
    assert (arr == 2).any()


def test_best_labelmap_nnunet_absent(cases_root):
    # a case id that was never written -> no corrected labelmap on disk
    cid = "case_never_written"
    assert not labels.corrected_path(cid).exists()
    arr, src = labels.best_labelmap_nnunet(cid)
    assert arr is None
    assert src is None


def test_best_labelmap_nnunet_rounds_float_labels(cases_root, write_nifti, make_volume):
    """corrected file stored as float should be rounded to uint8 {0,1,2}."""
    cid = "case_float_corrected"
    orch.ensure_case_dirs(cid)
    pv = orch.case_root(cid) / "previews" / "volume.nii.gz"
    vol = np.zeros((3, 4, 4), np.uint16)
    write_nifti(vol, pv)
    lab = np.zeros((3, 4, 4), np.float32)
    lab[:, 1:3, :] = 1.0
    lab[0, 1, 1] = 2.0
    # write the corrected labelmap directly as float via nibabel (bypass write_label_nifti's uint8 cast)
    nib.save(nib.Nifti1Image(lab, np.eye(4)), str(labels.corrected_path(cid)))

    arr, src = labels.best_labelmap_nnunet(cid)
    assert src == "corrected"
    assert arr.dtype == np.uint8
    assert set(np.unique(arr).tolist()) == {0, 1, 2}
