"""Unit tests for consensus.py — the per-eye multi-scan voting + native-frame
clipping + nifti write helper.

Strategy for determinism/speed:
  * The voting INVARIANT (a voxel in >=2 of 3 binary masks survives; a voxel in
    only 1 does not) is exercised both directly against the module's own formula
    and end-to-end through build_consensus().
  * For the end-to-end path we hand build_consensus() THREE cases that share the
    EXACT same volume image (so registration's best-of-identity guard picks the
    identity transform — no warp, no slow/divergent optimiser path), but each with
    its OWN scar mask. That isolates the vote + native-clip + write behaviour from
    the registration internals while still running the real code path.

All arrays are tiny (<=24 voxels/axis). No network/GPU/SAM2/torch/real data.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import SimpleITK as sitk

import consensus
import labels
import orchestration as orch


# ── local helpers (kept out of conftest per the task rules) ───────────────────
# NOTE on geometry: build_consensus() canonicalises the reference image (origin 0,
# identity direction) and stamps that geometry onto the voted consensus, then
# resamples it back into each member's RAW (non-canon) native image. nibabel's
# default RAS affine (diag([+,+,+])) is read by SimpleITK with a (-1,-1,1)
# direction, so the canon consensus and the raw native image DISAGREE in physical
# space and the native-frame consensus resamples to nothing. Using an LPS-positive
# affine (diag([-,-,+])) makes SimpleITK read identity direction + origin 0 for the
# raw image — matching _canon — so the native-frame path is geometrically valid.
# (See the test-level note returned to the caller: this is a real latent geometry
# coupling in build_consensus, not just a test artefact.)
_LPS_AFFINE = np.diag([-0.02, -0.02, 0.04, 1.0]).astype(float)


def _make_consensus_member(make_case, cid, scar_box, base_vol, affine=_LPS_AFFINE):
    """Write a case that shares `base_vol` (identical image → identity alignment)
    but has its own cornea+scar labelmap derived from scar_box."""
    shape = base_vol.shape
    lab = np.zeros(shape, np.uint8)
    # a generous cornea band so every scar voxel is inside the cornea
    lab[:, 8:16, :] = consensus.REF_CORNEA
    f0, f1, d0, d1, l0, l1 = scar_box
    lab[f0:f1, d0:d1, l0:l1] = consensus.REF_SCAR
    return make_case(cid=cid, vol=base_vol.copy(), lab=lab, affine=affine)


def _read_arr(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _sitk_box(box):
    """Convert a numpy (frame,depth,lateral) half-open box (f0,f1,d0,d1,l0,l1) to a
    tuple of slices addressing a SimpleITK-read array, whose axes are the REVERSE
    voxel order (lateral, depth, frame). Indexing a sitk-read array with the
    original numpy slices would silently go out of bounds (size-mismatched axes) and
    make assertions pass vacuously — so always go through this for read-back checks."""
    f0, f1, d0, d1, l0, l1 = box
    return (slice(l0, l1), slice(d0, d1), slice(f0, f1))


# ── 1. voting arithmetic invariant (direct, no registration) ──────────────────
def test_majority_vote_threshold_three_masks():
    """Reproduce the exact module rule: consensus = votes >= floor(n/2)+1.
    For n=3 the threshold is 2: a voxel set in >=2 masks is in, only-1 is out."""
    n = 3
    thr = math.floor(n / 2) + 1
    assert thr == 2

    shape = (4, 4, 4)
    m1 = np.zeros(shape, bool)
    m2 = np.zeros(shape, bool)
    m3 = np.zeros(shape, bool)
    # voxel A: all three agree   -> in consensus
    m1[0, 0, 0] = m2[0, 0, 0] = m3[0, 0, 0] = True
    # voxel B: exactly two agree -> in consensus
    m1[1, 1, 1] = m2[1, 1, 1] = True
    # voxel C: only one          -> NOT in consensus
    m3[2, 2, 2] = True

    stack = np.stack([m1, m2, m3]).astype(np.uint8)
    votes = stack.sum(axis=0)
    consensus_mask = votes >= thr

    assert consensus_mask[0, 0, 0]      # 3/3 in
    assert consensus_mask[1, 1, 1]      # 2/3 in
    assert not consensus_mask[2, 2, 2]  # 1/3 out
    assert int(consensus_mask.sum()) == 2


def test_majority_vote_two_masks_needs_both():
    """n=2 -> threshold floor(2/2)+1 = 2: a voxel must be in BOTH masks."""
    n = 2
    thr = math.floor(n / 2) + 1
    assert thr == 2
    shape = (3, 3, 3)
    a = np.zeros(shape, bool); b = np.zeros(shape, bool)
    a[0, 0, 0] = b[0, 0, 0] = True   # both -> in
    a[1, 1, 1] = True                # only a -> out
    votes = np.stack([a, b]).astype(np.uint8).sum(0)
    cons = votes >= thr
    assert cons[0, 0, 0]
    assert not cons[1, 1, 1]
    assert int(cons.sum()) == 1


def test_consensus_clipped_to_reference_cornea():
    """Even a unanimous voxel is dropped if it lies outside the reference cornea
    (the `& ref_cornea` term in build_consensus)."""
    shape = (3, 3, 3)
    m = np.ones(shape, bool)            # everyone votes everywhere
    votes = np.stack([m, m, m]).astype(np.uint8).sum(0)
    ref_cornea = np.zeros(shape, bool)
    ref_cornea[1, 1, 1] = True          # cornea = single voxel
    n = 3
    cons = (votes >= math.floor(n / 2) + 1) & ref_cornea
    assert int(cons.sum()) == 1
    assert cons[1, 1, 1]


# ── 2. end-to-end voting through build_consensus ──────────────────────────────
def test_build_consensus_majority_vote_end_to_end(make_case, make_volume):
    """Three identical-image scans, distinct scar masks. The consensus scar in the
    written labelmap must contain voxels seen by >=2 scans and exclude voxels seen
    by only one."""
    shape = (6, 24, 20)
    base = make_volume(shape=shape, fill=20, cornea_band=(8, 16))

    # scar boxes are numpy (f0,f1,d0,d1,l0,l1) = (frame,depth,lateral)
    shared = (2, 4, 11, 13, 6, 9)        # in A & B & C  -> votes=3 -> in consensus
    a_only = (2, 4, 11, 13, 14, 17)      # in A only     -> votes=1 -> NOT in consensus
    a = _make_consensus_member(make_case, "case_aa", shared, base)
    b = _make_consensus_member(make_case, "case_bb", shared, base)
    c = _make_consensus_member(make_case, "case_cc", shared, base)

    # give A an extra unique-only scar block by re-writing its label (shared + a_only)
    a_lab = np.zeros(shape, np.uint8)
    a_lab[:, 8:16, :] = consensus.REF_CORNEA
    a_lab[2:4, 11:13, 6:9] = consensus.REF_SCAR     # shared
    a_lab[2:4, 11:13, 14:17] = consensus.REF_SCAR   # A-only
    labels.write_label_nifti(a_lab, orch.case_root(a) / "previews" / "volume.nii.gz",
                             labels.corrected_path(a))

    cons_cid = "case_cons"
    orch.ensure_case_dirs(cons_cid)
    report = consensus.build_consensus([a, b, c], cons_cid, reference=a)

    assert report["n_scans"] == 3
    assert report["agreement_threshold"] == 2
    assert report["reference"] == a
    # the report's native scar volume per scan is non-zero (the biomarker)
    assert all(p["scar_volume_mm3"] > 0 for p in report["per_scan"])

    # read the written consensus labelmap; sitk axes are (lateral,depth,frame)
    cons_arr = _read_arr(labels.corrected_path(cons_cid))
    scar = cons_arr == consensus.REF_SCAR
    # shared region seen by all 3 -> present in consensus (votes 3 >= thr 2)
    assert scar[_sitk_box(shared)].all()
    # A-only region -> NOT in consensus (only 1 vote < thr)
    assert not scar[_sitk_box(a_only)].any()
    # the consensus scar is EXACTLY the shared block, nothing more
    assert int(scar.sum()) == (4 - 2) * (13 - 11) * (9 - 6)
    # labels are mutually exclusive: a voxel is cornea XOR scar XOR background
    cornea = cons_arr == consensus.REF_CORNEA
    assert not (scar & cornea).any()
    assert set(np.unique(cons_arr)).issubset({0, consensus.REF_CORNEA, consensus.REF_SCAR})


# ── 3. FOV / cornea clipping in the per-scan native consensus map ─────────────
def test_cons_native_clipped_to_member_fov_and_cornea(make_case, make_volume):
    """The per-scan cons_native.nii.gz keeps the consensus scar only where the
    member has (a) image data (volume>0) AND (b) its own cornea. A consensus voxel
    outside the member's FOV or cornea must be absent in that member's native map."""
    import nibabel as nib
    shape = (6, 24, 20)
    base = make_volume(shape=shape, fill=20, cornea_band=(8, 16))

    shared = (2, 4, 11, 13, 6, 9)        # the unanimous consensus scar block
    a = _make_consensus_member(make_case, "case_fa", shared, base)
    b = _make_consensus_member(make_case, "case_fb", shared, base)
    c = _make_consensus_member(make_case, "case_fc", shared, base)

    # member B: punch a hole in its FOV (volume==0) over PART of the consensus scar
    # (lateral cols 6:8) -> those voxels must be dropped from B's native map; col 8
    # still has data -> kept.
    b_vol = base.copy()
    b_vol[2:4, 11:13, 6:8] = 0
    bvp = orch.case_root(b) / "previews" / "volume.nii.gz"
    nib.save(nib.Nifti1Image(b_vol, nib.load(str(bvp)).affine), str(bvp))

    # member C: carve a NON-cornea hole inside the consensus-scar column (lateral
    # col 6 set to background) -> that voxel is outside C's cornea -> dropped.
    c_lab = np.zeros(shape, np.uint8)
    c_lab[:, 8:16, :] = consensus.REF_CORNEA
    c_lab[2:4, 11:13, 6:9] = consensus.REF_SCAR
    c_lab[2:4, 11:13, 6:7] = 0
    labels.write_label_nifti(c_lab, orch.case_root(c) / "previews" / "volume.nii.gz",
                             labels.corrected_path(c))

    cons_cid = "case_cons_fov"
    orch.ensure_case_dirs(cons_cid)
    consensus.build_consensus([a, b, c], cons_cid, reference=a)

    scans_dir = orch.case_root(cons_cid) / "scans"

    # reference A: full FOV + full cornea -> the whole consensus scar is in its native map
    a_nat = _read_arr(scans_dir / a / "cons_native.nii.gz")
    a_scar = a_nat == consensus.REF_SCAR
    assert a_scar.any()
    assert a_scar[_sitk_box(shared)].all()

    # member B: FOV hole over cols 6:8 -> no scar there; col 8 (in-FOV) -> scar kept
    b_nat = _read_arr(scans_dir / b / "cons_native.nii.gz")
    b_scar = b_nat == consensus.REF_SCAR
    assert not b_scar[_sitk_box((2, 4, 11, 13, 6, 8))].any()   # dropped (no data)
    assert b_scar[_sitk_box((2, 4, 11, 13, 8, 9))].all()       # kept (in FOV)

    # member C: non-cornea hole over col 6 -> no scar there; cols 7:9 (cornea) -> kept
    c_nat = _read_arr(scans_dir / c / "cons_native.nii.gz")
    c_scar = c_nat == consensus.REF_SCAR
    assert not c_scar[_sitk_box((2, 4, 11, 13, 6, 7))].any()   # dropped (outside cornea)
    assert c_scar[_sitk_box((2, 4, 11, 13, 7, 9))].all()       # kept (inside cornea)
    # every native map is a valid 0/1/2 labelmap; scar never lands on background
    for nat in (a_nat, b_nat, c_nat):
        assert set(np.unique(nat)).issubset({0, consensus.REF_CORNEA, consensus.REF_SCAR})


def test_cons_native_nonempty_under_nonidentity_native_geometry(make_case, make_volume):
    """REGRESSION (geometry coupling): build_consensus stamps CANON geometry (origin 0, identity
    direction) on the voted consensus + the scan→ref transforms, then pulls it back into each
    member's native frame. If the native image is resampled in its RAW geometry, a non-identity
    direction (real OCT carries a rotated/flipped direction; nibabel's default RAS affine is read
    by SimpleITK as a (-1,-1,1) direction) sends every output point to the wrong physical place and
    cons_native comes out with ZERO scar — silently. The fix resamples onto a CANON'd copy of the
    native grid, so cons_native must carry scar regardless of the native direction. Here every
    member uses nibabel's default RAS affine (the geometry that previously produced empty maps)."""
    shape = (6, 24, 20)
    base = make_volume(shape=shape, fill=20, cornea_band=(8, 16))
    ras = np.diag([0.02, 0.02, 0.04, 1.0]).astype(float)   # SimpleITK reads this as a flipped direction
    shared = (2, 4, 11, 13, 6, 9)
    a = _make_consensus_member(make_case, "case_ga", shared, base, affine=ras)
    b = _make_consensus_member(make_case, "case_gb", shared, base, affine=ras)
    c = _make_consensus_member(make_case, "case_gc", shared, base, affine=ras)

    cons_cid = "case_cons_geom"
    orch.ensure_case_dirs(cons_cid)
    consensus.build_consensus([a, b, c], cons_cid, reference=a)
    scans_dir = orch.case_root(cons_cid) / "scans"
    for m in (a, b, c):
        nat = _read_arr(scans_dir / m / "cons_native.nii.gz")
        assert (nat == consensus.REF_SCAR).sum() > 0, f"{m}: cons_native lost all scar (geometry-coupling regression)"
        assert nat[_sitk_box(shared)].max() == consensus.REF_SCAR


# ── 4. _write nifti round-trip with NO axis flip ──────────────────────────────
def test_write_roundtrips_array_no_axis_flip(tmp_path, make_volume):
    """_write(arr_zyx, ref_img, dst): reading it back yields the SAME array with no
    transpose/flip — GetArrayFromImage(read) == written array, element for element."""
    shape = (5, 7, 9)
    # an asymmetric, fully-distinct array so any axis swap/flip would change values
    arr = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape) % 3
    # mark a single unique voxel so a flip is unmistakable
    arr[0, 0, 0] = 0
    arr[shape[0] - 1, 0, 0] = 2     # high-z corner distinct from low-z corner
    arr[0, shape[1] - 1, 0] = 1

    ref = sitk.GetImageFromArray(np.zeros(shape, np.uint8))
    ref.SetSpacing((0.04, 0.02, 0.02))
    ref.SetOrigin((1.0, 2.0, 3.0))

    dst = tmp_path / "sub" / "written.nii.gz"
    consensus._write(arr, ref, dst)
    assert dst.exists()

    back = _read_arr(dst)
    assert back.shape == arr.shape
    assert np.array_equal(back, arr)              # identical, no flip/transpose
    # geometry copied from the reference image
    read_img = sitk.ReadImage(str(dst))
    assert read_img.GetSpacing() == pytest.approx((0.04, 0.02, 0.02))
    assert read_img.GetSize() == (shape[2], shape[1], shape[0])  # sitk size is x,y,z


def test_write_float_dtype_preserves_values(tmp_path):
    """_write with sitkFloat32 keeps the float payload (the warped-volume path) and
    still round-trips without an axis flip."""
    shape = (4, 6, 8)
    arr = (np.arange(np.prod(shape), dtype=np.float32).reshape(shape) * 0.5)
    ref = sitk.GetImageFromArray(np.zeros(shape, np.float32))
    ref.SetSpacing((0.04, 0.02, 0.02))
    dst = tmp_path / "vol.nii.gz"
    consensus._write(arr, ref, dst, dtype=sitk.sitkFloat32)
    back = _read_arr(dst)
    assert back.shape == arr.shape
    assert np.allclose(back, arr, atol=1e-4)


# ── 5. small pure helpers (deterministic, no IO) ──────────────────────────────
def test_dice_and_frac_helpers():
    a = np.zeros((4, 4, 4), bool); b = np.zeros((4, 4, 4), bool)
    a[0:2, 0, 0] = True            # 2 voxels
    b[1:3, 0, 0] = True            # 2 voxels, overlap = 1
    # dice = 2*inter/(|a|+|b|) = 2*1/4 = 0.5
    assert consensus._dice(a, b) == pytest.approx(0.5)
    # empty / empty -> 0 (no div by zero)
    z = np.zeros((2, 2), bool)
    assert consensus._dice(z, z) == 0.0
    # _frac(part, whole) = |part & whole| / |whole|
    whole = np.zeros((4,), bool); whole[0:4] = True
    part = np.zeros((4,), bool); part[0:1] = True
    assert consensus._frac(part, whole) == pytest.approx(0.25)
    assert consensus._frac(part, z.reshape(4)) == 0.0   # empty whole -> 0


def test_inverse_rigid_identity_roundtrips():
    """_inverse_rigid of an identity (Euler3D) transform leaves points unmoved."""
    ident = sitk.Euler3DTransform()
    inv = consensus._inverse_rigid(ident)
    p = (0.3, -0.2, 0.5)
    out = inv.TransformPoint(p)
    assert out == pytest.approx(p, abs=1e-9)


def test_inverse_rigid_inverts_translation():
    """The inverse of a pure translation transform undoes it (used to pull the voted
    consensus back into a scan's native frame)."""
    t = sitk.TranslationTransform(3, (0.1, -0.05, 0.2))
    # wrap as a composite (matches the CompositeTransform branch indirectly via GetInverse)
    inv = consensus._inverse_rigid(t)
    moved = t.TransformPoint((0.0, 0.0, 0.0))
    back = inv.TransformPoint(moved)
    assert back == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
