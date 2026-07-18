"""Unit tests for debug_align.py (the Debug tab's replicate-alignment comparison).

Covers the parts that are silently wrong rather than loudly broken:
  * parse_case / groups — scheme B MUST win over scheme A, or 97 real cases vanish
  * registration._rigid_intensity's new kwargs DEFAULT to the shipped constants (consensus guard)
  * the metric's two load-bearing properties (fixed transform-independent mask; blur matching)
  * the brute-force translation search recovers a known synthetic shift
  * the composite is magenta=fixed / green=moving and shares ONE window

All synthetic + tiny. No real case data, no network, no GPU.
"""
from __future__ import annotations

import inspect
import time

import numpy as np
import pytest
import scipy.ndimage as ndi
import SimpleITK as sitk

import debug_align as da
import registration as reg
import settings


# ── naming schemes ───────────────────────────────────────────────────────────
@pytest.mark.parametrize("name,eye,key", [
    ("case_cs001_os_v1", "cs001_os", (1, 0)),
    ("case_cs001_os_v3", "cs001_os", (3, 0)),
    ("case_cs005_od_v9", "cs005_od", (9, 0)),
    # Scheme B: the greedy /_v(\d+)$/ mis-reads this as eye "cs030_od_v1" replicate 2.
    ("case_cs030_od_v1_2", "cs030_od", (1, 2)),
    ("case_cs030_od_v1_3", "cs030_od", (1, 3)),
    ("case_cs030_od_v1", "cs030_od", (1, 0)),
])
def test_parse_case_both_schemes(name, eye, key):
    assert da.parse_case(name) == (eye, key)


def test_parse_case_rejects_non_cases():
    assert da.parse_case("case_oct_real") == (None, None)
    assert da.parse_case("random_dir") == (None, None)


def test_scheme_b_groups_with_its_own_eye_not_a_phantom():
    """The regression that silently drops 97 cases: v1, v1_2 and v1_3 are ONE eye."""
    eyes = {da.parse_case(n)[0] for n in
            ("case_cs030_od_v1", "case_cs030_od_v1_2", "case_cs030_od_v1_3")}
    assert eyes == {"cs030_od"}


def _mkcase(root, cid, shape=(6, 6, 4)):
    import nibabel as nib
    d = root / cid / "previews"
    d.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4)), str(d / "volume.nii.gz"))


def test_groups_only_multi_replicate_eyes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CASES_ROOT", tmp_path)
    for cid in ("case_cs001_os_v1", "case_cs001_os_v2",
                "case_cs030_od_v1", "case_cs030_od_v1_2", "case_cs030_od_v1_3",
                "case_lonely_od_v1"):
        _mkcase(tmp_path, cid)
    (tmp_path / "case_cs001_os_v1_consensus").mkdir(parents=True)   # must be skipped
    g = {x["eye"]: x["cases"] for x in da.groups()}
    assert "lonely_od" not in g                      # single replicate -> not offered
    assert g["cs001_os"] == ["case_cs001_os_v1", "case_cs001_os_v2"]
    assert g["cs030_od"] == ["case_cs030_od_v1", "case_cs030_od_v1_2", "case_cs030_od_v1_3"]


def test_groups_skips_cases_without_a_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CASES_ROOT", tmp_path)
    _mkcase(tmp_path, "case_cs009_os_v1")
    (tmp_path / "case_cs009_os_v2").mkdir(parents=True)   # no previews/volume.nii.gz
    assert da.groups() == []


# ── the consensus guard: shipped constants must remain the DEFAULTS ──────────
def test_rigid_intensity_defaults_are_the_shipped_constants():
    """debug_align adds kwargs to _rigid_intensity. If a default drifts, every existing caller
    (align_transform -> the post-SAM2 consensus lifecycle, validated at cornea Dice 0.978) silently
    changes behaviour. This is the tripwire."""
    p = inspect.signature(reg._rigid_intensity).parameters
    assert p["learning_rate"].default == 0.8
    assert p["smoothing_sigmas"].default == (2.0, 1.0, 0.0)
    assert p["seed"].default == 1


def test_rigid_intensity_new_params_are_keyword_only():
    """Keyword-only, so no positional caller can accidentally bind them to fixed_mask's slot."""
    p = inspect.signature(reg._rigid_intensity).parameters
    for n in ("learning_rate", "smoothing_sigmas", "seed"):
        assert p[n].kind is inspect.Parameter.KEYWORD_ONLY


def test_the_2_constant_fix_constants():
    assert da.FIX_LR == 0.03
    assert da.FIX_SIGMAS == (0.04, 0.02, 0.0)


# ── metric ───────────────────────────────────────────────────────────────────
def _blob(shape=(40, 50, 16), seed=0):
    """A smooth bright slab in a dark box — enough structure for Otsu + NCC."""
    rng = np.random.default_rng(seed)
    v = rng.random(shape, np.float32) * 40.0
    v[8:32, 14:30, 3:13] += 600.0
    return ndi.gaussian_filter(v, 1.0)


_SP = [0.05, 0.05, 0.1]


def test_identity_against_itself_is_perfect():
    fi = da.to_sitk(_blob(), _SP)
    assert da.score(fi, fi)["primary"] == pytest.approx(1.0, abs=1e-6)
    assert da.score(fi, fi)["frac_out"] == pytest.approx(0.0, abs=1e-9)


def test_eval_mask_is_transform_independent():
    """The whole comparison is rigged if a transform can change what gets scored."""
    fi = da.to_sitk(_blob(), _SP)
    mi = da.to_sitk(_blob(seed=1), _SP)
    n = [da.score(fi, mi, R=np.eye(3), t=t)["n_voxels"]
         for t in ([0, 0, 0], [0.3, 0, 0], [0, -0.4, 0.2])]
    assert len(set(n)) == 1


def test_eval_mask_spans_the_edge_not_just_tissue():
    """A tissue-ONLY mask is provably wrong here (speckle does not repeat between acquisitions);
    the signal is the bright/dark EDGE, so the dilated mask must be strictly larger than tissue."""
    fi = da.to_sitk(_blob(), _SP)
    _, _, masks, _ = da._fixed_side(fi)
    assert masks["dil0.2"].sum() > masks["tissue"].sum()
    assert masks["dil0.4"].sum() > masks["dil0.2"].sum()


def test_true_shift_beats_identity_and_a_wrong_shift():
    v = _blob()
    shift = [0.0, 6.0, 0.0]                       # voxels, depth axis
    vm = ndi.shift(v, shift, order=1, mode="constant", cval=0.0)
    fi, mi = da.to_sitk(v, _SP), da.to_sitk(vm, _SP)
    t_true = [0.0, shift[1] * _SP[1], 0.0]        # p_mov = p_fix + t
    s_true = da.score(fi, mi, R=np.eye(3), t=t_true)["primary"]
    s_id = da.score(fi, mi)["primary"]
    s_wrong = da.score(fi, mi, R=np.eye(3), t=[0.0, -0.5, 0.0])["primary"]
    assert s_true > s_id + da.BLUR_FLOOR
    assert s_true > s_wrong + da.BLUR_FLOOR


def test_out_of_fov_is_penalised_not_dropped():
    fi = da.to_sitk(_blob(), _SP)
    s = da.score(fi, fi, R=np.eye(3), t=[1.5, 0.0, 0.0])   # evict most of the volume
    assert s["frac_out"] > 0.0
    assert s["primary"] < 1.0


# ── transform helpers ────────────────────────────────────────────────────────
def test_extract_rigid_round_trips_origin_centered():
    e = sitk.Euler3DTransform()
    e.SetCenter((1.0, 2.0, 3.0))
    e.SetRotation(0.0, 0.0, 0.2)
    e.SetTranslation((0.1, -0.2, 0.05))
    R, t_eff = da.extract_rigid(e)
    x = np.array([0.7, -1.3, 2.2])
    assert np.allclose(R @ x + t_eff, np.array(e.TransformPoint(tuple(x))), atol=1e-9)


def test_extract_rigid_rejects_a_real_composite():
    c = sitk.CompositeTransform([sitk.Euler3DTransform(), sitk.Euler3DTransform()])
    with pytest.raises(ValueError):
        da.extract_rigid(c)


def test_ang():
    assert da.ang(np.eye(3)) == pytest.approx(0.0)
    c, s = np.cos(0.3), np.sin(0.3)
    assert da.ang(np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])) == pytest.approx(np.rad2deg(0.3))


# ── brute force ──────────────────────────────────────────────────────────────
def test_bruteforce_recovers_a_known_shift():
    v = _blob(shape=(48, 64, 24))
    lag = np.array([2, -3, 1])
    vm = ndi.shift(v, -lag, order=1, mode="constant", cval=0.0)   # vm(p) = v(p + lag)
    t, info = da.bruteforce_translation(v, vm, _SP)
    # fixed[p] ~ moving[p + d] and moving(p) = fixed(p + lag)  =>  d = -lag
    assert info["lag_vox"] == [int(-x) for x in lag]
    assert np.allclose(t, -lag * np.asarray(_SP), atol=max(_SP) * 0.6)
    assert not info["on_window_edge"]
    assert info["box_ncc_peak"] > info["box_ncc_identity"]


def test_bruteforce_is_deterministic():
    v, w = _blob(shape=(48, 64, 24)), _blob(shape=(48, 64, 24), seed=2)
    a, _ = da.bruteforce_translation(v, w, _SP)
    b, _ = da.bruteforce_translation(v, w, _SP)
    assert np.array_equal(a, b)


def test_bruteforce_raises_on_a_too_small_volume():
    tiny = np.zeros((6, 6, 3), np.float32)
    with pytest.raises(ValueError):
        da.bruteforce_translation(tiny, tiny, _SP)


# ── tissue cleaning / geometry ───────────────────────────────────────────────
def test_clean_tissue_drops_short_runs_keeps_the_band():
    tis = np.zeros((4, 100, 2), bool)
    tis[:, 10:13, :] = True      # 3-voxel speckle run  -> must go
    tis[:, 40:80, :] = True      # 40-voxel band        -> must stay
    out = da._clean_tissue(tis, [0.05, 0.005, 0.1])   # k = 0.05/0.005 = 10 voxels
    assert not out[:, 10:13, :].any()
    assert out[:, 45:75, :].all()


def test_view_geometry_indices_are_in_bounds_and_ordered():
    v = _blob(shape=(60, 80, 20))
    g = da.view_geometry(v, _SP)
    for k, hi in (("lat", 60), ("depth", 80), ("frames", 20),
                  ("zoom_lat", 60), ("zoom_depth", 80)):
        a, b = g[k]
        assert 0 <= a < b <= hi, k
    assert 0 <= g["frame"] < 20
    assert 0 <= g["apex_lat"] < 60


# ── render ───────────────────────────────────────────────────────────────────
def test_composite_is_magenta_fixed_green_moving():
    f = np.full((4, 5), 100.0, np.float32)
    m = np.zeros((4, 5), np.float32)
    rgb = da._composite(f, m, 0.0, 100.0, 0.05, 0.05)
    assert (rgb[..., 0] == 255).all() and (rgb[..., 2] == 255).all()   # fixed -> R+B = magenta
    assert (rgb[..., 1] == 0).all()                                    # no moving -> no green
    rgb2 = da._composite(m, f, 0.0, 100.0, 0.05, 0.05)
    assert (rgb2[..., 1] == 255).all()                                 # moving -> green
    assert (rgb2[..., 0] == 0).all() and (rgb2[..., 2] == 0).all()
    rgb3 = da._composite(f, f, 0.0, 100.0, 0.05, 0.05)                 # agreement -> white
    assert (rgb3 == 255).all()


def test_composite_respects_the_shared_window():
    """Both images MUST use the fixed volume's window; a per-image stretch would make a brightness
    difference look like a misalignment."""
    f = np.full((4, 5), 100.0, np.float32)
    m = np.full((4, 5), 50.0, np.float32)
    rgb = da._composite(f, m, 0.0, 100.0, 0.05, 0.05)
    assert (rgb[..., 1] == 127).all()       # 50/100 -> mid grey, NOT re-stretched to 255


def test_aspect_resize_uses_true_mm_aspect():
    """~13x anisotropy: drawn 1 voxel = 1 pixel the picture lies about the cornea's shape."""
    rgb = np.zeros((100, 100, 3), np.uint8)
    out = da._aspect_resize(rgb, row_mm=0.01, col_mm=0.04)   # 1 mm tall x 4 mm wide
    assert out.shape[1] == da.RENDER_WIDTH
    assert out.shape[0] == pytest.approx(da.RENDER_WIDTH / 4, rel=0.02)


def test_window_from_fixed_ignores_zeros():
    v = np.zeros((10, 10, 10), np.float32)
    v[2:8, 2:8, 2:8] = 500.0
    lo, hi = da.window_from_fixed(v)
    assert lo > 0.0 and hi >= lo


def test_render_root_is_never_inside_the_case_store():
    """review_cases/ is irreplaceable user data; debug PNGs are disposable."""
    assert settings.CASES_ROOT not in da._RENDER_ROOT.parents
    assert da._RENDER_ROOT != settings.CASES_ROOT


# ── surface residual ─────────────────────────────────────────────────────────
# The geometric truth, and the metric methods should be RANKED on: NCC scored the 2-constant fix and
# brute-force translation as a tie (+0.0127) while their surface residuals differ ~6x, because a pure
# translation cannot remove a TILT and NCC over a dilated mask barely registers one.
_RSP = [0.01, 0.005, 0.04]      # lat, depth, frames (mm) -> clip_guard 4, clean_tissue run 10


def _slab(surf, shape=(40, 200, 30), seed=0, thick=80):
    """A bright slab whose anterior surface sits at depth `surf` (scalar or per-frame) in a dark
    speckled box. A per-frame `surf` lets a TILT be built exactly."""
    rng = np.random.default_rng(seed)
    v = (rng.random(shape, np.float32) * 20.0).astype(np.float32)
    s = np.broadcast_to(np.asarray(surf, int), (shape[2],))
    for f in range(shape[2]):
        v[:, int(s[f]):int(s[f]) + thick, f] += 800.0
    return ndi.gaussian_filter(v, (1.0, 2.0, 0.0))


def _ref_for(v, lat=20):
    return da.surface_reference(v, _RSP, {"apex_lat": lat}, da._fixed_masks(v, _RSP))


def _shift_frames(v, shifts):
    """Roll each frame along DEPTH — preserves the speckle exactly, so the ONLY difference between
    fixed and moving is the surface offset under test."""
    out = np.empty_like(v)
    for f in range(v.shape[2]):
        out[:, :, f] = np.roll(v[:, :, f], int(shifts[f]), axis=1)
    return out


def test_surface_residual_is_zero_against_itself():
    v = _slab(60)
    r = da.surface_residual(_ref_for(v), v)
    assert r["resid_vox"] == pytest.approx(0.0, abs=1e-9)
    assert r["tilt_vox"] == pytest.approx(0.0, abs=1e-9)


def test_surface_residual_recovers_a_known_constant_depth_shift():
    """A pure depth offset must read back as exactly that offset, with NO tilt."""
    v = _slab(60)
    r = da.surface_residual(_ref_for(v), _shift_frames(v, np.full(30, 5)))
    assert r["resid_vox"] == pytest.approx(5.0, abs=0.5)
    assert r["tilt_vox"] == pytest.approx(0.0, abs=0.5)


def test_surface_residual_magnitude_is_symmetric_in_direction():
    v = _slab(60)
    ref = _ref_for(v)
    assert da.surface_residual(ref, _shift_frames(v, np.full(30, -6)))["resid_vox"] \
        == pytest.approx(6.0, abs=0.5)
    assert da.surface_residual(ref, _shift_frames(v, np.full(30, +6)))["resid_vox"] \
        == pytest.approx(6.0, abs=0.5)


def test_surface_residual_measures_a_tilt_a_translation_cannot_remove():
    """THE point of this metric: a surface tilted across frames reads a real tilt_vox. This is the
    case a pure translation structurally cannot fix and NCC scores as a tie."""
    nfr, a = 30, 0.5                                 # 0.5 vox of extra shift per frame
    v = _slab(60, shape=(40, 200, nfr))
    r = da.surface_residual(_ref_for(v), _shift_frames(v, np.round(a * np.arange(nfr))))
    span = (da.RESID_TILT_HI - da.RESID_TILT_LO) * (nfr - 1)
    assert r["tilt_vox"] == pytest.approx(a * span, abs=1.5)   # d[f] = -a*f -> tilt = +a*span


def test_surface_residual_tilt_sign_follows_the_fringe_convention():
    """positive tilt = the moving surface is SHALLOWER at low frames = green fringe there."""
    nfr = 30
    v = _slab(60, shape=(40, 200, nfr))
    ref = _ref_for(v)
    down = da.surface_residual(ref, _shift_frames(v, np.round(0.5 * np.arange(nfr))))
    up = da.surface_residual(ref, _shift_frames(v, -np.round(0.5 * np.arange(nfr))))
    assert down["tilt_vox"] > 4.0
    assert up["tilt_vox"] < -4.0


def test_surface_residual_um_uses_the_depth_spacing():
    v = _slab(60)
    r = da.surface_residual(_ref_for(v), _shift_frames(v, np.full(30, 4)))
    assert r["resid_um"] == pytest.approx(r["resid_vox"] * _RSP[1] * 1000.0, rel=1e-9)
    assert r["resid_um"] == pytest.approx(20.0, abs=3.0)          # 4 vox x 5 um


def test_surface_residual_search_is_constrained_to_the_fixed_surface():
    """THE reason wedge.py was wrong. A resampled moving volume has a hard out-of-FOV zero step
    ABOVE the tissue; unconstrained, the steepest-rise detector locks onto that step instead of the
    cornea. Here a deliberately STRONGER spurious edge sits ~130 vox above the surface — outside the
    +-90 window — and must be ignored."""
    v = _slab(150, shape=(40, 300, 30))
    mv = v.copy()
    mv[:, 20:26, :] += 5000.0
    mv = ndi.gaussian_filter(mv, (1.0, 2.0, 0.0))
    r = da.surface_residual(_ref_for(v), mv)
    assert r["resid_vox"] < 3.0, "detector locked onto the out-of-FOV step, not the cornea"


def test_surface_residual_flags_a_saturated_search_window():
    """The +-90 constraint that saves the detector from the out-of-FOV step also CLAMPS a genuinely
    huge offset: the mean is then a LOWER BOUND, not a measurement. Observed on cs005_od v1 vs v9
    (different lateral spacing / FOV): every method pegged at exactly 90.0. Flag it — never average
    clamped values into a plausible-looking number.

    Scope, honestly: this catches an offset AT or JUST BEYOND the bound (the argmax pins to the
    window edge), which is the regime real pairs land in — identity's worst measured residual is
    ~67 vox against a 90 bound. It CANNOT catch an offset so far out that the window contains only
    background, where the argmax lands somewhere random inside; nothing short of a wider search
    could. resid_saturated=False is therefore not a certificate, it is the absence of one alarm."""
    v = _slab(60, shape=(40, 400, 30))
    inside = da.surface_residual(_ref_for(v), _shift_frames(v, np.full(30, 88)))
    assert inside["resid_vox"] == pytest.approx(88.0, abs=1.0)   # still a real measurement
    assert inside["resid_saturated"] is False
    beyond = da.surface_residual(_ref_for(v), np.asarray(
        _shift_frames(v, np.full(30, da.RESID_SEARCH_VOX + 5))))
    assert beyond["resid_saturated"] is True
    assert beyond["resid_vox"] == pytest.approx(float(da.RESID_SEARCH_VOX), abs=1.0)  # clamped


def test_surface_reference_gates_clipped_frames():
    """A CLIPPED apex (tissue running to depth 0) has no edge to detect, so the steepest rise there
    is noise. wedge2.py has no such gate — it survived only by hardcoding a lateral that happens to
    be unclipped; at the apex lateral this tool renders, a clipped frame threw |resid| to 43 vox.
    Those frames must be DROPPED, not measured."""
    surf = np.full(30, 60)
    surf[10:13] = 0
    ref = _ref_for(_slab(surf, shape=(40, 200, 30)))
    assert not ref["frame_ok"][10:13].any()
    assert ref["frame_ok"][20]


def test_surface_reference_drops_the_edge_frames():
    ref = _ref_for(_slab(60))
    assert not ref["frame_ok"][:da.RESID_EDGE_FRAMES].any()
    assert not ref["frame_ok"][-da.RESID_EDGE_FRAMES:].any()


def test_surface_residual_reports_none_when_too_few_valid_frames():
    """Never invent a number: too few usable frames -> nulls, which the UI renders as '-'."""
    ref = _ref_for(_slab(60))
    ref["frame_ok"] = np.zeros_like(ref["frame_ok"])
    ref["frame_ok"][5] = True
    r = da.surface_residual(ref, _slab(60))
    assert r["resid_vox"] is None and r["resid_um"] is None and r["tilt_vox"] is None


def test_surface_reference_anchors_on_the_rendered_sagittal_lateral():
    """The number must explain the picture: it is measured where the sagittal panel is drawn."""
    v = _slab(60)
    g = da.view_geometry(v, _RSP)
    assert _ref_for(v, lat=g["apex_lat"])["lat"] == g["apex_lat"]


def test_fixed_masks_reuse_matches_computing_them_inline():
    """The residual and the geometry share ONE tissue mask (~1.3 s on a real volume). Passing it in
    must be identical to letting view_geometry compute its own."""
    v = _slab(60)
    assert da.view_geometry(v, _RSP) == da.view_geometry(v, _RSP, da._fixed_masks(v, _RSP))


# ── 3-D interactive replicate-agreement volumes ──────────────────────────────
def test_dbg_affine_orients_like_the_main_viewer():
    """niivue derives the initial camera pose from the affine; it must match the app's preview volumes
    (axcodes L,I,P: lat->X, depth->Z, frames->Y) so the cornea shows the right way up."""
    import nibabel as nib
    aff = da._dbg_affine(0.02)
    assert nib.aff2axcodes(aff) == ("L", "I", "P")
    # isotropic spacing recovered from the affine columns
    assert np.allclose(np.linalg.norm(aff[:3, :3], axis=0), [0.02, 0.02, 0.02])


def test_write_nifti_round_trips_on_the_iso_grid(tmp_path):
    """Written .nii.gz must load back with the same voxels, the iso spacing, and the debug affine —
    exactly what niivue fetches from the token-exempt view endpoint."""
    import nibabel as nib
    arr = (np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5))
    p = tmp_path / "fixed_iso.nii.gz"
    da._write_nifti(p, arr, 0.02, np.uint16)
    img = nib.load(str(p))
    assert np.array_equal(np.asarray(img.dataobj), arr.astype(np.uint16))
    assert np.allclose(img.header.get_zooms()[:3], [0.02, 0.02, 0.02])
    assert nib.aff2axcodes(img.affine) == ("L", "I", "P")


def test_write_nifti_clips_uint16_range(tmp_path):
    import nibabel as nib
    arr = np.array([[[-5.0, 70000.0]]], np.float32)
    p = tmp_path / "clip.nii.gz"
    da._write_nifti(p, arr, 0.02, np.uint16)
    got = np.asarray(nib.load(str(p)).dataobj)
    assert got.min() == 0 and got.max() == 65535


def test_build_disagree_gates_to_the_tissue_edge_within_coverage():
    """The gate = dilated fixed tissue AND moving coverage. Disagreement outside coverage (FOV
    eviction) or far from tissue must be zeroed, never mistaken for a real difference."""
    f = np.zeros((10, 10, 10), np.float32)
    f[3:7, 3:7, 3:7] = 100.0
    m = f.copy()
    m[5, 5, 5] = 900.0                              # a genuine in-tissue difference
    tissue = f > 0
    cov = np.ones_like(tissue, bool)
    cov[:2, :, :] = False                           # a rim the moving does NOT cover
    diff, gate = da.build_disagree(f, m, tissue, cov, 0.02)
    assert diff[5, 5, 5] == pytest.approx(800.0)    # the real difference survives
    assert not diff[:2, :, :].any()                 # uncovered rim is zeroed
    assert (diff[~gate] == 0.0).all()


def _v3d_inputs(v):
    masks = da._fixed_masks(v, _RSP)
    geom = da.view_geometry(v, _RSP, masks)
    iso = da.DBG_ISO_MM
    f_iso = da._iso_crop(v, _RSP, geom, iso, order=1)
    t_iso = da._iso_crop(masks["tissue"].astype(np.float32), _RSP, geom, iso,
                         order=0, antialias=False) > 0.5
    return geom, iso, f_iso, t_iso


def test_build_method_volumes_writes_gettable_nifti(tmp_path):
    v = _slab(60, shape=(40, 160, 24))
    geom, iso, f_iso, t_iso = _v3d_inputs(v)
    m_res = _shift_frames(v, np.full(24, 8))
    out = da.build_method_volumes(f_iso, t_iso, m_res, _RSP, geom, iso,
                                  tmp_path, "job123", "identity", scale=None)
    v3 = out["volumes3d"]
    assert out["scale"] > 0
    assert v3["moving"] == "/api/debug/align/view/job123/identity_moving_iso.nii.gz"
    assert v3["disagree"] == "/api/debug/align/view/job123/identity_disagree_iso.nii.gz"
    assert (tmp_path / "identity_moving_iso.nii.gz").exists()   # GET-able by the view endpoint
    assert (tmp_path / "identity_disagree_iso.nii.gz").exists()
    assert 0.0 <= v3["disagree_mean"] <= 1.0
    # RAW shared scale for the client's cal_max (intensity units, NOT the [0,1] mean).
    assert v3["disagree_max"] == pytest.approx(out["scale"])


def test_build_method_volumes_disagreement_is_cooler_for_a_better_alignment(tmp_path):
    """THE point of the disagreement volume: a misaligned pair reads a HIGHER disagree_mean than an
    aligned one AT THE SAME (identity-derived) scale — the numeric twin of 'identity hotter'."""
    v = _slab(60, shape=(40, 160, 24))
    geom, iso, f_iso, t_iso = _v3d_inputs(v)
    worse = da.build_method_volumes(f_iso, t_iso, _shift_frames(v, np.full(24, 8)), _RSP, geom, iso,
                                    tmp_path, "job1", "identity", scale=None)
    better = da.build_method_volumes(f_iso, t_iso, v, _RSP, geom, iso,
                                     tmp_path, "job1", "fixed", scale=worse["scale"])  # SHARED scale
    assert worse["volumes3d"]["disagree_mean"] > better["volumes3d"]["disagree_mean"]
    assert better["volumes3d"]["disagree_mean"] == pytest.approx(0.0, abs=1e-6)   # perfect match cool


def test_build_method_volumes_shares_one_iso_grid(tmp_path):
    """fixed / moving / disagree must land on ONE lattice so niivue overlays them voxel-perfect."""
    import nibabel as nib
    v = _slab(60, shape=(40, 160, 24))
    geom, iso, f_iso, t_iso = _v3d_inputs(v)
    da._write_nifti(tmp_path / "fixed_iso.nii.gz", f_iso, iso, np.uint16)
    da.build_method_volumes(f_iso, t_iso, _shift_frames(v, np.full(24, 6)), _RSP, geom, iso,
                            tmp_path, "job1", "identity", scale=None)
    shapes = {tuple(nib.load(str(tmp_path / n)).shape) for n in
              ("fixed_iso.nii.gz", "identity_moving_iso.nii.gz", "identity_disagree_iso.nii.gz")}
    assert len(shapes) == 1
    assert f_iso.shape == next(iter(shapes))


# ── orphan render dirs ───────────────────────────────────────────────────────
def test_sweep_removes_stale_dirs_only(tmp_path, monkeypatch):
    """_prune_jobs only knows the in-memory _JOBS, so on restart every prior job dir is orphaned and
    _MAX_KEPT_JOBS never applies across processes (observed: 17 dirs / 34 MB)."""
    import os
    monkeypatch.setattr(da, "_RENDER_ROOT", tmp_path)
    monkeypatch.setattr(da, "_JOBS", {"livejob": {"started": 0.0, "running": True}})
    old = tmp_path / "oldjob"
    old.mkdir()
    (old / "a.png").write_bytes(b"x")
    fresh = tmp_path / "freshjob"
    fresh.mkdir()
    live = tmp_path / "livejob"
    live.mkdir()
    stale = time.time() - da._JOB_TTL_S - 60
    for d in (old, live):
        os.utime(d, (stale, stale))       # `live` is stale on disk but STILL RUNNING here
    assert da.sweep_render_root() == 1
    assert not old.exists()               # orphan from a previous process -> gone
    assert fresh.exists()                 # younger than the TTL: another sidecar may be writing it
    assert live.exists()                  # a live job in THIS process is never swept


def test_sweep_runs_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(da, "_SWEPT", False)
    monkeypatch.setattr(da, "sweep_render_root", lambda *a, **k: calls.append(1))
    da._sweep_once()
    da._sweep_once()
    da._sweep_once()
    assert len(calls) == 1


def test_sweep_never_raises_into_a_job(monkeypatch):
    """Housekeeping must never fail a comparison."""
    def boom(*a, **k):
        raise OSError("boom")
    monkeypatch.setattr(da, "_SWEPT", False)
    monkeypatch.setattr(da, "sweep_render_root", boom)
    da._sweep_once()          # must not raise


def test_sweep_is_a_noop_when_the_root_does_not_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(da, "_RENDER_ROOT", tmp_path / "nope")
    assert da.sweep_render_root() == 0


# ── consensus: N-replicate min/excess decomposition ──────────────────────────
def test_consensus_decompose_agreement_is_white():
    """Where every replicate has the SAME intensity, excess is 0 -> pure white (the consensus glow),
    with alpha = that intensity."""
    I = [np.full((3, 3, 2), 0.7, np.float32) for _ in range(4)]
    cols = [np.asarray(da.CONSENSUS_PALETTE[i], np.float32) / 255.0 for i in range(4)]
    rgb, alpha = da.consensus_decompose(I, cols)
    assert np.allclose(rgb, 0.7, atol=1e-6)          # R==G==B==shared -> white/grey
    assert np.allclose(alpha, 0.7, atol=1e-6)


def test_consensus_decompose_single_replicate_sticks_out_in_its_hue():
    """A voxel bright in ONE replicate and dark in the rest -> shared 0 -> pure replicate hue."""
    I = [np.zeros((2, 2, 2), np.float32) for _ in range(3)]
    I[1][0, 0, 0] = 1.0                                # only replicate 1 (index 1) sticks out
    cols = [np.asarray(da.CONSENSUS_PALETTE[i], np.float32) / 255.0 for i in range(3)]
    rgb, alpha = da.consensus_decompose(I, cols)
    assert np.allclose(rgb[0, 0, 0], cols[1], atol=1e-6)   # exactly replicate 1's colour
    assert alpha[0, 0, 0] == pytest.approx(1.0)
    assert np.allclose(rgb[1, 1, 1], 0.0)                  # nobody there -> black/transparent
    assert alpha[1, 1, 1] == pytest.approx(0.0)


def test_consensus_decompose_alpha_is_the_max():
    rng = np.random.default_rng(0)
    I = [rng.random((4, 4, 3), np.float32) for _ in range(5)]
    cols = [np.asarray(da.CONSENSUS_PALETTE[i], np.float32) / 255.0 for i in range(5)]
    _rgb, alpha = da.consensus_decompose(I, cols)
    assert np.allclose(alpha, np.max(np.stack(I, 0), axis=0))


def test_consensus_decompose_clips_to_unit_cube():
    """Two replicates both sticking out in bright hues can sum past 1 -> must clip, never overflow."""
    I = [np.ones((2, 2, 2), np.float32), np.ones((2, 2, 2), np.float32)]
    cols = [np.array([1.0, 1.0, 0.0], np.float32), np.array([0.0, 1.0, 1.0], np.float32)]
    # shared = 1 everywhere -> excess 0 -> pure white; but test the raw-excess branch too:
    I2 = [np.ones((2, 2, 2), np.float32), np.zeros((2, 2, 2), np.float32)]
    rgb, _ = da.consensus_decompose(I2, cols)
    assert rgb.max() <= 1.0 and rgb.min() >= 0.0
    rgb2, _ = da.consensus_decompose(I, cols)
    assert np.allclose(rgb2, 1.0)                      # full agreement -> white


def test_consensus_colors_labels_both_schemes_and_marks_reference():
    a = da.consensus_colors("cs001_os", ["case_cs001_os_v1", "case_cs001_os_v2", "case_cs001_os_v3"])
    assert [r["label"] for r in a] == ["v1", "v2", "v3"]
    assert a[0]["is_ref"] and not a[1]["is_ref"] and not a[2]["is_ref"]
    b = da.consensus_colors("cs030_od", ["case_cs030_od_v1", "case_cs030_od_v1_2", "case_cs030_od_v1_3"])
    assert [r["label"] for r in b] == ["v1", "v1_2", "v1_3"]
    assert all(len(r["color"]) == 3 for r in a + b)


def test_consensus_palette_is_distinct_and_agree_is_white():
    """The legend must be readable: every replicate hue distinct, and none is the white agreement."""
    assert da.AGREE_COLOR == (255, 255, 255)
    assert len(set(da.CONSENSUS_PALETTE)) == len(da.CONSENSUS_PALETTE)   # all distinct
    assert da.AGREE_COLOR not in da.CONSENSUS_PALETTE
    assert len(da.CONSENSUS_PALETTE) >= 9                                # scales to the 9-rep eye


def test_write_rgba_nifti_is_rgba32_and_round_trips(tmp_path):
    """niivue's 3-D RGBA path keys off DT_RGBA32 (2304); the volume must load back byte-for-byte with
    the iso spacing and the debug (L,I,P) affine."""
    import nibabel as nib
    rgb = np.zeros((3, 4, 5, 3), np.float32)
    rgb[0, 0, 0] = [1.0, 0.0, 0.0]
    alpha = np.zeros((3, 4, 5), np.float32)
    alpha[0, 0, 0] = 1.0
    p = tmp_path / "consensus_rgba.nii.gz"
    da._write_rgba_nifti(p, rgb, alpha, 0.02)
    img = nib.load(str(p))
    assert int(img.header["datatype"]) == 2304 and int(img.header["bitpix"]) == 32
    assert np.allclose(img.header.get_zooms()[:3], [0.02, 0.02, 0.02])
    assert nib.aff2axcodes(img.affine) == ("L", "I", "P")
    got = np.asarray(img.dataobj)
    assert int(got["R"][0, 0, 0]) == 255 and int(got["A"][0, 0, 0]) == 255
    assert int(got["G"][0, 0, 0]) == 0 and int(got["R"][1, 1, 1]) == 0


def _mkcase_vol(root, cid, vol):
    import nibabel as nib
    d = root / cid / "previews"
    d.mkdir(parents=True, exist_ok=True)
    # affine spacing = _RSP (lat, depth, frames) so window/geometry behave like a real preview
    aff = np.diag([_RSP[0], _RSP[1], _RSP[2], 1.0])
    nib.save(nib.Nifti1Image(np.asarray(vol, np.float32), aff), str(d / "volume.nii.gz"))


def test_start_consensus_rejects_unknown_method_and_single_replicate(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "CASES_ROOT", tmp_path)
    _mkcase_vol(tmp_path, "case_cs001_os_v1", _slab(60, shape=(40, 160, 24)))
    with pytest.raises(ValueError):                       # only one replicate
        da.start_consensus("cs001_os", "fixed")
    _mkcase_vol(tmp_path, "case_cs001_os_v2", _slab(60, shape=(40, 160, 24), seed=1))
    with pytest.raises(ValueError):                       # unknown method
        da.start_consensus("cs001_os", "bogus")


def test_consensus_end_to_end_identity(tmp_path, monkeypatch):
    """The whole path on synthetic data: 3 replicates -> a GET-able RGBA volume on the iso lattice,
    with a mostly-agreeing region reading whitish (R~=G~=B)."""
    import nibabel as nib
    monkeypatch.setattr(settings, "CASES_ROOT", tmp_path)
    monkeypatch.setattr(da, "_RENDER_ROOT", tmp_path / "render")
    base = _slab(60, shape=(40, 160, 24))
    for i, cid in enumerate(("case_cs007_os_v1", "case_cs007_os_v2", "case_cs007_os_v3")):
        _mkcase_vol(tmp_path, cid, _slab(60, shape=(40, 160, 24), seed=i))   # aligned surfaces
    job_id = da.start_consensus("cs007_os", "identity")
    for _ in range(200):
        v = da.consensus_job_view(job_id)
        if v["status"] != "running":
            break
        time.sleep(0.05)
    assert v["status"] == "done", v.get("error")
    assert v["reference"] == "case_cs007_os_v1"
    assert [r["label"] for r in v["replicates"]] == ["v1", "v2", "v3"]
    assert v["agree_color"] == [255, 255, 255]
    assert v["iso_mm"] == da.DBG_ISO_MM
    # The RGBA volume the view route would serve exists at job_dir/<basename>.
    name = v["volume"].rsplit("/", 1)[-1]
    p = da.job_dir(job_id) / name
    assert p.exists() and name.endswith(".nii.gz")
    img = nib.load(str(p))
    assert int(img.header["datatype"]) == 2304
    assert tuple(img.shape) == tuple(v["geometry"]["iso_shape"])
    # 2-D composites exist and are named as the view route expects.
    for s in ("bscan", "sagittal"):
        assert (da.job_dir(job_id) / f"consensus_{s}.png").exists()
    # Whiteness: over voxels with real signal, aligned identical-surface slabs read grey/white
    # (channels ~equal), i.e. the agreement colour dominates.
    arr = np.asarray(img.dataobj)
    R = arr["R"].astype(np.float32); Gc = arr["G"].astype(np.float32); B = arr["B"].astype(np.float32)
    A = arr["A"].astype(np.float32)
    m = A > 60
    assert m.sum() > 0
    chroma = np.stack([R, Gc, B], -1)[m]
    whiteness = 1.0 - (chroma.max(-1) - chroma.min(-1)) / np.maximum(chroma.max(-1), 1.0)
    assert float(whiteness.mean()) > 0.6      # mostly-agreeing -> mostly white/grey
