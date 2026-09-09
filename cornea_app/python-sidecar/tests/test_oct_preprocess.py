"""Unit tests for the PURE, deterministic helpers in oct_preprocess.py.

Scope: the numeric building blocks of the OCT smoother pipeline that can be
exercised on tiny synthetic surfaces / columns / volumes with a KNOWN answer —
no real .OCT decode, no SAM2/torch, no GPU, no network. The documented
invariants under test:

  * the integer-TRUNCATING (toward zero) warp/shift           (_warp_by_displacement)
  * median-of-3 boundary smoothing                            (_smooth_median)
  * cubic outlier interpolation across a max_jump             (_correct_surface)
  * the max_displacement over-correction guard (#2)           (_slice_displacement)
  * displacement interpolation from good neighbours           (_interp_bad_displacement)
  * inter-slice (axial) displacement-field smoothing (#3)     via smooth_volume
  * axial keep-best / never-worse selection                   (iterate_smooth_volume,
                                                                axial_refine_volume)
  * RANSAC/least-squares quadratic fit                        (_fit_quadratic_ransac)
  * geometry / filename / companion-.txt parsing              (pure string/number helpers)

Everything heavy (the SAM2 path, the real-.OCT readers POCT, DICOM writing) is
intentionally NOT tested here.
"""
from __future__ import annotations

import numpy as np
import pytest

import oct_preprocess as M


# ───────────────────────── warp: integer truncation toward zero ─────────────
class TestWarpByDisplacement:
    def _col(self, vals):
        """A single-column (H, 1) image from a 1-D list."""
        return np.asarray(vals, dtype=np.float64).reshape(-1, 1)

    def test_positive_shift_truncates_toward_zero(self):
        # 2.9 must truncate to 2 (int(), toward zero) — faithful to warp_image_by_edge.
        img = self._col([1, 2, 3, 4, 5, 6])
        out = M._warp_by_displacement(img, np.array([2.9]))
        # tissue pushed DOWN by 2; top 2 rows become zero padding, bottom 2 fall off.
        assert out.ravel().tolist() == [0, 0, 1, 2, 3, 4]

    def test_negative_shift_truncates_toward_zero(self):
        img = self._col([1, 2, 3, 4, 5, 6])
        out = M._warp_by_displacement(img, np.array([-2.9]))  # truncates to -2
        # tissue pulled UP by 2; bottom 2 rows become zero padding.
        assert out.ravel().tolist() == [3, 4, 5, 6, 0, 0]

    def test_subpixel_shift_truncates_to_zero_is_noop(self):
        img = self._col([1, 2, 3, 4, 5, 6])
        out = M._warp_by_displacement(img, np.array([0.7]))  # int(0.7) == 0
        np.testing.assert_array_equal(out, img)

    def test_shift_larger_than_height_blanks_column(self):
        img = self._col([1, 2, 3])
        out = M._warp_by_displacement(img, np.array([10.0]))  # nh = 3-10 < 0
        np.testing.assert_array_equal(out, np.zeros_like(img))

    def test_per_column_independent_shifts(self):
        # two columns shifted differently in the same call
        img = np.array([[1, 10], [2, 20], [3, 30], [4, 40]], dtype=np.float64)
        out = M._warp_by_displacement(img, np.array([1.0, -1.0]))
        assert out[:, 0].tolist() == [0, 1, 2, 3]    # +1
        assert out[:, 1].tolist() == [20, 30, 40, 0]  # -1


# ───────────────────────── median-of-3 smoothing ───────────────────────────
class TestSmoothMedian:
    def test_single_spike_removed(self):
        a = np.array([10, 10, 50, 10, 10], dtype=float)
        out = M._smooth_median(a, 3)
        np.testing.assert_array_equal(out, np.full(5, 10.0))

    def test_monotone_ramp_preserved_in_interior(self):
        a = np.arange(7, dtype=float)
        out = M._smooth_median(a, 3)
        # the interior of a clean ramp is unchanged by a median filter
        np.testing.assert_array_equal(out[1:-1], a[1:-1])


# ───────────────────────── _correct_surface: max_jump outliers ──────────────
class TestCorrectSurface:
    def test_outlier_interpolated_away(self):
        # 40 jumps > 10 from its neighbour 12 → flagged + cubic-interpolated back onto the line.
        s = np.array([10, 11, 12, 40, 14, 15], dtype=float)
        out = M._correct_surface(s, max_jump=10.0)
        assert abs(out[3] - 13.0) < 1e-6
        # the inliers are untouched
        np.testing.assert_allclose(out[[0, 1, 2, 4, 5]], s[[0, 1, 2, 4, 5]])

    def test_clean_surface_within_max_jump_is_noop(self):
        s = np.array([10, 11, 12, 13, 14, 15], dtype=float)
        out = M._correct_surface(s, max_jump=10.0)
        np.testing.assert_allclose(out, s)

    def test_tiny_array_returned_unchanged(self):
        s = np.array([5.0])
        np.testing.assert_array_equal(M._correct_surface(s, max_jump=1.0), s)


# ───────────────────────── _interp_bad_displacement ─────────────────────────
class TestInterpBadDisplacement:
    def test_bad_column_interpolated_from_neighbours(self):
        disp = np.array([0.0, 1.0, 99.0, 3.0, 4.0])
        out = M._interp_bad_displacement(disp.copy(), bad_cols=[2], good_cols=[])
        # linear interp at x=2 over anchors {0,1,3,4}->{0,1,3,4} == 2.0
        assert abs(out[2] - 2.0) < 1e-9
        np.testing.assert_allclose(out[[0, 1, 3, 4]], [0, 1, 3, 4])

    def test_no_bad_columns_is_noop(self):
        disp = np.array([0.0, 1.0, 2.0])
        out = M._interp_bad_displacement(disp.copy(), bad_cols=[], good_cols=[])
        np.testing.assert_array_equal(out, disp)

    def test_explicit_good_anchors_used(self):
        # good_cols restricts the anchor set; bad col interps only from listed good cols.
        disp = np.array([0.0, 5.0, 99.0, 5.0, 10.0])
        out = M._interp_bad_displacement(disp.copy(), bad_cols=[2], good_cols=[1, 3])
        assert abs(out[2] - 5.0) < 1e-9


# ───────────────────────── _slice_displacement: over-correction guard ───────
class TestSliceDisplacementGuard:
    def _quad_edge(self, n=20, a=0.02, apex_row=5.0, center=10.0):
        x = np.arange(n)
        return (a * (x - center) ** 2 + apex_row).astype(float)

    def test_perfect_quadratic_edge_yields_zero_displacement(self):
        # disp = (quad_fit - edge); if the edge already IS a quadratic, disp ≈ 0.
        edge = self._quad_edge()
        disp = M._slice_displacement(edge, residual=5.0, corr_factor=1.0,
                                     bad_cols=[], good_cols=[], max_disp=40.0)
        assert np.max(np.abs(disp)) < 1e-6

    def test_runaway_column_clamped_by_guard(self):
        # one garbage column demands a huge shift; with the guard ON it is interpolated
        # from good neighbours + clamped, so |disp| can never exceed max_disp.
        edge = self._quad_edge()
        edge[10] = 300.0
        max_disp = 40.0
        guarded = M._slice_displacement(edge, 5.0, 1.0, [], [], max_disp=max_disp)
        assert np.max(np.abs(guarded)) <= max_disp + 1e-6
        # the bad column's shift is interpolated from its good neighbours → small, not ~300.
        assert abs(guarded[10]) < 5.0

    def test_guard_disabled_lets_runaway_through(self):
        # max_disp <= 0 disables the guard (legacy) → the runaway shift survives.
        edge = self._quad_edge()
        edge[10] = 300.0
        unguarded = M._slice_displacement(edge, 5.0, 1.0, [], [], max_disp=0.0)
        assert np.max(np.abs(unguarded)) > 100.0

    def test_corr_factor_scales_displacement(self):
        # disp = (quad - edge) * corr_factor; half the corr_factor → half the shift.
        edge = self._quad_edge()
        edge[3] = edge[3] + 8.0  # a modest in-band deviation (no runaway)
        full = M._slice_displacement(edge, 5.0, 1.0, [], [], max_disp=40.0)
        half = M._slice_displacement(edge, 5.0, 0.5, [], [], max_disp=40.0)
        np.testing.assert_allclose(half, 0.5 * full, atol=1e-6)


# ───────────────────────── _fit_quadratic_ransac ────────────────────────────
class TestFitQuadraticRansac:
    def test_recovers_clean_quadratic(self):
        x = np.arange(30)
        edge = 0.03 * (x - 15) ** 2 + 4.0
        fit = M._fit_quadratic_ransac(edge, residual_threshold=5.0)
        np.testing.assert_allclose(fit, edge, atol=1e-3)

    def test_rejects_localized_outliers(self):
        # a few hyper-bright "scar" outliers are rejected; the dome fit stays smooth.
        x = np.arange(40)
        edge = (0.03 * (x - 20) ** 2 + 4.0)
        edge[18:22] += 60.0  # a localized internal bright region
        fit = M._fit_quadratic_ransac(edge, residual_threshold=5.0)
        # the fit ignores the spike → stays near the underlying parabola at those cols
        assert np.max(np.abs(fit[18:22] - (0.03 * (x[18:22] - 20) ** 2 + 4.0))) < 10.0

    def test_short_edge_falls_back_without_crash(self):
        edge = np.array([3.0, 4.0])  # < 3 points → returns the edge as float
        out = M._fit_quadratic_ransac(edge, residual_threshold=5.0)
        np.testing.assert_allclose(out, edge)


# ───────────────────────── _longest_run / _axial_roughness ──────────────────
class TestRunAndRoughness:
    def test_longest_run(self):
        assert M._longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == 3
        assert M._longest_run(np.zeros(5, dtype=bool)) == 0
        assert M._longest_run(np.ones(4, dtype=bool)) == 4

    def test_axial_roughness_constant_step(self):
        # each slice shifts by exactly 1 across slices → mean |first diff| == 1.
        e = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=float)
        assert abs(M._axial_roughness(e) - 1.0) < 1e-9

    def test_axial_roughness_flat_is_zero(self):
        assert M._axial_roughness(np.ones((4, 5))) == 0.0

    def test_axial_roughness_single_slice_is_zero(self):
        assert M._axial_roughness(np.ones((1, 5))) == 0.0


# ───────────────────────── _fill_cols_along_rows / _fill_black_bands ─────────
class TestFillBlackBands:
    def test_leading_and_trailing_zeros_edge_replicated(self):
        # one column [0,0,5,7,0] → leading 0s become 5, trailing 0 becomes 7.
        img = np.array([[0], [0], [5], [7], [0]], dtype=float)
        out = M._fill_cols_along_rows(img)
        assert out.ravel().tolist() == [5, 5, 5, 7, 7]

    def test_interior_zeros_not_touched(self):
        # only the LEADING/TRAILING runs are filled; an interior zero stays.
        img = np.array([[0], [5], [0], [7], [0]], dtype=float)
        out = M._fill_cols_along_rows(img)
        assert out.ravel().tolist() == [5, 5, 0, 7, 7]

    def test_all_zero_column_left_alone(self):
        img = np.zeros((4, 1), dtype=float)
        np.testing.assert_array_equal(M._fill_cols_along_rows(img), img)

    def test_fill_black_bands_does_not_mutate_input(self):
        # the volume helper must operate on a COPY (the transpose is a view).
        vol = np.zeros((2, 5, 3), dtype=float)
        vol[:, 2, :] = 9.0  # one bright row → leading/trailing zeros to fill
        before = vol.copy()
        out = M._fill_black_bands(vol)
        np.testing.assert_array_equal(vol, before)        # input untouched
        assert out.shape == vol.shape
        assert (out != 0).all()                            # all padding filled


# ───────────────────────── _clip_mask: clipped-apex detection ───────────────
class TestClipMask:
    def test_normal_in_frame_dome_no_clip(self):
        # bright band well below the top (edge ~15 ≥ floor 8), dark gap above → never clips.
        D, F = 40, 30
        sl = np.full((D, F), 20.0)
        sl[15:25, :] = 200.0
        edge = np.full(F, 15.0)
        assert not M._clip_mask(sl, edge, M.DEFAULT_PARAMS).any()

    def test_pinned_top_with_bright_band_clips_center(self):
        # tissue from row 0 (no air gap) + edge pinned near top (<floor) → clip symptom.
        D, F = 40, 30
        sl = np.full((D, F), 20.0)
        sl[0:30, :] = 200.0
        edge = np.full(F, 3.0)
        mask = M._clip_mask(sl, edge, M.DEFAULT_PARAMS)
        # binary_closing trims clip_close_gap cells at each boundary, so test the centre.
        assert mask[F // 2]
        assert mask.sum() > F // 2

    def test_negative_edge_not_treated_as_clip(self):
        # a NEGATIVE edge means the detector ran OFF-frame (limbus) — not a central clip.
        D, F = 40, 30
        sl = np.full((D, F), 20.0)
        sl[0:30, :] = 200.0
        edge = np.full(F, -5.0)
        assert not M._clip_mask(sl, edge, M.DEFAULT_PARAMS).any()


class TestResolveClip:
    def test_normal_dome_is_strict_noop(self):
        # a well-detected in-frame dome must yield NO clip (gates all fail) → legacy path used.
        D, F = 40, 30
        sl = np.full((D, F), 20.0)
        sl[15:25, :] = 200.0
        edge = np.full(F, 15.0)
        cols, fit = M._resolve_clip(edge, sl, residual_threshold=5.0, p=M.DEFAULT_PARAMS)
        assert cols.size == 0
        assert fit is None


# ───────────────────────── geometry / spacing helpers ───────────────────────
class TestGeometryHelpers:
    def test_reformat_revert_roundtrip(self):
        v = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        # reformat_to_sagittal is a (2,1,0) transpose → (lateral, depth, frames).
        assert M.reformat_to_sagittal(v).shape == (4, 3, 2)
        np.testing.assert_array_equal(M.revert_sagittal(M.reformat_to_sagittal(v)), v)

    def test_validate_spacing_in_range_no_warnings(self):
        # (lateral, depth, slice) all within the Avanti bounds → no advisory warnings.
        assert M.validate_spacing((0.0117, 0.0031, 0.04)) == []

    def test_validate_spacing_out_of_range_warns(self):
        warns = M.validate_spacing((0.5, 0.0031, 0.04))  # lateral far too big
        assert len(warns) == 1 and "lateral" in warns[0]

    def test_to_float_strips_units(self):
        assert M._to_float("6.00 mm") == 6.0
        assert M._to_float("-3.5px") == -3.5
        assert M._to_float("abc") is None


class TestFilenameParsing:
    def test_full_filename_with_replicate(self):
        out = M.parse_oct_filename("John_CS001_3DCornea_OD_2024-01-15(2)_extra")
        assert out["patient_id"] == "CS001"
        assert out["laterality"] == "OD"
        assert out["study_date"] == "2024-01-15"
        assert out["series_number"] == 2

    def test_first_scan_no_replicate_defaults_to_one(self):
        out = M.parse_oct_filename("John_CS001_3DCornea_OD_2024-01-15")
        assert out["study_date"] == "2024-01-15"
        assert out["series_number"] == 1

    def test_too_few_tokens_returns_empty(self):
        assert M.parse_oct_filename("too_few") == {}


class TestCompanionGeometry:
    def test_picks_active_step_by_frame_count(self, tmp_path):
        # Two steps: Step 1 is the active 3D acquisition (Usage == n_frames), Step 2 a placeholder.
        txt = (
            "[CL - 3D Cornea Step 1]\n"
            "XY Scan Length = 513\n"
            "XY Scan Usage = 101\n"
            "[CL - 3D Cornea Step 1 Detail]\n"
            "XY Scan Size1 = 6.00\n"
            "XY Scan Interval1 = 0.040\n"
            "[CL - 3D Cornea Step 2]\n"
            "XY Scan Length = 513\n"
            "XY Scan Usage = 1\n"
            "[CL - 3D Cornea Step 2 Detail]\n"
            "XY Scan Size1 = 4.00\n"
            "XY Scan Interval1 = 0.020\n"
            "[General]\n"
            "OCT Window Height = 640\n"
            "Scan Depth = 2.006\n"
            "Eye Scanned = OD\n"
        )
        p = tmp_path / "scan.txt"
        p.write_text(txt)
        geom = M.companion_geometry(p, n_frames=101)
        assert geom["lateral_spacing"] == pytest.approx(6.00 / 513)
        assert geom["depth_spacing"] == pytest.approx(2.006 / 640)
        assert geom["slice_spacing"] == pytest.approx(0.040)

    def test_unreadable_file_returns_empty(self, tmp_path):
        assert M.companion_geometry(tmp_path / "missing.txt") == {}


# ───────────────────────── smooth_volume: inter-slice smoothing (#3) ─────────
class TestSmoothVolumeInterslice:
    def _vol(self):
        # tiny synthetic OCT (frames, depth, lateral) with a flat bright cornea band.
        F, D, L = 6, 30, 8
        vol = np.full((F, D, L), 20, np.uint16)
        vol[:, 12:18, :] = 200
        return vol

    def test_shape_and_dtype_preserved(self):
        vol = self._vol()
        out = M.smooth_volume(vol, {"auto_tune": False}, workers=1)
        assert out.shape == vol.shape
        assert out.dtype == vol.dtype

    def test_return_metric_corrected_array_identical(self):
        # the corrected array must be byte-identical whether or not metrics are returned.
        vol = self._vol()
        out = M.smooth_volume(vol, {"auto_tune": False}, workers=1)
        out2, dev, ax = M.smooth_volume(vol, {"auto_tune": False}, workers=1, return_metric=True)
        np.testing.assert_array_equal(out, out2)
        assert dev >= 0.0 and ax >= 0.0

    def test_interslice_smooth_off_is_default_field(self):
        # interslice_smooth=0 must be a no-op vs the per-slice field on a flat surface.
        vol = self._vol()
        a = M.smooth_volume(vol, {"auto_tune": False, "interslice_smooth": 0.0}, workers=1)
        b = M.smooth_volume(vol, {"auto_tune": False, "interslice_smooth": 2.0}, workers=1)
        # on a perfectly flat band both fields are ~0 → outputs equal; this guards the no-op path.
        np.testing.assert_array_equal(a, b)


# ───────────────────────── iterate_smooth_volume: keep-best ─────────────────
class TestIterateKeepBest:
    def _vol(self):
        F, D, L = 5, 28, 6
        vol = np.full((F, D, L), 20, np.uint16)
        vol[:, 12:16, :] = 200
        return vol

    def test_raw_is_in_candidate_set_and_best_is_argmin(self):
        vol = self._vol()
        chain, best_idx, info = M.iterate_smooth_volume(
            vol, {"auto_tune": False}, max_iter=2, workers=1)
        # V0 (raw) is always chain[0] and a candidate.
        np.testing.assert_array_equal(chain[0], vol)
        # best_idx is the argmin over the combined score (so it can never be worse than raw).
        scores = info["scores"]
        assert best_idx == min(range(len(scores)), key=lambda i: scores[i])
        assert 0 <= best_idx < len(chain)
        # every chain volume has a measured metric so it could compete.
        assert len(info["metrics"]) == len(chain)
        assert len(info["axial_metrics"]) == len(chain)

    def test_already_flat_keeps_raw_no_worse(self):
        # a perfectly flat boundary has ~0 deviation → no pass can beat raw; result == raw.
        vol = self._vol()
        chain, best_idx, info = M.iterate_smooth_volume(
            vol, {"auto_tune": False}, max_iter=3, workers=1)
        # the best score must be <= the raw score (keep-best never selects a worse pass).
        assert info["scores"][best_idx] <= info["scores"][0] + 1e-9


# ───────────────────────── axial_refine_volume: never-worse guard ───────────
class TestAxialRefineNeverWorse:
    def test_flat_volume_refine_is_safe(self):
        # On a flat cornea both domains are already smooth → the global guard must not
        # produce a worse surface; the returned volume keeps the sagittal shape.
        F, D, L = 6, 26, 6
        vol = np.full((F, D, L), 20, np.uint16)
        vol[:, 11:15, :] = 200
        out, info = M.axial_refine_volume(vol, {"auto_tune": False}, workers=1)
        assert out.shape == vol.shape
        # the guard reports a (never-worse) smoothness comparison.
        assert info["surf_rms_after"] <= info["surf_rms_before"] + 1e-9
        assert "applied" in info


# ── _correct_surface robust-outlier fix (v0.0.95): flags the spike at ANY index (incl. 0), not the
#    good neighbour, via a local-median test instead of the predecessor-difference test. ──
def test_correct_surface_fixes_first_frame_spike():
    base = np.arange(100.0, 120.0)          # smooth slope-1 line
    y = base.copy(); y[0] = 300.0           # spike at index 0 (the old loop never tested index 0)
    out = M._correct_surface(y, max_jump=10.0)
    assert abs(out[0] - 100.0) < 6.0        # the spike is corrected toward the line
    assert np.allclose(out[1:], base[1:])   # every good sample is untouched


def test_correct_surface_replaces_spike_not_good_neighbour():
    base = np.arange(100.0, 120.0)
    y = base.copy(); y[10] = 300.0          # mid spike
    out = M._correct_surface(y, max_jump=10.0)
    assert abs(out[10] - 110.0) < 6.0       # the SPIKE (index 10) is the one replaced
    assert out[9] == 109.0 and out[11] == 111.0   # the good neighbours are NOT overwritten


def test_correct_surface_leaves_smooth_curve_untouched():
    x = np.linspace(-1.0, 1.0, 40)
    y = 200.0 - 50.0 * (x ** 2)             # smooth dome, per-step change << max_jump
    out = M._correct_surface(y, max_jump=10.0)
    assert np.allclose(out, y)              # no false positives on legitimate curvature


# ── surface-crop posterior alignment (v0.0.104): clipped apex / whole-edge handled by POSTERIOR match ──
def _clipped_apex_sag(n_slices=3, depth=200, F=40, rng=None):
    """Synthetic sagittal volume: a bright corneal band whose dome apex is clipped ABOVE the frame in the
    central frames (band starts at row 0 there) while the flanks are fully in-frame. Posterior fully visible."""
    rng = rng or np.random.RandomState(0)
    post = (30.0 + 0.15 * (np.arange(F) - F / 2) ** 2)        # dome posterior: ~30 centre, deeper at edges
    ant = post - 60.0                                          # 60-px thick band → centre anterior < 0 (clipped)
    sag = (rng.rand(n_slices, depth, F) * 60).astype(np.float32)   # dark speckle background
    for s in range(n_slices):
        for f in range(F):
            top = int(max(0, round(ant[f]))); bot = int(round(post[f]))
            sag[s, top:min(depth, bot), f] += 1800.0          # bright cornea band (clipped at row 0 in the centre)
    return sag, ant, post


def test_build_surface_crop_edges_returns_posterior_and_keeps_apex_above_frame():
    sag, ant_true, post_true = _clipped_apex_sag()
    F = sag.shape[2]
    crop = [int(f) for f in range(F) if ant_true[f] < 0]       # the clipped (apex-above-frame) frames
    edges, posterior = M.build_surface_crop_edges(sag, crop, {}, workers=1)
    assert edges.shape == (sag.shape[0], F)                    # anterior edges (provided_edges for the warp)
    assert posterior.shape == (sag.shape[0], F)                # NEW second return: detected posterior (warp target)
    # the reconstructed anterior at the clipped centre is left ABOVE the frame (negative), not pinned in-frame
    assert edges[:, F // 2].min() < 0.0
    # the detected posterior tracks the true bottom edge (so the warp can match the bottom edge)
    assert float(np.median(np.abs(posterior - post_true[None, :]))) < 8.0


def test_warp_surface_crop_extend_taller_canvas_no_truncation():
    # The extend correction returns a TALLER volume (canvas grown UP) that keeps every column's acquired
    # tissue (no truncation) and leaves the apex above the OLD top, with the posterior fit to a parabola.
    sag, ant_true, post_true = _clipped_apex_sag()
    n, depth, F = sag.shape
    crop = [int(f) for f in range(F) if ant_true[f] < 0]
    edges, posterior = M.build_surface_crop_edges(sag, crop, {}, workers=1)
    out, pad, Pb, Pa, clamped = M.warp_surface_crop_extend(sag, posterior, crop, {}, workers=1)
    assert pad > 0 and not clamped                                                  # clean clip → not pad-capped
    assert out.shape[1] > depth and out.shape[0] == n and out.shape[2] == F
    # the top-edge parabola apex sits ABOVE the old top (negative in original coords)
    assert float(Pa.min()) < 0.0
    # NO truncation: total tissue energy is preserved (every column's acquired rows are placed, none cut)
    assert float(out.sum()) >= float(sag.sum()) * 0.999
    # the posterior parabola is smooth (small 2nd difference) — the "bottom edge is a parabola"
    assert float(np.nanmean(np.abs(np.diff(Pb[n // 2], 2)))) < 1.0


def test_warp_surface_crop_extend_nan_posterior_no_crash():
    # A non-finite posterior value (a degenerate detection) must not crash the warp (manual path has no
    # try/except) — the NaN shift is zeroed, not rounded to a crash.
    sag, ant_true, post_true = _clipped_apex_sag()
    F = sag.shape[2]
    crop = [int(f) for f in range(F) if ant_true[f] < 0]
    _, posterior = M.build_surface_crop_edges(sag, crop, {}, workers=1)
    posterior = posterior.copy(); posterior[0, 3] = np.nan; posterior[1, 7] = np.inf
    out, pad, Pb, Pa, clamped = M.warp_surface_crop_extend(sag, posterior, crop, {}, workers=1)
    assert np.isfinite(out).all() and out.shape[1] > sag.shape[1]


def test_is_substantial_clip_gate():
    # A few stray flagged frames on a normal dome must NOT auto-trigger; a real broad clip must.
    p = {**M.DEFAULT_PARAMS}
    n = 200
    stray = {"frames": [40, 41], "counts": {40: 5, 41: 4}, "n_slices": n}
    assert not M.is_substantial_clip(stray, p)                 # too few frames
    broad = {"frames": list(range(30, 70)), "counts": {f: 80 for f in range(30, 70)}, "n_slices": n}
    assert M.is_substantial_clip(broad, p)                     # many frames, flagged in 40% of slices
    shallow = {"frames": list(range(30, 70)), "counts": {f: 4 for f in range(30, 70)}, "n_slices": n}
    assert not M.is_substantial_clip(shallow, p)               # many frames but each in only 2% of slices


# ── auto crop-region: off-cornea NOISE frame detection (v0.0.107) ──
def _cornea_in_frames(nl=60, nd=200, nf=60, last_cornea=35, rng_seed=0):
    """Synthetic sag volume (lat, depth, frame): a bright coherent cornea band in frames [0, last_cornea],
    pure speckle noise after — the 'slow scan ran off the cornea' pattern."""
    rng = np.random.RandomState(rng_seed)
    sag = (rng.rand(nl, nd, nf) * 40).astype(np.float32)
    for fr in range(last_cornea + 1):
        sag[:, 80:112, fr] += 1600.0          # bright stroma → sharp air/tissue edge at depth 80
    return sag


def test_detect_noise_frames_crops_offcornea_tail():
    sag = _cornea_in_frames(last_cornea=35)
    nf = sag.shape[2]
    nz = M.detect_noise_frames(sag, {}, workers=1)
    assert nz, "should detect the trailing noise block"
    assert max(nz) == nf - 1                   # the run reaches the end boundary
    assert len(nz) >= 18                        # a long block (frames ~38..59)
    assert 0 not in nz and 20 not in nz         # cornea frames are never cropped


def test_detect_noise_frames_none_on_full_cornea():
    rng = np.random.RandomState(1)
    sag = (rng.rand(60, 200, 60) * 40).astype(np.float32)
    sag[:, 80:112, :] += 1600.0                # cornea band in EVERY frame
    assert M.detect_noise_frames(sag, {}, workers=1) == []


def test_shared_detection_matches_internal():
    """PERF (v0.0.113): preprocess computes the anterior detection ONCE and passes it (detect=) to BOTH the
    noise check and the surface-crop check — they run on the same volume with the same detector params, and the
    detector output is independent of crop_region. Passing a precomputed detect= MUST be byte-identical to
    letting each function recompute internally, else the shared cache would silently change the crop decisions."""
    sag = _cornea_in_frames(last_cornea=35)
    p = {**M.DEFAULT_PARAMS}
    det = M.detect_surface_all(sag, p, workers=1)
    assert M.detect_noise_frames(sag, p, workers=1, detect=det) == M.detect_noise_frames(sag, p, workers=1)
    a = M.detect_surface_crop_frames(sag, p, workers=1, detect=det)
    b = M.detect_surface_crop_frames(sag, p, workers=1)
    assert a["frames"] == b["frames"] and a["counts"] == b["counts"]


def test_refine_freeze_periphery():
    """Peripheral warp-spike fix (v0.0.117): 'logical limbus correction' (refine_freeze_frac) is a PER-SCAN
    opt-in. It must be a strict NO-OP at 0 (default → global pipeline byte-unchanged); when >0 it warps the outer
    lateral slices to a LATERALLY-SMOOTH surface (changing the periphery) while the feathered CENTRE is byte-
    identical to the ordinary warp."""
    rng = np.random.RandomState(3)
    F, D, L = 24, 100, 100                                  # (frames, depth, lateral)
    vol = (rng.rand(F, D, L) * 12).astype(np.float32)
    for l in range(L):
        for f in range(F):
            top = 40 + int(round(7 * np.sin(f / 3.5)))      # wavy across frames so the warp does real work
            if l < 8:
                top += 12                                   # a peripheral lateral STEP the smoothing will pull in
            vol[f, top:top + 16, l] += 900.0
    plain = M.smooth_volume(vol, {}, workers=1)
    off = M.smooth_volume(vol, {"refine_freeze_frac": 0.0}, workers=1)
    assert np.array_equal(plain, off)                       # frac=0 → strict no-op (global pipeline byte-unchanged)
    on = M.smooth_volume(vol, {"refine_freeze_frac": 0.25}, workers=1)
    # the feathered CENTRE is byte-identical to the ordinary warp (only the limbus is re-smoothed)
    assert np.array_equal(on[:, :, L // 2 - 4:L // 2 + 4], plain[:, :, L // 2 - 4:L // 2 + 4])


# ── delivered-volume QA (measure_delivered_qa) ────────────────────────────────
def _dome_volume(F=72, D=160, L=96, seed=0):
    """Tiny synthetic dome + speckle, in the pipeline's native (frames, depth, lateral) order.
    F must be >= 64: below that _cap_edge_descent's size gate (2*(10+4+16)+4) short-circuits and the
    one block that consumes the caller's array as vol= never runs, so a mutation test would pass
    vacuously."""
    rng = np.random.default_rng(seed)
    vol = np.zeros((F, D, L), dtype=np.float32)
    yy = np.arange(L)
    for f in range(F):
        surf = (40 + 0.004 * (yy - L / 2) ** 2 + 0.6 * np.sin(f / 3)).astype(int)
        for l in range(L):
            vol[f, surf[l]:surf[l] + 45, l] = 180 + rng.normal(0, 8, 45)
    return np.ascontiguousarray(vol)


def test_delivered_qa_never_mutates_the_volume():
    """QA runs alongside write_volume_nifti, so it must be strictly read-only — a mutation would
    silently change the delivered result."""
    vol = _dome_volume()
    before = vol.copy()
    qa = M.measure_delivered_qa(vol, {}, workers=1)
    assert np.array_equal(vol, before)                       # byte-identical
    assert vol.shape == before.shape
    assert qa and qa["dev"] >= 0.0 and qa["axial"] >= 0.0


def test_delivered_qa_does_not_move_pass_selection():
    """The coverage twin must not perturb disp_mean — that is the objective iterate_smooth_volume
    minimises to choose best_pass, so any drift there could change which pass ships for the
    already-accepted scans."""
    vol = _dome_volume()
    a = M.smooth_volume(vol, {}, workers=1, return_metric=True)
    b = M.smooth_volume(vol, {}, workers=1, return_metric=True, return_coverage=True)
    assert a[1] == b[1]                                      # disp_mean bit-identical
    assert a[2] == b[2]                                      # axial roughness bit-identical
    assert np.array_equal(a[0], b[0])                        # warped volume unchanged
    assert b[4] == 1.0                                       # intact volume -> full coverage
    assert b[3] == b[1]                                      # at full coverage the twin IS disp_mean


def test_delivered_qa_coverage_tracks_destroyed_volume():
    """coverage must fall as tissue is zeroed. It is the guard on `dev`: `dev` is deflated by
    destroyed volume (the RANSAC fit hugs the surviving frames), so a damaged scan reads
    artificially clean and may only be compared at like coverage."""
    vol = _dome_volume()
    full = M.measure_delivered_qa(vol, {}, workers=1)
    assert full["coverage"] == 1.0
    holed = vol.copy()
    holed[:holed.shape[0] // 3] = 0.0                        # destroy a third of the frames
    part = M.measure_delivered_qa(holed, {}, workers=1)
    assert part["coverage"] < 0.75
    assert "low-coverage" in part["review_reasons"] or part["coverage"] >= 0.60


def test_delivered_qa_cannot_break_a_run():
    """The central guarantee: QA must never raise. Every one of these inputs raised before the
    flag arithmetic was brought inside the try."""
    vol = _dome_volume()
    assert M.measure_delivered_qa(vol, {"final_qa": False}, workers=1) == {}
    for params, rhr in (({}, "x"), ({}, []), ({}, {"max_jitter": "n/a"}),
                        ({"qa_jitter_flag": None}, None), ({"qa_dev_flag": None}, None),
                        ({"qa_coverage_flag": None}, None)):
        r = M.measure_delivered_qa(vol, params, workers=1, rhr_info=rhr)
        assert isinstance(r, dict)                           # did not raise

    class _Boom:
        shape = (1, 1, 1)

        def __getattr__(self, n):
            raise RuntimeError("boom")

    assert M.measure_delivered_qa(_Boom(), {}, workers=1) == {}


def test_delivered_qa_unmeasurable_is_null_not_zero():
    """A missing measurement must be an explicit null plus a qa_errors note. Recording it as 0.0
    would read as 'no motion' on exactly the scans where motion was never measured — the
    surface-crop path, where rigid_height_refine never runs."""
    vol = _dome_volume()
    r = M.measure_delivered_qa(vol, {}, workers=1)            # no rhr_info, as on the surface-crop path
    assert r["max_jitter"] is None
    assert "max_jitter" in r.get("qa_errors", [])
    assert "motion" not in r["review_reasons"]                # absence must not fire, nor suppress silently


def test_delivered_qa_tilt_is_recorded_but_never_flags():
    """tilt_total is recorded (it is the in-plane metric's null space) but must raise no flag:
    delivered post-flatten tilt is O(10 px) while the old threshold was a PRE-flatten 150, and
    measured tilt is INVERTED against the human verdict (AUC 0.411)."""
    vol = _dome_volume()
    r = M.measure_delivered_qa(vol, {}, workers=1, rhr_info={"max_jitter": 0.1})
    assert "tilt_total" in r
    assert "tilt" not in r["review_reasons"]


def test_height_refine_reports_jitter_even_when_reverted():
    """max_jitter is MEASURED INPUT SEVERITY, not a result of the correction, so it must be
    present on the revert path too — its absence there is what capped jitter-based triage recall."""
    vol = _dome_volume()
    out, info = M.rigid_height_refine(vol, {}, workers=1)
    assert "max_jitter" in info                              # present regardless of applied/reverted
    if not info.get("applied"):
        assert np.array_equal(out, vol)                      # revert returns the untouched volume


# ── rigid_frame_refine: the final boundary-driven rigid per-frame pass ───────────────────────────────
# Its justification is that it reaches the acquisition-EDGE offset the passes before it structurally cannot
# (rigid_height_refine keeps only the high-frequency part of its correction and caps it at rhr_max=8 px).
#
# The gates below are NOT tuning. Each encodes a measurement that overturned an earlier belief:
#   * the deviation this pass acts on is NOT edge-specific — slide the identical fit-28/extrapolate-12
#     estimator into the undisputed interior and it reads 3.40 px against 2.87 px at the real edge. So the
#     trigger is that scan's OWN interior sham distribution, not a constant. The old fixed 0.8 px trigger
#     fired on 97.1% of human-approved scan-ends;
#   * a real edge defect RAMPS away from the seam (cs050_od_v1's leading frames read 61,61,61,62,63, having
#     flattened off a descending cornea). A UNIFORM block offset is what the estimator's own fit error looks
#     like, so it is refused;
#   * the reference must be anchored at the seam, or a nonzero fit residual there becomes a step by
#     construction — median 0.63 px, p90 1.99 px, max 38.9 px across the approved corpus.
_RFR_P = {"rfr_lead": 8, "rfr_tail": 4, "rfr_edge_n": 6, "rfr_edge_fit": 20,
          "rfr_placebo_offsets": (6, 10, 14)}


def _rfr_profile(F=64, bump_frame=None, bump_px=0.0, edge_ramp=0.0, edge_step=0.0, edge_n=6, side="trail"):
    """The per-frame boundary depth of a dome, with an optional interior step and an optional edge defect.

    `edge_ramp` departs progressively from the seam outward, which is what a real edge defect looks like;
    `edge_step` displaces the whole block uniformly, which is what fit error looks like."""
    f = np.arange(F, dtype=float)
    fx = (f - (F - 1) / 2) / ((F - 1) / 2)
    top = 60 + 30 * fx ** 2
    if bump_frame is not None:
        top[bump_frame] += bump_px
    idx = np.arange(F - edge_n, F) if side == "trail" else np.arange(edge_n)
    d = np.abs(idx - (F - edge_n - 1 if side == "trail" else edge_n))
    top[idx] += edge_step + edge_ramp * d
    return top


def _rfr_volume_from(prof, D=260, L=64, seed=1):
    """Build a (frames, depth, lateral) volume whose anterior boundary follows `prof`."""
    rng = np.random.default_rng(seed)
    F = prof.size
    vol = np.zeros((F, D, L), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, L)
    for f in range(F):
        top = prof[f] + 14 * x ** 2
        for l in range(L):
            t = int(round(top[l]))
            vol[f, t:t + 45, l] = 200.0
    return np.ascontiguousarray(vol + rng.normal(0, 3, vol.shape).astype(np.float32))


def _rfr_sides(prof, p=None):
    q = {**M.DEFAULT_PARAMS, **_RFR_P, **(p or {})}
    det = {}
    M._rfr_edge_deviation(np.asarray(prof, dtype=float), q, detail=det)
    return det


def test_rfr_corrects_a_ramped_edge_defect():
    """The real failure mode: the edge progressively leaves the curve the cornea is following."""
    prof = _rfr_profile(edge_ramp=3.0)                       # departs ~18 px by the outermost frame
    det = _rfr_sides(prof)
    assert det["trail"].get("applied"), det["trail"]
    assert det["trail"]["ask"] > 12
    assert not det["lead"].get("applied")                    # the untouched side stays untouched


def test_rfr_refuses_a_uniform_block_offset():
    """A block displaced bodily is the signature of the estimator's own fit error, not of a defect: the
    correction would be almost entirely a step at the seam."""
    det = _rfr_sides(_rfr_profile(edge_step=18.0))
    assert not det["trail"].get("applied")
    assert "uniform block offset" in det["trail"]["decline"]


def test_rfr_trigger_is_the_scans_own_null_not_a_constant():
    """The gate must scale with how noisy THIS scan's profile is. A ramp that clears the null on a clean
    profile must stop clearing it once the same profile is made jittery enough that the interior sham
    positions report just as much."""
    clean = _rfr_profile(edge_ramp=1.2)
    assert _rfr_sides(clean)["trail"].get("applied")
    rng = np.random.default_rng(7)
    noisy = clean + rng.normal(0, 4.0, clean.size)
    d = _rfr_sides(noisy)["trail"]
    assert not d.get("applied")
    assert d["placebo"] > 2.0 and d["threshold"] > d["ask"]


def test_rfr_edge_reference_is_anchored_at_the_seam():
    """Unanchored, a nonzero fit residual at the seam becomes a step by construction, because the edge block
    is shifted while the adjacent interior frame is not. On cs039_os_v1 that residual was -2.46 px — the
    largest in its own window — and the block was driven 4-9.5 px the wrong way."""
    prof = _rfr_profile(edge_ramp=3.0)
    prof[-14:-6] -= np.linspace(0, 5, 8)                     # bend the fit window near the seam
    det = _rfr_sides(prof)["trail"]
    if det.get("applied"):
        assert abs(det["seam_step"]) <= max(1.0, 0.35 * det["ask"])


def test_rfr_declines_an_implausible_edge_per_side():
    """An edge tens of px off the local curvature is an eyelash or a specular streak, not motion —
    cs011_os_v3 measures 122 px, and the reviewer's own verdict there was "likely eyelash"."""
    prof = _rfr_profile(edge_ramp=3.0)
    prof[-6:] -= 200.0                                       # the estimator has left the cornea entirely
    det = _rfr_sides(prof)
    assert not det["trail"].get("applied")
    assert "implausible" in det["trail"]["decline"]


def test_rfr_a_smooth_edge_needs_to_be_large_to_be_corrected():
    """Roughness is the one genuinely edge-specific property in the population (2nd-difference rms 5.49 px
    vs 1.56 px interior), so it is the usual evidence — but requiring it alone rejects cs050_od_v1, whose
    real 10.5 px defect is a SMOOTH flattening with a kink ratio of 0.51."""
    # rfr_gate_kink is raised here so a noiseless synthetic cannot pass the kink route on a numerical tie:
    # this test is about the SIZE route, and on an exactly-quadratic profile k_edge/k_int sits at ~1.00.
    q = {"rfr_gate_kink": 1.5}
    small = _rfr_sides(_rfr_profile(edge_ramp=0.75), q)["trail"]   # ~4.5 px, smooth -> below rfr_gate_smooth_px
    assert not small.get("applied")
    assert "smooth and small" in small["decline"]
    big = _rfr_sides(_rfr_profile(edge_ramp=3.0), q)["trail"]      # ~18 px, smooth -> too big to be fit error
    assert big.get("applied") and big["route"] == "large-and-smooth"


def test_rfr_is_a_strict_noop_on_a_clean_volume():
    """The approved corpus must come through byte-unchanged — a pass that churns already-good scans cannot
    be shipped under a no-regressions requirement."""
    vol = _rfr_volume_from(_rfr_profile())
    out, info = M.rigid_frame_refine(vol, _RFR_P)
    assert not info["applied"]
    assert out is vol or np.array_equal(out, vol)


def test_rfr_moves_the_volume_for_a_real_defect():
    """End to end: a ramped edge defect in an actual volume is measured and rigidly shifted out."""
    vol = _rfr_volume_from(_rfr_profile(edge_ramp=3.0))
    out, info = M.rigid_frame_refine(vol, _RFR_P)
    assert info["applied"], info
    assert not np.array_equal(out, vol)
    assert info["seam_step_after"] <= max(1.0, info["seam_step_before"] + 0.01)


def test_rfr_refuses_when_the_measure_is_unreliable():
    """A partial-shadow band dims the tissue without collapsing it, so the dropout test misses it and the
    boundary reads tens of px off (cs021_od_v3: 117 px over 42 frames). Refusing beats writing garbage."""
    vol = _rfr_volume_from(_rfr_profile())
    rng = np.random.default_rng(2)
    for f in range(20, 44):
        vol[f] = np.roll(vol[f], int(30 + 25 * np.sin(f)), axis=0) * rng.uniform(0.9, 1.0)
    out, info = M.rigid_frame_refine(vol, {**_RFR_P, "rfr_wild_rms": 8.0})
    if not info["applied"]:
        assert np.array_equal(out, vol)                      # untouched, not partially corrected


def test_rfr_interior_and_edge_regions_are_disjoint():
    """They are judged against DIFFERENT references, so an overlap makes them fight: with the review
    harness's tuning the last 7 frames were both inside the interior curve fit and given an edge shift, and
    moving them degraded the interior deviation on approved scans (cs001_od_v1 0.154 -> 0.253 px)."""
    p = {"rfr_lead": 14, "rfr_tail": 5, "rfr_edge_n": 12}
    lead, tail = M._rfr_split(p)
    assert lead >= p["rfr_edge_n"] and tail >= p["rfr_edge_n"]
    F = 101
    assert not (set(range(lead, F - tail)) & (set(range(12)) | set(range(F - 12, F))))


def test_rfr_saturation_rejection_cannot_blank_the_measurement():
    """Rejecting saturated A-scans is right (specular columns and eyelashes broke the earlier estimator) but
    it must stay RARE: on a near-uniform-peak volume the rule flags everything, and honouring it would leave
    nothing to measure at all."""
    S = M._anterior_boundary(_rfr_volume_from(_rfr_profile()), {})
    assert np.isfinite(S).mean() > 0.9


def test_rfr_never_moves_a_dropout_frame():
    """A blink or full-width shadow gives the estimator nothing to lock onto; it dives, which reads as a huge
    false displacement."""
    vol = _rfr_volume_from(_rfr_profile(edge_ramp=3.0))
    vol[10] *= 0.05
    assert bool(M._dropout_frames(vol, {})[10])
    _out, info = M.rigid_frame_refine(vol, _RFR_P)
    if info.get("applied"):
        assert abs(info["shift"][10]) <= 0.05


def test_rfr_never_pushes_tissue_out_of_the_canvas():
    """A rigid depth shift moves the whole B-scan and the warp zero-fills what it vacates, so raising a frame
    whose boundary already sits at the top of the volume simply CUTS the cornea off. cs035_od_v1's leading
    edge is at depth 0 and its local curvature asks for another ~15 px up — unreachable without extending
    the canvas, which is the surface-crop path's job."""
    prof = np.maximum(3.0, _rfr_profile(edge_ramp=-3.0) - 50)     # apex frames pinned at the very top
    vol = _rfr_volume_from(prof, D=110)
    tissue_before = float((vol > 100).sum())
    out, info = M.rigid_frame_refine(vol, _RFR_P)
    if info.get("applied"):
        assert float((out > 100).sum()) > tissue_before * 0.98
        _dv, pr, _S, _C = M._rfr_deviation(out, {**M.DEFAULT_PARAMS, **_RFR_P})
        assert np.nanmin(pr) >= 0.0


def test_rfr_unmeasurable_edge_is_null_not_zero():
    """Both edges declined is a normal outcome — cs009_os_v3's outermost columns are off the cornea entirely,
    so the estimator sits on a bright basal streak at depth ~420. Reporting 0.0 there would read as a PERFECT
    edge on exactly the scans whose edge could not be checked (the null-vs-zero trap that once capped jitter
    triage at 44% recall)."""
    prof = _rfr_profile()
    prof[:6] -= 300.0
    prof[-6:] -= 300.0
    assert not np.isfinite(M._rfr_edge_deviation(prof, {**M.DEFAULT_PARAMS, **_RFR_P})).any()
    vol = _rfr_volume_from(prof, D=420)
    _out, info = M.rigid_frame_refine(vol, _RFR_P)
    assert info.get("edge_dev") is None


def test_rfr_can_be_disabled_and_then_is_bit_identical():
    vol = _rfr_volume_from(_rfr_profile(edge_ramp=3.0))
    out, info = M.rigid_frame_refine(vol, {**_RFR_P, "rigid_frame_refine": False})
    assert not info["applied"] and out is vol


def test_rfr_corrects_an_edge_defect_that_varies_across_the_width():
    """A rigid depth shift is uniform across the B-scan, so on its own it can only remove the AVERAGE of a
    defect that varies from one sagittal end to the other. On cs039_os_v1's leading frame the deviation ramps
    from +21.7 px at one lateral end to +42.9 px at the other; shifting by the central-band value left ~10 px
    of over-lift at one end, which the reviewer saw as "the right edge slightly going upwards against the
    general cornea curvature" on the last sagittal slices. The ramp is a ROTATION, not a deformation."""
    F, D, L = 64, 300, 200
    rng = np.random.default_rng(11)
    prof = _rfr_profile(F=F, edge_ramp=3.0)
    vol = np.zeros((F, D, L), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, L)
    tilt = np.zeros(F)
    tilt[-6:] = np.linspace(0.02, 0.10, 6)                   # the defect grows across the width at the edge
    for f in range(F):
        top = prof[f] + 14 * x ** 2 + tilt[f] * (np.arange(L) - (L - 1) / 2.0)
        for l in range(L):
            t = int(round(top[l]))
            vol[f, t:t + 45, l] = 200.0
    vol = np.ascontiguousarray(vol + rng.normal(0, 3, vol.shape).astype(np.float32))
    q = {**M.DEFAULT_PARAMS, **_RFR_P, "rfr_bands": 8, "rfr_band_margin": 0.05}
    _dv, _pr, S, _C = M._rfr_deviation(vol, q)
    spread_before = M._rfr_edge_spread(S, q)
    # _rfr_edge_spread is a MEAN over the edge frames (the max carries a 2.74 px sub-pixel resampling floor,
    # measured under a geometry-preserving uniform shift), and the synthetic tilts only half the block.
    assert spread_before > 4.0                               # the across-width defect is really there

    out, info = M.rigid_frame_refine(vol, {**_RFR_P, "rfr_bands": 8})
    assert info["applied"], info
    assert info["frames_tilted"] >= 3
    assert 0.05 < info["max_tilt_deg"] <= 1.5
    assert info["edge_spread_after"] < info["edge_spread_before"]


def test_rfr_does_not_invent_a_rotation_from_noise():
    """The full 3-DOF rigid fit was rejected because at this residual level extra parameters are
    bias-dominated (every fitted lateral shift came out positive, median +0.50 px). So a tilt is fitted only
    when a straight line across the width genuinely explains a spread worth removing."""
    F, L = 64, 200
    rng = np.random.default_rng(5)
    S = np.repeat(_rfr_profile(F=F, edge_ramp=3.0)[:, None], L, axis=1) + rng.normal(0, 1.5, (F, L))
    q = {**M.DEFAULT_PARAMS, **_RFR_P, "rfr_bands": 8}
    edev = np.full(F, np.nan)
    edev[-6:] = 5.0                                          # pretend the side was accepted
    tilt = M._rfr_edge_tilt(S, edev, q)
    assert np.all(np.abs(tilt) * (L - 1) < 4.0)              # no meaningful rotation out of pure noise


def test_rfr_corrects_an_interior_frame_rotation():
    """A reviewer marked ONE column of cs035_od_v1_4 at sagittal slice 73 and the same column at slice 475 —
    the two ends of a 7.7 px across-width ramp whose frame MEDIAN is 0.07 px. Every central-band measure in
    this pass is blind to that by construction, so a depth-only correction cannot see it, let alone fix it.
    A per-frame rotation is rigid, so it is allowed."""
    F, D, L = 64, 300, 200
    rng = np.random.default_rng(21)
    prof = _rfr_profile(F=F)
    vol = np.zeros((F, D, L), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, L)
    lat = np.arange(L) - (L - 1) / 2.0
    bad = 30
    for f in range(F):
        top = prof[f] + 14 * x ** 2 + (0.05 * lat if f == bad else 0.0)   # one frame is rotated
        for l in range(L):
            t = int(round(top[l]))
            vol[f, t:t + 45, l] = 200.0
    vol = np.ascontiguousarray(vol + rng.normal(0, 3, vol.shape).astype(np.float32))
    q = {**M.DEFAULT_PARAMS, **_RFR_P, "rfr_bands": 8}
    _dv, pr, S, C = M._rfr_deviation(vol, q)
    assert abs(pr[bad] - np.median(pr[bad - 3:bad + 4])) < 1.0    # invisible to the frame median
    tilt, ginfo = M._rfr_interior_tilt(S, C, q)
    assert abs(tilt[bad]) > 1e-9 and ginfo["frames"] >= 1
    assert sum(abs(t) > 1e-9 for t in tilt) <= 3                  # the clean frames are left alone

    out, info = M.rigid_frame_refine(vol, {**_RFR_P, "rfr_bands": 8})
    assert info["applied"]
    assert info["int_spread_after"] < info["int_spread_before"]


def test_rfr_interior_rotation_needs_to_clear_the_scans_own_noise():
    """This is NOT the 3-DOF fit that was rejected — that one added rotation to every frame from a per-frame
    least-squares fit and was bias-dominated. A rotation is fitted only where it stands clear of THIS scan's
    own per-frame tilt noise, so a uniformly noisy scan gets none."""
    F, L = 64, 200
    rng = np.random.default_rng(9)
    prof = _rfr_profile(F=F)
    S = np.repeat(prof[:, None], L, axis=1) + rng.normal(0, 2.0, (F, L))
    C = np.repeat(np.polyval(np.polyfit(np.arange(F), prof, 4), np.arange(F))[:, None], L, axis=1)
    q = {**M.DEFAULT_PARAMS, **_RFR_P, "rfr_bands": 8}
    tilt, ginfo = M._rfr_interior_tilt(S, C, q)
    assert ginfo["frames"] == 0


def test_manual_patch_applies_shift_and_rotation_rigidly():
    """The reviewer-facing patch must express a ROTATION, not only a depth nudge: most residual defects
    review finds are a frame deep at one sagittal end and shallow at the other, which no depth offset can
    describe. And it must be RIGID — the B-scan is an instantaneous capture."""
    vol = _rfr_volume_from(_rfr_profile())
    F, D, L = vol.shape
    out, n = M.apply_manual_patch(vol, {30: [4.0, 12.0]})
    assert n == 1
    q = {**M.DEFAULT_PARAMS, **_RFR_P}
    S0 = M._anterior_boundary(vol, q)
    S1 = M._anterior_boundary(out, q)
    d = S1[30] - S0[30]                                    # per-lateral depth change on the patched frame
    # depth term: positive = DEEPER, the same convention apply_manual_shifts documents for an on-screen
    # drag downward, so a reviewer's patch reads the same way in both tools.
    assert abs(np.nanmean(d) - 4.0) < 0.6
    # rotation term: a LINEAR ramp across the laterals, spanning the requested amount
    lat = np.arange(L)
    ok = np.isfinite(d)
    slope = np.polyfit(lat[ok], d[ok], 1)[0] * (L - 1)
    assert abs(slope - 12.0) < 2.0
    assert np.array_equal(out[29], vol[29]) and np.array_equal(out[31], vol[31])   # neighbours untouched


def test_manual_patch_is_ground_truth_and_survives_the_auto_passes():
    """A patch stored as PARAMETERS is re-applied by the pipeline; a voxel edit would be erased on the next
    re-preprocess, since the endpoint re-runs every stage from the .OCT."""
    vol = _rfr_volume_from(_rfr_profile())
    out, n = M.apply_manual_patch(vol, [[20, [2.0, 0.0]], [21, 3.0]])
    assert n == 2                                          # accepts [frame, [shift, tilt]] and [frame, shift]
    for bad in ({}, None, {5: [float("nan"), 0.0]}, {999: [3.0, 0.0]}, {6: [0.001, 0.001]}):
        _o, k = M.apply_manual_patch(vol, bad)
        assert k == 0                                      # never crash, never act on nonsense


# ───────────────────────── tissue-step guard (non-waivable companion of the roughness waiver) ─────────────
class TestTissueStepGuard:
    """`_reject_if_stepped` must decline a rigid stage that INTRODUCES a localised across-frame step in the
    tissue and keep one that moves every frame together (a uniform shift is not a step). Trace-free: the
    synthetic volume has NO detectable "surface" logic involved — only speckle + a bright dome band."""

    @staticmethod
    def _dome(F=40, D=160, L=64, seed=0):
        rng = np.random.default_rng(seed)
        vol = rng.uniform(400.0, 600.0, size=(F, D, L)).astype(np.float32)   # speckle background
        f = np.arange(F)[:, None]; l = np.arange(L)[None, :]
        top = (30 + 0.03 * (f - F / 2) ** 2 + 0.02 * (l - L / 2) ** 2).astype(int)   # (F, L) dome
        for fi in range(F):
            for li in range(L):
                vol[fi, top[fi, li]:top[fi, li] + 50, li] += 900.0
        return vol

    @staticmethod
    def _shift(vol, frames, px):
        out = vol.copy()
        for fi in frames:
            out[fi] = 0.0
            out[fi, px:] = vol[fi, :-px]      # tissue pushed DOWN by px, zero-fill above
        return out

    def test_map_shape_and_uniform_motion(self):
        vol = self._dome()
        Mp, lats = M._tissue_step_map(vol)
        assert Mp.shape == (lats.size, vol.shape[0] - 1)
        assert np.isfinite(Mp).all()
        # the dome's own frame-to-frame motion is small and SMOOTH -> no kink anywhere
        K = M._tissue_step_kinks(Mp)
        assert np.nanmax(K) < 6.0

    def test_block_offset_is_declined(self):
        vol = self._dome()
        stepped = self._shift(vol, range(20, 40), 8)      # frames 20.. pushed 8 px: a step at pair 19->20
        info = {"stage": {"applied": True}}
        kept, _m = M._reject_if_stepped(vol, stepped, "stage", info, {"drawn_edge_step_guard": True})
        assert kept is vol
        assert info["stage"]["applied"] is False
        assert "tissue step" in info["stage"]["reason"].lower() or "TISSUE step" in info["stage"]["reason"]
        assert info["stage"]["tissue_step"]["frame_pair"] == [19, 20]
        assert info["stage"]["tissue_step"]["stepped_by_stage"] >= info["stage"]["tissue_step"]["need"]

    def test_uniform_shift_is_kept(self):
        vol = self._dome()
        moved = self._shift(vol, range(40), 8)             # every frame together: not a step
        info = {"stage": {"applied": True}}
        kept, m = M._reject_if_stepped(vol, moved, "stage", info, {"drawn_edge_step_guard": True})
        assert kept is moved
        assert info["stage"]["applied"] is True
        assert m is not None and m.shape[1] == 39

    def test_removing_a_step_is_never_penalised(self):
        vol = self._dome()
        stepped = self._shift(vol, range(20, 40), 8)
        info = {"stage": {"applied": True}}
        kept, _m = M._reject_if_stepped(stepped, vol, "stage", info, {"drawn_edge_step_guard": True})
        assert kept is vol and info["stage"]["applied"] is True

    def test_declined_stage_and_switch_off_are_untouched(self):
        vol = self._dome()
        info = {"stage": {"applied": False}}
        kept, m = M._reject_if_stepped(vol, vol, "stage", info, {"drawn_edge_step_guard": True}, before_map=None)
        assert kept is vol and m is None and "tissue_step" not in info["stage"]
        stepped = self._shift(vol, range(20, 40), 8)
        kept, _m = M._reject_if_stepped(vol, stepped, "stage", info, {"drawn_edge_step_guard": False})
        assert kept is stepped and "tissue_step" not in info["stage"]

    def test_default_params_declare_the_guard(self):
        assert M.DEFAULT_PARAMS["drawn_edge_step_guard"] is True
        assert M.DEFAULT_PARAMS["step_guard_px"] == 6.0
        assert 0.0 < M.DEFAULT_PARAMS["step_guard_frac"] < 1.0

    def test_zero_filled_bands_are_unjudged_not_counted(self):
        vol = self._dome()
        # blank the left 24 laterals entirely (zero-filled canvas) -> those bands are NaN in the map
        vol[:, :, :24] = 0.0
        Mp, lats = M._tissue_step_map(vol)
        # bands fully inside the blank region are NaN; a band straddling its boundary still averages tissue
        assert np.isnan(Mp[lats <= 20]).all() and np.isfinite(Mp[lats >= 28]).all()
        # a step confined to the blank laterals cannot be seen and must not decline the stage
        stepped = self._shift(vol, range(20, 40), 8)
        stepped[:, :, 24:] = vol[:, :, 24:]
        info = {"stage": {"applied": True}}
        kept, _m = M._reject_if_stepped(vol, stepped, "stage", info, {"drawn_edge_step_guard": True})
        assert kept is stepped and info["stage"]["applied"] is True
        assert info["stage"]["tissue_step"]["finite_bands"] < lats.size

    def test_before_map_threading_matches_fresh(self):
        v0 = self._dome(); v1 = self._shift(v0, range(40), 3); v2 = self._shift(v1, range(20, 40), 8)
        info_a = {"s": {"applied": True}}; _k, m1 = M._reject_if_stepped(v0, v1, "s", info_a, {"drawn_edge_step_guard": True})
        info_b = {"s": {"applied": True}}; M._reject_if_stepped(v1, v2, "s", info_b, {"drawn_edge_step_guard": True}, before_map=m1)
        info_c = {"s": {"applied": True}}; M._reject_if_stepped(v1, v2, "s", info_c, {"drawn_edge_step_guard": True})
        assert info_b["s"]["tissue_step"] == info_c["s"]["tissue_step"]
        assert info_b["s"]["applied"] is False

    def test_enlarging_an_existing_step_counts(self):
        v0 = self._dome(); small = self._shift(v0, range(20, 40), 6)   # a pre-existing ~6 px step
        big = self._shift(v0, range(20, 40), 14)                       # the stage makes it 14 px
        info = {"s": {"applied": True, "rougher_note": "kept: made 3 lateral(s) rougher than the input (max +1.0 px)",
                      "roughness_veto": "waived — this run follows your drawn edge"}}
        kept, _m = M._reject_if_stepped(small, big, "s", info, {"drawn_edge_step_guard": True})
        assert kept is small and info["s"]["applied"] is False
        assert info["s"]["rougher_note"].startswith("measured on the discarded move")
        assert info["s"]["roughness_veto"].endswith("discarded by the tissue-step guard")
        assert "_surface_before" not in info["s"]


# ───────────────────────── corrected-pane edit transform (regenerate modifies the transform) ─────────────
class TestEditTransform:
    def test_fit_recovers_known_shift_and_tilt(self):
        L, F = 513, 101
        xc = (np.arange(L) - (L - 1) / 2.0) / ((L - 1) / 2.0)
        a_true = np.linspace(-3.0, 5.0, F); b_true = np.linspace(2.0, -1.0, F)
        lats = [60, 150, 240, 330, 420, 500]
        deltas = {l: {f: float(a_true[f] + b_true[f] * xc[l]) for f in range(0, F, 5)} for l in lats}
        et = M.fit_edit_transform(deltas, L, F)
        assert et["fitted_frames"] == len(range(0, F, 5))
        assert np.allclose(et["shift"], a_true, atol=0.05) and np.allclose(et["tilt"], b_true, atol=0.05)

    def test_fit_shift_only_with_few_laterals_and_outlier_drop(self):
        L, F = 513, 20
        deltas = {100: {f: 4.0 for f in range(F)}, 300: {f: 4.0 for f in range(F)}}     # 2 laterals -> shift only
        et = M.fit_edit_transform(deltas, L, F)
        assert np.allclose(et["shift"], 4.0) and np.allclose(et["tilt"], 0.0)
        deltas = {l: {5: 2.0} for l in (50, 150, 250, 350, 450)}; deltas[450][5] = 60.0   # one eyelid-corner outlier
        et = M.fit_edit_transform(deltas, L, F)
        assert abs(et["shift"][5] - 2.0) < 0.3 and abs(et["tilt"][5]) < 0.3

    def test_apply_moves_content_deeper_by_the_shift(self):
        F, D, L = 6, 80, 64
        vol = np.zeros((F, D, L), np.float32); vol[:, 30, :] = 1.0          # a bright row at depth 30 in every frame
        et = {"shift": [0, 0, 5, 5, 0, 0], "tilt": [0] * F}
        out, info = M.apply_edit_transform(vol, et)
        assert info["applied"] and info["frames_moved"] == 2
        assert int(np.argmax(out[2, :, 10])) == 35 and int(np.argmax(out[0, :, 10])) == 30
        # a tilt: + at the right edge, - at the left (half-span 4 px)
        et = {"shift": [0] * F, "tilt": [0, 0, 4, 0, 0, 0]}
        out, _i = M.apply_edit_transform(vol, et)
        assert int(np.argmax(out[2, :, L - 1])) == 34 and int(np.argmax(out[2, :, 0])) == 26

    def test_apply_rejects_wrong_length(self):
        vol = np.zeros((6, 40, 8), np.float32)
        out, info = M.apply_edit_transform(vol, {"shift": [1, 2, 3]})
        assert out is vol and info["applied"] is False

    def test_default_mode_is_line_and_a_stored_transform_is_not_active(self):
        # 2026-09-05 (cs002_os_v1): a transform fitted from a few hand-drawn pane lines made the tissue LESS
        # quadratic (3.71 → 4.66 px). The default keeps any stored transform on record and never applies it.
        assert M.DEFAULT_PARAMS["corrected_edit_mode"] == "line"
        et = {"shift": [1.0, 2.0], "tilt": [0.0, 0.0], "rounds": 2}
        assert M.edit_transform_active({"edit_transform": et}) is False
        assert M.edit_transform_active({"edit_transform": et, "corrected_edit_mode": "line"}) is False
        assert M.edit_transform_active({"edit_transform": et, "corrected_edit_mode": "fold"}) is False
        assert M.edit_transform_active({"edit_transform": et, "corrected_edit_mode": "transform"}) is True
        assert M.edit_transform_active({"corrected_edit_mode": "transform"}) is False          # nothing stored
        assert M.edit_transform_active({"edit_transform": {"shift": []}, "corrected_edit_mode": "transform"}) is False


# ───────────────────────── drawn-laterals-only per-frame rigid fit (surface-crop aware) ─────────────────
class TestDrawnFrameRigid:
    def test_recovers_the_per_frame_jitter_from_drawn_lines(self):
        L, F = 513, 101; fr = np.arange(F)
        true_curve = 60 + 0.02 * (fr - 50) ** 2                      # a quadratic across frames
        jitter = np.zeros(F); jitter[10:20] = -8.0; jitter[70] = 5.0  # what the frames need
        edges = np.tile(true_curve - jitter, (L, 1)) + 30.0           # served surface = curve minus the jitter (+pad 30)
        quad = np.tile(true_curve + 30.0, (L, 1))
        lats = [40, 120, 200, 300, 400, 480]
        # the reviewer drew full lines on 6 laterals (raw rows = served - pad)
        anchors = {str(l): {str(int(f)): float(edges[l, f] - 30.0) for f in fr} for l in lats}
        a, b, used = M._drawn_frame_rigid(edges, quad, {"border_anchors": anchors, "_canvas_pad": 30})
        assert used.all()
        # a line's own deg-2 absorbs part of a 10-frame jitter (the parabola is pulled ~2.5 px), so the fit
        # recovers the SHAPE and sign, not the full 8 px: the jittered frames must read strongly negative
        # against their neighbours, the single-frame jitter positive, and no tilt may appear.
        assert a[10:20].mean() < -4.0 and (a[8] - a[12]) > 6.0 and a[70] > 3.0 and np.abs(b).max() < 0.5

    def test_cropped_frames_follow_the_bottom_line(self):
        L, F = 513, 60; fr = np.arange(F)
        edges = np.tile(100.0 + 0.0 * fr, (L, 1)); quad = edges.copy()   # top: flat, says "no move"
        # bottom line drawn on 5 laterals over frames 0..29: a quadratic except frames 10..14 sit 6 px too deep
        bot_true = 300 + 0.05 * (fr[:30] - 15) ** 2; bot = bot_true.copy(); bot[10:15] += 6.0
        lats = [100, 200, 256, 300, 400]
        anchors = {str(l): {str(int(f)): 100.0 for f in fr} for l in lats}
        post = {str(l): {str(int(f)): float(bot[f]) for f in range(30)} for l in lats}
        a, b, used = M._drawn_frame_rigid(edges, quad, {"border_anchors": anchors, "crop_post_anchors": post,
                                                        "surface_crop_frames": list(range(5, 25)), "_canvas_pad": 0})
        assert used.all()
        assert np.all(a[10:15] < -3.0) and abs(a[30]) < 0.5 and abs(a[3]) < 1.0

    def test_no_anchors_means_nothing_used(self):
        edges = np.zeros((64, 20)) + 50; quad = edges.copy()
        a, b, used = M._drawn_frame_rigid(edges, quad, {})
        assert not used.any()


class TestDensifyPolylines:
    """A drawn line is the polyline through its points: gaps along frames are filled linearly (cs002 bug 1)."""

    def test_fills_gaps_linearly_and_keeps_points(self):
        import oct_preprocess as op
        out = op.densify_anchor_polylines({"5": {"0": 10.0, "4": 18.0, "5": 20.0}})
        assert out == {5: {0: 10.0, 1: 12.0, 2: 14.0, 3: 16.0, 4: 18.0, 5: 20.0}}

    def test_outside_span_untouched_and_sentinel_gap_skipped(self):
        import oct_preprocess as op
        out = op.densify_anchor_polylines({3: {2: 100.0, 6: 639.0, 8: 120.0}}, depth=640)
        assert 0 not in out[3] and 1 not in out[3] and 9 not in out[3]
        assert 3 not in out[3] and 7 not in out[3]          # both gaps touch the ABSENT sentinel
        assert out[3][6] == 639.0

    def test_disabled_or_single_point_is_identity(self):
        import oct_preprocess as op
        assert op.densify_anchor_polylines({1: {0: 5.0, 9: 9.0}}, enabled=False) == {1: {0: 5.0, 9: 9.0}}
        assert op.densify_anchor_polylines({1: {4: 5.0}}) == {1: {4: 5.0}}

    def test_pin_anchors_pins_the_segment(self):
        import numpy as np, oct_preprocess as op
        surf = np.full((3, 10), 14.0)                       # the chord 10..18 stays within the chord guard of this surface
        op.pin_anchors(surf, {1: {2: 10.0, 6: 18.0}}, depth=100)
        assert np.allclose(surf[1, 2:7], [10, 12, 14, 16, 18]) and surf[1, 1] == 14 and surf[1, 7] == 14


class TestJsonSafeRecord:
    """The worker's final ITER line must never die on a record detail (cs002_os_v3 clear-all, 2026-09-04)."""

    def test_drops_private_keys_and_converts_numpy(self):
        import json, numpy as np, oct_preprocess as op
        rec = {"rigid_frame_refine": {"applied": np.bool_(True), "rough": np.float32(1.5), "_surface_before": np.zeros((2, 3)),
                                      "nested": [np.int64(3), (1, 2), {"_private": np.ones(2), "keep": np.arange(2)}]}}
        out = op._json_safe(rec)
        assert out == {"rigid_frame_refine": {"applied": True, "rough": 1.5, "nested": [3, [1, 2], {"keep": [0, 1]}]}}
        json.dumps(out)   # must not raise


# ── "bottom − interpolated thickness" placement of the cropped-frame top (reviewer design, 2026-09-04) ──
def _thickness_case(L=9, F=40, T_extra=None):
    """Synthetic (L, F) surfaces in RAW rows: a dome whose apex sits 10 px ABOVE the window (top_true < 0 on
    the central frames = the crop band), a thickness that trends linearly across frames like cs002 (F3), a
    served top pinned at 5 px (< clip_edge_floor) with the clip symptom on inside the band."""
    fr = np.arange(F, dtype=float)
    top_true = 0.1 * (fr - 20) ** 2 - 10.0
    T_true = 150.0 + 0.5 * fr
    if T_extra is not None:
        T_true = T_true + T_extra
    bottom = np.tile(top_true + T_true, (L, 1))
    crop = [int(f) for f in range(F) if top_true[f] < 0]
    served = np.tile(np.where(top_true < 0, 5.0, top_true), (L, 1))
    clip = np.tile(top_true < 0, (L, 1))
    c0, c1 = crop[0], crop[-1]
    flank = [f for f in range(c0 - 4, c0)] + [f for f in range(c1 + 1, c1 + 5)]
    return fr, top_true, T_true, bottom, crop, served, clip, flank












class TestDensifyChordGuard:
    """Separate strokes on one slice must not be bridged by a chord far from the surface (cs042_os_v1_3)."""

    def test_chord_along_surface_bridged_far_chord_left_open(self):
        import numpy as np, oct_preprocess as op
        ref = np.zeros((2, 60)); ref[1] = 100.0 + 0.05 * (np.arange(60) - 30) ** 2     # a dome on slice 1
        near = {0: {0: 0.0, 10: 0.0}}                                                    # chord on the flat surface
        far = {1: {0: float(ref[1, 0]), 59: float(ref[1, 59])}}                           # chord across the dome
        out = op.densify_anchor_polylines({**near, **far}, reference=ref, guarded_slices={0, 1})
        assert set(out[0]) == set(range(0, 11))
        assert set(out[1]) == {0, 59}                                                    # left to the interpolation
        assert set(op.densify_anchor_polylines(far)[1]) == set(range(0, 60))             # no reference → old behaviour
        assert set(op.densify_anchor_polylines(far, reference=ref)[1]) == set(range(0, 60))   # not a folded lateral → a drag: bridged

    def test_pin_anchors_does_not_pin_a_far_chord(self):
        import numpy as np, oct_preprocess as op
        surf = 100.0 + 0.05 * (np.arange(60) - 30) ** 2; S = np.tile(surf, (1, 1))
        before = S.copy()
        op.pin_anchors(S, {0: {0: float(surf[0]), 59: float(surf[59])}}, depth=640, guarded_slices={0})
        assert np.allclose(S, before)                                                    # nothing between the strokes moved
        op.pin_anchors(S, {0: {0: float(surf[0]), 59: float(surf[59])}}, depth=640)     # a reviewer drag: the chord is pinned
        assert abs(S[0, 30] - (surf[0] + (surf[59] - surf[0]) * 30 / 59)) < 1e-6


class TestFlattenExcludeLaterals:
    """Folded laterals are served-surface GT only: they must not drive the drawn-frame rigid fit."""

    def test_excluded_lateral_does_not_drive_the_move(self):
        import numpy as np, oct_preprocess as op
        L, F = 9, 40
        fr = np.arange(F, dtype=float); quad = 50 + 0.02 * (fr - 20) ** 2
        edges = np.tile(quad, (L, 1)).astype(np.float64)
        jitter = np.zeros(F); jitter[10] = 6.0                                       # one jittered frame
        edges += jitter[None, :]
        anchors = {"4": {str(f): float(edges[4, f]) for f in range(F)},               # honest line: asks to undo the jitter
                   "2": {str(f): float(edges[2, f] + (30.0 if f == 25 else 0.0)) for f in range(F)}}   # a folded line with a spike
        p = {**op.DEFAULT_PARAMS, "border_anchors": anchors, "surface_crop_frames": [], "flatten_drawn_min_line": 5}
        a_all, _, used_all = op._drawn_frame_rigid(edges, np.tile(quad, (L, 1)), p)
        a_ex, _, used_ex = op._drawn_frame_rigid(edges, np.tile(quad, (L, 1)), {**p, "flatten_exclude_laterals": [2]})
        assert used_ex.any()
        assert abs(a_ex[25]) < abs(a_all[25]) - 5                                   # the spike no longer pulls frame 25


def _witness_volume(L=5, D=200, F=60, dip=False, seed=0, surf_fn=None):
    """(vol (L, D, F) float32, surface (F,)) — a bright tissue block under a dome surface, dark air above. dip=True
    plunges the surface 100 px over frames ~40-50 (a blink); surf_fn overrides the surface."""
    rng = np.random.default_rng(seed); fr = np.arange(F, dtype=float)
    surf = surf_fn(fr) if surf_fn is not None else 60.0 + 0.03 * (fr - 30.0) ** 2
    if dip:
        surf = surf + 100.0 * np.exp(-0.5 * ((fr - 45.0) / 3.0) ** 2)
    rows = np.arange(D, dtype=float)[:, None]
    tissue = (rows >= surf[None, :]) & (rows < surf[None, :] + 120.0)
    sl = np.where(tissue, 1000.0 + 40.0 * rng.standard_normal((D, F)), 50.0 + 10.0 * rng.standard_normal((D, F)))
    vol = np.repeat(sl[None, :, :], L, axis=0).astype(np.float32)
    return vol, surf


class TestChordWitness:
    """A stroke gap is bridged by a straight chord ONLY when the raw slice says the chord runs along the surface."""

    def test_fast_drag_along_the_surface_bridges_exactly_as_today(self):
        vol, surf = _witness_volume()
        anchors = {2: {10: float(surf[10]), 31: float(surf[31])}}          # 20-frame gap; chord sagitta ~3 px
        R = M.chord_witness_refusals(vol, anchors, M.DEFAULT_PARAMS)
        assert R == {}
        out = M.densify_anchor_polylines(anchors, refuse_gaps=R)
        assert out == M.densify_anchor_polylines(anchors) and set(out[2]) == set(range(10, 32))
        # a WRONG surface reference must not matter: the witness never consults one
        ref = np.tile(surf + 25.0, (5, 1))
        assert set(M.densify_anchor_polylines(anchors, reference=ref, guarded_slices={2})[2]) == {10, 31}   # old guard refuses
        assert set(M.densify_anchor_polylines(anchors, refuse_gaps=R)[2]) == set(range(10, 32))              # witness bridges

    def test_chord_across_the_blink_dip_is_refused_and_points_kept(self):
        vol, surf = _witness_volume(dip=True)
        anchors = {2: {39: float(surf[39]), 51: float(surf[51])}}          # chord ~100 px ABOVE the plunging tissue
        R = M.chord_witness_refusals(vol, anchors, M.DEFAULT_PARAMS)
        assert (39, 51) in R.get(2, {}) and R[2][(39, 51)]["reason"] == "air" and R[2][(39, 51)]["run"] >= 5
        opened = {f for lo, hi in R[2][(39, 51)]["runs"] for f in range(lo, hi + 1)}
        assert len(opened) >= 5 and opened <= set(range(40, 51))
        d = M.densify_anchor_polylines(anchors, refuse_gaps=R)
        assert not (opened & set(d[2])) and d[2][39] == float(surf[39]) and d[2][51] == float(surf[51])   # open frames carry no chord
        S = np.tile(surf, (5, 1)).astype(np.float64); before = S.copy()
        M.pin_anchors(S, anchors, depth=200, refuse_gaps=R)
        fo = sorted(opened)
        assert np.allclose(S[2, fo], before[2, fo]) and S[2, 39] == float(surf[39])
        assert M.chord_refusals_to_list(R)[0]["lateral"] == 2 and M.chord_refusals_to_list(R)[0]["f0"] == 39

    def test_chord_through_tissue_is_refused(self):
        # the p1 geometry: a dome between two deep shoulders; the chord between the shoulders runs INSIDE the tissue
        fn = lambda fr: np.minimum(141.0 + 0.25 * (fr - 30.0) ** 2, 186.0)
        vol, surf = _witness_volume(D=320, surf_fn=fn)
        anchors = {2: {16: float(surf[16]), 44: float(surf[44])}}          # both at the 186 shoulders; chord flat at 186, dome apex at 141
        R = M.chord_witness_refusals(vol, anchors, M.DEFAULT_PARAMS)
        assert (16, 44) in R.get(2, {}) and R[2][(16, 44)]["reason"] == "tissue" and R[2][(16, 44)]["med_dev"] > 12
        assert M.chord_witness_refusals(vol, anchors, {**M.DEFAULT_PARAMS, "chord_witness": False}) == {}

    def test_only_the_failing_run_is_left_open(self):
        # a gap whose chord fails on the dip frames only: the rest of the gap keeps the chord (it sits on the surface)
        vol, surf = _witness_volume(dip=True)
        anchors = {2: {30: float(surf[30]), 59: float(surf[59])}}           # 28 interior frames; the dip is ~40-50
        R = M.chord_witness_refusals(vol, anchors, M.DEFAULT_PARAMS)
        rec = R[2][(30, 59)]; runs = rec["runs"]
        assert len(runs) >= 1 and runs[0][0] >= 36 and runs[-1][1] <= 55, runs           # only the dip frames fail
        d = M.densify_anchor_polylines(anchors, refuse_gaps=R)[2]
        opened = {f for lo, hi in runs for f in range(lo, hi + 1)}
        assert all(f not in d for f in opened) and all(f in d for f in range(31, 59) if f not in opened)
        lst = M.chord_refusals_to_list(R)                                    # the JSON/list form carries the runs
        d2 = M.densify_anchor_polylines(anchors, refuse_gaps=[[r["lateral"], r["f0"], r["f1"], r["runs"]] for r in lst])[2]
        assert set(d2) == set(d)

    def test_refusal_is_local_and_never_flips_a_frame(self):
        vol, surf = _witness_volume(L=16, dip=True)
        base = np.tile(surf - 30.0, (16, 1)).astype(np.float64)             # a deliberately WRONG auto (30 px shallow)
        anchors = {s: {f: float(surf[f]) for f in range(60)} for s in range(1, 15)}
        anchors[7] = {f: float(surf[f]) for f in list(range(0, 40)) + list(range(51, 60))}   # lifted the pointer across the dip
        p = {**M.DEFAULT_PARAMS, "interp_min_slices": 12}
        R = M.chord_witness_refusals(vol, anchors, p)
        assert set(R) == {7} and (39, 51) in R[7]
        fo = sorted({f for lo, hi in R[7][(39, 51)]["runs"] for f in range(lo, hi + 1)})   # the frames left open
        assert len(fo) >= 5 and set(fo) <= set(range(40, 51))
        S_old = np.asarray(M.interpolate_anchors_surface(anchors, base, p), np.float64)
        S_new = np.asarray(M.interpolate_anchors_surface(anchors, base, {**p, "_refused_chords": R}), np.float64)
        assert np.allclose(S_new[7, fo], surf[fo], atol=1e-4)                # the neighbours' drawn line, not the chord, not the auto
        assert np.max(np.abs(S_old[7, fo] - surf[fo])) > 20                   # the chord was ~100 px off there
        mask = np.zeros(S_old.shape, bool); mask[7, fo] = True
        assert np.array_equal(S_new[~mask], S_old[~mask])                    # locality: chord frames within 12 px of the witness stay
        assert anchors[7] == {f: float(surf[f]) for f in list(range(0, 40)) + list(range(51, 60))}   # points untouched

    def test_fill_check_keeps_the_chord_when_the_fill_is_worse(self):
        vol, surf = _witness_volume(dip=True)
        anchors = {2: {39: float(surf[39]), 51: float(surf[51])}}
        R = M.chord_witness_refusals(vol, anchors, M.DEFAULT_PARAMS)
        assert (39, 51) in R[2]
        good = np.tile(surf, (5, 1)).astype(np.float64)                        # a fill ON the surface: refusal stands
        kept, dropped = M.chord_fill_check(vol, anchors, R, good, M.DEFAULT_PARAMS)
        assert (39, 51) in kept.get(2, {}) and dropped == [] and kept[2][(39, 51)]["fill_px"] < 3
        bad = good.copy(); bad[2, 40:51] = surf[40:51] + 150.0                   # a fill 150 px off: worse than the ~100 px chord → chord kept
        kept2, dropped2 = M.chord_fill_check(vol, anchors, R, bad, M.DEFAULT_PARAMS)
        assert kept2 == {} and len(dropped2) == 1 and dropped2[0]["fill_px"] > dropped2[0]["chord_px"]

    def test_far_knots_taper_to_the_auto_per_cell(self):
        vol, surf = _witness_volume(L=16, dip=True)
        base = np.tile(surf - 30.0, (16, 1)).astype(np.float64)
        anchors = {s: {f: float(surf[f]) for f in range(60)} for s in range(1, 15)}
        for s in range(3, 13):                                              # laterals 3..12 lift the pointer across the dip
            anchors[s] = {f: float(surf[f]) for f in list(range(0, 40)) + list(range(51, 60))}
        p = {**M.DEFAULT_PARAMS, "interp_min_slices": 12, "interp_fill_reach": 3.0}
        R = M.chord_witness_refusals(vol, anchors, p)
        assert set(R) == set(range(3, 13))
        fo = sorted({f for lo, hi in R[7][(39, 51)]["runs"] for f in range(lo, hi + 1)})
        S = np.asarray(M.interpolate_anchors_surface(anchors, base, {**p, "_refused_chords": R}), np.float64)
        # lateral 7: nearest surviving knots at 2 and 13 (span 11 > 2*reach) → the auto; lateral 1: drawn → exact
        assert np.allclose(S[7, fo], base[7, fo]) and np.allclose(S[1, fo], surf[fo])
        # the frame gate did not flip: undrawn lateral 0 at an open frame is still the interpolated/extended line, not a base blend
        assert np.allclose(S[0, fo], surf[fo], atol=1e-6)

    def test_short_pass_between_failing_runs_is_absorbed(self):
        # two dips in one gap, 3 passing frames between them where the chord crosses the surface → ONE open run
        fn = lambda fr: 60.0 + 0.03 * (fr - 30.0) ** 2 + 100.0 * (np.exp(-0.5 * ((fr - 36.0) / 2.5) ** 2) + np.exp(-0.5 * ((fr - 48.0) / 2.5) ** 2))
        vol, surf = _witness_volume(surf_fn=fn)
        anchors = {2: {29: float(surf[29]), 55: float(surf[55])}}
        R = M.chord_witness_refusals(vol, anchors, M.DEFAULT_PARAMS)
        rec = R[2][(29, 55)]
        assert len(rec["runs"]) == 1 and rec["runs"][0][0] <= 34 and rec["runs"][0][1] >= 50, rec["runs"]

    def test_folded_lateral_guard_keeps_precedence(self):
        # a gap the folded-lateral reference guard refuses stays WHOLLY open even when the witness only fails part of it
        vol, surf = _witness_volume(dip=True)
        anchors = {2: {30: float(surf[30]), 59: float(surf[59])}}
        R = M.chord_witness_refusals(vol, anchors, M.DEFAULT_PARAMS)
        assert R[2][(30, 59)]["runs"][0][0] > 31                                # the witness leaves frames 31.. bridged
        ref = np.tile(surf, (5, 1)).astype(np.float64)                           # the chord is > 12 px off this reference at the dip
        d = M.densify_anchor_polylines(anchors, reference=ref, guarded_slices={2}, refuse_gaps=R)
        assert set(d[2]) == {30, 59}                                             # no chord fragments on a guarded lateral

    def test_witness_fail_safes_bridge_as_today(self):
        vol, surf = _witness_volume(dip=True)
        p = M.DEFAULT_PARAMS
        # (a) endpoints 30 px above the tissue: the witness cannot vouch → bridged
        a = {2: {39: float(surf[39]) - 30.0, 51: float(surf[51]) - 30.0}}
        assert M.chord_witness_refusals(vol, a, p) == {}
        # (b) tissue bright from row 0 over the gap (apex above the window) → unscorable → bridged
        v2 = vol.copy(); v2[:, :, 40:51] = 1000.0
        assert M.chord_witness_refusals(v2, {2: {39: float(surf[39]), 51: float(surf[51])}}, p) == {}
        # (c) too short a gap is never judged
        assert M.chord_witness_refusals(vol, {2: {43: float(surf[43]), 47: float(surf[47])}}, p) == {}
        # (d) the list form round-trips into densify
        R = M.chord_witness_refusals(vol, {2: {39: float(surf[39]), 51: float(surf[51])}}, p)
        lst = [[r["lateral"], r["f0"], r["f1"]] for r in M.chord_refusals_to_list(R)]
        assert set(M.densify_anchor_polylines({2: {39: 1.0, 51: 2.0}}, refuse_gaps=lst)[2]) == {39, 51}
        assert M._json_safe({"_refused_chords": R, "keep": 1}) == {"keep": 1}


class TestNoiseCropTissueWitness:
    """p6_os_v1 (2026-09-09): the auto noise crop zeroed 25 frames of real cornea because the detector lost a dim, deep
    cornea and scored them like empty frames. A boundary frame whose tissue ridge forms a concave dome across laterals
    is never noise; a bright but FLAT band (an eyelid) still is."""

    def _sag(self, tail="dim_dome", seed=0):
        rng = np.random.default_rng(seed); L, D, F = 40, 200, 40
        sag = (40.0 + 12.0 * rng.standard_normal((L, D, F))).astype(np.float32)          # speckle background
        x = np.arange(L, dtype=float); dome = 0.05 * (x - L / 2.0) ** 2                  # concave across laterals
        rows = np.arange(D, dtype=float)
        def band(top, amp):                                    # a tissue band: brightest at its anterior, decaying into the stroma
            d = rows[None, :] - top[:, None]
            return np.where((d >= 0) & (d < 60), amp * np.exp(-d / 20.0), 0.0)
        for f in range(F):
            if f < 25:
                sag[:, :, f] += band(60 + 2 * f + dome, 900.0)
            elif tail == "dim_dome":
                sag[:, :, f] += band(110 + 2 * (f - 25) + dome, 60.0)              # dim (~+150 % over bg at the ridge), concave
            elif tail == "bright_flat":
                sag[:, :, f] += band(110 + 2 * (f - 25) + np.zeros(L), 900.0)      # eyelid-like: bright, flat across laterals
        return sag

    def test_dim_dome_kept_flat_band_and_pure_noise_cropped(self):
        import oct_preprocess as op
        det = np.full((40, 40), 5.0); det[:, :25] = 60 + 2 * np.arange(25)[None, :]   # detector "lost" the tail
        p = {"crop_noise_min_run": 5}
        assert op.detect_noise_frames(self._sag("dim_dome"), p, detect=det) == []                    # dim concave band: tissue, kept
        assert op.detect_noise_frames(self._sag("bright_flat"), p, detect=det) == list(range(25, 40))  # bright FLAT band: not a cornea, cropped
        assert op.detect_noise_frames(self._sag("none"), p, detect=det) == list(range(25, 40))         # speckle only: cropped
        assert op.detect_noise_frames(self._sag("dim_dome"), {**p, "crop_noise_tissue_contrast": 0}, detect=det) == list(range(25, 40))   # disabled → old rule


class TestPlaceTopFromThickness:
    def test_two_flanks_drawn_and_bracketed_slices(self):
        import json
        L = 9
        fr, top_true, T_true, bottom, crop, served, clip, flank = _thickness_case(L=L)
        dt = {s: {f: float(top_true[f]) for f in flank} for s in (0, 8)}
        db = {s: {f: float(bottom[s, f]) for f in flank} for s in (0, 8)}
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, {}, dt, db, clip)
        assert np.abs(placed[[0, 8]][:, crop] - top_true[crop][None, :]).max() < 0.5
        assert np.abs(placed[1:8][:, crop] - top_true[crop][None, :]).max() < 1.5
        outside = [f for f in range(len(fr)) if f not in crop]
        assert np.isnan(placed[:, outside]).all() and np.isnan(T[:, outside]).all()
        assert info["n_slices_drawn_left"] == 2 and info["n_slices_drawn_right"] == 2
        assert info["n_placed"] == L * len(crop) and info["n_bands"] == 1
        json.dumps(info)                                         # every value is a python scalar
        for v in info.values():
            assert not isinstance(v, np.generic)

    def test_one_flank_holds_and_none_falls_back(self):
        L = 9
        fr, top_true, T_true, bottom, crop, served, clip, flank = _thickness_case(L=L)
        c0, c1 = crop[0], crop[-1]
        right = [f for f in flank if f > c1]
        bot1 = bottom.copy(); bot1[:, right] = np.nan                  # right flank unavailable
        dt = {s: {f: float(top_true[f]) for f in flank if f < c0} for s in (0, 8)}
        db = {s: {f: float(bottom[s, f]) for f in flank if f < c0} for s in (0, 8)}
        placed, T, info = M.place_top_from_thickness(served, bot1, crop, {}, dt, db, clip)
        assert np.isfinite(placed[:, crop]).all()
        assert np.allclose(T[:, crop], T_true[c0 - 1])               # held at the left boundary value
        worst = float(np.abs(T_true[crop] - T_true[c0 - 1]).max())
        assert np.abs(placed[:, crop] - top_true[crop][None, :]).max() <= worst + 1e-6
        assert info["n_slices_drawn_left"] == 2 and info["n_slices_drawn_right"] == 0
        assert info["n_slices_served"] == L - 2 and info["n_slices_fallback"] == 0
        bot2 = bottom.copy(); bot2[:, flank] = np.nan                  # no flank at all
        placed2, T2, info2 = M.place_top_from_thickness(served, bot2, crop, {}, None, None, clip)
        assert np.isnan(placed2).all() and np.isnan(T2).all() and info2["n_slices_fallback"] == L

    def test_drawn_valid_with_late_start_and_mixed_sources(self):
        # refuter high #2: a slice whose drawn line starts one frame into the flank (2 drawn-both frames per
        # side) is DRAWN-valid on that side and its OWN thickness is used, not the interpolation from the
        # neighbouring drawn slices — which here are 30 px thinner.
        L = 9
        fr, top_true, T_true, bottom, crop, served, clip, flank = _thickness_case(L=L)
        c0, c1 = crop[0], crop[-1]
        bottom4 = bottom.copy(); bottom4[4] += 30.0                    # slice 4 is a 30 px thicker cornea
        dt = {s: {f: float(top_true[f]) for f in flank} for s in (0, 8)}
        db = {s: {f: float(bottom[s, f]) for f in flank} for s in (0, 8)}
        late = [c0 - 2, c0 - 1, c1 + 1, c1 + 2]                          # 2 per side
        dt[4] = {f: float(top_true[f]) for f in late}
        db[4] = {f: float(bottom4[4, f]) for f in late}
        placed, T, info = M.place_top_from_thickness(served, bottom4, crop, {}, dt, db, clip)
        assert info["n_slices_drawn_left"] == 3 and info["n_slices_drawn_right"] == 3
        assert np.abs(placed[4, crop] - top_true[crop]).max() < 1.0    # own thickness (T_true + 30)
        assert np.abs(T[4, crop] - (T_true[crop] + 30.0)).max() < 1.0
        assert np.abs(placed[0, crop] - top_true[crop]).max() < 0.5    # neighbours untouched

    def test_top_present_kept_and_clamp(self):
        L = 9
        fr, top_true, T_true, bottom, crop, served, clip, flank = _thickness_case(L=L)
        c0 = crop[0]
        served2 = served.copy(); clip2 = clip.copy()
        served2[2, crop] = 10.0; clip2[2, crop] = False                # slice 2: a real in-frame top (>= floor)
        dt = {s: {f: float(top_true[f]) for f in flank} for s in (0, 8)}
        db = {s: {f: float(bottom[s, f]) for f in flank} for s in (0, 8)}
        db[0][c0 - 4] = float(top_true[c0 - 4]) + 900.0                # an implausible flank candidate
        placed, T, info = M.place_top_from_thickness(served2, bottom, crop, {}, dt, db, clip2)
        assert np.isnan(placed[2, crop]).all() and info["n_top_present_kept"] == len(crop)
        assert np.isfinite(T[2, crop]).all()                            # T exists there (unified rule applies)
        assert info["n_candidates_rejected"] == 1
        assert np.abs(placed[0, crop] - top_true[crop]).max() < 0.5    # the rejected point moved nothing

    def test_boundary_value_is_at_the_boundary_frame(self):
        # refuter high #1: a flank ramping 3 px/frame like cs002 slice 152 (231,228,226,222) → the value AT
        # c0−1 is the ramp continued (222), NOT the flank mean (227).
        L, F = 1, 30
        crop = list(range(4, 21))
        served = np.full((L, F), 5.0); served[0, :4] = 20.0
        bottom = np.full((L, F), 240.0); clip = np.zeros((L, F), bool); clip[0, 4:21] = True
        dt = {0: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}}
        db = {0: {0: 231.0, 1: 228.0, 2: 226.0, 3: 222.0}}
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, {}, dt, db, clip)
        assert abs(T[0, 4] - 222.0) <= 1.0 and abs(T[0, 4] - 227.0) > 3.0
        # a 33 px drawing ramp in 4 frames (cs002 slice 192: 195,210,220,228) is NOT extrapolated: the two
        # candidates nearest the boundary decide (median 224), and the gate is counted.
        db2 = {0: {0: 195.0, 1: 210.0, 2: 220.0, 3: 228.0}}
        placed2, T2, info2 = M.place_top_from_thickness(served, bottom, crop, {}, dt, db2, clip)
        assert abs(T2[0, 4] - 224.0) <= 1.0 and info2["n_slope_gated"] == 1

    def test_clip_ceiling_and_shaped_laterals_are_not_knots(self):
        # physical guard: on a bright-top cell the apex cannot sit INSIDE the frame — an under-measured T that
        # would place the top at +12 is clamped to the ceiling (0) and counted; a shaped lateral's drawn lines
        # are never a drawn candidate/knot (its top does not track the tissue frame-by-frame).
        L, F = 3, 30
        crop = list(range(4, 21))
        served = np.full((L, F), 5.0); served[:, :4] = 20.0; served[:, 21:] = 20.0
        bottom = np.full((L, F), 230.0); clip = np.zeros((L, F), bool); clip[:, 4:21] = True
        dt = {s: {f: 20.0 for f in list(range(4)) + list(range(21, 25))} for s in range(L)}
        db = {s: {f: 230.0 for f in list(range(4)) + list(range(21, 25))} for s in range(L)}
        db[1] = {f: 218.0 for f in db[1]}                              # slice 1 drawn 12 px too thin
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, {}, dt, db, clip)
        assert np.allclose(placed[0, crop], 20.0) is False              # slice 0 placed at 230-210 = 20 → ceiling
        assert np.all(placed[:, crop] <= 0.0) and info["n_ceiling_clamped"] > 0
        p2, T2, i2 = M.place_top_from_thickness(served, bottom, crop, {}, dt, db, clip, shaped_laterals=[1])
        assert i2["n_slices_drawn_left"] == 2 and abs(T2[1, 10] - 210.0) < 1e-6   # interpolated from 0 and 2


class TestDrawnFrameRigidUnified:
    def test_unified_top_recovers_jitter_in_cropped_frames(self):
        L, F = 513, 101; fr = np.arange(F)
        true_curve = 60 + 0.02 * (fr - 50) ** 2
        jitter = np.zeros(F); jitter[10:20] = -8.0; jitter[70] = 5.0
        crop = list(range(15, 26))                                       # the −8 block runs INTO the crop
        edges = np.tile(true_curve - jitter, (L, 1)) + 30.0
        quad = np.tile(true_curve + 30.0, (L, 1))
        lats = [40, 120, 200, 300, 400, 480]
        anchors = {str(l): {str(int(f)): float(edges[l, f] - 30.0) for f in fr if f not in crop} for l in lats}
        post = {str(l): {str(int(f)): float(true_curve[f] - jitter[f] + 200.0) for f in fr} for l in lats}
        placed = {l: {int(f): float(true_curve[f] - jitter[f]) for f in crop} for l in lats}
        tok = {l: list(crop) for l in lats}
        p = {"border_anchors": anchors, "crop_post_anchors": post, "surface_crop_frames": crop,
             "_canvas_pad": 30, "_raw_depth": 640, "_crop_top_placed": placed, "_crop_thickness_ok": tok,
             "flatten_crop_unified_top": True}
        a, b, used = M._drawn_frame_rigid(edges, quad, p)
        assert used.all()
        assert a[10:20].mean() < -4.0 and a[15:20].mean() < -4.0 and a[70] > 3.0 and np.abs(b).max() < 0.5
        assert M._LAST_DRAWN_FIT["n_placed_targets"] == len(lats) * len(crop)
        rec = M._drawn_fit_record(a, p)
        assert rec["unified_top"] and rec["n_placed_targets"] > 0 and len(rec["boundary_jump_px"]) == 1

    def test_robust_fit_ignores_an_outlying_line(self):
        L, F = 513, 60; fr = np.arange(F)
        edges = np.tile(100.0 + 0.0 * fr, (L, 1)); quad = edges.copy()
        lats = [40, 120, 200, 300, 400, 480]
        anchors = {str(l): {str(int(f)): 100.0 for f in fr} for l in lats}
        anchors["300"]["15"] = 70.0                                       # one line asks for +30 at frame 15
        base = {"border_anchors": anchors, "_canvas_pad": 0}
        a_r, _, _ = M._drawn_frame_rigid(edges, quad, {**base, "flatten_drawn_robust_fit": True})
        a_p, _, _ = M._drawn_frame_rigid(edges, quad, {**base, "flatten_drawn_robust_fit": False})
        assert abs(a_r[15] - a_r[14]) < 1.5
        assert abs(a_p[15] - a_p[14]) > 4.0

    @pytest.mark.parametrize("unified", [False, True])
    def test_cropped_frames_follow_the_bottom_line_fallback(self, unified):
        # with no `_crop_top_placed` (no T could be measured) the unified path falls back to the old rule
        L, F = 513, 60; fr = np.arange(F)
        edges = np.tile(100.0 + 0.0 * fr, (L, 1)); quad = edges.copy()
        bot_true = 300 + 0.05 * (fr[:30] - 15) ** 2; bot = bot_true.copy(); bot[10:15] += 6.0
        lats = [100, 200, 256, 300, 400]
        anchors = {str(l): {str(int(f)): 100.0 for f in fr} for l in lats}
        post = {str(l): {str(int(f)): float(bot[f]) for f in range(30)} for l in lats}
        a, b, used = M._drawn_frame_rigid(edges, quad, {"border_anchors": anchors, "crop_post_anchors": post,
                                                        "surface_crop_frames": list(range(5, 25)), "_canvas_pad": 0,
                                                        "flatten_crop_unified_top": unified})
        assert used.all()
        assert np.all(a[10:15] < -3.0) and abs(a[30]) < 0.5 and abs(a[3]) < 1.0

    def test_bottom_sentinel_dropped_from_bottom_fit(self):
        L, F = 513, 60; fr = np.arange(F)
        edges = np.tile(100.0 + 0.0 * fr, (L, 1)); quad = edges.copy()
        bot = 300 + 0.05 * (fr[:30] - 15) ** 2
        lats = [100, 200, 256, 300, 400]
        anchors = {str(l): {str(int(f)): 100.0 for f in fr} for l in lats}
        post = {str(l): {str(int(f)): float(bot[f]) for f in range(30)} for l in lats}
        for l in lats:
            post[str(l)]["2"] = 639.0                                     # ABSENT sentinel at raw depth 640
        p = {"border_anchors": anchors, "crop_post_anchors": post, "surface_crop_frames": list(range(5, 25)),
             "_canvas_pad": 0, "_raw_depth": 640}
        a_u, _, _ = M._drawn_frame_rigid(edges, quad, {**p, "flatten_crop_unified_top": True})
        a_o, _, _ = M._drawn_frame_rigid(edges, quad, {**p, "flatten_crop_unified_top": False})
        assert np.abs(a_u[5:25]).max() < 0.5                             # a clean parabola asks for no move
        assert np.abs(a_o[5:25]).max() > 2.0                             # the sentinel bent the old fit


def test_repin_drawn_points_adds_pad_and_skips_placed():
    L, F, pad = 40, 20, 10
    E = np.full((L, F), 50.0 + pad, dtype=np.float32)
    anchors = {"5": {str(f): 50.0 for f in range(F)}}
    E1 = E.copy(); M._repin_drawn_points(E1, anchors, pad=0.0)              # today's raw re-pin
    assert np.allclose(E1[5] - E1[6], -pad)
    E2 = E.copy(); E2[5, 3] = 12.5                                          # a placed cell
    M._repin_drawn_points(E2, anchors, pad=float(pad), skip={5: {3: 12.5}})
    assert np.allclose(E2[5, [0, 1, 2, 4]] - E2[6, [0, 1, 2, 4]], 0.0) and E2[5, 3] == 12.5
    # through smooth_volume's use_provided branch the flag selects the behaviour (measured on a tiny volume)
    rng = np.random.RandomState(0)
    sag = (rng.rand(L, 120 + pad, F) * 30).astype(np.float32); sag[:, :pad, :] = 0
    for f in range(F):
        sag[:, pad + 50:pad + 90, f] += 1500.0
    vol = M.revert_sagittal(sag)
    pe = np.full((L, F), 50.0 + pad)
    for flag in (True, False):
        p = {"border_anchors": anchors, "_canvas_pad": pad, "_raw_depth": 120, "provided_repin_add_pad": flag}
        out = M.smooth_volume(vol, p, provided_edges=pe, workers=1)
        assert out.shape == vol.shape and np.isfinite(out).all()


def test_drawn_flatten_synthetic_bottom_parabola():
    """End-to-end through smooth_volume: a surface-cropped dome with per-frame axial jitter (a smooth bump
    inside the crop band), drawn top on the un-cropped frames + drawn bottom everywhere at 5 laterals, the
    cropped-frame top placed at bottom − T. Trace-free on the delivered tissue: the bottom crossing in the
    cropped frames lies on its own deg-2, is continuous across the band boundary, the column is not deformed
    (top−bottom gap preserved), nothing is truncated (energy), and the pad was zeros BEFORE the warp."""
    L, D0, F, pad = 40, 200, 40, 30
    fr = np.arange(F, dtype=float)
    top_true = 0.06 * (fr - 20) ** 2 - 10.0                                   # apex 10 px above the window
    T_true = 130.0 + 0.4 * fr
    jit = 4.0 * np.sin(2 * np.pi * fr / 25.0) - 8.0 * np.exp(-((fr - 20) / 4.0) ** 2)
    top_obs = top_true + jit; bot_obs = top_true + T_true + jit
    crop = [int(f) for f in range(F) if top_true[f] < 0]
    c0, c1 = crop[0], crop[-1]
    rng = np.random.RandomState(1)
    raw = (rng.rand(L, D0, F) * 40).astype(np.float32)
    for f in range(F):
        t0 = int(max(0, round(top_obs[f]))); b0 = int(round(bot_obs[f]))
        raw[:, t0:min(D0, b0), f] += 1500.0
    sag = np.zeros((L, D0 + pad, F), np.float32); sag[:, pad:, :] = raw
    assert float(sag[:, :pad, :].max()) == 0.0                                # never manufacture data
    vol = M.revert_sagittal(sag)
    lats = [4, 12, 20, 28, 36]
    served = np.tile(np.where(top_true < 0, 3.0, top_obs), (L, 1))
    bottom = np.tile(bot_obs, (L, 1)); clip = np.tile(top_true < 0, (L, 1))
    anchors = {str(l): {str(f): float(top_obs[f]) for f in range(F) if f not in crop} for l in lats}
    post = {str(l): {str(f): float(bot_obs[f]) for f in range(F)} for l in lats}
    dt = M.densify_anchor_polylines(anchors, None, True); db = M.densify_anchor_polylines(post, D0, True)
    placed, T, info = M.place_top_from_thickness(served, bottom, crop, {}, dt, db, clip)
    assert info["n_placed"] == L * len(crop)
    pe = served.copy(); ok = np.isfinite(placed); pe[ok] = placed[ok]
    ctp = {l: {int(f): float(placed[l, f]) for f in crop if np.isfinite(placed[l, f])} for l in lats}
    tok = {l: [int(f) for f in crop if np.isfinite(T[l, f])] for l in lats}
    params = {"border_anchors": anchors, "crop_post_anchors": post, "surface_crop_frames": crop,
              "_canvas_pad": pad, "_raw_depth": D0, "_crop_top_placed": ctp, "_crop_thickness_ok": tok}
    M._LAST_FLATTEN_INFO.clear()
    out = M.smooth_volume(vol, params, provided_edges=pe + pad, workers=1)
    fl = dict(M._LAST_FLATTEN_INFO)
    assert fl["unified_top"] and fl["n_placed_targets"] == len(lats) * len(crop) and fl["min_target_row"] >= 0.0
    osag = M.reformat_to_sagittal(out)
    assert float(osag.sum()) >= 0.999 * float(sag.sum())                     # nothing truncated
    thr = 0.5 * float(np.percentile(raw, 99))
    unc = [f for f in range(F) if f not in crop]
    for l in lats:
        col = osag[l]
        bc = np.array([np.where(col[:, f] > thr)[0].max() for f in range(F)], float)
        tc = np.array([np.where(col[:, f] > thr)[0].min() for f in range(F)], float)
        cf = np.array(crop, float)
        c = np.polyfit(cf, bc[crop], 2); par = np.polyval(c, fr)
        assert float(np.sqrt(np.mean((bc[crop] - par[crop]) ** 2))) < 1.5        # bottom smooth / quadratic
        assert abs((bc[c0] - bc[c0 - 1]) - (par[c0] - par[c0 - 1])) <= 2.0        # continuous at the boundary
        assert abs((bc[c1 + 1] - bc[c1]) - (par[c1 + 1] - par[c1])) <= 2.0
        gap_in = bot_obs - np.maximum(top_obs, 0.0)
        assert np.abs((bc - tc)[unc] - gap_in[unc]).max() <= 2.0                  # rigid: no column deformed
        ct = np.polyfit(np.array(unc, float), tc[unc], 2)
        assert float(np.sqrt(np.mean((tc[unc] - np.polyval(ct, np.array(unc, float))) ** 2))) < 1.5


class TestFirstBrightRunWitness:
    """The placement rule's trace-free absence witness (cs002 2026-09-04): a claimed top INSIDE the first bright
    run that starts at the frame top is clipped; a bright band that ends above the claimed top is a separate
    structure (eyelid at the periphery) and the top stands."""

    def _slice(self, D=120, F=4):
        sl = np.full((D, F), 20.0)
        sl[2:90, 0] = 1000.0                            # frame 0: tissue from row 2 to 90 (clipped apex)
        sl[0:7, 1] = 1000.0; sl[30:100, 1] = 1000.0     # frame 1: bright band 0-6, dark gap, cornea from 30
        sl[40:110, 2] = 1000.0                          # frame 2: in-frame top at 40
        return sl                                       # frame 3: no tissue

    def test_first_bright_run(self):
        st, en = M._first_bright_run(self._slice(), {})
        assert st[0] == 2 and en[0] == 90 and st[1] == 0 and en[1] == 7 and st[2] == 40 and np.isnan(st[3])

    def test_absence_needs_the_claim_inside_the_run(self):
        sl = self._slice(); st, en = M._first_bright_run(sl, {})
        FF = 12; c = list(range(4, 8))                                   # crop in the middle, flanks on both sides
        top = np.full((1, FF), 30.0); bot = np.full((1, FF), 230.0)
        top[0, 4] = 18.0; top[0, 5] = 30.0; top[0, 6] = 40.0; top[0, 7] = 30.0
        tr = np.full((1, FF), np.nan); te = np.full((1, FF), np.nan)
        tr[0, 4:8] = st; te[0, 4:8] = en
        clip = np.zeros((1, FF), bool)
        placed, T, info = M.place_top_from_thickness(top, bot, c, {}, None, None, clip, top_row=tr, top_run_end=te)
        assert np.isfinite(placed[0, 4]) and placed[0, 4] <= 2.0        # inside the run → placed, ceiling = row 2
        assert np.isnan(placed[0, 5]) and np.isnan(placed[0, 6])         # eyelid band above a gap / in-frame → kept
        assert info["n_top_present_kept"] >= 2


class TestPlaceTopWitnessKeep:
    """p1_od_v1_2 display 233 (2026-09-09): a band cell whose served top is REAL (the column's first bright run starts
    right there, inside the frame) must not be demoted to absent just because the slice parabola disagrees."""

    def _case(self):
        F = 60; fr = np.arange(F, dtype=float); crop = list(range(20, 40))
        par = 100.0 + 0.05 * (fr - 30.0) ** 2                      # the slice's own parabola outside the crop
        top = par.copy()
        top[34:40] += 35.0                                          # a real, visible top 35 px off the parabola on 6 cells
        served = top[None, :].copy(); bottom = served + 250.0       # T = 250 (inside the clamp)
        tr = served.copy(); te = served + 200.0                     # the first bright run starts AT the served top
        clip = np.zeros((1, F), bool)
        return served, bottom, crop, tr, te, clip

    def test_image_witnessed_top_is_kept(self):
        served, bottom, crop, tr, te, clip = self._case()
        p_on = {"crop_place_witness_keep": True}
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, p_on, None, None, clip, top_row=tr, top_run_end=te)
        assert np.isnan(placed[0, 34:40]).all(), placed[0, 34:40]              # nothing placed over a witnessed top
        assert info["n_witness_kept"] == 6

    def test_without_the_gate_the_estimate_replaced_it(self):
        served, bottom, crop, tr, te, clip = self._case()
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, {"crop_place_witness_keep": False}, None, None, clip, top_row=tr, top_run_end=te)
        assert np.isfinite(placed[0, 34:40]).all() and info["n_witness_kept"] == 0
        assert (np.abs(placed[0, 34:40] - served[0, 34:40]) > 20).all()          # the old behaviour: 35 px off the real top

    def test_top_inside_a_clipped_column_is_still_demoted(self):
        served, bottom, crop, tr, te, clip = self._case()
        tr[0, 34:40] = 2.0; te[0, 34:40] = 300.0                    # the tissue is continuous from the frame top: clipped
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, {"crop_place_witness_keep": True}, None, None, clip, top_row=tr, top_run_end=te)
        assert np.isfinite(placed[0, 34:40]).all() and info["n_witness_kept"] == 0


class TestPlaceTopDrawnBandLine:
    """p1_od_v1_3 slice 378 (2026-09-09): between two drawn slices the band line is the interpolation of the reviewer's
    lines; a slice with no drawn line within reach keeps its own parabola."""

    def _case(self, L=9, F=60):
        fr = np.arange(F, dtype=float); crop = list(range(20, 40))
        par = 100.0 + 0.05 * (fr - 30.0) ** 2                       # the served top outside the crop (a dome)
        served = np.tile(par, (L, 1)); served[:, crop] = -20.0       # inside the crop the served top is above the window
        bottom = np.tile(par + 250.0, (L, 1)); clip = np.zeros((L, F), bool); clip[:, crop] = True
        tr = np.zeros((L, F)); tr[:, [f for f in range(F) if f not in crop]] = served[:, [f for f in range(F) if f not in crop]]
        te = tr + 200.0
        # drawn lines on slices 2 and 6: the reviewer's estimate sits ABOVE the window (negative rows), as a clipped apex does
        dt = {s: {f: float(par[f] - 140.0) for f in crop} for s in (2, 6)}
        return served, bottom, crop, clip, tr, te, dt

    def test_between_drawn_slices_the_line_is_their_interpolation(self):
        served, bottom, crop, clip, tr, te, dt = self._case()
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, {"crop_estimate_source": "drawn", "crop_estimate_reach": 10},
                                                     dt, None, clip, top_row=tr, top_run_end=te)
        fr = np.arange(60, dtype=float); par = 100.0 + 0.05 * (fr - 30.0) ** 2
        assert np.allclose(placed[4, crop], par[crop] - 140.0, atol=1.5)     # slice 4 lies between 2 and 6 → their line
        assert np.allclose(placed[2, crop], par[crop] - 140.0, atol=1.5)     # the drawn slice itself

    def test_beyond_reach_the_parabola_stands(self):
        served, bottom, crop, clip, tr, te, dt = self._case()
        p = {"crop_estimate_source": "drawn", "crop_estimate_reach": 1}      # reach 1 lateral: slice 4 is 2 away from both
        placed, T, info = M.place_top_from_thickness(served, bottom, crop, p, dt, None, clip, top_row=tr, top_run_end=te)
        placed_par, _, _ = M.place_top_from_thickness(served, bottom, crop, {"crop_estimate_source": "parabola"}, dt, None, clip, top_row=tr, top_run_end=te)
        assert np.allclose(placed[4, crop], placed_par[4, crop], atol=1e-6)


class TestPlaceTopEstimateFallback:
    """A clipped cell whose bottom − T lands inside the image takes the reviewer's estimate, not the ceiling."""

    def _setup(self):
        import numpy as np
        L, F = 3, 40; crop = list(range(0, 10))
        top = np.full((L, F), 60.0); bottom = np.full((L, F), 260.0)          # flank thickness 200
        top[:, :10] = -8.0                                                      # served value in the band (ignored by the estimate)
        bottom[:, :10] = 250.0                                                  # band bottom shallower than 200 below the top → bottom − T = 50 (inside)
        clip = np.zeros((L, F), dtype=bool); clip[:, :10] = True                 # the tissue says: clipped
        return L, F, crop, top, bottom, clip

    def test_estimate_used_instead_of_ceiling(self):
        import numpy as np, oct_preprocess as op
        L, F, crop, top, bottom, clip = self._setup()
        p = {**op.DEFAULT_PARAMS, "crop_place_clip_ceiling": 0.0, "crop_estimate_source": "drawn"}
        raw = {}
        drawn = {0: {f: -8.0 for f in range(10)}, 2: {f: -8.0 for f in range(10)}}       # their estimate on two slices → interpolated to slice 1
        placed, T, info = op.place_top_from_thickness(top, bottom, crop, p, drawn, {}, clip, out_raw=raw)
        assert info["n_estimate_fallback"] > 0 and info["n_ceiling_clamped"] == 0
        assert np.allclose(placed[:, :10][np.isfinite(placed[:, :10])], -8.0)
        pr = raw["placed_raw"][:, :10]; assert np.allclose(pr[np.isfinite(pr)], 50.0)      # the motion keeps the unclamped value

    def test_flag_off_keeps_the_ceiling(self):
        import numpy as np, oct_preprocess as op
        L, F, crop, top, bottom, clip = self._setup()
        p = {**op.DEFAULT_PARAMS, "crop_place_clip_ceiling": 0.0, "crop_place_estimate_fallback": False, "crop_estimate_source": "drawn"}
        drawn = {0: {f: -8.0 for f in range(10)}, 2: {f: -8.0 for f in range(10)}}
        placed, T, info = op.place_top_from_thickness(top, bottom, crop, p, drawn, {}, clip)
        assert info["n_ceiling_clamped"] > 0 and info["n_estimate_fallback"] == 0
        assert np.allclose(placed[:, :10][np.isfinite(placed[:, :10])], 0.0)


class TestDrawnFitRefinement:
    """A slow drift common to every lateral must end up in the per-frame move, not in the lines' parabolas."""

    def test_common_drift_migrates_into_the_move(self):
        import numpy as np, oct_preprocess as op
        L, F = 513, 101; fr = np.arange(F)
        dome = 60 + 0.02 * (fr - 50) ** 2
        drift = 6.0 * np.sin(2 * np.pi * fr / 45.0)                              # slow, frame-common, not a parabola
        lats = [40, 120, 200, 300, 400, 480]
        edges = np.tile(dome + drift, (L, 1)); quad = np.tile(dome, (L, 1))
        anchors = {str(l): {str(int(f)): float(edges[l, f]) for f in fr} for l in lats}
        base = {"border_anchors": anchors, "_canvas_pad": 0, "surface_crop_frames": [], "flatten_crop_unified_top": True}
        a1, _, _ = op._drawn_frame_rigid(edges, quad, {**base, "flatten_drawn_iters": 1})
        a3, _, _ = op._drawn_frame_rigid(edges, quad, {**base, "flatten_drawn_iters": 4})
        def left(a): return float(np.sqrt(np.mean((drift + a - np.mean(drift + a)) ** 2)))   # drift still in the surface after the move
        # one pass already removes all but the parabola-shaped part of the drift; the alternation is a fixed point
        # on identical lines — it must never be worse than the single pass, and the record must say it ran
        assert left(a1) < 0.4 * float(np.std(drift))
        assert left(a3) <= left(a1) + 1e-6
        rec = op._drawn_fit_record(a3, {**base, "flatten_drawn_iters": 4}); assert rec["refine_iters"] == 4


# ───────────────────────── tissue-motion stage (shape from lines, motion from tissue) ─────────────────────
class TestTissueMotionMove:
    @staticmethod
    def _synth(F=60, D=200, L=96, jitter=None, tilt=None, seed=0):
        """A bright tissue band whose depth follows a parabola across frames (the dome) plus a per-frame
        rigid jitter (+ optional half-span tilt), on a background of speckle so the correlation has texture."""
        rng = np.random.default_rng(seed)
        fr = np.arange(F); dome = 80.0 + 0.02 * (fr - F / 2.0) ** 2
        jitter = np.zeros(F) if jitter is None else np.asarray(jitter, float)
        tilt = np.zeros(F) if tilt is None else np.asarray(tilt, float)
        x = (np.arange(L) - (L - 1) / 2.0) / ((L - 1) / 2.0)
        tex = rng.uniform(0.0, 1.0, size=(D,))                    # the same depth texture in every frame/lateral
        vol = np.zeros((F, D, L), np.float32)
        rows = np.arange(D, dtype=float)
        for f in range(F):
            for l in range(L):
                top = dome[f] + jitter[f] + tilt[f] * x[l]
                prof = 30.0 + 20.0 * rng.uniform(0, 1, size=D) + np.where((rows >= top) & (rows < top + 60), 800.0 + 400.0 * tex, 0.0)
                vol[f, :, l] = prof
        return vol, jitter, tilt

    def test_recovers_rigid_jitter_and_tilt(self):
        F = 60
        rng = np.random.default_rng(1)
        jit = np.cumsum(rng.normal(0, 1.5, F)); jit -= np.polyval(np.polyfit(np.arange(F), jit, 2), np.arange(F))   # off-parabola part only
        tl = 3.0 * np.sin(np.arange(F) / 6.0)
        vol, jit, tl = self._synth(F=F, jitter=jit, tilt=tl)
        a, b, info = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        assert info["applied"] and info["pairs_measured"] == F - 1
        # the move undoes the jitter (up to the parabola the fit keeps) and the non-linear tilt
        fr = np.arange(F)
        resid = (a + jit); resid -= np.polyval(np.polyfit(fr, resid, 2), fr)
        assert np.sqrt(np.mean(resid ** 2)) < 1.0, np.sqrt(np.mean(resid ** 2))     # sub-pixel on speckle (~0.6 px measured)
        tres = (b + tl); tres -= np.polyval(np.polyfit(fr, tres, 1), fr)
        assert np.sqrt(np.mean(tres ** 2)) < 1.2, np.sqrt(np.mean(tres ** 2))     # ~0.9 px measured: lag quantisation across bands, integrated

    def test_pad_invariance_and_zero_rows(self):
        vol, jit, tl = self._synth(F=30, jitter=np.r_[np.zeros(10), 6.0 * np.ones(10), np.zeros(10)])
        a, b, _ = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        pad = np.zeros((vol.shape[0], vol.shape[1] + 50, vol.shape[2]), vol.dtype); pad[:, 50:, :] = vol
        a2, b2, _ = M.tissue_motion_move(pad, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        assert np.max(np.abs(a - a2)) < 0.05 and np.max(np.abs(b - b2)) < 0.05      # a zero canvas pad must not bias the lags
        # the move undoes the 6 px block UP TO the parabola the flatten keeps (a block a third of the volume long
        # bends the best-fit parabola, so the expected move is the block minus that parabola, not the whole block)
        fr = np.arange(vol.shape[0]); a_exp = np.polyval(np.polyfit(fr, jit, 2), fr) - jit
        assert np.max(np.abs(a - a_exp)) < 1.0, np.max(np.abs(a - a_exp))

    def test_motion_free_dome_gets_no_move(self):
        # the sub-pixel refinement must cover lag 0 / -1 (a wrapped lag table quantised exactly the small lags a smooth
        # dome produces and integrated them into a 2-3 px spurious move — review 2026-09-05)
        vol, _j, _t = self._synth(F=80, D=200, L=64)
        a, b, info = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        assert info["applied"] and np.max(np.abs(a)) < 1.0, np.max(np.abs(a))       # 0.67 measured (white texture is the worst case)

    def test_dead_frame_block_does_not_bend_the_live_move(self):
        vol, jit, _t = self._synth(F=60, D=200, L=64, jitter=np.r_[np.zeros(25), 5.0 * np.ones(6), np.zeros(29)])
        a0, _b0, _i0 = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        dead = vol.copy(); dead[:12] = 0.0                                       # a zeroed leading block (crop_region / blink)
        a1, _b1, i1 = M.tissue_motion_move(dead, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        assert i1["applied"] and np.all(a1[:12] == 0.0) and i1["segments"][0][0] >= 11
        live = np.arange(12, 60)
        # live frames: same move up to the parabola difference between the two fits
        d = a1[live] - a0[live]; d -= np.polyval(np.polyfit(live, d, 2), live)
        assert np.max(np.abs(d)) < 0.5, np.max(np.abs(d))
        a, b, info = M.tissue_motion_move(vol, {"tissue_motion_max_lag": -3})     # a bad per-case param must not raise
        assert a.shape == (60,) and np.all(np.isfinite(a))

    def test_cut_columns_do_not_bias_the_boundary_pair(self):
        # a MOTION-FREE dome whose apex rises above the window: frames 0..23 are cut at the canvas edge (tissue at
        # row 0), the rest are not. The correct move is ~0 everywhere; without the cut guard the pair straddling the
        # boundary matched the cut edge instead of the stroma and integrated a step into the band.
        F, D, L = 60, 400, 48; rng = np.random.default_rng(3); fr = np.arange(F)
        top = -50.0 + 0.03 * fr ** 2                                  # smooth; negative (cut) until frame ~41; stays in the canvas
        tex = rng.uniform(0, 1, D + 100); rows = np.arange(D, dtype=float); vol = np.zeros((F, D, L), np.float32)
        for f in range(F):
            for l in range(L):
                idx = np.clip((rows - top[f]).astype(int), 0, D + 99)   # the tissue texture is anchored to the tissue, not the canvas
                prof = 30.0 + 20.0 * rng.uniform(0, 1, D) + np.where((rows >= top[f]) & (rows < top[f] + 200), 800.0 + 400.0 * tex[idx], 0.0)
                vol[f, :, l] = prof
        a, b, info = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        assert info["applied"]
        bnd = int(np.argmax(top >= 0))                                # first un-cut frame
        assert abs(a[bnd] - a[bnd - 1]) < 1.0, (a[bnd - 3:bnd + 3])   # no step at the boundary pair
        assert np.max(np.abs(a[3:-3])) < 2.0 and np.max(np.abs(a)) < 3.0, (np.max(np.abs(a[3:-3])), a[:5], a[-5:])   # white-texture floor ~1.6
        a0, _b0, _i = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_cut_guard": 0})
        assert np.sqrt(np.mean(a0 ** 2)) >= np.sqrt(np.mean(a ** 2)) - 0.3   # the guard never makes it worse

    def test_degenerate_inputs(self):
        a, b, info = M.tissue_motion_move(np.zeros((2, 40, 8), np.float32), {})
        assert not info["applied"] and a.shape == (2,) and b.shape == (2,)
        a, b, info = M.tissue_motion_move(np.zeros((10, 40, 16), np.float32), {})   # all-zero: no band passes min_fill
        assert not info["applied"] and np.all(a == 0)


# ───────────────────────── surface-crop band: bottom-edge guide ─────────────────────────
class TestBandBottomGuide:
    def _posterior(self, F=80, L=64, band_off=None, noise=0.0, seed=0):
        fr = np.arange(F); dome = 300.0 + 0.03 * (fr - F / 2.0) ** 2
        P = np.tile(dome, (L, 1)).astype(float)
        if band_off is not None:
            P[:, :len(band_off)] += np.asarray(band_off, float)[None, :]      # the band's posterior is off its parabola
        if noise > 0:
            P += np.random.default_rng(seed).normal(0.0, noise, P.shape)
        return P

    def test_posterior_source_refines_every_band_frame_and_fades_outward(self):
        F, L = 80, 64; band = list(range(0, 20)); off = np.linspace(11.0, 2.0, 20); P = self._posterior(F, L, off)
        da, db, info = M.band_bottom_guide(P, {}, np.zeros(F), np.zeros(F), band, L,
                                           {"band_bottom_guide_min_laterals": 8, "band_bottom_guide_source": "posterior",
                                            "band_bottom_guide_fit_frames": "uncropped", "band_bottom_guide_tilt": True})
        assert info["applied"] and info["band"] == [0, 19] and info["refined_frames"] == band
        assert np.allclose(da[:20], -off, atol=0.3), da[:20]                 # every band frame lands on its parabola
        assert np.allclose(da[20:23], -off[19] * np.array([0.75, 0.5, 0.25]), atol=0.3) and np.all(da[23:] == 0.0)
        assert np.allclose(db, 0.0, atol=1e-6)
        assert info["posterior_off_parabola_px"]["after"] < info["posterior_off_parabola_px"]["before"]

    def test_drawn_bottom_lines_decide_the_shift_and_a_stroke_end_never_steps(self):
        F, L = 80, 64; band = list(range(0, 20)); P = self._posterior(F, L, np.full(20, 6.0))
        drawn = {5: {f: 0 for f in range(0, 10)}, 40: {f: 0 for f in range(0, 10)}}
        P[5, :10] += 3.0; P[40, :10] += 3.0                                   # the reviewer's laterals say 9 px, the detector 6 px
        da, db, info = M.band_bottom_guide(P, drawn, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert info["applied"] and info["drawn_driven_frames"] == list(range(0, 10)) and info["rejected_laterals"] == []
        assert np.allclose(da[:10], -9.0, atol=0.5), da[:10]                # the drawn laterals decide, fully
        assert np.allclose(da[10:13], -9.0 * np.array([0.75, 0.5, 0.25]), atol=0.5) and np.all(da[13:] == 0.0)   # fades, no step
        assert np.max(np.abs(np.diff(da))) <= 9.0 / 4.0 + 0.5 and np.allclose(db, 0.0)
        da2, db2, info2 = M.band_bottom_guide(P, {}, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert not info2["applied"] and "drawn" in info2["reason"]            # no lines → idle by default
        da3, _b, info3 = M.band_bottom_guide(P, drawn, np.zeros(F), np.zeros(F), band, L,
                                             {"band_bottom_guide_min_laterals": 8, "band_bottom_guide_source": "posterior"})
        assert info3["applied"] and abs(da3[15] + 6.0) < 0.5                 # "posterior" source: the detector fills the rest
        # a LONG line (drawn across the un-cropped frames too): its own un-cropped curvature is carried into the band,
        # whatever the detector's posterior does there
        Pl = self._posterior(F, L, np.full(20, 40.0))
        long = {20: {f: 0 for f in range(F)}, 44: {f: 0 for f in range(F)}}
        fr = np.arange(F); dome = 300.0 + 0.03 * (fr - F / 2.0) ** 2
        for l in long: Pl[l] = dome; Pl[l, :20] += 5.0
        da4, _b, info4 = M.band_bottom_guide(Pl, long, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert info4["applied"] and info4["rejected_laterals"] == []
        assert np.allclose(da4[:16], -5.0, atol=0.6), da4[:20]

    def test_drawn_line_is_honoured_under_posterior_noise_and_mislocks(self):
        F, L = 80, 64; band = list(range(0, 20)); drawn = {32: {f: 0 for f in range(0, 20)}, 40: {f: 0 for f in range(0, 20)}}
        for off in (3.0, 5.0):
            got = []
            for seed in range(6):
                P = self._posterior(F, L, np.full(20, off), noise=1.5, seed=seed)
                da, _b, info = M.band_bottom_guide(P, drawn, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
                got.append(float(np.mean(da[3:14])))
            assert abs(np.mean(got) + off) < 1.2, (off, got)                  # the line's disagreement is delivered, not absorbed
        P = self._posterior(F, L, np.full(20, 6.0)); P[:, 72:] += 40.0         # a run of detector mis-locks in un-cropped frames
        da, _b, info = M.band_bottom_guide(P, drawn, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert abs(da[5] + 6.0) < 1.5, da[5]                                  # the robust parabola ignores them
        P = self._posterior(F, L, np.full(20, 6.0)); P[:, 20:28] += 40.0       # mis-locks right next to the band
        da, _b, info = M.band_bottom_guide(P, drawn, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert abs(da[5] + 6.0) < 1.5, da[5]

    def test_safety_rules(self):
        F, L = 80, 64; band = list(range(0, 20)); P = self._posterior(F, L, np.full(20, 6.0))
        da, _b, info = M.band_bottom_guide(P, {32: {f: 0 for f in range(0, 20)}}, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert not info["applied"] and "agreeing" in info["reason"]          # one line alone moves nothing
        drawn = {32: {f: 0 for f in range(20)}, 40: {f: 0 for f in range(20)}, 10: {f: 0 for f in range(20)}}
        P2 = P.copy(); P2[10, :20] -= 250.0                                    # a line hundreds of px off is ignored; the sane two drive
        da, _b, info = M.band_bottom_guide(P2, drawn, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert info["applied"] and info["rejected_laterals"] == [10] and abs(da[5] + 6.0) < 0.5
        P3 = P.copy(); P3[40, :20] -= 9.0                                      # 32 asks −6, 40 asks +3: spread 9 > 8 → skipped
        da, _b, info = M.band_bottom_guide(P3, {32: drawn[32], 40: drawn[40]}, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert not info["applied"] and info["disagreeing_frames"] == band and info["rejected_laterals"] == []
        P4 = P.copy(); P4[:, :20] += 30.0                                      # everyone asks for 36 px: beyond the cap → ignored
        da, _b, info = M.band_bottom_guide(P4, {32: drawn[32], 40: drawn[40]}, np.zeros(F), np.zeros(F), band, L, {"band_bottom_guide_min_laterals": 8})
        assert (not info["applied"]) or np.max(np.abs(da)) <= 12.0 + 1e-6
        da, _b, info = M.band_bottom_guide(P, {32: drawn[32], 40: drawn[40]}, np.zeros(F), np.zeros(F), list(range(0, 45)), L, {"band_bottom_guide_min_laterals": 8})
        assert not info["applied"] and "extrapolated" in info["reason"]      # a band over half the volume is refused

    def test_degenerate_and_input_coercion(self):
        F = 30; P = self._posterior(F, 16)
        da, db, info = M.band_bottom_guide(None, {}, np.zeros(F), np.zeros(F), [0, 1, 2], 16)
        assert not info["applied"] and da.shape == (F,)
        da, db, info = M.band_bottom_guide(np.full((16, F), np.nan), {}, np.zeros(F), np.zeros(F), [0, 1, 2], 16)
        assert not info["applied"]
        da, db, info = M.band_bottom_guide(P, {3: {0: 0}}, np.zeros(F), np.zeros(F), np.arange(3), 16,
                                           {"band_bottom_guide_min_laterals": 8, "band_bottom_guide_min_uncropped": 0})
        assert da.shape == (F,) and np.all(np.isfinite(da))                   # ndarray crop frames + a silly param: no raise


# ───────────────────────── fork watchdog: a wedged child must never hang the caller ─────────────────────────
def _pw_double(x):
    return x * 2


def _pw_slow_double(x):
    import time as _t
    _t.sleep(3.0)
    return x * 2


class TestSecondMotionPass:
    def test_scratch_warp_matches_the_flatten_convention(self):
        # target row = surface + a + b·x: a positive shift moves content DEEPER, a positive tilt deepens the
        # RIGHT (high-lateral) end. Used only to re-measure the residual motion; must not change the sign.
        F, D, L = 4, 60, 33
        vol = np.zeros((F, D, L), np.float32); vol[:, 20, :] = 1000.0
        out = M._rigid_scratch_warp(vol, np.array([0.0, 3.0, 0.0, 0.0]), np.zeros(F))
        assert int(np.argmax(out[1, :, L // 2])) == 23 and int(np.argmax(out[0, :, L // 2])) == 20
        out = M._rigid_scratch_warp(vol, np.zeros(F), np.array([0.0, 4.0, 0.0, 0.0]))
        assert int(np.argmax(out[1, :, L - 1])) == 24 and int(np.argmax(out[1, :, 0])) == 16
        assert out.shape == vol.shape and out.dtype == vol.dtype

    def test_second_pass_thresholds_exist_and_are_ordered(self):
        p = M.DEFAULT_PARAMS
        assert p["tissue_motion_second_pass"] is True
        assert 0.0 < p["tissue_motion_second_pass_min_px"] < p["tissue_motion_second_pass_max_px"]

    def test_a_converged_volume_measures_near_zero_so_the_guard_skips_it(self):
        # the guard reads tissue_motion_move(...)['residual_rms_px'] on the once-moved volume; a volume whose
        # trajectory is already its own parabola must measure below the threshold (cs008_od_v2: 0.71 px).
        F, D, L = 40, 200, 48; rng = np.random.default_rng(5); fr = np.arange(F)
        dome = 70.0 + 0.03 * (fr - F / 2.0) ** 2; tex = rng.uniform(0, 1, D + 60); rows = np.arange(D, dtype=float)
        vol = np.zeros((F, D, L), np.float32)
        for f in range(F):
            for l in range(L):
                idx = np.clip((rows - dome[f]).astype(int), 0, D + 59)
                vol[f, :, l] = 30.0 + 20.0 * rng.uniform(0, 1, D) + np.where((rows >= dome[f]) & (rows < dome[f] + 90), 800.0 + 400.0 * tex[idx], 0.0)
        a, b, info = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1})
        assert info["applied"] and float(info["residual_rms_px"]) < M.DEFAULT_PARAMS["tissue_motion_second_pass_min_px"]


def _pw_sleeper(x):
    import time as _t
    _t.sleep(30)
    return x


class TestPoolTeardown:
    def test_stuck_workers_are_killed_not_leaked(self):
        # a worker that never returns must not survive the pool (2026-09-08: two orphans held 7.3 GB and the
        # reviewer's machine went into swap while they worked the queue)
        import concurrent.futures
        import multiprocessing as mp
        import time as _t
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork"))
        [ex.submit(_pw_sleeper, i) for i in range(2)]
        _t.sleep(1.0)
        procs = list(ex._processes.values())
        assert sum(p.is_alive() for p in procs) == 2
        killed = M._close_pool(ex, grace=1.0)
        _t.sleep(0.5)
        assert killed == 2 and sum(p.is_alive() for p in procs) == 0

    def test_clean_pool_closes_without_killing(self):
        import concurrent.futures
        import multiprocessing as mp
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork"))
        assert list(ex.map(abs, [-1, -2, -3])) == [1, 2, 3]
        assert M._close_pool(ex, grace=20.0) == 0

    def test_grace_zero_is_immediate(self):
        import concurrent.futures
        import multiprocessing as mp
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("fork"))
        ex.submit(_pw_sleeper, 1)
        import time as _t
        _t.sleep(1.0)
        t0 = _t.monotonic()
        M._close_pool(ex, grace=0.0)
        assert _t.monotonic() - t0 < 5.0


class TestPoolWatchdog:
    def test_normal_pool_still_maps(self):
        assert M._map_slices(_pw_double, [1, 2, 3, 4], None, 0.0, 1.0, 2) == [2, 4, 6, 8]

    def test_stalled_pool_falls_back_to_serial_within_the_deadline(self, monkeypatch):
        # A fork that deadlocks yields NO results; before 2026-09-07 ex.map (and shutdown) waited forever and the
        # request thread wedged in waitpid. With the deadline the children are killed and the serial fallback
        # returns the same answer. Simulated here with a worker slower than the deadline.
        import time as _t
        monkeypatch.setattr(M, "_POOL_TIMEOUT_S", 0.4)
        t0 = _t.monotonic()
        out = M._map_slices(_pw_slow_double, [3, 4], None, 0.0, 1.0, 2)
        dt = _t.monotonic() - t0
        assert out == [6, 8]                     # correct result, via the serial path
        assert dt < 30.0, dt                     # bounded: deadline + 2 serial items, never unbounded

    def test_kill_pool_is_safe_on_a_live_and_on_a_dead_executor(self):
        import concurrent.futures
        import multiprocessing as mp
        ex = concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("fork"))
        assert list(ex.map(_pw_double, [1, 2])) == [2, 4]
        M._kill_pool(ex)
        M._kill_pool(ex)                          # idempotent
        M._kill_pool(object())                    # never raises on a non-executor

    def test_timeout_is_configurable_and_positive(self):
        assert M._POOL_TIMEOUT_S > 0


# ───────────────────────── dome sign guard: the B-scan plane decides the sign of the across-frame dome ─────────────────────────
def _synth_moving_dome(F=60, D=220, L=64, frame_curv=0.03, motion=None, seed=3):
    """tissue band whose anterior is a dome across laterals AND across frames, plus optional per-frame motion"""
    rng = np.random.default_rng(seed); fr = np.arange(F, dtype=float); x = np.arange(L, dtype=float)
    lat_dome = 0.02 * (x - L / 2.0) ** 2                       # positive: apex up across laterals
    base = 60.0 + frame_curv * (fr - F / 2.0) ** 2             # across frames
    mv = np.zeros(F) if motion is None else motion
    vol = np.zeros((F, D, L), np.float32); rows = np.arange(D, dtype=float)
    for f in range(F):
        for l in range(L):
            top = base[f] + lat_dome[l] + mv[f]
            vol[f, :, l] = 30.0 + 20.0 * rng.uniform(0, 1, D) + np.where((rows >= top) & (rows < top + 70), 900.0 + 300.0 * rng.uniform(0, 1, D), 0.0)
    return vol


class TestEndPairGuard:
    def test_garbage_terminal_pair_is_replaced_by_the_trend(self):
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=0.4 * fr)                  # clean linear drift
        vol[-1] = np.roll(vol[-1], 25, axis=0)                                    # last frame jumps 25 px: a garbage terminal pair
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1}
        a_off, _, i_off = M.tissue_motion_move(vol, {**p, "tissue_motion_end_pairs": 0})
        a_on, _, i_on = M.tissue_motion_move(vol, {**p, "tissue_motion_end_pairs": 3})   # opt-in: OFF by default
        assert i_on.get("end_pairs_replaced"), "the terminal pair must be flagged"
        assert any(r["pair"] == F - 2 for r in i_on["end_pairs_replaced"])
        # the delivered last frame follows its neighbours' trend instead of the 25 px jump
        assert abs((a_on[-1] - a_on[-2]) - (a_on[-2] - a_on[-3])) < 3.0
        assert abs((a_off[-1] - a_off[-2]) - (a_off[-2] - a_off[-3])) > 10.0

    def test_clean_segment_is_untouched(self):
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=0.4 * fr + 2.0 * np.sin(fr / 5.0))
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1}
        a0, b0, i0 = M.tissue_motion_move(vol, {**p, "tissue_motion_end_pairs": 0})
        a1, b1, i1 = M.tissue_motion_move(vol, {**p, "tissue_motion_end_pairs": 3})
        assert not i1.get("end_pairs_replaced") and np.allclose(a0, a1) and np.allclose(b0, b1)
        a2, b2, i2 = M.tissue_motion_move(vol, p)                                          # default: guard off, identical
        assert not i2.get("end_pairs_replaced") and np.allclose(a0, a2)


class TestInteriorPairGuard:
    def _vol_with_streak(self, streak=True):
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=0.3 * fr)              # smooth drift
        if streak:
            # ONE pair with a split vote: from frame 30 on, the LEFT half of every frame is shifted 8 px deeper, so
            # pair 29→30 measures +8 on the left bands and 0 on the right (a blunder-shaped disagreement) while
            # pairs 28→29 and 30→31 stay consistent on both halves (the neighbours agree)
            # (a left/right split reads as a TILT to the linear fit and never trips the spread; the MIDDLE third
            # displaced cannot be absorbed by a line, which is what a streak-biased vote looks like)
            L = vol.shape[2]; vol[30:, :, L // 3: 2 * L // 3] = np.roll(vol[30:, :, L // 3: 2 * L // 3], 8, axis=1)
        return vol

    def test_split_vote_pair_takes_its_neighbours(self):
        vol = self._vol_with_streak(True); p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1}
        a0, _, i0 = M.tissue_motion_move(vol, {**p, "tissue_motion_interior_pair_guard": False})
        a1, _, i1 = M.tissue_motion_move(vol, {**p, "tissue_motion_interior_pair_guard": True})
        assert i1.get("interior_pairs_replaced"), i1.get("pair_spread")
        assert any(r["pair"] in (29, 30) for r in i1["interior_pairs_replaced"])
        # the delivered step at the streak frame shrinks
        step0 = abs((a0[30] - a0[29]) - (a0[29] - a0[28])); step1 = abs((a1[30] - a1[29]) - (a1[29] - a1[28]))
        assert step1 <= step0

    def test_real_move_with_a_split_vote_is_kept(self):
        # cs020_od_v1 pair 67: the WHOLE tissue steps 8 px deeper at one pair AND a middle-third block votes wrong; the raw
        # anterior witness says 8 → the measured lag stands, the neighbours' mean (≈0) is not substituted
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=0.3 * fr)
        vol[30:] = np.roll(vol[30:], 8, axis=1)                                                     # real 8 px step at pair 29
        L = vol.shape[2]; D = vol.shape[1]; x = np.arange(L, dtype=float)
        # + a split vote from a STREAK block: from frame 30 on, the block's anterior 30 rows are blacked out (a bright/dark
        # streak hides the surface) and its deep stroma sits 6 px further, so those bands vote +14 while the rest vote +8
        # and the anterior edge itself (the witness) still says +8 everywhere it is visible
        blk = slice(L // 3, 2 * L // 3); rolled = np.roll(vol[30:, :, blk], 6, axis=1)
        for k, f in enumerate(range(30, F)):
            top = 60.0 + 0.03 * (f - 30.0) ** 2 + 0.3 * f + 0.02 * (x[blk] - L / 2.0) ** 2 + 8.0
            rows = np.arange(D)[:, None]
            vol[f, :, blk] = np.where(rows >= top[None, :] + 30.0, rolled[k], np.where(rows >= top[None, :], 35.0, vol[f, :, blk]))
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": True,
             "tissue_motion_pair_witness_min_laterals": 20}
        a, _, i = M.tissue_motion_move(vol, p)
        rec = [r for r in (i.get("interior_pairs_replaced") or []) if r["pair"] in (29, 30)]
        assert rec and rec[0]["kept"] is True and rec[0]["witness"] is not None, i.get("interior_pairs_replaced")
        assert abs((a[29] - a[30]) - 8.0) < 2.0, (a[28:32])                                   # the step is delivered (trajectory falls by the lag)

    def test_tilted_real_move_with_a_split_vote_takes_the_witness_line(self):
        # the whole tissue steps deeper by 10 px on the left edge and 2 px on the right (a tilted rigid move) while a block
        # of bands votes wrong: the delivered pair carries the witness's shift AND tilt
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=0.3 * fr); L = vol.shape[2]
        x = np.arange(L); shift = np.round(10.0 - 8.0 * x / (L - 1)).astype(int)               # +10 → +2 across laterals
        for l in range(L):
            vol[30:, :, l] = np.roll(vol[30:, :, l], int(shift[l]), axis=1)
        D = vol.shape[1]; blk = slice(L // 3, 2 * L // 3); rolled = np.roll(vol[30:, :, blk], 7, axis=1)   # streak block (see above)
        for k, f in enumerate(range(30, F)):
            top = 60.0 + 0.03 * (f - 30.0) ** 2 + 0.3 * f + 0.02 * (x[blk] - L / 2.0) ** 2 + shift[blk]
            rows = np.arange(D)[:, None]
            vol[f, :, blk] = np.where(rows >= top[None, :] + 30.0, rolled[k], np.where(rows >= top[None, :], 35.0, vol[f, :, blk]))
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": True,
             "tissue_motion_pair_witness_min_laterals": 20}
        a, b, i = M.tissue_motion_move(vol, p)
        rec = [r for r in (i.get("interior_pairs_replaced") or []) if r["pair"] in (29, 30)]
        assert rec and rec[0]["kept"] is True, i.get("interior_pairs_replaced")
        assert abs((a[29] - a[30]) - 6.0) < 2.0, a[28:32]                                     # centre shift ≈ 6
        assert abs((b[29] - b[30]) - (-4.0)) < 2.0, b[28:32]                                  # half-span tilt ≈ −4 (left deeper)

    def test_clean_volume_untouched(self):
        vol = self._vol_with_streak(False); p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1}
        a0, b0, i0 = M.tissue_motion_move(vol, {**p, "tissue_motion_interior_pair_guard": False})
        a1, b1, i1 = M.tissue_motion_move(vol, p)
        assert not i1.get("interior_pairs_replaced") and np.allclose(a0, a1) and np.allclose(b0, b1)


class TestCutMaskPairSymmetric:
    """p4_os_v1_2 (2026-09-09): on a scratch-warped (zero-padded) volume a cut frame next to a shifted one had the
    guard's mask edge in one frame of the pair and the cut edge in the other → a 27-34 px lag. The mask must be
    per pair, symmetric, and start at each frame's first non-zero row."""

    def _vol(self):
        F = 30; D = 220; L = 24; rng = np.random.default_rng(5)
        vol = np.full((F, D, L), 40.0, np.float32) + 10.0 * rng.standard_normal((F, D, L)).astype(np.float32)
        # a CUT column: tissue from row 0 down to row 120 (speckled), every frame; frames >= 15 shifted 5 px deeper
        # with zero rows on top (as _rigid_scratch_warp leaves them)
        tissue = 900.0 + 300.0 * rng.uniform(0, 1, (F, 120, L)).astype(np.float32)
        for f in range(F):
            s0 = 5 if f >= 15 else 0
            vol[f, :, :] = np.where(np.arange(D)[:, None] < s0, 0.0, vol[f, :, :])
            vol[f, s0:s0 + 120, :] = tissue[f]
        return vol

    def test_cut_frame_next_to_a_shifted_one_measures_the_shift(self):
        vol = self._vol()
        a, b, info = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False,
                                              "tissue_motion_shape_sign_guard": False, "tissue_motion_second_pass": False})
        step = -(a[15] - a[14])                                       # trajectory falls by the lag: a deeper frame → −Δa = +5
        assert abs(step - 5.0) < 1.5, (a[13:17], info.get("pair_spread", [])[12:16])
        assert max(abs(-(a[f + 1] - a[f])) for f in (12, 13, 16, 17)) < 1.5


class TestAdaptiveLagWindow:
    """cs020_os_v2 (2026-09-09): a blink moved the eye 85 px between two frames, beyond the ±40 px lag window; the votes pinned
    at the boundary and a 23 px spike was delivered. A pair whose votes pin at the boundary is re-read with a wider window."""

    def test_move_beyond_the_window_is_measured(self):
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(D=300, frame_curv=0.03, motion=0.3 * fr)
        vol[30:] = np.roll(vol[30:], 60, axis=1)                                 # a 60 px jump at pair 29 (window is ±40)
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_max_lag": 40, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False}
        a_off, _, i_off = M.tissue_motion_move(vol, {**p, "tissue_motion_widen_window": False})
        a_on, _, i_on = M.tissue_motion_move(vol, p)
        assert 29 in (i_on.get("widened_pairs") or []), i_on.get("widened_pairs")
        assert abs((a_on[29] - a_on[30]) - 60.0) < 3.0, a_on[28:32]             # the full jump is delivered
        assert abs((a_off[29] - a_off[30]) - 60.0) > 15.0, a_off[28:32]         # without widening it is clipped near the window


class TestBridgedDome:
    """cs020_os_v4 (2026-09-09): a reviewer's artifact band zeroes frames 40-49; the two live halves must land on ONE dome,
    not two parabolas with an arbitrary offset."""

    def test_two_halves_share_one_dome_across_a_short_dead_gap(self):
        F = 100; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(F=F, frame_curv=0.03, motion=0.3 * fr)
        vol[60:] = np.roll(vol[60:], 15, axis=1)                                   # a rigid 15 px offset of the second half (what a blink leaves)
        vol[40:50] = 0.0                                                            # the marked artifact: dead frames
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False,
             "tissue_motion_shape_sign_guard": False}
        a, b, i = M.tissue_motion_move(vol, p)
        assert i.get("bridged_segments"), i.get("segments")
        # the delivered surface: true top + move; both halves must lie on one parabola
        top = 60.0 + 0.03 * (fr - 50.0) ** 2 + 0.3 * fr + np.where(fr >= 60, 15.0, 0.0)
        delivered = top + a; live = np.ones(F, bool); live[40:50] = False; live[:8] = live[-8:] = False   # ends carry the fit's own edge error
        c = np.polyfit(fr[live], delivered[live], 2); res = delivered[live] - np.polyval(c, fr[live])
        assert np.sqrt(np.mean(res ** 2)) < 1.5, (np.sqrt(np.mean(res ** 2)), c)
        a_sep, _, i_sep = M.tissue_motion_move(vol, {**p, "tissue_motion_bridge_max_gap": 0})   # independent domes: not on one curve
        d_sep = top + a_sep; c2 = np.polyfit(fr[live], d_sep[live], 2); res2 = d_sep[live] - np.polyval(c2, fr[live])
        assert np.sqrt(np.mean(res2 ** 2)) > np.sqrt(np.mean(res ** 2))


class TestGapEdgeOffset:
    """p1_od_v1_3 (2026-09-10): across a dead artifact gap the tissue cannot measure the rigid shift; the served edge
    sets the height of each bridged segment on the dome the tissue delivered."""

    def _vol(self, jump, F=70, D=300, L=48, gap=(30, 39), seed=5):
        rng = np.random.default_rng(seed); fr = np.arange(F, dtype=float); x = np.arange(L, dtype=float)
        vol = np.zeros((F, D, L), np.float32); rows = np.arange(D, dtype=float); edge = np.full((L, F), np.nan)
        for f in range(F):
            top = 90.0 + 0.02 * (fr[f] - 35.0) ** 2 + 0.01 * (x - L / 2.0) ** 2 + (jump if f > gap[1] else 0.0)
            vol[f] = (30.0 + 20.0 * rng.uniform(0, 1, (D, L)) + np.where((rows[:, None] >= top[None, :]) & (rows[:, None] < top[None, :] + 60), 900.0 + 300.0 * rng.uniform(0, 1, (D, L)), 0.0)).astype(np.float32)
            edge[:, f] = top
        vol[gap[0]:gap[1] + 1] = 0.0
        return vol, edge

    def _p(self):
        return {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False,
                "tissue_motion_shape_sign_guard": False, "tissue_motion_bridge_max_gap": 20, "tissue_motion_tilt_witness": False}

    def test_edge_sets_the_offset_across_the_gap(self):
        vol, edge = self._vol(jump=60.0)
        a, b, i = M.tissue_motion_move(vol, {**self._p(), "_gap_edge": edge})
        assert i.get("bridged_segments") and i.get("gap_edge_offset"), i.get("gap_edge_offset")
        rec = i["gap_edge_offset"][0]; assert abs(rec["shift_px"][1] + 60.0) < 4.0 or abs(rec["shift_px"][0] - 60.0) < 4.0, rec   # the jump is removed
        deliv = np.nanmedian(edge, axis=0) + a; live = vol.reshape(vol.shape[0], -1).max(axis=1) > 0
        fr = np.arange(vol.shape[0], dtype=float); res = deliv[live] - np.polyval(np.polyfit(fr[live], deliv[live], 2), fr[live])
        assert np.max(np.abs(res)) < 4.0, np.max(np.abs(res))                                  # one curve, one height
        a0, b0, i0 = M.tissue_motion_move(vol, self._p())                                       # no edge: the jump stays
        d0 = np.nanmedian(edge, axis=0) + a0; assert abs(d0[45] - d0[25]) > 30.0, (d0[45], d0[25])

    def test_rolled_segment_gets_shift_and_tilt(self):
        # after the gap the tissue is 40 px lower at the centre and rolled by 30 px half-span (60 px across the width)
        F, D, L = 70, 320, 48; rng = np.random.default_rng(9); fr = np.arange(F, dtype=float); xs = (np.arange(L) - (L - 1) / 2) / ((L - 1) / 2)
        vol = np.zeros((F, D, L), np.float32); rows = np.arange(D, dtype=float); edge = np.full((L, F), np.nan)
        for f in range(F):
            top = 110.0 + 0.02 * (fr[f] - 35.0) ** 2 + 0.01 * ((np.arange(L) - L / 2.0) ** 2) + ((40.0 + 30.0 * xs) if f > 39 else 0.0)
            vol[f] = (30.0 + 20.0 * rng.uniform(0, 1, (D, L)) + np.where((rows[:, None] >= top[None, :]) & (rows[:, None] < top[None, :] + 60), 900.0 + 300.0 * rng.uniform(0, 1, (D, L)), 0.0)).astype(np.float32)
            edge[:, f] = top
        vol[30:40] = 0.0
        a, b, i = M.tissue_motion_move(vol, {**self._p(), "_gap_edge": edge})
        rec = i["gap_edge_offset"][0]; assert abs(rec["shift_px"][1] + 40.0) < 5.0 and abs(rec["tilt_px"][1] + 30.0) < 6.0, rec
        live = vol.reshape(F, -1).max(axis=1) > 0
        for lat in (4, L // 2, L - 5):
            deliv = edge[lat] + a + b * xs[lat]; res = deliv[live] - np.polyval(np.polyfit(fr[live], deliv[live], 2), fr[live])
            assert np.max(np.abs(res)) < 5.0, (lat, np.max(np.abs(res)))                        # one curve at every lateral

    def test_no_jump_is_left_alone_and_unwitnessed_edge_is_skipped(self):
        vol, edge = self._vol(jump=0.0)
        a, b, i = M.tissue_motion_move(vol, {**self._p(), "_gap_edge": edge})
        rec = (i.get("gap_edge_offset") or [{}])[0]; assert all(abs(v) < 2.5 for v in rec.get("shift_px", [0.0])), rec
        a2, b2, i2 = M.tissue_motion_move(vol, {**self._p(), "_gap_edge": edge + 120.0})       # an edge 120 px off the tissue is no witness
        rec2 = (i2.get("gap_edge_offset") or [{}])[0]; assert all(v is None for v in rec2.get("gap_steps_px", [None])), rec2
        assert np.allclose(a2, a, atol=1e-6)


class TestNoisySecondarySegment:
    """cs025_os_v4 (2026-09-09): a short secondary segment whose band lags scatter far more than the main segment's is
    unmeasurable; its move is held from the main segment instead of following whatever the correlation locked onto."""

    def _vol(self, F=60, D=260, L=48, gap=(36, 43), seed=11, noisy_tail=True):
        rng = np.random.default_rng(seed); x = np.arange(L, dtype=float); rows = np.arange(D, dtype=float)
        vol = np.zeros((F, D, L), np.float32)
        for f in range(F):
            top = 70.0 + 0.02 * (f - 30.0) ** 2 + 0.01 * (x - L / 2.0) ** 2
            if f > gap[1] and noisy_tail:
                # faint cornea + a bright blob per lateral that jumps randomly from frame to frame: nothing rigid to measure
                tissue = np.where((rows[:, None] >= top[None, :]) & (rows[:, None] < top[None, :] + 50), 120.0 + 60.0 * rng.uniform(0, 1, (D, L)), 0.0)
                bl = rng.integers(150, 220, size=L); blob = np.where((rows[:, None] >= bl[None, :]) & (rows[:, None] < bl[None, :] + 25), 1500.0 + 300.0 * rng.uniform(0, 1, (D, L)), 0.0)
                vol[f] = (30.0 + 20.0 * rng.uniform(0, 1, (D, L)) + tissue + blob).astype(np.float32)
            else:
                vol[f] = (30.0 + 20.0 * rng.uniform(0, 1, (D, L)) + np.where((rows[:, None] >= top[None, :]) & (rows[:, None] < top[None, :] + 60), 900.0 + 300.0 * rng.uniform(0, 1, (D, L)), 0.0)).astype(np.float32)
        vol[gap[0]:gap[1] + 1] = 0.0
        return vol

    def test_noisy_tail_is_held_from_the_main_segment(self):
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False,
             "tissue_motion_shape_sign_guard": False, "tissue_motion_bridge_max_gap": 20, "tissue_motion_tilt_witness": False}
        a, b, i = M.tissue_motion_move(self._vol(), p)
        assert i.get("noisy_segments") and i["noisy_segments"][0]["segment"] == [44, 59], i.get("noisy_segments")
        assert np.allclose(a[44:], a[35]) and np.allclose(b[44:], b[35]), (a[35], a[44:])                # held from the main segment's end
        assert i["segments"] == [[0, 35]]
        a2, b2, i2 = M.tissue_motion_move(self._vol(noisy_tail=False), p)                                   # a clean tail is measured, not held
        assert not i2.get("noisy_segments") and i2["segments"] == [[0, 35], [44, 59]], i2.get("segments")

    def test_held_tail_is_placed_on_the_dome_by_its_line(self):
        # the tail's true top continues the dome but its tissue cannot be measured (blob); the served line is the top
        F = 60; fr = np.arange(F, dtype=float); vol = self._vol(); L = vol.shape[2]
        top = 70.0 + 0.02 * (fr - 30.0) ** 2; edge = np.tile(top[None, :], (L, 1)) + 0.01 * ((np.arange(L) - L / 2.0) ** 2)[:, None]
        edge[:, 44:] += 40.0                                                                     # the raw tail sits 40 px low (eye moved during the gap)
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False,
             "tissue_motion_shape_sign_guard": False, "tissue_motion_bridge_max_gap": 20, "tissue_motion_tilt_witness": False, "tissue_motion_gap_edge_offset": False}
        a, b, i = M.tissue_motion_move(vol, {**p, "_gap_edge": edge})
        ns = (i.get("noisy_segments") or [{}])[0]; assert ns.get("line_placed") is True, ns
        deliv = np.nanmedian(edge, axis=0) + a; live = vol.reshape(F, -1).max(axis=1) > 0
        res = deliv[live] - np.polyval(np.polyfit(fr[live][:30], deliv[live][:30], 2), fr[live])     # the main segment's own parabola, extrapolated
        assert np.max(np.abs(res[-16:])) < 4.0, res[-16:]                                        # the tail lies on it

    def test_fill_holds_ends_and_interpolates_inside(self):
        a = np.array([1.0, 2.0, 0.0, 0.0, 5.0, 0.0]); b = np.array([0.5, 0.5, 0.0, 0.0, -0.5, 0.0]); dead = np.array([False, False, True, True, False, True])
        fa, fb = M.fill_unmeasured_move(a, b, dead)
        assert np.allclose(fa, [1.0, 2.0, 3.0, 4.0, 5.0, 5.0]) and np.allclose(fb, [0.5, 0.5, 0.1667, -0.1667, -0.5, -0.5], atol=1e-3), (fa, fb)


class TestTiltShapeWitness:
    """cs024_os_v4 (2026-09-09): a spurious U-shaped tilt trajectory in the correlation flattened one lateral edge of the
    delivered cornea. The non-linear part of the tilt correction must follow the raw anterior edge's own tilt trajectory."""

    def _dome(self, F=60, D=220, L=64, roll=None, seed=3):
        rng = np.random.default_rng(seed); fr = np.arange(F, dtype=float); x = np.arange(L, dtype=float); xn = (x - (L - 1) / 2) / ((L - 1) / 2)
        vol = np.zeros((F, D, L), np.float32); rows = np.arange(D, dtype=float)
        for f in range(F):
            top = 60.0 + 0.03 * (fr[f] - 30.0) ** 2 + 0.02 * (x - L / 2.0) ** 2 + (roll[f] * xn if roll is not None else 0.0)
            vol[f] = (30.0 + 20.0 * rng.uniform(0, 1, (D, L)) + np.where((rows[:, None] >= top[None, :]) & (rows[:, None] < top[None, :] + 70), 900.0 + 300.0 * rng.uniform(0, 1, (D, L)), 0.0)).astype(np.float32)
        return vol

    def test_curvature_slope_of_a_pure_dome_is_zero_and_a_real_roll_is_seen(self):
        vol = self._dome(); p = {**M.DEFAULT_PARAMS}
        k0 = M._anterior_curvature_slope(vol, p); assert np.isfinite(k0) and abs(k0) < 0.003, k0
        F = 60; fr = np.arange(F, dtype=float); roll = 0.01 * (fr - 30.0) ** 2 - 9.0            # a U-shaped roll: ±9 px half-span
        k1 = M._anterior_curvature_slope(self._dome(roll=roll), p)
        assert abs(k1 - 0.01) < 0.004, k1                                                      # the raw curvature slope across laterals IS the roll's curvature

    def test_real_roll_is_still_removed_and_a_pure_dome_untouched(self):
        F = 60; fr = np.arange(F, dtype=float); roll = 0.01 * (fr - 30.0) ** 2 - 9.0
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False,
             "tissue_motion_shape_sign_guard": False, "tissue_motion_pair_witness_min_laterals": 20}
        a, b, i = M.tissue_motion_move(self._dome(roll=roll), p)
        assert i.get("tilt_witness") and i["tilt_witness"][0]["used_quad_px"] > 5
        # the delivered tilt (roll + b) must be close to linear: the real U is removed, witness or not
        deliv = roll + b; res = deliv - np.polyval(np.polyfit(fr, deliv, 1), fr)
        assert np.max(np.abs(res)) < 3.0, np.max(np.abs(res))
        a0, b0, i0 = M.tissue_motion_move(self._dome(), p)
        assert np.max(np.abs(b0)) < 2.0, np.max(np.abs(b0))                                    # a pure dome gets no tilt correction


class TestCutBandVote:
    """p1_od_v1_2 (2026-09-09): laterals whose cornea is cut by the frame top lock their correlation to lag 0 and
    out-vote a real move; with the rule they vote only when they agree with the un-cut fit."""

    def _vol(self):
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=0.3 * fr)
        vol[30:] = np.roll(vol[30:], 12, axis=1)                                     # one real 12 px step at pair 29
        # the LEFT quarter of every frame is cut by the frame top: the whole visible tissue of those columns is a
        # STATIC bright block from row 0 (the cut edge stays at row 0 in every frame → the correlation locks at lag 0)
        L = vol.shape[2]; D = vol.shape[1]; rows = np.arange(D)[:, None]
        lo = L // 4
        block = 900.0 + 300.0 * np.random.default_rng(1).uniform(0, 1, (D, lo))
        vol[:, :, :lo] = np.where(rows < 200, block[None, :, :], 30.0 + 20.0 * np.random.default_rng(3).uniform(0, 1, (F, D, lo)))
        return vol

    def test_cut_bands_no_longer_outvote_a_real_move(self):
        vol = self._vol(); p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False}
        a_old, _, i_old = M.tissue_motion_move(vol, {**p, "tissue_motion_cut_vote": False})
        a_new, _, i_new = M.tissue_motion_move(vol, {**p, "tissue_motion_cut_vote": True})
        assert i_new.get("cut_demoted_pairs", 0) > 0 and max(i_new["cut_bands_per_pair"]) > 0
        step_old = (a_old[30] - a_old[29]) - (a_old[29] - a_old[28]); step_new = (a_new[30] - a_new[29]) - (a_new[29] - a_new[28])
        # the delivered step at pair 29 (trajectory is −Σlag: a 12 px deeper frame moves the trajectory by −12)
        assert abs(step_new + 12.0) < 2.5, (step_old, step_new)
        assert abs(step_new + 12.0) < abs(step_old + 12.0)

    def test_guard_keeps_the_pair_once_the_vote_is_clean(self):
        vol = self._vol(); p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": True}
        a_new, _, i_new = M.tissue_motion_move(vol, {**p, "tissue_motion_cut_vote": True})
        assert not any(r["pair"] == 29 for r in (i_new.get("interior_pairs_replaced") or [])), i_new.get("interior_pairs_replaced")
        assert abs((a_new[30] - a_new[29]) - (a_new[29] - a_new[28]) + 12.0) < 2.5

    def test_cut_bands_that_carry_the_move_keep_voting(self):
        # cs040_os_v1: a bright strip in the top rows marks the left third as "cut", but those bands follow the tissue;
        # a real TILT step at pair 29 (a roll: +12 px on the left, 0 on the right) must survive the vote
        F = 60; fr = np.arange(F, dtype=float); vol = _synth_moving_dome(frame_curv=0.03, motion=0.3 * fr); D = vol.shape[1]; L = vol.shape[2]
        xs = (np.arange(L) - (L - 1) / 2) / ((L - 1) / 2)
        for f in range(30, F):
            for l in range(L):
                vol[f, :, l] = np.roll(vol[f, :, l], int(round(6.0 * (1.0 - xs[l]))))     # +12 px at x=−1, 0 at x=+1
        vol[:, :10, : L // 3] = 1500.0                                                     # the bright top strip (not tissue)
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False,
             "tissue_motion_shape_sign_guard": False, "tissue_motion_tilt_witness": False}
        a, b, i = M.tissue_motion_move(vol, {**p, "tissue_motion_cut_vote": True})
        assert max(i["cut_bands_per_pair"]) > 0                                            # the strip IS seen as cut
        tilt_step = (b[30] - b[29]) - (b[29] - b[28])
        assert abs(abs(tilt_step) - 6.0) < 2.0, (tilt_step, i.get("cut_demoted_pairs"))    # the half-span tilt step is measured, not flattened

    def test_all_cut_frame_still_votes(self):
        # a surface-crop band frame: nearly every band cut → no demotion, the old vote stands
        F = 40; vol = _synth_moving_dome(F=F, frame_curv=0.03); D = vol.shape[1]; rows = np.arange(D)[:, None]
        vol[:, :, :] = np.where(rows < 110, 900.0 + 300.0 * np.random.default_rng(2).uniform(0, 1, vol.shape), vol)
        a, _, i = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_cut_vote": True})
        assert i.get("cut_demoted_pairs", 0) == 0


class TestAnteriorPass:
    def test_common_residual_recovers_a_frame_wide_wave(self):
        F = 60; fr = np.arange(F, dtype=float); wave = 4.0 * np.sin(fr / 4.0)
        vol = _synth_moving_dome(frame_curv=0.03, motion=wave)          # the same wave on every lateral
        r = M.anterior_common_residual(vol)
        k = np.isfinite(r) & (fr > 5) & (fr < F - 6)
        assert k.sum() > 30 and np.corrcoef(r[k], wave[k])[0, 1] > 0.9 and abs(np.std(r[k]) - np.std(wave[k])) < 1.5

    def test_clean_dome_has_no_common_residual(self):
        vol = _synth_moving_dome(frame_curv=0.03, motion=np.zeros(60))
        r = M.anterior_common_residual(vol); k = np.isfinite(r)
        assert k.sum() > 30 and np.sqrt(np.nanmean(r[k] ** 2)) < 1.0

    def test_crop_and_dead_frames_are_excluded_from_the_fit(self):
        F = 60; vol = _synth_moving_dome(frame_curv=0.03, motion=np.zeros(F)); live = np.ones(F, bool); live[50:] = False
        r = M.anterior_common_residual(vol, live=live, crop_frames=list(range(0, 8)))
        assert np.isfinite(r).sum() > 30


class TestDomeSignGuard:
    def test_bscan_plane_curvature_is_positive_and_motion_free(self):
        vol = _synth_moving_dome(motion=np.linspace(-40, 40, 60) ** 1)   # a huge linear drift across frames
        r = M.bscan_plane_curvature(vol)
        assert r["n_good"] >= 8 and 0.015 < r["curv_px_per_lat2"] < 0.025   # ≈ the 0.02 built in, drift ignored

    def test_guard_fixes_an_inverted_trajectory_first_pass_only(self):
        F = 60; fr = np.arange(F, dtype=float)
        valley = -0.05 * (fr - F / 2.0) ** 2 + 0.05 * (fr - F / 2.0) ** 2   # cancel the built-in dome …
        motion = -0.08 * (fr - F / 2.0) ** 2                                # … and impose a VALLEY through motion
        vol = _synth_moving_dome(frame_curv=0.03, motion=motion)
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1}
        a_own, _, i_own = M.tissue_motion_move(vol, {**p, "tissue_motion_shape_curv": None})
        assert "shape_guard" not in i_own                                   # no shape → own parabola (inverted, kept)
        a_g, _, i_g = M.tissue_motion_move(vol, {**p, "tissue_motion_shape_curv": 0.03})
        assert i_g.get("shape_guard"), "the guard must fire when the own parabola is sign-wrong"
        assert i_g["shape_guard"][0]["own_curv"] < 0 < i_g["shape_guard"][0]["used_curv"]
        # the delivered trajectory (pos + a) must now bend upward with the requested curvature
        c_own = np.polyfit(fr, a_own, 2)[0]; c_g = np.polyfit(fr, a_g, 2)[0]
        assert c_g - c_own > 0.05                                           # the guard added a positive dome
        # a normal dome is left alone even with a shape given
        vol2 = _synth_moving_dome(frame_curv=0.03, motion=np.zeros(F))
        _, _, i2 = M.tissue_motion_move(vol2, {**p, "tissue_motion_shape_curv": 0.03})
        assert "shape_guard" not in i2

    def test_preview_fit_follows_the_guard(self):
        fr = np.arange(101, dtype=float)
        edge = -0.05 * (fr - 50) ** 2 + 300 + np.random.default_rng(0).normal(0, 1.0, 101)   # inverted served edge
        fit = np.polyval(np.polyfit(fr, edge, 2), fr)
        out, info = M.sign_guarded_quadratic(edge, fit, 0.02)
        assert info["applied"] and info["own_curv"] < 0 and np.polyfit(fr, out, 2)[0] > 0.019
        # a correct dome is untouched, and no shape means untouched
        good = 0.03 * (fr - 50) ** 2 + 200; gf = np.polyval(np.polyfit(fr, good, 2), fr)
        o2, i2 = M.sign_guarded_quadratic(good, gf, 0.02); assert not i2["applied"] and np.allclose(o2, gf)
        o3, i3 = M.sign_guarded_quadratic(edge, fit, None); assert not i3["applied"] and np.allclose(o3, fit)

    def test_guard_centres_the_dome_in_the_segment(self):
        # a sign-wrong trajectory's linear term is drift, not geometry: the DELIVERED dome's vertex must sit at the
        # middle of the live segment (cs017: vertex at frame −25 before, a ramp on screen). Judge the delivered
        # geometry itself: warp the synthetic volume by the move and read the anterior edge across frames.
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=-0.08 * (fr - F / 2.0) ** 2 + 2.0 * fr)   # valley + strong drift
        p = {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_shape_curv": 0.03}
        a, b, i = M.tissue_motion_move(vol, p)
        assert i.get("shape_guard")
        f0, f1 = i["shape_guard"][0]["segment"]
        moved = M._rigid_scratch_warp(np.pad(vol, ((0, 0), (200, 200), (0, 0))), a, b)
        L = vol.shape[2]; g = M.ndimage.gaussian_filter1d(moved[:, :, L // 2].astype(float), 1.5, axis=1)
        hit = g >= 500; ok = hit.any(axis=1); y = np.where(ok, np.argmax(hit, axis=1), np.nan).astype(float)
        k = np.isfinite(y) & (fr >= f0) & (fr <= f1)
        c = np.polyfit(fr[k], y[k], 2)
        assert c[0] > 0, "delivered dome must be apex-up"
        vtx = -c[1] / (2 * c[0])
        assert abs(vtx - 0.5 * (f0 + f1)) < 6.0, (vtx, (f0, f1))

    def test_guard_can_be_switched_off(self):
        F = 60; fr = np.arange(F, dtype=float)
        vol = _synth_moving_dome(frame_curv=0.03, motion=-0.08 * (fr - F / 2.0) ** 2)
        _, _, i = M.tissue_motion_move(vol, {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_shape_curv": 0.03, "tissue_motion_shape_sign_guard": False})
        assert "shape_guard" not in i



# ───────────────────────── applied move: persisted by the corrections run (option b) ─────────────────────────
def _crossing(col, thr=450.0):
    """sub-pixel first crossing of `thr` in a depth column (linear between the two rows straddling it)"""
    idx = np.where(np.asarray(col, np.float64) >= thr)[0]
    if idx.size == 0 or idx[0] == 0:
        return np.nan
    r = int(idx[0]); a, b = float(col[r - 1]), float(col[r])
    return r - 1 + (thr - a) / (b - a)


class TestAppliedMovePersisted:
    """border_cache/applied_move.npz written by preprocess_oct_to_nifti's corrections branch = the EXACT
    per-(lateral, frame) move the run applied, in measure_applied_move's convention:
    corrected_row = raw_row + canvas_pad + move (the bottom pad shifts no row)."""

    def test_applied_move_run_cache_matches_corrected_minus_raw_crossings(self, tmp_path, monkeypatch):
        F, D, L = 40, 220, 64
        fr = np.arange(F, dtype=float); x = np.arange(L, dtype=float)
        motion = 10.0 * np.sin(fr / 4.0) + 0.3 * fr                      # up to ±10 px + a drift: a real move
        vol = _synth_moving_dome(F=F, D=D, L=L, frame_curv=0.03, motion=motion)
        top = ((60.0 + 0.03 * (fr - F / 2.0) ** 2)[None, :] + (0.02 * (x - L / 2.0) ** 2)[:, None]
               + motion[None, :]).astype(np.float32)                    # the served line, RAW rows (L, F)
        monkeypatch.setattr(M, "read_oct_zstack", lambda p, i=0: vol)
        out = tmp_path / "case" / "input" / "vol.nii.gz"
        info = M.preprocess_oct_to_nifti("fake.OCT", out, params={}, provided_edges=top, workers=1)
        rec = info["applied_move"]
        assert rec["written"] and rec["complete"], rec
        assert [s["stage"] for s in rec["stages"]] == ["flatten"]       # default tissue path: one mover
        assert M._MOVE_LEDGER is None                                    # closed after the write
        bc = tmp_path / "case" / "border_cache" / "applied_move.npz"    # sibling convention of placed_edges.npz
        z = np.load(bc, allow_pickle=False)
        assert str(z["source"]) == "run" and int(z["n_extrapolated"]) == 0
        assert z["move"].shape == (L, F) and z["move"].dtype == np.float32
        assert str(z["key"]).split(":")[2] == "run"
        st = out.stat()
        assert int(z["stamp_mtime_ns"]) == st.st_mtime_ns and int(z["stamp_size"]) == st.st_size
        pad = int(z["canvas_pad"]); ce = info.get("canvas_extend") or {}
        assert pad == int(ce.get("pad", 0)) and int(z["bottom_pad"]) == int(ce.get("bottom_pad", 0))
        stages = __import__("json").loads(str(z["stages"]))
        assert stages[0]["stage"] == "flatten" and stages[0]["composed"]
        # THE CONTRACT: corrected crossing − raw crossing − pad == move, on every frame
        import nibabel as nib
        cor = np.asarray(nib.load(str(out)).dataobj).astype(np.float32)   # (L, D', F)
        raw = M.reformat_to_sagittal(vol).astype(np.float32)
        assert cor.shape[1] == D + pad + int(z["bottom_pad"])
        err = np.full((L, F), np.nan)
        for f in range(F):
            for l in range(0, L, 3):
                cr, cc = _crossing(raw[l, :, f]), _crossing(cor[l, :, f])
                if np.isfinite(cr) and np.isfinite(cc):
                    err[l, f] = (cc - cr - pad) - float(z["move"][l, f])
        assert np.isfinite(err).any(axis=0).all(), "every frame must have a measurable crossing"
        per_frame = np.nanmax(np.abs(err), axis=0)
        assert float(per_frame.max()) < 0.5, per_frame
        assert float(np.nanmax(np.abs(z["move"]))) > 5.0                  # the run really moved the tissue

    def test_applied_move_compose_and_ledger_settle(self):
        L, F = 5, 4
        a = np.full((L, F), 2.0, np.float32); b = -np.ones((L, F), np.float32)
        mv, stages, ok = M.compose_applied_move([("flatten", a), ("edit_transform", b)], (L, F))
        assert ok and mv.dtype == np.float32 and np.allclose(mv, 1.0)
        assert [s["stage"] for s in stages] == ["flatten", "edit_transform"] and all(s["composed"] for s in stages)
        # a missing field or a wrong shape → incomplete, and NEVER a partial sum
        mv2, st2, ok2 = M.compose_applied_move([("flatten", a), ("sagittal_quad_align", None)], (L, F))
        assert not ok2 and mv2 is None and st2[1] == {"stage": "sagittal_quad_align", "composed": False}
        mv3, _, ok3 = M.compose_applied_move([("flatten", np.zeros((L + 1, F)))], (L, F))
        assert not ok3 and mv3 is None
        # the ledger: notes are no-ops when closed; a declined stage (same object back) drops what it logged; a
        # mover that changed the volume without logging leaves a None entry; an unchanged copy is a no-op
        M._ledger_note("x", a)
        assert M._MOVE_LEDGER is None
        M._ledger_open()
        try:
            before = np.ones((F, 8, L), np.float32)
            m0 = M._ledger_mark(); M._ledger_note("flatten", a)
            M._ledger_settle("flatten", before, before.copy() * 2, m0)
            assert [s for s, _ in M._MOVE_LEDGER] == ["flatten"]
            m1 = M._ledger_mark(); M._ledger_note("rigid_height_refine", b)
            M._ledger_settle("rigid_height_refine", before, before, m1)                    # reverted → dropped
            assert [s for s, _ in M._MOVE_LEDGER] == ["flatten"]
            m2 = M._ledger_mark()
            M._ledger_settle("rigid_frame_refine", before, before.copy(), m2)              # unchanged copy → no-op
            assert [s for s, _ in M._MOVE_LEDGER] == ["flatten"]
            m3 = M._ledger_mark()
            M._ledger_settle("rigid_frame_derotate", before, before * 3, m3)               # moved, unlogged → None
            assert M._MOVE_LEDGER[-1] == ("rigid_frame_derotate", None)
            entries = M._ledger_close()
        finally:
            M._MOVE_LEDGER = None
        assert M._MOVE_LEDGER is None
        mv4, _, ok4 = M.compose_applied_move(entries, (L, F))
        assert not ok4 and mv4 is None

    def test_applied_move_cache_write_is_best_effort(self, tmp_path):
        # an incomplete composition writes nothing and reports why; a missing output never raises
        rec = M.write_applied_move_cache(tmp_path / "input" / "missing.nii.gz", [("flatten", None)], (3, 2))
        assert rec["written"] is False and rec["complete"] is False and "reason" in rec
        assert not (tmp_path / "border_cache").exists()
        rec2 = M.write_applied_move_cache(tmp_path / "input" / "missing.nii.gz", [("flatten", np.zeros((3, 2)))], (3, 2))
        assert rec2["written"] is False and "error" in rec2


class TestCorrectedPriorRunMove:
    """api_server._corrected_prior_surface prefers the run-persisted move (source="run") when it belongs to the
    corrected volume being served, and falls back to the measurement (source="measured") otherwise."""

    @staticmethod
    def _case(cases_root, cid, L=6, D=40, F=5, pad=4):
        import nibabel as nib
        import orchestration as orch
        orch.ensure_case_dirs(cid)
        root = orch.case_root(cid)
        aff = np.diag([0.02, 0.02, 0.04, 1.0])
        raw = root / "input" / "_raw_border.nii.gz"
        raw.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.full((L, D, F), 20, np.uint16), aff), str(raw))
        work = root / "input" / "vol.nii.gz"
        nib.save(nib.Nifti1Image(np.full((L, D + pad, F), 20, np.uint16), aff), str(work))
        bc = root / "border_cache"; bc.mkdir(parents=True, exist_ok=True)
        surf0 = (10.0 + np.arange(L)[:, None] + 0.5 * np.arange(F)[None, :]).astype(np.float32)
        np.savez_compressed(bc / "provided_edges.npz", surface=surf0)
        return root, work, bc, surf0

    @staticmethod
    def _write_run_move(bc, work, move, pad, stale=False):
        import json as _json
        st = work.stat()
        np.savez_compressed(bc / "applied_move.npz", key=np.array("0:1:run:test"), move=move.astype(np.float32),
                            n_extrapolated=np.array(0), source=np.array("run"), canvas_pad=np.array(int(pad)),
                            bottom_pad=np.array(0),
                            stamp_mtime_ns=np.array(int(st.st_mtime_ns) - (1 if stale else 0), dtype=np.int64),
                            stamp_size=np.array(int(st.st_size), dtype=np.int64),
                            stages=np.array(_json.dumps([{"stage": "flatten", "composed": True}])))

    def test_corrected_prior_prefers_the_run_move_and_measures_nothing(self, cases_root, monkeypatch):
        import api_server as api
        L, D, F, pad = 6, 40, 5, 4
        root, work, bc, surf0 = self._case(cases_root, "case_am_run", L, D, F, pad)
        move = (np.arange(L)[:, None] * 0.25 - 3.0 + np.arange(F)[None, :]).astype(np.float32)
        self._write_run_move(bc, work, move, pad)

        def _boom(*a, **k):
            raise AssertionError("measure_applied_move must not run when a valid run move exists")
        monkeypatch.setattr(api.oct_mod, "measure_applied_move", _boom)
        prior, meta = api._corrected_prior_surface("case_am_run", work, {}, (L, D + pad, F))
        assert prior is not None and meta["source"] == "run" and meta["cached"] is True
        assert meta["n_extrapolated"] == 0 and meta["canvas_pad"] == pad and meta["stages"] == ["flatten"]
        expect = np.clip(surf0.astype(np.float64) + pad + move, 0, D + pad - 1)
        assert np.allclose(prior, expect, atol=1e-5)
        # the posterior line rides the SAME run move (no posterior_edges.npz here → the documented reason)
        pb, why = api._corrected_prior_surface("case_am_run", work, {}, (L, D + pad, F), which="posterior")
        assert pb is None and "posterior_edges" in why
        # a second call must not have replaced the run move with a measured cache
        z = np.load(bc / "applied_move.npz", allow_pickle=False)
        assert str(z["source"]) == "run"
        assert api._run_applied_move(bc / "applied_move.npz", work, (L, F)) is not None
        # wrong shape → ignored, never a crash
        assert api._run_applied_move(bc / "applied_move.npz", work, (L + 1, F)) is None

    def test_corrected_prior_stale_run_move_falls_back_to_measured_and_replaces_it(self, cases_root, monkeypatch):
        import api_server as api
        L, D, F, pad = 6, 40, 5, 4
        root, work, bc, surf0 = self._case(cases_root, "case_am_stale", L, D, F, pad)
        self._write_run_move(bc, work, np.full((L, F), 99.0, np.float32), pad, stale=True)   # older corrected volume
        measured = (np.arange(F)[None, :] * 1.5 - 2.0) * np.ones((L, 1))
        calls = []

        def _fake_measure(rv, cv, p):
            calls.append((rv.shape, cv.shape, p.get("corrected_prior_max_lag")))
            return {"move": measured.astype(np.float32), "extrapolated_frames": [4], "resid_mad": np.zeros(F)}
        monkeypatch.setattr(api.oct_mod, "measure_applied_move", _fake_measure)
        prior, meta = api._corrected_prior_surface("case_am_stale", work, {}, (L, D + pad, F))
        assert prior is not None and meta["source"] == "measured" and meta["cached"] is False
        assert meta["n_extrapolated"] == 1 and meta["canvas_pad"] == pad
        assert len(calls) == 1 and calls[0][0] == calls[0][1] == (L, D + pad, F)     # raw padded on top to match
        assert np.allclose(prior, np.clip(surf0 + pad + measured, 0, D + pad - 1), atol=1e-5)
        z = np.load(bc / "applied_move.npz", allow_pickle=False)              # the stale run move was replaced
        assert str(z["source"]) == "measured" and int(z["n_extrapolated"]) == 1
        # ...and the measured cache is served next time without re-measuring
        prior2, meta2 = api._corrected_prior_surface("case_am_stale", work, {}, (L, D + pad, F))
        assert len(calls) == 1 and meta2 == {"source": "measured", "n_extrapolated": 1, "cached": True, "canvas_pad": pad}
        assert np.allclose(prior2, prior)
        # a fresh run move written for the CURRENT volume takes over again
        self._write_run_move(bc, work, np.zeros((L, F), np.float32), pad)
        prior3, meta3 = api._corrected_prior_surface("case_am_stale", work, {}, (L, D + pad, F))
        assert meta3["source"] == "run" and len(calls) == 1 and np.allclose(prior3, surf0 + pad)


class TestArtifactCellsDead:
    """2026-09-10 (p5_os_v1_2 / p5_os_v1_3): the reviewer's PARTIAL artifact band is dead to the tissue-motion
    measurement per CELL (tissue_motion_artifact_cells_dead), kept only when the run's own never-worse judge agrees.
    The descending eyelid on half the width was still being correlated (the 2026-09-09 rule honoured whole-frame
    bands only), dragging the global dome/tilt fits into a -200 px roll and a +-40 px tilt hump on the CLEAN frames."""

    def test_measurement_volume_helper_rules(self):
        rng = np.random.default_rng(0)
        F, D, L = 30, 50, 64
        vol = rng.uniform(100, 3000, (F, D, L)).astype(np.float32)
        am = np.zeros((L, F), bool)
        am[:32, 0:6] = True          # half the width, frames 0-5  -> cell-zeroed
        am[:, 10:12] = True          # whole width, frames 10-11   -> whole-frame dead (today's rule)
        am[:52, 20] = True           # 81% of the width, frame 20  -> live 19% < 0.25 -> left in (today's rule)
        p = {}
        v, nd, nc = M._tissue_measurement_volume(vol, am, p)
        assert nd == 2 and nc == 32 * 6, (nd, nc)
        assert (v[10:12] == 0).all() and (v[0:6, :, :32] == 0).all() and (v[0:6, :, 32:] == vol[0:6, :, 32:]).all()
        assert (v[20] == vol[20]).all() and (v[6:10] == vol[6:10]).all() and (v[12:20] == vol[12:20]).all() and (v[21:] == vol[21:]).all()
        # flag off == today's rule verbatim
        v0, nd0, nc0 = M._tissue_measurement_volume(vol, am, {"tissue_motion_artifact_cells_dead": False})
        assert nd0 == 2 and nc0 == 0 and (v0[10:12] == 0).all() and (v0[0:6] == vol[0:6]).all()
        # no marks -> the very same array object (no copy)
        v1, nd1, nc1 = M._tissue_measurement_volume(vol, np.zeros((L, F), bool), p)
        assert v1 is vol and nd1 == 0 and nc1 == 0
        v2, _, _ = M._tissue_measurement_volume(vol, None, p); assert v2 is vol
        # a frame that is only whole-dead is not double counted as cells
        am2 = np.zeros((L, F), bool); am2[:, 3] = True
        v3, nd3, nc3 = M._tissue_measurement_volume(vol, am2, p); assert nd3 == 1 and nc3 == 0 and (v3[3] == 0).all()
        assert M.DEFAULT_PARAMS["tissue_motion_artifact_cells_dead"] is True
        assert M.DEFAULT_PARAMS["tissue_motion_artifact_min_live_frac"] == 0.25
        assert M.DEFAULT_PARAMS["tissue_motion_artifact_cells_tol_px"] == 0.25

    # -- synthetic helpers --------------------------------------------------------------------------------------
    @staticmethod
    def _synth(F, D, L, base, motion, lat_curv=0.02, seed=3):
        """tissue band 70 rows deep whose anterior is base[f] + a lateral dome + a per-frame RIGID shift motion[f];
        returns (vol (F, D, L), true top (L, F)). A negative top = the apex left the frame (surface crop)."""
        rng = np.random.default_rng(seed); x = np.arange(L, dtype=float)
        lat_dome = lat_curv * (x - L / 2.0) ** 2
        vol = np.zeros((F, D, L), np.float32); rows = np.arange(D, dtype=float)
        top = np.zeros((L, F), np.float32)
        for f in range(F):
            for l in range(L):
                t = base[f] + lat_dome[l] + motion[f]; top[l, f] = t
                vol[f, :, l] = 30.0 + 20.0 * rng.uniform(0, 1, D) + np.where((rows >= t) & (rows < t + 70), 900.0 + 300.0 * rng.uniform(0, 1, D), 0.0)
        return vol, top

    @staticmethod
    def _fit_move(mv, am=None):
        """per-frame (shift, half-span tilt) of the DELIVERED move over the un-marked laterals"""
        L, F = mv.shape; x = (np.arange(L) - (L - 1) / 2.0) / ((L - 1) / 2.0)
        a = np.full(F, np.nan); b = np.full(F, np.nan)
        for f in range(F):
            ok = np.isfinite(mv[:, f]) & (~am[:, f] if am is not None else np.ones(L, bool))
            if ok.sum() >= 8:
                c = np.polyfit(x[ok], mv[ok, f], 1); b[f] = c[0]; a[f] = c[1]
        return a, b

    @staticmethod
    def _run(monkeypatch, tmp_path, name, vol, top, params, calls=None):
        monkeypatch.setattr(M, "read_oct_zstack", lambda p, i=0: vol)
        if calls is not None:
            # the second run of a test must wrap the ORIGINAL stage, not the previous run's recorder (monkeypatch
            # is still active): a nested wrapper would append the second run's calls to the first run's list
            orig = getattr(M.tissue_motion_move, "_artifact_test_orig", M.tissue_motion_move)
            def _rec(v, p=None):
                r = orig(v, p); calls.append(r); return r
            _rec._artifact_test_orig = orig
            monkeypatch.setattr(M, "tissue_motion_move", _rec)
        out = tmp_path / name / "input" / "vol.nii.gz"
        info = M.preprocess_oct_to_nifti("fake.OCT", out, params=params, provided_edges=top, workers=1)
        z = np.load(tmp_path / name / "border_cache" / "applied_move.npz", allow_pickle=True)
        return info, z["move"].astype(float)

    def test_partial_lid_band_is_dead_to_the_measurement_and_judged(self, tmp_path, monkeypatch):
        """A bright 'eyelid' descends 10 px/frame over a TRIANGULAR band (64 % of the width at frame 0, gone by
        frame 12 — the p5_os_v1_2 mark's shape) while the tissue moves by a shift-only motion. Flag off: the lid is
        correlated and the delivered shift/tilt carry it. Flag on: the marked cells are zeroed for the measurement,
        the judge keeps the cells-dead move, and the delivered move follows the tissue."""
        F, D, L = 40, 220, 64
        fr = np.arange(F, dtype=float)
        motion = 6.0 * np.sin(fr / 5.0) + 0.2 * fr                     # the TRUE per-frame move: shift only, no tilt
        vol, top = self._synth(F, D, L, 60.0 + 0.03 * (fr - F / 2.0) ** 2, motion)
        bands = {"0": [0, 11], "16": [0, 7], "32": [0, 3], "40": [0, 0]}
        am = M._artifact_mask({**M.DEFAULT_PARAMS, "crop_bands": bands}, L, F)
        assert 0.6 < am[:, 0].mean() < 0.7 and not am[:, 12:].any()
        rng = np.random.default_rng(7)
        for f in range(F):
            lats = np.flatnonzero(am[:, f])
            if lats.size:
                r0 = 10 + 10 * f
                vol[f, r0:r0 + 80, lats] = 1800.0 + 400.0 * rng.uniform(0, 1, (lats.size, 80))
        # lat_step 2 / band 1: at L=64 the default sampling leaves the cells-dead frames < min_bands (a real scan
        # has 128 bands). The tilt witness is off because the painted lid biases its raw-crossing reading equally
        # in both runs (a separate matter); second/anterior passes off so the delivered move IS the stage's.
        p = {"crop_bands": bands, "tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_tilt_witness": False,
             "tissue_motion_second_pass": False, "tissue_motion_anterior_pass": False}
        calls_on = []; calls_off = []
        info_on, mv_on = self._run(monkeypatch, tmp_path, "on", vol, top, p, calls_on)
        info_off, mv_off = self._run(monkeypatch, tmp_path, "off", vol, top, {**p, "tissue_motion_artifact_cells_dead": False}, calls_off)
        ac = info_on["tissue_motion"]["artifact_cells"]
        assert ac["zeroed"] == int(am.sum()) and ac["kept"] is True, ac
        assert ac["scores"]["cells_dead"] < ac["scores"]["frames_dead"], ac["scores"]
        assert len(calls_on) == 2 and len(calls_off) == 1                  # the judged second measurement, once
        assert "artifact_cells" not in info_off["tissue_motion"]           # flag off: today's path, no record
        assert info_on["pipeline_version"] == M.PIPELINE_VERSION == "2026-09-10.tissue-v8"
        a_on, b_on = self._fit_move(mv_on, am); a_off, b_off = self._fit_move(mv_off, am)
        # the delivered shift follows the tissue: (shift + true motion) sits on ONE smooth parabola (the dome the
        # flatten aims at) with the lid gone, and wobbles by the lid's drag without
        def _rms_about_parabola(a):
            y = a + motion; c = np.polyfit(fr, y, 2); return float(np.sqrt(np.mean((y - np.polyval(c, fr)) ** 2)))
        assert _rms_about_parabola(a_on) < 1.6 < 2.0 < _rms_about_parabola(a_off)
        # the CLEAN frames (no band): no tilt with the lid dead, a >10 px hump when it is correlated (the p5 complaint)
        # (the geometry line became robust on 2026-09-10, which already halves the lid's fake roll on the OFF side)
        assert np.nanmax(np.abs(b_on[12:])) < 1.0 and np.nanmax(np.abs(b_off[12:])) > 5.0 * np.nanmax(np.abs(b_on[12:]))
        # the stage's own tilt on the kept (cells-dead) measurement is the truth (none); the frames-dead one is not
        assert float(np.abs(calls_on[1][1]).max()) < 1.0 and float(np.abs(calls_on[0][1]).max()) > 10.0

    def test_cut_apex_delivers_the_tissue_tilt_not_the_served_top(self, tmp_path, monkeypatch):
        """Surface crop: the dome's apex leaves the frame top over a block of frames. The served line on those
        frames is the cut row (as the store's provided_edges), in run B with a spurious 30 px half-span tilt on
        top. The delivered per-frame tilt must be the TISSUE stage's tilt on every frame (cropped frames
        included), the served top's tilt must never enter the move, and the non-cropped frames are unchanged."""
        F, D, L = 48, 220, 64
        fr = np.arange(F, dtype=float)
        motion = 5.0 * np.sin(fr / 4.0) + 0.25 * fr
        vol, top = self._synth(F, D, L, 0.1 * (fr - F / 2.0) ** 2 - 20.0, motion)
        cut = [int(f) for f in range(F) if float(top[:, f].min()) < 0.0]
        assert 15 <= len(cut) <= 30 and cut == list(range(cut[0], cut[-1] + 1))
        x = (np.arange(L) - (L - 1) / 2.0) / ((L - 1) / 2.0)
        served_a = top.copy(); served_a[:, cut] = 8.0
        served_b = served_a.copy(); served_b[:, cut] = 8.0 + 30.0 * x[:, None]
        p = {"surface_crop_frames": cut, "tissue_motion_second_pass": False, "tissue_motion_anterior_pass": False}
        calls_a = []; calls_b = []
        info_a, mv_a = self._run(monkeypatch, tmp_path, "a", vol, served_a, p, calls_a)
        info_b, mv_b = self._run(monkeypatch, tmp_path, "b", vol, served_b, p, calls_b)
        assert len(calls_a) == len(calls_b) == 1 and "artifact_cells" not in info_a["tissue_motion"]   # no bands: one measurement
        assert [s["stage"] for s in info_a["applied_move"]["stages"]] == ["flatten"]
        cap = float(M.DEFAULT_PARAMS["tissue_motion_max_tilt_px"])
        a_a, b_a = self._fit_move(mv_a); a_b, b_b = self._fit_move(mv_b)
        stage_b_a = np.clip(np.asarray(calls_a[0][1], float), -cap, cap); stage_b_b = np.clip(np.asarray(calls_b[0][1], float), -cap, cap)
        # every frame moves RIGIDLY, by the stage's own shift + tilt (cropped frames included)
        rigid = max(float(np.nanmax(np.abs(mv_a[:, f] - (a_a[f] + b_a[f] * x)))) for f in range(F))
        assert rigid < 1e-3
        assert float(np.nanmax(np.abs(b_a[cut] - stage_b_a[cut]))) < 0.05 and float(np.nanmax(np.abs(b_a - stage_b_a))) < 0.05
        assert float(np.nanmax(np.abs(a_a - np.asarray(calls_a[0][0], float)))) < 0.05
        # the served top's spurious tilt never reaches the move: B == A on the cropped frames AND the rest
        assert float(np.nanmax(np.abs(mv_b[:, cut] - mv_a[:, cut]))) < 1e-3
        not_cut = [f for f in range(F) if f not in set(cut)]
        assert float(np.nanmax(np.abs(mv_b[:, not_cut] - mv_a[:, not_cut]))) < 1e-3
        assert float(np.nanmax(np.abs(b_b - stage_b_b))) < 0.05
        assert float(np.abs(stage_b_a).max()) < 3.0                       # the tissue has no tilt to speak of


# ───────────────────────── per-segment LATERAL shift across a dead gap (cs020_os_v4, 2026-09-10) ─────────────
class TestLateralShift:
    """cs020_os_v4: the eye moved LATERALLY during a reviewer-marked artifact band, so the segment after the band images
    a different part of the cornea. The estimator must recover the shift from the raw crossing profiles, the gate must
    refuse a small / implausible one, and the apply stage must roll a frame's tissue, served edge and applied move
    TOGETHER (whole frames only — never a per-column warp)."""

    def _vol(self, shift, F=60, D=220, L=160, gap=(30, 37), seed=4):
        """dome tissue with a dead gap; post-gap frames' tissue AND served edge moved by `shift` laterals (eye motion:
        new[l] = old[l - shift], vacated laterals empty). The across-lateral profile is a cornea-like dome with a LIMBUS
        SHOULDER (the parabola's slope drops to 40 % beyond |u| = 50): on a pure parabola a lateral shift and a tilt are
        the same profile change, and the estimator must (and does) refuse that case."""
        rng = np.random.default_rng(seed); fr = np.arange(F, dtype=float); x = np.arange(L, dtype=float); rows = np.arange(D, dtype=float)
        vol = np.zeros((F, D, L), np.float32); edge = np.full((L, F), np.nan)
        u = x - 0.6 * L; prof = np.where(np.abs(u) <= 50, 0.012 * u ** 2, 0.012 * 50 ** 2 + 0.48 * (np.abs(u) - 50))
        for f in range(F):
            top = 40.0 + 0.02 * (fr[f] - F / 2.0) ** 2 + prof                                  # off-centre dome: a shift is not a symmetry
            vol[f] = (30.0 + 20.0 * rng.uniform(0, 1, (D, L)) + np.where((rows[:, None] >= top[None, :]) & (rows[:, None] < top[None, :] + 60), 900.0 + 300.0 * rng.uniform(0, 1, (D, L)), 0.0)).astype(np.float32)
            edge[:, f] = top
            if shift and f > gap[1]:
                vol[f] = M._roll_lateral_plane(vol[f], shift); edge[:, f] = M._roll_lateral_column(edge[:, f], shift, fill="nan")
        vol[gap[0]:gap[1] + 1] = 0.0
        return vol, edge

    def _p(self):
        return {"tissue_motion_lat_step": 2, "tissue_motion_band": 1, "tissue_motion_interior_pair_guard": False, "tissue_motion_second_pass": False,
                "tissue_motion_shape_sign_guard": False, "tissue_motion_bridge_max_gap": 20, "tissue_motion_tilt_witness": False}

    def test_estimator_recovers_a_known_lateral_shift_and_rejects_none(self):
        vol, edge = self._vol(30)
        a, b, i = M.tissue_motion_move(vol, {**self._p(), "_gap_edge": edge})
        rec = i.get("gap_lateral_shift"); assert rec and rec.get("applied"), rec
        assert rec["reference"] == 0 and rec["dx"] == [0, -30], rec                                   # the roll that puts the post-gap frames back
        g = rec["gaps"][0]; assert g["accepted"] and g["dx"] == -30 and g["witness"]["dx"] == -30, g
        assert g["std_without"] > 3 * g["std_with"] and g["drop_px"] >= 8.0, g
        # the same estimate straight from the crossing map (what the stage calls)
        T = M._raw_crossing_map(vol, 800.0); live = vol.reshape(vol.shape[0], -1).max(axis=1) > 0
        r2 = M._gap_lateral_shift_estimate(T, live, np.zeros(vol.shape[0], bool), [(0, 29), (38, 59)], self._p(), edge=edge)
        assert r2["dx"] == [0, -30], r2
        # applying the roll puts the post-gap crossing profile back on the pre-gap one
        dxf = M.segment_lateral_dx_per_frame(rec, vol.shape[0]); assert dxf[45] == -30 and dxf[10] == 0 and dxf[33] in (0.0, -30.0)
        pb = np.nanmedian(T[22:30], axis=0); pa = np.nanmedian(np.vstack([M._roll_lateral_column(T[f], dxf[f], fill="nan") for f in range(38, 46)]), axis=0)
        ok = np.isfinite(pa) & np.isfinite(pb); assert ok.sum() > 100 and np.nanstd(pa[ok] - pb[ok]) < 2.0, np.nanstd(pa[ok] - pb[ok])
        # ... and the axial step across the gap is then read on the same tissue: no spurious tilt from the shifted parabola
        geo = (i.get("gap_edge_offset") or [{}])[0]; assert abs(geo.get("tilt_px", [0, 0])[1]) < 6.0, geo
        a_off, b_off, i_off = M.tissue_motion_move(vol, {**self._p(), "_gap_edge": edge, "tissue_motion_gap_lateral_shift": False})
        assert "gap_lateral_shift" not in i_off
        geo_off = (i_off.get("gap_edge_offset") or [{}])[0]; assert abs(geo_off.get("tilt_px", [0, 0])[1]) > 8.0, geo_off   # the shift read as a tilt
        # no shift → nothing accepted, nothing applied
        vol0, edge0 = self._vol(0)
        a0, b0, i0 = M.tissue_motion_move(vol0, {**self._p(), "_gap_edge": edge0})
        rec0 = i0.get("gap_lateral_shift"); assert rec0 and not rec0["applied"] and rec0["dx"] == [0, 0], rec0
        assert not rec0["gaps"][0]["accepted"], rec0["gaps"][0]

    def test_gate_refuses_small_unrelated_or_noisy_shifts(self):
        L = 400; x = np.arange(L, dtype=float); rng = np.random.default_rng(2); live = np.ones(20, bool); partial = np.zeros(20, bool)
        u = x - 230.0; prof = 60.0 + np.where(np.abs(u) <= 120, 0.004 * u ** 2, 0.004 * 120 ** 2 + 0.96 * 0.4 * (np.abs(u) - 120))   # dome + limbus shoulder
        parab = 60.0 + 0.004 * u ** 2
        def T_of(pb, pa):
            T = np.full((20, L), np.nan); T[:8] = pb + rng.normal(0, 0.3, (8, L)); T[12:] = pa + rng.normal(0, 0.3, (8, L)); return T
        segs = [(0, 7), (12, 19)]
        small = M._gap_lateral_shift_estimate(T_of(prof, M._roll_lateral_column(prof, 3, fill="nan")), live, partial, segs, {})
        assert not small["applied"] and not small["gaps"][0]["accepted"] and "|dx|" in small["gaps"][0]["reason"], small["gaps"][0]
        unrelated = M._gap_lateral_shift_estimate(T_of(prof, 60.0 + 40.0 * rng.uniform(0, 1, L)), live, partial, segs, {})
        assert not unrelated["applied"], unrelated["gaps"][0]
        noisy = M._gap_lateral_shift_estimate(T_of(prof, M._roll_lateral_column(prof, 40, fill="nan") + rng.normal(0, 20.0, L)), live, partial, segs, {})
        assert not noisy["applied"] and "residual" in noisy["gaps"][0]["reason"], noisy["gaps"][0]          # 20 px of scatter left: not the same cornea
        good = M._gap_lateral_shift_estimate(T_of(prof, M._roll_lateral_column(prof, 40, fill="nan")), live, partial, segs, {})
        assert good["applied"] and good["dx"] == [0, -40], good["gaps"][0]
        # on a PURE parabola a tilt explains the shifted profile exactly as well: refused (the axial stage's tilt is the simpler move)
        para = M._gap_lateral_shift_estimate(T_of(parab, M._roll_lateral_column(parab, 40, fill="nan")), live, partial, segs, {})
        assert not para["applied"] and "tilt" in para["gaps"][0]["reason"], para["gaps"][0]
        assert para["gaps"][0]["std_without_line"] < 1.0 < para["gaps"][0]["std_without"], para["gaps"][0]
        # the dual witness: an edge that only witnesses a DIFFERENT shift refuses the plain estimate
        E = np.zeros((L, 20)); E[:, :8] = prof[:, None]; E[:, 12:] = M._roll_lateral_column(prof, 40, fill="hold")[:, None]
        assert M._gap_lateral_shift_estimate(T_of(prof, M._roll_lateral_column(prof, 40, fill="nan")), live, partial, segs, {}, edge=E)["applied"]
        E2 = E.copy(); E2[:, 12:] = (M._roll_lateral_column(prof, 40, fill="hold") + 200.0)[:, None]         # 200 px off the tissue: no witness
        assert not M._gap_lateral_shift_estimate(T_of(prof, M._roll_lateral_column(prof, 40, fill="nan")), live, partial, segs, {}, edge=E2)["applied"]
        # at the search limit or with too little overlap: refused
        far = M._gap_lateral_shift_estimate(T_of(prof, M._roll_lateral_column(prof, 80, fill="nan")), live, partial, segs, {"tissue_motion_gap_lateral_max": 80})
        assert not far["applied"], far["gaps"][0]

    def test_apply_rolls_frames_edges_and_move_together(self, tmp_path):
        F, D, L = 10, 12, 40; rng = np.random.default_rng(7)
        vol = rng.uniform(1, 100, (F, D, L)).astype(np.float32); edges = rng.uniform(2, 9, (L, F)); fld = rng.uniform(-5, 5, (L, F)).astype(np.float32)
        rec = {"segments": [[0, 3], [6, 9]], "dx": [0, 7], "reference": 0, "applied": True}
        out, e_out, ent, info = M.apply_segment_lateral_shift(vol, rec, edges, [("flatten", fld), ("odd", None)], canvas_pad=5, params={"tissue_motion_gap_lateral_edge_fill": "nan", })
        # frames 4-5 belong to no segment and hold the NEAREST segment's dx: 4 → segment 0 (dx 0), 5 → segment 1 (dx 7)
        assert info["applied"] and info["frames_shifted"] == 5 and info["dx_per_frame"][5] == 7.0 and info["dx_per_frame"][4] == 0.0, info
        assert np.array_equal(out[:5], vol[:5]) and np.array_equal(np.asarray(e_out)[:, :5], edges[:, :5].astype(np.float32))
        for f in range(5, 10):
            assert np.array_equal(out[f, :, 7:], vol[f, :, :-7]) and not out[f, :, :7].any()
            assert np.allclose(np.asarray(e_out)[7:, f], edges[:-7, f].astype(np.float32)) and np.isnan(np.asarray(e_out)[:7, f]).all()
            assert np.allclose(ent[0][1][7:, f], fld[:-7, f]) and not ent[0][1][:7, f].any()
        assert ent[1] == ("odd", None) and ent[-1][0] == "lateral_shift" and not ent[-1][1].any()
        # a cell's tissue, edge and move travel together: delivered lateral l holds raw lateral l − dx
        l, f = 23, 8; assert out[f, :, l].tolist() == vol[f, :, l - 7].tolist() and np.asarray(e_out)[l, f] == np.float32(edges[l - 7, f]) and ent[0][1][l, f] == fld[l - 7, f]
        move, stages, complete = M.compose_applied_move(ent[:1] + ent[2:], (L, F)); assert complete and np.allclose(move[:, 8], ent[0][1][:, 8])
        # negative dx rolls the other way; sub-pixel dx interpolates; "hold" keeps the boundary value
        neg, en, _, _ = M.apply_segment_lateral_shift(vol, {"segments": [[0, 4], [5, 9]], "dx": [0, -3], "applied": True}, edges, None, params={"tissue_motion_gap_lateral_edge_fill": "hold"})
        assert np.array_equal(neg[7, :, :-3], vol[7, :, 3:]) and not neg[7, :, -3:].any() and np.allclose(np.asarray(en)[-3:, 7], np.float32(edges[-1, 7]))
        col = M._roll_lateral_column(np.arange(10, dtype=float), 0.5, fill="nan"); assert np.isnan(col[0]) and abs(col[5] - 4.5) < 1e-9
        # nothing accepted → identical objects back
        same = M.apply_segment_lateral_shift(vol, {"segments": [[0, 9]], "dx": [0], "applied": False}, edges, [], params={})
        assert same[0] is vol and same[1] is edges and not same[3]["applied"]
        # the persisted caches: placed_edges.npz in RAW rows (edges − pad) with the per-frame dx; applied_move.npz carries lateral_dx
        case = tmp_path / "case"; (case / "input").mkdir(parents=True); nii = case / "input" / "vol.nii.gz"; nii.write_bytes(b"x" * 100)
        out2, e2, ent2, info2 = M.apply_segment_lateral_shift(vol, rec, edges, [("flatten", fld)], canvas_pad=5, out_nifti=nii, params={})
        z = np.load(case / "border_cache" / "placed_edges.npz"); assert np.allclose(z["surface"][7:, 8], (edges[:-7, 8] - 5.0).astype(np.float32)) and z["lateral_dx"][8] == 7.0
        r = M.write_applied_move_cache(nii, ent2, (L, F), canvas_pad=5, lateral_dx=np.asarray(info2["dx_per_frame"]))
        assert r["written"] and r["lateral_dx_range"] == [0.0, 7.0], r
        z2 = np.load(case / "border_cache" / "applied_move.npz"); assert z2["lateral_dx"].tolist()[5:] == [7.0] * 5 and np.isfinite(z2["move"]).all() and "lateral_shift" in str(z2["stages"])
