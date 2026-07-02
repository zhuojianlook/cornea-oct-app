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
