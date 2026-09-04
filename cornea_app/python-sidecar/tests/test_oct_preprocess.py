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
