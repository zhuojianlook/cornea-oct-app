"""Headless port of the OCT_Extraction preprocessing pipeline (from the user's
Streamlit scripts) used to produce the corrected volumes the cornea app consumes:

  1. read .OCT  (oct_converter POCT)            → raw B-scan z-stack
  2. oct_to_dicom (DICOMGeneratorlossless.py)   → uint16 multi-frame DICOM + geometry
  3. smooth_volume (DICOMSmootherSteps.py)      → corneal-edge + column correction,
                                                  3D active correction across slices

Streamlit/matplotlib UI and all visualization were dropped; only the numeric pipeline
remains, with the smoother parameters exposed via a params dict. Faithful to the
originals except: (a) the read contract is fixed to `read_oct_volume()[0].volume`
(the installed oct_converter returns volume objects, so step 2's `np.stack(frames)`
was a version bug); (b) the per-slice 3D-active correction is computed in O(N) by
caching each slice's edge once instead of reprocessing neighbours; (c) the previously
unused `corr_factor` now scales the column displacement (default 1.0 = unchanged).
"""
from __future__ import annotations

import math
import os
import re
from pathlib import Path

import numpy as np
import cv2
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Defaults mirror DICOMSmootherSteps.py's sidebar defaults + the lossless converter.
DEFAULT_PARAMS: dict = {
    "sigma": 2.0,                 # gaussian sigma for the column gradient
    "max_jump": 10.0,             # outlier clamp between adjacent columns
    "median_filter_size": 5,      # boundary median filter
    "d": 9,                       # bilateral filter diameter
    "sigmaColor": 75,
    "sigmaSpace": 75,
    "side_window": 10,            # intelligent side-correction window
    "side_threshold_factor": 2.0,
    "residual_threshold": 5.0,    # RANSAC quadratic residual
    "active_threshold": 5.0,      # 3D active correction across neighbouring slices
    "corr_factor": 1.0,           # scales the column-correction displacement (0..1)
    # ── over-correction guard (#2) ── A low-signal lateral column gets a garbage edge whose deviation
    # from the dome's quadratic is huge, so (quad-edge) demands a 100-360px shift that bends the edge
    # and (re-detected on the warped output) compounds every pass. Any per-column displacement beyond
    # max_displacement (px) is therefore NOT trusted: that column is treated as bad and its shift is
    # interpolated from its good neighbours, then hard-clamped. Real corrections are a few px (a raw
    # boundary deviates < ~17px from its fit), so this is a no-op on well-detected columns — clean scans
    # are unchanged; only the pathological lateral runaway is tamed.
    "max_displacement": 40.0,
    # Peripheral warp-spike fix ("logical limbus correction"): PER-SCAN OPT-IN (default 0 = OFF → global pipeline
    # byte-unchanged). Set >0 (e.g. 0.18) on a scan showing a limbus warp SPIKE/STREAK: the outer
    # refine_freeze_frac of lateral slices is warped to a LATERALLY-SMOOTH surface (a smooth continuation of the
    # reliable central dome) instead of the noisy per-slice detection, in ALL passes — removing spikes AND
    # single-column streaks without freezing/tearing. Off by default because a BLANKET application mildly regresses
    # ~1/3 of clean scans; applied per-scan it corrects spike scans cleanly. provided-edges (fix-columns) exempt.
    "refine_freeze_frac": 0.18,
    # FIX limbus (a): lateral sigma multiplier for the peripheral blend (see smooth_volume 3c). 1.0 = original.
    "refine_edge_sigma_mult": 2.0,
    # ── ping-pong axial refine (#2) ── After the sagittal correction, run the SAME correction in the
    # axial domain (flatten along lateral, per frame) and keep it per-frame where it makes the en-face
    # boundary smoother — cleans the 'hairy' axial boundary the sagittal pass leaves at noisy slice ends.
    # Confirmed on real scans to give the smoothest 3-D surface; a global guard makes it never worse.
    "axial_refine": True,
    # ── axial consistency (#3) ── Sagittal slices are corrected independently, so neighbouring slices
    # can shift inconsistently → the en-face/axial corneal boundary turns jagged ("hairier"). Smoothing
    # the per-column displacement FIELD across the slice (lateral) axis with this Gaussian sigma (px,
    # 0 = off) makes neighbours shift consistently → a smoother axial boundary, while the per-slice
    # quadratic still carries the real lateral curvature. Small sigma stays close to the per-slice fit.
    "axial_motion_correct": False,  # RETIRED default OFF (tested, not shipped): a per-frame RIGID axial de-motion is
                                  #   REDUNDANT with the flatten — the per-slice flatten ALREADY removes the dominant
                                  #   per-frame rigid inter-frame motion (CS004 OD rep1 raw 5.56px rigid → output 1.08px),
                                  #   so applying it up-front smooths the RAW but not the OUTPUT, and it injects B-scan
                                  #   tears. rep1's RESIDUAL non-smoothness is NON-RIGID INTRA-frame distortion (eye moving
                                  #   DURING a B-scan at saccades) which a rigid shift can't fix. Code kept for reference.
    "amc_dome_deg": 5,            # degree of the robust 2-D dome fit used as the motion-free reference
    "amc_smooth": 1.0,            # gaussian sigma (frames) on the estimated motion → kills 1-frame detection noise
    "amc_max_shift": 30.0,        # px cap on the per-frame rigid depth shift
    "amc_min_motion": 1.0,        # px: if the per-frame motion std is below this, the scan is motion-free → no-op
    "intra_frame_dewarp": False,  # RETIRED default OFF (tested, not shipped): correcting the raw B-scans before the
                                  #   flatten gets re-processed/washed by the flatten, and correcting post-flatten hits
                                  #   detector re-lock revert; the residual high-freq frame-direction motion steps resist
                                  #   both and the warp adds B-scan tears. A proper fix needs joint motion-estimation +
                                  #   volume re-slicing (major re-architecture) — not worth it when clean replicates exist.
                                  #   (The intra_frame_dewarp function re-warps only saccade-distorted B-scans onto the
                                  #   smooth dome using the RAW band edge, gated per-frame; kept for reference, default off.)
    "ifd_frame_med": 7,           # frame-window median for the motion-free per-lateral reference (rejects saccade frames)
    "ifd_frame_gauss": 2.0,       # gaussian (frames) on the reference
    "ifd_frame_thresh": 2.0,      # px: per-frame lateral-distortion level above which a frame is treated as saccade-warped
    "ifd_frame_soft": 1.0,        # px ramp for the per-frame gate
    "ifd_lat_smooth": 8.0,        # gaussian (lateral) on the correction shift → coherent B-scan re-warp (no en-face jag)
    "ifd_max_shift": 20.0,        # px cap on the per-column intra-frame correction
    "interslice_smooth": 3.0,     # (raised 1→3 with subpixel_warp) smooth the per-slice displacement across slices
                                  #   more, reducing slice-to-slice apex ripple; validated to also SMOOTH approved scans
    # ── fix-columns provided_edges LATERAL de-streak (see smooth_volume use_provided branch) ── The provided_edges
    # warp disables all lateral smoothing to honour the exact drag, but that also lets UN-dragged peripheral
    # re-detection jitter through → the warp shears the clean band into vertical spikes ("fuzzy" axial border).
    # Replace only single-lateral spikes (> gate off the robust lateral trend) with the trend, protecting a small
    # band around every drag point + pinning the exact drags. Off = median<=1 or gate<=0.
    "provided_edge_lat_median": 7,     # lateral median window for the robust trend; <=1 = de-streak OFF (legacy exact)
    "provided_edge_lat_smooth": 2.0,   # gaussian sigma (lateral) smoothing that trend; 0 = median-only trend
    "provided_edge_lat_gate": 2.0,     # px: replace a column with the trend only where it deviates by more than this
    "provided_edge_protect_lat": 2,    # ± laterals around each drag point kept untouched (correction guard)
    "provided_edge_protect_frame": 1,  # ± frames around each drag point kept untouched
    "subpixel_warp": True,        # flatten warp shifts columns by the FRACTIONAL displacement (linear interp in
                                  #   depth) instead of int-truncate → removes the 1-px lateral STAIRCASE ripple in
                                  #   the anterior boundary. Interp is confined to the <1px depth shift (lateral/
                                  #   frame crispness untouched). False = legacy int-truncate warp.
    # ── apex de-tear (#apex) ── Within ONE sagittal slice, the warp flattens the anterior to its smooth
    # RANSAC quadratic via disp = (quad - edge), so ANY high-frequency jitter/STEP in the detected `edge`
    # (frame axis) is injected straight into the warp. At the rough bright-speckle APEX the DP detector
    # locks a few px shallower on one flank and deeper on the other, leaving a ~5-6px STEP in `edge` at
    # the apex; the smooth quad then makes disp jump between adjacent frames, and the column-warp TEARS the
    # tissue into a V-notch there. A light gaussian along the FRAME axis of the displacement field removes
    # that injected 1-3-frame step (the true anterior surface is smooth, and a well-detected column already
    # sits on its own quad so disp≈0 there → this is a no-op on smooth frames) while the parabolic bulk
    # warp — which varies slowly over frames — is preserved. Mirrors interslice_smooth but along frames.
    # 0 = off (byte-identical). Small sigma stays faithful; clip/cut clamps are re-asserted after.
    "apex_frame_smooth": 1.5,
    "apex_smooth_gate": 3.0,   # px: apex frame-smoothing applies ONLY where the displacement deviates from its
                               # 5-frame median by more than this (a detector-jump TEAR); a smooth apex is a no-op
    # ── windowed re-detection (fix-columns "Confirm", tilt-aware surface prior) ── When a PRIOR surface
    # (per-frame expected depth) is supplied to detection, the gradient argmax is restricted to a small
    # window ±detect_window (depth voxels) around it per column, so a spurious peak (e.g. a reflection
    # ABOVE the cornea) outside the window can't be picked. The prior is built by MARCHING outward from
    # the user's anchored slice(s) (redetect_surface): each slice's prior is its already-resolved
    # neighbour's surface, so the window tracks the tilted cornea. detect_seed_window is the (generous)
    # window used on an anchored seed slice where the prior is only the interpolated anchors. These are
    # used ONLY by redetect_surface; the normal auto pipeline passes no prior (prior=None → original
    # unrestricted argmax), so it is byte-for-byte unchanged.
    "detect_window": 10.0,
    "detect_seed_window": 45.0,
    # ── NATIVE DP surface detector (default) ── A globally-smooth anterior-surface detector that matches a
    # careful manual trace far better than the per-column gradient-argmax + RANSAC-quadratic legacy path
    # (which picks wrong layers at low-signal edges and leaves a jagged boundary), so AUTO preprocessing needs
    # little/no manual fix-columns. Pipeline: anisotropic despeckle (more along depth) → dark→bright vertical
    # gradient GATED by "bright cornea tissue just below" (locks to the true anterior surface, not internal
    # layers or top speckle) → dynamic-programming shortest smooth path (per-frame depth step ≤ dp_max_jump),
    # then sub-voxel refined. detector="dp" (default) | "legacy" (the old _merged_side_edge).
    "detector": "dp",
    "dp_sigma_depth": 3.0,        # despeckle Gaussian sigma along DEPTH (heavier — speckle is fine-grained)
    "dp_sigma_frame": 3.0,        # despeckle Gaussian sigma along FRAMES. RAISED 1.2→3.0: a low value let the DP
                                  #   surface follow frame-to-frame speckle → COLUMN-LEVEL edge jitter/notches (the
                                  #   marked CS002 OS(2) defect: a stale auto-tuned 0.8 gave a 5px notch; 3.0 → 0.4px)
    "dp_below": 24,               # depth window (px) just BELOW a candidate used for the "bright tissue below" gate
    "dp_above_gate": True,        # gate the DP score on boundary CONTRAST (below − above) not (below − med): keeps
                                  #   the epithelium (dark air above) over a deeper second layer (bright above) at a
                                  #   specular apex → removes the apex V-notch (CS001 OS3). Verified to CORRECT the
                                  #   same deep-lock notch on the vetted scans (improvement, not degrade). False = legacy.
    "dp_max_jump": 10,            # DP: max surface depth change between adjacent frames (smoothness constraint)
    # ── BOUNDARY EXTRAPOLATION (RETIRED, default OFF) ── replaced the first/last few frames' surface with a
    # frame-direction quadratic extrapolation from the interior. The user marked this as introducing a WRONG
    # EDGE ANGLE vs the general corneal curvature (CS002 OS(2)/(3) "sagital right edge corrected to a wrong
    # angle"): extrapolating along the frame axis projects the interior parabola and can leave the tissue.
    # SUPERSEDED by _lateral_smooth_by_confidence (cross-SLICE smoothing that stays ON the detected tissue).
    # nb>0 re-enables the legacy behaviour.
    "boundary_extrap_nb": 0,      # frames at EACH end to replace with the interior extrapolation (0 = disabled)
    "boundary_extrap_degree": 2,  # 2=QUADRATIC (follows corneal curvature); 1=linear tangent (wrong edge angle)
    "boundary_extrap_max_dev": 15.0,  # clamp |extrap − nearest interior| (px) so the quadratic can't OVERSHOOT the
                                  #   few extrapolated frames off the cornea (a 4-frame descent is well under this)
    "boundary_extrap_span": 18,   # interior frames used for the fit adjacent to each boundary
    "boundary_extrap_lat_sigma": 6.0,  # cross-slice (lateral) gaussian on the boundary frames (3-D consistency)
    # ── CONFIDENCE-TAPERED LATERAL SMOOTHING ── the acquisition-edge frames (low SNR at the slow-scan extremes)
    # detect the anterior surface with cross-SLICE (lateral) jitter → a jagged B-scan top contour (marked CS002
    # OS(2) f99/100, OS(3) f0-20). Smooth the detected surface ACROSS SLICES with a gaussian whose sigma is
    # tapered by each frame's detection CONFIDENCE: strong on the noisy low-confidence edge frames, a strict
    # NO-OP on high-confidence interior frames (real anterior detail + approved scans preserved). Unlike the
    # retired frame-direction extrapolation this stays ON the detected tissue, so it cannot create a wrong edge
    # angle; the specular column + stromal opacities sit BELOW the surface and are untouched. False = disabled.
    "lat_conf_smooth": True,
    "lat_conf_sigma_max": 9.0,    # max cross-slice gaussian sigma (laterals) applied to the lowest-confidence frame
    "lat_conf_lo": 0.35,          # confidence (relative to the interior high-signal median) at/below which full smoothing
    "lat_conf_hi": 0.80,          # confidence at/above which NO smoothing (interior frames → strict no-op)
    # ── LATERAL DESPIKE ── remove NARROW, LARGE cross-slice surface excursions (the DP diving into a shadow/
    # dropout notch or climbing a reflection; marked CS002 OS3 f0 ~35px dive + f6 notch). A run of <= max_w
    # laterals deviating > dev px from a robust lateral median trend is a detection artifact (a smooth cornea
    # never produces one) → reset to the trend. Width-gated so a real limbus flank / smooth dome is a strict
    # no-op. Independent of frame confidence (catches a local spike on an otherwise-confident frame). False disables.
    "despike_lateral": True,
    "despike_win": 31,            # lateral median-trend window (odd; wider than any real narrow spike)
    "despike_dev": 13.0,          # |surface − trend| px above which a narrow run is a spike/notch
    "despike_max_w": 12,          # max lateral run width treated as a spike (a real limbus flank is longer → kept)
    "despike_pad": 2,             # laterals padded around each reset run (absorb flank jitter)
    # ── 2-D DOME-TREND DIP SUPPRESSION ── the moderate-width dips the 1-D despike misses: the anterior surface
    # pulled ~6-10px toward a sub-surface stromal opacity on the low-signal flank (marked CS002 OS3 lat 294-389
    # × frames 0-45). Clip points deviating > dip2d_thresh from a robust 2-D (lateral×frame) median dome trend
    # back to it. The trend follows the real smooth dome + steep limbus (monotonic → median = true value, no
    # lag) and the gentle corneal curvature keeps the apex un-flattened. False disables.
    "dip2d_suppress": True,
    "dip2d_thresh": 7.0,          # px deviation from the 2-D dome trend above which a point is a detection dip/bump
    "dip2d_lat_win": 41,          # lateral median window (odd)
    "dip2d_frame_win": 9,         # frame median window (odd)
    # ── POCKET-ROBUST DOME ── dark intra-stromal POCKETS (disease variant, user directive) must be ridden OVER
    # by the epithelial surface, not dipped into. Iterative ONE-SIDED robust gaussian smoothing pulls points
    # that dived DEEPER than the smooth dome back up to it, so the surface bridges pockets smoothly. One-sided
    # → apex + correct surface are strict no-ops; gentle curvature preserved. Applied uniformly (the true
    # epithelium is smooth on every scan). Handles the moderate/wide pocket-dips the median-clip dip2d cannot
    # (its trend gets contaminated by the dip). False disables.
    "robust_dome": True,
    "robust_dome_sig_lat": 15.0,  # lateral gaussian sigma for the smooth-dome estimate
    "robust_dome_sig_frame": 5.0, # frame gaussian sigma for the smooth-dome estimate
    "robust_dome_thr": 3.5,       # px DEEPER-than-dome above which a point is a pocket-dip → pulled up
    "robust_dome_iters": 4,       # robust re-estimation passes (de-contaminates the dome from the dip)
    "robust_dome_max_pull": 6.0,  # CLAMP px: cap the per-point lift so a large-pocket gaussian can't run away
                                  #   and FLATTEN a real steep frame-direction descent (marked CS002 OS3 curvature)
    # POCKET DARKNESS GATE: apply the dome lift ONLY where the tissue below is dark (a genuine pocket), so a
    # HEALTHY cornea (bright stroma below the epithelium everywhere) is a STRICT NO-OP and its natural
    # curvature is preserved — the ungated dome flattened the healthy CS003 OD start-frame curvature. frac
    # tuned so healthy scans lift ~0 pts while a pocket scan (CS002 OS3) lifts a meaningful set.
    "robust_dome_pocket_gate": True,
    "robust_dome_pocket_frac": 0.72,  # below-surface brightness < frac × frame-median stroma ⇒ pocket
    "robust_dome_below_lo": 4,    # depth px below the surface where the sub-surface sampling band starts
    "robust_dome_below_hi": 30,   # depth px below the surface where it ends
    # ── EDGE PARABOLA CONSTRAINT (DISABLED by default — v148) ── the INTENT was to snap the first/last
    # acquisition-edge frames to a robust per-slice PARABOLA of the reliable interior, so a motion artifact at
    # the acquisition edge could not leave a steep edge off the overall shape (user directive). In practice this
    # BACKFIRED: the cornea FLATTENS toward the limbus and is NOT a true parabola out there, so the interior-fit
    # parabola OVER-descends at the periphery. Hard-snapping the outer frames to that over-descending parabola
    # (while the frames just inside the margin stayed on the true, flatter surface) INJECTED a ~10px downward
    # STEP/V-notch right at the margin boundary (~frame 87) — the exact "steep edge" it was meant to remove
    # (measured CS003 OD1: right-flank step 9.5px ON → 2.2px OFF; visually a clean smooth descent OFF). The raw
    # detection already tracks the limbus flattening smoothly, so the correct behaviour is to leave it alone.
    # Kept as an opt-in param (default False) — no approved scan uses it (added v145, CS002 OS approved at v132).
    "parabola_edge": False,
    "parabola_edge_nb": 4,        # frames at EACH end snapped to the interior parabola
    "parabola_edge_deg": 2,       # 2 = parabola (the corneal cross-section model)
    # ── FRAME-EDGE OVER-DESCENT CAP (v149) ── the flatten quadratic over-descends the acquisition-edge frames
    # (~10-28px deeper than the true interior level; the cornea flattens at the limbus + a frame-0 inter-frame
    # MOTION STEP). Fit a robust deg-1 trend to the RAW interior PAST the motion-step block, extrapolate across
    # the edge, and one-sided soft-clamp the output to it (min(cur, expected+dev)): FLAT edge → lifted back up
    # (fixes it); genuinely DESCENDING limbus → expected descends with it → no-op (no upward hook). Per-slice
    # no-op when the edge is already on-trend. False disables.
    "frame_edge_cap": True,
    "frame_edge_nb": 10,          # first/last N frames eligible for the clamp (covers a motion-step block)
    "frame_edge_gap": 4,          # frames skipped PAST the block before the interior fit window (avoids contamination)
    "frame_edge_reach": 16,       # length of the robust deg-1 interior fit window (frames)
    "frame_edge_dev": 3.0,        # px deadband: only lift an edge deeper than the extrapolated trend by MORE than this
    "frame_edge_soft": 5.0,       # px over which the blend weight ramps 0→1 (soft gate, no hard threshold)
    "frame_edge_lat_med": 15,         # lateral MEDIAN window on the edge boundary → rejects narrow warp spikes before smoothing
    "frame_edge_lat_smooth": 40.0,   # gaussian sigma ACROSS slices on the lift field → smooth en-face/axial edge (no fuzz)
    "frame_edge_frame_smooth": 2.0,  # gaussian sigma ALONG frames on the lift field → no wavy sagittal edge
    "frame_edge_conf_frac": 0.06,    # do-no-harm gate: ramp the de-bump to 0 where tissue-contrast < frac× the frame's
                                     #   typical contrast. LOW now (0.06) so dim-but-real edge tissue is still smoothed;
                                     #   the frame_edge_max_shift clamp is the real safety net against no-signal columns
    "frame_edge_max_shift": 15.0,    # px: hard cap on the per-column surface shift (de-bump + rawcap) → a floating
                                     #   no-tissue column can never shove tissue out of frame (black bands); gate stays relaxed
    "frame_edge_rawcap": False,      # RETIRED v0.0.156 (was the v0.0.155 "very steep curvature near the ends" fix). The
                                     #   post-hoc over-descent cap DEGRADED approved scans: lifting the last ~10 frames toward a
                                     #   less-steep target FLATTENED the natural smooth trailing descent AND injected a KINK/step
                                     #   at the cap's engagement boundary (~frame 90) — the SAME boundary-step failure that
                                     #   retired parabola_edge (⑯). Verified on the actual app previews (rep1 trailing, v154 smooth
                                     #   vs v155 kinked). The steep peripheral descent is REAL smooth tissue; do not lift it.
                                     #   Kept the code (frame_edge_overdescent_cap) but default OFF so it never runs.
    "frame_edge_rawcap_nb": 16,      # first/last N frames the over-descent cap covers (spans the limbus plateau zone)
    "frame_edge_rawcap_gap": 4,      # frames skipped past the edge band before the clean-interior anchor window
    "frame_edge_rawcap_margin": 3.0, # px: how far above the reference the lifted edge lands (small residual deadband)
    "frame_edge_rawcap_fire": 10.0,  # px over-descent to START acting → a gentle legit edge (small excess) is a no-op;
                                     #   only a steep od2-type over-plunge (large excess vs the pre-flatten reference) fires
    "frame_edge_rawcap_ramp": 6.0,   # px over which the fire gate ramps 0→1 (soft, so no lateral on/off jag)
    "frame_edge_rawcap_max_shift": 14.0,  # px: hard cap on the per-column lift → tissue can never be shoved out of frame
    "frame_edge_curve_snap": False,  # CONDITIONAL edge→overall-corneal-curve snap — RETIRED default OFF (tested, not
                                     #   shipped): the per-slice frame-direction snap toward the corneal arc barely reduces
                                     #   the edge stair-steps (the detector re-finds them) AND re-ROUGHENS the en-face/axial
                                     #   boundary (undoes frame_boundary_lat_smooth) — a fundamental frame-vs-lateral
                                     #   tradeoff, same class of failure as the v155 over-descent cap. Code kept for reference
    "frame_edge_snap_nb": 18,        # first/last N edge frames considered
    "frame_edge_snap_thresh": 4.0,   # px: minimum deviation from the curve to start correcting (defines "obviously")
    "frame_edge_snap_soft": 3.0,     # px over which the correction gate ramps 0→1
    "frame_edge_snap_max_shift": 18.0,   # px: hard cap on the per-column snap shift
    "frame_edge_snap_deg": 3,        # degree of the robust overall-corneal-curve polynomial fit
    "frame_edge_snap_lat_med": 9,    # lateral median window on the deviation (kills isolated per-slice spikes)
    "frame_edge_snap_conf_frac": 0.20,   # tissue-contrast gate: skip columns that ran off the cornea (no reliable curve)
    "dp_smooth_weight": 0.0,      # DP step-magnitude penalty λ (cost += λ·|step|; score∈[0,1]). 0 = hard-cap only
                                  #   (legacy). Small λ (~0.02–0.06) removes the apex/flank V-notch by discouraging
                                  #   maxj-sized hops onto deeper layers, while a real steep descent still pays off.
    # ── specular-spike rejection (anterior) ── A thin ultra-bright VERTICAL specular reflection at the corneal
    # apex (a narrow bright line rising above the true epithelial dome, present in the RAW data) has a strong
    # dark→bright top edge with bright tissue below, so the DP path climbs onto it and warps a spike. This guard
    # runs AFTER the DP path (auto detection only, prior=None): where the detected anterior rises ABOVE a
    # laterally-robust median trend (which ignores narrow spikes) by > spike_min_height px over a CONTIGUOUS run
    # no wider than spike_max_width frames, that narrow upward excursion is a specular streak → its frames are
    # replaced by linear interpolation from the smooth surface on either side (the specular is left as a bright
    # artifact ABOVE the corrected surface). ONE-SIDED (only upward/shallower) and NARROW-run gated, so a genuine
    # wide/steep dome apex is untouched and a clean apex (edge already on the trend) is a strict no-op.
    # spec_spike_reject=False disables.
    "spec_spike_reject": True,
    "spec_spike_max_width": 6,    # max contiguous frame run treated as a (narrow) specular spike (incl. boxy blocks)
    "spec_spike_min_height": 8.0, # min px the edge must rise ABOVE the median trend to be a spike
    "spec_spike_trend_win": 21,   # frames in the robust median trend (must exceed spike width to survive it)
    # ── FIX apexspec: LATERAL (across-slice) specular-spike rejection on the ASSEMBLED surface ── The per-slice
    # spec_spike_reject above runs along FRAMES for each fixed lateral, so a narrow ultra-bright VERTICAL specular
    # streak at the corneal APEX — which sits at a fixed lateral band and spans a run of frames — is only partly
    # caught (each lateral slice climbs the streak independently, and a per-frame interpolation cannot make the
    # LATERAL profile smooth). The result is a jagged spike in the AXIAL (fixed-frame) B-scan where the detected
    # surface follows the streak. This guard runs on detect_surface_all's full (lateral, frame) surface: for each
    # FRAME column it takes a LATERALLY-ROBUST median trend (window > streak width, so it ignores the spike and
    # follows the true dome), finds NARROW contiguous lateral runs that rise ABOVE the trend by > min_height (the
    # specular streak lures the DP up = shallower), and replaces a small padded band around each such run with the
    # median trend — removing the up-spike AND the adjacent jitter the streak induces while the true smooth dome
    # (which the wide-window median already follows) is preserved. ONE-SIDED (upward-triggered) + NARROW-run gated
    # → a clean apex (surface already on the lateral trend) is a strict no-op. Auto detection only (prior=None).
    # apex_lateral_reject=False disables.
    "apex_lateral_reject": True,
    "apex_lat_trend_win": 41,     # lateral px in the robust median trend (must exceed the streak's lateral width)
    "apex_lat_min_height": 8.0,   # min px the surface must rise ABOVE the lateral trend to trigger
    "apex_lat_max_width": 18,     # max contiguous lateral run of up-spike triggers treated as a specular band
    "apex_lat_pad": 4,            # lateral px padded around each triggered run (absorbs the induced jitter flanks)
    "apex_lat_reject_down": False,  # OFF (byte-identical to legacy upward-only). A lateral down-notch reset can't
                                  #   cleanly fix the CS001 apex V-notch: the notch is at the apex→flank transition
                                  #   (guarding flanks also protects the notch) AND is a generic apex-specular
                                  #   feature the vetted scans share (unguarded → steps the vetted flanks). The
                                  #   proper fix is DP-detector-level (stop the deep lock at the specular apex).
    "apex_lat_down_min_height": 10.0,  # min px the surface must sit BELOW the lateral trend to trigger a down-notch reset
    "apex_lat_dip_recover": 5.0,  # a down-notch resets ONLY if the surface returns to within this px of the trend on
                                  #   BOTH sides of the run (true local dip); a descent (one side stays deep) is a no-op
    # ── 2-D SURFACE-REFINE pass ── final robust clean-up of LOCAL column-level edge-detection errors (patches a
    # few px off the smooth dome in BOTH lateral and frame directions — the axial_consistency pass only looks
    # laterally within a frame and with a narrow window, so a wider/frame-narrow notch slips through). Robust 2-D
    # median target + hard deviation gate → strict NO-OP on an already-smooth surface (approved scans unchanged).
    "surface_refine_2d": False,   # DEFAULT OFF: degraded a smooth approved scan by ~17px in testing (its 2-D
                                  #   median target flattens genuine curvature at frame edges). Under investigation.
    "srf_dev_thresh": 2.5,        # px: a column deviating MORE than this from the smooth 2-D target is corrected
    "srf_lat_med": 9,             # lateral median window (px) for the smooth target
    "srf_frame_med": 9,           # FRAME median window (frames) — catches a frame-narrow notch axcons cannot
    "srf_gauss": 1.5,             # light Gaussian on the target (removes the median's staircase)
    "srf_max_shift": 12.0,        # cap the per-column correction (px) so a mis-fit target can't tear tissue
    "srf_iters": 2,               # detect→correct passes (self-terminates when nothing exceeds the gate)
    "srf_min_coverage": 0.5,      # a frame with less cornea than this fraction is left untouched (off-eye guard)
    # ── AXIAL-CONSISTENCY pass ── final per-FRAME lateral clean-up of the anterior surface (kills slice-to-slice
    # waviness/spikes only visible in the AXIAL B-scan); strictly gated → strict NO-OP on an already-smooth scan.
    "axial_consistency": True,
    # ── FRAME-BOUNDARY lateral smoothing ── the first/last B-scans (acquisition edge) are low-signal, so their
    # anterior detection is jagged laterally and axial_consistency's gate skips them. Force lateral consistency on
    # just those `fbls_nb` edge frames (wide median + gaussian). No-op elsewhere → interior/approved scans unchanged.
    "frame_boundary_smooth": True,   # v0.0.157 DEFAULT ON (user-requested): the START/END acquisition-edge frames
                                  #   (first/last fbls_nb) carry a WAVY corneal top edge — the low-signal per-lateral
                                  #   detection jitters, so the warped band doesn't align to a smooth corneal curve
                                  #   ("poor alignment of corneal edge to curve at the start/ends of axial slices").
                                  #   frame_boundary_lat_smooth aligns those edge frames to a wide-median+gaussian
                                  #   lateral arc → cleaner en-face edge. Verified: edge-frame lateral roughness DROPS
                                  #   (rep1 0.75→0.64, rep3 1.16→0.70) — improves BOTH marked and approved scans; local
                                  #   to the edge frames (interior untouched). Earlier default-off was too conservative.
    "fbls_nb": 8,                 # number of frames at EACH end to lateral-smooth (feathered toward the interior)
    "fbls_med": 41,               # lateral median window (px) — wide enough to erase the low-signal jag
    "fbls_gauss": 16.0,           # lateral gaussian on the smooth target (clean curve)
    "fbls_max_shift": 16.0,       # cap the per-column correction (px)
    "fbls_min_coverage": 0.4,     # skip an edge frame with less cornea than this fraction (off-eye guard)
    "axcons_med_win": 15,         # lateral median-filter width (px) for the smooth target (kills spikes, keeps dome)
    "axcons_two_sided": False,    # correct only DOWNWARD notches (detector locked too deep); True also lifts up-spikes
    "axcons_gate": 3.0,           # px: correct only lateral columns deviating from the smooth target by more than this
    "axcons_max_shift": 10.0,     # px: hard clamp on the per-column depth nudge (bounded, never a re-flatten)
    "axcons_strength": 1.0,       # fraction of the gated deviation removed per column
    "axcons_min_frac": 0.02,      # frame no-op unless > this frac of lateral cols exceed the gate (ignore specks)
    "axcons_max_frac": 0.18,      # frame no-op if MORE than this frac of in-cornea cols exceed the gate (rough/off-cornea)
    "axcons_min_coverage": 0.5,   # frame no-op unless the cornea fills >= this frac of the lateral span
    "axcons_iters": 2,            # detect→nudge repeats (a deep notch needs 2; a smooth frame stays no-op)
    # ── NATIVE AUTO-TUNE ── The app tunes the DP params to EACH scan (no user input): a coordinate-descent
    # sweep scored by on-board surface confidence (contrast − weight·roughness) on sampled slices, run at the
    # start of preprocessing; the chosen dp_* are used for the warp AND persisted so the fix-columns baseline
    # matches. auto_tune=False keeps the fixed defaults.
    "auto_tune": True,
    "autotune_smooth_weight": 18.0,   # roughness penalty vs contrast in the auto-tune objective
    # ── fix-columns "Confirm" = LOCAL re-detection ── A user correction should change ONLY the corrected
    # ("pink line") region + a BAND of neighbouring slices around it (the detector uses neighbour comparison,
    # so they're re-detected too); the rest of the auto-detected surface is satisfactory and kept untouched.
    # redetect_frame_margin = blend margin (frames) on each side of the corrected frame span; redetect_slice_band
    # = how many neighbouring slices each side of the anchored slice(s) are re-detected, the correction blending
    # smoothly back to the auto edge across that band (no seam). Drag on more slices to widen the corrected span.
    "redetect_frame_margin": 8,
    # PROPAGATE-TO-NEARBY (user request): a correction also re-detects the SAME corrected frame columns on the
    # ±N neighbour slices around each anchored slice, seeded by (pulled toward) the drawn surface — so a fix at
    # a bad cornea edge improves the neighbouring slices' boundary too (the detector error usually spans a band).
    # Strictly confined to the corrected FRAME columns (the rest of every slice stays the auto baseline) and the
    # weight ramps to 0 at ±N so there is no seam. 0 = strictly local (only the edited slices). Was 0 in v0.0.50;
    # re-enabled in v0.0.51 now that the leak (corrections bleeding onto un-edited frames) is fixed.
    "redetect_slice_band": 20,
    # The user-drawn line is trusted: the seed re-detection only snaps to the nearest gradient within
    # ±redetect_seed_window depth px of the drag (1-2 px), instead of a generous search that could wander off
    # the line. The march to neighbouring slices then tracks the surface within ±detect_window.
    "redetect_seed_window": 2.0,
    # After interpolating the correction across the gap between anchored slices, re-detect the best edge within
    # ±redetect_interp_window px of that interpolated border on the un-anchored in-between slices. NARROW (< the
    # typical correction) so the snap can't fall back to the too-shallow auto edge, but wide enough to refine to
    # each slice's real gradient. 0 → pure interpolation (no refine).
    "redetect_interp_window": 3.0,
    # ── smooth_corrected_volume: re-detect the corrected surface + slice-smooth + re-warp (the "Smooth corrected
    # volume" button). smooth_slice_sigma = gaussian σ across SLICES (frame axis untouched → corrections kept);
    # smooth_max_shift caps the per-column warp; smooth_iters re-detect→warp rounds.
    "smooth_slice_sigma": 4.0,
    "smooth_frame_sigma": 2.0,   # GENTLE frame-direction smoothing: removes the per-column 1-2px detection errors
                                 # (the source of axial fuzziness) while broad manual corrections survive
    "smooth_max_shift": 20.0,
    "smooth_iters": 2,
    # ── generalize_surface: propagate the LEARNED correction to the WHOLE volume ── When the user corrects
    # a few slices, learn the systematic per-frame residual (anchor − auto) and interpolate it across all
    # slices (not just the ±redetect_slice_band local band). A frame is generalized only if corrected the same
    # direction on >= gen_min_slices slices with robust median residual > gen_min_resid px (so one-off edits
    # aren't globalized). gen_resid_cap clamps a wild mis-click; taper/sigma smooth the correction field.
    "gen_min_slices": 2,
    "gen_min_resid": 3.0,
    "gen_sign_frac": 0.7,
    "gen_resid_cap": 45.0,
    "gen_taper_slices": 20,
    "gen_frame_margin": 6,
    "gen_slice_sigma": 8.0,
    "gen_frame_sigma": 2.0,
    # ── clipped-apex handling ── In some scans the cornea sits so high in the acquisition window that the
    # dome APEX rises ABOVE depth 0 across the central frames. Those columns have tissue filling from row 0
    # with NO dark air gap and NO air→epithelium edge, so the detector pins the edge at the top (~5px) and
    # the quadratic CLAMPS its apex to ~0 instead of extrapolating it above the frame from the valid flanks.
    # Worse, the resulting (quad−edge) displacement is NEGATIVE and the warp pushes real epithelial rows OFF
    # the top of the frame (lost tissue). When enabled, such columns are detected, EXCLUDED from the
    # quadratic fit (so it extrapolates from the in-frame flanks; the apex may go <0) and their warp shift is
    # clamped ≥0 (no real tissue lost). Every gate is a strict no-op on a normal in-frame dome (which has a
    # dark gap above the surface), so a well-detected scan is byte-for-byte unchanged. clip_handling=False is
    # a hard kill-switch. Thresholds calibrated on real clipped eyes (CS005 OD) + controls (CS001/CS004).
    "clip_handling": True,
    "clip_top_rows": 5,        # depth rows averaged for the top-band brightness test
    "clip_edge_floor": 8.0,    # a column is 'pinned at top' (clip symptom) when its detected edge < this row
    "clip_top_frac": 0.5,      # ...and clipped only if mean(top rows)/colmax > this (tissue from row 0, no air gap)
    "clip_min_cols": 6,        # min clipped columns before a slice is treated as clipped (ignore isolated noise)
    "clip_min_run": 5,         # min CONTIGUOUS run of clipped columns (a real dome apex is contiguous)
    "clip_min_flank": 5,       # min VALID (in-frame) columns required on EACH side of the clip band — a 1-2px
                               # flank can't constrain a parabola and extrapolates to garbage (a one-sided
                               # limbus/edge-of-volume clip is intentionally left to the manual fix-columns tool)
    "clip_apex_floor": -60.0,  # reject the fit if its extrapolated apex is more than this far above the frame
                               # (a backstop against degenerate extrapolation; a real apex sits just above row 0)
    "clip_a_min": 0.008,       # accept only if the masked-valid parabola x² coef ∈ [a_min, a_max] (curvature band)
    "clip_a_max": 0.05,        # ...rejects the limbus/edge-of-volume false positive (too-steep, not a dome)
    "clip_flank_rms": 8.0,     # ...and the flank-inlier RMS ≤ this (a real dome's flanks fit a parabola well)
    "clip_inlier_frac": 0.6,   # ...and the RANSAC inlier fraction on the valid columns ≥ this
    "clip_close_gap": 4,       # fill internal gaps ≤ this in the clip mask (DP can fragment a clipped run)
    # ── surface-crop (manual, bottom-edge guidance) ── the "Detect surface crop" tool auto-suggests frames
    # clipped in ≥crop_min_slices sagittal slices; the user confirms the set (surface_crop_frames) and a re-run
    # reconstructs those frames by POSTERIOR CONTINUITY (build_surface_crop_edges) instead of the auto apex
    # extrapolation. A sticky oct_param, applied via the provided_edges warp path. 0 frames = feature inactive.
    "crop_min_slices": 3,
    # AUTO crop-region (off-cornea NOISE): the slow scan can run OFF the cornea, leaving leading/trailing frames
    # that are pure noise (no coherent cornea surface, ~zero edge contrast). Auto-detected + zeroed before SAM2
    # (like the manual #9 crop). Only LONG boundary blocks are cut, so a faint cornea EDGE is never removed.
    "auto_crop_region": True,
    "crop_noise_frac": 0.20,   # a frame is "no cornea" if its edge contrast < this fraction of the cornea frames'
    "crop_noise_min_run": 10,  # ...and only a contiguous boundary run of >= this many such frames is cropped
    "crop_noise_max_frac": 0.75,  # safety: never auto-crop more than this fraction of frames (a failed scan)
    "crop_margin": 6.0,        # reconstruct a marked frame only where the posterior-continuity surface sits this
                               # many px ABOVE the detected anterior (the clip symptom) — a strict no-op elsewhere
    # surface-crop CORRECTION (auto + manual): a clipped cornea (apex and/or a whole edge ABOVE the acquisition
    # window) is corrected by fitting the still-visible POSTERIOR (bottom) edge to a per-slice PARABOLA, aligning
    # each column to it (robust shift), and EXTENDING the depth canvas UPWARD so the above-old-top top-edge
    # parabola apex/edge and the cut-off columns are kept/visible (never truncated). SAM2 cornea verified on the
    # taller volume. Detection (detect_surface_crop_frames / _clip_mask) runs automatically; a substantial clip
    # (auto gate below) triggers the correction. Manual surface_crop_frames overrides the auto set.
    "auto_surface_crop": True,    # auto-detect + auto-correct a clipped cornea as part of preprocessing
    "crop_auto_min_frames": 6,    # auto gate: need >= this many frames flagged clipped (>= crop_min_slices slices)
    "crop_auto_min_slices": 12,   # auto gate: AND the most-clipped frame flagged in >= this many slices (ABSOLUTE —
                                  # a central apex clip only spans the central lateral slices, so a fraction-of-all
                                  # test wrongly rejects it; a few stray flags on a normal dome stay well under this)
    "crop_auto_max_span": 200,    # auto gate: AND the fitted posterior parabola spans <= this many depth-voxels
                                  # across frames. A clean clip has a near-flat posterior (span ~70-90); a much
                                  # larger span is a steep TILT / decentred scan (CS008 OD ~341) that the extend
                                  # warp would mangle — AUTO skips it (→ normal pipeline). A MANUAL crop ignores this.
    "crop_auto_max_pad": 75,      # auto gate: AND the REQUIRED upward extension <= this many rows. A genuine apex
                                  # clip needs only a modest pad (sweep: CS014/CS015/CS021 all ~46-48); a pad of
                                  # 100+ means the warp wants to shift the whole cornea up a long way = a decentred
                                  # / tilted / artefacted scan, not a localized clip (sweep false-fires CS005 OD(6)
                                  # pad 100, CS011 OD(3) pad 110 → gross SAM2 over-seg). pad is the CLEAN
                                  # discriminator (frac/span overlap), so AUTO skips a too-large pad (→ normal
                                  # pipeline). A MANUAL crop ignores this. Below crop_max_pad (the clamp cap).
    "crop_target_med": 11,        # robust median window (frames) on the detected posterior before the shift
    "crop_slice_smooth": 20.0,    # cross-slice gaussian on the posterior parabola. RAISED 2→20: a weak value let
                                  #   the per-slice parabola fit WOBBLE slice-to-slice → a low-frequency LATERAL
                                  #   UNDULATION ("very wavy" reconstruction); the cornea's curve is consistent
                                  #   across slices, so heavy cross-slice smoothing recovers that consistent shape.
    "crop_disp_smooth_slice": 40.0,  # anti-wave: gaussian on the per-column warp shift ACROSS slices. RAISED 6→40
                                  #   for the same reason — kills the lateral wave from a faint/noisy posterior.
    "crop_disp_smooth_frame": 1.5,  # across frames (lighter — real frame-direction shape is kept). 0 = disabled.
    "crop_pad_margin": 8,         # extra rows above the highest above-old-top point (breathing room)
    "crop_max_pad": 160,          # safety cap on the upward extension (rows). A required pad above this means a
                                  # SEVERE tilt / artifact, not a clean clip: AUTO skips it (→ normal pipeline);
                                  # a MANUAL crop is applied but clamped here (best-effort, no pathological volume)
    # ── DP scar-guard ── cross-check the DP anterior edge against the legacy ('old method') RANSAC-quadratic
    # surface, which is robust to a bright internal scar. Where DP dives >dp_scar_tol px DEEPER than legacy over
    # a run of >=dp_scar_min_run frames (the scar-lock signature), re-run DP confined to +/-dp_scar_window of the
    # legacy surface so it tracks the true (first) boundary. One-sided + run-gated → no-op on a clean scan.
    "dp_scar_guard": True,
    "dp_scar_tol": 18.0,       # DP must not sit more than this many px DEEPER than the legacy surface
    "dp_scar_window": 12.0,    # when it does, re-detect DP within +/- this of the legacy surface (excludes the scar)
    "dp_scar_min_run": 6,      # min contiguous run of deeper-than-tol frames to trigger (ignore isolated noise)
    "dp_scar_darker_margin": 0.05,  # adopt a pulled-back frame only if its above-band is this much DARKER (normalised)
                               # than the deep DP edge — confirms it's a true air->tissue surface, not a wrong fit
    # ── GLOBAL DE-TILT pre-alignment (defect ④) ── A few acquisitions come out with the whole cornea acquired
    # STRONGLY TILTED (a ~45° diagonal bright band, ~3 px/frame ≈ 300 px total across the frames) rather than a
    # centred dome. The cornea is FULLY PRESENT and continuous in the raw data — it's just tilted — but the
    # per-slice quadratic flatten fits each sagittal slice to ITS OWN tilted quadratic, so the tilt is PRESERVED,
    # and the near-row-0 flank makes the clip/crop handling mis-fire → a hard SURFACE CUT / V-notch. Fix: BEFORE
    # detection/flatten, robustly estimate the DOMINANT LINEAR tilt of the anterior surface in the FRAME direction
    # (the slope is near-identical for every lateral slice — a pure acquisition tilt) and REMOVE it by rigidly
    # shifting each frame's whole (depth,lateral) plane in depth, extending the depth canvas so nothing truncates.
    # The cornea then sits near-horizontal → the normal detector/flatten produce a smooth centred dome with no cut.
    # GATED so a normal scan is a strict NO-OP. NOTE: the frame-direction linear slope is NOT a reliable tilt signal
    # on its own — an OFF-CENTRE dome (apex captured at a non-central frame, e.g. frame ~10-20) has a large net
    # linear slope purely from geometry, indistinguishable from acquisition tilt by the anterior parabola alone
    # (tilt and apex-offset are the SAME linear term). So the ONLY honest discriminator is de-tilt's PURPOSE: it
    # only helps when the tilt runs the surface OFF THE TOP of the window (near row 0) at a frame end — that is the
    # clip/V-notch it exists to prevent. A dome that stays comfortably in-frame needs no de-tilt regardless of slope.
    # Hence the gate = (|total linear tilt| >= detilt_min_total) AND (>= detilt_clip_min_frames frames whose surface
    # sits within detilt_clip_row px of the top). This makes off-centre domes with no clip (the false positives) a
    # strict NO-OP while keeping de-tilt as the pre-step for a genuinely clipped, tilted acquisition.
    "auto_detilt": True,
    "detilt_min_total": 150.0,  # min |robust linear tilt| (px, across all frames) to trigger de-tilt (else no-op)
    "detilt_clip_row": 30.0,    # a frame surface within this many px of the top counts as clipped (tilt ran off-top)
    "detilt_clip_min_frames": 3,  # need >= this many clipped frames for de-tilt to apply (else the slope is dome geometry)
    "detilt_max_pad": 400,      # safety cap on the canvas extension (px) added top+bottom by the de-tilt shift
}
# Optovue Angiovue XR Avanti "3D Cornea" geometry (corrected from the companion .txt;
# the conversion script's hardcoded 0.00625/0.0078 implied a 4x4x4mm cube — wrong, the
# real volume is 6.00mm lateral x 4.04mm x 2.006mm depth). Array is (frames,rows,cols)
# = (101 slices, 640 depth, 513 lateral). All exposed/overridable via params.
DEPTH_SPACING = round(2.006 / 640, 7)     # rows  (axial / Scan Depth / OCT Window Height)
LATERAL_SPACING = round(6.00 / 513, 7)    # cols  (fast B-scan line / XY Scan Size1 / Length)
SLICE_SPACING = 0.040                      # frames(slow axis / XY Scan Interval1)
DEFAULT_SLICE_THICKNESS = SLICE_SPACING
DEFAULT_PIXEL_SPACING = (DEPTH_SPACING, LATERAL_SPACING)   # DICOM [row, col]
# NIfTI geometry to match the app's existing volumes: sitk spacing (x,y,z)=(lateral,depth,slice),
# direction as Slicer produced for these OPT volumes, origin 0.
NIFTI_SPACING = (LATERAL_SPACING, DEPTH_SPACING, SLICE_SPACING)
NIFTI_DIRECTION = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)


def auto_workers(reserve: int = 1) -> int:
    """Per-slice parallel worker count for a SINGLE scan's processing — use ALL available cores (leaving
    `reserve` for the main/IO thread), with NO arbitrary upper cap, so the app scales to whatever machine it
    runs on (e.g. a 24-thread Ryzen → 23 workers). When several scans run concurrently the caller passes an
    explicit, smaller `workers` so K scans × workers ≈ all cores (no oversubscription).

    CORNEA_WORKERS env caps this — a parallel sweep sets it to cores/N so N concurrent scans don't
    oversubscribe the CPU during preprocessing (the endpoint runs the CLI without an explicit worker count)."""
    env = os.environ.get("CORNEA_WORKERS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 2) - max(0, int(reserve)))


# ── 1) read .OCT ───────────────────────────────────────────────────────────
class MissingCompanionError(ValueError):
    """The .OCT's companion .txt filespec isn't next to it (POCT can't read without it)."""


def read_oct_zstack(oct_path: str | Path, volume_index: int = 0) -> np.ndarray:
    """Read one volume's B-scan stack from an .OCT file → (frames, H, W) float32.

    An .OCT may hold several captures; the original pipeline uses volume 0. The
    Optovue .OCT stores its dimensions in a companion .txt that MUST sit next to it —
    POCT fails without it, so we check up front and raise an actionable error."""
    from oct_converter.readers import POCT
    p = Path(oct_path)
    if not (p.with_suffix(".txt").exists() or p.with_suffix(".TXT").exists()):
        raise MissingCompanionError(
            f"'{p.name}' has no companion .txt next to it — an Optovue .OCT can't be read "
            "without it. Upload the .OCT together with its .txt (or load the whole folder).")
    vols = POCT(str(oct_path)).read_oct_volume()
    if not vols:
        raise ValueError(f"No OCT volumes found in {oct_path}")
    vi = volume_index if 0 <= volume_index < len(vols) else 0
    arr = np.asarray(vols[vi].volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Unexpected OCT volume shape {arr.shape} in {oct_path}")
    return arr


def oct_num_volumes(oct_path: str | Path) -> int:
    from oct_converter.readers import POCT
    return len(POCT(str(oct_path)).read_oct_volume())


# ── 2) OCT → DICOM (metadata from filename + companion .txt) ────────────────
def parse_oct_filename(filename: str) -> dict:
    base = os.path.splitext(os.path.basename(filename))[0]
    toks = base.split("_")
    if len(toks) < 5:
        return {}
    # The date token is "YYYY-MM-DD" optionally followed by a replicate suffix "(N)".
    # Parse the date even when there's no "(N)" so the FIRST scan isn't left date-less.
    m = re.match(r"(\d{4}-\d{2}-\d{2})(?:\s*\((\d+)\))?", toks[4])
    return {
        "patient_name": toks[0],
        "patient_id": toks[1],
        "study_description": toks[2],
        "laterality": toks[3],
        "study_date": m.group(1) if m else "",
        "series_number": int(m.group(2)) if (m and m.group(2)) else 1,
    }


def parse_companion_file(txt_path: str | Path) -> dict:
    data: dict = {}
    with open(txt_path, "r", encoding="utf8", errors="ignore") as f:
        for line in f:
            if "=" in line:
                key, val = [x.strip() for x in line.split("=", 1)]
                k = key.lower()
                if k == "eye scanned":
                    data["eye_scanned"] = val
                elif k == "scan depth":
                    data["scan_depth"] = _to_float(val)
                elif k == "physical video width":
                    data["physical_video_width"] = _to_float(val)
                elif k == "physical video height":
                    data["physical_video_height"] = _to_float(val)
    return data


def _to_float(val: str) -> float | None:
    try:
        return float(re.sub(r"[^0-9.\-]", "", val))
    except ValueError:
        return None


# ── per-scan voxel geometry from the companion .txt (the source of truth) ────
# The .OCT's companion .txt records the TRUE acquisition geometry. It varies per
# scan (e.g. XY Scan Size1 = 4.60mm for CS019, 6.00mm for CS015), so the geometry
# must be read per-scan, not hardcoded. The file lists several "[CL - 3D Cornea
# Step N]" blocks; only ONE is the active 3D acquisition — the Step whose
# "XY Scan Usage" equals the slice/frame count (the others are Usage=1 placeholders).
def _parse_companion_full(txt_path: str | Path):
    """Parse the companion .txt into (top-level dict, {step_num: detail dict}).

    Top-level: oct_window_height, scan_depth, eye_scanned.
    Per step: length (XY Scan Length), usage (XY Scan Usage), size1 (XY Scan
    Size1, mm), interval1 (XY Scan Interval1, mm)."""
    top: dict = {}
    steps: dict = {}
    cur_step, in_detail = None, False
    with open(txt_path, "r", encoding="utf8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            sm = re.match(r"\[CL - 3D Cornea Step (\d+)(\s+Detail)?\]", line)
            if sm:
                cur_step, in_detail = int(sm.group(1)), bool(sm.group(2))
                steps.setdefault(cur_step, {})
                continue
            if line.startswith("["):                 # a non-step section resets context
                cur_step, in_detail = None, False
            if "=" not in line:
                continue
            key, val = [x.strip() for x in line.split("=", 1)]
            k = key.lower()
            if cur_step is None:
                if k == "oct window height":
                    top["oct_window_height"] = _to_float(val)
                elif k == "scan depth":
                    top["scan_depth"] = _to_float(val)
                elif k == "eye scanned":
                    top["eye_scanned"] = val
            else:
                s = steps[cur_step]
                if not in_detail:
                    if k == "xy scan length":
                        s["length"] = _to_float(val)
                    elif k == "xy scan usage":
                        s["usage"] = _to_float(val)
                else:
                    if k == "xy scan size1":
                        s["size1"] = _to_float(val)
                    elif k == "xy scan interval1":
                        s["interval1"] = _to_float(val)
                    elif k == "xy scan usage1" and s.get("usage") is None:
                        s["usage"] = _to_float(val)
    return top, steps


def companion_geometry(txt_path: str | Path, n_frames: int | None = None) -> dict:
    """Derive per-scan voxel spacing (mm) from the companion .txt. Returns a dict
    with any of lateral_spacing / depth_spacing / slice_spacing that could be
    resolved (empty if the file is unreadable/unrecognised — caller falls back to
    the Avanti constants). Picks the active acquisition Step by frame count."""
    try:
        top, steps = _parse_companion_full(txt_path)
    except Exception:  # noqa: BLE001
        return {}
    if not steps:
        return {}

    def usage(s: dict) -> float:
        return s.get("usage") or 0.0

    active = None
    if n_frames:
        active = next((s for s in steps.values() if usage(s) == n_frames), None)
    if active is None:                                # else the most-acquired step
        active = max(steps.values(), key=usage, default=None)
    if not active:
        return {}
    geom: dict = {}
    size1, length = active.get("size1"), active.get("length")
    depth, win_h = top.get("scan_depth"), top.get("oct_window_height")
    interval1 = active.get("interval1")
    if size1 and length:
        geom["lateral_spacing"] = size1 / length
    if depth and win_h:
        geom["depth_spacing"] = depth / win_h
    if interval1:
        geom["slice_spacing"] = interval1
    return geom


# Plausible Avanti 3D-Cornea voxel-spacing ranges (mm); outside these we warn so a
# wrong-geometry volume can't silently corrupt the scar metric.
SPACING_BOUNDS = {"lateral": (0.0050, 0.0140), "depth": (0.0025, 0.0040), "slice": (0.020, 0.060)}


def validate_spacing(spacing_xyz) -> list:
    """Return human-readable warnings for any (lateral, depth, slice) spacing that
    falls outside the plausible Avanti range — purely advisory, never raises."""
    sp = [float(s) for s in spacing_xyz[:3]]
    names = ("lateral", "depth", "slice")
    warns = []
    for val, name in zip(sp, names):
        lo, hi = SPACING_BOUNDS[name]
        if not (lo <= val <= hi):
            warns.append(f"{name} spacing {val:.5f}mm outside Avanti range [{lo}, {hi}]")
    return warns


def oct_to_dicom(oct_path: str | Path, output_path: str | Path,
                 patient_name: str = "", patient_id: str = "", study_desc: str = "",
                 series_num: int = 1, orient_vec=None,
                 slice_thickness: float = DEFAULT_SLICE_THICKNESS,
                 pixel_spacing=DEFAULT_PIXEL_SPACING,
                 volume_index: int = 0) -> str:
    """Lossless OCT → uint16 multi-frame DICOM (DICOMGeneratorlossless.oct_to_dicom),
    with the read contract fixed to volume[volume_index].volume."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    num_frames, rows, cols = vol.shape

    # Multi-frame Grayscale Word Secondary Capture (valid, widely readable by Slicer/ITK).
    sop_class = "1.2.840.10008.5.1.4.1.1.7.3"
    sop_instance = generate_uid()
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.MediaStorageSOPInstanceUID = sop_instance
    ds.file_meta.ImplementationClassUID = generate_uid()
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = sop_instance
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "OPT"
    ds.PatientName = patient_name
    ds.PatientID = patient_id
    ds.StudyDescription = study_desc
    ds.SeriesDescription = f"{patient_name} Series {series_num}".strip()
    ds.SeriesNumber = series_num
    ds.NumberOfFrames = num_frames
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SliceThickness = str(slice_thickness)
    ds.SpacingBetweenSlices = str(slice_thickness)
    ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
    if orient_vec and len(orient_vec) == 6:
        ds.ImageOrientationPatient = [float(x) for x in orient_vec]
    ds.PixelData = vol.tobytes()

    os.makedirs(os.path.dirname(str(output_path)) or os.getcwd(), exist_ok=True)
    pydicom.dcmwrite(str(output_path), ds, write_like_original=False)
    return str(output_path)


def metadata_for(oct_filename: str, companion_txt: str | Path | None = None) -> dict:
    """Combine filename + companion-.txt metadata into oct_to_dicom kwargs."""
    fm = parse_oct_filename(oct_filename)
    comp = parse_companion_file(companion_txt) if companion_txt and Path(companion_txt).exists() else {}
    desc = (fm.get("study_description", "") + " " + comp.get("eye_scanned", fm.get("laterality", ""))).strip()
    return {
        "patient_name": fm.get("patient_name", ""),
        "patient_id": fm.get("patient_id", ""),
        "study_desc": desc,
        "series_num": fm.get("series_number", 1),
    }


# ── 3) smoother: corneal-edge + column correction (DICOMSmootherSteps.py) ───
def _histeq(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        lo, hi = img.min(), img.max()
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(img, np.uint8)
    return cv2.equalizeHist(img)


def reformat_to_sagittal(volume: np.ndarray) -> np.ndarray:
    return np.transpose(volume, (2, 1, 0))


def revert_sagittal(volume_sag: np.ndarray) -> np.ndarray:
    return np.transpose(volume_sag, (2, 1, 0))


def _detect_surface_gradient(img: np.ndarray, sigma: float,
                             prior: np.ndarray | None = None, window: float | None = None) -> np.ndarray:
    # Vectorized over columns: smooth each column along depth, take the gradient, and
    # the brightest rising edge → corneal surface row. (Same result as the per-column
    # loop in the original, but ~order-of-magnitude faster.)
    # prior (per-FRAME expected depth, length = n_frames) + window (depth voxels): when both given, each
    # column's argmax is restricted to depth rows [prior[f]-window, prior[f]+window], so a spurious
    # gradient peak (e.g. a reflection above the cornea) OUTSIDE the window can't be picked. This is how
    # the fix-columns marched re-detection (redetect_surface) tracks the tilted cornea. prior=None →
    # unrestricted argmax (the original auto behaviour — the normal pipeline never passes a prior).
    sm = ndimage.gaussian_filter1d(img.astype(np.float32), sigma=sigma, axis=0)
    grad = np.gradient(sm, axis=0)                       # (depth, frames)
    if prior is None or window is None or not (float(window) > 0):
        return np.argmax(grad, axis=0)
    H, W = grad.shape
    pr = np.asarray(prior, dtype=np.float32)
    lo = np.clip(np.round(pr - float(window)), 0, H - 1).astype(np.intp)   # (frames,)
    hi = np.clip(np.round(pr + float(window)) + 1, 1, H).astype(np.intp)   # (frames,) exclusive
    rows = np.arange(H)[:, None]                          # (depth, 1)
    mask = (rows >= lo[None, :]) & (rows < hi[None, :])   # (depth, frames)
    g = np.where(mask, grad, -np.inf)
    return np.argmax(g, axis=0)


def _correct_surface(surface_y: np.ndarray, max_jump: float) -> np.ndarray:
    surface_y = surface_y.astype(float)
    n = surface_y.size
    if n < 2:
        return surface_y
    # Flag the SPIKE itself (at ANY index, including 0) by its deviation from a LOCAL MEDIAN. The old
    # predecessor-difference test (abs(y[i]-y[i-1])>max_jump) had two real defects: (1) it never tested
    # index 0, so a first-frame spike was never corrected; (2) when y[i] is a spike and y[i+1] is the good
    # value, the |y[i+1]-y[i]| jump flagged the GOOD sample (the one that "jumps back") and interpolated it
    # away instead of the spike. A robust local-median test flags the actual outlier regardless of position;
    # smooth corneal curvature stays within max_jump of its local median so it is never flagged.
    k = max(3, int(2 * round(max_jump / 5.0)) + 1)      # small odd window (~5 for the default max_jump=10)
    k = min(k, n)
    med = ndimage.median_filter(surface_y, size=k)
    outlier = np.abs(surface_y - med) > max_jump
    valid = np.where(~outlier)[0]
    if len(valid) < 2:
        return surface_y
    f = interp1d(valid, surface_y[valid], kind="cubic", fill_value="extrapolate")
    out = surface_y.copy()
    out[outlier] = f(np.where(outlier)[0])
    return out


def _smooth_median(surface_y: np.ndarray, size: int) -> np.ndarray:
    return ndimage.median_filter(surface_y, size=size)


def _advanced_edge(img: np.ndarray, p: dict, prior: np.ndarray | None = None) -> np.ndarray:
    if img.dtype != np.uint8:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    filt = cv2.bilateralFilter(img, d=int(p["d"]), sigmaColor=int(p["sigmaColor"]), sigmaSpace=int(p["sigmaSpace"]))
    # prior present (fix-columns marched re-detection) → windowed argmax around it; else unrestricted.
    raw = _detect_surface_gradient(filt, sigma=p["sigma"], prior=prior,
                                   window=(p.get("detect_window") if prior is not None else None))
    corrected = _correct_surface(raw, max_jump=p["max_jump"])
    return _smooth_median(corrected, size=int(p["median_filter_size"]))


def _intelligent_side_correction(boundary: np.ndarray, window: int, thresh: float, side_fraction: float = 0.05) -> np.ndarray:
    corrected = boundary.copy().astype(float)
    W = len(boundary)
    for x in range(int(W * side_fraction)):
        s, e = x + 1, min(W, x + window)
        if s >= e:
            continue
        med = np.median(boundary[s:e])
        mad = np.median(np.abs(boundary[s:e] - med))
        if corrected[x] < med - thresh * mad:
            corrected[x] = med
    for x in range(int(W * (1 - side_fraction)), W):
        s, e = max(0, x - window), x
        if s >= e:
            continue
        med = np.median(boundary[s:e])
        mad = np.median(np.abs(boundary[s:e] - med))
        if corrected[x] < med - thresh * mad:
            corrected[x] = med
    return corrected.astype(int)


def _side_correction_quadratic_bias(boundary: np.ndarray, quadratic: np.ndarray, window: int, thresh: float,
                                    side_fraction: float = 0.05, bias_weight: float = 0.7) -> np.ndarray:
    corrected = boundary.copy().astype(float)
    W = len(boundary)
    for x in range(int(W * side_fraction)):
        s, e = x + 1, min(W, x + window)
        if s >= e:
            continue
        cand = bias_weight * quadratic[x] + (1 - bias_weight) * np.median(boundary[s:e])
        if abs(boundary[x] - quadratic[x]) > thresh:
            corrected[x] = cand
    for x in range(int(W * (1 - side_fraction)), W):
        s, e = max(0, x - window), x
        if s >= e:
            continue
        cand = bias_weight * quadratic[x] + (1 - bias_weight) * np.median(boundary[s:e])
        if abs(boundary[x] - quadratic[x]) > thresh:
            corrected[x] = cand
    return corrected.astype(int)


def _fit_quadratic_ransac(edge: np.ndarray, residual_threshold: float) -> np.ndarray:
    """Faithful to DICOMSmootherSteps.fit_quadratic_ransac: sklearn RANSAC quadratic fit of the
    corneal boundary (degree-2 polynomial, min_samples=0.3, fixed seed)."""
    x = np.arange(len(edge)).reshape(-1, 1)
    try:
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        ransac = RANSACRegressor(estimator=model, min_samples=0.3,
                                 residual_threshold=residual_threshold, random_state=42)
        ransac.fit(x, edge)
        return ransac.predict(x)
    except Exception:  # noqa: BLE001
        # RANSAC found no valid consensus (degenerate/noisy edge, e.g. an artifacted scan) → plain
        # degree-2 least squares so the scan still preprocesses instead of crashing the whole run.
        xv = np.arange(len(edge))
        if len(edge) >= 3:
            return np.polyval(np.polyfit(xv, np.asarray(edge, float), 2), xv)
        return np.asarray(edge, float)


def _clip_mask(sl: np.ndarray, edge: np.ndarray, p: dict) -> np.ndarray:
    """Per-frame boolean: columns where the corneal apex is ABOVE the frame (clipped). True iff the
    detected edge is pinned near the top AND the top band is bright tissue with no dark air gap. The raw
    slice sl=(depth, frames) is REQUIRED — the top-band brightness cannot be derived from `edge` alone.
    A normal in-frame dome has a dark gap above the surface (top/colmax small), so it never triggers."""
    top_rows = max(1, int(p.get("clip_top_rows", 5)))
    top = np.asarray(sl[:top_rows], dtype=np.float64).mean(axis=0)
    colmax = np.asarray(sl, dtype=np.float64).max(axis=0)
    colmax[colmax <= 0] = 1.0
    e = np.asarray(edge, dtype=np.float64)
    # 0 ≤ edge < floor: a clipped apex is pinned just BELOW the frame top. A NEGATIVE edge means the detector
    # already ran OFF-frame (a limbus/edge-of-volume slice where _correct_surface cubic-extrapolated past 0) —
    # that is NOT a central clipped dome and must not be treated as one.
    mask = (e >= 0.0) & (e < float(p.get("clip_edge_floor", 8.0))) & \
           (top / colmax > float(p.get("clip_top_frac", 0.5)))
    # Consolidate small gaps: the DP detector can place 1-3 columns of a clipped apex a few px DEEPER than the
    # top floor, fragmenting the run (so those columns leak into the dome fit as outliers → the inlier gate
    # fails and the clip is missed). A binary closing fills internal gaps ≤ clip_close_gap WITHOUT growing the
    # outer extent — restoring the contiguous central run the gates expect. No-op for the already-contiguous
    # legacy mask and for a normal scan's (empty/sparse) mask, so it can't create a false clip.
    gap = int(p.get("clip_close_gap", 4))
    if gap > 0 and mask.any():
        mask = ndimage.binary_closing(mask, structure=np.ones(2 * gap + 1, dtype=bool))
    return mask


def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of True in a 1-D boolean array (a real dome apex clips contiguously)."""
    best = run = 0
    for v in np.asarray(mask):
        run = run + 1 if v else 0
        if run > best:
            best = run
    return int(best)


def _resolve_clip(edge: np.ndarray, sl: np.ndarray, residual_threshold: float, p: dict):
    """Detect a clipped corneal apex in one sagittal slice and, if CONFIRMED, return the EXTRAPOLATING
    quadratic fit (fit to the in-frame flank columns only, predicted across the clipped band so the apex
    may go <0) plus the clipped column indices. Returns (clip_cols[int], clip_fit) or (empty, None) when
    the slice is not a confirmed central clip — every other case (incl. limbus/edge-of-volume false
    positives) falls back to the legacy per-slice fit, so well-detected scans are unchanged.

    Six gates (all must hold): (1) ≥clip_min_cols clipped columns; (2) a contiguous run ≥clip_min_run;
    (3) the clipped band is CENTRAL (centroid in 20–80% of frames — a dome apex, not an edge); (4) valid
    in-frame columns on BOTH sides; (5) enough valid columns to fit; (6) FIT-QUALITY: the masked-valid
    parabola's x² coef is in the corneal curvature band AND its flank-inlier RMS + inlier fraction look
    like a real dome (this is the decisive discriminator that rejects the steep limbus failure mode)."""
    edge = np.asarray(edge, dtype=np.float64)
    n = edge.size
    clip = _clip_mask(sl, edge, p)
    if int(clip.sum()) < int(p.get("clip_min_cols", 6)) or _longest_run(clip) < int(p.get("clip_min_run", 5)):
        return np.array([], dtype=int), None
    cols = np.where(clip)[0]
    centroid = float(cols.mean())
    if not (0.2 * n <= centroid <= 0.8 * n):                         # gate 3: central dome, not an edge/limbus
        return np.array([], dtype=int), None
    valid = ~clip
    lo, hi = int(cols.min()), int(cols.max())
    min_flank = int(p.get("clip_min_flank", 5))
    # gate 4: a real central clip has a SUBSTANTIAL in-frame flank on BOTH sides to anchor the parabola. A
    # 1-2 column flank (a one-sided limbus/edge-of-volume clip) extrapolates to garbage — leave those to the
    # manual fix-columns tool rather than fabricate an apex.
    if int(valid[:lo].sum()) < min_flank or int(valid[hi + 1:].sum()) < min_flank:
        return np.array([], dtype=int), None
    if int(valid.sum()) < max(3, int(np.ceil(0.3 * n)) + 1):       # gate 5: enough valid columns to fit
        return np.array([], dtype=int), None
    x = np.arange(n, dtype=np.float64)
    try:
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        ransac = RANSACRegressor(estimator=model, min_samples=0.3,
                                 residual_threshold=residual_threshold, random_state=42)
        ransac.fit(x[valid].reshape(-1, 1), edge[valid])
        fit = ransac.predict(x.reshape(-1, 1))                       # predict over ALL columns → extrapolate
        a = float(ransac.estimator_.named_steps["linearregression"].coef_[2])   # x² coefficient (curvature)
        inlier = ransac.inlier_mask_
        inlier_frac = float(inlier.mean()) if inlier.size else 0.0
        flank_rms = (float(np.sqrt(np.mean((edge[valid][inlier] - fit[valid][inlier]) ** 2)))
                     if inlier.any() else np.inf)
    except Exception:  # noqa: BLE001
        return np.array([], dtype=int), None
    if not (float(p.get("clip_a_min", 0.008)) <= a <= float(p.get("clip_a_max", 0.05))     # gate 6: real dome
            and flank_rms <= float(p.get("clip_flank_rms", 8.0))
            and inlier_frac >= float(p.get("clip_inlier_frac", 0.6))):
        return np.array([], dtype=int), None
    if not np.all(np.isfinite(fit)) or float(np.min(fit)) < float(p.get("clip_apex_floor", -60.0)):  # gate 7: sane apex
        return np.array([], dtype=int), None
    return cols.astype(int), fit


def _extrapolate_fit(edge: np.ndarray, clip_cols: np.ndarray, residual_threshold: float):
    """Re-fit the extrapolating parabola for a KNOWN set of clipped columns — used to CARRY a clip forward to
    iteration passes ≥1, which detect on a warped+filled volume where the 'no air gap' clip invariant no
    longer holds (so they must NOT re-detect). The column set is trusted from pass 0; no gates here, just a
    RANSAC fit on the in-frame columns predicted across the clip. Returns the fit over all columns or None."""
    edge = np.asarray(edge, dtype=np.float64); n = edge.size
    cc = np.asarray(clip_cols, dtype=int)
    if cc.size == 0:
        return None
    valid = np.ones(n, dtype=bool); valid[cc[(cc >= 0) & (cc < n)]] = False
    if int(valid.sum()) < 3:
        return None
    x = np.arange(n, dtype=np.float64)
    try:
        model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        ransac = RANSACRegressor(estimator=model, min_samples=0.3,
                                 residual_threshold=residual_threshold, random_state=42)
        ransac.fit(x[valid].reshape(-1, 1), edge[valid])
        return ransac.predict(x.reshape(-1, 1))
    except Exception:  # noqa: BLE001
        try:
            return np.polyval(np.polyfit(x[valid], edge[valid], 2), x)
        except Exception:  # noqa: BLE001
            return None


def _warp_by_displacement(img: np.ndarray, displacement: np.ndarray, subpixel: bool = False) -> np.ndarray:
    H, W = img.shape
    if subpixel:
        # SUB-PIXEL warp (subpixel_warp): shift each column by the FRACTIONAL displacement via linear
        # interpolation in depth, instead of truncating to int. The int-truncate warp quantises the
        # sub-pixel-detected surface into a 1-px lateral STAIRCASE (the "ripples" the user sees at zoom); the
        # fractional shift removes it, giving a smooth anterior boundary. Interpolation is confined to the
        # 1-D depth shift (<1 px), so lateral/frame crispness is untouched. out[r] = img[r − s] (linear).
        disp = np.asarray(displacement, dtype=np.float64)
        rows = np.arange(H, dtype=np.float64)
        src = rows[:, None] - disp[None, :]                       # (H, W) source row for each output (row, col)
        lo = np.floor(src).astype(np.int64)
        frac = src - lo
        cols = np.arange(W)[None, :]
        m0 = (lo >= 0) & (lo <= H - 1)
        m1 = (lo + 1 >= 0) & (lo + 1 <= H - 1)
        f = img.astype(np.float64)
        v0 = np.where(m0, f[np.clip(lo, 0, H - 1), cols], 0.0)
        v1 = np.where(m1, f[np.clip(lo + 1, 0, H - 1), cols], 0.0)
        out = v0 * (1.0 - frac) + v1 * frac
        out[~(m0 | m1)] = 0.0                                     # fully out-of-range rows → 0 (vacated, like the int path)
        if np.issubdtype(img.dtype, np.integer):
            out = np.rint(out)
        return out.astype(img.dtype)
    warped = np.zeros_like(img)
    for x in range(W):
        shift = int(displacement[x])   # truncate toward zero (faithful to warp_image_by_edge)
        if shift > 0:
            nh = H - shift
            if nh > 0:
                warped[shift:, x] = img[:nh, x]
        elif shift < 0:
            nh = H + shift
            if nh > 0:
                warped[:nh, x] = img[-shift:, x]
        else:
            warped[:, x] = img[:, x]
    return warped


def _fill_cols_along_rows(img: np.ndarray) -> np.ndarray:
    """In a sagittal slice (rows=depth, cols=frames), replace each column's LEADING/TRAILING zero run
    (the black padding a prior column-warp left) with the nearest real pixel. Used between iterative
    passes so the edge detector can't lock onto the black-band→tissue edge (which caused 100–360px
    runaway shifts on pass 2+). Pure edge-replication; only touches padding, never real tissue."""
    H, W = img.shape
    out = img.copy()
    nz = img != 0
    has = nz.any(axis=0)
    firstnz = np.argmax(nz, axis=0)
    lastnz = H - 1 - np.argmax(nz[::-1], axis=0)
    for x in range(W):
        if not has[x]:
            continue
        f, l = int(firstnz[x]), int(lastnz[x])
        if f > 0:
            out[:f, x] = img[f, x]
        if l < H - 1:
            out[l + 1:, x] = img[l, x]
    return out


def _fill_black_bands(volume: np.ndarray) -> np.ndarray:
    """Fill the warp's black padding throughout a (frames, depth, lateral) volume, in the SAME
    sagittal domain the warp operates on, so a re-fed (already-corrected) volume detects cleanly.
    Operates on a COPY — reformat_to_sagittal is a transpose VIEW, so writing through it would mutate
    the caller's stored chain volume (corrupting the kept pass + its previews)."""
    sag = reformat_to_sagittal(volume).copy()
    for i in range(sag.shape[0]):
        sag[i] = _fill_cols_along_rows(sag[i])
    return np.ascontiguousarray(revert_sagittal(sag))


def _dp_min_cost_path(score: np.ndarray, p: dict) -> np.ndarray:
    """Given a per-(depth, frame) score (higher = more boundary-like, already per-frame normalised), find the
    globally-smoothest maximum-score path (depth step ≤ dp_max_jump between adjacent frames) by dynamic
    programming, then a 3-point parabolic sub-voxel refine on the score profile. Returns the per-frame depth
    (float). Shared verbatim by the ANTERIOR (_detect_surface_dp) and POSTERIOR (_detect_bottom_edge)
    detectors so both trace a speckle-robust, jitter-free boundary identically."""
    score = np.asarray(score, dtype=np.float32)
    D, F = score.shape
    cost = (-score).astype(np.float32)
    maxj = max(1, min(int(p.get("dp_max_jump", 10)), D - 1))   # clamp to depth so offsets stay in-range (tiny D)
    offs = np.arange(-maxj, maxj + 1)
    # SMOOTHNESS PENALTY (dp_smooth_weight): the hard max-jump cap alone leaves ANY step ≤ maxj "free", so at a
    # shoulder the path hops maxj px between frames onto a deeper coherent layer → the apex/flank V-notch (CS001
    # OS3). Penalising the step MAGNITUDE (λ·|step|, score is per-frame ∈[0,1]) makes the DP prefer a smooth
    # descent, only taking a big step when the score gain justifies it. 0 = OFF (byte-identical hard-cap-only).
    sw = float(p.get("dp_smooth_weight", 0.0) or 0.0)
    step_pen = (sw * np.abs(offs)).astype(np.float32)[:, None] if sw > 0 else None
    dp = cost[:, 0].copy()
    back = np.empty((D, F), dtype=np.int32)
    for f in range(1, F):
        cand = np.full((offs.size, D), np.inf, dtype=np.float32)  # cand[k,d] = dp_prev[d+offs[k]]
        for k, o in enumerate(offs):
            if o < 0:
                cand[k, -o:] = dp[:D + o]
            elif o > 0:
                cand[k, :D - o] = dp[o:]
            else:
                cand[k, :] = dp
        if step_pen is not None:
            cand = cand + step_pen                                # add λ·|step| to every transition (broadcast over depth)
        kbest = np.argmin(cand, axis=0)
        dp = cand[kbest, np.arange(D)] + cost[:, f]
        back[:, f] = np.arange(D) + offs[kbest]
    surf = np.empty(F, dtype=np.int32)
    surf[F - 1] = int(np.argmin(dp))
    for f in range(F - 1, 0, -1):
        surf[f - 1] = back[surf[f], f]
    # 3-point parabolic sub-voxel refine on the score profile at each frame's chosen depth
    out = surf.astype(np.float32)
    fcols = np.arange(F)
    d0 = surf
    mid = np.clip(d0, 1, D - 2)
    a = score[mid - 1, fcols]; b = score[mid, fcols]; c = score[mid + 1, fcols]
    denom = (a - 2.0 * b + c)
    safe = np.abs(denom) > 1e-6
    shift = np.where(safe, 0.5 * (a - c) / np.where(safe, denom, 1.0), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    interior = (d0 >= 1) & (d0 <= D - 2)
    out[interior] = (mid + shift)[interior]
    return out


def _detect_bottom_edge(slice_img: np.ndarray, p: dict, prior: np.ndarray | None = None) -> np.ndarray:
    """POSTERIOR (bottom) corneal-edge detector for one sagittal slice (depth, frames), depth 0 = TOP — the
    MIRROR of _detect_surface_dp. Returns the per-frame posterior depth (float, sub-voxel).

    Where the anterior is a dark→bright gradient gated by bright tissue BELOW, the posterior is a BRIGHT→DARK
    gradient (intensity falls downward, cornea→aqueous) GATED by bright tissue ABOVE it (the corneal stroma
    sits above the dark anterior chamber). Same anisotropic despeckle, per-frame normalisation and DP smooth
    path. Used by the surface-crop reconstruction to GUIDE frames whose apex is cropped (no anterior surface)
    by their still-visible bottom edge. A `prior` restricts the search to ±detect_window around it."""
    img = ndimage.gaussian_filter(slice_img.astype(np.float32),
                                  sigma=(float(p.get("dp_sigma_depth", 3.0)), float(p.get("dp_sigma_frame", 1.2))))
    D, F = img.shape
    gy = np.gradient(img, axis=0)
    grad = np.clip(-gy, 0.0, None)                                 # -ve gradient = intensity falls downward (bright→dark)
    med = float(np.median(img))
    bw = max(2, int(p.get("dp_below", 24)))
    # mean of ~bw px ABOVE each point = the anterior's "below" filter on the depth-reversed slice (origin valid
    # by construction), flipped back — so the posterior gate mirrors the anterior exactly.
    above = ndimage.uniform_filter1d(img[::-1], size=bw, axis=0, origin=-(bw // 2))[::-1]
    score = grad * np.maximum(above - med, 0.0)                    # strong fall AND bright tissue above
    win = p.get("detect_window") if prior is not None else None
    if prior is not None and win is not None and float(win) > 0:
        pr = np.asarray(prior, dtype=np.float32)
        rows = np.arange(D)[:, None]
        mask = (rows >= (pr - float(win))[None, :]) & (rows <= (pr + float(win))[None, :])
        score = np.where(mask, score, 0.0)
    score = score / (score.max(axis=0, keepdims=True) + 1e-6)
    return _dp_min_cost_path(score, p)


def _detect_surface_dp(slice_img: np.ndarray, p: dict, prior: np.ndarray | None = None) -> np.ndarray:
    """NATIVE dynamic-programming anterior-surface detector (see DEFAULT_PARAMS['detector']).
    slice_img = (depth, frames), depth 0 = TOP. Returns the per-frame surface depth (float, sub-voxel).

    1) Despeckle with an ANISOTROPIC Gaussian (heavier along depth, where OCT speckle is fine-grained;
       lighter along frames, to keep the real lateral corneal shape).
    2) Score each (depth, frame) as a candidate anterior surface = a dark→bright vertical gradient GATED by
       the mean brightness just BELOW it (the cornea is bright tissue under a dark gap), so the score is high
       only at the true epithelial surface — not at internal layers or random top speckle. Column-normalised
       so a dim peripheral frame still yields a confident pick.
    3) Dynamic programming finds the globally-smoothest maximum-score path (depth step ≤ dp_max_jump between
       adjacent frames) — the speckle-robust, jitter-free surface. Then a 3-point parabolic sub-voxel refine.

    A `prior` (per-frame expected depth) restricts the search to ±detect_window around it (used by the
    fix-columns marched re-detection); prior=None is the normal global auto detection."""
    img = ndimage.gaussian_filter(slice_img.astype(np.float32),
                                  sigma=(float(p.get("dp_sigma_depth", 3.0)), float(p.get("dp_sigma_frame", 1.2))))
    D, F = img.shape
    gy = np.gradient(img, axis=0)                                  # +ve = intensity rises downward (dark→bright)
    np.clip(gy, 0.0, None, out=gy)
    med = float(np.median(img))
    bw = max(2, int(p.get("dp_below", 24)))
    below = ndimage.uniform_filter1d(img, size=bw, axis=0, origin=-(bw // 2))   # mean of ~bw px BELOW each point
    # ABOVE-DARK gate (dp_above_gate): the TRUE anterior surface has bright tissue BELOW and DARK air ABOVE; a
    # deeper internal/second-reflection layer (e.g. at a specular apex, CS001 OS3) has bright tissue on BOTH
    # sides. Gating on the boundary CONTRAST (below − above) instead of (below − med) suppresses the deeper layer
    # (small contrast) and keeps the epithelium (large contrast) → the DP no longer dives to the deeper layer and
    # steps at the apex. `above` = mean of ~bw px ABOVE each point. Default OFF → byte-identical (below − med).
    if bool(p.get("dp_above_gate", False)):
        # mean of ~bw px ABOVE each point = the "below" filter on the depth-reversed slice, flipped back (same
        # trick as _detect_bottom_edge, so the origin stays valid).
        above = ndimage.uniform_filter1d(img[::-1], size=bw, axis=0, origin=-(bw // 2))[::-1]
        score = gy * np.maximum(below - above, 0.0)               # dark→bright edge with bright BELOW and DARK ABOVE
    else:
        score = gy * np.maximum(below - med, 0.0)                  # strong edge AND bright tissue below (legacy)
    # restrict to a window around a prior, if supplied (fix-columns tilt-aware re-detection)
    win = p.get("detect_window") if prior is not None else None
    no_signal = None
    if prior is not None and win is not None and float(win) > 0:
        pr = np.asarray(prior, dtype=np.float32)
        rows = np.arange(D)[:, None]
        mask = (rows >= (pr - float(win))[None, :]) & (rows <= (pr + float(win))[None, :])
        score = np.where(mask, score, 0.0)
        no_signal = score.max(axis=0) <= 0.0                      # window holds NO boundary signal → keep the prior
    score = score / (score.max(axis=0, keepdims=True) + 1e-6)     # per-frame normalise → confident dim columns
    out = _dp_min_cost_path(score, p)
    if no_signal is not None and no_signal.any():
        # without a boundary in the window the DP cost is all-zero and argmin ties to row 0 (a false top-edge);
        # fall back to the prior there instead of collapsing the frame to the top.
        out = np.asarray(out, dtype=np.float32).copy()
        out[no_signal] = np.asarray(prior, dtype=np.float32)[no_signal]
    return out


def _legacy_surface(slice_img: np.ndarray, p: dict, prior: np.ndarray | None = None) -> np.ndarray:
    """The ORIGINAL ('old method') per-slice anterior surface: {hist-eq, raw} gradient-argmax, the better
    RANSAC-quadratic of the two, then a side-correction bias. RANSAC fits a smooth corneal dome and rejects a
    localized internal bright region (e.g. a hyper-reflective scar) as outliers, so it reliably holds the TRUE
    first surface where the DP path can be lured DEEPER onto the scar. Used both as the legacy detector and as
    the DP scar-guard's robust vicinity anchor."""
    edge_h = _advanced_edge(_histeq(slice_img), p, prior=prior)
    edge_r = _advanced_edge(slice_img, p, prior=prior)
    q_h = _fit_quadratic_ransac(edge_h, p["residual_threshold"])
    q_r = _fit_quadratic_ransac(edge_r, p["residual_threshold"])
    chosen = edge_h if np.sum((edge_h - q_h) ** 2) <= np.sum((edge_r - q_r) ** 2) else edge_r
    quad_prelim = _fit_quadratic_ransac(chosen, p["residual_threshold"])
    return _side_correction_quadratic_bias(chosen, quad_prelim,
                                           window=int(p["side_window"]), thresh=p["side_threshold_factor"])


def _above_brightness(img: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """Mean brightness in the ~8px band just ABOVE each frame's edge, normalised by the column max. LOW = dark
    air above (a true anterior epithelial surface); HIGH = bright tissue above (the edge sits under cornea / on
    an internal scar). The DP scar-guard uses this to confirm a pull-back actually lands on a true surface."""
    D, F = img.shape
    e = np.clip(np.round(np.asarray(edge)).astype(int), 0, D - 1)
    fc = np.arange(F)
    band = np.mean([img[np.clip(e - k, 0, D - 1), fc] for k in range(2, 10)], axis=0)
    cmax = np.maximum(img.max(axis=0), 1.0)
    return band / cmax


def _dp_scar_guard(slice_img: np.ndarray, dp_edge: np.ndarray, p: dict) -> np.ndarray:
    """Keep the DP anterior edge in the VICINITY of the legacy ('old method') surface so it can't lock onto a
    bright internal structure (a hyper-reflective SCAR) DEEPER than the true epithelial boundary.

    The DP score = dark->bright gradient x bright-tissue-below; when the true surface is dim and a scar inside
    the cornea is very bright, the scar's upper boundary outscores the surface and the DP smooth path follows it
    (verified on CS021 OD: DP sat ~30px deeper, onto the scar; legacy held the surface). The legacy RANSAC-
    quadratic is robust to this (rejects the scar as outliers). So: where the DP edge sits >dp_scar_tol px
    DEEPER than the legacy edge over a contiguous run (the scar-lock signature), RE-RUN the DP CONFINED to
    +/-dp_scar_window of the legacy surface — it then tracks the true (first) surface within the band, keeping
    DP's sub-voxel/smoothness strengths but excluding the out-of-band scar.

    ONE-SIDED (only catches DP diving DEEPER) and run-GATED, so on a normal scan — where the validated DP edge
    already sits at/above legacy — it is a strict no-op (returns dp_edge unchanged). dp_scar_guard=False disables.

    CLIP-SAFE: the trigger requires the legacy edge to be a VALID IN-FRAME surface (leg > clip_edge_floor). On a
    clipped-apex scan the legacy RANSAC-quadratic EXTRAPOLATES the apex ABOVE the frame (leg < 0), so a DP edge
    pinned near the top would otherwise read as ">tol deeper than legacy" and fire spuriously — fighting the
    clip-handling. Excluding leg<=floor frames makes the guard a no-op on clipped apexes (confirmed on
    CS005/CS008/CS020) while still firing on a genuine scar (true surface in-frame, DP dives onto the bright
    scar below it). Only the genuine scar frames are replaced (free DP kept everywhere else → no slice-wide
    side-effects on clip/limbus columns)."""
    leg = _legacy_surface(slice_img, p).astype(np.float64)
    dp = np.asarray(dp_edge, dtype=np.float64)
    D = int(slice_img.shape[0])
    floor = float(p.get("clip_edge_floor", 8.0))
    # genuine scar-lock: DP sits >tol DEEPER than a VALID IN-FRAME legacy surface (clip apexes have leg<=floor)
    stray = (dp - leg > float(p.get("dp_scar_tol", 18.0))) & (leg > floor)
    if _longest_run(stray) < int(p.get("dp_scar_min_run", 6)):
        return dp_edge                                    # no contiguous scar-lock run → DP is trusted (no-op)
    win = float(p.get("dp_scar_window", 12.0))
    prior = np.clip(leg, 0.0, D - 1).astype(np.float32)   # clamp so the windowed search box stays in-frame
    windowed = np.asarray(_detect_surface_dp(slice_img, {**p, "detect_window": win}, prior=prior), dtype=np.float64)
    # SELF-VALIDATE before adopting: only pull a frame back if the windowed (shallower) position genuinely has
    # DARKER tissue ABOVE it (dark air over the epithelium = the true anterior surface) than the current deep DP
    # edge (which sits UNDER bright cornea / on a scar). This keeps the guard from moving a genuinely irregular-
    # but-correct DP edge onto a wrong legacy fit, and removes the residual non-corrective changes seen on
    # clipped scans (the windowed position there is not darker-above, so it is not adopted).
    smooth = ndimage.gaussian_filter(slice_img.astype(np.float32),
                                     sigma=(float(p.get("dp_sigma_depth", 3.0)), 0.6))
    darker = _above_brightness(smooth, windowed) < _above_brightness(smooth, dp) - float(p.get("dp_scar_darker_margin", 0.05))
    adopt = stray & darker
    if not adopt.any():
        return dp_edge
    out = dp.copy()
    out[adopt] = windowed[adopt]                          # pull back ONLY the confirmed scar-lock frames
    return out.astype(np.float32)


def _reject_specular_spike(edge: np.ndarray, p: dict) -> np.ndarray:
    """Remove a THIN specular spike from a detected anterior edge (depth 0 = TOP, so a spike rises = the edge
    value DROPS below the smooth trend). A narrow ultra-bright vertical reflection at the apex lures the DP path
    up onto it; here we detect the resulting narrow upward excursion and interpolate the smooth dome across it.

    A laterally-robust MEDIAN trend (window > spike width) ignores the spike, so trend−edge is large only on the
    spike frames. Frames where edge sits > spec_spike_min_height ABOVE the trend, in a CONTIGUOUS run no wider
    than spec_spike_max_width, are replaced by a straight-line interpolation between the good frames bracketing
    the run. One-sided (upward only) + narrow-run gated → a wide/steep real apex and a clean apex are no-ops."""
    e = np.asarray(edge, dtype=np.float64)
    F = e.size
    win = int(p.get("spec_spike_trend_win", 15))
    if F < 5 or win < 3:
        return edge
    win = win if win % 2 else win + 1                       # odd kernel for median_filter
    trend = ndimage.median_filter(e, size=min(win, F if F % 2 else F - 1), mode="nearest")
    above = trend - e                                        # > 0 where the edge is shallower (higher) than trend
    cand = above > float(p.get("spec_spike_min_height", 8.0))
    if not cand.any():
        return edge                                         # no upward excursion anywhere → no-op
    maxw = int(p.get("spec_spike_max_width", 4))
    out = e.copy()
    changed = False
    f = 0
    while f < F:
        if not cand[f]:
            f += 1
            continue
        g = f
        while g < F and cand[g]:
            g += 1                                          # [f, g) is a contiguous candidate run
        if (g - f) <= maxw:                                 # NARROW upward run = a specular spike → interpolate
            lo, hi = f - 1, g                               # bracketing good frames
            if lo >= 0 and hi < F:
                out[f:g] = np.interp(np.arange(f, g), [lo, hi], [e[lo], e[hi]])
                changed = True
            elif lo >= 0:                                   # run touches the right edge → hold the left good value
                out[f:g] = e[lo]; changed = True
            elif hi < F:                                    # run touches the left edge → hold the right good value
                out[f:g] = e[hi]; changed = True
        f = g
    return out.astype(np.float32) if changed else edge


def _merged_side_edge(slice_img: np.ndarray, p: dict, prior: np.ndarray | None = None) -> np.ndarray:
    """The per-slice corrected anterior boundary. Default detector = the native DP path (_detect_surface_dp,
    matches a manual trace so AUTO preprocessing needs minimal correction); set params['detector']='legacy'
    for the original {hist-eq, raw} gradient-argmax + RANSAC-quadratic choice. When a prior surface is supplied
    (fix-columns marched re-detection) the underlying detection is windowed around it.

    The DP path is wrapped by the SCAR-GUARD (_dp_scar_guard): a cross-check against the legacy surface that
    pulls the DP edge back into the legacy's vicinity wherever DP has dived DEEPER onto a bright internal scar.
    Skipped when an external prior is supplied (the fix-columns re-detect carries its own user-seeded surface)."""
    if str(p.get("detector", "dp")).lower() != "legacy":
        dp = _detect_surface_dp(slice_img, p, prior=prior)
        if prior is None and bool(p.get("dp_scar_guard", True)):
            dp = _dp_scar_guard(slice_img, dp, p)
        # FIX specular: drop a thin bright vertical specular spike the DP climbed onto (auto detection only)
        if prior is None and bool(p.get("spec_spike_reject", True)):
            dp = _reject_specular_spike(dp, p)
        return dp
    return _legacy_surface(slice_img, p, prior=prior)


def _edge_worker(packed):
    sl, p = packed
    return _merged_side_edge(sl, p)


def _redetect_one_slice(sl: np.ndarray, prior: np.ndarray, window: float, p: dict,
                        light: bool = True) -> np.ndarray:
    """Re-detect one sagittal slice's corneal surface within ±window of a per-frame `prior`. sl=(depth,
    frames). light=False (seed slices) uses the full robust detector (_merged_side_edge: hist-eq/raw choice
    + side-correction). light=True (the MARCH, called for every slice) uses a fast windowed gradient argmax
    + outlier/median cleanup — no bilateral / hist-eq / double-RANSAC — which is reliable because the tight
    window around the resolved neighbour already excludes confounders (and is ~10x faster, so a 513-slice
    march finishes in seconds instead of minutes)."""
    pr = np.asarray(prior, dtype=np.float32)
    if not light:
        return _merged_side_edge(sl, {**p, "detect_window": float(window)}, prior=pr)
    raw = _detect_surface_gradient(sl, sigma=float(p["sigma"]), prior=pr, window=float(window))
    corrected = _correct_surface(raw, max_jump=float(p["max_jump"]))
    return _smooth_median(corrected, size=int(p["median_filter_size"]))


def _extrapolate_boundary_edges(edges: np.ndarray, p: dict) -> np.ndarray:
    """Replace the FIRST/LAST `boundary_extrap_nb` frames' per-slice surface with a robust frame-direction LINEAR
    extrapolation from the adjacent interior frames, then smooth the boundary frames across slices. The low-signal
    acquisition-edge frames detect noisily; the raw corneal shape is smooth across frames, so the interior
    extrapolation matches the true band position and gives the warp a smooth, cross-slice-consistent surface there
    (removes the jagged edge B-scans). edges = (n_lateral, n_frames), depth 0 = top. Strict no-op when the surface
    is already smooth (extrapolation ≈ detection) or nb<=0."""
    nb = int(p.get("boundary_extrap_nb", 4) or 0)
    if nb <= 0:
        return edges
    e = np.asarray(edges, dtype=np.float64).copy()
    if e.ndim != 2:
        return edges
    L, F = e.shape
    span = int(p.get("boundary_extrap_span", 18))
    if F < 2 * nb + 4 or L < 8:
        return edges
    span = min(span, (F - 2 * nb) // 2) if F - 2 * nb > 0 else span
    deg = int(p.get("boundary_extrap_degree", 2))                # QUADRATIC follows the corneal CURVATURE (a linear
    #   tangent puts the edge frames at the WRONG ANGLE vs the dome — the marked CS002 OS(2)/(3) defect)
    dev = float(p.get("boundary_extrap_max_dev", 25.0))          # clamp |extrap − nearest interior| so a spurious
    #   interior curvature can't OVERSHOOT the few extrapolated frames off the cornea
    for l in range(L):
        a = e[l]; vld = a > 1.0
        ff = [f for f in range(nb, nb + span) if vld[f]]
        if len(ff) >= max(6, deg + 3):
            co = np.polyfit(ff, a[ff], deg)
            anchor = a[nb] if vld[nb] else np.polyval(co, nb)
            for f in range(nb):
                if vld[f]:
                    e[l, f] = float(np.clip(np.polyval(co, f), anchor - dev, anchor + dev))
        lf = [f for f in range(F - nb - span, F - nb) if vld[f]]
        if len(lf) >= max(6, deg + 3):
            co = np.polyfit(lf, a[lf], deg)
            anchor = a[F - nb - 1] if vld[F - nb - 1] else np.polyval(co, F - nb - 1)
            for f in range(F - nb, F):
                if vld[f]:
                    e[l, f] = float(np.clip(np.polyval(co, f), anchor - dev, anchor + dev))
    sig = float(p.get("boundary_extrap_lat_sigma", 6.0) or 0.0)
    if sig > 0:
        xs = np.arange(L)
        for f in list(range(nb)) + list(range(F - nb, F)):
            col = e[:, f]; m = col > 1.0
            if int(m.sum()) > 20:
                filled = np.interp(xs, xs[m], col[m])
                sm = ndimage.gaussian_filter1d(filled, sigma=sig, mode="nearest")
                e[m, f] = sm[m]
    return e.astype(edges.dtype)


def _reject_apex_lateral_spike(edges: np.ndarray, p: dict) -> np.ndarray:
    """FIX apexspec: remove a narrow apex specular streak from the ASSEMBLED (lateral, frame) surface by making
    each FRAME column laterally smooth where a narrow up-spike (the DP climbing the streak) sits above a
    laterally-robust median trend. edges = (n_lateral, n_frames), depth 0 = TOP (a spike rising = value DROPS).

    Per frame column: a MEDIAN trend over `apex_lat_trend_win` laterals (wider than the streak) follows the true
    dome and ignores the spike. Contiguous lateral runs where (trend − surface) > apex_lat_min_height (surface is
    SHALLOWER = climbed the streak), no wider than apex_lat_max_width, are the specular band; a small pad
    (apex_lat_pad) around each is reset to the trend (absorbing the jitter the streak induces on its flanks). The
    kept-away specular pixels are left in the volume ABOVE the corrected surface (per the user's decision). Strictly
    one-sided (upward-triggered) + narrow-run gated, so a clean apex (surface already on the lateral trend) and a
    genuinely wide/steep dome are strict no-ops. apex_lateral_reject=False disables."""
    if not bool(p.get("apex_lateral_reject", True)):
        return edges
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2:
        return edges
    L, F = e.shape
    win = int(p.get("apex_lat_trend_win", 41))
    if L < 5 or win < 3:
        return edges
    win = win if win % 2 else win + 1
    ksize = min(win, L if L % 2 else L - 1)
    min_h = float(p.get("apex_lat_min_height", 8.0))
    maxw = int(p.get("apex_lat_max_width", 18))
    pad = int(p.get("apex_lat_pad", 4))
    # FIX apexnotch (CS001 OSbase/OS3 apex): the apex defect the user marked is NOT the DP climbing a streak UP;
    # it is scattered per-sagittal-slice DOWNWARD jitter — a handful of laterals lock ~15px too DEEP (into the
    # stroma) at the apex frame, on an otherwise-smooth lateral surface. The original reject was upward-only
    # (surface shallower) so it never fired (measured 0.0px change on those apex notches). Add a symmetric
    # DOWNWARD branch: narrow lateral runs sitting > down_min_h px DEEPER than the robust median trend are the
    # same specular/jitter class and are reset to the trend. Gated separately (apex_lat_reject_down) and with its
    # own (slightly higher) height so a broad genuine curvature / a real posterior dip is a strict no-op; the
    # NARROW-run width gate keeps a sustained steep limbus flank (a long run) untouched.
    down = bool(p.get("apex_lat_reject_down", True))
    down_min_h = float(p.get("apex_lat_down_min_height", 10.0))
    recov = float(p.get("apex_lat_dip_recover", 5.0))

    def _narrow_run_band(cand: np.ndarray) -> np.ndarray:
        """Boolean mask of the NARROW (<= maxw) contiguous True-runs of `cand`, padded by `pad`."""
        b = np.zeros(L, dtype=bool)
        i = 0
        while i < L:
            if not cand[i]:
                i += 1
                continue
            j = i
            while j < L and cand[j]:
                j += 1                                           # [i, j) contiguous run
            if (j - i) <= maxw:                                  # NARROW lateral run = specular/jitter → reset to trend
                b[max(0, i - pad):min(L, j + pad)] = True
            i = j
        return b

    def _local_dip_band(cand: np.ndarray, dev: np.ndarray) -> np.ndarray:
        """Like _narrow_run_band but for DOWN-notches (dev = surface − trend, +ve = deeper): accept a run ONLY if
        the surface RECOVERS to within `recov` px of the trend on BOTH lateral sides within a short window — a
        true LOCAL dip. A monotonic flank descent keeps one side deep (dev > recov) → rejected (never stepped)."""
        b = np.zeros(L, dtype=bool)
        i = 0
        while i < L:
            if not cand[i]:
                i += 1
                continue
            j = i
            while j < L and cand[j]:
                j += 1
            if (j - i) <= maxw:
                look = pad + 4
                lo_ok = any((i - k) >= 0 and dev[i - k] <= recov for k in range(1, look + 1))
                hi_ok = any((j + k) < L and dev[j + k] <= recov for k in range(0, look + 1))
                if lo_ok and hi_ok:                              # recovers on BOTH sides = isolated dip, not a slope
                    b[max(0, i - pad):min(L, j + pad)] = True
            i = j
        return b

    out = e.copy()
    changed = False
    for f in range(F):
        col = e[:, f]
        trend = ndimage.median_filter(col, size=ksize, mode="nearest")
        dev = col - trend                                        # +ve = surface DEEPER than trend
        cand_up = (trend - col) > min_h                          # surface shallower than trend = climbed a streak
        cand_dn = (dev > down_min_h) if down else np.zeros(L, dtype=bool)  # deeper = candidate down-notch
        if not (cand_up.any() or cand_dn.any()):
            continue                                             # clean column → no-op
        band = _narrow_run_band(cand_up) | _local_dip_band(cand_dn, dev)
        if band.any():
            out[band, f] = trend[band]
            changed = True
    return out.astype(np.float32) if changed else edges


def detect_surface_all(sag: np.ndarray, params: dict | None = None, workers: int | None = None,
                       progress=None) -> np.ndarray:
    """The robust auto-detected corneal surface for EVERY sagittal slice (n_slices, n_frames) — the same
    per-slice _merged_side_edge the preprocessing detects. This is the BASELINE for the local-band
    re-detection: the part of the volume the user has NOT corrected stays exactly this 'satisfactory' edge."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n = int(sag.shape[0])
    if workers is None:
        workers = auto_workers()
    edges = _map_slices(_edge_worker, [(np.ascontiguousarray(sag[i]).astype(np.float32), p) for i in range(n)],
                        progress, 0.0, 1.0, workers)
    out = np.array([(e[0] if isinstance(e, tuple) else e) for e in edges], dtype=np.float32)
    # FIX apexspec: LATERAL specular-spike reject on the assembled (lateral, frame) surface. Auto detection only;
    # a supplied per-slice prior means fix-columns re-detection, which carries its own user-seeded surface and must
    # not be laterally re-smoothed here (the per-worker _edge_worker never receives a prior, so this is the auto path).
    if bool(p.get("spec_spike_reject", True)):
        out = _reject_apex_lateral_spike(out, p)
    if bool(p.get("despike_lateral", True)):
        out = _despike_lateral_surface(out, p)
    if bool(p.get("dip2d_suppress", True)):
        out = _suppress_surface_dips_2d(out, p)
    if bool(p.get("robust_dome", True)):
        out = _robust_dome_smooth(out, p, vol=sag)          # sag=(lat,depth,frames) → pocket-darkness gate
    if bool(p.get("lat_conf_smooth", True)):
        out = _lateral_smooth_by_confidence(out, sag, p)
    # NOTE: _parabola_edge_constrain is DELIBERATELY NOT applied here. detect_surface_all is the detection
    # baseline used by the NOISE-CROP (detect_noise_frames), the surface-crop, the confidence scores and the
    # fix-columns re-detect — all of which need the TRUE tissue-following surface. Snapping the edge frames to
    # the parabola moves the surface off the faint edge tissue → its confidence collapses → the noise-crop
    # wrongly zeroes valid cornea at the edges (CS003 OD: 25 real frames cropped). The parabola-edge motion-
    # artifact correction is applied ONLY to the final WARP surface (smooth_volume), so the OUTPUT edges follow
    # the parabola while the detectors still see the real tissue.
    return out


def _parabola_edge_constrain(edges: np.ndarray, p: dict) -> np.ndarray:
    """EDGE MOTION-ARTIFACT correction (user directive): the first/last few acquisition-edge frames carry a
    MOTION ARTIFACT that steepens the detected surface so it no longer matches the cornea's overall shape.
    Snap those edge frames to a ROBUST per-slice PARABOLA fit of the RELIABLE interior — so the corrected
    cornea never has steep edges that deviate from the overall parabola. The corneal cross-section (depth vs
    frame) is parabolic (measured interior fit residual ~1px), so this is a well-posed, gentle correction that
    only touches the `parabola_edge_nb` edge frames at each end; the interior is untouched. Iterative outlier
    rejection keeps the fit itself immune to the very motion artifacts it is correcting. parabola_edge=False
    disables. edges=(n_lateral, n_frames)."""
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2:
        return edges
    L, F = e.shape
    nb = int(p.get("parabola_edge_nb", 4))
    deg = int(p.get("parabola_edge_deg", 2))
    margin = int(p.get("parabola_edge_margin", 14))        # how far in from each end the motion artifact can reach
    snap_dev = float(p.get("parabola_edge_snap_dev", 5.0)) # a near-edge frame deviating > this from the parabola
    #   is a motion artifact → snap it too (the artifact often extends inward past the very edge frames, else the
    #   un-snapped inner artifact frames leave a DISCONTINUITY/step where the snapped edge rejoins them)
    if nb <= 0 or F < 2 * nb + 8 or L < 4:
        return edges
    # fit the parabola to the RELIABLE interior only (exclude both edge margins so the artifact can't bias it)
    fint = np.arange(margin, F - margin) if F - 2 * margin >= deg + 8 else np.arange(nb, F - nb)
    out = e.copy()
    changed = False
    for l in range(L):
        a = e[l]
        vld = a > 1.0
        ff = fint[vld[fint]]
        if ff.size < deg + 6:
            continue
        xx = ff.astype(np.float64); yy = a[ff].astype(np.float64)
        try:
            co = np.polyfit(xx, yy, deg)
            for _ in range(3):                             # robust: drop motion outliers, refit
                r = yy - np.polyval(co, xx)
                sd = np.std(r) + 1e-6
                keep = np.abs(r) < 2.5 * sd
                if keep.sum() < deg + 6 or keep.all():
                    break
                xx, yy = xx[keep], yy[keep]
                co = np.polyfit(xx, yy, deg)
        except Exception:
            continue
        for f in list(range(margin)) + list(range(F - margin, F)):
            if not vld[f]:
                continue
            para = np.polyval(co, f)
            if f < nb or f >= F - nb or abs(a[f] - para) > snap_dev:   # very edge, OR a near-edge artifact
                out[l, f] = para
                changed = True
    return out.astype(edges.dtype) if changed else edges


def _robust_dome_smooth(edges: np.ndarray, p: dict, vol: np.ndarray | None = None) -> np.ndarray:
    """POCKET-ROBUST anterior surface: keep the epithelial surface a SMOOTH DOME that rides gracefully OVER
    dark intra-stromal POCKETS (part of the disease variant — user directive), instead of dipping into them.
    The epithelium is a smooth continuous boundary and pockets sit BELOW it, so a local DOWNWARD excursion
    (surface deeper than the smooth dome) is the detector being pulled into a pocket — a detection error, not
    anatomy. Iterative ONE-SIDED robust smoothing estimates the smooth dome (gaussian over lateral+frame) and
    pulls DEEPER-than-dome points up to it (clamped by max_pull so a large-pocket gaussian can't run away and
    flatten a steep frame-edge descent).

    POCKET GATE (critical — vol given): the raw gaussian lift also fires on a HEALTHY cornea's normal
    curvature (the gaussian trend lags the natural acquisition-edge descent), which FLATTENS a healthy
    surface (marked as a 'curvature' defect on the healthy CS003 OD start-frames). So the lift is APPLIED
    ONLY where the tissue just BELOW the original surface is DARK (< robust_dome_pocket_frac × the frame's
    median sub-surface brightness) — i.e. a genuine pocket. On a healthy cornea (bright stroma everywhere
    below the epithelium) the gate is empty → STRICT NO-OP → curvature preserved; over a dark pocket it fires
    → the surface rides over it. vol=None (no gate) keeps the legacy uniform behaviour. robust_dome=False
    disables. edges=(n_lateral, n_frames); vol=(n_lateral, depth, n_frames); depth 0 = top."""
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2:
        return edges
    L, F = e.shape
    if L < 24 or F < 5:
        return edges
    sig_lat = float(p.get("robust_dome_sig_lat", 15.0))
    sig_fr = float(p.get("robust_dome_sig_frame", 5.0))
    thr = float(p.get("robust_dome_thr", 3.5))
    iters = int(p.get("robust_dome_iters", 4))
    max_pull = float(p.get("robust_dome_max_pull", 6.0))
    if thr <= 0 or iters <= 0 or (sig_lat <= 0 and sig_fr <= 0):
        return edges
    e0 = e.copy()
    out = e.copy()
    valid = e > 1.0
    changed = False
    for _ in range(iters):
        trend = ndimage.gaussian_filter(out, (max(0.0, sig_lat), max(0.0, sig_fr)), mode="nearest")
        dip = ((out - trend) > thr) & valid                # deeper than the smooth dome → dived into a pocket
        if not dip.any():
            break
        out[dip] = trend[dip]
        changed = True
    if not changed:
        return edges
    if max_pull > 0:                                       # clamp total displacement from the original detection
        out = np.where(valid, np.clip(out, e0 - max_pull, e0 + max_pull), e0)
    # POCKET GATE: keep the lift ONLY where there is a dark pocket just below the original surface. On a
    # healthy cornea this is empty → no-op (its natural curvature is preserved, not flattened).
    if vol is not None and bool(p.get("robust_dome_pocket_gate", True)):
        v3 = np.asarray(vol)
        if v3.ndim == 3 and v3.shape[0] == L and v3.shape[2] == F:
            D = v3.shape[1]
            frac = float(p.get("robust_dome_pocket_frac", 0.72))
            lo = int(p.get("robust_dome_below_lo", 4)); hi = int(p.get("robust_dome_below_hi", 30))
            lifted = (e0 - out) > 0.5                       # points the dome pulled up (now shallower)
            keep = np.zeros((L, F), dtype=bool)
            for f in range(F):
                idx = np.where(lifted[:, f])[0]
                if idx.size == 0:
                    continue
                vf = v3[:, :, f]
                d = np.clip(np.round(e0[:, f]).astype(int), 0, max(0, D - hi - 1))
                below = np.array([vf[l, d[l] + lo:d[l] + hi].mean() if e0[l, f] > 1 else np.inf for l in range(L)])
                fin = below[np.isfinite(below)]
                if fin.size == 0:
                    continue
                bref = float(np.median(fin))
                for l in idx:
                    if below[l] < frac * bref:             # dark below → genuine pocket → keep the lift
                        keep[l, f] = True
            out = np.where(keep, out, e0)
            if not keep.any():
                return edges
    return out.astype(edges.dtype)


def _suppress_surface_dips_2d(edges: np.ndarray, p: dict) -> np.ndarray:
    """Clip LOCAL surface excursions that deviate from a robust 2-D (lateral × frame) MEDIAN trend of the
    dome by > dip2d_thresh px, in EITHER direction, back to the trend. This catches the moderate-WIDTH dips
    the 1-D _despike_lateral_surface misses — the anterior surface being pulled ~6-10px toward a sub-surface
    stromal opacity on the low-signal flank (marked CS002 OS3 lat 294-389 × frames 0-45). The epithelium is
    a smooth dome that rides over stromal scars, so a LOCAL deviation from the 2-D trend is a detection error;
    the median trend follows the real smooth dome + steep limbus flank (a monotonic descent → median = the
    true centre value, no lag) and the cornea's gentle curvature means the apex is NOT flattened (median lag
    << thresh over the window). Uses the FRAME axis too, so a dip wide in one axis but localized in the other
    still stands out against the trend. dip2d_suppress=False disables. edges=(n_lateral, n_frames)."""
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2:
        return edges
    L, F = e.shape
    thr = float(p.get("dip2d_thresh", 7.0))
    if L < 24 or F < 5 or thr <= 0:
        return edges
    lw = int(p.get("dip2d_lat_win", 41)); lw = lw if lw % 2 else lw + 1
    fw = int(p.get("dip2d_frame_win", 9)); fw = fw if fw % 2 else fw + 1
    lw = min(lw, L if L % 2 else L - 1)
    fw = min(fw, F if F % 2 else F - 1)
    valid = e > 1.0
    trend = ndimage.median_filter(e, size=(lw, fw), mode="nearest")
    dev = e - trend
    mask = (np.abs(dev) > thr) & valid
    if not mask.any():
        return edges
    out = e.copy()
    out[mask] = trend[mask]
    return out.astype(edges.dtype)


def _despike_lateral_surface(edges: np.ndarray, p: dict) -> np.ndarray:
    """Remove NARROW, LARGE lateral surface excursions (spikes/notches) from the assembled (lateral, frame)
    surface. The anterior corneal surface is smooth across slices, so a contiguous run of <= despike_max_w
    laterals that deviates > despike_dev px (either direction) from a robust lateral MEDIAN trend is a
    detection artifact — the DP dove into a shadow/dropout notch (marked CS002 OS3 f0 ~35px dive, f6 notch)
    or climbed a reflection — NOT anatomy. Reset those runs to the trend. This is deliberately LESS
    conservative than _reject_apex_lateral_spike's two-sided-recovery down-branch (which mis-gates on the
    cluttered low-signal edge frames and let these through) but stays SAFE via two invariants: (a) the
    median trend follows a real smooth dome / steep limbus, so a genuine surface gives dev≈0 → strict
    no-op; (b) the WIDTH gate excludes a real limbus flank (a LONG monotonic run) — only NARROW runs, which
    a smooth cornea never produces, are reset. Independent of frame confidence, so it catches a local spike
    on an otherwise-high-confidence frame (which the frame-level lat_conf taper misses). despike_lateral=False
    disables. edges=(n_lateral, n_frames); depth 0 = top."""
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2:
        return edges
    L, F = e.shape
    win = int(p.get("despike_win", 31)); win = win if win % 2 else win + 1
    ksize = min(win, L if L % 2 else L - 1)
    if L < 16 or ksize < 5:
        return edges
    dev_t = float(p.get("despike_dev", 13.0))
    maxw = int(p.get("despike_max_w", 12))
    pad = int(p.get("despike_pad", 2))
    out = e.copy()
    changed = False
    for f in range(F):
        col = e[:, f]
        valid = col > 1.0
        if int(valid.sum()) < 16:
            continue
        trend = ndimage.median_filter(col, size=ksize, mode="nearest")
        cand = (np.abs(col - trend) > dev_t) & valid                 # narrow-or-wide excursion candidates
        if not cand.any():
            continue
        band = np.zeros(L, dtype=bool)
        i = 0
        while i < L:
            if not cand[i]:
                i += 1
                continue
            j = i
            while j < L and cand[j]:
                j += 1                                               # [i, j) contiguous run
            if (j - i) <= maxw:                                      # NARROW run only → artifact, reset to trend
                band[max(0, i - pad):min(L, j + pad)] = True
            i = j
        if band.any():
            out[band, f] = trend[band]
            changed = True
    return out.astype(edges.dtype) if changed else edges


def _lateral_smooth_by_confidence(out: np.ndarray, sag: np.ndarray, p: dict) -> np.ndarray:
    """Cross-SLICE (lateral) smoothing of the detected anterior surface, with the gaussian sigma TAPERED by
    each frame's detection confidence. The low-signal acquisition-edge frames (slow-scan extremes) detect
    the surface with lateral jitter → a jagged B-scan top contour; the high-signal interior frames detect
    cleanly. So smooth ACROSS SLICES only where confidence is low, easing to a strict NO-OP on confident
    frames (real anterior detail + already-approved scans untouched). This stays ON the detected tissue —
    unlike a frame-direction extrapolation it cannot put the edge at a wrong angle. The specular column and
    stromal opacities sit BELOW the surface, so they are preserved. out=(n_slices, n_frames); depth 0 = top.
    Strict no-op when every frame is confident, or L/F too small, or lat_conf_smooth=False."""
    e = np.asarray(out, dtype=np.float64)
    if e.ndim != 2:
        return out
    L, F = e.shape
    if L < 16 or F < 5:
        return out
    sig_max = float(p.get("lat_conf_sigma_max", 9.0) or 0.0)
    if sig_max <= 0:
        return out
    lo = float(p.get("lat_conf_lo", 0.35))
    hi = float(p.get("lat_conf_hi", 0.80))
    if hi <= lo:
        hi = lo + 1e-3
    # per-FRAME confidence = anterior CONTRAST (bright below − dark above) on the B-scan at that frame; a
    # low-signal edge frame scores near 0. sag=(n_slices, depth, n_frames) → B-scan at fr is sag[:, :, fr].T.
    con = np.zeros(F, dtype=np.float64)
    for fr in range(F):
        bs = np.ascontiguousarray(sag[:, :, fr]).astype(np.float32).T   # (depth, n_slices)
        con[fr] = _surface_confidence(bs, e[:, fr])[0]
    pos = con[con > 0]
    if pos.size < 3:
        return out
    ref = float(np.median(np.sort(pos)[-max(5, pos.size // 4):]))       # interior/high-signal reference
    if ref <= 0:
        return out
    conf_norm = np.clip(con / ref, 0.0, 1.0)                            # 1 ≈ as clean as the interior; ~0 = noisy edge
    e2 = e.copy()
    xs = np.arange(L)
    for fr in range(F):
        w = float(np.clip((hi - conf_norm[fr]) / (hi - lo), 0.0, 1.0))  # smoothing weight: 1 below lo, 0 above hi
        if w <= 0.01:
            continue
        col = e[:, fr]; m = col > 1.0
        if int(m.sum()) < max(20, L // 8):
            continue
        filled = np.interp(xs, xs[m], col[m])
        sm = ndimage.gaussian_filter1d(filled, sigma=sig_max * w, mode="nearest")
        e2[m, fr] = (1.0 - w) * col[m] + w * sm[m]                      # blend eases the seam to the interior
    return e2.astype(out.dtype)


def detect_noise_frames(sag: np.ndarray, params: dict | None = None, workers: int | None = None,
                        detect: np.ndarray | None = None) -> list:
    """AUTO crop-region: off-cornea NOISE frames at the scan boundary. The slow scan can run OFF the cornea,
    leaving leading/trailing frames with NO coherent corneal surface (near-zero edge contrast — bright stroma
    below a sharp boundary vs dark above). Returns the sorted frame indices to crop (zeroed before SAM2). A
    normal full-cornea scan returns []. Only LONG (>= crop_noise_min_run) contiguous BOUNDARY runs of near-zero
    contrast are cropped, so a faint cornea EDGE (a few low frames that quickly recover) is never removed; a
    failed/all-noise scan (> crop_noise_max_frac) is left untouched for the user."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    det = detect if detect is not None else detect_surface_all(sag, p, workers=workers)
    vs = ndimage.gaussian_filter(sag, (1.0, 2.0, 0.5))
    nf = int(sag.shape[2])
    con = np.array([_surface_confidence(vs[:, :, fr].T, det[:, fr])[0] for fr in range(nf)])
    # Reference = the cornea frames' contrast, taken from the strictly-POSITIVE contrasts only (a noise frame
    # scores ~0/negative). Using positive-only avoids a mostly-off-cornea scan poisoning the reference toward 0
    # (which would disable cropping just when it's needed most).
    pos = np.sort(con[con > 0])
    if pos.size == 0:
        return []
    ref = float(np.median(pos[-max(5, pos.size // 4):]))
    low = con < ref * float(p.get("crop_noise_frac", 0.20))
    run = int(p.get("crop_noise_min_run", 10))
    out: list[int] = []
    f = 0; lead = []
    while f < nf and low[f]:
        lead.append(f); f += 1
    if len(lead) >= run:
        out += lead
    f = nf - 1; trail = []
    while f >= 0 and low[f]:
        trail.append(f); f -= 1
    if len(trail) >= run:
        out += trail
    if len(set(out)) > nf * float(p.get("crop_noise_max_frac", 0.75)):
        return []                                                # a failed/all-noise scan — leave it to the user
    return sorted(set(out))


def detect_surface_crop_frames(sag: np.ndarray, params: dict | None = None, workers: int | None = None,
                               detect: np.ndarray | None = None) -> dict:
    """AUTO-SUGGEST the surface-CROPPED frames (B-scan columns whose corneal apex rises ABOVE the acquisition
    window, so the frame has no anterior surface). Runs the validated per-slice clip detector (_clip_mask) over
    every sagittal slice and counts, per frame, how many slices flag it clipped. Returns
    {frames:[...], counts:{frame: n_slices}, n_slices, depth_vox}: `frames` = those clipped in ≥crop_min_slices
    slices (the default selection the user verifies/edits); `counts` drives a per-frame confidence bar. The
    posterior-continuity reconstruction (build_surface_crop_edges) is what actually corrects the confirmed set."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    edges = detect if detect is not None else detect_surface_all(sag, p, workers=workers)
    n, depth_vox, F = int(sag.shape[0]), int(sag.shape[1]), int(sag.shape[2])
    counts = np.zeros(F, dtype=int)
    for i in range(n):
        cm = _clip_mask(np.ascontiguousarray(sag[i]).astype(np.float32), edges[i], p)
        counts[np.asarray(cm, dtype=bool)] += 1
    min_slices = max(1, int(p.get("crop_min_slices", 3)))
    frames = [int(f) for f in range(F) if counts[f] >= min_slices]
    return {"frames": frames, "counts": {int(f): int(counts[f]) for f in range(F) if counts[f] > 0},
            "n_slices": n, "depth_vox": depth_vox, "n_frames": F}


def _crop_reconstruct_slice(slice_img: np.ndarray, anterior: np.ndarray, crop_frames, p: dict):
    """POSTERIOR-CONTINUITY reconstruction for ONE sagittal slice. Returns (anterior_out, bottom_edge,
    adopted_mask):
      • bottom_edge   — the detected posterior edge (ALWAYS returned, for the UI to display the guidance);
      • anterior_out  — the detected anterior with the actually-clipped marked frames replaced by
                        bottom_edge − thickness, thickness interpolated FROM the non-marked frames' gap (can go
                        ABOVE the frame / negative where the apex is cropped);
      • adopted_mask  — which frames were reconstructed.
    Shared by build_surface_crop_edges (whose POSTERIOR feeds the warp) and the UI preview endpoint. NOTE: the
    extend warp (warp_surface_crop_extend) flattens to its OWN per-slice posterior PARABOLA and shifts+extends
    the canvas, so this reconstructed anterior is a GUIDANCE view of the bottom-edge match, not the pixel-exact
    final surface (the result is taller and parabola-flattened)."""
    a = np.asarray(anterior, dtype=np.float64)
    F = a.size
    b = _detect_bottom_edge(np.ascontiguousarray(slice_img).astype(np.float32), p).astype(np.float64)
    out = a.copy()
    adopted = np.zeros(F, dtype=bool)
    cfin = [] if crop_frames is None else list(crop_frames)
    cf = np.array(sorted({int(f) for f in cfin if 0 <= int(f) < F}), dtype=int)
    if cf.size == 0:
        return out, b, adopted
    floor = float(p.get("clip_edge_floor", 8.0))
    margin = float(p.get("crop_margin", 6.0))
    marked = np.zeros(F, dtype=bool); marked[cf] = True
    valid = (~marked) & np.isfinite(a) & (a >= floor)             # NON-marked frames with a trustworthy anterior
    if int(valid.sum()) < 3:
        return out, b, adopted
    t = b - a                                                     # corneal thickness (posterior - anterior)
    tv = t[valid]
    med = float(np.median(tv)); mad = float(np.median(np.abs(tv - med))) + 1e-6
    keep = valid.copy(); keep[valid] = np.abs(tv - med) <= 4.0 * 1.4826 * mad   # drop posterior mis-locks
    if int(keep.sum()) < 3:
        keep = valid
    fk = np.where(keep)[0].astype(np.float64)
    t_interp = np.interp(cf.astype(np.float64), fk, t[keep])      # thickness from non-marked flanks (held at ends)
    recon = b[cf] - t_interp
    clipped_here = recon < (a[cf] - margin)                       # reconstruct only where it sits ABOVE the detected anterior
    idx = cf[clipped_here]
    out[idx] = recon[clipped_here]
    adopted[idx] = True
    return out, b, adopted


def build_surface_crop_edges(sag: np.ndarray, crop_frames, params: dict | None = None,
                             workers: int | None = None):
    """Returns (anterior_edges, posterior_edges), each (n_slices, n_frames). `anterior_edges` feeds the warp as
    provided_edges, where the user-confirmed surface-CROPPED frames are reconstructed by POSTERIOR CONTINUITY;
    `posterior_edges` is the detected bottom edge per slice — the warp's alignment target for the crop path
    (the apex/edge may be clipped above the frame, but the posterior is still visible, so it is what we match).

    A cropped frame has no anterior surface (its apex is above the window), so its placement is taken from its
    still-visible BOTTOM (posterior) edge: effective_anterior(f) = posterior(f) − thickness(f), where
    thickness(f) is INTERPOLATED FROM the NON-cropped frames' (posterior − anterior) gap — never measured
    inside the cropped band. Flattening this array then lands every frame's posterior on ONE smooth curve, so a
    cropped frame aligns to the non-cropped frames by MATCHING THEIR BOTTOM EDGE (posterior continuity).

    Per slice: the NON-marked frames supply the thickness curve (robustly de-spiked, interpolated across the
    marked band). A marked frame is reconstructed only where it is ACTUALLY clipped in that slice — i.e. the
    posterior-continuity surface sits >= crop_margin px ABOVE the detected anterior (the detector pinned below
    the true, above-frame apex). On a slice where that marked frame's anterior is genuinely in-frame (a
    peripheral/limbus slice, where detected ~ posterior-thickness) the detected anterior is kept untouched —
    floor-independent (works whether the clip pins the detector at row ~5 or ~20). Frames not in `crop_frames`
    are never altered."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    edges = detect_surface_all(sag, p, workers=workers)            # (n_slices, n_frames) anterior
    F = int(sag.shape[2])
    posterior = np.zeros_like(edges)                               # detected bottom edge (warp target for the crop path)
    cf = np.array(sorted({int(f) for f in (crop_frames or []) if 0 <= int(f) < F}), dtype=int)
    for i in range(int(sag.shape[0])):
        out, b, _adopted = _crop_reconstruct_slice(sag[i], edges[i], cf, p)
        edges[i] = out.astype(np.float32)
        posterior[i] = b.astype(np.float32)
    return edges, posterior


def _fill_nan_1d(a: np.ndarray) -> np.ndarray:
    """Linearly interpolate NaNs in a 1-D array (held at the ends). All-NaN → zeros."""
    a = np.asarray(a, dtype=np.float64).copy()
    idx = np.arange(a.size); m = np.isfinite(a)
    if not m.any():
        return np.zeros_like(a)
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def is_substantial_clip(crop_info: dict, params: dict | None = None) -> bool:
    """Auto gate: True only when the detected clip is big enough to auto-trigger the surface-crop correction
    (a few stray flagged frames on a NORMAL dome must NOT fire). Needs >= crop_auto_min_frames flagged frames
    AND the most-clipped frame flagged in >= crop_auto_min_slices ABSOLUTE sagittal slices (a central/edge clip
    only spans the central/edge slices, so an absolute count — not a fraction of all slices — is the right test;
    the Avanti grid is a fixed 513 slices)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    frames = crop_info.get("frames") or []
    counts = crop_info.get("counts") or {}
    if len(frames) < int(p.get("crop_auto_min_frames", 6)):
        return False
    peak = max((int(v) for v in counts.values()), default=0)
    return peak >= int(p.get("crop_auto_min_slices", 12))


def warp_surface_crop_extend(sag: np.ndarray, posterior: np.ndarray, crop_frames, params: dict | None = None,
                             workers: int | None = None, detect: np.ndarray | None = None):
    """Correct a CLIPPED cornea (apex and/or a whole edge above the acquisition window) and return a TALLER
    volume. Per slice: fit the still-visible POSTERIOR (bottom) edge to a parabola (Pb), align each column to it
    with a ROBUST clipped shift (a posterior mis-lock can't blow up the canvas), and derive the top-edge parabola
    Pa = Pb − thickness (thickness from the in-frame flanks; its apex/edge may sit ABOVE the old top). The depth
    canvas is EXTENDED UPWARD by `pad` so every above-old-top point and every up-shifted (cut-off) column is kept
    — nothing is truncated. Returns (out_sag, pad, Pb, Pa) where out_sag is (n_slices, depth+pad+extra, n_frames)
    in the SAME per-voxel spacing (the window just spans more depth). SAM2 cornea verified on the taller volume."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n, depth, F = sag.shape
    cf = set(int(f) for f in (crop_frames or []))
    md = float(p.get("max_displacement", 40.0)) or 40.0
    res = float(p.get("residual_threshold", 5.0))
    det = detect if detect is not None else detect_surface_all(sag, p, workers=workers)
    # posterior parabola per slice (RANSAC = robust to a clip mis-lock), smoothed across slices for 3-D consistency
    Pb = np.stack([_fit_quadratic_ransac(posterior[i].astype(np.float64), res) for i in range(n)])
    Pb = ndimage.gaussian_filter1d(Pb, sigma=float(p.get("crop_slice_smooth", 2.0)), axis=0)
    # corneal thickness per slice from the NON-cut-off (in-frame) flanks → top-edge parabola (apex/edge may be <0)
    floor = float(p.get("clip_edge_floor", 8.0)); Ts = np.full(n, np.nan)
    for i in range(n):
        nonc = [f for f in range(F) if f not in cf and np.isfinite(det[i, f]) and det[i, f] >= floor]
        if len(nonc) >= 3:
            Ts[i] = float(np.median(posterior[i, nonc] - det[i, nonc]))
    Ts = _fill_nan_1d(Ts); Pa = Pb - Ts[:, None]
    # robust per-column shift: parabola − median-smoothed posterior, clipped so an outlier can't inflate the pad
    post_rob = np.stack([ndimage.median_filter(posterior[i].astype(np.float64), size=int(p.get("crop_target_med", 11)))
                         for i in range(n)])
    disp = np.nan_to_num(np.clip(Pb - post_rob, -md, md), nan=0.0, posinf=md, neginf=-md)  # never round a NaN shift
    # DISP SMOOTHING (anti-streak): the per-column shift = Pb(smooth) − posterior; when the posterior edge is FAINT
    # (a low-contrast bottom edge), its per-slice detection is noisy, so `disp` varies erratically slice-to-slice →
    # adjacent lateral columns shift by very different amounts → a vertical COMB artifact that mangles the extended
    # volume (even unclipped frames). A 2-D Gaussian over (slices, frames) removes that high-frequency noise while
    # preserving the low-frequency correction (recovering the clipped apex/edge as a coherent surface). On a
    # crisp-posterior scan `disp` is already smooth, so this is a near-no-op. Set the sigmas to 0 to disable.
    _ss = float(p.get("crop_disp_smooth_slice", 6.0)); _sf = float(p.get("crop_disp_smooth_frame", 1.5))
    if _ss > 0 or _sf > 0:
        disp = ndimage.gaussian_filter(disp, sigma=(_ss, _sf))
    Pa = np.nan_to_num(Pa, nan=0.0)
    raw_pad = int(np.ceil(max(0.0, -float(np.min(Pa)), -float(np.min(disp))))) + int(p.get("crop_pad_margin", 8))
    cap = int(p.get("crop_max_pad", 120))
    clamped = raw_pad > cap                                   # SEVERE clip/artifact — caller may skip (auto) or accept (manual)
    pad = min(raw_pad, cap)
    extra_bot = int(np.ceil(max(0.0, float(np.max((depth - 1) + np.maximum(0.0, disp))) - (depth - 1))))
    H2 = depth + pad + extra_bot
    out = np.zeros((n, H2, F), np.float32)
    for i in range(n):
        di = disp[i]
        for f in range(F):
            off = pad + int(round(di[f]))
            lo = max(0, off); hi = min(H2, off + depth)
            if hi > lo:
                out[i, lo:hi, f] = sag[i, lo - off:lo - off + (hi - lo), f]
    return out, int(pad), Pb, Pa, bool(clamped)


def _surface_confidence(sl_smooth: np.ndarray, edge: np.ndarray):
    """Score a detected edge WITHOUT ground truth: CONTRAST = bright tissue just below − dark just above (on a
    fixed-scale smoothed slice; high only on a real anterior boundary, since a deeper layer has bright cornea
    ABOVE it → low/negative contrast) and ROUGHNESS = mean |2nd difference| (jaggedness). Returns (contrast, roughness)."""
    D, F = sl_smooth.shape
    if D < 13:                                     # too shallow to sample ±6 px around the edge (never real OCT)
        return 0.0, (float(np.mean(np.abs(np.diff(np.asarray(edge, float), 2)))) if F >= 3 else 0.0)
    ei = np.clip(np.round(np.asarray(edge)).astype(int), 6, D - 7)
    fc = np.arange(F)
    below = np.mean([sl_smooth[ei + j, fc] for j in range(1, 7)], axis=0)
    above = np.mean([sl_smooth[ei - j, fc] for j in range(1, 7)], axis=0)
    return float(np.mean(below - above)), float(np.mean(np.abs(np.diff(np.asarray(edge, float), 2))))


def auto_tune_detector(sag: np.ndarray, params: dict | None = None, n_sample: int = 24):
    """The app tunes the native DP detector to THIS scan — no ground truth, no user input ('tuning performed
    by the app itself'). Coordinate-descent over the dp_* params, scoring each candidate on a spread of sampled
    sagittal slices by surface confidence (contrast − autotune_smooth_weight·roughness), and returns the best
    dp_* overrides (a dict) + its score. Sampling avoids the extreme periphery (limbus / cornea-out-of-frame)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if str(p.get("detector", "dp")).lower() == "legacy":
        return {}, 0.0
    n = int(sag.shape[0])
    lo, hi = int(0.15 * n), int(0.85 * n)
    if hi - lo < 3:
        lo, hi = 0, n - 1
    idxs = np.unique(np.linspace(lo, hi, min(int(n_sample), max(3, hi - lo))).round().astype(int))
    slices = [np.ascontiguousarray(sag[i]).astype(np.float32) for i in idxs]
    # Score contrast on a LIGHTLY-smoothed slice so OVER-smoothing the detector (which drifts the edge off the
    # sharp boundary) is penalised → the objective has an interior optimum that varies per scan, instead of
    # monotonically rewarding maximum smoothing. Roughness still penalises an under-smoothed jagged edge.
    sms = [ndimage.gaussian_filter(s, (1.0, 0.6)) for s in slices]
    sw = float(p.get("autotune_smooth_weight", 18.0))

    def score(pp: dict) -> float:
        cs = rs = 0.0
        for sl, sm in zip(slices, sms):
            c, r = _surface_confidence(sm, _detect_surface_dp(sl, pp))
            cs += c; rs += r
        k = max(1, len(slices))
        return cs / k - sw * (rs / k)

    # dp_sigma_frame floor RAISED 0.8→2.0 (min): a low frame-despeckle sigma let the DP surface chase frame-to-
    # frame speckle → column-level edge jitter (marked on CS002 OS(2)). 3.0 is the sweet spot; 4.0 over-smooths.
    grid = {"dp_sigma_depth": [2.0, 3.0, 4.0], "dp_sigma_frame": [2.0, 3.0, 4.0],
            "dp_below": [16, 24, 32], "dp_max_jump": [6, 10, 16]}
    # DETERMINISTIC: always start from the fixed DEFAULTS (not any persisted/incoming dp_*) and run coordinate
    # descent to a STABLE local optimum (idempotent), so the same raw scan always tunes to the same dp_* —
    # re-running a preprocess (or the steps filmstrip) never silently shifts the surface.
    best = {k: DEFAULT_PARAMS[k] for k in grid}
    best_s = score({**p, **best})
    for _pass in range(4):
        moved = False
        for key, vals in grid.items():
            bv, bs = best[key], best_s
            for v in vals:
                if v == best[key]:
                    continue
                s = score({**p, **best, key: v})
                if s > bs:
                    bs, bv = s, v
            if bv != best[key]:
                moved = True
            best[key], best_s = bv, bs
        if not moved:
            break
    return {k: (float(v) if isinstance(v, float) else int(v)) for k, v in best.items()}, float(best_s)


def redetect_surface(sag: np.ndarray, anchors: dict, params: dict | None = None,
                     baseline: np.ndarray | None = None, progress=None) -> np.ndarray:
    """LOCAL-BAND re-detection seeded by the user's fix-columns anchors.

    The auto-detected surface is KEPT everywhere ("the rest is satisfactory"); only a LOCAL BAND around the
    corrected ("pink line") region is re-detected — the corrected frames on the anchored slice(s) PLUS the
    neighbouring slices around that region (the detector uses neighbour comparison, so they need re-detection
    too), seeded by the user's drag and MARCHED outward until the re-detection re-converges to the auto edge
    (so the band auto-sizes to exactly where the correction matters). The band is spliced into the baseline
    with a smooth blend at its frame edges (no seam). This replaces the previous WHOLE-volume march, which
    re-detected every slice and so often replaced a good auto surface with a worse one.

    `sag` = sagittal volume (lateral, depth, frames), depth 0 = TOP. `anchors` = {slice: {frame: depth}}.
    `baseline` = the precomputed auto surface (n_slices, n_frames); if None it is detected here. Returns the
    surface: auto everywhere, locally corrected around the anchors."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n, depth, W = int(sag.shape[0]), int(sag.shape[1]), int(sag.shape[2])
    seed_win = float(p.get("redetect_seed_window", 2.0))     # snap tight to the user's drawn line (±1-2 px)
    march_win = float(p.get("detect_window", 10.0))
    fmargin = max(0, int(p.get("redetect_frame_margin", 8)))
    slice_band = max(0, int(p.get("redetect_slice_band", 30)))

    if baseline is not None and np.asarray(baseline).shape == (n, W):
        base = np.asarray(baseline, dtype=np.float32).copy()
    else:
        base = detect_surface_all(sag, p, progress=progress)
    surface = base.copy()

    # normalize anchors → {int slice: {int frame: float depth}} within bounds
    anc: dict[int, dict[int, float]] = {}
    for s_key, frames in (anchors or {}).items():
        try:
            s = int(s_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= s < n) or not isinstance(frames, dict):
            continue
        fm: dict[int, float] = {}
        for f_key, d in frames.items():
            try:
                f = int(f_key); dv = float(d)
            except (TypeError, ValueError):
                continue
            if 0 <= f < W and np.isfinite(dv):
                fm[f] = float(np.clip(dv, 0, depth - 1))
        if fm:
            anc[s] = fm
    if not anc:
        return surface  # no anchors → pure baseline (auto everywhere)

    # corrected frame region = the CONTIGUOUS RUNS of THIS SLICE's OWN anchored frames. CRITICAL: the region is
    # PER-SLICE, never a global union. Two earlier bugs lived here: (a) one [first,last] span bridged separate
    # edits with a straight line; (b) — the residual one — a SINGLE frame mask built from the UNION of every
    # slice's anchors (afr = {f for fm in anc.values() for f in fm}) was applied to EVERY anchored slice, so an
    # edit to frame F on slice A caused frame F to be re-detected (and shifted up to ±seed_win) on slices B,C,…
    # that the user never touched there ("un-edited parts of the edge become altered"). Now each slice's region
    # comes ONLY from its own anchored frames, so a frame is re-detected on a slice IFF the user edited it there.
    def _slice_runs(fm: dict) -> list[tuple[int, int]]:
        runs: list[tuple[int, int]] = []
        for f in sorted(fm):
            if runs and f == runs[-1][1] + 1:
                runs[-1] = (runs[-1][0], f)
            else:
                runs.append((f, f))
        return runs

    def _frame_weight(runs: list[tuple[int, int]]) -> np.ndarray:
        wf = np.zeros(W, dtype=np.float32)                  # 1 inside a run, ramping to 0 over ±fmargin, 0 elsewhere
        for (rs, re) in runs:
            lo, hi = max(0, rs - fmargin), min(W - 1, re + fmargin)
            for f in range(lo, hi + 1):
                if rs <= f <= re:
                    w = 1.0
                elif f < rs:
                    w = (f - lo + 1) / float(rs - lo + 1)
                else:
                    w = (hi - f + 1) / float(hi - re + 1)
                wf[f] = max(wf[f], w)
        return np.clip(wf, 0.0, 1.0)

    def _run_prior(fm: dict, slice_base: np.ndarray, runs: list[tuple[int, int]]) -> np.ndarray:
        """Prior = the slice's baseline, with each contiguous run's anchors interpolated WITHIN that run only
        (no straight line bridging separate runs). Frames outside the runs keep the baseline."""
        prior = slice_base.astype(np.float32).copy()
        for (rs, re) in runs:
            rf = sorted(f for f in fm if rs <= f <= re)
            if not rf:
                continue
            seg = np.arange(rs, re + 1)
            prior[rs:re + 1] = np.interp(seg, np.array(rf, float),
                                         np.array([fm[f] for f in rf], float)).astype(np.float32)
        return prior

    # SLICE band = ±slice_band around EACH anchored slice (NOT the [min,max] envelope). Each anchored slice gets
    # its own band; slices far from every anchored slice keep the baseline. slice_band=0 → ONLY anchored slices.
    anchored = sorted(anc.keys())
    ws = np.zeros(n, dtype=np.float32)                       # per-slice weight: 1 at an anchored slice → 0 at ±band
    for sa in anchored:
        for s in range(max(0, sa - slice_band), min(n - 1, sa + slice_band) + 1):
            d = abs(s - sa)
            ws[s] = max(ws[s], 1.0 if d == 0 else max(0.0, (slice_band + 1 - d) / float(slice_band + 1)))
    band_mask = ws > 0

    def _slice_weight(s: int) -> float:
        return float(ws[s])

    region_wf: dict[int, np.ndarray] = {}                   # slice -> its OWN per-frame region weight (no global union)
    redet_region: dict[int, np.ndarray] = {}               # slice -> full-W re-detected edge (region meaningful)

    def _splice(s: int, redet: np.ndarray, wf_s: np.ndarray) -> None:
        w = wf_s * _slice_weight(s)                          # combined frame×slice blend weight (frame weight is per-slice)
        surface[s] = base[s] * (1.0 - w) + redet.astype(np.float32) * w

    # 1) seed the anchored slice(s): each re-detects ONLY its own anchored frame-runs, windowed (±seed_win) around
    #    the user's drag, per-run prior (baseline between runs). Frames the user didn't touch on THIS slice keep base.
    for s, fm in anc.items():
        runs = _slice_runs(fm)
        wf_s = _frame_weight(runs)
        rmask = wf_s > 0
        prior = _run_prior(fm, base[s], runs)
        # light=True + tight seed_win: snap to the nearest gradient within ±1-2 px of the user's drawn line
        # (no robust side-correction/RANSAC that would pull the corrected edge away from where they drew it).
        redet = _redetect_one_slice(np.ascontiguousarray(sag[s]).astype(np.float32), prior, seed_win, p,
                                    light=True).astype(np.float32)
        # HARD-clamp the re-detected region to within ±seed_win of the prior so a low-signal frame can't drift.
        redet[rmask] = np.clip(redet[rmask], prior[rmask] - seed_win, prior[rmask] + seed_win)
        redet_region[s] = redet; region_wf[s] = wf_s; _splice(s, redet, wf_s)

    # 2) INTERPOLATE the correction across slices — fills the gaps BETWEEN anchored slices, and tapers to auto
    #    beyond ±slice_band of the outermost anchors. This REPLACES the previous re-detect march, which
    #    re-detected each un-anchored in-between slice and so snapped it back to the (too-shallow) auto edge — the
    #    correction never reached the slices between the ones the user fixed. Per frame, the anchored slices whose
    #    correction touches that frame are the interpolation KNOTS; the applied correction ((redet−base) already
    #    frame-weighted by region_wf, == what _splice adds at slice-weight 1) is linearly interpolated across
    #    slices between those knots and tapered beyond. A frame corrected on only ONE slice keeps the old ±band
    #    triangular taper (nothing to interpolate). Anchored slices are UNCHANGED — interp passes through their
    #    knot value, which equals the step-1 splice (base + corr at slice-weight 1).
    if slice_band > 0 and anchored:
        sidx = np.arange(n)
        # 2a) ACCURATE PRIOR — interpolate the RAW correction (anchor−base) across slices AND across frames, so the
        #     gaps between the slices the user fixed are filled with the true interpolated correction (not the auto
        #     edge). resid = per-frame residual interpolated over the anchored slices that touched that frame,
        #     tapered ±slice_band beyond the span; cov = the corrected region (frame weight) interpolated the same
        #     way. Then fill frame gaps (interp residual across frames within cov) + light smooth → no seams.
        resid = np.zeros((n, W), dtype=np.float32)
        cov = np.zeros((n, W), dtype=np.float32)
        def _spread(knots, vals):
            if len(knots) == 1:
                s0 = knots[0]
                return vals[0] * np.clip(1.0 - np.abs(sidx - s0) / (slice_band + 1), 0.0, 1.0).astype(np.float32)
            v = np.interp(sidx, knots, vals).astype(np.float32)
            lo, hi = knots[0], knots[-1]
            tap = np.where(sidx < lo, np.clip(1.0 - (lo - sidx) / (slice_band + 1), 0.0, 1.0),
                  np.where(sidx > hi, np.clip(1.0 - (sidx - hi) / (slice_band + 1), 0.0, 1.0), 1.0)).astype(np.float32)
            return v * tap
        for f in range(W):
            kr = [s for s in anchored if f in anc[s]]                    # slices that directly anchored frame f (raw drag)
            kc = [s for s in anchored if region_wf[s][f] > 1e-6]         # slices whose corrected region reaches f
            if kr:
                resid[:, f] = _spread(kr, np.array([anc[s][f] - base[s, f] for s in kr], dtype=np.float32))
            if kc:
                cov[:, f] = _spread(kc, np.array([region_wf[s][f] for s in kc], dtype=np.float32))
        cov = np.clip(cov, 0.0, 1.0)
        # fill residual across FRAMES within the corrected span of each slice (so margin frames between runs don't
        # sit at auto), then a light 2-D smooth of the correction field (no vertical seam / lateral step).
        for s in range(n):
            fcov = np.where(cov[s] > 1e-3)[0]
            if fcov.size >= 2:
                resid[s] = np.interp(np.arange(W), fcov, resid[s, fcov]).astype(np.float32)
                resid[s, :fcov[0]] = 0.0; resid[s, fcov[-1] + 1:] = 0.0
        resid = ndimage.gaussian_filter(resid, (2.0, 1.0))
        prior_surf = np.clip(base + resid, 0, depth - 1).astype(np.float32)
        # 2b) SMART re-detect: snap to the strongest RISING gradient NEAR the interpolated border — proximity-
        #     weighted so it refines to a nearby real edge but is NOT pulled back to the too-shallow auto edge
        #     (which, being a strong gradient, would win a plain argmax). window = ±redetect_interp_window;
        #     prox_sigma keeps it close to the interpolated prior; where no clear edge, the prior is kept.
        interp_win = float(p.get("redetect_interp_window", 3.0))
        prox_sigma = max(0.5, interp_win / 2.0)
        depthf = float(depth)
        for s in range(n):
            if s in anc:                                    # anchored slice → keep the user's exact drag (seeded)
                continue
            w = cov[s]
            if not (w > 1e-3).any():
                continue
            if interp_win > 0:
                sm = ndimage.gaussian_filter1d(sag[s].astype(np.float32), sigma=float(p["sigma"]), axis=0)
                grad = np.gradient(sm, axis=0)              # (depth, frames); rising edge = positive
                rows = np.arange(depth)[:, None]
                dist = rows - prior_surf[s][None, :]        # depth offset from the interpolated border
                inwin = np.abs(dist) <= interp_win
                prox = np.exp(-(dist ** 2) / (2.0 * prox_sigma ** 2))
                score = np.where(inwin, np.maximum(grad, 0.0) * prox, -1.0)
                redet = np.argmax(score, axis=0).astype(np.float32)
                # keep the interpolated prior where the window holds no real rising edge (score ~ 0)
                noedge = score.max(axis=0) <= 1e-6
                redet[noedge] = prior_surf[s][noedge]
            else:
                redet = prior_surf[s]
            surface[s] = base[s] * (1.0 - w) + redet * w    # blend into auto by the interpolated coverage weight
    if progress:
        progress(1.0)
    return surface


def generalize_surface(sag: np.ndarray, anchors: dict, params: dict | None = None,
                       baseline: np.ndarray | None = None, progress=None) -> np.ndarray:
    """GENERALIZE the user's fix-columns corrections to the WHOLE volume (all slices), so correcting a few
    representative slices propagates the CORRECTION PATTERN everywhere — not just the ±redetect_slice_band local
    march.

    Unlike redetect_surface (which keeps auto everywhere except a local band around each anchor), this LEARNS
    the systematic per-frame correction the user makes and interpolates it across ALL slices:
      1. For each frame, look at the anchored slices; if the user corrected it the SAME direction on
         >= gen_min_slices slices with a robust median |correction| > gen_min_resid, it's a "correction frame".
      2. The correction is the RESIDUAL (anchor − auto), NOT the absolute depth — so auto's own cross-slice
         curvature (dome/limbus) is preserved and only the learned correction is added on top. A slice the user
         anchored NEAR AUTO contributes a ~0 residual and stays near auto (no flat offset is ever imposed).
      3. Interpolate that residual across slices (edge-hold + taper to 0 beyond the anchored span) and across
         frames (fill gaps between correction frames, taper to 0 beyond the corrected range), then smooth.
      4. surface = auto + smoothed residual field.
    Validated on real data (CS004): reproduces held-out anchored slices to ~1.3px median (vs ~4-5px auto),
    smoothly. This is preview-first at the API layer (a separate generalize.npz, user-accepted). `sag` =
    (lateral/slice, depth, frames); `anchors` = {slice:{frame:depth}}; `baseline` = auto surface (n, W)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n, depth, W = int(sag.shape[0]), int(sag.shape[1]), int(sag.shape[2])
    base = (np.asarray(baseline, dtype=np.float64).copy()
            if baseline is not None and np.asarray(baseline).shape == (n, W)
            else detect_surface_all(sag, p, progress=progress).astype(np.float64))
    # normalize anchors → {int slice: {int frame: float depth}} within bounds (same as redetect_surface)
    anc: dict[int, dict[int, float]] = {}
    for s_key, frames in (anchors or {}).items():
        try:
            s = int(s_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= s < n) or not isinstance(frames, dict):
            continue
        fm: dict[int, float] = {}
        for f_key, d in frames.items():
            try:
                f = int(f_key); dv = float(d)
            except (TypeError, ValueError):
                continue
            if 0 <= f < W and np.isfinite(dv):
                fm[f] = float(np.clip(dv, 0, depth - 1))
        if fm:
            anc[s] = fm
    if not anc:
        return base.astype(np.float32)
    A = sorted(anc.keys())
    min_slices = max(1, int(p.get("gen_min_slices", 2)))
    min_resid = float(p.get("gen_min_resid", 3.0))
    sign_frac = float(p.get("gen_sign_frac", 0.7))
    resid_cap = float(p.get("gen_resid_cap", 45.0))     # clamp a wild mis-clicked anchor so it can't blow up
    taper_sl = max(1, int(p.get("gen_taper_slices", 20)))
    fmargin = max(1, int(p.get("gen_frame_margin", 6)))
    slice_sigma = float(p.get("gen_slice_sigma", 8.0))
    frame_sigma = float(p.get("gen_frame_sigma", 2.0))

    # STEP 1  learn the "correction frames" (CF) + per-slice residuals (median-robust, direction-consistent)
    CF: list[int] = []
    R: dict[int, dict[int, float]] = {}
    for f in range(W):
        sl = [s for s in A if f in anc[s]]
        if len(sl) < min_slices:
            continue
        res = np.array([float(np.clip(anc[s][f] - base[s, f], -resid_cap, resid_cap)) for s in sl])
        if np.median(res) > min_resid and float(np.mean(res > 0)) >= sign_frac:
            CF.append(f)
            R[f] = {s: float(np.clip(anc[s][f] - base[s, f], -resid_cap, resid_cap)) for s in sl}
    if not CF:
        return base.astype(np.float32)   # no systematic correction learned → auto unchanged

    # STEP 2  per-correction-frame residual, interpolated ACROSS SLICES (edge-hold, tapered beyond anchored span)
    RES = np.zeros((n, W), dtype=np.float64)
    sidx = np.arange(n)
    for f in CF:
        sf = sorted(R[f]); rv = np.array([R[f][s] for s in sf], dtype=np.float64)
        Rhat = np.interp(sidx, sf, rv)
        lo, hi = sf[0], sf[-1]
        Rhat = np.where(sidx < lo, Rhat[lo] * np.clip(1 - (lo - sidx) / taper_sl, 0, 1),
               np.where(sidx > hi, Rhat[hi] * np.clip(1 - (sidx - hi) / taper_sl, 0, 1), Rhat))
        RES[:, f] = Rhat

    # STEP 3  per slice, interpolate the residual ACROSS FRAMES (fill gaps between CF frames; taper beyond range)
    CFa = np.array(sorted(CF)); fx = np.arange(W)
    for s in range(n):
        row = np.interp(fx, CFa, RES[s, CFa])
        row = np.where(fx < CFa[0], RES[s, CFa[0]] * np.clip(1 - (CFa[0] - fx) / fmargin, 0, 1),
              np.where(fx > CFa[-1], RES[s, CFa[-1]] * np.clip(1 - (fx - CFa[-1]) / fmargin, 0, 1), row))
        RES[s] = row

    # STEP 4  smooth the correction field (slices × frames) and add to auto
    RES = ndimage.gaussian_filter(RES, (slice_sigma, frame_sigma))
    surface = np.clip(base + RES, 0, depth - 1)
    return surface.astype(np.float32)


def _interp_bad_displacement(disp: np.ndarray, bad_cols, good_cols) -> np.ndarray:
    """Replace the DISPLACEMENT (not the edge) at bad columns with a smooth interpolation from the
    GOOD anchor columns, so a bad column gets a correction consistent with its good neighbours.
    Interpolating the displacement (the correction field) rather than the detected edge avoids the
    overshoot that enlarged real curvature, and preserves the underlying tissue shape."""
    if not bad_cols:
        return disp
    W = len(disp)
    bad = [c for c in bad_cols if 0 <= c < W]
    if good_cols:
        anchors = sorted({c for c in good_cols if 0 <= c < W} - set(bad))
    else:
        bad_set = set(bad)
        anchors = [c for c in range(W) if c not in bad_set]
    if bad and len(anchors) >= 2:
        anchors = np.array(anchors)
        disp[bad] = np.interp(np.array(bad), anchors, disp[anchors])
    return disp


def _slice_displacement(active_edge, residual, corr_factor, bad_cols, good_cols, max_disp,
                        clip_cols=None, clip_fit=None, zero_cols=None):
    """The per-column shift that flattens one sagittal slice's boundary to its quadratic, WITH the
    over-correction guard (#2): a column whose demanded shift |quad-edge| exceeds max_disp is a runaway
    (a garbage low-signal edge the quadratic can't trust), so it is treated as bad and its shift is
    interpolated from the good (well-detected) columns, then hard-clamped — a runaway can no longer bend
    the slice by 100-360px or compound across passes. With NO runaway column (the normal case — a raw
    boundary deviates < ~17px from its fit) this is exactly the faithful (quad-edge)*corr_factor field,
    so well-detected scans/columns are unchanged. max_disp<=0 disables the guard (legacy).

    CLIPPED-APEX (clip_cols/clip_fit from _resolve_clip): on a slice whose dome apex is above the frame,
    the warp TARGET is the EXTRAPOLATING fit (clip_fit, fit to the in-frame flanks) instead of the
    apex-clamped legacy quadratic, and each clipped column's shift is clamped ≥0 — a clipped column's apex
    tissue is above the frame (not acquired), so its in-frame stroma must NOT be shifted UP off the top
    (the legacy code shifts it up and the warp truncates real epithelium). Clipped columns are also kept
    out of the runaway bad-set so their intentional ≈0 shift isn't interpolated away. clip_cols empty →
    byte-identical to the legacy path."""
    clip_cols = np.asarray(clip_cols, dtype=int) if clip_cols is not None else np.array([], dtype=int)
    zero_cols = np.asarray(zero_cols, dtype=int) if zero_cols is not None else np.array([], dtype=int)
    if (clip_cols.size or zero_cols.size) and clip_fit is not None:
        quad = np.asarray(clip_fit, dtype=np.float64)            # extrapolating fit (excludes clip/cut columns)
    else:
        quad = _fit_quadratic_ransac(active_edge, residual)
    disp = (quad - np.asarray(active_edge, dtype=np.float64)) * corr_factor
    if clip_cols.size:
        disp[clip_cols] = np.maximum(disp[clip_cols], 0.0)       # never shift a clipped column UP (lose tissue)
    if zero_cols.size:
        disp[zero_cols] = 0.0                                    # user-cut columns: clipped/unusable → leave as-is
    bad = set(int(c) for c in bad_cols)
    if max_disp and max_disp > 0:
        runaway = {int(c) for c in np.where(np.abs(disp) > max_disp)[0]}
        runaway -= set(int(c) for c in clip_cols) | set(int(c) for c in zero_cols)  # keep intentional clip/cut shifts
        bad |= runaway
    disp = _interp_bad_displacement(disp, sorted(bad), good_cols)  # runaway cols → good-neighbour shift
    if max_disp and max_disp > 0:
        np.clip(disp, -max_disp, max_disp, out=disp)              # backstop (e.g. an all-bad slice)
    return disp


def _disp_worker(packed):
    sl, active_edge, residual, corr_factor, bad_cols, good_cols, max_disp, clip_cols, clip_fit, zero_cols = packed
    return _slice_displacement(active_edge, residual, corr_factor, bad_cols, good_cols, max_disp,
                               clip_cols=clip_cols, clip_fit=clip_fit, zero_cols=zero_cols)


def _cap_edge_descent(disp_field: np.ndarray, active: np.ndarray,
                      clip_cols_list, zero_cols_list, p: dict, vol: np.ndarray | None = None) -> np.ndarray:
    """FRAME-EDGE OVER-DESCENT CAP (v149) — the recurring "edges too downward".

    The per-slice flatten target is a RANSAC QUADRATIC across frames. The cornea is parabolic only over
    the reliable interior; toward the acquisition edge the surface FLATTENS (limbus) and, worse, the
    first frames often carry an inter-frame MOTION STEP (a flat block acquired at a different axial eye
    position). The quadratic extrapolates the dome descent and pushes the output surface of those edge
    frames ~10-25px DEEPER than the true interior level (measured CS003 OD: raw frame-0 sits on-trend,
    but the warped output is +10..+28px). This is NOT the retired parabola_edge shelf (that pushed edges
    DOWN onto an over-descending parabola); here we only LIFT an over-descended edge back to the
    interior, one-sided, so it can never manufacture a downward step or an upward hook.

    Method (the crux is HOW the target is built): fit a robust deg-1 trend to `active` (= the INPUT/raw
    detected surface) over a window placed PAST the motion-step block (start = nb+gap, so a flat block
    at frames 0..~7 does not contaminate the fit — an earlier bug), and EXTRAPOLATE it across the edge
    frames → `expected`. `expected` is FLAT where the true periphery is flat (a motion-step edge → the
    over-descended output is pulled back up to the flat level, fixing the marked defect) and DESCENDS
    where the limbus genuinely descends (→ expected ≈ the real tissue → no lift → NO upward hook).

    SMOOTH 2-D BLEND (v149a — fixes the fuzzy en-face edge + wavy sagittal edge a hard per-column clamp
    caused): build a blend weight w = smoothstep((over-dev)/soft) and the target `expected` for the WHOLE
    edge block (all slices × edge frames) as 2-D fields, then GAUSSIAN-SMOOTH both across the slice
    (lateral) axis AND lightly across the frame axis before blending out_surface = (1-w)·cur + w·expected.
    A hard `min(cur, expected+dev)` per (slice,frame) alternated between cur and the clamp wherever cur
    crossed the threshold → a WAVE along frames + jaggedness across slices (the en-face boundary un-smooth
    at the very frames the cross-slice smoother had just fixed). Smoothing w+expected first makes the lift
    a coherent field → the lifted edge is smooth in BOTH directions, while w≈0 off the over-descent keeps
    it one-sided and a strict no-op on the interior / already-on-trend slices. Clip/user-cut cols exempt.
    frame_edge_cap=False disables."""
    if not bool(p.get("frame_edge_cap", True)):
        return disp_field
    fe_nb = int(p.get("frame_edge_nb", 10))          # edge frames eligible for the lift (covers the block)
    gap = int(p.get("frame_edge_gap", 4))            # frames skipped past the block before the fit window
    reach = int(p.get("frame_edge_reach", 16))       # length of the interior fit window
    dev = float(p.get("frame_edge_dev", 3.0))        # px deadband before any lift
    soft = float(p.get("frame_edge_soft", 5.0))      # px over which the blend weight ramps 0→1 (soft gate)
    sig_lat = float(p.get("frame_edge_lat_smooth", 12.0))    # gaussian sigma ACROSS slices (kills en-face fuzz)
    sig_frame = float(p.get("frame_edge_frame_smooth", 1.2)) # gaussian sigma ALONG frames (kills the sagittal wave)
    n, F = disp_field.shape[0], disp_field.shape[1]
    if fe_nb <= 0 or F < 2 * (fe_nb + gap + reach) + 4:
        return disp_field
    A = np.asarray(active, dtype=np.float64)
    out = disp_field.astype(np.float64).copy()
    CUR = A + out                                    # current (smooth) output surface field = flatten quad
    exempt = np.zeros((n, F), dtype=bool)
    for i in range(n):
        cc = clip_cols_list[i]; zc = zero_cols_list[i]
        if cc.size: exempt[i, cc] = True
        if zc.size: exempt[i, zc] = True

    def _fit(bx, bv):
        co = np.polyfit(bx, bv, 1)
        for _ in range(2):                           # robust: drop any residual block/outlier frames, refit
            r = bv - np.polyval(co, bx); sd = np.std(r) + 1e-6
            keep = np.abs(r) < 2.0 * sd
            if keep.sum() < 4 or keep.all():
                break
            co = np.polyfit(bx[keep], bv[keep], 1)
        return co

    for lead in (True, False):
        if lead:
            idx = np.arange(0, fe_nb)
            base = np.arange(fe_nb + gap, fe_nb + gap + reach)
        else:
            idx = np.arange(F - fe_nb, F)
            base = np.arange(F - fe_nb - gap - reach, F - fe_nb - gap)
        EXP = CUR[:, idx].copy()                     # per-lateral expected = interior linear extrapolation
        fit_ok = np.zeros(n, dtype=bool)
        for i in range(n):
            a = A[i]
            valid = (a[base] > 1.0) & np.isfinite(a[base])
            if valid.sum() < max(4, reach // 2):
                continue
            co = _fit(base[valid].astype(np.float64), a[base][valid].astype(np.float64))
            EXP[i] = np.polyval(co, idx.astype(np.float64))
            fit_ok[i] = True
        # TWO-SIDED DISTANCE FEATHER (v151): the low-signal acquisition-edge frames carry a jittery surface
        # in BOTH directions (up spikes + down over-descent + a frame-direction wave). A one-sided lift only
        # removes the DOWN over-descent, leaving the up-spikes/wave → the en-face (axial) border stays spiky
        # and the sagittal top edge stays wavy (user-reported). Instead blend the surface TOWARD the smooth
        # interior-dome extrapolation with a raised-cosine weight = 1 at the extreme edge tapering to 0 at the
        # nb boundary — two-sided, so up-spikes are pulled DOWN and over-descent pulled UP onto one smooth arc.
        # `expected` is the interior extrapolation (never over-descends), and both it and the weight are
        # gaussian-smoothed across the slice (lateral) axis below → the resulting edge is a clean arc in the
        # axial view AND smooth along frames. Feather+gate keep it a strict no-op on the reliable interior.
        _dist = (idx if lead else (F - 1 - idx)).astype(np.float64)
        wf = 0.5 * (1.0 + np.cos(np.pi * np.clip(_dist / float(fe_nb), 0.0, 1.0)))   # 1 at edge → 0 at nb
        W = np.tile(wf, (n, 1))
        W[~fit_ok] = 0.0
        W[(A[:, idx] <= 1.0) | exempt[:, idx]] = 0.0
        # DO-NO-HARM CONFIDENCE GATE (v150): at the extreme edge frames the surface signal is often
        # near-absent (the scan ran off the cornea into noise), so `active` floats in air there; lifting
        # THAT toward the interior trend manufactures a spike hanging in the dark (seen on approved OD1/
        # CS002 OS3 frame-0). Gate the lift by the tissue CONTRAST at the current surface (bright below −
        # dark above): keep the lift only where there is a real boundary, ramp it to 0 where the surface is
        # unreliable → the cap never invents an edge in a no-signal region (it leaves the pre-existing fuzz
        # rather than adding a new artifact). Reference is the per-edge-frame high-contrast level across
        # laterals, so a frame that is mostly real tissue gates out only its floating outliers.
        gate = np.ones((n, idx.size), dtype=np.float64)
        if vol is not None and idx.size:
            D = vol.shape[1]
            conf = np.zeros((n, idx.size), dtype=np.float64)
            for i in range(n):
                r = np.clip(np.round(A[i, idx]).astype(int), 6, D - 7)
                colv = vol[i]
                below = np.mean([colv[r + j, idx] for j in range(1, 7)], axis=0)
                above = np.mean([colv[r - j, idx] for j in range(1, 7)], axis=0)
                conf[i] = below - above
            ref = np.maximum(np.percentile(conf, 60, axis=0), 1e-3)     # typical real-tissue contrast per frame
            frac = float(p.get("frame_edge_conf_frac", 0.5))
            gate = np.clip(conf / (frac * ref), 0.0, 1.0)               # →0 where no tissue under the surface
            W = W * gate
        # DE-BUMP THE ANTERIOR BOUNDARY (v152 — the real fix): the warp flattens each lateral slice
        # INDEPENDENTLY, so at the low-signal acquisition-edge frames the per-lateral shifts are jittery and
        # the anterior boundary comes out JAGGED across lateral even though the RAW boundary is a clean smooth
        # arc — i.e. the preprocessing DEGRADES a boundary that was smoother before (user-reported, verified
        # raw-vs-processed). Fix: smooth the ACTUAL output boundary ACROSS LATERAL so the processed arc matches
        # the raw's smoothness, while staying ON the tissue. `base` = the boundary after a one-sided
        # over-descent correction (min(cur, interior-extrap+dev), per lateral); `target` = base gaussian-
        # smoothed across the slice axis (strong) + lightly along frames → a clean smooth arc that still hugs
        # the tissue (NOT a frame-extrapolation that could float off it). Feathered 1→0 from the extreme edge
        # to the nb boundary and gated only where there is genuinely NO tissue, so the interior is untouched.
        med_lat = int(p.get("frame_edge_lat_med", 9))
        base = ndimage.median_filter(CUR[:, idx], size=(max(1, med_lat), 1), mode="nearest")  # kill narrow spikes
        target = ndimage.gaussian_filter(base, sigma=(sig_lat, sig_frame), mode="nearest")     # then smooth the arc
        W = ndimage.gaussian_filter(W, sigma=(sig_lat, sig_frame), mode="nearest")
        # BOUNDED de-bump: the smooth target can sit far from `cur` where the surface floats in noise (off the
        # cornea); warping the tissue there by that full delta shoves it out of frame (BLACK BANDS — seen with
        # the gate off). Clamp the per-column de-bump shift to ±frame_edge_max_shift so a no-signal column can
        # never cause a catastrophic shift; this backstop lets the confidence gate be RELAXED enough to still
        # smooth the DIM-but-real edge tissue (a too-strong gate left a residual bump on the faint side).
        maxsh = float(p.get("frame_edge_max_shift", 15.0))
        delta = np.clip(W * (target - CUR[:, idx]), -maxsh, maxsh)
        newsurf = CUR[:, idx] + delta
        # bound the per-column de-bump shift so no column is ever moved more than frame_edge_max_shift — a
        # floating no-signal column can never be shoved out of frame in either direction (black bands).
        newsurf = np.clip(newsurf, CUR[:, idx] - maxsh, CUR[:, idx] + maxsh)
        for k, f in enumerate(idx):
            col = ~exempt[:, f]
            out[col, f] = newsurf[col, k] - A[col, f]
    # NOTE: the frame-direction OVER-DESCENT cap ("very steep curvature near the ends") is NOT applied here —
    # this runs inside the iterative flatten, so a displacement edit feeds the next iteration's global quad fit
    # and LEAKS into the interior (measured 3.5px mean / 30px max on approved CS002 OS3). It is a POST-HOC pass
    # instead (frame_edge_overdescent_cap, called once on the final volume) → strictly local, no cascade.
    for i in range(n):                               # re-assert tissue-preservation clamps
        cc = clip_cols_list[i]; zc = zero_cols_list[i]
        if cc.size: out[i][cc] = np.maximum(disp_field[i][cc], 0.0)
        if zc.size: out[i][zc] = disp_field[i][zc]
    return out


def _axial_roughness(edges: np.ndarray) -> float:
    """Mean |first-difference of the detected corneal boundary ACROSS sagittal slices| (axis 0) — i.e.
    how jagged the en-face / AXIAL boundary is. Per-slice correction is independent, so inconsistent
    inter-slice shifts make this grow ('hairier' axial view, #3); lower = smoother axial boundary."""
    e = np.asarray(edges, dtype=float)
    if e.ndim != 2 or e.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(e, axis=0))))


def _map_slices(worker, items, progress, lo, hi, workers):
    """Map a per-slice worker across slices on a spawn pool (no CUDA-fork issues),
    falling back to serial on any failure. Reports progress in [lo, hi]."""
    n = len(items)
    out = [None] * n
    try:
        import concurrent.futures
        import multiprocessing as mp
        # fork: children inherit this (clean, torch-free) process's memory — fast, no
        # re-import, no recursion. Safe because the heavy smoother runs in an isolated
        # subprocess (oct_preprocess CLI), never directly inside the CUDA-bearing sidecar.
        ctx = mp.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for i, r in enumerate(ex.map(worker, items, chunksize=8)):
                out[i] = r
                if progress:
                    progress(lo + (hi - lo) * (i + 1) / n)
        return out
    except Exception:
        for i, it in enumerate(items):
            out[i] = worker(it)
            if progress:
                progress(lo + (hi - lo) * (i + 1) / n)
        return out


def smooth_volume(volume: np.ndarray, params: dict | None = None, progress=None,
                  workers: int | None = None, return_metric: bool = False,
                  detect_volume: np.ndarray | None = None,
                  provided_edges: np.ndarray | None = None,
                  clip_report: dict | None = None,
                  fixed_clip_cols: list | None = None):
    """Apply the corneal-edge + column correction with 3D active correction to a
    (frames, H, W) volume; returns the corrected volume (same shape/dtype).

    Equivalent to DICOMSmootherSteps' process_slice_with_3d_active over every sagittal
    slice, but each slice's edge is computed once (O(N), not O(3N)) and the two
    independent per-slice phases are parallelised across CPU cores.

    return_metric=True → also return (mean per-column correction magnitude px, axial roughness px):
    the iterative-refinement convergence signal and the en-face boundary jaggedness (#3). The corrected
    array is identical either way.

    NOTE: the correction is no longer byte-identical to DICOMSmootherSteps — by design (the user asked
    to fix two failure modes): the OVER-CORRECTION GUARD (#2, max_displacement) interpolates+clamps a
    runaway lateral shift, and INTER-SLICE SMOOTHING (#3, interslice_smooth) smooths the displacement
    field across slices for a consistent axial boundary. Both are no-ops at their off values
    (max_displacement<=0, interslice_smooth=0) and the guard is a no-op on well-detected columns, so a
    clean scan is essentially unchanged; only the pathological lateral runaway/hairiness is tamed.

    detect_volume: if given, the corneal edge is DETECTED on this volume (e.g. a black-band-filled
    copy, so re-detection on a warped input isn't fooled by the warp's zero padding) while the warp is
    applied to `volume` itself — so the OUTPUT never contains the filled (fake-tissue) pixels, only the
    real data + honest zero padding. The cornea sits at the same row in both (filling only touches
    padding), so the detected displacement aligns `volume`'s cornea correctly.

    provided_edges (n_slices=lateral, n_frames): if given, USE these per-slice surface rows AS the detected
    edge instead of detecting — and SKIP the 3D-active snap + inter-slice smoothing. This is the fix-columns
    marched re-detection (redetect_surface) result: the warp then flattens EXACTLY to fit(provided_edges),
    which is the same edge+fit the scrub preview drew → preview == result by construction."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    sag = reformat_to_sagittal(volume)             # the volume to WARP (real data, never filled)
    det = reformat_to_sagittal(detect_volume) if detect_volume is not None else sag  # detect on this
    n = sag.shape[0]
    corr_factor = float(p.get("corr_factor", 1.0))
    active_threshold = float(p.get("active_threshold", 5.0))
    if workers is None:
        workers = auto_workers()

    use_provided = provided_edges is not None
    if use_provided:
        # the marched re-detected surface IS the edge; flatten directly to its fit (no snap/smooth) so the
        # warp matches the previewed border exactly.
        edges = np.asarray(provided_edges, dtype=np.float32)
        if edges.shape != (n, sag.shape[2]):
            raise ValueError(f"provided_edges shape {edges.shape} != expected {(n, sag.shape[2])}")
        # LATERAL DE-STREAK (fix-columns axial fuzz) — the provided (re-detected) surface jitters column-to-
        # column across LATERALS at low-SNR laterals the user did NOT drag; warping to it shears the clean band
        # into vertical spikes = a "fuzzy" axial/en-face border (the raw is clean there). Replace only the
        # single-lateral SPIKES (> _gate px off a robust lateral median→gaussian trend) with the trend, but:
        #   (1) never touch a column within ±_protect_lat laterals / ±_protect_frame frames of one of the user's
        #       DRAG points (their corrections + the immediate propagation zone are protected), and
        #   (2) PIN every exact drag point to its depth.
        # So only the re-detector's OWN un-dragged guesses are regularised (opted into); the actual drags are
        # honoured to the pixel. Smooths across LATERALS ONLY (axis 0) — the frame axis (the per-slice drag) is
        # never touched. Off via provided_edge_lat_median<=1 or provided_edge_lat_gate<=0. Validated on CS004:
        # peripheral streaks reduced, drag points 0px, central frames unchanged.
        _lem = int(p.get("provided_edge_lat_median", 7) or 0)
        _les = float(p.get("provided_edge_lat_smooth", 2.0) or 0.0)
        _gate = float(p.get("provided_edge_lat_gate", 2.0) or 0.0)
        _bl = int(p.get("provided_edge_protect_lat", 2) or 0)
        _bf = int(p.get("provided_edge_protect_frame", 1) or 0)
        if _lem > 1 and _gate > 0:
            _nl, _nf = edges.shape
            _trend = ndimage.median_filter(edges, size=(_lem, 1), mode="nearest")
            if _les > 0:
                _trend = ndimage.gaussian_filter1d(_trend.astype(np.float64), sigma=_les, axis=0,
                                                   mode="nearest").astype(np.float32)
            _spike = np.abs(edges - _trend) > _gate
            _prot = np.zeros((_nl, _nf), dtype=bool)
            _anc = p.get("border_anchors") or {}
            for _slat, _fm in _anc.items():
                try:
                    _li = int(_slat)
                except (TypeError, ValueError):
                    continue
                if not (0 <= _li < _nl) or not isinstance(_fm, dict):
                    continue
                _l0, _l1 = max(0, _li - _bl), min(_nl, _li + _bl + 1)
                for _sf in _fm:
                    try:
                        _fi = int(_sf)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= _fi < _nf:
                        _prot[_l0:_l1, max(0, _fi - _bf):min(_nf, _fi + _bf + 1)] = True
            _E = np.where(_spike & ~_prot, _trend, edges).astype(np.float32)
            for _slat, _fm in _anc.items():
                try:
                    _li = int(_slat)
                except (TypeError, ValueError):
                    continue
                if 0 <= _li < _nl and isinstance(_fm, dict):
                    for _sf, _dep in _fm.items():
                        try:
                            _fi = int(_sf)
                        except (TypeError, ValueError):
                            continue
                        if 0 <= _fi < _nf:
                            _E[_li, _fi] = float(_dep)
            edges = _E
        if progress:
            progress(0.5)
    else:
        # 1) per-slice corrected boundary (the expensive bilateral+edge+RANSAC) — parallel. Detected on
        #    `det` (the filled copy when iterating) so the warp's black padding can't fool the detector.
        #    The DP scar-guard (legacy cross-check) runs only on the RAW pass-0 detection (detect_volume is
        #    None) — the scar-lock is a raw-image phenomenon; later passes detect on a warped+filled volume
        #    where the surface is already flattened, so skipping the (costly) per-slice legacy there is safe.
        pe = {**p, "dp_scar_guard": False} if detect_volume is not None else p
        edges = np.array(_map_slices(_edge_worker, [(det[i], pe) for i in range(n)], progress, 0.0, 0.5, workers))
        # FIX apexspec: LATERAL specular-spike reject on the assembled (lateral, frame) edge that actually drives
        # the WARP — the per-slice _edge_worker climbs a narrow apex vertical specular streak independently per
        # lateral, leaving a jagged spike in the AXIAL/en-face surface that a per-frame reject can't smooth. Runs
        # on the auto path only (never provided_edges — handled in the use_provided branch); safe on every pass
        # (gated no-op where the surface already sits on the lateral trend). detect on a filled/warped copy
        # (detect_volume is not None) already sits flattened, so this is a no-op there too.
        if bool(p.get("spec_spike_reject", True)):
            edges = _reject_apex_lateral_spike(edges, p)
        # FIX jagged edge B-scans (marked CS002 OS(2)-(4) at axial slice 1-2): the FIRST/LAST few frames are
        # low-signal (acquisition edge), so their direct per-slice detection is noisy → a jagged warp. The raw
        # corneal shape is SMOOTH across frames, so replace those frames' surface with a frame-direction
        # extrapolation from the reliable INTERIOR frames (+ cross-slice smoothing) — the warp then flattens them
        # to the smooth interior-consistent shape. On a filled/warped later pass the surface already sits flat, so
        # it is a near-no-op there; on a genuinely smooth scan it is a strict no-op. boundary_extrap_nb=0 disables.
        if not use_provided and int(p.get("boundary_extrap_nb", 4) or 0) > 0:
            edges = _extrapolate_boundary_edges(edges, p)
        # FIX residual edge_detection (marked CS002 OS3 f0 ~35px dive + f6 notch): remove NARROW, LARGE
        # lateral surface excursions the DP baked in by diving into a shadow/dropout notch. Runs on the WARP
        # surface (a spiked surface → the warp pulls tissue up into a spike), independent of frame confidence
        # (a local spike can sit on an otherwise-high-confidence frame that lat_conf skips). Width-gated so a
        # real limbus flank / smooth dome is a strict no-op.
        if not use_provided and bool(p.get("despike_lateral", True)):
            edges = _despike_lateral_surface(edges, p)
        # FIX flank dips toward sub-surface opacities (marked CS002 OS3 lat 294-389 × frames 0-45): clip local
        # surface excursions >dip2d_thresh px from a robust 2-D (lateral×frame) dome trend back to it — the
        # moderate-WIDTH dips the 1-D despike misses where the surface is pulled toward a stromal scar/opacity.
        if not use_provided and bool(p.get("dip2d_suppress", True)):
            edges = _suppress_surface_dips_2d(edges, p)
        # POCKET-ROBUST dome (user directive: dark intra-stromal pockets are part of the disease and the
        # epithelial surface must ride smoothly OVER them, not dip in): iterative one-sided robust smoothing
        # pulls surface points that dived DEEPER than the smooth dome (into a pocket) back up to it, but ONLY
        # where the tissue below is DARK (a genuine pocket) — so on a HEALTHY cornea it is a strict no-op and
        # its natural curvature is preserved (the gate uses `det`, the detection volume).
        if not use_provided and bool(p.get("robust_dome", True)):
            edges = _robust_dome_smooth(edges, p, vol=det)
        # FIX jagged edge B-scans (the SUCCESSOR to the retired boundary extrapolation; marked CS002 OS(2)
        # f99/100, OS(3) f0-20): the low-signal acquisition-edge frames detect the surface with cross-SLICE
        # jitter → a jagged B-scan top contour. Smooth the assembled (lateral, frame) edge ACROSS SLICES with
        # a sigma tapered by each frame's detection confidence — strong on the noisy edge frames, a strict
        # NO-OP on confident interior frames. Stays ON the detected tissue (no wrong edge angle). On a
        # filled/warped later pass the surface is already flat (high confidence) → near-no-op there too.
        if not use_provided and bool(p.get("lat_conf_smooth", True)):
            edges = _lateral_smooth_by_confidence(edges, det, p)
        # EDGE PARABOLA CONSTRAINT — DISABLED by default (v148): the interior-fit parabola over-descends at the
        # limbus (cornea flattens there, not a true parabola), so hard-snapping the outer frames INJECTED a ~10px
        # downward step/V-notch at the margin boundary (~frame 87) — the exact steep edge it meant to remove. The
        # raw detection already tracks the limbus flattening smoothly. Opt-in only (default False). See DEFAULT_PARAMS.
        if not use_provided and bool(p.get("parabola_edge", False)):
            edges = _parabola_edge_constrain(edges, p)

    res = float(p["residual_threshold"])
    W = int(sag.shape[2])
    zero_cols_list = [np.array([], dtype=int) for _ in range(n)]
    # USER SURFACE CUT (re-run option): exclude a clipped surface from the fit so the flattening is robust.
    #   surface_cut = {"left": frame, "right": frame, "top": depth} — exclude frames < left, frames > right,
    #   and frames whose detected surface is ABOVE depth `top` (a clipped apex / reflection). Excluded columns
    #   are dropped from the quadratic fit (which extrapolates across them) and left UNWARPED (disp=0). Takes
    #   precedence over the auto clip-apex handling (the user is correcting manually). Not for provided_edges.
    cut = p.get("surface_cut") or {}
    cut_left = int(cut.get("left", 0) or 0); cut_right = int(cut.get("right", 0) or 0); cut_top = int(cut.get("top", 0) or 0)
    has_cut = (not use_provided) and (cut_left > 0 or (0 < cut_right < W - 1) or cut_top > 0)
    # 1.5) clipped-apex resolution (per slice): where the dome apex is above the frame, detect the clipped
    #   columns + an EXTRAPOLATING fit from the in-frame flanks. Gated + cheap: _resolve_clip early-exits on
    #   a normal in-frame dome (the overwhelming majority), so a well-detected scan is unchanged.
    #   DETECTION RUNS ONLY ON THE RAW ACQUISITION (detect_volume is None): the 'tissue at row 0, no air gap'
    #   clip invariant holds only on raw data — a re-fed/axial pass detects on a warped+filled copy where the
    #   warp itself manufactures that pattern, so detecting there would false-trigger on a NORMAL scan. Such
    #   passes instead REUSE the pass-0 clip columns via fixed_clip_cols (re-fitting on the current edges).
    #   Skipped entirely for provided_edges (the marched surface is authoritative) and when clip_handling off.
    clip_on = bool(p.get("clip_handling", True)) and not use_provided and (detect_volume is None) and not has_cut
    if has_cut:
        base_excl = set()
        if cut_left > 0:
            base_excl |= set(range(0, min(cut_left, W)))
        if 0 < cut_right < W - 1:
            base_excl |= set(range(cut_right + 1, W))
        clip_cols_list = [np.array([], dtype=int) for _ in range(n)]   # no apex >=0 clamp on a user cut
        clip_fit_list = []
        for i in range(n):
            excl = set(base_excl)
            if cut_top > 0:
                excl |= {int(f) for f in range(W) if float(edges[i][f]) < cut_top}
            cc = np.array(sorted(c for c in excl if 0 <= c < W), dtype=int)
            zero_cols_list[i] = cc
            clip_fit_list.append(_extrapolate_fit(edges[i], cc, res) if cc.size else None)
    elif clip_on:
        clip_resolved = [_resolve_clip(edges[i], det[i], res, p) for i in range(n)]
        clip_cols_list = [cr[0] for cr in clip_resolved]
        clip_fit_list = [cr[1] for cr in clip_resolved]
    elif fixed_clip_cols is not None and not use_provided and bool(p.get("clip_handling", True)):
        # carry-forward (iteration passes ≥1): reuse pass-0's clipped columns, refit on the current edges.
        clip_cols_list = [np.asarray(fixed_clip_cols[i], dtype=int) if i < len(fixed_clip_cols)
                          else np.array([], dtype=int) for i in range(n)]
        clip_fit_list = [(_extrapolate_fit(edges[i], clip_cols_list[i], res) if clip_cols_list[i].size else None)
                         for i in range(n)]
        clip_cols_list = [cc if (cf is not None) else np.array([], dtype=int)
                          for cc, cf in zip(clip_cols_list, clip_fit_list)]
    else:
        clip_cols_list = [np.array([], dtype=int) for _ in range(n)]
        clip_fit_list = [None for _ in range(n)]
    if clip_report is not None:
        cr_map = {int(i): [int(c) for c in clip_cols_list[i]] for i in range(n) if len(clip_cols_list[i])}
        clip_report["apex_clipped"] = {"slices": cr_map, "n_slices": len(cr_map),
                                       "n_frames_total": int(sum(len(v) for v in cr_map.values()))}
        clip_report["_clip_cols"] = clip_cols_list   # raw arrays for iteration carry-forward (internal)

    # 2) 3D active correction — faithful to DICOMSmootherSteps.process_slice_with_3d_active: snap
    #    each slice's edge toward the median of ITSELF + its available neighbours (boundaries
    #    included), where the deviation exceeds the threshold. SKIPPED for provided_edges (the marched
    #    surface is already the desired boundary; snapping would pull it off the user's correction).
    active = edges.copy()
    for i in (range(n) if not use_provided else range(0)):
        stack = [edges[i]]
        if i > 0:
            stack.append(edges[i - 1])
        if i < n - 1:
            stack.append(edges[i + 1])
        med = np.median(np.stack(stack), axis=0)
        dev = np.abs(edges[i] - med)
        snap = dev > active_threshold
        cc = clip_cols_list[i]; zc = zero_cols_list[i]
        if len(cc):
            snap[cc] = False                 # don't snap a clipped column toward neighbours — it stays extrapolated
        if len(zc):
            snap[zc] = False                 # user-cut columns are excluded from detection/fit too
        active[i][snap] = med[snap]

    # 3) per-slice displacement that flattens the boundary to its quadratic — parallel — WITH the
    #    over-correction guard (#2): a runaway shift (garbage low-signal edge) is interpolated from good
    #    neighbours + clamped, so it can't bend the edge or compound across passes.
    # provided_edges (marched re-detect): flatten EXACTLY to fit(surface) — no force/good columns and no
    # over-correction guard, so the warp equals what the preview drew (the real cornea ≈ its own quadratic,
    # so disp stays small anyway).
    max_disp = 0.0 if use_provided else float(p.get("max_displacement", 0.0) or 0.0)
    bad_cols = [] if use_provided else [int(c) for c in (p.get("force_columns") or [])]
    good_cols = [] if use_provided else [int(c) for c in (p.get("good_columns") or [])]
    items = [(sag[i], active[i], res, corr_factor, bad_cols, good_cols, max_disp,
              clip_cols_list[i], clip_fit_list[i], zero_cols_list[i]) for i in range(n)]
    disp_field = np.array(_map_slices(_disp_worker, items, progress, 0.5, 0.9, workers))  # (n_slices, n_frames)
    # SURFACE-CROP safety (provided_edges only): a reconstructed posterior-continuity column whose apex is
    # ABOVE the frame (provided edge < 0) must never be shifted UP — there's no acquired tissue above row 0,
    # so a negative shift would truncate real epithelium off the top. Clamp disp >= 0 exactly at those
    # above-frame columns. Confined to the provided path so the normal/legacy detect path is byte-unchanged.
    if use_provided:
        neg = edges < 0.0
        if neg.any():
            disp_field = np.where(neg, np.maximum(disp_field, 0.0), disp_field)
    # The per-pass metric is the mean per-column deviation of the boundary from its quadratic fit (the
    # iterative-refinement convergence signal + abs_floor calibration) — measured on the PRE-smoothing
    # field so its meaning is unchanged by #3's inter-slice smoothing (which only affects the warp).
    disp_mean = float(np.mean(np.abs(disp_field))) if disp_field.size else 0.0

    # 3a-apex) APEX DE-TEAR (#apex): smooth the displacement field along the FRAME axis (within each slice)
    #   with a small Gaussian. The warp injects any high-frequency STEP in the detected edge (disp = quad -
    #   edge) into the tissue; at the bright-speckle apex a ~5-6px edge step tears a V-notch. A light frame-
    #   axis smoothing removes that 1-3-frame injected step while the slowly-varying parabolic bulk warp is
    #   preserved (a well-detected column sits on its own quad → disp≈0 → no-op there). Applied BEFORE the
    #   over-correction backstop is done (disp_field already guarded) and BEFORE interslice smoothing so the
    #   two Gaussians (frame + lateral) compose. Skipped for provided_edges (the marched surface is exact).
    #   Clip/cut tissue-preservation clamps are re-asserted after (a clipped apex must not be shifted UP).
    afs = 0.0 if use_provided else float(p.get("apex_frame_smooth", 0.0) or 0.0)
    if afs > 0 and disp_field.shape[1] > 2:
        dsm = ndimage.gaussian_filter1d(disp_field.astype(np.float64), sigma=afs, axis=1, mode="nearest")
        # GATE (regression fix): apply the frame smoothing ONLY where the raw per-column displacement has a
        # genuine frame-direction JUMP/JITTER (the detector-jump that TEARS the apex into a V-notch on F/I) —
        # NOT everywhere. On a well-detected scan the displacement is already smooth along frames (deviation
        # from its 5-frame median < apex_smooth_gate px), so w=0 → strict NO-OP → the good-scan output is
        # byte-restored to the un-smoothed flatten. Only jittery (torn) apex columns are blended to the smooth
        # field, feathered by how far they exceed the gate — so the tear is removed without perturbing clean scans.
        _gate = float(p.get("apex_smooth_gate", 3.0))
        _med = ndimage.median_filter(disp_field.astype(np.float64), size=(1, 5), mode="nearest")
        _w = np.clip((np.abs(disp_field - _med) - _gate) / max(_gate, 1e-6), 0.0, 1.0)
        disp_field = disp_field.astype(np.float64) * (1.0 - _w) + dsm * _w
        for i in range(n):
            cc = clip_cols_list[i]
            if cc.size:
                disp_field[i][cc] = np.maximum(disp_field[i][cc], 0.0)
            zc = zero_cols_list[i]
            if zc.size:
                disp_field[i][zc] = 0.0

    # 3b) axial consistency (#3): smooth the displacement FIELD across the slice (lateral) axis so
    #     neighbouring sagittal slices shift consistently → a smoother en-face/axial boundary. The
    #     depth/frame axis is untouched (the per-slice quadratic governs it); sigma=0 → per-slice field.
    ism = 0.0 if use_provided else float(p.get("interslice_smooth", 0.0) or 0.0)
    if ism > 0 and n > 2:
        disp_field = ndimage.gaussian_filter1d(disp_field.astype(np.float64), sigma=ism, axis=0)
        # The inter-slice Gaussian re-mixes neighbouring slices, which can pull a clipped/cut column's shift
        # back below the per-slice tissue-preservation clamps applied in _disp_worker (clip: disp>=0 so an
        # above-frame epithelium apex isn't truncated; user-cut: disp==0). Re-assert them per slice so
        # smoothing can't silently re-introduce epithelial truncation on clipped eyes. (Ordinary columns —
        # the vast majority — keep the full inter-slice smoothing benefit.)
        for i in range(n):
            cc = clip_cols_list[i]
            if cc.size:
                disp_field[i][cc] = np.maximum(disp_field[i][cc], 0.0)
            zc = zero_cols_list[i]
            if zc.size:
                disp_field[i][zc] = 0.0

    # 3c) LOGICAL PERIPHERAL CORRECTION (refine_freeze_frac, PER-SCAN opt-in, default 0 = off): at the low-signal
    #     LIMBUS (outer refine_freeze_frac of lateral slices) the per-slice surface detection is unreliable, so a
    #     naive warp tears single-column streaks / wide spikes into the boundary. Instead of trusting the noisy
    #     local detection there, warp the limbus to a LATERALLY-SMOOTH surface: replace the peripheral displacement
    #     with a lateral-gaussian-smoothed version (a smooth continuation of the reliable central dome), feathered
    #     into the precise per-slice warp in the centre. Applied to ALL passes of an opted-in scan (pass 1 +
    #     refinement + axial) so the whole limbus is smoothly corrected — not frozen, not torn. Warped ONCE by the
    #     blended field, so there is no ghosting/seam. Off by default → global pipeline byte-identical.
    if not use_provided:
        _ff = float(p.get("refine_freeze_frac", 0.0) or 0.0)
        if _ff > 0 and n > 20:
            _edge = max(1, int(round(n * _ff)))
            # FIX limbus (a): the lateral peripheral blend smooths the DISPLACEMENT field toward the reliable
            # interior, but with sigma=_edge/3 and mode="nearest" the OUTERMOST few slices are still dominated
            # by their own noisy displacement (the boundary padding reflects the edge value), so a bad extreme-
            # lateral slice keeps its wobble. `refine_edge_sigma_mult` (default 2.0; 1.0 = original behaviour)
            # scales this lateral sigma; a value >1 ties the extreme limbus more strongly to the interior, damping the
            # residual peripheral wobble. `mode="reflect"` (was "nearest") stops the edge slice's own value
            # from being over-weighted at the boundary. Both no-ops at their defaults (mult=1 keeps the sigma;
            # reflect vs nearest is negligible in the interior where w<1), so central well-detected slices and
            # the regression scans are essentially unchanged.
            _sig = max(4.0, _edge / 3.0) * float(p.get("refine_edge_sigma_mult", 1.0) or 1.0)
            _sm = ndimage.gaussian_filter1d(disp_field.astype(np.float64), sigma=_sig, axis=0, mode="reflect")
            _feath = min(30, max(5, _edge // 2))
            _w = np.zeros(n, dtype=np.float64)             # w=1 → smoothed limbus, 0 → precise centre
            _w[:_edge] = 1.0
            _w[-_edge:] = 1.0
            _w[_edge:_edge + _feath] = np.linspace(1.0, 0.0, _feath)
            _w[-_edge - _feath:-_edge] = np.linspace(0.0, 1.0, _feath)
            disp_field = disp_field * (1.0 - _w[:, None]) + _sm * _w[:, None]
            for i in range(n):                             # re-assert tissue-preservation clamps after smoothing
                cc = clip_cols_list[i]
                if cc.size:
                    disp_field[i][cc] = np.maximum(disp_field[i][cc], 0.0)
                zc = zero_cols_list[i]
                if zc.size:
                    disp_field[i][zc] = 0.0
    # 3d) FRAME-EDGE OVER-DESCENT CAP (#edgecap, v149): the flatten quadratic extrapolates the dome parabola and
    #     pushes the first/last acquisition-edge frames ~10px DEEPER than the reliable interior trend (the cornea
    #     flattens at the limbus, it is not a parabola there) — the user's recurring "edges too downward". LIFT an
    #     over-descended edge back to the interior trend, one-sided (never pushes down → cannot recreate the
    #     retired parabola_edge margin shelf), gated + feathered (a slice already on-trend is a strict no-op).
    if not use_provided:
        disp_field = _cap_edge_descent(disp_field, active, clip_cols_list, zero_cols_list, p, vol=det)
    # 4) warp each slice by its guarded+smoothed displacement, then revert. sub-pixel (subpixel_warp) removes the
    #    int-truncate lateral staircase in the flattened anterior boundary (the "ripples" seen at zoom).
    _subpx = bool(p.get("subpixel_warp", False))
    warped = np.array([_warp_by_displacement(sag[i], disp_field[i], subpixel=_subpx) for i in range(n)])
    if progress:
        progress(1.0)
    corrected = revert_sagittal(warped)
    if return_metric:
        # disp_mean (deviation from fit, pre-smoothing) + axial roughness of the DETECTED boundary (the
        # en-face jaggedness the keep-best selection should also minimise, #3).
        return corrected, disp_mean, _axial_roughness(edges)
    return corrected


def _boundary_deviation(volume: np.ndarray, params: dict | None = None,
                        workers: int | None = None, detect_volume: np.ndarray | None = None):
    """Score a candidate volume's boundary quality on its own terms (no warp kept). Returns
    (in_plane_deviation, axial_roughness): the mean per-column deviation of the DETECTED boundary from
    its quadratic fit (how jagged WITHIN each sagittal slice), and the mean inter-slice first-difference
    (how jagged ACROSS slices = the en-face/axial 'hairiness', #3). Both in pixels; lower = better."""
    _, m, ax = smooth_volume(volume, params, workers=workers, return_metric=True, detect_volume=detect_volume)
    return float(m), float(ax)


def iterate_smooth_volume(volume: np.ndarray, params: dict | None = None,
                          max_iter: int = 5, min_improvement: float = 0.15,
                          abs_floor: float = 0.3, progress=None, workers: int | None = None,
                          inject_pass: int | None = None, inject_force=None, inject_good=None,
                          axial_weight: float = 0.5, clip_report: dict | None = None):
    """Iteratively re-apply smooth_volume to its own output, then KEEP THE BEST pass — the one whose
    detected corneal boundary deviates LEAST from a smooth fit (lowest "boundary deviation", px).

    Why keep-the-best rather than keep-the-last: each pass warps the boundary toward its quadratic
    fit, so the deviation usually SHRINKS pass over pass — but a pass can OVERSHOOT and produce a
    MORE deviant (worse) boundary than an earlier pass or even than the raw original (re-detection on
    an over-warped volume picks up a jagged edge). So we score EVERY candidate volume's deviation and
    select the minimum: a worse pass is never kept, and the result can never be more deviant than the
    raw input (raw is in the candidate set). This is the user's "compare so the subsequent border is
    not a more extreme deviation than the original".

    The search stops early (no more passes) once the deviation stops improving — it GREW vs the prior
    pass (overshoot), improved by < min_improvement (diminishing), fell below abs_floor (converged),
    or hit max_iter. But the FINAL choice is always argmin over all measured candidates.

    Returns (chain, best_idx, info): chain = [V0(raw), V1, …, Vm] every measured volume (for the UI
    pass-stepper); best_idx = index of the kept volume; info = {passes (corrected passes produced =
    len(chain)-1), best_pass, metrics (deviation px of each chain volume), stopped}."""
    max_iter = max(1, int(max_iter))
    # The iteration applies a manual column fix PER-PASS only (the user's "fix columns for a particular
    # iteration"): force_columns/good_columns are NOT global params here — they're injected at exactly
    # inject_pass (1-based) and absent on every other pass.
    base = dict(params or {})
    base.pop("force_columns", None)
    base.pop("good_columns", None)
    chain: list = [volume]       # V0 = raw, then each accepted pass
    rough: list = []             # rough[i] = in-plane boundary deviation of chain[i] (convergence signal)
    axial: list = []             # axial[i] = en-face/axial roughness of chain[i] (#3, folded into select)
    stopped = "max_iter"
    _clip_carry = None           # pass-0 clipped columns, carried to passes ≥1 (which can't re-detect a clip
                                 # on a warped+filled volume) so they keep extrapolating + never re-truncate
    for k in range(max_iter):
        lo = k / max_iter
        hi = (k + 1) / max_iter
        pp = dict(base)
        if inject_pass is not None and (k + 1) == int(inject_pass):
            pp["force_columns"] = [int(c) for c in (inject_force or [])]
            pp["good_columns"] = [int(c) for c in (inject_good or [])]
        # A re-fed pass (k>=1) runs on the PREVIOUS pass's warped output, whose black padding would
        # fool the edge detector into 100-360px runaway shifts — DETECT on a filled copy. But WARP the
        # real (unfilled) chain[k], so the output never carries the fill's fake pixels (only honest
        # zero padding). Pass 1 runs on raw with no fill → byte-identical to the faithful single pass.
        det = None if k == 0 else _fill_black_bands(chain[k])
        # Detect the clip ONLY on pass 0 (raw acquisition). Capture its clipped columns (into the caller's
        # clip_report when given, else a local dict) and carry them forward as fixed_clip_cols on passes ≥1.
        _cr = (clip_report if clip_report is not None else {}) if k == 0 else None
        nxt, r, ax = smooth_volume(chain[k], pp, progress=(
            (lambda f, lo=lo, hi=hi: progress(lo + (hi - lo) * f)) if progress else None),
            workers=workers, return_metric=True, detect_volume=det,   # r/ax = in-plane/axial of chain[k]
            clip_report=_cr, fixed_clip_cols=(_clip_carry if k >= 1 else None))
        if k == 0 and _cr is not None:
            _clip_carry = _cr.get("_clip_cols")                      # reuse these clipped columns on later passes
        rough.append(float(r)); axial.append(float(ax))
        # Force the iteration to REACH (and keep) the injected pass — never early-stop before it, or
        # the user's per-pass column fix would be silently discarded. Past the inject pass, the normal
        # keep-best stop logic resumes.
        force_reach = inject_pass is not None and (k + 1) <= int(inject_pass)
        # Stop producing more passes once the boundary stops getting smoother (but we've still
        # MEASURED chain[k], so it stays a candidate for the argmin below).
        if not force_reach and k >= 1:
            if r >= rough[k - 1]:
                stopped = "grew"; break          # chain[k] is MORE deviant than chain[k-1]
            if (rough[k - 1] - r) / max(rough[k - 1], 1e-9) < min_improvement:
                stopped = "diminishing"; break
        if not force_reach and r < abs_floor:
            stopped = "converged"
            chain.append(nxt)                    # a final tiny refinement is safe; keep + measure it
            break
        chain.append(nxt)                        # accept the next pass into the chain
    # Make sure EVERY chain volume has a measured deviation so it can compete in the argmin (the last
    # accepted pass is otherwise unmeasured when we stop by max_iter / converged).
    while len(rough) < len(chain):
        idx = len(rough)
        det = None if idx == 0 else _fill_black_bands(chain[idx])
        dev, ax = _boundary_deviation(chain[idx], base, workers=workers, detect_volume=det)
        rough.append(dev); axial.append(ax)
    # KEEP-THE-BEST by a COMBINED score: in-plane deviation + axial_weight × en-face/axial roughness
    # (#3). A pass that flattens each sagittal slice but leaves a HAIRIER axial boundary now loses to a
    # more axially-consistent pass — the old pure-in-plane argmin even preferred the hairiest pass.
    score = [rough[i] + axial_weight * axial[i] for i in range(len(chain))]
    best_idx = min(range(len(chain)), key=lambda i: score[i])
    info = {"passes": len(chain) - 1, "best_pass": best_idx,
            "metrics": [float(x) for x in rough], "axial_metrics": [float(x) for x in axial],
            "scores": [float(x) for x in score], "stopped": stopped}
    return chain, best_idx, info


# ── Ping-pong: axial correction after sagittal, for the hairy frames only (#2) ──────────────────────
# The sagittal correction flattens the boundary ALONG FRAMES (independently per lateral slice), so it
# leaves roughness ACROSS LATERAL — the en-face/"axial" boundary can look hairy where the sagittal slice
# was noisy at its ends. Running the SAME correction in the axial domain (flatten ALONG LATERAL, per
# frame) cleans those up. Empirically (real Avanti scans) a SINGLE axial pass after the sagittal one is
# the smoothest 3D surface; more ping-pong passes over-correct. Applying the axial result PER FRAME only
# where it actually reduces that frame's lateral roughness ("hairy frames only") is best + can't regress.
_FRAME_LATERAL_SWAP = (2, 1, 0)  # frames<->lateral (depth stays axis 1); makes axial slices the warp slices


def _axial_smooth_volume(volume: np.ndarray, params: dict | None, workers: int | None) -> np.ndarray:
    """Run smooth_volume in the AXIAL domain (flatten the boundary along the LATERAL axis, per frame) by
    swapping frames<->lateral, correcting, swapping back. Detects on a black-band-filled copy so the
    prior sagittal warp's padding can't fool the detector."""
    vt = np.ascontiguousarray(volume.transpose(*_FRAME_LATERAL_SWAP))
    # surface_cut / force_columns / good_columns / border_anchors are all defined in the SAGITTAL frame
    # domain; after the frame<->lateral swap their indices address the wrong axis, so strip them from the
    # axial pass (the sagittal pass already applied them). The axial pass runs a clean auto correction.
    _SAG_DOMAIN_KEYS = ("surface_cut", "force_columns", "good_columns", "border_anchors")
    pax = {k: v for k, v in (params or {}).items() if k not in _SAG_DOMAIN_KEYS}
    out = smooth_volume(vt, pax, workers=workers, detect_volume=_fill_black_bands(vt))
    return np.ascontiguousarray(out.transpose(*_FRAME_LATERAL_SWAP))


def _frame_boundary_surface(volume: np.ndarray, params: dict, workers: int | None) -> np.ndarray:
    """The corneal boundary B(frame, lateral) detected per FRAME (axial B-scan = depth×lateral) on a
    black-band-filled copy (so the warp padding can't fool detection). Shape (n_frames, n_lateral)."""
    vf = _fill_black_bands(volume)
    res = _map_slices(_edge_worker, [(vf[f], params) for f in range(vf.shape[0])], None, 0.0, 1.0, workers)
    return np.array([(r[0] if isinstance(r, tuple) else r) for r in res])


def _surface_rms(B: np.ndarray) -> float:
    """RMS deviation of the boundary surface from a smooth 2-D quadratic fit (3-D smoothness; lower=better)."""
    if B.ndim != 2 or B.size < 6:
        return 0.0
    ff, ll = np.mgrid[0:B.shape[0], 0:B.shape[1]].astype(float)
    A = np.c_[np.ones(B.size), ff.ravel(), ll.ravel(), ff.ravel() ** 2, ll.ravel() ** 2, (ff * ll).ravel()]
    coef, *_ = np.linalg.lstsq(A, B.ravel(), rcond=None)
    return float(np.sqrt(np.mean((B.ravel() - A @ coef) ** 2)))


def axial_refine_volume(v_sag: np.ndarray, params: dict | None = None, workers: int | None = None):
    """#2 ping-pong refine: after the sagittal correction, run an axial pass and KEEP it PER FRAME only
    where it reduces that frame's lateral boundary roughness (the user's 'axial correction for hairy
    axial slices'). A global guard then accepts the blend only if the whole 3-D surface got smoother — so
    this can never produce a worse surface than sagittal-only. Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    v_ax = _axial_smooth_volume(v_sag, p, workers)
    B_sag = _frame_boundary_surface(v_sag, p, workers)
    B_ax = _frame_boundary_surface(v_ax, p, workers)
    tvl_sag = np.mean(np.abs(np.diff(B_sag, axis=1)), axis=1)   # per-frame lateral roughness (sagittal)
    tvl_ax = np.mean(np.abs(np.diff(B_ax, axis=1)), axis=1)     # per-frame lateral roughness (axial)
    use = tvl_ax < tvl_sag                                       # frames the axial pass actually improved
    out = v_sag.copy()
    out[use] = v_ax[use]
    B_out = np.where(use[:, None], B_ax, B_sag)                  # blended surface (no re-detect needed)
    rms_before, rms_after = _surface_rms(B_sag), _surface_rms(B_out)
    if use.any() and rms_after <= rms_before:                    # global guard: only accept a smoother surface
        return out, {"frames_refined": int(use.sum()), "n_frames": int(B_sag.shape[0]),
                     "surf_rms_before": rms_before, "surf_rms_after": rms_after, "applied": True}
    return v_sag, {"frames_refined": 0, "n_frames": int(B_sag.shape[0]),
                   "surf_rms_before": rms_before, "surf_rms_after": rms_before, "applied": False}


# ── AXIAL-CONSISTENCY pass (# FIX axialcons) ────────────────────────────────────────────────────────
# The sagittal flatten corrects each of the 513 lateral columns independently (only 101 frame samples
# each), so their depth shifts are inconsistent → the anterior surface WAVES / spikes / notches across
# the LATERAL axis, visible ONLY in the AXIAL (B-scan) view. This pass takes the SAME per-sagittal-slice
# surface the label/validation uses (detect_surface_all → S[lateral, frame]); for each FRAME it laterally
# SMOOTHS that surface and applies a SMALL gated per-lateral-column DEPTH shift toward the smooth target,
# so the RED (pipeline) surface becomes laterally consistent. It NEVER re-flattens (max_shift clamp) and
# is a strict NO-OP where the surface is already smooth (gate) — so it can't regress good scans.
def _axcons_shift_from_edge(edge: np.ndarray, depth: int, p: dict) -> np.ndarray:
    """Given ONE frame's lateral surface profile `edge` (length = lateral; the detect_surface_all column for
    this frame) return the GATED per-lateral-column depth shift that pulls each jittery column onto the
    laterally-smoothed target. Returns 0 everywhere if the frame is already smooth / off-cornea (no-op)."""
    edge = np.asarray(edge, dtype=np.float64)
    L = edge.size
    sigma = float(p.get("axcons_sigma", 8.0) or 0.0)
    gate = float(p.get("axcons_gate", 2.0) or 0.0)
    max_shift = float(p.get("axcons_max_shift", 6.0) or 0.0)
    strength = float(p.get("axcons_strength", 1.0) or 0.0)
    min_frac = float(p.get("axcons_min_frac", 0.02) or 0.0)
    max_frac = float(p.get("axcons_max_frac", 0.25) or 1.0)
    min_cov = float(p.get("axcons_min_coverage", 0.5) or 0.0)
    if sigma <= 0 or max_shift <= 0 or strength <= 0 or L < 8:
        return np.zeros(L, dtype=np.float64)
    # A column with no cornea (all padding → edge at 0 or the frame top) must not vote or move. Treat only
    # in-frame, positive edges as valid; smooth the target over the VALID columns only (so off-cornea
    # limbus/background can't drag the smooth target).
    valid = np.isfinite(edge) & (edge > 1.0) & (edge < depth - 1)
    # COVERAGE GATE: a TRAILING/off-cornea frame (the slow scan ran off the eye) has the cornea covering only
    # a small part of the lateral span; its detected surface is not a smooth dome, so smoothing it makes a
    # meaningless target and forcing columns onto it tears tissue. Require the cornea to fill most of the
    # frame — otherwise leave the frame exactly as the sagittal pass left it (no-op). This is what keeps the
    # off-cornea limbus frames (the source of the regression) untouched.
    if int(valid.sum()) < max(8, int(min_cov * L)):
        return np.zeros(L, dtype=np.float64)
    xs = np.arange(L, dtype=np.float64)
    e_valid = np.interp(xs, xs[valid], edge[valid])           # fill invalid columns by lateral interpolation
    # TARGET = a robust lateral-smoothed surface: a wide MEDIAN filter (kills narrow spikes, keeps broad dome
    # curvature) FOLLOWED by a Gaussian (removes the median's own staircase). A genuinely smooth dome (any
    # width) reads deviation ≈ 0 → no-op; only the per-slice detector's narrow jitter deviates.
    med_win = int(p.get("axcons_med_win", 15) or 15)
    if med_win % 2 == 0:
        med_win += 1
    target = ndimage.median_filter(e_valid, size=med_win, mode="nearest")
    if sigma > 0:
        target = ndimage.gaussian_filter1d(target, sigma=sigma, mode="nearest")
    dev = e_valid - target                                   # + = surface sits DEEPER than the smooth target (a down-notch)
    # ONLY-DOWN option: the sagittal detector jitter that produces the AXIAL notches is a column locking a few
    # px too DEEP (into internal stroma) — a downward excursion. Correcting only downward notches (never
    # pushing a column deeper) avoids fighting a genuine shallow dome apex. axcons_two_sided=True corrects both.
    if not bool(p.get("axcons_two_sided", False)):
        dev = np.maximum(dev, 0.0)
    # GATE = SOFT-THRESHOLD: leave a gate-width dead-band untouched (a small wobble ≤ gate is genuine dome
    # micro-texture, NOT jitter → strict no-op), but apply the FULL excess beyond it so a real notch/spike
    # is pulled all the way onto the smooth target. corr = sign(dev)·max(|dev|−gate, 0).
    mag = np.abs(dev)
    excess = np.maximum(mag - gate, 0.0)
    corr = np.sign(dev) * excess
    corr[~valid] = 0.0                                        # never move an off-cornea / interpolated column
    n_over = float(np.count_nonzero(corr != 0.0))
    # Whole-frame no-op unless a real RUN of columns exceeds the gate (ignore a few stray specks → clean
    # frames untouched). A well-detected, laterally-smooth scan has 0 columns over the gate → strict no-op.
    if n_over <= min_frac * L:
        return np.zeros(L, dtype=np.float64)
    # MAX-FRAC GATE: genuine lateral JITTER is a SMALL fraction of spike columns on an otherwise-smooth dome.
    # If a LARGE fraction of columns deviate, the frame is fundamentally rough / off-cornea / mis-detected —
    # NOT jitter — so the smooth target is untrustworthy and flattening to it would tear tissue (this was the
    # regression on the trailing limbus frames). Leave such a frame untouched.
    if n_over > max_frac * max(1, int(valid.sum())):
        return np.zeros(L, dtype=np.float64)
    # shift = move each column toward the smooth target (depth). disp>0 shifts tissue DOWN (deeper); to pull
    # a too-deep column (dev>0) UP we need a NEGATIVE depth shift, so shift = -strength·corr, clamped small.
    shift = -strength * corr
    shift = np.clip(shift, -max_shift, max_shift)
    shift = np.round(shift)                                   # the warp truncates to int; round so a sub-px notch still moves
    return shift.astype(np.float64)


def axial_consistency_volume(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """FINAL lateral clean-up (# FIX axialcons): using the SAME per-sagittal-slice surface the label uses
    (detect_surface_all), apply a small GATED per-lateral-column depth shift per frame so the anterior
    surface is laterally consistent (no wave/spike/notch in the AXIAL view), keeping the sagittal flatten.
    Strict no-op on an already-smooth scan (gate). Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    n_frames = int(volume.shape[0])
    depth = int(volume.shape[1])
    iters = max(1, int(p.get("axcons_iters", 2) or 1))
    out = volume.copy()
    moved = np.zeros(n_frames, dtype=bool)
    n_moved_cols = 0
    # Iterate detect→nudge: a deep notch needs a couple of passes (the detector re-locks slightly after a
    # warp). Each pass re-detects the label surface on the CURRENT volume, so a frame that has become smooth
    # returns an all-zero shift and stops moving — the loop self-terminates on a good scan (strict no-op).
    for _ in range(iters):
        # detect on a black-band-filled copy so the sagittal warp's zero padding can't fool detection; use
        # the EXACT surface the label/validation reads (detect_surface_all → S[lateral, frame]).
        surf = detect_surface_all(reformat_to_sagittal(_fill_black_bands(out)), p, workers=workers)  # (lateral, frames)
        any_move = False
        for f in range(n_frames):
            sh = _axcons_shift_from_edge(surf[:, f], depth, p)
            if not np.any(sh != 0.0):
                continue
            # warp the B-scan (depth × lateral) by the per-lateral-column depth shift — SAME primitive as the
            # sagittal warp (_warp_by_displacement warps rows=depth by a per-column shift), applied laterally.
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), sh)
            moved[f] = True
            n_moved_cols += int(np.count_nonzero(sh != 0.0))
            any_move = True
        if not any_move:
            break
    n_moved_frames = int(moved.sum())
    return out, {"applied": bool(n_moved_frames), "frames_adjusted": n_moved_frames,
                 "n_frames": n_frames, "cols_adjusted": int(n_moved_cols)}


def axial_motion_correct(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """Correct slow-scan (frame-axis) inter-frame AXIAL EYE MOTION (v0.0.159).

    During the slow scan across frames the eye drifts/saccades AXIALLY, so each B-scan (frame) is acquired at
    a slightly different depth. The anterior surface then shows per-frame STEPS/WAVES across frames — a
    sagittal surface that is "obviously not smooth" — even though each individual B-scan is internally fine
    (measured CS004 OD rep1: motion 5.8px std, 89% of it a PER-FRAME RIGID displacement uniform across
    lateral, ±15px, with a physical drift+saccade trajectory; sibling replicates rep2/rep3 were motion-free).

    Model + fix: fit a robust smooth 3-D corneal dome T(lat,frame) (deg-`amc_dome_deg` 2-D poly, iterative
    2σ reject so the ±15px motion does not bias the fit); the per-frame RIGID motion is M(frame) = median
    over lateral of (S−T); rigidly shift each B-scan in depth by −M so its surface lands on the smooth dome.
    Because the shift is per-frame UNIFORM across lateral it corrects the sagittal steps WITHOUT roughening
    the en-face view (the failure mode of per-column smoothing). M is zero-centred (minimal canvas
    truncation) and a STRICT NO-OP when the motion is small (std < amc_min_motion) — motion-free scans
    (rep2/rep3, all approved) are byte-unchanged. Runs EARLY (before de-tilt/flatten) so the flatten sees
    de-motioned data. `volume` is (frames, depth, lateral). Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("axial_motion_correct", True)):
        return volume, {"applied": False}
    if workers is None:
        workers = auto_workers()
    F, depth = int(volume.shape[0]), int(volume.shape[1])
    try:
        S = detect_surface_all(reformat_to_sagittal(volume), p, workers=workers)  # (lat, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    L = int(S.shape[0])
    valid = (S > 1.0) & np.isfinite(S) & (S < depth - 1)
    if int(valid.sum()) < (L * F) // 4 or F < 12:
        return volume, {"applied": False}
    yy, xx = np.mgrid[0:L, 0:F].astype(np.float64)
    yn = yy / (L - 1) * 2 - 1; xn = xx / (F - 1) * 2 - 1
    deg = int(p.get("amc_dome_deg", 5))
    terms = [(a, b) for a in range(deg + 1) for b in range(deg + 1) if a + b <= deg]
    A = np.stack([(yn ** a) * (xn ** b) for a, b in terms], axis=-1)
    m = valid.copy()
    try:
        coef = np.linalg.lstsq(A[m], S[m], rcond=None)[0]
        for _ in range(4):
            r = S - A @ coef; sd = np.std(r[m]) + 1e-6
            mm = valid & (np.abs(r) < 2.0 * sd)
            if int(mm.sum()) < len(terms) + 5 or int(mm.sum()) == int(m.sum()):
                break
            m = mm; coef = np.linalg.lstsq(A[m], S[m], rcond=None)[0]
        T = A @ coef
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    dev = np.where(valid, S - T, np.nan)
    with np.errstate(all="ignore"):
        M = np.nanmedian(dev, axis=0)                       # per-frame rigid motion (frames,)
    M = np.where(np.isfinite(M), M, 0.0)
    M = ndimage.gaussian_filter1d(M, float(p.get("amc_smooth", 1.0)), mode="nearest")  # kill 1-frame noise
    M = M - np.median(M)                                     # zero-centre → minimal canvas truncation
    M = np.clip(M, -float(p.get("amc_max_shift", 30.0)), float(p.get("amc_max_shift", 30.0)))
    if float(np.std(M)) < float(p.get("amc_min_motion", 1.0)):   # motion-free → strict no-op
        return volume, {"applied": False, "motion_std": round(float(np.std(M)), 2)}
    out = volume.copy(); nadj = 0
    for f in range(F):
        if abs(M[f]) > 0.05:
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), np.full(L, -M[f]), subpixel=True)
            nadj += 1
    return out, {"applied": bool(nadj), "frames_adjusted": int(nadj),
                 "motion_std": round(float(np.std(M)), 2), "max_shift": round(float(np.max(np.abs(M))), 1)}


def intra_frame_dewarp(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """Correct INTRA-frame (within-B-scan) saccade distortion (v0.0.159) — the hard residual after the flatten.

    The per-slice flatten removes the per-frame RIGID axial motion, but it cannot touch distortion that
    happens DURING a single B-scan: at a saccade the eye moves mid-lateral-sweep, so that ONE B-scan's
    anterior surface is warped in the LATERAL direction (a ramp/step across lateral) away from the true smooth
    corneal shape (measured CS004 OD rep1: within-frame lateral spread spikes 1px→4-5px only at the saccade
    frames 20/42/58/78). This shows as a "not smooth" sagittal surface even though the neighbouring B-scans
    are clean. Fix: re-warp each column onto the smooth 3-D corneal surface, but derived so it aligns the ACTUAL
    tissue and only the genuinely-distorted frames move.

    Method (the crux — use the RAW band edge, not the post-processed surface): detect the anterior surface with
    the shape post-processors OFF (robust_dome / dip2d / lat_conf / despike / parabola off) so it tracks the
    true bright-band edge the warp will move. Build the motion-free reference T(lat,frame) = each lateral's
    surface robustly smoothed ALONG frames (median+gaussian, so a saccade-distorted frame is an outlier the
    smoother rejects → T at that frame is interpolated from the clean neighbours). Deviation dev = S − T is the
    intra-frame distortion; per-frame distortion level Dframe = robust lateral spread of dev. GATE by Dframe so
    only the saccade frames (Dframe > thresh) are corrected — a clean scan has no such frame → strict NO-OP
    (rep2/rep3, approved scans unchanged). The shift is smoothed across lateral (coherent B-scan re-warp, no
    en-face jag) and bounded. Runs EARLY (before de-tilt/flatten) so the flatten sees de-distorted B-scans.
    `volume` is (frames, depth, lateral). Returns (volume, info). intra_frame_dewarp=False disables."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("intra_frame_dewarp", True)):
        return volume, {"applied": False}
    if workers is None:
        workers = auto_workers()
    F, depth = int(volume.shape[0]), int(volume.shape[1])
    praw = {**p, "despike_lateral": False, "dip2d_suppress": False, "robust_dome": False,
            "lat_conf_smooth": False, "parabola_edge": False, "boundary_extrap_nb": 0, "frame_edge_cap": False}
    try:
        S = detect_surface_all(reformat_to_sagittal(volume), praw, workers=workers)  # RAW band edge (lat, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    L = int(S.shape[0])
    valid = (S > 1.0) & np.isfinite(S) & (S < depth - 1)
    if int(valid.sum()) < (L * F) // 4 or F < 12:
        return volume, {"applied": False}
    Sf = S.astype(np.float64)
    for f in range(F):                                   # fill invalid laterals per frame (interp) so smoothing is clean
        mm = valid[:, f]
        if int(mm.sum()) >= 8:
            Sf[~mm, f] = np.interp(np.where(~mm)[0], np.where(mm)[0], Sf[mm, f])
    fwin = int(p.get("ifd_frame_med", 7));  fwin += (fwin % 2 == 0)
    fg = float(p.get("ifd_frame_gauss", 2.0))
    # motion-free reference: each lateral's frame-trace robustly smoothed (rejects the distorted-frame outliers)
    T = ndimage.median_filter(Sf, size=(1, fwin), mode="nearest")
    if fg > 0:
        T = ndimage.gaussian_filter(T, sigma=(0.0, fg), mode="nearest")
    dev = np.where(valid, Sf - T, 0.0)
    # per-frame distortion level = robust lateral spread of the deviation (a tilted/warped B-scan reads high)
    with np.errstate(all="ignore"):
        Dframe = np.array([np.nanstd(np.where(valid[:, f], dev[:, f], np.nan)) for f in range(F)])
    Dframe = np.where(np.isfinite(Dframe), Dframe, 0.0)
    thr = float(p.get("ifd_frame_thresh", 2.0)); soft = float(p.get("ifd_frame_soft", 1.0))
    gate = np.clip((Dframe - thr) / max(1e-3, soft), 0.0, 1.0)     # 0 on clean frames → no-op
    if float(np.max(gate)) <= 0.0:
        return volume, {"applied": False, "max_frame_distortion": round(float(np.max(Dframe)), 2)}
    sig_lat = float(p.get("ifd_lat_smooth", 8.0)); cap = float(p.get("ifd_max_shift", 20.0))
    shift = -dev * gate[None, :]
    shift[:, gate > 0] = ndimage.gaussian_filter1d(shift[:, gate > 0], sigma=sig_lat, axis=0, mode="nearest")
    shift = np.clip(shift, -cap, cap); shift[~valid] = 0.0
    out = volume.copy(); nadj = 0
    for f in range(F):
        if np.any(shift[:, f] != 0.0):
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), shift[:, f], subpixel=True)
            nadj += 1
    return out, {"applied": bool(nadj), "frames_adjusted": int(nadj),
                 "max_frame_distortion": round(float(np.max(Dframe)), 2),
                 "saccade_frames": int(np.sum(gate > 0.3))}


def frame_boundary_lat_smooth(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """FIX the jagged border on the first/last B-scans (acquisition-edge frames). Those frames are LOW-SIGNAL, so
    the per-slice anterior detection wiggles laterally → the axial (B-scan) border looks jagged (the marked
    CS002 OS(2)-(4) defect at axial slice 1-2). The sagittal warp flattens WITHIN each slice but does not enforce
    CROSS-SLICE (lateral) consistency, and axial_consistency's "too-rough → skip" gate bails out on these very
    jagged frames — so they stay jagged. Here, ONLY on the first/last `fbls_nb` frames, force lateral consistency:
    take the detected surface across laterals, smooth it hard (wide median kills the jag + Gaussian), and shift
    each column so its border lands on that smooth curve (feathered toward the interior so there is no step). The
    smooth curve is the frame's OWN robust lateral trend, so real curvature is kept; only the jag is removed.
    Strict no-op when fbls_nb<=0 or a frame has too little cornea. Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    nb = int(p.get("fbls_nb", 4) or 0)
    if nb <= 0:
        return volume, {"applied": False, "frames_adjusted": 0}
    med = int(p.get("fbls_med", 31) or 1); gs = float(p.get("fbls_gauss", 12.0) or 0.0)
    cap = float(p.get("fbls_max_shift", 16.0) or 0.0); min_cov = float(p.get("fbls_min_coverage", 0.4) or 0.0)
    S = detect_surface_all(reformat_to_sagittal(_fill_black_bands(volume)), p, workers=workers)  # (lateral, frames)
    L, F = S.shape; depth = int(volume.shape[1])
    if med % 2 == 0:
        med += 1
    out = volume.copy(); nadj = 0
    for f in list(range(min(nb, F))) + list(range(max(0, F - nb), F)):
        a = S[:, f].astype(np.float64); vld = np.isfinite(a) & (a > 1.0) & (a < depth - 1)
        if int(vld.sum()) < max(20, int(min_cov * L)):
            continue
        xs = np.arange(L)
        interp = np.interp(xs, xs[vld], a[vld])
        tgt = ndimage.median_filter(interp, size=med, mode="nearest")       # wide median removes the lateral jag
        if gs > 0:
            tgt = ndimage.gaussian_filter1d(tgt, sigma=gs, mode="nearest")   # + gaussian for a clean smooth curve
        w = 1.0 - (min(f, F - 1 - f) / max(1, nb))                          # full at the very edge frame, taper in
        shift = np.where(vld, np.clip((tgt - a) * w, -cap, cap), 0.0)        # move each column's border onto tgt
        if not np.any(shift != 0.0):
            continue
        out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), shift, subpixel=True)
        nadj += 1
    return out, {"applied": bool(nadj), "frames_adjusted": nadj}


def frame_edge_overdescent_cap(volume: np.ndarray, ref_surface: np.ndarray | None,
                               params: dict | None = None, workers: int | None = None):
    """POST-HOC frame-direction OVER-DESCENT cap — the "very steep curvature near the ends" (user-reported).

    The per-slice flatten target is a RANSAC QUADRATIC across frames; it extrapolates the corneal dome PAST
    the point where the real surface flattens at the acquisition edge (the limbus plateau, or the slow scan
    running off the cornea) and pushes the last/first ~15 frames 10-20px DEEPER than the tissue actually goes
    → a sagittal anterior boundary that plunges into a steep hook the raw does NOT have (verified lat256
    CS003 OD: raw turns flat ~frame88 at depth ~235 while the warped output dives to ~260).

    Runs ONCE on the FINAL volume — NOT inside the iterative flatten (a displacement edit there feeds the next
    iteration's GLOBAL quad fit and leaks into the interior: measured 3.5px mean / 30px max on approved CS002
    OS3). Here only the edge frames are warped, so the interior is untouched by construction — no cascade.

    Per lateral, cap the edge frames' output depth so it never descends more than the REFERENCE (pre-flatten)
    boundary's OWN frame-direction descent + margin, anchored on a clean interior band past the over-descent
    onset: out ≤ S_anchor + (ref_shape − ref_anchor) + margin. Referencing the real tissue's descent
    (ref_surface, detected before the flatten) — not a linear interior extrapolation (the retired v149 EXP,
    which under-predicted a genuinely steep limbus and lifted it into the OD1 upward hook) — keeps a real steep
    periphery and removes only the quad's manufactured extra plunge. Strictly one-sided (lifts DOWN→UP only),
    gated to where BOTH surfaces exist, and bounded by max_shift → it can never invent a hook or shove tissue
    out of frame, and is a strict no-op on a scan with no over-descent. Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("frame_edge_rawcap", True)) or ref_surface is None:
        return volume, {"applied": False, "frames_adjusted": 0}
    if workers is None:
        workers = auto_workers()
    nb = int(p.get("frame_edge_rawcap_nb", 16) or 0)
    if nb <= 0:
        return volume, {"applied": False, "frames_adjusted": 0}
    gap = int(p.get("frame_edge_rawcap_gap", 4)); margin = float(p.get("frame_edge_rawcap_margin", 3.0))
    maxsh = float(p.get("frame_edge_rawcap_max_shift", 14.0)); med = int(p.get("frame_edge_lat_med", 15))
    sig_lat = float(p.get("frame_edge_lat_smooth", 40.0))
    fire = float(p.get("frame_edge_rawcap_fire", 10.0)); ramp = float(p.get("frame_edge_rawcap_ramp", 6.0))
    S = detect_surface_all(reformat_to_sagittal(_fill_black_bands(volume)), p, workers=workers)  # (lat, frames)
    R = np.asarray(ref_surface, dtype=np.float64)
    L, F = S.shape; depth = int(volume.shape[1])
    if R.shape != S.shape or F < 2 * (nb + gap + 8):
        return volume, {"applied": False, "frames_adjusted": 0}

    def smooth_lat(M):     # strong lateral smoothing (match the de-bump) + light frame smoothing of the ref shape
        return ndimage.gaussian_filter(ndimage.median_filter(M, size=(max(1, med), 1), mode="nearest"),
                                       sigma=(sig_lat, 1.5), mode="nearest")

    shifts = np.zeros((F, L), dtype=np.float64)
    for lead in (True, False):
        idx = np.arange(0, nb) if lead else np.arange(F - nb, F)
        aw = np.arange(nb + gap, nb + gap + 8) if lead else np.arange(F - nb - gap - 8, F - nb - gap)
        Rs = smooth_lat(R[:, idx])
        with np.errstate(all="ignore"):
            Ranch = np.nanmedian(np.where(R[:, aw] > 1.0, R[:, aw], np.nan), axis=1)
            Sanch = np.nanmedian(np.where(S[:, aw] > 1.0, S[:, aw], np.nan), axis=1)
        Ranch = ndimage.gaussian_filter(np.nan_to_num(Ranch), sigma=sig_lat, mode="nearest")
        Sanch = ndimage.gaussian_filter(np.nan_to_num(Sanch), sigma=sig_lat, mode="nearest")
        base_ref = Sanch[:, None] + (Rs - Ranch[:, None])               # reference-anchored expected depth
        excess = S[:, idx] - base_ref                                   # over-descent beyond the reference (px)
        # FIRE GATE: only act on a genuine steep over-plunge, NOT a gentle edge whose flatten shift is legitimate
        # (an approved scan's soft dome edge sits a few px below the pre-flatten reference by design). A soft
        # ramp on the over-descent magnitude → gentle edges (small excess) are a strict no-op; the steep od2-type
        # plunge (large excess) is lifted. Smoothed across lateral so the on/off boundary can't add a lateral jag.
        wfire = np.clip((excess - fire) / max(1e-3, ramp), 0.0, 1.0)
        wfire = ndimage.gaussian_filter(wfire, sigma=(sig_lat, 1.0), mode="nearest")
        target = base_ref + margin
        vld = (S[:, idx] > 1.0) & (S[:, idx] < depth - 1) & (R[:, idx] > 1.0) & np.isfinite(base_ref)
        sh = np.where(vld, np.clip(wfire * (target - S[:, idx]), -maxsh, 0.0), 0.0)  # lift only, gated
        for k, f in enumerate(idx):
            shifts[f] = sh[:, k]
    out = volume.copy(); nadj = 0
    for f in range(F):
        if np.any(shifts[f] != 0.0):
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), shifts[f], subpixel=True)
            nadj += 1
    return out, {"applied": bool(nadj), "frames_adjusted": nadj}


def frame_edge_curve_snap(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """POST-HOC CONDITIONAL frame-edge correction to the OVERALL corneal curve (v0.0.157).

    User's criterion (verbatim): "it should be conditional — if the edges are obviously deviating from the
    overall corneal curve it should be corrected"; and a real cornea's curvature gradient near the end has
    "very few if any cases where it does a change in direction … (physically unlikely)". So: the acquisition-
    edge frames sometimes carry stair-steps / gradient-direction REVERSALS where the detected border departs
    from the smooth convex corneal arc — physically implausible; correct THOSE toward the arc, and ONLY those.

    Per slice, fit a robust smooth OVERALL corneal curve (deg-3 poly, iterative 2σ outlier reject → the fit
    represents the reliable cornea and the deviating steps are the rejected outliers; verified stable, no edge
    blow-up because the fit is dominated by the smooth interior). In the first/last `nb` edge frames, measure
    the signed deviation dev = surface − curve; SOFT-GATE by |dev| so a small (on-curve) deviation is a strict
    NO-OP (an approved scan whose edges follow the curve is unchanged — verified near-no-op on CS003 OD rep1);
    where it OBVIOUSLY deviates, warp the column toward the curve (two-sided: an edge bumping ABOVE or plunging
    BELOW the arc is pulled back). A raised-cosine DISTANCE feather takes the correction to 0 at the interior
    boundary of the edge band, and the deviation itself is ~0 there, so the correction can NEVER create a
    boundary step/kink (the failure that retired parabola_edge ⑯ and the v155 cap ⑲ — a kink is itself a
    gradient reversal). A light lateral MEDIAN on the deviation kills isolated per-slice spikes so the snapped
    en-face boundary stays coherent. Only edge frames are warped → interior untouched by construction. A
    tissue-contrast gate skips columns where the scan ran off the cornea (no reliable curve there). Returns
    (volume, info). frame_edge_curve_snap=False disables."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("frame_edge_curve_snap", True)):
        return volume, {"applied": False, "frames_adjusted": 0}
    if workers is None:
        workers = auto_workers()
    nb = int(p.get("frame_edge_snap_nb", 18) or 0)
    thresh = float(p.get("frame_edge_snap_thresh", 4.0))     # px: only correct deviations OBVIOUSLY off the curve
    soft = float(p.get("frame_edge_snap_soft", 3.0))         # px over which the gate ramps 0→1
    maxsh = float(p.get("frame_edge_snap_max_shift", 18.0))
    deg = int(p.get("frame_edge_snap_deg", 3))
    med_lat = int(p.get("frame_edge_snap_lat_med", 9))
    conf_frac = float(p.get("frame_edge_snap_conf_frac", 0.20))
    S = detect_surface_all(reformat_to_sagittal(_fill_black_bands(volume)), p, workers=workers)  # (lat, frames)
    L, F = S.shape; depth = int(volume.shape[1])
    if nb <= 0 or F < 2 * nb + 8:
        return volume, {"applied": False, "frames_adjusted": 0}
    x = np.arange(F)
    eidx = np.r_[np.arange(0, nb), np.arange(F - nb, F)]
    wf = np.zeros(F)                                          # raised-cosine distance feather: 1 at edge → 0 at nb
    for f in range(nb):
        wf[f] = 0.5 * (1.0 + np.cos(np.pi * f / nb))
    for f in range(F - nb, F):
        wf[f] = 0.5 * (1.0 + np.cos(np.pi * (F - 1 - f) / nb))
    dev = np.zeros((L, F)); valid = np.zeros((L, F), dtype=bool)
    for i in range(L):
        s = S[i].astype(np.float64); m = (s > 1.0) & np.isfinite(s)
        if int(m.sum()) < deg + 4:
            continue
        c = np.polyfit(x[m], s[m], deg)
        for _ in range(4):                                   # robust: drop the deviating (outlier) frames, refit
            r = s - np.polyval(c, x); sd = np.std(r[m]) + 1e-6
            k = m & (np.abs(r) < 2.0 * sd)
            if int(k.sum()) < deg + 4 or int(k.sum()) == int(m.sum()):
                break
            c = np.polyfit(x[k], s[k], deg)
        C = np.polyval(c, x)
        for f in eidx:
            if m[f]:
                dev[i, f] = s[f] - C[f]; valid[i, f] = True
    dsm = dev.copy()                                         # light lateral median → kill isolated spikes, keep per-slice
    dsm[:, eidx] = ndimage.median_filter(dev[:, eidx], size=(max(1, med_lat), 1), mode="nearest")
    w = np.clip((np.abs(dsm) - thresh) / max(1e-3, soft), 0.0, 1.0)
    shift = np.clip(-dsm * w * wf[None, :], -maxsh, maxsh)
    shift[~valid] = 0.0
    # do-no-harm tissue gate: skip columns where there is no real boundary contrast (scan ran off the cornea)
    for f in eidx:
        col = shift[:, f]
        act = np.nonzero(col != 0.0)[0]
        if act.size == 0:
            continue
        r = np.clip(np.round(S[act, f]).astype(int), 6, depth - 7)
        bscan = volume[f]                                     # (depth, lateral)
        below = np.mean([bscan[np.clip(r + j, 0, depth - 1), act] for j in range(1, 7)], axis=0)
        above = np.mean([bscan[np.clip(r - j, 0, depth - 1), act] for j in range(1, 7)], axis=0)
        ref = max(float(np.percentile(below - above, 60)), 1e-3)
        gate = np.clip((below - above) / (conf_frac * ref), 0.0, 1.0)
        shift[act, f] = col[act] * gate
    out = volume.copy(); nadj = 0
    for f in eidx:
        if np.any(shift[:, f] != 0.0):
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), shift[:, f], subpixel=True)
            nadj += 1
    return out, {"applied": bool(nadj), "frames_adjusted": int(nadj)}


def surface_refine_2d(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """FINAL robust 2-D surface refinement (# FIX column-level edge errors). The anterior surface, detected as
    S(lateral, frame), is a SMOOTH 2-D dome. Bad edge detection leaves LOCAL PATCHES where the surface locks a
    few px off (a column into the stroma, or onto a bright fleck) — e.g. CS002 OS(2) lat ~360-380 × frames 16-18
    sit ~4px too deep. These slip past axial_consistency, whose lateral median window (15px) is NARROWER than the
    patch AND which never looks along frames; a patch 20-lat wide × 3-frame deep is invisible to it. Here the
    smooth target is a robust 2-D median over BOTH axes (frame window catches the frame-narrow notch, lateral
    window the lateral-narrow one) + a light Gaussian; every column whose surface deviates > srf_dev_thresh px is
    pulled fully onto the target via the same depth-warp primitive. HARD-gated on the deviation, so an already-
    smooth surface reads dev≈0 → strict NO-OP (approved scans byte-unchanged). Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    nf, depth = int(volume.shape[0]), int(volume.shape[1])
    thr = float(p.get("srf_dev_thresh", 2.5) or 0.0)
    cap = float(p.get("srf_max_shift", 12.0) or 0.0)
    lm = int(p.get("srf_lat_med", 9) or 1); fm = int(p.get("srf_frame_med", 9) or 1)
    gs = float(p.get("srf_gauss", 1.5) or 0.0)
    iters = max(1, int(p.get("srf_iters", 2) or 1))
    min_cov = float(p.get("srf_min_coverage", 0.5) or 0.0)
    if thr <= 0 or cap <= 0:
        return volume, {"applied": False, "frames_adjusted": 0, "cols_adjusted": 0}
    out = volume.copy(); moved = np.zeros(nf, dtype=bool); n_cols = 0
    for _ in range(iters):
        S = detect_surface_all(reformat_to_sagittal(_fill_black_bands(out)), p, workers=workers)  # (lateral, frames)
        nl = int(S.shape[0])
        valid = np.isfinite(S) & (S > 1.0) & (S < depth - 1)
        Sf = S.astype(np.float64)
        # per-frame lateral interpolation of invalid columns so the 2-D median/gaussian aren't poisoned by NaNs;
        # a frame with too little cornea (off-eye) is left out of the correction (its target is meaningless).
        cov = valid.sum(axis=0)
        for f in range(nf):
            m = valid[:, f]
            if int(m.sum()) >= max(8, int(min_cov * nl)):
                Sf[~m, f] = np.interp(np.where(~m)[0], np.where(m)[0], Sf[m, f]) if m.any() else Sf[~m, f]
        target = ndimage.median_filter(Sf, size=(lm, fm), mode="nearest")
        if gs > 0:
            target = ndimage.gaussian_filter(target, sigma=gs, mode="nearest")
        dev = Sf - target                                        # + = surface DEEPER than the smooth 2-D target
        okframe = (cov >= np.maximum(8, int(min_cov * nl)))[None, :]
        shift = np.where((np.abs(dev) > thr) & valid & okframe, -np.clip(dev, -cap, cap), 0.0)  # pull outliers onto target
        if not np.any(shift != 0.0):
            break
        any_move = False
        for f in range(nf):
            sh = shift[:, f]
            if not np.any(sh != 0.0):
                continue
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), sh, subpixel=True)
            moved[f] = True; n_cols += int(np.count_nonzero(sh != 0.0)); any_move = True
        if not any_move:
            break
    return out, {"applied": bool(moved.sum()), "frames_adjusted": int(moved.sum()), "cols_adjusted": int(n_cols)}


def smooth_corrected_volume(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """SMOOTH a manually-corrected volume by RE-DETECTING its surface (which is EASY + reliable now — the
    correction put the surface where it belongs, so the auto detector lands on it, unlike on the raw), smoothing
    that detection ACROSS SLICES, and re-warping each column onto the slice-smoothed surface.

    The fix-columns Run warps via provided_edges with inter-slice smoothing DISABLED (to honour the exact drag),
    which leaves the surface reliable but with residual per-column 1-2px DETECTION errors — jagged slice-to-slice
    (the axial/B-scan fuzziness) AND small wiggles within each sagittal slice. Here we re-detect that surface and
    smooth it 2-D: STRONG across slices (smooth_slice_sigma) + GENTLE across frames (smooth_frame_sigma) so the
    per-column noise is removed in BOTH the axial and sagittal views, then warp each column onto it (full warp,
    capped). The frame σ is small, so BROAD manual corrections (a multi-frame notch/dip) survive while only the
    single-frame noise is removed — validated on CS004: slice-to-slice roughness 1.49→0.26, per-column error
    0.5→0.2px, and the frame-68-70 correction depth shifts <0.1px. `volume` = (frames, depth, lateral). Returns
    (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    nf, depth = int(volume.shape[0]), int(volume.shape[1])
    sigma = float(p.get("smooth_slice_sigma", 4.0) or 0.0)
    fsigma = float(p.get("smooth_frame_sigma", 2.0) or 0.0)
    cap = float(p.get("smooth_max_shift", 20.0) or 0.0)
    iters = max(1, int(p.get("smooth_iters", 2) or 1))
    if sigma <= 0 or cap <= 0:
        return volume, {"applied": False, "frames_adjusted": 0, "cols_adjusted": 0}
    out = volume.copy(); moved = np.zeros(nf, dtype=bool); n_cols = 0
    for _ in range(iters):
        # re-detect on the (corrected) volume — reliable because the surface is where the user put it
        S = detect_surface_all(reformat_to_sagittal(_fill_black_bands(out)), p, workers=workers)  # (lateral, frames)
        Sf = S.astype(np.float64)
        # 2-D gaussian target: STRONG across SLICES (lateral, axis=0) to kill slice-to-slice jitter (the axial
        # fuzziness), plus GENTLE across FRAMES (axis=1) to remove the per-column 1-2px detection errors WITHIN
        # each sagittal slice (the source of that fuzziness). frame σ is small so BROAD manual corrections (a
        # multi-frame notch/dip) survive — validated: per-column error 0.5→0.2px while frame-68-70 depth shifts
        # <0.1px — only the single-frame noise is removed.
        target = ndimage.gaussian_filter(Sf, sigma=(sigma, fsigma), mode="nearest")
        shift = np.clip(target - Sf, -cap, cap)                 # full warp toward the 2-D-smoothed surface
        if not np.any(np.abs(shift) > 0.05):
            break
        any_move = False
        for f in range(nf):
            sh = shift[:, f]
            if not np.any(np.abs(sh) > 0.05):
                continue
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), sh, subpixel=True)
            moved[f] = True; n_cols += int(np.count_nonzero(np.abs(sh) > 0.05)); any_move = True
        if not any_move:
            break
    return out, {"applied": bool(moved.sum()), "frames_adjusted": int(moved.sum()), "cols_adjusted": int(n_cols)}


def apply_manual_shifts(volume: np.ndarray, shifts) -> tuple[np.ndarray, int]:
    """#2 fix-columns drag-to-correct: shift a specific frame (B-scan) UP/DOWN in DEPTH by an explicit
    pixel offset the annotator dragged in the fix-columns view — a per-frame manual ground-truth nudge
    applied ON TOP of the automatic boundary correction (so the user can fix any frame the auto-detect
    still placed wrong, especially the last few sagittal slices). `shifts` maps frame_index ->
    depth_pixels (positive = DOWN / deeper, matching the on-screen drag down); accepts a dict
    {frame: px} or a list of [frame, px] pairs. Vacated rows are zero-filled. Returns (volume,
    n_frames_shifted)."""
    pairs = []
    if isinstance(shifts, dict):
        pairs = list(shifts.items())
    elif isinstance(shifts, (list, tuple)):
        for item in shifts:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((item[0], item[1]))
    nz, depth = volume.shape[0], volume.shape[1]
    out = volume.copy()
    n = 0
    for f, px in pairs:
        try:
            fpx = float(px)
            if not math.isfinite(fpx):   # reject NaN/Infinity defensively (never crash the worker)
                continue
            fi, s = int(f), int(round(fpx))
        except (TypeError, ValueError, OverflowError):
            continue
        if not (0 <= fi < nz) or s == 0 or abs(s) >= depth:
            continue        # out-of-range shift = no-op (never erase the whole frame)
        b = out[fi]                      # (depth, lateral)
        shifted = np.zeros_like(b)       # vacated rows stay 0 (background)
        if s > 0 and s < depth:          # move pixels DOWN (toward larger depth index)
            shifted[s:, :] = b[:depth - s, :]
        elif s < 0 and -s < depth:       # move pixels UP
            shifted[:depth + s, :] = b[-s:, :]
        out[fi] = shifted
        n += 1
    return out, n


# ── NIfTI output (correct Avanti geometry, matching the app's existing volumes) ──
def write_volume_nifti(vol_zyx: np.ndarray, out_path: str | Path,
                       spacing_xyz=NIFTI_SPACING, direction=NIFTI_DIRECTION,
                       origin=(0.0, 0.0, 0.0)) -> str:
    """Write a (frames, rows, cols) = (z, y, x) array as a NIfTI with explicit spacing
    (mm) and direction — bypassing the multi-frame-DICOM spacing loss so the geometry
    that drives scar mm³ is exactly right. `origin` (sitk, mm) is a general override (default 0,0,0
    like every existing volume); the surface-crop EXTEND keeps the default because consensus
    registration canonicalises origin+direction (_canon) and aligns the cornea by the optimiser."""
    import os
    import SimpleITK as sitk
    img = sitk.GetImageFromArray(np.ascontiguousarray(vol_zyx))
    img.SetSpacing(tuple(float(s) for s in spacing_xyz))
    img.SetDirection(tuple(float(d) for d in direction))
    img.SetOrigin(tuple(float(o) for o in origin))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # Write atomically (tmp + replace) so a killed/crashed worker can never leave a truncated
    # NIfTI at the real path — a later reader (e.g. the raw-scrub cache) must never see a
    # half-written volume. The temp keeps the .nii.gz suffix so SimpleITK still gzips it.
    tmp = f"{out_path}.tmp.nii.gz"
    sitk.WriteImage(img, tmp)
    os.replace(tmp, str(out_path))
    return str(out_path)


def _resolve_spacing(params: dict | None, companion_txt: str | Path | None = None,
                     n_frames: int | None = None):
    """Resolve (lateral, depth, slice) spacing with precedence: explicit params >
    companion-.txt-derived per-scan geometry > Avanti constants. The companion is
    the per-scan source of truth (XY Scan Size1 varies 4–6mm between scans)."""
    geom = {}
    if companion_txt and Path(companion_txt).exists():
        geom = companion_geometry(companion_txt, n_frames)
    p = params or {}

    def pick(key: str, default: float) -> float:
        if p.get(key) is not None:
            return float(p[key])
        if geom.get(key) is not None:
            return float(geom[key])
        return default

    return (pick("lateral_spacing", LATERAL_SPACING),
            pick("depth_spacing", DEPTH_SPACING),
            pick("slice_spacing", SLICE_SPACING))


def raw_oct_to_nifti(oct_path: str | Path, out_nifti: str | Path,
                     volume_index: int = 0, params: dict | None = None,
                     companion_txt: str | Path | None = None) -> str:
    """Raw .OCT z-stack → NIfTI (no corrections) for inspection/scrubbing."""
    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    sp = _resolve_spacing(params, companion_txt, n_frames=vol.shape[0])
    return write_volume_nifti(vol, out_nifti, sp)


def _crop_lateral_indices(params: dict | None, n_lateral: int) -> np.ndarray:
    """LEGACY (#9 v1): the old full-slice crop — params['crop_lateral'] = flat lateral indices (0..512) to
    zero ENTIRELY. Kept so cases saved before the box crop still apply. Returns a clean in-bounds array."""
    raw = (params or {}).get("crop_lateral") or []
    out = sorted({int(c) for c in raw if 0 <= int(c) < int(n_lateral)})
    return np.array(out, dtype=int)


def _crop_region_box(params: dict | None, n_frames: int, n_lateral: int):
    """#9 v2 Crop: the BOX crop = certain FRAME columns over a RANGE of LATERAL slices. params['crop_region']
    = {'lateral': [lo, hi] (inclusive sagittal-slice range), 'frames': [int, …] (the marked frame columns)}.
    Returns (lat_lo, lat_hi, [frame indices]) clamped in-bounds, or None when nothing valid is selected."""
    r = (params or {}).get("crop_region")
    if not isinstance(r, dict):
        return None
    lat = r.get("lateral") or []
    if len(lat) != 2:
        return None
    lo, hi = sorted((int(lat[0]), int(lat[1])))
    lo = max(0, min(int(n_lateral) - 1, lo)); hi = max(0, min(int(n_lateral) - 1, hi))
    fs = sorted({int(f) for f in (r.get("frames") or []) if 0 <= int(f) < int(n_frames)})
    if not fs:
        return None
    return (lo, hi, fs)


def _apply_crop(corrected: np.ndarray, params: dict | None) -> tuple[np.ndarray, int]:
    """#9 Crop: ZERO a BOX = (lateral-slice range) × (frame columns) across ALL depth — i.e. remove certain
    columns within a slice for a range of slices, BEFORE SAM2. `corrected` is (frames, depth, lateral), so a
    box zeros corrected[frame, :, lat_lo:lat_hi+1] for each marked frame. Also honours the LEGACY full-slice
    crop (crop_lateral). The removed region is recorded so scar-alignment analytics exclude it. Returns
    (corrected, n_voxels_zeroed)."""
    n_frames, depth, n_lateral = corrected.shape
    box = _crop_region_box(params, n_frames, n_lateral)
    legacy = _crop_lateral_indices(params, n_lateral)
    if box is None and legacy.size == 0:
        return corrected, 0
    corrected = np.ascontiguousarray(corrected)
    zeroed = 0
    if box is not None:
        lo, hi, fs = box
        for f in fs:
            corrected[f, :, lo:hi + 1] = 0
        zeroed += len(fs) * (hi - lo + 1) * depth
    if legacy.size:
        corrected[:, :, legacy] = 0
        zeroed += int(legacy.size) * n_frames * depth
    return corrected, int(zeroed)


def estimate_global_tilt(det: np.ndarray, n_frames: int, p: dict):
    """Robustly estimate the DOMINANT LINEAR tilt (px per frame) of the anterior surface in the FRAME direction
    from the per-slice detection `det` (n_lateral, n_frames). Returns (slope, total_tilt, frame_depth) where
    slope is px/frame, total_tilt = slope*(n_frames-1) (the tilt across the whole B-scan span), and frame_depth
    is the robust median surface depth per frame (for diagnostics / gating).

    The tilt is a PURE ACQUISITION tilt: identical for every lateral slice, so a per-frame median over lateral
    slices is a very clean estimate. Only in-frame, positive edges contribute (a clipped/failed column reads ~0
    or the frame top). The linear slope is a robust (Theil-Sen-style) median of pairwise slopes so an off-cornea
    frame at either end can't skew it. A near-centred dome's frame-direction linear component is ~0 by symmetry."""
    e = np.asarray(det, dtype=np.float64)
    F = int(n_frames)
    # robust per-frame surface depth = median over lateral slices of the VALID (in-frame) edges
    valid = np.isfinite(e) & (e > 1.0) & (e < e.shape[1] * 1000)  # sanity; depth handled by caller's canvas
    fd = np.full(F, np.nan)
    for f in range(F):
        col = e[:, f]
        m = valid[:, f]
        if int(m.sum()) >= max(5, e.shape[0] // 20):
            fd[f] = float(np.median(col[m]))
    good = np.isfinite(fd)
    if int(good.sum()) < max(8, F // 3):
        return 0.0, 0.0, fd
    xs = np.arange(F, dtype=np.float64)[good]
    ys = fd[good]
    # Theil-Sen slope: median of pairwise slopes (robust to end frames running off the cornea)
    n = xs.size
    if n > 60:  # subsample pairs to bound cost while staying robust
        idx = np.linspace(0, n - 1, 60).astype(int)
        xs, ys = xs[idx], ys[idx]
        n = xs.size
    dx = xs[None, :] - xs[:, None]
    dy = ys[None, :] - ys[:, None]
    iu = np.triu_indices(n, k=1)
    sl = dy[iu] / dx[iu]
    slope = float(np.median(sl))
    total = slope * (F - 1)
    return slope, total, fd


def apply_global_detilt(vol: np.ndarray, slope: float, p: dict):
    """Remove a DOMINANT LINEAR FRAME-direction tilt from the raw volume (frames, depth, lateral) by rigidly
    shifting each frame's whole (depth,lateral) plane in DEPTH by -round(slope*(f-f_center)), extending the depth
    canvas top+bottom so no tissue is truncated. `slope` is px/frame (from estimate_global_tilt). Returns
    (out_vol, pad_top, shifts) where out_vol has depth = old_depth + pad_top + pad_bot. Constant across lateral →
    a pure rigid re-alignment that preserves the cornea's shape; only the per-frame depth offset changes."""
    F, D, L = vol.shape
    fc = (F - 1) / 2.0
    shifts = np.round(-slope * (np.arange(F, dtype=np.float64) - fc)).astype(int)  # +ve = move that frame DOWN
    cap = int(p.get("detilt_max_pad", 400))
    # A frame with shift s>0 moves tissue DOWN by s → needs pad_bot rows at the bottom. shift s<0 moves UP → needs
    # pad_top rows at the top. Clamp both to the safety cap so a runaway estimate can't inflate the canvas.
    pad_bot = int(min(cap, max(0, int(np.max(shifts)))))
    pad_top = int(min(cap, max(0, int(-np.min(shifts)))))
    H2 = D + pad_top + pad_bot
    out = np.zeros((F, H2, L), dtype=vol.dtype)
    for f in range(F):
        off = pad_top + int(shifts[f])                        # new top row of this frame's original row 0
        lo = max(0, off); hi = min(H2, off + D)
        if hi > lo:
            out[f, lo:hi, :] = vol[f, lo - off:lo - off + (hi - lo), :]
    return out, int(pad_top), shifts


def preprocess_oct_to_nifti(oct_path: str | Path, out_nifti: str | Path,
                            params: dict | None = None, volume_index: int = 0,
                            progress=None, companion_txt: str | Path | None = None,
                            max_iterations: int = 1, min_improvement: float = 0.15,
                            abs_floor: float = 0.3, iter_dir: str | Path | None = None,
                            inject_pass: int | None = None, inject_force=None, inject_good=None,
                            provided_edges: np.ndarray | None = None, workers: int | None = None) -> dict:
    """Full pipeline: read .OCT → smoother corrections → NIfTI with correct geometry.

    max_iterations<=1 → single pass. max_iterations>1 → iterative refinement (iterate_smooth_volume),
    auto-stopping when the boundary correction stops shrinking, then keeping the BEST pass (lowest
    in-plane deviation + axial roughness, #3). Both paths apply the over-correction guard (#2) +
    inter-slice smoothing (#3) — see smooth_volume (no longer byte-identical to DICOMSmootherSteps by
    design). FINALLY (#2 ping-pong) the chosen volume is AXIAL-refined: an axial correction pass kept
    per-frame where it makes the en-face boundary smoother (axial_refine param, default on; a global
    guard makes it never worse). When iter_dir is given, each INTERMEDIATE sagittal pass volume
    (V1..V(n-1)) is written there as pass_{k}.nii.gz so the UI can step through them; out_nifti is the
    axial-refined best (so the delivered volume can be slightly smoother than the last stepped pass).
    Returns {out, passes, metrics, applied, stopped}."""
    vol = read_oct_zstack(oct_path, volume_index).astype(np.uint16)
    sp = _resolve_spacing(params, companion_txt, n_frames=vol.shape[0])
    # AUTO crop-region: if no manual crop is set, auto-detect off-cornea NOISE frames (the slow scan ran off the
    # cornea) and zero them via the same #9 crop path before surface detection + SAM2. A normal full-cornea scan
    # detects none (no-op). Manual crop_region/crop_lateral overrides; the marched re-detect path is left alone.
    _pcr = {**DEFAULT_PARAMS, **(params or {})}
    # ── AXIAL MOTION CORRECTION (v0.0.159): remove slow-scan inter-frame eye motion UP-FRONT (auto path only) by
    # rigidly re-aligning each B-scan in depth to a smooth 3-D dome, so de-tilt/flatten see de-motioned data and
    # the sagittal surface comes out smooth. Strict NO-OP on a motion-free scan (approved scans unchanged).
    _amc_info = None
    if provided_edges is None and bool(_pcr.get("axial_motion_correct", False)):
        vol, _amc_info = axial_motion_correct(vol, params, workers=workers)
    # INTRA-frame saccade de-distortion (v0.0.159): re-warp only the genuinely saccade-distorted B-scans onto
    # the smooth 3-D dome before the flatten. Strict no-op on a clean scan.
    _ifd_info = None
    if provided_edges is None and bool(_pcr.get("intra_frame_dewarp", True)):
        vol, _ifd_info = intra_frame_dewarp(vol, params, workers=workers)
    _auto_cr = None                                                # auto crop-region box → surfaced in info → persisted
    # PERF: the noise-crop check and the surface-crop check BOTH run the ~12s anterior detector on the SAME raw
    # volume with the SAME (pre-auto-tune) detector params. The detector output is independent of crop_region /
    # crop_lateral (those only mask the volume via _apply_crop; they are NOT read by _edge_worker), so compute
    # detect_surface_all ONCE and reuse it for both. The cache is CLEARED the instant _apply_crop mutates `vol`
    # (then it recomputes on the cropped volume) — so a cropped scan is byte-identical, the common no-crop scan
    # saves a full detect pass.
    _det_cache: dict = {}

    def _cur_sag_det(_p):
        """(sag, anterior-detection) for the CURRENT `vol`, computed once and reused while `vol` is unchanged."""
        if "det" not in _det_cache:
            _s = reformat_to_sagittal(vol)
            _det_cache["sag"] = _s
            _det_cache["det"] = detect_surface_all(_s, _p, workers=workers)
        return _det_cache["sag"], _det_cache["det"]

    # ── FIX tilt (defect ④): GLOBAL DE-TILT pre-alignment. A strongly-tilted acquisition (~3 px/frame ≈ 300 px
    # total) is flattened per-slice to its OWN tilted quadratic, so the tilt is preserved AND the near-row-0 flank
    # trips the clip/crop handling into a hard cut/V-notch. Remove the dominant LINEAR frame-direction tilt FIRST,
    # by rigidly shifting each frame in depth (canvas extended so nothing truncates) → the cornea is near-horizontal
    # and the normal detector + surface-crop + flatten downstream produce a smooth centred dome with NO cut. GATED
    # on the robust total tilt so a normal near-centred scan is a strict NO-OP (its frame-direction linear
    # component is ~0). Runs only on the AUTO path (not fix-columns provided_edges, not manual crop/legacy).
    # ── CROP-APPROVAL WORKFLOW ── auto de-tilt / crop-region / surface-crop are DETECTED here but only APPLIED
    # when the user APPROVES (params["apply_proposals"]=True, set by the frontend's Approve-preprocessing button).
    # Otherwise they are reported in info["proposals"] (uncorrected output kept) so the UI shows the proposed
    # region PINK + glows the fix-columns / crop buttons — the user reviews/approves at the vetting step instead
    # of a silent shift. A manually-set crop_region / surface_crop_frames is still applied directly (manual wins).
    _approved = bool((params or {}).get("apply_proposals", False))
    _proposals: dict = {"detilt": None, "crop_region": None, "surface_crop": None}
    _crop_guard_removed: list[int] = []   # clipped-apex frames the guard pulled OUT of a destructive crop_region
    _detilt_info = None
    if provided_edges is None and _pcr.get("auto_detilt", True) \
            and str(_pcr.get("detector", "dp")).lower() != "legacy" \
            and (params or {}).get("crop_region") is None and (params or {}).get("crop_lateral") is None \
            and (params or {}).get("surface_crop_frames") is None:
        try:
            _sd, _dd = _cur_sag_det(params)
            _slope, _total, _fd = estimate_global_tilt(_dd, vol.shape[0], _pcr)
            # CLIP GATE: a large linear slope alone is NOT tilt — an off-centre dome has one purely from geometry.
            # De-tilt only helps when the tilt runs the surface off the TOP of the window (near row 0) at a frame
            # end (the clip/V-notch it exists to prevent). If the surface stays comfortably in-frame everywhere,
            # the slope is dome geometry → strict NO-OP (fixes the false de-tilt proposals on off-centre domes).
            _clip_ct = int(np.sum(np.isfinite(_fd) & (_fd < float(_pcr.get("detilt_clip_row", 30.0)))))
            if abs(_total) >= float(_pcr.get("detilt_min_total", 150.0)) \
                    and _clip_ct >= int(_pcr.get("detilt_clip_min_frames", 3)):
                if _approved:
                    vol, _pad_top, _shifts = apply_global_detilt(vol, _slope, _pcr)
                    _det_cache.clear()                        # vol changed → shared anterior detection is stale
                    _detilt_info = {"slope_per_frame": round(float(_slope), 4),
                                    "total_tilt": round(float(_total), 1),
                                    "pad_top": int(_pad_top),
                                    "new_depth": int(vol.shape[1])}
                else:                                         # PROPOSE: report it; leave the volume un-detilted
                    _proposals["detilt"] = {"total_tilt": round(float(_total), 1),
                                            "slope_per_frame": round(float(_slope), 4)}
        except Exception:  # noqa: BLE001 — de-tilt is best-effort; fall back to the normal pipeline
            _detilt_info = None

    if provided_edges is None and _pcr.get("auto_crop_region", True) \
            and (params or {}).get("crop_region") is None and (params or {}).get("crop_lateral") is None \
            and str(_pcr.get("detector", "dp")).lower() != "legacy":
        try:
            _s0, _d0 = _cur_sag_det(params)
            _nz = detect_noise_frames(_s0, params, workers=workers, detect=_d0)
            if _nz:
                # AUTO-APPLY off-cornea NOISE removal (NOT a proposal): zeroing junk frames with no cornea is
                # cleanup, not a "shift" of the cornea, and leaving it un-applied leaves garbage/spikes in the
                # peripheral frames (regressed CS001_OD scans). Only the cornea-RESHAPING corrections (surface-
                # crop, de-tilt) are proposals — the user reviews those. detect_noise_frames must not fire on a
                # cornea-out-the-top runout (that is a surface-crop, handled below), so real cornea is never cut.
                params = dict(params or {})
                params["crop_region"] = {"frames": _nz, "lateral": [0, int(vol.shape[2]) - 1], "auto": True}
                _auto_cr = params["crop_region"]               # persist it (sticky) so re-runs don't un-crop it
        except Exception:  # noqa: BLE001 — best-effort; fall back to no auto crop
            pass
    # STALE / MISCLASSIFIED-CROP GUARD: a FULL-WIDTH auto crop_region must NEVER destructively zero frames that
    # actually hold cornea. An old auto noise-crop (or a stale crop_region carried in params from a previous
    # algorithm version, e.g. one lacking the "auto" flag) that mis-fired on real cornea (a clipped apex, or just a
    # frame the old detector scored low) would otherwise (a) destroy real corneal tissue and (b) SUPPRESS the
    # surface-crop reconstruction — that block only sees frames still present, so a zeroed frame is never rebuilt.
    # RE-VALIDATE against the CURRENT noise detector: a frame survives the crop ONLY if detect_noise_frames still
    # flags it as genuine off-cornea noise (blink/runout). Everything else is cornea (clipped or normal) → pulled
    # OUT of the crop so it is preserved (and a clipped apex then flows to the surface-crop proposal). A real blink
    # crop (OD runout) is re-flagged by the current detector → kept intact. GATED to FULL-WIDTH crops (lateral spans
    # the whole frame = the auto/stale signature); a MANUAL sub-lateral box crop is left exactly as the user drew it
    # (its frames aren't full-frame noise, so a noise re-check would wrongly drop them).
    _cr0 = (params or {}).get("crop_region") if params else None
    if provided_edges is None and isinstance(_cr0, dict) and _cr0.get("frames") \
            and str(_pcr.get("detector", "dp")).lower() != "legacy":
        try:
            _lat = _cr0.get("lateral") or []
            _nlat = int(vol.shape[2])
            _full_width = len(_lat) == 2 and int(_lat[0]) <= 0 and int(_lat[1]) >= _nlat - 1
            _crf = [int(f) for f in (_cr0.get("frames") or [])]
            if _full_width and _crf:
                _sg, _dg = _cur_sag_det(params)
                _noise = set(detect_noise_frames(_sg, params, workers=workers, detect=_dg))
                _keep = [f for f in _crf if f in _noise]          # keep ONLY still-genuine-noise frames
                if len(_keep) != len(_crf):
                    _crop_guard_removed = [f for f in _crf if f not in _noise]  # cornea → caller drops from persisted crop
                    params = dict(params)
                    if _keep:
                        params["crop_region"] = {**_cr0, "frames": _keep}
                    else:
                        params.pop("crop_region", None)           # no genuine noise left → surface-crop handles any clip
        except Exception:  # noqa: BLE001 — guard is best-effort; fall back to applying the crop as given
            pass
    # #9 Crop RE-DETECT: zero the cropped box on the RAW volume BEFORE surface detection, so the anterior-edge
    # DP detector + RANSAC parabola fit + warp are all computed on the TRUNCATED volume — the removed
    # frame-columns no longer pull the surface (the DP smoothly bridges the gap; RANSAC drops any residual as
    # an outlier). The box is full-depth, so it stays zero through the depth-only warp; the final _apply_crop
    # re-asserts it (idempotent). No-op when no crop is set, so non-cropped preprocessing is byte-unchanged.
    if params and (params.get("crop_region") or params.get("crop_lateral")):
        vol, _ = _apply_crop(vol, params)
        _det_cache.clear()                                        # vol changed → shared anterior detection is stale
    # SURFACE-CROP (AUTO + manual): a clipped cornea (apex and/or a whole edge ABOVE the acquisition window) is
    # corrected by fitting the still-visible POSTERIOR (bottom) edge to a parabola, aligning each column to it,
    # and EXTENDING the depth canvas UPWARD so the above-old-top apex/edge + cut-off columns are kept (never
    # truncated) — producing a TALLER volume (SAM2 cornea verified on it). Detection runs AUTOMATICALLY; a
    # substantial clip triggers the correction. A manual surface_crop_frames set overrides the auto set. Skipped
    # for the marched fix-columns re-detect (provided_edges wins) and when auto_surface_crop is off.
    _crop_frames = (params or {}).get("surface_crop_frames") if params else None
    _auto_crop = False
    _pcc = {**DEFAULT_PARAMS, **(params or {})}
    if provided_edges is None and _crop_frames is None and _pcc.get("auto_surface_crop", True) \
            and str(_pcc.get("detector", "dp")).lower() != "legacy":
        try:
            _s1, _d1 = _cur_sag_det(params)                       # reused from the noise check when no crop was applied
            _ci = detect_surface_crop_frames(_s1, params, workers=workers, detect=_d1)
            if is_substantial_clip(_ci, params):
                # AUTO-APPLY (was: propose, apply only if the user approved). The clip detector is reliable (no
                # false-fire on non-clipped scans) and — with the strong cross-slice smoothing above — the
                # reconstruction is now clean, so a detected clip is corrected AUTOMATICALLY with no human confirm.
                # A pathological over-clip is still refused by the frac-frames sanity check below.
                _crop_frames = _ci["frames"]; _auto_crop = True
        except Exception:  # noqa: BLE001 — auto detection is best-effort; fall back to the normal pipeline
            pass
    if provided_edges is None and _crop_frames:
        # auto-tune the detector to this scan, then EXTEND-warp (taller volume): posterior parabola + canvas-up.
        params = dict(params or {})
        _pc = {**DEFAULT_PARAMS, **params}
        _sagv = reformat_to_sagittal(vol)
        _crop_tune: dict = {}
        if _pc.get("auto_tune", True) and str(_pc.get("detector", "dp")).lower() != "legacy":
            try:
                _best, _sc = auto_tune_detector(_sagv, params)
                params.update(_best); _pc = {**DEFAULT_PARAMS, **params}
                _crop_tune = {"params": _best, "score": round(float(_sc), 2)}
            except Exception:  # noqa: BLE001
                pass
        _det = detect_surface_all(_sagv, _pc, workers=workers)
        _, _posterior = build_surface_crop_edges(_sagv, _crop_frames, params, workers=workers)
        _out_sag, _pad, _Pb, _Pa, _clamped = warp_surface_crop_extend(_sagv, _posterior, _crop_frames, params,
                                                                      workers=workers, detect=_det)
        _lo, _hi = int(0.3 * _sagv.shape[0]), int(0.7 * _sagv.shape[0])
        _pb_span = float(np.ptp(np.median(_Pb[_lo:_hi], axis=0)))      # posterior span across frames (diagnostic)
        _frac = len(list(_crop_frames)) / max(1, int(_sagv.shape[2]))
        if _auto_crop and _frac > float(_pc.get("crop_auto_max_frac", 0.5)):
            # SANITY only: if MORE than half the frames are flagged clipped, this is a failed / fully-off-axis scan,
            # NOT a localized apex clip — reconstructing it is meaningless, so fall through to the normal pipeline
            # (keep-clipped). The old pad/span/clamped rejection is GONE: it existed because the un-smoothed warp
            # mangled large/steep clips, but the strong cross-slice smoothing now reconstructs them cleanly (OS0
            # span 302 / pad 120 reconstructs smooth), so a clean localized clip of any size is auto-corrected.
            _crop_frames = None
        else:
            corrected = revert_sagittal(_out_sag)              # (frames, depth+pad, lateral) — same per-voxel spacing
            _F_tot = int(_sagv.shape[2])
            _peak = max((int(v) for v in _ci.get("counts", {}).values()), default=-1) if _auto_crop else -1
            info = {"passes": 1, "best_pass": 1, "metrics": [], "axial_metrics": [], "stopped": "surface_crop",
                    "apex_clipped": {"slices": {}, "n_slices": 0, "n_frames_total": 0},
                    "surface_crop": {"n_frames": len(list(_crop_frames)), "pad": int(_pad),
                                     "auto": bool(_auto_crop), "clamped": bool(_clamped),
                                     "n_frames_total": _F_tot,
                                     "frac_frames": round(len(list(_crop_frames)) / max(1, _F_tot), 3),
                                     "peak_slices": _peak, "pb_span": round(_pb_span, 1)}}
            if _crop_tune:
                info["auto_tune"] = _crop_tune
            p_all = {**DEFAULT_PARAMS, **(params or {})}
            ms = p_all.get("manual_shifts")
            if ms:
                corrected, n_ms = apply_manual_shifts(corrected, ms)
                info["manual_shifts"] = {"n_frames": int(n_ms)}
            corrected, n_crop = _apply_crop(corrected, p_all)
            if n_crop:
                info["crop"] = {"n_voxels": n_crop}
            # Origin stays (0,0,0) like every other volume: consensus registration canonicalises origin+direction
            # (_canon) and aligns the cornea by the optimiser, so a taller clipped replicate registers to its
            # repeats by pose, not by header geometry (verified on the extended volume).
            write_volume_nifti(corrected, out_nifti, sp)
            info["out"] = str(out_nifti)
            if _auto_cr:
                info["auto_crop_region"] = _auto_cr
            if _detilt_info:
                info["detilt"] = _detilt_info
            info["proposals"] = _proposals
            if _crop_guard_removed:
                info["crop_guard_removed_frames"] = list(_crop_guard_removed)
            return info
    if provided_edges is not None:
        # fix-columns marched re-detection: a SINGLE same-canvas warp that flattens to the user-validated
        # surface, NO iteration and NO axial-refine — so the corrected volume matches the scrub preview exactly.
        corrected = smooth_volume(vol, params, progress=progress, provided_edges=provided_edges, workers=workers)
        info = {"passes": 1, "best_pass": 1, "metrics": [], "axial_metrics": [], "stopped": "redetect",
                "apex_clipped": {"slices": {}, "n_slices": 0, "n_frames_total": 0}}
        p_all = {**DEFAULT_PARAMS, **(params or {})}
        ms = p_all.get("manual_shifts")
        if ms:
            corrected, n_ms = apply_manual_shifts(corrected, ms)
            info["manual_shifts"] = {"n_frames": int(n_ms)}
        corrected, n_crop = _apply_crop(corrected, p_all)
        if n_crop:
            info["crop"] = {"n_voxels": n_crop}
        write_volume_nifti(corrected, out_nifti, sp)
        info["out"] = str(out_nifti)
        if _auto_cr:
            info["auto_crop_region"] = _auto_cr
        info["proposals"] = _proposals
        if _crop_guard_removed:
            info["crop_guard_removed_frames"] = list(_crop_guard_removed)
        return info
    # ── NATIVE AUTO-TUNE: the app tunes the DP detector to THIS scan before correcting (no user input). The
    # chosen dp_* are merged into params so the warp uses them AND surfaced in info["auto_tune"] so the caller
    # persists them to the case (→ the fix-columns baseline + steps recompute with the same tuned params).
    params = dict(params or {})
    _pa = {**DEFAULT_PARAMS, **params}
    auto_tune_info: dict = {}
    _dp_keys = ("dp_sigma_depth", "dp_sigma_frame", "dp_below", "dp_max_jump")
    # AUTO-HEAL stale tune: a prior run persisted dp_* → normally reuse (deterministic). BUT params tuned by an
    # OLD grid can carry a now-out-of-range dp_sigma_frame (e.g. the 0.8 that caused the column-jitter defect);
    # a value below the current grid floor (2.0) means the tune predates the fix → RE-TUNE instead of freezing it.
    _cached_dp = all(k in params for k in _dp_keys) and float(params.get("dp_sigma_frame", 0.0) or 0.0) >= 2.0
    if _pa.get("auto_tune", True) and str(_pa.get("detector", "dp")).lower() != "legacy" and not _cached_dp:
        # a stale cached set (fails the floor check) must be DROPPED so the fresh tune's choice isn't overridden
        for _k in _dp_keys:
            params.pop(_k, None)
        try:
            best, sc = auto_tune_detector(reformat_to_sagittal(vol), params)
            params.update(best)
            auto_tune_info = {"params": best, "score": round(float(sc), 2)}
        except Exception:  # noqa: BLE001 — tuning is best-effort; fall back to the fixed defaults
            auto_tune_info = {}
    elif _cached_dp:
        auto_tune_info = {"cached": True}             # reuse persisted dp_* (deterministic tune → identical)
    clip_report: dict = {}
    # PRE-FLATTEN reference surface (the real tissue's frame-direction shape) — captured here, on the
    # de-tilted/cropped volume the flatten is about to operate on, so the post-hoc over-descent cap can tell a
    # genuinely steep-but-real limbus (keep) from the flatten quad's manufactured over-plunge (remove).
    _rawcap_ref = None
    if provided_edges is None and bool(_pcr.get("frame_edge_rawcap", True)):
        try:
            _rawcap_ref = detect_surface_all(reformat_to_sagittal(vol), params, workers=workers)
        except Exception:  # noqa: BLE001 — reference is best-effort; cap simply no-ops without it
            _rawcap_ref = None
    if max_iterations and int(max_iterations) > 1:
        chain, best_idx, info = iterate_smooth_volume(
            vol, params, max_iter=int(max_iterations),
            min_improvement=min_improvement, abs_floor=abs_floor, progress=progress,
            inject_pass=inject_pass, inject_force=inject_force, inject_good=inject_good,
            clip_report=clip_report, workers=workers)
        corrected = chain[best_idx]                 # the BEST pass (least-deviant boundary)
        # Write EVERY corrected pass (V1..Vm) so the UI can step through them all and SEE why the
        # best was chosen (a worse pass is visibly more deviant). chain[0] = raw = context_raw.
        if iter_dir is not None and len(chain) > 1:
            idir = Path(iter_dir)
            idir.mkdir(parents=True, exist_ok=True)
            for k, pv in enumerate(chain[1:], start=1):
                write_volume_nifti(pv, idir / f"pass_{k}.nii.gz", sp)
    else:
        corrected, m, ax = smooth_volume(vol, params, progress=progress, return_metric=True,
                                         clip_report=clip_report, workers=workers)
        info = {"passes": 1, "best_pass": 1, "metrics": [float(m)], "axial_metrics": [float(ax)], "stopped": "single"}
    info["apex_clipped"] = clip_report.get("apex_clipped", {"slices": {}, "n_slices": 0, "n_frames_total": 0})
    if auto_tune_info:
        info["auto_tune"] = auto_tune_info          # tuned dp_* → caller persists to the case's oct_params
    # #2 ping-pong: refine the sagittally-corrected volume with an AXIAL pass, kept per-frame only where
    # it makes the en-face boundary smoother (and only if the whole 3-D surface improves). Confirmed on
    # real scans to give the smoothest 3-D corneal surface; never worse than sagittal-only.
    p_all = {**DEFAULT_PARAMS, **(params or {})}
    if p_all.get("axial_refine", True):
        corrected, ref = axial_refine_volume(corrected, params, workers=workers)
        info["axial_refine"] = ref
    # FIX axialcons: final AXIAL-consistency pass — the sagittal flatten corrects the 513 lateral columns
    # independently, so their shifts are inconsistent → lateral WAVINESS/spikes/notches visible only in the
    # AXIAL (B-scan) view. Per B-scan, apply a SMALL GATED per-column depth nudge onto a laterally-smoothed
    # surface. Strict no-op on an already-smooth scan (gate), so it can't regress good scans.
    if p_all.get("axial_consistency", True):
        corrected, axc = axial_consistency_volume(corrected, params, workers=workers)
        info["axial_consistency"] = axc
    # FIX column-level edge errors: a final robust 2-D surface-refine pass (both lateral AND frame directions)
    # that pulls LOCAL patches where the edge detector locked a few px off (invisible to the lateral-only
    # axial_consistency) onto the smooth 2-D dome. Hard-gated on deviation → strict no-op on a smooth scan.
    if p_all.get("surface_refine_2d", True):
        corrected, srf = surface_refine_2d(corrected, params, workers=workers)
        info["surface_refine_2d"] = srf
    # FIX jagged edge B-scans: lateral-smooth ONLY the first/last few (low-signal acquisition-edge) frames, whose
    # jagged border axial_consistency's gate leaves untouched. Gated to the boundary frames → no-op on the interior.
    if p_all.get("frame_boundary_smooth", True):
        corrected, fbs = frame_boundary_lat_smooth(corrected, params, workers=workers)
        info["frame_boundary_smooth"] = fbs
    # FIX the "very steep curvature near the ends": POST-HOC frame-direction over-descent cap. RETIRED default
    # OFF (v0.0.156) — it flattened the real smooth descent + injected a boundary kink; superseded by the
    # conditional curve-snap below. Kept behind frame_edge_rawcap (default False) for reference only.
    if p_all.get("frame_edge_rawcap", False):
        corrected, foc = frame_edge_overdescent_cap(corrected, _rawcap_ref, params, workers=workers)
        info["frame_edge_overdescent_cap"] = foc
    # CONDITIONAL edge→overall-corneal-curve snap (v0.0.157): only where the acquisition-edge border OBVIOUSLY
    # deviates from the smooth corneal arc (stair-steps / physically-implausible gradient-direction reversals),
    # pull it onto the arc. Deviation-gated + distance-feathered → a strict no-op on on-curve (approved) edges,
    # and structurally cannot create a boundary kink. Local to the edge frames (interior untouched).
    if p_all.get("frame_edge_curve_snap", True):
        corrected, fcs = frame_edge_curve_snap(corrected, params, workers=workers)
        info["frame_edge_curve_snap"] = fcs
    # #2 fix-columns drag-to-correct: apply the annotator's explicit per-frame manual depth nudges LAST,
    # so they override whatever the auto-correction left for those frames (manual ground truth wins).
    ms = p_all.get("manual_shifts")
    if ms:
        corrected, n_ms = apply_manual_shifts(corrected, ms)
        info["manual_shifts"] = {"n_frames": int(n_ms)}
    corrected, n_crop = _apply_crop(corrected, p_all)
    if n_crop:
        info["crop"] = {"n_voxels": n_crop}
    write_volume_nifti(corrected, out_nifti, sp)
    info["out"] = str(out_nifti)
    if _auto_cr:
        info["auto_crop_region"] = _auto_cr
    if _detilt_info:
        info["detilt"] = _detilt_info
    info["proposals"] = _proposals
    if _crop_guard_removed:
        info["crop_guard_removed_frames"] = list(_crop_guard_removed)
    if _amc_info:
        info["axial_motion_correct"] = _amc_info
    if _ifd_info:
        info["intra_frame_dewarp"] = _ifd_info
    return info


# ── Diagnostic: render EVERY processing step for the central sagittal slice ──
# (mirrors the Streamlit generate_visualization_steps filmstrip; adds coronal steps on request).
_C_RED, _C_GREEN, _C_BLUE, _C_MAGENTA = (255, 64, 64), (64, 220, 96), (90, 150, 255), (235, 90, 235)


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array to PNG with only stdlib (no preview_io dependency)."""
    import struct
    import zlib
    rgb = np.ascontiguousarray(np.asarray(rgb, np.uint8))
    H, W, _ = rgb.shape
    sl = np.empty((H, 1 + W * 3), np.uint8)
    sl[:, 0] = 0
    sl[:, 1:] = rgb.reshape(H, W * 3)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(sl.tobytes(), 6)) + chunk(b"IEND", b""))


def _gray_rgb(img2d: np.ndarray) -> np.ndarray:
    g = np.asarray(img2d, np.float32)
    f = g[np.isfinite(g)]
    lo, hi = (float(np.percentile(f, 1)), float(np.percentile(f, 99))) if f.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    u = (np.clip((g - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)
    return np.stack([u, u, u], -1)


def _draw_curve(rgb: np.ndarray, y_per_x: np.ndarray, color, dashed: bool = False) -> np.ndarray:
    H, W = rgb.shape[:2]
    for x in range(min(W, len(y_per_x))):
        if dashed and (x // 5) % 2:
            continue
        yy = int(round(float(y_per_x[x])))
        for dy in (-1, 0, 1):
            if 0 <= yy + dy < H:
                rgb[yy + dy, x] = color
    return rgb


def _disp_resize(rgb: np.ndarray, px_aspect: float = 1.0, base_h: int = 480) -> np.ndarray:
    # MORPHOLOGICALLY-CORRECT block-replication (#2): each frame column is `px_aspect`× as wide as a depth
    # row (= slice_spacing / depth_spacing ≈ 12.8 for the Avanti), so the sagittal slice shows at its true
    # ~2:1 LANDSCAPE shape instead of the squashed portrait the old fixed 460×320 box produced. Integer
    # replication keeps every frame column exactly the same width (uniform AND crisp). base_h targets the
    # depth (row) display height. Overlays (red/green/blue curves) are drawn at native res BEFORE this, so
    # they scale with the image; the viewer renders image-rendering: pixelated.
    H, W = rgb.shape[:2]
    if H == 0 or W == 0:
        return rgb
    kh = max(1, round(base_h / H))
    kw = max(1, round(kh * float(px_aspect)))
    if kw == 1 and kh == 1:
        return rgb
    out = np.repeat(np.repeat(rgb, kh, axis=0), kw, axis=1)
    return np.ascontiguousarray(out)


def border_curves(oct_path, params=None, volume_index=0, companion_txt=None, slice_index=None):
    """Per-frame DETECTED corneal surface + RANSAC best-fit for ONE sagittal slice — as coordinate arrays
    (depth row per frame), so the UI can draw + drag them. Same detection as preprocess_steps (the
    side-corrected merged edge + its quadratic fit). reformat slice = (depth, frames), so edge[frame] is
    a depth row in [0, depth_vox); the displayed sagittal preview has depth 0 at the TOP (flipud+rot90 CW),
    so the UI maps x=frame/n_frames, y=edge/depth_vox."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    res = float(p["residual_threshold"])
    vol = read_oct_zstack(oct_path, volume_index)
    sag = reformat_to_sagittal(vol)                 # (lateral, depth, frames)
    n = sag.shape[0]
    idx = n // 2 if slice_index is None else max(0, min(n - 1, int(slice_index)))
    sl = sag[idx].astype(np.float32)                # (depth, frames)
    depth_vox, n_frames = sl.shape
    edge = _merged_side_edge(sl, p)                 # depth row per frame
    clip_cols, clip_fit = (_resolve_clip(edge, sl, res, p) if p.get("clip_handling", True)
                           else (np.array([], dtype=int), None))
    fit = clip_fit if clip_fit is not None else _fit_quadratic_ransac(edge, res)   # extrapolates above-frame on a clip
    return {
        "slices": int(n), "index": int(idx), "n_frames": int(n_frames), "depth_vox": int(depth_vox),
        "edge": [float(v) for v in edge], "fit": [float(v) for v in fit],
        "clipped": [int(c) for c in clip_cols],
    }


def preprocess_steps(oct_path, params=None, volume_index=0, companion_txt=None,
                     bad_cols=None, workers=None, slice_index=None):
    """Return (n_sagittal_slices, idx, [(label, rgb_uint8, kind, branch)]) for every per-slice
    preprocessing step on ONE sagittal slice (the central one, or `slice_index` so the user can
    inspect the detected border + fit on any slice). Faithful to the per-slice pipeline; the final
    warp reflects the current bad-column selection so the filmstrip shows exactly what a re-run does.
    `kind` is "stage" or "decision"; `branch` is the decision outcome text (for the tree)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    res = float(p["residual_threshold"]); cf = float(p.get("corr_factor", 1.0)); at = float(p.get("active_threshold", 5.0))
    vol = read_oct_zstack(oct_path, volume_index)
    sag = reformat_to_sagittal(vol)                 # (lateral, depth, frames)
    # Mirror preprocess: auto-tune the DP detector to THIS scan so the filmstrip's detection + warp match what
    # a real preprocess produces (preview == result), even BEFORE the first preprocess has persisted the dp_*.
    # Deterministic (auto_tune_detector seeds from defaults), so this yields the same dp_* the warp used.
    if p.get("auto_tune", True) and str(p.get("detector", "dp")).lower() != "legacy":
        try:
            _best, _ = auto_tune_detector(sag, p)
            p = {**p, **_best}
        except Exception:  # noqa: BLE001 — best-effort; fall back to the fixed defaults
            pass
    n = sag.shape[0]
    idx = n // 2 if slice_index is None else max(0, min(n - 1, int(slice_index)))
    sl = sag[idx].astype(np.float32)
    steps = []
    _si = [0]
    # Morphologically-correct display aspect (#2): a frame column is (slice_spacing / depth_spacing)× as
    # wide as a depth row, so the sagittal slice renders at its true ~2:1 landscape shape. Per-scan geometry
    # from the companion .txt when available, else the Avanti constants.
    geom = companion_geometry(companion_txt, n_frames=sag.shape[2]) if companion_txt else {}
    depth_sp = float(geom.get("depth_spacing") or DEPTH_SPACING)
    frame_sp = float(geom.get("slice_spacing") or SLICE_SPACING)
    px_aspect = (frame_sp / depth_sp) if depth_sp > 0 else 1.0

    def add(label, rgb, kind="stage", branch="", lane="full"):
        # lane: "full" spans the tree; "dp" / "legacy" are the two parallel detector branches.
        _si[0] += 1
        steps.append((f"{_si[0]}. {label}", _disp_resize(rgb, px_aspect), kind, branch, lane))

    add(f"Original — sagittal slice {idx}/{n}", _gray_rgb(sl))
    # The FINAL anterior edge = the guarded DP path (what auto preprocessing uses). We now ALWAYS render
    # BOTH detector branches (#2 "the program runs both methods") as a decision tree that converges at the
    # DP scar-guard cross-check, regardless of the detector param.
    merged = _merged_side_edge(sl, p)

    # ── DP lane (the native detector auto preprocessing uses) ──
    img = ndimage.gaussian_filter(sl, sigma=(float(p.get("dp_sigma_depth", 3.0)), float(p.get("dp_sigma_frame", 1.2))))
    add("DP · despeckle — anisotropic Gaussian (heavier along depth)", _gray_rgb(img), lane="dp")
    gy = np.gradient(img, axis=0); np.clip(gy, 0.0, None, out=gy)
    bw = max(2, int(p.get("dp_below", 24)))
    below = ndimage.uniform_filter1d(img, size=bw, axis=0, origin=-(bw // 2))
    score = gy * np.maximum(below - float(np.median(img)), 0.0)
    score = score / (score.max(axis=0, keepdims=True) + 1e-6)
    add("DP · surface score — dark→bright gradient × bright-tissue-below", _gray_rgb(score * 255.0),
        kind="decision", lane="dp",
        branch="locks to the anterior epithelium (a real edge AND bright cornea just below), not internal layers or top speckle")
    dp_raw = _detect_surface_dp(sl, p)               # DP BEFORE the scar-guard (to show the guard's effect)
    add("DP · detected surface (red) — smooth path", _draw_curve(_gray_rgb(sl), dp_raw, _C_RED),
        kind="decision", lane="dp",
        branch=f"dynamic-programming shortest smooth MAX-score path (per-frame depth step ≤ {int(p.get('dp_max_jump', 10))}) + 3-point sub-voxel refine")

    # ── Legacy lane (the 'old method' — and the DP scar-guard's reference surface) ──
    heq = _histeq(sl)
    add("Legacy · histogram equalized", _gray_rgb(heq), lane="legacy")
    filt = cv2.bilateralFilter(cv2.normalize(heq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
                               int(p["d"]), int(p["sigmaColor"]), int(p["sigmaSpace"]))
    add("Legacy · bilateral filtered", _gray_rgb(filt), lane="legacy")
    raw_edge = _detect_surface_gradient(filt, p["sigma"])
    add("Legacy · gradient-argmax edge (red)", _draw_curve(_gray_rgb(heq), raw_edge, _C_RED), lane="legacy")
    legacy_edge = _legacy_surface(sl, p)
    add("Legacy · side-corrected RANSAC surface (green)", _draw_curve(_gray_rgb(sl), legacy_edge, _C_GREEN),
        kind="decision", lane="legacy",
        branch="per side: keep the hist-eq OR raw edge, whichever fits its RANSAC quadratic best (robust to a bright internal scar)")

    # ── Merge: DP scar-guard cross-check — both lanes converge on the final edge ──
    guard_ov = _draw_curve(_draw_curve(_gray_rgb(sl), legacy_edge, _C_GREEN), merged, _C_RED)
    n_pulled = int(np.count_nonzero(np.abs(np.asarray(dp_raw) - np.asarray(merged)) > 0.5))
    add("Scar-guard cross-check — red = final DP (guarded), green = legacy reference", guard_ov,
        kind="decision", lane="full",
        branch=(f"where raw DP dives > {float(p.get('dp_scar_tol', 18)):.0f}px DEEPER than legacy (scar-lock signature), "
                f"DP is re-detected within ±{float(p.get('dp_scar_window', 12)):.0f}px of legacy → {n_pulled} frame(s) pulled "
                "back; the final anterior edge = guarded DP"))

    # clipped-apex: if the dome apex is above the frame, fit/extrapolate from the in-frame flanks so the
    # filmstrip's fit + final warp match a real clip-aware re-run (preview == result).
    clip_cols, clip_fit = (_resolve_clip(merged, sl, res, p) if p.get("clip_handling", True)
                           else (np.array([], dtype=int), None))
    quad = clip_fit if clip_fit is not None else _fit_quadratic_ransac(merged, res)
    im6 = _draw_curve(_gray_rgb(sl), merged, _C_GREEN)
    _clip_note = f"; apex clipped → extrapolated from in-frame flanks ({len(clip_cols)} cols)" if len(clip_cols) else ""
    add("Quadratic warp-target fit — green = edge, blue = fit", _draw_curve(im6, quad, _C_BLUE, dashed=False),
        kind="decision", branch=f"RANSAC quadratic (residual ≤ {res:.0f}px); degree-2 polyfit fallback if RANSAC fails{_clip_note}")
    nb = [merged]
    if idx > 0:
        nb.append(_merged_side_edge(sag[idx - 1].astype(np.float32), p))
    if idx < n - 1:
        nb.append(_merged_side_edge(sag[idx + 1].astype(np.float32), p))
    med = np.median(np.stack(nb), axis=0)
    dvv = np.abs(merged - med); snap = dvv > at
    if len(clip_cols):
        snap[clip_cols] = False                 # don't snap a clipped column toward neighbours
    active_e = merged.copy(); active_e[snap] = med[snap]
    n_snapped = int(np.count_nonzero(snap))
    quad_a = clip_fit if clip_fit is not None else _fit_quadratic_ransac(active_e, res)
    im7 = _draw_curve(_gray_rgb(sl), active_e, _C_MAGENTA)
    add("3D active correction — magenta = corrected, blue = fit", _draw_curve(im7, quad_a, _C_BLUE, dashed=False),
        kind="decision", branch=f"snap cols deviating > {at:.0f}px from the 3-slice neighbour median → {n_snapped} snapped")
    # Final warp: same logic as smooth_volume — with the over-correction guard (#2) + clipped-apex handling
    # so the filmstrip matches a real re-run (runaway shift interpolated from good neighbours + clamped;
    # clipped columns extrapolated + never shifted up).
    max_disp = float(p.get("max_displacement", 0.0) or 0.0)
    disp = _slice_displacement(active_e, res, cf, [int(c) for c in (bad_cols or [])],
                               [int(c) for c in (p.get("good_columns") or [])], max_disp,
                               clip_cols=clip_cols, clip_fit=clip_fit)
    warped = _warp_by_displacement(sag[idx], disp)
    guard = f"over-correction guard: clamp |shift| > {max_disp:.0f}px (interp from good neighbours)" if max_disp > 0 else "no over-correction guard (max_displacement=0)"
    add("Final corrected — column warp", _gray_rgb(warped.astype(np.float32)), kind="decision", branch=guard)
    return n, idx, steps


# ── CLI: run the heavy pipeline in an isolated subprocess (called by the sidecar,
#    so the fork-based parallelism never touches the sidecar's CUDA/torch state) ──
if __name__ == "__main__":
    import argparse
    import json as _json
    ap = argparse.ArgumentParser(description="OCT preprocessing worker")
    ap.add_argument("mode", choices=["raw", "preprocess", "steps", "border"])
    ap.add_argument("oct_path")
    ap.add_argument("out_nifti")   # mode=steps: OUTPUT DIR for step PNGs; mode=border: OUTPUT JSON file
    ap.add_argument("--params", default="{}")
    ap.add_argument("--volume-index", type=int, default=0)
    ap.add_argument("--companion-txt", default="")
    ap.add_argument("--bad-cols", default="[]")
    ap.add_argument("--slice-index", type=int, default=-1)     # which sagittal slice for steps (-1 = central)
    ap.add_argument("--max-iter", type=int, default=1)        # >1 = iterative refinement
    ap.add_argument("--min-improvement", type=float, default=0.15)
    ap.add_argument("--abs-floor", type=float, default=0.3)
    ap.add_argument("--iter-dir", default="")                 # where to write intermediate pass NIfTIs
    ap.add_argument("--inject-pass", type=int, default=0)     # apply the column fix at ONLY this pass (1-based; 0=none)
    ap.add_argument("--inject-force", default="[]")           # bad frame indices for the injected pass
    ap.add_argument("--inject-good", default="[]")            # good/anchor frame indices for the injected pass
    ap.add_argument("--provided-edges", default="")           # .npz with 'surface' (lateral,frames): fix-columns marched re-detect
    ap.add_argument("--workers", type=int, default=0)         # per-scan parallel worker cap (0 = auto); set <full when running scans concurrently
    a = ap.parse_args()
    _p = _json.loads(a.params)
    _comp = a.companion_txt or None
    if a.mode == "raw":
        raw_oct_to_nifti(a.oct_path, a.out_nifti, volume_index=a.volume_index, params=_p, companion_txt=_comp)
    elif a.mode == "steps":
        _si = None if a.slice_index < 0 else int(a.slice_index)
        _n, _idx, _steps = preprocess_steps(a.oct_path, params=_p, volume_index=a.volume_index, companion_txt=_comp,
                                            bad_cols=_json.loads(a.bad_cols or "[]"), slice_index=_si)
        _outdir = Path(a.out_nifti)
        _outdir.mkdir(parents=True, exist_ok=True)
        for old in _outdir.glob("step_*.png"):   # clear stale steps from a prior run
            old.unlink()
        _entries = []
        for _i, (_label, _rgb, _kind, _branch, _lane) in enumerate(_steps):
            _fn = f"step_{_i:02d}.png"
            (_outdir / _fn).write_bytes(_png_bytes(_rgb))
            _entries.append({"label": _label, "file": _fn, "kind": _kind, "branch": _branch, "lane": _lane})
        # New shape: {slices, index, steps}; the API reader tolerates the legacy list too.
        (_outdir / "labels.json").write_text(_json.dumps({"slices": _n, "index": _idx, "steps": _entries}))
    elif a.mode == "border":
        _si = None if a.slice_index < 0 else int(a.slice_index)
        _bc = border_curves(a.oct_path, params=_p, volume_index=a.volume_index, companion_txt=_comp, slice_index=_si)
        Path(a.out_nifti).write_text(_json.dumps(_bc))
    else:
        _pe = None
        if a.provided_edges:
            _pe = np.load(a.provided_edges)["surface"]
        _info = preprocess_oct_to_nifti(
            a.oct_path, a.out_nifti, params=_p, volume_index=a.volume_index, companion_txt=_comp,
            max_iterations=a.max_iter, min_improvement=a.min_improvement, abs_floor=a.abs_floor,
            iter_dir=(a.iter_dir or None),
            inject_pass=(a.inject_pass or None), inject_force=_json.loads(a.inject_force or "[]"),
            inject_good=_json.loads(a.inject_good or "[]"), provided_edges=_pe,
            workers=(a.workers if a.workers and a.workers > 0 else None))
        # Single machine-readable line the sidecar parses for the per-pass convergence report.
        print("ITER " + _json.dumps(_info))
    print("OK " + str(a.out_nifti))
