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

import json
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
    # Fix-columns (provided_edges) over-correction guard. The corrections warp flattens each lateral's DRAWN
    # surface to its deg-2 quadratic; at the limbus/frame-edge the drawn plunge is a runaway from that parabola
    # and the flatten REMOVES it, collapsing the reviewer's corrected dome ~3x ("left edge on the corrected
    # RESULT stays uncorrected"). Applying the same guard the auto path uses (interpolate a runaway edge disp
    # from the reliable interior + clamp) keeps the dome+limbus. 40 = mirror the auto path; 0 = raw disp (needed by
    # the corrections-path rigid ROTATION fit below, which wants the true per-column quad-minus-edge deviation).
    "provided_max_displacement": 0.0,
    # Fix-columns (provided_edges) DOME-PRESERVING flatten target (gaussian-smooth of the drawn surface instead of a
    # deg-2 parabola). SUPERSEDED by the corrections-path rigid ROTATION fit (rigid_frame_rotate) — the reviewer's
    # spec is a per-frame rigid rotation/translation onto the quadratic, which removes inter-frame motion the smooth
    # target keeps. Left at 0 (off); set >0 only to fall back to the smooth-target approach.
    #
    # RE-TESTED 2026-08-24 (cs046) and RE-REJECTED, on the DELIVERED volume, judged by the reviewer: enabling this
    # (sigma=6) makes the outer edge descend onto the drawn anchors ~20px more, BUT the corrected border is no longer
    # a clean QUADRATIC — it stays wavy (deg-2 residual on the reliable central slice 0.85px -> 4.03px), because the
    # smooth-of-the-drawn target preserves the per-frame motion that the rigid ROTATION removes. The ~20-40px outer-
    # frame "lift" that motivated the re-test is the CORRECT removal of inter-frame motion (the raw plunge is largely
    # motion, not curvature), not a defect. So this stays 0; rigid_frame_rotate owns the corrections path. The earlier
    # "3x collapse" that reopened this was a gradient-trace artifact ([[mistakes]] #14). See [[cornea-corrections-dome-flatten]].
    "provided_flatten_smooth": 0.0,
    # Corrections-path rigid ROTATION (the reviewer's algorithm): "interpolate the manual edge GT, find the best
    # axial rotation/translation to correct it toward a quadratic". rigid_frame_warp (below) already fits ONE per-
    # frame depth shift (translation) as the median across laterals; with this on, it additionally fits a per-frame
    # TILT (a true rigid B-scan rotation about the surface, linear in lateral — no per-column shear/deformation) via
    # a robust least-squares line, so each B-scan is rigidly shifted+tilted onto the smooth dome. Removes inter-
    # frame torsion a pure shift cannot → a clean smooth quadratic that follows the drawn dome+limbus (verified
    # cs046). Provided/corrections path only. False = median-shift-only (old). rigid_frame_rotate_max clamps the tilt.
    "rigid_frame_rotate": True,
    "rigid_frame_rotate_max": 20.0,   # max |tilt| (px) across a lateral half-width — guards a low-signal frame
    # LIMBUS / frame-edge band smoothing of the WARP TARGET (corrections path). The outermost frames (the steep
    # limbus descent at the faint FOV edge) carry ~1.7x the centre's detection roughness (corner_edge_retrace's
    # relaxed DP), and the flatten reproduces it as a bumpy corrected limbus. Smooth ONLY the outer band (across
    # slices then across frames, light) with a taper into the interior so the steep descent + smooth centre are
    # preserved but the ±1-2px wiggles go. Reviewer chose GENTLE-AVERAGE (no anchor re-pin — edge corrections are
    # approximate by design). 0 frames = OFF. Applied to provided_edges (corrections path) ONLY; auto path unchanged.
    # SUPERSEDED by dense_pure_interp (below): the edge-band smoothing was a band-aid that smoothed the JITTERY
    # re-detection at the limbus but drifted off the reviewer's drawn values. Pure interpolation of the drawn
    # values is smooth AND exact, so this defaults OFF (0). Kept for sparse-anchor scans where interpolation
    # across large gaps is unreliable — re-enable per-scan there.
    "edge_band_smooth_frames": 0,
    "edge_band_smooth_sigma_slice": 2.5,   # across-slice (en-face) gaussian — kills slice-to-slice edge jitter
    "edge_band_smooth_sigma_frame": 1.0,   # across-frame (within-slice) gaussian — LIGHT (heavier flattens the apex)
    "edge_band_smooth_taper": 8,           # frames over which the smoothing weight ramps to 0 at the interior
    "edge_band_smooth_ends": "low",        # "low" (frame0 = display RIGHT), "high", or "both"; low-only is descent-safe
    # PURE INTERPOLATION between manually corrected slices (the reviewer's fix). On the DENSE-anchor corrections
    # path, serve interpolate_anchors_surface (per-frame linear interp of the drawn values) instead of re-detecting
    # between anchors — smooth AND passes through the drawn edge exactly. interp_min_slices = how many slices must
    # draw a frame before that frame is interpolated (else it keeps auto — the mid-dome the reviewer left alone).
    "dense_pure_interp": True,
    "interp_min_slices": 12,
    "interp_frame_taper": 3.0,
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
    "axial_motion_correct": True,  # DEFAULT ON (v0.0.190): PROPER rigid inter-frame motion registration. A B-scan is
                                  #   acquired near-instantaneously so its internal geometry is truth and must NOT be
                                  #   deformed (user constraint); only the SLOW-scan inter-frame eye drift/saccade shifts
                                  #   each B-scan in depth. Estimate that per-frame RIGID depth offset by aligning each
                                  #   B-scan's surface to a robust smooth 3-D dome (deg-`amc_dome_deg` 2-D poly, iterative
                                  #   2σ reject → saccade-robust) and shift each frame uniformly → the sagittal flattens
                                  #   WITHOUT per-column deformation. Was retired when the per-column flatten removed the
                                  #   motion BY DEFORMING; with rigid_frame_warp that deformation is gone, so AMC becomes
                                  #   the primary motion corrector. Strict no-op on motion-free scans (byte-unchanged).
    "amc_dome_deg": 5,            # degree of the robust 2-D dome fit used as the motion-free reference
    "amc_smooth": 1.0,            # gaussian sigma (frames) on the estimated motion → kills 1-frame detection noise
    "amc_max_shift": 30.0,        # px cap on the per-frame rigid depth shift
    "amc_min_motion": 1.0,        # px: if the per-frame motion std is below this, the scan is motion-free → no-op
    "rigid_frame_warp": True,     # DEFAULT ON (v0.0.190): enforce the instantaneous-B-scan constraint IN the flatten —
                                  #   replace each frame's per-lateral displacement with ONE robust rigid depth shift
                                  #   (median over reliable laterals) so the flatten can never per-column-deform a B-scan.
                                  #   Composes with axial_motion_correct (both are rigid per-frame shifts). Auto path only
                                  #   (a manual fix-columns provided_edges warp still honours the user's exact drawn border).
    "rigid_frame_smooth": 1.5,    # gaussian sigma (frames) on the per-frame rigid shift → kills frame-to-frame detection
                                  #   jitter (adjacent B-scans ~40ms apart, so real motion is smooth); 0 = raw per-frame median
    "rigid_height_refine": True,  # DEFAULT ON (v0.0.191): final RIGID per-frame height cleanup for a smoother sagittal —
                                  #   iteratively re-estimate the leftover per-frame depth JITTER vs a robust smooth dome,
                                  #   keep only its HIGH-FREQ part (dome trajectory preserved, NOT flattened) and rigidly
                                  #   shift each B-scan by it. Self-gated: kept only if surface roughness drops. Rigid only.
    "final_qa": True,             # measure boundary deviation + axial roughness + global tilt on the volume that is
                                  #   actually WRITTEN (the pass metrics describe a pre-rigid intermediate that is never
                                  #   delivered). Telemetry ONLY — emitted into oct_iter.final_qa, gates nothing.
                                  #   Costs one extra detector pass; set False to skip.
    "qa_jitter_flag": 3.0,        # advisory review flag: residual per-frame depth jitter (px). ADVISORY, never rejects.
    "qa_dev_flag": 2.0,           # advisory review flag: delivered boundary deviation (px)
    "qa_coverage_flag": 0.60,     # advisory: below this valid-column fraction, `dev` is not trustworthy — a zeroed /
                                  #   black-padded volume measures artificially CLEAN, so flag rather than rank it.
                                  # NOTE: there is deliberately NO qa_tilt_flag. tilt_total is RECORDED (it is the
                                  # in-plane metric's null space) but flags nothing: delivered post-flatten tilt is
                                  # O(10 px) whereas the old 150.0 was copied from detilt_min_total (a PRE-flatten
                                  # trigger), and measured tilt is INVERTED vs the human verdict (AUC 0.411).
    "rhr_iters": 4,               # iterations of the dome-fit → median-residual → subtract loop
    "rhr_smooth": 3.0,            # gaussian sigma (frames) splitting removable jitter (HF) from the real dome trajectory (LF)
    "rhr_max": 8.0,               # px cap on the per-frame height correction
    "rigid_frame_derotate": True, # DEFAULT ON (v0.0.192, drift-aware v0.0.193): SECONDARY per-frame rigid ROTATION
                                  #   (inter-frame eye torsion) after the axial (depth) alignment — flattens the per-frame
                                  #   surface TILT that a depth-shift can't reach, by rotating the whole B-scan (rigid, no
                                  #   deformation). v0.0.193: removes the full frame-VARIATION of the tilt incl. a slow
                                  #   PROGRESSIVE-ROTATION drift (per-scan motion; keeps the constant real decentration DC).
    "rfd_smooth": 1.5,            # gaussian sigma (frames) — light denoise of the per-frame tilt (the drift is kept intact)
    "rfd_max_deg": 3.0,           # cap on the per-frame rotation (degrees) — real inter-frame torsion is < ~1.5°
    "rfd_ref_sigma": 9.0,         # v0.0.198 ROBUST rotational reference: level each frame's tilt to a SMOOTH-across-frames
                                  #   baseline (median-prefilter + gaussian σ frames), NOT a single global DC median. σ≈9
                                  #   keeps the real, slow decentration/astigmatism trend (>>σ) and marks the fast per-frame
                                  #   tilt swings (torsion motion, ~a few frames) for removal. A single DC ref is not robust
                                  #   (worked on centred scans by luck) — it can't represent a tilt that varies frame-to-frame.
    "rfd_iters": 3,               # v0.0.198 CLOSED-LOOP: rotate → re-detect → re-measure the tilt, repeat, so each frame's
                                  #   tilt actually REACHES the reference (the open-loop single-shot formula under-delivered).
    # SAGITTAL QUADRATIC ALIGN: a final RIGID per-frame depth shift that flattens the across-frame anterior surface
    # to its per-lateral deg-2 quadratic — "the corrected result should fit a nice smooth quadratic". rigid_height_
    # refine/derotate/refine keep the deg-5 dome + remove jitter, leaving ~4px off a quadratic; this targets the
    # quadratic directly (~1.2px). Shift-only (free tilt overfits ~884px); self-gated on central off-quad.
    # DEFAULT OFF (v0.0.218): forcing the surface onto a deg-2 quadratic SQUASHES a genuinely steep dome — a real
    # cornea is deg-5, not a parabola, so "off-quad" is largely anatomy, not motion. On cs020 (real 306px dome) this
    # + rigid_sagittal_mc flattened the corrected dome to 97px (0.32x), the "edge doesn't follow the corneal
    # curvature" the reviewer saw. The corrections path now mirrors the dome-PRESERVING normal path (validated: 15
    # approved scans keep dome ratio ~1.0). Re-enable per-scan ONLY for a genuine across-frame undulation.
    "sag_quad_align": False,
    "sqa_iters": 8,               # alternating fit rounds (target = per-lateral quad → per-frame median shift)
    "sqa_max_shift": 30.0,        # clamp on the per-frame depth TRANSLATION (px)
    "sqa_max_deg": 0.0,           # per-frame ROTATION cap (deg). 0 = translation-only: a clamped tilt over-fit the
                                  #   edge and put NEW steps at peripheral laterals that the aggregate step-guard,
                                  #   masked by a pre-existing weak-corner detection artefact, could not see. Raise
                                  #   it again only with a per-lateral (not aggregate) step guard.
    "sqa_smooth": 1.0,            # gaussian sigma (frames) on the accumulated shift/tilt — inter-frame motion is smooth
    "sqa_min_gain": 0.3,          # keep only if central off-quad drops by >= this (px), else a strict no-op
    "sqa_edge_pad": 8,            # frames excluded per end from the quadratic fit → the edge follows the CENTRAL
                                  #   curvature (extrapolated), not the deep-edited edge that would drag it steep
    "sqa_edge_slope_frac": 0.0,   # edge target slope as a fraction of the central slope (1=follow curvature, 0=flat);
                                  #   <1 flattens the residual edge descent (motion) the reviewer wants removed
    "sqa_step_tol": 12.0,         # sagittal-step guard: reject a move whose worst per-lateral frame jump grows by >this
    "sqa_step_abs": 25.0,         #   AND exceeds this absolute (px) — never introduce a gross localized sagittal step
    # corrections path: let sag_quad_align be the SOLE fine axial correction (skip rigid_height_refine/derotate/
    # rigid_frame_refine there) so every axial move answers to the sagittal edge, not each stage's own aggregate metric.
    # DEFAULT OFF (v0.0.218): with sag_quad_align off (it flattened real domes), the corrections path must run the
    # SAME dome-preserving passes as the normal path (rigid_height_refine[guided] + derotate + rigid_frame_refine),
    # which are proven on 15 approved scans. sole_sag=True is only meaningful with sag_quad_align back on.
    "corrections_sole_sag": False,
    # EDGE-FRAME GUARD (v0.0.218, corrections path): the outermost few acquisition-edge frames sit at the faint FOV
    # corner where the de-jitter can't trust the detection — they dip below the smooth dome ("first two columns of
    # the left edge don't follow the corneal curvature"). Fit each lateral's interior surface, extrapolate to the
    # edge, measure the REAL tissue anterior there, and rigidly shift the outer frames back onto that curvature.
    # Tapered (no step), rigid (anatomy preserved), self-gated on the tissue off-dome. See edge_frame_guard().
    # DEFAULT OFF (v0.0.218): the edge dip is NON-RIGID across laterals — on cs020 most laterals dip ~+20px while the
    # si<70 corner is slightly LIFTED, a LOCAL feature no single per-frame shift (or shift+tilt) can isolate. The
    # uniform guard fixed slice 19 (si 494) but over-lifted/flattened the slice-472 (si 41) corner. A rigid per-frame
    # move cannot put every lateral's outermost edge on the curve at once; the honest fixes are to accept the small
    # faint-corner dip or to CROP the outermost frames. The self-gate below now checks the FULL lateral range so this
    # stage can never again silently improve the centre while harming the periphery.
    "edge_guard": False,
    "edge_guard_frames": 2,        # outermost frames per end shifted at full weight
    "edge_guard_taper": 3,         # frames over which the shift ramps 0->1 inward (prevents a sagittal step)
    "edge_guard_deg": 4,           # degree of the per-lateral interior dome fit that is extrapolated to the edge
    "edge_guard_max_shift": 40.0,  # px clamp on the per-frame edge shift
    "edge_guard_min_gain": 2.0,    # keep only if |tissue off-dome| at the outer frames drops by >= this (px)
    # RECONCILE-TO-MANUAL-LINE (v0.0.218, corrections path): de-jitter the interior, but where the de-jitter's
    # per-frame correction is UNRELIABLE (faint FOV corner = low anterior edge strength) pin the surface back to the
    # user's interpolated drawn line (provided_edges). Reviewer's spec: "for areas where the warp has high variance,
    # pin to the interpolated manual line." DEFAULT OFF (v0.0.218): validated in isolation (roughness held at 1.60,
    # edge on the line) but the per-frame pin shift COLLAPSES the surface detection inside the live pipeline volume
    # (off-line 9.8->295, roughness ->0.38 degenerate) — a depth-orientation/canvas mismatch vs the on-disk volume
    # it was prototyped against; the self-gate correctly declines it, so it currently no-ops. Parked pending an
    # orientation fix. The reviewer chose plain de-jitter (no reconcile) meanwhile. See reconcile_manual_line().
    "reconcile_line": False,
    "reconcile_contrast_hw": 8,    # px half-window above/below the surface for the tissue-vs-background contrast
    "reconcile_rel_lo": 0.2,       # edge strength below this * median => fully unreliable (w->0, pin to line)
    "reconcile_rel_hi": 0.7,       # edge strength above this * median => fully reliable (w->1, keep de-jitter)
    "reconcile_w_smooth": 1.5,     # gaussian (frames) on the reliability weight (smooth taper -> no sagittal step)
    "reconcile_max_shift": 60.0,   # px clamp on the per-frame pin shift
    "reconcile_min_gain": 2.0,     # keep only if the low-reliability frames move >= this (px) toward the line
    "reconcile_rough_tol": 0.25,   # ...and interior roughness rises by no more than this (px)
    # CORRECTED-RESULT GENERALIZE (v0.0.219): reconstruct the corrected surface from the reviewer's drawn edges +
    # across-lateral interpolation, DP fallback beyond the drawn range (see generalize_corrected_surface). Used to
    # SERVE the corrected pane's edge where the DP fails on steep limbus descents. No global toggle — active
    # whenever corrected_edge_anchors exist.
    "corrected_generalize_taper": 8,  # laterals over which the interpolated edge tapers back to the DP outside the range
    "corrected_generalize_smooth": 2.0,  # gaussian (frames) to blend the interp↔DP across-frame notch; drawn pts re-pinned
    "csa_anchor_tol": 1.0,           # px: (legacy) old smooth-align declined a move shifting the drawn anchors by >this
    # Corrected-mode "Smooth to N & re-run" (align_corrected_to_smooth) — apply the reviewer's corrected-slice edits
    # as a per-frame rigid shift+TILT (rotation). csa_from_detected: measure the residual vs the DETECTED corrected
    # surface (True, so a real tilt has a residual to propagate) not the reconstruction (False = old, always declined).
    # csa_tilt_min_laterals: 2 WIDE-SPREAD edited slices already define a tilt line. csa_min_improve: the move must
    # cut the median |drawn-detected| deviation by at least this (never-worse guard) or it declines.
    "csa_from_detected": True,
    "csa_tilt_min_laterals": 2,
    "csa_min_improve": 0.3,
    "csa_max_tilt": 30.0,            # px: cap the tilt SWING across the width (was 20 — too tight for a real rotation);
                                     # the never-worse guard + wide-spread requirement keep a 2-point fit from running away
    "seg_clip_anterior": True,       # SAM2 label's anterior is clipped onto the reconstructed corneal surface (see api_server segment_sam2)
    "rigid_frame_refine": True,   # DEFAULT ON (v0.0.217): FINAL residual per-frame rigid depth correction, driven by the
                                  #   ANTERIOR BOUNDARY (DP-independent) rather than by the DP surface + smooth dome that
                                  #   drive AMC/rhr/rfd. Reaches what those structurally cannot: rigid_height_refine keeps
                                  #   only the HIGH-FREQ part of its correction (σ=rhr_smooth, cap rhr_max=8px), so a LOW-freq
                                  #   20-40px offset at the ACQUISITION-EDGE frames survives it — the "edge doesn't follow the
                                  #   corneal curvature, overtly steep" defect found on scan after scan in review. Edge frames
                                  #   are referenced to the cornea's OWN LOCAL CURVATURE (a polynomial extrapolates the wrong
                                  #   way there). Self-gated on the re-measured boundary: never worse, byte no-op if smooth.
    "rfr_lead": 14,               # leading frames excluded from the per-lateral reference fit (they must not shape the curve
    "rfr_tail": 5,                #   they are judged against); trailing likewise. Asymmetric: the scan start is the noisier end
    "rfr_edge_n": 12,             # how many frames per end get the LOCAL-CURVATURE reference instead of the interior curve
    "rfr_edge_fit": 28,           # frames just inside the edge whose quadratic continues the cornea's actual curvature
    "rfr_lat_lo": 0.20,           # central lateral band used for every per-frame median — the dim, speckled periphery is
    "rfr_lat_hi": 0.80,           #   unreliable (its per-slice roughness runs ~4x the centre) and drags the number off
    "rfr_min_dev": 1.0,           # px: interior frames deviating less than this are left alone (sub-pixel churn is noise)
    "rfr_edge_tau": 6.0,          # frames: seam-weighting decay of the edge reference fit, weight exp(-|f-seam|/tau).
                                  #   A PLAIN least-squares fit minimises average error over its window, but this curve is
                                  #   only ever used to EXTRAPOLATE from its seam end, and nothing pinned that endpoint —
                                  #   on cs039_os_v1 its residual there was -2.46px (the largest in the window) and the
                                  #   edge was pushed 4-9.5px the WRONG way, which a reviewer marked as a step
    "rfr_gate_abs_px": 3.0,       # px: floor on the ask, replacing the old fixed 0.8px trigger. That trigger sat an order
                                  #   of magnitude BELOW this statistic's own noise: slide the identical fit-28/extrapolate-12
                                  #   estimator into the undisputed interior and it reads 3.40px median where the real edge
                                  #   reads 2.87px. It fired on 97.1% of human-approved scan-ends
    "rfr_gate_smooth_px": 6.0,    # px: a SMOOTH edge block (no kink) is corrected only above this — cs050_od_v1's real
                                  #   defect is a smooth 10.5px flattening with a kink ratio of 0.51, so "must be kinked"
                                  #   alone rejects genuine failures; 6px sits well clear of the ~3.4px interior sham median
    "rfr_gate_ratio": 2.0,        # ...and the ask must also beat 2.0x THIS scan's own sham-edge median (see rfr_placebo_offsets),
                                  #   so the trigger is calibrated per scan instead of by a constant that fits no scan
    "rfr_placebo_offsets": (12, 18, 24, 30, 36),  # frames inboard at which the sham edges are measured — same window
                                  #   lengths and lever arm, but no acquisition edge, so their spread IS the null
    "rfr_gate_kink": 1.0,         # the edge block's 2nd-difference rms must exceed this x the interior's. Roughness is the
                                  #   ONE thing genuinely edge-specific (5.49px vs 1.56px, while the MEDIAN 2nd difference is
                                  #   identical at 1.00px — the excess is a kink in a few frames, not an offset in all of
                                  #   them). A smooth edge merely sitting off an extrapolation is not a defect
    "rfr_seam_step_px": 1.0,      # px: floor on the step the pass may create where the edge block meets the interior.
                                  #   Approved corneas have none, so a step here is by definition this pass's own artefact
    "rfr_interior_tilt": True,    # DEFAULT ON: the INTERIOR frames also get a rigid rotation where one stands clear of
                                  #   the scan's own per-frame tilt noise. A reviewer marked one column of cs035_od_v1_4 at
                                  #   sagittal slice 73 AND at slice 475 — the two ends of a 7.7px across-width ramp whose
                                  #   frame-MEDIAN is 0.07px, so every central-band measure here was blind to it. This is not
                                  #   the rejected 3-DOF fit: no lateral translation, and it fires only above a per-scan null
    "rfr_int_tilt_min_px": 3.0,   # px of across-width ramp below which no interior rotation is fitted
    "rfr_int_tilt_ratio": 4.0,    # ...and it must also stand this far clear of the median interior frame's own ramp
    # Acceptance-gate tolerances. Each is set above that metric's measured sub-pixel RESAMPLING FLOOR: a
    # uniform per-frame shift moves the whole volume rigidly, so any drift it produces is interpolation
    # artefact, and an exact-INTEGER shift (no interpolation) drifts ~0, which identifies the mechanism.
    # The previous single 0.25px tolerance was below three of the five floors — the edge one by 11x.
    "rfr_gate_rms_px": 0.05,          # interior rms; the pass's own real corrections move it up by max 0.009
    "rfr_gate_step_px": 0.25,         # seam step, per-scan allowance; real-plan headroom p99 0.507
    "rfr_gate_edge_spread_px": 0.75,  # edge spread (MEAN over frames; the MAX aggregate is unusable)
    "rfr_gate_int_spread_px": 0.10,   # interior spread (MEAN); at 0.25 a 6px interior rotation is invisible
    "rfr_gate_bsd_px": 0.5,           # along-frame banded roughness — the one gate the tilt fitters cannot game
    "rfr_jit_sigma": 2.0,         # frames: low-pass separating the edge block's frame-to-frame JITTER from its overall
                                  #   offset. The offset is only as good as a quadratic extrapolated 12 frames past its
                                  #   window (hence the sham-null gate), but that error is SMOOTH in frame index — a
                                  #   polynomial cannot zigzag — so the jitter is real per-frame motion and is trusted
                                  #   even on blocks whose offset is refused. Before this, a declined block kept its wobble
    "rfr_jit_min_px": 1.2,        # px floor on a jitter correction
    "rfr_jit_ratio": 4.0,         # ...and it must stand this far clear of the scan's own INTERIOR jitter rms
                                  #   (cs044_os_v1: edge 1.72px vs interior 0.18px, a 10x ratio, on the marked frames)
    "rfr_int_max_tilt_deg": 1.0,  # cap on the interior rotation (tighter than the edge: rigid_frame_derotate already ran)
    "rfr_edge_tilt": True,        # DEFAULT ON: correct the accepted edge blocks with a rigid depth shift AND a lateral
                                  #   TILT. A uniform shift can only remove the AVERAGE of a defect that varies across the
                                  #   B-scan's width; on cs039_os_v1's leading frame the deviation ramps 21.7->42.9px, so
                                  #   shifting by the central-band value left ~10px of over-lift at one sagittal end — the
                                  #   "right edge goes upwards against the corneal curvature near slice 513" report. The ramp
                                  #   is a ROTATION of the B-scan, not a deformation, so it is permitted; same quantity and
                                  #   same application (a linear per-column depth ramp) as rigid_frame_derotate
    "rfr_edge_max_tilt_deg": 1.5, # cap on that rotation. Real inter-frame torsion is under ~1.5 deg; cs039 needs 0.92
    "rfr_edge_tilt_min_px": 4.0,  # px of across-width spread below which no rotation is fitted — otherwise per-frame
                                  #   estimator noise becomes a spurious rotation, which is what sank the full 3-DOF fit
    "rfr_edge_tilt_ratio": 1.5,   # the across-width ramp must also beat 1.5x THIS scan's own sham-edge tilt null.
                                  #   A constant threshold fits no scan: cs044_os_v1_3's null p90 is 4.8px and its two
                                  #   real rotations read 9.1 and 11.2, while cs044_os_v1_2's noisier periphery pushes
                                  #   the null to 7.8px, so its largest edge tilt (6.9) is INSIDE the null and inert
    "rfr_edge_tilt_fit": 0.35,    # a straight line must explain the spread this well (residual/spread) — the cleanest
                                  #   separator between a real rotation (0.09-0.16) and a line through noise (0.3-0.6)
    "rfr_edge_tilt_smooth": 1.5,  # gaussian sigma (frames) on the fitted rotation sequence. B-scans are ~40ms apart so a
                                  #   real torsion trajectory is SMOOTH (cs039_os_v1: 20.7, 16.5, 7.4, 3.4px, one sign);
                                  #   a line fitted to noisy band medians ALTERNATES (cs044_os_v1: +10.0, +4.7, -12.0,
                                  #   -12.7px). Low-passing keeps the first and cancels the second
    "rfr_bands": 10,              # lateral bands used to see ACROSS the width (the central-band profile cannot)
    "rfr_band_margin": 0.05,      # outermost lateral fraction excluded from those bands — the boundary estimate there is
                                  #   noise (cs039_os_v1 laterals 2 and 508 read 193px and 115px off the arc)
    "rfr_seam_step_frac": 0.35,   # ...but the cap is max(px, frac x ask): a real edge defect BEGINS at the seam and grows
                                  #   outward, so a small step there is the price of removing a large one. What this refuses
                                  #   is a block moved UNIFORMLY, where the step is most of the correction — an offset, which
                                  #   is what the fit-error failure mode looks like, rather than a kink
    "rfr_edge_min_dev": 0.8,      # px: legacy per-frame floor. The side-level gates decide whether a block moves at all
    "rfr_max_shift": 45.0,        # px cap on the per-frame rigid depth shift
    "rfr_edge_wild_px": 45.0,     # an EDGE frame further than this off the local corneal curvature is an eyelash /
                                  #   specular streak, not motion — that SIDE's edge correction is declined, not clamped
    "rfr_sat_frac": 0.98,         # A-scans peaking above this x the 99.5th pct are SATURATED (specular/eyelash) → not measured
    "rfr_sat_max_frac": 0.05,     # ...but only if the rule stays RARE; above this share it is misfiring, so it is ignored
    "rfr_dropout_frac": 0.55,     # frames below this x the median signal have collapsed (blink/shadow) → never measured or moved
    "rfr_wild_px": 25.0,          # sanity: an interior frame deviating more than this is the MEASURE failing, not eye motion
    "rfr_wild_max": 3,            #   this many such frames (or rfr_wild_rms overall) refuses the pass instead of writing garbage
    "rfr_wild_rms": 8.0,
    "rfr_regress_tol": 1.15,      # non-regression on the INTERIOR: multiplicative (it lives at 0.15-0.5 px, where a
                                  #   0.001 px wobble is numerical noise, so a relative tolerance is the right shape)
    "rfr_edge_regress_px": 0.25,  # non-regression on the EDGE: ABSOLUTE px, because its before-value can be tens of px
                                  #   and a multiplicative tolerance there would quietly permit a several-px worsening
    "rfr_canvas_margin": 4.0,     # px: never shift a frame so far that its boundary leaves the canvas — the warp
                                  #   zero-fills what it vacates, so that would CUT the cornea off (cs035_od_v1, whose
                                  #   leading edge already sits at depth 0). Extending the canvas is surface-crop's job
    "crop_incomplete_cornea": False,  # DEFAULT OFF (v0.0.197): RETIRED. SAM2 fuses axial+coronal+sagittal by 2-of-3 majority
                                  #   vote, so a truncated cornea in the SAGITTAL view alone is outvoted by the two intact
                                  #   views → no crop needed. (The v0.0.196 crop was also inconsistent at the very edges.)
                                  #   Code kept behind the flag; the volume stays full-width with honest black edges (v195).
    "crop_cornea_max_frac": 0.06, # cap: never trim more than this fraction of laterals per side (keeps a clean scan near-full)
    "crop_cornea_band": 110,      # corneal-band depth (px below the anterior surface) checked for black = incomplete cornea
    "crop_cornea_black_thr": 0.15, # a lateral is "incomplete" if >this fraction of its band is black at ANY frame
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
    # ── fix-columns provided_edges INTER-SLICE (lateral) displacement smoothing ── The dominant cause of the
    # "jagged/spiky" axial edge: the provided_edges warp flattens each lateral INDEPENDENTLY (its own per-frame
    # quadratic), so ~1px per-lateral warp jitter turns the surface into a sawtooth in the en-face/axial view
    # (crisp rendering makes every tooth visible). This smooths the WARP DISPLACEMENT across laterals (same
    # mechanism as interslice_smooth on the auto path, sigma matched to it) so neighbouring slices shift
    # coherently → the axial edge drops from ~1.3px teeth to the raw's ~0.25px. Validated on CS004: central
    # teeth 1.4→0.22px (= raw), dragged sagittal slices visually unchanged. 0 = off (legacy per-slice warp).
    "provided_edge_ism": 3.0,
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
    # ── GATED FAINT→ONSET SNAP (faint_snap_frac) ── The DP can settle on a faint pre-epithelial reflection ABOVE
    # the true anterior surface (auto sits on intensity ~760, the true epithelium on ~1840 — learned from CS004
    # GT). TARGETED post-detection correction: ONLY where the detected point is DIM (< faint_snap_frac × col-max)
    # do we look just below for the ONSET of the sustained bright band and snap to it. Fires only on faint points
    # → a surface already on bright tissue is left byte-untouched (no overshoot on clean scans, verified on the
    # approved CS001-CS003); snaps to the FIRST sustained-bright depth → never dives into the stroma. See
    # _detect_surface_dp. Validated on CS004: 10%→50% of the user's corrections now matched within 2px, median
    # error 4.05→2.0px, with the approved scans' median shift 0px. 0 = off (byte-identical legacy detector).
    "faint_snap_frac": 0.45,      # a detected point is "faint" (snap candidate) if its intensity < this × col-max
    "faint_snap_onset": 0.55,     # band ONSET = intensity rises past this × col-max ...
    "faint_snap_sustain": 0.42,   #   ... AND the next faint_snap_sustain_px stay above this × col-max
    "faint_snap_range": 14,       # search at most this many px below the faint point for the onset
    "faint_snap_sustain_px": 6,   # window (px) over which the band brightness must be sustained
    "faint_snap_coherent": True,  # apply the snap on the ASSEMBLED surface + smooth the correction across laterals
                                  #   (robust: no per-column jitter). False = the per-slice snap (jittery at edges)
    "faint_snap_lat_smooth": 2.0, # gaussian sigma (lateral) on the snap CORRECTION → laterally-coherent surface
    # ── EDGE REGULARIZATION (low-SNR FOV-boundary laterals) ── see _edge_regularize_surface. At the extreme
    # lateral edges the cornea exits the FOV → weak signal → jagged sagittal border (CS001 OD__4 lat 0/1). Smooth
    # the surface ACROSS FRAMES (depth-preserving) with a sigma tapered by each lateral's confidence: strong on
    # faint edge laterals, a strict no-op on the confident interior. Off via edge_regularize=False.
    "edge_regularize": True,      # smooth faint edge laterals' border across frames (depth-preserving)
    "edge_reg_frame_sigma": 20.0, # base gaussian sigma (frames) at zero confidence; scales down with confidence
    "edge_reg_conf_thr": 0.8,     # laterals with band-brightness confidence >= this are untouched (interior)
    "edge_reg_outer_band": 15,    # the outermost N laterals per edge get a FLOOR sigma even when bright/confident
    "edge_reg_outer_floor": 4.0,  # floor frame-sigma on that outer band → de-jitters the tissue-bearing FOV edge
    # ── EDGE DOME CONSTRAINT (downward-hook removal) ── see _edge_dome_constrain. At the FOV-boundary laterals the
    # detector can DIVE a few px DEEPER than the smooth corneal dome (following a faint deeper structure at low SNR)
    # → the border 'does not follow the general curve' (CS001 OD__4/__5, user-marked laterals 0-13). Per frame, fit
    # a robust quadratic TREND past the edge and pull ONLY the laterals sitting >gate px BELOW it up onto the trend.
    # DOWNWARD-only by construction → never pushes a shallower real LIMBUS flattening down (the _parabola_edge trap).
    "edge_dome_constrain": True,  # remove downward hooks at the extreme lateral edges (auto path only)
    "edge_dome_ew": 18,           # number of outer laterals per edge that may be pulled onto the trend
    "edge_dome_band": 100,        # laterals just past the edge used to fit the robust quadratic dome trend
    "edge_dome_gate": 1.2,        # only laterals detected > this many px DEEPER than the trend are snapped up
    # ── FAINT-EDGE DOME FOLLOW (too-shallow correction) ── see _edge_dome_follow. The OPPOSITE defect to the dome
    # CONSTRAINT: at the faint FOV edge the DP locks onto a dim PRE-epithelial reflection ABOVE the true epithelium,
    # so the border FLATTENS off the corneal curve (CS001 OD, user: "doesn't follow the overall corneal curvature").
    # A-scans confirm the true epithelium is BRIGHTER (~1100-1300 vs ~400-900) and a few px DEEPER, at the dome. Per
    # (lateral,frame) in the edge band, if a dome-bounded window below the surface holds a SUSTAINED band ≥ratio× the
    # surface brightness, snap DOWN onto it; the whole edge patch is then 2-D (lateral×frame) smoothed so it follows
    # the curve without jitter, interior byte-untouched. DESCEND-only, dome-bounded → never dives into stroma.
    "edge_dome_follow": True,     # snap the too-shallow faint edge down onto the true epithelium (auto path only)
    "edge_follow_band": 28,       # outer laterals per edge examined for the too-shallow → snap-down correction
    "edge_follow_fit": 90,        # reliable interior laterals (just past the band) fitted for the dome window bound
    "edge_follow_ratio": 1.4,     # snap only where the sustained band below is >= this * the surface brightness
    "edge_follow_search": 20,     # max px below the surface to search for that brighter epithelium band
    "edge_follow_pad": 6,         # allow the search to reach dome+this many px (dome-bounded, no stroma dive)
    "edge_follow_sustain": 5,     # px window a band must stay bright over to count (rejects single-px specular)
    "edge_follow_med": 5,         # lateral median width on the snapped patch (kills per-lateral argmax jitter)
    "edge_follow_med_frame": 3,   # frame-direction median width on the snapped patch (kills per-frame jitter)
    "edge_follow_smooth": 1.6,    # lateral gaussian sigma on the snapped patch
    "edge_follow_smooth_frame": 1.6,  # frame-direction gaussian sigma on the snapped patch
    # ── FRAME-EDGE EPITHELIUM SNAP ── the FRAME-axis sibling of edge_follow: same faint pre-epithelial float,
    # but at the LEFT/RIGHT FRAME edges of the B-scan (edge_follow only covers the LATERAL FOV edges). Reuses
    # edge_follow_ratio/search/sustain; descend-only, frame-direction-dome-bounded. Reviewer cs046: "the left
    # edge floats above the tissue." See _frame_edge_epithelium_snap.
    "frame_edge_snap": True,      # snap the faint frame-edge surface DOWN onto the brighter epithelium
    "frame_edge_band": 12,        # first/last N frames treated as the low-signal edge band
    "frame_edge_fit": 40,         # interior frames used to fit the frame-direction dome that bounds the search
    "frame_edge_pad": 14,         # search may reach frame-dome + this many px (looser than the lateral pad: the
                                  #   cornea descends toward the frame edge faster than the interior quadratic)
    "frame_edge_spike_tol": 2.5,  # a snapped frame deeper than BOTH neighbours by > this (px) is median-despiked
    # ── STEEP-LIMBUS CORNER RE-TRACE (dp-v6, _corner_edge_retrace) ── the frame_edge_snap above is depth-capped and
    # dome-bounded, so where the epithelium plunges toward the limbus (~14+px/frame) it flatlines tens of px too
    # shallow. A local relaxed-DP re-trace of the frame-edge band follows that descent onto the real band; matches the
    # reviewer's manual anchors (cs046 corner 21.5→3.7px). Descend-gated + whole-lateral tissue-validity skip → strict
    # no-op on a flat/already-correct corner, so it never degrades an approved scan.
    "corner_retrace": True,       # master enable / kill-switch (False = strict byte-identical no-op)
    "corner_band": 13,            # first/last N frames re-traced
    "corner_anchor_med": 4,       # interior frames whose median seeds the DP anchor depth
    "corner_maxdown": 30,         # relaxed per-frame DESCENT cap (vs dp_max_jump=10) — follows the steep limbus
    "corner_maxup": 4,            # small upward slack
    "corner_smooth_w": 0.03,      # |step| path-smoothness penalty
    "corner_up_pen": 0.3,         # extra penalty per px of ASCENT (enforces near-monotone descent)
    "corner_sustain": 10,         # sustained-brightness / band-thickness window
    "corner_above": 16,           # window ABOVE for the dark->bright onset score
    "corner_ridge": 6,            # snap the onset down to the local brightness ridge (GT sits ~ridge into the band)
    "corner_bright_ratio": 1.10,  # pick must be >= this x brighter than where BASE sat
    "corner_tissue_frac": 0.40,   # pick >= this x interior-epithelium brightness (real 0.52-0.80x; dark air 0.14x)
    "corner_ref_valid_k": 2.5,    # interior epithelium must exceed background by this many MAD, else skip the lateral
    "corner_tol": 2.0,            # min descent past BASE to bother re-tracing
    "corner_max_descent": 130,    # absolute descent bound below the anchor (anti-runaway)
    "corner_sig_depth": 1.5,      # depth gaussian before scanning (namespaced — NOT dp_sigma_depth=3.0, which oversmooths the faint onset)
    "corner_sig_frame": 1.2,      # frame gaussian before scanning (namespaced)
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
    # ── UNTRUSTED-SURFACE REPAIR ── (surface-break + artifact correction; reviewer-directed, 2026-08)
    # The dominant defect in the reviewed corpus is per-frame UNDULATION: the detected surface departing from the
    # clean quadratic a corneal B-scan should follow. Measured over 4,033 frames its median is 1.23 px and its p99
    # is 135 px, and the p99 frames are not shape at all — they are frames where the detector emits a boundary on
    # image regions that contain NO cornea (uniform speckle at the acquisition edge) or that are OCCLUDED by an
    # eyelash/eyelid/reflection. Neither is correctable by moving the frame, so no rigid stage can help; the fix is
    # to stop trusting the surface there and replace those columns with the frame's own robust quadratic.
    #
    # Four independent signals, each scored against the SCAN'S OWN median rather than a constant (the absolute
    # levels vary far too much between scans to threshold globally — coherence separates hallucinating frames from
    # calm ones at AUC 0.93 while overlapping badly in absolute value):
    #   coherence  — an A-scan that stops resembling its lateral neighbours: no band to lock onto
    #   shadow     — tissue under the surface much darker than the rest of the slice: the beam was blocked
    #   bright     — signal in the air ABOVE the surface: the blocker itself
    #   jump       — the surface departing sharply from its own frame's quadratic: the "sharp V"
    # Validated against the reviewer's 20 marked artifact locations: 17/20 caught (edge 4/4, eyelid 4/4,
    # reflection 2/2, eyelash 4/5, motion 2/4 — a motion artifact leaves the tissue bright and coherent, so it is
    # a CORRECTION problem for the rigid pass, not a detection problem for this).
    "surface_repair": False,        # DEFAULT OFF until the corpus regression is reviewed
    "srep_coh_drop": 0.06,          # coherence this far below the scan median = untrusted (keeps 97.3% of calm
                                    #   frames, refuses 46.4% of hallucinating ones)
    "srep_shadow_frac": 0.85,       # tissue below the surface dimmer than this x the slice's own level
    "srep_bright_frac": 1.25,       # air above the surface brighter than this x the slice's own level
    "srep_jump_px": 6.0,            # surface this far off its frame's robust quadratic
    "srep_max_frac": 0.45,          # NEVER replace more than this fraction of one frame's laterals. A mark can
                                    #   span most of the frame axis (cs020_os_v4: app columns 19-101), and past
                                    #   that point there is not enough trusted surface left to fit a quadratic
                                    #   to — replacing it all would be inventing a surface, not repairing one.
    "srep_min_trusted": 120,        # a frame needs at least this many trusted laterals to be repaired at all
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
    # ── EDGE-TO-CURVATURE constraints ── (reviewer-directed, 2026-08). Over half the scans the reviewer called
    # "near acceptable" were rejected for one thing: the edges do not continue the overall corneal curvature —
    # "the edge columns need to be better fitted to the overall curvature", "corners of frames dont always
    # follow", "left and right cornea edge seem to change their slope signs". Measured against the frame's own
    # quadratic, near-acceptable scans sit 13.73 px off at the FRAME ends and 6.64 px off at the LATERAL ends,
    # against 2.68 and 1.66 px on approved scans.
    #
    # parabola_edge (already present, default False) fixes the frame axis; at parabola_edge_nb=14 it takes the
    # near-acceptable set from 11.83 to 1.35 px — better than the approved scans' current 2.65. The 4-frame
    # default was too narrow, which matches the reviewer's wording ("early/late frames", "near the start of the
    # sagittal slices"). It saturates by 14, so this is a plateau rather than a knob to keep turning.
    #
    # lat_edge_parabola is its twin in the lateral direction: 6.64 -> 2.30 px at a 12% edge width.
    #
    # BOTH are gated on the interior actually BEING a parabola, because snapping onto a bad fit is worse than
    # leaving the edge alone. The three scans where the frame-axis version misbehaved say so exactly:
    # cs029_os_v5 interior residual 1.65 px (20.24 -> 0.39), cs025_os_v2 4.82 px (8.15 -> 14.91, WORSE),
    # cs011_od_v2 17.77 px (52.56 -> 38.42, no help).
    # ── SURFACE-CROP PATH: padding fill + the final rigid passes (both DEFAULT OFF, pending the A/B) ──
    # warp_surface_crop_extend builds the taller canvas with zeros, so the padded rows sit at ~33 against an OCT
    # background floor of ~511 just below. The DP detector prefers that cliff to the epithelium (row 27 vs the
    # true ~199), which is why the surface-crop branch returns early and skips the three rigid smoothing passes.
    # crop_pad_fill="background" tiles the scan's own background into those rows (no value fabricated) and
    # restores detection; surface_crop_finish then runs the passes, and is HARD-GATED on the fill because
    # running them on a zero-padded volume applies real shifts computed from the artifact.
    #
    # BOTH DEFAULT ON as of the A/B over all 39 canvas-extended scans. The decisive evidence was not the
    # roughness gain (+8.3% median per scan, 28/36 improved, sign-test p=0.0006, 4/4 unpadded controls
    # byte-identical) but WHERE THE SURFACE LANDS on the delivered file: with zeros padding, 89-93% of columns
    # detect on the padding cliff instead of the cornea (case_cs024_os_v2 median surface row 17.6 against a
    # corneal band at ~163; case_cs008_od_v3 144.2 with 92.9% of columns on the cliff). With the fill it is
    # 0-3%. Everything downstream that re-detects — SAM2 seeding, delivered-QA, the review metrics — was
    # reading a cliff on those scans. NOTE the trap this also explains: undulation CANNOT diagnose it, because
    # a detector tracking a cliff flatly across every lateral scores an excellent undulation (cs008_od_v3 read
    # 1.34 px while 92.9% wrong).
    # The change is uneven (8 of 39 regress on roughness, one by 54%), so the store rollout goes through
    # .work/guarded_rerun.py --only-if-better, which keeps the old output per scan wherever it measures worse.
    # POSTERIOR (bottom) edge for the surface-crop reconstruction. It sits a near-constant distance below the
    # anterior, so the detection is re-run with prior = anterior + robust median thickness, which rejects
    # mis-locks onto the iris / specular bands. crop_post_anchors is the reviewer's manual override, keyed
    # {slice: {frame: depth}} exactly like border_anchors, and WINS over any detection.
    "crop_post_prior": True,
    "crop_post_prior_win": 40.0,   # px: search window around anterior+thickness in the second pass
    "crop_post_anchors": None,
    "crop_pad_fill": "background",  # "zeros" (pre-A/B behaviour) | "background"
    "crop_pad_fill_src": 24,      # rows below the pad used as the background source block
    "surface_crop_finish": True,  # run the final rigid passes on the surface-crop path
    # ...but NOT derotate, which the ablation identified as the pass that degrades per-frame shape on these
    # volumes (undul 13.06 -> 3.74 when dropped, with roughness still improving 1.461 -> 1.127). See the call
    # site in preprocess_oct_to_nifti. Untouched on the main path.
    "surface_crop_derotate": False,
    # Same three passes on the FIX-COLUMNS (provided_edges) path, per the reviewer's directive that correcting
    # the detected edge must "still allow the axial transforms to occur". Costs the exact scrub-preview match
    # that branch used to guarantee; False restores it.
    "redetect_finish": True,
    "edge_fit_max_resid": 2.5,    # px: interior quadratic residual above this = do not snap anything to it
    "lat_edge_parabola": False,   # DEFAULT OFF pending review
    "lat_edge_frac": 0.12,        # fraction of laterals at EACH end treated as edge
    "lat_edge_max_move": 40.0,    # px: never move a column further than this
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
    "axcons_two_sided": True,     # correct BOTH directions — a narrow up-spike is as much an axial discontinuity as a
                                  #   down-notch on a smooth cornea; the smooth median+gaussian target follows the broad
                                  #   dome so a genuine apex reads dev≈0 (untouched), only narrow jitter spikes deviate
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
    # each slice's real gradient. ±1 px BY REVIEWER DIRECTIVE (2026-08-17): find the best edge within 1 px of the
    # drawn/interpolated border, never snap back to auto. 0 → pure interpolation (no refine).
    "redetect_interp_window": 1.0,
    # DENSE-ANCHOR routing: when the fix-columns anchors span the volume with no gap wider than
    # 2×redetect_slice_band (and the span reaches both ends), SERVE the local-redetect connect-the-dots
    # surface — tight linear interpolation of the drawn corrections between adjacent anchored slices — even
    # when border_generalize is set. The smoothed/gated generalize field is tuned to spread a FEW corrections
    # robustly across laterals and so attenuates ~40% of the drawn correction between dense anchors (crisp AT
    # the anchors, soft/level-shifted BETWEEN → reads as "not interpolating"). Sparse anchors keep generalize
    # (its robustness + wide lateral spread are why it exists). Verified redetect still drives the rigid median
    # shift with dense volume-spanning anchors (cs007: 4.2px shift / 74% laterals). False → always honour
    # border_generalize (old behaviour).
    "dense_anchor_redetect": True,
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
    # WHICH frame rule decides the clipped set. "geom" = the validated flank-extrapolated APEX rule
    # (_sc_geom_frames: micro-F1 0.931 / precision 0.937 / recall 0.925 over the 37 GT surface-crop scans, 0
    # false frames on the CS010 peripheral-limbus trap, 0 of 129 vetted non-clipped scans firing). "count" =
    # the legacy _clip_mask per-slice tally + hysteresis — a complete, no-migration revert. The geom rule needs
    # the surface evidence threaded from the pipeline (see detect_surface_crop_frames); where that is
    # unavailable it falls back to "count" on its own.
    "crop_detect": "geom",
    # The three params below are read by the "count" rule only. They existed as inline .get() fallbacks;
    # declared here so they are discoverable (values identical to those fallbacks).
    "crop_hys_min_slices": 1,     # count rule: weak floor for the hysteresis run-extend
    "crop_frame_close_gap": 4,    # count rule: HALF the maximum interior hole bridged (reach = 2x this = 8)
    # "this scan is a write-off" cap on the AUTO correction: refuse to reconstruct when more than this fraction
    # of frames is flagged, because that is a failed / fully-off-axis acquisition rather than a localized clip.
    # Per rule, because the two rules have very different over-selection behaviour — see the veto site in
    # preprocess_oct_to_nifti for the measurements behind 0.75. (Both existed only as inline .get() fallbacks;
    # declared here so they are discoverable. Values unchanged for the count rule.)
    "crop_auto_max_frac": 0.5,        # count rule
    # geom rule: 0.60 sits just ABOVE the largest ground-truth crop the user has ever marked (0.574 = 58 of 101
    # frames on p1_od_v1), so every real clip in the corpus is admitted, and just BELOW the over-reaches.
    # Calibrated the hard way: at 0.75 the rule "corrected" cs015_od_v1 (74 frames, 0.73) and p5_os_v1_4 (70,
    # 0.69) — both steeply tilted, largely-off-window acquisitions — and the reconstruction came out WORSE than
    # leaving them clipped (surface detached across the volume). A >60%-clipped scan is not a localized apex
    # clip; keep-clipped-clean is the right answer there.
    "crop_auto_max_frac_geom": 0.60,
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
    "crop_rigid_disp": True,      # v0.0.202 RIGID surface-crop warp: collapse the per-column shift to ONE depth
                                  #   shift per frame (median over laterals) so an axial B-scan is repositioned
                                  #   RIGIDLY, never per-column-DEFORMED (surface-crop scans obey the rigid rule too).
                                  #   crop_disp_smooth_slice is then moot. False = legacy per-column warp (A/B only).
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
    # GUIDED RE-DETECTION: half-width (px) of the window the detector may search around the prior. TIGHT BY
    # REVIEWER DIRECTIVE (2026-08-17): the prior IS the reviewer's drawn/interpolated correction, so refine to the
    # best REAL edge WITHIN ±1 px of it and NEVER snap back to the auto edge — a wider search wanders off the
    # corrected line and "wastes the reviewer's effort" (observed on cs046: the left edge reverted to auto, ~0.5px
    # of a ~3.4px correction surviving). The old wide ±40 px "find the true boundary" width was validated with
    # prior=PLAIN-AUTO (no correction fed in); once the reviewer has corrected the edge, the prior IS the target.
    "guided_window": 1.0,
    # ADAPTIVE GUIDED WINDOW (the reviewer's refinement): the fixed ±guided_window is the RIGHT width where a
    # confident corneal edge exists (an off prior is still recovered), but at a FAINT frame — no real boundary in
    # the window — a wide search wanders onto a spurious speckle gradient. So per frame, shrink the effective
    # window toward `guided_window_min` as the best-edge strength drops below the slice's own strong-edge
    # reference, i.e. HUG the smooth generalize interpolation where there is nothing better to lock onto. On a
    # strong edge it is a strict no-op (full window). ONLY the guided re-detect uses it; the tight seed-Confirm
    # (redetect_seed_window) and the normal auto-detect are untouched.
    "guided_adaptive": True,
    "guided_window_min": 1.0,    # floor half-window (px) at a faint frame — hug the prior within ±1 px
    "guided_conf_lo": 0.20,      # best-edge < this fraction of the slice's strong edge → fully tight
    "guided_conf_hi": 0.50,      # best-edge > this fraction → full window; linear ramp between
    "guided_conf_pctl": 80.0,    # percentile of per-frame edge strength taken as the slice's "strong edge"
    # Extra rows kept above the highest drawn point when the canvas is extended for a corrected above-window
    # apex, so the epithelium does not sit flush against the new top edge.
    "crop_pad_margin": 8,
    # Hand rigid_height_refine a guided re-detection of the corrected volume rather than letting it re-detect
    # with detect_surface_all.
    # DEFAULT OFF, on evidence. The theory was sound — the stage shifts every frame onto a dome fitted through
    # whatever surface it reads, so a surface that is wrong on the corrected frames should misplace exactly
    # those frames. Measured on the reviewer's own anchors it changed nothing: median shape error 7.94 px
    # self-detected vs 7.95 px guided, while leaving the surface rougher (1.656 -> 1.722). The stage fits a
    # DEGREE-5 2-D dome and then keeps only the HIGH-FREQUENCY residual, so local accuracy in its input is
    # low-passed away before it reaches the applied shift. Kept as a switch, not a default: it costs a full
    # detection pass per run and bought nothing.
    "redetect_guided_finish": False,
    # Run axial_motion_correct on the CORRECTIONS path too. Off: it regressed the scan it was meant to fix
    # (dev 3.95 -> 10.86) and the cause is not yet understood. See the gate for the reasoning.
    "redetect_amc": False,
    # RIGID SAGITTAL MOTION CORRECTION on the corrections path: remove inter-frame drift the flatten keeps, by
    # a per-frame rigid depth shift, GATED on the sphere-excess ratio so it fires only on genuine drift.
    # Validated: complaint scan 166->34 px (undulation improved), two approved scans untouched (below gate).
    # DEFAULT OFF (v0.0.218): the sphere prediction (smc_sag_ratio 0.444) UNDER-predicts a genuinely steep dome, so
    # smc_excess_gate can't distinguish real anatomy from drift — on cs020 it read the real 306px dome as 3.6x the
    # ~85px prediction, fired, and squashed the dome to 97px (the reviewer's "edge doesn't follow the corneal
    # curvature"). A pure de-jitter that PRESERVES the dome (the normal-path passes) is correct for a smooth dome;
    # re-enable this per-scan only for a scan with a real across-frame UNDULATION (wavy, not a clean steep dome).
    "rigid_sagittal_mc": False,
    "smc_sag_ratio": 0.444,       # across-frame sag / lateral sag for a sphere over the Avanti 4.04 vs 6.00 mm axes
    "smc_excess_gate": 2.0,       # fire only when measured across-frame sag exceeds this x the sphere prediction
    "smc_max_shift": 140.0,       # cap on the per-frame depth shift (px)
    "smc_tilt_gate": 40.0,        # min edge-swing (px) of MONOTONIC tilt drift before the rotation fires.
                                  #   40 sits in the gap between the approved-scan range (measured 0.2-31 px of
                                  #   natural tilt drift) and clear motion (cs042 59 px), so it fires only on
                                  #   genuine drift and leaves approved scans frozen — same philosophy as the
                                  #   depth gate. Lower it toward 20 to also remove mild residual torsion on
                                  #   borderline scans (benign in testing, but it then changes approved output).
    "smc_tilt_smooth": 1.5,       # gaussian smoothing (frames) of the per-frame tilt estimate
    "smc_tilt_deg": 1,            # degree of the tilt-DRIFT GATE fit. 1 = the MONOTONIC (early-vs-late) drift
                                  #   that is the reviewer's rotation signature. deg>=2 also fits a mid-scan BOW,
                                  #   which on a clean scan is low-freq detection noise (cs028: 0.2px monotonic
                                  #   drift but a ~25px deg-3 bow) — firing on it sheared and harmed the scan.
    # Guided POSTERIOR detection: half-width of the window the bottom-edge detector may search around the
    # reviewer-informed prior, and how much the interpolated residual is smoothed across frames.
    # 60, MEASURED — not a guess, and not the 30 first tried. Held out the reviewer's slice-214 posterior
    # points and built the prior from slice 195 alone: auto scored median 14.6 px, +/-15 gave 30.0 and +/-30
    # gave 41.0 (both WORSE than auto), +/-60 gave 11.5 with 31.9% of points within 5 px against auto's 25.5%.
    # A tight window pins the search beside a prior that is itself wrong, so it cannot recover; a wide one
    # lets the detector find the real bright->dark gradient. Same shape as the anterior, where +/-5 lost to +/-40.
    "post_guided_window": 60.0,
    "post_prior_smooth": 3.0,
    # How far beyond the outermost corrected slice a posterior correction is still allowed to guide detection,
    # fading linearly to nothing. Beyond it, guidance is withheld entirely rather than weakened to a no-op.
    "post_taper_slices": 20,
    # Master switch for guided posterior detection in the surface-crop preview. False = the old behaviour
    # (detect the bottom edge with no knowledge of the reviewer's posterior corrections).
    "post_guided": True,
}


def load_param_overrides(path: str | None = None) -> dict:
    """Fold GLOBALLY TUNED detector parameters into DEFAULT_PARAMS, in place.

    WHY A FILE AND NOT AN IN-MEMORY UPDATE. Preprocessing runs two ways — in-process from the sidecar, and as
    THIS module invoked as a CLI in a subprocess (_run_oct_worker, which keeps fork-based parallelism away
    from the sidecar's CUDA state). A tuned value held only in the sidecar's memory would apply to the first
    and be silently absent from the second, so the same scan would preprocess differently depending on which
    path ran it. Reading the file at import means every process that touches the pipeline sees the same
    detector, including subprocesses that inherit only the environment.

    The path comes from CORNEA_PARAM_OVERRIDES so the DATA DIRECTORY IS RESOLVED IN EXACTLY ONE PLACE (the
    sidecar's settings), rather than re-derived here from XDG/CORNEA_DATA_DIR — that duplication is precisely
    what once pointed a relaunched app at the wrong case store.

    Only keys that already exist in DEFAULT_PARAMS are accepted, and only scalars: the file is written by the
    tuner, but it is still on disk, and an unknown key silently doing nothing is a worse failure than a
    rejected one. Returns what was applied."""
    src = path or os.environ.get("CORNEA_PARAM_OVERRIDES") or ""
    if not src:
        return {}
    try:
        with open(src, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    applied: dict = {}
    for k, v in (data.get("params") or data).items():
        if k not in DEFAULT_PARAMS or not isinstance(v, (int, float, bool, str)):
            continue
        applied[k] = v
    DEFAULT_PARAMS.update(applied)
    return applied


_PARAM_OVERRIDES_APPLIED = load_param_overrides()
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
                             prior: np.ndarray | None = None, window: float | None = None,
                             adapt: dict | None = None) -> np.ndarray:
    # Vectorized over columns: smooth each column along depth, take the gradient, and
    # the brightest rising edge → corneal surface row. (Same result as the per-column
    # loop in the original, but ~order-of-magnitude faster.)
    # prior (per-FRAME expected depth, length = n_frames) + window (depth voxels): when both given, each
    # column's argmax is restricted to depth rows [prior[f]-window, prior[f]+window], so a spurious
    # gradient peak (e.g. a reflection above the cornea) OUTSIDE the window can't be picked. This is how
    # the fix-columns marched re-detection (redetect_surface) tracks the tilted cornea. prior=None →
    # unrestricted argmax (the original auto behaviour — the normal pipeline never passes a prior).
    # adapt (guided re-detect only): {min_window, conf_lo, conf_hi, pctl} → shrink the per-frame window toward
    # min_window where the full-window pick's edge is WEAK vs the slice's own strong edge, so a faint frame hugs
    # the prior rather than chasing a spurious gradient. None → the fixed-window behaviour above (all other callers).
    sm = ndimage.gaussian_filter1d(img.astype(np.float32), sigma=sigma, axis=0)
    grad = np.gradient(sm, axis=0)                       # (depth, frames)
    if prior is None or window is None or not (float(window) > 0):
        return np.argmax(grad, axis=0)
    H, W = grad.shape
    pr = np.asarray(prior, dtype=np.float32)
    rows = np.arange(H)[:, None]                          # (depth, 1)

    def _win_argmax(win_vec: np.ndarray) -> np.ndarray:
        lo = np.clip(np.round(pr - win_vec), 0, H - 1).astype(np.intp)     # (frames,)
        hi = np.clip(np.round(pr + win_vec) + 1, 1, H).astype(np.intp)     # (frames,) exclusive
        mask = (rows >= lo[None, :]) & (rows < hi[None, :])                # (depth, frames)
        return np.argmax(np.where(mask, grad, -np.inf), axis=0)

    d_full = _win_argmax(np.full(W, float(window), dtype=np.float32))
    if not adapt:
        return d_full
    # ADAPTIVE WINDOW: gate the effective half-window by how strong the full-window pick's edge is, measured
    # against the slice's own strong-edge reference (a high percentile of per-frame edge strengths). A faint
    # frame (weak best-edge) shrinks toward min_window → the surface stays on the prior (the smooth generalize
    # interpolation); a confident frame keeps the full window → a genuinely-off prior is still corrected.
    g_full = np.clip(grad[d_full, np.arange(W)], 0.0, None)                # edge strength at the full pick
    finite = g_full[np.isfinite(g_full)]
    R = float(np.percentile(finite, float(adapt.get("pctl", 80.0)))) if finite.size else 0.0
    if R <= 1e-6:
        return d_full                                                     # no measurable edge anywhere → don't gate
    lo_t = float(adapt.get("conf_lo", 0.20)); hi_t = float(adapt.get("conf_hi", 0.50))
    w = np.clip((g_full / R - lo_t) / max(hi_t - lo_t, 1e-6), 0.0, 1.0)   # 0 weak→tight, 1 strong→wide
    mn = float(adapt.get("min_window", 1.0))
    eff = mn + (float(window) - mn) * w                                   # per-frame effective half-window
    return _win_argmax(eff.astype(np.float32))


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
    # fails and the clip is missed). A binary closing fills internal gaps WITHOUT growing the outer extent —
    # restoring the contiguous central run the gates expect. No-op for the already-contiguous legacy mask and
    # for a normal scan's (empty/sparse) mask, so it can't create a false clip.
    # REACH: the structuring element is ones(2*clip_close_gap+1), which closes holes up to 2*clip_close_gap
    # wide — i.e. 8 at the default of 4, NOT 4 (measured: widths 3-8 fill, 9 does not). Read the param as
    # "half the maximum gap bridged".
    # OR with the pre-closing mask so the closing can only FILL gaps, never ERODE. binary_closing's erosion
    # step (border_value=0) sheds the first ~gap frames of a clip that runs to the FRAME-ARRAY EDGE — a
    # WHOLE-EDGE clip (cornea off the top along an entire edge), not a central apex. Without this, an edge
    # clip loses its boundary frames and the surface-crop stops short of the true edge (CS002_OS(5)). In the
    # interior, closing is extensive (⊇ mask), so central apex clips are byte-unchanged.
    gap = int(p.get("clip_close_gap", 4))
    if gap > 0 and mask.any():
        mask = mask | ndimage.binary_closing(mask, structure=np.ones(2 * gap + 1, dtype=bool))
    return mask


# ── SURFACE-CROP FRAME RULE ("geom"): flank-extrapolated APEX recovery ────────────────────────────────────
# The legacy rule (_clip_mask + per-frame slice tally) asks "is the detected edge pinned near the top AND is
# the top band bright". That conflates a clipped central APEX with a PERIPHERAL LIMBUS graze, and it reads
# only the geometry it is handed — so on a motion-corrected volume it fires on frames whose apex is actually
# in-frame (CS010: 7 false frames) while MISSING the tapering tails of real clips.
#
# This rule instead estimates the corneal APEX per frame on BOTH geometries and asks the user's actual
# question: "is the apex within ~2 px of the RAW top". Validated over the 37 GT surface-crop scans (user
# marks) at micro-F1 0.931 / precision 0.937 / recall 0.925, with 0 false frames on the CS010 limbus-graze
# trap and 0 of 129 vetted non-clipped scans firing. Reference implementation and the design history live in
# .work/wf/surface_crop_detect.py; these helpers are a faithful transplant of it (see tests).
#
# Why the apex must be EXTRAPOLATED rather than read off: the DP detector PINS a clipped apex inconsistently
# anywhere in rows 0-8, so a naive min(S_raw) <= 2 test has recall 0.23. Fitting the descending dome FLANKS
# (excluding the pinned plateau) and extrapolating to the vertex recovers where the apex WOULD be, including
# above row 0. Why both geometries: axial_motion_correct shifts a clipped apex DOWN to ~row M[f], so the
# motion-corrected apex plus M[f] recovers the raw-top-relative apex; and on strongly TILTED scans raw
# detection mislocates the apex outright (cs008_od_v3: raw apex 17-20 is wrong) so the mc flanks carry the
# signal. Reading only one geometry is what makes the legacy rule fail in both directions.
_SC_ALGO_VERSION = "apex-recover-v1"
# Thresholds are MODULE constants, deliberately NOT DEFAULT_PARAMS keys: they are a single jointly-validated
# operating point, and a per-case oct_params copy would silently drift individual scans off it.
_SC_A_RAW = 5.0        # seed: smoothed raw apex near the top (the user's criterion, on the SMOOTHED profile
                       #   so a 1-2 px detector spike cannot qualify)
_SC_A_BAND = 3         # seed: >= N laterals of the raw apex at row <= 2 (a real clip is a BAND, not a column)
_SC_A_M = 0.0          # seed: motion-recovered raw apex (rmc + M) at/above the top
_SC_A_VY = 2.0         # seed: two-flanked dome vertex extrapolates to the top
_SC_A_VYNEG = -8.0     # seed: one-sided vertex extrapolates well ABOVE the frame (trusted only when strong)
_SC_A_DEEP = 1.0       # seed: motion-corrected apex essentially at the top
_SC_W_RAW = 4.0        # weak (run-extend): raw apex
_SC_W_MC = 6.0         # weak: motion-corrected apex
_SC_W_M = 2.0          # weak: motion-recovered apex
_SC_W_VY = 4.0         # weak: two-flanked dome vertex
_SC_GAP = 3            # bridge gaps of <= this many frames between kept runs (tapering tails)
_SC_MED_K = 7          # lateral median-smoothing window
_SC_RAW_NEAR = 30.0    # physical-consistency gate: raw apex this shallow → trust it, frame is clip-eligible
_SC_RAW_FAR = 70.0     # ...or up to here IF the motion-recovered apex is also sane
_SC_RAWAPEX_FLOOR = -12.0   # a real clipped apex is never hundreds of px above the frame (broken S_mc excursion)


def _sc_lateral_median(A: np.ndarray, k: int = _SC_MED_K) -> np.ndarray:
    """Median-smooth EVERY frame's lateral profile at once: A=(laterals, frames) → same shape, float64.

    Bit-identical to the reference per-column loop (`np.median(a[max(0,i-h):min(n,i+h+1)])` for each i) but
    ~260x faster (4.6 s → 0.018 s per geometry per scan), which is what keeps the rule free in the pipeline
    AND in the /oct-surface-crop/detect request thread. Two details are load-bearing for exactness:
      • the input dtype is PRESERVED (no float64 upcast first). The 2*h truncated EDGE windows have EVEN
        length, so their median is a mean of two elements — computed in float32 vs float64 that differs in
        the last bits, and the equality test in tests/ would fail.
      • the interior uses the full odd-length window (median = exact middle element), the edges are computed
        explicitly with the same truncated bounds as the reference.
    """
    A = np.asarray(A)
    L = int(A.shape[0]); h = int(k) // 2
    out = np.empty(A.shape, dtype=np.float64)
    if L > 2 * h:
        win = np.lib.stride_tricks.sliding_window_view(A, 2 * h + 1, axis=0)   # (L-2h, frames, k)
        out[h:L - h] = np.median(win, axis=-1)
        edge_rows = list(range(0, h)) + list(range(L - h, L))
    else:
        edge_rows = list(range(L))          # window never fits: every row is a truncated edge window
    for i in edge_rows:
        out[i] = np.median(A[max(0, i - h):min(L, i + h + 1)], axis=0)
    return out


def _sc_flank_vertex(s: np.ndarray, sm: np.ndarray, rmin: float, ax: int):
    """Fit a parabola to the descending dome FLANKS (excluding the pinned near-apex plateau and detector
    spikes) and return (vertex_row, two_sided, curvature). The vertex may be NEGATIVE = apex above the
    window, which is exactly the clipped case the tally-based rule cannot see."""
    L = len(s); x = np.arange(L)
    nospike = np.abs(s - sm) < 8
    flank = nospike & (sm > rmin + 6.0)
    two = (int((flank & (x < ax)).sum()) >= 25) & (int((flank & (x > ax)).sum()) >= 25)
    vy = np.inf; curv = 0.0
    if flank.sum() >= 30:
        A = np.polyfit(x[flank], s[flank], 2); curv = float(A[0])
        if A[0] > 1e-6:                      # opens downward in row-space = a real dome
            vy = A[2] - A[1] ** 2 / (4 * A[0])
    return float(vy), bool(two), curv


def _sc_frame_evidence(s_mc, smc, s_raw, sraw, m: float):
    """(seed, weak) for ONE frame. `seed` = a confident clip; `weak` = enough to EXTEND a run that already
    contains a seed. Smoothed profiles are passed in (hoisted out of the frame loop)."""
    rmc = float(smc.min()); ax = int(smc.argmin())
    vy, two, curv = _sc_flank_vertex(s_mc, smc, rmc, ax)
    rraw = float(sraw.min()); n2 = int((sraw <= 2).sum())
    rawapex = rmc + m
    # PHYSICAL-CONSISTENCY gate: raw is the truth geometry, so a clip needs the raw apex near the top. The one
    # exception is a heavy-motion frame where the raw detector mislocates the apex moderately deep (rraw<=70)
    # yet the motion-recovered apex is a plausible SMALL clip. This rejects broken S_mc excursions (rmc
    # plunging hundreds of px negative while the true raw surface sits deep in-frame) without excluding any
    # genuine clip (verified: 0 of 937 GT frames excluded).
    if not ((rraw <= _SC_RAW_NEAR) or (rraw <= _SC_RAW_FAR and rawapex >= _SC_RAWAPEX_FLOOR)):
        return False, False
    seed = ((rraw <= _SC_A_RAW) or (n2 >= _SC_A_BAND) or (rawapex <= _SC_A_M) or
            (two and curv > 0 and vy <= _SC_A_VY) or (curv > 0 and vy <= _SC_A_VYNEG) or
            (rmc <= _SC_A_DEEP))
    weak = ((rraw <= _SC_W_RAW) or (rmc <= _SC_W_MC) or (rawapex <= _SC_W_M) or
            (n2 >= 1) or (two and curv > 0 and vy <= _SC_W_VY))
    return bool(seed), bool(weak)


def _sc_geom_frames(S_mc: np.ndarray, S_raw: np.ndarray, M: np.ndarray) -> list[int]:
    """The clipped FRAME set. S_mc / S_raw are (laterals, frames) anterior-surface rows on the
    motion-corrected and RAW geometries (row 0 = top of the window); M is the per-frame rigid depth shift
    axial_motion_correct APPLIED, oriented so raw_row = corrected_row + M[f].

    Seeds are deliberately strict, so the peripheral-limbus trap (top-grazing pixels but the apex genuinely
    in-frame) fires NONE of them: its smoothed raw apex stays >= 6, it has no raw top band, its
    motion-recovered apex is > 0 and its dome vertex extrapolates in-frame. Real clips are CONTIGUOUS runs
    around the apex frame, so each seed is GROWN along the frame axis through the looser weak criterion and
    short gaps are bridged. A scan with NO seed emits nothing — which is what keeps the trap empty rather
    than relying on a per-scan exception."""
    S_mc = np.asarray(S_mc); S_raw = np.asarray(S_raw)
    M = np.asarray(M, dtype=np.float64).ravel()
    F = int(S_mc.shape[1])
    SM_mc = _sc_lateral_median(S_mc); SM_raw = _sc_lateral_median(S_raw)
    seed = np.zeros(F, bool); weak = np.zeros(F, bool)
    for i in range(F):
        seed[i], weak[i] = _sc_frame_evidence(S_mc[:, i], SM_mc[:, i], S_raw[:, i], SM_raw[:, i], float(M[i]))
    # frame-axis HYSTERESIS: keep each maximal weak run only if it contains a seed
    keep = np.zeros(F, bool); i = 0
    while i < F:
        if weak[i]:
            j = i
            while j < F and weak[j]:
                j += 1
            if seed[i:j].any():
                keep[i:j] = True
            i = j
        else:
            i += 1
    idx = np.nonzero(keep)[0]
    for a, b in zip(idx[:-1], idx[1:]):        # bridge short breaks between kept runs (tapering tails)
        if b - a <= _SC_GAP:
            keep[a:b + 1] = True
    return [int(f) for f in np.nonzero(keep)[0]]


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


def _extrapolate_fit(edge: np.ndarray, clip_cols: np.ndarray, residual_threshold: float, degree: int = 2):
    """Re-fit the extrapolating polynomial for a KNOWN set of clipped columns — used to CARRY a clip forward to
    iteration passes ≥1, which detect on a warped+filled volume where the 'no air gap' clip invariant no
    longer holds (so they must NOT re-detect). The column set is trusted from pass 0; no gates here, just a
    RANSAC fit on the in-frame columns predicted across the clip. Returns the fit over all columns or None.

    `degree` (default 2 = the legacy parabola): a symmetric deg-2 cannot hold a dome whose apex sits near a
    frame edge (a crop that keeps only one side of the cornea), so it collapses the short steep flank — set a
    higher degree (e.g. 4) for the crop_bands path to preserve that captured curvature (validated: deg-4 recovers
    the left flank +53→+74px matching the target, max across-frame 2nd-diff ~1px; deg-5 starts to oscillate)."""
    edge = np.asarray(edge, dtype=np.float64); n = edge.size
    cc = np.asarray(clip_cols, dtype=int)
    if cc.size == 0:
        return None
    valid = np.ones(n, dtype=bool); valid[cc[(cc >= 0) & (cc < n)]] = False
    if int(valid.sum()) < 3:
        return None
    x = np.arange(n, dtype=np.float64)
    deg = max(1, int(degree))
    if deg >= 3:
        # High degree is only used on a CLEAN target (the frame-smoothed provided surface) where RANSAC's
        # subset selection is both unnecessary (no outliers to reject) and UNSTABLE — a deg-4 through 30% of
        # points (min_samples=0.3) diverges (observed: apex flips to the frame edge, flank +187px). Fit ALL
        # valid points with a plain least-squares polynomial, which is well-constrained and matches the target.
        try:
            return np.polyval(np.polyfit(x[valid], edge[valid], deg), x)
        except Exception:  # noqa: BLE001
            return None
    try:
        model = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
        ransac = RANSACRegressor(estimator=model, min_samples=0.3,
                                 residual_threshold=residual_threshold, random_state=42)
        ransac.fit(x[valid].reshape(-1, 1), edge[valid])
        return ransac.predict(x.reshape(-1, 1))
    except Exception:  # noqa: BLE001
        try:
            return np.polyval(np.polyfit(x[valid], edge[valid], deg), x)
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
    # GATED FAINT→ONSET SNAP (faint_snap_frac): the DP can settle on a faint pre-epithelial reflection ABOVE the
    # true surface — a dim point with the bright cornea only further below (measured on CS004: auto sits on
    # intensity ~760, the user's true surface on ~1840). This is a TARGETED, GATED correction: for each frame,
    # ONLY where the detected point is DIM (< faint_snap_frac × column-max) do we look just below for the ONSET of
    # the sustained bright band and snap to it. Because it fires only on faint points, a surface already on bright
    # tissue is left byte-untouched (no overshoot on clean scans); because it snaps to the FIRST sustained-bright
    # depth (not the brightest), it lands on the epithelial onset, never diving into the stroma / a deeper layer.
    # Global auto only (prior is None → not the fix-columns windowed re-detect). 0 = off (legacy).
    # NOTE: the DEFAULT is the laterally-COHERENT snap (_faint_snap_coherent), applied on the ASSEMBLED surface by
    # detect_surface_all + smooth_volume — it smooths the snap CORRECTION across laterals so it can't inject the
    # per-column jitter this per-slice version does at faint acquisition-edge frames (CS001 OD slice 1). This
    # per-slice path stays as a fallback (faint_snap_coherent=False).
    _fsf = float(p.get("faint_snap_frac", 0.0) or 0.0)
    if _fsf > 0 and prior is None and not bool(p.get("faint_snap_coherent", True)):
        # check against a LIGHTLY-smoothed RAW slice (not the heavy despeckle `img`, whose Gaussian blends the
        # faint point with the nearby band and hides the faintness); the sustained-window mean rejects speckle.
        chk = ndimage.gaussian_filter1d(slice_img.astype(np.float32), 1.0, axis=0)
        cmax = chk.max(axis=0) + 1e-6
        onf = float(p.get("faint_snap_onset", 0.55)); suf = float(p.get("faint_snap_sustain", 0.42))
        rng = int(p.get("faint_snap_range", 14)); sus = max(2, int(p.get("faint_snap_sustain_px", 6)))
        fc = np.arange(F)
        s0 = np.clip(np.round(np.asarray(out)).astype(int), 0, D - 1)
        faint = chk[s0, fc] < _fsf * cmax                          # only DIM detections are candidates for the snap
        if np.any(faint):
            hi = onf * cmax; lo = suf * cmax
            csum = np.concatenate([np.zeros((1, F), np.float32), np.cumsum(chk, axis=0)], axis=0)  # (D+1,F)
            out = np.asarray(out, dtype=np.float32).copy()
            found = np.zeros(F, bool)
            for k in range(1, rng + 1):                            # vectorised: first sustained-bright onset below
                dk = np.clip(s0 + k, 0, D - 1)
                d2 = np.clip(dk + sus, 0, D)
                roll = (csum[d2, fc] - csum[dk, fc]) / np.maximum(d2 - dk, 1)
                hit = faint & ~found & (chk[dk, fc] > hi) & (roll > lo)
                out[hit] = dk[hit].astype(np.float32); found |= hit
    return out


def _faint_snap_coherent(surf: np.ndarray, vol: np.ndarray, p: dict) -> np.ndarray:
    """Laterally-COHERENT faint→onset snap — the ROBUST version of the per-slice snap in _detect_surface_dp.
    Same gated correction (where a detected point is DIM, snap DOWN to the ONSET of the sustained bright band
    below — fixes the auto detector's ~4px shallow bias), but computed on the ASSEMBLED (lateral, frame) surface
    and then the CORRECTION (delta) is SMOOTHED across laterals. The per-slice snap fires independently per
    column, so at faint acquisition-edge frames adjacent slices snap to slightly different onsets → per-lateral
    jitter = an axial/en-face "fuzzy" border (CS001 OD slice 1: surface roughness 0.18→1.02). Smoothing the delta
    (not the whole surface — un-snapped columns stay exact) removes that jitter while keeping the deeper on-
    epithelium position: the snap can no longer inject sagittal slice-to-slice misalignment anywhere.
    surf=(L,F) detected surface; vol=(L,D,F) volume it was detected on. faint_snap_frac<=0 → no-op."""
    fsf = float(p.get("faint_snap_frac", 0.0) or 0.0)
    if fsf <= 0 or surf.ndim != 2:
        return surf
    L, D, F = vol.shape
    onf = float(p.get("faint_snap_onset", 0.55)); suf = float(p.get("faint_snap_sustain", 0.42))
    rng = int(p.get("faint_snap_range", 14)); sus = max(2, int(p.get("faint_snap_sustain_px", 6)))
    latsig = float(p.get("faint_snap_lat_smooth", 2.0) or 0.0)
    surf = np.asarray(surf, dtype=np.float32)
    snapped = surf.copy()
    fc = np.arange(F)
    for li in range(L):
        chk = ndimage.gaussian_filter1d(vol[li].astype(np.float32), 1.0, axis=0)   # lightly-smoothed raw (D,F)
        cmax = chk.max(axis=0) + 1e-6
        s0 = np.clip(np.round(surf[li]).astype(int), 0, D - 1)
        faint = chk[s0, fc] < fsf * cmax                                           # DIM detections only
        if not np.any(faint):
            continue
        hi = onf * cmax; lo = suf * cmax
        csum = np.concatenate([np.zeros((1, F), np.float32), np.cumsum(chk, axis=0)], axis=0)
        found = np.zeros(F, bool)
        for k in range(1, rng + 1):                                                # first sustained-bright onset below
            dk = np.clip(s0 + k, 0, D - 1); d2 = np.clip(dk + sus, 0, D)
            roll = (csum[d2, fc] - csum[dk, fc]) / np.maximum(d2 - dk, 1)
            hit = faint & ~found & (chk[dk, fc] > hi) & (roll > lo)
            snapped[li, hit] = dk[hit].astype(np.float32); found |= hit
    delta = snapped - surf
    if latsig > 0:
        delta = ndimage.gaussian_filter1d(delta.astype(np.float64), latsig, axis=0, mode="nearest")
    return (surf + delta).astype(np.float32)


def _edge_regularize_surface(surf: np.ndarray, vol: np.ndarray, p: dict) -> np.ndarray:
    """EDGE REGULARIZATION for low-SNR FOV-boundary laterals. At the extreme lateral edges the cornea is exiting
    the field of view — the OCT signal is weak, so the per-slice surface jitters across frames (a jagged sagittal
    border, e.g. CS001 OD__4 lateral 0/1). Fix: smooth the surface ACROSS FRAMES (axis 1) with a sigma tapered by
    each LATERAL's detection confidence — strong on the faint edge laterals, a strict NO-OP on the confident
    interior. Crucially this is the FRAME direction, so it removes the jitter WITHOUT changing the lateral's depth
    (no over-shallowing of the descending dome) and without pulling toward the interior. Confidence = mean
    brightness of the band just below the surface, normalised by the scan's p75 (low where the cornea is faint).
    surf=(L,F); vol=(L,D,F) the volume surf was detected on. edge_regularize=False or edge_reg_frame_sigma<=0 →
    no-op. Complements _lateral_smooth_by_confidence (which handles low-confidence FRAMES across laterals)."""
    if not bool(p.get("edge_regularize", True)) or surf.ndim != 2:
        return surf
    base = float(p.get("edge_reg_frame_sigma", 20.0) or 0.0)
    thr = float(p.get("edge_reg_conf_thr", 0.8) or 0.0)
    if base <= 0 or thr <= 0:
        return surf
    L, D, F = vol.shape
    fc = np.arange(F)
    band = np.empty((L, F), np.float32)
    for li in range(L):
        s = np.clip(np.round(surf[li]).astype(int), 0, D - 9)
        band[li] = np.mean([vol[li][np.clip(s + k, 0, D - 1), fc] for k in range(2, 9)], axis=0)
    conf = np.clip(band / (np.percentile(band, 75) + 1e-6), 0.0, 1.0).mean(axis=1)   # per-lateral confidence
    # OUTER-BAND FLOOR: the confidence taper is a NO-OP on high-confidence laterals — but the very outermost
    # laterals near the FOV boundary can be BRIGHT (high confidence) yet still carry frame-to-frame detection
    # JITTER at the low-SNR margin (CS001 OD__4/__5 marked edge: conf ~0.95 → escapes the taper → jitter ~3px,
    # rougher than the interior). So on the outermost `outer` laterals of EACH edge, apply a small FLOOR sigma
    # regardless of confidence: it de-jitters the tissue-bearing edge while a gentle sigma preserves the slowly
    # varying corneal dome (real anatomy changes over ~20+ frames; the floor only removes 1-few-frame jitter).
    outer = int(p.get("edge_reg_outer_band", 15)); floor = float(p.get("edge_reg_outer_floor", 4.0) or 0.0)
    out = surf.astype(np.float64).copy()
    for li in range(L):
        sig = base * max(0.0, (thr - conf[li]) / thr)        # 0 above thr (confident interior); grows as conf→0
        ed = min(li, L - 1 - li)
        if floor > 0 and ed < outer:
            # TAPERED floor: FULL at the FOV boundary (ed=0), fading LINEARLY to 0 at the band seam (ed=outer).
            # A HARD floor on 0..outer-1 (and none at `outer`) frame-smooths one side of the seam but not the
            # other → the smoothing pulls the low-SNR EDGE-FRAME value toward its neighbours on the inner side,
            # leaving a ~floor-px STEP in the anterior surface exactly at lateral `outer` on the acquisition-edge
            # frames (CS001 OD, "step near the ends of axial slice 1/101 & 101/101"). Tapering makes the smoothing
            # continuous across the seam → the step becomes a gentle ramp → no visible discontinuity, while the
            # extreme edge (where jitter is worst) still gets the full floor.
            sig = max(sig, floor * (outer - ed) / outer)
        if sig > 0.3:
            out[li] = ndimage.gaussian_filter1d(surf[li].astype(np.float64), sig, mode="nearest")
    return out.astype(surf.dtype)


def _edge_dome_constrain(surf: np.ndarray, p: dict) -> np.ndarray:
    """EDGE DOME CONSTRAINT — pull DOWNWARD hooks at the FOV-boundary laterals back onto the general corneal
    curve. At the extreme lateral edges the detector can DIVE a few px DEEPER than the smooth dome (following a
    faint deeper structure at low SNR) → the sagittal/axial border 'doesn't follow the general curve' (CS001 OD
    scans 4/5, user-marked laterals 0-13). Per FRAME, fit a robust quadratic TREND to the confident band just
    PAST each edge (skipping the hook itself), then snap ONLY the edge laterals that sit MORE THAN edge_dome_gate
    px BELOW that trend up onto it. DOWNWARD-only by construction → it can never push a shallower real LIMBUS
    flattening DOWN (the trap that retired _parabola_edge_constrain), never over-descends, and touches only the
    outer edge_dome_ew laterals (interior byte-untouched). Validated: CS001 OD__4/__5 dives removed, 0 downward
    moves on any scan, interior 0px. edge_dome_constrain=False → off. surf=(L,F)."""
    if not bool(p.get("edge_dome_constrain", True)) or surf.ndim != 2:
        return surf
    ew = int(p.get("edge_dome_ew", 18)); band = int(p.get("edge_dome_band", 100))
    gate = float(p.get("edge_dome_gate", 1.2) or 0.0)
    L, F = surf.shape
    if gate <= 0 or L < 2 * (ew + 30):
        return surf
    out = surf.astype(np.float64).copy()
    for fr in range(F):
        y = surf[:, fr].astype(np.float64)
        for lo, edge in ((0, np.arange(0, ew)), (L - ew, np.arange(L - ew, L))):
            b0 = ew if lo == 0 else lo - band
            m = np.arange(max(0, b0), min(L, b0 + band))
            if m.size < 25:
                continue
            co = np.polyfit(m, y[m], 2)
            for _ in range(2):                               # robust: drop outliers, refit the trend
                r = y[m] - np.polyval(co, m); sd = np.std(r) + 1e-6
                m = m[np.abs(r) < 2.5 * sd]
                if m.size < 20:
                    break
                co = np.polyfit(m, y[m], 2)
            fit = np.polyval(co, edge)
            hit = (y[edge] - fit) > gate                     # DOWNWARD dive only → pull up to the trend
            if np.any(hit):
                out[edge[hit], fr] = fit[hit]
    return out.astype(surf.dtype)


def _edge_dome_follow(surf: np.ndarray, vol: np.ndarray, p: dict) -> np.ndarray:
    """FAINT-EDGE DOME FOLLOW — where the cornea is EXITING the FOV (extreme lateral edges, weak signal) the DP
    detector locks onto a faint PRE-epithelial reflection ABOVE the true surface and the boundary FLATTENS off the
    corneal curve (CS001 OD, user: "the edge doesn't follow the overall corneal curvature"). Verified on the
    A-scans: the edge surface sits on DIM signal (~400-900) while the TRUE epithelium is BRIGHTER (~1100-1300) and
    a few px DEEPER, right where the dome predicts. TRIGGER = the faint-snap signature: within the edge band, a
    laterally-guided window below the surface holds a SUSTAINED band markedly BRIGHTER than the surface itself
    (peak ≥ `edge_follow_ratio`× the surface intensity). When it fires, SNAP the surface DOWN to that band
    (descend-only), bounded above by the robust dome fit of the reliable interior laterals + `edge_follow_pad` so
    it can never dive past the epithelium into stroma. The trigger self-limits: it stops at the first lateral whose
    surface already sits on the bright band (nothing brighter below → no change), so the snapped edge joins the
    correct interior with NO seam and follows the curve WITHOUT the over-descent V-notch that retired
    _parabola_edge_constrain. Auto path only. edge_dome_follow=False → off. surf=(L,F); vol=(L,D,F)."""
    if not bool(p.get("edge_dome_follow", True)) or surf.ndim != 2:
        return surf
    ew = int(p.get("edge_follow_band", 28)); fitb = int(p.get("edge_follow_fit", 90))
    pad = int(p.get("edge_follow_pad", 6)); sustain = int(p.get("edge_follow_sustain", 5))
    search = int(p.get("edge_follow_search", 20)); ratio = float(p.get("edge_follow_ratio", 1.4))
    L, F = surf.shape; D = vol.shape[1]
    if L < 2 * (ew + fitb) or ratio <= 1.0:
        return surf
    ker = np.ones(max(1, sustain)) / max(1, sustain)
    mwl = int(p.get("edge_follow_med", 5)); mwf = int(p.get("edge_follow_med_frame", 3))
    sgl = float(p.get("edge_follow_smooth", 1.6)); sgf = float(p.get("edge_follow_smooth_frame", 1.6))
    anchor = 8
    out = surf.astype(np.float64).copy()

    # ── pass 1: per (lateral, frame) in each edge band, snap the RAW target down to the bright band ──
    snap = surf.astype(np.float64).copy()                   # raw snap field (= surf where the trigger doesn't fire)
    fired_any = False
    for fr in range(F):
        y = surf[:, fr].astype(np.float64)
        for edge, fitrng in ((np.arange(0, ew), np.arange(ew, ew + fitb)),
                             (np.arange(L - ew, L), np.arange(L - ew - fitb, L - ew))):
            m = np.array(fitrng)                            # robust quadratic dome of the reliable interior band
            if m.size < 20:
                continue
            co = np.polyfit(m, y[m], 2)
            for _ in range(2):                              # drop outliers, refit
                r = y[m] - np.polyval(co, m); sd = np.std(r) + 1e-6; m = m[np.abs(r) < 2.5 * sd]
                if m.size < 15:
                    break
                co = np.polyfit(m, y[m], 2)
            dome = np.polyval(co, edge)
            for k, li in enumerate(edge):
                if not np.isfinite(y[li]):                  # surface extrapolated OUT of frame (clipped apex) → no band below to follow
                    continue
                s = int(round(float(y[li])))
                sc = max(0, min(D - 1, s))                  # clamp for indexing (a negative/above-FOV depth has no valid slice)
                surf_int = float(np.mean(vol[li, sc:min(D, sc + 2), fr]))         # brightness AT the surface
                s_lo = max(0, s + 2)
                s_hi = int(min(D - 1, min(round(dome[k]) + pad, s + search)))    # dome-bounded search below
                if s_hi <= s_lo + sustain:
                    continue
                seg = vol[li, s_lo:s_hi + 1, fr].astype(np.float64)
                if seg.size <= sustain:                     # too few depth samples for a valid sustained-run convolution
                    continue
                run = np.convolve(seg, ker, mode="valid")   # sustained brightness (epithelium ≫ pre-reflection)
                pk = int(np.argmax(run))
                if run[pk] < ratio * (surf_int + 1e-6):     # nothing markedly brighter below → surface is right
                    continue
                snap[li, fr] = float(min(s_lo + pk + (sustain - 1) // 2, s_hi))  # snap DOWN to the bright band
                fired_any = True
    if not fired_any:
        return surf

    def _med(a: np.ndarray, w: int, ax: int) -> np.ndarray:
        if w <= 1:
            return a
        r = w // 2; b = np.pad(a, [(r, r) if i == ax else (0, 0) for i in range(a.ndim)], mode="edge")
        sl = [slice(None)] * a.ndim
        stack = []
        for j in range(w):
            sl[ax] = slice(j, j + a.shape[ax]); stack.append(b[tuple(sl)])
        return np.median(np.stack(stack, 0), axis=0)

    def _gauss(a: np.ndarray, sg: float, ax: int) -> np.ndarray:
        if sg <= 0:
            return a
        rad = max(1, int(3 * sg)); x = np.arange(-rad, rad + 1)
        k = np.exp(-x * x / (2 * sg * sg)); k /= k.sum()
        b = np.pad(a, [(rad, rad) if i == ax else (0, 0) for i in range(a.ndim)], mode="edge")
        return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), ax, b)

    # ── pass 2: 2-D smooth each edge patch (lat×frame) with the untouched interior as a lateral anchor ──
    for edge, inner in ((np.arange(0, ew), np.arange(ew, ew + anchor)),
                        (np.arange(L - ew, L), np.arange(L - ew - anchor, L - ew))):
        left = edge[0] == 0
        idx = np.concatenate([edge, inner]) if left else np.concatenate([inner, edge])
        patch = snap[idx, :].copy()                         # (ew+anchor, F) raw snap + interior anchor rows
        patch = _med(patch, mwl, 0); patch = _med(patch, mwf, 1)
        patch = _gauss(patch, sgl, 0); patch = _gauss(patch, sgf, 1)
        out[edge, :] = patch[:ew, :] if left else patch[anchor:, :]
    return out.astype(surf.dtype)


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


def guided_posterior_row(sag: np.ndarray, si: int, anchors: dict, params: dict | None = None,
                         window: float | None = None) -> np.ndarray | None:
    """The POSTERIOR (bottom) edge of slice `si`, DETECTED but guided by the reviewer's posterior corrections.

    THE GAP THIS CLOSES. crop_post_anchors had exactly one reader in the pipeline — a per-frame override inside
    the surface-crop reconstruction — so a posterior correction changed the line on the slice it was drawn on
    and nowhere else. Re-running could not improve the bottom edge anywhere, by construction, which is the
    reviewer's "bottom edge detection does not seem to improve after iterations".

    THE MECHANISM is the reviewer's own: the correction says WHERE TO LOOK, the detector still decides where
    the edge IS. Their residual (drawn depth − auto depth) is measured on the anchored slices, interpolated
    across slices to `si`, added to si's own auto posterior to form a prior, and _detect_bottom_edge then
    searches for the real bright→dark gradient within a window of it. Interpolating the DRAWN DEPTHS instead
    would just spread a hand-drawn line — measured on the anterior, that reproduced auto exactly on a held-out
    slice while guided detection cut the median error from 11.1 px to 3.0 px.

    Cheap by design: it detects only the anchored slices plus `si`, not the volume, so it is affordable inside
    a per-slice preview. Returns None when there is nothing to guide with."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    n, _D, F = int(sag.shape[0]), int(sag.shape[1]), int(sag.shape[2])
    if not (0 <= si < n) or not isinstance(anchors, dict) or not anchors:
        return None
    # anchored slices → {slice: {frame: depth}}, skipping the absent sentinel (no posterior on that frame)
    anc: dict[int, dict[int, float]] = {}
    for s_key, frames in anchors.items():
        try:
            sa = int(s_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= sa < n) or not isinstance(frames, dict):
            continue
        fm = {}
        for f_key, d in frames.items():
            try:
                fi = int(f_key); dv = float(d)
            except (TypeError, ValueError):
                continue
            if 0 <= fi < F and np.isfinite(dv) and 0.0 <= dv < _D - 1:
                fm[fi] = dv
        if fm:
            anc[sa] = fm
    if not anc:
        return None
    # residual per anchored slice, on its OWN auto posterior — the residual is what generalises, not the depth
    res: dict[int, np.ndarray] = {}
    for sa, fm in anc.items():
        b = _detect_bottom_edge(np.ascontiguousarray(sag[sa]).astype(np.float32), p)
        r = np.full(F, np.nan)
        for fi, dv in fm.items():
            if np.isfinite(b[fi]):
                r[fi] = dv - float(b[fi])
        if np.isfinite(r).any():
            res[sa] = r
    if not res:
        return None
    # interpolate each frame's residual across the anchored slices (edge-hold beyond the anchored span, the
    # same shape generalize_surface uses); with one anchored slice this is a constant carry, which is the most
    # a single slice can honestly support
    keys = sorted(res)
    R = np.full(F, np.nan)
    for f in range(F):
        vals = [(sa, res[sa][f]) for sa in keys if np.isfinite(res[sa][f])]
        if not vals:
            continue
        if len(vals) == 1:
            R[f] = vals[0][1]
        else:
            xs = np.array([v[0] for v in vals], dtype=np.float64)
            ys = np.array([v[1] for v in vals], dtype=np.float64)
            R[f] = float(np.interp(float(si), xs, ys))             # edge-hold outside the span
    if not np.isfinite(R).any():
        return None
    # TAPER BEYOND THE ANCHORED SPAN. Without it the residual is edge-held forever, so a correction on slice
    # 195 was carried unchanged to slice 300 and moved the detected posterior by 166 px on average — not a
    # refinement, a relocation. generalize_surface tapers for exactly this reason; a correction is evidence
    # about the cornea NEAR where it was drawn, and claiming it 100 slices away is not supported by anything.
    # Outside the taper the function returns None rather than a zero residual: with no correction in reach
    # there is nothing to guide with, and the caller should use its ordinary detection rather than a windowed
    # search around a prior that is merely auto by another name.
    _lo, _hi = min(keys), max(keys)
    _tap = float(p.get("post_taper_slices", p.get("gen_taper_slices", 20)) or 0.0)
    if si < _lo:
        _d = float(_lo - si)
    elif si > _hi:
        _d = float(si - _hi)
    else:
        _d = 0.0
    if _tap <= 0 and _d > 0:
        return None
    _w = 1.0 if _d <= 0 else max(0.0, 1.0 - _d / _tap)
    if _w <= 1e-3:
        return None
    R = R * _w
    # smooth across frames so a couple of corrected frames do not produce a step in the prior
    Rf = R.copy()
    ok = np.isfinite(Rf)
    if ok.sum() >= 2:
        Rf = np.interp(np.arange(F), np.where(ok)[0], Rf[ok])
    else:
        Rf = np.where(ok, Rf, float(np.nanmean(Rf)))
    sig = float(p.get("post_prior_smooth", 3.0) or 0.0)
    if sig > 0:
        Rf = ndimage.gaussian_filter1d(Rf, sigma=sig, mode="nearest")
    sl = np.ascontiguousarray(sag[si]).astype(np.float32)
    prior = _detect_bottom_edge(sl, p).astype(np.float64) + Rf
    win = float(window if window is not None else p.get("post_guided_window", 30.0))
    return _detect_bottom_edge(sl, {**p, "detect_window": win}, prior=prior.astype(np.float32))


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
                        light: bool = True, adapt: dict | None = None) -> np.ndarray:
    """Re-detect one sagittal slice's corneal surface within ±window of a per-frame `prior`. sl=(depth,
    frames). light=False (seed slices) uses the full robust detector (_merged_side_edge: hist-eq/raw choice
    + side-correction). light=True (the MARCH, called for every slice) uses a fast windowed gradient argmax
    + outlier/median cleanup — no bilateral / hist-eq / double-RANSAC — which is reliable because the tight
    window around the resolved neighbour already excludes confounders (and is ~10x faster, so a 513-slice
    march finishes in seconds instead of minutes)."""
    pr = np.asarray(prior, dtype=np.float32)
    if not light:
        return _merged_side_edge(sl, {**p, "detect_window": float(window)}, prior=pr)
    raw = _detect_surface_gradient(sl, sigma=float(p["sigma"]), prior=pr, window=float(window), adapt=adapt)
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
    out = _surface_post_passes(out, sag, p)
    # #9 v3: make the DISPLAYED surface IGNORE any marked artifact band — the raw detector dives into the artifact
    # inside the band; replace it with a smooth reconstruction from the cornea on either side. Last, so nothing
    # re-introduces the dive. No-op without crop_bands; does not affect the flatten (which excludes the band).
    out = _reconstruct_surface_over_bands(out, p)
    return out


def _frame_edge_epithelium_snap(surf: np.ndarray, vol: np.ndarray, p: dict) -> np.ndarray:
    """FRAME-EDGE faint-edge snap — the FRAME-axis sibling of _edge_dome_follow. At the low-signal LEFT/RIGHT
    frame edges of a B-scan the DP detector locks onto a faint PRE-epithelial reflection ABOVE the true, brighter
    epithelium so the displayed border floats off the tissue (reviewer cs046: "the left edge floats above the
    tissue"). _edge_dome_follow only covers the LATERAL FOV edges. Per LATERAL, in each frame-edge band, if a
    SUSTAINED band below the surface is markedly brighter (peak >= edge_follow_ratio x the surface intensity)
    within a frame-direction-dome-bounded window, SNAP the surface DOWN to it (descend-only). Same safety as
    _edge_dome_follow: it can never dive past the epithelium (dome+pad bound), fires only on the faint-snap
    signature, and is a strict no-op where the surface already sits on the bright band. frame_edge_snap=False →
    off. surf=(L,F); vol=(L,D,F)."""
    if not bool(p.get("frame_edge_snap", True)) or surf.ndim != 2:
        return surf
    ewf = int(p.get("frame_edge_band", 12))            # first/last N FRAMES = the edge band
    fitf = int(p.get("frame_edge_fit", 40))            # interior frames for the frame-direction dome bound
    # dome bound uses a LARGER pad than the lateral sibling: the cornea descends toward the frame edge faster than
    # the interior quadratic predicts, so a tight pad stops ~2px short of the real epithelium (cs046). 14px still
    # can't reach the posterior (≫30px away), so it stays descend-safe.
    pad = int(p.get("frame_edge_pad", 14)); sustain = int(p.get("edge_follow_sustain", 5))
    search = int(p.get("edge_follow_search", 20)); ratio = float(p.get("edge_follow_ratio", 1.4))
    L, F = surf.shape; D = vol.shape[1]
    if F < 2 * ewf + 10 or ratio <= 1.0:
        return surf
    ker = np.ones(max(1, sustain)) / max(1, sustain)
    out = surf.astype(np.float64).copy()
    fired = False
    for li in range(L):
        y = surf[li, :].astype(np.float64)
        for edge, fitrng in ((np.arange(0, ewf), np.arange(ewf, min(F, ewf + fitf))),
                             (np.arange(max(0, F - ewf), F), np.arange(max(0, F - ewf - fitf), max(0, F - ewf)))):
            m = np.array([f for f in fitrng if np.isfinite(y[f]) and 0 <= y[f] < D - 1])
            if m.size < 10:
                continue
            co = np.polyfit(m, y[m], 2)
            for _ in range(2):                          # drop outliers, refit (the frame-direction dome of the interior)
                r = y[m] - np.polyval(co, m); sd = np.std(r) + 1e-6; m = m[np.abs(r) < 2.5 * sd]
                if m.size < 8:
                    break
                co = np.polyfit(m, y[m], 2)
            dome = np.polyval(co, edge)
            for k, f in enumerate(edge):
                if not np.isfinite(y[f]):               # extrapolated out of frame (clipped apex) → no band to follow
                    continue
                s = int(round(float(y[f]))); sc = max(0, min(D - 1, s))
                surf_int = float(np.mean(vol[li, sc:min(D, sc + 2), f]))       # brightness AT the surface
                s_lo = max(0, s + 2)
                s_hi = int(min(D - 1, min(round(float(dome[k])) + pad, s + search)))   # dome-bounded search below
                if s_hi <= s_lo + sustain:
                    continue
                seg = vol[li, s_lo:s_hi + 1, f].astype(np.float64)
                if seg.size <= sustain:
                    continue
                run = np.convolve(seg, ker, mode="valid")                     # sustained brightness (epithelium ≫ reflection)
                pk = int(np.argmax(run))
                if run[pk] < ratio * (surf_int + 1e-6):                        # nothing markedly brighter below → surface is right
                    continue
                out[li, f] = float(min(s_lo + pk + (sustain - 1) // 2, s_hi))  # snap DOWN to the bright band
                fired = True
    if not fired:
        return surf
    # ISOLATED-SPIKE GUARD: on a noisy edge frame the snap can over-dive a SINGLE frame onto a stray bright blob
    # (cs046 lat300 frame 5). Kill only frames that sit DEEPER than BOTH frame-neighbours by >spike_tol — a
    # sustained descent (every real snap) is never an outlier vs its neighbours, so it is untouched.
    tol = float(p.get("frame_edge_spike_tol", 2.5))
    o = out.copy()
    for li in range(L):
        for f in list(range(1, ewf)) + list(range(max(1, F - ewf), F - 1)):
            a, b, c = o[li, f - 1], o[li, f], o[li, f + 1]
            if b - a > tol and b - c > tol:                             # isolated DOWNWARD spike → median it out
                out[li, f] = float(np.median([a, b, c]))
    return out.astype(surf.dtype)


def _corner_edge_retrace(surf, vol, p=None):
    """Relaxed-DP re-trace of the anterior surface in the LEFT/RIGHT frame-edge band, to follow a
    steep limbal descent the depth-capped global DP (dp_max_jump=10) flatlines over. Per lateral, per
    edge: anchor to the trusted interior, run a near-MONOTONE-descent DP over the un-normalised
    dark->bright onset score, and splice the trace in ONLY where it descends past BASE onto genuine
    markedly-brighter epithelium. Interior frames untouched. No GT. Safe/descend-gated:
      * whole lateral skipped unless its interior surface itself sits on real bright tissue
        ((ref-median) >= ref_valid_k*MAD)  -> no-op on a floating/flat corner;
      * per frame accepted only if it descends past BASE (>tol), is bright_ratio x brighter than where
        BASE sat, AND >= tissue_frac of THIS lateral's interior-epithelium brightness (rejects dark
        air/speckle); dark->bright onset + tot-descent bound stop it diving onto deeper stroma.
    surf=(L,F) BASE; vol=(L,D,F). corner_retrace=False -> strict no-op.

    SELECTED (cs046 steep-limbus corner) via a 4-approach fan-out judged against the reviewer's manual
    anchors: local relaxed-DP won over greedy band-follow on SAFETY — band-follow false-descended ~25px
    onto dark speckle (its gate is only relative to the current surface brightness), whereas this
    lateral-validity + onset + tissue-frac gate blocks that. corner=3.68px vs anchors (baseline 21.5),
    interior byte-safe, no-op on flat corners. corner_retrace kill-switch + dp-v6 algo bump."""
    p = p or {}
    if not bool(p.get("corner_retrace", True)):
        return surf
    out = np.asarray(surf, np.float64).copy()
    if out.ndim != 2:
        return out
    L, F = out.shape; D = vol.shape[1]
    band     = int(p.get("corner_band", 13))
    amed     = int(p.get("corner_anchor_med", 4))
    maxdown  = int(p.get("corner_maxdown", 30))
    maxup    = int(p.get("corner_maxup", 4))
    smooth_w = float(p.get("corner_smooth_w", 0.03))
    up_pen   = float(p.get("corner_up_pen", 0.3))
    sustain  = int(p.get("corner_sustain", 10))
    above    = int(p.get("corner_above", 16))
    ridge    = int(p.get("corner_ridge", 6))
    bright_ratio = float(p.get("corner_bright_ratio", 1.10))
    tissue_frac  = float(p.get("corner_tissue_frac", 0.40))
    ref_valid_k  = float(p.get("corner_ref_valid_k", 2.5))
    tol      = float(p.get("corner_tol", 2.0))
    max_desc = float(p.get("corner_max_descent", 130))
    # HARDENING 1 (mandatory): corner-namespaced smoothing. The prototype read dp_sigma_depth/frame,
    # which DEFAULT_PARAMS sets to 3.0/3.0 -> over-smooths the faint onset and regresses the corner to
    # baseline (18px). Namespaced keys decouple it from the DP detector's global smoothing.
    sig_d = float(p.get("corner_sig_depth", 1.5)); sig_f = float(p.get("corner_sig_frame", 1.2))
    # HARDENING 2: interior reference window as a FRACTION of F (not hardcoded 35/66), generalizing to
    # any frame count; clamped to the trusted interior.
    rf0 = p.get("corner_ref_f0", None); rf1 = p.get("corner_ref_f1", None)
    if rf0 is None: rf0 = int(round(0.35 * F))
    if rf1 is None: rf1 = int(round(0.65 * F))
    ref_f0 = int(max(band + amed, rf0)); ref_f1 = int(min(F - band - amed, rf1))
    if ref_f1 <= ref_f0:
        ref_f0 = max(0, F // 2 - 3); ref_f1 = min(F, F // 2 + 3)
    if F < band + amed + 4 or maxdown <= 0:
        return out
    sw = sustain // 2

    def _wmeans(im, wb, wa):
        Dl, Fl = im.shape
        cs = np.concatenate([np.zeros((1, Fl), np.float32), np.cumsum(im, 0)], 0)
        d = np.arange(Dl)
        d2 = np.minimum(d + wb, Dl); below = (cs[d2] - cs[d]) / np.maximum(d2 - d, 1)[:, None]
        d0 = np.maximum(d - wa, 0);  ab = (cs[d] - cs[d0]) / np.maximum(d - d0, 1)[:, None]
        return below, ab

    def _dp(score, cols, anchor_depth, dmin, dmax):
        Dl = score.shape[0]; W = len(cols); BIG = 1e9
        cost = (-score[:, cols]).astype(np.float32)
        for j in range(W):
            m = np.ones(Dl, bool); m[int(max(0, dmin[j])):int(min(Dl, dmax[j] + 1))] = False; cost[m, j] += BIG
        ad = int(max(0, min(Dl - 1, round(anchor_depth))))
        c0 = np.full(Dl, BIG, np.float32); c0[ad] = 0.0; cost[:, 0] = c0
        offs = np.arange(-maxup, maxdown + 1)
        # asymmetric penalty: ascent (offs<0) costs up_pen/px -> near-monotone descent. Sign verified
        # against the shift convention below (o>0 == descent by o).
        pen = (smooth_w * np.abs(offs) + up_pen * np.maximum(0, -offs)).astype(np.float32)[:, None]
        dp = cost[:, 0].copy(); back = np.empty((Dl, W), np.int32)
        for j in range(1, W):
            cand = np.full((offs.size, Dl), np.inf, np.float32)
            for k, o in enumerate(offs):
                if o > 0:   cand[k, o:] = dp[:Dl - o]
                elif o < 0: cand[k, :Dl + o] = dp[-o:]
                else:       cand[k, :] = dp
            cand = cand + pen
            kb = np.argmin(cand, axis=0); dp = cand[kb, np.arange(Dl)] + cost[:, j]; back[:, j] = np.arange(Dl) - offs[kb]
        path = np.empty(W, np.int32); path[W - 1] = int(np.argmin(dp))
        for j in range(W - 1, 0, -1): path[j - 1] = back[path[j], j]
        return path

    for li in range(L):
        im = ndimage.gaussian_filter(vol[li].astype(np.float32), (sig_d, sig_f))
        below, ab = _wmeans(im, sustain, above)
        sc = np.clip(below - ab, 0.0, None)          # un-normalised onset score (normalising kills the faint edge)
        med = float(np.median(im)); mad = float(np.median(np.abs(im - med))) + 1e-6
        rr = []
        for f in range(ref_f0, min(ref_f1, F)):
            v = out[li, f]
            if not np.isfinite(v):
                continue
            d = int(round(v)); lo = max(0, d - sw); rr.append(im[lo:min(D, lo + sustain), f].mean())
        ref = float(np.median(rr)) if rr else 0.0
        if (ref - med) < ref_valid_k * mad:          # interior not on real tissue -> untrustworthy lateral, skip
            continue
        for side in ("R", "L"):
            if side == "R":
                free = list(range(F - band, F)); af = F - band - 1
                aref = np.nanmedian(out[li, af - amed + 1:af + 1]); cols = [af] + free
            else:
                free = list(range(band - 1, -1, -1)); af = band
                aref = np.nanmedian(out[li, af:af + amed]); cols = [af] + free
            if not np.isfinite(aref):                # clipped/no interior anchor -> skip side
                continue
            dmin = np.full(len(cols), aref - 6); dmax = np.full(len(cols), aref + max_desc)
            path = _dp(sc, cols, aref, dmin, dmax)
            for jj, f in enumerate(cols):
                if jj == 0:
                    continue
                if not np.isfinite(out[li, f]):      # extrapolated/clipped frame -> leave as-is
                    continue
                dp_d = int(path[jj]); b_d = int(round(out[li, f]))
                if dp_d - b_d <= tol:                                # must DESCEND past BASE
                    continue
                pd = dp_d
                if ridge > 0:
                    seg = im[dp_d:min(D, dp_d + ridge + 1), f]
                    if seg.size: pd = dp_d + int(np.argmax(seg))
                lo = max(0, pd - sw); dp_b = float(im[lo:min(D, lo + sustain), f].mean())
                bl = max(0, b_d - 1); b_b = float(im[bl:min(D, b_d + 2), f].mean())
                if dp_b < bright_ratio * (b_b + 1e-6):               # markedly brighter than BASE
                    continue
                if dp_b < tissue_frac * ref:                         # real epithelium, not dark air/speckle
                    continue
                out[li, f] = float(pd)
    return out.astype(surf.dtype)


def _surface_post_passes(out: np.ndarray, sag: np.ndarray, p: dict) -> np.ndarray:
    """The whole-volume tail every detected surface gets: specular reject, despike, dip suppression, robust
    dome smoothing, lateral confidence smoothing, faint-onset snap, the two edge-dome passes, untrusted-column
    repair and edge regularization.

    EXTRACTED so that a surface produced by any other route gets IDENTICAL treatment. Comparing a raw per-slice
    detection against this polished one is not a comparison of detectors, it is a comparison of one detector
    with and without smoothing — measured on real scans, that alone accounted for most of an apparent +13%
    roughness penalty against guided re-detection (+13% -> +2.5% once both were passed through here)."""
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
    # FAINT→ONSET SNAP (laterally COHERENT): fix the auto detector's ~4px shallow bias — it locks onto a faint
    # pre-epithelial reflection ABOVE the true surface; snap DIM detections down to the sustained bright-band
    # onset, with the correction smoothed across laterals so it never injects per-column jitter. Auto baseline
    # only (this function never receives a prior → it IS the auto path). See _faint_snap_coherent.
    if float(p.get("faint_snap_frac", 0.0) or 0.0) > 0 and bool(p.get("faint_snap_coherent", True)):
        out = _faint_snap_coherent(out, sag, p)
    # EDGE DOME CONSTRAINT (BEFORE edge-regularize): pull per-frame DOWNWARD hooks at the FOV-boundary laterals
    # back onto the general corneal curve (CS001 OD__4/__5). DOWNWARD-only → never over-descends a real limbus.
    # Runs FIRST so the subsequent frame-direction edge-regularize has the final say on edge smoothness (else it
    # would re-roughen the floor-smoothed outer laterals). See _edge_dome_constrain.
    out = _edge_dome_constrain(out, p)
    # FAINT-EDGE DOME FOLLOW (AFTER the downward-hook constraint, BEFORE the frame-smoothing): where the edge
    # surface floats on a faint PRE-epithelial reflection ABOVE the true (brighter) epithelium, snap it DOWN onto
    # that sustained band so the border FOLLOWS the corneal curve instead of flattening off it (CS001 OD, user:
    # "the edge doesn't follow the overall corneal curvature"). 2-D smoothed; interior byte-untouched. Complements
    # _edge_dome_constrain (which fixes the opposite, too-DEEP, defect). See _edge_dome_follow.
    out = _edge_dome_follow(out, sag, p)
    # FRAME-EDGE faint snap (frame-axis sibling of _edge_dome_follow): the SAME faint pre-epithelial float, but at
    # the LEFT/RIGHT FRAME edges of the B-scan (reviewer cs046: "the left edge floats above the tissue") — the
    # lateral-only _edge_dome_follow never touches it. Descend-only, frame-dome-bounded, same ratio trigger.
    out = _frame_edge_epithelium_snap(out, sag, p)
    # STEEP-LIMBUS CORNER RE-TRACE (dp-v6): the frame-edge snap above is depth-capped/dome-bounded and flatlines
    # where the epithelium plunges toward the limbus (~14+px/frame) — a local relaxed-DP re-trace follows that
    # descent onto the real band, matching the reviewer's manual anchors (cs046 corner 21.5→3.7px). Descend-gated,
    # whole-lateral tissue-validity skip, no-op on flat corners → never degrades a scan whose corner is already right.
    out = _corner_edge_retrace(out, sag, p)
    # EDGE REGULARIZATION: smooth the faint FOV-boundary laterals' jagged border across frames (depth-preserving),
    # a strict no-op on the confident interior. Handles the sagittal edge-slice jitter (CS001 OD__4 lateral 0/1)
    # AND, via the outer-band floor sigma, the BRIGHT tissue-bearing FOV edge (CS001 OD__4 visual-left / array-right).
    # UNTRUSTED-SURFACE REPAIR: replace columns the image gives no reason to trust (no band, an
    # eyelash/eyelid shadow, a reflection, or a sharp V) with the frame's own robust quadratic, so
    # every later stage sees a corneal surface rather than a trace over speckle. Default OFF.
    out = _repair_untrusted_surface(out, sag, p)
    out = _edge_regularize_surface(out, sag, p)
    # NOTE: _parabola_edge_constrain is DELIBERATELY NOT applied here. detect_surface_all is the detection
    # baseline used by the NOISE-CROP (detect_noise_frames), the surface-crop, the confidence scores and the
    # fix-columns re-detect — all of which need the TRUE tissue-following surface. Snapping the edge frames to
    # the parabola moves the surface off the faint edge tissue → its confidence collapses → the noise-crop
    # wrongly zeroes valid cornea at the edges (CS003 OD: 25 real frames cropped). The parabola-edge motion-
    # artifact correction is applied ONLY to the final WARP surface (smooth_volume), so the OUTPUT edges follow
    # the parabola while the detectors still see the real tissue.
    return out


def guided_redetect_all(sag: np.ndarray, prior: np.ndarray, params: dict | None = None,
                        window: float | None = None) -> np.ndarray:
    """Re-detect EVERY slice inside ±window of `prior`, then apply the standard surface tail.

    THE REVIEWER'S IDEA, and it measures better than the alternatives. Interpolating a drawn correction across
    the volume only spreads a hand-drawn line — on a held-out corrected slice it produced a surface identical
    to auto, because it never looked at the image there. This instead uses the correction to say WHERE TO LOOK
    and lets the detector decide WHERE THE EDGE IS, so the other 512 slices are detected rather than guessed.

    Measured against the reviewer's own corrections (prior = plain auto, so no correction even fed in), the
    median error on three corrected slices went 27.4→19.2, 11.1→3.0 and 14.9→12.0 px, and the surface sat on
    14-20% stronger gradients on every scan tried. Wider windows beat tight ones (±5px reproduced the prior's
    mistakes; ±40px found the real boundary), which is the signature of detection rather than interpolation.

    NOT SAFE UNCONDITIONALLY — at the delivered-volume level it helped two approved scans and hurt two others.
    Callers must guard it; see the api layer, which keeps this surface only when it measures better."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    win = float(window if window is not None else p.get("guided_window", 40.0))
    adapt = None
    if bool(p.get("guided_adaptive", True)):
        adapt = {"min_window": float(p.get("guided_window_min", 1.0)),
                 "conf_lo": float(p.get("guided_conf_lo", 0.20)),
                 "conf_hi": float(p.get("guided_conf_hi", 0.50)),
                 "pctl": float(p.get("guided_conf_pctl", 80.0))}
    pr = np.asarray(prior, dtype=np.float32)
    out = np.empty_like(pr)
    for s in range(int(sag.shape[0])):
        out[s] = _redetect_one_slice(np.ascontiguousarray(sag[s]).astype(np.float32),
                                     pr[s], win, p, light=True, adapt=adapt)
    return _surface_post_passes(out, sag, p)


def surface_gradient_score(sag: np.ndarray, S: np.ndarray, stride: int = 8) -> float:
    """Mean |d intensity / d depth| at the detected surface — how strong a boundary it is sitting on.

    A LABEL-FREE quality signal, and the detector's own objective measured independently of how the answer was
    reached, so it can judge a surface on the ~170 approved scans where no ground truth exists. Never used
    alone: a search that wanders finds strong gradients on other structures, so it is always paired with a
    roughness check (see shift_roughness)."""
    L, D, F = int(sag.shape[0]), int(sag.shape[1]), int(sag.shape[2])
    tot, n = 0.0, 0
    for s in range(0, L, max(1, int(stride))):
        d = np.clip(np.rint(np.asarray(S[s], dtype=np.float64)), 1, D - 2).astype(int)
        c = np.arange(F)
        g = np.abs(sag[s][d + 1, c].astype(np.float64) - sag[s][d - 1, c].astype(np.float64))
        tot += float(np.nansum(g)); n += int(np.isfinite(g).sum())
    return tot / max(1, n)


def shift_roughness(S: np.ndarray) -> float:
    """Across-frame jitter of the PER-FRAME SHIFT the warp will actually apply.

    Deliberately not the roughness of the surface itself. Each B-scan gets ONE rigid depth shift, taken as a
    median across all 513 laterals, so per-lateral jitter that is uncorrelated between laterals never reaches
    the volume. The reviewer approved the OUTPUT, so this — not the surface — is what a regression guard has
    to protect. On real scans the two disagree sharply: one scan's surface roughened 10% while its delivered
    shift series smoothed by 20%."""
    m = np.median(np.asarray(S, dtype=np.float64), axis=0)
    return float(np.mean(np.abs(np.diff(m, n=2))))


def anchor_error(S: np.ndarray, anchors: dict, depth: int) -> tuple[float, int]:
    """(median |surface − reviewer anchor|, n) over anchors that are actually detectable.

    Anchors ABOVE the captured window are excluded: on a surface-cropped cornea the reviewer legitimately
    draws the apex at negative depth, and no detector can return a row that does not exist. Scoring against
    them adds a large constant no candidate can reduce, which drowns the differences that matter — on one
    scan those 13 points alone moved the apparent error from 23 to 27 px."""
    errs: list[float] = []
    for s_key, frames in (anchors or {}).items():
        try:
            s = int(s_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= s < S.shape[0]) or not isinstance(frames, dict):
            continue
        for f_key, d in frames.items():
            try:
                f = int(f_key); dv = float(d)
            except (TypeError, ValueError):
                continue
            if 0 <= f < S.shape[1] and 0.0 <= dv < depth - 1 and np.isfinite(S[s, f]):
                errs.append(abs(float(S[s, f]) - dv))
    if not errs:
        return float("nan"), 0
    return float(np.median(errs)), len(errs)


def _repair_untrusted_surface(edges: np.ndarray, sag: np.ndarray, p: dict) -> np.ndarray:
    """Replace surface columns the image gives no reason to trust with the frame's own robust quadratic.

    This is a DETECTION repair, not a warp: it changes what the pipeline believes the surface is, so every later
    stage (the rigid fit, the crop, the labels) sees a corneal surface instead of a boundary traced over speckle
    or an eyelash. Nothing is deformed, so it is orthogonal to the rigid-only constraint rather than bounded by it.

    A column is untrusted when ANY of four signals fires, each measured against the SCAN'S OWN median:
      * its A-scan stops resembling its lateral neighbours (no band present to lock onto);
      * the tissue beneath the surface is much darker than the rest of that slice (the beam was blocked);
      * there is signal in the air above the surface (the blocker);
      * the surface sits far off its own frame's robust quadratic (the "sharp V").

    The replacement is the quadratic fitted to that frame's TRUSTED columns only, so the repaired region continues
    the cornea the frame itself shows. Frames with too few trusted columns, or where too much of the frame would
    be replaced, are left ALONE — past that point there is no evidence left to interpolate from and a "repair"
    would be an invention. edges=(n_lateral, n_frames), sag=(n_lateral, depth, n_frames).
    """
    if not bool(p.get("surface_repair", False)):
        return edges
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2 or sag is None or sag.ndim != 3 or sag.shape[0] != e.shape[0] or sag.shape[2] != e.shape[1]:
        return edges
    L, F = e.shape
    if L < 60 or F < 8:
        return edges
    v = np.asarray(sag, dtype=np.float32)
    D = v.shape[1]
    lat = np.arange(L, dtype=np.float64)

    # 1 — lateral coherence: does each A-scan resemble its neighbour?
    A = ndimage.uniform_filter1d(v, size=5, axis=1)
    A = A - A.mean(axis=1, keepdims=True)
    nrm = np.sqrt((A * A).sum(axis=1))
    coh = (A[:-1] * A[1:]).sum(axis=1) / np.maximum(nrm[:-1] * nrm[1:], 1e-6)
    coh = np.vstack([coh, coh[-1:]])
    coh = ndimage.uniform_filter(coh, size=(6, 3), mode="nearest")
    coh_bad = coh < (float(np.median(coh)) - float(p.get("srep_coh_drop", 0.06)))

    # 2/3 — the frame's own robust quadratic, then brightness below and above it
    exp = np.full((L, F), np.nan)
    dev = np.full((L, F), np.nan)
    for f in range(F):
        y = e[:, f]
        g = np.isfinite(y)
        if int(g.sum()) < 40:
            continue
        c0 = np.polyfit(lat[g], y[g], 2)
        r0 = y[g] - np.polyval(c0, lat[g])
        k = np.abs(r0) <= np.percentile(np.abs(r0), 80)
        q = np.polyval(np.polyfit(lat[g][k], y[g][k], 2), lat)
        exp[:, f] = q
        dev[:, f] = np.abs(y - q)
    # Gather both brightness windows with fancy indexing rather than a per-column loop: 513x101 columns is
    # 51,813 Python iterations per scan, which cost about two minutes a scan when this first ran over the corpus.
    base = np.where(np.isfinite(exp), np.nan_to_num(exp, nan=0.0), 0.0).astype(np.int64)
    ok = np.isfinite(exp)
    ii = np.arange(L)[:, None, None]
    ff = np.arange(F)[None, None, :]

    def _window(off_lo, off_hi):
        # One behavioural difference from the loop this replaced: an offset falling outside the canvas is
        # CLAMPED to the edge row rather than dropped from the average, so near the top and bottom of the volume
        # the window repeats the edge instead of shrinking. That is the steadier of the two — a shrinking window
        # changes its own noise level exactly where the surface is least reliable — and both are then normalised
        # by the scan median, so the threshold moves with it either way.
        offs = np.arange(off_lo, off_hi)[None, :, None]
        zz = np.clip(base[:, None, :] + offs, 0, D - 1)
        vals = v[ii, zz, ff].mean(axis=1)
        return np.where(ok, vals, np.nan)

    below = _window(5, 60)
    above = _window(-70, -10)
    def _norm(X):
        med = float(np.nanmedian(X)) if np.isfinite(X).any() else 1.0
        return ndimage.uniform_filter(np.nan_to_num(X, nan=med), size=(6, 3), mode="nearest") / max(med, 1e-6)
    below_n, above_n = _norm(below), _norm(above)
    bad = (coh_bad
           | (below_n < float(p.get("srep_shadow_frac", 0.85)))
           | (above_n > float(p.get("srep_bright_frac", 1.25)))
           | (np.nan_to_num(dev, nan=0.0) > float(p.get("srep_jump_px", 6.0))))

    out = e.copy()
    max_frac = float(p.get("srep_max_frac", 0.45))
    min_trusted = int(p.get("srep_min_trusted", 120))
    for f in range(F):
        m = bad[:, f]
        good = ~m & np.isfinite(e[:, f])
        if int(good.sum()) < min_trusted or int(m.sum()) == 0:
            continue
        if m.mean() > max_frac:
            continue                      # too little left to fit — leave the frame exactly as detected
        c0 = np.polyfit(lat[good], e[good, f], 2)
        r0 = e[good, f] - np.polyval(c0, lat[good])
        k = np.abs(r0) <= np.percentile(np.abs(r0), 90)
        q = np.polyval(np.polyfit(lat[good][k], e[good, f][k], 2), lat)
        out[m, f] = q[m]
    return out.astype(edges.dtype, copy=False)


def _interior_parabola_ok(edges: np.ndarray, p: dict, axis: str = "frame") -> bool:
    """Is this scan's interior actually a parabola? If not, nothing should be snapped onto it.

    Both edge constraints work by pulling the edges onto a quadratic fitted to the reliable interior. That is
    only meaningful when the interior IS quadratic — on cs011_od_v2 the interior fit leaves a 17.77 px residual,
    and snapping its edges to that made the surface no better (52.56 -> 38.42 px). Measured per scan, so a
    difficult volume opts itself out instead of being detected by a constant."""
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2:
        return False
    L, F = e.shape
    prof = np.nanmedian(e[int(0.2 * L):int(0.8 * L)], axis=0) if axis == "frame" else \
        np.nanmedian(e[:, int(0.2 * F):int(0.8 * F)], axis=1)
    n = prof.size
    x = np.arange(n, dtype=np.float64)
    g = np.isfinite(prof)
    g[:max(4, n // 12)] = False
    g[-max(4, n // 12):] = False
    if int(g.sum()) < 20:
        return False
    c = np.polyfit(x[g], prof[g], 2)
    r = prof[g] - np.polyval(c, x[g])
    return float(np.sqrt(np.mean(r ** 2))) <= float(p.get("edge_fit_max_resid", 2.5))


def _lat_edge_parabola(edges: np.ndarray, p: dict) -> np.ndarray:
    """Pull each frame's OUTER LATERALS onto the quadratic fitted to that frame's trusted middle.

    The frame-axis twin of _parabola_edge_constrain, and necessary because half the reviewer's edge complaints
    are about the lateral ends WITHIN a B-scan — "the left edge marked is a very different curvature from the
    general corneal curvature", "corners of frames dont always follow overall curvature" — which no correction
    along the frame axis can reach.

    The correction is TAPERED to zero where it meets the untouched interior, so the repaired edge joins it
    without a step, and bounded by lat_edge_max_move so a wild column cannot drag the edge somewhere absurd.
    edges=(n_lateral, n_frames)."""
    if not bool(p.get("lat_edge_parabola", False)):
        return edges
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 2 or not _interior_parabola_ok(e, p, axis="lateral"):
        return edges
    L, F = e.shape
    nb = int(float(p.get("lat_edge_frac", 0.12)) * L)
    if nb < 4 or L < 8 * nb:
        return edges
    lat = np.arange(L, dtype=np.float64)
    lo_f, hi_f = int(0.15 * L), int(0.85 * L)
    max_move = float(p.get("lat_edge_max_move", 40.0))
    out = e.copy()
    for f in range(F):
        y = out[:, f]
        g = np.isfinite(y)
        mid = g.copy()
        mid[:lo_f] = False
        mid[hi_f:] = False
        if int(mid.sum()) < 200:
            continue
        c0 = np.polyfit(lat[mid], y[mid], 2)
        r = y[mid] - np.polyval(c0, lat[mid])
        k = np.abs(r) <= np.percentile(np.abs(r), 90)
        q = np.polyval(np.polyfit(lat[mid][k], y[mid][k], 2), lat)
        for sl, w in ((slice(0, nb), np.linspace(1.0, 0.0, nb)),
                      (slice(L - nb, L), np.linspace(0.0, 1.0, nb))):
            seg = out[sl, f]
            mv = np.clip(q[sl] - seg, -max_move, max_move)
            out[sl, f] = seg + mv * w
    return out.astype(edges.dtype, copy=False)


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
                               detect: np.ndarray | None = None,
                               sc_s_mc: np.ndarray | None = None, sc_s_raw: np.ndarray | None = None,
                               sc_shift: np.ndarray | None = None) -> dict:
    """AUTO-SUGGEST the surface-CROPPED frames (B-scan columns whose corneal apex rises ABOVE the acquisition
    window, so the frame has no anterior surface). Returns
    {frames:[...], counts:{frame: n_slices}, lateral_by_frame:{...}, rule, n_slices, depth_vox, n_frames}:
    `frames` = the default selection the user verifies/edits; `counts` drives a per-frame confidence bar. The
    posterior-continuity reconstruction (build_surface_crop_edges) is what actually corrects the confirmed set.

    TWO rules, selected by crop_detect:
      • "geom" (DEFAULT) — the validated flank-extrapolated APEX rule (_sc_geom_frames, micro-F1 0.931 on the
        37 GT scans, 0 false frames on the CS010 limbus trap). It needs the caller to pass the surface
        EVIDENCE, because this function is handed only ONE geometry: sc_s_raw (anterior on the RAW geometry),
        sc_s_mc (anterior AFTER axial_motion_correct) and sc_shift (that call's per-frame shift M, oriented
        raw_row = corrected_row + M[f]). Without all three it silently falls back to "count".
      • "count" — the legacy per-slice _clip_mask tally + hysteresis. Still computed either way, because
        `counts` and `lateral_by_frame` feed the confidence bar, is_substantial_clip and the axial overlay.
    `rule` in the result says which one produced `frames`."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    edges = detect if detect is not None else detect_surface_all(sag, p, workers=workers)
    n, depth_vox, F = int(sag.shape[0]), int(sag.shape[1]), int(sag.shape[2])
    counts = np.zeros(F, dtype=int)
    # Keep the FULL (lateral, frame) clip mask, not just the per-frame tally. The tally answers "is this
    # B-scan clipped", which is what the sagittal view needs (there a frame IS a column). The AXIAL view
    # shows one B-scan as lateral×depth, so there the clipped region is a set of LATERAL columns within the
    # frame — the transpose of this mask. Without it the axial view has nothing to draw.
    mask = np.zeros((n, F), dtype=bool)
    for i in range(n):
        cm = np.asarray(_clip_mask(np.ascontiguousarray(sag[i]).astype(np.float32), edges[i], p), dtype=bool)
        mask[i, :cm.size] = cm[:F]
        counts[cm] += 1
    # HYSTERESIS frame selection (not a bare per-frame count >= min_slices threshold). A physically-continuous
    # clip has its per-frame clipped-slice count DIP below crop_min_slices at interior frames (a few px of
    # detector noise) and at the marginal frame-array EDGE frames — a hard threshold then punches interior
    # HOLES into the clipped band and DROPS the edge frames (validated across the surface-crop set: the misses
    # are ALL under-detection, never over). Fix: keep any contiguous run of even-weakly-clipped frames
    # (count >= crop_hys_min_slices) that CONTAINS a strong seed (count >= crop_min_slices). This bridges
    # interior holes and extends the band to the array edge in one pass, while an isolated weak run with no
    # strong seed (stray noise) is still rejected — so it cannot introduce a false clip. No-op when the
    # thresholded set is already a solid contiguous run.
    # RULE SELECTION. crop_detect="geom" (default) uses the validated flank-extrapolated APEX rule, which needs
    # the caller to supply the surface EVIDENCE (both geometries + the motion shift) because this function only
    # receives ONE geometry. When that evidence is absent — every existing caller that passes detect= only, the
    # legacy detector path, an axial_motion_correct no-op — we fall back to the count rule below, so those call
    # sites are byte-unchanged. crop_detect="count" forces the legacy rule outright: a complete, no-migration
    # revert. The geom branch is wrapped so a failure DEGRADES to the count rule rather than aborting a
    # preprocess run, and `rule` is reported so the UI/QA can tell which one produced the frames.
    rule = "count"
    frames: list[int] | None = None
    sm_top = None
    if str(p.get("crop_detect", "geom")).lower() == "geom" \
            and sc_s_mc is not None and sc_s_raw is not None and sc_shift is not None:
        try:
            frames = [int(f) for f in _sc_geom_frames(sc_s_mc, sc_s_raw, sc_shift) if 0 <= int(f) < F]
            # Per-lateral distance to the RAW top, taking whichever geometry places the surface higher. On a
            # TILTED scan the raw detector mislocates the apex tens of px deep (cs008_od_v3: rows 17-20 for a
            # frame whose apex is actually cut off), so a raw-only profile would report NO clipped laterals
            # there; the motion-corrected surface plus that frame's shift recovers the true raw row. Used only
            # for the axial overlay below, never for selection.
            _mm = np.asarray(sc_shift, dtype=np.float64).ravel()
            sm_top = np.minimum(_sc_lateral_median(np.asarray(sc_s_raw)),
                                _sc_lateral_median(np.asarray(sc_s_mc)) + _mm[None, :])
            rule = "geom"
        except Exception:  # noqa: BLE001 — never let the new rule break a preprocess run
            frames = None
            sm_top = None
            rule = "count-fallback"
    min_slices = max(1, int(p.get("crop_min_slices", 3)))
    weak_min = max(1, int(p.get("crop_hys_min_slices", 1)))
    strong = counts >= min_slices
    if frames is not None:
        pass                                    # geom rule already decided the frame set
    elif strong.any():
        weak = counts >= weak_min
        lbl, _n = ndimage.label(weak)
        strong_labels = set(int(x) for x in np.unique(lbl[strong]) if x > 0)
        sel = np.array([lbl[f] in strong_labels for f in range(F)], dtype=bool)
        # GAP-FILL: bridge interior holes where the count dropped to 0 (below even the weak floor) — a
        # physically-continuous apex still clips there but a few px of detector noise zeroed the frame's
        # tally. Unioned with the pre-close mask so the band edges are never eroded (same boundary-safe idiom
        # as _clip_mask). A gap wider than the reach (a genuine break between two separate clips, or a
        # diagonal fragmentation) is left alone.
        # REACH: ones(2*crop_frame_close_gap+1) closes holes up to 2*crop_frame_close_gap wide — 8 frames at
        # the default of 4, NOT 4. Read the param as "half the maximum hole bridged".
        g = int(p.get("crop_frame_close_gap", 4))
        if g > 0:
            sel = sel | ndimage.binary_closing(sel, structure=np.ones(2 * g + 1, dtype=bool))
        frames = [int(f) for f in range(F) if sel[f]]
    else:
        frames = []
    # Per-frame clipped LATERALS, emitted only for the detected frames so the payload stays bounded
    # (a clipped frame is typically clipped over a contiguous run of laterals around the apex).
    lateral_by_frame = {int(f): [int(i) for i in np.nonzero(mask[:, f])[0]] for f in frames}
    # UI FALLBACK for the geom rule: the geom frame set is NOT derived from `mask`, and 22% of geom-selected
    # frames have an all-False _clip_mask column (that disagreement is the whole point of the new rule). The
    # axial view draws lateral_by_frame, so without this it renders NOTHING on exactly the frames the rule
    # improves — which reads to the reviewing user as "the fix did nothing". Derive those laterals from the
    # already-smoothed raw apex profile instead. Computed AFTER `frames` is final so it cannot influence
    # selection, and `counts` is left untouched so is_substantial_clip / peak_slices keep their old evidence.
    if sm_top is not None:
        _floor = float(p.get("clip_edge_floor", 8.0))
        for f in frames:
            if not lateral_by_frame.get(f) and f < sm_top.shape[1]:
                prof = sm_top[:, f]
                sel_lat = prof <= _floor
                if not sel_lat.any():
                    # Both geometries put this frame's whole profile below the floor — the rule still selected
                    # it (e.g. from an extrapolated dome vertex above the window, where no lateral is literally
                    # pinned at the top). Mark the band around the frame's own apex so the overlay shows WHERE
                    # the reconstruction applies. The argmin always qualifies, so this is never empty.
                    sel_lat = prof <= float(np.nanmin(prof)) + _floor
                lateral_by_frame[f] = [int(i) for i in np.nonzero(sel_lat)[0]]
    return {"frames": frames, "counts": {int(f): int(counts[f]) for f in range(F) if counts[f] > 0},
            "lateral_by_frame": lateral_by_frame, "rule": rule,
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
    sl = np.ascontiguousarray(slice_img).astype(np.float32)
    # A caller that has already computed a BETTER posterior for this slice (guided_posterior_row — the
    # reviewer's corrections used as a search prior) passes it in rather than having it re-detected here with
    # no knowledge of those corrections. The manual per-frame override below still applies on top, so a point
    # the reviewer drew on THIS slice continues to win outright over any detection, guided or not.
    _pre = p.get("_post_row_detected")
    if _pre is not None and np.asarray(_pre).shape == (sl.shape[1],):
        b = np.asarray(_pre, dtype=np.float64).copy()
    else:
        b = _detect_bottom_edge(sl, p).astype(np.float64)
    # ── THICKNESS-PRIOR SECOND PASS ──────────────────────────────────────────────────────────────────────
    # The posterior edge sits a NEAR-CONSTANT distance below the anterior — corneal thickness varies slowly
    # across a slice — so a detection far from that offset is a mis-lock, not anatomy. The first pass above is
    # unconstrained and does mis-lock (onto the iris, a specular band, or the anterior itself), and because the
    # reconstruction is `posterior − thickness`, a mis-locked posterior moves the reconstructed apex by the
    # same amount. _detect_bottom_edge already accepts a PRIOR that restricts its search — it was simply never
    # given one here. So: measure the robust thickness from the frames where the anterior is trustworthy, then
    # re-detect with prior = anterior + that thickness.
    # Self-limiting by construction: with no trustworthy anterior frames there is no prior and the first pass
    # stands. crop_post_prior=False disables.
    if bool(p.get("crop_post_prior", True)) and np.isfinite(a).any():
        _floor0 = float(p.get("clip_edge_floor", 8.0))
        _cf0 = set(int(f) for f in (crop_frames if crop_frames is not None else []))
        _ok = np.array([np.isfinite(a[f]) and a[f] >= _floor0 and f not in _cf0 for f in range(F)])
        if int(_ok.sum()) >= 3:
            _t0 = (b - a)[_ok]
            _t0 = _t0[np.isfinite(_t0) & (_t0 > 0)]
            if _t0.size >= 3:
                _med_t = float(np.median(_t0))
                _prior = np.where(np.isfinite(a), a + _med_t, np.nan)
                _prior = _fill_nan_1d(_prior)
                _win = float(p.get("crop_post_prior_win", 40.0))
                b2 = _detect_bottom_edge(sl, {**p, "detect_window": _win}, prior=_prior).astype(np.float64)
                # keep the constrained result only where it is finite and inside the canvas
                _good = np.isfinite(b2) & (b2 > 0) & (b2 < sl.shape[0] - 1)
                b = np.where(_good, b2, b)
    # ── MANUAL OVERRIDE ──────────────────────────────────────────────────────────────────────────────────
    # The reviewer's dragged posterior points WIN over any detection, on the same principle as border anchors:
    # where a human has said where the edge is, that is ground truth and the detector's opinion is irrelevant.
    # Applied per frame, so a couple of corrections fix a slice without redrawing all of it.
    _pa = p.get("_post_anchor_row")
    if isinstance(_pa, dict) and _pa:
        for _f, _d in _pa.items():
            try:
                _fi = int(_f); _dv = float(_d)
            except (TypeError, ValueError):
                continue
            if 0 <= _fi < F and np.isfinite(_dv):
                # ABSENT SENTINEL: a value at or below the canvas floor means the reviewer said there is NO
                # posterior edge on this frame — the bottom edge genuinely leaves the image there. Clamping it
                # back to depth-1 (as this did) manufactured a flat edge pinned along the floor, and since the
                # reconstruction is posterior - thickness, that phantom edge dragged the rebuilt apex with it.
                # NaN instead, so every downstream mask treats the frame as having no measurement.
                b[_fi] = float("nan") if _dv >= sl.shape[0] - 1 else float(np.clip(_dv, 0.0, sl.shape[0] - 1))
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
    valid = valid & np.isfinite(b)                                # frames with no posterior contribute nothing
    if int(valid.sum()) < 3:
        return out, b, adopted
    tv = t[valid]
    med = float(np.median(tv)); mad = float(np.median(np.abs(tv - med))) + 1e-6
    keep = valid.copy(); keep[valid] = np.abs(tv - med) <= 4.0 * 1.4826 * mad   # drop posterior mis-locks
    if int(keep.sum()) < 3:
        keep = valid
    fk = np.where(keep)[0].astype(np.float64)
    t_interp = np.interp(cf.astype(np.float64), fk, t[keep])      # thickness from non-marked flanks (held at ends)
    recon = b[cf] - t_interp
    # a cropped frame whose posterior is absent has nothing to be rebuilt FROM — leave its detected anterior
    # alone rather than propagating a NaN into the surface
    _has_post = np.isfinite(b[cf])
    clipped_here = _has_post & (recon < (a[cf] - margin))         # reconstruct only where it sits ABOVE the detected anterior
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
    cf = np.array(sorted({int(f) for f in (crop_frames if crop_frames is not None else []) if 0 <= int(f) < F}), dtype=int)
    # per-slice manual posterior anchors ({slice: {frame: depth}}), normalised once
    _panch = p.get("crop_post_anchors") or {}
    _panch = _panch if isinstance(_panch, dict) else {}
    for i in range(int(sag.shape[0])):
        _row = _panch.get(str(i)) or _panch.get(i) or None
        out, b, _adopted = _crop_reconstruct_slice(sag[i], edges[i], cf, {**p, "_post_anchor_row": _row})
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
    cf = set(int(f) for f in (crop_frames if crop_frames is not None else []))
    md = float(p.get("max_displacement", 40.0)) or 40.0
    res = float(p.get("residual_threshold", 5.0))
    det = detect if detect is not None else detect_surface_all(sag, p, workers=workers)
    # posterior parabola per slice (RANSAC = robust to a clip mis-lock), smoothed across slices for 3-D consistency
    Pb = np.stack([_fit_quadratic_ransac(posterior[i].astype(np.float64), res) for i in range(n)])
    Pb = ndimage.gaussian_filter1d(Pb, sigma=float(p.get("crop_slice_smooth", 2.0)), axis=0)
    # Corneal thickness → top-edge parabola Pa = Pb − thickness (apex/edge may sit <0 = ABOVE the old top).
    # PER-FRAME thickness PROPAGATED from the ADJACENT un-cut frames, not a global median: measure thickness
    # only where the anterior is valid (un-cut, in-window), then interpolate ACROSS FRAMES into the cut band,
    # HELD at the ends. A cut frame therefore inherits the thickness of its NEAREST un-cut neighbours — the
    # corneal shape varies smoothly frame-to-frame — instead of a global median that a thin peripheral/limbus
    # frame drags down (under-reconstructing the apex, so the extended canvas fell short of the true above-
    # window apex). Smoothed across slices for 3-D consistency; a slice with <3 valid frames falls back to the
    # cross-slice-filled per-slice median.
    floor = float(p.get("clip_edge_floor", 8.0))
    allf = np.arange(F, dtype=np.float64)
    Th = np.full((n, F), np.nan); Ts = np.full(n, np.nan)
    for i in range(n):
        thick = posterior[i] - det[i]
        valid = np.array([(f not in cf) and np.isfinite(det[i, f]) and det[i, f] >= floor for f in range(F)])
        if int(valid.sum()) >= 3:
            Ts[i] = float(np.median(thick[valid]))
            Th[i] = np.interp(allf, allf[valid], thick[valid])   # held at ends → nearest-un-cut extrapolation into the cut band
    Ts = _fill_nan_1d(Ts)
    bad = ~np.isfinite(Th).all(axis=1)
    Th[bad] = Ts[bad, None]
    Th = ndimage.gaussian_filter1d(Th, sigma=float(p.get("crop_slice_smooth", 2.0)), axis=0)
    Pa = Pb - Th
    # robust per-column shift: parabola − median-smoothed posterior, clipped so an outlier can't inflate the pad
    post_rob = np.stack([ndimage.median_filter(posterior[i].astype(np.float64), size=int(p.get("crop_target_med", 11)))
                         for i in range(n)])
    disp_col = np.nan_to_num(np.clip(Pb - post_rob, -md, md), nan=0.0, posinf=md, neginf=-md)  # per-column (lateral,frame)
    if bool(p.get("crop_rigid_disp", True)):
        # RIGID B-SCAN CONSTRAINT (v0.0.202): disp_col[i,f] varies across lateral i WITHIN a single axial frame f, so
        # applying it column-by-column would DEFORM the instantaneous B-scan. Surface-crop scans are NO EXCEPTION to the
        # rigid rule (a B-scan is captured near-instantaneously → its internal geometry is TRUTH): collapse to ONE depth
        # shift PER FRAME (robust median over the laterals) so every lateral of a frame moves TOGETHER — a pure rigid
        # depth reposition + canvas extend, ZERO per-column warp. The posterior lands on its smooth parabola in
        # aggregate and the reconstructed apex is repositioned coherently; the frame's real lateral shape is preserved.
        # Smooth the per-frame shift across FRAMES only (the old 2-D lateral+frame Gaussian's lateral axis is now moot);
        # the median already kills the faint-posterior per-slice noise the lateral smoothing existed to remove.
        disp_f = np.array([float(np.median(disp_col[:, f])) for f in range(F)])
        _sf = float(p.get("crop_disp_smooth_frame", 1.5))
        if _sf > 0:
            disp_f = ndimage.gaussian_filter1d(disp_f, sigma=_sf, mode="nearest")
        disp = np.repeat(disp_f[None, :], n, axis=0)                            # constant across lateral → RIGID per frame
    else:  # LEGACY per-column warp (deforms the B-scan) — kept only for A/B comparison; NOT the default.
        _ss = float(p.get("crop_disp_smooth_slice", 6.0)); _sf = float(p.get("crop_disp_smooth_frame", 1.5))
        disp = ndimage.gaussian_filter(disp_col, sigma=(_ss, _sf)) if (_ss > 0 or _sf > 0) else disp_col
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
    # The vacated rows above the tissue are HARD ZEROS, and an OCT background floor is not zero — measured on
    # case_cs005_od_v2 the padding sits at 33 against 511 just below it. That step is a far stronger edge than
    # the epithelium, so the DP detector locks onto it: detection lands at row 27 instead of ~199 (verified with
    # the scan's own tuned dp_*, and with auto_tune re-run on the padded volume). Every later stage that
    # re-detects is then reading an artifact, which is why the surface-crop branch has to return early and skip
    # the three rigid smoothing passes — see preprocess_oct_to_nifti.
    # Filling those rows by TILING this scan's own background (real noise from just below the pad; no tissue
    # invented, no distribution synthesised) removes the cliff: detection returns to row 199 with 0% of columns
    # left in the padding, and rigid_frame_derotate — a strict no-op on the zero-padded volume — then adjusts
    # 100 frames for roughness 1.073 -> 0.859.
    # DEFAULT OFF: this changes DELIVERED voxels on the 39 canvas-extended scans, so it stays opt-in until the
    # A/B is reviewed. crop_pad_fill="background" enables it.
    if str(p.get("crop_pad_fill", "zeros")).lower() == "background" and pad > 0:
        out = _fill_pad_background(out, int(pad), int(p.get("crop_pad_fill_src", 24)))
    return out, int(pad), Pb, Pa, bool(clamped)


def _fill_pad_background(out: np.ndarray, pad: int, src: int = 24, first_valid: np.ndarray | None = None
                         ) -> np.ndarray:
    """Replace each column's empty leading rows with a mirrored tiling of ITS OWN background just below them.

    out is (n_slices, depth, n_frames).

    PER-COLUMN, not per-volume, and that distinction is the whole correctness of this function. Every frame is
    copied in at its own offset (`off = pad + round(disp[f])` in warp_surface_crop_extend), so a downward-
    displaced frame has MORE empty rows than `pad`. A version of this that sampled a single global block at
    rows [pad : pad+src] read those still-empty rows as if they were background and copied zeros over zeros —
    silently doing nothing for exactly the frames that needed it (measured: 48 of 101 frames left with a zero
    band up to 20 rows, and detection then follows the residual cliff on some of them).

    `first_valid` is the per-(slice, frame) index of the first real row; when omitted it is measured from the
    data. The source block is taken from just BELOW that boundary, so it is always real signal.
    """
    if out.ndim != 3:
        return out
    n, H, F = out.shape
    src = max(2, int(src))
    if first_valid is None:
        # a column's leading run of near-zero rows; threshold off the volume's own level so it is scale-free
        lvl = float(np.median(out[out > 0])) if np.any(out > 0) else 0.0
        thr = 0.05 * lvl
        nz = out > thr
        any_nz = nz.any(axis=1)
        first_valid = np.where(any_nz, nz.argmax(axis=1), 0).astype(np.int64)   # (n, F)
    fv = np.clip(np.asarray(first_valid, dtype=np.int64), 0, H - 1)
    hi = int(fv.max())
    if hi <= 0:
        return out
    hi = min(hi, H - 2)
    rows = np.arange(hi)[None, :]                                    # (1, hi)
    for f in range(F):
        lo = fv[:, f][:, None]                                       # (n, 1) first real row of this column
        if int(lo.max()) <= 0:
            continue
        avail = np.maximum(np.minimum(src, H - lo), 1)                # rows of real signal below the boundary
        # mirrored tiling: row r takes lo + ((lo-1-r) mod avail), so the join at the boundary is continuous
        idx = lo + ((lo - 1 - rows) % avail)
        idx = np.clip(idx, 0, H - 1)
        block = np.take_along_axis(out[:, :, f], idx, axis=1)         # (n, hi)
        mask = rows < lo
        cur = out[:, :hi, f]
        out[:, :hi, f] = np.where(mask, block, cur)
    return out


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


def surface_confidence_map(sag: np.ndarray, surface: np.ndarray, p: dict | None = None) -> np.ndarray:
    """Per-(slice, frame) confidence in [0,1] that the SERVED anterior edge is a real corneal boundary — GT-free,
    measured on the edge actually displayed (not a fresh detection). It is the per-frame form of
    _surface_confidence's CONTRAST: mean(6px just BELOW the edge) - mean(6px just ABOVE), on a lightly-smoothed
    slice, auto-scaled to THIS scan's own typical boundary contrast (median of the positive contrasts) and
    clipped to [0,1]. ~1 on a bright interior boundary (bright stroma below, dark air above); -> 0 where the edge
    FLOATS above the epithelium in speckle (the faint frame-edge / limbus corner) or where an interpolated edge
    has drifted off tissue. This is exactly the "areas that are low confidence" signal the fix-columns editor
    uses to ask the reviewer for more edge corrections. Read-only; one vectorised smoothing pass; no detector run.

    Prototyped on cs046 (the faint-left-corner scan): interior median 1.00 with a 1.1% false-flag rate at a 0.35
    threshold; the known FLOATING left corner fires on ~40 slices while the clean right corner fires on 0; and
    96% of frames where the auto edge is >8px off the reviewer's manual GT read <0.35. The air-above (from
    _above_brightness) and roughness (2nd-difference) factors were tried and DROPPED as empirically harmful: air
    crushed the good interior to ~0.3 (a floating edge has dark air above it too, so it does not discriminate),
    and roughness penalised the legitimate STEEP limbus descent -> it would nag on a correctly-traced corner."""
    _ = p  # reserved for future tuning; the contrast metric is self-normalising and needs no params today
    sag = np.asarray(sag, np.float32)
    surf = np.asarray(surface, np.float32)
    if sag.ndim != 3 or surf.ndim != 2:
        return np.ones(surf.shape if surf.ndim == 2 else (0, 0), np.float32)
    n, D, F = sag.shape
    if surf.shape != (n, F) or D < 13 or F < 1:
        return np.ones((n, F), np.float32)
    sm = ndimage.gaussian_filter(sag, (0.0, 1.0, 0.6))          # per-slice depth/frame despeckle (matches auto_tune)
    ei = np.clip(np.round(surf).astype(int), 6, D - 7)          # (n, F) edge row, kept off the borders
    ni = np.arange(n)[:, None]; fi = np.arange(F)[None, :]
    below = np.mean([sm[ni, ei + j, fi] for j in range(1, 7)], axis=0)
    above = np.mean([sm[ni, ei - j, fi] for j in range(1, 7)], axis=0)
    raw_c = below - above
    ref = float(np.median(raw_c[raw_c > 0])) if np.any(raw_c > 0) else 1.0
    return np.clip(raw_c / max(ref, 1e-6), 0.0, 1.0).astype(np.float32)


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


def pin_anchors(surface: np.ndarray, anchors: dict, depth: int,
                crop_max_pad: float = 120.0) -> np.ndarray:
    """Force `surface` to the reviewer's DRAWN depth at every anchored (slice, frame), IN PLACE.

    Ground truth: the reviewer's fix-columns line is what the surface IS at the frames they drew. A detector
    (any window: local redetect, generalize, guided) may refine the surface AROUND those frames, but must
    never override one — otherwise the strongest-nearby gradient wins and the correction is silently ignored
    (guided's 40 px search did exactly this: it landed 7-17 px off a manifestly-correct line). Pinning makes a
    correction STICK where it was drawn, so iterating corrections converges instead of re-detecting each time.

    Applies the SAME anchor normalization as redetect_surface, so pinned == what the reviewer meant:
      * ABSENT sentinel (depth >= depth-1) → skipped (the reviewer said no surface there; do not assert one).
      * ABOVE-canvas (negative) anchors → clamped to [-crop_max_pad, depth-1] and honored (a surface-cropped
        apex lives above the window and can only be given at negative depth).
    Non-drawn frames are left untouched. Returns the same array."""
    if surface is None or not anchors:
        return surface
    S = np.asarray(surface)
    if S.ndim != 2:
        return surface
    L, F = int(S.shape[0]), int(S.shape[1])
    for s_key, frames in anchors.items():
        try:
            s = int(s_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= s < L) or not isinstance(frames, dict):
            continue
        for f_key, d in frames.items():
            try:
                f = int(f_key); dv = float(d)
            except (TypeError, ValueError):
                continue
            if 0 <= f < F and np.isfinite(dv) and dv < depth - 1:
                S[s, f] = float(np.clip(dv, -float(crop_max_pad), depth - 1))
    return S


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
    # PER-SLICE seed windows. The two kinds of correction mean different things: a shaped quadratic IS the
    # surface (window 0 — follow it exactly), while hand-drawn edge anchors only say roughly where the surface
    # is, so they are seeds the detector snaps to within a couple of px. One global window forced every
    # anchored slice to be read the same way, so a scan carrying both had to pick one — and the app resolved
    # that by DISCARDING the other slices' anchors on commit. Keyed by slice; anything absent uses the global.
    seed_win_by_slice: dict[int, float] = {}
    for _k, _v in (p.get("redetect_seed_window_slices") or {}).items():
        try:
            seed_win_by_slice[int(_k)] = float(_v)
        except (TypeError, ValueError):
            continue
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
            # ABSENT SENTINEL: an anchor at or below the canvas floor means the reviewer said there is NO
            # top surface on that frame (it happens — some frames have none at all). Skipped entirely rather
            # than clamped to depth-1, which would assert a real surface lying along the bottom and drag the
            # flatten's per-frame alignment onto it.
            if 0 <= f < W and np.isfinite(dv) and dv < depth - 1:
                # ABOVE-CANVAS anchors are legal and necessary. A surface-cropped cornea has its apex above
                # the captured window by definition, so the curve through it can only be specified at
                # NEGATIVE depth. Clamping those to 0 (as this did) silently flattened the drawn apex onto
                # the top row: the correction was accepted and then did the opposite of what was drawn.
                # The floor is crop_max_pad, the same cap that bounds how far warp_surface_crop_extend will
                # extend the canvas, so nothing can be asked for that the reconstruction cannot deliver — and
                # a negative surface then DRIVES that extension (raw_pad is computed from exactly this).
                fm[f] = float(np.clip(dv, -float(p.get("crop_max_pad", 120)), depth - 1))
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
        # light=True + tight seed window: snap to the nearest gradient within ±1-2 px of the user's drawn line
        # (no robust side-correction/RANSAC that would pull the corrected edge away from where they drew it).
        sw = float(seed_win_by_slice.get(s, seed_win))        # 0 on a slice whose curve was shaped exactly
        redet = _redetect_one_slice(np.ascontiguousarray(sag[s]).astype(np.float32), prior, sw, p,
                                    light=True).astype(np.float32)
        # HARD-clamp the re-detected region to within ±sw of the prior so a low-signal frame can't drift.
        redet[rmask] = np.clip(redet[rmask], prior[rmask] - sw, prior[rmask] + sw)
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
            # ABSENT SENTINEL: an anchor at or below the canvas floor means the reviewer said there is NO
            # top surface on that frame (it happens — some frames have none at all). Skipped entirely rather
            # than clamped to depth-1, which would assert a real surface lying along the bottom and drag the
            # flatten's per-frame alignment onto it.
            if 0 <= f < W and np.isfinite(dv) and dv < depth - 1:
                # ABOVE-CANVAS anchors are legal and necessary. A surface-cropped cornea has its apex above
                # the captured window by definition, so the curve through it can only be specified at
                # NEGATIVE depth. Clamping those to 0 (as this did) silently flattened the drawn apex onto
                # the top row: the correction was accepted and then did the opposite of what was drawn.
                # The floor is crop_max_pad, the same cap that bounds how far warp_surface_crop_extend will
                # extend the canvas, so nothing can be asked for that the reconstruction cannot deliver — and
                # a negative surface then DRIVES that extension (raw_pad is computed from exactly this).
                fm[f] = float(np.clip(dv, -float(p.get("crop_max_pad", 120)), depth - 1))
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


def smooth_surface_edge_band(surface: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Light smoothing of the faint LIMBUS / frame-edge bands of a served surface (lateral, frame) — the WARP
    TARGET (provided_edges) on the corrections path.

    The outermost frames are the steep limbus descent at the low-SNR FOV edge, where the detector
    (corner_edge_retrace's relaxed DP) leaves ~1.7x the centre's roughness. The rigid flatten reproduces that
    faithfully → a bumpy corrected limbus while the dome is smooth (the exact "not smooth at the right end,
    tissue too" the reviewer reported on cs048). This smooths ONLY the outer `edge_band_smooth_frames` at BOTH
    frame ends: an across-SLICE (en-face) gaussian kills the slice-to-slice jitter, then a light across-FRAME
    gaussian removes the within-B-scan wiggles. A linear taper ramps the smoothing weight to 0 at the interior
    edge of the band so there is no step where it meets the un-smoothed centre. The frame gaussian is light
    (σ≈2) so a steep monotone descent is preserved (smoothing a line keeps the line) — only the ±1-2px bumps go.

    GENTLE-AVERAGE by design: NO anchor re-pin (the reviewer chose to let the smoothing lightly average the
    drawn edge too, since fix-columns edge corrections are approximate). Gated by edge_band_smooth_frames
    (0 = OFF → returns the input unchanged). Auto path never calls this; corrections path only."""
    p = params or {}
    nf = int(p.get("edge_band_smooth_frames", 0) or 0)
    if nf <= 0:
        return surface
    s = np.asarray(surface, dtype=np.float64)
    if s.ndim != 2:
        return surface
    L, F = s.shape
    if F < 2 * nf + 4 or L < 3:
        return surface
    s = s.copy()
    sig_s = float(p.get("edge_band_smooth_sigma_slice", 2.5))   # across-slice (en-face) — the DOMINANT fix
    sig_f = float(p.get("edge_band_smooth_sigma_frame", 1.0))   # across-frame — LIGHT (heavier flattens the apex)
    taper = max(1, int(p.get("edge_band_smooth_taper", 8)))
    # WHICH frame end(s) to smooth. DEFAULT "low" = frame 0, which the scaleX(-1) display puts on the RIGHT (the
    # reviewer's "right end"). Only the low end by default: on a decentred dome (cs048 apex at frame ~9) the HIGH
    # end holds the deep steep descent + outliers, and smoothing there flattens real curvature (measured −20px
    # descent, 200px moves). "both"/"high" are opt-in for a centred dome whose other limbus is also rough.
    ends = str(p.get("edge_band_smooth_ends", "low"))
    bands = {"low": [(0, nf)], "high": [(F - nf, F)], "both": [(0, nf), (F - nf, F)]}.get(ends, [(0, nf)])
    for lo, hi in bands:
        idx = np.arange(lo, hi)
        n = len(idx)
        band = s[:, idx]
        # across-SLICE first (kills slice-to-slice edge jitter, preserves each B-scan's curvature), then a LIGHT
        # across-FRAME pass (removes within-B-scan wiggles; kept small so the corneal apex is not flattened).
        sm = ndimage.gaussian_filter1d(band, sig_s, axis=0, mode="nearest")
        if sig_f > 0:
            sm = ndimage.gaussian_filter1d(sm, sig_f, axis=1, mode="nearest")
        # weight = 1 at the FRAME edge, ramps to 0 over `taper` frames at the INTERIOR boundary of the band
        if lo == 0:                                  # low band: interior boundary is its high (last) index
            dist = (n - 1 - np.arange(n)) / taper
        else:                                        # high band: interior boundary is its low (first) index
            dist = np.arange(n) / taper
        w = np.clip(dist, 0.0, 1.0)[None, :]
        s[:, idx] = band * (1.0 - w) + sm * w
    return s.astype(surface.dtype if surface.dtype.kind == "f" else np.float32)


def interpolate_anchors_surface(anchors, baseline: np.ndarray, params: dict | None = None) -> np.ndarray:
    """PURE per-frame interpolation of the reviewer's drawn edge across slices — "draw a pure interpolation
    between the manually corrected slices" (reviewer directive, cs048).

    For each FRAME drawn on >= interp_min_slices slices, the surface at that frame becomes a linear
    interpolation of ONLY the drawn depths across slices — EXACT at every drawn slice, a straight line
    between. This is the OPPOSITE of re-detecting between anchors: redetect_surface interpolates each slice's
    whole border (drawn frames MIXED with auto for the undrawn ones) then re-detects within a window, so the
    auto detector's jitter at the faint limbus leaks in even though the drawn limbus values are smooth
    (measured cs048: served jitter 5-12px vs drawn-value trend ~0.1px). Interpolating the drawn values
    directly removes that leak. Frames drawn on fewer slices keep the auto BASELINE (the reviewer left them to
    auto — usually the well-lit mid-dome), blended across the frame axis by a gaussian on the per-frame
    on/off weight (interp_frame_taper) so there is no seam. Anchors may be str- or int-keyed."""
    p = params or {}
    min_slices = int(p.get("interp_min_slices", 12))
    taper = float(p.get("interp_frame_taper", 3.0))
    base = np.asarray(baseline, dtype=np.float64)
    if base.ndim != 2:
        return baseline
    L, F = base.shape
    A: dict[int, dict[int, float]] = {}
    for s, fv in (anchors or {}).items():
        try:
            si = int(s)
        except (TypeError, ValueError):
            continue
        row: dict[int, float] = {}
        for f, d in (fv or {}).items():
            try:
                row[int(f)] = float(d)
            except (TypeError, ValueError):
                continue
        if row:
            A[si] = row
    if not A:
        return baseline
    pure = base.copy()
    wf = np.zeros(F)
    xs = np.arange(L)
    for f in range(F):
        pts = sorted((s, A[s][f]) for s in A if f in A[s])
        if len(pts) >= 2:
            sls = np.array([q[0] for q in pts], dtype=np.float64)
            dep = np.array([q[1] for q in pts], dtype=np.float64)
            pure[:, f] = np.interp(xs, sls, dep)                 # exact at drawn slices, linear between
            if len(pts) >= min_slices:
                wf[f] = 1.0
    if wf.max() <= 0.0:
        return baseline                                          # no frame drawn densely enough → leave auto
    wf = ndimage.gaussian_filter1d(wf, taper)                    # taper the on/off across frames (no seam)
    out = pure * wf[None, :] + base * (1.0 - wf[None, :])
    return out.astype(baseline.dtype if baseline.dtype.kind == "f" else np.float32)


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
                        clip_cols=None, clip_fit=None, zero_cols=None, flatten_target=None):
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
    if flatten_target is not None:
        # FIX-COLUMNS DOME-PRESERVE: the flatten target is a lightly-SMOOTHED copy of the reviewer's own drawn
        # surface (not a deg-2 RANSAC parabola). The parabola cannot represent a real cornea's shape at the
        # frame edge — it FLATTENS the limbus plunge the reviewer drew, collapsing the corrected dome ~3x (the
        # "left edge on the corrected result stays uncorrected"). Flattening to a smooth of the drawn line
        # instead removes only per-frame jitter and keeps the dome + limbus. See smooth_volume use_provided.
        quad = np.asarray(flatten_target, dtype=np.float64)
    elif (clip_cols.size or zero_cols.size) and clip_fit is not None:
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
    (sl, active_edge, residual, corr_factor, bad_cols, good_cols, max_disp,
     clip_cols, clip_fit, zero_cols, flatten_target) = packed
    return _slice_displacement(active_edge, residual, corr_factor, bad_cols, good_cols, max_disp,
                               clip_cols=clip_cols, clip_fit=clip_fit, zero_cols=zero_cols,
                               flatten_target=flatten_target)


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
                  return_coverage: bool = False,
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
            # Gate the frame-axis snap on the interior actually being a parabola — see
            # _interior_parabola_ok. Then apply the lateral-direction twin, which reaches the edge the
            # frame-axis version cannot (the reviewer's "left edge"/"corners of frames").
            if _interior_parabola_ok(edges, p, axis="frame"):
                edges = _parabola_edge_constrain(edges, p)
            edges = _lat_edge_parabola(edges, p)
        # FAINT→ONSET SNAP (laterally coherent) on the WARP edges — same correction as detect_surface_all, so the
        # flattened OUTPUT surface lands on the epithelium AND stays laterally smooth (no axial fuzziness from the
        # snap). Auto path only; the fix-columns provided_edges path carries the user's own surface and is skipped.
        if float(p.get("faint_snap_frac", 0.0) or 0.0) > 0 and bool(p.get("faint_snap_coherent", True)):
            edges = _faint_snap_coherent(edges, det, p)
        # EDGE DOME CONSTRAINT on the WARP edges (BEFORE edge-regularize) — pull per-frame DOWNWARD hooks at the
        # FOV-boundary laterals onto the general dome curve (CS001 OD__4/__5). DOWNWARD-only; auto path only (the
        # provided fix-columns edges are the user's GROUND-TRUTH surface → NEVER touched). See _edge_dome_constrain.
        if not use_provided:
            edges = _edge_dome_constrain(edges, p)
        # FAINT-EDGE DOME FOLLOW on the WARP edges (see detect_surface_all) — snap the too-shallow faint edge DOWN
        # onto the brighter epithelium band so the flattened OUTPUT border follows the corneal curve. Auto path only.
        if not use_provided:
            edges = _edge_dome_follow(edges, det, p)
        # EDGE REGULARIZATION on the WARP edges — smooth the faint FOV-boundary laterals across frames so the
        # flattened OUTPUT sagittal border is smooth there too (depth preserved), incl. the outer-band floor that
        # de-jitters the BRIGHT tissue-bearing FOV edge. Runs LAST so it has the final say. Auto path only.
        edges = _edge_regularize_surface(edges, det, p)

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
    # #9 v3 ARTIFACT CROP (per-lateral): the reviewer marked a per-slice band [lo,hi] on several laterals,
    # interpolated across laterals (_artifact_bands). Those frames are EXCLUDED from THIS lateral's fit and left
    # UNWARPED — the SAME mechanism as surface_cut (zero_cols), but the excluded set varies per lateral. Union
    # into zero_cols_list and refit clip_fit (RANSAC extrapolating across the artifact + any existing clip/cut) so
    # the cornea is fit to the REMAINING frames and the artifact can't bend the flatten. Downstream already handles
    # per-lateral zero_cols: _disp_worker gives them disp=0, and rigid_frame_warp drops them from the per-frame
    # median (~clamped) then re-asserts disp=0. Not for provided_edges (the marched surface is authoritative).
    # crop_bands MUST be honored on the provided_edges (fix-columns re-run) path TOO — NOT gated `not use_provided`.
    # This block only sets zero_cols + the flatten FIT target; it never edits the authoritative `edges` surface, so
    # honoring it here doesn't touch the user's marched surface. WITHOUT it, _slice_displacement fits
    # _fit_quadratic_ransac over ALL frames; the deep flat artifact-hold (frames ~47-100 held at ~140px in the
    # provided surface) captures the RANSAC consensus and the rising apex (frames 0-14) is discarded as outliers →
    # the DOMED provided surface is flattened into a monotonic descent (cs007 frame-dome collapsed +81→+11px). The
    # exclusion below refits over the cornea frames only (as the auto path does) → the dome is preserved. No-op when
    # there are no crop_bands (_artifact_bands returns None) → uncropped scans are byte-identical. (The surface_cut/
    # clip siblings above stay gated `not use_provided` by design; only crop_bands is un-gated.)
    _art_bands = _artifact_bands(p, W, n)
    if _art_bands is not None:
        # A crop keeps only PART of the cornea (apex near a frame edge), so the flatten target must be a
        # higher-degree fit — a symmetric deg-2 parabola collapses the short steep flank (cs007 left flank
        # +76→+38px). deg-4 preserves it (+74px, matching the target) with negligible oscillation (~1px).
        _cfd = max(1, int(p.get("crop_fit_degree", 4)))
        for i in range(n):
            _af = _art_bands[i]
            if not _af.size:
                continue
            _zc = np.array(sorted(set(int(c) for c in zero_cols_list[i]) | set(int(c) for c in _af)), dtype=int)
            zero_cols_list[i] = _zc
            _excl = np.array(sorted(set(int(c) for c in _zc) | set(int(c) for c in clip_cols_list[i])), dtype=int)
            _fit = _extrapolate_fit(edges[i], _excl, res, degree=_cfd) if _excl.size else None
            if _fit is not None:
                clip_fit_list[i] = _fit
            else:
                # extrapolate failed (<3 frames survive the artifact) → the lateral is almost entirely artifact.
                # DON'T leave clip_fit None, or _slice_displacement RANSAC-fits over ALL frames incl. the artifact
                # and flattens the few survivors to a contaminated quadratic. Use a FLAT fit at the surviving
                # edge's median so the artifact can't bend it; the frames are zeroed anyway. All-artifact/NaN →
                # leave as-is (survivors, if any, keep the prior fit; the whole lateral is a write-off).
                _keep = np.ones(W, dtype=bool)
                _keep[_excl[(_excl >= 0) & (_excl < W)]] = False
                _kv = edges[i][_keep]; _kv = _kv[np.isfinite(_kv)]
                if _kv.size:
                    clip_fit_list[i] = np.full(W, float(np.median(_kv)), dtype=np.float64)
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
    # ── FIX-COLUMNS DOME-PRESERVE (root cause of "left edge on the corrected RESULT stays uncorrected") ──
    # The flatten target on BOTH paths is a per-lateral deg-2 RANSAC quadratic across frames. On the AUTO path
    # the detected surface IS ~parabolic, so quad ≈ surface and the dome is preserved. But the reviewer's DRAWN
    # surface is NOT a parabola at the frame edge — the cornea plunges at the limbus — so quad(drawn) sits far
    # ABOVE the drawn plunge (measured disp ~-76px) and the warp lifts the corner up to the parabola, collapsing
    # the whole corrected sagittal dome ~3x (cs046 slice452: drawn descent 94px -> 18px result). The measured
    # AUTO dome is ~80px (ratio ~1.0); the corrections path must MIRROR that. Fix: on the provided path the
    # flatten target is a lightly-SMOOTHED copy of the DRAWN surface (gaussian σ=provided_flatten_smooth across
    # frames) instead of its parabola — it removes only per-frame jitter and KEEPS the dome + limbus (cs046
    # restored to ~77px, ≈ AUTO). The auto path is byte-unchanged (flatten_target stays None there).
    # provided_flatten_smooth=0 restores the old (parabola-flattening) behaviour.  See [[cornea-corrections-dome-flatten]].
    max_disp = (float(p.get("provided_max_displacement", 40.0) or 0.0) if use_provided
                else float(p.get("max_displacement", 0.0) or 0.0))
    bad_cols = [] if use_provided else [int(c) for c in (p.get("force_columns") or [])]
    good_cols = [] if use_provided else [int(c) for c in (p.get("good_columns") or [])]
    _pfs = float(p.get("provided_flatten_smooth", 2.0) or 0.0)
    if use_provided and _pfs > 0:
        # smooth the DRAWN edge across frames per lateral → the dome-preserving flatten target (keeps the
        # limbus plunge, removes jitter). Not applied where clip/cut columns need the extrapolating clip_fit.
        _ft = ndimage.gaussian_filter1d(np.asarray(edges, np.float64), _pfs, axis=1, mode="nearest")
        flatten_targets = [(None if (clip_cols_list[i].size or zero_cols_list[i].size) else _ft[i])
                           for i in range(n)]
    else:
        flatten_targets = [None] * n
    items = [(sag[i], active[i], res, corr_factor, bad_cols, good_cols, max_disp,
              clip_cols_list[i], clip_fit_list[i], zero_cols_list[i], flatten_targets[i]) for i in range(n)]
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
    # Tissue-column-restricted twin of disp_mean + the valid fraction, for QA ONLY. Computed HERE, at the
    # same pre-smoothing point, so that at full coverage it is EXACTLY disp_mean (any later apex de-tear /
    # inter-slice smoothing would otherwise make the two incomparable — verified: 1.6202 vs 1.6202).
    # HONEST LIMIT: restricting the mean does NOT remove the destroyed-volume bias. Measured on a synthetic
    # dome, zeroing 12 of 24 frames drops the metric 1.62 -> 0.16 whether or not the dead columns are
    # masked out, because the deviation is a residual from a RANSAC quadratic fit ALONG the frame axis:
    # with half the frames gone the fit has fewer points to satisfy and hugs the survivors more tightly,
    # so the residual shrinks on the columns that remain. Masking only removes the trivially-zero terms.
    # `coverage` is therefore the real guard: it does not repair `dev`, it tells you when `dev` cannot be
    # trusted or compared. Never rank two scans on `dev` at different coverage.
    _dev_valid = disp_mean
    _coverage = 1.0
    if return_coverage and disp_field.size:
        _valid = np.zeros(disp_field.shape, dtype=bool)
        for _i in range(min(n, disp_field.shape[0])):
            _v = np.any(sag[_i] != 0, axis=0)[:disp_field.shape[1]].copy()   # column carries tissue?
            for _bad in (zero_cols_list[_i], clip_cols_list[_i]):            # deliberately-suppressed
                if getattr(_bad, "size", 0):
                    _b = np.asarray(_bad, dtype=int)
                    _v[_b[(_b >= 0) & (_b < _v.size)]] = False
            _valid[_i, :_v.size] = _v
        _nv = int(_valid.sum())
        _dev_valid = float(np.mean(np.abs(disp_field[_valid]))) if _nv else float("nan")
        _coverage = _nv / float(_valid.size)

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
    ism = float(p.get("provided_edge_ism", 0.0) or 0.0) if use_provided else float(p.get("interslice_smooth", 0.0) or 0.0)
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
    # 3d) RIGID-B-SCAN constraint (#rigidframe): an AXIAL B-scan (frame) is captured near-instantaneously, so its
    #     internal geometry is REAL — the correction of the sagittal (time-domain) inter-frame motion must only
    #     shift the WHOLE B-scan in depth, NEVER deform it per-column. The per-sagittal-slice flatten is already
    #     ~rigid on high-SNR interior frames (per-lateral disp spread ≈ detection noise), but on faint boundary
    #     frames the columns disagree and it bends the real geometry (the apex/limbus notch, per-frame disp spread
    #     3-6px ≫ the ~0 rigid part). Replace each frame's per-lateral displacement with ONE robust rigid shift
    #     (median over the reliable laterals, excluding clipped/user-cut columns whose disp is a clamp not a motion
    #     sample), then re-assert the tissue-preservation clamps. Interior barely changes (≤ noise); the boundary
    #     deformation is removed at the source. rigid_frame_warp=False → off (legacy per-column).
    #
    #     APPLIES ON THE CORRECTIONS PATH TOO (was: auto path only). The rule is a statement about the DATA, not
    #     about which code path produced the surface: a B-scan is captured near-instantaneously, so its internal
    #     geometry is real on a corrected scan exactly as it is on an auto one. Excluding use_provided meant that
    #     correcting a scan silently opted it out of the constraint and warped it per-column — the reviewer's
    #     corrections were the one case guaranteed to deform the B-scans.
    #
    #     CONSEQUENCE, deliberately accepted: the delivered volume no longer sits exactly on the drawn line at
    #     every column, because honouring a drawn line per-column IS the deformation. The drawn surface now
    #     determines ONE shift per frame (the median over reliable laterals), which is the most a rigid
    #     transform can take from it. Preview and result therefore differ per-column by design; the alternative
    #     is bending instantaneously-captured tissue to match a hand-drawn curve.
    if bool(p.get("rigid_frame_warp", False)):
        clamped = np.zeros(disp_field.shape, dtype=bool)          # (lateral, frame) clip/user-cut columns
        for i in range(n):
            cc = clip_cols_list[i]; zc = zero_cols_list[i]
            if len(cc):
                clamped[i, np.clip(np.asarray(cc, dtype=int), 0, disp_field.shape[1] - 1)] = True
            if len(zc):
                clamped[i, np.clip(np.asarray(zc, dtype=int), 0, disp_field.shape[1] - 1)] = True
        df = disp_field.astype(np.float64)
        # RIGID per-frame move fit across laterals. Default (auto path): TRANSLATION only (median depth shift). On
        # the CORRECTIONS path (use_provided) additionally fit a per-frame ROTATION (tilt) — the reviewer's spec:
        # "interpolate the manual edge GT and find the best axial rotation/TRANSLATION to correct it toward a
        # quadratic". df here is (per-lateral-quadratic - drawn edge), i.e. how far each column is from the smooth
        # dome; the least-squares line (translation t + tilt θ) across laterals is the SHARED rigid inter-frame move
        # (depth shift + torsion) that best carries the B-scan onto that dome. It is still a rigid B-scan move
        # (linear in lateral = a small rotation about the surface — NOT a per-column shear, so the instantaneously-
        # captured B-scan is never deformed) and removes inter-frame TILT a pure median shift cannot. The tilt is
        # clamped so a low-signal frame can't shear the scan. rigid_frame_rotate=False → median-only (old behaviour).
        _rotate = use_provided and bool(p.get("rigid_frame_rotate", True))
        _lat = np.arange(n, dtype=np.float64) - (n - 1) / 2.0     # lateral axis, centred (rotation pivot)
        _half = max(1.0, float(np.abs(_lat).max()))
        _maxtilt = float(p.get("rigid_frame_rotate_max", 20.0))   # max |tilt| across a half-width (px)
        deltas = np.full(df.shape[1], np.nan)                     # per-frame translation (depth shift at centre)
        slopes = np.zeros(df.shape[1])                            # per-frame rotation (tilt, px per lateral)
        for f in range(df.shape[1]):
            good = np.isfinite(df[:, f]) & (~clamped[:, f])
            if int(good.sum()) < 8:
                continue
            if _rotate:
                x = _lat[good]; y = df[good, f]; th = 0.0; t = float(np.median(y))
                try:                                              # robust deg-1 fit (drop outliers, refit twice)
                    c = np.polyfit(x, y, 1)
                    for _ in range(2):
                        r = y - np.polyval(c, x); sd = float(np.std(r)) + 1e-6
                        keep = np.abs(r) < 2.5 * sd
                        if int(keep.sum()) >= 8 and not keep.all():
                            x = x[keep]; y = y[keep]; c = np.polyfit(x, y, 1)
                        else:
                            break
                    th = float(c[0]); t = float(c[1])             # slope (tilt) + intercept at lateral centre
                except Exception:  # noqa: BLE001
                    th = 0.0; t = float(np.median(df[good, f]))
                if abs(th) * _half > _maxtilt:                    # clamp the tilt so it can't shear the scan
                    th = np.sign(th) * _maxtilt / _half
                deltas[f] = t; slopes[f] = th
            else:
                deltas[f] = float(np.median(df[good, f]))         # ONE rigid depth shift for the whole B-scan
        val = np.isfinite(deltas)
        if int(val.sum()) >= 4:
            # the inter-frame motion is SMOOTH (adjacent B-scans are ~40ms apart), so a per-frame value that
            # jitters frame-to-frame is DETECTION NOISE, not motion → smoothing the per-frame shift/tilt across
            # frames removes that jitter (the frames stay rigidly aligned, no B-scan deformation) and recovers a
            # smooth sagittal border. rigid_frame_smooth=0 → raw per-frame fit (jittery).
            xs = np.arange(df.shape[1], dtype=np.float64)
            deltas = np.interp(xs, xs[val], deltas[val])
            sig = float(p.get("rigid_frame_smooth", 1.5) or 0.0)
            if sig > 0:
                deltas = ndimage.gaussian_filter1d(deltas, sigma=sig, mode="nearest")
            if _rotate:
                slopes = np.interp(xs, xs[val], slopes[val])
                if sig > 0:
                    slopes = ndimage.gaussian_filter1d(slopes, sigma=sig, mode="nearest")
                df[:] = deltas[None, :] + slopes[None, :] * _lat[:, None]   # per-frame translation + rotation
            else:
                df[:] = deltas[None, :]
        disp_field = df
        for i in range(n):                                        # re-assert clip (disp>=0) + user-cut (disp==0)
            cc = clip_cols_list[i]
            if len(cc):
                disp_field[i][cc] = np.maximum(disp_field[i][cc], 0.0)
            zc = zero_cols_list[i]
            if len(zc):
                disp_field[i][zc] = 0.0
        # ...and re-assert the ABOVE-WINDOW clamp on the corrections path. It was applied to disp_field further
        # up, but the rigid replacement above overwrites every column with the frame's median — which can be
        # negative and would shift a column whose apex is already above row 0 further up, truncating real
        # epithelium off the top. This clamp is the reason that assignment cannot simply stand.
        if use_provided:
            neg = edges < 0.0
            if neg.any():
                disp_field = np.where(neg, np.maximum(disp_field, 0.0), disp_field)
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
        if return_coverage:
            # disp_mean is deliberately LEFT UNCHANGED: it is the objective iterate_smooth_volume
            # minimises to pick best_pass, so altering it would change which pass ships and could move
            # the 130 accepted scans. Pass selection keeps the historical number; QA also gets the
            # tissue-column-restricted twin and the valid fraction (both computed above, pre-smoothing).
            return corrected, disp_mean, _axial_roughness(edges), _dev_valid, _coverage
        return corrected, disp_mean, _axial_roughness(edges)
    return corrected


def _boundary_deviation(volume: np.ndarray, params: dict | None = None,
                        workers: int | None = None, detect_volume: np.ndarray | None = None,
                        return_coverage: bool = False):
    """Score a candidate volume's boundary quality on its own terms (no warp kept). Returns
    (in_plane_deviation, axial_roughness): the mean per-column deviation of the DETECTED boundary from
    its quadratic fit (how jagged WITHIN each sagittal slice), and the mean inter-slice first-difference
    (how jagged ACROSS slices = the en-face/axial 'hairiness', #3). Both in pixels; lower = better."""
    if return_coverage:   # QA only — adds the tissue-column-restricted metric + valid fraction
        _, m, ax, mv, cov = smooth_volume(volume, params, workers=workers, return_metric=True,
                                          return_coverage=True, detect_volume=detect_volume)
        return float(m), float(ax), float(mv), float(cov)
    _, m, ax = smooth_volume(volume, params, workers=workers, return_metric=True, detect_volume=detect_volume)
    return float(m), float(ax)


def measure_delivered_qa(volume: np.ndarray, params: dict | None = None, workers: int | None = None,
                         rhr_info: dict | None = None, path: str | None = None,
                         reported_dev: float | None = None) -> dict:
    """Measure quality on the volume that is ACTUALLY WRITTEN — the only honest QA point.

    Why this exists: metrics[]/axial_metrics[]/scores[]/best_pass are computed inside
    iterate_smooth_volume and returned BEFORE rigid_height_refine, rigid_frame_derotate, the GT anchors
    and the crops mutate the volume. The delivered result is therefore never scored.

    COMPARABILITY: `dev` is the coverage-corrected deviation (tissue-bearing columns only), so it is NOT
    numerically identical to metrics[]. The honest comparison target is metrics[best_pass] — the pass the
    delivered volume actually descends from, NOT metrics[-1]: iterate_smooth_volume picks best_pass by
    argmin, and in 38 of 271 scans the last pass is not the best (worst gap 3.30 px, case_cs017_os_v4).
    The caller passes it in as `reported_dev` so the two sit side by side.

    `coverage` is the fraction of (slice, frame) columns that carried tissue, and it is LOAD-BEARING:
    `dev` is DEFLATED by destroyed volume. A zeroed crop_region frame or the black padding of a
    surface-crop canvas removes points from the RANSAC quadratic fit that `dev` is the residual of, so
    the fit hugs the survivors more tightly and the deviation shrinks — measured, zeroing half the
    frames takes the metric 1.62 -> 0.16, i.e. a badly damaged scan reads ~10x CLEANER than an intact
    one. Restricting the mean to tissue columns does NOT repair this (verified: near-identical either
    way); it only makes the number well-defined. So `coverage` is a guard, not a correction:
    NEVER rank two scans on `dev` at different coverage, and never rank across `path` strata (a
    surface-crop scan is measured on an extended, partly-padded canvas). Below qa_coverage_flag the
    scan is flagged `low-coverage` precisely because its `dev` is not believable.

    `needs_review` is advisory ONLY — nothing gates, rejects or branches on it. Absent values are
    recorded as explicit null with a note in `qa_errors`, so "not measured" can never be read as "clean".

    Returns {} if disabled or if the primary measurement fails. QA must never break a preprocessing run:
    every leg, including the flag arithmetic, is inside a try."""
    try:
        p = {**DEFAULT_PARAMS, **(params or {})}
        if not p.get("final_qa", True):
            return {}
        out: dict = {"qa_errors": []}

        def _f(v, default):
            """float() that cannot raise — a malformed param must not take down the run."""
            try:
                f = float(v)
                return f if np.isfinite(f) else default
            except (TypeError, ValueError):
                return default

        # Measuring params, NOT correcting params: iterate_smooth_volume strips these before scoring
        # (they steer the warp, so leaving them in changes the number — measured -28% from a surface_cut
        # alone on identical voxels). Strip them here too so QA scores geometry, not user overrides.
        qp = dict(params or {})
        for _k in ("force_columns", "good_columns", "surface_cut"):
            qp.pop(_k, None)

        det_vol = _fill_black_bands(volume)          # same treatment the pass-scorer gives a warped volume
        try:
            _dev_all, ax, dev, cov = _boundary_deviation(volume, qp, workers=workers,
                                                         detect_volume=det_vol, return_coverage=True)
        except Exception:  # noqa: BLE001
            return {}
        if not (np.isfinite(dev) and np.isfinite(ax)):
            # A NaN would silently SUPPRESS needs_review (nan > x is False) and, worse, json.dumps writes
            # a bare NaN that starlette's allow_nan=False renderer rejects — one bad scan would 500 the
            # whole /api/cases/list. Drop the keys instead.
            out["qa_errors"].append("dev")
        else:
            out["dev"] = round(float(dev), 4)
            out["axial"] = round(float(ax), 4)
            out["score"] = round(float(dev) + 0.5 * float(ax), 4)   # same combination as the pass score
            out["dev_uncorrected"] = round(float(_dev_all), 4)      # the historical, coverage-blind number
        out["coverage"] = round(float(cov), 4) if np.isfinite(cov) else None
        if path:
            out["path"] = str(path)                  # stratum: never pool across diminishing/grew/surface_crop
        if reported_dev is not None:
            out["reported_dev"] = round(_f(reported_dev, float("nan")), 4)

        # TILT — the in-plane metric's documented null space (a pure frame-direction tilt contributes
        # exactly 0 px to dev). Detected on det_vol, matching the dev leg: every delivered volume carries
        # black padding from the rigid shifts, and detecting on the unfilled volume measures the padding.
        # dp_scar_guard off: it is 10-13x slower here and returns the same tilt (44.73 vs 44.66 measured).
        try:
            _tp = {**p, "dp_scar_guard": False}
            det = detect_surface_all(reformat_to_sagittal(det_vol), _tp, workers=workers)
            _slope, _total, _fd = estimate_global_tilt(det, int(volume.shape[0]), _tp)
            # estimate_global_tilt returns 0.0 when too few frames yield a robust depth — indistinguishable
            # from a genuinely level scan. Treat an all-zero result as unmeasured rather than as "level".
            if _total is None or not np.isfinite(_total) or (_slope == 0.0 and _total == 0.0):
                out["tilt_total"] = None
                out["qa_errors"].append("tilt")
            else:
                out["tilt_total"] = round(abs(float(_total)), 2)
        except Exception:  # noqa: BLE001
            out["tilt_total"] = None
            out["qa_errors"].append("tilt")

        # MOTION — max_jitter is measured INPUT severity. It is unavailable on paths where
        # rigid_height_refine has not run (notably surface-crop, which returns before it), and a missing
        # value must read as UNKNOWN, not as "no motion": it is the strongest known discriminator.
        jit = (rhr_info if isinstance(rhr_info, dict) else {}).get("max_jitter")
        if jit is None:
            out["max_jitter"] = None
            out["qa_errors"].append("max_jitter")
        else:
            out["max_jitter"] = _f(jit, None)
            if out["max_jitter"] is None:
                out["qa_errors"].append("max_jitter")

        reasons = []
        if out.get("max_jitter") is not None and out["max_jitter"] > _f(p.get("qa_jitter_flag"), 3.0):
            reasons.append("motion")
        if out.get("dev") is not None and out["dev"] > _f(p.get("qa_dev_flag"), 2.0):
            reasons.append("boundary")
        if out.get("coverage") is not None and out["coverage"] < _f(p.get("qa_coverage_flag"), 0.60):
            reasons.append("low-coverage")           # dev is not trustworthy on this scan
        # NOTE: tilt is recorded but deliberately does NOT raise a flag. Measured on the store, delivered
        # (post-flatten) tilt is O(10 px) while qa_tilt_flag was copied from detilt_min_total=150, a
        # PRE-flatten trigger — so the threshold could essentially never fire. Worse, measured tilt is
        # INVERTED against the human verdict (AUC 0.411: accepted scans median 70, rejected 48), so
        # flagging on it would preferentially flag GOOD scans. Left unflagged pending real calibration.
        out["needs_review"] = bool(reasons)
        out["review_reasons"] = reasons
        if not out["qa_errors"]:
            out.pop("qa_errors")
        return out
    except Exception:  # noqa: BLE001
        return {}


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


def _anterior_boundary(volume: np.ndarray, p: dict) -> np.ndarray:
    """The ANTERIOR CORNEAL BOUNDARY per (frame, lateral) — the quantity a reviewer actually judges.

    Deliberately INDEPENDENT of the DP detector used everywhere else in this module, because the DP detector
    is precisely what fails where the residual defects live: it flattens off the descending cornea at the
    acquisition-edge frames (so a residual computed from it there has the OPPOSITE SIGN to the real error)
    and it is pulled by specular columns. Two earlier alternatives were tried and rejected:

    * a plain threshold-crossing estimator locks onto eyelashes and saturation streaks;
    * the corneal band's intensity CENTROID is immune to all of that, but it tracks the centre of MASS, which
      can sit still while the boundary moves. Correcting cs041_os_v1 to flatten its centroid drove the
      anterior boundary from 0.48 px rms to 1.67 px — it manufactured the very bumps the review kept finding.

    So: first crossing of the halfway level between each A-scan's own background and its own peak, with
    SATURATED columns rejected outright (NaN) rather than measured. `volume` is (frames, depth, lateral);
    returns (frames, lateral) with NaN where there is no usable boundary."""
    sm = ndimage.uniform_filter1d(volume.astype(np.float32, copy=False), size=7, axis=1)
    bg = np.percentile(sm, 20, axis=1)
    pk = sm.max(axis=1)
    above = sm >= (bg + 0.5 * (pk - bg))[:, None, :]
    idx = np.argmax(above, axis=1).astype(np.float64)
    idx[~above.any(axis=1)] = np.nan
    sat = pk > np.percentile(pk, 99.5) * float(p.get("rfr_sat_frac", 0.98))
    # Saturation artefacts are RARE by definition. If this rule flags a large share of the A-scans it is not
    # finding specular columns — the peak intensity is simply near-uniform (a clipped or synthetic volume) —
    # and honouring it would blank the whole measurement. Ignore it in that case rather than measure nothing.
    if sat.mean() <= float(p.get("rfr_sat_max_frac", 0.05)):
        idx[sat] = np.nan
    return idx


def _dropout_frames(volume: np.ndarray, p: dict) -> np.ndarray:
    """Frames whose tissue signal has COLLAPSED — a blink, a full-width shadow, a dropout band.

    The boundary estimate has nothing to lock onto in such a frame and dives to the bottom of the window,
    which reads as a huge apparent displacement: on cs021_od_v3 a dropout band produced a 117 px "deviation"
    over 42 frames and would have moved every one of them to the clamp, chasing an artefact. A frame below
    `rfr_dropout_frac` of the scan-median signal is excluded from BOTH the reference fit and the correction —
    with no tissue there is nothing to align."""
    sig = np.nanmedian(volume.max(axis=1), axis=1).astype(np.float64)     # (frames,)
    med = float(np.nanmedian(sig))
    return (sig < float(p.get("rfr_dropout_frac", 0.55)) * med) if med > 0 else np.zeros(sig.shape, bool)


def _lateral_frame_curve(S: np.ndarray, lead: int, tail: int, deg: int = 4) -> np.ndarray:
    """Per-LATERAL robust degree-`deg` fit of the boundary ACROSS frames, fitted on the INTERIOR frames only
    and evaluated everywhere. Excluding the acquisition edges from the fit is the point: they must not
    influence the curve they are then judged against. S is (frames, lateral); returns the same shape."""
    F, L = S.shape
    x = np.arange(F, dtype=np.float64)
    interior = np.zeros(F, bool)
    interior[lead:F - tail] = True
    out = np.full((F, L), np.nan)
    need = 4 * (deg + 1)
    for i in range(L):
        y = S[:, i]
        ok = np.isfinite(y) & interior
        if int(ok.sum()) < need:
            continue
        c = np.polyfit(x[ok], y[ok], deg)
        for _ in range(2):
            r = y - np.polyval(c, x)
            s = np.nanstd(r[ok])
            if not np.isfinite(s) or s <= 0:
                break
            ok2 = ok & (np.abs(r) < 2.0 * s)
            if int(ok2.sum()) < need:
                break
            c = np.polyfit(x[ok2], y[ok2], deg)
            ok = ok2
        out[:, i] = np.polyval(c, x)
    return out


def _rfr_split(p: dict):
    """(lead, tail) frames excluded from the interior — i.e. the boundary between the two corrections.

    The two regions MUST be disjoint. They are measured against different references and correct different
    things, so an overlap makes them fight: with the tuning inherited from the review harness (lead 14,
    tail 5, edge 12) the last 7 frames were BOTH inside the interior curve fit AND given an edge shift, so
    moving them changed the very curve the interior was judged against. Measured on the approved corpus, that
    showed up as the interior deviation degrading (e.g. cs001_od_v1 0.154 -> 0.253 px) on scans where the
    edge correction was otherwise a clear win. Widening the exclusion to at least `rfr_edge_n` removes it."""
    edge_n = int(p.get("rfr_edge_n", 12))
    return max(int(p.get("rfr_lead", 14)), edge_n), max(int(p.get("rfr_tail", 5)), edge_n)


def _rfr_from_surface(S: np.ndarray, p: dict):
    """dev / prof / C from an already-known boundary map. Everything downstream reads only these."""
    L = int(S.shape[1])
    lead, tail = _rfr_split(p)
    band = slice(int(float(p.get("rfr_lat_lo", 0.20)) * L), int(float(p.get("rfr_lat_hi", 0.80)) * L))
    C = _lateral_frame_curve(S, lead, tail)
    with np.errstate(all="ignore"):
        dev = np.nanmedian((S - C)[:, band], axis=1)
        prof = np.nanmedian(S[:, band], axis=1)
    return dev, prof, C


def _rfr_predict_surface(S: np.ndarray, shift: np.ndarray, tilt: np.ndarray) -> np.ndarray:
    """The boundary map a RIGID plan produces, computed ANALYTICALLY rather than by re-detecting.

    Scoring by re-detection is not translation-invariant. `_warp_by_displacement` interpolates each A-scan,
    which moves where `_anterior_boundary`'s half-max crossing lands by a fraction of a pixel, and the
    acceptance metrics amplify that: measured over 378 (scan, control) pairs, a UNIFORM per-frame shift —
    which moves the whole volume as one rigid body and cannot change any geometry — drifts edge_spread by
    3.47 px at p90, and the composite gate then DISCARDS the result on 54% of approved scans. A net-zero
    round trip (+0.5 px then -0.5 px, ending exactly where it started) is discarded on 38%. The artefact is a
    function of the interpolation PHASE, not the distance: -0.5 and -1.5 px drift identically, while an exact
    integer shift drifts 0.0000 because it does not interpolate at all.

    Because every correction this pass makes is rigid, its effect on the boundary is exact and closed-form —
    a depth shift moves the whole A-scan column, a tilt adds a linear ramp — so predicting it removes the
    artefact by construction instead of tolerating it, and saves the second detection pass as well."""
    L = int(S.shape[1])
    lat = np.arange(L, dtype=np.float64) - (L - 1) / 2.0
    return S - shift[:, None] - tilt[:, None] * lat[None, :]


def _rfr_deviation(volume: np.ndarray, p: dict):
    """Per-frame boundary deviation `dev` (interior) and `prof` (the per-frame boundary depth used at the
    edges). Both are medians over the CENTRAL laterals: the periphery is dim and speckled, its estimate is
    unreliable, and including it drags the per-frame number off (peripheral per-slice roughness runs ~4x the
    centre). Dropout frames are NaN in both."""
    S = _anterior_boundary(volume, p)
    drop = _dropout_frames(volume, p)
    S[drop] = np.nan
    dev, prof, C = _rfr_from_surface(S, p)
    dev[drop] = np.nan
    prof[drop] = np.nan
    return dev, prof, S, C


def _r2(v):
    """Round for telemetry, but keep an unmeasurable value an explicit null rather than a plausible 0.0."""
    return None if not np.isfinite(v) else round(float(v), 2)


def _r3(v):
    return None if not np.isfinite(v) else round(float(v), 3)


def _rfr_side_windows(F: int, side: str, p: dict, inward: int = 0):
    """(fit window, predicted block, seam frame) for one side, optionally slid `inward` frames off the edge.

    `inward > 0` produces a SHAM edge in the undisputed interior — the null control this pass is gated on.
    Same window lengths, same lever arm, same estimator: the only difference is that there is no acquisition
    edge there, so whatever it reports is the estimator's own extrapolation error and nothing else."""
    en, nf = int(p.get("rfr_edge_n", 12)), int(p.get("rfr_edge_fit", 28))
    if side == "lead":
        fit = np.arange(en + inward, en + inward + nf)
        blk = np.arange(inward, en + inward)
        seam = en + inward
    else:
        fit = np.arange(F - en - nf - inward, F - en - inward)
        blk = np.arange(F - en - inward, F - inward)
        seam = F - en - inward - 1
    if fit[0] < 0 or fit[-1] >= F or blk[0] < 0 or blk[-1] >= F:
        return None
    return fit, blk, seam


def _rfr_edge_fit(prof: np.ndarray, fit: np.ndarray, seam: int, p: dict):
    """The reference curve for one side: a SEAM-WEIGHTED quadratic over `fit`, evaluated everywhere.

    Weighted, not plain least squares, because plain least squares minimises AVERAGE error over the window
    while this curve is used only to EXTRAPOLATE from its seam end. Nothing pins the endpoint, and on
    cs039_os_v1 that endpoint residual was -2.46 px — the largest in its own window, since the profile is flat
    over frames 78-84 and then descends fast over 85-88. Extrapolating from there pushed the edge block down
    4-9.5 px, which a reviewer marked as a step. Weighting by exp(-|f - seam| / rfr_edge_tau) pins the fit
    where it is about to extrapolate from while the far frames still stabilise the curvature.

    NOTE ON numpy.polyfit's `w`: it multiplies the RESIDUALS, so the effective weight is w**2. Passing sqrt of
    the intended weight is deliberate — passing the weight directly halves the decay constant, and the result
    is sensitive to it (on cs039 that variant asks +11.7 px where this one asks +0.6).

    The quadratic is kept — a linear reference was measured and rejected. The interior window's uncurved slope
    is 1.193 px/frame against a true edge-block slope of 2.201, because a corneal dome's slope grows with
    eccentricity; dropping the curvature term injects ~6 px of spurious deviation across the block, which both
    misses real defects (it cancels them) and invents new ones."""
    ok = np.isfinite(prof[fit])
    if int(ok.sum()) < 12:
        return None
    fx = fit.astype(np.float64)
    tau = float(p.get("rfr_edge_tau", 6.0) or 0.0)
    x = np.arange(prof.size, dtype=np.float64)
    try:
        if tau > 0:
            w = np.sqrt(np.exp(-np.abs(fx[ok] - float(seam)) / tau))
            c = np.polyfit(fx[ok], prof[fit][ok], 2, w=w)
        else:
            c = np.polyfit(fx[ok], prof[fit][ok], 2)
    except Exception:  # noqa: BLE001
        return None
    return np.polyval(c, x)


def _rfr_second_diff_rms(prof: np.ndarray, idx: np.ndarray) -> float:
    v = prof[idx]
    v = v[np.isfinite(v)]
    if v.size < 5:
        return float("nan")
    return float(np.sqrt(np.mean(np.diff(v, 2) ** 2)))


def _rfr_hf(v: np.ndarray, sigma: float) -> np.ndarray:
    """The frame-to-frame (high-frequency) part of a per-frame series: v minus a gaussian low-pass of it.

    NaN-SAFE. Replacing a missing frame with 0 and then differencing against the smoothed series invents a
    deviation the size of the local level — measured at 6.4 px on a dropout frame, which the pass would then
    have "corrected" by moving a B-scan that has no tissue to align. Missing entries are filled from their
    neighbours for the smoothing only, and returned as NaN so nothing downstream can act on them."""
    x = np.asarray(v, dtype=np.float64)
    ok = np.isfinite(x)
    if not ok.any():
        return np.full(x.shape, np.nan)
    idx = np.arange(x.size, dtype=np.float64)
    filled = np.interp(idx, idx[ok], x[ok])
    hf = filled - ndimage.gaussian_filter1d(filled, sigma, mode="nearest")
    return np.where(ok, hf, np.nan)


def _rfr_interior_jitter(dev: np.ndarray, p: dict) -> float:
    """RMS frame-to-frame jitter of the INTERIOR per-frame deviation — this scan's own null for the wobble
    measure. On cs044_os_v1 it is 0.18 px while the trailing edge block sits at 1.72 px, a 10x ratio."""
    F = int(dev.size)
    lead, tail = _rfr_split(p)
    seg = dev[lead:F - tail]
    if seg.size < 12 or not np.isfinite(seg).any():
        return float("nan")
    hf = _rfr_hf(seg, float(p.get("rfr_jit_sigma", 2.0)))
    return float(np.sqrt(np.mean(hf ** 2)))


def _rfr_edge_deviation(prof: np.ndarray, p: dict, a_int: np.ndarray | None = None, detail: dict | None = None,
                        dev: np.ndarray | None = None):
    """Per-frame deviation of the EDGE frames from the cornea's own local curvature. NaN outside acted-on sides.

    Every gate here exists because a measured belief turned out to be wrong.

    ANCHORED AT THE SEAM. The reference is translated so it passes through the CORRECTED interior value at the
    seam frame, and the deviation is measured against that. Without it a nonzero fit residual at the seam
    becomes a hard step by construction, because `_rfr_plan` writes the edge deviation into the 12 edge frames
    while the adjacent interior frame keeps its own near-zero one. Measured across the 138 approved scans, the
    unanchored version induces a seam step of median 0.63 px, p90 1.99 px and max 38.9 px — on volumes a
    clinician had already signed off. Anchoring makes a step impossible rather than unlikely.

    GATED AGAINST THE SCAN'S OWN NULL. The old fixed 0.8 px trigger was an order of magnitude below this
    statistic's noise floor. Slide the identical fit-and-extrapolate estimator into the undisputed interior and
    it reads just as large: across the approved corpus the sham-edge median is 3.40 px against 2.87 px at the
    real edge, and a sweep of sham positions 6/12/20/28/36 frames inboard gives 6.57/6.75/5.99/5.95/5.90 px
    against 6.99 px at the edge — a paired difference of -0.08 px. What the statistic mostly measures is the
    error of pushing a 28-frame quadratic 12 frames past its own window, which is why it fired on 97.1% of
    approved scan-ends. So the trigger is now per scan and per side: the real ask must beat that side's own
    sham distribution, not a constant.

    REQUIRES A KINK, NOT AN OFFSET. The one thing that IS genuinely edge-specific is roughness: the edge
    block's second-difference rms is 5.49 px against 1.56 px in the interior, while the MEDIAN second
    difference is identical at 1.00 px — the excess lives in a minority of frames as a kink. It tracks
    instability of the boundary estimate (lateral scatter rho +0.29..+0.39) rather than signal loss (peak
    intensity runs the wrong way). A smooth edge that merely sits off an extrapolation is not a defect.

    NOT GATED on how much the correction re-slopes the edge, though that sounds right: it was measured and it
    is backwards. The false correction on cs039 re-slopes by 0.49 px/frame while the eye-verified true one on
    cs050_od_v1 re-slopes by 0.67. Genuine edge defects ARE slope defects — cs050's leading frames read
    61,61,61,62,63, having flattened off a descending cornea — so such a rule blocks exactly what it should fix.
    """
    F = int(prof.size)
    out = np.full(F, np.nan)
    if int(p.get("rfr_edge_n", 12)) <= 0:
        return out
    ai = a_int if a_int is not None else np.zeros(F)
    dev_full = dev
    wild = float(p.get("rfr_edge_wild_px", 45.0))
    gate_abs = float(p.get("rfr_gate_abs_px", 3.0))
    gate_ratio = float(p.get("rfr_gate_ratio", 1.2))
    kink_ratio = float(p.get("rfr_gate_kink", 1.0))
    smooth_abs = float(p.get("rfr_gate_smooth_px", 6.0))
    step_px = float(p.get("rfr_seam_step_px", 1.0))
    step_frac = float(p.get("rfr_seam_step_frac", 0.35))
    shams = [int(v) for v in (p.get("rfr_placebo_offsets") or (12, 18, 24, 30, 36))]

    for side in ("lead", "trail"):
        rec = {"side": side}
        w = _rfr_side_windows(F, side, p)
        if w is None:
            continue
        fit, blk, seam = w
        pred = _rfr_edge_fit(prof, fit, seam, p)
        if pred is None or not np.isfinite(prof[seam]):
            rec["decline"] = "unmeasurable"
            if detail is not None:
                detail[side] = rec
            continue
        anchor = (float(prof[seam]) - float(ai[seam])) - float(pred[seam])
        dev = np.where(np.isfinite(prof[blk]), prof[blk] - (pred[blk] + anchor), np.nan)
        if not np.isfinite(dev).any():
            rec["decline"] = "unmeasurable"
            if detail is not None:
                detail[side] = rec
            continue
        ask = float(np.nanmax(np.abs(dev)))
        rec["ask"] = round(ask, 2)

        # the frame that touches the seam — the step the reviewer would see
        near = dev[0] if side == "trail" else dev[-1]
        rec["seam_step"] = None if not np.isfinite(near) else round(float(near), 2)

        placebo = []
        for j in shams:
            ws = _rfr_side_windows(F, side, p, inward=j)
            if ws is None:
                continue
            f2, b2, s2 = ws
            pr2 = _rfr_edge_fit(prof, f2, s2, p)
            if pr2 is None or not np.isfinite(prof[s2]):
                continue
            an2 = float(prof[s2]) - float(pr2[s2])
            d2 = np.where(np.isfinite(prof[b2]), prof[b2] - (pr2[b2] + an2), np.nan)
            if np.isfinite(d2).any():
                placebo.append(float(np.nanmax(np.abs(d2))))
        pmed = float(np.median(placebo)) if placebo else float("nan")
        rec["placebo"] = None if not np.isfinite(pmed) else round(pmed, 2)
        thr = max(gate_abs, gate_ratio * pmed) if np.isfinite(pmed) else gate_abs
        rec["threshold"] = round(float(thr), 2)

        # JITTER, measured separately and trusted further. The block-level ask is dominated by how well a
        # quadratic extrapolates 12 frames past its window, which is why it must be gated against a sham null.
        # But that error is SMOOTH in frame index — a polynomial cannot zigzag — so the frame-to-frame
        # component of the deviation is real per-frame motion regardless of how wrong the reference's overall
        # level is. cs044_os_v1: the trailing block's jitter is 3.21 px max / 1.72 rms against 0.62 / 0.18 in
        # the same scan's interior, and it peaks on exactly the frames a reviewer marked as "small corneal
        # surface unevenness". Before this, a DECLINED block kept its jitter along with its offset.
        jit = _rfr_hf(dev, float(p.get("rfr_jit_sigma", 2.0)))
        null_j = _rfr_interior_jitter(dev_full, p) if dev_full is not None else float("nan")
        thr_j = max(float(p.get("rfr_jit_min_px", 1.2)),
                    float(p.get("rfr_jit_ratio", 4.0)) * (null_j if np.isfinite(null_j) else 0.0))
        rec["jitter_rms"] = round(float(np.sqrt(np.mean(jit ** 2))), 2)
        rec["jitter_null"] = _r2(null_j)
        rec["jitter_thr"] = round(float(thr_j), 2)

        if ask > wild:
            rec["decline"] = "implausible (artefact, e.g. an eyelash)"
        elif not (ask > thr):
            rec["decline"] = "within this scan's own null"
        elif np.isfinite(near) and abs(float(near)) > max(step_px, step_frac * ask):
            # PROPORTIONAL, not absolute. A genuine edge defect begins AT the seam and grows outward —
            # cs050_od_v1's leading frames read 61,61,61,62,63, having flattened off a descending cornea — so
            # its deviation at the seam-adjacent frame is necessarily nonzero, and an absolute 1px cap
            # rejected exactly the corrections that were verified correct (cs039 lead 1.33 against a 32.6px
            # defect, cs050 lead 1.44 against 10.5px). What must be refused is a block moved essentially
            # UNIFORMLY, where the step is most of the correction: that is an offset, not a kink, and an
            # offset is what the fit-error failure mode looks like.
            rec["decline"] = "uniform block offset — the step would be most of the correction"
        else:
            k_edge = _rfr_second_diff_rms(prof, blk)
            k_int = _rfr_second_diff_rms(prof, fit)
            kinked = (np.isfinite(k_edge) and np.isfinite(k_int) and k_int > 0
                      and k_edge > kink_ratio * k_int)
            rec["kink"] = None if not (np.isfinite(k_edge) and np.isfinite(k_int) and k_int > 0) \
                else round(k_edge / k_int, 2)
            # A KINK, OR SOMETHING TOO BIG TO BE NOISE. Roughness is the one property that is genuinely
            # edge-specific in the population (2nd-difference rms 5.49 px at the edge against 1.56 px in the
            # interior, while the MEDIAN 2nd difference is identical at 1.00 px), so it is the usual evidence.
            # But requiring it alone rejects a real failure mode: cs050_od_v1's leading edge flattens SMOOTHLY
            # off a descending cornea — frames read 61,61,61,62,63 — a 10.5 px defect with a kink ratio of
            # 0.51. So a smooth block may still be corrected, but only when it is far too large to be the
            # estimator's own error: 6 px, against an interior sham median of ~3.4 px across the corpus.
            if not (kinked or ask > smooth_abs):
                rec["decline"] = "smooth and small — off the extrapolation, but within estimator error"
            else:
                out[blk] = dev
                rec["applied"] = True
                rec["route"] = "kink" if kinked else "large-and-smooth"
        if not rec.get("applied") and rec.get("decline") != "implausible (artefact, e.g. an eyelash)":
            # the block offset stays, but its frame-to-frame wobble does not have to
            big = np.abs(jit) > thr_j
            if big.any() and float(np.max(np.abs(jit))) > thr_j:
                out[blk] = np.where(big, jit, 0.0)
                rec["jitter_frames"] = int(big.sum())
                rec["route"] = "jitter-only"
        if detail is not None:
            detail[side] = rec
    return out


def _rfr_lateral_bands(L: int, p: dict):
    """Lateral bands used to see ACROSS the width. The per-frame profile everything else uses is a median
    over the central laterals, which by construction cannot see a defect that varies from one side of the
    B-scan to the other. The outermost `rfr_band_margin` fraction is excluded: there the boundary estimate is
    unreliable (on cs039_os_v1 laterals 2 and 508 read -193 px and -115 px off the arc, pure estimator noise)."""
    mrg = float(p.get("rfr_band_margin", 0.05))
    n = int(p.get("rfr_bands", 10))
    lo, hi = int(mrg * L), int((1.0 - mrg) * L)
    if hi - lo < 8 * n:
        return []
    edges = np.linspace(lo, hi, n + 1).astype(int)
    return [(edges[i], edges[i + 1]) for i in range(n)]


def _rfr_band_edge_dev(S: np.ndarray, p: dict):
    """Edge deviation per (lateral band, frame), against each band's OWN seam-anchored local curvature.

    Returns (bands, centres, dev) with dev shape (n_bands, n_frames), NaN outside the edge blocks."""
    F, L = int(S.shape[0]), int(S.shape[1])
    bands = _rfr_lateral_bands(L, p)
    dev = np.full((len(bands), F), np.nan)
    if not bands:
        return bands, np.zeros(0), dev
    x = np.arange(F, dtype=np.float64)
    for bi, (lo, hi) in enumerate(bands):
        with np.errstate(all="ignore"):
            pr = np.nanmedian(S[:, lo:hi], axis=1)
        for side in ("lead", "trail"):
            w = _rfr_side_windows(F, side, p)
            if w is None:
                continue
            fit, blk, seam = w
            pred = _rfr_edge_fit(pr, fit, seam, p)
            if pred is None or not np.isfinite(pr[seam]):
                continue
            dev[bi, blk] = pr[blk] - (pred[blk] + (pr[seam] - pred[seam]))
    return bands, np.array([(lo + hi) / 2.0 for lo, hi in bands]), dev


def _rfr_banded_second_diff(S: np.ndarray, p: dict) -> float:
    """Roughness ALONG the frame axis, measured inside each lateral band.

    The one acceptance metric the tilt fitters cannot satisfy by construction. `_rfr_edge_tilt` and
    `_rfr_interior_tilt` least-squares-fit a line ACROSS bands and subtract it, so any metric of across-band
    spread is partly self-fulfilling — it improves whenever a tilt is fitted, which is exactly when an
    independent check is needed. This differences along the OTHER axis.

    It is also the sharpest detector of the failure that motivated the tilt smoothing: on a 6 px ALTERNATING
    edge rotation (the "+10.0, +4.7, -12.0, -12.7 px" fit that spurious noise produced on cs044_os_v1) it
    fires on 96.8% of scans against edge_spread's 75%. Correctly silent on a benign smooth dome ramp, which
    has no second difference at all — and, by the same token, blind to a SMOOTH rotation, so it complements
    edge_spread rather than replacing it."""
    L = int(S.shape[1])
    bands = _rfr_lateral_bands(L, p)
    if not bands:
        return float("nan")
    with np.errstate(all="ignore"):
        V = np.array([np.nanmedian(S[:, lo:hi], axis=1) for lo, hi in bands])
        D = V[:, :-2] - 2.0 * V[:, 1:-1] + V[:, 2:]
        if not np.isfinite(D).any():
            return float("nan")
        return float(np.sqrt(np.nanmean(D ** 2)))


def _rfr_edge_spread(S: np.ndarray, p: dict) -> float:
    """Worst across-the-width spread of the edge deviation, in px. This is the quantity a reviewer sees as
    'the edge goes upward against the corneal curvature only on the far sagittal slices' — a defect that is
    invisible to every central-band measure, because the central band is exactly where it vanishes."""
    _b, _c, dev = _rfr_band_edge_dev(S, p)
    if dev.size == 0 or not np.isfinite(dev).any():
        return float("nan")
    have = np.isfinite(dev).any(axis=0)              # frames outside every edge block are all-NaN by design
    if not have.any():
        return float("nan")
    with np.errstate(all="ignore"):
        rng = np.nanmax(dev[:, have], axis=0) - np.nanmin(dev[:, have], axis=0)
    # MEAN over the edge frames, not max. Measured under a uniform per-frame shift — a correction that moves
    # the whole volume rigidly and therefore cannot change any geometry — the max drifts by 2.74 px (p90,
    # 20 scans) purely from sub-pixel resampling perturbing where the boundary estimator's half-max crossing
    # lands; the mean drifts 0.74. An exact INTEGER shift, which does not interpolate, drifts ~0 for both,
    # which is what identifies the mechanism. The mean is also the better detector: on an alternating-sign
    # 1 degree per-frame rotation it separates from the control on 100% of scans against the max's 86%.
    return float(np.nanmean(rng)) if np.isfinite(rng).any() else float("nan")


def _rfr_band_residual(S: np.ndarray, C: np.ndarray, p: dict):
    """Per-frame boundary residual measured in each lateral band. (centres, values(bands, frames)).

    Everything else in this pass collapses the laterals to one median, which cannot see a defect that is
    deep on one side of the B-scan and shallow on the other — and that is precisely how a per-frame ROTATION
    presents. On cs035_od_v1_4 a reviewer marked one column at sagittal slice 73 and the same column at
    slice 475: the two ends of a 7.7 px ramp whose frame-median is 0.07 px, i.e. invisible to every other
    measure here."""
    F, L = int(S.shape[0]), int(S.shape[1])
    bands = _rfr_lateral_bands(L, p)
    if not bands:
        return np.zeros(0), np.zeros((0, F))
    R = S - C
    with np.errstate(all="ignore"):
        vals = np.array([np.nanmedian(R[:, lo:hi], axis=1) for lo, hi in bands])
    return np.array([(lo + hi) / 2.0 for lo, hi in bands]), vals


def _rfr_fit_tilt(ctr: np.ndarray, v: np.ndarray, p: dict):
    """Robust (slope, spread, residual-rms) of one frame's across-width residual. slope is px per lateral."""
    ok = np.isfinite(v)
    if int(ok.sum()) < 4:
        return None
    c = np.polyfit(ctr[ok], v[ok], 1)
    for _ in range(2):
        r = v - np.polyval(c, ctr)
        sd = np.nanstd(r[ok])
        if not np.isfinite(sd) or sd <= 0:
            break
        ok2 = ok & (np.abs(r) < 2.0 * sd)
        if int(ok2.sum()) < 4:
            break
        c = np.polyfit(ctr[ok2], v[ok2], 1)
        ok = ok2
    resid = v[ok] - np.polyval(c, ctr[ok])
    return float(c[0]), float(np.nanmax(v[ok]) - np.nanmin(v[ok])), float(np.sqrt(np.mean(resid ** 2)))


def _rfr_precheck_tilt(m: dict, p: dict) -> np.ndarray:
    """Every rotation the pass would apply, interior AND edge — consulted before the early-out, because a scan
    can need a rotation while needing no depth shift at all. Two reviewer findings landed here: cs035_od_v1_4's
    marked column carries a 7.7 px across-width ramp with a 0.07 px frame median, and cs044_os_v1_3's marked
    columns carry 11.2 px and 9.1 px ramps on a scan whose every block offset sits inside its own null. Leaving
    the edge tilt out of this check made the second case exit as "nothing to correct" before it was computed."""
    n = int(m["prof"].size)
    out = np.zeros(n)
    try:
        out = _rfr_interior_tilt(m["S"], m["C"], p)[0]
    except Exception:  # noqa: BLE001
        pass
    try:
        et = _rfr_edge_tilt(m["S"], np.full(n, np.nan), p)
        out = np.where(np.abs(et) > 1e-9, et, out)
    except Exception:  # noqa: BLE001
        pass
    return out


def _rfr_interior_spread(S: np.ndarray, C: np.ndarray, p: dict) -> float:
    """Worst across-the-width spread of the INTERIOR per-frame residual, in px. The quantity a reviewer sees
    as 'a very small notch' that appears at one sagittal slice and reverses at another."""
    F = int(S.shape[0])
    lead, tail = _rfr_split(p)
    ctr, vals = _rfr_band_residual(S, C, p)
    if vals.size == 0:
        return float("nan")
    with np.errstate(all="ignore"):
        rng = np.nanmax(vals[:, lead:F - tail], axis=0) - np.nanmin(vals[:, lead:F - tail], axis=0)
    # MEAN, for the same reason as the edge version: the max carries a 0.48 px resampling floor and detects
    # per-frame noise on only 43% of scans, while the mean carries 0.04 px and detects it on 93%.
    return float(np.nanmean(rng)) if np.isfinite(rng).any() else float("nan")


def _rfr_interior_tilt(S: np.ndarray, C: np.ndarray, p: dict):
    """Per-frame lateral TILT of the INTERIOR frames, as a depth ramp in px per lateral column.

    WHY THIS IS NOT THE 3-DOF FIT THAT WAS REJECTED. That attempt added lateral translation AND rotation to
    EVERY frame from a per-frame least-squares fit, and it was bias-dominated: 100% of fitted lateral shifts
    came out positive (median +0.50 px, an estimator artefact) and the boundary rms rose 0.49 -> 0.84 px. Here
    there is no lateral translation, and a rotation is fitted only where it stands clear of THIS scan's own
    per-frame tilt noise — the same per-scan null the edge gates use, rather than a constant that fits no scan.

    On cs035_od_v1_4 the marked column measures a 7.7 px ramp across the width (0.34 deg) while the median
    interior frame measures 0.5 px, so it stands ~6x clear. `rigid_frame_derotate` runs earlier and removes
    the bulk of the inter-frame torsion, but it is driven by the DP surface and levels tilt to a smooth
    across-frames baseline; what survives it is exactly this kind of isolated single-frame residual."""
    F, L = int(S.shape[0]), int(S.shape[1])
    tilt = np.zeros(F, dtype=np.float64)
    info = {"frames": 0, "max_px": 0.0}
    if not bool(p.get("rfr_interior_tilt", True)):
        return tilt, info
    lead, tail = _rfr_split(p)
    ctr, vals = _rfr_band_residual(S, C, p)
    if vals.shape[0] < 4:
        return tilt, info
    fits = {}
    for f in range(lead, F - tail):
        r = _rfr_fit_tilt(ctr, vals[:, f], p)
        if r is not None:
            fits[f] = r
    if len(fits) < 12:
        return tilt, info
    width = float(L - 1)
    across = np.array([abs(v[0]) * width for v in fits.values()])
    null = float(np.median(across))                       # this scan's own per-frame tilt noise
    thr = max(float(p.get("rfr_int_tilt_min_px", 3.0)), float(p.get("rfr_int_tilt_ratio", 4.0)) * null)
    sx = float(p.get("oct_spacing_lateral", 0.0078))
    sz = float(p.get("oct_spacing_depth", 0.0031))
    max_slope = np.tan(np.radians(float(p.get("rfr_int_max_tilt_deg", 1.0)))) * (sx / sz)
    for f, (slope, spread, rms) in fits.items():
        if abs(slope) * width < thr:
            continue
        # a straight line must actually explain it, or this is noise being shaped into a rotation
        if rms > 0.5 * max(spread, 1e-6):
            continue
        tilt[f] = float(np.clip(slope, -max_slope, max_slope))
    info = {"frames": int(np.sum(np.abs(tilt) > 1e-9)), "null_px": round(null, 2),
            "threshold_px": round(thr, 2),
            "max_px": round(float(np.max(np.abs(tilt)) * width), 2),
            "max_deg": round(float(np.degrees(np.arctan(np.max(np.abs(tilt)) * sz / sx))), 3)}
    return tilt, info


def _in_edge_block(f: int, F: int, p: dict) -> bool:
    en = int(p.get("rfr_edge_n", 12))
    return f < en or f >= F - en


def _rfr_edge_tilt_null(S: np.ndarray, p: dict) -> float:
    """This scan's own null for the edge across-width tilt: the same fit run at interior SHAM positions.

    The block-offset gate already works this way, and the tilt needs it for the same reason — the statistic
    has a per-scan floor set by how noisy that scan's band medians are, and a constant threshold fits no scan.
    Measured: cs044_os_v1_3's null p90 is 4.80 px while its two genuinely rotated frames read 9.07 and 11.16;
    on cs044_os_v1_2, whose periphery is far noisier, the null runs to 7.83 px and the largest edge tilt
    (6.86) sits INSIDE it and must not be acted on."""
    F, L = int(S.shape[0]), int(S.shape[1])
    bands = _rfr_lateral_bands(L, p)
    if len(bands) < 4:
        return float("nan")
    ctr = np.array([(lo + hi) / 2.0 for lo, hi in bands])
    prof_b = [np.nanmedian(S[:, lo:hi], axis=1) for lo, hi in bands]
    vals = []
    for side in ("lead", "trail"):
        for j in [int(v) for v in (p.get("rfr_placebo_offsets") or (12, 18, 24, 30, 36))]:
            w = _rfr_side_windows(F, side, p, inward=j)
            if w is None:
                continue
            fit, blk, seam = w
            rows = []
            for pr in prof_b:
                pred = _rfr_edge_fit(pr, fit, seam, p)
                if pred is None or not np.isfinite(pr[seam]):
                    rows.append(np.full(len(blk), np.nan))
                else:
                    rows.append(pr[blk] - (pred[blk] + (pr[seam] - pred[seam])))
            sub = np.array(rows)
            for i in range(sub.shape[1]):
                v = sub[:, i]
                ok = np.isfinite(v)
                if int(ok.sum()) >= 4:
                    vals.append(abs(float(np.polyfit(ctr[ok], v[ok], 1)[0] * (L - 1))))
    return float(np.percentile(vals, 90)) if vals else float("nan")


def _rfr_edge_tilt(S: np.ndarray, edev: np.ndarray, p: dict, allow: np.ndarray | None = None):
    """Per-frame lateral TILT of the accepted edge blocks, as a depth ramp in px per lateral column.

    A rigid depth shift is uniform across the B-scan, so it can only ever remove the AVERAGE of a defect that
    varies across the width. On cs039_os_v1's leading frame the deviation ramps from +21.7 px at one lateral
    end to +42.9 px at the other; shifting by the central-band value (+32.8) corrected the centre and left
    ~10 px of over-lift at one end and under-lift at the other. The reviewer saw exactly that: "the last few
    sagittal frames near 513 have the right edge slightly going upwards against the general cornea curvature".

    That across-width ramp is a ROTATION of the B-scan, not a deformation, so it is allowed — it is the same
    quantity `rigid_frame_derotate` corrects, applied here to the acquisition-edge frames it does not reach,
    and applied the same way (a linear per-column depth ramp; distances within the B-scan are preserved).
    Measured on that frame the tilt is 20.7 px across the width = 0.92 degrees, comfortably inside real
    inter-frame torsion.

    Returns a per-frame slope array (px per lateral column), zero wherever the tilt is not supported."""
    F, L = int(S.shape[0]), int(S.shape[1])
    tilt = np.zeros(F, dtype=np.float64)
    if not bool(p.get("rfr_edge_tilt", True)):
        return tilt
    bands, ctr, dev = _rfr_band_edge_dev(S, p)
    if len(bands) < 4:
        return tilt
    sx = float(p.get("oct_spacing_lateral", 0.0078))
    sz = float(p.get("oct_spacing_depth", 0.0031))
    max_slope = np.tan(np.radians(float(p.get("rfr_edge_max_tilt_deg", 1.5)))) * (sx / sz)
    mid = (L - 1) / 2.0
    # Fitted for EVERY edge frame, not only those whose block offset was accepted. A declined block used to
    # get no rotation at all, because edev is NaN there — so a frame that was rotated but not displaced was
    # invisible to this pass. A reviewer found exactly that on cs044_os_v1_3: display columns 7 and 11 carry
    # clean 11.2 px and 9.1 px across-width ramps (linear-fit residuals 0.93 and 1.28) on a scan whose block
    # offsets were all within their own null.
    null = _rfr_edge_tilt_null(S, p)
    thr_null = (float(p.get("rfr_edge_tilt_ratio", 1.5)) * null) if np.isfinite(null) else 0.0
    for f in range(F):
        if not _in_edge_block(f, F, p):
            continue

        v = dev[:, f]
        ok = np.isfinite(v)
        if int(ok.sum()) < 4:
            continue
        c = np.polyfit(ctr[ok], v[ok], 1)
        for _ in range(2):                    # a single bad band must not set the rotation
            r = v - np.polyval(c, ctr)
            sd = np.nanstd(r[ok])
            if not np.isfinite(sd) or sd <= 0:
                break
            ok2 = ok & (np.abs(r) < 2.0 * sd)
            if int(ok2.sum()) < 4:
                break
            c = np.polyfit(ctr[ok2], v[ok2], 1)
            ok = ok2
        slope = float(c[0])
        resid = v[ok] - np.polyval(c, ctr[ok])
        spread = float(np.nanmax(v[ok]) - np.nanmin(v[ok])) if int(ok.sum()) else 0.0
        # Only rotate when a straight line across the width genuinely explains the spread, and when there is
        # a spread worth removing. Otherwise this fits per-frame estimator noise into a spurious rotation —
        # the failure mode that sank the full 3-DOF fit (every fitted lateral shift came out positive).
        across = abs(slope) * (L - 1)
        if across < max(float(p.get("rfr_edge_tilt_min_px", 4.0)), thr_null):
            continue
        # A straight line must genuinely explain it. This is what separates a real rotation from a line
        # fitted through noisy band medians: on cs044_os_v1_3 the two real ones have residuals 0.93 and 1.28
        # against spreads of 9.8 and 8.1, while the marginal frames sit at 1.9-3.5.
        if float(np.sqrt(np.mean(resid ** 2))) > float(p.get("rfr_edge_tilt_fit", 0.35)) * spread:
            continue
        tilt[f] = float(np.clip(slope, -max_slope, max_slope))

    # ALTERNATION TEST, replacing a blanket low-pass. Low-passing the sequence does cancel the noise pattern
    # it was aimed at, but it also cancels an ISOLATED genuine rotation — cs044_os_v1_3's two real frames are
    # single-frame and a gaussian would have attenuated 11.2 px to ~3 and dropped them. What actually
    # distinguishes noise is that it ALTERNATES against comparably-sized neighbours, so test for that
    # directly and leave an isolated, well-explained, above-null rotation alone.
    for f in range(F):
        if abs(tilt[f]) <= 1e-9:
            continue
        nb = [tilt[g] for g in (f - 1, f + 1) if 0 <= g < F and _in_edge_block(g, F, p)]
        opp = [x for x in nb if x * tilt[f] < 0 and abs(x) > 0.5 * abs(tilt[f])]
        if len(opp) >= 1 and len(nb) >= 2 and all(x * tilt[f] < 0 for x in nb):
            tilt[f] = 0.0
    return tilt


def _rfr_plan(dev: np.ndarray, edev: np.ndarray, p: dict) -> np.ndarray:
    """The per-frame RIGID depth shift to apply: interior frames levelled by their own deviation from the
    per-lateral corneal curve, edge frames by their deviation from the anchored local corneal curvature."""
    F = int(dev.size)
    lead, tail = _rfr_split(p)
    min_dev, clamp = float(p.get("rfr_min_dev", 1.0)), float(p.get("rfr_max_shift", 45.0))
    a = np.zeros(F, dtype=np.float64)
    for f in range(lead, F - tail):
        if np.isfinite(dev[f]) and abs(dev[f]) > min_dev:
            a[f] = float(np.clip(dev[f], -clamp, clamp))
    # An accepted side is applied WHOLE — including its sub-threshold frames near the seam. Thresholding
    # frame-by-frame inside an accepted block is what would reintroduce a step between a moved frame and its
    # unmoved neighbour; the side-level gates above already decided whether this block should move at all.
    for f in range(F):
        if np.isfinite(edev[f]):
            a[f] = float(np.clip(edev[f], -clamp, clamp))
    return a


def _rfr_interior_plan(dev: np.ndarray, p: dict) -> np.ndarray:
    """The interior-only shift, needed BEFORE the edge reference can be anchored to a corrected seam."""
    return _rfr_plan(dev, np.full(dev.size, np.nan), p)


def rigid_frame_refine(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """FINAL residual rigid per-frame correction driven by the ANTERIOR BOUNDARY (v0.0.217).

    `axial_motion_correct` -> `rigid_frame_warp` -> `rigid_height_refine` -> `rigid_frame_derotate` already
    align the frames, but all four are driven by the DP-detected surface and by a smooth 3-D dome, and
    `rigid_height_refine` additionally keeps only the HIGH-FREQUENCY part of its correction (gaussian
    sigma=`rhr_smooth`, capped at `rhr_max`=8 px) so that the real dome trajectory survives. That is the right
    call in the interior, and it is exactly why a LOW-frequency, LARGE offset at the acquisition-EDGE frames
    survives every one of them. Ten scans reviewed one-by-one with the user found that same defect over and
    over — "the left-most edge of the sagittal sections doesn't follow the overall corneal curvature, overtly
    steep" (CS018_OD_3), the same on CS023_OS_v2, both edges on CS041_OS_v1 at up to 40 px — plus smaller
    interior bumps (CS002_OS_3 at sagittal slice 182).

    Each frame is moved RIGIDLY in depth by one displacement for the whole B-scan, so every A-scan shifts
    together and nothing inside a B-scan is deformed. A B-scan is captured instantaneously; its internal
    geometry is truth.

    WHY DEPTH-TRANSLATION ONLY. The full in-plane rigid group is {lateral translation, depth translation,
    rotation}; the pivot adds no freedom (a rotation about any point equals one about the centre plus a
    translation) and the order is irrelevant. That complete 3-DOF fit was implemented and measured: it
    explains 11% of the residual against 8% for depth-only, but APPLYING it made things WORSE — at this
    residual level the extra parameters are bias-dominated (100% of fitted lateral shifts came out positive,
    median +0.50 px, an estimator artefact rather than motion) and the boundary rms rose from 0.49 to 0.84 px.
    The rotation term is already covered, self-gated, by `rigid_frame_derotate`.

    WHAT IT CANNOT DO, by design: the residual left after this varies from one sagittal slice to the next
    (across the width a tilt explains 0-17% of it, a quartic only 13%), so no rigid per-frame transform can
    reach it. Removing it would mean deforming the B-scan, which is forbidden.

    Guarded four ways, each from an observed failure: dropout frames are never moved; an implausible
    measurement refuses the whole pass rather than writing garbage; the shift is clamped; and the boundary
    deviation is RE-MEASURED after the move, with the result discarded unless it actually improved. A scan
    that arrives smooth is returned byte-unchanged. `volume` is (frames, depth, lateral)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("rigid_frame_refine", True)):
        return volume, {"applied": False}
    F, L = int(volume.shape[0]), int(volume.shape[2])
    lead, tail = _rfr_split(p)
    if F < lead + tail + 24 or L < 32:
        return volume, {"applied": False, "reason": "too few frames"}
    k = slice(lead, F - tail)

    def _metrics(S_in, want_detail=False):
        """Every quantity the pass plans from and is judged on, computed from a boundary map.

        Called ONCE on the detected surface and then on the ANALYTICALLY PREDICTED surface, never by
        re-detecting the warped voxels — see `_rfr_predict_surface` for why that matters (the composite gate
        discarded a provably harmless rigid translation on 54% of approved scans, purely from sub-pixel
        interpolation moving the half-max crossing).

          rms        interior boundary deviation; catches per-frame NOISE (detected on 100% of scans)
          seam_step  the jump across each seam; catches a BLOCK OFFSET (100%)
          edge/int_spread  worst across-width spread; edge_spread is the only one of these that catches a
                     ROTATION (100%), which rms and int_spread miss entirely
          far_dev    TELEMETRY ONLY, no longer a gate — see the acceptance block
        """
        dv, pr, Cc = _rfr_from_surface(S_in, p)
        det = {} if want_detail else None
        ai = _rfr_interior_plan(dv, p)
        ed = _rfr_edge_deviation(pr, p, a_int=ai, detail=det, dev=dv)
        with np.errstate(all="ignore"):
            r = float(np.sqrt(np.nanmean(dv[k] ** 2))) if np.isfinite(dv[k]).any() else float("nan")
            # NaN, not 0.0, when a seam cannot be measured. Initialising to 0.0 and only raising it inside
            # the finite check made an UNMEASURABLE seam score as a PERFECT one: blanking frames 11/12/88/89
            # on cs001_od gives step 0.000 against a 0.541 baseline, so destroying a seam reads as improving
            # it. Same null-vs-zero trap as edge_dev and max_jitter before it.
            step = float("nan")
            for side in ("lead", "trail"):
                w = _rfr_side_windows(F, side, p)
                if w is None:
                    continue
                _f, blk, seam = w
                j = blk[0] if side == "trail" else blk[-1]
                if np.isfinite(pr[j]) and np.isfinite(pr[seam]):
                    # the profile's own local slope at the seam, so a smooth descent is not read as a step
                    nb = np.arange(max(0, seam - 5), min(F, seam + 6))
                    gd = np.isfinite(pr[nb])
                    sl = np.polyfit(nb[gd], pr[nb][gd], 1)[0] if int(gd.sum()) >= 4 else 0.0
                    v = abs(float(pr[j] - pr[seam] - sl * (j - seam)))
                    step = v if not np.isfinite(step) else max(step, v)
            L4 = int(S_in.shape[1])
            band = slice(int(float(p.get("rfr_lat_lo", 0.20)) * L4), int(float(p.get("rfr_lat_hi", 0.80)) * L4))
            en = int(p.get("rfr_edge_n", 12))
            res = np.nanmedian((S_in - Cc)[:, band], axis=1)[np.r_[np.arange(en), np.arange(F - en, F)]]
            far = float(np.sqrt(np.nanmean(res ** 2))) if np.isfinite(res).any() else float("nan")
        return {"dev": dv, "prof": pr, "edev": ed, "a_int": ai, "rms": r, "S": S_in, "C": Cc,
                "seam_step": float(step), "far_dev": far, "detail": det,
                "edge_spread": _rfr_edge_spread(S_in, p),
                "int_spread": _rfr_interior_spread(S_in, Cc, p),
                "bsd": _rfr_banded_second_diff(S_in, p),
                "n_finite": int(np.isfinite(pr).sum())}

    try:
        _S0 = _anterior_boundary(volume, p)
        _S0[_dropout_frames(volume, p)] = np.nan
        m0 = _metrics(_S0, want_detail=True)
    except Exception:  # noqa: BLE001 — a refinement must never break a preprocess run
        return volume, {"applied": False, "reason": "measure failed"}
    dev0, edev0, prof0, rms0 = m0["dev"], m0["edev"], m0["prof"], m0["rms"]
    with np.errstate(all="ignore"):
        emax0 = float(np.nanmax(np.abs(edev0))) if np.isfinite(edev0).any() else float("nan")

    n_big = int(np.sum(np.abs(np.nan_to_num(dev0[k])) > float(p.get("rfr_wild_px", 25.0))))
    # SANITY. Tens of px across many frames is not eye motion, it is the measure failing — typically on a
    # partial-shadow band that dims the tissue without collapsing it, so the dropout test misses it. Refuse.
    if n_big >= int(p.get("rfr_wild_max", 3)) or not np.isfinite(rms0) or rms0 > float(p.get("rfr_wild_rms", 8.0)):
        return volume, {"applied": False, "reason": "measure unreliable",
                        "frames_over_25px": n_big, "dev_rms": None if not np.isfinite(rms0) else round(rms0, 3)}

    a = _rfr_plan(dev0, edev0, p)
    # CANVAS GUARD. A rigid depth shift moves the whole B-scan, so a frame whose boundary already sits at the
    # top of the volume cannot be raised further without pushing tissue OUT of it — the warp zero-fills what
    # it vacates, so the cornea would simply be cut off. On cs035_od_v1 the leading edge is already at depth 0
    # and the local curvature asks for another ~15 px up; without this the pass truncates real tissue to chase
    # a target it physically cannot reach (and the edge deviation does not improve anyway: 15.32 -> 15.44).
    # Reaching that geometry needs the canvas EXTENDED, which is the surface-crop path's job, not this one.
    margin = float(p.get("rfr_canvas_margin", 4.0))
    depth = int(volume.shape[1])
    if not np.any(np.abs(a) > 0.05) and not np.any(np.abs(_rfr_precheck_tilt(m0, p)) > 1e-9):
        return volume, {"applied": False,
                        "reason": "already smooth" if np.isfinite(emax0) else "nothing to correct",
                        "dev_rms": round(rms0, 3),
                        "edge_dev": _r2(emax0), "edge_measurable": bool(np.isfinite(emax0)),
                        "seam_step": round(m0["seam_step"], 2), "far_dev": _r2(m0["far_dev"]),
                        "sides": m0["detail"]}

    # The accepted edge blocks also get a lateral TILT where the defect varies across the B-scan's width.
    # That ramp is a rigid ROTATION (the B-scan's internal distances are preserved), applied exactly as
    # rigid_frame_derotate applies one; the INTERIOR keeps a pure depth shift, because adding free parameters
    # to the interior was measured and is bias-dominated at that residual level.
    try:
        b_tilt = _rfr_edge_tilt(m0["S"], edev0, p)
    except Exception:  # noqa: BLE001
        b_tilt = np.zeros(F)
    try:
        i_tilt, i_info = _rfr_interior_tilt(m0["S"], m0["C"], p)
        b_tilt = np.where(np.abs(b_tilt) > 1e-9, b_tilt, i_tilt)   # edge blocks keep their own fit
    except Exception:  # noqa: BLE001
        i_info = {"frames": 0}
    lat_off = np.arange(L, dtype=np.float64) - (L - 1) / 2.0

    # CANVAS PRECONDITION, checked on the FULL-WIDTH plan INCLUDING the rotation. The previous version
    # clamped against `prof0`, the CENTRAL-BAND MEDIAN, while the constraint — tissue leaving the volume —
    # applies across the whole width; and it ran BEFORE the tilt was added, so a rotation could lift a
    # lateral end out of the canvas with nothing checking. Measured on the approved corpus, cs027_os was
    # applied with no complaint and lost 33 tissue voxels off the top. The leak is small on vetted scans
    # (2 of 47 lose any tissue, under 0.2%) but unbounded on the damaged scans this pass targets — and until
    # now it was masked by far_dev accidentally vetoing those same scans, protection that disappeared when
    # far_dev correctly became telemetry.
    #
    # Only the TRUSTED laterals count: at the extreme columns the boundary estimate is noise (readings 190 px
    # off the arc), and a minimum taken over those would refuse corrections on the strength of garbage.
    _bands = _rfr_lateral_bands(L, p)
    if _bands:
        trusted = np.arange(_bands[0][0], _bands[-1][1])
        Spred = _rfr_predict_surface(m0["S"], a, b_tilt)
        with np.errstate(all="ignore"):
            top = np.nanmin(Spred[:, trusted], axis=1)
            bot = np.nanmax(Spred[:, trusted], axis=1)
        for f in range(F):
            if abs(a[f]) <= 0.05 and abs(b_tilt[f]) <= 1e-9:
                continue
            if np.isfinite(top[f]) and top[f] < margin:
                a[f] -= (margin - float(top[f]))            # lower the frame until it fits
            if np.isfinite(bot[f]) and bot[f] > depth - 1 - margin:
                a[f] += float(bot[f]) - (depth - 1 - margin)
    a[np.abs(a) <= 0.05] = 0.0

    out = volume.copy()
    for f in range(F):
        if abs(a[f]) <= 0.05 and abs(b_tilt[f]) <= 1e-6:
            continue
        d = a[f] + b_tilt[f] * lat_off
        out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), -d, subpixel=True)

    # NON-REGRESSION, scored against references this pass did NOT use. cs041_os_v1 arrived with an already
    # excellent boundary (0.48 px rms) and an earlier centroid-driven pass "improved" its own driving metric
    # while making the boundary 3.5x worse; the version before this one then repeated the mistake at the edge,
    # re-measuring the result with the same quadratic that had produced it.
    try:
        m1 = _metrics(_rfr_predict_surface(m0["S"], a, b_tilt))
    except Exception:  # noqa: BLE001
        m1 = None
    if m1 is None:
        return volume, {"applied": False, "reason": "could not verify the result"}
    rms1, step1, far1 = m1["rms"], m1["seam_step"], m1["far_dev"]
    step0, far0 = m0["seam_step"], m0["far_dev"]
    # TOLERANCES. Each sits above that metric's measured RESAMPLING FLOOR — the drift it shows under a
    # uniform per-frame shift, which cannot change geometry at all — and each metric is kept only for the
    # insult it demonstrably detects. Measured over 14-20 approved scans (floor = control p90 |drift|;
    # power = fraction of scans separating the insult from the control):
    #   rms        floor 0.015  catches per-frame NOISE 100%
    #   seam_step  floor 0.748  catches a BLOCK OFFSET 100%, noise 86%
    #   far_dev    floor 0.114  catches a BLOCK OFFSET 93%, noise 100%
    #   edge_mean  floor 0.742  catches ROTATION 100% — and nothing else here does (rms 0%, int_mean 0%)
    #   int_mean   floor 0.037  catches per-frame NOISE 93%
    # A legitimate smooth dome change (a 10 px ramp across the frame axis) trips none of them (0-14%),
    # so they do not fire on real geometry.
    # TOLERANCES. Scoring is now ANALYTIC, so the resampling floor is float64 round-off (1e-13) rather than
    # the 0.7-3.5 px it was under re-detection. Every tolerance is therefore set by POWER and by the headroom
    # the pass's OWN real corrections need — not by a noise floor. Measured over 63 approved scans (headroom
    # = how far the metric moves up under the pass's genuine plan, n=47):
    #   rms       +0.05  headroom max 0.009  -> 1.5px interior bump 94%, 0.3px per-frame noise 92%
    #   seam_step +0.25  headroom p99 0.507  -> 3px block offset 87%
    #   edge      +0.75  headroom max 0.609  -> 6px alternating rotation 75%, 4px smooth 46%
    #   int       +0.10  headroom max 0.060  -> 6px interior rotation 86%, 12px 100%
    #   bsd       +0.50  headroom max 0.465  -> 6px alternating rotation 97%, silent on a benign dome ramp
    # An earlier draft loosened these to clear the OLD floor, costing 86 points of detection on a 6px
    # interior rotation and 46 on the cs044_os_v1 alternating mode in exchange for nothing.
    step_cap = float(p.get("rfr_seam_step_px", 1.0))
    bad = []
    if not np.isfinite(rms1) or rms1 > rms0 + float(p.get("rfr_gate_rms_px", 0.05)):
        bad.append(f"interior {rms0:.3f}->{rms1:.3f}")
    # A step at the seam is the reviewer's actual complaint, so it is a hard cap rather than a tolerance —
    # and it is capped in ABSOLUTE terms because the pass must not be allowed to introduce one at all.
    if np.isfinite(step0) and not np.isfinite(step1):
        bad.append("seam step became unmeasurable")
    elif np.isfinite(step1) and step1 > max(step_cap, (step0 if np.isfinite(step0) else 0.0)
                                            + float(p.get("rfr_gate_step_px", 0.25))):
        bad.append(f"seam step {step0:.2f}->{step1:.2f}")
    # Losing measurable frames is damage that every nan-aggregate here would otherwise score as improvement.
    if int(m1.get("n_finite", 0)) < int(m0.get("n_finite", 0)):
        bad.append(f"frames became unmeasurable {m0.get('n_finite')}->{m1.get('n_finite')}")
    bsd0, bsd1 = m0.get("bsd", float("nan")), m1.get("bsd", float("nan"))
    if np.isfinite(bsd0) and np.isfinite(bsd1) and bsd1 > bsd0 + float(p.get("rfr_gate_bsd_px", 0.5)):
        bad.append(f"along-frame roughness {bsd0:.2f}->{bsd1:.2f}")
    # far_dev is TELEMETRY, not a gate. It scores the edge frames against `_lateral_frame_curve` extrapolated
    # into them — which is exactly the reference the edge path exists BECAUSE it is invalid there (see
    # `_rfr_edge_deviation`: "a polynomial extrapolates the wrong way"). So it vetoes precisely the
    # corrections that are right: of 47 approved scans, 14 had an edge block accepted and only 3 survived —
    # 10 of the 11 rejections were this term, including cs050_od_v1, whose correction was verified by eye.
    # It is also redundant for detection: rms catches per-frame noise on 100% of scans and seam_step catches
    # a block offset on 100%, which is everything far_dev was contributing.
    # ACROSS-THE-WIDTH. Every other metric here is a central-band median, which is blind by construction to a
    # defect that only shows on the far sagittal slices — the one the reviewer found after the depth-only
    # version shipped. This is the measure that can see it.
    sp0, sp1 = m0.get("edge_spread", float("nan")), m1.get("edge_spread", float("nan"))
    if np.isfinite(sp0) and np.isfinite(sp1) and sp1 > sp0 + float(p.get("rfr_gate_edge_spread_px", 0.75)):
        bad.append(f"across-width edge spread {sp0:.2f}->{sp1:.2f}")
    ip0, ip1 = m0.get("int_spread", float("nan")), m1.get("int_spread", float("nan"))
    if np.isfinite(ip0) and np.isfinite(ip1) and ip1 > ip0 + float(p.get("rfr_gate_int_spread_px", 0.10)):
        bad.append(f"across-width interior spread {ip0:.2f}->{ip1:.2f}")
    if bad:
        return volume, {"applied": False, "reason": "would regress: " + "; ".join(bad),
                        "dev_rms_before": round(rms0, 3), "dev_rms_after": _r3(rms1),
                        "seam_step_before": _r2(step0), "seam_step_after": _r2(step1),
                        "bsd_before": _r2(bsd0), "bsd_after": _r2(bsd1),
                        "far_dev_before": _r2(far0), "far_dev_after": _r2(far1),
                        "edge_spread_before": _r2(sp0), "edge_spread_after": _r2(sp1),
                        "int_spread_before": _r2(ip0), "int_spread_after": _r2(ip1),
                        "interior_tilt": i_info, "sides": m0["detail"]}
    edge_n = int(p.get("rfr_edge_n", 12))
    return out, {"applied": True, "frames_moved": int(np.sum(np.abs(a) > 0.05)),
                 "max_shift": round(float(np.max(np.abs(a))), 2),
                 "edge_shift_max": round(float(np.max(np.abs(np.r_[a[:edge_n], a[-edge_n:]]))), 2),
                 "dev_rms_before": round(rms0, 3), "dev_rms_after": _r3(rms1),
                 "seam_step_before": _r2(step0), "seam_step_after": _r2(step1),
                 "bsd_before": _r2(m0.get("bsd", float("nan"))), "bsd_after": _r2(m1.get("bsd", float("nan"))),
                 "far_dev_before": _r2(far0), "far_dev_after": _r2(far1),
                 "edge_spread_before": _r2(sp0), "edge_spread_after": _r2(sp1),
                 "max_tilt_deg": round(float(np.degrees(np.arctan(
                     np.max(np.abs(b_tilt)) * float(p.get("oct_spacing_depth", 0.0031))
                     / float(p.get("oct_spacing_lateral", 0.0078))))), 2),
                 "frames_tilted": int(np.sum(np.abs(b_tilt) > 1e-6)),
                 "int_spread_before": _r2(ip0), "int_spread_after": _r2(ip1),
                 "interior_tilt": i_info,
                 "sides": m0["detail"],
                 "shift": [round(float(v), 3) for v in a]}


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
    # GATE = SOFT-THRESHOLD: corr = sign(dev)·max(|dev|−gate, 0). Leave a gate-width dead-band (a wobble ≤ gate is
    # genuine dome micro-texture → no-op) and remove the excess beyond it. CRITICAL that this stays SOFT, not a
    # hard "|dev|>gate → pull fully to the dome": a continuity pass must be CONTINUOUS in dev, else two adjacent
    # columns straddling the gate (e.g. 3.1 vs 2.9) would be corrected by 3.1 vs 0 and the pass would CREATE a
    # ~gate step where there was none. Soft caps every residual at ≤ gate and can never introduce a gate-boundary
    # step. A real notch (|dev| ≫ gate) is still pulled to within gate of the dome; combined with the sub-pixel
    # apply + two-sided detection this smoothly flattens both up-spikes and down-notches.
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
    # SUB-PIXEL: keep the FRACTIONAL shift (no np.round) — the apply site warps with subpixel=True, so a notch is
    # pulled smoothly onto the dome instead of snapping to the nearest integer row (which would re-quantise the
    # very axial staircase this pass exists to remove). Matches the sub-pixel main warp.
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
            # SUB-PIXEL (matches the main warp): a fractional notch correction lands the column smoothly on the
            # dome instead of int-truncating it into a fresh 1-px staircase.
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), sh, subpixel=True)
            moved[f] = True
            n_moved_cols += int(np.count_nonzero(sh != 0.0))
            any_move = True
        if not any_move:
            break
    n_moved_frames = int(moved.sum())
    return out, {"applied": bool(n_moved_frames), "frames_adjusted": n_moved_frames,
                 "n_frames": n_frames, "cols_adjusted": int(n_moved_cols)}


def rigid_sagittal_motion_correct(volume: np.ndarray, params: dict | None = None,
                                  workers: int | None = None):
    """Remove inter-frame axial DRIFT that survives the flatten, by a per-frame rigid DEPTH shift — GATED so
    it only fires on genuine drift and never touches a normal scan.

    THE PROBLEM (measured). The corrections-path flatten lands each B-scan on its own across-frame quadratic
    and keeps it, so a slow drift that accumulated into a smooth across-frame bow is preserved as if it were
    corneal curvature. On the complaint scan the across-frame sag was 166 px against an anatomical ~34 —
    ~130 px of drift, shaped like anatomy, which no dome/quadratic fit can tell apart from real shape.

    THE DRIFT-PROOF REFERENCE. Each B-scan is captured instantaneously, so its LATERAL arc is untouched by
    inter-frame motion. A near-spherical cornea has across-frame sag ≈ SAG_RATIO × lateral sag (constant in
    radius). So the lateral sag — which drift cannot corrupt — predicts what the across-frame sag SHOULD be.
    Excess over that prediction is drift.

    THE CORRECTION is exactly the reviewer's spec: one rigid depth shift per B-scan (nothing deformed), moving
    each frame so the per-frame surface trajectory follows a smooth curve scaled to the sphere-predicted sag.
    Because drift is common across laterals (the whole B-scan moved), a single per-frame shift corrects every
    lateral at once, and the lateral arc — hence lateral sag — is invariant under it.

    THE GATE. Fires only when (measured across-frame sag) / (SAG_RATIO × lateral sag) exceeds
    `smc_excess_gate`. Validated: complaint scan 4.9× (fires → 166→34 px, undulation IMPROVED 4.2→0.6, lateral
    held); two approved scans 1.28× and 0.92× (do NOT fire — left byte-identical). A normal cornea's
    variation sits ~0.9–1.3×, so the 2.0 gate has wide margin. Under-correcting a mildly-drifted scan (leaving
    acquired geometry) is the safe failure; distorting a good scan toward a sphere model is not, which is why
    the gate is deliberately conservative. Returns (volume, info); info.applied is False when the gate holds.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("rigid_sagittal_mc", True)):
        return volume, {"applied": False, "reason": "disabled"}
    if workers is None:
        workers = auto_workers()
    ratio = float(p.get("smc_sag_ratio", 0.444))
    gate = float(p.get("smc_excess_gate", 2.0))
    max_shift = float(p.get("smc_max_shift", 140.0))
    try:
        sag = reformat_to_sagittal(volume)                      # (lateral, depth, frames)
        S = detect_surface_all(sag, p, workers=workers).astype(np.float64)   # (lateral, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False, "reason": "detect failed"}
    L, F = S.shape
    if F < 12 or L < 32:
        return volume, {"applied": False, "reason": "too small"}
    lo, hi = int(0.25 * L), int(0.75 * L)
    r = np.array([np.nanmedian(S[lo:hi, f]) for f in range(F)], dtype=np.float64)   # per-frame ref depth
    if not np.isfinite(r).all():
        r = np.interp(np.arange(F), np.where(np.isfinite(r))[0], r[np.isfinite(r)]) \
            if np.isfinite(r).any() else None
        if r is None:
            return volume, {"applied": False, "reason": "no reference"}
    # LATERAL sag — drift-proof, measured within B-scans, median over central frames
    xL = np.arange(L, dtype=np.float64)
    lat = []
    for f in range(F // 3, 2 * F // 3):
        col = S[:, f]
        ok = np.isfinite(col) & (col > 1)
        if ok.sum() >= L * 0.5:
            lat.append(float(np.ptp(ndimage.gaussian_filter1d(np.interp(xL, xL[ok], col[ok]), 10.0, mode="nearest"))))
    lat_sag = float(np.median(lat)) if lat else float("nan")
    if not np.isfinite(lat_sag) or lat_sag < 8:
        return volume, {"applied": False, "reason": "lateral sag unmeasurable"}
    xs = np.arange(F, dtype=np.float64)
    trend = np.poly1d(np.polyfit(xs, r, min(4, F - 1)))(xs)      # smooth across-frame trajectory
    cur_sag = float(np.ptp(trend))
    target_sag = ratio * lat_sag
    excess = cur_sag / max(target_sag, 1e-6)

    # ── DEPTH component (y): sphere de-bow, gated on the across-frame excess ratio ──────────────────────
    depth_shift = np.zeros(F)
    if excess > gate:
        scaled = (trend - trend.mean()) * (target_sag / cur_sag) + trend.mean()
        depth_shift = scaled - r

    # ── TILT component (r): remove the VARIATION of the per-frame lateral tilt ──────────────────────────
    # Each B-scan's arc has a linear-in-lateral tilt m_f. A rigid eye in a fixed pose has a CONSTANT tilt
    # across frames (its static torsion/decentration); any VARIATION is the eye rotating during the scan —
    # motion. The reviewer's "trough on early slices, bump on later slices at the same column" is exactly a
    # tilt that flips sign across frames, which a static cornea cannot produce. So keep the mean tilt (real,
    # small) and remove the variation, as a per-frame rotation (a depth shear linear in lateral; the lateral
    # component is sub-pixel at these angles, so the arc SHAPE — the anatomy — is preserved).
    xc = (L - 1) / 2.0
    xn = np.arange(L, dtype=np.float64) - xc
    tlo, thi = int(0.2 * L), int(0.8 * L)
    tilt = np.full(F, np.nan)
    for f in range(F):
        col = S[:, f]
        ok = np.isfinite(col) & (col > 1)
        ok[:tlo] = False; ok[thi:] = False
        if int(ok.sum()) > 20:
            A = np.vstack([np.ones(int(ok.sum())), xn[ok], xn[ok] ** 2]).T
            try:
                coef = np.linalg.lstsq(A, col[ok], rcond=None)[0]
                tilt[f] = coef[1]                               # depth-px per lateral-px (the tilt slope)
            except Exception:  # noqa: BLE001
                pass
    tilt_gate_px = float(p.get("smc_tilt_gate", 20.0))          # min edge-swing DRIFT (px) to act on
    rot_slope = np.zeros(F)
    tilt_drift_px = 0.0
    if np.isfinite(tilt).sum() >= max(8, F // 3):
        tf = np.interp(xs, xs[np.isfinite(tilt)], tilt[np.isfinite(tilt)])
        # TWO metrics, two jobs — conflating them broke both directions.
        #   GATE on a LOW-ORDER fit: the reviewer's rotation is a SMOOTH monotonic drift, while the per-frame
        #   tilt estimate also carries high-freq WIGGLE from detection noise. Keying the DECISION on total
        #   variation fired on a clean scan (cs028: 0.2 px real drift, ~21 px wiggle) and sheared it by noise
        #   (lateral 218->188). A degree-`smc_tilt_deg` fit sees only the real drift, so cs028 no longer fires.
        #   APPLY a lightly-smoothed tilt: a low-order fit is too rigid to REMOVE the real drift fully (it left
        #   cs042 at 13.8 px residual and dev 1.63 vs 0.62), because genuine motion has a steeper transition
        #   than deg-3 can follow. Only scans that PASS the gate reach this, and those have real drift, so
        #   removing their lightly-smoothed tilt (incl. any micro-wiggle, which is also motion) is correct.
        deg = int(p.get("smc_tilt_deg", 3))
        gate_curve = np.poly1d(np.polyfit(xs, tf, min(deg, F - 1)))(xs)
        tilt_drift_px = float(np.ptp(gate_curve - np.median(gate_curve)) * L)   # GATE metric (noise-robust)
        apply_curve = ndimage.gaussian_filter1d(tf, float(p.get("smc_tilt_smooth", 1.5)), mode="nearest")
        if tilt_drift_px > tilt_gate_px:
            rot_slope = -(apply_curve - np.median(apply_curve))  # remove the full smoothed drift (rigid rot)

    info = {"applied": False, "excess": round(excess, 2), "gate": gate,
            "across_frame_sag": round(cur_sag, 1), "lateral_sag": round(lat_sag, 1),
            "target_sag": round(target_sag, 1),
            "tilt_drift_px": round(tilt_drift_px, 1), "tilt_gate_px": tilt_gate_px}
    depth_on = excess > gate
    # SOLE-SAG: the DEPTH component (coarse inter-frame drift) stays — it is a uniform per-frame shift that cannot
    # make a lateral-specific step — but the TILT component defers to sag_quad_align, which fits the rotation
    # against the sagittal edge with a step guard. This stage's tilt keyed only on its own drift metric and put a
    # 68 px swing on the acquisition-edge frames → a 128 px step at the peripheral laterals.
    tilt_on = tilt_drift_px > tilt_gate_px and not bool(p.get("corrections_sole_sag", False))
    if not depth_on and not tilt_on:
        info["reason"] = "within normal range (no drift, no tilt)"
        return volume, info

    # zero-centre the depth shift (minimal canvas use); clip. Rotation is already zero-mean (drift - mean).
    depth_shift = np.clip(depth_shift - (np.median(depth_shift) if depth_on else 0.0), -max_shift, max_shift)
    out = volume.copy(); nadj = 0
    for f in range(F):
        disp = np.full(L, depth_shift[f]) + rot_slope[f] * xn   # per-lateral: translation + rotation(shear)
        if float(np.max(np.abs(disp))) > 0.05:
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), disp, subpixel=True)
            nadj += 1
    info.update({"applied": bool(nadj), "frames_adjusted": int(nadj),
                 "depth_applied": bool(depth_on), "tilt_applied": bool(tilt_on),
                 "max_shift": round(float(np.max(np.abs(depth_shift))), 1) if depth_on else 0.0,
                 "max_tilt_swing": round(float(np.max(np.abs(rot_slope)) * L), 1) if tilt_on else 0.0,
                 "sag_after_expected": round(float(target_sag), 1)})
    return out, info


def generalize_corrected_surface(surface: np.ndarray, anchors, params: dict | None = None) -> np.ndarray:
    """Reconstruct the CORRECTED-result anterior surface from the reviewer's DRAWN edges (pixel-accurate GT) +
    ACROSS-LATERAL interpolation between them, falling back to the given `surface` (the DP detection) where no
    drawn laterals bracket a point.

    WHY: on a steep limbus descent the DP detector fails — a deeper high-gradient layer out-scores the epithelium
    ~3x and per-frame score-normalisation buries the true anterior, so the DP wanders onto a faint band ~200px off
    (cs020 lat 132/152). The reviewer's drawn edge is right to a pixel. So instead of re-tuning the DP, the reviewer
    draws a SPARSE set of representative laterals and this LINEARLY INTERPOLATES the edge across laterals between
    them (validated leave-one-out ~2-5px on cs020's dense edge frames; an un-drawn in-between lateral lands on the
    real tissue). Beyond the drawn lateral range the interpolated value is TAPERED back to the DP over
    `corrected_generalize_taper` laterals so no lateral step forms; drawn points are pinned EXACTLY. Per frame, so a
    frame drawn on only a couple of laterals reconstructs just their span and everything else keeps the DP.
    surface = (lateral, frames); anchors = {lateral: {frame: depth}} in corrected-output depth (0 = TOP)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    S = np.asarray(surface, dtype=np.float64).copy()
    if S.ndim != 2:
        return np.asarray(surface, dtype=np.float32)
    L, F = S.shape
    anc: dict[int, dict[int, float]] = {}
    for lk, fm in (anchors or {}).items():
        try:
            l = int(lk)
        except (TypeError, ValueError):
            continue
        if not (0 <= l < L) or not isinstance(fm, dict):
            continue
        row: dict[int, float] = {}
        for fk, dv in fm.items():
            try:
                f = int(fk); d = float(dv)
            except (TypeError, ValueError):
                continue
            if 0 <= f < F and math.isfinite(d):
                row[f] = d
        if row:
            anc[l] = row
    if not anc:
        return np.asarray(surface, dtype=np.float32)
    A = np.full((L, F), np.nan)
    for l, row in anc.items():
        for f, d in row.items():
            A[l, f] = d
    taper = max(0, int(p.get("corrected_generalize_taper", 8)))
    lat = np.arange(L)
    for f in range(F):
        dl = np.nonzero(np.isfinite(A[:, f]))[0]
        if dl.size == 0:
            continue
        if dl.size == 1:
            S[dl[0], f] = A[dl[0], f]
            continue
        lo, hi = int(dl[0]), int(dl[-1])
        S[lo:hi + 1, f] = np.interp(lat[lo:hi + 1], dl.astype(np.float64), A[dl, f])
        # taper to the DP just OUTSIDE the drawn lateral range so no step at the boundary
        for edge, step in ((lo, -1), (hi, 1)):
            base = float(S[edge, f])
            for k in range(1, taper + 1):
                li = edge + step * k
                if not (0 <= li < L):
                    break
                w = 1.0 - k / (taper + 1.0)
                S[li, f] = w * base + (1.0 - w) * float(surface[li, f])
    # FRAME-DIRECTION notch clean-up: where a lateral's drawn-frame coverage starts/stops (e.g. a taper lateral is
    # interpolated only at the edge frames + DP elsewhere), the interp↔DP boundary leaves a small across-frame step.
    # A light gaussian along frames blends it; the drawn points are then re-pinned EXACTLY so the reviewer's GT is
    # kept. Only laterals the reconstruction TOUCHED are smoothed → pure-DP laterals are byte-untouched.
    _sm = float(p.get("corrected_generalize_smooth", 2.0) or 0.0)
    if _sm > 0:
        touched = np.any(np.abs(S - np.asarray(surface, dtype=np.float64)) > 1e-6, axis=1)
        if touched.any():
            S[touched] = ndimage.gaussian_filter1d(S[touched], _sm, axis=1, mode="nearest")
    for l, row in anc.items():                              # exact pins (belt-and-suspenders, after the smooth)
        for f, d in row.items():
            S[l, f] = d
    return S.astype(np.float32)


def align_corrected_to_smooth(volume: np.ndarray, params: dict | None = None,
                              workers: int | None = None) -> tuple[np.ndarray, dict]:
    """PROPAGATE the reviewer's TRUSTED CURVES across the volume — the "Smooth to trusted slices" action.

    The reviewer works the before/after view in two passes. First they fix the RAW detection frame-by-frame and
    re-run until the corrected result is roughly right. Then, on the CORRECTED result, they either EDIT the
    surface line on a sagittal slice (their DRAWN curve IS the good curvature they want there) or APPROVE a slice
    (its detected corrected border is already good). Those edited + approved slices (LATERALS) are ground truth,
    and this makes EVERY frame's rigid depth + tilt agree with them — so the whole corrected result follows the
    curvature the reviewer defined.

    The target is the reviewer's OWN curves, NEVER an auto-fit. An earlier version fitted a smooth curve to the
    detection and moved the tissue onto it; measured on real motion scans that DECLINED or worsened, because an
    auto-fit cannot separate motion from real shape and the residual is largely per-frame detection jitter. A
    DRAWN curve removes that ambiguity: it is unambiguous truth, so the align just spreads it.

    Mechanism: detect the current corrected surface S[lateral, frame]. For each trusted lateral build its GOOD
    across-frame curve — an edited slice's DRAWN line (its drags over the detection, lightly smoothed to blend),
    an approved slice's lightly-smoothed detection. Per frame, fit the rigid depth SHIFT + TILT that best moves
    the current surface at the trusted laterals onto their good curves (least-squares over the laterals; a bare
    median when too few / clustered), then warp each B-scan by that per-frame shift+tilt. Only the two rigid DOF
    move; the within-frame arc is untouched. Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    max_shift = float(p.get("csa_max_shift", 80.0))          # px — clip guard on the per-frame depth shift
    nF, depth, L = volume.shape
    try:
        S = detect_surface_all(reformat_to_sagittal(volume), p, workers=workers).astype(np.float64)   # (lat, frames)
        # CURRENT surface = the DETECTED corrected surface (where the tissue actually is). The residual that drives
        # the per-frame rigid move is (drawn - detected) = the correction the reviewer made. An earlier version used
        # the RECONSTRUCTION (drawn edges baked in) to dodge a floating DP edge, but then current == target at every
        # edited lateral → residual ~0 → the move collapsed and was declined, so a real tilt (one end too high, the
        # other too low) never propagated. A ROBUST fit below rejects a floating-detection outlier lateral instead.
        # csa_from_detected=False restores the old reconstruction behaviour.
        _cea = p.get("corrected_edge_anchors")
        if _cea and not bool(p.get("csa_from_detected", True)):
            S = generalize_corrected_surface(S, _cea, p).astype(np.float64)
    except Exception:  # noqa: BLE001 — no surface → nothing to move
        return volume, {"applied": False, "reason": "detect failed"}
    Ls, F = S.shape
    if F < 12 or Ls < 32:
        return volume, {"applied": False, "reason": "too small"}
    xc = (L - 1) / 2.0
    xn = np.arange(L, dtype=np.float64) - xc
    xs = np.arange(F, dtype=np.float64)
    from scipy.signal import savgol_filter

    def _smooth_curve(v):
        # a good across-frame curve is smooth; fill sparse points + drop residual detection wobble.
        ok = np.isfinite(v) & (v > 1) & (v < depth - 1)
        if int(ok.sum()) < max(4, F // 4):
            return None
        vi = np.interp(xs, xs[ok], v[ok])
        w = min(31, F)
        if w % 2 == 0:
            w -= 1
        return savgol_filter(vi, w, 3, mode="nearest") if w >= 5 else vi

    # ── gather the trusted GOOD curves: lateral -> good depth per frame ──────────────────────────────────────
    good: dict[int, np.ndarray] = {}
    n_edit = 0; n_appr = 0
    # EDITED slices: the reviewer's DRAWN corrected line = the detection with their drags applied, blended smooth.
    _edit = p.get("corrected_edge_anchors")
    if isinstance(_edit, dict):
        for lk, fmap in _edit.items():
            try:
                le = int(lk)
            except (TypeError, ValueError):
                continue
            if not (0 <= le < L) or not isinstance(fmap, dict):
                continue
            base = _smooth_curve(S[le, :])
            if base is None:
                continue
            drawn = base.copy()
            for fk, dv in fmap.items():
                try:
                    fi = int(fk); dd = float(dv)
                except (TypeError, ValueError):
                    continue
                if 0 <= fi < F and math.isfinite(dd):
                    drawn[fi] = dd                           # the drag wins exactly where the reviewer drew
            gd = _smooth_curve(drawn)                        # smooth so isolated drags blend into a curve...
            if gd is None:
                continue
            for fk, dv in fmap.items():                      # ...then RE-PIN the exact drawn frames: a single-frame
                try:                                         # edit must NOT be smoothed away (savgol over ~31 frames
                    fi = int(fk); dd = float(dv)             # washes a lone point out → residual 0 → nothing to
                except (TypeError, ValueError):              # propagate, so the whole correction was silently lost).
                    continue
                if 0 <= fi < F and math.isfinite(dd):
                    gd[fi] = dd
            good[le] = gd; n_edit += 1
    # APPROVED slices: the detected corrected border, lightly smoothed (its detection IS the target).
    for v in (p.get("corrected_trusted_laterals") or []):
        try:
            la = int(v)
        except (TypeError, ValueError):
            continue
        if not (0 <= la < L) or la in good:
            continue
        sc = _smooth_curve(S[la, :])
        if sc is not None:
            good[la] = sc; n_appr += 1
    trusted = sorted(k for k, val in good.items() if val is not None)
    if not trusted:
        return volume, {"applied": False, "reason": "no trusted slices", "edited_slices": n_edit, "approved_slices": n_appr}

    # ── per-frame rigid depth-shift + tilt to move the current surface onto the good curves ──────────────────
    # residual per frame = good(drawn) - S(detected) at the trusted laterals. Fit ONE rigid move per frame: depth
    # SHIFT (intercept) + TILT (slope across laterals), ROBUST (drop >2.5σ laterals, refit) so a single floating-
    # detection lateral — whose residual is DETECTOR error not drift — can't drag the whole volume (the failure the
    # reconstruction hack was papering over). A TILT still needs a WIDE spread so a slope fit to clustered edits
    # can't blow up when extrapolated across all laterals; both are hard-clamped below.
    max_tilt = float(p.get("csa_max_tilt", 20.0))            # px — cap the tilt SWING across the width
    min_tlat = int(p.get("csa_tilt_min_laterals", 2))        # 2 WIDE-SPREAD laterals already define a tilt line
    shift = np.full(F, np.nan); tilt = np.zeros(F)
    for f in range(F):
        lats = []; resid = []
        for la in trusted:
            c = S[la, f]; g = good[la][f]
            if np.isfinite(c) and 1 < c < depth - 1 and math.isfinite(g):
                lats.append(float(la)); resid.append(g - c)   # how far to move frame f AT this lateral
        if not resid:
            continue
        rr = np.array(resid); xnl = np.array(lats) - xc
        sh = float(np.median(rr)); ti = 0.0                   # robust depth shift (fallback)
        if rr.size >= min_tlat and (xnl.max() - xnl.min()) >= 0.45 * L:   # wide + populated → a reliable tilt
            m = np.ones(rr.size, bool); cf = None
            for _ in range(3):                                            # robust: drop outlier laterals, refit
                try:
                    cf = np.polyfit(xnl[m], rr[m], 1)                     # [slope(tilt), intercept(shift@centre)]
                except Exception:  # noqa: BLE001
                    cf = None; break
                r2 = rr - np.polyval(cf, xnl); sd = float(np.std(r2[m])) + 1e-6
                nm = np.abs(r2) < 2.5 * sd
                if int(nm.sum()) >= min_tlat and not np.array_equal(nm, m):
                    m = nm
                else:
                    break
            if cf is not None:
                sh = float(cf[1]); ti = float(cf[0])
        shift[f] = sh; tilt[f] = ti
    if not np.isfinite(shift).any():
        return volume, {"applied": False, "reason": "no residuals", "edited_slices": n_edit, "approved_slices": n_appr}
    fs = np.nonzero(np.isfinite(shift))[0]
    shift = np.interp(xs, fs.astype(np.float64), shift[fs])
    tilt = np.interp(xs, fs.astype(np.float64), tilt[fs])
    # SPREAD sparse edits across the corner they belong to: a reviewer who draws ONE point (or a short drag) per
    # slice means "the tilt is here", not "only at this exact frame". Hold each directly-drawn frame's move across
    # ±csa_spread frames (nearest-drawn wins) so a single point fixes the whole corner — otherwise it is a lone
    # spike the frame-smoothing below shrinks to ~40%, under-correcting (the "must I draw every frame?" problem).
    _spread = int(p.get("csa_spread", 6))
    _ef = sorted({int(_fk) for _fm in ((p.get("corrected_edge_anchors") or {}).values())
                  if isinstance(_fm, dict) for _fk in _fm
                  if isinstance(_fk, (int, str)) and str(_fk).lstrip("-").isdigit() and 0 <= int(_fk) < F})
    if _ef and _spread > 0:
        _efa = np.array(_ef)
        _sh0 = shift.copy(); _ti0 = tilt.copy()
        for f in range(F):
            _j = int(np.argmin(np.abs(_efa - f)))
            if abs(_efa[_j] - f) <= _spread:                     # inside a drawn frame's corner → hold its move
                shift[f] = _sh0[_efa[_j]]; tilt[f] = _ti0[_efa[_j]]
    # motion is smooth frame-to-frame; a jagged per-frame shift/tilt is fit noise → lightly smooth both, then clamp.
    shift = np.clip(ndimage.gaussian_filter1d(shift, 1.5, mode="nearest"), -max_shift, max_shift)
    tilt = np.clip(ndimage.gaussian_filter1d(tilt, 1.5, mode="nearest"), -max_tilt / max(1.0, L), max_tilt / max(1.0, L))
    # NEVER-WORSE guard: apply the move only if it brings the drawn anchors CLOSER to the depths the reviewer drew
    # — not "don't move them" (the old guard, which killed every real correction). For each drawn anchor:
    #   pre  = |drawn - detected|                       (how far off the corrected result was)
    #   post = |drawn - (detected + move)| = |resid - move|   (how far off AFTER the rigid move)
    # Fire only if the move reduces the median deviation. A rigid shift+tilt that genuinely captures the reviewer's
    # tilt drops post well below pre; a spurious move (e.g. a lone floating-detection lateral that survived the
    # robust fit) does not, and is declined. No warp/re-detect needed — predicted from the move directly.
    _anchor_improved = False   # did the reviewer's DRAWN anchors get closer? (set below; lets a valid tilt through
                               #   the tilt-blind off-quadratic guard, which a rotation pivoting at centre can't move)
    _anc_edit = p.get("corrected_edge_anchors")
    if isinstance(_anc_edit, dict) and _anc_edit:
        _pre = []; _post = []
        for _lk, _fm in _anc_edit.items():
            try:
                _le = int(_lk)
            except (TypeError, ValueError):
                continue
            if not (0 <= _le < L) or not isinstance(_fm, dict):
                continue
            for _fk, _dv in _fm.items():
                try:
                    _fi = int(_fk); _dd = float(_dv)
                except (TypeError, ValueError):
                    continue
                if not (0 <= _fi < F) or not (np.isfinite(S[_le, _fi]) and math.isfinite(_dd)):
                    continue
                _r = _dd - float(S[_le, _fi])                          # drawn - detected (residual to fix)
                _mv = float(shift[_fi] + tilt[_fi] * xn[_le])          # the rigid move at this anchor
                _pre.append(abs(_r)); _post.append(abs(_r - _mv))
        _pm = float(np.median(_pre)) if _pre else 0.0
        _qm = float(np.median(_post)) if _post else 0.0
        if _pre and _qm >= _pm - float(p.get("csa_min_improve", 0.3)):   # must reduce deviation to fire
            return volume, {"applied": False, "trusted_slices": len(trusted), "edited_slices": n_edit,
                            "approved_slices": n_appr, "dev_before": round(_pm, 1), "dev_after": round(_qm, 1),
                            "reason": "declined — rigid move does not reduce deviation from the drawn line"}
        _anchor_improved = bool(_pre)   # passed the anchor never-worse check → the drawn edges got closer
    out = volume.copy(); nadj = 0
    for f in range(F):
        disp = shift[f] + tilt[f] * xn                       # per-lateral: translation + rotation(shear)
        if float(np.max(np.abs(disp))) > 0.05:
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), disp, subpixel=True)
            nadj += 1
    # NEVER-WORSE guard on the SAGITTAL edge — the metric the reviewer actually judges. Re-detect the result and
    # keep it ONLY if the across-frame surface got MORE quadratic; otherwise hand the input back unchanged.
    lo, hi = int(0.2 * L), int(0.8 * L)
    def _offquad(SS):
        ag = np.array([np.nanmedian(SS[lo:hi, ff]) for ff in range(F)])
        okm = np.isfinite(ag)
        if int(okm.sum()) < F * 0.5:
            return None
        ag = np.interp(xs, xs[okm], ag[okm])
        return float(np.sqrt(np.mean((ag - np.poly1d(np.polyfit(xs, ag, 2))(xs)) ** 2)))
    oq0 = _offquad(S)
    try:
        oq1 = _offquad(detect_surface_all(reformat_to_sagittal(out), p, workers=workers))
    except Exception:  # noqa: BLE001
        oq1 = None
    info = {"trusted_slices": len(trusted), "edited_slices": n_edit, "approved_slices": n_appr,
            "off_quad_before": round(oq0, 2) if oq0 is not None else None,
            "off_quad_after": round(oq1, 2) if oq1 is not None else None,
            "max_shift": round(float(np.max(np.abs(shift))), 1),
            "max_tilt_swing": round(float(np.max(np.abs(tilt)) * L), 1)}
    # Decline on the (central-lateral) off-quadratic metric ONLY when the reviewer did NOT improve their drawn
    # anchors — i.e. approved-only smooth-align. A per-frame TILT pivots at the lateral centre, so it barely moves
    # the central-median surface this metric watches; vetoing a rotation that already pulled the drawn edges onto
    # their line (anchor never-worse passed) would kill exactly the correction the reviewer just drew.
    if oq0 is not None and oq1 is not None and oq1 >= oq0 - 0.1 and not _anchor_improved:
        info.update({"applied": False, "reason": "declined — sagittal edge not improved"})
        return volume, info
    info.update({"applied": bool(nadj), "frames_adjusted": int(nadj)})
    return out, info


def axial_motion_correct(volume: np.ndarray, params: dict | None = None, workers: int | None = None,
                         detect: np.ndarray | None = None):
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
        # `detect` lets the caller SHARE an anterior detection it already ran on this same (pre-correction)
        # volume — the codebase's existing reuse idiom (detect= on detect_surface_crop_frames and
        # warp_surface_crop_extend). The surface-crop "geom" rule needs that same RAW-geometry surface, so
        # passing it in keeps the pipeline pass-count neutral instead of adding a third detect pass.
        S = detect if detect is not None else \
            detect_surface_all(reformat_to_sagittal(volume), p, workers=workers)  # (lat, frames)
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
    # `shift` = the per-frame rigid depth shift APPLIED (each B-scan moved by -M[f]). Surface-crop detection
    # runs on this de-tilted volume, where a clipped apex that was pinned at the raw frame TOP now sits at
    # ~M[f]; adding M[f] back recovers the raw-top-relative surface for the "edge within N px of the top"
    # clip criterion. Purely informational (callers only log the summary fields).
    return out, {"applied": bool(nadj), "frames_adjusted": int(nadj),
                 "motion_std": round(float(np.std(M)), 2), "max_shift": round(float(np.max(np.abs(M))), 1),
                 "shift": [round(float(x), 3) for x in M]}


def rigid_height_refine(volume: np.ndarray, params: dict | None = None, workers: int | None = None,
                        detect: np.ndarray | None = None):
    """RIGID per-frame HEIGHT refinement for a smoother sagittal (v0.0.191).

    After the rigid flatten each B-scan sits at ONE depth (a single per-frame height). Any residual ESTIMATION
    JITTER in those heights (AMC + rigid_frame_warp leave a small ±few-px error) leaves the sagittal anterior
    surface slightly WAVY — the user's "the rearranging of the axial 2-D image heights is not perfect". Because
    the B-scan must not deform, the ONLY lever is a better per-frame height. Estimate the leftover per-frame
    error by ITERATIVELY aligning the detected surface to a robust smooth 3-D dome (deg-`amc_dome_deg` 2-D poly,
    re-fit each iteration so the ±jitter doesn't bias it), then keep ONLY the HIGH-FREQUENCY component of the
    accumulated correction (subtract a gaussian-`rhr_smooth` low-pass) so the TRUE smooth dome trajectory is
    preserved — we remove jitter, we do NOT flatten the dome — and apply it as a per-frame RIGID depth shift
    (uniform across lateral → zero B-scan deformation). SELF-GATED: re-detect after the shift and KEEP the
    result only if the per-frame surface roughness actually DROPS, else a strict no-op (never worse). Most of
    the residual sagittal roughness is PER-LATERAL (detection noise / real astigmatism) and is irreducible by
    any rigid shift; this pass only recovers the removable COMMON-motion part. `volume`=(frames,depth,lateral)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    F, depth = int(volume.shape[0]), int(volume.shape[1])
    if F < 12:
        return volume, {"applied": False}
    # `detect` = a surface of THIS volume the caller already has, same reuse idiom as axial_motion_correct.
    # It matters most on the CORRECTIONS path: this stage aligns every frame to a smooth dome fitted through
    # the detected surface, so wherever the detector is wrong the frame is moved to the wrong height — and the
    # frames the reviewer corrected are, by definition, the ones the detector got wrong. Re-detecting here
    # discarded the correction and re-introduced the error it was drawn to fix, which is the "better on some
    # slices, worse on others" the reviewer reported. A caller that has a better surface passes it in.
    _shape_ok = (detect is not None and np.asarray(detect).ndim == 2
                 and int(np.asarray(detect).shape[1]) == F)
    try:
        S = (np.asarray(detect, dtype=np.float32) if _shape_ok
             else detect_surface_all(reformat_to_sagittal(volume), p, workers=workers))  # (lat, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    L = int(S.shape[0])
    valid = (S > 1.0) & np.isfinite(S) & (S < depth - 1)
    if int(valid.sum()) < (L * F) // 4:
        return volume, {"applied": False}

    def _rough(surf):
        sm = np.where((surf > 1) & (surf < depth - 1) & np.isfinite(surf), surf, np.nan)
        with np.errstate(all="ignore"):
            return float(np.nanmean([np.nanmean(np.abs(np.diff(sm[l, :], 2))) for l in range(10, L - 10, 4)]))

    yy, xx = np.mgrid[0:L, 0:F].astype(np.float64)
    yn = yy / (L - 1) * 2 - 1; xn = xx / (F - 1) * 2 - 1
    deg = int(p.get("amc_dome_deg", 5))
    terms = [(a, b) for a in range(deg + 1) for b in range(deg + 1) if a + b <= deg]
    A = np.stack([(yn ** a) * (xn ** b) for a, b in terms], axis=-1)
    Sc = np.where(valid, S, np.nan).astype(np.float64)
    Mcum = np.zeros(F)
    for _ in range(int(p.get("rhr_iters", 4))):
        v = np.isfinite(Sc) & (Sc > 1) & (Sc < depth - 1)
        if int(v.sum()) < len(terms) + 5:
            break
        try:
            coef = np.linalg.lstsq(A[v], Sc[v], rcond=None)[0]
        except Exception:  # noqa: BLE001
            break
        T = A @ coef
        with np.errstate(all="ignore"):
            step = np.nanmedian(np.where(v, Sc - T, np.nan), axis=0)
        step = np.where(np.isfinite(step), step, 0.0)
        Sc = Sc - step[None, :]; Mcum += step
    sig = float(p.get("rhr_smooth", 3.0) or 0.0)
    jitter = (Mcum - ndimage.gaussian_filter1d(Mcum, sig, mode="nearest")) if sig > 0 else Mcum
    jitter = np.clip(jitter, -float(p.get("rhr_max", 8.0)), float(p.get("rhr_max", 8.0)))
    if float(np.max(np.abs(jitter))) < 0.15:       # no meaningful jitter → strict no-op
        return volume, {"applied": False, "max_jitter": round(float(np.max(np.abs(jitter))), 2)}
    r0 = _rough(S)
    out = volume.copy(); nadj = 0
    for f in range(F):
        if abs(jitter[f]) > 0.02:
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), np.full(L, -jitter[f]), subpixel=True)
            nadj += 1
    try:                                            # SELF-GATE: keep only if the surface got smoother
        S2 = detect_surface_all(reformat_to_sagittal(out), p, workers=workers)
        r1 = _rough(S2)
    except Exception:  # noqa: BLE001
        r1 = r0 + 1.0
    if not (r1 < r0):                               # no improvement → revert (never worse)
        # max_jitter is the MEASURED input motion severity, not a result of the correction — so it must be
        # reported even when the correction is reverted. Omitting it here made the field structurally absent
        # on exactly the scans whose refinement failed (the ones most likely to be bad), which capped any
        # jitter-based triage at ~44% recall. The revert still returns the untouched volume.
        return volume, {"applied": False, "max_jitter": round(float(np.max(np.abs(jitter))), 2),
                        "rough_before": round(r0, 3), "rough_after": round(r1, 3)}
    return out, {"applied": bool(nadj), "frames_adjusted": int(nadj), "max_jitter": round(float(np.max(np.abs(jitter))), 2),
                 "rough_before": round(r0, 3), "rough_after": round(r1, 3)}


def sagittal_quad_align(volume: np.ndarray, params: dict | None = None, workers: int | None = None,
                        detect: np.ndarray | None = None):
    """RIGID per-frame depth SHIFT that flattens the SAGITTAL (across-frame) anterior surface to its per-lateral
    QUADRATIC — the reviewer's spec that the corrected result "fit a nice smooth quadratic".

    An axial B-scan is captured instantaneously, so between frames the eye may only TRANSLATE/ROTATE the whole
    scan; in the slow-scan (time) direction a sphere's profile is an arc ~ a deg-2 quadratic, so any deviation
    from that quadratic which is COMMON to all laterals is inter-frame motion, not anatomy. Alternating fit:
    target = each lateral's own deg-2 fit across frames; the per-frame shift = the MEDIAN across laterals of
    (target - surface), so only the shared (motion) component is removed — a shared depth shift cannot touch a
    per-lateral difference, so real per-lateral anatomy (astigmatism, irregularity) is preserved by construction.

    Distinct from rigid_height_refine, which keeps ONLY the high-frequency jitter and PRESERVES the deg-5 dome —
    measured to leave the sagittal surface ~4 px off a quadratic (worse than the input edge's ~3 px); this pass
    targets the quadratic directly (~1.2 px on the same scan). Fits per-frame TRANSLATION + ROTATION jointly (one
    order-free least-squares rigid move per frame); the rotation is a TRUE rigid B-scan rotation about the surface
    pivot — never a per-column shear, which would deform the instantaneous B-scan — hard-clamped to sqa_max_deg (a
    small torsion; an unconstrained tilt overfits catastrophically, measured 884 px swing). Reuses the same rotate
    machinery as rigid_frame_derotate. SELF-GATED: re-detect after the move and keep it only if the aggregate
    off-quadratic actually drops. volume=(frames,depth,lateral)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("sag_quad_align", True)):
        return volume, {"applied": False, "reason": "disabled"}
    if workers is None:
        workers = auto_workers()
    F, depth = int(volume.shape[0]), int(volume.shape[1])
    if F < 12:
        return volume, {"applied": False}
    _shape_ok = (detect is not None and np.asarray(detect).ndim == 2
                 and int(np.asarray(detect).shape[1]) == F)
    try:
        S = (np.asarray(detect, dtype=np.float32) if _shape_ok
             else detect_surface_all(reformat_to_sagittal(volume), p, workers=workers))  # (lat, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    L = int(S.shape[0]); fr = np.arange(F)
    valid0 = (S > 1.0) & np.isfinite(S) & (S < depth - 1)
    if int(valid0.sum()) < (L * F) // 4:
        return volume, {"applied": False}
    # The corrected surface must fit the OVERALL corneal curvature — so the quadratic is fit to the CENTRAL frames
    # and EXTRAPOLATED to the acquisition-edge frames, and the edge is flattened/measured against THAT, not a fit
    # the deep-edited edge itself drags steep (the "edge too steep, doesn't follow the curvature" the reviewer sees
    # once rigid_frame_refine's edge constraint is gone). sqa_edge_pad = frames excluded from the fit at each end.
    _pad = max(0, int(p.get("sqa_edge_pad", 8)))

    def _cquad(e: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """deg-2 fit to the CENTRAL frames, evaluated over the central span; the acquisition-edge frames get a
        LINEAR continuation at the central slope — NOT the parabola extrapolation, which accelerates and leaves the
        edge ~2x too steep (a cornea's limbus flattens, it does not steepen). Value-matched at the fit boundary."""
        okc = ok & (fr >= _pad) & (fr < F - _pad)
        fit_ok = okc if int(okc.sum()) >= 20 else ok
        T = np.polyval(np.polyfit(fr[fit_ok], e[fit_ok], 2), fr)
        if _pad > 0 and int(okc.sum()) >= 20:
            m = float(np.polyfit(fr[fit_ok], e[fit_ok], 1)[0])   # robust central slope
            m *= float(p.get("sqa_edge_slope_frac", 0.3))        # flatten the edge toward horizontal (motion, not
            #   anatomy): 1.0 = follow the central slope, 0.0 = a flat edge. The reviewer wants the residual edge
            #   descent removed, so the target edge is a small fraction of the central slope.
            lo_a, hi_a = _pad, F - _pad - 1                       # fit-boundary frames
            T[:_pad] = T[lo_a] + m * (fr[:_pad] - lo_a)           # leading edge: near-flat continuation
            T[F - _pad:] = T[hi_a] + m * (fr[F - _pad:] - hi_a)   # trailing edge: near-flat continuation
        return T

    _edge_mask = (fr < _pad) | (fr >= F - _pad)      # the acquisition-edge frames
    def _offq(surf: np.ndarray) -> float:
        """Median over CENTRAL laterals of |surface - central-frames quadratic|, but taken as the WORSE of the
        all-frame median and the EDGE-frame median. A plain all-frame median is swamped by the well-fit interior
        and never sees a too-steep edited edge (only ~8 frames), so sag would not fire to pull the edge back onto
        the overall curvature. Central laterals only (0.15-0.85 L): the FOV-edge laterals fade (detection artefact)."""
        vals = []
        for l in range(int(0.15 * L), int(0.85 * L), 3):
            e = surf[l]; ok = (e > 1.0) & np.isfinite(e) & (e < depth - 1)
            if int(ok.sum()) < 20:
                continue
            r = np.abs(e - _cquad(e, ok))
            allm = float(np.nanmedian(r[ok]))
            eok = ok & _edge_mask
            vals.append(max(allm, float(np.nanmedian(r[eok])) if int(eok.sum()) >= 4 else allm))
        return float(np.nanmedian(vals)) if vals else float("nan")

    oq0 = _offq(S)
    if not np.isfinite(oq0):
        return volume, {"applied": False}
    # Fit per-frame TRANSLATION toward the per-lateral quadratic BOTH ways — shift-only and joint shift+ROTATION —
    # then keep whichever fitted surface is more quadratic. Order-free (jointly re-solved each round); translation =
    # MEDIAN residual (robust — an lstsq intercept gets dragged by FOV-edge laterals); rotation = 2σ-trimmed slope,
    # hard-clamped to sqa_max_deg. So the rotation fires on torsion scans and is a strict no-op on the rest (a forced
    # tilt on a torsion-free scan makes it worse — measured 2.25→2.82 on this scan).
    xc = (L - 1) / 2.0
    xnl = (np.arange(L) - xc) / (L / 2.0)                       # normalised lateral, [-1, 1]
    n_iter = max(1, int(p.get("sqa_iters", 8)))
    max_shift = float(p.get("sqa_max_shift", 30.0))
    amax = np.radians(float(p.get("sqa_max_deg", 3.0)))        # pixel-space rotation cap (same convention as derotate)
    tilt_cap = float(L) * amax / 2.0                           # |tilt at xn=1| a <=amax rotation produces
    sig = float(p.get("sqa_smooth", 1.0))

    def _fit(use_tilt: bool):
        shift = np.zeros(F); tilt = np.zeros(F)
        for _ in range(n_iter):
            C = S + shift[None, :] + (tilt[None, :] * xnl[:, None] if use_tilt else 0.0)
            T = np.empty_like(C)
            for l in range(L):
                ok = valid0[l]
                if int(ok.sum()) < 20:
                    T[l] = C[l]; continue
                T[l] = _cquad(C[l], ok)          # central-frames quadratic, extrapolated to the edges
            R = np.where(valid0, T - C, np.nan)
            if use_tilt:
                for f in range(F):
                    ok = valid0[:, f]
                    if int(ok.sum()) < 60:
                        continue
                    rr = R[ok, f]; xx = xnl[ok]; A = np.c_[np.ones(int(ok.sum())), xx]
                    cf = np.linalg.lstsq(A, rr, rcond=None)[0]
                    resid = rr - A @ cf; keep = np.abs(resid) < 2.5 * (np.std(resid) + 1e-6)
                    dt = float(np.linalg.lstsq(A[keep], rr[keep], rcond=None)[0][1]) if int(keep.sum()) > 40 else float(cf[1])
                    shift[f] += float(np.nanmedian(rr - dt * xx)); tilt[f] += dt   # median intercept (robust)
                tilt = np.clip(ndimage.gaussian_filter1d(tilt, sig, mode="nearest"), -tilt_cap, tilt_cap)
            else:
                with np.errstate(all="ignore"):
                    step = np.nanmedian(R, axis=0)
                shift = shift + np.where(np.isfinite(step), step, 0.0)
            shift = ndimage.gaussian_filter1d(shift, sig, mode="nearest")
        shift = np.clip(shift - np.median(shift), -max_shift, max_shift)
        return shift, np.clip(tilt, -tilt_cap, tilt_cap)

    shift_s, _ = _fit(False)                                    # translation only
    shift_j, tilt_j = _fit(True)                               # translation + rotation
    if _offq(S + shift_s[None, :]) <= _offq(S + shift_j[None, :] + tilt_j[None, :] * xnl[:, None]) + 0.05:
        shift, tilt = shift_s, np.zeros(F)                     # rotation did not help → translation only
    else:
        shift, tilt = shift_j, tilt_j
    alpha = np.clip(2.0 * tilt / float(L), -amax, amax)        # per-frame rotation (rad); adds tilt L*alpha/2, as derotate
    if float(np.max(np.abs(shift))) < 0.1 and float(np.degrees(np.max(np.abs(alpha)))) < 0.05:
        return volume, {"applied": False, "off_quad_before": round(oq0, 2), "reason": "no move needed"}
    # APPLY per frame: a TRUE rigid rotation about the surface pivot (never a per-column shear), then a depth shift.
    rc = (L - 1) / 2.0
    scen = np.full(F, (depth - 1) / 2.0)
    for f in range(F):
        col = S[:, f]; mk = (col > 1) & (col < depth - 1) & np.isfinite(col)
        if int(mk.sum()) > 40:
            scen[f] = float(np.median(col[mk]))
    out = volume.copy(); nadj = 0
    for f in range(F):
        moved = False
        img = np.ascontiguousarray(volume[f].T)                 # (lateral, depth)
        if abs(alpha[f]) > 1e-4:
            ct, st = np.cos(alpha[f]), np.sin(alpha[f])
            Mrot = np.array([[ct, st], [-st, ct]]); piv = np.array([rc, scen[f]]); offr = piv - Mrot @ piv
            img = ndimage.affine_transform(img, Mrot, offset=offr, order=1, mode="constant", cval=0.0)
            moved = True
        frame_dl = np.ascontiguousarray(img.T)                  # back to (depth, lateral)
        if abs(shift[f]) > 0.05:
            frame_dl = _warp_by_displacement(frame_dl, np.full(L, shift[f]), subpixel=True)
            moved = True
        if moved:
            out[f] = frame_dl; nadj += 1
    if nadj == 0:
        return volume, {"applied": False, "off_quad_before": round(oq0, 2), "reason": "no move needed"}
    # SELF-GATE: re-detect the delivered surface and keep the move only if the off-quad really dropped (never worse).
    try:
        S2 = detect_surface_all(reformat_to_sagittal(out), p, workers=workers)
        oq1 = _offq(S2)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False, "reason": "redetect failed"}
    # SAGITTAL-EDGE STEP GUARD: a rigid move must never introduce a gross LOCALISED across-frame step (a per-lateral
    # frame-to-frame jump), even at the peripheral laterals the central off-quad metric under-weights.
    def _mstep(SS):
        with np.errstate(all="ignore"):
            per_lat = np.nanmax(np.abs(np.diff(np.where((SS > 1) & np.isfinite(SS), SS, np.nan), axis=1)), axis=1)
        per_lat = per_lat[np.isfinite(per_lat)]
        return float(np.percentile(per_lat, 98)) if per_lat.size else 0.0
    st0, st1 = _mstep(S), _mstep(S2)
    _step_abs = float(p.get("sqa_step_abs", 25.0))
    # Only reject a step this move INTRODUCED: st0 must have been small (a pre-existing large jump is a detection
    # artefact at a weak corner, not something the rigid move made — and it must not veto a legitimate flatten).
    if st0 < _step_abs and st1 > st0 + float(p.get("sqa_step_tol", 12.0)) and st1 > _step_abs:
        return volume, {"applied": False, "off_quad_before": round(oq0, 2),
                        "step_before": round(st0, 1), "step_after": round(st1, 1), "reason": "introduced sagittal step"}
    if not (np.isfinite(oq1) and oq1 < oq0 - float(p.get("sqa_min_gain", 0.3))):
        return volume, {"applied": False, "off_quad_before": round(oq0, 2),
                        "off_quad_after": round(float(oq1), 2), "reason": "no off-quad gain"}
    return out, {"applied": True, "frames_adjusted": int(nadj),
                 "off_quad_before": round(oq0, 2), "off_quad_after": round(float(oq1), 2),
                 "max_shift": round(float(np.max(np.abs(shift))), 1),
                 "max_deg": round(float(np.degrees(np.max(np.abs(alpha)))), 2),
                 "step_before": round(st0, 1), "step_after": round(st1, 1)}


def edge_frame_guard(volume: np.ndarray, params: dict | None = None, workers: int | None = None,
                     detect: np.ndarray | None = None):
    """RIGID per-frame depth SHIFT on ONLY the outermost K acquisition-edge frames, so the faint FOV-corner frames
    CONTINUE the interior corneal curvature instead of dipping below it (the reviewer's "first two columns of the
    left edge slightly don't follow the corneal curvature").

    At the extreme frames the anterior signal is weak and AMBIGUOUS: the DP detector latches onto a shallow false
    edge (reads the corner as LIFTED) while the true tissue sits DEEPER, dipping ~10-35 px below the smooth dome
    (measured on cs020 peripheral slices; the de-jitter passes amplify a ~7px raw dip to ~36px). So this stage does
    NOT trust the DP surface at the edge. Per lateral it fits the RELIABLE interior surface (frames [pad, F-pad])
    with a deg-`edge_guard_deg` polynomial and extrapolates to the outer frames = the target the edge SHOULD sit
    on; it then measures the ACTUAL tissue anterior at the outer frames by a windowed dark->bright gradient searched
    around that target, and shifts each outer frame by the MEDIAN across central laterals of (target - tissue). The
    shift is TAPERED over `edge_guard_taper` frames (no sagittal step introduced — every touched frame lands on the
    same smooth dome), RIGID (one depth shift per frame -> per-lateral anatomy preserved), and SELF-GATED on the
    robust tissue off-dome, kept only if |off-dome| at the outer frames drops. Corrections-path stage (weak FOV
    corners are where the provided-edges de-jitter is least reliable). volume=(frames,depth,lateral)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("edge_guard", True)):
        return volume, {"applied": False, "reason": "disabled"}
    if workers is None:
        workers = auto_workers()
    F, depth = int(volume.shape[0]), int(volume.shape[1])
    K = max(1, int(p.get("edge_guard_frames", 2)))
    taper = max(1, int(p.get("edge_guard_taper", 3)))
    deg = int(p.get("edge_guard_deg", 4))
    pad = K + taper
    if F < 3 * pad + 6:
        return volume, {"applied": False}
    _shape_ok = (detect is not None and np.asarray(detect).ndim == 2
                 and int(np.asarray(detect).shape[1]) == F)
    try:
        sag = reformat_to_sagittal(volume)                          # (lat, depth, frames)
        S = (np.asarray(detect, dtype=np.float32) if _shape_ok
             else detect_surface_all(sag, p, workers=workers))      # (lat, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    L = int(S.shape[0]); fr = np.arange(F)
    valid = (S > 1.0) & np.isfinite(S) & (S < depth - 1)
    lo_c, hi_c = int(0.15 * L), int(0.85 * L)
    outer = (fr < K) | (fr >= F - K)
    win = (fr < pad) | (fr >= F - pad)
    # taper weight: 1.0 on the outermost K frames, ramping to 0 across the next `taper` frames, 0 in the interior.
    w = np.zeros(F)
    for f in range(F):
        d = min(f, F - 1 - f)
        w[f] = 1.0 if d < K else ((pad - d) / float(taper) if d < pad else 0.0)

    def _tissue_ant(img: np.ndarray, target: np.ndarray) -> np.ndarray:
        """windowed dark->bright anterior for one sagittal slice (depth,frames) near `target`, at the win frames."""
        out = np.full(F, np.nan)
        for f in np.where(win)[0]:
            t = target[f]
            if not np.isfinite(t):
                continue
            lo = int(max(0, t - 45)); hi = int(min(depth - 1, t + 55))
            if hi - lo < 8:
                continue
            c = ndimage.gaussian_filter1d(img[:, f].astype(np.float32), 2.5)
            out[f] = lo + int(np.argmax(np.gradient(c[lo:hi])))
        return out

    def _resid_field(SS: np.ndarray, vol_sag: np.ndarray, lat_lo: int = lo_c, lat_hi: int = hi_c):
        """per-lateral (interior-extrapolated dome) - (measured tissue anterior), at the win frames."""
        vv = (SS > 1.0) & np.isfinite(SS) & (SS < depth - 1)
        R = np.full((L, F), np.nan)
        for l in range(lat_lo, lat_hi):
            ok = vv[l] & (fr >= pad) & (fr < F - pad)
            if int(ok.sum()) < max(24, deg + 4):
                continue
            T = np.polyval(np.polyfit(fr[ok], SS[l][ok], deg), fr)
            ta = _tissue_ant(vol_sag[l], T)
            R[l] = np.where(win & np.isfinite(ta), T - ta, np.nan)
        return R

    def _offdome(R: np.ndarray) -> float:
        """90th-percentile over ALL laterals of |dome - tissue| at the OUTER frames (0 = tissue on the dome). Full
        lateral range + a high percentile so a rigid shift that fixes the centre but OVER-LIFTS the peripheral
        corner (the cs020 slice-472 flattening) is scored as WORSE and the self-gate declines it."""
        vals = []
        for l in range(L):
            oo = outer & np.isfinite(R[l])
            if int(oo.sum()) >= 1:
                vals.append(float(np.nanmedian(np.abs(R[l][oo]))))
        return float(np.nanpercentile(vals, 90)) if vals else float("nan")

    # shift estimate: common per-frame residual over the CENTRAL laterals (the majority); the self-gate below then
    # judges it over the FULL lateral range so a peripheral over-lift vetoes it.
    Rc = _resid_field(S, sag)
    with np.errstate(all="ignore"):
        step = np.nanmedian(Rc, axis=0)
    step = np.where(np.isfinite(step), step, 0.0)
    cap = float(p.get("edge_guard_max_shift", 40.0))
    shift = np.clip(step * w, -cap, cap)
    if float(np.max(np.abs(shift))) < 0.5:
        return volume, {"applied": False, "reason": "no move needed"}
    od0 = _offdome(_resid_field(S, sag, 0, L))          # FULL lateral range for the gate
    out = volume.copy(); nadj = 0
    for f in np.where(np.abs(shift) >= 0.3)[0]:
        out[f] = _warp_by_displacement(np.ascontiguousarray(volume[f]), np.full(L, shift[f]), subpixel=True)
        nadj += 1
    if nadj == 0:
        return volume, {"applied": False}
    try:
        sag2 = reformat_to_sagittal(out)
        S2 = detect_surface_all(sag2, p, workers=workers)
        od1 = _offdome(_resid_field(S2, sag2, 0, L))     # FULL lateral range for the gate
    except Exception:  # noqa: BLE001
        return volume, {"applied": False, "reason": "redetect failed"}
    if not (np.isfinite(od1) and np.isfinite(od0) and abs(od1) < abs(od0) - float(p.get("edge_guard_min_gain", 2.0))):
        return volume, {"applied": False,
                        "off_dome_before": round(float(od0), 2) if np.isfinite(od0) else None,
                        "off_dome_after": round(float(od1), 2) if np.isfinite(od1) else None,
                        "reason": "no edge gain"}
    return out, {"applied": True, "frames_adjusted": int(nadj),
                 "off_dome_before": round(float(od0), 2), "off_dome_after": round(float(od1), 2),
                 "max_shift": round(float(np.max(np.abs(shift))), 1)}


def reconcile_manual_line(volume: np.ndarray, provided_edges: np.ndarray, params: dict | None = None,
                          workers: int | None = None):
    """De-jitter the interior, but PIN the surface back to the user's interpolated manual line (provided_edges)
    where the de-jitter's per-frame correction is UNRELIABLE (the reviewer's "for areas where the warp has high
    variance, pin to the interpolated manual line").

    The de-jitter passes (rigid_height_refine/derotate/refine) re-detect the surface and shift each frame onto a
    smooth dome. At the faint FOV-corner frames the DETECTION is weak, so they SMOOTHLY lift the surface off the
    near-perfect drawn line (measured: up to ~39px at frame 100 on cs020) — not a spike, so smoothing can't catch
    it. The reliability signal is the anterior EDGE STRENGTH (tissue-minus-background contrast at the surface):
    high in the interior, collapsing at the corner. Per frame:
      D[f]   = median over central laterals of (dejittered_surface - provided_edges)   # the de-jitter's off-line move
      w[f]   = edge-strength reliability in [0,1] (per-scan adaptive: 0.2..0.7 of the median contrast)
      shift  = (w[f]-1) * D[f]     # where reliable keep the de-jitter (shift 0); where unreliable UNDO it -> pin to line
    RIGID (one depth shift per frame -> anatomy preserved); the reliability tapers smoothly so no sagittal step is
    introduced. SELF-GATED: keep only if the surface moves toward the drawn line at the low-reliability frames while
    the interior roughness does not rise. Corrections path only (needs provided_edges). volume=(frames,depth,lateral)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not bool(p.get("reconcile_line", True)) or provided_edges is None:
        return volume, {"applied": False, "reason": "disabled or no line"}
    if workers is None:
        workers = auto_workers()
    F, depth = int(volume.shape[0]), int(volume.shape[1])
    if F < 24:
        return volume, {"applied": False}
    try:
        sag = reformat_to_sagittal(volume)                       # (lat, depth, frames)
        S = detect_surface_all(sag, p, workers=workers).astype(float)   # (lat, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    PE = np.asarray(provided_edges, dtype=float)
    if PE.shape != S.shape:
        return volume, {"applied": False, "reason": "shape"}
    L = int(S.shape[0]); fr = np.arange(F)
    lo_c, hi_c = int(0.15 * L), int(0.85 * L)
    _hw = int(p.get("reconcile_contrast_hw", 8))                 # half-window (px) above/below the surface
    # per-frame anterior edge strength = tissue(below) - background(above), median over central laterals
    C = np.full((L, F), np.nan)
    for l in range(lo_c, hi_c):
        img = sag[l]; srow = S[l]
        for f in range(F):
            s = srow[f]
            if not np.isfinite(s):
                continue
            si = int(s)
            if si - _hw < 0 or si + _hw >= depth:
                continue
            C[l, f] = float(np.mean(img[si + 2:si + _hw, f]) - np.mean(img[si - _hw:si - 2, f]))
    with np.errstate(all="ignore"):
        rel = np.nanmedian(C[lo_c:hi_c], axis=0)
    if not np.isfinite(rel).any():
        return volume, {"applied": False, "reason": "no contrast"}
    rel_ref = float(np.nanmedian(rel))                          # per-scan adaptive reliability scale
    if rel_ref <= 1e-3:
        return volume, {"applied": False, "reason": "flat contrast"}
    lo_r = float(p.get("reconcile_rel_lo", 0.2)) * rel_ref
    hi_r = float(p.get("reconcile_rel_hi", 0.7)) * rel_ref
    w = np.clip((rel - lo_r) / (hi_r - lo_r + 1e-6), 0.0, 1.0)
    w = np.where(np.isfinite(w), w, 1.0)
    w = ndimage.gaussian_filter1d(w, float(p.get("reconcile_w_smooth", 1.5)), mode="nearest")
    # de-jitter's per-frame off-line move (common component over central laterals)
    D = np.zeros(F)
    for f in range(F):
        r = (S[lo_c:hi_c, f] - PE[lo_c:hi_c, f])
        r = r[np.isfinite(r)]
        if r.size > 20:
            D[f] = float(np.median(r))
    cap = float(p.get("reconcile_max_shift", 60.0))
    shift = np.clip((w - 1.0) * D, -cap, cap)                   # undo the de-jitter where unreliable
    if float(np.max(np.abs(shift))) < 0.5:
        return volume, {"applied": False, "reason": "no move needed"}
    # metrics BEFORE: off-line at the low-reliability (w<0.5) frames + interior roughness
    unrel = w < 0.5
    def _offline(SS):
        if not unrel.any():
            return 0.0
        vals = [abs(float(np.nanmedian((SS[lo_c:hi_c, f] - PE[lo_c:hi_c, f])[
            np.isfinite(SS[lo_c:hi_c, f] - PE[lo_c:hi_c, f])]))) for f in np.where(unrel)[0]]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if vals else 0.0
    def _rough(SS):
        v = []
        for l in range(int(0.2 * L), int(0.8 * L), 3):
            e = SS[l]; ok = np.isfinite(e) & (e > 1)
            if int(ok.sum()) < 80:
                continue
            v.append(float(np.sqrt(np.nanmean(np.diff(e[ok], 2) ** 2))))
        return float(np.nanmedian(v)) if v else float("nan")
    off0, rgh0 = _offline(S), _rough(S)
    out = volume.copy(); nadj = 0
    for f in np.where(np.abs(shift) >= 0.3)[0]:
        out[f] = _warp_by_displacement(np.ascontiguousarray(volume[f]), np.full(L, shift[f]), subpixel=True)
        nadj += 1
    if nadj == 0:
        return volume, {"applied": False}
    try:
        S2 = detect_surface_all(reformat_to_sagittal(out), p, workers=workers).astype(float)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False, "reason": "redetect failed"}
    off1, rgh1 = _offline(S2), _rough(S2)
    # keep only if it pulls the unreliable frames toward the line AND does not worsen interior roughness
    if not (off1 < off0 - float(p.get("reconcile_min_gain", 2.0))
            and np.isfinite(rgh1) and rgh1 <= rgh0 + float(p.get("reconcile_rough_tol", 0.25))):
        return volume, {"applied": False, "off_line_before": round(off0, 1), "off_line_after": round(off1, 1),
                        "rough_before": round(rgh0, 2), "rough_after": round(rgh1, 2), "reason": "no net gain"}
    return out, {"applied": True, "frames_adjusted": int(nadj), "off_line_before": round(off0, 1),
                 "off_line_after": round(off1, 1), "rough_before": round(rgh0, 2), "rough_after": round(rgh1, 2),
                 "max_shift": round(float(np.max(np.abs(shift))), 1), "unreliable_frames": int(unrel.sum())}


def rigid_frame_derotate(volume: np.ndarray, params: dict | None = None, workers: int | None = None):
    """Per-frame rigid ROTATION correction — the SECONDARY inter-frame motion component (v0.0.192).

    Between two axial B-scans (~40 ms) the eye's motion is not purely IN/OUT (depth) — there is also a small
    TORSION / in-plane ROTATION. In the B-scan (lateral×depth) plane that rotation appears as a TILT of the
    anterior surface across the lateral axis; in the sagittal view it is a residual that varies LINEARLY across
    lateral (which a single per-frame depth shift — rigid_height_refine — cannot remove, and which a per-COLUMN
    correction must NOT touch because that would deform the instantaneous B-scan). A true rigid ROTATION of the
    whole B-scan removes it WITHOUT deforming it (rotation preserves every internal distance/angle). Biggest in
    the first/last few frames (the eye settling/drifting at scan start/end), matching where the residual lives.

    ROBUST REFERENCE + CLOSED LOOP (v0.0.198). Per frame measure the lateral tilt b (linear coeff of a per-frame
    quadratic a·xn²+b·xn+c, independent of the dome curvature a) and level it to a SMOOTH-across-frames baseline
    b_ref (median-prefilter + gaussian σ=`rfd_ref_sigma`), NOT a single global DC=median(b). The DC reference is
    not robust — it assumes the real corneal tilt is one constant on every frame (it varies smoothly with slow-scan
    position), so it drags legitimately-different frames to the median AND under-removes localised torsion (measured:
    the old single-shot removed only ~¼ of the marked tilt swing). b_ref keeps the slow real decentration/astigmatism
    trend (>>σ, consistent across replicates) and marks the fast per-frame tilt swings (torsion motion) as b−b_ref.
    The rotation is CLOSED-LOOP: rotate → re-detect → re-measure → repeat (`rfd_iters`) so each frame's tilt actually
    REACHES b_ref (the open-loop formula stopped ¼ of the way). The net per-frame angle is applied ONCE to the
    original B-scan (rotate the (lateral,depth) image; swept-out regions BLACK cval=0, NOT edge-replicated — honest
    black beats invented edge data). SELF-GATED on a nearest-filled iterate: keep only if surface roughness drops.
    Δx (lateral translation) is NOT estimated — collinear with the tilt on the dome flanks, destabilises the fit
    (tested: worse). Runs AFTER rigid_height_refine (the "initial axial alignment"). volume=(frames,depth,lateral)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    F, depth = int(volume.shape[0]), int(volume.shape[1])
    if F < 12:
        return volume, {"applied": False}
    try:
        S = detect_surface_all(reformat_to_sagittal(volume), p, workers=workers)  # (lat, frames)
    except Exception:  # noqa: BLE001
        return volume, {"applied": False}
    L = int(S.shape[0])
    valid = (S > 1.0) & np.isfinite(S) & (S < depth - 1)
    if int(valid.sum()) < (L * F) // 4:
        return volume, {"applied": False}

    def _rough(surf):
        sm = np.where((surf > 1) & (surf < depth - 1) & np.isfinite(surf), surf, np.nan)
        with np.errstate(all="ignore"):
            return float(np.nanmean([np.nanmean(np.abs(np.diff(sm[l, :], 2))) for l in range(10, L - 10, 4)]))

    # PER-FRAME lateral tilt from a per-frame quadratic fit a·xn²+b·xn+c → b = tilt (independent of the dome
    # curvature a). ROBUST REFERENCE (v0.0.198): level each frame's tilt to a SMOOTH-across-frames baseline b_ref
    # (median-prefilter + gaussian σ=rfd_ref_sigma), NOT a single global DC=median(b). A single DC assumes the real
    # corneal tilt is one constant on every frame — it isn't (it varies smoothly with the slow-scan position), so a
    # DC reference drags legitimately-different frames to the median AND leaves localised torsion only partly
    # removed (worked on centred scans by luck). b_ref keeps the slow, real decentration/astigmatism trend (>>σ)
    # and marks the fast per-frame tilt swings (torsion motion) as b−b_ref for removal.
    xc = (L - 1) / 2.0
    xnl = (np.arange(L) - xc) / (L / 2.0)                # normalised lateral, [-1,1]
    rc = (L - 1) / 2.0
    amax = np.radians(float(p.get("rfd_max_deg", 3.0)))
    ref_sig = float(p.get("rfd_ref_sigma", 9.0) or 0.0)
    n_iter = max(1, int(p.get("rfd_iters", 3)))

    def _tilt(surf):                                    # per-frame lateral tilt b, 2σ-trimmed quadratic
        b = np.full(F, np.nan)
        for f in range(F):
            col = surf[:, f]; mk = (col > 1) & (col < depth - 1) & np.isfinite(col)
            if int(mk.sum()) < 60:
                continue
            try:
                c2 = np.polyfit(xnl[mk], col[mk], 2)
                res = col[mk] - np.polyval(c2, xnl[mk]); sd = np.std(res) + 1e-6
                keep = np.abs(res) < 2.0 * sd
                if int(keep.sum()) > 40:
                    idx = np.where(mk)[0][keep]; c2 = np.polyfit(xnl[idx], col[idx], 2)
                b[f] = c2[1]
            except Exception:  # noqa: BLE001
                continue
        return b

    def _ref(b):                                        # robust SMOOTH-across-frames baseline (keeps the DC + slow trend)
        g = np.isfinite(b)
        if int(g.sum()) < max(8, F // 4):
            return None
        bi = np.interp(np.arange(F), np.where(g)[0], b[g])
        bi = ndimage.median_filter(bi, size=5, mode="nearest")       # kill single-frame detector spikes
        return ndimage.gaussian_filter1d(bi, ref_sig, mode="nearest") if ref_sig > 0 else bi

    def _pivots(surf):                                  # per-frame surface pivot depth (lateral centre)
        sc = np.full(F, (depth - 1) / 2.0)
        for f in range(F):
            col = surf[:, f]; mk = (col > 1) & (col < depth - 1) & np.isfinite(col)
            if int(mk.sum()) > 40:
                sc[f] = float(np.median(col[mk]))
        return sc

    r0 = _rough(S)
    # CLOSED-LOOP: rotate → re-detect → re-measure the tilt until each frame reaches b_ref. The open-loop single-shot
    # rotation under-delivered (measured: removed only ~1/4 of the marked tilt swing). Accumulate the NET per-frame
    # angle on a NEAREST-filled iterate (stable re-detection — black corners would confuse the surface detector),
    # then apply the net rotation ONCE to the original B-scan with honest BLACK fill.
    det = volume.copy()
    Scur = S
    total_alpha = np.zeros(F)
    it_done = 0
    for it_done in range(1, n_iter + 1):
        b = _tilt(Scur)
        bref = _ref(b)
        if bref is None:
            break
        resid = np.where(np.isfinite(b), b - bref, 0.0)
        sig = float(p.get("rfd_smooth", 1.5) or 0.0)     # light denoise of the RESIDUAL (single-frame detect noise)
        if sig > 0:
            resid = ndimage.gaussian_filter1d(resid, sig, mode="nearest")
        alpha = np.clip(-2.0 * resid / L, -amax, amax)   # radians: rotation whose z-shift α·(x-xc) cancels the tilt
        if float(np.degrees(np.max(np.abs(alpha)))) < 0.03:
            break
        scen = _pivots(Scur)                             # DYNAMIC surface pivot (no spurious lateral shear)
        for f in range(F):
            if abs(alpha[f]) <= 1e-4:
                continue
            ct, st = np.cos(alpha[f]), np.sin(alpha[f])
            Mrot = np.array([[ct, st], [-st, ct]]); piv = np.array([rc, scen[f]]); offr = piv - Mrot @ piv
            fd = np.ascontiguousarray(det[f].T)
            det[f] = np.ascontiguousarray(ndimage.affine_transform(fd, Mrot, offset=offr, order=1, mode="nearest").T)
        total_alpha += alpha
        try:
            Scur = detect_surface_all(reformat_to_sagittal(det), p, workers=workers)
        except Exception:  # noqa: BLE001
            break
    total_alpha = np.clip(total_alpha, -amax, amax)
    max_deg = float(np.degrees(np.max(np.abs(total_alpha))))
    if max_deg < 0.03:                                   # nothing meaningful to rotate → strict no-op
        return volume, {"applied": False, "max_deg": round(max_deg, 2)}
    scen0 = _pivots(S)                                   # pivot on the ORIGINAL surface for the net one-shot rotation
    out = volume.copy(); nrot = 0
    for f in range(F):
        if abs(total_alpha[f]) <= 1e-4:
            continue
        ct, st = np.cos(total_alpha[f]), np.sin(total_alpha[f])
        Mrot = np.array([[ct, st], [-st, ct]]); piv = np.array([rc, scen0[f]]); offr = piv - Mrot @ piv
        fr_ld = np.ascontiguousarray(volume[f].T)        # (lateral, depth); BLACK fill (no invented edge data)
        out[f] = np.ascontiguousarray(ndimage.affine_transform(fr_ld, Mrot, offset=offr, order=1, mode="constant", cval=0.0).T)
        nrot += 1
    r1 = _rough(Scur)                                    # SELF-GATE on the nearest-filled iterate (stable corners)
    if not (r1 < r0):
        return volume, {"applied": False, "rough_before": round(r0, 3), "rough_after": round(r1, 3)}
    return out, {"applied": True, "frames_rotated": int(nrot), "max_deg": round(max_deg, 2), "iters": int(it_done),
                 "rough_before": round(r0, 3), "rough_after": round(r1, 3)}


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


def apply_manual_patch(volume: np.ndarray, patch) -> tuple[np.ndarray, int]:
    """Per-frame RIGID ground-truth patch: {frame: [depth_px, tilt_px_across_width]}.

    The reviewer-facing sibling of `manual_shifts`, which is depth-only and integer. Most of the residual
    defects review actually finds are ROTATIONS — a frame deep at one sagittal end and shallow at the other —
    and no depth nudge can express one. This applies both terms as a single sub-pixel rigid move: a uniform
    depth offset plus a linear across-lateral ramp, which is a rotation of the B-scan, so the instantaneous
    B-scan geometry is preserved exactly as the auto passes preserve it.

    STICKY and applied LAST, like `manual_shifts`: it is ground truth and must outrank every automatic guard.
    Storing a patch as PARAMETERS rather than baking it into voxels is the whole point — the preprocess
    endpoint re-runs every stage from the .OCT, so a voxel edit is erased on the next run, whereas this is
    re-applied. It also means the accumulated patches ARE the corpus the automatic passes must later learn to
    reproduce: each entry is a measured (shift, rotation) the reviewer accepted.

    `tilt_px_across_width` is the depth change from one lateral edge to the other; positive = deeper at
    increasing lateral index. Returns (volume, n_frames_patched)."""
    pairs = []
    if isinstance(patch, dict):
        pairs = list(patch.items())
    elif isinstance(patch, (list, tuple)):
        for item in patch:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((item[0], item[1]))
    F, depth, L = int(volume.shape[0]), int(volume.shape[1]), int(volume.shape[2])
    lat = np.arange(L, dtype=np.float64) - (L - 1) / 2.0
    out = volume.copy()
    n = 0
    for f, val in pairs:
        try:
            fi = int(f)
            if isinstance(val, (list, tuple)):
                sh = float(val[0])
                tl = float(val[1]) if len(val) > 1 else 0.0
            else:
                sh, tl = float(val), 0.0
        except (TypeError, ValueError, OverflowError, IndexError):
            continue
        if not (0 <= fi < F) or not (math.isfinite(sh) and math.isfinite(tl)):
            continue
        if abs(sh) < 0.02 and abs(tl) < 0.02:
            continue
        if abs(sh) >= depth:
            continue                      # never erase a whole frame
        d = sh + (tl / max(1.0, L - 1)) * lat
        out[fi] = _warp_by_displacement(np.ascontiguousarray(out[fi]), d, subpixel=True)
        n += 1
    return out, n


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


def apply_axial_surface_gt(volume: np.ndarray, axial_anchors, params: dict | None = None,
                           workers: int | None = None) -> tuple[np.ndarray, dict]:
    """AXIAL fix-tool GT (v0.0.186). The annotator opened an AXIAL B-scan (fixed FRAME, lateral×depth) and dragged
    the anterior corneal surface across LATERALS onto the true band where the auto-detector got it wrong — a notch
    at the apex/limbus of the first/last low-SNR frames that the SAGITTAL fix-columns tool (which corrects along the
    FRAME axis) structurally cannot reach. This applies that correction as a POST-HOC ADDITIVE per-frame warp on the
    FINISHED corrected volume: the frame_boundary_lat_smooth / apply_manual_shifts template, but the target is the
    user's ABSOLUTE drawn curve instead of a smoothed one. It never re-runs the sagittal flatten (no provided_edges,
    no re-detect) → no double-warp, no interior cascade. It is IDEMPOTENT + STICKY: it stores the TARGET depth (not a
    fixed shift) and each run RE-DETECTS the current surface with the SAME detector the annotator drew against and
    re-diffs, so re-runs land on the target every time. The correction is LOCAL: it spans only the drawn lateral
    range, raised-cosine feathered back to the detected surface over `axial_gt_feather` laterals on each side, so
    undrawn laterals are WYSIWYG-unchanged and no step forms. axial_anchors = {str(frame): {str(lateral): depth}} in
    the corrected-volume depth space (0 = TOP). volume = (frames, depth, lateral). Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not isinstance(axial_anchors, dict) or not axial_anchors:
        return volume, {"applied": False, "frames_adjusted": 0}
    if workers is None:
        workers = auto_workers()
    nF, depth, L = volume.shape
    fm = int(p.get("axial_gt_feather", 14) or 0)
    try:
        surf = detect_surface_all(reformat_to_sagittal(_fill_black_bands(volume)), p, workers=workers)  # (lat, frames)
    except Exception:  # noqa: BLE001 — best effort; without a current surface we cannot diff → no-op
        return volume, {"applied": False, "frames_adjusted": 0}
    out = volume.copy(); nadj = 0; xs = np.arange(L, dtype=np.float64)
    for f_key, lat_map in axial_anchors.items():
        try:
            f = int(f_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= f < nF) or not isinstance(lat_map, dict) or not lat_map:
            continue
        dl, dv = [], []                                          # drawn (lateral, depth) anchors for this frame
        for l_key, d in lat_map.items():
            try:
                li, dd = int(l_key), float(d)
            except (TypeError, ValueError):
                continue
            if 0 <= li < L and math.isfinite(dd):
                dl.append(li); dv.append(dd)
        if len(dl) < 3:                                          # too few anchors → don't inject a garbage warp
            continue
        order = np.argsort(dl); dl = np.asarray(dl)[order].astype(np.float64); dv = np.asarray(dv)[order]
        lo_l, hi_l = int(dl[0]), int(dl[-1])
        cur = surf[:, f].astype(np.float64)
        t_dense = np.interp(xs, dl, dv)                          # drawn target spread across laterals (flat outside)
        w = np.zeros(L, dtype=np.float64)                        # correction weight: 1 inside the drawn span
        w[lo_l:hi_l + 1] = 1.0
        for k in range(1, fm + 1):                               # raised-cosine feather back to the detected surface
            ww = 0.5 * (1.0 + math.cos(math.pi * k / max(1, fm)))
            if lo_l - k >= 0:
                w[lo_l - k] = max(w[lo_l - k], ww)
            if hi_l + k < L:
                w[hi_l + k] = max(w[hi_l + k], ww)
        good = np.isfinite(cur) & (cur > 1.0) & (cur < depth - 1)   # never move an off-cornea / padding column
        shift = np.where(good, w * (t_dense - cur), 0.0)           # + = deeper; local, feathered, WYSIWYG elsewhere
        shift[~np.isfinite(shift)] = 0.0
        shift = np.clip(shift, -(depth - 2), depth - 2)
        if np.any(np.abs(shift) > 1e-6):
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), shift, subpixel=True)
            nadj += 1
    return out, {"applied": nadj > 0, "frames_adjusted": int(nadj)}


def apply_sagittal_surface_gt(volume: np.ndarray, sag_anchors, params: dict | None = None,
                              workers: int | None = None) -> tuple[np.ndarray, dict]:
    """CORRECTED-RESULT sagittal fix-tool GT. The annotator opened the before/after view and dragged the
    anterior surface on the CORRECTED result along the FRAME axis (a fixed lateral, across frames) — the axis
    where residual INTER-FRAME drift lives (a trough on the early frames, a bump on the later ones = a
    per-frame depth offset the auto rigid alignment left behind, and the very thing the reviewer sees).

    Why this is a POST-HOC warp and not a change to the original GT: editing the raw provided_edges CANNOT fix
    this. Measured end-to-end, a +12 px nudge to the raw surface moved the corrected surface by 0.66 px
    (gain 0.055) — the rigid inter-frame alignment re-detects the real surface and re-smooths it, absorbing
    the nudge. So the correction is applied where it lands: a per-frame RIGID depth shift (+ a tilt when
    several laterals are drawn on one frame) on the FINISHED corrected volume. It is the SAGITTAL sibling of
    apply_axial_surface_gt: IDEMPOTENT + STICKY — it stores the absolute TARGET depth and each run re-detects
    the current surface with the same detector and re-diffs, so re-runs land on the target every time; the
    per-frame shift is interpolated across gaps between drawn frames and raised-cosine feathered back to zero
    beyond the drawn span so no step forms at the edges. sag_anchors = {str(lateral): {str(frame): depth}} in
    the corrected-output depth space (0 = TOP). volume = (frames, depth, lateral). Returns (volume, info)."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not isinstance(sag_anchors, dict) or not sag_anchors:
        return volume, {"applied": False, "frames_adjusted": 0}
    if workers is None:
        workers = auto_workers()
    nF, depth, L = volume.shape
    # NO feather by default. The correction ALIGNS drifted frames to their (un-drawn, smooth) neighbours, so a
    # hard band edge forms NO step — the drawn frames land ON the boundary trend. Feathering instead spreads the
    # shift onto those correct neighbours, diluting the fix (a single-frame drift only half-corrected). The
    # cross-lateral guard already guarantees an applied correction is a genuine alignment, so no smoothing is due.
    fm = int(p.get("sag_gt_feather", 0) or 0)
    try:
        surf = detect_surface_all(reformat_to_sagittal(_fill_black_bands(volume)), p, workers=workers)  # (lat, frames)
    except Exception:  # noqa: BLE001 — without a current surface we cannot diff → no-op
        return volume, {"applied": False, "frames_adjusted": 0}
    # gather the drawn anchors PER FRAME (across the drawn laterals): f -> {lateral: absolute target depth}
    per_frame: dict[int, dict[int, float]] = {}
    for l_key, frame_map in sag_anchors.items():
        try:
            li = int(l_key)
        except (TypeError, ValueError):
            continue
        if not (0 <= li < L) or not isinstance(frame_map, dict):
            continue
        for f_key, d in frame_map.items():
            try:
                f = int(f_key); dd = float(d)
            except (TypeError, ValueError):
                continue
            if 0 <= f < nF and math.isfinite(dd):
                per_frame.setdefault(f, {})[li] = dd
    if not per_frame:
        return volume, {"applied": False, "frames_adjusted": 0}
    xc = (L - 1) / 2.0
    shift = np.zeros(nF, dtype=np.float64)          # per-frame rigid depth shift (+ = deeper)
    tilt = np.zeros(nF, dtype=np.float64)           # per-frame tilt slope (depth px per lateral px)
    has = np.zeros(nF, dtype=bool)
    for f, lat_map in per_frame.items():
        ll = np.array(sorted(lat_map), dtype=np.int64)
        tt = np.array([lat_map[int(l)] for l in ll], dtype=np.float64)
        cur = surf[ll, f].astype(np.float64)                     # current corrected surface at those laterals
        good = np.isfinite(cur) & (cur > 1.0) & (cur < depth - 1)  # never diff against an off-cornea column
        if good.sum() < 1:
            continue
        llg = ll[good].astype(np.float64); resid = tt[good] - cur[good]   # move each drawn point by resid
        if llg.size >= 3 and (llg.max() - llg.min()) >= 8:       # enough lateral spread → fit shift + tilt
            A = np.vstack([np.ones(llg.size), llg - xc]).T
            a, b = np.linalg.lstsq(A, resid, rcond=None)[0]
            shift[f] = float(a); tilt[f] = float(b)
        else:                                                    # one lateral (or clustered) → pure per-frame shift
            shift[f] = float(np.median(resid)); tilt[f] = 0.0
        has[f] = True
    fs = np.nonzero(has)[0]
    if fs.size == 0:
        return volume, {"applied": False, "frames_adjusted": 0}
    # interpolate the per-frame shift/tilt across gaps between drawn frames, then feather to 0 beyond the span
    allf = np.arange(nF, dtype=np.float64)
    shift_i = np.interp(allf, fs.astype(np.float64), shift[fs])
    tilt_i = np.interp(allf, fs.astype(np.float64), tilt[fs])
    lo_f, hi_f = int(fs.min()), int(fs.max())
    wf = np.zeros(nF, dtype=np.float64); wf[lo_f:hi_f + 1] = 1.0
    for k in range(1, fm + 1):
        ww = 0.5 * (1.0 + math.cos(math.pi * k / max(1, fm)))
        if lo_f - k >= 0:
            wf[lo_f - k] = max(wf[lo_f - k], ww)
        if hi_f + k < nF:
            wf[hi_f + k] = max(wf[hi_f + k], ww)
    shift_i *= wf; tilt_i *= wf
    xs = np.arange(L, dtype=np.float64)
    # ── CROSS-LATERAL GUARD (rigid-or-accept) ────────────────────────────────────────────────────────────────
    # A per-frame shift/tilt moves the WHOLE B-scan across all 513 laterals. It is only a LEGAL, useful rigid
    # correction if it makes the drawn frames MORE consistent with the smooth surface implied by the un-corrected
    # frames just outside the drawn band — measured over ALL good laterals, not just the one the reviewer drew on.
    #   * A real inter-frame DRIFT (or a genuinely TILTED frame) is off across the whole B-scan, so the move
    #     reduces that deviation → APPLY (the tissue moves, rigidly).
    #   * An EDGE-SPECIFIC defect (surface wrong only at the periphery while the centre is already right) cannot
    #     be fixed by any whole-frame move: shifting to fix the edge corrupts the centre, so the deviation RISES
    #     → DECLINE and leave it. Per the standing rule (corrections are rigid axial shifts/rotations only), a
    #     periphery no rigid move can reach is accepted as-is, never patched by a non-rigid deformation.
    # Whole-correction decision (one coherent defect per draw), so the feather rides the core consistently.
    band = np.nonzero(wf > 1e-6)[0]
    ref_lo = int(max(0, (int(band.min()) if band.size else 0) - 1))
    ref_hi = int(min(nF - 1, (int(band.max()) if band.size else nF - 1) + 1))
    span = max(1, ref_hi - ref_lo)
    db = da = 0.0; nn = 0
    for f in fs:                                                  # the ANCHORED frames = the core of the correction
        disp = shift_i[f] + tilt_i[f] * (xs - xc)
        w = (f - ref_lo) / span
        expected = (1.0 - w) * surf[:, ref_lo] + w * surf[:, ref_hi]   # smooth surface from the band boundaries
        cur_f = surf[:, f]
        gx = (np.isfinite(cur_f) & (cur_f > 1.0) & (cur_f < depth - 1)
              & np.isfinite(expected) & (expected > 1.0) & (expected < depth - 1))
        if gx.sum() < max(8, int(0.1 * L)):
            continue
        db += float(np.sum(np.abs(cur_f[gx] - expected[gx])))
        da += float(np.sum(np.abs(cur_f[gx] + disp[gx] - expected[gx])))
        nn += int(gx.sum())
    dev_before = (db / nn) if nn else 0.0
    dev_after = (da / nn) if nn else 0.0
    # meaningful improvement required: a periphery-only defect makes dev_after >= dev_before → decline
    apply_ok = nn > 0 and (dev_after < dev_before - 0.4) and (dev_after < 0.92 * dev_before)
    out = volume.copy(); nadj = 0
    if apply_ok:
        for f in range(nF):
            if abs(shift_i[f]) < 1e-6 and abs(tilt_i[f]) < 1e-9:
                continue
            disp = shift_i[f] + tilt_i[f] * (xs - xc)
            disp = np.clip(disp, -(depth - 2), depth - 2)
            out[f] = _warp_by_displacement(np.ascontiguousarray(out[f]), disp, subpixel=True)
            nadj += 1
    return out, {"applied": nadj > 0, "frames_adjusted": int(nadj),
                 "declined": bool(not apply_ok and nn > 0),
                 "dev_before": round(dev_before, 2), "dev_after": round(dev_after, 2),
                 "n_drawn_frames": int(fs.size),
                 "max_shift": float(np.max(np.abs(shift_i))) if nadj else 0.0,
                 "max_tilt_swing": float(np.max(np.abs(tilt_i)) * L) if nadj else 0.0}


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


def _artifact_bands(params: dict | None, n_frames: int, n_lateral: int):
    """#9 v3 ARTIFACT CROP: a PER-LATERAL frame band, marked on several slices and INTERPOLATED across laterals.

    A time-domain artifact (e.g. an eyelid closing during the slow scan) occupies a frame band [lo, hi] whose
    extent VARIES per sagittal slice (lateral) — so a single uniform box (crop_region) can't describe it. The
    reviewer marks the band [lo, hi] on a few laterals; params['crop_bands'] = {str(lateral): [lo, hi]} (inclusive
    frame indices). This LINEARLY interpolates lo and hi across laterals BETWEEN the marked slices, so the
    artifact volume fills in without marking every slice. NO crop outside the marked lateral span — the reviewer
    marks where the artifact is present, so its lateral extent is exactly [min marked lateral, max marked lateral]
    (mark the boundary slices to control it). One mark → that lateral only.

    Returns a list of length n_lateral, each an int ndarray of frame indices to exclude/zero for that lateral
    (possibly empty), or None when no valid band is defined. Consumed as per-lateral zero_cols in the fit (so the
    cornea is fit to the remaining frames) and zeroed by _apply_crop before SAM2."""
    raw = (params or {}).get("crop_bands")
    if not isinstance(raw, dict) or not raw:
        return None
    marks = []
    for k, v in raw.items():
        try:
            lat = int(k)
            if not isinstance(v, (list, tuple)) or len(v) != 2:
                continue
            lo, hi = int(v[0]), int(v[1])
        except (TypeError, ValueError):
            continue
        if not (0 <= lat < int(n_lateral)):
            continue
        lo, hi = sorted((lo, hi))
        lo = max(0, min(int(n_frames) - 1, lo)); hi = max(0, min(int(n_frames) - 1, hi))
        marks.append((lat, float(lo), float(hi)))
    if not marks:
        return None
    marks.sort()
    lats = np.array([m[0] for m in marks], dtype=np.float64)
    los = np.array([m[1] for m in marks], dtype=np.float64)
    his = np.array([m[2] for m in marks], dtype=np.float64)
    out = [np.array([], dtype=int) for _ in range(int(n_lateral))]
    lo_lat, hi_lat = int(lats[0]), int(lats[-1])
    for x in range(lo_lat, hi_lat + 1):
        if lats.size == 1:
            a, b = los[0], his[0]
        else:
            a = float(np.interp(x, lats, los)); b = float(np.interp(x, lats, his))
        # round-half-UP (floor(x+0.5)), NOT Python's banker's round(): the frontend preview interpolates with the
        # SAME formula but rounds via JS Math.round (half-up). Matching it keeps preview == what actually gets
        # cropped at half-integer interpolated boundaries (frames are >=0, so floor(x+0.5) == Math.round(x)).
        a = int(math.floor(a + 0.5)); b = int(math.floor(b + 0.5))
        a = max(0, min(int(n_frames) - 1, a)); b = max(0, min(int(n_frames) - 1, b))
        if b >= a:
            out[x] = np.arange(a, b + 1, dtype=int)
    return out


def _reconstruct_surface_over_bands(surface: np.ndarray, params: dict | None) -> np.ndarray:
    """Make the DETECTED surface IGNORE the marked artifact bands (#9 v3). The raw detector tracks the artifact
    (a deep bright structure) INSIDE the band — the displayed edge dives into it instead of ignoring it. Per
    lateral, replace the band frames' surface with a smooth reconstruction from the NON-artifact frames on either
    side: linear-interp between the two boundary values when the band is interior, or a FLAT HOLD from whichever
    side exists when it runs to a frame edge. The band is cropped (zeroed) anyway, so this only makes the DISPLAYED
    edge honest — it no longer follows the artifact. `surface` is (n_lateral, n_frames). Off (no crop_bands) → no-op.

    NOTE this does NOT touch the flatten: smooth_volume detects its own edges and EXCLUDES the band from the fit
    via zero_cols (see the _artifact_bands injection there), so the warp already ignores the artifact; this fixes
    the detection the reviewer SEES (detect_surface_all → the baseline / oct-border curves)."""
    if not isinstance(surface, np.ndarray) or surface.ndim != 2:
        return surface
    n_lat, n_frames = surface.shape
    bands = _artifact_bands(params, n_frames, n_lat)
    if bands is None:
        return surface
    out = surface.copy().astype(np.float64)
    for lat in range(n_lat):
        fs = bands[lat]
        if not fs.size:
            continue
        lo, hi = int(fs.min()), int(fs.max())
        left = out[lat, max(0, lo - 5):lo]
        right = out[lat, hi + 1:min(n_frames, hi + 6)]
        lval = float(np.median(left[np.isfinite(left)])) if np.isfinite(left).any() else None
        rval = float(np.median(right[np.isfinite(right)])) if np.isfinite(right).any() else None
        if lval is None and rval is None:
            continue
        if lval is not None and rval is not None:
            out[lat, lo:hi + 1] = np.linspace(lval, rval, hi - lo + 1)   # interior band → interpolate across
        else:
            out[lat, lo:hi + 1] = lval if lval is not None else rval     # band runs to an edge → flat hold
    return out.astype(surface.dtype)


def fit_quadratic_excluding_bands(edge, params: dict | None, lateral: int, n_lateral: int):
    """The cyan quadratic FIT for a slice, EXCLUDING this lateral's artifact-band frames (#9 v3). detect_surface_all
    fills the band with a FLAT-HOLD reconstruction of the cornea-boundary value; fitting a parabola THROUGH that long
    flat run drags it off the real cornea (the reviewer sees a "blue line looks off"). Fit the quadratic to the
    NON-band (cornea) frames only and extrapolate across the band. Falls back to the plain RANSAC fit when this
    lateral has no band. Returns a per-frame array over ALL frames."""
    edge = np.asarray(edge, dtype=np.float64)
    nf = edge.size
    p = params or DEFAULT_PARAMS
    res = float(p.get("residual_threshold", 8.0))
    bands = _artifact_bands(p, nf, n_lateral)
    band = bands[lateral] if (bands is not None and 0 <= lateral < len(bands)) else None
    if band is None or not band.size:
        return _fit_quadratic_ransac(edge, res)
    keep = np.ones(nf, dtype=bool)
    keep[band[(band >= 0) & (band < nf)]] = False
    if int(keep.sum()) < 3:
        return _fit_quadratic_ransac(edge, res)
    xs = np.arange(nf, dtype=np.float64)
    try:
        return np.polyval(np.polyfit(xs[keep], edge[keep], 2), xs)
    except Exception:  # noqa: BLE001
        return _fit_quadratic_ransac(edge, res)


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
    bands = _artifact_bands(params, n_frames, n_lateral)         # #9 v3: per-lateral interpolated artifact band
    if box is None and legacy.size == 0 and bands is None:
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
    if bands is not None:
        # corrected is (frames, depth, lateral): zero this lateral's interpolated artifact frames across all depth.
        for lat in range(n_lateral):
            fs = bands[lat]
            if fs.size:
                corrected[fs, :, lat] = 0
                zeroed += int(fs.size) * depth
    return corrected, int(zeroed)


def _crop_incomplete_cornea(corrected: np.ndarray, params: dict | None = None, workers: int | None = None):
    """Minimal LATERAL dimension-crop (v0.0.196): the rigid derotate BLACK-fills the swept-out corners, which at
    the EXTREME laterals can leave the corneal BAND partially black — a partial cornea in a sagittal slice confuses
    SAM2. Trim ONLY the few edge laterals whose band is incomplete, so every REMAINING sagittal slice contains the
    FULL cornea. Interior black (deep, below the cornea) is left untouched; capped at `crop_cornea_max_frac` per
    side so a clean/non-rotated scan barely moves. `corrected` is (frames, depth, lateral). Returns
    (cropped, (n_left, n_right))."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    if workers is None:
        workers = auto_workers()
    F, D, L = corrected.shape
    cap = int(float(p.get("crop_cornea_max_frac", 0.06)) * L)
    if cap < 1:
        return corrected, (0, 0)
    band = int(p.get("crop_cornea_band", 110))
    blk = float(p.get("crop_cornea_black_thr", 0.15))
    try:
        S = detect_surface_all(reformat_to_sagittal(corrected), p, workers=workers)   # (lat, frames)
    except Exception:  # noqa: BLE001
        return corrected, (0, 0)

    cc = S[L // 2]                                            # centre-lateral surface = "is there a cornea this frame?"
    cmid = L // 2

    def _incomplete(l):
        col = S[l]
        for f in range(F):
            cd = cc[f]                                         # SKIP frames the whole B-scan lacks a cornea for
            if not (np.isfinite(cd) and 1 < cd < D - 1):       # (a damaged / black TAIL frame is a frame problem,
                continue                                       #  not a rotation edge — cropping laterals can't fix it)
            clo = int(cd); chi = min(D, clo + band)
            if float((corrected[f, clo:chi, cmid] < 1).mean()) > blk:
                continue                                       # centre band itself black at this frame → skip
            d = col[f]                                         # the frame HAS a cornea (centre is intact):
            if not (np.isfinite(d) and 1 < d < D - 1):
                return True                                    # ...but this lateral has no surface → incomplete
            lo = int(d); hi = min(D, lo + band)
            if float((corrected[f, lo:hi, l] < 1).mean()) > blk:
                return True                                    # ...this lateral's band partly black → incomplete
        return False

    lo = 0
    while lo < cap and _incomplete(lo):
        lo += 1
    hi = L
    while (L - hi) < cap and _incomplete(hi - 1):
        hi -= 1
    if lo == 0 and (L - hi) == 0:
        return corrected, (0, 0)
    return np.ascontiguousarray(corrected[:, :, lo:hi]), (int(lo), int(L - hi))


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
    # ── SURFACE-CROP EVIDENCE (crop_detect="geom"). The apex rule needs the anterior surface on BOTH the RAW
    # and the motion-corrected geometry plus the shift between them, and only THIS point in the pipeline has
    # both. Build S_raw BEFORE axial_motion_correct and hand it to AMC (which needs the same detection), then
    # re-detect immediately AFTER AMC. Capturing S_mc here — after AMC and the dewarp but BEFORE the de-tilt
    # and before _apply_crop — is what makes the evidence match the geometry the rule was validated on: the
    # de-tilt rebases every row by pad_top+shifts[f] and _apply_crop zeroes whole frames, either of which
    # would feed the rule a surface in a different basis than its thresholds assume.
    _S_raw = _S_mc = _M = None
    _want_sc = (provided_edges is None
                and (params or {}).get("surface_crop_frames") is None    # a manual GT set: never auto-detect
                and _pcr.get("auto_surface_crop", True)
                and str(_pcr.get("crop_detect", "geom")).lower() == "geom"
                and str(_pcr.get("detector", "dp")).lower() != "legacy"
                and bool(_pcr.get("axial_motion_correct", False))        # no AMC → no S_mc/M → count rule
                and not bool(_pcr.get("intra_frame_dewarp", True)))      # IFD would move rows after AMC
    if _want_sc:
        try:
            _S_raw = detect_surface_all(reformat_to_sagittal(vol), params, workers=workers)
        except Exception:  # noqa: BLE001 — hoisting this pass out of AMC must not remove AMC's own guard:
            _S_raw = None  # on failure do NOT pass detect= to AMC (it re-detects and degrades to a no-op)
    # AXIAL MOTION CORRECTION — now on the CORRECTIONS path too.
    #
    # This is the stage whose whole job is removing inter-frame axial drift, and `provided_edges is None`
    # excluded it from every "Correct & re-run". Measured consequence on a corrected scan: across-frame sag
    # 162 px against an anatomical ~85, i.e. ~75 px of drift left in, while two approved scans that took the
    # AUTO path (AMC ran: 98 and 100 frames adjusted) sat at 93 and 86 px — right on anatomy. That residue is
    # exactly the "corrected pane still isn't smooth" the reviewer reported. Correcting the border must not
    # cost the scan its motion correction.
    #
    # AUTO PATH ONLY here (provided_edges is None). Running AMC BEFORE the flatten on the corrections path was
    # tried and regressed the scan (dev 3.95 -> 10.86): provided_edges is in RAW coordinates and AMC moves the
    # volume beneath it, so re-basing the surface through AMC's shift — though the sign was right — still fed
    # the flatten and the downstream detectors a surface in a shifted basis they did not expect. The
    # corrections path instead removes drift AFTER the flatten (see the provided_edges branch below), where
    # AMC operates on a finished volume exactly as it does on any other, with no basis to reconcile.
    if provided_edges is None and bool(_pcr.get("axial_motion_correct", False)):
        vol, _amc_info = axial_motion_correct(vol, params, workers=workers, detect=_S_raw)
    # INTRA-frame saccade de-distortion (v0.0.159): re-warp only the genuinely saccade-distorted B-scans onto
    # the smooth 3-D dome before the flatten. Strict no-op on a clean scan.
    _ifd_info = None
    if provided_edges is None and bool(_pcr.get("intra_frame_dewarp", True)):
        vol, _ifd_info = intra_frame_dewarp(vol, params, workers=workers)
    if _want_sc and _S_raw is not None:
        # Only a motion correction that was actually APPLIED gives a meaningful (S_mc, M) pair. On a no-op AMC
        # returns the volume unmodified, so S_mc would equal S_raw with M=0 — a configuration the rule was
        # never validated in; fall back to the count rule there rather than guess.
        if _amc_info and _amc_info.get("applied") and _amc_info.get("shift") is not None:
            try:
                _sag_mc = reformat_to_sagittal(vol)
                _S_mc = detect_surface_all(_sag_mc, params, workers=workers)
                _M = np.asarray(_amc_info["shift"], dtype=np.float64)
                if _S_mc.shape != (int(vol.shape[2]), int(vol.shape[0])) or _M.size != int(vol.shape[0]):
                    _S_mc = _M = None                      # shape guard: wrong axes ⇒ do not feed the rule
            except Exception:  # noqa: BLE001
                _S_mc = _M = None
    _auto_cr = None                                                # auto crop-region box → surfaced in info → persisted
    # PERF: the noise-crop check and the surface-crop check BOTH run the ~12s anterior detector on the SAME raw
    # volume with the SAME (pre-auto-tune) detector params. The detector output is independent of crop_region /
    # crop_lateral (those only mask the volume via _apply_crop; they are NOT read by _edge_worker), so compute
    # detect_surface_all ONCE and reuse it for both. The cache is CLEARED the instant _apply_crop mutates `vol`
    # (then it recomputes on the cropped volume) — so a cropped scan is byte-identical, the common no-crop scan
    # saves a full detect pass.
    _det_cache: dict = {}
    if _S_mc is not None:
        # PASS-COUNT NEUTRAL: the surface-crop evidence pass above already detected the anterior on exactly the
        # current `vol` (post-AMC, post-dewarp, pre-de-tilt, pre-crop) with the same `params`, which is what
        # _cur_sag_det would compute on first use. Seed it rather than paying for a second identical pass. Any
        # later mutation of `vol` clears the cache as before, so this cannot go stale.
        _det_cache["sag"] = _sag_mc
        _det_cache["det"] = _S_mc

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
            # The evidence variables are kept OUTSIDE _det_cache on purpose: the cache is cleared when the
            # de-tilt or _apply_crop mutates `vol`, but the rule must still see the pre-de-tilt geometry it was
            # validated on. If _apply_crop ran, `_s1`/`_d1` are the cropped re-detect while the evidence is not
            # — that is fine, since the evidence alone decides `frames` and `_d1` only feeds counts.
            _ci = detect_surface_crop_frames(_s1, params, workers=workers, detect=_d1,
                                             sc_s_mc=_S_mc, sc_s_raw=_S_raw, sc_shift=_M)
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
        # RULE-AWARE cap. The 0.5 default was calibrated against the legacy count rule, which could flag most
        # of a noisy scan. It is too tight for the user's OWN ground truth: 6 GT surface-crop scans exceed
        # frac 0.45 and one reaches 0.574 (p1_od_v1, 58 of 101 frames genuinely clipped), so a 0.5 cap refuses
        # to correct scans that ARE correctable. The geom rule earns the looser cap: its worst-case selection
        # over the 37 GT scans is 0.545, and over all 129 vetted non-clipped scans it selects NOTHING (frac
        # 0.000 everywhere) — there is no creep toward the threshold to guard against. A genuinely failed /
        # fully-off-axis scan still lands near 1.0 and is still refused. 0.75 matches crop_noise_max_frac,
        # which encodes the same "this scan is a write-off" judgement.
        # NOTE the _auto_crop guard on reading _ci: on the MANUAL path the user supplied surface_crop_frames
        # directly, the detector never ran, and _ci does not exist. The cap is an AUTO-only sanity gate anyway
        # (a manual crop is a deliberate human decision and is never refused), so resolve it inside the guard.
        _cap = 0.0
        if _auto_crop:
            _cap = float(_pc.get("crop_auto_max_frac_geom", 0.75)) if _ci.get("rule") == "geom" \
                else float(_pc.get("crop_auto_max_frac", 0.5))
        if _auto_crop and _frac > _cap:
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
                                     "peak_slices": _peak, "pb_span": round(_pb_span, 1),
                                     # WHICH rule produced `frames`, and the algorithm version. This payload is
                                     # a persisted SNAPSHOT that the mark editors prefer over a live detection,
                                     # so without a version stamp an improved detector stays invisible on every
                                     # already-processed scan and there is no way to tell a stale set apart.
                                     "rule": _ci.get("rule") if _auto_crop else "manual",
                                     "algo": _SC_ALGO_VERSION,
                                     # WHICH frames were treated as clipped. Only the COUNT used to be
                                     # recorded, so an auto-detected crop could not be shown, reviewed or
                                     # edited — the UI had nothing to mark and the user could not tell
                                     # WHERE the pipeline thought the apex was missing. These are ARRAY
                                     # frame indices, the same space as oct_params.surface_crop_frames,
                                     # so an auto set can be loaded straight into the editor and amended.
                                     "frames": sorted(int(f) for f in _crop_frames)}}
            if _crop_tune:
                info["auto_tune"] = _crop_tune
            p_all = {**DEFAULT_PARAMS, **(params or {})}
            # ── FINAL RIGID SMOOTHING on the surface-crop path (opt-in) ──────────────────────────────────────
            # This branch returns before the main path's three rigid passes, so every canvas-extended scan ships
            # WITHOUT them. Measured cost: on an unpadded control those passes take roughness 0.750 -> 0.601,
            # and 0 of the 39 canvas-extended scans in the store are approved against 170/269 elsewhere, 64% of
            # their rejections citing smoothness. (Confounded — extension only fires on clipped-apex scans,
            # which are harder to begin with — but they are also the only scans denied the smoothing.)
            #
            # It CANNOT simply be enabled: these passes re-detect, and on a zero-padded volume the detector
            # locks onto the padding cliff (row 27 instead of ~199). Run there, rigid_height_refine happily
            # "improves" 95 frames against that artifact — real shifts computed from a false surface, which is
            # worse than skipping it. So this is gated on the padding actually having been filled
            # (crop_pad_fill="background"), which is what restores detection to the true epithelium.
            # Order matches the main path: rigid passes first, then the user's GT (axial anchors, manual
            # shifts, reviewer patches) so GT still wins.
            if (bool(p_all.get("surface_crop_finish", False))
                    and str(p_all.get("crop_pad_fill", "zeros")).lower() == "background"
                    and bool(p_all.get("rigid_frame_warp", True))):
                if p_all.get("rigid_height_refine", True):
                    corrected, _rhr = rigid_height_refine(corrected, params, workers=workers)
                    info["rigid_height_refine"] = _rhr
                # DEROTATE IS OFF ON THIS PATH BY DEFAULT — measured, not assumed. A five-arm ablation over
                # the scans where surface_crop_finish degraded per-frame shape (.work/stage_ablate.py) put the
                # damage squarely on this pass, and showed that dropping it keeps the whole roughness gain:
                #   case_cs005_od_v2  all three: undul 13.06 rough 1.178 | minus derotate: undul  3.74 rough 1.127
                #   case_cs014_os_v1  all three: undul  4.57 rough 0.876 | minus derotate: undul  1.12 rough 0.825
                # i.e. undulation returns to its no-op baseline (3.72 / 1.04) while roughness still improves
                # 1.461->1.127 and 1.231->0.825. Strictly better than running it.
                # Edge-loss was 0.0% in every arm, so this is NOT rotation pushing tissue off the canvas (the
                # obvious hypothesis, and it is refuted); the pass resamples each column by a different
                # fractional offset and the detector's sub-pixel response follows, though the size of the
                # effect is not fully explained. Scoped to THIS path only: derotate is untouched on the main
                # path, where the 269 unpadded scans it serves are 63% approved.
                if p_all.get("surface_crop_derotate", False):
                    corrected, _rfd = rigid_frame_derotate(corrected, params, workers=workers)
                    info["rigid_frame_derotate"] = _rfd
                if p_all.get("rigid_frame_refine", True):
                    corrected, _rfr = rigid_frame_refine(corrected, params, workers=workers)
                    info["rigid_frame_refine"] = _rfr
                # RE-FILL after the passes. Each of them warps frames, and _warp_by_displacement fills the rows
                # a shift vacates with ZEROS (see ~line 1531) — so the cliff this whole path exists to remove is
                # cut afresh, per frame, after the extension-time fill has already run. Measured on the
                # delivered file: re-detecting case_cs024_os_v2 gave undulation 1.56 px before this branch and
                # 22.94 px after it, and case_cs008_od_v3 1.34 -> 41.77 px — i.e. the output was left HARDER to
                # detect on than the zero-padded volume it replaced, which would silently damage every
                # downstream consumer that re-detects (SAM2 seeding, QA, the review metrics).
                # Per-column off each column's own first-valid row, so a frame displaced further than the
                # original pad is covered too.
                corrected = revert_sagittal(_fill_pad_background(reformat_to_sagittal(corrected), 0,
                                                                 int(p_all.get("crop_pad_fill_src", 24))))
            _axa = p_all.get("axial_anchors")   # AXIAL fix-tool GT (see main path) — applies on the surface-crop path too
            if _axa:
                corrected, _axinfo = apply_axial_surface_gt(corrected, _axa, params, workers=workers)
                info["axial_anchors"] = _axinfo
            # "Smooth to trusted slices": propagate the reviewer's edited (drawn) + approved curves across the
            # volume. It CONSUMES corrected_edge_anchors as the edited GT, so the standalone per-frame warp below
            # is skipped in that mode (else the edits would apply twice).
            if p_all.get("corrected_smooth_align"):
                corrected, _csa = align_corrected_to_smooth(corrected, params, workers=workers)
                info["corrected_smooth_align"] = _csa
            elif p_all.get("corrected_edge_anchors"):   # standalone CORRECTED-edge fix-tool (guarded rigid axial move)
                corrected, _ceinfo = apply_sagittal_surface_gt(corrected, p_all.get("corrected_edge_anchors"), params, workers=workers)
                info["corrected_edge_anchors"] = _ceinfo
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
            if _amc_info:                     # was dropped on this path — recorded only on the normal path
                info["axial_motion_correct"] = _amc_info
            if _ifd_info:
                info["intra_frame_dewarp"] = _ifd_info
            write_volume_nifti(corrected, out_nifti, sp)
            # DELIVERED-VOLUME QA on the surface-crop path too. This branch hardcodes metrics=[] /
            # apex_clipped=0 (it never runs the iterate loop, so it has no per-pass metrics), which left
            # these scans with NO quality number at all — that, not genuine cleanliness, is why this class
            # looked "perfectly specific": no other detector could physically fire on these manifests.
            # Measured AFTER the write: QA costs ~3x the volume in peak RAM and a Linux OOM kill is not
            # catchable, so it must never run at the one moment when the output does not yet exist.
            # rigid_height_refine never runs on this path, so max_jitter is recorded as an explicit null.
            _fqa = measure_delivered_qa(corrected, params, workers=workers, path="surface_crop")
            if _fqa:
                info["final_qa"] = _fqa
            info["out"] = str(out_nifti)
            if _auto_cr:
                info["auto_crop_region"] = _auto_cr
            if _detilt_info:
                info["detilt"] = _detilt_info
            info["proposals"] = _proposals
            if _crop_guard_removed:
                info["crop_guard_removed_frames"] = list(_crop_guard_removed)
            return info
    _crop_recon_info = None
    # ── POSTERIOR-DERIVED CURVATURE ON THE CORRECTIONS PATH ─────────────────────────────────────────────
    # THE REVIEWER'S MODEL, in their words: the bottom line "allows for curvature correction even when the top
    # edge (red) is not present". On a surface-cropped frame there IS no anterior to detect or to draw, so the
    # posterior is the only observation of the cornea's shape there, and the anterior is recoverable from it as
    # posterior − thickness (thickness interpolated from the un-cropped flanks).
    #
    # build_surface_crop_edges does exactly that, and already reads crop_post_anchors per slice — so the
    # reviewer's own orange line feeds it. It was simply unreachable from here: its only call site sat inside
    # the `provided_edges is None` branch, which is why marking amber columns and correcting the bottom line
    # both appeared to do nothing on a re-run. One gate, both symptoms.
    #
    # PRECEDENCE: the reviewer's RED edge wins wherever they drew one, including above the window — there it is
    # their hypothesis for the clipped apex, which is a statement about this cornea and outranks a
    # reconstruction inferred from thickness. The reconstruction fills the remaining cropped frames only.
    if provided_edges is not None:
        _cf_src = (params or {}).get("surface_crop_frames") if params else None
        if _cf_src:
            try:
                _pe = np.asarray(provided_edges, dtype=np.float32).copy()
                _Lp, _Fp = int(_pe.shape[0]), int(_pe.shape[1])
                _cf = np.array(sorted({int(f) for f in _cf_src if 0 <= int(f) < _Fp}), dtype=int)
                if _cf.size:
                    _rec, _ = build_surface_crop_edges(reformat_to_sagittal(vol), _cf.tolist(),
                                                       params, workers=workers)
                    _rec = np.asarray(_rec, dtype=np.float32)
                    if _rec.shape == _pe.shape:
                        # where the reviewer drew a red anchor — those (slice, frame) keep their own value
                        _drawn = np.zeros(_pe.shape, dtype=bool)
                        for _sk, _fm in ((params or {}).get("border_anchors") or {}).items():
                            try:
                                _si2 = int(_sk)
                            except (TypeError, ValueError):
                                continue
                            if not (0 <= _si2 < _Lp) or not isinstance(_fm, dict):
                                continue
                            for _fk in _fm:
                                try:
                                    _fi2 = int(_fk)
                                except (TypeError, ValueError):
                                    continue
                                if 0 <= _fi2 < _Fp:
                                    _drawn[_si2, _fi2] = True
                        _take = np.isfinite(_rec[:, _cf]) & (~_drawn[:, _cf])
                        _pe[:, _cf] = np.where(_take, _rec[:, _cf], _pe[:, _cf])
                        provided_edges = _pe
                        _crop_recon_info = {"n_frames": int(_cf.size),
                                            "n_columns_rebuilt": int(_take.sum()),
                                            "from": "posterior continuity (bottom line)"}
            except Exception:  # noqa: BLE001 — best-effort: fall back to the anterior as supplied
                _crop_recon_info = None

    # ── CANVAS EXTENSION ON THE CORRECTIONS PATH ────────────────────────────────────────────────────────
    # When the reviewer draws the anterior ABOVE the captured window (negative depth — the definition of a
    # surface-cropped apex), the volume has to grow upward or there is nowhere for that tissue to go. The
    # auto path does this at :7241, but that branch is gated `provided_edges is None`, so correcting a scan
    # skipped it entirely: the drawn line was understood well enough to be protected by the disp>=0 clamp
    # (:4336) and then nothing was done with it, which is exactly the reviewer's report — "the corrected
    # result never shifts those columns outside the image area".
    #
    # A CANVAS OPERATION ONLY, deliberately. The auto branch extends AND warps (flattening to its own
    # posterior parabola) and then returns; running it here would mean two warps composed, or the extension
    # deciding the alignment instead of the correction. Growing the canvas and re-basing the surface leaves
    # the moving to the rigid flatten below — which is the reviewer's own rule: corrections fix the detected
    # edge, the rigid transforms do the moving.
    _ext_pad = 0
    if provided_edges is not None:
        _pe = np.asarray(provided_edges, dtype=np.float32)
        _above = float(np.nanmin(_pe)) if np.isfinite(_pe).any() else 0.0
        if _above < 0.0:
            _pex = {**DEFAULT_PARAMS, **(params or {})}
            # headroom for the deepest drawn point plus a margin, capped exactly where the reconstruction is
            # capped — asking for more than warp_surface_crop_extend could ever deliver would be incoherent.
            _ext_pad = int(min(float(_pex.get("crop_max_pad", 120)),
                               np.ceil(-_above) + float(_pex.get("crop_pad_margin", 8))))
            if _ext_pad > 0:
                _F, _D, _L = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
                _tall = np.zeros((_F, _D + _ext_pad, _L), dtype=vol.dtype)
                _tall[:, _ext_pad:, :] = vol
                # Fill the new rows with real background rather than zeros: a zero block against ~33-valued
                # OCT background is a cliff the detector locks onto (it mistook row 27 for the epithelium on
                # padded volumes), and every later stage would then measure that cliff instead of the cornea.
                if str(_pex.get("crop_pad_fill", "zeros")).lower() == "background":
                    # (out, pad, src) — src is how many rows of real background to sample, NOT the depth.
                    # first_valid is left to default: the pad here is uniform (no per-frame displacement has
                    # been applied yet), so every column's first real row is exactly _ext_pad.
                    _tall = revert_sagittal(_fill_pad_background(
                        reformat_to_sagittal(_tall), _ext_pad, int(_pex.get("crop_pad_fill_src", 24))))
                vol = _tall
                # The surface is in OLD-canvas rows; every row moved down by _ext_pad, so the drawn apex that
                # was at a negative depth is now a real row inside the volume. This is what makes the flatten
                # able to act on it at all.
                provided_edges = _pe + float(_ext_pad)
    if provided_edges is not None:
        # fix-columns marched re-detection: a SINGLE same-canvas warp that flattens to the user-validated
        # surface, NO iteration and NO axial-refine — so the corrected volume matches the scrub preview exactly.
        corrected = smooth_volume(vol, params, progress=progress, provided_edges=provided_edges, workers=workers)
        info = {"passes": 1, "best_pass": 1, "metrics": [], "axial_metrics": [], "stopped": "redetect",
                "apex_clipped": {"slices": {}, "n_slices": 0, "n_frames_total": 0}}
        # Record AMC here too. It is reported on the auto and surface-crop branches but was never recorded on
        # this one, so "did motion correction run?" was unanswerable from the manifest for every corrected
        # scan — which is how a missing stage stayed hidden until the anatomy said so.
        if _amc_info:
            info["axial_motion_correct"] = _amc_info
        if _crop_recon_info:
            info["crop_reconstruction"] = _crop_recon_info
        if _ext_pad:
            # Recorded, because a taller output is the most visible thing a run can do and a silent change of
            # canvas height is indistinguishable from a bug. The reviewer asked for exactly this and needs to
            # be able to confirm it happened.
            info["canvas_extend"] = {"pad": int(_ext_pad), "reason": "corrected apex above the window",
                                     "depth_before": int(vol.shape[1] - _ext_pad),
                                     "depth_after": int(vol.shape[1])}
        p_all = {**DEFAULT_PARAMS, **(params or {})}
        # ── FINAL RIGID PASSES on the fix-columns path ───────────────────────────────────────────────────────
        # REVIEWER DIRECTIVE: "not to manipulate the columns directly in sagittal view but to correct the
        # detected edge / corneal curvature. This will inform detector and still allow the axial transforms to
        # occur." Until now the second half did not happen: this branch returned before the main path's rigid
        # passes, so a corrected edge bought a single warp onto the user's surface and nothing else — the same
        # early-return shape as the surface-crop branch.
        #
        # Why running them here is coherent rather than double-correcting: the user's surface enters as
        # provided_edges and fixes WHERE THE CORNEA IS; the rigid passes then place each frame's pose
        # (translation + rotation) against the smooth dome implied by that surface. Those are different
        # quantities. And each pass is self-gated on its own roughness measure, so where the corrected surface
        # already leaves nothing to remove they are strict no-ops.
        #
        # TRADE-OFF, stated because it is real: the branch's original comment justified skipping them as "NO
        # iteration and NO axial-refine — so the corrected volume matches the scrub preview exactly". That
        # exactness is now traded away — the delivered volume may differ from the scrub preview by the rigid
        # pose correction. redetect_finish=False restores the old behaviour.
        # Runs BEFORE the user's GT (axial anchors, manual shifts, reviewer patches) so GT still wins last.
        #
        # ── AXIAL MOTION CORRECTION, on the CORRECTED volume ─────────────────────────────────────────────
        # The flatten above lands each frame on fit(provided_edges) — its OWN per-slice lateral quadratic —
        # independently per frame. That removes within-frame deviation but NOT frame-to-frame drift, because
        # provided_edges encodes the surface where it actually sits in the drifted acquisition. So a corrected
        # scan keeps its inter-frame drift: measured, across-frame sag 162 px vs an anatomical ~85 (sag ratio
        # 2.10 vs 0.44), which is the "corrected pane still isn't smooth" the reviewer reported.
        #
        # Two candidates were tried here. Plain AMC (redetect_amc) removes only ~6%: its degree-5 dome fit
        # absorbs a smooth drift as if it were anatomy, and it cannot use the drift-proof lateral geometry.
        # rigid_sagittal_motion_correct does — it predicts the true across-frame sag from the lateral arc
        # (which inter-frame motion cannot touch) and shifts each frame to it, GATED so it fires only when the
        # measured sag exceeds the sphere prediction by smc_excess_gate. Validated: complaint scan 166->34 px
        # (undulation improved 4.2->0.6, lateral held), two approved scans below the gate and left untouched.
        # Rigid (one depth shift per frame), so the B-scan is never deformed — the reviewer's exact spec.
        if provided_edges is not None and bool(p_all.get("rigid_sagittal_mc", True)):
            corrected, _smc = rigid_sagittal_motion_correct(corrected, params, workers=workers)
            info["rigid_sagittal_mc"] = _smc
        if provided_edges is not None and bool(p_all.get("redetect_amc", False)) \
                and bool(p_all.get("axial_motion_correct", True)):
            corrected, _amc2 = axial_motion_correct(corrected, params, workers=workers)
            info["axial_motion_correct_post"] = _amc2
        if bool(p_all.get("rigid_frame_warp", True)) and bool(p_all.get("redetect_finish", True)):
            # SINGLE SAGITTAL OPTIMISER (reviewer directive): rigid_height_refine / derotate / rigid_frame_refine each
            # do axial corrections toward their OWN aggregate targets (dome roughness, boundary), which pass their
            # self-gates while introducing a LOCALISED peripheral sagittal STEP — an over-rotated acquisition-edge
            # frame becomes a big depth jump at a far lateral (measured: 74 px at xn=-0.87, 4 px at centre). With
            # corrections_sole_sag the corrections path SKIPS them and lets sag_quad_align be the SOLE fine axial
            # correction — one per-frame translation+rotation judged directly by the sagittal edge (with a step
            # guard) — so every axial move answers to a smooth sagittal surface. corrections_sole_sag=False restores
            # the multi-stage passes.
            _sole = bool(p_all.get("corrections_sole_sag", True))
            if p_all.get("rigid_height_refine", True) and not _sole:
                # GIVE IT AN ACCURATE SURFACE instead of letting it re-detect. This stage fits a smooth dome
                # through the detected surface and shifts every frame onto it, so the surface it reads decides
                # where each frame lands — and detect_surface_all is wrong precisely on the frames the reviewer
                # corrected. Guided re-detection (windowed light search around the auto prior, then the same
                # post-passes) measured markedly closer to the reviewer's own anchors on the corrected volume:
                # median error 11.1 -> 3.0 px on a held-out corrected slice, 31% -> 69% of points within 5 px.
                # Best-effort: any failure falls back to the stage's own detection, i.e. the old behaviour.
                _rhr_det = None
                if bool(p_all.get("redetect_guided_finish", True)):
                    try:
                        _cs = reformat_to_sagittal(corrected).astype(np.float32)
                        _rhr_det = guided_redetect_all(_cs, detect_surface_all(_cs, p_all, workers=workers),
                                                       p_all)
                        del _cs
                    except Exception:  # noqa: BLE001
                        _rhr_det = None
                corrected, _rhr = rigid_height_refine(corrected, params, workers=workers, detect=_rhr_det)
                _rhr["surface"] = "guided" if _rhr_det is not None else "self-detected"
                info["rigid_height_refine"] = _rhr
            if p_all.get("rigid_frame_derotate", True) and not _sole:
                corrected, _rfd = rigid_frame_derotate(corrected, params, workers=workers)
                info["rigid_frame_derotate"] = _rfd
            if p_all.get("rigid_frame_refine", True) and not _sole:
                corrected, _rfr = rigid_frame_refine(corrected, params, workers=workers)
                info["rigid_frame_refine"] = _rfr
            # FINAL: flatten the sagittal (across-frame) surface to its per-lateral QUADRATIC with a rigid per-frame
            # translation + (clamped, gated) rotation — the reviewer's "corrected result should fit a nice smooth
            # quadratic". With corrections_sole_sag this is the ONLY fine axial pass; self-gated on the sagittal edge.
            if p_all.get("sag_quad_align", True):
                corrected, _sqa = sagittal_quad_align(corrected, params, workers=workers)
                info["sagittal_quad_align"] = _sqa
            # EDGE-FRAME GUARD: nudge the outermost acquisition-edge frames back onto the interior corneal curvature
            # (the faint FOV corner where the de-jitter dips them below the dome). Rigid, tapered, self-gated.
            if p_all.get("edge_guard", True):
                corrected, _efg = edge_frame_guard(corrected, params, workers=workers)
                info["edge_frame_guard"] = _efg
            # RECONCILE TO THE MANUAL LINE: keep the de-jitter in the reliable interior, but pin the faint FOV-corner
            # frames back to the user's drawn line (provided_edges) where the de-jitter's correction is unreliable.
            if provided_edges is not None and p_all.get("reconcile_line", True):
                corrected, _rec = reconcile_manual_line(corrected, provided_edges, params, workers=workers)
                info["reconcile_manual_line"] = _rec
        _axa = p_all.get("axial_anchors")   # AXIAL fix-tool GT (see main path) — applies on the fix-columns path too
        if _axa:
            corrected, _axinfo = apply_axial_surface_gt(corrected, _axa, params, workers=workers)
            info["axial_anchors"] = _axinfo
        # "Smooth to trusted slices" propagates the reviewer's edited + approved curves; it CONSUMES
        # corrected_edge_anchors as the edited GT, so the standalone warp is skipped in that mode (no double-apply).
        if p_all.get("corrected_smooth_align"):
            corrected, _csa = align_corrected_to_smooth(corrected, params, workers=workers)
            info["corrected_smooth_align"] = _csa
        elif p_all.get("corrected_edge_anchors"):   # standalone CORRECTED-edge fix-tool (guarded rigid axial move)
            corrected, _ceinfo = apply_sagittal_surface_gt(corrected, p_all.get("corrected_edge_anchors"), params, workers=workers)
            info["corrected_edge_anchors"] = _ceinfo
        ms = p_all.get("manual_shifts")
        if ms:
            corrected, n_ms = apply_manual_shifts(corrected, ms)
            info["manual_shifts"] = {"n_frames": int(n_ms)}
        corrected, n_crop = _apply_crop(corrected, p_all)
        if n_crop:
            info["crop"] = {"n_voxels": n_crop}
        write_volume_nifti(corrected, out_nifti, sp)
        # QA on the fix-columns path too. Like the surface-crop branch this hardcodes metrics=[] /
        # axial_metrics=[] (it is a single same-canvas warp, no iterate loop), so without this the
        # delivered volume carries no quality number at all. Latent today (0/308 are 'redetect') but
        # this is the write path every user-RESCUED scan takes, i.e. the Phase 1 loop's own output —
        # exactly the volumes whose quality most needs measuring.
        _fqa = measure_delivered_qa(corrected, params, workers=workers, path="redetect")
        if _fqa:
            info["final_qa"] = _fqa
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
    # RIGID-B-SCAN mode disables the per-column axial post-passes: axial_refine / axial_consistency / fbls /
    # surface_refine_2d / frame_edge_curve_snap all NUDGE individual lateral columns to smooth the detected B-scan
    # surface — they exist only to repair the DEFORMATION the per-column main warp injected. A rigid main warp
    # preserves the real instantaneous B-scan geometry, so those repairs would now BEND the true geometry. Off.
    _rigid = bool(p_all.get("rigid_frame_warp", False))
    if p_all.get("axial_refine", True) and not _rigid:
        corrected, ref = axial_refine_volume(corrected, params, workers=workers)
        info["axial_refine"] = ref
    # FIX axialcons: final AXIAL-consistency pass — the sagittal flatten corrects the 513 lateral columns
    # independently, so their shifts are inconsistent → lateral WAVINESS/spikes/notches visible only in the
    # AXIAL (B-scan) view. Per B-scan, apply a SMALL GATED per-column depth nudge onto a laterally-smoothed
    # surface. Strict no-op on an already-smooth scan (gate), so it can't regress good scans.
    if p_all.get("axial_consistency", True) and not _rigid:
        corrected, axc = axial_consistency_volume(corrected, params, workers=workers)
        info["axial_consistency"] = axc
    # FIX column-level edge errors: a final robust 2-D surface-refine pass (both lateral AND frame directions)
    # that pulls LOCAL patches where the edge detector locked a few px off (invisible to the lateral-only
    # axial_consistency) onto the smooth 2-D dome. Hard-gated on deviation → strict no-op on a smooth scan.
    if p_all.get("surface_refine_2d", True) and not _rigid:
        corrected, srf = surface_refine_2d(corrected, params, workers=workers)
        info["surface_refine_2d"] = srf
    # FIX jagged edge B-scans: lateral-smooth ONLY the first/last few (low-signal acquisition-edge) frames, whose
    # jagged border axial_consistency's gate leaves untouched. Gated to the boundary frames → no-op on the interior.
    if p_all.get("frame_boundary_smooth", True) and not _rigid:
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
    if p_all.get("frame_edge_curve_snap", True) and not _rigid:
        corrected, fcs = frame_edge_curve_snap(corrected, params, workers=workers)
        info["frame_edge_curve_snap"] = fcs
    # RIGID-mode sagittal cleanup: with the per-column post-passes gated off, the only remaining lever for a smooth
    # sagittal is a better PER-FRAME HEIGHT. Re-estimate + remove the residual per-frame jitter as a rigid depth shift
    # (self-gated, never worse; keeps the dome, no B-scan deformation). Runs ONLY under rigid (the deforming path
    # already smooths the sagittal by warping). Before the user-GT anchors so those still win.
    if _rigid and p_all.get("rigid_height_refine", True):
        corrected, rhr = rigid_height_refine(corrected, params, workers=workers)
        info["rigid_height_refine"] = rhr
    # SECONDARY rigid alignment: per-frame ROTATION (inter-frame eye torsion) — flattens the residual surface TILT
    # jitter that the depth-only rigid_height_refine leaves (a rigid rotation, so still zero B-scan deformation).
    # Runs after the axial alignment, self-gated. Rigid only; before user-GT anchors.
    if _rigid and p_all.get("rigid_frame_derotate", True):
        corrected, rfd = rigid_frame_derotate(corrected, params, workers=workers)
        info["rigid_frame_derotate"] = rfd
    # FINAL residual pass, driven by the ANTERIOR BOUNDARY instead of the DP surface + dome that drive all three
    # passes above. It is the only one that can reach the LOW-frequency acquisition-EDGE offset (rigid_height_refine
    # deliberately keeps only the high-freq part, capped at 8 px), which review found on scan after scan. Runs last
    # of the rigid passes, self-gated on the re-measured boundary, still before the user-GT anchors so those win.
    if _rigid and p_all.get("rigid_frame_refine", True):
        corrected, rfr = rigid_frame_refine(corrected, params, workers=workers)
        info["rigid_frame_refine"] = rfr
    # AXIAL fix-tool GT: apply the annotator's axial-plane surface corrections (drawn on an axial B-scan across
    # laterals) as a post-hoc additive per-frame warp onto the corrected result — reaches the apex/limbus defects
    # the sagittal fix-columns tool can't. Sticky + idempotent (re-diffs the drawn target vs the re-detected surface).
    _axa = p_all.get("axial_anchors")
    if _axa:
        corrected, _axinfo = apply_axial_surface_gt(corrected, _axa, params, workers=workers)
        info["axial_anchors"] = _axinfo
    # "Smooth to trusted slices" propagates the reviewer's edited + approved curves across the whole volume; it
    # CONSUMES corrected_edge_anchors as the edited GT, so the standalone per-frame warp below is skipped in that
    # mode (else the same edits apply twice).
    if p_all.get("corrected_smooth_align"):
        corrected, _csa = align_corrected_to_smooth(corrected, params, workers=workers)
        info["corrected_smooth_align"] = _csa
    # CORRECTED-RESULT sagittal fix-tool GT (STANDALONE, when not smooth-aligning): the reviewer dragged the
    # anterior surface on the CORRECTED result along the FRAME axis. A per-frame RIGID axial shift (+ tilt),
    # GUARDED by a cross-lateral check: applied only when the rigid move makes the frame more consistent across
    # laterals; declined when it can't. Sticky + idempotent.
    elif p_all.get("corrected_edge_anchors"):
        corrected, _ceinfo = apply_sagittal_surface_gt(corrected, p_all.get("corrected_edge_anchors"), params, workers=workers)
        info["corrected_edge_anchors"] = _ceinfo
    # #2 fix-columns drag-to-correct: apply the annotator's explicit per-frame manual depth nudges LAST,
    # so they override whatever the auto-correction left for those frames (manual ground truth wins).
    ms = p_all.get("manual_shifts")
    if ms:
        corrected, n_ms = apply_manual_shifts(corrected, ms)
        info["manual_shifts"] = {"n_frames": int(n_ms)}
    # Reviewer-accepted rigid patches (depth + rotation), applied after everything else because they are GT.
    mp = p_all.get("manual_patch")
    if mp:
        corrected, n_mp = apply_manual_patch(corrected, mp)
        info["manual_patch"] = {"n_frames": int(n_mp)}
    corrected, n_crop = _apply_crop(corrected, p_all)
    if n_crop:
        info["crop"] = {"n_voxels": n_crop}
    # MINIMAL lateral crop: drop only the extreme laterals whose corneal band the rigid rotation left partially
    # black, so SAM2 sees a FULL cornea in every sagittal slice (rigid path only — no rotation ⇒ no edge black).
    if _rigid and p_all.get("crop_incomplete_cornea", True):
        corrected, _cic = _crop_incomplete_cornea(corrected, params, workers=workers)
        if _cic[0] or _cic[1]:
            info["crop_incomplete_cornea"] = {"left": _cic[0], "right": _cic[1], "new_lateral": int(corrected.shape[2])}
    write_volume_nifti(corrected, out_nifti, sp)
    # DELIVERED-VOLUME QA (telemetry only, gates nothing): every mutation above — the rigid stages, the GT
    # anchors, the crops — happened AFTER the pass score was computed, so nothing has scored what was just
    # written. Measured AFTER the write, not before: QA costs ~3x the volume in peak RAM and a Linux OOM
    # kill cannot be caught by try/except, so it must not run while the output does not yet exist. Nothing
    # below mutates `corrected` (only `info`), so measuring here is equivalent.
    # reported_dev is metrics[best_pass] — the pass the delivered volume descends from — NOT metrics[-1]:
    # in 38 of 271 scans the last pass is not the best one (worst gap 3.30 px).
    _mets = info.get("metrics") or []
    _bp = info.get("best_pass")
    _rep = _mets[_bp] if (isinstance(_bp, int) and 0 <= _bp < len(_mets)) else (_mets[-1] if _mets else None)
    _fqa = measure_delivered_qa(corrected, params, workers=workers,
                                rhr_info=info.get("rigid_height_refine"),
                                path=info.get("stopped"), reported_dev=_rep)
    if _fqa:
        info["final_qa"] = _fqa
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


def _plot_series(series, colors, H=170, W=440):
    """Line plot of 1..k equal-domain series (per-frame profiles) on a dark canvas → HxWx3 uint8.
    Used for the derotate tilt-reference detail. Auto y-range with a zero baseline when it spans 0."""
    arrs = [np.asarray(s, float) for s in series]
    canvas = np.full((H, W, 3), 18, np.uint8)
    fin = [a[np.isfinite(a)] for a in arrs if np.isfinite(a).any()]
    if not fin:
        return canvas
    allv = np.concatenate(fin)
    lo, hi = float(allv.min()), float(allv.max())
    if hi <= lo:
        hi = lo + 1.0
    m = 0.12 * (hi - lo); lo -= m; hi += m
    if lo < 0.0 < hi:                                            # zero baseline
        yz = int((1.0 - (0.0 - lo) / (hi - lo)) * (H - 1))
        if 0 <= yz < H:
            canvas[yz, :] = (70, 70, 70)
    for a, col in zip(arrs, colors):
        Ln = len(a)
        if Ln < 2:
            continue
        for xi in range(W):
            fi = xi * (Ln - 1) / (W - 1); i0 = int(fi); i1 = min(Ln - 1, i0 + 1); t = fi - i0
            v = a[i0] * (1 - t) + a[i1] * t
            if not np.isfinite(v):
                continue
            yi = int((1.0 - (v - lo) / (hi - lo)) * (H - 1))
            for dy in (-1, 0, 1):
                if 0 <= yi + dy < H:
                    canvas[yi + dy, xi] = col
    return canvas


def _tilt_profile_and_ref(S, depth, p):
    """Per-frame lateral tilt b(f) + the ROBUST smooth-across-frames reference b_ref(f) — mirrors
    rigid_frame_derotate's reference, for the steps-viewer tilt detail. S = (lateral, frames)."""
    L, F = S.shape
    xc = (L - 1) / 2.0; xn = (np.arange(L) - xc) / (L / 2.0)
    b = np.full(F, np.nan)
    for f in range(F):
        col = S[:, f]; mk = (col > 1) & (col < depth - 1) & np.isfinite(col)
        if int(mk.sum()) >= 60:
            try:
                b[f] = np.polyfit(xn[mk], col[mk], 2)[1]
            except Exception:  # noqa: BLE001
                pass
    g = np.isfinite(b)
    if int(g.sum()) < max(8, F // 4):
        return b, b
    bi = np.interp(np.arange(F), np.where(g)[0], b[g])
    sig = float(p.get("rfd_ref_sigma", 9.0) or 0.0)
    ref = (ndimage.gaussian_filter1d(ndimage.median_filter(bi, size=5, mode="nearest"), sig, mode="nearest")
           if sig > 0 else bi)
    return b, ref


def _derotate_pivot_overlay(v, S, p):
    """Central axial B-scan (depth×lateral) with the detected surface (red) + the derotate rotation PIVOT
    (magenta ✚ = lateral centre, that frame's median surface depth). v=(frames,depth,lateral), S=(lateral,frames)."""
    try:
        F, depth, L = v.shape
        f = F // 2
        fr = v[f].astype(np.float32)                            # (depth, lateral)
        rgb = _gray_rgb(fr)
        col = S[:, f]                                           # depth per lateral
        for x in range(L):
            d = col[x]
            if np.isfinite(d) and 0 < d < depth:
                for dy in (-1, 0, 1):
                    if 0 <= int(d) + dy < depth:
                        rgb[int(d) + dy, x] = _C_RED
        mk = (col > 1) & (col < depth - 1) & np.isfinite(col)
        pd = int(np.median(col[mk])) if int(mk.sum()) > 40 else depth // 2
        pc = (L - 1) // 2
        for k in range(-8, 9):                                  # magenta cross at the pivot
            if 0 <= pd + k < depth:
                rgb[pd + k, pc] = _C_MAGENTA
            if 0 <= pc + k < L:
                rgb[pd, pc + k] = _C_MAGENTA
        return rgb
    except Exception:  # noqa: BLE001
        return None


def preprocess_steps(oct_path, params=None, volume_index=0, companion_txt=None,
                     bad_cols=None, workers=None, slice_index=None):
    """Faithful visual filmstrip of EVERY step the DEFAULT (rigid) preprocessing performs, on ONE sagittal
    slice (central or `slice_index`):
      A. Surface DETECTION — how the DP detector finds the anterior epithelium (what is being detected).
      B. Rigid INTER-FRAME correction — the ACTUAL default stages (axial motion-correct → intra-frame dewarp
         → rigid flatten → height-refine → derotate), each as the detected surface BEFORE→AFTER + the
         decision (applied?/metrics/gate), plus the derotate's smooth tilt-reference + rotation-pivot detail.
    Returns (n_slices, idx, [(label, rgb, kind, branch, lane)]). kind∈{stage,decision}; lane∈{full,dp,legacy}.
    NOTE: this re-runs the real stage functions (not the retired per-column warp) so preview == result."""
    p = {**DEFAULT_PARAMS, **(params or {})}
    vol = read_oct_zstack(oct_path, volume_index)
    sag = reformat_to_sagittal(vol)                             # (lateral, depth, frames)
    # Mirror preprocess: auto-tune the DP detector to THIS scan so the detection + correction match a real run.
    if p.get("auto_tune", True) and str(p.get("detector", "dp")).lower() != "legacy":
        try:
            _best, _ = auto_tune_detector(sag, p); p = {**p, **_best}
        except Exception:  # noqa: BLE001
            pass
    n = sag.shape[0]
    idx = n // 2 if slice_index is None else max(0, min(n - 1, int(slice_index)))
    sl = sag[idx].astype(np.float32)
    geom = companion_geometry(companion_txt, n_frames=sag.shape[2]) if companion_txt else {}
    depth_sp = float(geom.get("depth_spacing") or DEPTH_SPACING)
    frame_sp = float(geom.get("slice_spacing") or SLICE_SPACING)
    px_aspect = (frame_sp / depth_sp) if depth_sp > 0 else 1.0
    steps = []; _si = [0]

    def add(label, rgb, kind="stage", branch="", lane="full", resize=True):
        _si[0] += 1
        steps.append((f"{_si[0]}. {label}", _disp_resize(rgb, px_aspect) if resize else rgb, kind, branch, lane))

    def surf_overlay(volnow, color=_C_RED):
        sg = reformat_to_sagittal(volnow); j = max(0, min(sg.shape[0] - 1, idx))
        s2 = sg[j].astype(np.float32)
        try:
            e = _merged_side_edge(s2, p)
        except Exception:  # noqa: BLE001
            e = np.full(s2.shape[1], np.nan)
        return _draw_curve(_gray_rgb(s2), e, color)

    # ── A. SURFACE DETECTION (what is being detected) ──
    add(f"Original — sagittal slice {idx}/{n}", _gray_rgb(sl))
    img = ndimage.gaussian_filter(sl, sigma=(float(p.get("dp_sigma_depth", 3.0)), float(p.get("dp_sigma_frame", 1.2))))
    add("Detect · despeckle — anisotropic Gaussian (heavier along depth)", _gray_rgb(img))
    gy = np.gradient(img, axis=0); np.clip(gy, 0.0, None, out=gy)
    bw = max(2, int(p.get("dp_below", 24)))
    below = ndimage.uniform_filter1d(img, size=bw, axis=0, origin=-(bw // 2))
    score = gy * np.maximum(below - float(np.median(img)), 0.0)
    score = score / (score.max(axis=0, keepdims=True) + 1e-6)
    add("Detect · surface score — dark→bright gradient × bright-tissue-below", _gray_rgb(score * 255.0),
        kind="decision",
        branch="locks to the anterior epithelium (a real edge AND bright cornea just below), not internal layers or top speckle")
    dp_raw = _detect_surface_dp(sl, p)
    add("Detect · DP surface (red) — smooth MAX-score path", _draw_curve(_gray_rgb(sl), dp_raw, _C_RED),
        kind="decision",
        branch=f"dynamic-programming shortest smooth path (per-frame depth step ≤ {int(p.get('dp_max_jump', 10))}) + 3-point sub-voxel refine")
    try:
        legacy_edge = _legacy_surface(sl, p); merged = _merged_side_edge(sl, p)
        n_pulled = int(np.count_nonzero(np.abs(np.asarray(dp_raw) - np.asarray(merged)) > 0.5))
        add("Detect · scar-guard cross-check — red = final, green = legacy reference",
            _draw_curve(_draw_curve(_gray_rgb(sl), legacy_edge, _C_GREEN), merged, _C_RED),
            kind="decision", lane="full",
            branch=(f"where raw DP dives > {float(p.get('dp_scar_tol', 18)):.0f}px DEEPER than the legacy RANSAC surface "
                    f"(scar-lock signature), DP is re-detected near legacy → {n_pulled} frame(s) pulled back; "
                    "the final anterior edge = guarded DP"))
    except Exception:  # noqa: BLE001
        pass

    # ── B. RIGID INTER-FRAME CORRECTION (the actual default pipeline; re-runs the real stage functions) ──
    _rigid = bool(p.get("rigid_frame_warp", True))
    _blank = (np.zeros((sl.shape[0], sl.shape[1], 3), np.uint8) + 26)
    v = vol.copy()
    add("Rigid pipeline · input surface (red) — wavy from inter-frame eye motion", surf_overlay(v),
        kind="stage", lane="full",
        branch=("each axial B-scan is captured near-instantaneously — its internal geometry is TRUTH; only its "
                "RIGID pose between frames (depth + tilt) is corrected, never its shape"))
    try:
        v, amc = axial_motion_correct(v, p, workers=workers)
    except Exception:  # noqa: BLE001
        amc = {"applied": False}
    if amc.get("applied"):
        add("① Axial motion-correct — coarse per-frame DEPTH shift onto a robust 3-D dome", surf_overlay(v),
            kind="decision", branch=f"APPLIED — {amc.get('frames_adjusted', '?')} frames · motion σ={amc.get('motion_std', '?')}px · max shift {amc.get('max_shift', '?')}px")
    else:
        add("① Axial motion-correct — no-op", _blank, kind="decision", branch="SKIPPED — the dome fit found negligible per-frame depth motion")
    try:
        v, ifd = intra_frame_dewarp(v, p, workers=workers)
    except Exception:  # noqa: BLE001
        ifd = {"applied": False}
    if ifd.get("applied"):
        add("② Intra-frame dewarp — undo within-B-scan saccade distortion", surf_overlay(v),
            kind="decision", branch=f"APPLIED — saccade frames {ifd.get('saccade_frames', '?')} re-warped onto the motion-free reference")
    else:
        add("② Intra-frame dewarp — no-op", _blank, kind="decision", branch="SKIPPED — no frame exceeds the intra-frame saccade-distortion gate")
    try:
        v, mflat, _axf = smooth_volume(v, p, return_metric=True, workers=workers)
        add("③ Flatten — RIGID per-frame depth alignment (constant across lateral → no B-scan deformation)", surf_overlay(v),
            kind="decision", branch=f"APPLIED — per-frame shift onto the DP surface's smooth dome; boundary deviation {float(mflat):.2f}px (keep-best over passes)")
    except Exception as _e:  # noqa: BLE001
        add("③ Flatten — error", surf_overlay(v), kind="decision", branch=str(_e)[:70])
    if _rigid and p.get("rigid_height_refine", True):
        try:
            v, rhr = rigid_height_refine(v, p, workers=workers)
        except Exception:  # noqa: BLE001
            rhr = {"applied": False}
        if rhr.get("applied"):
            add("④ Rigid height-refine — remove residual per-frame height JITTER (rigid depth shift)", surf_overlay(v),
                kind="decision", branch=f"APPLIED — {rhr.get('frames_adjusted', '?')} frames · max jitter {rhr.get('max_jitter', '?')}px · sag roughness {rhr.get('rough_before', '?')}→{rhr.get('rough_after', '?')}")
        else:
            add("④ Rigid height-refine — no-op", _blank, kind="decision", branch="SKIPPED — self-gate: no roughness improvement")
    if _rigid and p.get("rigid_frame_derotate", True):
        b_before = b_ref = piv_png = None
        try:
            _Sb = detect_surface_all(reformat_to_sagittal(v), p, workers=workers)
            b_before, b_ref = _tilt_profile_and_ref(_Sb, v.shape[1], p)
            piv_png = _derotate_pivot_overlay(v, _Sb, p)       # pivot on the PRE-derotate frame + surface
        except Exception:  # noqa: BLE001
            pass
        try:
            v, rfd = rigid_frame_derotate(v, p, workers=workers)
        except Exception:  # noqa: BLE001
            rfd = {"applied": False}
        if rfd.get("applied"):
            add("⑤ Rigid derotate — level per-frame TILT (inter-frame torsion) by a rigid rotation", surf_overlay(v),
                kind="decision", branch=f"APPLIED — {rfd.get('frames_rotated', '?')} frames · max {rfd.get('max_deg', '?')}° · {rfd.get('iters', '?')} closed-loop iters · sag roughness {rfd.get('rough_before', '?')}→{rfd.get('rough_after', '?')}")
            if b_before is not None and b_ref is not None:
                add("⑤ · tilt reference — per-frame tilt b(f) [red] vs ROBUST smooth reference [blue]",
                    _plot_series([b_before, b_ref], [_C_RED, _C_BLUE]), kind="decision", lane="full", resize=False,
                    branch=("the reference is a SMOOTH-across-frames baseline (median-prefilter + gaussian σ≈9), NOT a single "
                            "global median — it keeps the slow real decentration/astigmatism and rotates away only the fast per-frame torsion (red−blue)"))
            if piv_png is not None:
                add("⑤ · rotation pivot (magenta ✚) on an axial B-scan — on the surface, lateral centre", piv_png,
                    kind="decision", lane="full", resize=False,
                    branch="the rotation axis is the surface point (not the array centre) → pure tilt-levelling, no spurious lateral shear; the angle itself is pivot-independent")
        else:
            add("⑤ Rigid derotate — no-op", _blank, kind="decision", branch="SKIPPED — self-gate: no roughness improvement")
    if _rigid and p.get("rigid_frame_refine", True):
        try:
            v, rfr = rigid_frame_refine(v, p, workers=workers)
        except Exception:  # noqa: BLE001
            rfr = {"applied": False, "reason": "error"}
        if rfr.get("applied"):
            add("⑥ Boundary refine — residual rigid depth shift, measured on the ANTERIOR BOUNDARY", surf_overlay(v),
                kind="decision",
                branch=(f"APPLIED — {rfr.get('frames_moved', '?')} frames · max {rfr.get('max_shift', '?')}px "
                        f"(edge {rfr.get('edge_shift_max', '?')}px) · boundary deviation "
                        f"{rfr.get('dev_rms_before', '?')}→{rfr.get('dev_rms_after', '?')}px; the EDGE frames are "
                        "referenced to the cornea's own local curvature, not to a polynomial that extrapolates there"))
        else:
            add("⑥ Boundary refine — no-op", _blank, kind="decision",
                branch=f"SKIPPED — {rfr.get('reason', 'self-gate: boundary already smooth')}")
    add("Final corrected surface (red)", surf_overlay(v), kind="decision", lane="full",
        branch="every frame's rigid pose (depth + tilt) corrected; the instantaneous B-scan geometry preserved throughout")
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
