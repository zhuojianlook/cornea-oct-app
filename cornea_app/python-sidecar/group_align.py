"""group_align — Step 4 "Aligned": group-wise 3-D alignment of a patient+eye group's replicate scans.

ELEMENTS 1-4 of the design (study wf_b197dcb9-d9a; prototype A = scratchpad/wf_align/A, prototype B =
wf_align/B): (1) MEMBER LOADING — every replicate as a `MemberData` (corrected volume, the served anterior line in
CORRECTED rows, a validity mask, the posterior, scar, spacing, provenance) plus the group rule and the reference
choice; (2) BAND EXTRACTION — the corneal band flattened to the served line (`BandData`) with the matcher's
features and `band_similarity`, the masked local-NCC map prototype A validated; (3) PAIR REGISTRATION —
`register_pair`: coarse 3-D masked FFT-NCC on the COARSE pyramids (the 240-row band at ×4×4 with the tissue mask,
prototype A's coarse configuration — a SEED, never a rejection) → per-frame fine registration (masked FFT-NCC
for the frame's lateral/depth shift, a per-lateral local-NCC depth search for the tilt) → projection onto the
admissible RIGID model (integer frame offset, per-B-scan axial shift + tilt, one lateral shift per live segment)
→ match quality after the transform (`warp_band` + band_similarity against the adjacent-frame ceiling) → the
FINE stage's verdict ('no_correspondence' when < 30 % of the overlapping frames measure or the match is poor on
both counts); (4) GROUP CONSISTENCY — `register_group` registers every member to the reference and reports the
transitivity of mov2→mov1→ref against mov2→ref. Nothing here writes a transform back to the store or the app.
Verified 2026-09-10 on COPIES of CS001_OS (v1 reference, scratchpad/impl_fix/acceptance_cs001.json, 22/22):
v2→v1 df −9, dx median −36.6 (segments −6.9 | −39.7 split at frame 42 = prototype A's saccade), a ∈ [−75.5,
−23.2]; v3→v1 df −7, dx −29.5, a ∈ [−38.3, +47.1]; coarse NCC 0.780 / 0.786 (df sharpness 0.037 / 0.035,
unimodal, seed = the full peak; the fine-band pyramid gave 0.698 / 0.725); matched 0.379 / 0.341 > 0.5 after the
transform = 1.125× / 1.013× the BAND-space adjacent-frame ceiling of v1 (0.3367 on the (−8, 120) posterior-capped
band — the acceptance bar is ≥ 1.0) and 1.72× / 1.55× prototype A's ORIGINAL-space ceiling (0.2202); measured
85/92 and 85/94 frames; transitivity a 0.82 px / b 1.68 px / dx 1.17 laterals rms over 82 frames (v3→v2 coarse
0.835); ~10-11 s per pair at OMP 3. A synthetic 20-lateral saccade at frame 20/40 (coarse peak 0.65 here, 0.49 on
the fine-band pyramid, i.e. below the retired 0.5 gate) registers exactly (segments −5.09 | +15.09, a −3.99).

ROUND 9 (2026-09-10, fix_r0 — the BETWEEN-SCAN geometry; reviewer directive "match the scan, not the noise"). The
engine above was developed on CS001_OS (near-identical replicate geometry) and refused 7 of 8 real pairs of P5_OS /
CS032_OS. Diagnosis (wf_bs/diag_q1, q2_lat_band, q3_truth, synth_plan): it matched SPECKLE where only STRUCTURE is
shared (every P5 / CS032 coarse surface a plateau), its search ranges and caps were within-scan priors (true offsets
up to 140 laterals / 48 frames, line-demanded tilts 53-200 px), the P5 bands were capped by a carried posterior that
was the bright-band bottom, CS032's members have two lateral spacings (11.50 / 12.28 µm), and the segment model
fragmented a slow real lateral wave. The fix, an explicit model (PairParams' docstring): E1 noise floor from the air
above the served line; E2 a carried posterior validated (MIN_POSTERIOR_THICKNESS) else trace-free; E3 clip-aware band
masks; E4 lateral resampling by the header ratio onto the reference's spacing (resample_member_lateral,
common_lateral_grid — a sampling-grid change recorded in PairResult.lateral_scale / lateral_offset_*); E5 two
features per band (STRUCTURE sigma 5 in-plane = the verdict; SPECKLE sigma 1.5 = the report), each judged against its
own same-scan ceiling (BAND_SPACE_CEILING 'rows_-8_120_struct_sigma5'), never smoothed along frames; E6 coarse ±300 /
±200 / ±50 with top-K seeds scored by a cheap fine pass; E7 the served lines' own tilt b_lines(dx) as the prior, the cap
on the residual, the per-lateral search re-centred on the fitted line, a fit that does not hold unmeasured; E8 the pose
angle at the coarse seed ('pose_beyond_frame_rigid', register_group's reference = most partners within pose_max_deg);
E9 dx served per frame through the step-aware fill, decidable frames only, single own wins corroborated (decisive /
run ≥ 3 / anchored, never a dome-ridge alias), 'dx_excursion' reported; E10 the speckle match reported and used as a
second witness (min_relative_match_speckle); E11 min_relative_match 0.75 on the structure feature (cross-patient null
0.60-0.67); E12 the witness rule's two alias holes. Verified on the COPIES (scratchpad/wf_bs/fix_r0/acceptance_out):
CS001 v2→v1 df −9 / dx −36 / rel_struct 1.01 / rel_speckle 1.32, v3→v1 −7 / −29 / 1.00 / 1.27, all six ordered pairs ok;
P5 reference v1_3 by the pose rule, v1 15° non-contributing, the sibling pairs at the landmark witness's offsets
(v1_2→v1_3 +22 / −110, v1_4→v1_3 +24 / −54) at rel_struct 0.94 / rel_speckle 0.95-1.06; CS032 (v1_3 / v1_4 resampled
×1.0678 onto 548 laterals) v1_3→v1 −10 / +2, v1_4→v1 +20 / +38, v1_2→v1 +35 / −43 at rel_struct 0.90-0.96; the
cross-patient controls refused. Every synthetic test runs the same engine at the synthetic structure scale
(tests/test_group_align.py SYN_SIGMA).

ROUND 9b (2026-09-11, the restarted round 0). The round-9 engine (d950feda) reproduced the real acceptance but regressed the
synthetic batteries (wf_fix3 strict per-frame refuting 49 → 476, probe_target 8 → 122 wrong rows): the STRUCTURE feature's
per-frame tilt carries a ~1 px mean / 3+ px tail error (the speckle engine: 0.3 px), its per-frame gate statistic saturates
and is degenerate along the dome ridge, and the E9 per-frame model let junk / alias frames through where the round-8 witness
rules no longer reached (a served value with no overlap was 'undecidable' and never arbitrated; a decisive single own win
needed no second witness; an axial-only alias run had no witness; ridge-guarded frames next to a saccade cut were
interpolated on the wrong side without a verdict). The fix keeps the STRUCTURE verdict and brings the SPECKLE feature in
where it demonstrably correlates — the reviewer's "final refinement": E10 (a) the fine stage re-runs the per-lateral depth
search on the speckle feature around the structure line and accepts it only where the speckle column NCC reaches
speckle_refine_min_col (PairResult.per_frame_speckle_ncc / speckle_refined; a matching frame 0.75-0.85, a junk frame 0.23;
tilt error max 3.2 → 0.7 px on the synthetic rows), (b) a structure-WEAK frame is re-measured on the speckle feature and
kept when that peak is sound and decisive ('speckle_rescued'), (c) a speckle _FrameScorer is the arbitration's SECOND
WITNESS: a decisive single / weak own win is served only when the speckle prefers it where it can speak
(quality['speckle_witness'] counts; silent on a frame whose speckle ceiling is under speckle_ceiling_min), with a
'decisive_gain' clause (own beats a passing served value by ≥ decisive_gain: a real 1-frame excursion next to a junk
neighbour) and the dome-ridge test before every decisive clause; E9(b) the evaluated-cells criterion no longer blocks the
arbitration (dec_arb: a 115-lateral wrong served shift on a last frame is arbitrated against its own 65); E12 an ISOLATED
axial-only run of ≤ 2 frames (an own win the fill had rejected, or a kept 2-frame step) needs the speckle witness against
the fill WITHOUT it, or — its statistic saturated — a measured neighbour agreeing in a and b, else 'axial_residual'; the
BOUNDARY pass runs at least once and offers discredited / ridge-guarded frames the other side's value. Verified on the
COPIES (scratchpad/wf_bs/fix_r0/acceptance_r9b) and the synthetic batteries (batteries2/): see fix_r0/notes.md.

ROUND 10 (2026-09-11, fix_r1 — the round-0 refutations of 547769ad; verify_refute_r0 / verify_regress_r0 / verify_code_r0).
R1 the pose is a REPORT ('pose_high') when the per-frame rigid model demonstrably fits (no verdict, rel_struct ≥ pose_fit_min_rel
0.85): a decentring of 150-250 laterals on a dome always implies 8.5-15° between the B-scan planes and a tilt about the FRAME
axis is exactly the per-frame model (the synthetic B:dx+170…+250 pairs register to 2 laterals / 2 px); 'pose_beyond_frame_rigid'
is the non-contributing verdict only when the model does not fit (P5_OS v1 at 15° / rel 0.74) and it supersedes the match-level
verdicts only (never an alias run's 'dx_residual'). R2 the pose is read at the SERVED transform (the served lines' tilt at each
frame's served dx; the seed's and the fine stage's angles stay in quality) — a garbage half-search seed on a df-bound pair read
16° where the served transform implies 0.02 px. R3 the tilt's PRECISION (the line's residual over the effective number of
independent local-NCC windows and the x-spread of the fitted laterals, PairResult.b_se): a frame whose tilt standard error exceeds
tilt_se_max (3 px) is served the step-aware Savitzky-Golay trend of every kept frame's tilt with its a re-fitted at the overlap's
centre (quality['tilt_low_precision']) — a half-overlap frame measured its tilt to ±5-13 px and served it (depth errors 5-10 px
over the overlap on 10-18 frames). R5 a served-line slope error under the cap is MEASURED: a weak frame (or every frame of a
weak-coarse pair) is re-matched with the moving B-scan pre-sheared by ±12 … max_tilt_px px half-span ('fine_shear_rescued'), the
per-lateral search is centred on the winning shear, and a poor or grid-edge-confined first line fit triggers a WIDE coarse depth
search (± max_tilt_px in 3 px, 'fine_wide_search') whose line also re-centres the fine search (a 35-px rotation was rms-rejected
on 92 of 94 frames). R6 the dx mode guard accepts a COHERENT ramp of ≥ seg_hold frames (consecutive frames within dx_step) as a
sustained mode: a real ±25-lateral wave over 40-60 frames is one mode (its steep parts were dropped from the knots and served a
straight line up to 5 laterals off). E7 REVISED: the tilt-residual cap is the dome-ridge ALIAS test — a frame whose dx is FAR from
its sound neighbours' (> seg_dx_step) and whose tissue tilt disagrees with the served lines AT ITS OWN dx by more than max_tilt_px
is a ridge alias (folded into ridge_alias: not a knot, interpolated, judged, named when it fails; CS032 v1_2→v1 frames 4-7 at
dx −140 / lines −96 were served ok because the lines' TREND had rejected those very frames' b_lines as outliers); an on-trend
frame's disagreement is measured and reported ('tilt_residual_high'): the still-clipped P5 members' reconstructed lines disagree
with the tissue by 24-76 px on whole pairs that match at 0.83-0.95, so a residual verdict refused three of the six sibling
directions; 'tilt_beyond_max' stays for |b| beyond the sanity cap. ALIAS HOLES: a ridge-guarded frame is judged under what it is
served and named when it fails (it was 'dead'); an INTERIOR isolated axial spike of ≤ 2 frames at a SATURATED statistic (both
kept neighbours disagreeing) is unwitnessed whatever the speckle vote; the ridge test's a-jump reads the served a at both
partnered neighbours (two junk neighbours let a 2-frame alias at dx −74 / a +41 through); a ridge-blocked single is named only
when its served value fails the gate proper (not merely grazes it: a pair at rel 0.91 was refused on a 0.299-vs-0.30 frame).
register_group: a transitivity pair whose df disagrees with the composition of the direct pairs by ≥ 2 frames is re-run seeded
at the composed df and kept when it registers as well (the group as a df witness on a flat df profile). Verified on the COPIES
(scratchpad/wf_bs/fix_r1/acceptance_out, out/driver_*_p8.log) and the synthetic batteries (fix_r1/batteries, refute/results):
see fix_r1/notes.md.

CONVENTIONS (the whole module, and every consumer):
  * volumes as nibabel loads them:  V[l, z, f] = (lateral 513, depth D, frame 101); D = 640 or canvas-padded.
  * served anterior line S[l, f] in CORRECTED rows (float, NaN = unknown).
  * spacing from the NIfTI header ≈ (0.0078 lateral, 0.00313 depth, 0.04 frame) mm.
  * display sagittal slice s ↔ raw lateral 512 − s (reporting only; `display_slice_to_lateral`).
  * shift convention everywhere: MOVING index + shift = REFERENCE index.
  * band row k ↔ depth offset (k + row0) from the served line; the served line sits at band row −row0.

Importable WITHOUT FastAPI: api_server / settings / metrics_export are only imported lazily, and the group rule is
copied here verbatim (tests/test_group_align.py pins the copy to api_server._group_members).

READ-ONLY on the store except two best-effort caches under <case>/border_cache (surface_corrected.npz for a legacy
scan's detection, applied_move.npz for a measured move — the api_server-compatible file), both skipped with
`write_cache=False`; a validation run on a COPY of a case folder writes them in the copy (manifest paths are
remapped to the copy, see `case_local_path`).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass, field, replace as _dc_replace
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.fft as sfft
import scipy.ndimage as ndi

import oct_preprocess as op

# ── prototype A constants (wf_align/A/bandlib.py, finelib.py) ────────────────────────────────────────────────────
BAND_ROWS_DEFAULT = (-8, 120)  # FINE band rows relative to the served line; A matched on 240 rows (posterior included)
R_SKIP = 8            # rows just under the surface excluded from MATCHING: the flat specular line carries no lateral
                      # information and pins the depth offset to 0 (prototype A)
LOCAL_WIN = (33, 25)  # (lateral, depth) window of the local NCC — the match metric of prototype A
# The COARSE stage has its own band + pyramid, prototype A's coarse_register configuration (stage2_coarse.py): the
# 240-row band (posterior included), block means ×4 lateral × ×4 depth, the Otsu TISSUE mask with NO posterior cap
# (A's Mmatch). Measured on the CS001_OS copies (scratchpad/impl_fix/coarse_variants.json, v1 reference): this gives
# coarse NCC 0.780 (v2) / 0.786 (v3) with df sharpness 0.037 / 0.035, against 0.698 / 0.725 for the fine band
# ((−8, 120), ×4×2, posterior-capped). The PLAIN band mask (no tissue cap: aqueous below the posterior kept) reaches
# 0.92 / 0.93 but only because the shared depth profile (bright stroma → dark aqueous) correlates at EVERY shift:
# df sharpness collapses to 0.007 (< coarse_min_sharp 0.03 → 'coarse_multimodal') and, un-normalised, v2→v1 picks a
# wrong peak (dx −10, df −7) — so "band" stays an option (PairParams.coarse_mask), not the default.
COARSE_ROWS_DEFAULT = (-8, 240)
COARSE_DS = (4, 4)    # coarse pyramid block size (lateral, depth) for the coarse FFT-NCC
COARSE_MASK_DEFAULT = "tissue"   # "tissue" (Otsu tissue rule, no posterior cap) | "band" (canvas & valid only)
# ── ROUND 9 (2026-09-10, between-scan geometry; reviewer directive "match the scan, not the noise") ────────────
# TWO features per band. Speckle is an independent realisation per acquisition: adjacent frames of ONE scan share
# it partially (that is the same-scan ceiling), replicate scans do NOT — so a between-scan transform must be read
# from STRUCTURE at scales above the speckle. The STRUCTURE feature is sqrt(I − nf) smoothed IN-PLANE with sigma 5
# px (≥ 3-5 speckle widths; a normalised convolution inside the match mask, so the bright specular line above
# R_SKIP never leaks into the matched rows) — used by the coarse pyramid, the fine stage, the arbitration's judge
# and the primary (verdict) quality; the SPECKLE feature is prototype A's sigma-1.5 image — used only for the
# reported speckle-scale match (and the optional refinement). NEITHER feature is ever smoothed or averaged along
# the FRAME axis (the denoising study: that erases the frame-specific information the frame offset is read from;
# every Gaussian on an (L, T, F) array here has sigma[2] == 0 — _assert_no_frame_smoothing). Each feature has its
# own local-NCC window (≈ 4 × its correlation length) and its own same-scan ceiling, so a match is always judged
# against the SAME-feature ceiling (a smoother image cannot inflate its relative match).
FEATURE_SIGMA_STRUCT = (5.0, 5.0)    # (lateral, depth) px — the structure feature's in-plane Gaussian
FEATURE_SIGMA_SPECKLE = (1.5, 1.5)   # prototype A's speckle-stabilised feature (the speckle report)
FEATURE_SIGMA = FEATURE_SIGMA_SPECKLE   # legacy name (the speckle feature)
LOCAL_WIN_STRUCT = (65, 49)          # (lateral, depth) local-NCC window of the structure feature
LOCAL_WIN_SPECKLE = (33, 25)         # prototype A's window (the speckle feature)
LOCAL_WIN = LOCAL_WIN_STRUCT         # the DEFAULT matcher window (band_similarity / pair_quality / the judge)
GAP_ROWS = 20         # a sustained dark run of this many rows below 0.7×Otsu ends a column's tissue (posterior)
MIN_TISSUE_ROWS = 40  # a column with fewer tissue rows carries no cornea (background / artifact)
# E2 — a CARRIED posterior line (border_cache/posterior_edges.npz) is accepted as the band's posterior cap only when
# its thickness over the served line is anatomically plausible: median ≥ MIN_POSTERIOR_THICKNESS px (= the tissue
# rule's own minimum MIN_TISSUE_ROWS + R_SKIP + GAP_ROWS), p10 ≥ MIN_POSTERIOR_P10 px and median ≥ half the
# trace-free thickness; otherwise the trace-free estimate is used ('trace_free_fallback'). P5_OS v1_2 / v1_3 carried
# the BRIGHT-BAND bottom (33-47 px median, p10 22) as their posterior — 53-59 % of the valid band was capped away and
# 43-56 % of the columns fell under min_tissue_rows (match_frac 0.18 / 0.14 vs 0.64 / 0.69 uncapped).
MIN_POSTERIOR_THICKNESS = 68.0
MIN_POSTERIOR_P10 = 48.0
NCC_THRESHOLDS = (0.3, 0.5, 0.7)
# Prototype A's adjacent-frame ceiling on case_cs001_os_v1 (same scan, frames 40 µm apart, local NCC in the band
# tissue below R_SKIP): gap 1 → mean 0.285, 45.1 % > 0.3, 22.0 % > 0.5, 6.5 % > 0.7; gap 2 → mean 0.182, 12.3 % > 0.5.
# Two DIFFERENT scans rigidly aligned reached 0.34-0.35 > 0.5 (1.56-1.59× that ceiling), so the ceiling is the
# speckle floor of the metric, not a target the pair engine must beat.
PROTOTYPE_A_CEILING = {"gap1": {"ncc_mean": 0.2848, "matched_frac_0.3": 0.4510, "matched_frac_0.5": 0.2202,
                                "matched_frac_0.7": 0.0653},
                       "gap2": {"ncc_mean": 0.1821, "matched_frac_0.3": 0.3031, "matched_frac_0.5": 0.1234,
                                "matched_frac_0.7": 0.0309}}
# A measured that ceiling in ORIGINAL space (unflattened adjacent frames). This module's local_ncc reproduces it
# there to 3 decimals (verify_e12, 2026-09-10: gap 1 mean 0.2844, 22.25 % > 0.5; gap 2 0.1818, 12.49 %). The SAME
# metric on the FLATTENED band (band_similarity(v1, v1, frame_offset=1)) is higher, because flattening removes the
# frame-to-frame surface slope that misaligns speckle by 1-2 px in original space — so a band-space match must be
# judged against the band-space ceiling below, never against PROTOTYPE_A_CEILING. The ceiling also depends on the
# MATCH MASK: with the member loaded by default (posterior="auto" → the trace-free posterior caps the mask; 49 % of
# the 240-row band matched) vs. posterior=False (the Otsu tissue rule alone keeps deeper, lower-SNR rows; 68 %).
# A pair's match % must be compared with the ceiling measured under the SAME band_rows and posterior setting.
BAND_SPACE_CEILING = {"rows_-8_240": {"gap1": {"ncc_mean": 0.3418, "matched_frac_0.3": 0.5576, "matched_frac_0.5": 0.2947,
                                               "matched_frac_0.7": 0.0870},
                                      "gap2": {"ncc_mean": 0.2563, "matched_frac_0.3": 0.4262, "matched_frac_0.5": 0.2001,
                                               "matched_frac_0.7": 0.0487}},
                      "rows_-8_120": {"gap1": {"ncc_mean": 0.3672, "matched_frac_0.3": 0.5998, "matched_frac_0.5": 0.3367,
                                               "matched_frac_0.7": 0.1027},
                                      "gap2": {"ncc_mean": 0.2775, "matched_frac_0.3": 0.4633, "matched_frac_0.5": 0.2307,
                                               "matched_frac_0.7": 0.0571}},
                      # the same two bands with posterior=False (no cap; Otsu tissue rule only)
                      "rows_-8_240_no_posterior": {"gap1": {"ncc_mean": 0.3057, "matched_frac_0.5": 0.2490}},
                      "rows_-8_120_no_posterior": {"gap1": {"ncc_mean": 0.3558, "matched_frac_0.5": 0.3237}},
                      # ROUND 9: the STRUCTURE feature (sigma 5, normalised convolution inside the match mask, window
                      # 65 × 49) on the same (−8, 120) posterior-capped band of case_cs001_os_v1 — the VERDICT ceiling
                      # (measured on the copies 2026-09-10 by the implementer; pinned by the verifier). The speckle
                      # entries above stay the SPECKLE report's pin (window 33 × 25, sigma 1.5).
                      "rows_-8_120_struct_sigma5": {"gap1": {"ncc_mean": 0.7405, "matched_frac_0.3": 0.9529,
                                                             "matched_frac_0.5": 0.8769, "matched_frac_0.7": 0.6936}}}

# ── the source-filename rule (metrics_export._NAME_RE, copied so the module has no sidecar-settings import) ────
# preprocessed_CS001_14145_3D Cornea_OD_2024-07-11 (2)_0.dcm  /  P5_10861_3D Cornea_OS_2022-09-27_10.43.37_1.OCT
_NAME_RE = re.compile(
    r"(?P<pid>[A-Za-z]+\d+)_(?P<dev>\d+)_.*?_(?P<eye>O[DS])_(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:[ _]*\((?P<variant>\d+)\))?",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Group rule (api_server._group_id_norm / _group_members, copied verbatim; pinned by the test suite)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def parse_case_meta(source: str | None) -> dict:
    """patient_id / eye / date / variant from the source filename — metrics_export.parse_case_meta when the
    sidecar package is importable (it pulls settings/orchestration), else the identical regex copied above."""
    try:
        import metrics_export  # noqa: WPS433 — lazy: pulls settings + orchestration
        return metrics_export.parse_case_meta(source)
    except Exception:  # noqa: BLE001 — stand-alone use (no sidecar settings): the copied rule
        meta = {"patient_id": "", "eye": "", "date": "", "variant": ""}
        if not source:
            return meta
        m = _NAME_RE.search(Path(source).name)
        if m:
            meta.update(patient_id=m.group("pid").upper(), eye=m.group("eye").upper(),
                        date=m.group("date"), variant=m.group("variant") or "")
        return meta


def group_id_norm(gid: str) -> str:
    """Normalise a patient+eye group id ("CS001_OD", "cs001|od", "CS001 OD") to lowercase "patient_eye"
    (api_server._group_id_norm, verbatim)."""
    return re.sub(r"[\s|/:,+]+", "_", str(gid or "").strip()).strip("_").lower()


def group_key(manifest: dict | None) -> tuple[str, str, str] | None:
    """(normalised group id, patient, EYE) of ONE case from its manifest — the sidebar's grouping rule
    (api_server._group_members): manifest patient_id / eye first, else the source filename (oct_source, then
    companion_txt) parsed by parse_case_meta; consensus cases (manifest.consensus_cases), cases without a source
    and cases with an unknown eye never join a group → None."""
    m = manifest or {}
    if not m or m.get("consensus_cases"):
        return None
    src = m.get("oct_source") or m.get("companion_txt")
    if not src:
        return None
    meta: dict = {}
    try:
        meta = parse_case_meta(src)
    except Exception:  # noqa: BLE001
        pass
    pid = str(m.get("patient_id") or meta.get("patient_id") or "").strip()
    eye = str(m.get("eye") or meta.get("eye") or "").strip()
    if not pid or not eye or eye == "?":
        return None
    return group_id_norm(f"{pid}_{eye}"), pid, eye.upper()


def read_manifest(case_dir: str | os.PathLike) -> dict:
    p = Path(case_dir) / "manifest.json"
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001
        return {}


def group_members(group_id: str, root: str | os.PathLike) -> tuple[list[str], dict]:
    """Case ids under `root` (a cases/ directory) that resolve to the patient+eye group `group_id` — the SAME
    rule api_server._group_members applies to settings.CASES_ROOT, so the group the reviewer sees in the sidebar
    is the group that gets aligned. Sorted by case id; `*_consensus` dirs and unreadable manifests skipped.
    Returns (members, {"patient", "eye"})."""
    want = group_id_norm(group_id)
    members: list[str] = []
    key: dict = {"patient": None, "eye": None}
    root = Path(root)
    if not want or not root.exists():
        return members, key
    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.endswith("_consensus"):
            continue
        gk = group_key(read_manifest(child))
        if gk is None or gk[0] != want:
            continue
        members.append(child.name)
        key = {"patient": gk[1], "eye": gk[2]}
    return members, key


def default_cases_root() -> Path:
    """settings.CASES_ROOT (lazy import; honours CORNEA_DATA_DIR) — the sidecar's own case store."""
    import settings  # noqa: WPS433
    return Path(settings.CASES_ROOT)


def resolve_case_dir(case: str | os.PathLike, root: str | os.PathLike | None = None) -> Path:
    """A case DIRECTORY (has manifest.json) as given, else <root or settings.CASES_ROOT>/<case id>."""
    p = Path(case)
    if p.is_dir() and (p / "manifest.json").exists():
        return p
    base = Path(root) if root is not None else default_cases_root()
    return base / str(case)


def case_local_path(case_dir: Path, path: str | None, cid: str | None = None) -> Path | None:
    """Resolve a manifest path (absolute, into the store) against `case_dir`, so a COPY of a case folder reads
    ITS OWN files: (1) the path itself when it lies inside case_dir; (2) the same tail after the '/<cid>/' segment
    under case_dir when that file exists (a copy made elsewhere); (3) the original path when it exists (read-only
    fallback); else None."""
    if not path:
        return None
    p = Path(path)
    cd = Path(case_dir).resolve()
    try:
        if p.resolve().is_relative_to(cd) and p.exists():
            return p
    except Exception:  # noqa: BLE001
        pass
    names = [n for n in (cid, cd.name) if n]
    parts = p.parts
    for n in names:
        if n in parts:
            i = len(parts) - 1 - parts[::-1].index(n)
            cand = cd.joinpath(*parts[i + 1:])
            if cand.exists():
                return cand
    if len(parts) >= 2:
        cand = cd.joinpath(*parts[-2:])           # <case>/input/<file>
        if cand.exists():
            return cand
    return p if p.exists() else None


def display_slice_to_lateral(s: int, n_lateral: int = 513) -> int:
    """Display sagittal slice s (what the reviewer sees) ↔ raw lateral index n_lateral − 1 − s (512 − s)."""
    return int(n_lateral) - 1 - int(s)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Element 1 — MemberData
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class MemberData:
    """One replicate scan of a patient+eye group, in CORRECTED space.

    volume     (L, D, F) float32 — manifest.input_volume, the pipeline OUTPUT (never corrected_volume).
    served     (L, F) float64 — the served anterior line in CORRECTED rows (NaN = unknown), the corrected-pane rule
               (see load_member); `served_source` names the file it came from, `move_source` how it was carried.
    valid      (L, F) bool — served finite AND inside the canvas (2 < row < D−3) AND not in a reviewer crop band
               (oct_params.crop_bands via oct_preprocess._artifact_mask) AND the A-scan is not zero-filled.
    posterior  (L, F) float64 in corrected rows or None; `posterior_source` = "posterior_edges" (the run's served
               bottom edge, carried like the top) | "trace_free" (per-A-scan threshold crossing, wf_align
               read_groups.posterior_trace_free) | None.
    scar       (L, F) float fraction or None; `scar_source` = "labelmap" (segmentation/<cid>_corrected.nii.gz, label
               2 among labels 1|2 per A-scan) | "proxy" (attach_scar_proxy from the band) | None.
    spacing    (3,) mm per voxel from the NIfTI header (lateral, depth, frame).
    canvas_pad top rows the pipeline added to the canvas (0 on a 640-row scan); `bottom_pad` likewise.
    lateral_dx (F,) per-frame lateral translation the run applied last (applied_move.npz; zeros when none).
    move       (L, F) the per-frame rigid axial move raw → corrected (corrected = raw + canvas_pad + move) or None.
    surface_crop_frames  frames whose anterior was clipped at the top of the raw window (oct_params); valid there
               (the band below the served estimate is tissue) — the pair engine may down-weight them.
    meta       {patient, eye, vetted, rejected_unfixable, difficult, ...}; `timings` seconds per step."""
    cid: str
    case_dir: Path
    group: str | None
    volume: np.ndarray
    served: np.ndarray
    valid: np.ndarray
    spacing: np.ndarray
    canvas_pad: int = 0
    bottom_pad: int = 0
    lateral_dx: np.ndarray | None = None
    move: np.ndarray | None = None
    move_source: str | None = None
    served_source: str | None = None
    posterior: np.ndarray | None = None
    posterior_source: str | None = None
    scar: np.ndarray | None = None
    scar_source: str | None = None
    surface_crop_frames: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    crop_band_mask: np.ndarray | None = None
    meta: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(v) for v in self.volume.shape)  # type: ignore[return-value]

    @property
    def valid_area(self) -> int:
        return int(self.valid.sum())

    def dome_apex(self) -> tuple[float, float]:
        """(lateral, frame) of the dome apex: the minimum of a quadric fitted to the served line over the valid
        cells (the shallowest point, in corrected rows). Falls back to the argmin of the fitted quadric on the
        grid when the fit is not convex in both axes."""
        return _dome_apex(self.served, self.valid)


def _dome_apex(served: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    L, F = served.shape
    ok = valid & np.isfinite(served)
    lc = (np.arange(L) - (L - 1) / 2.0) / max(1.0, (L - 1) / 2.0)
    fc = (np.arange(F) - (F - 1) / 2.0) / max(1.0, (F - 1) / 2.0)
    LL, FF = np.meshgrid(lc, fc, indexing="ij")
    if ok.sum() < 12:
        return (L - 1) / 2.0, (F - 1) / 2.0
    A = np.stack([np.ones(ok.sum()), LL[ok], FF[ok], LL[ok] ** 2, FF[ok] ** 2, LL[ok] * FF[ok]], 1)
    c = np.linalg.lstsq(A, served[ok], rcond=None)[0]
    det = 4 * c[3] * c[4] - c[5] ** 2
    if c[3] > 0 and det > 0:
        l0 = (-2 * c[4] * c[1] + c[5] * c[2]) / det
        f0 = (-2 * c[3] * c[2] + c[5] * c[1]) / det
        if -1.5 <= l0 <= 1.5 and -1.5 <= f0 <= 1.5:
            return float(l0 * (L - 1) / 2.0 + (L - 1) / 2.0), float(f0 * (F - 1) / 2.0 + (F - 1) / 2.0)
    Q = c[0] + c[1] * LL + c[2] * FF + c[3] * LL ** 2 + c[4] * FF ** 2 + c[5] * LL * FF
    Q = np.where(ok, Q, np.inf)
    i, j = np.unravel_index(int(np.argmin(Q)), Q.shape)
    return float(i), float(j)


def _load_nifti(path: Path):
    import nibabel as nib
    img = nib.load(str(path))
    return img, np.asarray(img.dataobj).astype(np.float32)


def _stamp(path: Path) -> tuple[int, int]:
    st = path.stat()
    return int(st.st_mtime_ns), int(st.st_size)


def run_applied_move(mv_path: Path, work: Path, shape) -> tuple[np.ndarray, int, int, np.ndarray, list] | None:
    """(move (L, F) float64, canvas_pad, bottom_pad, lateral_dx (F,), stage names) from a RUN-written
    border_cache/applied_move.npz (oct_preprocess.write_applied_move_cache, source="run") that belongs to the
    corrected volume `work`: the stamp recorded at write time (st_mtime_ns + st_size of the delivered NIfTI) must
    equal work.stat() now — api_server._run_applied_move's rule. None for anything else (absent, a measured cache,
    a stale run move, a wrong shape, a malformed file)."""
    if not mv_path.exists():
        return None
    try:
        z = np.load(mv_path, allow_pickle=False)
        if "source" not in z.files or str(z["source"]) != "run":
            return None
        mt, sz = _stamp(work)
        if int(z["stamp_mtime_ns"]) != mt or int(z["stamp_size"]) != sz:
            return None
        mv = np.asarray(z["move"], dtype=np.float64)
        if mv.shape != tuple(int(v) for v in shape) or not np.isfinite(mv).all():
            return None
        try:
            stages = [str(s.get("stage")) for s in json.loads(str(z["stages"])) if isinstance(s, dict)] \
                if "stages" in z.files else []
        except Exception:  # noqa: BLE001
            stages = []
        F = int(shape[1])
        ldx = (np.asarray(z["lateral_dx"], dtype=np.float64).reshape(-1) if "lateral_dx" in z.files
               else np.zeros(F))
        if ldx.size != F:
            ldx = np.zeros(F)
        pad = int(z["canvas_pad"]) if "canvas_pad" in z.files else 0
        bpad = int(z["bottom_pad"]) if "bottom_pad" in z.files else 0
        return mv, pad, bpad, ldx, stages
    except Exception:  # noqa: BLE001
        return None


def measured_applied_move(case_dir: Path, work: Path, raw_path: Path, vol: np.ndarray, params: dict,
                          manifest: dict, write_cache: bool = True) -> tuple[np.ndarray, int, dict] | None:
    """The per-frame rigid move raw → corrected MEASURED detector-free (oct_preprocess.measure_applied_move,
    whole-A-scan NCC per lateral + a robust line per frame), api_server._corrected_prior_surface's measured
    branch: the raw volume padded at the TOP by the canvas pad (D − raw depth), the lag window widened to cover
    the pad (+16) and the run's deepest tissue move, the result cached to border_cache/applied_move.npz with the
    api_server-compatible key ("<raw mtime_ns>:<work mtime_ns>:<max_lag>:<min_cols>", source="measured"; a valid
    RUN move is never overwritten because this branch is only reached when none exists). Returns
    (move (L, F), canvas_pad, info) or None when the raw volume is missing / not the same canvas."""
    if not raw_path.exists():
        return None
    L, D, F = (int(v) for v in vol.shape)
    import nibabel as nib
    rimg = nib.load(str(raw_path))
    rshape = tuple(int(v) for v in rimg.shape[:3])
    if rshape[0] != L or rshape[2] != F or rshape[1] > D:
        return None
    pad = int(D - rshape[1])
    p = {**op.DEFAULT_PARAMS, **(params or {})}
    lag = max(int(p.get("corrected_prior_max_lag", 48) or 48), pad + 16)
    try:
        tm = ((manifest.get("oct_iter") or {}).get("tissue_motion") or {})
        sr = tm.get("shift_range") if tm.get("applied") else None
        if sr and len(sr) == 2 and float(sr[1]) > 0:
            lag = max(lag, pad + 16 + int(math.ceil(float(sr[1]))))
    except Exception:  # noqa: BLE001
        pass
    key = f"{raw_path.stat().st_mtime_ns}:{work.stat().st_mtime_ns}:{p.get('corrected_prior_max_lag')}:{p.get('corrected_prior_min_cols')}"
    mv_path = case_dir / "border_cache" / "applied_move.npz"
    if mv_path.exists():
        try:
            z = np.load(mv_path, allow_pickle=False)
            if "key" in z.files and str(z["key"]) == key and z["move"].shape == (L, F):
                return (np.asarray(z["move"], dtype=np.float64), pad,
                        {"source": "measured", "cached": True, "n_extrapolated": int(z["n_extrapolated"])})
        except Exception:  # noqa: BLE001
            pass
    rv = np.asarray(rimg.dataobj).astype(np.float32)
    if pad > 0:
        rv = np.concatenate([np.zeros((L, pad, F), dtype=rv.dtype), rv], axis=1)
    r = op.measure_applied_move(rv, vol, {**p, "corrected_prior_max_lag": lag})
    move = np.asarray(r["move"], dtype=np.float64)
    info = {"source": "measured", "cached": False, "n_extrapolated": len(r.get("extrapolated_frames") or []),
            "extrapolated_frames": [int(v) for v in (r.get("extrapolated_frames") or [])], "max_lag": lag}
    if write_cache:
        try:
            mv_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = mv_path.with_name("applied_move.tmp.npz")
            np.savez_compressed(tmp, key=np.array(key), move=move.astype(np.float32),
                                n_extrapolated=np.array(info["n_extrapolated"]), source=np.array("measured"))
            os.replace(tmp, mv_path)
        except Exception:  # noqa: BLE001 — a read-only store just re-measures next time
            pass
    return move, pad, info


def detect_corrected_surface(case_dir: Path, vol: np.ndarray, work: Path, dp_params: dict,
                             workers: int = 3, write_cache: bool = True) -> tuple[np.ndarray, dict]:
    """LAST fallback (legacy scans with no border_cache, e.g. case_cs001_os_v1): the app's own detector on the
    CORRECTED volume — oct_preprocess.detect_surface_all with the manifest's dp_* params (17-19 s at workers=3 on
    a 513×640×101 volume; prototype A stage 1). Cached to border_cache/surface_corrected.npz keyed by the
    input_volume stamp (mtime_ns + size) and the dp params; a copy of the case folder caches in the copy."""
    bc = case_dir / "border_cache"
    cache = bc / "surface_corrected.npz"
    mt, sz = _stamp(work)
    sig = json.dumps({k: dp_params[k] for k in sorted(dp_params)}, sort_keys=True, default=str)
    L, D, F = (int(v) for v in vol.shape)
    if cache.exists():
        try:
            z = np.load(cache, allow_pickle=False)
            if (int(z["stamp_mtime_ns"]) == mt and int(z["stamp_size"]) == sz and str(z["params_sig"]) == sig
                    and z["surface"].shape == (L, F)):
                return np.asarray(z["surface"], dtype=np.float64), {"cached": True, "detect_s": float(z["detect_s"])}
        except Exception:  # noqa: BLE001
            pass
    t0 = time.time()
    S = np.asarray(op.detect_surface_all(vol, params=dict(dp_params), workers=workers), dtype=np.float64)
    dt = time.time() - t0
    if write_cache:
        try:
            bc.mkdir(parents=True, exist_ok=True)
            tmp = bc / "surface_corrected.tmp.npz"
            np.savez_compressed(tmp, surface=S.astype(np.float32), stamp_mtime_ns=np.array(mt, dtype=np.int64),
                                stamp_size=np.array(sz, dtype=np.int64), params_sig=np.array(sig),
                                source=np.array("detect_surface_all"), detect_s=np.array(dt))
            os.replace(tmp, cache)
        except Exception:  # noqa: BLE001
            pass
    return S, {"cached": False, "detect_s": dt}


def posterior_trace_free(vol: np.ndarray, ant: np.ndarray, valid: np.ndarray, tmax: int = 400) -> np.ndarray:
    """Trace-free posterior crossing per (l, f) (wf_align/read_groups.posterior_trace_free): the first depth
    ≥ ant+15 where the depth-smoothed (σ=3) A-scan falls below bg + 0.3·(band_peak − bg), band_peak = p90 of rows
    ant+3..ant+60, bg = p10 of rows ant+260..ant+450. Returns THICKNESS in rows, NaN where not measurable (a
    shallow canvas has no deep background window → all NaN → no posterior)."""
    L, D, F = vol.shape
    thick = np.full((L, F), np.nan)
    idx = np.arange(D)[:, None]
    for l in range(L):
        okf = valid[l] & np.isfinite(ant[l])
        if not okf.any():
            continue
        sl = ndi.gaussian_filter1d(vol[l], 3.0, axis=0)
        ai = np.clip(np.rint(np.nan_to_num(ant[l], nan=0.0)), 0, D - 1).astype(int)
        rel = idx - ai[None, :]
        band = np.where((rel >= 3) & (rel <= 60), sl, np.nan)
        deep = np.where((rel >= 260) & (rel <= 450), sl, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)      # all-NaN columns → NaN, by design
            peak = np.nanpercentile(band, 90, axis=0)
            bg = np.nanpercentile(deep, 10, axis=0)
        thr = bg + 0.3 * (peak - bg)
        below = (sl < thr[None, :]) & (rel >= 15) & (rel <= tmax)
        has = below.any(axis=0)
        first = np.argmax(below, axis=0)
        t = (first - ai).astype(float)
        t[~has] = np.nan
        t[~okf] = np.nan
        t[~np.isfinite(peak) | ~np.isfinite(bg) | (peak <= bg + 1)] = np.nan
        thick[l] = t
    return thick


def load_member(case: str | os.PathLike, root: str | os.PathLike | None = None, *, posterior: str | bool = "auto",
                scar: bool = True, workers: int = 3, write_cache: bool = True) -> MemberData:
    """Load one replicate as a MemberData (design element 1).

    `case` is a case directory (or a COPY of one) or a case id under `root` (default settings.CASES_ROOT).

    THE SERVED LINE (corrected rows) — the corrected-pane rule, api_server._corrected_prior_surface:
      1. border_cache/placed_edges.npz['surface'] when its mtime ≥ provided_edges.npz's, else
         provided_edges.npz['surface'] — RAW rows — carried by  + canvas_pad + move, where the move is the RUN
         move (border_cache/applied_move.npz, source="run", stamp == the delivered NIfTI; run_applied_move) …
      2. … else the MEASURED move (measured_applied_move: baseline.npz is the served line when there is no
         correction curve; oct_preprocess.measure_applied_move raw=_raw_border vs corrected, cached).
      3. Legacy scans with no border_cache (case_cs001_os_v1): oct_preprocess.detect_surface_all on the corrected
         volume with the manifest's dp_* params, cached to border_cache/surface_corrected.npz (detect_corrected_surface).
    Prototype A (wf_align/A/stage1_surfaces.py) validated the chain on CS001_OS v2/v3: detect-on-corrected vs
    baseline + measured move agree to median |diff| 0.46-0.68 px (p90 ≈ 1-2 px).

    `posterior`: "auto" → posterior_edges.npz carried like the top when the run wrote one AND its thickness is
    plausible (E2: median ≥ MIN_POSTERIOR_THICKNESS, p10 ≥ MIN_POSTERIOR_P10, median ≥ half the trace-free
    thickness — meta['posterior_check']), else the trace-free estimate (posterior_trace_free, ~2 s;
    posterior_source 'trace_free_fallback' when a carried line was rejected); False → skip. `scar`: read
    segmentation/<cid>_corrected.nii.gz when it exists and matches the volume grid. `write_cache=False` never
    touches border_cache."""
    t_all = time.time()
    case_dir = resolve_case_dir(case, root)
    m = read_manifest(case_dir)
    if not m:
        raise FileNotFoundError(f"no manifest.json under {case_dir}")
    cid = str(m.get("case_id") or case_dir.name)
    timings: dict = {}
    gk = group_key(m)
    group = gk[0] if gk else None
    work = case_local_path(case_dir, m.get("input_volume"), cid)
    if work is None:
        raise FileNotFoundError(f"{cid}: manifest.input_volume not found ({m.get('input_volume')})")
    t0 = time.time()
    img, vol = _load_nifti(work)
    timings["load_volume"] = time.time() - t0
    L, D, F = (int(v) for v in vol.shape)
    spacing = np.asarray(img.header.get_zooms()[:3], dtype=np.float64)
    params = dict(m.get("oct_params") or {})
    dp_params = {k: v for k, v in params.items() if k.startswith("dp_")}
    bc = case_dir / "border_cache"
    raw_path = case_dir / "input" / "_raw_border.nii.gz"

    # ── served line chain ───────────────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    served = None; served_source = None; move = None; move_source = None; move_info: dict = {}
    canvas_pad = 0; bottom_pad = 0; lateral_dx = np.zeros(F)
    raw_surface = None
    prov, placed, base = bc / "provided_edges.npz", bc / "placed_edges.npz", bc / "baseline.npz"
    src = None
    if prov.exists():
        src = prov
        try:
            if placed.exists() and placed.stat().st_mtime_ns >= prov.stat().st_mtime_ns:
                src = placed
        except OSError:
            pass
    elif base.exists():
        src = base
    if src is not None:
        try:
            raw_surface = np.asarray(np.load(src, allow_pickle=False)["surface"], dtype=np.float64)
            if raw_surface.shape != (L, F):
                raw_surface = None
            else:
                served_source = src.name
        except Exception:  # noqa: BLE001
            raw_surface = None
    if raw_surface is not None:
        run = run_applied_move(bc / "applied_move.npz", work, (L, F))
        if run is not None:
            move, canvas_pad, bottom_pad, lateral_dx, stages = run
            move_source = "run"; move_info = {"source": "run", "stages": stages}
        else:
            meas = measured_applied_move(case_dir, work, raw_path, vol, params, m, write_cache=write_cache)
            if meas is not None:
                move, canvas_pad, move_info = meas
                move_source = "measured"
        if move is not None:
            served = np.asarray(op.carry_correction_curve(raw_surface + float(canvas_pad), move, D), dtype=np.float64)
    if served is None and raw_path.exists():
        # 2b. NO cached baseline (the scan was never opened in the border editor): detect on the RAW volume — the
        #     same surface the app's step-3 view serves (border_cache/baseline.npz is exactly that, cached) — and
        #     carry it with the measured move. Detecting on the CORRECTED volume instead (step 3 below) agrees in
        #     the median but fails locally (cs001_od_v4: median 0.7 px, max 183 px), which is the "bad border" the
        #     reviewer saw in the scrub. Same cost (one detection), and the cache is written for the app to reuse.
        try:
            import nibabel as nib
            raw_arr = np.ascontiguousarray(np.asanyarray(nib.load(str(raw_path)).dataobj))
            base_surf = np.asarray(op.detect_surface_all(raw_arr, dict(dp_params), workers=workers), dtype=np.float64)
            if base_surf.shape == (L, F):
                raw_surface = base_surf
                served_source = "baseline (detected on the raw volume)"
                if write_cache:
                    try:
                        import os as _os
                        bc.mkdir(parents=True, exist_ok=True)
                        tmp = bc / "baseline.tmp.npz"
                        np.savez_compressed(tmp, surface=base_surf.astype(np.float32),
                                            raw_mtime=float(_os.path.getmtime(raw_path)),
                                            params_sig=str(m.get("_baseline_params_sig") or "group_align"))
                        _os.replace(tmp, bc / "baseline.npz")
                    except OSError:
                        pass
                meas = measured_applied_move(case_dir, work, raw_path, vol, params, m, write_cache=write_cache)
                if meas is not None:
                    move, canvas_pad, move_info = meas
                    move_source = "measured"
                    served = np.asarray(op.carry_correction_curve(raw_surface + float(canvas_pad), move, D), dtype=np.float64)
            del raw_arr
        except Exception as exc:  # noqa: BLE001 — fall through to the corrected-volume detector
            move_info = {"raw_baseline_error": f"{type(exc).__name__}: {exc}"}
    if served is None:
        served, dinfo = detect_corrected_surface(case_dir, vol, work, dp_params, workers=workers,
                                                 write_cache=write_cache)
        served_source = "detect_surface_all"; move_source = None; move_info = dinfo
        try:                                         # the pad is still knowable from the raw canvas when present
            import nibabel as nib
            if raw_path.exists():
                rd = int(nib.load(str(raw_path)).shape[1])
                canvas_pad = int(D - rd) if D > rd else 0
        except Exception:  # noqa: BLE001
            pass
    timings["served_line"] = time.time() - t0

    # ── validity ────────────────────────────────────────────────────────────────────────────────────────────
    t0 = time.time()
    band_mask = op._artifact_mask(params, L, F)
    signal = vol.max(axis=1) > 0
    valid = np.isfinite(served) & (served > 2) & (served < D - 3) & ~band_mask & signal
    scf = np.array(sorted({int(f) for f in (params.get("surface_crop_frames") or []) if 0 <= int(f) < F}), dtype=int)
    timings["valid"] = time.time() - t0

    # ── posterior ───────────────────────────────────────────────────────────────────────────────────────────
    post = None; post_source = None; post_check: dict | None = None
    if posterior:
        t0 = time.time()
        pe = bc / "posterior_edges.npz"
        if pe.exists() and move is not None:
            try:
                pr = np.asarray(np.load(pe, allow_pickle=False)["surface"], dtype=np.float64)
                raw_depth = D - canvas_pad
                if pr.shape == (L, F):
                    pr = np.where(~np.isfinite(pr) | (pr >= raw_depth - 1) | (pr <= 0), np.nan, pr)
                    post = np.where(np.isfinite(pr), pr + float(canvas_pad) + move, np.nan)
                    post = np.where(post > served + 1, post, np.nan)
                    post_source = "posterior_edges"
            except Exception:  # noqa: BLE001
                post = None
        if post is not None and posterior in ("auto", True):
            # E2 (round 9) — VALIDATE the carried line against the trace-free estimate: a carried posterior that is
            # anatomically too thin (P5_OS v1_2 / v1_3: the bright-band bottom 33-47 px below the anterior, the
            # trace-free posterior 156 / 199 px) caps away half the band; it is REJECTED (kept in meta with its
            # statistics) and the trace-free line takes its place ('trace_free_fallback').
            thick_c = (post - served)[valid]
            thick_c = thick_c[np.isfinite(thick_c)]
            thick_tf = posterior_trace_free(vol, served, valid)
            tf_vals = thick_tf[valid & np.isfinite(thick_tf)]
            med_c = float(np.median(thick_c)) if thick_c.size else float("nan")
            p10_c = float(np.percentile(thick_c, 10)) if thick_c.size else float("nan")
            med_tf = float(np.median(tf_vals)) if tf_vals.size else float("nan")
            ok_c = (thick_c.size > 0 and med_c >= MIN_POSTERIOR_THICKNESS and p10_c >= MIN_POSTERIOR_P10
                    and (not np.isfinite(med_tf) or med_c >= 0.5 * med_tf))
            post_check = {"carried_median_px": med_c, "carried_p10_px": p10_c, "trace_free_median_px": med_tf,
                          "min_median_px": MIN_POSTERIOR_THICKNESS, "min_p10_px": MIN_POSTERIOR_P10, "accepted": bool(ok_c)}
            if not ok_c:
                post = None
                if np.isfinite(thick_tf).any():
                    post = served + thick_tf
                    post_source = "trace_free_fallback"
                else:
                    post_source = None
        if post is None and posterior in ("auto", True) and post_source is None:
            thick = posterior_trace_free(vol, served, valid)
            if np.isfinite(thick).any():
                post = served + thick
                post_source = "trace_free"
        timings["posterior"] = time.time() - t0

    # ── scar ────────────────────────────────────────────────────────────────────────────────────────────────
    scar_arr = None; scar_source = None
    if scar:
        lab_p = case_dir / "segmentation" / f"{cid}_corrected.nii.gz"
        if lab_p.exists():
            try:
                import nibabel as nib
                lab = np.rint(np.asarray(nib.load(str(lab_p)).dataobj)).astype(np.uint8)
                if lab.shape == vol.shape:
                    cornea = (lab == 1) | (lab == 2)
                    n_c = cornea.sum(axis=1).astype(np.float64)
                    n_s = (lab == 2).sum(axis=1).astype(np.float64)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        scar_arr = np.where(n_c > 0, n_s / np.maximum(n_c, 1), 0.0)
                    scar_source = "labelmap"
            except Exception:  # noqa: BLE001
                scar_arr = None
    timings["total"] = time.time() - t_all
    meta = {"patient": gk[1] if gk else None, "eye": gk[2] if gk else None,
            "vetted": bool(m.get("preproc_vetted")), "rejected_unfixable": bool(m.get("rejected_unfixable")),
            "difficult": bool(m.get("difficult_scan")), "review_flags": m.get("review_flags"),
            "input_volume": str(work), "served_source": served_source, "move": move_info,
            "pipeline_version": (m.get("oct_iter") or {}).get("pipeline_version"),
            "crop_bands": params.get("crop_bands"), "n_crop_band_cells": int(band_mask.sum()),
            "n_surface_crop_frames": int(scf.size), "dp_params": dp_params, "posterior_check": post_check,
            "lateral_resample": 1.0, "lateral_offset": 0}
    return MemberData(cid=cid, case_dir=case_dir, group=group, volume=vol, served=served,
                      valid=valid, spacing=spacing, canvas_pad=int(canvas_pad), bottom_pad=int(bottom_pad),
                      lateral_dx=np.asarray(lateral_dx, dtype=np.float64), move=move, move_source=move_source,
                      served_source=served_source, posterior=post, posterior_source=post_source, scar=scar_arr,
                      scar_source=scar_source, surface_crop_frames=scf, crop_band_mask=band_mask, meta=meta,
                      timings=timings)


def load_group(group_id: str, root: str | os.PathLike | None = None, **kw) -> list[MemberData]:
    """Every member of a patient+eye group (group_members) loaded with load_member(**kw)."""
    base = Path(root) if root is not None else default_cases_root()
    members, _ = group_members(group_id, base)
    return [load_member(base / cid, **kw) for cid in members]


def choose_reference(members: Iterable[MemberData], tie_tol: float = 0.02) -> str:
    """The reference member's cid (the design's rule): the member with the LARGEST VALID AREA (valid.sum(): the
    most usable corneal surface, so every other member has the most to overlap with); members within `tie_tol`
    (2 %) of that area tie, and the tie goes to the MOST CENTRAL DOME — the apex of the quadric fitted to the
    served line closest to the volume centre in normalised (lateral, frame) coordinates — because a centred
    dome keeps the periphery of every other member inside the reference's field of view after alignment."""
    ms = list(members)
    if not ms:
        raise ValueError("choose_reference: no members")
    areas = np.array([m.valid_area for m in ms], dtype=np.float64)
    best = float(areas.max())
    cand = [i for i, a in enumerate(areas) if a >= (1.0 - tie_tol) * best]
    if len(cand) == 1:
        return ms[cand[0]].cid
    def centrality(i: int) -> float:
        L, _, F = ms[i].shape
        l0, f0 = ms[i].dome_apex()
        return math.hypot((l0 - (L - 1) / 2.0) / max(1.0, (L - 1) / 2.0), (f0 - (F - 1) / 2.0) / max(1.0, (F - 1) / 2.0))
    cand.sort(key=lambda i: (centrality(i), -areas[i], ms[i].cid))
    return ms[cand[0]].cid


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Element 2 — BandData
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class BandData:
    """The corneal band of one member, flattened to its served line (design element 2).

    row0        band row k ↔ depth offset (k + row0) from the served line; `surface_row` = −row0 (8 by default).
    band        (L, T, F) float32 — the corrected volume sampled at depth S[l, f] + row0 + k, linear in depth
                (integer + sub-pixel shift per A-scan), 0 outside the canvas.
    mask        (L, T, F) bool — the sampled row is inside the canvas AND the cell (l, f) is valid AND, when the
                member carries a posterior, the row lies above it.
    feat        (L, T, F) float32 — the STRUCTURE feature (round 9, the matcher's feature): sqrt(max(I − nf, 0))
                smoothed per B-scan with sigma_struct (5, 5) px as a normalised convolution inside match_mask,
                zero-mean / unit-variance per frame inside match_mask, 0 elsewhere. Never smoothed along frames.
    feat_speckle (L, T, F) float32 — prototype A's SPECKLE feature: the same sqrt image with a plain (1.5, 1.5)
                Gaussian per B-scan, normalised the same way — the reported speckle-scale match only.
    tissue      (L, T, F) bool — prototype A's tissue rule: rows from the surface down to the POSTERIOR (the first
                sustained run of GAP_ROWS rows below 0.7×Otsu of a smoothed feature ends the column; posterior
                de-spiked by a (15, 3) median across laterals/frames), columns with < MIN_TISSUE_ROWS rows or a
                dark top dropped.
    match_mask  (L, T, F) bool — mask & tissue & (offset ≥ R_SKIP): the cells the matcher and band_similarity use.
    posterior_row (L, F) float — the band bottom in band rows (member posterior when present, else the Otsu bottom).
    coarse / coarse_mask — the COARSE stage's own pyramid: the member sampled on `coarse_rows` (COARSE_ROWS_DEFAULT
                (−8, 240) — the posterior included), features + per-frame normalisation inside the coarse mask
                (`coarse_mask_mode` "tissue" = the Otsu tissue rule with NO posterior cap, prototype A's Mmatch;
                "band" = canvas & valid only), block means over `coarse_ds` (COARSE_DS ×4 lateral × ×4 depth).
                Independent of the fine band above (band / feat / match_mask are untouched by the coarse settings).
    noise_floor  the volume's background level (median of a ×4 subsample — background-dominated); otsu_thr.
    served / valid  (L, F) the member's served line (CORRECTED rows, NaN = unknown) and validity, carried so the
                pair engine (register_pair) can turn a band-space depth offset into a corrected-space axial move
                (Δz = dz_band + S_ref − S_mov) and warp_band can carry the moving line onto the reference grid."""
    cid: str
    row0: int
    band: np.ndarray
    mask: np.ndarray
    feat: np.ndarray
    tissue: np.ndarray
    match_mask: np.ndarray
    posterior_row: np.ndarray
    coarse: np.ndarray
    coarse_mask: np.ndarray
    coarse_ds: tuple[int, int]
    spacing: np.ndarray
    noise_floor: float
    otsu_thr: float
    timings: dict = field(default_factory=dict)
    served: np.ndarray | None = None
    valid: np.ndarray | None = None
    coarse_rows: tuple[int, int] = COARSE_ROWS_DEFAULT
    coarse_mask_mode: str = COARSE_MASK_DEFAULT
    feat_speckle: np.ndarray | None = None
    sigma_struct: tuple[float, float] = FEATURE_SIGMA_STRUCT
    sigma_speckle: tuple[float, float] = FEATURE_SIGMA_SPECKLE
    lateral_scale: float = 1.0          # E4: the member was resampled by this factor onto the pair's lateral grid
    lateral_offset: int = 0             # E4: original lateral l ↔ grid lateral l · lateral_scale + lateral_offset

    def feature(self, name: str = "struct") -> np.ndarray:
        """The feature image by name: 'struct' (feat) or 'speckle' (feat_speckle; feat when absent)."""
        if name == "struct":
            return self.feat
        if name == "speckle":
            return self.feat_speckle if self.feat_speckle is not None else self.feat
        raise ValueError(f"unknown feature {name!r} (struct | speckle)")

    @property
    def surface_row(self) -> int:
        return -int(self.row0)

    @property
    def n_rows(self) -> int:
        return int(self.band.shape[1])

    @property
    def n_frames(self) -> int:
        return int(self.band.shape[2])

    def offsets(self) -> np.ndarray:
        """Depth offset from the served line of every band row."""
        return np.arange(self.n_rows) + int(self.row0)

    def frame(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        """(band image (L, T), match mask (L, T)) of frame f — the per-B-scan band image."""
        return self.band[:, :, f], self.match_mask[:, :, f]


def noise_floor(vol: np.ndarray, served: np.ndarray | None = None, valid: np.ndarray | None = None,
                rows_above: tuple[int, int] = (120, 20)) -> float:
    """Background level (intensity units). E1 (round 9): the median of the NON-ZERO voxels in the AIR above the
    served line — rows served − rows_above[0] … served − rows_above[1] of the valid columns (a ×2 frame / ×4 lateral
    subsample) — the same tissue-free estimate on every member whatever its zero fraction or canvas pad. The
    whole-volume median is background-dominated only when the canvas carries no pad: on CS032_OS v1_4 (49.7 % zero
    voxels after the canvas extension) it read 63 against 616 from the air (±0.04 relative match and a flipped
    coarse mode on the v1_2 pairs). Fallback (no served line, or no air rows): the median of the non-zero voxels
    of the subsample; an all-zero volume → 0."""
    vol = np.asarray(vol)
    if served is not None:
        L, D, F = vol.shape
        S = np.asarray(served, float)
        V = np.ones(S.shape, bool) if valid is None else np.asarray(valid, bool)
        vals = []
        for f in range(0, F, 2):
            for l in range(0, L, 4):
                if not V[l, f] or not np.isfinite(S[l, f]):
                    continue
                z1 = int(S[l, f]) - int(rows_above[1]); z0 = max(0, z1 - int(rows_above[0] - rows_above[1]))
                if z1 <= z0:
                    continue
                col = vol[l, z0:z1, f]; col = col[col > 0]
                if col.size:
                    vals.append(col)
        if vals:
            v = np.concatenate(vals)
            if v.size >= 50:
                return float(np.median(v))
    sub = vol[::4, ::4, ::4]
    nz = sub[sub > 0]
    return float(np.median(nz)) if nz.size else 0.0


def _otsu(v: np.ndarray, nb: int = 256) -> float:
    v = np.asarray(v, dtype=np.float64)
    if v.size < 2 or not np.isfinite(v).any() or float(v.max()) <= float(v.min()):
        return float(v.max()) if v.size else 0.0
    h, e = np.histogram(v, bins=nb)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = np.roll(np.cumsum((h * c)[::-1])[::-1] / np.maximum(w1, 1), -1)
    var = w0[:-1] * w1[:-1] * (m0[:-1] - m1[:-1]) ** 2
    return float(c[int(np.argmax(var))])


def sample_band(vol: np.ndarray, served: np.ndarray, band_rows: tuple[int, int] = BAND_ROWS_DEFAULT
                ) -> tuple[np.ndarray, np.ndarray]:
    """Flattened band B[l, k, f] = V[l, S[l, f] + row0 + k, f] (linear in depth) + the inside-canvas mask.
    Prototype A band_extract, with the row origin generalised to band_rows[0] (rows above the surface kept)."""
    L, D, F = vol.shape
    row0, row1 = int(band_rows[0]), int(band_rows[1])
    T = row1 - row0
    B = np.zeros((L, T, F), np.float32); M = np.zeros((L, T, F), bool)
    r = (np.arange(T) + row0)[None, :]
    for f in range(F):
        s = served[:, f]
        okc = np.isfinite(s)
        zf = np.where(okc, s, 0.0)[:, None] + r
        z0 = np.floor(zf).astype(np.int64); w = (zf - z0).astype(np.float32)
        ok = (z0 >= 0) & (z0 + 1 <= D - 1) & okc[:, None]
        z0c = np.clip(z0, 0, D - 2)
        sl = vol[:, :, f]
        b = (1 - w) * np.take_along_axis(sl, z0c, axis=1) + w * np.take_along_axis(sl, z0c + 1, axis=1)
        B[:, :, f] = np.where(ok, b, 0.0); M[:, :, f] = ok
    return B, M


def block_mean(a: np.ndarray, ds: tuple[int, int]) -> np.ndarray:
    """Block-mean downsample along the first two axes (lateral, depth) by ds; frames untouched (prototype A)."""
    L, R, F = a.shape
    dl, dr = int(ds[0]), int(ds[1])
    Lc, Rc = (L // dl) * dl, (R // dr) * dr
    a = a[:Lc, :Rc]
    return a.reshape(L // dl, dl, R // dr, dr, F).mean(axis=(1, 3))


def _assert_no_frame_smoothing(sigma) -> None:
    """Round 9 invariant: a Gaussian that touches an (L, T, F) array never smooths along the FRAME axis."""
    s = tuple(float(v) for v in np.atleast_1d(sigma))
    if len(s) >= 3 and s[2] != 0.0:
        raise ValueError(f"frame-axis smoothing is forbidden (sigma {s}): the frame offset is read from it")


def _band_masks(B: np.ndarray, M: np.ndarray, row0: int, nf: float, gap_rows: int, min_tissue_rows: int
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Prototype A band_features' MASKS on a sampled band (B, M) — CLIP-AWARE (E3, round 9): returns (X = the
    sqrt(max(I − nf, 0)) feature image, M' = the band mask with the ZERO-sampled rows removed (the canvas pad / the
    rows a surface crop vacated are not band), tissue mask, match mask = tissue below R_SKIP, post_k band row where
    the tissue ends, Otsu threshold). The tissue rule: rows from the surface down to the POSTERIOR (the first
    sustained run of gap_rows rows below 0.7×Otsu of a smoothed feature ends the column — a run may not start inside
    the specular / epithelial rows NOR inside the zero pad; de-spiked by a (15, 3) median), columns with
    < min_tissue_rows tissue rows COUNTED FROM THE SERVED LINE (a clipped column keeps its depth geometry: the served
    reconstruction stays the anchor) or a dark top dropped — the dark-top test reads the first 20 NON-ZERO rows below
    max(served + R_SKIP, first non-zero row + R_SKIP) (a column clipped by 18+ px was dropped whole before)."""
    L, T, F = B.shape
    M = M & (B > 0)
    X = np.sqrt(np.clip(B - nf, 0, None)).astype(np.float32)
    Xt = ndi.gaussian_filter(X, (4, 3, 0))
    thr = _otsu(Xt[M][::7]) if M.any() else 0.0
    bright = (Xt > 0.7 * thr) & M
    dark = ~bright
    k_surf = -row0                                          # band row of the served line
    gap = max(1, min(int(gap_rows), T - 1))
    c = np.cumsum(dark, axis=1, dtype=np.int32); c = np.pad(c, ((0, 0), (1, 0), (0, 0)))
    run = (c[:, gap:, :] - c[:, :-gap, :]) == gap           # a full dark run STARTS here
    first_nz = np.where(M.any(axis=1), np.argmax(M, axis=1), T)            # (L, F) first sampled non-zero row
    rows_run = np.arange(run.shape[1])[None, :, None]
    run &= rows_run >= np.maximum(k_surf + R_SKIP + 4, first_nz[:, None, :] + R_SKIP + 4)
    has = run.any(axis=1)
    post_k = np.where(has, np.argmax(run, axis=1), T).astype(np.float64)   # band row where the tissue ends
    post_k = ndi.median_filter(post_k, size=(min(15, L), min(3, F)), mode="nearest")
    rows = np.arange(T)[None, :, None]
    tis = (rows < post_k[:, None, :]) & (rows >= k_surf) & M
    start = np.maximum(k_surf + R_SKIP, first_nz + R_SKIP)                 # (L, F)
    cum_b = np.cumsum(bright, axis=1, dtype=np.int32); cum_b = np.pad(cum_b, ((0, 0), (1, 0), (0, 0)))
    lo = np.clip(start, 0, T); hi = np.clip(start + 20, 0, T)
    li = np.arange(L)[:, None]; fi = np.arange(F)[None, :]
    n_bright = cum_b[li, hi, fi] - cum_b[li, lo, fi]
    top_ok = (n_bright > 0.5 * np.maximum(hi - lo, 1)) & (hi > lo)
    tis &= ((post_k - k_surf) >= min(int(min_tissue_rows), T - k_surf - 1))[:, None, :] & top_ok[:, None, :]
    offs = np.arange(T) + row0
    match = tis & (offs[None, :, None] >= R_SKIP)
    return X, M, tis, match, post_k, float(thr)


def _speckle_feature(X: np.ndarray, sigma) -> np.ndarray:
    """Prototype A's speckle-stabilised feature: the sqrt image smoothed per B-scan with `sigma` (lateral, depth)."""
    _assert_no_frame_smoothing((float(sigma[0]), float(sigma[1]), 0.0))
    Xs = np.empty_like(X)
    for f in range(X.shape[2]):
        Xs[:, :, f] = ndi.gaussian_filter(X[:, :, f], (float(sigma[0]), float(sigma[1])))
    return Xs


def _struct_feature(X: np.ndarray, match: np.ndarray, sigma) -> np.ndarray:
    """Round 9 STRUCTURE feature: the sqrt image smoothed per B-scan with `sigma` (lateral, depth; default 5 px —
    ≥ 3-5 speckle widths) as a NORMALISED CONVOLUTION inside the match mask (gauss(X·m) / gauss(m)), so the bright
    specular line above R_SKIP and the zero pad never leak into the matched rows; 0 outside the mask. Never along
    frames."""
    _assert_no_frame_smoothing((float(sigma[0]), float(sigma[1]), 0.0))
    out = np.zeros_like(X)
    sg = (float(sigma[0]), float(sigma[1]))
    for f in range(X.shape[2]):
        m = match[:, :, f].astype(np.float32)
        if not m.any():
            continue
        num = ndi.gaussian_filter(X[:, :, f] * m, sg)
        den = ndi.gaussian_filter(m, sg)
        out[:, :, f] = np.where(m > 0, num / np.maximum(den, 1e-3), 0.0)
    return out


def _band_features(B: np.ndarray, M: np.ndarray, row0: int, nf: float, sigma, gap_rows: int, min_tissue_rows: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Legacy entry (prototype A band_features): (Xs SPECKLE feature smoothed with `sigma`, tissue, match, post_k,
    Otsu threshold) on the clip-aware masks of _band_masks."""
    X, M2, tis, match, post_k, thr = _band_masks(B, M, row0, nf, gap_rows, min_tissue_rows)
    return _speckle_feature(X, sigma), tis, match, post_k, thr


def _normalise_frames(Xs: np.ndarray, match: np.ndarray) -> np.ndarray:
    """Zero-mean / unit-variance per frame inside `match`, 0 elsewhere (a frame with < 2 cells or no variance → 0)."""
    feat = np.zeros_like(Xs)
    for f in range(Xs.shape[2]):
        mm = match[:, :, f]
        n = int(mm.sum())
        if n < 2:
            continue
        v = Xs[:, :, f][mm]
        mu = float(v.mean()); sd = float(v.std())
        if sd < 1e-6:
            continue
        feat[:, :, f] = np.where(mm, (Xs[:, :, f] - mu) / sd, 0.0)
    return feat


def extract_band(member: MemberData, band_rows: tuple[int, int] = BAND_ROWS_DEFAULT, *, use_posterior: bool = True,
                 coarse_rows: tuple[int, int] = COARSE_ROWS_DEFAULT, coarse_ds: tuple[int, int] = COARSE_DS,
                 coarse_mask: str = COARSE_MASK_DEFAULT, sigma: tuple[float, float] = FEATURE_SIGMA_STRUCT,
                 sigma_speckle: tuple[float, float] | None = FEATURE_SIGMA_SPECKLE,
                 gap_rows: int = GAP_ROWS, min_tissue_rows: int = MIN_TISSUE_ROWS) -> BandData:
    """Design element 2: the member's corneal band flattened to its served line, with the matcher's features.

    band_rows = (row0, row1) relative to the served line (default (−8, +120): 8 rows of air/specular above, 120
    rows = 0.38 mm of stroma below; prototype A matched on 240 rows so the posterior was inside the band at the
    periphery). `use_posterior` caps the mask at the member's posterior when it carries one ("or up to the
    posterior when available"). `min_tissue_rows` (40): a column whose tissue ends sooner carries no cornea
    (prototype A). ROUND 9: `sigma` is the STRUCTURE feature's in-plane Gaussian (FEATURE_SIGMA_STRUCT (5, 5) —
    BandData.feat, the matcher's feature, a normalised convolution inside the match mask), `sigma_speckle` the
    SPECKLE feature's (FEATURE_SIGMA_SPECKLE (1.5, 1.5) — BandData.feat_speckle, the report; None skips it); the
    masks are clip-aware (_band_masks: zero-sampled rows are not band, the dark-top test reads non-zero rows) and
    the noise floor is read from the air above the served line (noise_floor). The COARSE pyramid (BandData.coarse)
    is built from the STRUCTURE feature on `coarse_rows` (default (−8, 240)) with `coarse_ds` block means and the
    `coarse_mask` rule ("tissue": Otsu tissue rule, no posterior cap — prototype A's coarse configuration; "band":
    canvas & valid only), so the coarse settings never touch the fine band; it is sampled once more from the volume
    (~2-3 s) unless it coincides with the fine band. Runs in ~7-10 s on a 513×640×101 volume."""
    t_all = time.time(); timings: dict = {}
    vol, S = member.volume, member.served
    L, D, F = vol.shape
    row0, row1 = int(band_rows[0]), int(band_rows[1])
    if row1 <= row0:
        raise ValueError(f"extract_band: empty band {band_rows}")
    c_row0, c_row1 = int(coarse_rows[0]), int(coarse_rows[1])
    if c_row1 <= c_row0:
        raise ValueError(f"extract_band: empty coarse band {coarse_rows}")
    if coarse_mask not in ("tissue", "band"):
        raise ValueError(f"extract_band: coarse_mask must be 'tissue' or 'band', got {coarse_mask!r}")
    sg = (float(sigma[0]), float(sigma[1]))
    sg_sp = None if sigma_speckle is None else (float(sigma_speckle[0]), float(sigma_speckle[1]))
    T = row1 - row0
    t0 = time.time()
    B, M = sample_band(vol, S, (row0, row1))
    M &= member.valid[:, None, :]
    offs = np.arange(T) + row0
    post_capped = bool(use_posterior and member.posterior is not None)
    if post_capped:
        pr = member.posterior - S                            # posterior in offset rows
        pr = np.where(np.isfinite(pr), pr, np.inf)
        M &= offs[None, :, None] < pr[:, None, :]
    timings["sample"] = time.time() - t0
    # ── clip-aware masks (prototype A's tissue rule) + the two features ─────────────────────────────────────
    t0 = time.time()
    nf = noise_floor(vol, S, member.valid)
    X, M, tis, match, post_k, thr = _band_masks(B, M, row0, nf, gap_rows, min_tissue_rows)
    if post_capped:
        pr = member.posterior - S
        post_row = np.where(np.isfinite(pr), pr - row0, post_k)
    else:
        post_row = post_k
    timings["features"] = time.time() - t0
    # ── per-frame normalisation inside the match mask ────────────────────────────────────────────────────────
    t0 = time.time()
    feat = _normalise_frames(_struct_feature(X, match, sg), match)
    feat_sp = _normalise_frames(_speckle_feature(X, sg_sp), match) if sg_sp is not None else None
    timings["normalise"] = time.time() - t0
    # ── the coarse stage's own band + pyramid (prototype A: 240 rows, ×4×4, tissue mask, no posterior cap) ───
    t0 = time.time()
    if (c_row0, c_row1) == (row0, row1) and coarse_mask == "tissue" and not post_capped:
        feat_c, match_c = feat, match                         # the fine band IS the coarse band
    else:
        Bc, Mc = sample_band(vol, S, (c_row0, c_row1))
        Mc &= member.valid[:, None, :]
        if coarse_mask == "tissue":
            Xc, _, _, match_c, _, _ = _band_masks(Bc, Mc, c_row0, nf, gap_rows, min_tissue_rows)
        else:
            Mc &= Bc > 0
            Xc = np.sqrt(np.clip(Bc - nf, 0, None)).astype(np.float32)
            match_c = Mc & ((np.arange(c_row1 - c_row0) + c_row0)[None, :, None] >= R_SKIP)
        feat_c = _normalise_frames(_struct_feature(Xc, match_c, sg), match_c)
    coarse = block_mean(feat_c * match_c, coarse_ds)
    coarse_mask_arr = block_mean(match_c.astype(np.float32), coarse_ds) > 0.5
    timings["coarse"] = time.time() - t0
    timings["total"] = time.time() - t_all
    return BandData(cid=member.cid, row0=row0, band=B, mask=M, feat=feat, tissue=tis, match_mask=match,
                    posterior_row=post_row, coarse=coarse.astype(np.float32), coarse_mask=coarse_mask_arr,
                    coarse_ds=(int(coarse_ds[0]), int(coarse_ds[1])), spacing=np.asarray(member.spacing),
                    noise_floor=nf, otsu_thr=float(thr), timings=timings,
                    served=np.asarray(S, dtype=np.float64).copy(), valid=np.asarray(member.valid, dtype=bool).copy(),
                    coarse_rows=(c_row0, c_row1), coarse_mask_mode=str(coarse_mask), feat_speckle=feat_sp,
                    sigma_struct=sg, sigma_speckle=(sg_sp if sg_sp is not None else FEATURE_SIGMA_SPECKLE),
                    lateral_scale=float((member.meta or {}).get("lateral_resample", 1.0) or 1.0),
                    lateral_offset=int((member.meta or {}).get("lateral_offset", 0) or 0))


def resample_member_lateral(m: MemberData, factor: float, L_out: int | None = None) -> tuple[MemberData, int]:
    """E4 (round 9): the member on a lateral grid `factor` times FINER (factor = spacing_mov / spacing_ref > 1 → more
    laterals): volume / served / posterior / scar / move linearly (NaN-aware), valid & crop mask by nearest;
    spacing[0] /= factor; then padded (zeros + invalid) or cropped SYMMETRICALLY to L_out laterals. Returns
    (member, lateral_offset) with  grid lateral = original lateral · factor + lateral_offset  (meta['lateral_resample']
    / ['lateral_offset']). A change of SAMPLING GRID onto a common physical lateral spacing — recorded in the
    transform (PairResult.lateral_scale / lateral_offset_*), never a per-column warp of the tissue. CS032_OS: v1_3 /
    v1_4 (12.281 µm) against v1 / v1_2 (11.501 µm) → factor 1.0678, 513 → 548 laterals (34.8 laterals of drift across a
    frame that a per-segment rigid dx could not fit; the cross pairs rose 0.627 → 0.903, 0.649 → 0.894 …)."""
    L, D, F = m.shape
    factor = float(factor)
    Ln = int(round(L * factor))
    src = np.clip(np.arange(Ln) / factor, 0, L - 1)
    i0 = np.floor(src).astype(int); w = (src - i0).astype(np.float32); i1 = np.minimum(i0 + 1, L - 1)
    if abs(factor - 1.0) < 1e-12:
        vol = np.asarray(m.volume, np.float32)
    else:
        vol = (1 - w)[:, None, None] * m.volume[i0] + w[:, None, None] * m.volume[i1]

    def lin(a):
        if a is None:
            return None
        a = np.asarray(a, float)
        a0 = a[i0]; a1 = a[i1]
        out = (1 - w)[:, None] * a0 + w[:, None] * a1
        both = np.isfinite(a0) & np.isfinite(a1)
        return np.where(both, out, np.where(np.isfinite(a0), a0, a1))

    def near(a):
        return None if a is None else np.asarray(a)[np.rint(src).astype(int)]
    served = lin(m.served); valid = near(m.valid) & np.isfinite(served)
    post = lin(m.posterior); scar = lin(m.scar); move = lin(m.move); cbm = near(m.crop_band_mask)
    off = 0
    if L_out is not None and int(L_out) != Ln:
        L_out = int(L_out)
        if L_out > Ln:
            off = (L_out - Ln) // 2

            def pad(a, fill):
                if a is None:
                    return None
                o = np.full((L_out,) + a.shape[1:], fill, dtype=a.dtype); o[off:off + Ln] = a; return o
            vol = pad(vol, 0); served = pad(served, np.nan); valid = pad(valid, False)
            post = pad(post, np.nan); scar = pad(scar, 0.0); move = pad(move, np.nan); cbm = pad(cbm, False)
        else:
            c0 = (Ln - L_out) // 2; off = -c0
            cut = lambda a: None if a is None else a[c0:c0 + L_out]  # noqa: E731
            vol, served, valid, post, scar, move, cbm = (cut(v) for v in (vol, served, valid, post, scar, move, cbm))
    sp = np.array(m.spacing, float); sp[0] = sp[0] / factor
    lat_dx = None if m.lateral_dx is None else np.asarray(m.lateral_dx, float) * factor
    meta = dict(m.meta or {})
    base_scale = float(meta.get("lateral_resample", 1.0) or 1.0); base_off = int(meta.get("lateral_offset", 0) or 0)
    meta.update(lateral_resample=base_scale * factor, lateral_offset=int(round(base_off * factor)) + int(off),
                lateral_grid=int(vol.shape[0]))
    mm = MemberData(cid=m.cid, case_dir=m.case_dir, group=m.group, volume=np.ascontiguousarray(vol.astype(np.float32)),
                    served=np.asarray(served, float), valid=np.asarray(valid, bool), spacing=sp, canvas_pad=m.canvas_pad,
                    bottom_pad=m.bottom_pad, lateral_dx=lat_dx, move=move, move_source=m.move_source,
                    served_source=m.served_source, posterior=post, posterior_source=m.posterior_source, scar=scar,
                    scar_source=m.scar_source, surface_crop_frames=m.surface_crop_frames, crop_band_mask=cbm,
                    meta=meta, timings=dict(m.timings or {}))
    return mm, int(off)


def common_lateral_grid(members: Iterable[MemberData], reference: str, tol: float = 0.005) -> tuple[list, dict]:
    """E4: every member of a group on the REFERENCE's physical lateral spacing — each member whose header lateral
    spacing differs from the reference's by more than `tol` (relative) is resampled by s = spacing_m / spacing_ref
    (the HEADER ratio, never fitted), and every member is embedded centred on a common grid of
    L_out = max_m round(L_m · s_m) laterals (zero + invalid pad). Returns (members, {cid: {'scale', 'offset',
    'L_out', 'resampled'}})."""
    ms = list(members)
    ref = next((m for m in ms if m.cid == reference), None)
    if ref is None:
        raise ValueError(f"common_lateral_grid: reference {reference!r} not among {[m.cid for m in ms]}")
    sp_ref = float(ref.spacing[0])
    scales = {}
    for m in ms:
        s = float(m.spacing[0]) / sp_ref if sp_ref > 0 else 1.0
        scales[m.cid] = s if abs(s - 1.0) > float(tol) else 1.0
    L_out = max(int(round(m.shape[0] * scales[m.cid])) for m in ms)
    out = []; rec = {}
    for m in ms:
        s = scales[m.cid]
        if s == 1.0 and m.shape[0] == L_out:
            out.append(m); rec[m.cid] = {"scale": 1.0, "offset": 0, "L_out": L_out, "resampled": False}
            continue
        mm, off = resample_member_lateral(m, s, L_out=L_out)
        out.append(mm); rec[m.cid] = {"scale": s, "offset": int(off), "L_out": L_out, "resampled": s != 1.0}
    return out, rec


def scar_proxy(band: BandData, min_voxels: int = 2000, k_sigma: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """PROXY scar map when a case has no labelmap (prototype A finelib.scar_proxy): hyper-reflective stromal
    patches — cells of a (6, 4, 1)-smoothed feature more than k_sigma SD above the stromal mean, inside the
    stroma (30 rows below the surface to 15 rows above the band bottom), connected components ≥ min_voxels.
    Returns (proxy (L, T, F) bool, fraction (L, F) of band tissue rows flagged)."""
    Xs = ndi.gaussian_filter(np.where(band.tissue, band.feat, 0.0).astype(np.float32), (6, 4, 1))
    offs = band.offsets()[None, :, None]
    strom = band.tissue & (offs >= 30) & (np.arange(band.n_rows)[None, :, None] < (band.posterior_row[:, None, :] - 15))
    if not strom.any():
        return np.zeros_like(band.tissue), np.zeros(band.tissue.shape[::2], dtype=np.float64)
    v = Xs[strom]; mu, sd = float(v.mean()), float(v.std())
    pr = (Xs > mu + k_sigma * sd) & strom
    lab, n = ndi.label(pr)
    if n:
        sizes = ndi.sum(pr, lab, index=np.arange(1, n + 1))
        keep = np.zeros(n + 1, bool); keep[1:] = sizes >= min_voxels
        pr = keep[lab]
    n_t = band.tissue.sum(axis=1).astype(np.float64)
    frac = np.where(n_t > 0, pr.sum(axis=1) / np.maximum(n_t, 1), 0.0)
    return pr, frac


def attach_scar_proxy(member: MemberData, band: BandData) -> MemberData:
    """Fill member.scar from the band proxy when the case has no labelmap (scar_source → "proxy")."""
    if member.scar is None:
        _, frac = scar_proxy(band)
        member.scar = frac; member.scar_source = "proxy"
    return member


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Similarity — the masked local NCC of prototype A (finelib.local_ncc / match_metrics)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def local_ncc(A: np.ndarray, B: np.ndarray, mask: np.ndarray, win: tuple[int, int] = LOCAL_WIN) -> np.ndarray:
    """Masked local normalised cross-correlation of two 2-D images over a (lateral, depth) window (prototype A):
    windowed means/variances taken over the MASKED pixels only (uniform_filter with a constant border), NaN where
    fewer than half the window is masked. Invariant to a per-image affine intensity change."""
    m = mask.astype(np.float32)
    A = np.asarray(A, np.float32); B = np.asarray(B, np.float32)
    uf = lambda x: ndi.uniform_filter(x, win, mode="constant")  # noqa: E731
    n = uf(m); nn = np.maximum(n, 1e-6)
    ma = uf(A * m) / nn; mb = uf(B * m) / nn
    vaa = uf(A * A * m) / nn - ma ** 2; vbb = uf(B * B * m) / nn - mb ** 2; vab = uf(A * B * m) / nn - ma * mb
    with np.errstate(invalid="ignore", divide="ignore"):
        ncc = vab / np.sqrt(np.clip(vaa, 1e-9, None) * np.clip(vbb, 1e-9, None))
    ncc[n < 0.5] = np.nan
    return ncc


@dataclass
class SimilarityResult:
    """band_similarity output: ncc (L, T, n) local-NCC map (NaN outside eval), eval (L, T, n) bool, the frame
    pairs (frames_a[i] ↔ frames_b[i]) and summary stats (ncc_mean, matched_frac_<thr>, n_eval, per_frame_frac_0.5)."""
    ncc: np.ndarray
    eval: np.ndarray
    frames_a: np.ndarray
    frames_b: np.ndarray
    stats: dict


def band_similarity(band_a: BandData, band_b: BandData, window: tuple[int, int] = LOCAL_WIN, *,
                    frame_offset: int = 0, frames: Iterable[int] | None = None, min_eval: int = 500,
                    use: str = "match", feature: str = "struct") -> SimilarityResult:
    """The masked local-NCC map between two flattened bands, frame by frame (prototype A match_metrics /
    band_ncc_upper_bound): frame f of `band_a` against frame f + frame_offset of `band_b`, inside the joint mask
    (`use` = "match": match_mask — tissue below R_SKIP, the validated matcher mask; "mask": the geometric mask), on
    the `feature` image ('struct' = BandData.feat, the round-9 structure feature, window LOCAL_WIN_STRUCT;
    'speckle' = feat_speckle, prototype A's, window LOCAL_WIN_SPECKLE — pass the matching window).
    Bands must share (L, T). No registration happens here — the pair engine resamples `band_b` onto `band_a`'s
    grid first; band_similarity(v1, v1, frame_offset=1) is the adjacent-frame CEILING of that feature: on
    case_cs001_os_v1 speckle mean 0.342 / 29.5 % > 0.5 for gap 1 on 240 band rows (BAND_SPACE_CEILING; prototype
    A's 0.285 / 22.0 % is the same metric in ORIGINAL space, PROTOTYPE_A_CEILING, reproduced by local_ncc to 3
    decimals); the structure feature's ceiling is BAND_SPACE_CEILING['rows_-8_120_struct_sigma5']."""
    if band_a.band.shape[:2] != band_b.band.shape[:2]:
        raise ValueError(f"band_similarity: band shapes differ {band_a.band.shape} vs {band_b.band.shape}")
    ma_all = band_a.match_mask if use == "match" else band_a.mask
    mb_all = band_b.match_mask if use == "match" else band_b.mask
    fa_img = band_a.feature(feature); fb_img = band_b.feature(feature)
    Fa, Fb = band_a.n_frames, band_b.n_frames
    fr = list(range(Fa)) if frames is None else [int(f) for f in frames]
    pairs = [(f, f + int(frame_offset)) for f in fr if 0 <= f < Fa and 0 <= f + int(frame_offset) < Fb]
    L, T = band_a.band.shape[:2]
    n = len(pairs)
    ncc = np.full((L, T, n), np.nan, np.float32); ev = np.zeros((L, T, n), bool)
    pf = {t: np.full(n, np.nan) for t in NCC_THRESHOLDS}
    pf_n = np.zeros(n, np.int64); pf_mean = np.full(n, np.nan)
    for i, (fa, fb) in enumerate(pairs):
        e = ma_all[:, :, fa] & mb_all[:, :, fb]
        if int(e.sum()) < min_eval:
            continue
        nc = local_ncc(fa_img[:, :, fa], fb_img[:, :, fb], e, window)
        ok = e & np.isfinite(nc)
        ncc[:, :, i] = np.where(ok, nc, np.nan); ev[:, :, i] = ok
        if ok.sum():
            v = nc[ok]
            pf_n[i] = int(ok.sum()); pf_mean[i] = float(v.mean())
            for t in NCC_THRESHOLDS:
                pf[t][i] = float(np.mean(v > t))
    vals = ncc[ev]
    # per-PAIR records (the arbitration's judge assembles pair-level numbers from them: n_eval, the matched
    # fractions and the mean NCC of a frame set are exact sums over these)
    stats = {"n_eval": int(ev.sum()), "n_pairs": n, "ncc_mean": float(np.nanmean(vals)) if vals.size else float("nan"),
             "per_frame_n_eval": pf_n, "per_frame_ncc_mean": pf_mean}
    for t in NCC_THRESHOLDS:
        stats[f"matched_frac_{t}"] = float(np.mean(vals > t)) if vals.size else float("nan")
        stats[f"per_frame_frac_{t}"] = pf[t]
    return SimilarityResult(ncc=ncc, eval=ev, frames_a=np.array([p[0] for p in pairs], dtype=int),
                            frames_b=np.array([p[1] for p in pairs], dtype=int), stats=stats)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Masked NCC over ALL integer shifts via FFT (Padfield 2012) — prototype A bandlib, for the coarse stage next
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def fft_shape(shape_f, max_shift) -> tuple[int, ...]:
    return tuple(sfft.next_fast_len(int(n + 2 * s + 1)) for n, s in zip(shape_f, max_shift))


def prep_fixed(f: np.ndarray, mf: np.ndarray, shape) -> list:
    """Fixed-side spectra for masked NCC (reused across many moving images / segment masks)."""
    f = np.asarray(f, np.float64) * mf; mf = mf.astype(np.float64)
    axes = tuple(range(f.ndim))
    return [sfft.rfftn(a, shape, axes=axes) for a in (mf, f, f * f)]


def masked_ncc_fft(f, mf, m, mm, max_shift, min_overlap: int = 1, fixed_fft=None, shape=None):
    """Padfield 2012 masked NCC for EVERY integer shift: exact per-shift means/variances over the OVERLAP of the
    two masks. Returns (ncc, n) of shape 2·max_shift+1 per axis, indexed by shift s = idx − max_shift with
    MOVING + s = FIXED (prototype A selftest: a (5, −3) roll is recovered exactly)."""
    if shape is None:
        shape = fft_shape(m.shape if f is None else f.shape, max_shift)
    axes = tuple(range(m.ndim))
    Fa = fixed_fft if fixed_fft is not None else prep_fixed(f, mf, shape)
    m = np.asarray(m, np.float64) * mm; mm = mm.astype(np.float64)
    Fb = [np.conj(sfft.rfftn(b, shape, axes=axes)) for b in (mm, m, m * m)]

    def xc(A, Bc):
        return sfft.irfftn(A * Bc, shape, axes=axes)
    N = xc(Fa[0], Fb[0]); Sf = xc(Fa[1], Fb[0]); Sm = xc(Fa[0], Fb[1])
    Sff = xc(Fa[2], Fb[0]); Smm = xc(Fa[0], Fb[2]); Sfm = xc(Fa[1], Fb[1])
    N = np.maximum(np.round(N), 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        num = Sfm - Sf * Sm / N
        den = np.sqrt(np.clip(Sff - Sf * Sf / N, 0, None) * np.clip(Smm - Sm * Sm / N, 0, None))
        ncc = np.where((N >= min_overlap) & (den > 1e-9), num / den, -1.0)
    idx = [np.arange(-s, s + 1) % n for s, n in zip(max_shift, shape)]
    return ncc[np.ix_(*idx)], N[np.ix_(*idx)]


def subpix_peak(ncc: np.ndarray, pk) -> tuple[float, ...]:
    """Parabolic sub-pixel refinement along each axis around the integer peak index tuple pk."""
    out = []
    for ax, i in enumerate(pk):
        if 0 < i < ncc.shape[ax] - 1:
            idx = list(pk)
            idx[ax] = i - 1; y0 = ncc[tuple(idx)]; idx[ax] = i; y1 = ncc[tuple(idx)]; idx[ax] = i + 1; y2 = ncc[tuple(idx)]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
            out.append(float(i + np.clip(d, -1, 1)))
        else:
            out.append(float(i))
    return tuple(out)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Element 3 — pair registration: coarse FFT-NCC → per-frame fine → rigid projection → quality
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
@dataclass
class PairParams:
    """Knobs of register_pair — defaults are the design's / prototype A's values (wf_align/A finelib.py:
    fine_win (64, 20), seg_win (12, 8), gate_ncc 0.3, robust_line 4·MAD, min_n 30). Units: laterals / px / frames.

    ROUND 9 (2026-09-10) — the BETWEEN-SCAN geometry, an explicit model. A replicate pair differs by (i) an integer
    frame offset df, (ii) a per-frame lateral shift dx[f] (a slow real lateral wave: CS032 ±25 laterals over ~40
    frames, CS001 ±5, plus real saccade steps), (iii) a per-B-scan rigid axial move a[f] + b[f]·x, and (iv) a change
    of lateral SAMPLING when the headers disagree (CS032 v1_3 / v1_4 12.281 µm against v1 / v1_2 11.501 µm: the
    moving member is resampled onto the reference's physical spacing by the header ratio before matching —
    lateral_scale_tol, recorded in PairResult.lateral_scale / lateral_offset_*, dx then in REFERENCE laterals on the
    common grid about the reference centre). The two served anterior lines are the geometric prior: on a dome, a
    lateral decentring Δ implies a tilt difference between the two B-scans of about Δ · half-span / R (170 laterals
    ≈ 1.3 mm → ≈ 100 px half-span at R 7-9 mm), so the within-scan cap of 40 px on |b| is WRONG between scans: the
    cap (max_tilt_px) applies to the RESIDUAL |b − b_lines(dx)|, b_lines = the half-span slope of S_ref(l + dx) −
    S_mov(l) over the valid overlap (measured residual median 2-13 px, p90 ≤ 17 on every registrable pair; the
    reference's own dome slope at offset dx under-predicts by 37-190 px and is NOT the prior), plus a sanity cap
    max_tilt_abs_px. A pair whose lines demand a pose angle theta = atan(median|b_lines| · sp_depth / (half-span ·
    sp_lateral)) beyond pose_max_deg (P5_OS v1 against its siblings: 168-192 px ≈ 15°) cannot be represented by a
    per-frame rigid transform: 'pose_beyond_frame_rigid', a member-level non-contributing verdict (never
    'no_correspondence' — it overlaps). Search ranges are between-scan priors (coarse ±300 / ±200 / ±50, fine ±96 /
    ±40 / ±10; max_dx = the coarse half-range; P5 / CS032 offsets reach 140 laterals / 48 frames), the coarse
    surface is a plateau on such pairs so its top-K separated maxima (coarse_seed_k / coarse_seed_tol) are each
    tried by a cheap fine pass ('coarse_seeds', 'coarse_reseeded'), and every match is judged on the STRUCTURE
    feature (sigma 5 in-plane, never along frames) against the reference's own STRUCTURE ceiling; the speckle
    match is reported beside it (quality['match_structure'] / ['match_speckle']). A per-frame verdict is taken
    only on a DECIDABLE frame (structure ceiling ≥ frame_ceiling_min and ≥ frame_eval_min_frac of its cells
    evaluated; the rest are served the fill and listed — CS001 v1→v2 was refused by one frame whose ceiling was
    0.09-0.11: a verdict taken on noise). The lateral shift is served PER FRAME (dx_trend: the trusted frames' dx
    put through the same step-aware MAD fill as a / b; the piecewise-constant segment model and its 'fragmented'
    escalation are retired — dx_segments keeps the saccade / step runs with their medians for the summary).

    THE DECISION TREE of register_pair (round 6 — one uniform, evidence-based rule: no NCC class and no frame
    quorum ever decides what a measured frame is served; CS001 never reaches the old 'strong' class, max 0.88;
    round 7 amended it after five refutations, R1-R5 below):
      1. coarse_register SEEDS (never rejects); fine_register measures every frame (dx, dz, a, b, peak NCC); weak
         frames (peak < vote_ncc) and pinned frames are re-searched (widened window, far range; decisive only). A
         WEAK and FLAT coarse peak (< coarse_min_ncc AND df sharpness < coarse_min_sharp) re-seeds the fine stage over
         df0 − 1 / df0 / df0 + 1 and keeps the df whose frames match best ('coarse_reseeded', quality['df_reseed'] —
         R5: a garbage df served neighbour frames as 12 one-frame segments at rel 0.52); a FRAGMENTED pair (more than
         fragment_frac of ≥ 4 segments are one-frame segments, quality['fragmented']) must reach
         fragmented_min_relative_match instead of min_relative_match.
      2. FIRST PROJECTION: a / b robust step-aware fill over the kept frames; trusted frames (kept, NCC ≥ vote_ncc,
         not pinned) vote for their live segment's shift (split_segments' step / hold rule; a run of ≥ seg_hold live
         frames with nothing trusted is DEAD: 'dx_unmeasured_run', interpolated like a crop band). A pinned frame
         (its peak at the widened search edge) is a bound: it splits, never votes, is never served as measured.
      2b. SUB-STEP PLATEAUS: inside a base segment, two plateaus of the trusted frames' own dx — each seg_hold window
         agreeing within substep_agree — a step ≥ substep_agree apart are a candidate cut (_plateau_cuts); it is
         accepted when the cluster medians score better than the served constant on the window frames by
         arbitration_margin (cell-weighted per-frame gate ratio; quality['plateau_candidates'] / ['plateau_cuts']).
         The median of a bimodal segment lands between its modes and serves every frame a few laterals off while
         passing the gate everywhere (round-6 refutation B); a ramp never qualifies (its windows are not plateaus).
         A candidate cut k is PLACED at k − 1 / k / k + 1 by scoring the frames next to it under the two cluster
         values, moving only across a NON-VOTER — a weak frame the plateau windows do not cover (R3: served the
         wrong cluster 5 laterals off at ratio 0.0, unarbitrated because 5 < substep_dx); a voter stays where the
         plateau evidence put it (CS001 v3 frame 13, own 0.95 between the clusters).
      3. ARBITRATION of every CONTRADICTION: a partnered frame with a finite fine measurement (pinned included) whose
         served value differs from it by more than the bar — substep_dx laterals for dx, axial_tol_px for a / b
         (fill-rejected frames) — or whose served value FAILS the per-frame gate while its measurement differs at
         all (> 1 lateral / 0.5 px: a 5-lateral plateau under the bar scores ≈ 0 of its ceiling on synthetic speckle)
         — is scored on that frame alone (the per-frame gate ratio: band-space matched fraction after the warp / the
         reference's own adjacent-frame ceiling on that frame) under the SERVED value and under its OWN measurement
         — own dx with the served a / b, own a / b with the served dx, and BOTH jointly on every axial
         contradiction whose own dx differs from the served dx at all (> 1 lateral; round 8: an axial contradiction
         is judged at the frame's OWN lateral shift as well as at the served one — an in-bar 5-lateral transient with
         an a +5 px step scored 0.000 at the served dx and lost) (a reference frame's warp depends on its partner's
         (dx, a, b) only, so only the contradicting frames cost a second pass, batched and memoised). A served value
         NEAR the gate (ratio below frame_match_frac + arbitration_margin) counts as failing it for the contradiction
         rule (round 8: a served ratio of 0.263 passed the 0.25 gate against the frame's own 1.000, 6 laterals off):
           own WINS    ⇔ own ≥ frame_match_frac and own BEATS served: lexicographic over ratio → frac_0.7 →
                         ncc_mean, arbitration_margin applied to the DECIDING statistic (_better; R1: the gate ratio
                         saturates at 1.0 on a periodic texture for an alias and the truth alike, the finer
                         statistics separate them) → served its own value.
                         dx: an agreeing run (substep_agree) of winners is its own SEGMENT (the median of the run;
                         a single frame included — min_segment_frames 1; quality['dx_segments_held'] folds runs
                         shorter than seg_hold into their segment for the saccade-level view). A single winner is
                         legitimate ANCHORED: its own dx within substep_dx of an anchored neighbour's measurement or
                         between two (a ramp / a saccade in flight the constant misses; anchors = frames served
                         within the bar of their own dx, arbitrated frames whose own value scored ≥ frame_match_frac,
                         and singles anchored by a frame other than the one they vouch for); an unanchored INTERIOR
                         single is 'dx_residual' (REJECT: a 1-frame lateral excursion nothing vouches for — two
                         aliases vouching only for each other included). A single at a RUN END (the volume ends, a
                         saccade / plateau cut) is anchored the same way — by the neighbour on the side it has, or by
                         its base segment's TREND (round 8: a line over ≥ 3 anchored frames within seg_hold frames on
                         its inner side, extrapolated to it — a saccade in flight the cut ends); its position no
                         longer exempts it: UNANCHORED, it is judged by the WITNESS rule below like any short run
                         (refused when its gate statistic saturated and no neighbour agrees in a and b; unsaturated,
                         the lexicographic verdict ranked it — CS001 v3 frames 99 / 100, own −8.3 / −4.9 against
                         frame 98's −30.4, stay served; quality['unanchored_end_singles']). The peak
                         NCC plays no part: a weak winner is decided exactly like a sound one (an all-weak run is
                         flagged 'weak_segment'). a / b: the winner keeps its measured a / b (kept; the fill is
                         redone through it) — the per-frame rigid axial model admits any a[f], b[f].
                         RUNS decide as ONE unit: contiguous contradicting frames whose own values agree share the
                         same evidence — a run holding a winner is served its own values on every frame, the
                         'neither' frames included (verdict 'run'; their own value beats the served one, short of the
                         bar). A PINNED frame whose bound wins is refused: 'dx_beyond_max' (|bound| > max_dx) or
                         'dx_at_search_edge' (quality['pinned_contradictions']). An own a / b candidate whose tilt
                         exceeds max_tilt_px is not admissible (the cap is the prior on a rigid B-scan move; a garbage
                         alias fit can score on its own frame with a −50 px tilt) — decided on the served value alone.
                         The JOINT candidate must also beat the served value on ncc_mean and its ratio may not fall
                         below the best single candidate's. WITNESS (R1, round 8): every carved run of ≤ 2 frames —
                         INTERIOR or touching a base boundary (the volume ends, a saccade / plateau cut), a joint
                         winner or a run the MAD fill KEPT (two adjacent aliases agree with each other and are never
                         rejected) alike — and every BASE segment of ≤ 2 frames (split_segments' short-side rule
                         makes two agreeing measured frames next to a cut / the run end a base segment of their own:
                         trusted voters, own == served, never a contradiction) whose gate statistic SATURATED on its
                         winning record or under the final served value (frac_0.5 ≥ 1 − margin: the metric could not
                         rank it) needs a measured neighbour agreeing in a AND b within axial_tol_px with the run's
                         SERVED a / b, else the run is 'dx_residual' (quality['joint_unwitnessed_runs'], the kept ones
                         also under ['kept_unwitnessed_runs']; the tested runs in ['witness_runs']; R1: two alias
                         frames at a 13.7 / −6.2, b 17 px — or a kept pair at a 43 / b 31 px — against neighbours at
                         3 / 0 px; round 8 at_end: a kept alias pair at the volume start served 36 laterals / 9 px /
                         16 px off, an alias single on each side of a saccade cut 18 / 11 laterals off, all with ok
                         True — position exempted them). RE-FILL (R2): every non-kept LIVE frame the
                         redone fill moves by more than refill_change_px is arbitrated NEXT round against its previous
                         interpolation and its own measurement — the best is served, a previous interpolation that
                         wins is PINNED as a knot of the fill (quality['refill_changed_frames'] / ['axial_pinned_
                         frames'], verdict 'previous'); a measured frame rejected by the MAD fill was otherwise
                         re-served 6-19 px off at ratio 0.000. A frame of a DEAD run (≥ seg_hold unmeasured frames,
                         no sound peak) is interpolated by design: the redone fill moves it WITHOUT a verdict (round 8:
                         a junk gap next to an axial winner was 'axial_residual' on its unmeasured frames); a measured
                         weak frame of a dead run is still arbitrated on its own evidence ('dx_untrusted_run' when a
                         run of them scores under neither value).
           served WINS ⇔ otherwise with either value scoring → the served value stays, both ratios recorded
                         (quality['arbitrated_frames']; a fill-rejected frame kept on the fill is listed under
                         quality['axial_rejected_frames']).
           NEITHER     ⇔ both ratios below frame_match_frac → a contiguous run of ≥ 2 such frames agreeing with
                         each other is a verdict with NO quorum: 'low_frame_match' (dx, a sound frame in the run),
                         'dx_untrusted_run' (dx, all weak) or 'axial_residual' (a / b only) — REJECT; a pinned one is
                         'dx_at_search_edge'; a single one is reported (quality['unscored_frames']) and counts toward
                         the quorum in 5.
      4. BOUNDARY: an unmeasured (or weak, unserved) frame next to a TRANSITION of the served shift (a segment cut or
         an override run's edge) is scored under the other side's value too and joins it when that wins the same way.
         Every round STARTS by scoring every judged frame whose served (dx, a, b) changed since its last score — a
         plateau re-projection, a carve, a merge, a boundary reassignment, a re-fill (R3; quality['rescored_frames']).
      5. The FINAL transform is scored on every judged frame (memoised — only changed frames cost anything) and the
         pair-level numbers (matched fractions, coverage, ncc_mean, surface residual) are assembled from the
         per-frame records; a CONTIGUOUS run of ≥ 2 judged frames below frame_match_frac — measured or not — refuses
         the pair with NO quorum ('low_frame_match', the run in quality['low_frame_match_runs']; R4: 2-4 unmeasured
         frames served the other side of a saccade, or a garbage gap shorter than a hold — a run of ≥ seg_hold
         unmeasured frames is dead and not judged); scattered SINGLE failures refuse by the quorum max(min_bad_frames,
         frame_bad_frac × judged). The subset path judges every unmeasured live frame and every neighbour of a failing
         frame. The global verdicts (no_correspondence, low_relative_match, dx_at_search_edge, dx_beyond_max,
         tilt_beyond_max) apply as before.
      THE INVARIANT, precisely: a live partnered frame is served its own measurement (a winner), or a value that
         scored at least as well as its own on that frame within arbitration_margin (recorded), or the pair is refused
         naming the frames — for every frame beyond the bars, every frame whose served value fails the gate, and
         every sub-step plateau held ≥ seg_hold frames. INSIDE the bars a segment's constant is served BY DESIGN
         even where the frame's own value would score higher: one lateral shift per segment is the model, and the
         per-frame ratio is sensitive to a 1-5-lateral drift on real speckle (CS001: the own value beats the served
         constant by 0.05-1.3 on ~55 % of the frames, all within substep_dx; prototype A's residual MAD 2.6-4.2). A
         slow drift of a few laterals inside a segment stays one segment; a per-frame lateral shift is not the model.
      STILL SERVED BY DESIGN (round 7-8, sanctioned and listed): drift ramps as piecewise constants (a frame up to
         ~3.3 laterals off its own inside the bar); an alias texture whose period equals the shift (0 and 7 are the
         same value to the metric; a tie keeps the served value); the in-bar per-segment constant above (a served
         value that passes the gate by more than the margin); a / b interpolated across a zeroed crop band or a dead
         run (dead frames, no measurement); a 1-frame 'neither' frame (its own value scores under frame_match_frac
         too) and a single unmeasured frame, both under the quorum; a kept tilt beyond max_tilt_px (inadmissible own:
         a single 'neither' frame). Runs of 2-4 frames with NO fine measurement are no longer served ok when they fail
         the gate (R4 refuses them). CONSERVATIVE by the witness rule: a real 1-2-frame lateral + axial excursion
         whose a / b sit more than axial_tol_px from both neighbours — in the interior, at the volume ends or next
         to a cut alike (round 8: a 1-frame −70 | −50 step with a +25 px at frame 0 is indistinguishable from an
         alias single there) — is refused 'dx_residual' at a saturated gate statistic (named, never served wrong).

    ROUND 10 — the round-0 refutations (see the module docstring): tilt_se_max (R3), pose_fit_min_rel (R1), shear_rescue /
    shear_step / shear_margin and dz_wide / dz_wide_step (R5), group_df_reseed (register_group); the mode guard's coherent ramp
    (R6) and the tilt-residual alias test (E7 revised) carry no new knob. dx CONVENTION on a resampled pair: dx_applied is in
    REFERENCE laterals on the common grid about the two centred members' common centre — the ORIGINAL reference lateral of a
    moving original lateral l is  l · lateral_scale + lateral_offset_mov + dx − lateral_offset_ref  (quality['lateral_geometry']).

    ROUND 9b — the SPECKLE feature as the final refinement and the second witness (see the module docstring):
    speckle_refine / speckle_refine_dz / speckle_refine_min_col  the fine stage's per-lateral depth search re-run on the
                      speckle feature ± speckle_refine_dz px around the structure line, accepted where the mean best
                      per-lateral speckle column NCC ≥ speckle_refine_min_col (PairResult.per_frame_speckle_ncc).
    speckle_witness / speckle_ceiling_min  the speckle _FrameScorer as the arbitration's second witness; silent on a frame
                      whose speckle ceiling (frac_0.5) is below speckle_ceiling_min.
    decisive_gain     an own win beating a passing served value by this much (ratio) is decisive when the witness backs it.
    axial_witness     an isolated axial-only run of ≤ 2 frames needs the witness (or, saturated, a neighbour) — 'axial_residual'.

    coarse_rows / coarse_ds / coarse_mask  the COARSE stage's own band and pyramid (extract_band; prototype A's
                      coarse configuration): rows (−8, 240) — the posterior included — block means ×4 lateral ×
                      ×4 depth, the Otsu tissue mask with NO posterior cap ("tissue"; "band" = canvas & valid only,
                      see COARSE_ROWS_DEFAULT). CS001_OS copies, v1 reference: coarse NCC 0.780 (v2) / 0.786 (v3).
                      Applied when the members are MemberData (extract_band runs here); a BandData already carries
                      its pyramid and is used as is (coarse_register reports the band's coarse_rows / coarse_ds).
    coarse_max_dx     ±laterals of the coarse lateral search (60 → ±15 coarse cells at COARSE_DS[0] = 4).
    coarse_max_dz     ±px of the coarse band-space depth search (40 → ±10 coarse rows at COARSE_DS[1] = 4; both
                      bands are flattened to their own served line, so the depth offset is the served-line
                      disagreement — prototype A: +0.5 / +0.9 px on CS001_OS).
    coarse_max_df     ±frames of the coarse frame-offset search (12; prototype A: df −9 / −7).
    coarse_attempt_ncc  the coarse peak is USABLE as the fine stage's seed when its masked NCC ≥ this (0.30) and it
                      does not sit on a search bound; otherwise the two halves of the moving frame range are
                      searched separately and the better half's peak seeds the fine stage ('coarse_split_seed').
                      A moving scan with a mid-volume lateral saccade has two lateral modes, so the single global
                      peak ≈ p_major · ρ can fall below 0.5 while every frame still registers (a synthetic
                      20-lateral saccade at frame 20/40: 0.494; the fine stage resolves it exactly) — the coarse
                      stage therefore never rejects a pair; 'no_correspondence' is the FINE stage's verdict.
    coarse_min_ncc    peak masked NCC below which the coarse peak is flagged 'coarse_weak' (0.5; prototype A
                      0.71-0.81; CS001_OS 0.78 / 0.79). A flag, not a rejection.
    coarse_excl       cells (coarse dx/dz) / frames around the peak excluded from the 'best off-peak' (3).
    coarse_min_sharp  df-profile sharpness (peak − best value ≥ coarse_excl FRAMES away) below which the coarse
                      peak is flagged multimodal (0.03; CS001_OS 0.037 / 0.035 on the 240-row pyramid). The 3-D
                      sharpness is reported only: a saccade inside the moving scan is a real second lateral mode
                      at the coarse scale. A flag, not a rejection.
    min_measured_frac  'no_correspondence' when fewer than this fraction of the overlapping frames measure after
                      the fine stage + projection (0.30), or when the after-transform match is poor on BOTH
                      counts (matched fraction < low_match AND coverage < low_coverage).
    fine_win_dx/dz    per-frame search window around the coarse peak (laterals / px; A used 64 / 20; 48 covers
                      the −35-lateral saccade inside case_cs001_os_v2 from a −38 coarse peak).
    fine_dz_search    ±px of the per-lateral local-NCC depth search around the frame's peak (6).
    local_win         (lateral, depth) window of the local NCC (LOCAL_WIN = (33, 25)).
    ncc_floor         per-frame masked NCC below which the frame is unmeasured (0.3, prototype A's gate).
    min_windows       laterals with a valid local-NCC window a frame needs (None → max(16, L // 10)).
    min_mask_cells    match-mask cells a B-scan needs to enter the fine stage (None → max(256, 2 % of L·T)).
    seg_dx_step       a step in dx[f] of at least this many laterals … (12)
    seg_hold          … held over at least this many frames on both sides splits a lateral segment (5).
    live_frac         a frame whose valid laterals cover less than this fraction is DEAD (reviewer band) (0.5) —
                      UNLESS the fine stage measured a peak on it: a measured frame is live whatever its lateral
                      fraction (round-2 refutation A: at true dx 65 only 49 % of the laterals overlap, every
                      frame measured 65.2 at NCC 0.94 and the projection served the clamped seed instead).
    sg_window/order   Savitzky-Golay trend across frames for the MAD rejection of a[f] and b[f] (11 / 2).
    mad_k, mad_floor  reject |residual − median| > k·MAD, the threshold never below mad_floor px (4.0 / 1.0).
    axial_step_px / axial_step_hold  the a[f] / b[f] trend is fitted PER RUN between sustained axial steps — a jump
                      of ≥ axial_step_px between consecutive measured frames held over axial_step_hold frames on
                      both sides (6 px / 2; _step_runs) — so a rigid axial saccade / blink is fitted on both sides
                      instead of bridged (round-4 refutation 1: an a-step of 10-25 px measured at NCC 0.97 was
                      rejected on 8 frames and served a ramp up to 11 px off, ok True). CS001_OS: no consecutive
                      a/b step meets the rule (largest sustained 5.8 px), the real pairs are untouched.
    substep_dx        the CONTRADICTION bar for dx (6 laterals): a frame whose own fine dx sits more than this from
                      the served shift is arbitrated (decision tree step 3); below it the constant is served (real
                      per-frame drift: CS001 ±5 laterals inside a segment, prototype A's residual MAD 2.6-4.2).
    substep_agree     frames whose own dx agree within this (3 laterals) form ONE run: a winner run (a segment or an
                      override run), a merge into a neighbouring segment, a 'neither' run — the engine's 'same value'
                      tolerance. Also the sub-step PLATEAU rule's step and window spread (decision tree 2b: on CS001
                      a step ≥ 3 with both 5-frame windows agreeing within 3 fires nowhere on v2, only at 13 on v3 —
                      the 7-12 plateau, frames 0-6 being unpartnered — and at 70 on v3→v2 where both cluster medians
                      sit within 1 lateral of the served constant, rejected as nothing to gain;
                      scratchpad/wf_fix3/fix_r1/probe/plateau_probe*.log) and the quality=False sample's threshold
                      (every frame whose own dx differs from the served shift by more than this is scored).
    axial_tol_px      the contradiction bar for a / b (3 px): a fill-rejected frame served more than this from its
                      own measurement is arbitrated; own wins → kept (served as measured).
    arbitration_margin  own beats served when its per-frame record beats the served one by more than this on the
                      DECIDING statistic — the first of ratio / frac_0.7 / ncc_mean on which the two differ by more
                      than the margin (_better; round 7: a saturated ratio, 1.0 vs 1.0, is decided on frac_0.7 then
                      ncc_mean) (0.05). Calibration (scratchpad/wf_fix3/fix_r0/probe_*_ratios.log): synthetic frames whose
                      served value equals their own within 1 lateral / 0.5 px differ by ≤ 0.05 (sd ≤ 0.008); on
                      CS001 the metric is far more sensitive (a 1-lateral change moves a frame's ratio by up to
                      0.24; a 3-5-lateral drift by up to +1.3), and every contradicting frame decided by +0.13 …
                      +1.8 (one served-win at −0.11) — the margin is well inside the evidence on both.
    min_segment_frames  an agreeing run of own-winning frames is its own SEGMENT (dx_segments) when it holds at
                      least this many frames (1: EVERY winner run, a single frame included — dx_segments is then an
                      exact description of the transform: every frame of a segment is served the segment's shift).
                      A winner run shorter than this is instead served per frame inside its segment (dx_applied
                      only; quality['dx_override_runs'] with (r0, r1, median, at_end)) — e.g. 5 = seg_hold keeps
                      dx_segments at the saccade / plateau level but a reader that assumes a segment's frames share
                      its shift is then wrong on the override frames. The transform is the same either way; the
                      saccade / plateau-level segmentation (winner runs shorter than seg_hold folded into their
                      segment) is always published as quality['dx_segments_held'] (CS001: [42] / [13, 41] there
                      against [13, 14, 39, …] / [13, 18, 19, 38, …] in dx_segments).
    max_tilt_px       |b[f]| cap, px half-span (oct_preprocess tissue_motion_max_tilt_px = 40).
    max_dx            |segment dx| above which a MEASURED segment is refused ('dx_beyond_max', ok False) — a cap
                      that is a verdict, never a clamp (80 laterals = 0.62 mm; a clamped shift is a confidently
                      wrong transform: round-2 refutation A, true dx 65 clamped to 60 → matched 0.001, ok True).
    coarse_widen      factor by which a coarse search axis whose peak sits ON its bound is widened and the search
                      re-run (2.0; ≤ 1 disables) — the adaptive window of oct_preprocess's tissue_motion
                      (tissue_motion_widen_*): a shift beyond the window is MEASURED, not clipped. Capped by the
                      pyramid (dx ≤ Lc − 1, dz ≤ Tc − 1 cells, df ≤ Fc // 2 frames).
    fine_widen        factor by which a FRAME's fine window (fine_win_dx / dz) is widened and the frame re-measured
                      when its peak sits within edge_tol of the window edge (2.0; ≤ 1 disables). A frame still
                      pinned after widening is 'at the search edge' (PairResult.dx_at_edge).
    edge_tol          laterals / px inside the window edge that still count as pinned (1; oct_preprocess:
                      |lag| ≥ max_lag − 1).
    edge_frac         a MEASURED segment with at least this fraction of its measured frames pinned after widening
                      is unmeasurable: 'dx_at_search_edge', ok False (0.5).
    min_relative_match  band-space relative match (matched_frac_0.5 / the reference's own adjacent-frame ceiling
                      on the same frames) below which the pair is 'low_relative_match' → 'no_correspondence'
                      (0.5). Real pairs sit at 1.0-1.1 (CS001 v2 1.125, v3 1.013; synthetic ≥ 0.9); a wrong
                      transform at 0.001-0.03 (round-2 refutations A/B: true dx 65 clamped to 60 → 0.004 of the
                      ceiling with coverage 0.42 — the 'matched < 0.3 AND coverage < 0.3' rule alone let it pass).
    quality_subset_frames  the CHEAP acceptance check of the quality=False path (register_group's transitivity
                      pairs): ~this many evenly spaced partnered frames PLUS every contradicting frame, scored
                      against the reference's own adjacent-frame ceiling on the SAME frames (12; 0 disables and
                      leaves the measured-frames criterion alone — never the default). Decides exactly like the
                      full path (low_match / low_relative_match / the per-frame gate); ~1/8 of the full cost.
    far_search        a WEAK frame — cells but no peak ≥ vote_ncc in its home window: unmeasured, or an in-window
                      alias — is re-measured with the widened window (×fine_widen) and, still weak, over the FAR
                      range |dx| ≤ max_dx (True) — the same adaptive rule as a pinned frame: a second lateral mode
                      beyond the window (a saccade of 55-75 laterals from the coarse seed, whose in-window alias
                      reaches 0.37; round-3 refutation C) is MEASURED, never silently inherited from the majority.
                      Only weak frames are re-searched, so the cost is bounded (CS001: 1 of 92 / 0 of 94 frames).
    far_ncc_floor / rescue_margin  a rescue must be DECISIVE: the wider search's peak ≥ far_ncc_floor (0.5) and
                      ≥ the home-window peak + rescue_margin (0.2), else the home measurement stands. Far aliases
                      on unrelated same-geometry volumes peak at ≤ 0.38 (synthetic); true beyond-window peaks
                      0.91-0.98; on CS001 v2 frame 100 (home 0.49) a far alias at 0.53 / dx 275 would otherwise
                      replace a real match and drag three neighbours off the a/b trend.
    vote_ncc          a frame VOTES for its segment's lateral shift in the first projection (and is live whatever
                      its lateral fraction) only when its peak NCC ≥ this (0.5) AND its rigid model held
                      (`measured`: a/b fitted and kept by the MAD rule) and it is not pinned — PairResult.dx_trusted.
                      Below it a finite dx is EVIDENCE for the arbitration (decision tree step 3), never a vote — and
                      there it is decided exactly like a sound frame (its own value scores or it does not; an all-weak
                      winner run is flagged 'weak_segment'): the class never decides what a frame is served.
    frame_match_frac  the per-frame acceptance bar (0.25): a frame — served, or a candidate value in the
                      arbitration — whose band-space matched fraction (> 0.5 local NCC, after the transform) is
                      below this fraction of the reference's own per-frame adjacent-frame ceiling does not score
                      (a frame served tens of laterals off matches ≈ 0.0-0.05 of its ceiling; CS001 v3's frames
                      8-12, served 8-11 laterals from their measurement, sat at 0.10-0.48 — their own measurement
                      scores 0.35-0.89 and is served now).
    min_bad_frames / min_bad_frames_subset / frame_bad_frac  'low_frame_match' (ok False) when at least
                      max(min_bad_frames, frame_bad_frac × judged frames) judged frames are bad — 5 on the full
                      check, 3 on the subset check, 10 % — the residual quorum for SCATTERED single bad frames. A
                      CONTIGUOUS run of ≥ 2 judged frames failing the gate refuses the pair with no quorum (round 7,
                      R4 — a garbage gap shorter than a hold included; it used to pass under the quorum), as does a
                      contiguous agreeing run of ≥ 2 contradicting frames that score under neither value (decision
                      tree step 3). Frames of an unmeasured run (dead by measurement) are not judged for the quorum
                      unless they measured a peak ≥ vote_ncc.
    refill_change_px  a non-kept LIVE frame whose served a or b moves by more than this (0.5 px) when the fill is redone
                      through an axial winner is CHANGED: arbitrated next round against its previous interpolation and
                      its own measurement (round 7, R2); a frame of a dead run is interpolated by design, never judged.
    coarse_reseed     re-seed the fine stage over df0 ± 1 when the coarse peak is weak (< coarse_min_ncc) AND flat
                      (sharpness < coarse_min_sharp), keeping the df with the best mean per-frame peak NCC (True;
                      round 7, R5 — three fine passes, only on such pairs; CS001 0.78 / 0.037 never qualifies).
    fragment_frac / fragmented_min_relative_match  a pair of ≥ 4 segments with more than fragment_frac (0.25) of them
                      one-frame segments is FRAGMENTED and must reach fragmented_min_relative_match (0.75) instead of
                      min_relative_match (CS001 v2 / v3: 45 / 62 % one-frame segments at 1.17 / 1.08; a garbage df 0.52).
    low_match / low_coverage  matched fraction (local NCC > 0.5 of the overlapped reference band) / coverage
                      below which the pair is flagged low_match (0.30 / 0.30 — prototype A's rule). The match
                      bar of the CS001 acceptance is the BAND-SPACE relative match: matched_frac_0.5 / the
                      reference's adjacent-frame ceiling of the same band ≥ 1.0 (v2 1.125, v3 1.013).

    PARTIAL OVERLAP (2026-09-12, CS001_OD): two scans of one eye may image regions offset by most of the scan width
    ("the right of the scar and the left of the scar with some common area": v4 / v5 / v6 against v2 sit ≈ 390 of 513
    laterals apart, sharing ≈ 110-120 laterals). Such a pair is REGISTERED ON THE OVERLAP and reported as a fraction;
    only a true non-overlap is refused. min_overlap_laterals (96 ≈ 19 % of 513) / min_overlap_frame_frac (0.40) are the
    overlap BAR: the coarse search covers dx ±(L − min_overlap_laterals) (coarse_max_dx None), a coarse shift needs
    the bar's cells at the moving mask's density (coarse_min_overlap None — _overlap_floor_cells — instead of E6's 30 %
    of the moving cells, which hid the true peak of a 23 % overlap), max_dx is L − min_overlap_laterals (max_dx None:
    'dx_beyond_max' only beyond the bar), the fine stage / coverage / relative_match are evaluated on the overlap as
    they always were (coverage = the fraction of the REFERENCE covered, 0.18-0.23 on those pairs — informational), and
    PairResult.overlap_laterals / overlap_fraction (of the moving scan's laterals on the common grid) go into the record.
    'no_overlap' (beside 'no_correspondence'): the coarse seed or the served shift leaves fewer than the bar's laterals /
    frames (the measured offset is in quality['overlap']), or the fine stage measured fewer than min_measured_frac of the
    partnered frames (no overlapping structure at all; a refused pair also records the OVERLAP-AGNOSTIC coarse peak,
    coarse_unrestricted, so a beyond-bar match is named with its offset). The alias risk the 30 % rule guarded
    against is held by the fine stage's structure match on the overlap (≥ min_relative_match of the same-scan ceiling)
    and the top-K coarse seeds each scored by the fine stage (a coarse peak that is not the single separated maximum
    within coarse_seed_tol is confirmed or replaced by the seeds' df-scores). 'partial_overlap' (informational): the
    served overlap covers less than half of the moving laterals.

    BAR EDGE (2026-09-12, refutation R1 — bar_margin / bar_edge_run; bar_edge_check): the fine window is capped at max_dx, so a
    true offset a few laterals BEYOND the bar (CS001_OD v6 → v1: 92 shared against the 96 bar, truth −421) is measured ~11
    laterals short INSIDE the bar and served ok (−409.5, matched 0.75 against 0.87 at the truth), and a saccade that carries
    half the frames beyond the bar (v5 → v1 frames 53-100, truth −430..−465) is served the in-bar fill (−409..−415) as ok.
    Now a pair whose served |dx| median, or a run of ≥ bar_edge_run partnered frames, lies within bar_margin laterals of
    max_dx, or whose OVERLAP-AGNOSTIC coarse peak (coarse_unrestricted, computed for EVERY pair now) lies beyond the bar by
    more than one coarse cell, is JUDGED: the served transform is scored with pair_quality on the scope's frames against
    beyond-bar candidates (the served dx shifted outward by one coarse cell at a time up to max_dx + 2·bar_margin, and the
    scope's own coarse peak — the whole-volume one, or the run's frame-masked one — when it lies beyond the bar at the same
    df); the in-bar value is CONFIRMED only when it beats every scoreable candidate outright (a tie refuses; informational
    'near_search_edge'), else the pair is refused 'dx_at_search_edge' with the winning candidate in quality['overlap']
    ['offset'] (source 'bar_edge_judge') and the scope's frames listed under quality['beyond_bar_frames'] — unmeasured
    (live / measured False), never served the fill as ok — and 'no_overlap' beside it when they exceed 1 −
    min_overlap_frame_frac of the partnered frames. The record is quality['bar_edge']. A pair already refused by another
    verdict is not judged (its unrestricted peak is still recorded so the reason names it — R2, overlap_reason)."""
    coarse_rows: tuple[int, int] = COARSE_ROWS_DEFAULT
    coarse_ds: tuple[int, int] = COARSE_DS
    coarse_mask: str = COARSE_MASK_DEFAULT
    coarse_max_dx: int | None = None  # ±laterals of the coarse lateral search; None → L − min_overlap_laterals (every offset that
    #                                   keeps the overlap bar's laterals; capped by the pyramid: Lc − 1 cells) — PARTIAL OVERLAP
    coarse_max_dz: int = 200          # ±px of the coarse band-space depth search (capped: Tc − 1 cells)
    coarse_max_df: int = 50           # ±frames of the coarse frame-offset search (capped: Fc // 2)
    coarse_min_overlap: float | None = None   # None → the ABSOLUTE overlap bar (min_overlap_laterals × min_overlap_frame_frac of the
    #                                   live frames at the moving mask's density, halved — _overlap_floor_cells); a float = the legacy
    #                                   fraction of the MOVING cells (E6's 0.30, kept for the synthetic suite)
    min_overlap_laterals: int = 96    # PARTIAL OVERLAP (2026-09-12): the fewest laterals two scans must SHARE (≈ 19 % of 513 = 0.75 mm) —
    #                                   the coarse search runs over dx ±(L − this), max_dx = L − this, and a coarse seed / served shift
    #                                   leaving fewer laterals is 'no_overlap' (the measured offset stays in the record)
    min_overlap_frame_frac: float = 0.40   # … and the fewest FRAMES the overlap must span (fraction of the moving frames)
    bar_margin: int = 12              # BAR EDGE (2026-09-12, R1): a served |dx| (its median, or a run of ≥ bar_edge_run frames) within this many
    #                                   laterals of max_dx is AT THE SEARCH EDGE and judged against beyond-bar candidates (bar_edge_check)
    bar_edge_run: int = 5             # … the run length of edge-zone frames that triggers the judge on its own
    coarse_attempt_ncc: float = 0.30
    coarse_min_ncc: float = 0.5
    coarse_excl: int = 3
    coarse_min_sharp: float = 0.015   # df-profile sharpness below which 'coarse_multimodal' (structure feature)
    coarse_seed_k: int = 4            # E6: up to K separated coarse maxima within coarse_seed_tol are seeds
    coarse_seed_tol: float = 0.03     # … of the peak NCC; each is scored by a cheap fine pass, the best is kept
    min_measured_frac: float = 0.30
    fine_win_dx: int = 96             # per-frame lateral window around the seed (laterals)
    fine_win_dz: int = 40             # per-frame depth window around the seed (px)
    fine_dz_search: int = 10          # ±px of the per-lateral local-NCC depth search around the frame's dz
    fine_rms_recentre: float | None = None   # px: a line fit worse than this re-centres the per-lateral search on the
    #                                          fitted a + b·x (None → fine_dz_search / 2)
    fine_rms_max: float | None = None        # px: a frame whose fit still exceeds this is UNMEASURED (None → fine_dz_search)
    feature_sigma_struct: tuple[float, float] = FEATURE_SIGMA_STRUCT   # (lateral, depth) px, extract_band
    feature_sigma_speckle: tuple[float, float] = FEATURE_SIGMA_SPECKLE
    local_win: tuple[int, int] = LOCAL_WIN_STRUCT       # the STRUCTURE feature's local-NCC window (the verdict)
    local_win_speckle: tuple[int, int] = LOCAL_WIN_SPECKLE   # the speckle report's window
    ncc_floor: float = 0.3
    min_windows: int | None = None
    min_mask_cells: int | None = None
    seg_dx_step: float = 12.0
    seg_hold: int = 5
    live_frac: float = 0.5
    sg_window: int = 11
    sg_order: int = 2
    mad_k: float = 4.0
    mad_floor: float = 1.0
    axial_step_px: float = 6.0
    axial_step_hold: int = 2
    dx_trend: bool = True             # E9: dx served PER FRAME (the a/b fill applied to the trusted frames' dx)
    dx_step: float = 6.0              # laterals: the dx fill's sustained-step rule …
    dx_step_hold: int = 3             # … held over this many frames on both sides (a 2-frame lateral step is arbitrated, not kept:
    #                                   two adjacent period aliases with agreeing a / b were otherwise kept as a saccade)
    substep_dx: float = 6.0
    substep_agree: float = 3.0
    axial_tol_px: float = 3.0
    arbitration_margin: float = 0.05
    min_segment_frames: int = 1
    max_tilt_px: float = 40.0         # px half-span: cap on the RESIDUAL tilt |b − b_lines(dx)| (E7)
    max_tilt_abs_px: float = 300.0    # px half-span: sanity cap on |b| itself
    max_dx: float | None = None       # laterals: |dx| above which a measured shift is refused; None → L − min_overlap_laterals
    coarse_widen: float = 2.0
    fine_widen: float = 2.0
    edge_tol: int = 1
    edge_frac: float = 0.5
    min_relative_match: float = 0.75      # STRUCTURE relative match (cross-patient null 0.60-0.67; weakest true pair 0.90)
    min_relative_match_speckle: float = 0.40   # E10: the speckle report as a second witness (null 0.24-0.30; true ≥ 0.62)
    quality_subset_frames: int = 12
    far_search: bool = True
    far_ncc_floor: float = 0.7        # structure scale: an unrelated cornea's per-frame peak reaches 0.6-0.7
    rescue_margin: float = 0.2
    vote_ncc: float = 0.7             # structure scale: real frames 0.75-0.95, the cross-patient null 0.6-0.7
    frame_match_frac: float = 0.25
    min_bad_frames: int = 5
    min_bad_frames_subset: int = 3
    frame_bad_frac: float = 0.10
    low_match: float = 0.30
    low_coverage: float = 0.30
    refill_change_px: float = 0.5
    coarse_reseed: bool = True
    ridge_axial_tol: float = 8.0      # E9 ridge guard (px): a candidate that jumps in BOTH dx (> seg_dx_step / substep_dx)
    #                                   and a (> this) against its neighbours is the dome-ridge alias of a smooth feature
    excursion_max_dx: float = 36.0    # laterals: a decisive unanchored single is a microsaccade only within this of the fill
    #                                   (≤ 0.3 mm); farther, or more than max_excursions of them, the frames are aliases
    max_excursions: int = 3           # … per pair, or 8 % of the partnered frames if more ('dx_residual' beyond)
    decisive_own_min: float = 0.5     # E9: a single-frame own win is DECISIVE only when its own ratio reaches this (a junk
    #                                   frame scores under both values; its 'win' at 0.27 is chance)
    frame_ceiling_min: float = 0.15   # E9(b): a frame whose structure per-frame ceiling (frac_0.5) is below this is
    #                                   UNDECIDABLE — served the fill, never arbitrated, never refused
    frame_eval_min_frac: float = 0.10   # … or whose evaluated cells are fewer than this fraction of its reference cells
    pose_max_deg: float = 8.0         # E8: pose angle (from b_lines at the coarse seed) beyond which the pair is
    #                                   'pose_beyond_frame_rigid' (a member-level, non-contributing verdict)
    pose_overlap_min: float = 0.70    # E8: the pose verdict supersedes the match verdicts when rel_struct reaches this (the
    #                                   pair overlaps: P5 v1 0.73-0.79 against its siblings; the cross-patient null 0.56-0.67)
    lateral_scale_tol: float = 0.005  # E4: relative lateral-spacing difference above which the moving member is resampled
    speckle_report: bool = True       # E10: report the speckle-scale match next to the structure match
    speckle_refine: bool = True       # E10: refine (a, b) per frame on the SPECKLE feature where it correlates (fine stage:
    #                                   the per-lateral depth search re-run on feat_speckle around the structure line)
    speckle_refine_min_rel: float = 0.5   # (reporting) pair-level speckle relative match the report calls 'correlating'
    speckle_refine_dz: int = 3        # px: the speckle refinement's per-lateral depth search around the structure line
    speckle_refine_min_col: float = 0.25   # mean best per-lateral speckle column NCC a frame needs for the refinement (a
    #                                   matching frame 0.35-0.85, an uncorrelated / junk frame 0.05-0.25)
    speckle_witness: bool = True      # E10: the speckle scorer as the SECOND WITNESS of single / weak own wins
    speckle_ceiling_min: float = 0.10 # a frame whose speckle ceiling (frac_0.5) is below this has no speckle witness
    decisive_gain: float = 0.30       # E9: an own win beating the served value by this much (ratio) is decisive when the
    #                                   speckle witness corroborates it (or is silent on that frame)
    axial_witness: bool = True        # E12: an ISOLATED axial-only own-win run of <= 2 frames needs the speckle witness (or,
    #                                   when it is silent, must not saturate the gate statistic) — else 'axial_residual'
    # ── ROUND 10 (2026-09-11, fix_r1: the round-0 refutations R1 / R2 / R3 / R5 / R6 and the two alias holes) ────────
    tilt_se_max: float = 3.0          # R3: px half-span — a frame whose tilt STANDARD ERROR (the line's residual over the
    #                                   effective number of independent local-NCC windows across the fitted laterals and their
    #                                   x-spread: a half-overlap frame has ~4 independent samples on half a lever arm) exceeds
    #                                   this is served the robust across-frame TREND of the precise frames' tilt, its a re-fitted
    #                                   at the overlap's centre (quality['tilt_low_precision']); a half-overlap tilt was measured
    #                                   to ±5-13 px and SERVED (the served depth over the overlap 5-10 px off on 10-18 frames)
    pose_fit_min_rel: float = 0.85    # R1: a pair beyond pose_max_deg whose per-frame rigid model FITS — no REJECT verdict and
    #                                   rel_struct ≥ this — is served ok with the informational 'pose_high'; the non-contributing
    #                                   'pose_beyond_frame_rigid' is the verdict only when the model does not fit (a decentring of
    #                                   150-250 laterals on a dome always implies 8.5-15° between the B-scan planes — a tilt about
    #                                   the FRAME axis is exactly the per-frame rigid model; P5_OS v1 stays non-contributing at
    #                                   rel_struct 0.73)
    shear_rescue: bool = True         # R5: a frame WEAK on the plain frame-level FFT is re-matched with the moving B-scan pre-SHEARED
    shear_margin: float = 0.1         #     … the winning shear must reach vote_ncc + shear_margin (a best-of-7 search on a junk
    #                                   frame reaches vote_ncc alone) and beat the plain peak by rescue_margin
    shear_step: int = 12              #     by ± shear_step … max_tilt_px px half-span (a served-line slope error shears the flattened
    #                                   band: the rigid-shift FFT of a band sheared by 35 px peaks at 0.5 and every frame was weak);
    #                                   the winning shear must reach vote_ncc and beat the plain peak by rescue_margin, and the
    #                                   per-lateral search is then centred on it ('fine_shear_rescued', quality['shear_frames'])
    group_df_reseed: bool = True      # register_group: a transitivity pair whose df disagrees with the composition of the two direct
    #                                   pairs by ≥ 2 frames is re-run seeded at the COMPOSED df (the group as a df witness on a flat
    #                                   df profile) and kept when it registers as well (CS032 v1_4→v1_3: +32 direct, +30 composed,
    #                                   the flattest df profile of the three at sharpness 0.004)
    dz_wide: bool = True              # R5: when the first per-lateral line fit is poor (rms > rms_recentre) a WIDE coarse depth
    dz_wide_step: int = 3             #     search (dz ± max_tilt_px in dz_wide_step px) fits the line the ±fine_dz_search grid
    #                                   cannot reach, and the fine search is re-centred on it too (a 35-px served-line slope error
    #                                   under the cap was rms-rejected on 92 of 94 frames and the pair refused)

    def resolved(self, L: int) -> "PairParams":
        """PARTIAL OVERLAP (2026-09-12): the params with the lateral ranges made CONCRETE for a pair of L laterals on the common
        grid — coarse_max_dx / max_dx None → L − min_overlap_laterals (the whole admissible range: every offset that leaves the
        overlap bar's laterals). Returns self when both are set (the synthetic suite pins the legacy 300 / 300)."""
        if self.coarse_max_dx is not None and self.max_dx is not None:
            return self
        adm = max(1, int(L) - int(self.min_overlap_laterals))
        return _dc_replace(self, coarse_max_dx=(int(self.coarse_max_dx) if self.coarse_max_dx is not None else int(adm)),
                           max_dx=(float(self.max_dx) if self.max_dx is not None else float(adm)))

    def as_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @property
    def rms_recentre(self) -> float:
        return float(self.fine_rms_recentre) if self.fine_rms_recentre is not None else float(self.fine_dz_search) / 2.0

    @property
    def rms_max(self) -> float:
        return float(self.fine_rms_max) if self.fine_rms_max is not None else float(self.fine_dz_search)


def _pair_params(params) -> PairParams:
    if params is None:
        return PairParams()
    if isinstance(params, PairParams):
        return params
    return PairParams(**{k: v for k, v in dict(params).items() if k in PairParams.__dataclass_fields__})


@dataclass
class PairResult:
    """register_pair output: the RIGID transform of a moving member onto the reference plus its evidence.

    Convention (module-wide): MOVING index + shift = REFERENCE index. For moving voxel (l, z, f):
        f_ref = f + df;   l_ref = l + dx_applied[f];   z_ref = z + a[f] + b[f] · x(l),  x(l) = (l − (L−1)/2) / ((L−1)/2)
    i.e. per B-scan a rigid axial shift + tilt (the preprocessing rule: an axial B-scan is captured instantaneously,
    so only a rigid per-frame move is admissible), ONE lateral shift per live segment (like the pipeline's
    lateral_shift stage), an integer frame offset. a[f] is the TOTAL axial move in corrected rows: it already
    contains the band-space depth offset and the served-line difference (Δz = dz_band + S_ref − S_mov); dz0 is
    the coarse stage's constant band-space depth offset (the prior of the fine search), NOT applied on top.

    df            integer frame offset (prototype A on CS001_OS: v2→v1 −9, v3→v1 −7).
    dz0           coarse band-space depth offset, px (moving band row + dz0 = reference band row).
    dx_segments   [(f0, f1, dx)] — moving frames f0 ≤ f < f1 (a live segment between reviewer bands / measured
                  saccades / sub-step plateaus, or an own-winning run of the arbitration — a single frame included)
                  share the lateral shift dx (the segment MEDIAN of the measured per-frame dx, |dx| ≤ max_dx): an
                  EXACT description of the applied transform with the default min_segment_frames 1 (a run shorter
                  than min_segment_frames is otherwise served per frame: quality['dx_override_runs']). The
                  saccade / plateau-level segmentation — winner runs shorter than seg_hold folded into their
                  segment — is quality['dx_segments_held'] (CS001: [42] / [13, 41] cuts).
    a, b          (F,) per-frame axial shift (px) and tilt (px half-span, |b| ≤ max_tilt_px on interpolated frames):
                  robustified against a STEP-AWARE Savitzky-Golay trend (MAD rejection per run between sustained
                  axial steps) and filled by interpolation (held at the ends); a_raw / b_raw are the FINE stage's
                  measurements (NaN only where the fine stage measured nothing); `measured` (F,) says which frames
                  are served their own a / b (kept by the fill, or a fill-rejected frame whose own measurement WON
                  the arbitration). A fill-rejected frame is served the interpolation only when that scores at
                  least as well on its frame (quality['arbitrated_frames'] / ['axial_rejected_frames']).
    dx_per_frame  (F,) MEASURED per-frame lateral shift (NaN unmeasured) — dx_applied (F,) is what is APPLIED: the
                  segment's shift, or the frame's own measurement on an override run; prototype A: median −34.8
                  (v2) / −28.4 (v3), a real −35-lateral saccade at v2 frame 42 splits v2 into two segments. A frame
                  contradicting its segment's shift by more than substep_dx (or under it when the served value fails
                  the per-frame gate) is arbitrated on its own frame (PairParams): served its own measurement (a
                  segment of ≥ min_segment_frames frames, else an override) when that scores better, else the
                  segment's shift, both ratios recorded.
    dz_per_frame  (F,) frame-level band-space depth offset from the fine FFT search (NaN unmeasured).
    dz_band       (L, F) per-lateral band-space depth offset of the local-NCC search (moving laterals; NaN = no window).
    dx_at_edge    (F,) the frame's fine peak sits within edge_tol of its search window edge AFTER the adaptive
                  widening (fine_widen) — such a frame's dx is a bound, not a measurement.
    dx_trusted    (F,) the frame VOTES in the first projection: measured (a/b kept), peak NCC ≥ vote_ncc and not
                  pinned. Every other finite dx is reported (dx_per_frame) and is EVIDENCE for the arbitration,
                  decided like any other frame ('weak_segment' marks an all-weak winner run).
    dx_search     (F,) which search measured the frame: 0 the home window, 1 the widened window (×fine_widen),
                  2 the far range |dx| ≤ max_dx (far_search); −1 nothing measured.
    live          (F,) the frame belongs to a live segment: its transform is measured or bridged between measured
                  frames of the same segment. False = DEAD — a reviewer band / zero-filled frame (live_frames) or
                  an unmeasured run (quality['unmeasured_runs']): the served a / b / dx_applied there is an
                  interpolation between neighbours, unsupported by any measurement of that frame.
    per_frame_ncc (F,) peak masked NCC of the per-frame FFT search (prototype A median 0.80-0.81);
    per_frame_local_ncc (F,) mean of the best per-lateral local-NCC column scores; n_windows (F,) laterals with a
                  valid window; fit_rms (F,) rms residual of the line fit (the non-rigid remainder, A median 1.2-1.4 px).
    ncc_coarse, peak_sharpness, coarse  the coarse stage: ncc_coarse = the FULL-volume peak masked NCC on the
                  coarse pyramids (CS001_OS: 0.780 / 0.786); coarse = dict with the seed the fine stage used
                  (dx0 / dz0 / df0, seed_ncc, seed_source 'full' | 'half_0' | 'half_1'), the full peak ('full'),
                  the half searches when they ran ('split'), the df profile / axes / bounds.
    matched_frac_0_5 / _0_3, coverage, ncc_mean  quality AFTER the transform: warp_band → band_similarity against
                  the reference band (fractions of the OVERLAPPED reference match-mask voxels with local NCC > 0.5 /
                  > 0.3; coverage = overlapped / all reference match-mask voxels; prototype A original-space rigid:
                  0.34-0.35 > 0.5, coverage 0.85-0.89, ncc_mean 0.38). ceiling = band_similarity(ref, ref,
                  frame_offset=1).stats of the reference (the adjacent-frame speckle ceiling of the SAME metric in
                  BAND space; CS001_OS v1 on the (−8, 120) posterior-capped band: 0.3367 > 0.5);
                  relative_match = matched_frac_0_5 / ceiling['matched_frac_0.5'] in BAND space — the acceptance
                  bar is ≥ 1.0 (CS001_OS: 1.125 / 1.013). quality['relative_match_orig_space_prototype_A'] is the
                  same match against prototype A's ORIGINAL-space ceiling (0.2202; A called 1.56-1.59× on its own
                  rigid match; CS001_OS here 1.72 / 1.55) — two different denominators, both reported.
    quality       dict with the raw stats (n_eval, matched_of_ref, surface_residual_rms_px = rms of the carried
                  moving served line minus the reference line over the overlap; A rigid 2.3 / 2.8 px) and the two
                  ceilings under clear names: ceiling_band_space / relative_match_band_space and
                  ceiling_orig_space_prototype_A / relative_match_orig_space_prototype_A.
    flags         'no_correspondence' — the FINE stage's verdict (never the coarse stage's): fewer than
                  min_measured_frac of the overlapping frames measured, or matched_frac_0_5 < low_match AND
                  coverage < low_coverage, or the band-space relative match < min_relative_match
                  ('low_relative_match'), or the per-frame gate judged no frame ('unjudged'), or no overlap at all;
                  the result keeps its evidence (per-frame NCC, measured frames, the projected a/b/dx) but `ok` is
                  False and the transform must not be applied. 'dx_at_search_edge' — a measured segment whose
                  frames still pin at the WIDENED fine window (edge_frac), or a pinned frame whose bound the served
                  value does not fit (quality['pinned_contradictions']): the shift is a bound, not a measurement;
                  ok False. 'dx_beyond_max' — a measured segment's |dx| > max_dx, or a winning frame's / a pinned
                  frame's |dx| > max_dx (quality['beyond_max_frames']); ok False (never clamped).
                  'tilt_beyond_max' — a kept frame's |b| > max_tilt_px; ok False (never clamped:
                  quality['tilt_beyond_max_frames']). 'dx_residual' — an unanchored interior frame whose own dx wins
                  the arbitration: within substep_dx of no anchored neighbour's measurement and between none
                  (quality['dx_residual_runs']); ok False: a 1-frame lateral excursion nothing vouches for — or a
                  carved run of ≤ 2 frames (interior or at a base boundary; a joint winner, or a pair the MAD fill
                  kept) or a base segment of ≤ 2 frames at a saturated gate statistic with no measured neighbour
                  agreeing in a and b with its served a / b (quality['joint_unwitnessed_runs']; the kept ones also
                  in ['kept_unwitnessed_runs']; an unanchored end single that the witness rule kept is listed in
                  ['unanchored_end_singles'], served).
                  'low_frame_match' — ≥ max(min_bad_frames,
                  frame_bad_frac × judged) judged frames match below frame_match_frac of their own per-frame
                  ceiling (quality['bad_frames']), or a CONTIGUOUS run of ≥ 2 judged frames does (measured or not;
                  quality['bad_runs'] / ['low_frame_match_runs']), or a contiguous agreeing run of ≥ 2 contradicting
                  sound frames scores under neither its measurement nor the served value; ok False. 'dx_untrusted_run' — such a run of WEAK frames (peak < vote_ncc;
                  quality['dx_untrusted_runs']); 'axial_residual' — such a run of fill-rejected frames whose a / b
                  score under neither the fill nor their measurement (quality['axial_residual_runs']); ok False.
                  REJECT_FLAGS lists the eight verdicts.
                  'dx_unmeasured_run' — ≥ seg_hold consecutive live frames measured nothing trusted even over the
                  full range (quality['unmeasured_runs']): they are DEAD (the segments split there, their
                  transform is interpolated like a crop band's) — reported, not refused; frames of such a run
                  with a peak ≥ vote_ncc are still judged by the gate. 'arbitrated' — at least one frame or plateau
                  cut was arbitrated (quality['arbitrated_frames']: served / own ratios and the verdict per frame —
                  'own' / 'served' / 'neither' / 'run' (joined a winner run) / 'neighbour' (a boundary frame) — with
                  'joint' when the joint (dx, a, b) candidate decided; frames reassigned across a transition under
                  quality['boundary_frames']; 'neither' frames that joined a run under quality['joined_frames'];
                  single 'neither' frames under quality['unscored_frames']; own-winning runs shorter than
                  min_segment_frames under quality['dx_override_runs']; the plateau cuts and their scores under
                  quality['plateau_cuts'] / ['plateau_candidates'] (with 'placed_from'); frames the redone fill moved
                  and re-arbitrated under quality['refill_changed_frames'], those pinned to a previous interpolation
                  (verdict 'previous') under ['axial_pinned_frames']; frames re-scored at a round start under
                  ['rescored_frames']; quality['weak_contradictions'] is kept for the round-5 diagnostics and is always
                  empty). 'coarse_reseeded' — the weak-and-flat coarse peak's df was replaced by a neighbour df whose
                  frames match better (quality['df_reseed']); quality['fragmented'] / ['one_frame_segments'] carry the
                  fragmentation rule's numbers. 'weak_segment' — a segment or override run served on an
                  agreeing run of frames below vote_ncc whose own shift scores (quality['weak_segments']). 'fine_rescued' /
                  'fine_far_search' — frames unmeasured in the home window measured in the widened / full-range
                  search. 'coarse_weak' (full coarse peak < coarse_min_ncc), 'coarse_on_bound' (after any
                  widening), 'coarse_widened' (a bound axis was widened ×coarse_widen and re-searched),
                  'coarse_multimodal', 'coarse_split_seed' (the full peak was unusable, a frame-half seeded the
                  fine stage), 'fine_widened' (frames re-measured with the widened window), 'few_measured_frames'
                  (< half the overlapping frames measured), 'low_match'.
                  quality['decision_source'] says which acceptance check decided: 'full' (quality=True) or
                  'subset' (quality=False: the cheap check on quality_subset_frames frames plus every arbitrated
                  frame, numbers under quality['subset']); the top-level matched_frac_0_5 / coverage /
                  relative_match stay NaN on the subset path.
    timings       seconds per stage; params the PairParams used; shape (L, T, F) of the bands.
    overlap_laterals / overlap_fraction  PARTIAL OVERLAP (2026-09-12): the laterals the two scans share under the served
                  shift (median over the partnered live frames of L − |dx_applied|) and that as a fraction of the moving
                  scan's laterals on the common grid (CS001_OD v5 → v2: ≈ 110 laterals, 0.21). quality['overlap'] = {bar,
                  seed, served, verdict 'ok' | 'below_bar' | 'none_measured' | 'poor_match', offset (the measured offset
                  of a below-bar pair), unrestricted_peak (the overlap-agnostic coarse peak of a refused pair)}. Flags:
                  'no_overlap' (beside 'no_correspondence') — the seed / served shift leaves fewer laterals or frames
                  than the bar, or nothing measured; 'partial_overlap' — the served overlap is under half the laterals
                  (informational, ok stays True). BAR EDGE (R1): 'dx_at_search_edge' also when the bar-edge judge
                  (bar_edge_check, quality['bar_edge']) finds a beyond-bar candidate that the served in-bar value does
                  not beat — quality['beyond_bar_frames'] are then unmeasured (live / measured False) and the winner is
                  quality['overlap']['offset'] (source 'bar_edge_judge'); 'near_search_edge' (informational) when the
                  served value was judged and confirmed."""
    ref_cid: str
    mov_cid: str
    df: int
    dz0: float
    dx_segments: list
    a: np.ndarray
    b: np.ndarray
    dx_per_frame: np.ndarray
    dx_applied: np.ndarray
    dz_per_frame: np.ndarray
    a_raw: np.ndarray
    b_raw: np.ndarray
    measured: np.ndarray
    per_frame_ncc: np.ndarray
    per_frame_local_ncc: np.ndarray
    n_windows: np.ndarray
    fit_rms: np.ndarray
    dz_band: np.ndarray
    dx_at_edge: np.ndarray
    dx_trusted: np.ndarray
    dx_search: np.ndarray
    live: np.ndarray
    ncc_coarse: float
    peak_sharpness: float
    coarse: dict
    matched_frac_0_5: float
    matched_frac_0_3: float
    coverage: float
    ncc_mean: float
    ceiling: dict
    relative_match: float
    quality: dict
    flags: list
    timings: dict
    params: dict
    shape: tuple
    # ── round 9: the between-scan geometry record ────────────────────────────────────────────────────────────
    lateral_scale: float = 1.0          # E4: moving laterals were resampled by this factor (spacing_mov / spacing_ref)
    lateral_offset_mov: int = 0         # E4: grid lateral of the moving member = original · lateral_scale + offset
    lateral_offset_ref: int = 0         # E4: grid lateral of the reference = original + offset (a centred pad)
    pose_angle_deg: float = float("nan")   # E8: atan(median|b_lines| · sp_depth / (half-span · sp_lateral)) at the seed
    b_lines: np.ndarray | None = None   # (F,) half-span slope the two SERVED LINES imply at the frame's dx (px)
    a_lines: np.ndarray | None = None   # (F,) its intercept (px)
    per_frame_speckle_ncc: np.ndarray | None = None   # E10: mean best per-lateral SPECKLE column NCC at the structure line
    speckle_refined: np.ndarray | None = None         # E10: the frame's a / b were refined on the speckle feature
    b_se: np.ndarray | None = None                    # R3 (round 10): the per-frame tilt standard error (px half-span)
    # PARTIAL OVERLAP (2026-09-12): the laterals the two scans share under the SERVED shift (median over the partnered live
    # frames of L − |dx_applied|, common-grid laterals) and that as a fraction of the moving scan's laterals; the bar, the
    # seed's / served / overlap-agnostic offsets and the verdict live in quality['overlap']
    overlap_laterals: float = float("nan")
    overlap_fraction: float = float("nan")

    @property
    def n_frames(self) -> int:
        return int(self.a.size)

    @property
    def ok(self) -> bool:
        return not (REJECT_FLAGS & set(self.flags))

    def frame_partner(self, f: int) -> int | None:
        """Reference frame of moving frame f, None outside the reference."""
        fr = int(f) + int(self.df)
        return fr if 0 <= fr < int(self.shape[2]) else None

    def summary(self) -> dict:
        """Compact, JSON-serialisable digest (the acceptance numbers)."""
        m = self.measured
        med = lambda v: (float(np.nanmedian(v[m])) if m.any() and np.isfinite(v[m]).any() else float("nan"))  # noqa: E731
        rng = lambda v: ([float(np.nanmin(v)), float(np.nanmax(v))] if np.isfinite(v).any() else [None, None])  # noqa: E731
        return {"ref": self.ref_cid, "mov": self.mov_cid, "df": int(self.df), "dz0": round(float(self.dz0), 3),
                "ncc_coarse": round(float(self.ncc_coarse), 4), "peak_sharpness": round(float(self.peak_sharpness), 4),
                "n_df_peaks": self.coarse.get("n_df_peaks"), "coarse_dx0": self.coarse.get("dx0"),
                "coarse_seed_source": self.coarse.get("seed_source"), "coarse_seed_ncc": self.coarse.get("seed_ncc"),
                "coarse_rows": self.coarse.get("coarse_rows"), "coarse_ds": self.coarse.get("coarse_ds"),
                "coarse_split": self.coarse.get("split"),
                "dx_segments": [(int(f0), int(f1), round(float(dx), 2)) for f0, f1, dx in self.dx_segments],
                "dx_frame_median": round(med(self.dx_per_frame), 2), "dx_frame_range": rng(self.dx_per_frame),
                "dz_frame_median": round(med(self.dz_per_frame), 3),
                "a_range": rng(self.a), "b_range": rng(self.b), "measured_frames": int(m.sum()),
                "overlap_frames": int(sum(1 for f in range(self.n_frames) if self.frame_partner(f) is not None)),
                "ncc_frame_median": round(med(self.per_frame_ncc), 4),
                "local_ncc_frame_median": round(med(self.per_frame_local_ncc), 4),
                "fit_rms_median": round(med(self.fit_rms), 3),
                "matched_frac_0.5": round(float(self.matched_frac_0_5), 4), "matched_frac_0.3": round(float(self.matched_frac_0_3), 4),
                "coverage": round(float(self.coverage), 4), "ncc_mean": round(float(self.ncc_mean), 4),
                "ceiling": {k: round(float(v), 4) for k, v in self.ceiling.items() if not isinstance(v, np.ndarray)},
                "relative_match": round(float(self.relative_match), 3),
                "ceiling_band_space_0.5": self.quality.get("ceiling_band_space", {}).get("matched_frac_0.5"),
                "relative_match_band_space": self.quality.get("relative_match_band_space"),
                "ceiling_orig_space_prototype_A_0.5": self.quality.get("ceiling_orig_space_prototype_A"),
                "relative_match_orig_space_prototype_A": self.quality.get("relative_match_orig_space_prototype_A"),
                "relative_to_prototype_A_orig_ceiling": self.quality.get("relative_to_prototype_A_orig_ceiling"),
                "measured_frac": self.quality.get("measured_frac"),
                "decision_source": self.quality.get("decision_source"),
                "subset": self.quality.get("subset"),
                "coarse_widened_axes": self.coarse.get("widened_axes"),
                "fine_widened_frames": self.quality.get("fine_widened_frames"),
                "dx_at_edge_frames": int(np.asarray(self.dx_at_edge, bool).sum()),
                "dx_trusted_frames": int(np.asarray(self.dx_trusted, bool).sum()),
                "live_frames": int(np.asarray(self.live, bool).sum()),
                "dx_search_counts": {str(k): int((np.asarray(self.dx_search) == k).sum()) for k in (0, 1, 2)},
                "dx_mad_segments": self.quality.get("dx_mad_segments"),
                "dx_residual_runs": self.quality.get("dx_residual_runs"),
                "low_frame_match_runs": self.quality.get("low_frame_match_runs"),
                "dx_untrusted_runs": self.quality.get("dx_untrusted_runs"),
                "axial_residual_runs": self.quality.get("axial_residual_runs"),
                "axial_rejected_frames": self.quality.get("axial_rejected_frames"),
                "unmeasured_runs": self.quality.get("unmeasured_runs"),
                "n_contradicting": self.quality.get("n_contradicting"),
                "arbitration_verdicts": _verdict_counts(self.quality.get("arbitrated_frames")),
                "weak_segments": self.quality.get("weak_segments"),
                "dx_override_runs": self.quality.get("dx_override_runs"),
                "dx_segments_held": self.quality.get("dx_segments_held"),
                "plateau_cuts": self.quality.get("plateau_cuts"),
                "pinned_contradictions": self.quality.get("pinned_contradictions"),
                "beyond_max_frames": self.quality.get("beyond_max_frames"),
                "joined_frames": self.quality.get("joined_frames"),
                "boundary_frames": self.quality.get("boundary_frames"),
                "unscored_frames": self.quality.get("unscored_frames"),
                "scorer_calls": self.quality.get("scorer_calls"), "scorer_frames": self.quality.get("scorer_frames"),
                "tilt_beyond_max_frames": self.quality.get("tilt_beyond_max_frames"),
                "bad_frames": self.quality.get("bad_frames"), "n_frames_evaluated": self.quality.get("n_frames_evaluated"),
                "frame_ratio_min": self.quality.get("frame_ratio_min"),
                "sharpness_3d": self.coarse.get("sharpness_3d"),
                "surface_residual_rms_px": self.quality.get("surface_residual_rms_px"),
                # PARTIAL OVERLAP (2026-09-12)
                "overlap_laterals": (round(float(self.overlap_laterals), 1) if np.isfinite(self.overlap_laterals) else None),
                "overlap_fraction": (round(float(self.overlap_fraction), 4) if np.isfinite(self.overlap_fraction) else None),
                "overlap_verdict": (self.quality.get("overlap") or {}).get("verdict"),
                # round 9: the between-scan geometry record
                "lateral_scale": round(float(self.lateral_scale), 5),
                "lateral_offsets": [int(self.lateral_offset_mov), int(self.lateral_offset_ref)],
                "pose_angle_deg": (round(float(self.pose_angle_deg), 2) if np.isfinite(self.pose_angle_deg) else None),
                "raster_rotation_deg": self.quality.get("raster_rotation_deg"),
                "b_lines_median": (round(float(np.nanmedian(self.b_lines)), 2) if self.b_lines is not None and np.isfinite(self.b_lines).any() else None),
                "tilt_residual_median": self.quality.get("tilt_residual_median_px"),
                "coarse_seeds": self.coarse.get("seed_scores"),
                "undecidable_frames": self.quality.get("undecidable_frames"),
                "dead_runs": self.quality.get("dead_runs"),
                "match_structure": self.quality.get("match_structure"), "match_speckle": self.quality.get("match_speckle"),
                "speckle_refined_frames": (int(np.asarray(self.speckle_refined, bool).sum()) if self.speckle_refined is not None else None),
                "speckle_col_median": self.quality.get("speckle_col_median"), "speckle_witness": self.quality.get("speckle_witness"),
                "axial_unwitnessed_runs": self.quality.get("axial_unwitnessed_runs"),
                "superseded_flags": self.quality.get("superseded_flags"),
                # round 10
                "pose_angle_seed_deg": self.quality.get("pose_angle_seed_deg"),
                "tilt_low_precision_frames": len((self.quality.get("tilt_low_precision") or {}).get("frames") or []),
                "tilt_se_median_px": self.quality.get("tilt_se_median_px"),
                "wide_search_frames": len(self.quality.get("wide_search_frames") or []),
                "shear_frames": len(self.quality.get("shear_frames") or {}),
                "flags": list(self.flags), "timings": {k: round(float(v), 2) for k, v in self.timings.items()}}

    def to_dict(self) -> dict:
        """Everything, JSON-serialisable (arrays → lists, NaN kept as float nan)."""
        def conv(v):
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, dict):
                return {kk: conv(vv) for kk, vv in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(vv) for vv in v]
            if isinstance(v, (np.floating, np.integer, np.bool_)):
                return v.item()
            return v
        return {k: conv(v) for k, v in self.__dict__.items()}


def _verdict_counts(arb: dict | None) -> dict | None:
    """{'own': n, 'served': n, 'neither': n, 'run': n, 'neighbour': n, 'previous': n} over quality['arbitrated_frames'] (dx and axial)."""
    if not arb:
        return None
    out: dict = {}
    for rec in arb.values():
        for k in ("dx", "axial"):
            v = (rec.get(k) or {}).get("verdict")
            if v:
                out[v] = out.get(v, 0) + 1
    return out


REJECT_FLAGS = frozenset({"no_correspondence", "dx_at_search_edge", "dx_beyond_max", "tilt_beyond_max", "dx_residual",
                          "axial_residual", "low_frame_match", "dx_untrusted_run",
                          "pose_beyond_frame_rigid"})   # PairResult.ok is False
# the MATCH-LEVEL verdicts a pose verdict SUPERSEDES (E8): a pair whose served lines demand a pose beyond pose_max_deg is
# registered for the record only — it overlaps, so it is never 'no_correspondence'. ROUND 10: the per-frame arbitration
# verdicts (dx_residual, axial_residual, low_frame_match, …) are never superseded — a pose explains a poor global match, not
# an alias run (a periodic-texture alias pair lost its 'dx_residual' to the pose read at its served alias shifts)
_POSE_SUPERSEDES = frozenset({"no_correspondence", "low_relative_match", "unjudged", "low_match", "few_measured_frames",
                              "low_speckle_match", "no_overlap"})


# ── helpers ────────────────────────────────────────────────────────────────────────────────────────────────────
def _as_band(x, band_rows: tuple[int, int], p: PairParams | None = None) -> BandData:
    if isinstance(x, BandData):
        return x
    if isinstance(x, MemberData):
        p = p or PairParams()
        return extract_band(x, band_rows=band_rows, coarse_rows=tuple(p.coarse_rows), coarse_ds=tuple(p.coarse_ds),
                            coarse_mask=str(p.coarse_mask), sigma=tuple(p.feature_sigma_struct),
                            sigma_speckle=(tuple(p.feature_sigma_speckle) if p.speckle_report else None))
    raise TypeError(f"register_pair: expected BandData or MemberData, got {type(x).__name__}")


def _shift2(img: np.ndarray, dx: int, dz: int, fill=0):
    """out[l, k] = img[l − dx, k − dz] (moving + shift = fixed), `fill` outside."""
    L, T = img.shape
    out = np.full_like(img, fill)
    dx, dz = int(dx), int(dz)
    if abs(dx) >= L or abs(dz) >= T:
        return out
    out[max(0, dx):min(L, L + dx), max(0, dz):min(T, T + dz)] = img[max(0, -dx):min(L, L - dx), max(0, -dz):min(T, T - dz)]
    return out


def _interp_nan(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Linear interpolation of a 1-D series with NaN gaps at fractional positions x: NaN outside the finite
    range and wherever either integer neighbour of x is NaN (no bridging of gaps)."""
    y = np.asarray(y, float); x = np.asarray(x, float)
    idx = np.flatnonzero(np.isfinite(y))
    out = np.full(x.shape, np.nan)
    if idx.size < 2:
        return out
    ok = np.isfinite(x) & (x >= idx[0]) & (x <= idx[-1])
    if not ok.any():
        return out
    xo = x[ok]
    lo = np.clip(np.floor(xo).astype(int), 0, y.size - 1); hi = np.clip(lo + 1, 0, y.size - 1)
    v = np.interp(xo, idx, y[idx])
    v[~(np.isfinite(y[lo]) & np.isfinite(y[hi]))] = np.nan
    out[ok] = v
    return out


def robust_line(x: np.ndarray, y: np.ndarray, iters: int = 3, min_n: int = 30, k: float = 4.0):
    """Robust line y ≈ a + b·x (prototype A finelib.robust_line): iterated least squares with 4·MAD rejection.
    Returns (a, b, rms, keep) — NaN when fewer than min_n points survive."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(y) & np.isfinite(x)
    if m.sum() < min_n:
        return np.nan, np.nan, np.nan, m
    co = None
    for _ in range(iters):
        co = np.polyfit(x[m], y[m], 1)
        rr = y[m] - np.polyval(co, x[m])
        sd = 1.4826 * np.median(np.abs(rr - np.median(rr))) + 1e-6
        keep = m.copy(); keep[m] = np.abs(rr) <= k * sd
        if keep.sum() < min_n:
            break
        m = keep
    rr = y[m] - np.polyval(co, x[m])
    return float(co[1]), float(co[0]), float(np.sqrt(np.mean(rr ** 2))), m


def _step_runs(vals: np.ndarray, step: float | None, hold: int = 2) -> list[tuple[int, int]]:
    """[(i0, i1)] runs of a MEASURED series (index order, gaps already dropped) between SUSTAINED steps: a jump of
    ≥ step between consecutive values whose `hold` values on either side agree with each other (spread ≤ step / 2)
    and all lie ≥ step from the other side's median — split_segments' step/hold rule on the axial series (a rigid
    per-frame axial saccade / blink is admissible: round-2 refutation 1, an a-step of 15-25 px measured at NCC
    0.97 was bridged by the SG trend and the eight frames around it rejected and served a ramp 11 px off).
    No step (`step` None / ≤ 0) → one run."""
    n = int(vals.size)
    if n == 0:
        return []
    hold = max(1, int(hold))
    if not step or step <= 0 or n < 2 * hold:
        return [(0, n)]
    cuts = []
    for t in range(hold, n - hold + 1):
        if abs(float(vals[t] - vals[t - 1])) < float(step):
            continue
        pre = vals[t - hold:t]; post = vals[t:t + hold]
        if np.ptp(pre) > 0.5 * float(step) or np.ptp(post) > 0.5 * float(step):
            continue
        mp, mq = float(np.median(pre)), float(np.median(post))
        if min(float(np.min(np.abs(post - mp))), float(np.min(np.abs(pre - mq)))) < float(step):
            continue
        cuts.append(t)
    b = [0] + cuts + [n]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1)]


def _fill_kept(v: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Linear interpolation of v over the kept frames, ends held (zeros when nothing is kept)."""
    v = np.asarray(v, float); kept = np.asarray(kept, bool) & np.isfinite(v)
    if not kept.any():
        return np.zeros(v.size)
    x = np.arange(v.size)
    return np.interp(x, x[kept], v[kept])


def _sg_trend(v: np.ndarray, keep: np.ndarray, window: int = 11, order: int = 2, step: float | None = None,
              hold: int = 2) -> np.ndarray:
    """The step-aware Savitzky-Golay TREND of a per-frame series over the `keep` frames, read on EVERY frame: fitted per
    run of the kept frames between sustained steps (_step_runs), a run shorter than 3 frames its own trend, the gaps
    between runs bridged and the ends held (_robust_fill's trend; round 10 also serves it as the tilt of a frame whose own
    tilt is imprecise — a smoothing of a TRANSFORM parameter across frames, never of any image)."""
    from scipy.signal import savgol_filter
    v = np.asarray(v, float); F = v.size; x = np.arange(F)
    idx = np.flatnonzero(np.asarray(keep, bool) & np.isfinite(v))
    if idx.size == 0:
        return np.zeros(F)
    tr = np.full(F, np.nan)
    for i0, i1 in _step_runs(v[idx], step, hold):
        fi = idx[i0:i1]
        if fi.size >= 3:
            # the run's frames bridged and held over the WHOLE frame axis, smoothed, read inside the run's range
            vi = np.interp(x, fi, v[fi])
            w = min(int(window), fi.size if fi.size % 2 == 1 else fi.size - 1)
            w = max(w, 3)
            o = min(int(order), w - 1)
            sm = savgol_filter(vi, w, o, mode="interp") if w <= F else vi
            tr[fi[0]:fi[-1] + 1] = sm[fi[0]:fi[-1] + 1]
        else:
            tr[fi] = v[fi]
    fin = np.isfinite(tr)
    return np.interp(x, x[fin], tr[fin])                  # gaps between runs bridged, the ends held


def _robust_fill(v: np.ndarray, ok: np.ndarray, window: int = 11, order: int = 2, k: float = 4.0,
                 floor: float = 1.0, iters: int = 3, step: float | None = None, hold: int = 2,
                 cap: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """MAD rejection of a per-frame series against a STEP-AWARE Savitzky-Golay trend, then linear interpolation
    across frames with the ends held — the design's robustify-and-fill rule for a[f] and b[f]. The trend is fitted
    per run of the kept frames between sustained steps (_step_runs: a jump ≥ `step` held over `hold` frames on
    both sides — the axial sibling of the lateral saccade rule; None disables), so a rigid axial step is fitted
    on both sides instead of bridged; a run shorter than 3 frames is its own trend (a run of ≥ 2 consecutive
    measured frames that agree with each other is never rejected merely because the trend cannot follow a
    step). Each iteration rebuilds the runs and the trend from the frames kept so far and re-tests EVERY measured
    frame against it (a neighbour an outlier had dragged off the trend is re-admitted once the outlier is gone;
    the MAD is global over the kept residuals, its threshold never below `floor`). Returns (filled (F,), kept (F,)
    bool)."""
    v = np.asarray(v, float); F = v.size
    ok = np.asarray(ok, bool) & np.isfinite(v)
    keep = ok.copy()

    for _ in range(max(1, int(iters))):
        n = int(keep.sum())
        if n < 3:
            break
        r = v - _sg_trend(v, keep, window, order, step, hold)
        med = float(np.median(r[keep])); mad = 1.4826 * float(np.median(np.abs(r[keep] - med)))
        thr = max(k * mad, float(floor))
        if cap is not None:
            thr = min(thr, float(cap))                          # a majority of aliases must not inflate the threshold
        new = ok & (np.abs(r - med) <= thr)
        if (new == keep).all():
            break
        keep = new
    if not keep.any():
        return np.zeros(F), keep
    return _fill_kept(v, keep), keep


def live_frames(band: BandData, live_frac: float = 0.5) -> np.ndarray:
    """(F,) frames whose valid laterals (band.mask has any row) cover ≥ live_frac of the width; the others are
    DEAD — a reviewer crop band or a zero-filled frame — and split the lateral segments. register_pair ORs this
    with the frames the fine stage measured: a measured frame is live whatever its lateral fraction (the lateral
    fraction only gates frames whose measurement is absent or below the floor)."""
    cols = band.mask.any(axis=1)
    return cols.mean(axis=0) >= float(live_frac)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[(f0, f1)] maximal runs of True in a 1-D bool array (f0 ≤ f < f1)."""
    m = np.asarray(mask, bool)
    if not m.any():
        return []
    d = np.diff(np.r_[0, m.astype(int), 0])
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def split_segments(dx: np.ndarray, live: np.ndarray, step: float = 12.0, hold: int = 5) -> list[tuple[int, int]]:
    """Live segments [f0, f1) of the moving frame axis: maximal runs of `live` frames, each split — recursively,
    largest step first — wherever the measured per-frame lateral shift dx[f] (NaN = unmeasured) STEPS by ≥ `step`
    laterals and the step is HELD over the `hold` frames on both sides of the cut: every measured frame of the
    post-window lies ≥ step from the pre-window's median and vice versa (each window needs ≥ 3 measured frames;
    a 3-frame transient inside a 5-frame window is NOT a saccade — the FIRST projection; register_pair's
    arbitration then carves out every frame whose own measurement proves better than this projection, see
    PairParams). Candidates are ranked by the median jump; cuts within 10 % of `step` of the best tie and go to
    the largest step between neighbouring measured frames. The design's saccade rule: prototype A saw a real
    −35-lateral saccade inside case_cs001_os_v2 at frame 42, which no reviewer band marks (this rule cuts v2 at 42
    and v3 at 41). A side SHORTER than `hold` — the cut lies within hold frames of the run's end, where a saccade
    has no 'return' to distinguish it from a transient — needs ≥ 2 measured frames that agree with each other
    (spread ≤ step / 2) and all lie ≥ step from the other side (round-3 refutation D); the interior rule is
    unchanged."""
    dx = np.asarray(dx, float); live = np.asarray(live, bool); F = dx.size
    hold = max(1, int(hold))
    runs = []
    f = 0
    while f < F:
        if not live[f]:
            f += 1; continue
        g = f
        while g < F and live[g]:
            g += 1
        runs.append((f, g)); f = g
    out: list[tuple[int, int]] = []

    def local_step(kk: int, f0: int, f1: int) -> float:
        """|first measured dx at/after the cut − last measured dx before it| — locates the cut inside a tie."""
        pre = dx[f0:kk]; post = dx[kk:f1]
        pre = pre[np.isfinite(pre)]; post = post[np.isfinite(post)]
        return abs(float(post[0]) - float(pre[-1])) if pre.size and post.size else 0.0

    def rec(f0: int, f1: int) -> None:
        jumps: dict[int, float] = {}
        short_n = min(2, hold)
        st = float(step)
        for kk in range(f0 + 1, f1):
            pre_short = kk - f0 < hold; post_short = f1 - kk < hold      # the side touches the run's end
            pre_i = np.arange(max(f0, kk - hold), kk); post_i = np.arange(kk, min(f1, kk + hold))
            pre_i = pre_i[np.isfinite(dx[pre_i])]; post_i = post_i[np.isfinite(dx[post_i])]
            need_pre = short_n if pre_short else min(3, hold)
            need_post = short_n if post_short else min(3, hold)
            if pre_i.size < need_pre or post_i.size < need_post:
                continue
            pre = dx[pre_i]; post = dx[post_i]
            if (pre_short and np.ptp(pre) > 0.5 * st) or (post_short and np.ptp(post) > 0.5 * st):
                continue                                       # a short side must agree with itself
            mp, mq = float(np.median(pre)), float(np.median(post))
            sustained = min(float(np.min(np.abs(post - mp))), float(np.min(np.abs(pre - mq))))
            if sustained < st:                             # not held over the whole window on both sides
                continue
            jumps[kk] = abs(mq - mp)
        best_j = max(jumps.values()) if jumps else 0.0
        if jumps:
            # every cut within 10 % of the step of the best median jump ties (the hold windows straddle the same
            # step); the tie goes to the cut with the largest step between its neighbouring measured frames
            ties = [kk for kk, j in jumps.items() if j >= best_j - 0.1 * float(step)]
            best_k = max(ties, key=lambda kk: (local_step(kk, f0, f1), -kk))
            rec(f0, best_k); rec(best_k, f1)
        else:
            out.append((int(f0), int(f1)))
    for f0, f1 in runs:
        rec(f0, f1)
    return out


def _agreeing_runs(vals: np.ndarray, agree: float) -> list[tuple[int, int]]:
    """[(r0, r1)] maximal runs of consecutive FINITE values whose SPREAD (max − min) stays within `agree` — every
    frame of a run lies within `agree` of the run's median, so a run served its median is never more than that
    off any of its frames (a slow ramp becomes several short runs, never one run whose ends sit half its span
    off the median)."""
    v = np.asarray(vals, float); n = int(v.size)
    out: list[tuple[int, int]] = []
    f = 0
    while f < n:
        if not np.isfinite(v[f]):
            f += 1; continue
        lo = hi = float(v[f]); g = f + 1
        while g < n and np.isfinite(v[g]) and max(hi, float(v[g])) - min(lo, float(v[g])) <= float(agree):
            lo = min(lo, float(v[g])); hi = max(hi, float(v[g])); g += 1
        out.append((int(f), int(g))); f = g
    return out


def _coherent_runs(vals: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """[(r0, r1)] maximal runs of consecutive FINITE values whose CONSECUTIVE differences stay within `tol` — a ramp of any
    total span is ONE coherent run (round 10, R6: a real lateral wave of ±25 laterals over 40-60 frames moves 2.5-4 laterals
    per frame; _agreeing_runs' spread rule split it into 1-2-frame runs, so the mode guard dropped every frame of its steep
    parts from the knots and the fill served them a straight line up to 5 laterals off their own decisive measurement)."""
    v = np.asarray(vals, float); n = int(v.size)
    out: list[tuple[int, int]] = []
    f = 0
    while f < n:
        if not np.isfinite(v[f]):
            f += 1; continue
        g = f + 1
        while g < n and np.isfinite(v[g]) and abs(float(v[g]) - float(v[g - 1])) <= float(tol):
            g += 1
        out.append((int(f), int(g))); f = g
    return out


def _carve_segments(base: list, forced: np.ndarray, agree: float, live: np.ndarray) -> tuple[list, list]:
    """The base segmentation re-cut around every agreeing run of FORCED frames — frames served a pinned lateral
    shift (their own measurement that won the arbitration, or a neighbouring segment's shift they were reassigned
    to): each run becomes its own segment, base cuts strictly inside a run are dropped. Returns (segments, carved)
    with carved = [(r0, r1, at_end)]; at_end says the run touches a boundary of the base segmentation (a live-run
    end or a cut) — the place a single contradicting frame may legitimately sit."""
    bounds = {int(b) for seg in base for b in seg}
    live = np.asarray(live, bool)
    runs = _agreeing_runs(np.where(live, forced, np.nan), agree)
    cuts = set(bounds); carved: list = []
    for r0, r1 in runs:
        carved.append((int(r0), int(r1), bool(r0 in bounds or r1 in bounds)))
        for c in [c for c in cuts if r0 < c < r1]:
            cuts.discard(c)
        cuts.add(int(r0)); cuts.add(int(r1))
    segs: list = []
    for f0, f1 in _runs(live):
        cs = sorted({f0, f1} | {c for c in cuts if f0 < c < f1})
        segs += [(int(cs[i]), int(cs[i + 1])) for i in range(len(cs) - 1)]
    return segs, carved


def _drop_cuts(segs: list, drop: set) -> list:
    """Merge consecutive segments across the cuts in `drop`."""
    out: list = []
    for f0, f1 in segs:
        if out and out[-1][1] == f0 and f0 in drop:
            out[-1] = (out[-1][0], f1)
        else:
            out.append((f0, f1))
    return out


def _add_cuts(segments: list, cuts) -> list:
    """`segments` [(f0, f1)] with every cut of `cuts` that lies strictly inside a segment applied."""
    out: list = []
    for f0, f1 in segments:
        cs = sorted({int(f0), int(f1)} | {int(c) for c in cuts if f0 < c < f1})
        out += [(cs[i], cs[i + 1]) for i in range(len(cs) - 1)]
    return out


def _plateau_cuts(dx: np.ndarray, segments: list, step: float, hold: int, spread: float, min_n: int = 3) -> list[int]:
    """Candidate SUB-STEP cuts inside `segments`: a step of at least `step` in the measured dx (NaN = not a voter) held
    over `hold` frames on both sides — every measured frame of each hold window lies ≥ step from the other window's
    median, split_segments' sustained rule — where BOTH windows are PLATEAUS (spread ≤ `spread`, ≥ min_n measured
    frames). A ramp never qualifies (its windows are not plateaus, and consecutive windows do not separate); a
    bimodal segment does. Adjacent candidates straddling the same step tie like split_segments' cuts: the largest step
    between neighbouring measured frames wins. register_pair accepts a candidate only when the cluster medians score
    better than the served constant on the window frames (the evidence gate)."""
    dx = np.asarray(dx, float); hold = max(1, int(hold))
    cand: list[tuple[int, float]] = []
    for f0, f1 in segments:
        for kk in range(int(f0) + hold, int(f1) - hold + 1):
            pre = dx[kk - hold:kk]; post = dx[kk:kk + hold]
            pre = pre[np.isfinite(pre)]; post = post[np.isfinite(post)]
            if pre.size < min_n or post.size < min_n:
                continue
            if np.ptp(pre) > float(spread) or np.ptp(post) > float(spread):
                continue
            mp, mq = float(np.median(pre)), float(np.median(post))
            if min(float(np.min(np.abs(post - mp))), float(np.min(np.abs(pre - mq)))) < float(step):
                continue
            cand.append((int(kk), abs(float(post[0]) - float(pre[-1]))))
    out: list[int] = []
    i = 0
    while i < len(cand):                                        # consecutive candidates: the largest local step wins
        j = i
        while j + 1 < len(cand) and cand[j + 1][0] == cand[j][0] + 1:
            j += 1
        out.append(max(cand[i:j + 1], key=lambda t: (t[1], -t[0]))[0])
        i = j + 1
    return out


def _segment_dx(dx: np.ndarray, segments: list[tuple[int, int]], max_dx: float, fallback: float
                ) -> tuple[list, np.ndarray]:
    """[(f0, f1, dx)] with the segment MEDIAN of the measured dx — NEVER clamped: dx_applied equals the measured
    segment median exactly (register_pair refuses a measured segment beyond max_dx with 'dx_beyond_max'); only a
    segment with nothing measured takes the nearest measured segment's value, else `fallback` = the coarse dx0
    (clipped to ±max_dx — the one place the cap is applied, to a seed, not a measurement) — and the (F,)
    per-frame applied dx (dead frames hold the nearest segment's value)."""
    F = dx.size
    vals = []
    for f0, f1 in segments:
        v = dx[f0:f1]; v = v[np.isfinite(v)]
        vals.append(float(np.median(v)) if v.size else np.nan)
    vals = np.array(vals, float)
    if np.isfinite(vals).any():
        cen = np.array([(f0 + f1 - 1) / 2.0 for f0, f1 in segments])
        okv = np.isfinite(vals)
        vals = np.interp(cen, cen[okv], vals[okv]) if okv.sum() >= 1 else vals
    else:
        vals = np.full(len(segments), float(np.clip(fallback, -float(max_dx), float(max_dx))))
    applied = np.full(F, np.nan)
    for (f0, f1), v in zip(segments, vals):
        applied[f0:f1] = v
    if np.isfinite(applied).any():
        idx = np.flatnonzero(np.isfinite(applied))
        applied = np.interp(np.arange(F), idx, applied[idx])
    else:
        applied = np.full(F, float(np.clip(fallback, -max_dx, max_dx)))
    return [(int(f0), int(f1), float(v)) for (f0, f1), v in zip(segments, vals)], applied


# ── the arbitration's judge: one frame under one candidate (dx, a, b), memoised ────────────────────────────────
class _FrameScorer:
    """The band-space match of ONE moving frame under a candidate (dx, a, b) — the per-frame gate ratio (matched
    fraction > 0.5 after the warp / the reference's own adjacent-frame ceiling on that frame) — memoised per
    (frame, dx, a, b). warp_band resamples a reference frame from its moving partner with that partner's own
    df / dx_applied / a / b only, so a frame's score under a candidate value does not depend on what the rest of
    the transform serves: every candidate is scored once (pair_quality on just the frames that need it, batched)
    and the pair-level numbers of the final transform are assembled from the same per-frame records (_assemble).
    The per-frame ceiling — the mean of the reference's adjacent-frame pairs (f−1, f) and (f, f+1) — is completed
    lazily for the frames scored (a given ceiling with 'per_frame_frac_0.5', e.g. register_group's, is used as is)."""

    def __init__(self, ref: BandData, mov: BandData, template: PairResult, ceiling: dict | None, window,
                 feature: str = "struct") -> None:
        self.ref, self.mov, self.template, self.window = ref, mov, template, tuple(window)
        self.feature = str(feature)
        self.F = int(ref.n_frames); self.df = int(template.df)
        c = dict(ceiling or {})
        pf = c.get("per_frame_frac_0.5")
        pf = np.asarray(pf, float) if pf is not None else np.full(self.F, np.nan)
        if pf.shape != (self.F,):
            pf = np.full(self.F, np.nan)
        self.c_frac = pf.copy(); self.c_n = np.full(self.F, np.nan); self.c_known = np.isfinite(pf)
        self.c_agg = {k: float(v) for k, v in c.items() if not isinstance(v, np.ndarray)}
        self.cache: dict = {}
        self.n_ref_frame = ref.match_mask.sum(axis=(0, 1)).astype(float)
        self.n_calls = 0; self.n_scored = 0; self.time_s = 0.0

    @staticmethod
    def key(f: int, dx: float, a: float, b: float) -> tuple:
        return (int(f), round(float(dx), 4), round(float(a), 4), round(float(b), 4))

    def ceiling_dict(self) -> dict:
        d = dict(self.c_agg); d["per_frame_frac_0.5"] = self.c_frac
        return d

    def ensure_ceiling(self, frames_ref) -> None:
        F = self.F
        need = sorted({i for fr in frames_ref for i in (int(fr) - 1, int(fr)) if 0 <= i < F - 1 and not self.c_known[i]})
        if not need:
            return
        cs = band_similarity(self.ref, self.ref, window=self.window, frame_offset=1, frames=need, feature=self.feature)
        st = cs.stats
        self.c_frac[cs.frames_a] = np.asarray(st["per_frame_frac_0.5"], float)
        self.c_n[cs.frames_a] = np.asarray(st["per_frame_n_eval"], float)
        self.c_known[cs.frames_a] = True

    def ceiling_over(self, frames_ref) -> float:
        """matched_frac_0.5 of the ceiling over the pairs (f, f+1), f in frames_ref (cell-weighted; NaN if none)."""
        idx = sorted({int(fr) for fr in frames_ref if 0 <= int(fr) < self.F - 1})
        n = np.array([self.c_n[i] for i in idx], float); v = np.array([self.c_frac[i] for i in idx], float)
        ok = np.isfinite(n) & np.isfinite(v) & (n > 0)
        if not ok.any():
            return float(self.c_agg.get("matched_frac_0.5", float("nan")))
        return float(np.sum(v[ok] * n[ok]) / np.sum(n[ok]))

    def absorb(self, q: dict, frames_mov, dx: np.ndarray, a: np.ndarray, b: np.ndarray) -> None:
        """Record the per-frame numbers of a pair_quality result for `frames_mov` scored under (dx, a, b)."""
        for f in frames_mov:
            f = int(f); fr = f + self.df
            if not (0 <= fr < self.F):
                continue
            rec = {"frac_0.5": float(q["frame_frac_0.5"][fr]), "frac_0.3": float(q["frame_frac_0.3"][fr]),
                   "frac_0.7": float(q["frame_frac_0.7"][fr]), "n_eval": float(q["frame_n_eval"][fr]),
                   "ncc_mean": float(q["frame_ncc_mean"][fr]), "resid_ss": float(q["frame_resid_ss"][fr]),
                   "resid_n": float(q["frame_resid_n"][fr]), "ceiling": float(q["frame_ceiling_0.5"][fr])}
            ff, fc = rec["frac_0.5"], rec["ceiling"]
            rec["ratio"] = float(ff / fc) if np.isfinite(ff) and np.isfinite(fc) and fc > 0 else float("nan")
            self.cache[self.key(f, dx[f], a[f], b[f])] = rec

    def score(self, frames_mov, dx: np.ndarray, a: np.ndarray, b: np.ndarray) -> dict:
        """{f: record} for every moving frame in frames_mov under (dx[f], a[f], b[f]) — cached, else one batched
        pair_quality over the missing frames."""
        frames = [int(f) for f in frames_mov]
        # an UNPARTNERED frame (no reference frame at f + df) has no record: a NaN record (round 9: a harness scoring frames
        # outside the overlap crashed pair_quality's frame index)
        nan_rec = {"frac_0.5": float("nan"), "frac_0.3": float("nan"), "frac_0.7": float("nan"), "n_eval": 0.0,
                   "ncc_mean": float("nan"), "resid_ss": 0.0, "resid_n": 0.0, "ceiling": float("nan"), "ratio": float("nan")}
        for f in frames:
            if not (0 <= f + self.df < self.F) and self.key(f, dx[f], a[f], b[f]) not in self.cache:
                self.cache[self.key(f, dx[f], a[f], b[f])] = dict(nan_rec)
        miss = [f for f in frames if self.key(f, dx[f], a[f], b[f]) not in self.cache]
        if miss:
            t0 = time.time()
            fr_list = [f + self.df for f in miss]
            self.ensure_ceiling(fr_list)
            alt = _dc_replace(self.template, dx_applied=np.asarray(dx, float), a=np.asarray(a, float), b=np.asarray(b, float))
            q, _ = pair_quality(self.ref, self.mov, alt, ceiling=self.ceiling_dict(), window=self.window, frames=fr_list,
                                feature=self.feature)
            self.absorb(q, miss, dx, a, b)
            self.n_calls += 1; self.n_scored += len(miss); self.time_s += time.time() - t0
        return {f: self.cache[self.key(f, dx[f], a[f], b[f])] for f in frames}


def _assemble(recs, n_ref: float) -> dict:
    """Pair-level match numbers of a frame set from its per-frame records (exact: the matched fractions and the
    mean NCC are cell-weighted sums; coverage = evaluated cells / n_ref; the surface residual its rms)."""
    recs = list(recs)
    n = float(sum(r["n_eval"] for r in recs))
    out = {"n_eval": int(round(n)), "coverage": n / max(1.0, float(n_ref))}
    for k, src in (("matched_frac_0.3", "frac_0.3"), ("matched_frac_0.5", "frac_0.5"), ("matched_frac_0.7", "frac_0.7"),
                   ("ncc_mean", "ncc_mean")):
        out[k] = (float(sum(r[src] * r["n_eval"] for r in recs if r["n_eval"] > 0 and np.isfinite(r[src]))) / n
                  if n > 0 else float("nan"))
    ss = float(sum(r["resid_ss"] for r in recs)); nr = float(sum(r["resid_n"] for r in recs))
    out["surface_residual_rms_px"] = float(np.sqrt(ss / nr)) if nr > 0 else float("nan")
    return out


_DECIDING_STATS = ("ratio", "frac_0.7", "ncc_mean")


def _better(cand: dict | None, base: dict | None, margin: float) -> tuple[int, str]:
    """The arbitration's COMPARISON of two per-frame records (round 7, refutation R1): lexicographic over ratio (matched
    fraction > 0.5 / the frame's ceiling — the gate statistic), then frac_0.7, then ncc_mean, `margin` applied to the
    DECIDING statistic — the first on which the two differ by more than the margin. Returns (+1 cand wins / −1 base wins /
    0 tie on all three, the deciding statistic or 'tie'). The gate ratio SATURATES (frac_0.5 = 1.0 on a periodic texture
    for an alias and for the truth alike); the finer statistics do not. A finite value beats a missing one."""
    for k in _DECIDING_STATS:
        cv = float(cand.get(k, np.nan)) if cand is not None else float("nan")
        bv = float(base.get(k, np.nan)) if base is not None else float("nan")
        fc, fb = bool(np.isfinite(cv)), bool(np.isfinite(bv))
        if fc and not fb:
            return 1, k
        if fb and not fc:
            return -1, k
        if not fc and not fb:
            continue
        if cv - bv > float(margin):
            return 1, k
        if bv - cv > float(margin):
            return -1, k
    return 0, "tie"


# ── (a) coarse ─────────────────────────────────────────────────────────────────────────────────────────────────
def _overlap_floor_cells(mask: np.ndarray, ds, bar_laterals: int, bar_frame_frac: float) -> int:
    """PARTIAL OVERLAP (2026-09-12): the ABSOLUTE overlap floor of the coarse search in PYRAMID CELLS — the cells an overlap
    of bar_laterals × bar_frame_frac of the live frames holds at this mask's density (mask cells per live (lateral cell,
    frame) column), HALVED because both masks must hold a cell (the tissue depth differs between scans). The floor keeps
    tiny-overlap aliases out of the search (E6's reason for the 30 % rule); the geometric bar (overlap_of) decides."""
    n = int(mask.sum())
    if n == 0:
        return 1
    live_f = int(mask.any(axis=(0, 1)).sum()); live_l = int(mask.any(axis=(1, 2)).sum())
    density = n / float(max(1, live_l * live_f))
    k_lat = max(1, int(np.ceil(float(bar_laterals) / float(ds[0]))))
    k_fr = max(1, int(np.ceil(float(bar_frame_frac) * live_f)))
    return max(1, int(0.5 * k_lat * k_fr * density))


def overlap_of(L: int, Fm: int, F: int, dx: float, df: int, bar_laterals: int, bar_frame_frac: float) -> dict:
    """PARTIAL OVERLAP (2026-09-12): the overlap an offset (dx laterals, df frames) implies between a moving scan of Fm frames
    and a reference of F frames on a common grid of L laterals: laterals = L − |dx| (clipped to [0, L]), fraction = of the
    moving scan's L, frames = the moving frames with a reference partner (f + df inside the reference), frame_fraction = of
    Fm; admissible = both bars met."""
    dxf = float(dx) if np.isfinite(dx) else float("nan")
    lat = float(np.clip(float(L) - abs(dxf), 0.0, float(L))) if np.isfinite(dxf) else float("nan")
    dfi = int(round(float(df))) if np.isfinite(df) else 0
    frames = int(max(0, min(int(Fm), int(F) - dfi) - max(0, -dfi)))
    frac = lat / max(1.0, float(L)) if np.isfinite(lat) else float("nan")
    ffrac = frames / max(1.0, float(Fm))
    adm = bool(np.isfinite(lat) and lat >= float(bar_laterals) and ffrac >= float(bar_frame_frac))
    return {"laterals": lat, "fraction": frac, "frames": frames, "frame_fraction": float(ffrac), "admissible": adm,
            "dx": dxf, "df": dfi}


def _coarse_peak(ref: BandData, mov_coarse: np.ndarray, mov_mask: np.ndarray, ms, ds, min_overlap_cells: int = 1
                 ) -> tuple[np.ndarray, np.ndarray, dict]:
    """One masked FFT-NCC search of a moving pyramid (+ mask) against the reference pyramid: (ncc, N, peak dict
    with dx0 / dz0 / df0 in full-resolution units, the integer peak index, its NCC, overlap, on_bound). A shift
    counts only when it overlaps ≥ min_overlap_cells (PARTIAL OVERLAP 2026-09-12: the absolute floor of
    _overlap_floor_cells by default — E6's 30 % of the moving cells, which a tiny-overlap df-bound alias needed on
    CS032 v1_4→v1_3 at 0.20, hid the true peak of a 23 % overlap on CS001_OD; the legacy fraction stays available
    through PairParams.coarse_min_overlap)."""
    n_mov = int(mov_mask.sum())
    ncc, N = masked_ncc_fft(ref.coarse, ref.coarse_mask, mov_coarse, mov_mask, ms,
                            min_overlap=max(1, int(min_overlap_cells)))
    pk = tuple(int(v) for v in np.unravel_index(int(np.argmax(ncc)), ncc.shape))
    peak = float(ncc[pk])
    sp = subpix_peak(ncc, pk)
    d = {"dx0": float((sp[0] - ms[0]) * ds[0]), "dz0": float((sp[1] - ms[1]) * ds[1]), "df0": int(pk[2] - ms[2]),
         "ncc": peak, "pk": pk, "overlap": float(N[pk]), "n_mov_cells": n_mov,
         "on_bound": bool(any(i == 0 or i == n - 1 for i, n in zip(pk, ncc.shape)))}
    return ncc, N, d


def _coarse_maxima(ncc: np.ndarray, ms, ds, k: int, tol: float, excl: int) -> list[dict]:
    """E6 top-K seeding: the SEPARATED local maxima of the coarse NCC volume (a 3×3×3 neighbourhood; ≥ `excl` cells
    apart in some axis — Chebyshev distance) within `tol` of the peak, best first, at most `k` of them; a maximum ON a
    search bound is dropped when an off-bound one exists (it is a bound, not a measurement). Each: dx0 / dz0 / df0
    (sub-cell in dx / dz, integer df), ncc, on_bound, the integer index. The coarse surface of a between-scan pair
    is a PLATEAU (P5 / CS032: 2-8 maxima within 0.01-0.03), so every candidate is tried by the fine stage."""
    peak = float(ncc.max())
    if not np.isfinite(peak) or peak <= -1.0:
        return []
    mx = ndi.maximum_filter(ncc, size=3, mode="nearest")
    cand = np.argwhere((ncc == mx) & (ncc >= peak - float(tol)) & (ncc > -1.0))
    if cand.size == 0:
        return []
    vals = ncc[tuple(cand.T)]
    order = np.argsort(-vals, kind="stable")
    keep: list = []
    for i in order:
        c = cand[i]
        if all(int(np.max(np.abs(c - kk))) >= int(excl) for kk in keep):
            keep.append(c)
        if len(keep) >= 4 * max(1, int(k)):
            break
    out = []
    for c in keep:
        pk = tuple(int(v) for v in c)
        sp = subpix_peak(ncc, pk)
        out.append({"dx0": float((sp[0] - ms[0]) * ds[0]), "dz0": float((sp[1] - ms[1]) * ds[1]), "df0": int(pk[2] - ms[2]),
                    "ncc": float(ncc[pk]), "pk": pk, "on_bound": bool(any(i == 0 or i == n - 1 for i, n in zip(pk, ncc.shape)))})
    off = [s for s in out if not s["on_bound"]]
    if off:
        out = off
    return out[:max(1, int(k))]


def _count_maxima(ncc: np.ndarray, tol: float, excl: int) -> int:
    """The number of SEPARATED local maxima (3×3×3 neighbourhood, ≥ excl cells apart in some axis) within `tol` of
    the coarse peak — on-bound ones included (the plateau statistic reported as 'n_maxima_within_0.03')."""
    peak = float(ncc.max())
    if not np.isfinite(peak) or peak <= -1.0:
        return 0
    mx = ndi.maximum_filter(ncc, size=3, mode="nearest")
    cand = np.argwhere((ncc == mx) & (ncc >= peak - float(tol)) & (ncc > -1.0))
    if cand.size == 0:
        return 0
    vals = ncc[tuple(cand.T)]
    keep: list = []
    for i in np.argsort(-vals, kind="stable"):
        c = cand[i]
        if all(int(np.max(np.abs(c - kk))) >= int(excl) for kk in keep):
            keep.append(c)
        if len(keep) >= 64:
            break
    return len(keep)


def _b_lines(ref_served: np.ndarray, mov_served: np.ndarray, mov_valid: np.ndarray | None, dx: float, xc: np.ndarray,
             min_n: int, k: float = 4.0) -> tuple[float, float]:
    """(a_lines, b_lines): the intercept / half-span slope (px) of the robust line through S_ref(l + dx) − S_mov(l)
    over the valid overlap — what the two SERVED LINES imply for the axial move at lateral shift dx (round 9's tilt
    prior: the fine stage's tilt follows it to 2-13 px; the reference's own dome at offset dx does not)."""
    lat = np.arange(mov_served.size, dtype=float)
    d = _interp_nan(np.asarray(ref_served, float), lat + float(dx)) - np.asarray(mov_served, float)
    if mov_valid is not None:
        d = np.where(np.asarray(mov_valid, bool), d, np.nan)
    a, b, _, _ = robust_line(xc, d, min_n=min_n, k=k)
    return float(a), float(b)


def pose_angle_deg(b_half_span_px: float, spacing, n_lateral: int) -> float:
    """E8: the pose angle (degrees) a tilt of b px half-span implies between two B-scan planes: atan(b · sp_depth /
    (half-span laterals · sp_lateral)). P5_OS v1 against its siblings: 168-192 px at 7.797 µm → ≈ 15°; CS032 pairs
    ≤ 83 px at 11.5 µm → ≤ 5°."""
    hs = max(1.0, (int(n_lateral) - 1) / 2.0)
    sp = np.asarray(spacing, float)
    if not np.isfinite(b_half_span_px) or sp.size < 2 or sp[0] <= 0:
        return float("nan")
    return float(np.degrees(np.arctan(abs(float(b_half_span_px)) * float(sp[1]) / (hs * float(sp[0])))))


def coarse_register(ref: BandData, mov: BandData, params: PairParams | dict | None = None) -> dict:
    """Design element 3(a): the global 3-D shift (dx0 laterals, dz0 band rows, df0 frames) of the moving band
    onto the reference by Padfield masked FFT-NCC (masked_ncc_fft) on the coarse pyramids (BandData.coarse: the
    240-row band, block means ×4 lateral × ×4 depth, tissue mask without a posterior cap — prototype A's coarse
    configuration) over dx ±coarse_max_dx (PARTIAL OVERLAP 2026-09-12: None → ±(L − min_overlap_laterals), every offset
    that keeps the overlap bar's laterals), dz ±coarse_max_dz, df ±coarse_max_df, with the peak refined sub-cell
    (subpix_peak) in dx and dz and kept INTEGER in df — a peak on a search bound widens every axis ×coarse_widen
    and searches again ('widened_axes'; 'max_shift_full' is the search actually used, 'max_shift_initial' the
    configured one). Also the peak's SHARPNESS = peak − the best value ≥
    coarse_excl cells/frames away in some axis, whether the peak sits on a search bound, the df profile (max over
    dx, dz per df) and its number of local maxima within 0.1 of the peak (n_df_peaks; 1 = unimodal).

    The SEED handed to the fine stage (dx0 / dz0 / df0 of the returned dict): the full-volume peak when it is
    USABLE (NCC ≥ coarse_attempt_ncc and not on a search bound; seed_source 'full'); otherwise the two halves of
    the moving frame range are searched separately (the moving pyramid masked to frames [0, F/2) and [F/2, F))
    and the better half's peak seeds the fine stage (seed_source 'half_0' | 'half_1', the two searches under
    'split') — a mid-volume saccade splits the lateral mode in two, and one half is then unimodal. 'ncc' is
    ALWAYS the full-volume peak (the reported coarse NCC), 'seed_ncc' the seed's. Prototype A (×4×4 on 240 rows,
    CS001_OS): v2→v1 (dx −37.8, dz +0.7, df −9), v3→v1 (−31.8, +0.9, −7), peak NCC 0.71-0.81, unimodal df
    profiles; this module on the copies: 0.780 / 0.786, sharpness 0.037 / 0.035. The overlap FLOOR of the search
    (masked_ncc_fft's min_overlap) is the absolute bar's cells at the moving mask's density (_overlap_floor_cells;
    'min_overlap_cells' / 'min_overlap_rule'), or the legacy fraction of the moving cells when coarse_min_overlap is
    set; the GEOMETRIC bar (overlap_of) is applied to the seeds — 'overlap' (the seed's), 'no_overlap' (no admissible
    seed), 'beyond_bar_peak' (a primary peak the bar rejected, with its offset), 'overlap_bar'."""
    L_full = int(ref.band.shape[0])
    p = _pair_params(params).resolved(L_full)
    t0 = time.time()
    ds = tuple(int(v) for v in ref.coarse_ds)
    if ref.coarse.shape[:2] != mov.coarse.shape[:2]:
        raise ValueError(f"coarse_register: pyramid shapes differ {ref.coarse.shape} vs {mov.coarse.shape}")
    Lc, Tc, Fc = mov.coarse.shape
    F_ref = int(ref.n_frames); Fm_full = int(mov.n_frames)
    bar_l = int(p.min_overlap_laterals); bar_f = float(p.min_overlap_frame_frac)
    legacy_floor = p.coarse_min_overlap is not None

    def floor_of(mask: np.ndarray) -> int:
        """The FFT overlap floor in cells for this moving mask: the absolute bar (default) or the legacy fraction."""
        if legacy_floor:
            return max(1, int(float(p.coarse_min_overlap) * int(mask.sum())))
        return _overlap_floor_cells(mask, ds, bar_l, bar_f)
    ovl = floor_of(mov.coarse_mask)
    ms = (max(1, min(int(p.coarse_max_dx) // ds[0], Lc - 1)), max(1, min(int(p.coarse_max_dz) // ds[1], Tc - 1)),
          max(1, min(int(p.coarse_max_df), max(1, Fc // 2))))
    ms_initial = ms
    ncc, N, full = _coarse_peak(ref, mov.coarse, mov.coarse_mask, ms, ds, ovl)
    # ADAPTIVE WINDOW (oct_preprocess tissue_motion_widen_*): a peak ON a search bound is a bound, not a
    # measurement — widen EVERY axis ×coarse_widen (capped by the pyramid) and search again; the widened search
    # space is a superset, so its peak is at least as good. All axes, not just the pinned one: a true shift
    # beyond the window can leave a garbage peak pinned on ANOTHER axis (round-2 refutation, true dx 75: the
    # ±60 full peak sat on the df bound at dx 51; with dx widened too the peak is found at 75). Refutation A:
    # true dx 62-75 pinned at 60 (then the halves pinned too); widened to ±120 the peak is found at 62-75.
    widened_axes: list[str] = []
    if full["on_bound"] and float(p.coarse_widen) > 1.0 and full["n_mov_cells"] > 0:
        caps = (Lc - 1, Tc - 1, max(1, Fc // 2))
        ms_w = list(ms)
        for ax, name in enumerate(("dx", "dz", "df")):
            if ms[ax] < caps[ax]:
                ms_w[ax] = max(1, min(int(round(ms[ax] * float(p.coarse_widen))), caps[ax]))
                widened_axes.append(name)
        if widened_axes:
            ms = tuple(ms_w)
            ncc, N, full = _coarse_peak(ref, mov.coarse, mov.coarse_mask, ms, ds, ovl)
    n_mov = full["n_mov_cells"]
    pk = full["pk"]; peak = full["ncc"]
    dx0, dz0, df0, on_bound = full["dx0"], full["dz0"], full["df0"], full["on_bound"]
    seed_ncc, seed_source, split = peak, "full", None
    usable = np.isfinite(peak) and peak >= float(p.coarse_attempt_ncc) and not on_bound and n_mov > 0
    if not usable and Fc >= 4:
        half = Fc // 2
        split = []
        for j, (f0, f1) in enumerate(((0, half), (half, Fc))):
            mm = mov.coarse_mask.copy(); mm[:, :, :f0] = False; mm[:, :, f1:] = False
            if not mm.any():
                continue
            _, _, h = _coarse_peak(ref, mov.coarse, mm, ms, ds, floor_of(mm))
            h = {k: v for k, v in h.items() if k != "pk"}
            h.update(frames=(int(f0), int(f1)), source=f"half_{j}")
            split.append(h)
        if split:
            best = max(split, key=lambda h: (np.isfinite(h["ncc"]) and not h["on_bound"], h["ncc"]))
            dx0, dz0, df0 = best["dx0"], best["dz0"], best["df0"]
            seed_ncc, seed_source = float(best["ncc"]), str(best["source"])
    # E6: the top-K separated maxima within coarse_seed_tol of the peak — every one is a candidate seed for the fine
    # stage (register_pair scores each by a cheap fine pass and keeps the best); the primary seed leads the list
    seeds = _coarse_maxima(ncc, ms, ds, int(p.coarse_seed_k), float(p.coarse_seed_tol), int(p.coarse_excl)) if n_mov > 0 else []
    prim = {"dx0": float(dx0), "dz0": float(dz0), "df0": int(df0), "ncc": float(seed_ncc), "on_bound": bool(on_bound and seed_source == "full"),
            "source": seed_source}
    seeds = [prim] + [dict(s, source="maximum") for s in seeds
                      if not (abs(s["dx0"] - prim["dx0"]) < ds[0] and abs(s["dz0"] - prim["dz0"]) < ds[1] and s["df0"] == prim["df0"])]
    seeds = seeds[:max(1, int(p.coarse_seed_k))]
    # PARTIAL OVERLAP (2026-09-12): the GEOMETRIC bar on the seeds — a seed whose offset leaves fewer than min_overlap_laterals
    # laterals or min_overlap_frame_frac of the moving frames is NOT admissible (the widened search can reach such offsets:
    # they are measured and kept in the record, never seeded); the best admissible separated maximum leads instead, and a
    # pair with no admissible seed at all is 'no_overlap' (register_pair refuses it with the offset in the record)
    beyond_bar = None
    prim_ov = overlap_of(L_full, Fm_full, F_ref, prim["dx0"], prim["df0"], bar_l, bar_f)
    if not prim_ov["admissible"]:
        beyond_bar = {**{k: prim[k] for k in ("dx0", "dz0", "df0", "ncc", "source")}, "overlap": prim_ov}
        adm_seeds = [s_ for s_ in seeds[1:] if overlap_of(L_full, Fm_full, F_ref, s_["dx0"], s_["df0"], bar_l, bar_f)["admissible"]]
        if adm_seeds:
            best_adm = adm_seeds[0]                        # the separated maxima come best first
            dx0, dz0, df0 = float(best_adm["dx0"]), float(best_adm["dz0"]), int(best_adm["df0"])
            seed_ncc, seed_source = float(best_adm["ncc"]), "admissible_maximum"
            prim = dict(best_adm, source=seed_source)
            seeds = [prim] + [s_ for s_ in adm_seeds if s_ is not best_adm]
            prim_ov = overlap_of(L_full, Fm_full, F_ref, prim["dx0"], prim["df0"], bar_l, bar_f)
        else:
            # no admissible separated maximum near the peak: the best ADMISSIBLE cell of the search volume seeds the fine stage
            # instead (a beyond-bar peak can be a small-overlap alias — CS001_OD v4 → v2 peaked at −449 laterals after the
            # widening while its siblings register at −387 / −393; the fine stage judges the admissible candidate, the
            # beyond-bar peak stays in the record) — only a volume with no usable admissible cell at all is 'no_overlap'
            dx_ax = (np.arange(-ms[0], ms[0] + 1) * ds[0]).astype(float)
            df_ax = np.arange(-ms[2], ms[2] + 1)
            ok_dx = np.abs(dx_ax) <= float(L_full - bar_l)
            ok_df = np.array([overlap_of(L_full, Fm_full, F_ref, 0.0, int(d_), bar_l, bar_f)["admissible"] for d_ in df_ax], bool)
            sub = np.where(ok_dx[:, None, None] & ok_df[None, None, :], ncc, -1.0)
            v_adm = float(sub.max()) if sub.size else -1.0
            if np.isfinite(v_adm) and v_adm > -1.0 and v_adm >= float(p.coarse_attempt_ncc):
                pk2 = tuple(int(v) for v in np.unravel_index(int(np.argmax(sub)), sub.shape))
                sp2 = subpix_peak(ncc, pk2)
                dx0 = float(np.clip((sp2[0] - ms[0]) * ds[0], -float(L_full - bar_l), float(L_full - bar_l)))
                dz0, df0 = float((sp2[1] - ms[1]) * ds[1]), int(pk2[2] - ms[2])
                seed_ncc, seed_source = float(ncc[pk2]), "admissible_peak"
                prim = {"dx0": dx0, "dz0": dz0, "df0": df0, "ncc": seed_ncc, "source": seed_source,
                        "on_bound": bool(any(i == 0 or i == n - 1 for i, n in zip(pk2, ncc.shape)))}
                seeds = [prim]
                prim_ov = overlap_of(L_full, Fm_full, F_ref, prim["dx0"], prim["df0"], bar_l, bar_f)
            else:
                seeds = [prim]
    else:
        seeds = [prim] + [s_ for s_ in seeds[1:] if overlap_of(L_full, Fm_full, F_ref, s_["dx0"], s_["df0"], bar_l, bar_f)["admissible"]]
    no_overlap = not prim_ov["admissible"]
    ii = [np.abs(np.arange(n) - i) for i, n in zip(pk, ncc.shape)]
    ex = int(p.coarse_excl)
    # 3-D sharpness: best value ≥ coarse_excl cells/frames away in ANY axis. A real saccade inside the moving scan
    # (CS001_OS v2: −7 → −40 laterals at frame 42) is a genuine second LATERAL mode at this scale (0.665 vs the
    # 0.698 peak on v2→v1), resolved by the per-frame fine stage — so it is REPORTED, not flagged.
    far = (ii[0][:, None, None] >= ex) | (ii[1][None, :, None] >= ex) | (ii[2][None, None, :] >= ex)
    off3 = float(ncc[far].max()) if far.any() else float("nan")
    sharp3 = peak - off3 if np.isfinite(off3) else float("nan")
    # df sharpness (the FLAGGED one): the frame offset is the coarse stage's only non-refinable output, so the
    # peak of the df profile (max over dx, dz per df) must stand ≥ coarse_min_sharp above the best value
    # ≥ coarse_excl frames away (CS001_OS: 0.060 / 0.069) and be the profile's only local maximum within 0.1.
    prof = ncc.max(axis=(0, 1)); df_axis = np.arange(-ms[2], ms[2] + 1)
    far_df = ii[2] >= ex
    off_df = float(prof[far_df].max()) if far_df.any() else float("nan")
    sharp_df = peak - off_df if np.isfinite(off_df) else float("nan")
    n_pk = 0
    for i in range(prof.size):
        if prof[i] <= -1.0 or prof[i] < peak - 0.1:
            continue
        if (i == 0 or prof[i] >= prof[i - 1]) and (i == prof.size - 1 or prof[i] >= prof[i + 1]):
            n_pk += 1
    n_sep = _count_maxima(ncc, 0.03, int(p.coarse_excl)) if n_mov > 0 else 0
    return {"dx0": dx0, "dz0": dz0, "df0": df0, "ncc": peak, "seed_ncc": float(seed_ncc), "seed_source": seed_source,
            "full": {"dx0": full["dx0"], "dz0": full["dz0"], "df0": full["df0"], "ncc": peak, "on_bound": bool(on_bound)},
            "split": split, "sharpness": float(sharp_df), "off_peak_df": off_df,
            "sharpness_3d": float(sharp3), "off_peak_3d": off3, "on_bound": bool(on_bound),
            "overlap": float(N[pk]), "n_mov_cells": n_mov, "df_profile": prof.astype(np.float32),
            "df_axis": df_axis, "n_df_peaks": int(n_pk), "max_shift_full": (ms[0] * ds[0], ms[1] * ds[1], ms[2]),
            "max_shift_initial": (ms_initial[0] * ds[0], ms_initial[1] * ds[1], ms_initial[2]),
            "widened_axes": widened_axes or None, "coarse_ds": ds, "coarse_rows": tuple(int(v) for v in mov.coarse_rows),
            "coarse_mask_mode": str(mov.coarse_mask_mode), "seeds": seeds, "n_maxima_within_0.03": int(n_sep),
            "min_overlap_frac": (float(p.coarse_min_overlap) if legacy_floor else None), "min_overlap_cells": int(ovl),
            "min_overlap_rule": ("legacy_fraction" if legacy_floor else "absolute"),
            "overlap": prim_ov, "no_overlap": bool(no_overlap), "beyond_bar_peak": beyond_bar,
            "overlap_bar": {"laterals": bar_l, "frame_frac": bar_f, "max_dx": float(p.max_dx), "coarse_max_dx": int(p.coarse_max_dx)},
            "time_s": time.time() - t0}


def coarse_unrestricted(ref: BandData, mov: BandData, params: PairParams | dict | None = None) -> dict:
    """PARTIAL OVERLAP (2026-09-12): the OVERLAP-AGNOSTIC coarse peak — the whole pyramid range in dx (±(Lc − 1) cells), the
    configured dz / df ranges, with a LOW cell floor (half the bar's laterals × half its frame fraction at the mask's
    density): the record of WHERE the best match lies when the admissible search found no correspondence. A peak beyond
    the overlap bar names a refused pair 'no_overlap' with its offset (register_pair); never a seed."""
    L_full = int(ref.band.shape[0])
    p = _pair_params(params).resolved(L_full)
    t0 = time.time()
    ds = tuple(int(v) for v in ref.coarse_ds)
    Lc, Tc, Fc = mov.coarse.shape
    bar_l = int(p.min_overlap_laterals); bar_f = float(p.min_overlap_frame_frac)
    floor = _overlap_floor_cells(mov.coarse_mask, ds, max(4, bar_l // 2), 0.5 * bar_f)
    ms = (max(1, Lc - 1), max(1, min(int(p.coarse_max_dz) // ds[1], Tc - 1)), max(1, min(int(p.coarse_max_df), max(1, Fc // 2))))
    if int(mov.coarse_mask.sum()) == 0:
        return {"dx0": float("nan"), "dz0": float("nan"), "df0": 0, "ncc": float("nan"), "overlap": None, "beyond_bar": False,
                "max_shift": (ms[0] * ds[0], ms[1] * ds[1], ms[2]), "min_overlap_cells": int(floor), "time_s": time.time() - t0}
    _, _, pk = _coarse_peak(ref, mov.coarse, mov.coarse_mask, ms, ds, floor)
    ov = overlap_of(L_full, int(mov.n_frames), int(ref.n_frames), pk["dx0"], pk["df0"], bar_l, bar_f)
    return {"dx0": float(pk["dx0"]), "dz0": float(pk["dz0"]), "df0": int(pk["df0"]), "ncc": float(pk["ncc"]), "on_bound": bool(pk["on_bound"]),
            "overlap": ov, "beyond_bar": (not ov["admissible"]), "max_shift": (ms[0] * ds[0], ms[1] * ds[1], ms[2]),
            "min_overlap_cells": int(floor), "time_s": time.time() - t0}


# ── (b) fine, per frame ────────────────────────────────────────────────────────────────────────────────────────
def _shift2_cols(img: np.ndarray, dx: int, dz_cols: np.ndarray, fill=0):
    """out[l, k] = img[l − dx, k − dz_cols[l − dx]] (moving + shift = fixed) — a per-COLUMN integer depth shift of the
    moving B-scan (the shift of column l' of the moving image is dz_cols[l']), `fill` outside. Used to centre the
    per-lateral depth search on a candidate rigid TILT a + b·x (E7): a rigid transform evaluated column by column,
    never a warp of the tissue."""
    L, T = img.shape
    dx = int(dx)
    ls = np.arange(L) - dx
    okl = (ls >= 0) & (ls < L)
    dzc = np.zeros(L, int)
    dzc[okl] = np.asarray(dz_cols, int)[ls[okl]]
    KK = np.arange(T)[None, :] - dzc[:, None]
    okk = (KK >= 0) & (KK < T)
    src = img[np.clip(ls, 0, L - 1)[:, None], np.clip(KK, 0, T - 1)]
    return np.where(okl[:, None] & okk, src, fill)


def _column_dz_scores(Xf, Mf, Xm, Mm, dxi: int, dz_grid, win, min_rows: int = 8, centre: np.ndarray | None = None) -> np.ndarray:
    """(n_dz, L) per-REFERENCE-lateral mean local NCC of the moving B-scan shifted by (dxi, dz) onto the
    reference B-scan, NaN where fewer than min_rows rows carry a valid window. `centre` (L,) int per MOVING lateral
    adds a per-column depth offset (E7: the search re-centred on a fitted rigid line); None = a constant shift."""
    L, T = Xf.shape
    S = np.full((len(dz_grid), L), np.nan)
    for j, dz in enumerate(dz_grid):
        if centre is None:
            MMs = _shift2(Mm, dxi, int(dz), False)
        else:
            MMs = _shift2_cols(Mm, dxi, np.asarray(centre, int) + int(dz), False)
        e = Mf & MMs
        if int(e.sum()) < 4 * min_rows:
            continue
        Ms = _shift2(Xm, dxi, int(dz), 0.0) if centre is None else _shift2_cols(Xm, dxi, np.asarray(centre, int) + int(dz), 0.0)
        nc = local_ncc(Xf, Ms, e, win)
        v = e & np.isfinite(nc)
        cnt = v.sum(axis=1)
        col = np.where(v, nc, 0.0).sum(axis=1) / np.maximum(cnt, 1)
        S[j] = np.where(cnt >= min_rows, col, np.nan)
    return S


def _best_dz(S: np.ndarray, dz_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per lateral: the dz of the best column score with a parabolic sub-pixel refinement; (dz (L,), score (L,))."""
    n, L = S.shape
    nS = np.where(np.isfinite(S), S, -np.inf)
    j = np.argmax(nS, axis=0)
    best = nS[j, np.arange(L)]
    ok = np.isfinite(best)
    dz = dz_grid[j].astype(float)
    inner = ok & (j > 0) & (j < n - 1)
    if inner.any():
        li = np.flatnonzero(inner)
        y0 = S[j[li] - 1, li]; y1 = S[j[li], li]; y2 = S[j[li] + 1, li]
        good = np.isfinite(y0) & np.isfinite(y2)
        den = y0 - 2 * y1 + y2
        with np.errstate(invalid="ignore", divide="ignore"):
            d = np.where(good & (np.abs(den) > 1e-12), 0.5 * (y0 - y2) / den, 0.0)
        dz[li] = dz[li] + np.clip(d, -1, 1)
    dz[~ok] = np.nan
    best = np.where(ok, best, np.nan)
    return dz, best


def fine_register(ref: BandData, mov: BandData, coarse: dict, params: PairParams | dict | None = None, *,
                  fft_only: bool = False) -> dict:
    """Design element 3(b): per moving frame f (reference partner f + df0), (1) the frame's lateral/depth shift
    (dx[f], dz[f]) and peak masked NCC from a masked FFT-NCC of the moving band B-scan against the reference
    B-scan (the STRUCTURE feature), searched within ±fine_win_dx / ±fine_win_dz of the coarse peak (prototype A
    fine_pair: NCC median 0.80-0.81, 92-94 of 101 frames valid on CS001_OS); (2) the per-lateral band-space depth
    offset dz_band(l) from a masked LOCAL-NCC (local_ncc) search over dz[f] ± fine_dz_search at the frame's lateral
    shift, each reference lateral taking the dz of its best column score (parabolic sub-pixel); (3) the projection
    onto the rigid B-scan model: Δz(l) = dz_band(l) + S_ref(l + dx[f], f + df) − S_mov(l, f) fitted as a robust
    line over x = (l − (L−1)/2)/((L−1)/2) → a[f] (intercept), b[f] (half-span slope), fit_rms[f] (the non-rigid
    remainder; A median 1.2-1.4 px). ROUND 9 (E7): the two SERVED LINES' own line at the frame's dx is fitted as
    well — a_lines[f] / b_lines[f] (_b_lines) — and when the first fit leaves fit_rms > rms_recentre (fine_dz_search
    / 2) the per-lateral search is RE-RUN centred column by column on the fitted rigid line a + b·x (one extra
    _column_dz_scores; the better fit is kept); a frame whose fit_rms still exceeds rms_max (fine_dz_search) is
    UNMEASURED — a line disagreement the search cannot reach is never served as a confident wrong tilt (Q2 c-A: a
    25-px line rotation was served b −22 / a −7 at 0.97× the ceiling with ok True). A frame with < min_mask_cells
    in either mask, peak NCC < ncc_floor or < min_windows laterals with a valid window is UNMEASURED (NaN
    everywhere). `fft_only` (E6 seed scoring) stops after step (1): dx / dz / ncc per frame only.

    ADAPTIVE WINDOW (oct_preprocess tissue_motion_widen_*): a frame whose integer peak sits within edge_tol of
    its search window edge (in dx or dz) is re-measured with the window widened ×fine_widen ('widened_frames');
    a frame still pinned after that has at_edge[f] True — its dx is a bound, not a measurement, and register_pair
    refuses a segment made of such frames ('dx_at_search_edge')."""
    p = _pair_params(params).resolved(int(ref.band.shape[0]))
    t0 = time.time()
    L, T, F = ref.band.shape
    Fm = mov.n_frames
    dx0, dz0, df = float(coarse["dx0"]), float(coarse["dz0"]), int(coarse["df0"])
    min_cells = int(p.min_mask_cells) if p.min_mask_cells else max(256, int(0.02 * L * T))
    min_win = int(p.min_windows) if p.min_windows else max(16, L // 10)
    lat = np.arange(L, dtype=float); hs = max(1.0, (L - 1) / 2.0); xc = (lat - (L - 1) / 2.0) / hs
    o = {k: np.full(Fm, np.nan) for k in ("dx", "dz", "ncc", "local_ncc", "a", "b", "rms", "win_dx", "a_lines", "b_lines", "rms_first",
                                          "b_se", "x_mean")}
    o["n_windows"] = np.zeros(Fm, int); o["n_overlap"] = np.zeros(Fm)
    o["dz_band"] = np.full((L, Fm), np.nan)
    o["measured"] = np.zeros(Fm, bool); o["at_edge"] = np.zeros(Fm, bool)
    o["search"] = np.full(Fm, -1, np.int8); o["has_cells"] = np.zeros(Fm, bool)
    o["recentred"] = np.zeros(Fm, bool); o["rms_rejected"] = np.zeros(Fm, bool); o["wide_searched"] = np.zeros(Fm, bool)
    o["shear"] = np.zeros(Fm)
    o["speckle_refined"] = np.zeros(Fm, bool); o["speckle_col"] = np.full(Fm, np.nan); o["speckle_rescued"] = np.zeros(Fm, bool)
    o["t_fft"] = 0.0; o["t_local"] = 0.0; o["t_speckle"] = 0.0
    win_s = tuple(int(v) for v in p.local_win)
    wide_ok = bool(p.dz_wide) and not fft_only
    cap_w = max(int(p.fine_dz_search), int(round(float(p.max_tilt_px)))); step_w = max(1, int(p.dz_wide_step))
    shear_ok = bool(p.shear_rescue) and not fft_only
    # a WEAK coarse peak is the pair-level signature of sheared bands (a served-line slope error shears the whole moving band:
    # coarse NCC 0.25 on a 25-px rotation): every frame then tries the shear grid — an in-window ALIAS at 0.53-0.61 stays
    # above vote_ncc and would otherwise never be re-matched (the sheared truth peaks at 0.93-0.96)
    shear_all = shear_ok and (not np.isfinite(float(coarse.get("ncc", np.nan))) or float(coarse.get("ncc", np.nan)) < float(p.coarse_min_ncc))
    shear_grid = [float(s_) for k_ in range(1, cap_w // max(1, int(p.shear_step)) + 1) for s_ in (k_ * int(p.shear_step), -k_ * int(p.shear_step))]
    dz_off = np.arange(-int(p.fine_dz_search), int(p.fine_dz_search) + 1)
    tol = int(p.edge_tol)
    rms_re = float(p.rms_recentre); rms_max = float(p.rms_max)
    ref_feat = ref.feat; mov_feat = mov.feat
    # E10: the SPECKLE refinement of the rigid line — the per-lateral depth search re-run on the speckle feature (prototype
    # A's sigma-1.5 image, window local_win_speckle) centred column by column on the structure fit, accepted only where the
    # speckle demonstrably correlates on that frame (mean best column NCC ≥ speckle_refine_min_col) and stays inside the
    # refinement window; the structure fit stands everywhere else (a rotated pose, a junk frame, uncorrelated speckle)
    sp_ok = bool(p.speckle_refine) and ref.feat_speckle is not None and mov.feat_speckle is not None and not fft_only
    ref_sp = ref.feat_speckle if sp_ok else None; mov_sp = mov.feat_speckle if sp_ok else None
    dz_sp = np.arange(-int(p.speckle_refine_dz), int(p.speckle_refine_dz) + 1); wsp = tuple(p.local_win_speckle)

    def window(win_dx: float, win_dz: float, abs_dx: float | None = None) -> dict:
        ms = (min(L - 1, int(abs(dx0)) + int(win_dx) + 2), min(T - 1, int(abs(dz0)) + int(win_dz) + 2))
        shape = fft_shape((L, T), ms)
        ax0 = np.arange(-ms[0], ms[0] + 1); ax1 = np.arange(-ms[1], ms[1] + 1)
        win_ok = (np.abs(ax0 - dx0) <= win_dx)[:, None] & (np.abs(ax1 - dz0) <= win_dz)[None, :]
        # the EFFECTIVE bounds of the search (window ∩ the FFT range ∩ the absolute cap |dx| ≤ abs_dx when given)
        b = [max(-ms[0], dx0 - win_dx), min(ms[0], dx0 + win_dx), max(-ms[1], dz0 - win_dz), min(ms[1], dz0 + win_dz)]
        if abs_dx is not None:
            win_ok &= (np.abs(ax0) <= float(abs_dx))[:, None]
            b[0] = max(b[0], -float(abs_dx)); b[1] = min(b[1], float(abs_dx))
        return {"ms": ms, "shape": shape, "win_ok": win_ok, "bounds": tuple(b), "win_dx": float(win_dx)}

    KEYS = ("dx", "dz", "ncc", "local_ncc", "a", "b", "rms", "win_dx", "n_windows", "n_overlap", "measured", "at_edge", "search",
            "a_lines", "b_lines", "rms_first", "recentred", "rms_rejected", "speckle_refined", "speckle_col", "speckle_rescued",
            "b_se", "x_mean", "wide_searched", "shear")

    def reset(f: int) -> None:
        for k in ("dx", "dz", "ncc", "local_ncc", "a", "b", "rms", "win_dx", "a_lines", "b_lines", "rms_first", "b_se", "x_mean"):
            o[k][f] = np.nan
        o["n_windows"][f] = 0; o["n_overlap"][f] = 0.0; o["dz_band"][:, f] = np.nan
        o["measured"][f] = False; o["at_edge"][f] = False; o["search"][f] = -1
        o["recentred"][f] = False; o["rms_rejected"][f] = False; o["wide_searched"][f] = False; o["shear"][f] = 0.0
        o["speckle_refined"][f] = False; o["speckle_col"][f] = np.nan; o["speckle_rescued"][f] = False

    def snapshot(f: int) -> dict:
        d = {k: o[k][f] for k in KEYS}; d["dz_band"] = o["dz_band"][:, f].copy()
        return d

    def restore(f: int, d: dict) -> None:
        for k in KEYS:
            o[k][f] = d[k]
        o["dz_band"][:, f] = d["dz_band"]

    def rescue(f: int, W: dict, level: int) -> bool:
        """Re-measure a WEAK frame (nothing, or a peak below vote_ncc, in its window) in the wider window W: kept
        only when the wider search is DECISIVE — its peak reaches far_ncc_floor AND beats the home-window peak by
        rescue_margin — else the frame's original measurement is restored (CS001 v2 frame 100: home 0.49, a far
        alias at 0.53 and dx 275 would otherwise replace a real match and drag three neighbours off the trend)."""
        d = snapshot(f)
        home = float(d["ncc"]) if np.isfinite(d["ncc"]) else -1.0
        reset(f); measure(f, W, max(float(p.far_ncc_floor), home + float(p.rescue_margin)), level)
        if np.isfinite(o["dx"][f]):
            return True
        restore(f, d)
        return False

    def fit_line(dz_m: np.ndarray, S_ref_at: np.ndarray, S_mov: np.ndarray, f: int):
        dz_tot = dz_m + S_ref_at - S_mov
        if mov.valid is not None:
            dz_tot = np.where(mov.valid[:, f], dz_tot, np.nan)
        return robust_line(xc, dz_tot, min_n=min_win, k=p.mad_k)

    def measure(f: int, W: dict, floor: float, level: int) -> None:
        """One search of frame f inside window W: the peak must reach `floor`; `level` (0 home window, 1
        widened, 2 full range) is recorded in o['search'] when the frame measures."""
        fr = f + df
        if not (0 <= fr < F):
            return
        ms, shape, win_ok, bnd = W["ms"], W["shape"], W["win_ok"], W["bounds"]
        Xf, Mf = ref_feat[:, :, fr], ref.match_mask[:, :, fr]
        Xm, Mm = mov_feat[:, :, f], mov.match_mask[:, :, f]
        n_f, n_m = int(Mf.sum()), int(Mm.sum())
        if n_f < min_cells or n_m < min_cells:
            return
        o["has_cells"][f] = True
        t1 = time.time()
        Fa = prep_fixed(Xf, Mf, shape)
        # the overlap must reach 20 % of the LARGER frame mask (round 9: a structure feature has ~1/10 of the speckle
        # feature's independent samples — an 18-lateral overlap of a 128-lateral band aliased at NCC 0.78-0.86)
        ncc, N = masked_ncc_fft(None, None, Xm, Mm, ms, min_overlap=max(1, int(0.20 * max(n_m, n_f))), fixed_fft=Fa, shape=shape)
        pk = np.unravel_index(int(np.argmax(np.where(win_ok, ncc, -1.0))), ncc.shape)
        pk_ncc = float(ncc[pk])
        if sp_ok and pk_ncc < float(p.vote_ncc):
            # E10 RESCUE: a frame WEAK on the structure feature (a per-frame noise burst blurs the smooth structure) is
            # re-measured on the SPECKLE feature in the same window; the speckle peak replaces it only when it is SOUND
            # (≥ vote_ncc) and beats the structure peak by rescue_margin — where the speckle demonstrably correlates
            Fa_s = prep_fixed(ref_sp[:, :, fr], Mf, shape)
            ncc_s, N_s = masked_ncc_fft(None, None, mov_sp[:, :, f], Mm, ms, min_overlap=max(1, int(0.20 * max(n_m, n_f))), fixed_fft=Fa_s, shape=shape)
            pk_s = np.unravel_index(int(np.argmax(np.where(win_ok, ncc_s, -1.0))), ncc_s.shape)
            pk_ncc_s = float(ncc_s[pk_s])
            if pk_ncc_s >= float(p.vote_ncc) and pk_ncc_s >= pk_ncc + float(p.rescue_margin):
                ncc, N, pk, pk_ncc = ncc_s, N_s, pk_s, pk_ncc_s
                o["speckle_rescued"][f] = True
        centre0 = None
        if shear_ok and (pk_ncc < float(p.vote_ncc) or shear_all):
            # ROUND 10 (R5) SHEAR RESCUE: a moving band flattened to a served line with a SLOPE error against the reference's is
            # sheared in band space, and a rigid-shift FFT cannot match a sheared B-scan (35 px half-span: peak 0.5, every
            # frame weak). The FFT is re-run with the moving B-scan pre-sheared column by column by ±shear_step … max_tilt_px
            # (a rigid candidate tilt evaluated per column, never a warp of the tissue — the same _shift2_cols the re-centring
            # uses); the winning shear must be sound and decisive, and it centres the per-lateral search below
            best = (pk_ncc, ncc, N, pk, 0.0)
            for s_ in shear_grid:
                dzc_ = np.rint(s_ * xc).astype(int)
                ncc_h, N_h = masked_ncc_fft(None, None, _shift2_cols(Xm, 0, dzc_, 0.0), _shift2_cols(Mm, 0, dzc_, False), ms,
                                            min_overlap=max(1, int(0.20 * max(n_m, n_f))), fixed_fft=Fa, shape=shape)
                pk_h = np.unravel_index(int(np.argmax(np.where(win_ok, ncc_h, -1.0))), ncc_h.shape)
                v_h = float(ncc_h[pk_h])
                if v_h > best[0]:
                    best = (v_h, ncc_h, N_h, pk_h, float(s_))
            if best[4] != 0.0 and best[0] >= float(p.vote_ncc) + float(p.shear_margin) and best[0] >= pk_ncc + float(p.rescue_margin):
                pk_ncc, ncc, N, pk = best[0], best[1], best[2], best[3]
                o["shear"][f] = best[4]
                centre0 = np.rint(best[4] * xc).astype(int)
        o["t_fft"] += time.time() - t1
        if pk_ncc < float(floor):
            o["ncc"][f] = pk_ncc
            return
        sp = subpix_peak(ncc, pk)
        dxf = float(sp[0] - ms[0]); dzf = float(sp[1] - ms[1])
        pdx, pdz = int(pk[0]) - ms[0], int(pk[1]) - ms[1]         # the INTEGER peak decides the edge test
        o["at_edge"][f] = bool(pdx <= bnd[0] + tol or pdx >= bnd[1] - tol or pdz <= bnd[2] + tol or pdz >= bnd[3] - tol)
        o["win_dx"][f] = W["win_dx"]; o["search"][f] = int(level)
        o["dx"][f] = dxf; o["dz"][f] = dzf; o["ncc"][f] = pk_ncc; o["n_overlap"][f] = float(N[pk])
        if fft_only:
            return
        # the SERVED LINES' own line at this dx (E7): the tilt prior, known before any tissue matching
        S_mov = np.asarray(mov.served[:, f], float)
        S_ref_at = _interp_nan(ref.served[:, fr], lat + dxf)
        a_l, b_l = _b_lines(ref.served[:, fr], S_mov, None if mov.valid is None else mov.valid[:, f], dxf, xc, min_win, p.mad_k)
        o["a_lines"][f] = a_l; o["b_lines"][f] = b_l
        # per-lateral band-space depth offset by local NCC
        t1 = time.time()
        dxi = int(round(dxf)); dz_grid = int(round(dzf)) + dz_off
        S = _column_dz_scores(Xf, Mf, Xm, Mm, dxi, dz_grid, win_s, centre=centre0)
        dz_r, sc_r = _best_dz(S, dz_grid)                       # on reference laterals
        dz_m = np.full(L, np.nan)
        lr = np.arange(L); lm = lr - dxi
        inside = (lm >= 0) & (lm < L)
        dz_m[lm[inside]] = dz_r[inside] + (0.0 if centre0 is None else centre0[lm[inside]])
        # the per-lateral estimates piling up at the grid's edge: the search was CONFINED (the truth lies beyond ±fine_dz_search)
        _fin_r = inside & np.isfinite(dz_r)
        edge_conf = float(np.mean(np.abs(dz_r[_fin_r] - int(round(dzf))) >= float(p.fine_dz_search) - 1.0)) if _fin_r.any() else 0.0
        nw = int(np.isfinite(dz_m).sum())
        o["n_windows"][f] = nw
        o["local_ncc"][f] = float(np.nanmean(sc_r)) if np.isfinite(sc_r).any() else np.nan
        if nw < min_win:
            o["t_local"] += time.time() - t1
            return
        a, b, rms, keep = fit_line(dz_m, S_ref_at, S_mov, f)
        o["rms_first"][f] = rms
        win_used = int(win_s[0])
        if np.isfinite(a) and np.isfinite(rms) and (rms > rms_re or edge_conf > 0.15):
            # E7: re-centre the per-lateral search on a fitted RIGID line (a per-column shift of the candidate tilt); the
            # better fit is kept. ROUND 10 (R5): the candidate lines are the first fit AND the line of a WIDE coarse
            # per-lateral search (dz ± max_tilt_px in dz_wide_step px) — a 10-40 px line-vs-tissue disagreement lies outside
            # the ±fine_dz_search grid, so the first fit was confined and re-centring on it never reached the tissue
            cands = [(a, b)]
            if wide_ok:
                dz_w = int(round(dzf)) + np.arange(-cap_w, cap_w + 1, step_w)
                Sw = _column_dz_scores(Xf, Mf, Xm, Mm, dxi, dz_w, win_s, centre=centre0)
                dz_rw, _ = _best_dz(Sw, dz_w)
                dz_mw = np.full(L, np.nan); dz_mw[lm[inside]] = dz_rw[inside] + (0.0 if centre0 is None else centre0[lm[inside]])
                o["wide_searched"][f] = True
                if int(np.isfinite(dz_mw).sum()) >= min_win:
                    aw, bw, _rw, _kw = fit_line(dz_mw, S_ref_at, S_mov, f)
                    if np.isfinite(aw) and np.isfinite(bw) and (abs(aw - a) > 2.0 or abs(bw - b) > 2.0):
                        cands.append((aw, bw))
            for ac, bc in cands:
                pred = ac + bc * xc - (S_ref_at - S_mov)             # band-space dz the line predicts, per moving lateral
                pred = np.where(np.isfinite(pred), pred, 0.0)
                centre = np.clip(np.rint(pred - int(round(dzf))), -T, T).astype(int)
                S2 = _column_dz_scores(Xf, Mf, Xm, Mm, dxi, dz_grid, win_s, centre=centre)
                dz_r2, sc_r2 = _best_dz(S2, dz_grid)
                dz_m2 = np.full(L, np.nan)
                dz_m2[lm[inside]] = dz_r2[inside] + centre[lm[inside]]
                if int(np.isfinite(dz_m2).sum()) >= min_win:
                    a2, b2, rms2, keep2 = fit_line(dz_m2, S_ref_at, S_mov, f)
                    if np.isfinite(a2) and (not np.isfinite(rms) or rms2 < rms):
                        a, b, rms, keep, dz_m = a2, b2, rms2, keep2, dz_m2
                        o["recentred"][f] = True
                        o["local_ncc"][f] = float(np.nanmean(sc_r2)) if np.isfinite(sc_r2).any() else o["local_ncc"][f]
                        o["n_windows"][f] = int(np.isfinite(dz_m2).sum())
        o["t_local"] += time.time() - t1
        o["dz_band"][:, f] = dz_m
        if not np.isfinite(a):
            return
        if np.isfinite(rms) and rms > rms_max:
            o["rms_rejected"][f] = True; o["rms"][f] = rms
            return                                               # a fit the rigid model does not hold: unmeasured
        if sp_ok:
            t2 = time.time()
            pred = a + b * xc - (S_ref_at - S_mov)                 # the structure line, per moving lateral, in band rows
            pred = np.where(np.isfinite(pred), pred, 0.0)
            centre = np.clip(np.rint(pred), -T, T).astype(int)
            Ssp = _column_dz_scores(ref_sp[:, :, fr], Mf, mov_sp[:, :, f], Mm, dxi, dz_sp, wsp, centre=centre)
            dz_rs, sc_s = _best_dz(Ssp, dz_sp)
            col = float(np.nanmean(sc_s)) if np.isfinite(sc_s).any() else float("nan")
            o["speckle_col"][f] = col
            if np.isfinite(col) and col >= float(p.speckle_refine_min_col):
                dz_ms = np.full(L, np.nan)
                dz_ms[lm[inside]] = dz_rs[inside] + centre[lm[inside]]
                fin_s = inside & np.isfinite(dz_rs)
                n_ws = int(fin_s.sum())
                edge_frac = float(np.mean(np.abs(dz_rs[fin_s]) >= float(p.speckle_refine_dz))) if n_ws else 1.0
                if n_ws >= min_win and edge_frac <= 0.3:
                    a3, b3, rms3, keep3 = fit_line(dz_ms, S_ref_at, S_mov, f)
                    if (np.isfinite(a3) and abs(a3 - a) <= float(p.speckle_refine_dz) and abs(b3 - b) <= float(p.speckle_refine_dz)):
                        a, b, rms, dz_m, keep = a3, b3, rms3, dz_ms, keep3
                        win_used = int(wsp[0])
                        o["speckle_refined"][f] = True
                        o["dz_band"][:, f] = dz_m
            o["t_speckle"] += time.time() - t2
        # ROUND 10 (R3): the tilt's PRECISION — the slope's standard error from the line's residual over the EFFECTIVE number
        # of independent per-lateral samples (the fitted laterals / the local-NCC window width: adjacent columns share a
        # window) and the spread of x over the fitted laterals (a half-overlap frame has half the lever arm); register_pair
        # serves a frame its own tilt only when it is precise (tilt_se_max)
        km = np.asarray(keep, bool) if keep is not None else np.zeros(L, bool)
        kx = xc[km] if km.any() else xc[np.isfinite(dz_m)]
        if kx.size >= 2 and np.isfinite(rms):
            n_eff = max(1.0, float(kx.size) / float(max(1, win_used)))
            sd_x = float(np.std(kx))
            sig = float(rms) * float(np.sqrt(n_eff / max(n_eff - 2.0, 1.0)))
            o["b_se"][f] = sig / (float(np.sqrt(n_eff)) * max(sd_x, 0.05))
            o["x_mean"][f] = float(np.mean(kx))
        o["a"][f] = a; o["b"][f] = b; o["rms"][f] = rms
        o["measured"][f] = True

    W1 = window(float(p.fine_win_dx), float(p.fine_win_dz))
    for f in range(Fm):
        measure(f, W1, float(p.ncc_floor), 0)
    pinned = np.flatnonzero(o["at_edge"] & np.isfinite(o["dx"]))
    is_weak = lambda: o["has_cells"] & ~(np.nan_to_num(o["ncc"], nan=-1.0) >= float(p.vote_ncc)) & ~o["at_edge"]  # noqa: E731
    o["widened_frames"] = []; o["rescued_frames"] = []; o["far_frames"] = []
    widen = float(p.fine_widen)
    far = bool(p.far_search) and not fft_only
    W2 = window(float(p.fine_win_dx) * widen, float(p.fine_win_dz) * widen) if widen > 1.0 else None
    if W2 is not None and W2["ms"] == W1["ms"] and W2["bounds"] == W1["bounds"]:
        W2 = None
    # the FAR range: every shift up to the policy cap max_dx — |dx| ≤ max_dx exactly (a mode beyond it is refused
    # anyway, and searching further only invites aliases)
    W3 = window(abs(dx0) + float(p.max_dx), float(p.fine_win_dz) * max(1.0, widen), abs_dx=float(p.max_dx)) if far else None

    def rescue_frames(frames) -> list[int]:
        """The adaptive re-search of WEAK frames (nothing, or an alias, in the home window): the widened window,
        then the far range; a rescue must be decisive (rescue). Returns the frames whose measurement changed.
        Exposed as o['rescue_frames'] so register_pair can re-search a frame whose in-window peak reached
        vote_ncc but whose rigid fit the projection rejected (round-4 refutation 2: an alias at 0.58 with a
        MAD-rejected fit broke a run and left a frame measured 65 at NCC 0.95 alone)."""
        changed = []
        for f in frames:
            f = int(f)
            if not o["has_cells"][f]:
                continue
            done = W2 is not None and rescue(f, W2, 1)
            if done:
                o["rescued_frames"].append(f)
            elif W3 is not None and rescue(f, W3, 2):
                done = True; o["far_frames"].append(f)
            if done:
                changed.append(f)
        return changed

    if pinned.size and W2 is not None and not fft_only:
        for f in pinned:
            reset(int(f)); measure(int(f), W2, float(p.ncc_floor), 1)
        o["widened_frames"] = [int(f) for f in pinned]
    if far:
        # a WEAK frame is re-searched like a pinned one (its peak may lie beyond the window: a second lateral
        # mode, whose in-window alias reaches 0.37); a rescue must reach far_ncc_floor — the wider the search,
        # the likelier an alias — else the original measurement stands
        rescue_frames(np.flatnonzero(is_weak()))
    o["rescue_frames"] = rescue_frames
    o["time_s"] = time.time() - t0
    o["df"] = df; o["ms"] = W1["ms"]
    return o


# ── (d) quality: resample the moving band by the transform, re-run band_similarity ───────────────────────────
def warp_band(mov: BandData, res: PairResult, ref: BandData, frames: Iterable[int] | None = None, *,
              features: Iterable[str] = ("struct",)) -> BandData:
    """The moving band resampled onto the REFERENCE band grid by the PairResult's rigid transform (frame offset,
    per-frame lateral shift, per-frame axial shift + tilt): reference cell (l_r, k_r, f_r) ← moving
    (l_r − dx_applied[f], k_r + S_ref(l_r) − S_mov(l_m) − a[f] − b[f]·x(l_m), f = f_r − df), linear (order 1)
    in the band, feat, band, mask, match_mask and tissue; reference frames without a partner stay empty. The
    returned BandData's `served` is the moving line CARRIED onto the reference grid (S_mov(l_m) + Δz, NaN outside
    the overlap — prototype A aligned_surface), `valid` its finite cells. `frames` restricts the warp to those
    REFERENCE frames (the cheap subset quality check); the others stay empty. `features`: which feature images to
    carry — 'struct' (feat, always) and/or 'speckle' (feat_speckle; the speckle report)."""
    L, T, F = ref.band.shape
    Fm = mov.n_frames
    lat = np.arange(L, dtype=float)
    hs = max(1.0, (L - 1) / 2.0)
    kk = np.arange(T, dtype=float)
    want_sp = ("speckle" in set(features)) and mov.feat_speckle is not None
    feat = np.zeros((L, T, F), np.float32); band = np.zeros((L, T, F), np.float32)
    feat_sp = np.zeros((L, T, F), np.float32) if want_sp else None
    mask = np.zeros((L, T, F), bool); match = np.zeros((L, T, F), bool); tis = np.zeros((L, T, F), bool)
    served = np.full((L, F), np.nan)
    fr_list = range(F) if frames is None else [int(v) for v in frames if 0 <= int(v) < F]
    for fr in fr_list:
        f = fr - int(res.df)
        if not (0 <= f < Fm):
            continue
        dxs = float(res.dx_applied[f])
        lm = lat - dxs
        inside = (lm >= 0) & (lm <= L - 1)
        Sm_at = _interp_nan(mov.served[:, f], lm)
        dzl = float(res.a[f]) + float(res.b[f]) * (lm - (L - 1) / 2.0) / hs
        Sr = np.asarray(ref.served[:, fr], float)
        off = Sr - Sm_at - dzl                                   # band-row offset per reference lateral
        okl = inside & np.isfinite(off)
        if not okl.any():
            continue
        KK = kk[None, :] + np.where(okl, off, 0.0)[:, None]
        LL = np.broadcast_to(np.where(okl, lm, 0.0)[:, None], (L, T))
        coords = [LL, KK]
        # 'grid-constant': interpolate up to the band's edge and blend with 0 beyond it — plain 'constant' returns
        # cval for ANY coordinate past the last index without interpolating, so an identity warp with a ≈ 5e-8
        # dropped the band's last row in every column (1 % of the match mask; the identity coverage was 0.939
        # against the metric's own support of 0.949). A cell whose centre lies more than half outside is dropped.
        feat[:, :, fr] = ndi.map_coordinates(mov.feat[:, :, f], coords, order=1, cval=0.0, mode="grid-constant")
        if want_sp:
            feat_sp[:, :, fr] = ndi.map_coordinates(mov.feat_speckle[:, :, f], coords, order=1, cval=0.0, mode="grid-constant")
        band[:, :, fr] = ndi.map_coordinates(mov.band[:, :, f], coords, order=1, cval=0.0, mode="grid-constant")
        m_ = ndi.map_coordinates(mov.mask[:, :, f].astype(np.float32), coords, order=1, cval=0.0, mode="grid-constant") > 0.5
        mm_ = ndi.map_coordinates(mov.match_mask[:, :, f].astype(np.float32), coords, order=1, cval=0.0, mode="grid-constant") > 0.5
        tt_ = ndi.map_coordinates(mov.tissue[:, :, f].astype(np.float32), coords, order=1, cval=0.0, mode="grid-constant") > 0.5
        mask[:, :, fr] = m_ & okl[:, None]; match[:, :, fr] = mm_ & okl[:, None]; tis[:, :, fr] = tt_ & okl[:, None]
        feat[:, :, fr][~match[:, :, fr]] = 0.0
        if want_sp:
            feat_sp[:, :, fr][~match[:, :, fr]] = 0.0
        served[:, fr] = np.where(okl, Sm_at + dzl, np.nan)
    # a stand-in pyramid from the warped FINE band (the coarse band is not carried): never fed to coarse_register
    coarse = block_mean(feat * match, ref.coarse_ds)
    coarse_mask = block_mean(match.astype(np.float32), ref.coarse_ds) > 0.5
    post = np.where(np.isfinite(served), _carry_rows(mov.posterior_row, res, L, F, Fm), np.nan)
    return BandData(cid=f"{mov.cid}->{ref.cid}", row0=ref.row0, band=band, mask=mask, feat=feat, tissue=tis,
                    match_mask=match, posterior_row=post, coarse=coarse.astype(np.float32), coarse_mask=coarse_mask,
                    coarse_ds=ref.coarse_ds, spacing=np.asarray(mov.spacing), noise_floor=mov.noise_floor,
                    otsu_thr=mov.otsu_thr, timings={}, served=served, valid=np.isfinite(served),
                    coarse_rows=(int(ref.row0), int(ref.row0) + T), coarse_mask_mode="warped", feat_speckle=feat_sp,
                    sigma_struct=mov.sigma_struct, sigma_speckle=mov.sigma_speckle,
                    lateral_scale=mov.lateral_scale, lateral_offset=mov.lateral_offset)


def _carry_rows(rows: np.ndarray, res: PairResult, L: int, F: int, Fm: int) -> np.ndarray:
    """An (L, Fm) per-cell row quantity of the moving band carried onto the reference grid (lateral shift +
    frame offset only — band rows are relative to the served line)."""
    out = np.full((L, F), np.nan)
    lat = np.arange(L, dtype=float)
    for fr in range(F):
        f = fr - int(res.df)
        if not (0 <= f < Fm):
            continue
        out[:, fr] = _interp_nan(np.asarray(rows[:, f], float), lat - float(res.dx_applied[f]))
    return out


def pair_quality(ref: BandData, mov: BandData, res: PairResult, ceiling: dict | None = None,
                 window: tuple[int, int] = LOCAL_WIN, frames: Iterable[int] | None = None,
                 feature: str = "struct") -> tuple[dict, BandData]:
    """Design element 3(d): warp_band(mov) → band_similarity against the reference band inside the joint match
    mask: matched fractions (> 0.5, > 0.3) of the OVERLAPPED reference voxels, coverage = overlapped / all
    reference match-mask voxels, mean local NCC, the residual of the carried moving line to the reference line
    (rms px over the overlap), and the reference's adjacent-frame ceiling (band_similarity(ref, ref,
    frame_offset=1), computed unless given) for the relative match — all on the `feature` image ('struct', the
    verdict, window LOCAL_WIN_STRUCT; 'speckle', the report, window LOCAL_WIN_SPECKLE — pass the matching window
    and ceiling). `frames` restricts everything — the warp, the match, the coverage denominator and the ceiling
    (when computed here) — to those reference frames: the cheap check of register_pair's quality=False path.
    Returns (quality dict, the warped band)."""
    t0 = time.time()
    fr_sub = None if frames is None else [int(v) for v in frames]
    w = warp_band(mov, res, ref, frames=fr_sub, features=("struct", feature))
    sim = band_similarity(ref, w, window=window, frames=fr_sub, feature=feature)
    n_ref = int(ref.match_mask.sum()) if fr_sub is None else int(ref.match_mask[:, :, fr_sub].sum())
    st = sim.stats
    q = {"n_ref": n_ref, "n_eval": int(st["n_eval"]), "coverage": float(st["n_eval"]) / max(1, n_ref),
         "ncc_mean": float(st["ncc_mean"]), "n_pairs": int(st["n_pairs"])}
    for t in NCC_THRESHOLDS:
        q[f"matched_frac_{t}"] = float(st[f"matched_frac_{t}"])
        q[f"matched_of_ref_{t}"] = float(st[f"matched_frac_{t}"]) * float(st["n_eval"]) / max(1, n_ref) \
            if np.isfinite(st[f"matched_frac_{t}"]) else float("nan")
    q["per_frame_frac_0.5"] = np.asarray(st["per_frame_frac_0.5"], float)
    F = ref.n_frames
    frame_frac = np.full(F, np.nan)                            # by REFERENCE frame (NaN: not evaluated)
    frame_frac[sim.frames_a] = q["per_frame_frac_0.5"]
    q["frame_frac_0.5"] = frame_frac

    def by_frame(vals, fill=np.nan):
        arr = np.full(F, fill, float); arr[sim.frames_a] = np.asarray(vals, float)
        return arr
    # per-frame records by REFERENCE frame (NaN / 0 where not evaluated): the arbitration assembles the numbers of
    # any frame set from them (_assemble) instead of re-running the match
    q["frame_frac_0.3"] = by_frame(st["per_frame_frac_0.3"]); q["frame_frac_0.7"] = by_frame(st["per_frame_frac_0.7"])
    q["frame_n_eval"] = by_frame(st["per_frame_n_eval"], 0.0); q["frame_ncc_mean"] = by_frame(st["per_frame_ncc_mean"])
    q["frame_n_ref"] = ref.match_mask.sum(axis=(0, 1)).astype(float)
    d_all = w.served - ref.served
    okd_all = np.isfinite(d_all)
    q["frame_resid_ss"] = np.sum(np.where(okd_all, d_all, 0.0) ** 2, axis=0); q["frame_resid_n"] = okd_all.sum(axis=0).astype(float)
    d = d_all if fr_sub is None else d_all[:, fr_sub]
    okd = np.isfinite(d)
    q["surface_residual_rms_px"] = float(np.sqrt(np.mean(d[okd] ** 2))) if okd.any() else float("nan")
    q["surface_overlap_frac"] = float(okd.mean())
    q["frames"] = fr_sub
    # the per-frame ceiling: ceiling['per_frame_frac_0.5'] (F,) by reference frame f = the pair (f, f + 1) —
    # computed here when the given ceiling lacks it (register_group's shared ceiling carries it)
    if ceiling is None or "per_frame_frac_0.5" not in ceiling:
        cs = band_similarity(ref, ref, window=window, frame_offset=1, frames=fr_sub, feature=feature)
        c = cs.stats
        c_pairs = np.full(F, np.nan); c_pairs[cs.frames_a] = np.asarray(c["per_frame_frac_0.5"], float)
        if ceiling is None:
            ceiling = {k: float(v) for k, v in c.items() if not isinstance(v, np.ndarray)}
        ceiling = dict(ceiling); ceiling["per_frame_frac_0.5"] = c_pairs
    c_pairs = np.asarray(ceiling["per_frame_frac_0.5"], float)
    frame_ceiling = np.full(F, np.nan)
    for fr in range(F):                                        # a frame's ceiling: its pairs with both neighbours
        v = [c_pairs[i] for i in (fr - 1, fr) if 0 <= i < c_pairs.size and np.isfinite(c_pairs[i])]
        if v:
            frame_ceiling[fr] = float(np.mean(v))
    q["frame_ceiling_0.5"] = frame_ceiling
    # Two ceilings, two names. BAND space (like-for-like: the same metric on the same flattened band, the
    # reference against its own next frame; CS001_OS v1 (−8, 120) posterior-capped: 0.3367 > 0.5) — the CS001
    # acceptance bar is relative_match_band_space ≥ 1.0 (v2 0.379 → 1.125, v3 0.341 → 1.013). ORIGINAL space:
    # prototype A judged its rigid match (0.34-0.35 > 0.5) against the unflattened adjacent-frame ceiling
    # (0.2202, PROTOTYPE_A_CEILING gap 1) and called it 1.56-1.59×; flattening removes the frame-to-frame surface
    # slope, so that ceiling is lower and the ratio larger (1.72 / 1.55 here). `relative_match` = the band-space one.
    q["ceiling"] = dict(ceiling)
    q["ceiling_band_space"] = dict(ceiling)
    cf = float(ceiling.get("matched_frac_0.5", float("nan")))
    q["relative_match"] = q["matched_frac_0.5"] / cf if cf and np.isfinite(cf) and cf > 0 else float("nan")
    q["relative_match_band_space"] = q["relative_match"]
    q["ceiling_orig_space_prototype_A"] = float(PROTOTYPE_A_CEILING["gap1"]["matched_frac_0.5"])
    q["relative_match_orig_space_prototype_A"] = q["matched_frac_0.5"] / q["ceiling_orig_space_prototype_A"]
    q["relative_to_prototype_A_orig_ceiling"] = q["relative_match_orig_space_prototype_A"]   # legacy name
    q["feature"] = str(feature); q["window"] = (int(window[0]), int(window[1]))
    q["time_s"] = time.time() - t0
    return q, w


def overlap_reason(ov: dict | None, flags=(), dx_median=None, overlap_fraction=None) -> str:
    """R2 (2026-09-12): the reviewer-facing reason of a refused pair's overlap. When the pair was refused by the bar, by the
    bar-edge judge or by correspondence AND the record holds a correspondence BEYOND the bar — the overlap-agnostic coarse
    peak (quality['overlap']['unrestricted_peak']) or the bar-edge judge's winner (['offset'] with source 'bar_edge_judge') —
    the text names THAT: 'best correspondence at ≈ −449 laterals (12% overlap, below the 19% bar)' (+ ', the in-bar value ≈
    −410 laterals scored 0.75 against 0.87' after the judge), never the admissible seed the fine stage was handed (which
    stays in the record: ['seed'] / ['served']). Otherwise the plain 'offset ≈ −406 laterals, overlap 20%'. '' when nothing
    is known."""
    ov = ov if isinstance(ov, dict) else {}
    fl = set(flags or [])
    bar = ov.get("bar") or {}
    bar_l = bar.get("laterals"); L_ = bar.get("L")
    try:
        bar_txt = (f"{float(bar_l) / float(L_):.0%} bar" if (bar_l and L_) else (f"{int(bar_l)}-lateral bar" if bar_l else "bar"))
    except (TypeError, ValueError):
        bar_txt = "bar"

    def _fin(v) -> bool:
        try:
            return v is not None and np.isfinite(float(v))
        except (TypeError, ValueError):
            return False
    off = ov.get("offset") or {}
    src_ = off.get("source")
    u = ov.get("unrestricted_peak") or {}
    refused = bool(fl & {"no_overlap", "no_correspondence", "dx_at_search_edge", "dx_beyond_max"})
    beyond = None
    if src_ in ("unrestricted_peak", "bar_edge_judge", "beyond_bar_peak") and _fin(off.get("dx")):
        beyond = dict(off)
    elif refused and u.get("beyond_bar") and _fin(u.get("dx0")):
        beyond = dict(u.get("overlap") or {}, dx=float(u["dx0"]), source="unrestricted_peak")
    if beyond is not None:
        s = f"best correspondence at ≈ {float(beyond['dx']):+.0f} laterals"
        frac = beyond.get("fraction")
        if _fin(frac):
            # one decimal when the overlap and the bar round to the same integer percent (v6 → v1: 18.5 % against the 18.7 % bar)
            try:
                bar_frac = float(bar_l) / float(L_) if (bar_l and L_) else None
            except (TypeError, ValueError):
                bar_frac = None
            if bar_frac is not None and abs(float(frac) - bar_frac) < 0.01:
                s += f" ({float(frac):.1%} overlap, below the {bar_frac:.1%} bar)"
            else:
                s += f" ({float(frac):.0%} overlap, below the {bar_txt})"
        if beyond.get("source") == "bar_edge_judge" and _fin(beyond.get("served_dx")) and _fin(beyond.get("served_score")) and _fin(beyond.get("score")):
            s += f", the in-bar value ≈ {float(beyond['served_dx']):+.0f} laterals scored {float(beyond['served_score']):.2f} against {float(beyond['score']):.2f}"
        return s
    plain = ov.get("offset") or ov.get("served") or ov.get("seed") or {}
    dx = dx_median if _fin(dx_median) else plain.get("dx", plain.get("dx0"))
    frac = overlap_fraction if _fin(overlap_fraction) else plain.get("fraction")
    parts = []
    if _fin(dx):
        parts.append(f"offset ≈ {float(dx):+.0f} laterals")
    if _fin(frac):
        parts.append(f"overlap {float(frac):.0%}")
    if ov.get("verdict") == "none_measured":
        parts.append("no overlapping structure measured")
    return ", ".join(parts)


BAR_EDGE_TIE = 0.005   # bar_edge_check: the served in-bar value must beat the best beyond-bar candidate by more than this (matched_frac_0.5)


def bar_edge_check(ref: BandData, mov: BandData, res: PairResult, params: PairParams | dict | None = None, *,
                   ceiling: dict | None = None, window: tuple[int, int] | None = None, unrestricted: dict | None = None) -> dict:
    """BAR EDGE (2026-09-12, refutation R1): judge a served transform whose lateral shift sits at the search edge — see
    PairParams' BAR EDGE paragraph. Triggers: 'median' (the served |dx| median over the partnered live frames ≥ max_dx −
    bar_margin), 'runs' (a run of ≥ bar_edge_run partnered frames in that zone), 'unrestricted_peak' (the overlap-agnostic
    coarse peak beyond the bar by more than one coarse cell, at any df). Scopes: the whole pair (median / unrestricted
    triggers) and every zone run. Per scope the SERVED transform and every beyond-bar candidate — the served per-frame dx
    shifted outward by k coarse cells while |median| + k·cell ≤ max_dx + 2·bar_margin, and the scope's own coarse peak over
    the whole pyramid range (the run's frames alone for a run) when it lies beyond max_dx at the served df — are scored with
    pair_quality on the scope's reference frames (matched_frac_0.5 of the overlapped reference voxels); a candidate is
    scoreable when it evaluates at least 30 % of the served value's cells. The scope is CONFIRMED when the served value is
    finite and beats every scoreable candidate outright (by more than BAR_EDGE_TIE; nothing scoreable beyond the bar confirms it too);
    otherwise the beyond-bar winner is the best correspondence and the scope's frames are beyond the bar. Returns the record:
    verdict 'not_at_edge' | 'confirmed' | 'refused', triggers, scopes (each with served / candidates / best / confirmed),
    beyond_bar_frames, best (the winning candidate of the largest refused scope: signed dx, overlap, scores)."""
    t0 = time.time()
    p = _pair_params(params)
    L, T, F = ref.band.shape; Fm = int(mov.n_frames); df = int(res.df)
    p = p.resolved(L)
    max_dx = float(p.max_dx); margin = float(p.bar_margin); zone_lo = max_dx - margin
    bar_l = int(p.min_overlap_laterals); bar_f = float(p.min_overlap_frame_frac)
    ds = tuple(int(v) for v in ref.coarse_ds); step = max(1, int(ds[0]))
    win = tuple(window) if window is not None else tuple(p.local_win)
    partnered = np.array([0 <= f + df < F for f in range(Fm)], bool)
    dxa = np.asarray(res.dx_applied, float)
    fin = partnered & np.isfinite(dxa)
    live = np.asarray(res.live, bool) if res.live is not None and np.asarray(res.live).size == Fm else np.ones(Fm, bool)
    absdx = np.abs(dxa)
    sel_med = fin & live
    if not sel_med.any():
        sel_med = fin
    med = float(np.median(absdx[sel_med])) if sel_med.any() else float("nan")
    sign = float(np.sign(np.median(dxa[sel_med]))) if sel_med.any() else 1.0
    sign = sign if sign != 0 else 1.0
    zone = fin & (absdx >= zone_lo)
    runs = [(int(f0), int(f1)) for f0, f1 in _runs(zone) if f1 - f0 >= int(p.bar_edge_run)]
    u = dict(unrestricted) if isinstance(unrestricted, dict) else {}
    u_dx = float(u.get("dx0", np.nan)) if u.get("dx0") is not None else float("nan")
    u_beyond = bool(u.get("beyond_bar")) and np.isfinite(u_dx) and abs(u_dx) > max_dx + step
    triggers: list = []
    if np.isfinite(med) and med >= zone_lo:
        triggers.append("median")
    if runs:
        triggers.append("runs")
    if u_beyond:
        triggers.append("unrestricted_peak")
    rec: dict = {"edge_zone_abs_dx": float(zone_lo), "max_dx": max_dx, "bar_margin": margin, "coarse_cell": int(step),
                 "median_abs_dx": (round(med, 2) if np.isfinite(med) else None), "n_zone_frames": int(zone.sum()),
                 "zone_frames": [int(f) for f in np.flatnonzero(zone)], "runs": runs,
                 "unrestricted_dx0": (round(u_dx, 2) if np.isfinite(u_dx) else None), "unrestricted_df0": u.get("df0"),
                 "unrestricted_beyond": bool(u_beyond), "triggers": triggers, "verdict": "not_at_edge", "scopes": [],
                 "beyond_bar_frames": [], "best": None, "time_s": 0.0}
    if not triggers:
        rec["time_s"] = time.time() - t0
        return rec
    scopes: list = []
    if "median" in triggers or "unrestricted_peak" in triggers:
        scopes.append(("pair", [int(f) for f in np.flatnonzero(fin)], med))
    for f0, f1 in runs:
        sel = np.zeros(Fm, bool); sel[f0:f1] = True; sel &= fin
        scopes.append((f"run {f0}-{f1}", [int(f) for f in np.flatnonzero(sel)], float(np.median(absdx[sel]))))
    Lc, Tc, Fc = mov.coarse.shape
    ms_u = (max(1, Lc - 1), max(1, min(int(p.coarse_max_dz) // ds[1], Tc - 1)), max(1, min(int(p.coarse_max_df), max(1, Fc // 2))))
    beyond_frames: set = set()
    best_overall = None
    for name, frames, med_s in scopes:
        if not frames or not np.isfinite(med_s):
            continue
        frames_ref = [int(f) + df for f in frames]
        srec: dict = {"scope": name, "n_frames": len(frames), "frames": [int(frames[0]), int(frames[-1])], "median_abs_dx": round(float(med_s), 2)}
        try:
            q_s, _ = pair_quality(ref, mov, res, ceiling=ceiling, window=win, frames=frames_ref)
        except Exception as e:  # noqa: BLE001 — an unscorable scope cannot confirm the served value
            srec.update(error=f"{type(e).__name__}: {e}", confirmed=False, candidates=[], best=None)
            scopes_rec = rec["scopes"]; scopes_rec.append(srec)
            beyond_frames.update(frames)
            continue
        served = {"matched_frac_0.5": float(q_s["matched_frac_0.5"]), "n_eval": int(q_s["n_eval"]), "ncc_mean": float(q_s["ncc_mean"]),
                  "dx": round(float(sign * med_s), 2)}
        srec["served"] = served
        cands: list = []
        k = 1
        while med_s + k * step <= max_dx + 2.0 * margin + 1e-9:
            if med_s + k * step > max_dx:
                cands.append({"shift": float(sign * k * step), "source": "outward_shift", "abs_dx": float(med_s + k * step)})
            k += 1
        # the scope's OWN correspondence over the whole pyramid range: the whole-volume unrestricted peak for the pair, the
        # frame-masked coarse peak for a run (never a seed — a candidate the judge scores)
        pk = None
        if name == "pair":
            if u and np.isfinite(u_dx):
                pk = {"dx0": u_dx, "df0": int(u.get("df0", df)), "ncc": float(u.get("ncc", np.nan))}
        else:
            try:
                mm = np.zeros_like(mov.coarse_mask)
                fr_ = [int(f) for f in frames if 0 <= int(f) < Fc]        # the coarse pyramid keeps full frames
                mm[:, :, fr_] = mov.coarse_mask[:, :, fr_]
                if mm.any():
                    floor = _overlap_floor_cells(mm, ds, max(4, bar_l // 2), 0.5 * bar_f)
                    _, _, pkd = _coarse_peak(ref, mov.coarse, mm, ms_u, ds, floor)
                    pk = {"dx0": float(pkd["dx0"]), "df0": int(pkd["df0"]), "ncc": float(pkd["ncc"])}
            except Exception as e:  # noqa: BLE001 — the coarse candidate is optional
                srec["coarse_error"] = f"{type(e).__name__}: {e}"
        if pk is not None:
            srec["coarse_peak"] = {"dx0": round(float(pk["dx0"]), 2), "df0": int(pk["df0"]), "ncc": (round(float(pk["ncc"]), 4) if np.isfinite(pk["ncc"]) else None),
                                   "beyond": bool(abs(float(pk["dx0"])) > max_dx), "same_df": int(pk["df0"]) == df}
            if int(pk["df0"]) == df and abs(float(pk["dx0"])) > max_dx and np.isfinite(pk["dx0"]):
                shift = float(pk["dx0"]) - float(sign * med_s)
                if not any(abs(cd["shift"] - shift) < 1.0 for cd in cands):
                    cands.append({"shift": float(shift), "source": "coarse_peak", "abs_dx": float(abs(pk["dx0"])), "ncc": float(pk["ncc"])})
        for cd in cands:
            dxa2 = dxa.copy(); dxa2[frames] = dxa2[frames] + cd["shift"]
            res2 = _dc_replace(res, dx_applied=dxa2)
            try:
                q, _ = pair_quality(ref, mov, res2, ceiling=ceiling, window=win, frames=frames_ref)
                cd.update({"matched_frac_0.5": float(q["matched_frac_0.5"]), "n_eval": int(q["n_eval"]), "ncc_mean": float(q["ncc_mean"])})
            except Exception as e:  # noqa: BLE001
                cd.update({"matched_frac_0.5": float("nan"), "n_eval": 0, "ncc_mean": float("nan"), "error": f"{type(e).__name__}: {e}"})
            cd["dx"] = round(float(sign * med_s + cd["shift"]), 2)
            cd["scoreable"] = bool(np.isfinite(cd["matched_frac_0.5"]) and cd["n_eval"] > 0 and cd["n_eval"] >= 0.3 * max(1, served["n_eval"]))
        scoreable = [cd for cd in cands if cd["scoreable"]]
        best = max(scoreable, key=lambda cd: (cd["matched_frac_0.5"], cd["n_eval"])) if scoreable else None
        served_ok = bool(np.isfinite(served["matched_frac_0.5"]) and served["n_eval"] > 0)
        if best is None:
            confirmed = served_ok
        else:
            # the in-bar value must BEAT the best beyond-bar candidate outright (a tie within BAR_EDGE_TIE refuses): CS001_OD v5 → v2
            # frames 56-92 served −410.5 at 0.859 against 0.815 one coarse cell beyond the bar and falling outward — confirmed;
            # v6 → v1 served −409.5 at 0.749 against 0.869 at −421 — refused
            confirmed = served_ok and served["matched_frac_0.5"] >= best["matched_frac_0.5"] + BAR_EDGE_TIE
        srec.update(candidates=[{k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in cd.items()} for cd in cands],
                    best=({k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in best.items()} if best else None),
                    confirmed=bool(confirmed))
        rec["scopes"].append(srec)
        if not confirmed:
            beyond_frames.update(int(f) for f in frames)
            if best is not None and (best_overall is None or len(frames) > best_overall["n_frames"]):
                best_overall = {"n_frames": len(frames), "scope": name, "dx": float(best["dx"]), "abs_dx": float(best["abs_dx"]),
                                "score": float(best["matched_frac_0.5"]), "served_score": float(served["matched_frac_0.5"]),
                                "served_dx": float(served["dx"]), "source": str(best["source"]), "n_eval": int(best["n_eval"])}
    rec["beyond_bar_frames"] = sorted(int(f) for f in beyond_frames)
    rec["verdict"] = "refused" if beyond_frames else "confirmed"
    rec["best"] = best_overall
    rec["time_s"] = time.time() - t0
    return rec


def _empty_pair(ref: BandData, mov: BandData, coarse: dict, flags: list, p: PairParams, timings: dict,
                geom: dict | None = None, overlap: dict | None = None) -> PairResult:
    L, T, F = ref.band.shape; Fm = mov.n_frames
    nanF = np.full(Fm, np.nan)
    g = geom or {}
    ov = (overlap or {}).get("offset") or (overlap or {}).get("seed") or {}
    return PairResult(ref_cid=ref.cid, mov_cid=mov.cid, df=int(coarse["df0"]), dz0=float(coarse["dz0"]), dx_segments=[],
                      a=nanF.copy(), b=nanF.copy(), dx_per_frame=nanF.copy(), dx_applied=nanF.copy(),
                      dz_per_frame=nanF.copy(), a_raw=nanF.copy(), b_raw=nanF.copy(), measured=np.zeros(Fm, bool),
                      per_frame_ncc=nanF.copy(), per_frame_local_ncc=nanF.copy(), n_windows=np.zeros(Fm, int),
                      fit_rms=nanF.copy(), dz_band=np.full((L, Fm), np.nan), dx_at_edge=np.zeros(Fm, bool),
                      dx_trusted=np.zeros(Fm, bool), dx_search=np.full(Fm, -1, np.int8), live=np.zeros(Fm, bool),
                      ncc_coarse=float(coarse["ncc"]),
                      peak_sharpness=float(coarse["sharpness"]), coarse=coarse, matched_frac_0_5=float("nan"),
                      matched_frac_0_3=float("nan"), coverage=0.0, ncc_mean=float("nan"), ceiling={},
                      relative_match=float("nan"), quality=({"overlap": overlap} if overlap else {}), flags=list(flags), timings=timings,
                      params=p.as_dict(),
                      shape=(L, T, F), lateral_scale=float(g.get("scale", 1.0)), lateral_offset_mov=int(g.get("offset_mov", 0)),
                      lateral_offset_ref=int(g.get("offset_ref", 0)), pose_angle_deg=float(g.get("pose", float("nan"))),
                      b_lines=nanF.copy(), a_lines=nanF.copy(),
                      overlap_laterals=float(ov.get("laterals", float("nan")) if ov.get("laterals") is not None else float("nan")),
                      overlap_fraction=float(ov.get("fraction", float("nan")) if ov.get("fraction") is not None else float("nan")))


def _pair_geometry(ref, mov, p: PairParams, band_rows) -> tuple[BandData, BandData, dict, list]:
    """E4: the two members on ONE lateral grid. MemberData with lateral spacings that differ by more than
    lateral_scale_tol: the MOVING member is resampled by the header ratio onto the reference's physical spacing and
    both are embedded centred on a common grid (common_lateral_grid); the bands are extracted on that grid. BandData
    are used as they are (a spacing mismatch is flagged 'lateral_scale_mismatch', informational). Returns
    (ref_band, mov_band, geometry record, flags)."""
    flags: list = []
    geom = {"scale": 1.0, "offset_mov": 0, "offset_ref": 0, "L_out": None, "resampled": False}
    if isinstance(ref, MemberData) and isinstance(mov, MemberData):
        s = float(mov.spacing[0]) / float(ref.spacing[0]) if float(ref.spacing[0]) > 0 else 1.0
        if abs(s - 1.0) > float(p.lateral_scale_tol):
            (ref_m, mov_m), rec = common_lateral_grid([ref, mov], ref.cid, float(p.lateral_scale_tol))
            geom = {"scale": float(rec[mov.cid]["scale"]), "offset_mov": int(rec[mov.cid]["offset"]),
                    "offset_ref": int(rec[ref.cid]["offset"]), "L_out": int(rec[mov.cid]["L_out"]), "resampled": True}
            flags.append("lateral_resampled")
            return _as_band(ref_m, band_rows, p), _as_band(mov_m, band_rows, p), geom, flags
    ref_b = _as_band(ref, band_rows, p); mov_b = _as_band(mov, band_rows, p)
    geom.update(scale=float(mov_b.lateral_scale) / max(1e-9, float(ref_b.lateral_scale)), offset_mov=int(mov_b.lateral_offset),
                offset_ref=int(ref_b.lateral_offset), resampled=bool(mov_b.lateral_scale != 1.0))
    if ref_b.spacing.size and mov_b.spacing.size and float(ref_b.spacing[0]) > 0:
        s = float(mov_b.spacing[0]) / float(ref_b.spacing[0])
        if abs(s - 1.0) > float(p.lateral_scale_tol):
            flags.append("lateral_scale_mismatch")
    return ref_b, mov_b, geom, flags


def _pose_of(ref_b: BandData, mov_b: BandData, coarse: dict, p: PairParams) -> tuple[float, float, np.ndarray]:
    """E8: (pose angle in degrees, median |b_lines| in px half-span, b_lines per moving frame) at the coarse SEED
    (dx0, df0): the tilt the two served lines demand between the B-scan planes, before any tissue matching."""
    L = int(ref_b.band.shape[0]); F = int(ref_b.n_frames); Fm = int(mov_b.n_frames)
    lat = np.arange(L, dtype=float); xc = (lat - (L - 1) / 2.0) / max(1.0, (L - 1) / 2.0)
    min_win = int(p.min_windows) if p.min_windows else max(16, L // 10)
    dx0, df0 = float(coarse["dx0"]), int(coarse["df0"])
    bl = np.full(Fm, np.nan)
    for f in range(0, Fm, 2):
        fr = f + df0
        if not (0 <= fr < F):
            continue
        _, b = _b_lines(ref_b.served[:, fr], mov_b.served[:, f], None if mov_b.valid is None else mov_b.valid[:, f],
                        dx0, xc, min_win, p.mad_k)
        bl[f] = b
    fin = np.isfinite(bl)
    if not fin.any():
        return float("nan"), float("nan"), bl
    med = float(np.median(np.abs(bl[fin])))
    return pose_angle_deg(med, ref_b.spacing, L), med, bl


def register_pair(ref, mov, params: PairParams | dict | None = None, *, ceiling: dict | None = None,
                  ceiling_speckle: dict | None = None, quality: bool = True,
                  band_rows: tuple[int, int] = BAND_ROWS_DEFAULT, coarse: dict | None = None) -> PairResult:
    """Design element 3: register a moving member onto the reference — a TWO-STAGE decision. (a) coarse_register
    on the coarse pyramids (the STRUCTURE feature) only has to yield USABLE seeds: the peak (NCC ≥
    coarse_attempt_ncc and not on a search bound, else the better frame-half's peak, 'coarse_split_seed') and, on a
    plateau, its top-K separated maxima (E6), each scored by a cheap FFT-only fine pass — the seed whose frames match
    best is kept ('coarse_reseeded' when it is not the peak; quality['df_reseed'] / coarse['seed_scores']); it flags
    ('coarse_weak', 'coarse_on_bound', 'coarse_multimodal') but never rejects — a mid-volume saccade halves the
    global peak while every frame still registers. (b) fine_register per frame (weak frames re-searched: widened
    window, then the full lateral range, decisive only; E7: the served lines' own tilt a_lines / b_lines per frame,
    the per-lateral search re-centred on the fitted rigid line when the fit is poor, a frame whose rigid fit does
    not hold unmeasured). (c) the FIRST rigid projection — a[f] / b[f] robustified against the step-aware
    Savitzky-Golay trend and filled by interpolation (_robust_fill; the interpolated TILT is clamped on its
    RESIDUAL to the lines, never the lines' own tilt), live segments from the dead frames (live_frames) split at
    measured saccades (split_segments), and — E9 — the lateral shift served PER FRAME: the trusted frames' dx
    (measured, NCC ≥ vote_ncc) put through the same step-aware MAD fill inside each live segment (dx_trend; a
    rejected or unmeasured frame is interpolated), a run of ≥ seg_hold frames measuring nothing trusted dead
    ('dx_unmeasured_run', PairResult.live). (d) the quality pass on the served transform — the full check
    (quality=True: every partnered frame, the pair-level numbers as before) or the CHEAP one on a frame subset
    (quality=False, a transitivity pair) — followed by the ARBITRATION (PairParams' decision tree) on the DECIDABLE
    frames only (E9(b): a structure per-frame ceiling ≥ frame_ceiling_min and ≥ frame_eval_min_frac of the frame's
    reference cells evaluated; the rest are served the fill, listed in quality['undecidable_frames'], never refused):
    every frame whose served (dx, a, b) contradicts its own fine measurement beyond the bars (substep_dx /
    axial_tol_px) is scored under both on its own frame and is served whichever scores better (own: a knot of the
    fill / its measured a / b; served: recorded), an unmeasured frame next to a cut is offered the neighbouring
    shift, and the final transform is scored where it changed. INVARIANT: every decidable live partnered frame is
    served its own measurement, or a value that scores at least as well on that frame (within
    arbitration_margin), or the pair is refused with a REJECT flag naming the frames — never a clamp, never a value
    silently inherited from a neighbour. ok is never decided on the measured-frame count alone. 'no_correspondence'
    is declared HERE: fewer than min_measured_frac of the overlapping frames measured, or matched_frac_0.5 <
    low_match AND coverage < low_coverage, or the band-space STRUCTURE relative match < min_relative_match, or
    nothing judged, or no overlapping frame at all — the evidence stays on the result, `ok` is False. A measured
    segment pinned at the WIDENED fine window ('dx_at_search_edge') or beyond max_dx ('dx_beyond_max'), a kept
    RESIDUAL tilt |b − b_lines| beyond max_tilt_px or |b| beyond max_tilt_abs_px ('tilt_beyond_max'), an isolated
    interior lateral excursion that scores ('dx_residual'), a contiguous run served by neither its measurement nor
    the projection ('low_frame_match' / 'dx_untrusted_run' / 'axial_residual') and a bad minority of frames by
    the per-frame ceiling ('low_frame_match') are refused the same way. E8: a pair whose served lines demand a
    pose beyond pose_max_deg is registered for the record only and flagged 'pose_beyond_frame_rigid' (the match
    verdicts it supersedes are kept under quality['superseded_flags']; it is never 'no_correspondence'). E10: the
    speckle-scale match of the served transform is reported beside the structure match (quality['match_structure']
    / ['match_speckle'], each with its own same-scan ceiling). `ref` / `mov` are BandData (with served lines) or
    MemberData (E4: resampled onto one lateral grid when their header spacings differ — PairResult.lateral_scale /
    lateral_offset_*; dx is then in REFERENCE laterals on the common grid about the reference centre — then
    extract_band(band_rows, coarse_rows/ds/mask from the params)). `coarse` — a coarse_register result of this pair
    to reuse (register_group's pose matrix)."""
    p = _pair_params(params)
    t_all = time.time(); timings: dict = {}
    t0 = time.time()
    ref_b, mov_b, geom, flags = _pair_geometry(ref, mov, p, band_rows)
    timings["bands"] = time.time() - t0
    if ref_b.served is None or mov_b.served is None:
        raise ValueError("register_pair: the bands must carry the served line (BandData.served; extract_band sets it)")
    if ref_b.band.shape[:2] != mov_b.band.shape[:2]:
        raise ValueError(f"register_pair: band shapes differ {ref_b.band.shape} vs {mov_b.band.shape}")
    L, T, F = ref_b.band.shape; Fm = mov_b.n_frames
    hs = max(1.0, (L - 1) / 2.0)
    p = p.resolved(L)                                        # PARTIAL OVERLAP: coarse_max_dx / max_dx None → L − min_overlap_laterals
    bar_l = int(p.min_overlap_laterals); bar_f = float(p.min_overlap_frame_frac)
    # (a) coarse — seeds and flags, never a rejection (except 'no_overlap': no admissible seed at all)
    c = dict(coarse) if coarse is not None else coarse_register(ref_b, mov_b, p)
    timings["coarse"] = c.get("time_s", 0.0)
    if not np.isfinite(c["ncc"]) or c["ncc"] < p.coarse_min_ncc:
        flags.append("coarse_weak")
    if c["on_bound"]:
        flags.append("coarse_on_bound")
    if (np.isfinite(c["sharpness"]) and c["sharpness"] < p.coarse_min_sharp) or c["n_df_peaks"] != 1:
        flags.append("coarse_multimodal")
    if c.get("seed_source", "full") != "full":
        flags.append("coarse_split_seed")
    if c.get("widened_axes"):
        flags.append("coarse_widened")
    n_overlap = sum(1 for f in range(Fm) if 0 <= f + int(c["df0"]) < F)
    if c["n_mov_cells"] == 0 or not np.isfinite(c["ncc"]) or c["ncc"] <= -1.0 or n_overlap == 0 or bool(c.get("no_overlap")):
        # nothing to register: an empty band, no frame pairing, or (PARTIAL OVERLAP) no coarse seed inside the overlap bar —
        # the measured offset stays in the record (coarse['beyond_bar_peak'] / quality['overlap'])
        flags += ["no_correspondence", "no_overlap"]
        timings["total"] = time.time() - t_all
        ov_seed = c.get("overlap") or overlap_of(L, Fm, F, c["dx0"], c["df0"], bar_l, bar_f)
        bb_ = (c.get("beyond_bar_peak") or {})
        ov_rec = {"verdict": "below_bar" if bool(c.get("no_overlap")) else "none", "bar": {"laterals": bar_l, "frame_frac": bar_f, "L": int(L), "max_dx": float(p.max_dx)},
                  "seed": ov_seed, "offset": (dict(bb_["overlap"], ncc=bb_.get("ncc"), source="beyond_bar_peak") if bb_.get("overlap") else ov_seed), "served": None}
        return _empty_pair(ref_b, mov_b, c, flags, p, timings, geom, overlap=ov_rec)
    # E8: the pose the two served lines demand at the coarse seed (ROUND 10, R2: re-read at the FINAL transform below —
    # the seed can be a garbage half-search on a df-bound pair; the fine stage's b_lines at each frame's measured dx is the pose)
    t0 = time.time()
    theta, b_lines_med, b_lines_seed = _pose_of(ref_b, mov_b, c, p)
    theta_seed = float(theta)
    timings["pose"] = time.time() - t0
    # (b) fine — E6: every coarse seed (the peak, its separated maxima on a plateau, and df0 ± 1 when the peak is weak
    # and flat: round 7, R5) is scored by a cheap FFT-only pass (the mean per-frame peak NCC over the partnered
    # frames); the best seed gets the full fine pass
    seeds = [dict(s) for s in (c.get("seeds") or [])]
    if not seeds:
        seeds = [{"dx0": float(c["dx0"]), "dz0": float(c["dz0"]), "df0": int(c["df0"]), "ncc": float(c.get("seed_ncc", c["ncc"])), "source": "full"}]
    weak_flat = bool(np.isfinite(c["ncc"]) and c["ncc"] < float(p.coarse_min_ncc)
                     and np.isfinite(c["sharpness"]) and c["sharpness"] < float(p.coarse_min_sharp))
    if weak_flat and bool(p.coarse_reseed):
        for d in (int(c["df0"]) - 1, int(c["df0"]) + 1):
            if not any(int(s["df0"]) == d and abs(float(s["dx0"]) - float(c["dx0"])) < 1.0 for s in seeds):
                seeds.append({"dx0": float(c["dx0"]), "dz0": float(c["dz0"]), "df0": int(d), "ncc": float("nan"), "source": "df_reseed"})
    seed_scores: list = []
    t0 = time.time()
    if len(seeds) > 1:
        for s in seeds:
            cs_ = dict(c); cs_.update(dx0=float(s["dx0"]), dz0=float(s["dz0"]), df0=int(s["df0"]))
            oo = fine_register(ref_b, mov_b, cs_, p, fft_only=True)
            part = np.array([0 <= f + int(s["df0"]) < F for f in range(Fm)], bool)
            v = np.clip(np.nan_to_num(np.asarray(oo["ncc"], float), nan=0.0), 0.0, None)
            sc = float(np.mean(v[part])) if part.any() else -1.0
            n_meas = int(np.isfinite(oo["dx"][part]).sum()) if part.any() else 0
            seed_scores.append({"dx0": round(float(s["dx0"]), 2), "dz0": round(float(s["dz0"]), 2), "df0": int(s["df0"]),
                                "coarse_ncc": (round(float(s["ncc"]), 4) if np.isfinite(float(s.get("ncc", np.nan))) else None),
                                "source": str(s.get("source", "maximum")), "score": round(sc, 4), "n_peaks": n_meas})
        best_i = max(range(len(seeds)), key=lambda i: (round(seed_scores[i]["score"], 6), seed_scores[i]["n_peaks"], -i))
        c["seed_scores"] = seed_scores
        c["df_reseed"] = {"scores": {f"{seed_scores[i]['df0']}@{seed_scores[i]['dx0']}": seed_scores[i]["score"] for i in range(len(seeds))},
                          "seed_df": int(c["df0"]), "df": int(seeds[best_i]["df0"])}
        if best_i != 0:
            c["dx0"], c["dz0"], c["df0"] = float(seeds[best_i]["dx0"]), float(seeds[best_i]["dz0"]), int(seeds[best_i]["df0"])
            flags.append("coarse_reseeded")
            n_overlap = sum(1 for f in range(Fm) if 0 <= f + int(c["df0"]) < F)
    timings["seed_scoring"] = time.time() - t0
    o = fine_register(ref_b, mov_b, c, p)
    timings["fine"] = o["time_s"]; timings["fine_fft"] = o["t_fft"]; timings["fine_local_ncc"] = o["t_local"]
    timings["fine_speckle"] = o.get("t_speckle", 0.0)
    df = int(c["df0"])
    # R2: the pose at the fine stage — the median |b_lines| over the measured frames (each at its own measured dx); the
    # verdict's pose is re-read at the SERVED dx of the final transform below (an alias-measured frame's lines at its raw dx
    # inflate this median on a periodic texture)
    _blm = np.abs(np.asarray(o["b_lines"], float))[np.asarray(o["measured"], bool)]
    _blm = _blm[np.isfinite(_blm)]
    theta_fine = pose_angle_deg(float(np.median(_blm)), ref_b.spacing, L) if _blm.size >= 5 else float("nan")
    if np.isfinite(theta_fine):
        theta = theta_fine; b_lines_med = float(np.median(_blm))
    pose_flag = bool(np.isfinite(theta) and theta > float(p.pose_max_deg))
    xc_ = (np.arange(L, dtype=float) - (L - 1) / 2.0) / hs
    min_win_ = int(p.min_windows) if p.min_windows else max(16, L // 10)
    if o["widened_frames"]:
        flags.append("fine_widened")
    if n_overlap == 0:
        flags += ["no_correspondence", "no_overlap"]
        timings["total"] = time.time() - t_all
        ov_seed = overlap_of(L, Fm, F, c["dx0"], c["df0"], bar_l, bar_f)
        return _empty_pair(ref_b, mov_b, c, flags, p, timings, dict(geom, pose=theta),
                           overlap={"verdict": "below_bar", "bar": {"laterals": bar_l, "frame_frac": bar_f, "L": int(L), "max_dx": float(p.max_dx)}, "seed": ov_seed, "offset": ov_seed, "served": None})
    # (c) rigid projection — the FIRST projection: the per-frame rigid model (a[f] / b[f] robustified against the
    # step-aware SG trend; a frame rejected in either series is served the interpolation for now), then the lateral
    # shift per frame inside each live segment. The arbitration in (d) starts from here.
    t0 = time.time()
    tol = float(p.axial_tol_px); cap = float(p.max_tilt_px); cap_abs = float(p.max_tilt_abs_px)
    b_lines = np.asarray(o["b_lines"], float); a_lines = np.asarray(o["a_lines"], float)
    bl_fin = np.isfinite(b_lines)
    # the lines' tilt as a ROBUST trend across frames: a served-line defect on a few frames (CS032 v1_3→v1 frames 99-100:
    # b_lines 80 / 97 px against 30 px on every neighbour, the tissue at 36 / 30) is not the tissue's disagreement with
    # the lines
    if bl_fin.sum() >= 5:
        # a wide Savitzky-Golay trend (21 frames) with 3·MAD rejection: a served-line excursion of several frames
        # (CS032 v1_3→v1 frames 83-89: b_lines 64-85 px against 25-32 on either side, the tissue smooth at 27-50) is
        # rejected and bridged; the tissue is then judged against the lines' smooth tilt
        bl_fill, _ = _robust_fill(b_lines, bl_fin, window=21, order=2, k=3.0, floor=2.0, step=None)
    else:
        bl_fill = _fill_kept(b_lines, bl_fin) if bl_fin.any() else np.zeros(Fm)

    # ROUND 10: a measured frame is FAR when its dx lies more than seg_dx_step from the median dx of the SOUND frames within
    # ±seg_hold frames of it (its own excluded; the coarse seed when it has no such neighbour) — the dome-ridge alias test's
    # lateral half
    _sound0 = np.isfinite(o["dx"]) & ~o["at_edge"] & (np.nan_to_num(o["ncc"], nan=-1.0) >= float(p.vote_ncc))
    dx_far = np.zeros(Fm, bool)
    for f in np.flatnonzero(np.isfinite(o["dx"])):
        f = int(f)
        nb_ = [float(o["dx"][g]) for g in range(max(0, f - 2 * int(p.seg_hold)), min(Fm, f + 2 * int(p.seg_hold) + 1)) if g != f and _sound0[g]]
        ref_dx = float(np.median(nb_)) if len(nb_) >= 2 else float(c["dx0"])
        dx_far[f] = abs(float(o["dx"][f]) - ref_dx) > float(p.seg_dx_step)

    b_meas_arr = np.asarray(o["b"], float).copy()   # the fine stage's MEASURED tilt (before any trend replaces it)

    def tilt_residual_own(f: int) -> float:
        """|b_measured − b_lines| at the frame's OWN dx (the lines' tilt there; NaN without a line fit) — the MEASURED tilt, never
        a trend: a trend drawn through alias frames bends toward the lines at their alias dx and hides them."""
        b = float(b_meas_arr[f])
        return abs(b - float(b_lines[f])) if (np.isfinite(b) and np.isfinite(b_lines[f])) else float("nan")

    def admissible_b(f: int) -> bool:
        """ROUND 10: an own tilt candidate is admissible when |b| is within the sanity cap and — on a frame whose dx is FAR from
        its neighbours' (an alias suspect) — its residual to the served lines AT ITS OWN dx is within max_tilt_px (E7's cap as
        the dome-ridge alias test: a lateral alias Δ carries the lines' tilt at Δ, tens of px off the tissue's); an on-trend
        frame's disagreement with the lines is measured and reported, never a verdict (the still-clipped P5 members' lines are
        reconstructions 24-76 px off the tissue on pairs that match at 0.83-0.95)."""
        b = float(o["b"][f])
        if not np.isfinite(b) or abs(b) > cap_abs:
            return False
        if not dx_far[f]:
            return True
        r_ = tilt_residual_own(f)
        if not np.isfinite(r_):
            r_ = abs(b - float(bl_fill[f])) if bl_fin.any() else abs(b)
        return r_ <= cap

    lowprec = np.zeros(Fm, bool)                 # R3: frames served the tilt TREND (their own tilt is imprecise)

    def fill() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        m0 = o["measured"].copy()
        ncc_ = np.nan_to_num(o["ncc"], nan=-1.0)
        # round 9: the fill is seeded by SOUND frames only (peak ≥ vote_ncc); a weak frame's a / b is evidence for the
        # arbitration, never a knot (two junk frames whose chance a / b sat near a real axial step broke the step rule and
        # dragged seven real frames into a dead run). ROUND 10: a frame whose tilt is INADMISSIBLE (its residual to the served
        # lines beyond max_tilt_px) never seeds the fill either — served the fill ('tilt_residual_single'), or a run of them
        # is the 'tilt_beyond_max' verdict; and the tilt fill's rejection statistics come from the PRECISE frames alone (the
        # trend-served frames sit on the trend by construction and would shrink the MAD to nothing)
        adm_ = np.array([bool(m0[f]) and admissible_b(int(f)) for f in range(Fm)], bool)   # an alias frame never seeds
        m_seed = m0 & (ncc_ >= float(p.vote_ncc)) & adm_ & ~o["at_edge"]
        _, ka = _robust_fill(o["a"], m_seed, p.sg_window, p.sg_order, p.mad_k, p.mad_floor, step=p.axial_step_px, hold=p.axial_step_hold)
        seed_b = m_seed & ~lowprec
        if lowprec.any() and int(seed_b.sum()) >= 3:
            _, kb = _robust_fill(o["b"], seed_b, p.sg_window, p.sg_order, p.mad_k, p.mad_floor, step=p.axial_step_px, hold=p.axial_step_hold)
            kb = kb | (m_seed & lowprec)
        else:
            _, kb = _robust_fill(o["b"], m_seed, p.sg_window, p.sg_order, p.mad_k, p.mad_floor, step=p.axial_step_px, hold=p.axial_step_hold)
        return m0, ka & kb, ncc_

    measured0, kept, ncc = fill()
    # a frame the fill rejected while its in-window peak was found is an alias suspect (its rigid fit does not hold
    # with its neighbours'): re-search it like a weak frame — widened window, then the far range; a rescue must be
    # DECISIVE (≥ home peak + rescue_margin), so a frame whose home peak cannot be beaten at all is not re-searched
    # (round-4 refutation 2, 'I +65|0@4' frame 2: an alias at 0.58 replaced by the true 65 at 0.95)
    refit = np.flatnonzero(measured0 & ~kept & ~o["at_edge"] & (ncc + float(p.rescue_margin) <= 1.0))
    if refit.size and bool(p.far_search) and callable(o.get("rescue_frames")) and o["rescue_frames"](refit):
        measured0, kept, ncc = fill()
    # ROUND 10 (R3): the TILT PRECISION gate — a frame whose tilt standard error exceeds tilt_se_max (a half-overlap frame:
    # ~4 independent local-NCC windows on half a lever arm) is served the robust across-frame TREND of the precise frames'
    # tilt (the step-aware Savitzky-Golay trend the fill already uses; over every kept frame when fewer than 5 are precise)
    # and its a re-fitted at the overlap's centre x̄ (a + b·x̄ is the precisely measured depth there): a transform-parameter
    # smoothing across frames, never an image smoothing; the measured values stay in quality['tilt_low_precision']
    b_se = np.asarray(o["b_se"], float); x_mean = np.nan_to_num(np.asarray(o["x_mean"], float), nan=0.0)
    a_meas = o["a"].copy(); b_meas = o["b"].copy()
    lowprec |= measured0 & np.isfinite(b_se) & (b_se > float(p.tilt_se_max))
    tilt_lowprec: dict = {"frames": [], "b_measured": [], "b_served": [], "b_se": [], "a_measured": []}
    if lowprec.any():
        # the trend is read from EVERY kept frame (the first fill's MAD-consistent set, the imprecise ones included): the
        # Savitzky-Golay window averages ~11 frames — the imprecise tilts' noise drops by ~3× while a real 3-px-per-frame tilt
        # ramp (P5_OS v1_3→v1_4: the lines' tilt runs −31 → +84 px over 35 frames) is followed; a trend through a handful of
        # scattered precise frames flattened that ramp to 0 and made 13 frames a dead run
        precise = kept & ~lowprec & (np.nan_to_num(o["ncc"], nan=-1.0) >= float(p.vote_ncc))
        src_b = (kept | (measured0 & lowprec & _sound0)) & ~dx_far          # never through an alias suspect
        b_tr = _sg_trend(np.where(measured0, o["b"], np.nan), src_b, p.sg_window, p.sg_order, step=p.axial_step_px, hold=p.axial_step_hold)
        for f in np.flatnonzero(lowprec):
            f = int(f)
            o["b"][f] = float(b_tr[f]); o["a"][f] = float(a_meas[f] + (b_meas[f] - b_tr[f]) * x_mean[f])
            tilt_lowprec["frames"].append(f); tilt_lowprec["b_measured"].append(round(float(b_meas[f]), 2))
            tilt_lowprec["b_served"].append(round(float(b_tr[f]), 2)); tilt_lowprec["b_se"].append(round(float(b_se[f]), 2))
            tilt_lowprec["a_measured"].append(round(float(a_meas[f]), 2))
        tilt_lowprec["source"] = "all_kept_frames"; tilt_lowprec["n_precise"] = int(precise.sum())
        measured0, kept, ncc = fill()
    kept0 = kept.copy()
    a_raw = np.where(measured0, o["a"], np.nan); b_raw = np.where(measured0, o["b"], np.nan)
    b_resid = np.where(measured0 & bl_fin, np.minimum(np.abs(b_raw - bl_fill), np.abs(b_raw - b_lines)), np.nan)   # the TISSUE's disagreement with the lines

    def _jitter(v: np.ndarray, k_: np.ndarray) -> float:
        """Median |Δ| of a per-frame series between adjacent kept frames (the pair's own frame-to-frame noise)."""
        ki_ = np.flatnonzero(k_ & np.isfinite(v))
        if ki_.size < 6:
            return 0.0
        d_ = np.abs(np.diff(v[ki_]))[np.diff(ki_) == 1]
        return float(np.median(d_)) if d_.size >= 5 else 0.0
    # the witness and ridge tolerances scale with the pair's own frame-to-frame a / b jitter (a still-clipped P5 member
    # jitters 5-8 px between adjacent frames; the synthetic bands < 1 px, so the floors stand there)
    jit_a = _jitter(o["a"], kept); jit_b = _jitter(o["b"], kept)
    w_tol_a = max(tol, 2.0 * jit_a); w_tol_b = max(tol, 2.0 * jit_b)
    ridge_tol = max(float(p.ridge_axial_tol), 2.0 * jit_a)
    a_pin = np.full(Fm, np.nan); b_pin = np.full(Fm, np.nan)   # a fill-rejected frame PINNED to a previous interpolation (R2)

    def fills(k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """a / b served: kept frames keep their measurement, pinned frames their pinned value (a previous interpolation
        that won the arbitration — a knot for the interpolation like a kept frame), the rest is interpolated (ends held);
        only the interpolated tilt is capped — on its RESIDUAL to the served lines' tilt (E7) and by the sanity cap — a
        kept tilt beyond the cap is a verdict ('tilt_beyond_max'), never clamped."""
        pin = np.isfinite(a_pin) & ~k
        va = np.where(pin, a_pin, o["a"]); vb = np.where(pin, b_pin, o["b"])
        af = _fill_kept(va, k | pin); bf = _fill_kept(vb, k | pin)
        b_int = np.clip(np.clip(bf - bl_fill, -cap, cap) + bl_fill, -cap_abs, cap_abs)
        return af, np.where(k | pin, bf, b_int)

    a_fill, b_fill = fills(kept)
    finite = np.isfinite(o["dx"])
    pinned = finite & o["at_edge"]                               # a bound, not a measurement: splits, never votes, never serves
    # ROUND 10: a FAR frame (dx_far: > seg_dx_step from its sound neighbours') VOTES only inside a sustained mode — an agreeing run
    # of ≥ dx_step_hold frames or a coherent ramp of ≥ seg_hold (the mode guard's rule, applied to the voters too): an isolated far
    # frame that the MAD fill happened to keep (a junk frame at noise 200) broke a dead run and became a knot
    _sound_dx = np.where(_sound0, o["dx"], np.nan)
    sustained_mode = np.zeros(Fm, bool)
    for r0_, r1_ in _agreeing_runs(_sound_dx, float(p.substep_agree)):
        if r1_ - r0_ >= int(p.dx_step_hold):
            sustained_mode[r0_:r1_] = True
    for r0_, r1_ in _coherent_runs(_sound_dx, float(p.dx_step)):
        if r1_ - r0_ >= max(int(p.dx_step_hold), int(p.seg_hold)):
            sustained_mode[r0_:r1_] = True
    trusted = finite & kept & ~pinned & (ncc >= float(p.vote_ncc)) & (~dx_far | sustained_mode)   # votes for its segment's shift
    sound = finite & ~pinned & (ncc >= float(p.vote_ncc))       # a dx measurement good enough to be served alone
    weak = finite & ~pinned & ~sound                             # below vote_ncc: evidence for the arbitration only
    partnered = np.array([0 <= f + df < F for f in range(Fm)], bool)
    lateral_live = live_frames(mov_b, p.live_frac)
    agree = float(p.substep_agree); bar = float(p.substep_dx)
    near = 1.0            # laterals / px below which two values are the same for the arbitration (nothing to score)
    min_seg = max(1, int(p.min_segment_frames)); hold = max(1, int(p.seg_hold))
    dx_vote = np.where(trusted, o["dx"], np.nan)
    plateau_fixed: dict = {}    # {segment start: value} — a plateau-cut cluster that keeps the served constant
    use_trend = bool(p.dx_trend)
    P0_dx = np.full(Fm, float(c["dx0"]))                          # the seed stands in for the served dx at this point
    tv = np.flatnonzero(trusted)
    if tv.size:
        P0_dx = np.interp(np.arange(Fm), tv, o["dx"][tv])
    # E9(d): a run of ≥ 2 WEAK frames (peak < vote_ncc) whose a / b the fill rejected — and whose own dx does not contradict
    # the trusted frames' trend (a coherent weak lateral excursion stays judged: 'dx_untrusted_run') — is DEAD — interpolated by design,
    # never arbitrated, never judged (P5 v1_3→v1_2: a 5-frame dip at NCC 0.42-0.52 with a_raw 20-30 px off the trend)
    weak_rej = measured0 & ~kept & (ncc < float(p.vote_ncc)) & ~(finite & (np.abs(o["dx"] - P0_dx) > bar))
    dead_runs = [(int(f0), int(f1)) for f0, f1 in _runs(weak_rej) if f1 - f0 >= 2]
    dead_weak = np.zeros(Fm, bool)
    for f0, f1 in dead_runs:
        dead_weak[f0:f1] = True

    def segment_fill(f0: int, f1: int, forced: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
        """E9: the lateral shift served per frame inside [f0, f1): the trusted frames' own dx and the forced frames'
        pinned values put through the step-aware MAD fill (a forced frame is a knot, never rejected; the rejection threshold
        never below substep_dx — inside the contradiction bar the per-frame model serves the measurement); returns
        (applied (f1−f0,), kept (f1−f0,), the sustained-step cuts inside the segment in frame indices)."""
        n = int(f1 - f0)
        idx = np.arange(f0, f1)
        pin = np.isfinite(forced[f0:f1])
        v = np.where(pin, forced[f0:f1], np.where(trusted[f0:f1], o["dx"][f0:f1], np.nan))
        # the mode guard: a far frame (> seg_dx_step from the coarse seed) is a knot only inside an agreeing run of
        # ≥ dx_step_hold consecutive trusted frames (a sustained lateral mode); an isolated far frame is arbitrated
        far = np.isfinite(v) & ~pin & (np.abs(v - float(c["dx0"])) > float(p.seg_dx_step))
        if far.any():
            # a sustained mode: an AGREEING run (spread ≤ substep_agree) of ≥ dx_step_hold frames — or, ROUND 10 (R6), a COHERENT
            # ramp (consecutive frames within dx_step of each other) of ≥ seg_hold frames: a real lateral wave ramps 2.5-4
            # laterals per frame over tens of frames and is one mode; two or three period aliases within 6 laterals of each
            # other are not (the spread rule split the wave's steep parts into 1-2-frame runs and dropped them from the knots)
            sustained = np.zeros(n, bool)
            for r0, r1 in _agreeing_runs(np.where(np.isfinite(v) & ~pin, v, np.nan), agree):
                if r1 - r0 >= int(p.dx_step_hold):
                    sustained[r0:r1] = True
            for r0, r1 in _coherent_runs(np.where(np.isfinite(v), v, np.nan), float(p.dx_step)):
                if r1 - r0 >= max(int(p.dx_step_hold), int(p.seg_hold)):
                    sustained[r0:r1] = True
            v = np.where(far & ~sustained, np.nan, v)
        ok = np.isfinite(v)
        if not ok.any():
            return np.full(n, np.nan), np.zeros(n, bool), []
        _, k = _robust_fill(v, ok, p.sg_window, p.sg_order, p.mad_k, max(float(p.mad_floor), float(p.substep_dx)),
                            step=float(p.dx_step), hold=int(p.dx_step_hold), cap=float(p.seg_dx_step))
        k = (k | pin) & ok
        k &= ~step_dropped[f0:f1]                              # a short step the scorer did not corroborate: not a knot
        applied = _fill_kept(v, k)
        ki = np.flatnonzero(k)
        cuts: list = []
        if ki.size >= 2:
            runs_k = _step_runs(v[ki], float(p.dx_step), int(p.dx_step_hold))
            for j, (i0, i1) in enumerate(runs_k):
                if j > 0:
                    cuts.append(int(idx[ki[i0]]))
                    # the gap between the two runs is a STEP, not a ramp: each gap frame takes the nearer run's end value (a
                    # ramp across a saccade served every gap frame half a period off and made any alias 'decisive')
                    g_lo, g_hi = int(ki[i0 - 1]), int(ki[i0])
                    for g in range(g_lo + 1, g_hi):
                        applied[g] = v[g_lo] if (g - g_lo) <= (g_hi - g) else v[g_hi]
            # every kept step run that is SHORT (< seg_hold frames) or LARGE (≥ seg_dx_step from a neighbouring run) is a
            # candidate for the scorer's corroboration against its neighbours' values (project() records, the loop scores)
            for j, (i0, i1) in enumerate(runs_k):
                if len(runs_k) < 2 or pin[ki[i0:i1]].any():
                    continue
                v_run = float(np.median(v[ki[i0:i1]]))
                v_lo = float(v[ki[i0 - 1]]) if j > 0 else None
                v_hi = float(v[ki[i1]]) if j < len(runs_k) - 1 else None
                jumps = [abs(v_run - x) for x in (v_lo, v_hi) if x is not None]
                if (i1 - i0 < hold) or (jumps and max(jumps) >= float(p.seg_dx_step)):
                    fr_s = [int(idx[ki[t]]) for t in range(i0, i1)]
                    short_steps.append((fr_s, v_lo if v_lo is not None else v_hi, v_hi if v_hi is not None else v_lo))
        return applied, k, cuts

    def project(forced: np.ndarray, extra_cuts: list) -> dict:
        """Live segments + the shift served on each frame, plus the per-frame OVERRIDES. `forced` (Fm,) pins a frame's
        served shift (NaN = free): its own measurement that won the arbitration, or a neighbouring value it was reassigned
        to. A trusted, pinned or forced frame is live whatever its lateral fraction; a run of ≥ seg_hold live partnered
        frames with none of those is DEAD ('dx_unmeasured_run': interpolated like a crop band). The BASE segmentation is the
        first projection's: split_segments' step / hold rule over the trusted (and pinned) frames plus the accepted
        sub-step plateau cuts (`extra_cuts`). Every agreeing run (substep_agree) of forced frames of at least
        min_segment_frames frames is carved into its own SEGMENT; a carved run whose shift equals an adjacent remainder's
        within 1 lateral merges into it. E9 (dx_trend): inside each segment the served dx is the step-aware MAD fill of
        the trusted frames' dx with the forced frames as knots (segment_fill) — dx_segments lists the sustained-step runs
        with their medians; otherwise (dx_trend False) one shift per segment = the median of its trusted frames' dx, the
        round-8 model. A SHORTER forced run is not a segment: its frames are served their own value inside their segment
        (dx_applied only — 'override_runs')."""
        owned = np.isfinite(forced)
        live = lateral_live | trusted | pinned | owned
        runs_dead = [(f0, f1) for f0, f1 in _runs(live & partnered & ~(trusted | pinned | owned)) if f1 - f0 >= int(p.seg_hold)]
        for f0, f1 in runs_dead:
            live[f0:f1] = False
        base = split_segments(np.where(trusted | pinned, o["dx"], np.nan), live, step=p.seg_dx_step, hold=p.seg_hold)
        if extra_cuts:
            base = _add_cuts(base, extra_cuts)
        base_vals, _ = _segment_dx(dx_vote, base, p.max_dx, c["dx0"])
        base_vals = [(f0, f1, plateau_fixed.get(int(f0), v)) for f0, f1, v in base_vals]
        bounds = {int(b) for seg in base for b in seg}
        runs_all = _agreeing_runs(np.where(live, forced, np.nan), agree)
        long_runs = [(r0, r1) for r0, r1 in runs_all if r1 - r0 >= min_seg]
        short_runs = [(r0, r1) for r0, r1 in runs_all if r1 - r0 < min_seg]
        forced_long = np.full(Fm, np.nan)
        for r0, r1 in long_runs:
            forced_long[r0:r1] = forced[r0:r1]
        segs, carved = _carve_segments(base, forced_long, agree, live)
        carved_set = {(r0, r1) for r0, r1, _ in carved}

        def base_value(f: int) -> float:
            for b0, b1, v in base_vals:
                if b0 <= f < b1:
                    return float(v)
            return float(np.clip(c["dx0"], -float(p.max_dx), float(p.max_dx)))
        vals = [float(np.median(forced[f0:f1])) if (f0, f1) in carved_set else base_value(f0) for f0, f1 in segs]
        merged: set = set()
        drop: set = set()
        for i, (f0, f1) in enumerate(segs):
            if (f0, f1) not in carved_set:
                continue
            best = None
            for j in (i - 1, i + 1):
                if not (0 <= j < len(segs)):
                    continue
                g0, g1 = segs[j]
                adjacent = (g1 == f0) if j == i - 1 else (g0 == f1)
                if (g0, g1) in carved_set or not adjacent:
                    continue
                d = abs(vals[i] - vals[j])
                if d <= near and (best is None or d < best[0]):
                    best = (d, f0 if j == i - 1 else f1, j)
            if best is not None:
                drop.add(best[1]); merged.add((f0, f1)); vals[i] = vals[best[2]]   # the neighbour's shift
        if drop:
            # rebuild: merged runs disappear into their neighbour (which keeps its value and grows)
            out: list = []
            for (f0, f1), v in zip(segs, vals):
                if out and out[-1][1] == f0 and (f0 in drop):
                    out[-1] = (out[-1][0], f1, out[-1][2])
                else:
                    out.append((f0, f1, v))
            dx_segments = [(int(f0), int(f1), float(v)) for f0, f1, v in out]
        else:
            dx_segments = [(int(f0), int(f1), float(v)) for (f0, f1), v in zip(segs, vals)]
        applied = np.full(Fm, np.nan)
        dx_kept = np.zeros(Fm, bool)
        short_steps.clear()
        if use_trend:
            # E9: per-frame serving inside each segment; segments with nothing kept take the segment value
            seg_out: list = []
            for f0, f1, v in dx_segments:
                ap, k_, cuts = segment_fill(f0, f1, forced)
                if np.isfinite(ap).any():
                    applied[f0:f1] = ap; dx_kept[f0:f1] = k_
                    pieces = sorted({int(f0), int(f1)} | {int(cc) for cc in cuts if f0 < cc < f1})
                    for i in range(len(pieces) - 1):
                        a0, a1 = pieces[i], pieces[i + 1]
                        kk = np.flatnonzero(k_[a0 - f0:a1 - f0]) + a0
                        seg_out.append((int(a0), int(a1), float(np.median(ap[kk - f0] if kk.size else ap[a0 - f0:a1 - f0]))))
                else:
                    applied[f0:f1] = v
                    seg_out.append((int(f0), int(f1), float(v)))
            dx_segments = seg_out
        else:
            for f0, f1, v in dx_segments:
                applied[f0:f1] = v
        if np.isfinite(applied).any():                          # dead frames: interpolated between segments
            idx = np.flatnonzero(np.isfinite(applied))
            applied = np.interp(np.arange(Fm), idx, applied[idx])
        else:
            applied = np.full(Fm, float(np.clip(c["dx0"], -float(p.max_dx), float(p.max_dx))))
        overrides: list = []
        for r0, r1 in short_runs:                               # served per frame inside their segment
            applied[r0:r1] = forced[r0:r1]
            overrides.append((int(r0), int(r1), float(np.median(forced[r0:r1])), bool(r0 in bounds or r1 in bounds)))
        singles = [(r0, r1) for r0, r1, at_end in carved if r1 - r0 == 1 and not at_end and (r0, r1) not in merged]
        singles += [(r0, r1) for r0, r1, _, at_end in overrides if r1 - r0 == 1 and not at_end]
        end_singles = [(r0, r1) for r0, r1, at_end in carved if r1 - r0 == 1 and at_end and (r0, r1) not in merged]
        end_singles += [(r0, r1) for r0, r1, _, at_end in overrides if r1 - r0 == 1 and at_end]
        # the HELD (saccade / plateau-level) view: winner runs shorter than seg_hold folded into their segment
        held_forced = np.full(Fm, np.nan)
        for r0, r1 in runs_all:
            if r1 - r0 >= hold:
                held_forced[r0:r1] = forced[r0:r1]
        segs_h, _ = _carve_segments(base, held_forced, agree, live)
        held = [(int(f0), int(f1), float(np.median(held_forced[f0:f1])) if np.isfinite(held_forced[f0:f1]).all() else base_value(f0))
                for f0, f1 in segs_h]
        return {"live": live, "unmeasured_runs": runs_dead, "base": base, "segments": [sg[:2] for sg in dx_segments],
                "dx_segments": dx_segments, "dx_segments_held": held, "dx_applied": applied, "carved": carved, "merged": merged,
                "override_runs": overrides, "interior_singles": sorted(singles), "end_singles": sorted(end_singles),
                "dx_kept": dx_kept}

    forced = np.full(Fm, np.nan)
    plateau_cuts: list = []
    step_dropped = np.zeros(Fm, bool)           # short kept steps the scorer did not corroborate (round 9)
    short_steps: list = []                       # [(frames, value before, value after)] recorded by project()
    P = project(forced, plateau_cuts)
    # E9 RIDGE GUARD (round 9): on a dome the structure feature is degenerate along the ridge dx ↔ (a, b) — a lateral shift
    # Δ with the matching tilt / axial change reproduces the same smooth image, so a short run of frames can measure a
    # coherent ALIAS point of the ridge (CS032 v1_2→v1 frames 5-7: dx −139 / a −300 against the trend 0 / −275; P5 v1_2→v1_3
    # frame 61: dx −170 / a 177 against −114 / 116). A real lateral saccade does not move the eye AXIALLY by 0.1-1 mm
    # between adjacent frames: a base segment shorter than seg_hold whose frames jump in BOTH dx (> seg_dx_step) and a
    # (> ridge_axial_tol) against the adjacent long segment's served values is the alias — its frames stop voting and are
    # interpolated (quality['ridge_runs']); the same test guards every short own win in the arbitration below.
    ridge_alias = np.zeros(Fm, bool); ridge_runs: list = []
    # ROUND 10: the tilt-residual alias test — a FAR frame whose tissue tilt disagrees with the lines at its own dx beyond the cap
    alias_far = measured0 & partnered & dx_far & np.array([not admissible_b(int(f)) for f in range(Fm)], bool)
    if alias_far.any():
        ridge_alias |= alias_far
        ridge_runs += [(int(f0), int(f1)) for f0, f1 in _runs(alias_far)]
    base0 = list(P["base"])
    exc_max = float(p.excursion_max_dx)
    for i, (f0, f1) in enumerate(base0):
        if f1 - f0 >= hold or not any(trusted[f] for f in range(f0, f1)):
            continue
        tf = [f for f in range(f0, f1) if trusted[f]]
        dx_s = float(np.median(o["dx"][tf])); a_s = float(np.median(o["a"][tf]))
        nb = [base0[j] for j in (i - 1, i + 1) if 0 <= j < len(base0) and base0[j][1] - base0[j][0] >= hold
              and (base0[j][1] == f0 or base0[j][0] == f1)]
        alias = False
        if nb:
            alias = True
            for g0, g1 in nb:
                gb = g1 - 1 if g1 == f0 else g0                    # the neighbour's boundary frame
                if abs(dx_s - float(P["dx_applied"][gb])) <= float(p.seg_dx_step) or abs(a_s - float(a_fill[gb])) <= ridge_tol:
                    alias = False
        # ROUND 10: an EXCURSION-AND-RETURN — a run shorter than seg_hold whose shift lies more than excursion_max_dx (0.3 mm)
        # from the served shift on BOTH sides while the two sides agree with each other within excursion_max_dx — is the
        # dome-ridge alias whatever its a: on a dome the alias measurement is self-consistent (its tilt IS the lines' tilt at the
        # alias shift, its a the ridge's), only the lateral discontinuity betrays it (CS032 v1_2→v1 frames 5-8 at dx −140
        # between 14 and −16: a 1.2-mm saccade and back within 160 ms is no eye movement)
        lo_f, hi_f = int(f0) - 1, int(f1)
        live_lo = 0 <= lo_f and partnered[lo_f] and bool(P["live"][lo_f]); live_hi = hi_f < Fm and partnered[hi_f] and bool(P["live"][hi_f])
        if not alias:
            if live_lo and live_hi:
                d_lo, d_hi = float(P["dx_applied"][lo_f]), float(P["dx_applied"][hi_f])
                if abs(dx_s - d_lo) > exc_max and abs(dx_s - d_hi) > exc_max and abs(d_lo - d_hi) <= exc_max:
                    alias = True
            elif (live_lo or live_hi) and f1 - f0 <= 2:
                # one live side only (the other dead / unpartnered): a 1-2-frame run farther than excursion_max_dx from its only
                # live neighbour is the alias too (CS032 v1_2→v1 frames 5-6 at dx −140 as the FIRST live frames, the next 16 at 0)
                d_nb = float(P["dx_applied"][lo_f if live_lo else hi_f])
                if abs(dx_s - d_nb) > exc_max:
                    alias = True
        if alias:
            ridge_alias[f0:f1] = True; ridge_runs.append((int(f0), int(f1)))
    ridge_runs = sorted(set(ridge_runs))
    if ridge_alias.any():
        trusted &= ~ridge_alias; sound &= ~ridge_alias; kept &= ~ridge_alias; kept0 &= ~ridge_alias
        dx_vote = np.where(trusted, o["dx"], np.nan)
        a_fill, b_fill = fills(kept)
        P = project(forced, plateau_cuts)
    timings["project"] = time.time() - t0
    res = PairResult(ref_cid=ref_b.cid, mov_cid=mov_b.cid, df=df, dz0=float(c["dz0"]), dx_segments=P["dx_segments"],
                     a=a_fill, b=b_fill, dx_per_frame=o["dx"], dx_applied=P["dx_applied"], dz_per_frame=o["dz"],
                     a_raw=a_raw, b_raw=b_raw, measured=kept, per_frame_ncc=o["ncc"], per_frame_local_ncc=o["local_ncc"],
                     n_windows=o["n_windows"], fit_rms=np.where(measured0 | o["rms_rejected"], o["rms"], np.nan), dz_band=o["dz_band"],
                     dx_at_edge=o["at_edge"].copy(), dx_trusted=trusted.copy(), dx_search=o["search"].copy(),
                     live=P["live"].copy(),
                     ncc_coarse=float(c["ncc"]), peak_sharpness=float(c["sharpness"]), coarse=c,
                     matched_frac_0_5=float("nan"), matched_frac_0_3=float("nan"), coverage=float("nan"),
                     ncc_mean=float("nan"), ceiling=dict(ceiling or {}), relative_match=float("nan"), quality={},
                     flags=flags, timings=timings, params=p.as_dict(), shape=(L, T, F),
                     lateral_scale=float(geom.get("scale", 1.0)), lateral_offset_mov=int(geom.get("offset_mov", 0)),
                     lateral_offset_ref=int(geom.get("offset_ref", 0)), pose_angle_deg=float(theta),
                     b_lines=b_lines.copy(), a_lines=a_lines.copy(),
                     per_frame_speckle_ncc=np.asarray(o["speckle_col"], float).copy(), speckle_refined=o["speckle_refined"].copy())
    # (d) quality + ARBITRATION (PairParams' decision tree). Pass 1 scores the SERVED transform — the full check
    # (quality=True: every partnered frame, the pair-level numbers as before) or the CHEAP one (quality=False:
    # ~quality_subset_frames evenly spaced frames PLUS every frame whose own dx differs from the served shift by more
    # than substep_agree — the engine's own 'same value' tolerance — PLUS every contradiction). Sub-step plateaus are
    # cut first; every contradiction is then arbitrated on its own frame (pinned frames included); unmeasured frames
    # next to a transition are offered the neighbouring value; the final transform is scored where it changed
    # (memoised) and its pair-level numbers assembled per frame. ok is NEVER decided on the measured-frame count
    # alone (refutation B).
    t0 = time.time()
    C_dx = partnered & finite & (np.abs(o["dx"] - P["dx_applied"]) > bar)
    C_ab = partnered & measured0 & ~kept & ((np.abs(a_fill - o["a"]) > tol) | (np.abs(b_fill - o["b"]) > tol))
    C = C_dx | C_ab
    win = tuple(p.local_win)
    scorer: _FrameScorer | None = None; q1: dict | None = None; source = "none"
    pool = np.zeros(Fm, bool)
    if quality:
        q1, _ = pair_quality(ref_b, mov_b, res, ceiling=ceiling, window=win, feature="struct")
        scorer = _FrameScorer(ref_b, mov_b, res, q1["ceiling"], win, feature="struct")
        scorer.absorb(q1, np.flatnonzero(partnered), res.dx_applied, res.a, res.b)
        pool = partnered.copy(); source = "full"
        timings["quality"] = q1["time_s"]
    elif int(p.quality_subset_frames) > 0:
        fr_all = [f for f in range(Fm) if partnered[f]]
        step = max(1, len(fr_all) // int(p.quality_subset_frames))
        off = partnered & finite & (np.abs(o["dx"] - P["dx_applied"]) > agree)
        unmeasured = partnered & ~finite & P["live"]                    # no fine measurement, not in a dead run (R4)
        sub = sorted(set(fr_all[step // 2::step]) | {int(f) for f in np.flatnonzero(C | off | unmeasured)})
        scorer = _FrameScorer(ref_b, mov_b, res, ceiling, win, feature="struct")
        scorer.score(sub, res.dx_applied, res.a, res.b)
        pool[sub] = True; source = "subset"
        timings["quality_subset"] = scorer.time_s
    fm = float(p.frame_match_frac); margin = float(p.arbitration_margin)
    c_min = float(p.frame_ceiling_min); e_min = float(p.frame_eval_min_frac)

    # E9(b) DECIDABILITY: a verdict is taken only on a frame whose STRUCTURE per-frame ceiling reaches frame_ceiling_min
    # and whose evaluated cells reach frame_eval_min_frac of its reference cells (read from its record under the served
    # value); the rest are served the fill and reported — never refused (CS001 v1→v2 frame 90: ceiling 0.09-0.11)
    def decidable_of(dx_arr: np.ndarray, a_arr: np.ndarray, b_arr: np.ndarray, eval_rule: bool = True) -> np.ndarray:
        """`eval_rule` False: the CEILING criterion alone — the arbitration's eligibility (a frame whose SERVED value leaves
        too few evaluated cells is not undecidable, its served value is what does not overlap: a 115-lateral wrong served
        shift on a last frame was never arbitrated against its own 65 at ratio 1.28); the full rule is the quorum's."""
        out = np.ones(Fm, bool)
        if scorer is None:
            return out
        for f in np.flatnonzero(partnered):
            fr = int(f) + df
            rec = scorer.cache.get(scorer.key(int(f), dx_arr[f], a_arr[f], b_arr[f]))
            if rec is None:
                continue
            ce = float(rec.get("ceiling", np.nan)); ne = float(rec.get("n_eval", 0.0))
            nr = float(scorer.n_ref_frame[fr]) if 0 <= fr < scorer.n_ref_frame.size else 0.0
            if (np.isfinite(ce) and ce < c_min) or (eval_rule and nr > 0 and ne < e_min * nr):
                out[f] = False
        return out

    decidable = decidable_of(res.dx_applied, res.a, res.b)
    dec_arb = decidable_of(res.dx_applied, res.a, res.b, eval_rule=False)

    def ratio_of(rec) -> float:
        return float(rec["ratio"]) if rec is not None and np.isfinite(rec.get("ratio", np.nan)) else float("nan")

    # E10 — the SPECKLE scorer, the arbitration's SECOND WITNESS: the same per-frame records on the speckle feature (window
    # local_win_speckle, the reference's own speckle ceiling), scored only for the frames the arbitration asks about; silent
    # on a frame whose speckle ceiling is below speckle_ceiling_min. A single or weak own win is served only when the speckle
    # witness corroborates it where it can speak (a junk frame's chance win, a dome-ridge alias of the smooth structure
    # feature, a periodic alias: not corroborated); where the pair's speckle does not correlate (a rotated pose) it is silent
    # and the structure verdict stands as before.
    scorer_sp: _FrameScorer | None = None
    if scorer is not None and bool(p.speckle_witness) and mov_b.feat_speckle is not None and ref_b.feat_speckle is not None:
        scorer_sp = _FrameScorer(ref_b, mov_b, res, ceiling_speckle, tuple(p.local_win_speckle), feature="speckle")
    sp_counts = {"own": 0, "served": 0, "tie": 0, "silent": 0}

    def sp_prefers(f: int, dx_c, a_c, b_c, dx_s, a_s, b_s) -> tuple[int, bool]:
        """(vote, usable): +1 the speckle witness prefers the candidate (dx_c, a_c, b_c)[f] over the served (dx_s, a_s,
        b_s)[f] (lexicographic _better on the speckle records, the candidate reaching the gate), −1 the served value, 0 a tie;
        usable False = silent (no speckle scorer, or the frame's speckle ceiling under speckle_ceiling_min)."""
        if scorer_sp is None:
            return 0, False
        f = int(f)
        rc = scorer_sp.score([f], np.asarray(dx_c, float), np.asarray(a_c, float), np.asarray(b_c, float))[f]
        rs = scorer_sp.score([f], np.asarray(dx_s, float), np.asarray(a_s, float), np.asarray(b_s, float))[f]
        ce = float(rc.get("ceiling", np.nan))
        if not (np.isfinite(ce) and ce >= float(p.speckle_ceiling_min)):
            sp_counts["silent"] += 1
            return 0, False
        rcr, rsr = ratio_of(rc), ratio_of(rs)
        if not ((np.isfinite(rcr) and rcr >= fm) or (np.isfinite(rsr) and rsr >= fm)):
            # BLIND: neither value reaches the gate at the speckle scale on this frame (a noise-burst frame whose structure
            # still measures) — the witness is silent, the structure decides as before
            sp_counts["silent"] += 1
            return 0, False
        v, _ = _better(rc, rs, margin)
        if v > 0 and np.isfinite(ratio_of(rc)) and ratio_of(rc) >= fm:
            sp_counts["own"] += 1
            return 1, True
        if v < 0:
            sp_counts["served"] += 1
            return -1, True
        sp_counts["tie"] += 1
        return 0, True

    def sp_backs(f: int, dx_c, a_c, b_c) -> bool:
        """The speckle witness backs the candidate on frame f: it prefers it, or it is silent there."""
        vote, usable = sp_prefers(f, dx_c, a_c, b_c, served_dx_now[0], a_fill, b_fill)
        return (not usable) or vote > 0

    served_dx_now = [res.dx_applied]                     # the served dx the witness compares against (updated per round)

    def decide(s_rec, cands: list) -> tuple[str, str | None, dict | None]:
        """The verdict of one frame: `s_rec` its record under the SERVED value, `cands` [(name, record)] its candidates.
        A candidate that reaches the gate (ratio ≥ frame_match_frac) and beats the served record — _better: lexicographic
        ratio → frac_0.7 → ncc_mean, the margin on the deciding statistic (R1: the gate ratio saturates) — wins: the best
        such candidate by the same comparison → ('own', name, record); every candidate and the served value below the
        gate → ('neither', None, the best candidate); else ('served', None, the best candidate)."""
        s_ = ratio_of(s_rec)
        recs = [(n, r) for n, r in cands if r is not None]
        passing = [(n, r) for n, r in recs if np.isfinite(ratio_of(r)) and ratio_of(r) >= fm]
        best = None
        for n, r in passing:
            if best is None or _better(r, best[1], margin)[0] > 0:
                best = (n, r)
        if best is not None and ((not np.isfinite(s_)) or _better(best[1], s_rec, margin)[0] > 0):
            return "own", best[0], best[1]
        top = best[1] if best is not None else (max(recs, key=lambda nr: (bool(np.isfinite(ratio_of(nr[1]))), ratio_of(nr[1])))[1] if recs else None)
        if best is None and ((not np.isfinite(s_)) or s_ < fm):
            return "neither", None, top
        return "served", None, top

    def verdict(s_rec, c_rec) -> str:
        return decide(s_rec, [("own", c_rec)])[0]

    def ratio_gain(frames, dx_new, a_new, b_new, dx_old, a_old, b_old) -> float:
        """Cell-weighted mean of (ratio under the new values − ratio under the old) over `frames` (NaN if none)."""
        if not frames:
            return float("nan")
        s_rec = scorer.score(frames, dx_old, a_old, b_old); n_rec = scorer.score(frames, dx_new, a_new, b_new)
        num = den = 0.0
        for f in frames:
            rs, rn, n = s_rec[f]["ratio"], n_rec[f]["ratio"], n_rec[f]["n_eval"]
            if np.isfinite(rs) and np.isfinite(rn) and n > 0:
                num += n * (rn - rs); den += n
        return num / den if den > 0 else float("nan")

    t1 = time.time()
    plateau_rec: list = []
    # SUB-STEP PLATEAU cuts (decision tree step 2b) — the round-8 piecewise-constant model's rule: inside a base
    # segment, two plateaus of the trusted frames' own dx a step ≥ substep_agree apart are cut when a cluster's
    # median scores better than the served constant on that cluster's window frames. With dx served per frame (E9,
    # dx_trend) a plateau step is already served exactly, so the rule is skipped.
    if scorer is not None and not use_trend:
        cands0 = _plateau_cuts(dx_vote, P["base"], agree, hold, agree)
        cands = list(cands0); placement: dict = {}
        if cands:
            base_c = _add_cuts(P["base"], cands)
            vals_c, _ = _segment_dx(dx_vote, base_c, p.max_dx, c["dx0"])
            alt0 = P["dx_applied"].copy()
            for f0, f1, v in vals_c:
                alt0[f0:f1] = v
            refined: list = []
            for kk in cands:
                vL, vR = float(alt0[kk - 1]), float(alt0[kk])
                seg = next(((f0, f1) for f0, f1 in P["base"] if f0 < kk < f1), None)
                if seg is None or abs(vL - vR) <= near:
                    refined.append(int(kk)); continue
                f0, f1 = seg
                fr2 = [f for f in (kk - 1, kk) if partnered[f] and f0 <= f < f1 and not np.isfinite(dx_vote[f])]
                if not fr2:
                    refined.append(int(kk)); continue
                aL = P["dx_applied"].copy(); aL[fr2] = vL
                aR = P["dx_applied"].copy(); aR[fr2] = vR
                sL = scorer.score(fr2, aL, a_fill, b_fill); sR = scorer.score(fr2, aR, a_fill, b_fill)

                def tot(cut: int) -> float:
                    t = 0.0
                    for f in fr2:
                        r_ = sL[f] if f < cut else sR[f]
                        if np.isfinite(r_["ratio"]):
                            t += float(r_["ratio"])
                    return t
                opts = [kk] + ([kk - 1] if not np.isfinite(dx_vote[kk - 1]) else []) + ([kk + 1] if not np.isfinite(dx_vote[kk]) else [])
                opts = [c_ for c_ in opts if f0 < c_ < f1]
                best_c = max(opts, key=lambda c_: (round(tot(c_), 6), c_ == kk))
                refined.append(int(best_c))
            placement = {int(k0): int(k1) for k0, k1 in zip(cands0, refined)}
            cands = sorted(set(refined))
            base_c = _add_cuts(P["base"], cands)
            vals_c, _ = _segment_dx(dx_vote, base_c, p.max_dx, c["dx0"])
            served_dx = P["dx_applied"]
            alt = served_dx.copy()
            for f0, f1, v in vals_c:
                alt[f0:f1] = v
            seg_of = {int(f0): (int(f0), int(f1)) for f0, f1, v in vals_c}
            gains: dict = {}
            for kk in cands:
                rec = {"cut": int(kk), "left": float(alt[kk - 1]), "right": float(alt[kk]), "served": float(served_dx[kk]),
                       "gain_left": float("nan"), "gain_right": float("nan"), "accepted": False,
                       "placed_from": sorted(int(k0) for k0, k1 in placement.items() if k1 == kk)}
                for side, W, f_at in (("left", range(kk - hold, kk), kk - 1), ("right", range(kk, kk + hold), kk)):
                    W = [f for f in W if 0 <= f < Fm and partnered[f] and trusted[f]]
                    if abs(float(alt[f_at] - served_dx[f_at])) > near:
                        g_ = ratio_gain(W, alt, a_fill, b_fill, served_dx, a_fill, b_fill)
                        rec[f"gain_{side}"] = float(g_)
                        seg = next((sg for sg in seg_of.values() if sg[0] <= f_at < sg[1]), None)
                        if seg is not None:
                            gains.setdefault(seg, []).append(g_)
                plateau_rec.append(rec)
            wins = {seg for seg, gl in gains.items() if any(np.isfinite(g_) and g_ > margin for g_ in gl)}
            for rec in plateau_rec:
                kk = rec["cut"]
                rec["accepted"] = any(seg in wins for seg in seg_of.values() if seg[1] == kk or seg[0] == kk)
            plateau_cuts = [r_["cut"] for r_ in plateau_rec if r_["accepted"]]
            if plateau_cuts:
                base_a = _add_cuts(P["base"], plateau_cuts)
                for f0, f1 in base_a:
                    if (f0, f1) not in wins and any(f0 == kk or f1 == kk for kk in plateau_cuts):
                        plateau_fixed[int(f0)] = float(served_dx[f0])
                P = project(forced, plateau_cuts)
    arb: dict = {}; neither = np.zeros(Fm, bool); win_ab = np.zeros(Fm, bool)
    boundary: list = []; propagated: list = []; joined: list = []; pinned_bad: list = []
    decided = np.zeros(Fm, bool); decided_ab = np.zeros(Fm, bool)
    changed_ab = np.zeros(Fm, bool)          # served a / b moved by > refill_change_px since the frame was last decided (R2)
    refill_changed: list = []; pinned_ab: list = []; rescored_rounds: list = []
    a_prev = a_fill.copy(); b_prev = b_fill.copy()   # what a changed frame was served BEFORE the redo that changed it
    n_rounds = 0
    part_idx = np.flatnonzero(partnered)
    weak_wins: list = []                                 # frames whose served value fails and whose own win is weakly supported
    ridge_blocked_all: list = []                         # decisive joint singles blocked by the ridge test (round 9b)
    uncorroborated_steps: list = []                      # short kept steps the scorer did not corroborate (round 9)
    C_dx_all = C_dx.copy(); C_ab_all = C_ab.copy()     # every contradiction of every round (the verdict bookkeeping)
    chg = float(p.refill_change_px)

    def bad_under(dx_arr: np.ndarray, a_arr: np.ndarray, b_arr: np.ndarray, thresh: float | None = None) -> np.ndarray:
        """Frames ALREADY scored under (dx, a, b) whose ratio is below `thresh` — the gate frame_match_frac by default; fm +
        arbitration_margin is 'near the gate' (round 8: a served ratio of 0.263 passed the 0.25 gate against the frame's own
        1.000, 6 laterals off, and was never arbitrated) — cache lookups, nothing scored."""
        t_ = fm if thresh is None else float(thresh)
        out = np.zeros(Fm, bool)
        for f in np.flatnonzero(partnered):
            rec = scorer.cache.get(scorer.key(int(f), dx_arr[f], a_arr[f], b_arr[f]))
            if rec is not None and np.isfinite(rec["ratio"]) and rec["ratio"] < t_:
                out[f] = True
        return out

    def dead_of(PP_: dict) -> np.ndarray:
        """Frames of a DEAD run (≥ seg_hold unmeasured live frames, quality['unmeasured_runs']; E9(d): a run of ≥ 2 weak
        fill-rejected frames) without a sound peak: their transform is interpolated by design and they are never judged —
        not by the quorum, not by a re-fill verdict (round 8: a junk gap next to an axial winner was refused
        'axial_residual' on its unmeasured dead frames)."""
        d = np.zeros(Fm, bool)
        for f0, f1 in PP_["unmeasured_runs"]:
            d[f0:f1] = True
        d &= ~(finite & (ncc >= float(p.vote_ncc)))
        # ROUND 10: a RIDGE-GUARDED frame is not dead — it is interpolated (or offered the other side by the boundary pass)
        # and JUDGED under what it is served; failing the gate it is named 'dx_residual' (two ridge-guarded aliases before a
        # saccade cut were interpolated on the wrong side, 30 laterals off, and hidden by the informational 'ridge_alias')
        return d | dead_weak

    def rescore(dx_arr: np.ndarray, a_arr: np.ndarray, b_arr: np.ndarray) -> list:
        """Round 7, refutation R3: every frame of the pool — on the subset path also every frame now off the served shift
        by more than substep_agree, and every re-filled frame — whose CURRENT served (dx, a, b) has not been scored yet (a
        plateau re-projection, a carve, a merge, a boundary reassignment or a re-fill changed it since) is scored, batched,
        before the contradictions are derived: a frame served a value that was never judged is otherwise invisible to the
        gate (bad_under reads the cache) until the final pass, where it only counts toward the quorum."""
        off_now = partnered & finite & (np.abs(o["dx"] - dx_arr) > agree)
        cand_ = np.flatnonzero((pool | off_now | changed_ab) & partnered)
        miss = [int(f) for f in cand_ if scorer.key(int(f), dx_arr[f], a_arr[f], b_arr[f]) not in scorer.cache]
        if miss:
            scorer.score(miss, dx_arr, a_arr, b_arr)
            pool[miss] = True
        return miss

    def redo_fill() -> None:
        """The a / b fill redone through the winners and the pins; every partnered non-kept frame whose served a / b moved
        by more than refill_change_px is CHANGED (round 7, refutation R2): it is arbitrated next round against its previous
        interpolation and its own measurement — never re-served unjudged (a measured frame rejected by the MAD fill and
        served 1-3 px off was re-served 6-19 px off through a neighbour's axial win at ratio 0.000 against its own 1.02)."""
        nonlocal a_fill, b_fill, kept
        a_before, b_before = a_fill, b_fill
        kept = kept0 | win_ab
        a_fill, b_fill = fills(kept)
        moved = (partnered & ~kept & ~np.isfinite(a_pin) & ~dead_of(P)
                 & ((np.abs(a_fill - a_before) > chg) | (np.abs(b_fill - b_before) > chg)))
        for f in np.flatnonzero(moved):
            f = int(f)
            a_prev[f] = a_before[f]; b_prev[f] = b_before[f]
            changed_ab[f] = True; decided_ab[f] = False
            if f not in refill_changed:
                refill_changed.append(f)

    # the arbitration LOOP: a projection can create contradictions of its own (a carve changes what a dead run is
    # served; a merge moves a cut; a re-fill moves a fill-rejected frame), so contradictions are re-derived against the
    # CURRENT served value until none is left (4 rounds at most; CS001 needs one or two) — a frame is decided once per
    # served value
    while scorer is not None and n_rounds < 4:
        # STEP CORROBORATION (round 9): a kept lateral step run that is short (< seg_hold frames) or large (≥ seg_dx_step from
        # a neighbouring run) is corroborated by the scorer — its frames under their own value against the neighbours'
        # values (the bridge across a middle run, the adjacent run's value at an end); a step whose own value does not
        # beat the alternative by the margin on the MAJORITY of its frames is not a measurement (a period alias scores like
        # the truth; a real saccade or transient wins on every frame) — dropped and re-projected (quality['uncorroborated_steps'])
        if short_steps:
            dropped_now = False
            for fr_s, v_lo, v_hi in list(short_steps):
                fr_s = [f for f in fr_s if partnered[f] and not step_dropped[f]]
                if not fr_s:
                    continue
                served_dx = P["dx_applied"]
                alt = served_dx.copy()
                for f in fr_s:
                    alt[f] = v_lo if abs(f - fr_s[0]) <= abs(fr_s[-1] - f) else v_hi
                s_rec = scorer.score(fr_s, served_dx, a_fill, b_fill); n_rec = scorer.score(fr_s, alt, a_fill, b_fill)
                pool[fr_s] = True
                n_win = sum(1 for f in fr_s if _better(s_rec[f], n_rec[f], margin)[0] > 0)
                wins = n_win * 2 > len(fr_s)                        # the MAJORITY of the run's frames prefer the step
                v_run = float(np.median(served_dx[fr_s]))
                if (len(fr_s) < int(p.seg_hold) and v_lo is not None and v_hi is not None and abs(v_run - v_lo) > float(p.excursion_max_dx)
                        and abs(v_run - v_hi) > float(p.excursion_max_dx) and abs(v_lo - v_hi) <= float(p.excursion_max_dx)):
                    wins = False                                    # ROUND 10: an excursion-and-return of < seg_hold frames is no step
                if not wins:
                    step_dropped[fr_s] = True; dropped_now = True
                    uncorroborated_steps.append((int(fr_s[0]), int(fr_s[-1]) + 1, round(float(np.median(served_dx[fr_s])), 2)))
            if dropped_now:
                P = project(forced, plateau_cuts)
        served_dx = P["dx_applied"]; served_dx_now[0] = served_dx
        rescored_rounds.append(rescore(served_dx, a_fill, b_fill))
        decidable &= decidable_of(served_dx, a_fill, b_fill)
        dec_arb &= decidable_of(served_dx, a_fill, b_fill, eval_rule=False)
        # a served value failing the gate OR within the margin above it ('near the gate') is a contradiction when the frame's
        # measurement differs at all (round 8: a served ratio of 0.263 vs the gate 0.25 against the frame's own 1.000)
        bad_now = bad_under(served_dx, a_fill, b_fill, fm + margin)
        C_dx = partnered & finite & dec_arb & ~ridge_alias & ~decided & ~np.isfinite(forced) & (
            (np.abs(o["dx"] - served_dx) > bar) | (bad_now & (np.abs(o["dx"] - served_dx) > near)))
        own_ab = measured0 & (((np.abs(a_fill - o["a"]) > tol) | (np.abs(b_fill - o["b"]) > tol))
                              | (bad_now & ((np.abs(a_fill - o["a"]) > 0.5) | (np.abs(b_fill - o["b"]) > 0.5))))
        C_ab = partnered & dec_arb & ~dead_weak & ~ridge_alias & ~kept & ~decided_ab & ~np.isfinite(a_pin) & (own_ab | changed_ab)
        if not C_dx.any() and not C_ab.any():
            # round 9b: the BOUNDARY pass (below) must run at least once when a ridge-guarded or unmeasured live frame sits next
            # to a transition of the served shift — such frames are never contradictions, and the loop used to break here
            # before offering them the other side's value (two ridge-guarded aliases before a saccade cut were interpolated
            # on the wrong side, 30 laterals off, unjudged)
            if n_rounds > 0 or not ((ridge_alias | (~finite & P["live"])) & partnered).any():
                break
        n_rounds += 1
        C |= C_dx | C_ab; C_dx_all |= C_dx; C_ab_all |= C_ab
        Cf = np.flatnonzero(C_dx | C_ab)
        served_rec = scorer.score(Cf, served_dx, a_fill, b_fill)
        Cd = np.flatnonzero(C_dx); Ca = np.flatnonzero(C_ab)
        Ca_own = np.flatnonzero(C_ab & measured0); Ca_prev = np.flatnonzero(C_ab & changed_ab)
        # the JOINT candidate (own dx, own a, own b) is built for EVERY measured axial contradiction whose own dx differs from
        # the served dx at all (> near), not only for C_dx & C_ab: an axial contradiction must be judged at the frame's OWN
        # lateral shift as well as at the served one (round 8, refuter: an in-bar +5-lateral transient with an a +5 px step on a
        # period-7 texture — own a / b at the served dx scored 0.000 and 'served' won at 0.61 while the joint scores 1.0)
        Cj = np.flatnonzero(C_ab & measured0 & ~pinned & (np.abs(o["dx"] - served_dx) > near))
        own_dx_rec: dict = {}; own_ab_rec: dict = {}; prev_rec: dict = {}; joint_rec: dict = {}
        alt_a = a_fill.copy(); alt_b = b_fill.copy()
        if Cd.size:                                            # the lateral shift: own dx with the served a / b
            alt = served_dx.copy(); alt[Cd] = o["dx"][Cd]
            own_dx_rec = scorer.score(Cd, alt, a_fill, b_fill)
        if Ca_own.size:                                        # the axial move: own a / b with the served dx
            alt_a[Ca_own] = o["a"][Ca_own]; alt_b[Ca_own] = o["b"][Ca_own]
            own_ab_rec = scorer.score(Ca_own, served_dx, alt_a, alt_b)
        if Ca_prev.size:                                       # a re-filled frame: its PREVIOUS interpolation (R2)
            pa = a_fill.copy(); pb = b_fill.copy(); pa[Ca_prev] = a_prev[Ca_prev]; pb[Ca_prev] = b_prev[Ca_prev]
            prev_rec = scorer.score(Ca_prev, served_dx, pa, pb)
        if Cj.size:                                            # both: own dx AND own a / b (refutation E)
            altj = served_dx.copy(); altj[Cj] = o["dx"][Cj]
            joint_rec = scorer.score(Cj, altj, alt_a, alt_b)
        vd = {int(f): decide(served_rec[f], [("own", own_dx_rec[f])]) for f in Cd}
        # E9 CORROBORATION (round 9): the per-frame structure ratio is a NOISY statistic (adjacent frames of one real pair
        # differ by 0.1-0.2) and degenerate along the dome ridge, so an own / joint win on a SINGLE frame whose served value
        # passes the gate is accepted only when it is (1) DECISIVE — the served record is near or under the gate — or (2)
        # part of an agreeing run of ≥ 2 contradicting frames, or (3) ANCHORED by a kept neighbour served its own dx within
        # substep_dx of the candidate — and in (2) / (3) not a ridge alias (its a within ridge_axial_tol of the neighbours'
        # served a); a run needs ≥ 3 frames (two adjacent period aliases agree with each other: round 8, D1). Otherwise the
        # frame is served the fill and the verdict recorded 'served' with 'uncorroborated' True
        # (CS001 v3→v1 frame 14: own −13 against neighbours −4.6 / −1.9 at 0.567 vs 0.491 — a verdict taken on noise).
        runs_c = {int(f): (int(r0), int(r1)) for r0, r1 in _agreeing_runs(np.where(C_dx & ~pinned, o["dx"], np.nan), agree)
                  for f in range(r0, r1)}
        f_first, f_last = (int(part_idx[0]), int(part_idx[-1])) if part_idx.size else (-1, -1)

        def a_jump(f: int, a_cand: float) -> float:
            # ROUND 10: the jump against the SERVED a of both partnered neighbours (kept or interpolated — it is what is served
            # there); reading kept neighbours only let a 2-frame alias at dx −74 / a +41 px between two junk frames through
            vals = [abs(float(a_cand) - float(a_fill[g])) for g in (f - 1, f + 1) if 0 <= g < Fm and partnered[g] and np.isfinite(a_fill[g])]
            return min(vals) if vals else 0.0

        def near_trusted(f: int) -> bool:
            """ROUND 10: a 2-frame run is corroborated only when a TRUSTED frame within ±2·seg_hold measures within excursion_max_dx
            of it (a 2-frame alias at dx −140 inside a dead region of junk frames won as a 'run' against an interpolation that
            failed there; nothing within 0.3 mm of it was ever measured)"""
            w_ = 2 * int(p.seg_hold)
            return any(trusted[g] and abs(float(o["dx"][g]) - float(o["dx"][f])) <= float(p.excursion_max_dx)
                       for g in range(max(0, f - w_), min(Fm, f + w_ + 1)) if g != f)

        def served_support(f: int) -> int:
            """The served shift's SUPPORT at f: trusted / owned frames within ±2·seg_hold of f served within seg_dx_step of f's
            served shift (a short candidate run is overruled by the excursion prior only when the served side is a SUSTAINED mode
            with more support than the candidate: ≥ seg_hold knots and more than the run's frames)."""
            w_ = 2 * int(p.seg_hold)
            return sum(1 for g in range(max(0, f - w_), min(Fm, f + w_ + 1)) if g != f and partnered[g] and (trusted[g] or np.isfinite(forced[g]))
                       and abs(float(served_dx[g]) - float(served_dx[f])) <= float(p.seg_dx_step))

        def own_ratio_of(f: int) -> tuple[float, float]:
            r_ = vd.get(f)
            rec_ = r_[2] if r_ is not None else None
            return (ratio_of(rec_), float(rec_.get("frac_0.5", np.nan)) if rec_ is not None else float("nan"))

        def corroborated_dx(f: int, a_cand: float, cand_dx=None, cand_a=None, cand_b=None) -> tuple[bool, str]:
            sr_ = ratio_of(served_rec[f]); orr, ofr = own_ratio_of(f)
            served_fails = (not np.isfinite(sr_)) or sr_ < fm + margin
            # the speckle witness on the candidate (own dx with the served a / b, or the joint own (dx, a, b)) — E10
            c_dx = cand_dx if cand_dx is not None else np.where(np.arange(Fm) == f, o["dx"], served_dx)
            c_a = cand_a if cand_a is not None else a_fill; c_b = cand_b if cand_b is not None else b_fill
            backed = lambda: sp_backs(f, c_dx, c_a, c_b)  # noqa: E731 — scored only when it decides
            # RIDGE (round 9b): a candidate whose a jumps by more than ridge_tol from both kept neighbours on a run shorter than 3
            # frames is the dome-ridge alias of the smooth feature (dx 91 / a 29 / b 57 next to a saccade scored decisively on a
            # 128-lateral band) — never decisive; a pure dx candidate carries the served a (no jump)
            aj0 = a_jump(f, a_cand); r0_ = runs_c.get(f)
            n_run0 = (r0_[1] - r0_[0]) if r0_ is not None else 1
            # a run touching the END of the partnered range keeps the round-8 end-single rule (the witness / anchoring below)
            at_end0 = (r0_ is not None and (r0_[0] <= f_first or r0_[1] - 1 >= f_last)) or (r0_ is None and (f <= f_first or f >= f_last))
            if (n_run0 < int(p.seg_hold) and not at_end0 and abs(float(o["dx"][f]) - float(served_dx[f])) > float(p.excursion_max_dx)
                    and served_support(f) >= max(int(p.seg_hold), n_run0 + 1)):
                return False, "excursion_far"                        # uncorroborated: served the fill, judged by the gate
            if aj0 > ridge_tol and (r0_ is None or r0_[1] - r0_[0] < 3):
                # the served value FAILS the gate proper as well (ROUND 10: not merely grazes it — CS032 v1_3→v1_2 frame 56 at
                # 0.299 against the 0.30 near-gate bar refused a pair at rel 0.91): nothing admissible scores on this frame — a
                # 1-frame joint excursion nothing corroborates (round 8's 'dx_residual' for an unwitnessed joint single): NAMED
                return False, ("ridge_decisive" if ((not np.isfinite(sr_)) or sr_ < fm) else "ridge")
            # DECISIVE: the served value fails or grazes the gate while the frame's own value — a SOUND measurement (peak ≥
            # vote_ncc; a weak frame is evidence only) — reaches decisive_own_min of its ceiling AND low_match in absolute
            # terms (a low per-frame ceiling must not inflate junk into a win) — and the speckle witness backs it (E10)
            if (served_fails and bool(sound[f]) and np.isfinite(orr) and orr >= float(p.decisive_own_min)
                    and np.isfinite(ofr) and ofr >= float(p.low_match)):
                if backed():
                    return True, "decisive"
                return False, "speckle_contradicts"
            # DECISIVE GAIN: the own value beats a served value that passes the gate by decisive_gain (a real 1-frame excursion
            # next to a junk neighbour: own 1.0 against 0.36), with the speckle witness backing it
            if (bool(sound[f]) and np.isfinite(orr) and np.isfinite(sr_) and orr - sr_ >= float(p.decisive_gain)
                    and orr >= float(p.decisive_own_min) and np.isfinite(ofr) and ofr >= float(p.low_match) and backed()):
                return True, "decisive_gain"
            aj = a_jump(f, a_cand)
            r_ = runs_c.get(f)
            # ROUND 10: the microsaccade prior of the single-frame rule applied to every SHORT run (< seg_hold): a candidate farther
            # than excursion_max_dx (0.3 mm) from the served shift on a run of < seg_hold frames is the dome-ridge alias whatever
            # it scores (on a dome the alias is self-consistent in a and tilt; CS032 v1_2→v1 frames 5-6 at −140 against 0 won as a
            # 2-frame 'run'); a decisive single beyond it is 'dx_residual' by the excursion rule below
            if r_ is not None and aj <= ridge_tol:
                n_run = r_[1] - r_[0]
                # ≥ 3 agreeing SOUND frames corroborate; a WEAK frame of a run (peak < vote_ncc) needs the speckle witness too
                # (two junk frames' chance aliases agree with each other at the structure scale); a 2-frame run only where
                # the served value FAILS on one of its frames (two adjacent period aliases agree with each other while the
                # served truth scores: round 8, D1)
                weak_here = not bool(sound[f])
                if n_run >= 3 and (not weak_here or (np.isfinite(orr) and orr >= float(p.decisive_own_min) and backed())):
                    return True, "run"
                if (n_run == 2 and any(((not np.isfinite(ratio_of(served_rec.get(x)))) or ratio_of(served_rec.get(x)) < fm + margin)
                                       for x in range(r_[0], r_[1]) if x in served_rec)
                        and (not weak_here or (np.isfinite(orr) and orr >= float(p.decisive_own_min))) and near_trusted(f) and backed()):
                    return True, "run"
            for g in (f - 1, f + 1):
                if (0 <= g < Fm and partnered[g] and trusted[g] and abs(float(o["dx"][g] - served_dx[g])) <= near
                        and abs(float(o["dx"][f] - o["dx"][g])) <= bar and aj <= ridge_tol):
                    return True, "anchored"
            if served_fails:
                return False, "weak_win"                             # the served value fails, the own is weakly supported: no value to serve
            return False, ("ridge" if aj > ridge_tol else "uncorroborated")

        uncorr: dict = {}
        ridge_blocked: list = []
        alt_dx_own = served_dx.copy()
        if Cd.size:
            alt_dx_own[Cd] = o["dx"][Cd]
        for f in Cd:
            f = int(f)
            if vd[f][0] == "own":
                okc, why = corroborated_dx(f, float(a_fill[f]), alt_dx_own, a_fill, b_fill)
                if not okc:
                    uncorr[f] = why; vd[f] = ("served", None, vd[f][2])
                    if why == "weak_win":
                        weak_wins.append(int(f))
                    if why == "ridge_decisive":
                        ridge_blocked.append(int(f))
        # an own a / b candidate beyond the tilt cap is not admissible (E7: the cap is on the RESIDUAL to the served lines'
        # tilt, plus the sanity cap on |b|; a garbage alias fit can score on its frame with a tilt of −50 px): decided on the
        # other candidates
        adm = {int(f): bool(measured0[f] and admissible_b(int(f))) for f in Ca}
        va: dict = {}
        for f in Ca:
            f = int(f)
            cands = ([("own", own_ab_rec[f])] if adm[f] else []) + ([("previous", prev_rec[f])] if changed_ab[f] else [])
            va[f] = decide(served_rec[f], cands)
            if va[f][0] == "own" and va[f][1] == "own":
                # the same corroboration for an axial single: decisive, an agreeing run (marked below), an end frame, or a
                # correction within ridge_axial_tol; a lone jump of tens of px against a served value that scores is not a knot
                sr_ = ratio_of(served_rec[f]); orr = ratio_of(va[f][2])
                in_run = any(0 <= g < Fm and C_ab[g] and measured0[g] and abs(float(o["a"][g] - o["a"][f])) <= tol
                             and abs(float(o["b"][g] - o["b"][f])) <= tol for g in (f - 1, f + 1))
                decisive_ab = ((not np.isfinite(sr_)) or sr_ < fm + margin) and np.isfinite(orr) and orr >= float(p.decisive_own_min)
                gain_ab = np.isfinite(sr_) and np.isfinite(orr) and orr - sr_ >= float(p.decisive_gain) and orr >= float(p.decisive_own_min)
                small = abs(float(o["a"][f] - a_fill[f])) <= ridge_tol and abs(float(o["b"][f] - b_fill[f])) <= ridge_tol
                # E10: a decisive axial single needs the speckle witness (a junk / periodic frame's depth alias scores on the
                # structure feature); an end frame is no longer exempt — the witness rule below judges an isolated axial run
                if not (in_run or small or ((decisive_ab or gain_ab) and sp_backs(f, served_dx, alt_a, alt_b))):
                    uncorr[f] = "axial_uncorroborated"; va[f] = ("served", None, va[f][2])
        # the JOINT candidate (own dx, own a, own b) on a frame contradicting in both: it wins when it reaches the gate,
        # beats the served record (lexicographic), beats it on ncc_mean too (R1: the continuous score) and its ratio is
        # not below the best single candidate's (a joint that does not beat own-dx-alone changes a / b for nothing)
        joint: set = set()
        for f in Cj:
            f = int(f)
            if not adm[f]:
                continue
            jr, sr = joint_rec[f], served_rec[f]
            v_j = decide(sr, [("joint", jr)])[0]
            best_single = None                              # the single candidates: own dx, own a / b, the previous interpolation
            for r_ in (own_dx_rec.get(f), own_ab_rec.get(f), prev_rec.get(f)):
                if r_ is not None and (best_single is None or _better(r_, best_single, margin)[0] > 0):
                    best_single = r_
            s_ncc, j_ncc = float(sr.get("ncc_mean", np.nan)), float(jr.get("ncc_mean", np.nan))
            ncc_ok = (not np.isfinite(s_ncc)) or (np.isfinite(j_ncc) and j_ncc > s_ncc)
            if v_j == "own" and ncc_ok and (best_single is None or ratio_of(jr) >= ratio_of(best_single) - 1e-9):
                vd_prev = vd.get(f); vd[f] = ("own", "joint", jr)
                okc, why = corroborated_dx(f, float(o["a"][f]), altj, alt_a, alt_b)
                if not okc and vd_prev is not None:
                    vd[f] = vd_prev
                if okc:
                    vd[f] = ("own", "joint", jr); va[f] = ("own", "joint", jr); joint.add(f)
                else:
                    uncorr[f] = why
                    if why == "ridge_decisive" and f not in ridge_blocked:
                        ridge_blocked.append(int(f))
                    if f in vd and vd[f][0] == "own":
                        vd[f] = ("served", None, vd[f][2])
        # a joint winner that was not a dx contradiction (in-bar own dx) is a dx winner from here on: recorded, decided, and a
        # member of the dx runs / anchoring like any other
        joint_only = sorted(f for f in joint if not C_dx[f])
        if joint_only:
            C_dx[joint_only] = True; C_dx_all[joint_only] = True; C[joint_only] = True
        for f in sorted(set(int(f) for f in Cd) | set(joint_only)):
            f = int(f); sr = served_rec[f]; v, name, rec_ = vd[f]
            odr = own_dx_rec.get(f, rec_)
            arb.setdefault(f, {"served": ratio_of(sr), "served_ncc": float(sr.get("ncc_mean", np.nan)), "weak": bool(weak[f]), "pinned": bool(pinned[f])})
            arb[f]["dx"] = {"own": ratio_of(odr), "own_ncc": float(odr.get("ncc_mean", np.nan)) if odr is not None else float("nan"), "verdict": v,
                            "own_dx": float(o["dx"][f]), "served_dx": float(served_dx[f]), "round": n_rounds, "joint": f in joint,
                            "decided_on": (_better(rec_, sr, margin)[1] if rec_ is not None else None),
                            "own_frac": (float(rec_.get("frac_0.5", np.nan)) if rec_ is not None else float("nan")),
                            "uncorroborated": uncorr.get(f)}
            if pinned[f]:                                      # a bound the served value does not fit: a verdict, never served
                if v in ("own", "neither"):
                    pinned_bad.append(f)
            elif v == "own":
                forced[f] = o["dx"][f]
            elif v == "neither":
                neither[f] = True
        decided[Cd] = True
        if joint_only:
            decided[joint_only] = True
        ridge_blocked_all += [f for f in ridge_blocked if f not in ridge_blocked_all]
        pins_now = False
        for f in Ca:
            f = int(f); sr = served_rec[f]; v, name, rec_ = va[f]
            arb.setdefault(f, {"served": ratio_of(sr), "served_ncc": float(sr.get("ncc_mean", np.nan)), "weak": bool(weak[f]), "pinned": bool(pinned[f])})
            own_r = own_ab_rec.get(f)
            arb[f]["axial"] = {"own": (ratio_of(own_r) if adm[f] else float("nan")),
                               "own_ncc": (float(own_r.get("ncc_mean", np.nan)) if own_r is not None else float("nan")),
                               "previous": (ratio_of(prev_rec[f]) if changed_ab[f] else None),
                               "verdict": (v if v != "own" else ("own" if name in ("own", "joint") else "previous")), "candidate": name,
                               "own_a": float(o["a"][f]), "served_a": float(a_fill[f]), "own_b": float(o["b"][f]), "served_b": float(b_fill[f]),
                               "previous_a": (float(a_prev[f]) if changed_ab[f] else None), "previous_b": (float(b_prev[f]) if changed_ab[f] else None),
                               "joint": f in joint, "admissible": adm[f], "changed": bool(changed_ab[f]), "round": n_rounds,
                               "decided_on": (_better(rec_, sr, margin)[1] if rec_ is not None else None)}
            if v == "own" and name in ("own", "joint"):
                win_ab[f] = True
            elif v == "own" and name == "previous":
                a_pin[f] = a_prev[f]; b_pin[f] = b_prev[f]; pins_now = True
                if f not in pinned_ab:
                    pinned_ab.append(f)
            elif v == "neither":
                neither[f] = True
        decided_ab[Ca] = True; changed_ab[Ca] = False
        # RUNS are decided as ONE unit (refutation C): contiguous contradicting frames whose own values agree share
        # the same evidence — a run holding a winner is served its own values on every frame, the 'neither' frames
        # included (their own value scores better than the served one, only short of the bar)
        for r0, r1 in _agreeing_runs(np.where(C_dx & ~pinned, o["dx"], np.nan), agree):
            fr_ = list(range(r0, r1))
            if len(fr_) >= 2 and any(np.isfinite(forced[x]) for x in fr_):
                for x in fr_:
                    if not np.isfinite(forced[x]):
                        forced[x] = o["dx"][x]; neither[x] = False
                        arb[x]["dx"]["verdict"] = "run"; joined.append(int(x))
        if Ca_own.size:
            in_ab = np.zeros(Fm, bool); in_ab[Ca_own] = True
            in_ab &= np.array([bool(adm.get(int(f), False)) for f in range(Fm)], bool)
            for r0, r1 in _runs(in_ab):
                fr_ = list(range(r0, r1))
                agree_ab = all(abs(float(o["a"][x] - o["a"][x - 1])) <= tol and abs(float(o["b"][x] - o["b"][x - 1])) <= tol for x in fr_[1:])
                if len(fr_) >= 2 and agree_ab and win_ab[fr_].any():
                    for x in fr_:
                        if not win_ab[x]:
                            win_ab[x] = True; arb[x]["axial"]["verdict"] = "run"; joined.append(int(x))
                            a_pin[x] = np.nan; b_pin[x] = np.nan
                            neither[x] = bool(C_dx[x]) and vd.get(x, ("served",))[0] == "neither" and not np.isfinite(forced[x])
        # PROPAGATION (the subset path scores a sample): a winner's verdict extends to its UNSCORED neighbours
        # whose own measurement agrees with the run and contradicts the same served value — the same evidence,
        # the same verdict; they are scored under what they are served in the final pass
        for r0, r1 in _agreeing_runs(forced, agree):
            run_vals = [float(forced[x]) for x in range(r0, r1)]
            for f, step in ((r0 - 1, -1), (r1, 1)):
                while (0 <= f < Fm and partnered[f] and finite[f] and not pinned[f] and not decided[f] and dec_arb[f]
                       and not np.isfinite(forced[f])
                       and scorer.key(f, served_dx[f], a_fill[f], b_fill[f]) not in scorer.cache
                       and abs(float(o["dx"][f] - served_dx[f])) > near
                       and max(run_vals + [float(o["dx"][f])]) - min(run_vals + [float(o["dx"][f])]) <= agree):
                    forced[f] = o["dx"][f]; decided[f] = True; propagated.append(int(f)); run_vals.append(float(o["dx"][f]))
                    f += step
        if (win_ab & ~kept).any() or pins_now:
            redo_fill()
        P = project(forced, plateau_cuts)
        # BOUNDARY: an unmeasured (or weak, unserved) frame next to a TRANSITION of the served shift (a segment cut or
        # an override run's edge) is offered the value on the other side
        # ROUND 9b: a DISCREDITED frame — arbitrated, its own value neither served nor credible (own ratio under the gate) — is
        # as uninformative as an unmeasured one: it is offered the other side's value too (a junk alias next to a saccade cut
        # was otherwise served the wrong side, 24 laterals off, with 'served' passing the gate)
        lost = np.array([bool(decided[f] and not np.isfinite(forced[f]) and f in arb and "dx" in arb[f]
                              and not (np.isfinite(arb[f]["dx"].get("own", np.nan)) and arb[f]["dx"]["own"] >= fm)) for f in range(Fm)], bool)
        cand = partnered & ~pinned & ~np.isfinite(forced) & dec_arb & (~sound | lost | ridge_alias)
        applied = P["dx_applied"]; live_now = P["live"]
        trans = [i for i in range(1, Fm) if live_now[i] and live_now[i - 1] and abs(float(applied[i] - applied[i - 1])) > bar]
        moved = False
        for i in trans:
            for frames, alt_v in ((range(i - 1, -1, -1), float(applied[i])), (range(i, Fm), float(applied[i - 1]))):
                run = []
                for f in frames:
                    if not cand[f] or np.isfinite(forced[f]):
                        break
                    run.append(int(f))
                if not run:
                    continue
                alt = applied.copy(); alt[run] = alt_v
                s_rec = scorer.score(run, applied, a_fill, b_fill); n_rec = scorer.score(run, alt, a_fill, b_fill)
                pool[run] = True                                 # scored under the served value: judged (R4)
                for f in run:                                   # from the transition outward, contiguous
                    if verdict(s_rec[f], n_rec[f]) != "own":
                        break
                    forced[f] = alt_v; boundary.append(f); decided[f] = True; moved = True
                    arb[f] = {"served": ratio_of(s_rec[f]), "served_ncc": float(s_rec[f].get("ncc_mean", np.nan)), "weak": bool(weak[f]), "pinned": False,
                              "dx": {"own": ratio_of(n_rec[f]), "own_ncc": float(n_rec[f].get("ncc_mean", np.nan)), "verdict": "neighbour",
                                     "own_dx": float(alt_v), "served_dx": float(applied[f]), "round": n_rounds, "joint": False}}
        if moved:
            P = project(forced, plateau_cuts)
    C_dx, C_ab = C_dx_all, C_ab_all
    owned = np.isfinite(forced)
    served_final = P["dx_applied"]
    # R2 / ROUND 10: the POSE of the served transform — the served lines' half-span tilt at each partnered frame's SERVED dx
    # (a robust line per frame, no images), the median |b_lines| over the measured frames → theta; the seed's and the
    # fine stage's estimates stay in quality (a garbage half-search seed on a df-bound pair read 16° where the served
    # transform implies 0.02 px)
    b_lines_served = np.full(Fm, np.nan)
    for f in np.flatnonzero(partnered & (measured0 | P["live"])):
        f = int(f); fr_ = f + df
        if not (0 <= fr_ < F) or not np.isfinite(served_final[f]):
            continue
        _, b_lines_served[f] = _b_lines(ref_b.served[:, fr_], mov_b.served[:, f], None if mov_b.valid is None else mov_b.valid[:, f],
                                        float(served_final[f]), xc_, min_win_, p.mad_k)
    _bls = np.abs(b_lines_served[measured0 & partnered]); _bls = _bls[np.isfinite(_bls)]
    if _bls.size >= 5:
        theta = pose_angle_deg(float(np.median(_bls)), ref_b.spacing, L); b_lines_med = float(np.median(_bls))
    pose_flag = bool(np.isfinite(theta) and theta > float(p.pose_max_deg))
    # WITNESS (R1, round 8) — computed FIRST, the anchoring below builds on it. A JOINT winner (own dx AND own a / b) on a run
    # of at most two frames whose a / b sat beyond axial_tol_px of the fill and whose gate statistic SATURATED on its frame
    # (frac_0.5 ≥ 1 − arbitration_margin: the metric could not rank it against a better candidate) is a 1-2-frame
    # lateral+axial excursion nothing corroborates: it needs a MEASURED neighbour agreeing in a AND b (within axial_tol_px)
    # with the run's served a / b, else the run is 'dx_residual' (round 7, refutation R1). Round 8: every short carved run
    # whose gate statistic saturates under the FINAL served value is tested, kept or joint alike; every base segment of ≤ 2
    # frames that stands in the final segmentation is tested too; the neighbour across a cut is a witness.
    # ROUND 9 (E12, the two alias holes of probe_target / probe_inject): (i) a short run is saturation-tested under EACH of its
    # frames' OWN (dx, a, b) as well as under the served value (a 2-frame base segment served its MEDIAN scored 0.0 where each
    # own value scored 1.0, and the witness never ran); (ii) a neighbour g is a witness only when it agrees in a AND b with
    # the run's served a / b at BOTH run frames and is not itself in an unwitnessed short run — the witness pass is iterated to
    # a fixed point, so two alias runs vouching only for each other stay unwitnessed.
    unwitnessed: list = []; kept_unwitnessed: list = []
    carved_short = [(int(r0), int(r1)) for r0, r1, at_end in P["carved"] if r1 - r0 <= 2 and (r0, r1) not in P["merged"]]
    carved_all = {(int(r0), int(r1)) for r0, r1, _ in P["carved"]}
    final_segs = {(int(f0), int(f1)) for f0, f1 in P["segments"]}
    base_short = [(int(f0), int(f1)) for f0, f1 in P["base"] if f1 - f0 <= 2 and (int(f0), int(f1)) in final_segs
                  and (int(f0), int(f1)) not in carved_all and any(measured0[f] and partnered[f] for f in range(f0, f1))]
    short_runs = sorted(set(carved_short) | set(base_short))
    # E12 (round 9b) — AXIAL-ONLY short runs: a run of ≤ 2 frames served their OWN a / b by an axial win the fill had rejected,
    # ISOLATED from every adjacent measured frame by more than max(axial_step_px, 2 × the pair's a-jitter) in a or b, is an
    # alias suspect exactly like a joint winner: the speckle witness must back it on one of its frames where it can speak;
    # where the witness is silent the saturation rule below applies. Unwitnessed → 'axial_residual' (quality['axial_unwitnessed_runs'])
    axial_short: list = []; axial_unwitnessed: list = []; axial_dropped: list = []
    if scorer is not None and bool(p.axial_witness):
        # an axial own-win run the fill had rejected, OR a KEPT run (the fill's own 2-frame axial step rule admits a 2-frame
        # plateau: two injected aliases at a +17 px agreeing with each other were kept as a step and never judged)
        ax_win = (win_ab | kept0) & measured0 & partnered
        step_ax = max(float(p.axial_step_px), 2.0 * jit_a, 2.0 * jit_b)
        # maximal runs of frames served their OWN a / b that AGREE with each other (within axial_tol_px); an isolated step run
        # differs from every adjacent measured frame by more than step_ax in a or b
        for r0, r1 in _runs(ax_win):
            f = r0
            while f < r1:
                g = f + 1
                while g < r1 and abs(float(o["a"][g] - o["a"][g - 1])) <= tol and abs(float(o["b"][g] - o["b"][g - 1])) <= tol:
                    g += 1
                if g - f <= 2 and (int(f), int(g)) not in set(short_runs):
                    nbrs = [h for h in (f - 1, g) if 0 <= h < Fm and partnered[h] and measured0[h] and np.isfinite(o["a"][h]) and np.isfinite(o["b"][h])]
                    # ISOLATED = a spike / plateau: the run differs from every adjacent measured frame by more than step_ax in a
                    # (or in b) with the neighbours on the SAME side — a frame whose a and b lie between its two neighbours' is a
                    # ramp (CS001 v3→v1 frame 40, the saccade in flight: a 9 → 2 → −5), not an alias
                    def _iso(series, val):
                        d = [float(val) - float(series[h]) for h in nbrs]
                        return all(abs(v) > step_ax for v in d) and (len(d) == 1 or (d[0] > 0) == (d[1] > 0))
                    if nbrs and (_iso(o["a"], a_fill[f]) or _iso(o["b"], b_fill[f])):
                        axial_short.append((int(f), int(g)))
                f = g
        for r0, r1 in axial_short:
            fr_ = [f for f in range(r0, r1) if partnered[f]]
            if not fr_:
                continue
            k_wo = kept.copy(); k_wo[r0:r1] = False                 # the fill WITHOUT the run: what the run replaced
            a_prev_fill, b_prev_fill = fills(k_wo)
            rec_ax = scorer.score(fr_, served_final, a_fill, b_fill)
            sat = any(np.isfinite(rec_ax[f]["frac_0.5"]) and rec_ax[f]["frac_0.5"] >= 1.0 - margin for f in fr_)
            if scorer_sp is not None:
                # where the speckle witness can speak, SATURATION is the speckle statistic's (the structure feature's frac_0.5
                # sits near 1.0 on a real pair — its saturation is not the signature of an alias; the speckle's is)
                rec_sp = scorer_sp.score(fr_, served_final, a_fill, b_fill)
                if any(np.isfinite(rec_sp[f]["ceiling"]) and rec_sp[f]["ceiling"] >= float(p.speckle_ceiling_min) for f in fr_):
                    sat = any(np.isfinite(rec_sp[f]["frac_0.5"]) and rec_sp[f]["frac_0.5"] >= 1.0 - margin for f in fr_)
            votes = [sp_prefers(f, served_final, a_fill, b_fill, served_final, a_prev_fill, b_prev_fill) for f in fr_]
            usable = any(u for _, u in votes)
            # INTERIOR = a KEPT measurement on both sides (a junk / dead / rejected neighbour is no witness of the interior:
            # a real 1-frame +25 px excursion next to a dead junk gap keeps the vote)
            interior = all(0 <= h < Fm and partnered[h] and kept[h] and np.isfinite(o["a"][h]) and np.isfinite(o["b"][h])
                           for h in (r0 - 1, r1))
            if usable and any(v > 0 for v, u in votes if u):
                if sat and interior:
                    # ROUND 10 (E12 ii): an INTERIOR spike of ≤ 2 frames disagreeing with BOTH measured neighbours at a
                    # SATURATED statistic — the metric cannot rank it against the value it replaced, and no neighbour agrees
                    # with it: a vote taken at saturation is not a witness (two injected aliases at a +17 px scored 1.0 on
                    # both features and were served); a run at the volume END keeps the vote (a real end step has one side)
                    axial_unwitnessed.append((int(r0), int(r1)))
                continue                                         # the speckle witness PREFERS the run's own a / b: witnessed
            if usable and any(v < 0 for v, u in votes if u):
                # the speckle witness prefers the fill WITHOUT the run: the run is DROPPED and its frames served the fill (a
                # still-clipped P5 member measures 1-frame axial spikes of 10-16 px the speckle does not see) — a correction,
                # recorded, never a refusal (the second witness overruled the first)
                axial_dropped.append((int(r0), int(r1)))
                kept[r0:r1] = False; win_ab[r0:r1] = False; a_pin[r0:r1] = np.nan; b_pin[r0:r1] = np.nan
                continue
            if sat:
                # a TIE (or a silent witness) at a SATURATED statistic (structure, or speckle where it speaks: a periodic patch
                # aliases in depth too) — nothing can rank the run and no neighbour agrees with an isolated one: unwitnessed
                axial_unwitnessed.append((int(r0), int(r1)))
        if axial_dropped:
            a_fill, b_fill = fills(kept)
    if scorer is not None and short_runs:
        fr_s = [f for r0, r1 in short_runs for f in range(r0, r1) if partnered[f]]
        rec_s = scorer.score(fr_s, served_final, a_fill, b_fill) if fr_s else {}
        # (i) each frame under its OWN measurement (dx, a, b) where it has one
        fr_own = [f for f in fr_s if finite[f] and measured0[f]]
        rec_own: dict = {}
        if fr_own:
            odx = served_final.copy(); oa = a_fill.copy(); ob = b_fill.copy()
            odx[fr_own] = o["dx"][fr_own]; oa[fr_own] = o["a"][fr_own]; ob[fr_own] = o["b"][fr_own]
            rec_own = scorer.score(fr_own, odx, oa, ob)
        tested: dict = {}
        # round 9b: where the SPECKLE witness can speak, SATURATION is the speckle statistic's (the structure feature's
        # frac_0.5 sits near 1.0 on a real pair; a periodic alias saturates both)
        rec_s_sp = scorer_sp.score(fr_s, served_final, a_fill, b_fill) if (scorer_sp is not None and fr_s) else {}
        rec_own_sp: dict = {}
        if scorer_sp is not None and fr_own:
            rec_own_sp = scorer_sp.score(fr_own, odx, oa, ob)
        for r0, r1 in short_runs:
            fr_ = [f for f in range(r0, r1) if partnered[f]]
            sp_ok_run = any(f in rec_s_sp and np.isfinite(rec_s_sp[f].get("ceiling", np.nan)) and rec_s_sp[f]["ceiling"] >= float(p.speckle_ceiling_min) for f in fr_)
            if sp_ok_run:
                fracs = [float(rec_s_sp[f]["frac_0.5"]) for f in fr_ if f in rec_s_sp and np.isfinite(rec_s_sp[f]["frac_0.5"])]
                fracs += [float(rec_own_sp[f]["frac_0.5"]) for f in fr_ if f in rec_own_sp and np.isfinite(rec_own_sp[f]["frac_0.5"])]
            else:
                fracs = [float(rec_s[f]["frac_0.5"]) for f in fr_ if np.isfinite(rec_s[f]["frac_0.5"])]
                fracs += [float(arb[f]["dx"].get("own_frac", np.nan)) for f in fr_ if f in arb and "dx" in arb[f]]
                fracs += [float(rec_own[f]["frac_0.5"]) for f in fr_ if f in rec_own and np.isfinite(rec_own[f]["frac_0.5"])]
            if any(np.isfinite(v) and v >= 1.0 - margin for v in fracs):
                tested[(r0, r1)] = fr_
        # (ii) the witness pass to a fixed point
        in_run = {}
        for (r0, r1) in tested:
            for f in range(r0, r1):
                in_run[f] = (r0, r1)
        status = {k: False for k in tested}                    # witnessed?
        changed_w = True
        while changed_w:
            changed_w = False
            for (r0, r1), fr_ in tested.items():
                if status[(r0, r1)] or not fr_:
                    continue
                ok_w = False
                for g in (r0 - 1, r1):
                    if not (0 <= g < Fm and partnered[g] and measured0[g] and np.isfinite(o["a"][g]) and np.isfinite(o["b"][g])):
                        continue
                    if g in in_run and not status[in_run[g]]:
                        continue                                 # itself in an unwitnessed short run: no witness
                    if all(abs(float(o["a"][g] - a_fill[x])) <= w_tol_a and abs(float(o["b"][g] - b_fill[x])) <= w_tol_b for x in (r0, r1 - 1)):
                        ok_w = True; break
                if ok_w:
                    status[(r0, r1)] = True; changed_w = True
        for (r0, r1), fr_ in tested.items():
            if not status[(r0, r1)]:
                unwitnessed.append((int(r0), int(r1)))
                if not any(f in arb and bool(arb[f].get("dx", {}).get("joint")) for f in fr_):
                    kept_unwitnessed.append((int(r0), int(r1)))
    unw_frames = sorted({int(f) for r0, r1 in unwitnessed for f in range(r0, r1)})
    # ANCHORING: a single dx winner is a 1-frame lateral excursion unless ANCHORED: its own dx lies within substep_dx of an
    # anchored neighbour's measurement or between two (a ramp / a saccade in flight the constant misses). Anchors
    # are frames whose measurement is not in question — served within the bar of their own dx (the base, a run, an
    # end frame), or arbitrated with their own value scoring ≥ frame_match_frac — and singles anchored by a frame
    # other than the one they would vouch for. An unanchored INTERIOR single is 'dx_residual' (REJECT). Round 8: a
    # single at a RUN END is judged by the WITNESS rule above like any short run; a frame the witness rule refuses
    # vouches for NOTHING. The end singles' own neighbour / TREND test is reported: quality['unanchored_end_singles'].
    singles = [r0 for r0, r1 in P["interior_singles"]]; end_singles = [r0 for r0, r1 in P["end_singles"]]
    credible = np.array([bool(f in arb and "dx" in arb[f] and np.isfinite(arb[f]["dx"]["own"]) and arb[f]["dx"]["own"] >= fm)
                         for f in range(Fm)], bool)
    anchored = partnered & finite & ~pinned & ((np.abs(o["dx"] - served_final) <= bar) | credible)
    anchored[singles] = False
    if unw_frames:
        anchored[unw_frames] = False
    pending = set(int(f) for f in singles)
    changed = True
    while changed and pending:
        changed = False
        for f in sorted(pending):
            w = [float(o["dx"][g]) for g in (f - 1, f + 1) if 0 <= g < Fm and anchored[g]]
            if w and min(w) - bar <= float(o["dx"][f]) <= max(w) + bar:
                anchored[f] = True; pending.discard(f); changed = True
    # ROUND 9: with the corroboration rule every single own win is DECISIVE (its served value failed or grazed the gate),
    # run-corroborated or anchored — an unanchored decisive single is a real 1-frame lateral excursion the evidence
    # demands (a microsaccade of 3-13 laterals lasts one frame), served and REPORTED ('dx_excursion',
    # quality['dx_excursions']); 'dx_residual' (REJECT) is the WITNESS rule's alias verdict below
    # a microsaccade prior: a served excursion sits within excursion_max_dx of the fill it replaced and such frames are few
    # (max_excursions); an alias texture produces many, far off — those are named 'dx_residual'
    exc_ok: list = []; exc_bad: list = []
    for f in sorted(pending):
        f = int(f)
        srv_ = float(arb[f]["dx"]["served_dx"]) if f in arb and "dx" in arb[f] else float("nan")
        if np.isfinite(srv_) and abs(float(o["dx"][f]) - srv_) <= float(p.excursion_max_dx):
            exc_ok.append(f)
        else:
            exc_bad.append(f)
    n_exc_max = max(int(p.max_excursions), int(np.ceil(0.08 * max(1, int(partnered.sum())))))   # ≈ 8 per 100 frames
    if len(exc_ok) > n_exc_max:
        exc_bad = sorted(set(exc_bad) | set(exc_ok)); exc_ok = []
    dx_excursions = [(f, f + 1) for f in exc_ok]
    residual_runs: list = [(f, f + 1) for f in exc_bad]
    base_of = {int(f): (int(b0), int(b1)) for b0, b1 in P["base"] for f in range(int(b0), int(b1))}

    def trend_at(f: int) -> list:
        """The base segment's trend at an end single: a line over the anchored frames within seg_hold frames on the
        single's inner side(s) (≥ 3 of them, inside its base segment), extrapolated to f."""
        b0, b1 = base_of.get(int(f), (int(f), int(f) + 1))
        vals: list = []
        for side in (-1, 1):
            hs_ = [h for h in range(f + side, f + side * (hold + 1), side) if b0 <= h < b1 and h != f and anchored[h]]
            if len(hs_) >= 3:
                coef = np.polyfit(np.asarray(hs_, float), np.asarray(o["dx"][hs_], float), 1)
                vals.append(float(np.polyval(coef, float(f))))
        return vals

    unanchored_end: list = []
    for f in end_singles:
        w = [float(o["dx"][g]) for g in (f - 1, f + 1) if 0 <= g < Fm and anchored[g]] + trend_at(int(f))
        if not (w and min(w) - bar <= float(o["dx"][f]) <= max(w) + bar):
            unanchored_end.append(int(f))
    residual_runs = sorted(set(residual_runs) | set(unwitnessed))
    # round 9b: a ridge-blocked decisive joint single (its served value fails the gate, its own a jumps from both neighbours)
    # that ended up served the fill is a named refusal like an unwitnessed joint run
    _dead_now = dead_of(P)                                 # a junk frame of a dead run is interpolated by design, never named
    rb = [int(f) for f in ridge_blocked_all if partnered[f] and not np.isfinite(forced[f]) and not _dead_now[f]]
    residual_runs = sorted(set(residual_runs) | {(f, f + 1) for f in rb})
    weak_segments = [(int(f0), int(f1), float(v)) for f0, f1, v in P["dx_segments"] if all(weak[f] and owned[f] for f in range(f0, f1))]
    weak_segments += [(int(r0), int(r1), float(v)) for r0, r1, v, _ in P["override_runs"] if all(weak[f] for f in range(r0, r1))]
    # E7: the tilt cap is on the RESIDUAL to the served lines' tilt (a kept frame whose tissue disagrees with the lines by
    # more than max_tilt_px half-span), plus the sanity cap on |b|; a frame without a line fit is judged on |b| ≤ cap
    tilt_all = [int(f) for f in np.flatnonzero(measured0 & partnered & np.isfinite(b_raw)) if abs(float(b_raw[f])) > cap_abs]
    tilt_high = [int(f) for f in np.flatnonzero(kept & partnered & ~dx_far) if np.isfinite(tilt_residual_own(int(f))) and tilt_residual_own(int(f)) > cap]
    # the verdict needs a RUN of ≥ 2 consecutive frames beyond the residual cap, or ≥ 5 % of the kept frames scattered — a
    # single frame beyond it is a transition of the served line (P5 v1_2→v1_3 frame 52: the still-clipped reconstruction's
    # tilt changes by 35 px over frames 52-54 while the tissue turned two frames earlier), reported, not a refusal
    tilt_mask = np.zeros(Fm, bool); tilt_mask[tilt_all] = True
    tilt_runs = [(int(f0), int(f1)) for f0, f1 in _runs(tilt_mask) if f1 - f0 >= 2]
    tilt_frames = (tilt_all if (tilt_runs or len(tilt_all) >= max(2, int(np.ceil(0.05 * max(1, int((measured0 & partnered).sum())))))) else [])
    tilt_singles = [f for f in tilt_all if f not in tilt_frames]
    pin_beyond = sorted(int(f) for f in pinned_bad if abs(float(o["dx"][f])) > float(p.max_dx))
    pin_edge = sorted(int(f) for f in pinned_bad if abs(float(o["dx"][f])) <= float(p.max_dx))
    res.dx_segments = P["dx_segments"]; res.dx_applied = P["dx_applied"]; res.a = a_fill; res.b = b_fill
    res.measured = kept; res.live = P["live"].copy()
    # segment verdicts on the FINAL transform
    edge_segments: list = []; beyond_segments: list = []; dx_mad: list = []
    for f0, f1, v in res.dx_segments:
        fin = finite[f0:f1]
        if fin.any() and float(np.mean(o["at_edge"][f0:f1][fin])) >= float(p.edge_frac):
            edge_segments.append((int(f0), int(f1)))
        sel = trusted[f0:f1] | owned[f0:f1]
        if not sel.any():
            dx_mad.append(None)
            continue
        dx_mad.append(round(float(np.median(np.abs(o["dx"][f0:f1] - res.dx_applied[f0:f1])[sel])), 3))
        if abs(float(v)) > float(p.max_dx):
            beyond_segments.append((int(f0), int(f1)))
    beyond_frames = sorted(int(f) for f in np.flatnonzero((owned | trusted) & partnered & (np.abs(res.dx_applied) > float(p.max_dx)))) + pin_beyond
    # the FINAL scores: every judged frame under what is served now (cache hits except the changed frames)
    ratios: dict = {}; bad_frames: list = []; n_frames_eval = 0; low_frames = False; ratio_min = float("nan")
    low_runs: list = []; untrusted_runs: list = []; axial_runs: list = []; unscored: list = []; bad_runs: list = []
    undecidable: list = []
    if scorer is not None:
        judged = (pool | C | changed_ab) & partnered
        for f in boundary + propagated:
            judged[f] = True
        jf = [int(f) for f in np.flatnonzero(judged)]
        final_rec = scorer.score(jf, res.dx_applied, res.a, res.b)
        decidable &= decidable_of(res.dx_applied, res.a, res.b)
        undecidable = sorted(int(f) for f in np.flatnonzero(partnered & ~decidable))
        dead = dead_of(P)                                   # a frame with a sound peak in a dead run is judged
        for f in jf:
            r_ = final_rec[f]["ratio"]
            if np.isfinite(r_):
                ratios[f] = float(r_)
        # the subset path (round 7, refutation R4): every frame contiguous with a failing frame is judged too — a run of
        # failures is a verdict, and a sample that stops at one of them cannot see the run
        if source == "subset":
            seen = set(jf)
            while True:
                nb = sorted({g for f in ratios if ratios[f] < fm and not dead[f] and decidable[f] for g in (f - 1, f + 1)
                             if 0 <= g < Fm and partnered[g] and not dead[g] and g not in seen})
                if not nb:
                    break
                rec_ = scorer.score(nb, res.dx_applied, res.a, res.b)
                final_rec.update(rec_); jf += nb; seen |= set(nb); judged[nb] = True
                for f in nb:
                    if np.isfinite(rec_[f]["ratio"]):
                        ratios[f] = float(rec_[f]["ratio"])
            decidable &= decidable_of(res.dx_applied, res.a, res.b)
            undecidable = sorted(int(f) for f in np.flatnonzero(partnered & ~decidable))
        # a 'weak_win' frame (the served fill fails the gate, the own value too weakly supported to serve) is served the
        # fill and judged by the quorum like a single 'neither' frame — recorded, never a refusal on its own
        res.quality["weak_win_frames"] = sorted(set(int(f) for f in weak_wins))
        # ROUND 10: a ridge-guarded frame served a value that FAILS the gate (interpolated, or the other side's value offered
        # by the boundary pass and lost) is named — 'dx_residual' — never served silently
        ridge_bad = np.zeros(Fm, bool)
        for f in np.flatnonzero(ridge_alias & partnered):
            f = int(f)
            if f in ratios and ratios[f] < fm and decidable[f] and not np.isfinite(forced[f]) and not dead[f]:
                ridge_bad[f] = True
        quorum = [f for f in ratios if not dead[f] and decidable[f]]
        n_frames_eval = len(quorum)
        bad_frames = sorted(f for f in quorum if ratios[f] < fm)
        min_bad = int(p.min_bad_frames) if source == "full" else int(p.min_bad_frames_subset)
        need = max(min_bad, int(np.ceil(float(p.frame_bad_frac) * n_frames_eval)))
        low_frames = n_frames_eval > 0 and len(bad_frames) >= need
        ratio_min = min(ratios.values()) if ratios else float("nan")
        bad_mask = np.zeros(Fm, bool); bad_mask[bad_frames] = True
        bad_runs = [(int(f0), int(f1)) for f0, f1 in _runs(bad_mask) if f1 - f0 >= 2]
        in_bad_run = np.zeros(Fm, bool)
        for f0, f1 in bad_runs:
            in_bad_run[f0:f1] = True
        if (ridge_bad & ~in_bad_run).any():
            residual_runs = sorted(set(residual_runs) | {(int(f0), int(f1)) for f0, f1 in _runs(ridge_bad & ~in_bad_run)})
        # a ridge-blocked single INSIDE a contiguous bad run is named by that run ('low_frame_match'), not twice
        rb_set = {int(f) for f in rb}
        residual_runs = sorted(r_ for r_ in residual_runs if not (r_[1] - r_[0] == 1 and r_[0] in rb_set and in_bad_run[r_[0]]))
        # round 9: with a frame offset of at least seg_hold the overlap boundary is one member's own first / last frames —
        # habitually the poorest of a scan — so a bad run of ≤ 3 frames touching the boundary is not a no-quorum refusal (it
        # still counts toward the quorum; quality['edge_bad_runs']); with df ≈ 0 the round-7 rule stands
        edge_bad: list = []
        if abs(int(df)) >= int(p.seg_hold) and part_idx.size:
            keep_runs = []
            for f0, f1 in bad_runs:
                if f1 - f0 <= 3 and (f0 == int(part_idx[0]) or f1 == int(part_idx[-1]) + 1):
                    edge_bad.append((f0, f1))
                else:
                    keep_runs.append((f0, f1))
            bad_runs = keep_runs
        res.quality["edge_bad_runs"] = edge_bad

        # NEITHER runs: ≥ 2 contiguous frames that scored under neither value, agreeing with each other — a
        # verdict with no quorum; a single such frame is reported and counts toward the quorum above
        def agrees(f: int, g: int) -> bool:
            if bool(C_dx[f]) != bool(C_dx[g]) or bool(C_ab[f]) != bool(C_ab[g]):
                return False
            if C_dx[f] and abs(float(o["dx"][f] - o["dx"][g])) > agree:
                return False
            if C_ab[f] and (abs(float(o["a"][f] - o["a"][g])) > tol or abs(float(o["b"][f] - o["b"][g])) > tol):
                return False
            return True

        neither_final = neither & ~owned & decidable & np.array([f in ratios and ratios[f] < fm for f in range(Fm)], bool)
        for f0, f1 in _runs(neither_final):
            r0 = f0
            for g in range(f0 + 1, f1 + 1):
                if g == f1 or not agrees(g - 1, g):
                    fr_ = list(range(r0, g))
                    if len(fr_) >= 2:
                        if C_dx[r0] and any(sound[x] for x in fr_):
                            low_runs.append((int(r0), int(g)))
                        elif C_dx[r0]:
                            untrusted_runs.append((int(r0), int(g), round(float(np.median(o["dx"][fr_])), 2)))
                        else:
                            axial_runs.append((int(r0), int(g)))
                    else:
                        unscored.append(int(r0))
                    r0 = g
        # round 7, refutation R4: a CONTIGUOUS run of ≥ 2 judged frames failing the gate under the FINAL served value —
        # measured or not — is a verdict with no quorum ('low_frame_match', the run named in quality['low_frame_match_
        # runs']); the quorum stays for scattered single failures.
        covered = [(int(u[0]), int(u[1])) for u in untrusted_runs] + [(int(u[0]), int(u[1])) for u in axial_runs] + [tuple(u) for u in low_runs]
        for f0, f1 in bad_runs:
            if not any(u0 <= f0 and f1 <= u1 for u0, u1 in covered):
                low_runs.append((int(f0), int(f1)))
        if axial_unwitnessed:
            axial_runs = sorted(set(axial_runs) | set(axial_unwitnessed))
        # the pair-level numbers of the FINAL transform, assembled from the per-frame records
        if source == "full":
            recs = scorer.score([int(f) for f in np.flatnonzero(partnered)], res.dx_applied, res.a, res.b)
            agg = _assemble(recs.values(), float(ref_b.match_mask.sum()))
            q = dict(q1); q.update(agg)
            n_ref = float(q["n_ref"])
            for t in NCC_THRESHOLDS:
                q[f"matched_of_ref_{t}"] = (q[f"matched_frac_{t}"] * agg["n_eval"] / max(1.0, n_ref)
                                           if np.isfinite(q[f"matched_frac_{t}"]) else float("nan"))
            cf = float(q["ceiling"].get("matched_frac_0.5", float("nan")))
            q["relative_match"] = q["matched_frac_0.5"] / cf if cf and np.isfinite(cf) and cf > 0 else float("nan")
            q["relative_match_band_space"] = q["relative_match"]
            q["relative_match_orig_space_prototype_A"] = q["matched_frac_0.5"] / q["ceiling_orig_space_prototype_A"]
            q["relative_to_prototype_A_orig_ceiling"] = q["relative_match_orig_space_prototype_A"]
            ffin = np.full(F, np.nan)
            for f, r_ in recs.items():
                ffin[int(f) + df] = r_["frac_0.5"]
            q["frame_frac_0.5"] = ffin
            for k in ("frame_frac_0.3", "frame_frac_0.7", "frame_n_eval", "frame_ncc_mean", "frame_n_ref", "frame_resid_ss", "frame_resid_n"):
                q.pop(k, None)
            res.matched_frac_0_5 = q["matched_frac_0.5"]; res.matched_frac_0_3 = q["matched_frac_0.3"]
            res.coverage = q["coverage"]; res.ncc_mean = q["ncc_mean"]; res.ceiling = dict(q["ceiling"])
            res.relative_match = q["relative_match"]; res.quality = dict(q)
        else:
            fr_ref = [f + df for f in jf]
            n_ref_sub = float(sum(scorer.n_ref_frame[fr] for fr in fr_ref))
            agg = _assemble(final_rec.values(), n_ref_sub)
            c05 = scorer.ceiling_over(fr_ref)
            res.quality = {"subset": {"matched_frac_0.5": agg["matched_frac_0.5"], "matched_frac_0.3": agg["matched_frac_0.3"],
                                      "coverage": agg["coverage"], "ncc_mean": agg["ncc_mean"],
                                      "relative_match": (agg["matched_frac_0.5"] / c05 if np.isfinite(c05) and c05 > 0 else float("nan")),
                                      "n_eval": agg["n_eval"], "n_ref": int(n_ref_sub),
                                      "surface_residual_rms_px": agg["surface_residual_rms_px"], "ceiling_0.5": c05,
                                      "frames": [int(v) for v in fr_ref]}}
    timings["arbitration"] = time.time() - t1
    if scorer is not None:
        timings["scoring"] = scorer.time_s
    timings["quality_total"] = time.time() - t0
    measured_frac = float(kept.sum()) / max(1, n_overlap)
    if kept.sum() < 0.5 * max(1, n_overlap):
        flags.append("few_measured_frames")
    if o["rescued_frames"]:
        flags.append("fine_rescued")
    if o["far_frames"]:
        flags.append("fine_far_search")
    if P["unmeasured_runs"]:
        flags.append("dx_unmeasured_run")
    if arb or plateau_cuts:
        flags.append("arbitrated")
    if weak_segments:
        flags.append("weak_segment")
    if bool(o["recentred"].any()):
        flags.append("fine_recentred")
    if dx_excursions:
        flags.append("dx_excursion")
    if tilt_singles:
        flags.append("tilt_residual_single")
    if tilt_high:
        flags.append("tilt_residual_high")
    if tilt_lowprec["frames"]:
        flags.append("tilt_low_precision")
    if bool(o["wide_searched"].any()):
        flags.append("fine_wide_search")
    if bool((np.asarray(o["shear"]) != 0).any()):
        flags.append("fine_shear_rescued")
    if ridge_runs:
        flags.append("ridge_alias")
    if bool(o["speckle_refined"].any()):
        flags.append("speckle_refined")
    if bool(o["speckle_rescued"].any()):
        flags.append("speckle_rescued")
    if axial_dropped:
        flags.append("axial_dropped")
    poor_match = False; low_rel = False; unjudged = False; n_one = 0
    if scorer is not None:
        qq = res.quality["subset"] if source == "subset" else res.quality
        mf = qq["matched_frac_0.5"]; rel = qq["relative_match"]
        low_m = (not np.isfinite(mf)) or mf < p.low_match
        low_c = qq["coverage"] < p.low_coverage
        if low_m or low_c:
            flags.append("low_match")
        poor_match = low_m and low_c
        n_one = sum(1 for f0, f1, _ in res.dx_segments if f1 - f0 == 1)
        low_rel = (not np.isfinite(mf)) or (np.isfinite(rel) and rel < float(p.min_relative_match))
        if low_rel:
            flags.append("low_relative_match")
        unjudged = n_frames_eval == 0
    # E10: the SPECKLE-scale match of the served transform, reported beside the structure match (each against its own
    # same-scan ceiling; the verdict is the structure match's)
    t2 = time.time()
    if scorer is not None and source == "full":
        res.quality = dict(res.quality)
        res.quality["match_structure"] = {"matched_frac_0.5": float(res.matched_frac_0_5), "matched_frac_0.3": float(res.matched_frac_0_3),
                                          "ncc_mean": float(res.ncc_mean), "coverage": float(res.coverage),
                                          "ceiling_0.5": float(res.ceiling.get("matched_frac_0.5", float("nan"))),
                                          "relative_match": float(res.relative_match), "feature": "struct", "window": list(win),
                                          "sigma": list(ref_b.sigma_struct)}
        if bool(p.speckle_report) and mov_b.feat_speckle is not None and ref_b.feat_speckle is not None:
            try:
                wsp = tuple(p.local_win_speckle)
                cs_sp = ceiling_speckle
                if cs_sp is None or "per_frame_frac_0.5" not in cs_sp:
                    sim_c = band_similarity(ref_b, ref_b, window=wsp, frame_offset=1, feature="speckle")
                    cs_sp = {k: float(v) for k, v in sim_c.stats.items() if not isinstance(v, np.ndarray)}
                    cp = np.full(F, np.nan); cp[sim_c.frames_a] = np.asarray(sim_c.stats["per_frame_frac_0.5"], float)
                    cs_sp["per_frame_frac_0.5"] = cp
                qs, _ = pair_quality(ref_b, mov_b, res, ceiling=cs_sp, window=wsp, feature="speckle")
                res.quality["match_speckle"] = {"matched_frac_0.5": float(qs["matched_frac_0.5"]), "matched_frac_0.3": float(qs["matched_frac_0.3"]),
                                                "ncc_mean": float(qs["ncc_mean"]), "coverage": float(qs["coverage"]),
                                                "ceiling_0.5": float(cs_sp.get("matched_frac_0.5", float("nan"))),
                                                "relative_match": float(qs["relative_match"]), "feature": "speckle", "window": list(wsp),
                                                "sigma": list(ref_b.sigma_speckle)}
            except Exception as e:  # noqa: BLE001 — the report never breaks the verdict
                res.quality["match_speckle"] = {"error": str(e)}
    timings["speckle_report"] = time.time() - t2
    low_sp = False
    msp_ = (res.quality.get("match_speckle") or {}) if isinstance(res.quality, dict) else {}
    if scorer is not None and source == "full" and "relative_match" in msp_ and np.isfinite(msp_["relative_match"]):
        low_sp = float(msp_["relative_match"]) < float(p.min_relative_match_speckle)
        if low_sp:
            flags.append("low_speckle_match")
    # E8: the raster rotation the per-frame dx trend implies (a slope of dx over frames = a rotation about depth)
    raster_deg = None
    tf = np.flatnonzero(trusted & partnered)
    if tf.size >= 8:
        coef = np.polyfit(tf.astype(float), np.asarray(o["dx"][tf], float), 1)
        sp = np.asarray(ref_b.spacing, float)
        if sp.size >= 3 and sp[2] > 0:
            raster_deg = float(np.degrees(np.arctan(float(coef[0]) * float(sp[0]) / float(sp[2]))))
    res.quality = dict(res.quality)
    res.quality["frame_ratios"] = {int(f): round(r_, 3) for f, r_ in ratios.items()}
    res.quality["arbitrated_frames"] = {int(f): v for f, v in sorted(arb.items())}
    res.quality["measured_frac"] = measured_frac
    res.quality["n_overlap_frames"] = int(n_overlap)
    res.quality["decision_source"] = source
    res.quality["fine_widened_frames"] = list(o["widened_frames"])
    res.quality["rescued_frames"] = list(o["rescued_frames"]); res.quality["far_frames"] = list(o["far_frames"])
    res.quality["recentred_frames"] = [int(f) for f in np.flatnonzero(o["recentred"])]
    res.quality["rms_rejected_frames"] = [int(f) for f in np.flatnonzero(o["rms_rejected"])]
    res.quality["dx_mad_segments"] = dx_mad
    res.quality["edge_segments"] = edge_segments; res.quality["beyond_max_segments"] = beyond_segments
    res.quality["beyond_max_frames"] = beyond_frames; res.quality["pinned_contradictions"] = sorted(int(f) for f in pinned_bad)
    res.quality["unmeasured_runs"] = P["unmeasured_runs"]
    res.quality["dead_runs"] = dead_runs; res.quality["ridge_runs"] = ridge_runs
    res.quality["uncorroborated_steps"] = uncorroborated_steps
    res.quality["ab_jitter_px"] = [round(jit_a, 2), round(jit_b, 2)]; res.quality["witness_tol_px"] = [round(w_tol_a, 2), round(w_tol_b, 2)]
    res.quality["dx_residual_runs"] = residual_runs; res.quality["dx_excursions"] = dx_excursions
    res.quality["low_frame_match_runs"] = low_runs; res.quality["dx_untrusted_runs"] = untrusted_runs
    res.quality["axial_residual_runs"] = axial_runs; res.quality["unscored_frames"] = unscored
    res.quality["weak_segments"] = weak_segments; res.quality["weak_contradictions"] = []   # no class drop any more (always empty)
    res.quality["dx_override_runs"] = [(r0, r1, round(v, 3), at_end) for r0, r1, v, at_end in P["override_runs"]]
    res.quality["dx_segments_held"] = [(f0, f1, round(v, 3)) for f0, f1, v in P["dx_segments_held"]]
    res.quality["dx_kept"] = [int(f) for f in np.flatnonzero(P.get("dx_kept", np.zeros(Fm, bool)))]
    res.quality["plateau_cuts"] = plateau_cuts; res.quality["plateau_candidates"] = plateau_rec
    res.quality["boundary_frames"] = boundary; res.quality["propagated_frames"] = propagated; res.quality["joined_frames"] = sorted(set(joined))
    res.quality["arbitration_rounds"] = int(n_rounds)
    res.quality["refill_changed_frames"] = sorted(refill_changed); res.quality["axial_pinned_frames"] = sorted(pinned_ab)
    res.quality["joint_unwitnessed_runs"] = unwitnessed; res.quality["kept_unwitnessed_runs"] = kept_unwitnessed
    res.quality["axial_witness_runs"] = axial_short; res.quality["axial_unwitnessed_runs"] = axial_unwitnessed
    res.quality["axial_dropped_runs"] = axial_dropped
    res.quality["speckle_witness"] = dict(sp_counts)
    res.quality["speckle_refined_frames"] = [int(f) for f in np.flatnonzero(o["speckle_refined"])]
    res.quality["speckle_rescued_frames"] = [int(f) for f in np.flatnonzero(o["speckle_rescued"])]
    _spc = np.asarray(o["speckle_col"], float)
    res.quality["speckle_col_median"] = (round(float(np.nanmedian(_spc[measured0])), 3) if (measured0.any() and np.isfinite(_spc[measured0]).any()) else None)
    res.quality["witness_runs"] = short_runs; res.quality["unanchored_end_singles"] = unanchored_end
    res.quality["rescored_frames"] = sorted({int(f) for rr in rescored_rounds for f in rr})
    res.quality["bad_runs"] = bad_runs; res.quality["fragmented"] = False; res.quality["one_frame_segments"] = int(n_one)
    res.quality["df_reseed"] = c.get("df_reseed"); res.quality["coarse_seeds"] = c.get("seed_scores")
    res.quality["axial_rejected_frames"] = [int(f) for f in np.flatnonzero(C_ab & measured0 & ~win_ab)]
    res.quality["tilt_beyond_max_frames"] = tilt_frames; res.quality["tilt_residual_singles"] = tilt_singles
    res.quality["tilt_residual_high_frames"] = tilt_high; res.quality["dx_far_frames"] = [int(f) for f in np.flatnonzero(dx_far & partnered)]
    res.quality["tilt_alias_frames"] = [int(f) for f in np.flatnonzero(alias_far)]
    res.quality["tilt_residual_median_px"] = (round(float(np.nanmedian(b_resid)), 2) if np.isfinite(b_resid).any() else None)
    res.quality["tilt_residual_p90_px"] = (round(float(np.nanpercentile(b_resid, 90)), 2) if np.isfinite(b_resid).any() else None)
    res.quality["b_lines_median_px"] = (round(float(np.nanmedian(np.abs(b_lines))), 2) if np.isfinite(b_lines).any() else None)
    res.quality["pose_angle_deg"] = float(theta) if np.isfinite(theta) else None
    res.quality["pose_angle_seed_deg"] = float(theta_seed) if np.isfinite(theta_seed) else None
    res.quality["pose_angle_fine_deg"] = float(theta_fine) if np.isfinite(theta_fine) else None
    res.quality["b_lines_served_median_px"] = (round(float(np.nanmedian(np.abs(b_lines_served))), 2) if np.isfinite(b_lines_served).any() else None)
    res.quality["pose_b_lines_median_px"] = float(b_lines_med) if np.isfinite(b_lines_med) else None
    res.quality["tilt_low_precision"] = tilt_lowprec
    res.quality["tilt_se_median_px"] = (round(float(np.nanmedian(b_se[measured0])), 2) if (measured0.any() and np.isfinite(b_se[measured0]).any()) else None)
    res.quality["wide_search_frames"] = [int(f) for f in np.flatnonzero(o["wide_searched"])]
    res.quality["shear_frames"] = {int(f): float(o["shear"][f]) for f in np.flatnonzero(np.asarray(o["shear"]) != 0)}
    res.b_se = b_se.copy(); res.pose_angle_deg = float(theta)
    res.quality["raster_rotation_deg"] = (round(raster_deg, 2) if raster_deg is not None else None)
    res.quality["undecidable_frames"] = undecidable
    res.quality["decidable_frac"] = (float(np.mean([decidable[f] for f in np.flatnonzero(partnered)])) if partnered.any() else float("nan"))
    res.quality["lateral_geometry"] = dict(geom)
    res.quality["trusted_frac"] = float(trusted.sum()) / max(1, n_overlap)
    res.quality["bad_frames"] = bad_frames; res.quality["n_frames_evaluated"] = int(n_frames_eval)
    res.quality["frame_ratio_min"] = ratio_min
    res.quality["n_contradicting"] = int(C.sum()); res.quality["n_dx_contradicting"] = int(C_dx.sum())
    res.quality["n_axial_contradicting"] = int(C_ab.sum())
    res.quality["scorer_calls"] = int(scorer.n_calls) if scorer is not None else 0
    res.quality["scorer_frames"] = int(scorer.n_scored) if scorer is not None else 0
    # the verdicts (REJECT_FLAGS): the fine stage's 'no_correspondence' (too few frames measured, a poor match on
    # both counts, a match below half the reference's own ceiling, or nothing judged), a segment or a pinned frame at
    # the widened search edge, a shift beyond max_dx, a kept residual tilt beyond the cap, an unanchored interior lateral
    # excursion, a run served by neither its measurement nor the fill, a bad minority of frames — never a clamp,
    # never a value served that scores worse than the frame's own measurement
    if unjudged:
        flags.append("unjudged")
    if measured_frac < float(p.min_measured_frac) or poor_match or low_rel or unjudged or low_sp:
        flags.append("no_correspondence")
    if edge_segments or pin_edge:
        flags.append("dx_at_search_edge")
    if beyond_segments or beyond_frames:
        flags.append("dx_beyond_max")
    if tilt_frames:
        flags.append("tilt_beyond_max")
    if residual_runs:
        flags.append("dx_residual")
    if axial_runs:
        flags.append("axial_residual")
    if untrusted_runs:
        flags.append("dx_untrusted_run")
    if low_frames or low_runs:
        flags.append("low_frame_match")
    # PARTIAL OVERLAP (2026-09-12): the overlap the SERVED shift implies (median over the partnered live frames of L − |dx|),
    # the record, and the 'no_overlap' verdict — the served shift leaves fewer laterals / frames than the bar, or the fine
    # stage measured nothing (no overlapping structure); a refused pair also records the overlap-agnostic coarse peak so a
    # match beyond the bar is named with its offset. 'partial_overlap' (informational) under half the moving laterals.
    _sel_ov = partnered & np.asarray(res.live, bool) & np.isfinite(res.dx_applied)
    if not _sel_ov.any():
        _sel_ov = partnered & np.isfinite(res.dx_applied)
    _dx_med = float(np.median(np.abs(res.dx_applied[_sel_ov]))) if _sel_ov.any() else float("nan")
    _ov_served = overlap_of(L, Fm, F, _dx_med, int(df), bar_l, bar_f)
    _ov_served["dx_abs_median"] = _dx_med; _ov_served["n_frames_used"] = int(_sel_ov.sum())
    res.overlap_laterals = float(_ov_served["laterals"]); res.overlap_fraction = float(_ov_served["fraction"])
    ov_rec = {"verdict": "ok", "bar": {"laterals": bar_l, "frame_frac": bar_f, "max_dx": float(p.max_dx), "L": int(L)},
              "seed": c.get("overlap"), "served": _ov_served, "measured_frac": float(measured_frac), "offset": None}
    if np.isfinite(_ov_served["laterals"]) and not _ov_served["admissible"]:
        ov_rec["verdict"] = "below_bar"; ov_rec["offset"] = dict(_ov_served, source="served")
    # BAR EDGE (R1): the OVERLAP-AGNOSTIC coarse peak is computed for EVERY pair now — the judge below needs it, and a refused
    # pair's reason names it (R2, overlap_reason)
    u_ = None
    try:
        t_u = time.time()
        u_ = coarse_unrestricted(ref_b, mov_b, p)
        ov_rec["unrestricted_peak"] = u_
        timings["coarse_unrestricted"] = time.time() - t_u
    except Exception as e:  # noqa: BLE001 — the record never breaks the verdict
        ov_rec["unrestricted_error"] = f"{type(e).__name__}: {e}"
    if "no_correspondence" in flags:
        if measured_frac < float(p.min_measured_frac):
            ov_rec["verdict"] = "none_measured"
        elif ov_rec["verdict"] == "ok":
            ov_rec["verdict"] = "poor_match"
        if u_ is not None and u_.get("beyond_bar") and np.isfinite(u_.get("ncc", np.nan)) and (not np.isfinite(c["ncc"]) or float(u_["ncc"]) > float(c["ncc"])):
            ov_rec["verdict"] = "below_bar"
            ov_rec["offset"] = dict(u_["overlap"], ncc=float(u_["ncc"]), source="unrestricted_peak")
    # BAR EDGE (R1): a pair that is ok so far and whose served shift sits at the search edge is judged against beyond-bar
    # candidates (bar_edge_check) — refused 'dx_at_search_edge' unless the in-bar value demonstrably beats them
    be_rec = None
    if scorer is not None and source == "full" and not (REJECT_FLAGS & set(flags)):
        try:
            t_b = time.time()
            be_rec = bar_edge_check(ref_b, mov_b, res, p, ceiling=ceiling, window=tuple(win), unrestricted=u_)
            timings["bar_edge"] = time.time() - t_b
            if be_rec["verdict"] == "refused":
                flags.append("dx_at_search_edge")
                bb = [int(f) for f in be_rec["beyond_bar_frames"]]
                res.live = np.asarray(res.live, bool).copy(); res.measured = np.asarray(res.measured, bool).copy()
                for f in bb:
                    res.live[f] = False; res.measured[f] = False
                n_part = int(partnered.sum())
                if len(bb) > (1.0 - bar_f) * max(1, n_part):
                    ov_rec["verdict"] = "below_bar"
                    if "no_overlap" not in flags:
                        flags.append("no_overlap")
                best_ = be_rec.get("best")
                if best_ is not None:
                    ov_rec["offset"] = dict(overlap_of(L, Fm, F, float(best_["dx"]), int(df), bar_l, bar_f), source="bar_edge_judge",
                                            score=float(best_["score"]), served_score=float(best_["served_score"]), served_dx=float(best_["served_dx"]),
                                            scope=str(best_["scope"]), candidate_source=str(best_["source"]))
                elif u_ is not None and u_.get("beyond_bar") and u_.get("overlap"):
                    ov_rec["offset"] = dict(u_["overlap"], ncc=float(u_.get("ncc", np.nan)), source="unrestricted_peak")
            elif be_rec["verdict"] == "confirmed":
                flags.append("near_search_edge")
        except Exception as e:  # noqa: BLE001 — a judge that cannot run is a refusal, never a silent ok
            be_rec = {"verdict": "error", "error": f"{type(e).__name__}: {e}"}
            flags.append("dx_at_search_edge")
    res.quality["bar_edge"] = be_rec
    res.quality["beyond_bar_frames"] = list((be_rec or {}).get("beyond_bar_frames") or [])
    if ov_rec["verdict"] in ("below_bar", "none_measured") and "no_overlap" not in flags:
        flags.append("no_overlap")
    if np.isfinite(res.overlap_fraction) and 0.0 < res.overlap_fraction < 0.5:
        flags.append("partial_overlap")
    res.quality["overlap"] = ov_rec
    if pose_flag:
        # E8 / ROUND 10 (R1): a pose beyond pose_max_deg is a REPORT ('pose_high') when the per-frame rigid model demonstrably
        # fits — no REJECT verdict and rel_struct ≥ pose_fit_min_rel (a tilt about the frame axis IS the per-frame model: a
        # decentring of 150-250 laterals on a dome implies 8.5-15° and registers to 2 laterals / 2 px); otherwise the verdict
        # 'pose_beyond_frame_rigid' supersedes the match verdicts when the pair DOES overlap (rel_struct ≥ pose_overlap_min:
        # P5_OS v1 at 15° / 0.73), and a pair that also fails the overlap bar keeps its 'no_correspondence' (the cross-patient
        # controls: 33° / 16° at 0.56-0.65)
        rel_now = float(res.relative_match) if np.isfinite(res.relative_match) else float("nan")
        if not np.isfinite(rel_now) and source == "subset":
            rel_now = float((res.quality.get("subset") or {}).get("relative_match", float("nan")))
        rejects_now = [fl for fl in flags if fl in REJECT_FLAGS]
        fits = (not rejects_now) and (not low_sp) and np.isfinite(rel_now) and rel_now >= float(p.pose_fit_min_rel)
        if fits:
            res.quality["superseded_flags"] = []
            flags.append("pose_high")
        else:
            overlaps = (not (poor_match or unjudged)) and np.isfinite(rel_now) and rel_now >= float(p.pose_overlap_min)
            sup = [fl for fl in flags if fl in _POSE_SUPERSEDES] if overlaps else []
            res.quality["superseded_flags"] = sup
            flags[:] = [fl for fl in flags if fl not in sup] + ["pose_beyond_frame_rigid"]
    timings["total"] = time.time() - t_all
    return res


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Element 4 — group consistency
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def transitivity_check(pair_1r: PairResult, pair_2r: PairResult, pair_21: PairResult,
                       tol: tuple[float, float, float] = (1.0, 2.0, 2.0)) -> dict:
    """Design element 4's diagnostic: compose mov2→mov1 (pair_21) with mov1→ref (pair_1r) and compare with the
    direct mov2→ref (pair_2r) on the frames all three measured. df must agree exactly; the composed axial move
    of moving-2 lateral l is a_21 + a_1r + b_1r·dx_21/hs + (b_21 + b_1r)·x(l) (the lateral shift moves x by
    dx_21/hs before the second tilt), compared as rms/max over frames of a, b and the per-frame dx (and the
    applied per-frame dx). Prototype A (CS001_OS, 92 frames): a rms 0.90 px, b rms 1.63 px, dx rms 1.59 laterals;
    `ok` = df exact and (a rms, b rms, dx rms) within tol = (1, 2, 2). All three pairs must live on ONE lateral
    grid (register_group's common grid)."""
    F2 = pair_2r.n_frames; F1 = pair_1r.n_frames
    L = int(pair_2r.shape[0]); hs = max(1.0, (L - 1) / 2.0)
    da, db, ddx, ddxa = [], [], [], []
    for f in range(F2):
        f1 = f + int(pair_21.df)
        if not (0 <= f1 < F1):
            continue
        if not (pair_2r.measured[f] and pair_21.measured[f] and pair_1r.measured[f1]):
            continue
        dx21 = float(pair_21.dx_applied[f])
        a_c = pair_21.a[f] + pair_1r.a[f1] + pair_1r.b[f1] * dx21 / hs
        b_c = pair_21.b[f] + pair_1r.b[f1]
        da.append(pair_2r.a[f] - a_c); db.append(pair_2r.b[f] - b_c)
        ddx.append(pair_2r.dx_per_frame[f] - (pair_21.dx_per_frame[f] + pair_1r.dx_per_frame[f1]))
        ddxa.append(pair_2r.dx_applied[f] - (dx21 + pair_1r.dx_applied[f1]))
    rms = lambda v: float(np.sqrt(np.nanmean(np.square(v)))) if len(v) and np.isfinite(v).any() else float("nan")  # noqa: E731
    mx = lambda v: float(np.nanmax(np.abs(v))) if len(v) and np.isfinite(v).any() else float("nan")  # noqa: E731
    df_c = int(pair_21.df) + int(pair_1r.df)
    out = {"frames_compared": len(da), "df_direct": int(pair_2r.df), "df_composed": df_c, "df_ok": int(pair_2r.df) == df_c,
           "a_rms_px": rms(np.asarray(da, float)), "a_max_px": mx(np.asarray(da, float)), "b_rms_px": rms(np.asarray(db, float)),
           "b_max_px": mx(np.asarray(db, float)), "dx_rms": rms(np.asarray(ddx, float)), "dx_max": mx(np.asarray(ddx, float)),
           "dx_applied_rms": rms(np.asarray(ddxa, float)), "dx_applied_max": mx(np.asarray(ddxa, float)),
           "pair_21_ncc_coarse": float(pair_21.ncc_coarse), "pair_21_flags": list(pair_21.flags), "tol": list(tol)}
    out["ok"] = bool(out["df_ok"] and len(da) > 0 and out["a_rms_px"] <= tol[0] and out["b_rms_px"] <= tol[1]
                     and out["dx_rms"] <= tol[2])
    return out


@dataclass
class GroupResult:
    """register_group output — behaves as the design's {cid: PairResult} mapping (non-reference members only)
    and carries the reference cid, the member list, the reference's adjacent-frame ceiling (structure feature),
    the transitivity diagnostics (one dict per checked triple, transitivity_check) and timings. ROUND 9:
    `non_contributing` {cid: reason} — members whose pair with the reference is 'pose_beyond_frame_rigid' (E8) or
    refused; `pose` {(mov, ref): theta degrees} over every ORDERED pair (the pose matrix the reference choice used);
    `lateral_grid` {cid: {scale, offset, L_out, resampled}} (E4); `ceiling_speckle` the reference's speckle ceiling."""
    reference: str
    members: list
    pairs: dict
    ceiling: dict
    transitivity: list
    timings: dict
    non_contributing: dict = field(default_factory=dict)
    pose: dict = field(default_factory=dict)
    lateral_grid: dict = field(default_factory=dict)
    ceiling_speckle: dict | None = None
    reference_rule: dict = field(default_factory=dict)

    def __getitem__(self, cid: str) -> PairResult:
        return self.pairs[cid]

    def __contains__(self, cid) -> bool:
        return cid in self.pairs

    def __iter__(self):
        return iter(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def keys(self):
        return self.pairs.keys()

    def items(self):
        return self.pairs.items()

    def values(self):
        return self.pairs.values()

    def summary(self) -> dict:
        return {"reference": self.reference, "members": list(self.members),
                "pairs": {cid: r.summary() for cid, r in self.pairs.items()},
                "ceiling": {k: (round(float(v), 4) if not isinstance(v, np.ndarray) else None) for k, v in self.ceiling.items()},
                "ceiling_speckle": ({k: (round(float(v), 4) if not isinstance(v, np.ndarray) else None) for k, v in self.ceiling_speckle.items()}
                                    if self.ceiling_speckle else None),
                "transitivity": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()} for t in self.transitivity],
                "non_contributing": dict(self.non_contributing),
                "pose": {f"{m}->{r}": (round(float(v), 2) if v is not None and np.isfinite(v) else None) for (m, r), v in self.pose.items()},
                "lateral_grid": dict(self.lateral_grid), "reference_rule": dict(self.reference_rule),
                "timings": {k: round(float(v), 2) for k, v in self.timings.items()}}


def overlap_note(r: PairResult) -> str:
    """PARTIAL OVERLAP (2026-09-12): ' (no_overlap: …)' for a refused pair whose overlap is below the bar or unmeasured, ' (at the
    search edge: …)' after a bar-edge refusal, where … is overlap_reason — R2: 'best correspondence at ≈ −449 laterals (12%
    overlap, below the 19% bar)' when the record holds a correspondence beyond the bar, else 'offset ≈ −403 laterals, overlap
    21%'; ' (overlap 21%)' for any other pair with a measured overlap; '' when nothing is known."""
    q = r.quality if isinstance(r.quality, dict) else {}
    ov = q.get("overlap") or {}
    frac = float(r.overlap_fraction) if r.overlap_fraction is not None else float("nan")
    reason = overlap_reason(ov, r.flags, dx_median=None, overlap_fraction=None)
    if "no_overlap" in r.flags:
        return f" (no_overlap: {reason})" if reason else " (no_overlap)"
    if "dx_at_search_edge" in r.flags and (q.get("bar_edge") or {}).get("verdict") in ("refused", "error") and reason:
        return f" (at the search edge: {reason})"
    if reason.startswith("best correspondence"):
        return f" ({reason})"
    if np.isfinite(frac):
        return f" (overlap {frac:.0%})"
    return ""


def _band_area(b: BandData) -> int:
    return int(b.valid.sum()) if b.valid is not None else int(b.mask.any(axis=1).sum())


def _feature_ceiling(ref_b: BandData, window, feature: str) -> dict:
    """The reference's own adjacent-frame ceiling on one feature, with the per-frame array the per-frame gate uses."""
    csim = band_similarity(ref_b, ref_b, window=tuple(window), frame_offset=1, feature=feature)
    cst = csim.stats
    ceiling = {k: float(v) for k, v in cst.items() if not isinstance(v, np.ndarray)}
    c_pairs = np.full(ref_b.n_frames, np.nan)                  # by reference frame f: the pair (f, f + 1)
    c_pairs[csim.frames_a] = np.asarray(cst["per_frame_frac_0.5"], float)
    ceiling["per_frame_frac_0.5"] = c_pairs
    return ceiling


def register_group(members: Iterable, reference: str | None = None, params: PairParams | dict | None = None, *,
                   band_rows: tuple[int, int] = BAND_ROWS_DEFAULT, transitivity: bool = True, max_triples: int = 3,
                   quality: bool = True) -> GroupResult:
    """Design element 4: register every member of a group onto the reference with register_pair, the reference's
    adjacent-frame ceilings (structure and speckle) computed ONCE and shared by every pair, then the transitivity
    diagnostic for up to max_triples pairs of non-reference members (mov2→mov1 registered with the CHEAP quality
    check — quality=False — composed with mov1→ref, compared with mov2→ref; transitivity_check). ROUND 9: (E4)
    MemberData members are put on ONE lateral grid — the provisional reference's physical spacing (the header
    ratio; common_lateral_grid), every member centred on a common canvas — before the bands are extracted, so every
    pair and every transitivity triple composes on the same grid; (E8) the coarse stage runs on every ORDERED pair
    and the pose angle the served lines demand at each seed is tabulated (`pose`): when `reference` is None the
    reference is the member with the MOST partners within pose_max_deg, ties → the largest valid area → the most
    central dome (choose_reference's rule as the tie-break) — P5_OS: v1 (≈ 15° against its three siblings) can no
    longer be the reference the largest-area rule picked; a member whose pair is 'pose_beyond_frame_rigid' is
    NON-CONTRIBUTING (`non_contributing`, registered for the record). `members` are MemberData (bands extracted
    here) or BandData with served lines (used as they are; no grid change); a member whose stage found no
    correspondence is skipped from the triples."""
    t_all = time.time(); timings: dict = {}
    ms = list(members)
    if not ms:
        raise ValueError("register_group: no members")
    p = _pair_params(params)
    t0 = time.time()
    mdata: list[MemberData] = []
    bdata: dict[str, BandData] = {}
    for m in ms:
        if isinstance(m, MemberData):
            mdata.append(m)
        elif isinstance(m, BandData):
            if m.served is None:
                raise ValueError(f"register_group: band {m.cid} carries no served line")
            bdata[m.cid] = m
        else:
            raise TypeError(f"register_group: expected MemberData or BandData, got {type(m).__name__}")
    all_members = len(mdata) == len(ms)
    cids = [m.cid for m in ms]
    if reference is not None and reference not in cids:
        raise ValueError(f"register_group: reference {reference!r} is not a member of {cids}")
    grid_rec: dict = {}
    bands: dict[str, BandData] = {}

    def build_bands(ref_cid: str) -> None:
        nonlocal grid_rec, bands
        bands = {}
        if all_members:
            grid_members, grid_rec = common_lateral_grid(mdata, ref_cid, float(p.lateral_scale_tol))
            for gm in grid_members:
                bands[gm.cid] = _as_band(gm, band_rows, p)
        else:
            for m in ms:
                bands[m.cid] = bdata[m.cid] if m.cid in bdata else _as_band(m, band_rows, p)
            grid_rec = {c: {"scale": 1.0, "offset": 0, "L_out": None, "resampled": False} for c in bands}
    # the PROVISIONAL reference (the round-8 rule) fixes the lateral grid; the pose rule may move it
    provisional = reference if reference is not None else (choose_reference(mdata) if all_members
                                                              else max(cids, key=lambda c: (_band_area(bdata[c]) if c in bdata else 0, c)))
    build_bands(provisional)
    timings["bands"] = time.time() - t0
    # E8: the coarse stage on every ORDERED pair → the pose matrix (and the seeds reused by register_pair)
    t0 = time.time()
    coarse_cache: dict = {}; pose: dict = {}
    for r_ in cids:
        for m_ in cids:
            if m_ == r_:
                continue
            c = coarse_register(bands[r_], bands[m_], p)
            theta, _, _ = _pose_of(bands[r_], bands[m_], c, p)
            coarse_cache[(m_, r_)] = c; pose[(m_, r_)] = float(theta)
    timings["pose_matrix"] = time.time() - t0
    rule: dict = {"provisional": provisional, "pose_max_deg": float(p.pose_max_deg)}
    if reference is None:
        partners = {c: sum(1 for o_ in cids if o_ != c and np.isfinite(pose[(o_, c)]) and pose[(o_, c)] <= float(p.pose_max_deg)) for c in cids}
        rule["partners_within_pose"] = dict(partners)
        best_n = max(partners.values())
        cand = [c for c in cids if partners[c] == best_n]
        if len(cand) == 1:
            reference = cand[0]
        elif all_members:
            reference = choose_reference([m for m in mdata if m.cid in cand])
        else:
            reference = max(cand, key=lambda c: (_band_area(bands[c]), c))
        rule["reference"] = reference
        if reference != provisional and all_members:
            # the grid was the provisional reference's spacing: re-grid when the final reference's differs
            sp_prov = float(next(m.spacing[0] for m in mdata if m.cid == provisional))
            sp_ref = float(next(m.spacing[0] for m in mdata if m.cid == reference))
            if abs(sp_ref / sp_prov - 1.0) > float(p.lateral_scale_tol):
                t0 = time.time()
                build_bands(reference)
                coarse_cache = {}
                for m_ in cids:
                    if m_ != reference:
                        c = coarse_register(bands[reference], bands[m_], p)
                        coarse_cache[(m_, reference)] = c
                        pose[(m_, reference)] = float(_pose_of(bands[reference], bands[m_], c, p)[0])
                timings["regrid"] = time.time() - t0
    t0 = time.time()
    ref_b = bands[reference]
    ceiling = _feature_ceiling(ref_b, tuple(p.local_win), "struct")
    ceiling_sp = (_feature_ceiling(ref_b, tuple(p.local_win_speckle), "speckle")
                  if (bool(p.speckle_report) and ref_b.feat_speckle is not None) else None)
    timings["ceiling"] = time.time() - t0
    pairs: dict[str, PairResult] = {}
    others = [c for c in cids if c != reference]
    non_contrib: dict = {}
    for c in others:
        t0 = time.time()
        pairs[c] = register_pair(ref_b, bands[c], p, ceiling=ceiling, ceiling_speckle=ceiling_sp, quality=quality,
                                 coarse=coarse_cache.get((c, reference)))
        if grid_rec.get(c, {}).get("resampled") and "lateral_resampled" not in pairs[c].flags:
            pairs[c].flags.insert(0, "lateral_resampled")
        timings[f"pair {c}"] = time.time() - t0
        if "pose_beyond_frame_rigid" in pairs[c].flags:
            non_contrib[c] = f"pose_beyond_frame_rigid ({pairs[c].pose_angle_deg:.1f} deg > {p.pose_max_deg} deg)"
        elif not pairs[c].ok:
            non_contrib[c] = "refused: " + ", ".join(fl for fl in pairs[c].flags if fl in REJECT_FLAGS) + overlap_note(pairs[c])
    trans: list[dict] = []
    if transitivity and len(others) >= 2:
        from itertools import combinations
        n_done = 0
        for c1, c2 in combinations(others, 2):
            if n_done >= int(max_triples):
                break
            if not (pairs[c1].ok and pairs[c2].ok):
                continue
            t0 = time.time()
            p21 = register_pair(bands[c1], bands[c2], p, quality=False, coarse=coarse_cache.get((c2, c1)))
            if not p21.ok and quality:
                # ROUND 10: the cheap subset path judges ~12 frames plus every contradiction and refuses on a single decisive
                # excursion it cannot corroborate; the FULL path decides the transitivity pair before the composition is given up
                p21_full = register_pair(bands[c1], bands[c2], p, quality=True, coarse=coarse_cache.get((c2, c1)))
                if p21_full.ok:
                    p21 = p21_full
            rec = {"ref": reference, "mov1": c1, "mov2": c2, "pair_21_decision_source": p21.quality.get("decision_source"),
                   "pair_21_subset": p21.quality.get("subset")}
            if p21.ok:
                rec.update(transitivity_check(pairs[c1], pairs[c2], p21))
                # ROUND 10: the GROUP as a df witness — a transitivity pair whose df disagrees with the composition of the two
                # direct pairs by ≥ 2 frames is re-run seeded at the composed df (dx0 = the composed per-frame dx median); the
                # re-run replaces it when it registers as well (its subset relative match within 0.05 of the original's)
                df_c = int(rec["df_composed"]); df_d = int(rec["df_direct"])
                p1r, p2r = pairs[c1], pairs[c2]
                df21_seed = int(p2r.df) - int(p1r.df)             # the transitivity pair's df the two DIRECT pairs imply
                if bool(p.group_df_reseed) and abs(df_c - df_d) >= 2 and coarse_cache.get((c2, c1)) is not None:
                    comp = []
                    for f in range(p2r.n_frames):
                        f1 = f + df21_seed
                        if 0 <= f1 < p1r.n_frames and p2r.measured[f] and p1r.measured[f1]:
                            comp.append(float(p2r.dx_applied[f]) - float(p1r.dx_applied[f1]))
                    if len(comp) >= 5:
                        c21 = dict(coarse_cache[(c2, c1)])
                        seed = {"dx0": float(np.median(comp)), "dz0": float(c21.get("dz0", 0.0)), "df0": df21_seed, "ncc": float("nan"), "on_bound": False, "source": "group_df"}
                        c21.update(dx0=seed["dx0"], df0=df21_seed, seed_source="group_df", seeds=[seed])
                        t1 = time.time()
                        p21b = register_pair(bands[c1], bands[c2], p, quality=(p21.quality.get("decision_source") == "full"), coarse=c21)
                        def _rel(pp):
                            v = float(pp.relative_match) if np.isfinite(pp.relative_match) else float("nan")
                            return v if np.isfinite(v) else float((pp.quality.get("subset") or {}).get("relative_match", float("nan")))
                        rel_a = _rel(p21); rel_b = _rel(p21b)
                        rec["df_group_reseed"] = {"df_seed": df21_seed, "df_original": int(p21.df), "dx0_seed": round(seed["dx0"], 2), "ok": bool(p21b.ok),
                                                  "rel_original": rel_a, "rel_reseeded": rel_b, "time_s": time.time() - t1, "accepted": False}
                        if p21b.ok and int(p21b.df) == df21_seed and np.isfinite(rel_b) and (not np.isfinite(rel_a) or rel_b >= rel_a - 0.05):
                            p21 = p21b
                            rec.update(transitivity_check(pairs[c1], pairs[c2], p21))
                            rec["df_group_reseed"]["accepted"] = True
                            rec["pair_21_decision_source"] = p21.quality.get("decision_source"); rec["pair_21_subset"] = p21.quality.get("subset")
            else:
                rec.update({"ok": False, "frames_compared": 0, "pair_21_flags": list(p21.flags),
                            "pair_21_ncc_coarse": float(p21.ncc_coarse)})
            rec["time_s"] = time.time() - t0
            trans.append(rec); n_done += 1
        timings["transitivity"] = sum(t["time_s"] for t in trans)
    timings["total"] = time.time() - t_all
    return GroupResult(reference=reference, members=cids, pairs=pairs, ceiling=ceiling, transitivity=trans,
                       timings=timings, non_contributing=non_contrib, pose=pose, lateral_grid=grid_rec,
                       ceiling_speckle=ceiling_sp, reference_rule=rule)


__all__ = [
    "MemberData", "BandData", "SimilarityResult", "load_member", "load_group", "choose_reference", "extract_band",
    "band_similarity", "local_ncc", "sample_band", "scar_proxy", "attach_scar_proxy", "posterior_trace_free",
    "run_applied_move", "measured_applied_move", "detect_corrected_surface", "group_members", "group_key",
    "group_id_norm", "parse_case_meta", "resolve_case_dir", "case_local_path", "display_slice_to_lateral",
    "masked_ncc_fft", "prep_fixed", "subpix_peak", "block_mean", "fft_shape", "noise_floor",
    "resample_member_lateral", "common_lateral_grid", "pose_angle_deg",
    "PairParams", "PairResult", "GroupResult", "register_pair", "register_group", "coarse_register", "fine_register",
    "coarse_unrestricted", "overlap_of", "overlap_note", "overlap_reason", "bar_edge_check",
    "pair_quality", "warp_band", "transitivity_check", "split_segments", "live_frames", "robust_line",
    "BAND_ROWS_DEFAULT", "R_SKIP", "LOCAL_WIN", "LOCAL_WIN_STRUCT", "LOCAL_WIN_SPECKLE", "FEATURE_SIGMA_STRUCT",
    "FEATURE_SIGMA_SPECKLE", "COARSE_DS", "COARSE_ROWS_DEFAULT", "COARSE_MASK_DEFAULT",
    "MIN_POSTERIOR_THICKNESS", "MIN_POSTERIOR_P10",
    "PROTOTYPE_A_CEILING", "BAND_SPACE_CEILING", "REJECT_FLAGS",
]
