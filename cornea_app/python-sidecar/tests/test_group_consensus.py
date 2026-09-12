"""Consensus v2 revision 3 (group_consensus + group_job's wiring) — tests RE-BASED to what n voters can GUARANTEE
(the synthetic-truth refutation of 2026-09-11, verifier journal wf_46b7702a): the honest majority inherits its voters'
own errors, so the bar is "κ* within the largest HONEST replicate's own error", never "5 % on 90 %".

Two generators:
  * model L (here): tissue exact, the dome error only in the LINE (the original implementer model; kept for the
    rule-level tests: coverage, vote-only, legacy scale, algebra, cache).
  * model T (tests/_truth_model.py = the refuter's refute_truth.py, copied verbatim): every replicate's TISSUE carries a
    smooth per-frame rigid error (quadratic err_px at the frame ends + cubic + wobble + tilt error), lines follow the
    tissue (+ noise, spikes), pairs = exact tissue-to-tissue rigid transforms, crops, a saccade, single-coverage regions.
    The guarantees are proven on model T.

Guarantees tested
  n = 3 (CS001's n): κ* = the per-lateral MEDIAN → |κ* − κ_true| ≤ Δκ(h + 1 px) on ≥ 95 % of the 3-vote laterals for the
        refuter's sweep h ∈ {3, 5, 8} × w ∈ {8, 15, 22, 30} (h = the honest ±error, w = the wrong member's; +1 px = the
        honest replicates' own cubic / wobble / noise leaking into their quadratic fit); the wrong member never decides
        alone when w > h; every member's tissue after the apply within ~0.35·(h + 1) + 0.2 px RMS of the truth.
  n = 4: the wrong member is never in the majority; κ* within Δκ(max honest + 1 px) on the 4-vote laterals; a wrong
        ANCHOR is out-voted too and the calibrated axial tie-break (κ_axial·ρ) never picks the anchor on a 2-vote lateral.
  apply: the APPLIED δa / δb are smooth along frames (max |d²| ≤ 0.1 / 0.2 px) and the dome part of the residual ≤ 1 px;
        a 10 px LINE dip on 6 frames changes δa by ≤ 0.1 px (line error never moves tissue) and is flagged.
  reference sensitivity on model T: the shape spread (beyond a pose + a smooth dome move) is small; the raw spread is the
        anchors' own dome error (its dome part in κ equals the injected difference).
  the shared wrong dome (every replicate's along-frame curvature 1.6× / 0.6× the across-lateral one) is flagged by the
        axial witness; a legitimate toricity 1.28 is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

import group_align as ga
import group_consensus as gc_
import group_job as gj

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))          # tests/ is not on sys.path under this pytest config
import _truth_model as tm  # noqa: E402  (the refuter's model-T generator, copied verbatim)

# ── model L (the original implementer generator) ─────────────────────────────────────────────────────────────
L, D, F = 257, 8, 81
SPACING = (0.0115, 0.0031, 0.04)          # mm: lateral (companion-derived, trusted), depth, frame
KAPPA_F, KAPPA_L = 0.11, 0.10             # 1/mm along frames / across laterals (toric within 10 %)
C2_F = KAPPA_F * SPACING[2] ** 2 / (2 * SPACING[1])      # px / frame²
C2_L = KAPPA_L * SPACING[0] ** 2 / (2 * SPACING[1])      # px / lateral²
Z0 = 120.0
FIT_MARGIN_PX = 1.0                        # the honest replicates' cubic (0.4–0.8 px) + wobble (0.5 px) + noise leak into c2


def truth(l, f):
    """The true dome in the reference's coordinates (rows)."""
    l = np.asarray(l, float); f = np.asarray(f, float)
    return Z0 + C2_L * (l - (L - 1) / 2.0) ** 2 + C2_F * (f - (F - 1) / 2.0) ** 2


def pose(df: int, dx: float, a0: float, a1: float, b: float) -> dict:
    fr = np.arange(F, dtype=float)
    return {"df": int(df), "dx": np.full(F, float(dx)), "a": a0 + a1 * (fr - (F - 1) / 2.0), "b": np.full(F, float(b))}


POSES = {"R": pose(0, 0.0, 0.0, 0.0, 0.0), "A": pose(3, -12.0, 6.0, 0.05, 3.0), "B": pose(-2, 8.0, -5.0, -0.02, -2.5)}


def make_member(cid: str, T: dict, *, beta: float, alpha: float = 0.0, tilt0: float = 0.0, tilt1: float = 0.0, seed: int = 0,
                spacing=SPACING, crop=None, kappa_f_scale: float = 1.0) -> ga.MemberData:
    """Served line = the member's TISSUE dome in its own coordinates (the truth pulled back through its pose) + a smooth
    per-frame dome error alpha + beta (f − fc)² + (tilt0 + tilt1 (f − fc))·x(l) + 0.4 px noise (model L: the error is in
    the LINE only). kappa_f_scale ≠ 1 scales the along-frame curvature of the TISSUE itself (a shared wrong dome)."""
    rng = np.random.default_rng(seed)
    lat = np.arange(L, dtype=float); fr = np.arange(F, dtype=float)
    LL, FF = np.meshgrid(lat, fr, indexing="ij")
    hs = (L - 1) / 2.0
    z_true = (Z0 + C2_L * (LL + T["dx"][None, :] - hs) ** 2 + kappa_f_scale * C2_F * (FF + T["df"] - (F - 1) / 2.0) ** 2)
    tissue = z_true - T["a"][None, :] - T["b"][None, :] * (LL - hs) / hs
    u = FF - (F - 1) / 2.0
    err = alpha + beta * u ** 2 + (tilt0 + tilt1 * u) * (LL - hs) / hs
    served = tissue + err + rng.normal(0.0, 0.4, size=tissue.shape)
    valid = np.ones((L, F), bool)
    if crop:
        for (l0, l1, f0, f1) in crop:
            valid[l0:l1, f0:f1] = False
    return ga.MemberData(cid=cid, case_dir=Path("/nonexistent"), group="syn", volume=np.zeros((L, D, F), np.float32),
                         served=served, valid=valid, spacing=np.asarray(spacing, float))


BETAS_L = {"R": 0.0015, "A": -0.0012, "B": 0.0010}       # ±2–3 px at the frame ends: independent small errors
HF_L = (F - 1) / 2.0
H_MAX_L_PX = max(abs(b) for b in BETAS_L.values()) * HF_L ** 2   # the largest honest error of model L (2.4 px)


def group(wrong: str | None = "A", *, shared_scale: float = 1.0, spacing=SPACING, beta_wrong: float = 0.011) -> tuple[dict, dict]:
    """(members by cid, transforms by cid) — R the anchor, A / B moved by their poses; `wrong` gets the 15–20 px dome."""
    betas = dict(BETAS_L)
    if wrong:
        betas[wrong] = beta_wrong                              # 0.011 × 40² ≈ 17.6 px at the frame ends
    crops = {"R": [(100, 111, 30, 41)], "A": [(0, 20, 0, F), (0, L, 0, 5)], "B": [(L - 30, L, 0, F), (0, L, F - 8, F)]}
    tilts = {"R": (0.3, 0.005), "A": (-0.4, 0.01), "B": (0.5, -0.008)}
    ms = {c: make_member(c, POSES[c], beta=betas[c], alpha=0.5 * i, tilt0=tilts[c][0], tilt1=tilts[c][1], seed=10 + i, spacing=spacing,
                         crop=crops[c], kappa_f_scale=shared_scale) for i, c in enumerate(["R", "A", "B"])}
    return ms, {c: dict(POSES[c]) for c in ms}


def placed(ms: dict, trs: dict, order=("R", "A", "B")):
    members = [ms[c] for c in order if c in trs]
    canvas = gj.union_canvas(members, trs)
    Lc, _Dc, Fc = canvas["shape"]
    lines = {}; colmask = np.zeros((Lc, Fc), bool); cols = {}
    for m in members:
        _v, _k, line, col = gj.place_on_canvas(m, trs[m.cid], canvas, volume=False)
        lines[m.cid] = line; colmask |= col; cols[m.cid] = col
    return members, canvas, lines, colmask, cols


def truth_on_canvas(canvas: dict) -> np.ndarray:
    l0, z0, f0 = canvas["origin"]; Lc, _Dc, Fc = canvas["shape"]
    LL, FF = np.meshgrid(np.arange(Lc) + l0, np.arange(Fc) + f0, indexing="ij")
    return truth(LL, FF) - z0


def run_v2(ms, trs, vote_only=None):
    members, canvas, lines, colmask, cols = placed(ms, trs)
    cons = gc_.consensus_v2(ms["R"], members, trs, lines, colmask, canvas, vote_only=vote_only,
                            pair_info={c: {"ok": True, "rel": 1.2, "ncc_coarse": 0.9} for c in trs if c != "R"})
    return cons, members, canvas, lines, colmask, cols


# ── model T helpers ───────────────────────────────────────────────────────────────────────────────────────────
def dk(px: float) -> float:
    """Δκ (1/mm) of a dome error of `px` at the frame ends of the model-T frames."""
    return float(gc_.kappa_frames(px / tm.HF ** 2, tm.SP))


def kappa_error(run) -> np.ndarray:
    """κ* − κ_true per canvas lateral (model T: the truth's c2_f(l) carries the asphericity)."""
    canvas = run["canvas"]; l0 = canvas["origin"][0]; Lc = canvas["shape"][0]
    return run["cons"]["kappa_star"] - gc_.kappa_frames(tm.c2f_true(np.arange(Lc) + l0), tm.SP)


def build3(h: float, w: float) -> tuple[dict, dict]:
    """Three model-T replicates: R −h, B +h (honest), A +w (wrong) px at the frame ends; no saccade."""
    pose_B = dict(tm.POSE["B"]); pose_B.pop("saccade", None)
    err = {"R": -h, "B": h, "A": w}
    reps = {}
    for i, c in enumerate(["R", "A", "B"]):
        p = pose_B if c == "B" else tm.POSE[c]
        reps[c] = tm.make_rep(c, **p, err_px=err[c], gamma_px=tm.GAMMA[c], wob_px=tm.WOB[c], tilt_err=tm.TILT_ERR[c], crops=tm.CROPS[c], seed=100 + i)
    return reps, {c: tm.pair_to(reps[c], reps["R"]) for c in reps}


def assert_smooth(fin: dict) -> None:
    for c, t in fin.items():
        sz = t["sizes"]
        assert sz["smooth_ok"], (c, sz["delta_a_d2_max"], sz["delta_b_d2_max"])
        assert sz["delta_a_d2_max"] <= gj.SMOOTH_D2_A_PX and sz["delta_b_d2_max"] <= gj.SMOOTH_D2_B_PX, (c, sz["delta_a_d2_max"], sz["delta_b_d2_max"])
        # the applied field IS a_final − a_pair (what the verifier checks) and holds outside the covered frames
        cov = t["covered"]
        assert np.allclose(t["a"][cov] - t["a_pair"][cov], t["delta_a"][cov]) and np.allclose(t["b"][cov] - t["b_pair"][cov], t["delta_b"][cov])
        held = ~cov
        assert np.allclose(t["a"][held], t["a_pair"][held], equal_nan=True) and np.isnan(t["delta_a"][held]).all()


# ── n = 3: the median is bounded by the largest HONEST error (the refuter's h / w sweep) ──────────────────────
@pytest.mark.parametrize("h", [3.0, 5.0, 8.0])
@pytest.mark.parametrize("w", [8.0, 15.0, 22.0, 30.0])
def test_three_replicates_median_is_bounded_by_the_honest_error(h, w):
    reps, trs = build3(h, w)
    run = tm.run_consensus(reps, trs, order=["R", "A", "B"])
    cons = run["cons"]
    e = kappa_error(run)
    nv = cons["n_votes"]; src = cons["dome_source"]
    sel3 = (nv == 3) & np.isfinite(e)
    assert sel3.sum() >= 300
    assert set(np.unique(src[sel3])) == {"median3"}
    bound = dk(h) + dk(FIT_MARGIN_PX)
    within = np.mean(np.abs(e[sel3]) <= bound)
    assert within >= 0.95, f"h={h} w={w}: |κ*−κ| ≤ Δκ({h}+{FIT_MARGIN_PX} px) on only {within:.1%} (median rel {np.median(e[sel3]) / tm.KF:+.3f})"
    # the median IS one of the votes, and never the wrong member's when it is the extreme (w > h + margin)
    K = {c: cons["votes"][c]["kappa_lat"] for c in cons["voter_ids"]}
    stack = np.stack([K[c] for c in cons["voter_ids"]], 0)
    assert np.all(np.min(np.abs(stack[:, sel3] - cons["kappa_star"][sel3]), axis=0) < 1e-12)
    if w > h + FIT_MARGIN_PX:
        picked_A = np.mean(np.abs(K["A"][sel3] - cons["kappa_star"][sel3]) < 1e-12)
        assert picked_A <= 0.05, f"the wrong member decided {picked_A:.1%} of the 3-vote laterals"
    assert cons["dome_verdict"] == "median_of_three"
    assert cons["source_counts"]["reference"] == 0 and cons["source_counts"]["median"] == 0
    # the apply: every member's tissue lands within the honest error of the truth; the applied field is smooth
    fin, ap = tm.apply_and_measure(reps, trs, run)
    assert_smooth(fin)
    bar = 0.35 * (h + FIT_MARGIN_PX) + 0.2
    # the dome part of the residual beyond the move scales with the honest error (the median has no averaging, so the
    # 3-vote / 2-vote transition at the canvas edges steps c2* by the votes' spread): 1.0 px is the bar for CS001-like
    # replicates (dissents 2–4 px, h ≤ 5 here); at h = 8 px it reaches ~1.1 px
    dome_bar = gj.AFTER_RMS_BAR_PX if h <= 5.0 else 1.5
    for c in ("R", "A", "B"):
        assert ap[c]["after_rms_minus_const"] <= bar, (c, ap[c]["after_rms_minus_const"], bar)
        assert fin[c]["sizes"]["dome_rms_after_px"] <= dome_bar, (c, fin[c]["sizes"]["dome_rms_after_px"])


# ── n = 4: the wrong member (or the wrong ANCHOR) is out-voted; the calibrated tie-break never picks the anchor ──
def test_four_replicates_outvote_the_wrong_member():
    err = {"R": -8.0, "A": 22.0, "B": 6.0, "C": -5.0}
    reps, trs = tm.build_group(err)                        # saccade in B, crops, single-coverage regions
    run = tm.run_consensus(reps, trs)
    cons = run["cons"]; e = kappa_error(run); nv = cons["n_votes"]
    sel4 = (nv == 4) & np.isfinite(e)
    assert sel4.sum() >= 300 and set(np.unique(cons["dome_source"][sel4])) == {"majority"}
    bound = dk(8.0) + dk(FIT_MARGIN_PX)
    assert np.mean(np.abs(e[sel4]) <= bound) >= 0.95
    vA = next(v for v in cons["voters"] if v["cid"] == "A")
    assert all(f is not None and f < 0.05 for f in vA["in_majority_by_band"]), vA["in_majority_by_band"]
    assert vA["dissent_bands"] >= 3 and vA["correction_at_frame_ends_px"] > 8.0
    assert vA["flags"] == [] and vA["witnessed"] is True          # corrected, witnessed by three others: no warning
    assert cons["dome_verdict"] == "witnessed_majority" and cons["flags"] == []
    assert all(b["verdict"] == "witnessed" for b in cons["bands"])
    assert cons["toricity"]["calibrated"] and 1.0 <= cons["toricity"]["rho"] <= 1.4
    fin, ap = tm.apply_and_measure(reps, trs, run)
    assert_smooth(fin)
    for c in tm.ORDER:
        assert ap[c]["after_rms_minus_const"] <= 1.5, (c, ap[c]["after_rms_minus_const"])
        assert ap[c]["after_peak_group_const"] <= 4.5, (c, ap[c]["after_peak_group_const"])
        assert fin[c]["sizes"]["dome_rms_after_px"] <= gj.AFTER_RMS_BAR_PX
    assert ap["_consensus_vs_truth"]["rms_minus_const"] <= 1.5 and ap["_divot_residual_rms_minus_const"] <= 2.0
    # between-member tissue consistency after the apply
    assert all(v <= 0.6 for v in ap["_pair_tissue_rms_after"].values()), ap["_pair_tissue_rms_after"]


def test_wrong_anchor_is_outvoted_and_the_calibrated_tie_break_never_picks_it():
    reps, trs = tm.build_group({"R": 22.0, "A": -8.0, "B": 6.0, "C": -5.0})
    run = tm.run_consensus(reps, trs)
    cons = run["cons"]; e = kappa_error(run); nv = cons["n_votes"]; src = cons["dome_source"]
    sel4 = (nv == 4) & np.isfinite(e)
    assert np.mean(np.abs(e[sel4]) <= dk(8.0) + dk(FIT_MARGIN_PX)) >= 0.95
    vR = next(v for v in cons["voters"] if v["cid"] == "R")
    assert all(f is not None and f < 0.05 for f in vR["in_majority_by_band"][1:]), vR["in_majority_by_band"]
    assert vR["in_majority_by_band"][0] < 0.5                        # band 0: 2-vote edge laterals where the saccade member's own vote is off too
    # the calibrated tie-break: κ_axial × ρ (ρ ≈ the true toricity × asphericity, not 1) picks the honest voter
    ax = src == "axial"
    assert ax.sum() >= 10 and not cons["in_majority"]["R"][ax].any()
    assert np.all(np.abs(e[ax]) <= dk(8.0) + dk(FIT_MARGIN_PX))
    assert cons["toricity"]["calibrated"] and cons["toricity"]["n_laterals"] >= gc_.RHO_MIN_LATERALS
    fin, ap = tm.apply_and_measure(reps, trs, run)
    assert_smooth(fin)
    assert ap["R"]["before_rms"] > 5.0 and ap["R"]["after_rms_minus_const"] <= 1.6     # the anchor's own 22 px dome is corrected
    for c in ("A", "B", "C"):
        assert ap[c]["after_rms_minus_const"] <= 1.6, (c, ap[c]["after_rms_minus_const"])


# ── the apply: LINE error never moves tissue (a 10 px dip on 6 frames), and it is flagged ────────────────────
def test_line_dip_does_not_move_tissue_and_is_flagged():
    err = {"R": -8.0, "A": 22.0, "B": 6.0, "C": -5.0}
    reps, trs = tm.build_group(err)
    run = tm.run_consensus(reps, trs)
    fin = gj.final_transforms(run["members"], trs, run["canvas"], run["lines"], run["cons"]["curve"])
    f0, n = 44, 6
    for label, dip in [("all_laterals", (f0, n, 10.0, 0, tm.L)), ("lateral_band", (f0, n, 10.0, 120, 260))]:
        reps_d, trs_d = tm.build_group(err, dip=dip, dip_member="C")
        run_d = tm.run_consensus(reps_d, trs_d)
        fin_d = gj.final_transforms(run_d["members"], trs_d, run_d["canvas"], run_d["lines"], run_d["cons"]["curve"])
        for c in tm.ORDER:
            d_a = np.nanmax(np.abs(fin_d[c]["delta_a"] - fin[c]["delta_a"])); d_b = np.nanmax(np.abs(fin_d[c]["delta_b"] - fin[c]["delta_b"]))
            assert d_a <= 0.1 and d_b <= 0.05, (label, c, d_a, d_b)
            assert fin_d[c]["sizes"]["line_off_consensus_frames"] == (n if c == "C" else 0), (label, c)
        assert_smooth(fin_d)


# ── the shared wrong dome: only the axial witness can tell; a legitimate toricity is not flagged ──────────────
def test_shared_wrong_dome_is_flagged_by_the_axial_witness():
    ms, trs = group(wrong=None, shared_scale=1.6)          # model L: the tissue's along-frame curvature 1.6× the across-lateral one
    cons, *_ = run_v2(ms, trs)
    assert cons["source_counts"]["median3"] >= 0.8 * cons["voted_laterals"]          # they agree with each other …
    assert "dome_axial_mismatch" in cons["flags"], cons["flags"]                      # … but not with their own B-scans
    ratios = [b["ratio"] for b in cons["bands"]]
    assert all(np.isfinite(r) and r > gc_.RATIO_ENVELOPE[1] for r in ratios), ratios
    assert all(b["verdict"] == "mismatch" for b in cons["bands"]) and cons["axial_verdict"] == "mismatch"


@pytest.mark.parametrize("scale, flagged", [(1.6, True), (0.6, True), (1.28 / (tm.KF / tm.KL), False)])
def test_shared_wrong_dome_model_t(scale, flagged):
    reps, trs = tm.build_group({"R": -8.0, "A": 6.0, "B": 5.0, "C": -5.0}, kf_scale=scale)
    cons = tm.run_consensus(reps, trs)["cons"]
    assert ("dome_axial_mismatch" in cons["flags"]) is flagged, (scale, cons["flags"], [b["ratio"] for b in cons["bands"]])
    assert all((b["verdict"] == "mismatch") is flagged for b in cons["bands"])


# ── model L: the wrong member is corrected to within the honest error; the rest of the rule ───────────────────
def test_model_l_median_corrects_the_wrong_member_within_the_honest_error():
    ms, trs = group(wrong="A")
    cons, members, canvas, lines, colmask, cols = run_v2(ms, trs)
    c2s = cons["coef"][:, 2]
    fit = np.isfinite(c2s)
    assert fit.sum() >= 0.9 * canvas["shape"][0]
    h_c2 = max(abs(b) for b in BETAS_L.values())                          # the largest honest error in px/frame²
    within = np.abs(c2s[fit] - C2_F) <= h_c2 + 0.0005 / HF_L ** 2 * HF_L ** 2 * 0 + 0.0005
    assert within.mean() >= 0.90, f"c2 within the honest error on only {within.mean():.2%} (median {np.median(c2s[fit]):.5f} vs truth {C2_F:.5f})"
    assert cons["source_counts"]["median3"] >= 0.8 * cons["voted_laterals"], cons["source_counts"]
    vA = next(v for v in cons["voters"] if v["cid"] == "A")
    assert vA["dissent_bands"] >= 3 and vA["correction_at_frame_ends_px"] > 8.0
    assert all((f is None) or f < 0.2 for f in vA["in_majority_by_band"][1:4]), vA["in_majority_by_band"]
    assert vA["flags"] == [] and vA["witnessed"] is True
    assert "dome_axial_mismatch" not in cons["flags"] and "dome_majority_failed" not in cons["flags"], cons["flags"]
    assert cons["axial"]["scale_trusted"] and cons["axial_verdict"] == "witnessed" and cons["dome_verdict"] == "median_of_three"
    Zc = truth_on_canvas(canvas)
    both = np.isfinite(cons["curve"])
    d = cons["curve"][both] - Zc[both]
    assert np.sqrt(np.mean(d * d)) <= 0.45 * H_MAX_L_PX + 1.0, np.sqrt(np.mean(d * d))
    # the APPLY step lands the wrong member's LINE dome within the honest error of the truth
    fin = gj.final_transforms(members, trs, canvas, lines, cons["curve"])
    assert_smooth(fin)
    tA = fin["A"]
    _v, _k, line_after, col_after = gj.place_on_canvas(ms["A"], {"df": tA["df"], "dx": tA["dx"], "a": tA["a"], "b": tA["b"]}, canvas, volume=False)
    dome_after = gj.member_dome(line_after, col_after)["curve"]
    both = np.isfinite(dome_after) & np.isfinite(Zc)
    d = dome_after[both] - Zc[both]
    assert np.sqrt(np.mean(d * d)) <= 0.45 * H_MAX_L_PX + 1.0, f"wrong member's final dome off the truth by {np.sqrt(np.mean(d * d)):.2f} px rms"
    dome_before = gj.member_dome(lines["A"], cols["A"])["curve"]
    both = np.isfinite(dome_before) & np.isfinite(Zc)
    assert np.max(np.abs(dome_before[both] - Zc[both])) > 12.0


def test_no_extrapolation_beyond_coverage():
    ms, trs = group(wrong="A")
    cons, members, canvas, lines, colmask, cols = run_v2(ms, trs)
    finite = np.isfinite(cons["curve"])
    assert not finite[~colmask].any()                      # never a value where nobody covers
    line_cov = np.zeros_like(colmask)
    for ln in lines.values():
        line_cov |= np.isfinite(ln)
    assert finite.sum() >= 0.95 * line_cov.sum()           # and (almost) everywhere a line covers
    Lc = canvas["shape"][0]
    assert len(cons["dome_source"]) == Lc and set(np.unique(cons["dome_source"])) <= set(gc_.DOME_SOURCES)


def test_legacy_spacing_is_unverified_and_two_voter_split_keeps_the_reference():
    sp = (4.0 / 513.0, 0.0031, 0.04)
    ms, trs = group(wrong="A", spacing=sp, beta_wrong=0.02)  # 32 px at the frame ends: beyond 2 tau of the reference
    trs.pop("B"); ms.pop("B")                                # two voters that disagree everywhere: nothing is decided
    cons, *_ = run_v2(ms, trs)
    assert "lateral_scale_unverified" in cons["flags"]
    assert not cons["axial"]["scale_trusted"] and cons["axial_verdict"].endswith("_scale_unverified")
    assert all("ratio_at_scan_size_mm" in b for b in cons["bands"] if np.isfinite(b["ratio"]))
    assert not cons["toricity"]["calibrated"]                # no decided lateral → no calibrated witness, no continuity
    assert cons["source_counts"]["reference"] >= 0.8 * cons["voted_laterals"], cons["source_counts"]
    assert "dome_majority_failed" in cons["flags"]
    vA = next(v for v in cons["voters"] if v["cid"] == "A")
    assert "dome_unwitnessed" in vA["flags"]                 # nobody witnesses either side: the reviewer must look


def test_two_voter_split_falls_to_continuity_then_calibrated_axial():
    """A 2-vote lateral that disagrees takes the nearest decided lateral's value (or the calibrated axial pick), never
    the anchor by default: R and A disagree on the left third only (A's own dome bent there), B covers the rest."""
    ms, trs = group(wrong=None)
    # bend A's line on its left 90 laterals by a 48 px dome error (a locally wrong own dome, beyond 2 tau), and hide B there
    u = np.arange(F, dtype=float) - (F - 1) / 2.0
    ms["A"].served[:90] += 0.03 * u[None, :] ** 2
    ms["B"].valid[:150] = False
    cons, *_ = run_v2(ms, trs)
    src = cons["dome_source"]; nv = cons["n_votes"]
    two = nv == 2
    assert two.sum() >= 40
    assert cons["source_counts"]["reference"] == 0
    assert (cons["source_counts"]["continuity"] + cons["source_counts"]["axial"]) >= 20, cons["source_counts"]
    # wherever the split was decided, κ* sits with the honest votes, not A's bent one
    kap = cons["kappa_star"]; KA = cons["votes"]["A"]["kappa_lat"]
    dec = two & np.isin(src, ["continuity", "axial"])
    assert np.all(np.abs(kap[dec] - KAPPA_F) < np.abs(KA[dec] - KAPPA_F))


def test_vote_only_member_votes_without_a_transform():
    ms, trs = group(wrong="A")
    tA = trs.pop("A")
    vote_only = [(ms["A"], {"df": tA["df"], "dx": tA["dx"], "a": np.zeros(F), "b": np.zeros(F)})]
    cons, members, canvas, *_ = run_v2(ms, trs, vote_only=vote_only)
    assert [m.cid for m in members] == ["R", "B"]
    vA = next(v for v in cons["voters"] if v["cid"] == "A")
    assert vA["role"] == "vote_only" and "vote_only" in vA["flags"]
    assert cons["source_counts"]["median3"] >= 0.8 * cons["voted_laterals"]      # three voters: the median
    c2s = cons["coef"][:, 2]; fit = np.isfinite(c2s)
    assert (np.abs(c2s[fit] - C2_F) <= max(abs(b) for b in BETAS_L.values()) + 0.0005).mean() >= 0.9


# ── the reference sensitivity ─────────────────────────────────────────────────────────────────────────────────
def _provider_from_poses(trs: dict):
    def provider(anchor: str, mov: str):
        T = gc_.compose_transforms(trs[mov], gc_.invert_transform(trs[anchor], L, F), L)
        return {"ok": True, "df": T["df"], "dx": T["dx"], "a": T["a"], "b": T["b"], "rel": 1.2, "ncc_coarse": 0.9, "flags": []}
    return provider


def test_reference_sensitivity_model_l_spread_is_small():
    ms, trs = group(wrong=None)
    cons, members, canvas, lines, colmask, cols = run_v2(ms, trs)
    ctx = {"curve": cons["curve"], "canvas": canvas, "cons": cons}
    sens = gj.reference_sensitivity_stage("syn", ms["R"], ms, ["R", "A", "B"], trs, ctx, _provider_from_poses(trs), log=lambda *_a: None)
    assert sens["anchors_skipped"] == []
    for c in ("A", "B"):
        r = sens["anchors"][c]
        assert r["skipped"] is None and r["n_cells"] > 0.5 * np.isfinite(cons["curve"]).sum(), r
        assert r["shape_spread_px"] <= 1.0 and r["spread_px"] <= 1.5, (c, r["spread_px"], r["shape_spread_px"])
        assert r["shape_spread_px"] <= r["spread_px"] + 1e-9
        assert r["contributing"] and all(np.isfinite(k) for k in r["kappa_by_band"])
        rt = r["pair_roundtrip"]
        assert rt["df_error"] == 0 and rt["a_rms_px"] < 1e-9 and rt["b_rms_px"] < 1e-9 and rt["dx_rms"] < 1e-9
    assert sens["shape_spread_max_px"] <= 1.0
    assert all(s is not None and s <= 0.05 * KAPPA_F for s in sens["kappa_band_spread"]), sens["kappa_band_spread"]


def test_reference_sensitivity_model_t_shape_spread_and_dome_part():
    """Model T (the refuter's FAIL 4): the raw spread IS the anchors' own dome error — its dome part in κ equals the
    injected anchor-vs-R difference; the SHAPE spread (beyond a pose + a smooth dome move) is what the result keeps."""
    err = {"R": -8.0, "A": 22.0, "B": 6.0, "C": -5.0}
    reps, trs = tm.build_group(err)
    run = tm.run_consensus(reps, trs)
    ctx = {"curve": run["cons"]["curve"], "canvas": run["canvas"], "cons": run["cons"]}
    sens = gj.reference_sensitivity_stage("syn", reps["R"].md, {c: reps[c].md for c in tm.ORDER}, tm.ORDER, trs, ctx,
                                          tm.provider_from_reps(reps), log=lambda *_a: None)
    assert sens["anchors_skipped"] == []
    for c in ("A", "B", "C"):
        r = sens["anchors"][c]
        expected = dk(err[c] - err["R"])
        assert abs(abs(r["dome_part_kappa"]) - abs(expected)) <= 0.25 * abs(expected) + 0.002, (c, r["dome_part_kappa"], expected)
        assert r["shape_spread_px"] <= r["spread_px"] + 1e-9
        assert r["shape_spread_px"] <= (2.5 if c == "B" else 0.8), (c, r["shape_spread_px"])   # B: the saccade anchor's own grid
    assert sens["anchors"]["A"]["spread_px"] > 3.0                                       # the raw number is the 30 px dome difference
    assert all(s is not None and s <= 0.04 * tm.KF for s in sens["kappa_band_spread"]), sens["kappa_band_spread"]


def test_reference_sensitivity_skips_an_anchor_without_a_served_pair():
    ms, trs = group(wrong=None)
    cons, members, canvas, lines, colmask, cols = run_v2(ms, {k: v for k, v in trs.items() if k != "B"})
    ctx = {"curve": cons["curve"], "canvas": canvas, "cons": cons}
    sens = gj.reference_sensitivity_stage("syn", ms["R"], ms, ["R", "A", "B"], {k: v for k, v in trs.items() if k != "B"}, ctx,
                                          _provider_from_poses(trs), log=lambda *_a: None)
    assert "B" in sens["anchors_skipped"] and "no accepted pair" in sens["anchors"]["B"]["skipped"]
    assert sens["anchors"]["A"]["spread_px"] is not None


# ── the record ────────────────────────────────────────────────────────────────────────────────────────────────
def test_summary_carries_the_revision_and_the_guarantee():
    ms, trs = group(wrong=None)
    cons, *_ = run_v2(ms, trs)
    s = gc_.summary_of(cons)
    assert s["version"] == 2 and s["revision"] == gc_.CONSENSUS_REVISION == 3
    assert "median3" in s["dome"]["rule"] and "honest" in s["dome"]["guarantee"]
    assert s["dome"]["n_voters"] == 3 and s["dome"]["toricity"]["calibrated"] in (True, False)
    assert set(s["dome"]["sources"]) == set(gc_.DOME_SOURCES)
    import json
    json.dumps(s)                                           # JSON-able


# ── transform algebra and the pair cache ─────────────────────────────────────────────────────────────────────
def test_transform_algebra_roundtrip():
    T = POSES["A"]
    Tinv = gc_.invert_transform(T, L, F)
    I = gc_.compose_transforms(T, Tinv, L)
    ok = np.isfinite(I["dx"])
    assert I["df"] == 0 and ok.sum() >= F - abs(T["df"])
    assert np.allclose(I["dx"][ok], 0) and np.allclose(I["a"][ok], 0, atol=1e-9) and np.allclose(I["b"][ok], 0)
    l, f = 100.0, 40.0
    hs = (L - 1) / 2.0
    zA = truth(l + T["dx"][int(f)], f + T["df"]) - T["a"][int(f)] - T["b"][int(f)] * (l - hs) / hs
    zR = zA + T["a"][int(f)] + T["b"][int(f)] * (l - hs) / hs
    assert abs(zR - truth(l + T["dx"][int(f)], f + T["df"])) < 1e-9
    fR = int(f) + T["df"]; lR = l + T["dx"][int(f)]
    zA2 = zR + Tinv["a"][fR] + Tinv["b"][fR] * (lR - hs) / hs
    assert abs(zA2 - zA) < 1e-9


@dataclass
class _Res:
    ref_cid: str
    mov_cid: str
    df: int
    a: np.ndarray
    measured: np.ndarray
    flags: list
    dx_segments: list
    shape: tuple
    quality: dict = field(default_factory=dict)
    b_lines: np.ndarray | None = None


def test_pair_cache_roundtrip(tmp_path):
    cache = gc_.PairCache(tmp_path / "pairs", "md5x", "ph1", ctor=_Res)
    r = _Res("ref", "mov", -3, np.array([1.0, np.nan, 2.5]), np.array([True, False, True]), ["arbitrated"], [(0, 3, -1.5)], (5, 6, 7),
             {"k": [1, 2], "n": None, "arr": np.array([0.5, np.nan]), "sub": {"frames": [4, 5], "m": np.array([True, False])}}, None)
    cache.put(r)
    back = cache.get("ref", "mov")
    assert back is not None and back.df == -3 and back.flags == ["arbitrated"] and back.dx_segments == [(0, 3, -1.5)] and back.shape == (5, 6, 7)
    assert np.array_equal(back.measured, r.measured) and back.measured.dtype == bool
    assert np.allclose(back.a, r.a, equal_nan=True) and back.b_lines is None
    q = back.quality
    assert q["k"] == [1, 2] and q["n"] is None and q["sub"]["frames"] == [4, 5]
    assert isinstance(q["arr"], np.ndarray) and np.allclose(q["arr"], [0.5, np.nan], equal_nan=True)
    assert isinstance(q["sub"]["m"], np.ndarray) and q["sub"]["m"].dtype == bool and q["sub"]["m"].tolist() == [True, False]
    assert cache.get("ref", "other") is None
    other = gc_.PairCache(tmp_path / "pairs", "md5y", "ph1", ctor=_Res)
    assert other.get("ref", "mov") is None


# ── TISSUE GATE on the apply (2026-09-12, refutation R4) ──────────────────────────────────────────────────────
def test_tissue_gate_decide_holds_the_scan_whose_move_raises_its_disagreement():
    """The decision alone (tissue-edge arrays): three scans whose tissue edges disagree by a few px before; after the dome move two
    are corrected and one is moved 15 px AWAY at the frame ends — every scan's pairwise disagreement rises at first, the worst
    offender is held (its after edges are its before edges), the rest are re-judged and stay; the group mean after ends below
    before (ok)."""
    Lc, Fc = 120, 61
    rng = np.random.default_rng(3)
    u = (np.arange(Fc) - (Fc - 1) / 2.0) / ((Fc - 1) / 2.0)
    base = 100.0 + 0.002 * (np.arange(Lc)[:, None] - Lc / 2) ** 2 + 6.0 * u[None, :] ** 2
    noise = lambda: rng.normal(0.0, 0.2, size=(Lc, Fc))  # noqa: E731
    before = {"R": base - 2.0 * u[None, :] ** 2 + noise(), "A": base + 2.0 * u[None, :] ** 2 + noise(), "B": base + noise()}
    after = {"R": base + noise(), "A": base + 15.0 * u[None, :] ** 2 + noise(), "B": base + noise()}
    before["R"][:10, :] = np.nan; after["R"][:10, :] = np.nan                     # an uncovered strip: NaN never counts
    g = gj.tissue_gate_decide(before, after)
    assert g["held"] == ["A"] and g["per_scan"]["A"]["held"] and not g["per_scan"]["R"]["held"] and not g["per_scan"]["B"]["held"], g
    pa = g["per_scan"]["A"]
    assert pa["after_initial_px"] > pa["before_px"] + gj.TISSUE_GATE_SCAN_PX and pa["reason"]["rise_px"] > 3.0
    assert abs(pa["after_px"] - pa["before_px"]) <= 0.5                       # held: judged at its before placement against the moved rest
    for c in ("R", "B"):
        assert g["per_scan"][c]["after_px"] < g["per_scan"][c]["before_px"], (c, g["per_scan"][c])
    assert g["ok"] and g["group_after_px"] < g["group_before_px"] and g["group_after_initial_px"] > g["group_before_px"]
    assert g["iterations"] == 2 and g["threshold_scan_px"] == 0.3 and g["threshold_group_px"] == 0.1
    per, mat = gj.pairwise_tissue_disagreement(before)
    assert set(per) == {"R", "A", "B"} and mat["R"]["A"]["n"] == (Lc - 10) * Fc and mat["A"]["R"]["mean_abs_px"] == mat["R"]["A"]["mean_abs_px"]
    # nothing to hold when every move helps
    g2 = gj.tissue_gate_decide(before, {"R": base + noise(), "A": base + noise(), "B": base + noise()})
    assert g2["held"] == [] and g2["ok"] and g2["iterations"] == 1
    import json
    json.dumps(gj.jsonable(g)); json.dumps(gj.jsonable(g2))


def _tissue_member(cid: str, T: dict, *, err_tissue: float = 0.0, err_line: float = 0.0, seed: int = 0, D: int = 260):
    """A member whose VOLUME carries a bright tissue step at its tissue surface. The dome error (px at the frame ends) sits in the
    TISSUE (err_tissue: a smooth residual the pair engine's placement LEFT — the pair transform is the acquisition pose alone, so
    the placed tissues disagree by their errors and the dome move is what corrects them) or only in the LINE (err_line — model L:
    the tissue is exact, the served line is biased, so the dome move would push exact tissue away). Returns (MemberData, pair
    transform → R = the pose)."""
    rng = np.random.default_rng(seed)
    lat = np.arange(L, dtype=float); fr = np.arange(F, dtype=float)
    LL, FF = np.meshgrid(lat, fr, indexing="ij")
    hs = (L - 1) / 2.0
    u = (FF - (F - 1) / 2.0) / HF_L
    z_true = Z0 + C2_L * (LL + T["dx"][None, :] - hs) ** 2 + C2_F * (FF + T["df"] - (F - 1) / 2.0) ** 2
    e_t = err_tissue * u ** 2; e_l = err_line * u ** 2
    tissue = z_true - T["a"][None, :] - T["b"][None, :] * (LL - hs) / hs - e_t
    line = tissue + e_l + rng.normal(0.0, 0.4, size=tissue.shape)
    z = np.arange(D, dtype=float)[None, :, None]
    vol = 25.0 + 5.0 * rng.standard_normal((L, D, F)) + 220.0 / (1.0 + np.exp(-(z - tissue[:, None, :]) / 0.8))
    md = ga.MemberData(cid=cid, case_dir=Path("/nonexistent"), group="syn", volume=np.clip(vol, 0, None).astype(np.float32),
                       served=line, valid=np.ones((L, F), bool), spacing=np.asarray(SPACING, float))
    pair = {"df": int(T["df"]), "dx": T["dx"].copy(), "a": T["a"].copy(), "b": T["b"].copy()}
    return md, pair


def test_apply_holds_the_dome_move_of_a_line_biased_scan_by_the_tissue_gate(tmp_path):
    """End to end (build_consensus → apply_transforms on rendered tissue volumes): R and B carry ±3 px dome errors in their placed
    TISSUE (the pair placement left them; the dome move corrects them), A's tissue is exact but its LINE carries an 18-px dome
    error — A's own dome follows its line, so its dome move would push its exact tissue ~15 px off the others. The gate (measured
    from the scrub images) HOLDS A's move (a_pair / b_pair served, δa = δb = 0, 'dome_move_held_by_tissue' with both numbers),
    keeps R's and B's, and the group's tissue disagreement after is below before. Line error never moves tissue."""
    mR, pR = _tissue_member("R", POSES["R"], err_tissue=-3.0, seed=1)
    mA, pA = _tissue_member("A", POSES["A"], err_line=18.0, seed=2)
    mB, pB = _tissue_member("B", POSES["B"], err_tissue=3.0, seed=3)
    members = [mR, mA, mB]
    trs = {"R": pR, "A": pA, "B": pB}
    ctx: dict = {}
    out = tmp_path / "align_min"; out.mkdir()
    logs: list = []
    cons = gj.build_consensus("syn", mR, members, trs, out, log=logs.append, ctx=ctx,
                              pair_info={c: {"ok": True, "rel": 1.1, "ncc_coarse": 0.9} for c in ("A", "B")})
    assert cons["n_members"] == 3
    summ = gj.apply_transforms("syn", mR, members, trs, ctx, out, log=logs.append)
    assert summ["held_scans"] == ["A"], (summ["held_scans"], [l_ for l_ in logs if "TISSUE GATE" in l_ or "tissue gate" in l_])
    tj = gj.read_json(out / gj.TRANSFORMS_JSON)
    gate = tj["tissue_gate"]
    assert gate["held"] == ["A"] and gate["ok"] and gate["group_after_px"] <= gate["group_before_px"] + gj.TISSUE_GATE_GROUP_PX
    assert gate["group_after_px"] < gate["group_before_px"], gate
    pa = gate["per_scan"]["A"]
    assert pa["held"] and pa["after_initial_px"] > pa["before_px"] + 3.0 and pa["reason"]["threshold_px"] == 0.3
    for c in ("R", "B"):
        assert not gate["per_scan"][c]["held"] and gate["per_scan"][c]["after_px"] < gate["per_scan"][c]["before_px"], (c, gate["per_scan"][c])
    tA = next(m for m in tj["members"] if m["cid"] == "A")
    assert tA["sizes"]["dome_move_held"] is True and tA["sizes"]["dome_move_held_by_tissue"]["after_px"] > tA["sizes"]["dome_move_held_by_tissue"]["before_px"] + 0.3
    assert tA["sizes"]["delta_a"]["peak"] == 0.0 and tA["sizes"]["delta_b"]["peak"] == 0.0 and tA["sizes"]["smooth_ok"]
    assert np.allclose(np.asarray(tA["a_final"], float), np.asarray(tA["a_pair"], float), equal_nan=True)
    assert np.allclose(np.asarray(tA["b_final"], float), np.asarray(tA["b_pair"], float), equal_nan=True)
    tB = next(m for m in tj["members"] if m["cid"] == "B")
    assert tB["sizes"]["dome_move_held"] is False and tB["sizes"]["delta_a"]["peak"] > 1.0     # B's residual tissue error IS corrected
    pm = {m["cid"]: m for m in summ["members"]}
    assert pm["A"]["dome_move_held"] is True and pm["B"]["dome_move_held"] is False and pm["A"]["rms_after_px"] == pm["A"]["rms_before_px"]
    # the held scan's AFTER scrub image is its BEFORE image; the meta carries the pairwise numbers
    scrub = out / gj.SCRUB_DIR
    a_after = np.load(scrub / gj.scrub_volume_name("after", "A"), mmap_mode="r"); a_before = np.load(scrub / gj.scrub_volume_name("before", "A"), mmap_mode="r")
    assert np.array_equal(np.asarray(a_after), np.asarray(a_before))
    meta = gj.read_json(scrub / gj.SCRUB_META)
    ts = meta["tissue_summary"]
    assert set(ts["pairwise_scan"]) == {"R", "A", "B"} and ts["pairwise_group"]["after"] <= ts["pairwise_group"]["before"] + 0.1
    assert abs(ts["pairwise_scan"]["A"]["after"] - ts["pairwise_scan"]["A"]["before"]) <= 0.5
    assert "tissue_pairwise" in meta and meta["tissue_pairwise"]["after"]["A"]["R"]["n"] > 0
    # the montage / volume still agree with the held transform: aligned_lines.npz carries a_pair as A's final a
    z = np.load(out / gj.ALIGNED_LINES)
    assert np.allclose(z["a_A"], z["a_pair_A"], equal_nan=True) and not np.allclose(z["a_B"], z["a_pair_B"], equal_nan=True)
    # all honest: nothing held, the group improves
    mA2, pA2 = _tissue_member("A", POSES["A"], err_tissue=2.0, seed=2)
    out2 = tmp_path / "align_min2"; out2.mkdir(); ctx2: dict = {}
    gj.build_consensus("syn", mR, [mR, mA2, mB], {"R": pR, "A": pA2, "B": pB}, out2, log=logs.append, ctx=ctx2,
                       pair_info={c: {"ok": True, "rel": 1.1, "ncc_coarse": 0.9} for c in ("A", "B")})
    summ2 = gj.apply_transforms("syn", mR, [mR, mA2, mB], {"R": pR, "A": pA2, "B": pB}, ctx2, out2, log=logs.append)
    g2 = gj.read_json(out2 / gj.TRANSFORMS_JSON)["tissue_gate"]
    assert summ2["held_scans"] == [] and g2["ok"] and g2["group_after_px"] < g2["group_before_px"], g2


# ── the roster reason (R2) and the transitive role ──────────────────────────────────────────────────────────────
def test_member_role_names_the_beyond_bar_correspondence_and_the_via_route():
    bar = {"laterals": 96, "frame_frac": 0.4, "L": 513, "max_dx": 417.0}
    base = {"cid": "case_x_v4", "is_reference": False, "ok": False, "reject_flags": ["no_correspondence", "dx_beyond_max"],
            "dx_median": -405.5, "overlap_fraction": 0.2018}
    # (a) refused by correspondence, the unrestricted coarse peak beyond the bar: the reason names THAT, never the admissible seed
    ov = {"verdict": "below_bar", "bar": bar, "seed": {"dx": -406.0, "fraction": 0.2086}, "served": {"dx": -405.5, "fraction": 0.2018},
          "offset": {"dx": -449.3, "fraction": 0.1242, "laterals": 63.7, "ncc": 0.916, "source": "unrestricted_peak"},
          "unrestricted_peak": {"dx0": -449.3, "df0": 6, "ncc": 0.916, "beyond_bar": True, "overlap": {"dx": -449.3, "fraction": 0.1242}}}
    role = gj.member_role({**base, "overlap": ov})
    assert role == "refused: no_correspondence, dx_beyond_max (best correspondence at ≈ -449 laterals (12% overlap, below the 19% bar))", role
    # (b) the unrestricted peak recorded but 'offset' not set (an older-style record): still named
    ov_b = {**ov, "offset": None, "verdict": "poor_match"}
    assert "best correspondence at ≈ -449 laterals (12% overlap, below the 19% bar)" in gj.member_role({**base, "overlap": ov_b})
    # (c) the bar-edge judge's winner: the in-bar value and both scores
    ov_c = {"verdict": "below_bar", "bar": bar, "served": {"dx": -409.5, "fraction": 0.2017},
            "offset": {"dx": -421.0, "fraction": 0.1793, "source": "bar_edge_judge", "score": 0.869, "served_score": 0.749, "served_dx": -409.5}}
    role_c = gj.member_role({**base, "reject_flags": ["dx_at_search_edge", "no_overlap"], "dx_median": float("nan"), "overlap_fraction": None, "overlap": ov_c})
    # (17.9 % against the 18.7 % bar: one decimal when the two round to the same integer percent)
    assert role_c == ("refused: dx_at_search_edge, no_overlap (best correspondence at ≈ -421 laterals (17.9% overlap, below the 18.7% bar), "
                      "the in-bar value ≈ -410 laterals scored 0.75 against 0.87)"), role_c
    # (d) no correspondence beyond the bar known: the plain offset
    ov_d = {"verdict": "poor_match", "bar": bar, "served": {"dx": -405.5, "fraction": 0.2018}, "offset": None,
            "unrestricted_peak": {"dx0": -404.0, "df0": 6, "ncc": 0.7, "beyond_bar": False, "overlap": {"dx": -404.0, "fraction": 0.21}}}
    assert gj.member_role({**base, "reject_flags": ["no_correspondence"], "overlap": ov_d}) == "refused: no_correspondence (offset ≈ -406 laterals, overlap 20%)"
    # (e) placed transitively
    rec_v = {**base, "overlap": ov, "via": "case_x_v5", "route": {"rel": 1.015}}
    assert gj.member_role(rec_v) == "contributing (via case_x_v5)"
    ro = gj.build_roster([{"cid": "case_x_v2", "is_reference": True, "ok": True}, rec_v, {**base, "overlap": ov}])
    assert [r["placed"] for r in ro] == [True, True, False] and ro[1]["via"] == "case_x_v5" and ro[1]["route"] == {"rel": 1.015} and ro[2]["via"] is None
    assert ga.overlap_reason(ov, ["no_correspondence"]) == "best correspondence at ≈ -449 laterals (12% overlap, below the 19% bar)"
    assert ga.overlap_reason({}, []) == "" and ga.overlap_reason(None, ["no_overlap"]) == ""
    assert ga.overlap_reason({"verdict": "none_measured", "bar": bar, "seed": {"dx": 300.0, "fraction": 0.4}}, ["no_overlap"]) == "offset ≈ +300 laterals, overlap 40%, no overlapping structure measured"
