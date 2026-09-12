"""Synthetic-truth generator (model T) — the refuter's own generator of 2026-09-11 (scratchpad wf_cons2/verify_truth/refute_truth.py),
copied VERBATIM as a test fixture; only the sidecar path, the log and main() were removed. Used by tests/test_group_consensus.py.

Model T ('tissue'): every replicate's preprocessing left a smooth per-frame RIGID error in its TISSUE (dome error
e_m(f) + tilt error t_m(f)·x); its served line follows its tissue (+ 0.4 px noise, 2 % 8-px spikes, optional line
dip). The pair engine registers TISSUE to TISSUE exactly (pairs = P_R^-1 ∘ P_m). This is the reviewer's situation
("tissue registration maps every member's line onto the reference's line").
Model L ('line'): the implementer's test model — tissue exact, the dome error only in the LINE.

Truth: toric aspheric dome in physical units (κ_f 0.125, κ_l 0.11 /mm; asphericity −10 % of c2_f at the lateral
edges; a real 6 px divot at lateral 300), spacing (0.0117, 0.0031, 0.040) mm (trusted, not the legacy cube).
4 replicates R (anchor), A, B, C with independent smooth dome errors (px at the frame ends): see GROUPS.
Crops (laterals / frames / a band), a saccade (B: dx jumps 12 laterals at frame 60), df / dx offsets that leave
single-coverage frames and laterals on the union canvas.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "3")

SIDECAR = str(Path(__file__).resolve().parents[1])
if SIDECAR not in sys.path:
    sys.path.insert(0, SIDECAR)
import numpy as np  # noqa: E402

import group_align as ga  # noqa: E402
import group_consensus as gc_  # noqa: E402
import group_job as gj  # noqa: E402

L, D, F = 400, 8, 101
HS = (L - 1) / 2.0
FC = (F - 1) / 2.0
HF = FC
SP = (0.0117, 0.0031, 0.040)
KF, KL = 0.125, 0.11
C2F = KF * SP[2] ** 2 / (2 * SP[1])          # px / frame²
C2L = KL * SP[0] ** 2 / (2 * SP[1])          # px / lateral²
Z0 = 150.0
ASPH = 0.10
DIVOT_L, DIVOT_PX, DIVOT_W = 300.0, 6.0, 5.0
LOG = []


def log(*a):
    LOG.append(" ".join(str(x) for x in a))


def c2f_true(l):
    x = (np.asarray(l, float) - HS) / HS
    return C2F * (1.0 - ASPH * x ** 2)


def truth(l, f, *, kf_scale: float = 1.0):
    """The true anterior surface in TRUTH coordinates (rows)."""
    l = np.asarray(l, float); f = np.asarray(f, float)
    z = Z0 + C2L * (l - HS) ** 2 + kf_scale * c2f_true(l) * (f - FC) ** 2
    return z + DIVOT_PX * np.exp(-((l - DIVOT_L) / DIVOT_W) ** 2)


@dataclasses.dataclass
class Rep:
    cid: str
    df: int
    dx: np.ndarray            # (F,) own frame → truth lateral shift (truth lateral = own + dx)
    A: np.ndarray             # (F,) z_truth = z_own + A(f) + B(f)·x(l)   (tissue → truth)
    B: np.ndarray
    tissue: np.ndarray        # (L, F) own tissue anterior surface
    served: np.ndarray        # (L, F) served line
    valid: np.ndarray
    e_dome: np.ndarray        # (F,) the dome error (px)
    a_pose: np.ndarray = None # (F,) the acquisition pose only (no error): the truth in R's coordinates uses THIS
    b_pose: np.ndarray = None
    md: object = None

    def P(self):
        return {"df": int(self.df), "dx": self.dx.copy(), "a": self.A.copy(), "b": self.B.copy()}


def make_rep(cid, *, df=0, dx0=0.0, saccade=None, a0=0.0, a1=0.0, b0=0.0, err_px=0.0, gamma_px=0.0, wob_px=0.0,
             tilt_err=(0.0, 0.0), crops=(), seed=0, error_in="tissue", kf_scale=1.0, spike_frac=0.02, noise=0.4,
             dip=None):
    rng = np.random.default_rng(seed)
    fr = np.arange(F, dtype=float); u = fr - FC
    dx = np.full(F, float(dx0))
    if saccade is not None:                       # (frame, jump laterals)
        dx[int(saccade[0]):] += float(saccade[1])
    a_pose = a0 + a1 * u; b_pose = np.full(F, float(b0))
    # the smooth per-frame dome error: quadratic (err_px at the frame ends) + cubic (gamma_px at the ends) + slow wobble
    e = err_px * (u / HF) ** 2 + gamma_px * (u / HF) ** 3 + wob_px * np.sin(2 * np.pi * u / F + 0.7)
    t = tilt_err[0] + tilt_err[1] * u / HF        # tilt error at the lateral half-span, px
    lat = np.arange(L, dtype=float)
    LL, FF = np.meshgrid(lat, fr, indexing="ij")
    x = (LL - HS) / HS
    z_true = truth(LL + dx[None, :], FF + df, kf_scale=kf_scale)
    if error_in == "tissue":
        A = a_pose + e; B = b_pose + t              # the error is in the tissue: z_truth = z_own + A + B x
        tissue = z_true - A[None, :] - B[None, :] * x
        line = tissue.copy()
    else:
        A = a_pose; B = b_pose
        tissue = z_true - A[None, :] - B[None, :] * x
        line = tissue + e[None, :] + t[None, :] * x   # the error only in the LINE
    line = line + rng.normal(0.0, noise, size=line.shape)
    if spike_frac > 0:
        sp = rng.random(line.shape) < spike_frac
        line[sp] += rng.choice([-8.0, 8.0], size=int(sp.sum())) * rng.uniform(0.5, 1.5, size=int(sp.sum()))
    if dip is not None:                            # (f0, n_frames, px, l0, l1): a LINE defect only
        f0, n, px, l0, l1 = dip
        line[l0:l1, f0:f0 + n] += px
    valid = np.ones((L, F), bool)
    for (l0, l1, f0, f1) in crops:
        valid[l0:l1, f0:f1] = False
    rep = Rep(cid, df, dx, A, B, tissue, line, valid, e, a_pose, b_pose)
    rep.md = ga.MemberData(cid=cid, case_dir=Path("/nonexistent"), group="syn", volume=np.zeros((L, D, F), np.float32),
                           served=line, valid=valid, spacing=np.asarray(SP, float))
    return rep


def pair_to(m: Rep, R: Rep) -> dict:
    """The exact TISSUE-to-tissue rigid pair m → R (moving index + shift = reference index), my own formula."""
    df = m.df - R.df
    dx = np.full(F, np.nan); a = np.full(F, np.nan); b = np.full(F, np.nan)
    for f in range(F):
        fR = f + df
        if not (0 <= fR < F):
            continue
        d = m.dx[f] - R.dx[fR]
        dx[f] = d
        b[f] = m.B[f] - R.B[fR]
        a[f] = m.A[f] - R.A[fR] - R.B[fR] * d / HS
    return {"df": int(df), "dx": dx, "a": a, "b": b}


def tissue_md(rep: Rep):
    """A MemberData whose served line IS the tissue (to place the tissue by a transform)."""
    return dataclasses.replace(rep.md, served=rep.tissue.copy(), valid=np.ones((L, F), bool))


def truth_on_canvas(canvas, R: Rep, kf_scale=1.0):
    """The truth in R's (= canvas) coordinates: the truth carried by R's acquisition POSE only (a_pose, b_pose — NOT
    R's dome error: R's tissue = this − e_R). A whole-volume constant is the anchor's coordinate freedom."""
    l0, z0, f0 = canvas["origin"]; Lc, _Dc, Fc = canvas["shape"]
    LLc, FFc = np.meshgrid(np.arange(Lc) + l0, np.arange(Fc) + f0, indexing="ij")
    Z = np.full((Lc, Fc), np.nan)
    for fc in range(Fc):
        f = fc + f0
        if not (0 <= f < F):
            continue
        l = LLc[:, fc].astype(float)
        Z[:, fc] = truth(l + R.dx[f], f + R.df, kf_scale=kf_scale) - R.a_pose[f] - R.b_pose[f] * (l - HS) / HS - z0
    return Z


def placed(reps: dict, trs: dict, order):
    members = [reps[c].md for c in order if c in trs]
    canvas = gj.union_canvas(members, trs)
    Lc, _Dc, Fc = canvas["shape"]
    lines = {}; colmask = np.zeros((Lc, Fc), bool); cols = {}
    for m in members:
        _v, _k, line, col = gj.place_on_canvas(m, trs[m.cid], canvas, volume=False)
        lines[m.cid] = line; colmask |= col; cols[m.cid] = col
    return members, canvas, lines, colmask, cols


def rms(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v * v))) if v.size else float("nan")


# ── the groups ────────────────────────────────────────────────────────────────────────────────────────────────
CROPS = {"R": [(180, 200, 20, 34)],                         # a crop band
         "A": [(0, 40, 0, F), (0, L, 0, 6)],                 # 40 laterals + 6 frames cropped
         "B": [(L - 25, L, 0, F), (0, L, F - 10, F)],        # 25 laterals + 10 frames
         "C": [(150, 170, 60, 80), (0, L, 0, 3)]}
POSE = {"R": dict(df=0, dx0=0.0, a0=0.0, a1=0.0, b0=0.0),
        "A": dict(df=4, dx0=-15.0, a0=7.0, a1=0.06, b0=3.5),
        "B": dict(df=-6, dx0=9.0, saccade=(60, 12.0), a0=-6.0, a1=-0.03, b0=-2.0),
        "C": dict(df=2, dx0=22.0, a0=3.0, a1=0.02, b0=1.5)}
TILT_ERR = {"R": (0.8, 0.6), "A": (-1.5, 0.9), "B": (1.2, -0.7), "C": (-0.6, 0.4)}
GAMMA = {"R": 0.6, "A": -0.8, "B": 0.5, "C": -0.4}
WOB = {"R": 0.5, "A": 0.6, "B": -0.4, "C": 0.5}
ORDER = ["R", "A", "B", "C"]


def build_group(err: dict, *, error_in="tissue", kf_scale=1.0, dip=None, seed0=100, dip_member=None):
    reps = {}
    for i, c in enumerate(ORDER):
        reps[c] = make_rep(c, **POSE[c], err_px=err[c], gamma_px=GAMMA[c], wob_px=WOB[c], tilt_err=TILT_ERR[c],
                           crops=CROPS[c], seed=seed0 + i, error_in=error_in, kf_scale=kf_scale,
                           dip=(dip if c == dip_member else None))
    trs = {c: pair_to(reps[c], reps["R"]) for c in ORDER}
    return reps, trs


def run_consensus(reps, trs, anchor="R", order=ORDER):
    members, canvas, lines, colmask, cols = placed(reps, trs, order)
    cons = gc_.consensus_v2(reps[anchor].md, members, trs, lines, colmask, canvas,
                            pair_info={c: {"ok": True, "rel": 1.2, "ncc_coarse": 0.9} for c in trs if c != anchor})
    return dict(cons=cons, members=members, canvas=canvas, lines=lines, colmask=colmask, cols=cols)


def check_c2(run, label):
    cons = run["cons"]; canvas = run["canvas"]
    l0 = canvas["origin"][0]; Lc = canvas["shape"][0]
    c2s = cons["coef"][:, 2]; fit = np.isfinite(c2s)
    lat_ref = np.arange(Lc) + l0
    truth_c2 = c2f_true(np.clip(lat_ref, 0, L - 1))          # beyond R's laterals the truth continues (clip = edge value ±)
    truth_c2 = c2f_true(lat_ref)
    rel = (c2s - truth_c2) / truth_c2
    within = np.abs(rel[fit]) <= 0.05
    # per dome source
    src = cons["dome_source"]
    by_src = {}
    for s in gc_.DOME_SOURCES:
        sel = fit & (src == s)
        if sel.any():
            by_src[s] = dict(n=int(sel.sum()), within5=float(np.mean(np.abs(rel[sel]) <= 0.05)), med_rel=float(np.median(rel[sel])),
                             max_rel=float(np.max(np.abs(rel[sel]))))
    res = dict(label=label, fitted=int(fit.sum()), Lc=int(Lc), within5_frac=float(within.mean()), median_rel=float(np.median(rel[fit])),
               p95_abs_rel=float(np.percentile(np.abs(rel[fit]), 95)), max_abs_rel=float(np.max(np.abs(rel[fit]))),
               by_source=by_src, source_counts=cons["source_counts"], flags=cons["flags"], dome_verdict=cons["dome_verdict"],
               axial_verdict=cons["axial_verdict"],
               bands=[(b["band"], round(b["kappa_frames"], 4), round(b["kappa_axial"], 4), round(b["ratio"], 3), b["verdict"]) for b in cons["bands"]],
               voters=[(v["cid"], v["role"], [None if not np.isfinite(k) else round(k, 4) for k in v["kappa_by_band"]],
                        v["in_majority_by_band"], v["dissent_bands"], v["correction_at_frame_ends_px"], v["witnessed"], v["flags"]) for v in cons["voters"]])
    # the c2 within 5 % restricted to laterals with ≥ 3 votes
    nv = cons["n_votes"]
    sel3 = fit & (nv >= 3)
    res["within5_frac_3plus_votes"] = float(np.mean(np.abs(rel[sel3]) <= 0.05)) if sel3.any() else None
    res["n_3plus"] = int(sel3.sum())
    return res


def apply_and_measure(reps, trs, run, *, kf_scale=1.0):
    """final_transforms → place every member's TISSUE by its final transform → error to the truth (in R's coordinates)."""
    cons, canvas, lines = run["cons"], run["canvas"], run["lines"]
    fin = gj.final_transforms(run["members"], trs, canvas, lines, cons["curve"])
    Zt = truth_on_canvas(canvas, reps["R"], kf_scale)
    out = {}
    placed_t = {}
    for m in run["members"]:
        c = m.cid; t = fin[c]
        Tf = {"df": t["df"], "dx": t["dx"], "a": t["a"], "b": t["b"]}
        _v, _k, tis_after, col = gj.place_on_canvas(tissue_md(reps[c]), Tf, canvas, volume=False)
        _v, _k, tis_before, _c = gj.place_on_canvas(tissue_md(reps[c]), trs[c], canvas, volume=False)
        placed_t[c] = tis_after
        both = np.isfinite(tis_after) & np.isfinite(Zt)
        d_after = tis_after[both] - Zt[both]
        bothb = np.isfinite(tis_before) & np.isfinite(Zt)
        d_before = tis_before[bothb] - Zt[bothb]
        out[c] = dict(before_rms=rms(d_before), before_peak=float(np.max(np.abs(d_before))),
                      after_rms=rms(d_after), after_peak=float(np.max(np.abs(d_after))), after_mean=float(np.mean(d_after)),
                      after_rms_minus_const=rms(d_after - np.mean(d_after)),
                      jitter=t["sizes"]["delta_a_jitter"], delta_a=t["sizes"]["delta_a"], dome_after=t["sizes"]["dome_rms_after_px"],
                      profile_beyond_tilt=t["sizes"]["profile_beyond_tilt_frames"], covered=t["sizes"]["covered_frames"])
    # a whole-group constant (the anchor's coordinate freedom): the mean over ALL members' cells
    allm = np.concatenate([(placed_t[c][np.isfinite(placed_t[c]) & np.isfinite(Zt)] - Zt[np.isfinite(placed_t[c]) & np.isfinite(Zt)]) for c in placed_t])
    const = float(np.mean(allm))
    for c in placed_t:
        both = np.isfinite(placed_t[c]) & np.isfinite(Zt)
        d = placed_t[c][both] - Zt[both] - const
        out[c]["after_rms_group_const"] = rms(d); out[c]["after_peak_group_const"] = float(np.max(np.abs(d)))
    # between-member tissue consistency after
    cids = list(placed_t)
    pair_rms = {}
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            a, b = placed_t[cids[i]], placed_t[cids[j]]
            both = np.isfinite(a) & np.isfinite(b)
            pair_rms[f"{cids[i]}-{cids[j]}"] = rms(a[both] - b[both])
    out["_group_const"] = const; out["_pair_tissue_rms_after"] = pair_rms
    # consensus vs truth
    both = np.isfinite(cons["curve"]) & np.isfinite(Zt)
    d = cons["curve"][both] - Zt[both]
    out["_consensus_vs_truth"] = dict(rms=rms(d), rms_minus_const=rms(d - np.mean(d)), mean=float(np.mean(d)), peak=float(np.max(np.abs(d))))
    # the divot preserved? consensus − truth near lateral 300 (canvas) vs away
    l0 = canvas["origin"][0]
    lc = np.arange(canvas["shape"][0]) + l0
    near = (np.abs(lc - DIVOT_L) <= 6)
    cn = cons["curve"][near]; zn = Zt[near]; bn = np.isfinite(cn) & np.isfinite(zn)
    out["_divot_residual_rms_minus_const"] = rms((cn[bn] - zn[bn]) - np.mean(d))
    return fin, out


def provider_from_reps(reps):
    def provider(anchor, mov):
        T = pair_to(reps[mov], reps[anchor])
        return {"ok": True, "df": T["df"], "dx": T["dx"], "a": T["a"], "b": T["b"], "rel": 1.2, "ncc_coarse": 0.9, "flags": []}
    return provider
