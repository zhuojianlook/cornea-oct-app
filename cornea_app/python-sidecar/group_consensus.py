"""group_consensus — consensus v2: MAJORITY OWN DOME, AXIAL-CHECKED (element 5; reviewer's point 2026-09-11:
"technically no single replicate is a reference").

The pose rule's reference is only the COORDINATE ANCHOR. The provisional consensus (group_job.consensus_curve on the
tissue-PLACED served lines) inherited the anchor's smooth dome almost entirely: the tissue registration maps every
member's line onto the anchor's line (up to line error), so the other members voted only through line noise and the
curvature moved 28 % when the anchor changed (prototype study wf_e5). Here the DOME (the along-frame quadratic
coefficient c2 per lateral) is decided by a MAJORITY of the members' OWN domes and the PLACEMENT (c1, c0) by the
tissue-placed lines:

  (a) DOME VOTERS = the reference + every member whose pair is ok (+ vote-only members: a refused pair with a strong
      relative match, VOTE_ONLY_REL / VOTE_ONLY_NCC — its own line is independent of the refused transform; it gets no
      transform). Per voter and per OWN lateral: a robust quadratic along its OWN frames of its OWN served line on its
      own evidence cells (valid — crop bands / zero A-scans excluded — and not a surface-crop frame) → c2_m(l) with a
      standard error; κ_m = 2 c2 dz/df² [1/mm] with the MEMBER's spacing; carried to canvas laterals by df and the
      median dx only (never a / b: those would map the vote onto the anchor's dome).
  (b) THE DECISION κ*(lc) per lateral, by the number of votes n (revision 3, after the synthetic-truth refutation of
      2026-09-11 — with n = 3 the tightest PAIR has no information about which pair is right: honest errors ±h at the
      frame ends let a wrong member +w win whenever w < 3h, biasing κ* by 6–18 % with no flag):
        n = 1 → 'single' (that voter's own dome: its error is that voter's own error).
        n = 2 → the se⁻²-weighted mean when the two sit within 2τ ('majority'); otherwise the CALIBRATED axial tie-break
                ('axial', below) when one exists, else the value of the nearest decided lateral ('continuity'), and only
                when nothing is decided anywhere the reference's own vote ('reference', flagged).
        n = 3 → the per-lateral MEDIAN of the three ('median3'; se-agnostic). Its error is bounded by the LARGEST HONEST
                error: with R −h, B +h, A +w (w > h) the median is B (error h), whereas the tightest pair (R, A) errs up to
                ~2h. So with three replicates the dome is accurate to the honest replicates' OWN error; a wrong replicate
                is out-voted only with ≥ 4 replicates or a trusted axial witness — and the honest majority itself
                inherits its voters' errors (median +4 %, p95 8.5 % with 5–8 px honest errors): a 5 % / 90 % bar is not
                what n voters can guarantee; the bar is max(honest error).
        n ≥ 4 → τ = max(TAU_REL × median|κ|, TAU_ABS); k = ⌊n/2⌋+1; the tightest k-subset; range ≤ 2τ → the se⁻²-weighted
                mean of the subset ('majority'; a single wrong member never wins); no majority → the calibrated axial
                tie-break, else continuity, else the reference (as for n = 2).
      CALIBRATED AXIAL TIE-BREAK: κ_axial sits ~14 % below κ_frames on a legitimately toric cornea, so comparing the
      votes with the raw witness picked the WRONG anchor (2-vote canvas edges). The tie-break compares with κ_axial(l)·ρ
      where ρ = the group's own median κ*/κ_axial over the majority-decided laterals (a calibrated toricity; ≥
      RHO_MIN_LATERALS decided laterals; ρ also cancels an unverified lateral scale, so it needs no trusted header): the
      voter nearest κ_axial·ρ + every voter within τ of it ('axial'). Group flag dome_majority_failed when 'reference' /
      'median' decide > UNDECIDED_FRAC of the voted laterals. c2* is smoothed across laterals (running median SMOOTH_MED
      + Savitzky-Golay SMOOTH_SG: quadratics change little between laterals). c1 from the tissue-placed lines with c2*
      pinned (robust, smoothed alike); c0 refitted with c2*, c1 pinned and only LIGHTLY smoothed (C0_MED / C0_SG: divots
      and the limbus are real). Evaluated on covered cells only (no extrapolation).
  (c) AXIAL WITNESS per lateral band (N_BANDS fifths of the reference's laterals): κ_frames = 2 c2* dz/df² (median over
      the band) vs κ_axial = median over voters and frames of 2 q2 dz/dl_m² from each member's WITHIN-B-scan
      quadratic across ± AXIAL_HALF laterals about the band centre (mapped by its median dx, its OWN lateral
      spacing); the ratio must sit inside the toricity envelope RATIO_ENVELOPE (CS001 measured 1.0–1.2: sphere-vs-
      parabola bias ≤ 7 %, toricity ≤ 10 %, off-axis factor at the periphery) → 'witnessed', else group flag
      dome_axial_mismatch with the numbers. A header lateral spacing equal to the legacy 4.0/513 mm cube
      (LEGACY_LATERAL_MM: the legacy .dcm path stamped a 4 × 4 × 4 mm cube while the true XY scan size varies
      4.6–6.0 mm; the frame interval 0.040 mm is reliable) is NOT trusted: the ratio is also reported under
      SCAN_SIZES_MM (it scales with dl²), the group is flagged lateral_scale_unverified and the axial tie-break is
      unavailable.
  (d) REFERENCE SENSITIVITY (group_job.reference_sensitivity_stage): the whole thing re-run with every other member
      as the anchor and its consensus mapped into the served coordinates by the served pair → the spread (RMS over
      covered cells). The record keeps the rule's anchor as the served one.

The APPLY step (group_job.final_transforms): every member moves by the smooth-to-smooth shift + tilt of (consensus −
its OWN dome) — LINE ERROR NEVER MOVES TISSUE — fitted per frame on a FIXED inlier set of laterals and smoothed along
frames (revision 3: the per-frame 3×MAD inlier set flipped frame to frame and stepped the applied δa / δb).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
from scipy.signal import savgol_filter

CONSENSUS_VERSION = 2
CONSENSUS_REVISION = 3    # 3: n = 3 → per-lateral median; calibrated axial tie-break (ρ); continuity fallback; smooth apply
RHO_MIN_LATERALS = 30     # decided laterals needed to calibrate the toricity ρ = κ*/κ_axial for the tie-break
TAU_REL = 0.20            # majority tolerance on κ, relative to the median |κ| of the voters
TAU_ABS = 0.015           # … and its absolute floor, 1/mm (≈ 12 % of a 0.128 cornea)
SE_FLOOR = 0.002          # 1/mm — floor of a vote's standard error (the se⁻² weights)
RATIO_ENVELOPE = (0.75, 1.35)   # κ_frames / κ_axial: toricity envelope of the axial witness
N_BANDS = 5
LEGACY_LATERAL_MM = 4.0 / 513.0
LEGACY_TOL_MM = 2e-6
SCAN_SIZES_MM = (4.0, 4.6, 6.0)  # the ratio re-expressed for these XY scan sizes when the header scale is legacy
MIN_FRAMES = 30           # own-dome fit: evidence frames per lateral
MIN_SPAN = 40             # … spanning at least this many frames
AXIAL_HALF = 100          # laterals either side of the band centre in the within-B-scan quadratic
AXIAL_MIN_LAT = 60
AXIAL_MIN_SPAN = 120
SMOOTH_MED = 31           # c2* / c1 across laterals: running median + Savitzky-Golay (order 2)
SMOOTH_SG = 151
C0_MED = 5                # c0 across laterals: LIGHT robust smoothing only
C0_SG = 15
VOTE_ONLY_REL = 1.0       # a refused pair votes (dome only) when relative_match ≥ this …
VOTE_ONLY_NCC = 0.5       # … and ncc_coarse ≥ this
UNDECIDED_FRAC = 0.20     # > this fraction of voted laterals decided by the reference fallback → dome_majority_failed
DISSENT_BANDS = 3         # a voter whose own κ is off κ* by > τ in this many bands 'dissents'
CORRECTION_PX = 8.0       # … and whose implied dome correction at the frame ends exceeds this, unwitnessed → dome_unwitnessed
SPACING_MISMATCH_REL = 0.02
FRAME_SPACING_MISMATCH_REL = 0.01
DOME_SOURCES = ("majority", "median3", "axial", "continuity", "reference", "single", "median", "none")
DECIDED_SOURCES = ("majority", "median3", "axial")   # decisions with information (the continuity fallback copies them)


# ── small helpers ─────────────────────────────────────────────────────────────────────────────────────────────
def x_norm(l, L: int) -> np.ndarray:
    """The engine's tilt coordinate x(l) = (l − (L−1)/2) / ((L−1)/2)."""
    hs = max(1.0, (L - 1) / 2.0)
    return (np.asarray(l, float) - (L - 1) / 2.0) / hs


def kappa_frames(c2_px_per_frame2, spacing) -> np.ndarray:
    """Along-frame quadratic coefficient (px/frame²) → curvature κ = 2 c2 dz/df² [1/mm] (1/R of a sphere)."""
    dz, df = float(spacing[1]), float(spacing[2])
    return 2.0 * np.asarray(c2_px_per_frame2, float) * dz / df ** 2


def kappa_axial(q2_px_per_lat2, spacing) -> np.ndarray:
    """Across-lateral quadratic coefficient (px/lateral²) → κ = 2 q2 dz/dl² [1/mm]."""
    dl, dz = float(spacing[0]), float(spacing[1])
    return 2.0 * np.asarray(q2_px_per_lat2, float) * dz / dl ** 2


def c2_from_kappa(kappa, spacing) -> np.ndarray:
    dz, df = float(spacing[1]), float(spacing[2])
    return np.asarray(kappa, float) * df ** 2 / (2.0 * dz)


def lateral_spacing_is_legacy(spacing) -> bool:
    """The header lateral spacing is the legacy 4.0/513 mm cube (the physical lateral scale is unverified)."""
    return abs(float(spacing[0]) - LEGACY_LATERAL_MM) < LEGACY_TOL_MM


def robust_polyfit(x, y, deg: int, k: float = 3.0, iters: int = 6, min_n: int = 6, floor: float = 0.5):
    """Least squares with iterative MAD rejection (|r| > k·1.4826·MAD, MAD floored at `floor` px).
    Returns (coef high→low, rms over the kept points, n kept, kept mask) or (None, nan, n, mask)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    need = max(min_n, deg + 2)
    if ok.sum() < need:
        return None, float("nan"), int(ok.sum()), ok
    keep = ok.copy()
    c = None
    for _ in range(iters):
        c = np.polyfit(x[keep], y[keep], deg)
        r = y - np.polyval(c, x)
        mad = float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        thr = max(k * 1.4826 * mad, k * floor)
        new = ok & (np.abs(r) <= thr)
        if new.sum() < need or np.array_equal(new, keep):
            break
        keep = new
    r = y - np.polyval(c, x)
    rms = float(np.sqrt(np.mean(r[keep] ** 2)))
    return c, rms, int(keep.sum()), keep


def fit_along_frames(S: np.ndarray, ok: np.ndarray, *, min_frames: int = MIN_FRAMES, min_span: int = MIN_SPAN) -> dict:
    """Per lateral the robust quadratic z = c2 u² + c1 u + c0 along the member's OWN frames, u = f − (F−1)/2
    (px/frame², px/frame, px). se2 = the standard error of c2 (rms × sqrt((XᵀX)⁻¹₀₀) on the kept points)."""
    L, F = S.shape
    u = np.arange(F, dtype=float) - (F - 1) / 2.0
    c2 = np.full(L, np.nan); c1 = np.full(L, np.nan); c0 = np.full(L, np.nan)
    se2 = np.full(L, np.nan); rms = np.full(L, np.nan); n = np.zeros(L, int); span = np.zeros(L, int)
    for l in range(L):
        m = ok[l] & np.isfinite(S[l])
        if m.sum() < min_frames:
            continue
        idx = np.flatnonzero(m)
        span[l] = int(idx[-1] - idx[0] + 1)
        if span[l] < min_span:
            continue
        c, r, k, keep = robust_polyfit(u[m], S[l, m], 2)
        if c is None or not np.all(np.isfinite(c)):
            continue
        c2[l], c1[l], c0[l] = c; rms[l] = r; n[l] = k
        uk = u[m][keep]
        X = np.stack([uk ** 2, uk, np.ones(uk.size)], 1)
        try:
            cov = np.linalg.inv(X.T @ X)
            se2[l] = max(r, 0.5) * float(np.sqrt(max(cov[0, 0], 0.0)))
        except np.linalg.LinAlgError:
            pass
    return {"c2": c2, "c1": c1, "c0": c0, "se2": se2, "rms": rms, "n": n, "span": span, "fc": (F - 1) / 2.0}


def smooth_across_laterals(v: np.ndarray, med: int = SMOOTH_MED, sg: int = SMOOTH_SG) -> np.ndarray:
    """Robust smoothing across laterals over the span of finite values (gaps interpolated for the filters): running
    median `med` then Savitzky-Golay `sg` / order 2. NaN outside the span."""
    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan)
    fin = np.isfinite(v)
    idx = np.flatnonzero(fin)
    if idx.size == 0:
        return out
    span = np.arange(idx[0], idx[-1] + 1)
    x = np.interp(span, idx, v[idx])
    n = span.size
    if n >= 3 and med >= 3:
        x = ndi.median_filter(x, size=min(med, n if n % 2 == 1 else n - 1), mode="nearest")
    win = min(sg, n if n % 2 == 1 else n - 1)
    if win >= 5:
        x = savgol_filter(x, win, 2, mode="interp")
    out[span] = x
    return out


def median_dx(T: dict) -> float:
    """The median lateral shift of a transform over its finite frames (0 when none)."""
    dx = np.asarray(T.get("dx", np.zeros(0)), float)
    dx = dx[np.isfinite(dx)]
    return float(np.median(dx)) if dx.size else 0.0


def lateral_bands(L_ref: int, l0: int, n: int = N_BANDS) -> list[dict]:
    """N equal fifths of the reference's laterals, in reference and canvas laterals (canvas = reference − l0)."""
    edges = np.linspace(0, L_ref, n + 1).astype(int)
    return [{"band": k, "laterals_ref": [int(edges[k]), int(edges[k + 1])],
             "laterals_canvas": [int(edges[k]) - int(l0), int(edges[k + 1]) - int(l0)],
             "centre_ref": 0.5 * (int(edges[k]) + int(edges[k + 1]) - 1)} for k in range(n)]


def own_evidence(m) -> np.ndarray:
    """The member's own evidence cells: valid (crop bands and zero A-scans excluded) and not a surface-crop frame."""
    ok = np.asarray(m.valid, bool) & np.isfinite(np.asarray(m.served, float))
    scf = np.asarray(getattr(m, "surface_crop_frames", np.zeros(0, int)), int)
    if scf.size:
        ok = ok.copy(); ok[:, scf[(scf >= 0) & (scf < ok.shape[1])]] = False
    return ok


# ── (a) own-dome votes ────────────────────────────────────────────────────────────────────────────────────────
def own_dome_votes(voters: list, canvas: dict) -> dict:
    """voters = [(member, transform {df, dx…}, role)]; per voter the own along-frame quadratic per own lateral on its
    own evidence, κ with its OWN spacing, carried to canvas laterals by round(median dx) − l0. Returns cid →
    {kappa_lat (Lc,), se_lat (Lc,), c2_own (L,), se2_own, rms_own, n_own, dx_median, spacing, role, n_voting}."""
    l0 = int(canvas["origin"][0]); Lc = int(canvas["shape"][0])
    out: dict = {}
    for m, T, role in voters:
        S = np.asarray(m.served, float); ok = own_evidence(m)
        L = S.shape[0]
        fit = fit_along_frames(S, ok)
        sp = np.asarray(m.spacing, float)
        kap = kappa_frames(fit["c2"], sp); se = kappa_frames(fit["se2"], sp)
        dxm = median_dx(T)
        kappa_lat = np.full(Lc, np.nan); se_lat = np.full(Lc, np.nan); own_lat = np.full(Lc, -1, int)
        lc = np.arange(L) + int(round(dxm)) - l0
        inside = (lc >= 0) & (lc < Lc) & np.isfinite(kap)
        kappa_lat[lc[inside]] = kap[inside]; se_lat[lc[inside]] = se[inside]; own_lat[lc[inside]] = np.arange(L)[inside]
        out[m.cid] = {"kappa_lat": kappa_lat, "se_lat": se_lat, "own_lat": own_lat, "c2_own": fit["c2"], "c1_own": fit["c1"],
                      "c0_own": fit["c0"], "se2_own": fit["se2"], "rms_own": fit["rms"], "n_own": fit["n"], "fc_own": fit["fc"],
                      "dx_median": dxm, "spacing": sp, "role": role, "n_voting": int(inside.sum()),
                      "line_rms_median_px": (float(np.nanmedian(fit["rms"])) if np.isfinite(fit["rms"]).any() else None)}
    return out


# ── (c) the axial witness (within-B-scan curvature, physical units) ───────────────────────────────────────────
def axial_witness(voters: list, votes: dict, bands: list, canvas: dict, *, half: int = AXIAL_HALF,
                  min_lat: int = AXIAL_MIN_LAT, min_span: int = AXIAL_MIN_SPAN) -> dict:
    """Per voter and band: the median over its own frames of κ_axial = 2 q2 dz/dl_m² where q2 is the robust quadratic
    across the laterals within ± half of the band centre (mapped to the member's OWN laterals by its median dx) on its
    own evidence cells. Group κ_axial(band) = median over voters; interpolated across band centres to every canvas
    lateral (the tie-break voter). Returns {per_member: {cid: {kappa: [nb], n_frames: [nb], mad: [nb]}}, group: [nb],
    lateral (Lc,), scale_trusted, lateral_spacing_mm: {cid}, legacy: [cid…]}."""
    l0 = int(canvas["origin"][0]); Lc = int(canvas["shape"][0])
    per: dict = {}; legacy: list = []; spacing_mm: dict = {}
    for m, T, role in voters:
        S = np.asarray(m.served, float); ok = own_evidence(m)
        L, F = S.shape
        sp = np.asarray(m.spacing, float); spacing_mm[m.cid] = float(sp[0])
        if lateral_spacing_is_legacy(sp):
            legacy.append(m.cid)
        dxm = votes[m.cid]["dx_median"] if m.cid in votes else median_dx(T)
        lat = np.arange(L, dtype=float)
        rec = {"kappa": [], "n_frames": [], "mad": [], "centre_own": []}
        for b in bands:
            cen = float(b["centre_ref"]) - dxm                    # the band centre in the member's OWN laterals
            w0 = (lat >= cen - half) & (lat <= cen + half)
            q2 = []
            for f in range(F):
                w = w0 & ok[:, f]
                if w.sum() < min_lat:
                    continue
                idx = np.flatnonzero(w)
                if idx[-1] - idx[0] < min_span:
                    continue
                c, _r, _n, _k = robust_polyfit(lat[w] - cen, S[w, f], 2, min_n=min_lat)
                if c is not None and np.isfinite(c[0]):
                    q2.append(c[0])
            if q2:
                kq = kappa_axial(np.asarray(q2), sp)
                rec["kappa"].append(float(np.median(kq))); rec["n_frames"].append(int(kq.size))
                rec["mad"].append(float(1.4826 * np.median(np.abs(kq - np.median(kq)))))
            else:
                rec["kappa"].append(float("nan")); rec["n_frames"].append(0); rec["mad"].append(float("nan"))
            rec["centre_own"].append(cen)
        per[m.cid] = rec
    nb = len(bands)
    group = np.full(nb, np.nan)
    for k in range(nb):
        vals = np.array([per[c]["kappa"][k] for c in per], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            group[k] = float(np.median(vals))
    cen_c = np.array([b["centre_ref"] - l0 for b in bands], float)
    okb = np.isfinite(group)
    lateral = (np.interp(np.arange(Lc), cen_c[okb], group[okb]) if okb.sum() >= 2
               else (np.full(Lc, group[okb][0]) if okb.sum() == 1 else np.full(Lc, np.nan)))
    return {"per_member": per, "group": group, "lateral": lateral, "scale_trusted": not legacy, "legacy": legacy,
            "lateral_spacing_mm": spacing_mm, "half_laterals": int(half)}


# ── (b) the majority ──────────────────────────────────────────────────────────────────────────────────────────
def tight_majority(vals: np.ndarray) -> tuple[np.ndarray, float, int]:
    """The tightest ⌊n/2⌋+1 subset of the finite values → (indices, range, k)."""
    v = np.asarray(vals, float)
    idx = np.flatnonzero(np.isfinite(v))
    n = idx.size
    if n == 0:
        return np.zeros(0, int), float("nan"), 0
    k = n // 2 + 1
    order = idx[np.argsort(v[idx], kind="stable")]
    best = None
    for i in range(n - k + 1):
        rng = float(v[order[i + k - 1]] - v[order[i]])
        if best is None or rng < best[1]:
            best = (order[i:i + k], rng)
    return best[0], best[1], k


def majority_kappa(votes: dict, ref_cid: str, Lc: int, kax_lat: np.ndarray | None, scale_trusted: bool, *,
                   tau_rel: float = TAU_REL, tau_abs: float = TAU_ABS, rho_min_laterals: int = RHO_MIN_LATERALS) -> dict:
    """Per canvas lateral the decided κ* (module docstring, rule (b)): n = 1 'single'; n = 3 the per-lateral MEDIAN
    ('median3'); n = 2 / n ≥ 4 the tightest ⌊n/2⌋+1 subset within 2τ ('majority'); otherwise — in this order — the
    CALIBRATED axial tie-break ('axial': the voter nearest κ_axial(l)·ρ, ρ = median κ*/κ_axial over the decided laterals,
    ≥ rho_min_laterals of them), the nearest decided lateral's value ('continuity'), the reference's own vote
    ('reference'), the median ('median'). `scale_trusted` is recorded only: ρ cancels the lateral scale. Returns {kappa
    (Lc,), source (Lc,) str, n_votes (Lc,), tau (Lc,), in_majority {cid: (Lc,) bool}, voters: [cid…], toricity: {rho,
    n_laterals, calibrated, scale_trusted}}."""
    cids = list(votes)
    K = np.stack([votes[c]["kappa_lat"] for c in cids], 0) if cids else np.zeros((0, Lc))
    SE = np.stack([votes[c]["se_lat"] for c in cids], 0) if cids else np.zeros((0, Lc))
    W = 1.0 / np.maximum(np.where(np.isfinite(SE), SE, np.inf), SE_FLOOR) ** 2
    kappa = np.full(Lc, np.nan); src = np.full(Lc, "none", dtype="<U10"); nv = np.zeros(Lc, int); tau = np.full(Lc, np.nan)
    inmaj = {c: np.zeros(Lc, bool) for c in cids}
    i_ref = cids.index(ref_cid) if ref_cid in cids else -1
    decided = np.zeros(Lc, bool)
    pending: list = []

    def _mark_near(l: int, value: float, t: float) -> None:
        v = K[:, l]
        near = np.isfinite(v) & (np.abs(v - value) <= t)
        for j2 in np.flatnonzero(near):
            inmaj[cids[j2]][l] = True

    for l in range(Lc):
        v = K[:, l] if K.size else np.zeros(0)
        fin = np.isfinite(v)
        n = int(fin.sum()); nv[l] = n
        if n == 0:
            continue
        if n == 1:
            j = int(np.flatnonzero(fin)[0])
            kappa[l] = v[j]; src[l] = "single"; inmaj[cids[j]][l] = True
            continue
        t = max(tau_rel * float(np.median(np.abs(v[fin]))), tau_abs); tau[l] = t
        if n == 3:                                     # the median: error ≤ the largest honest error
            kappa[l] = float(np.median(v[fin])); src[l] = "median3"; decided[l] = True
            _mark_near(l, kappa[l], t)
            continue
        sub, rng, k = tight_majority(v)
        if rng <= 2.0 * t:
            w = W[sub, l]; w = w if np.isfinite(w).all() and w.sum() > 0 else np.ones(sub.size)
            kappa[l] = float(np.sum(v[sub] * w) / np.sum(w)); src[l] = "majority"; decided[l] = True
            for j in sub:
                inmaj[cids[j]][l] = True
            continue
        pending.append(l)
    # the calibrated toricity ρ from the decided laterals (the tie-break's witness)
    rho = None; n_rho = 0
    if kax_lat is not None and decided.any():
        kx = np.asarray(kax_lat, float)
        okr = decided & np.isfinite(kappa) & np.isfinite(kx) & (kx > 0) & (kappa > 0)
        n_rho = int(okr.sum())
        if n_rho >= rho_min_laterals:
            rho = float(np.median(kappa[okr] / kx[okr]))
    still: list = []
    for l in pending:
        v = K[:, l]; fin = np.isfinite(v); t = tau[l]
        if rho is not None and np.isfinite(kax_lat[l]):
            target = float(kax_lat[l]) * rho
            d = np.where(fin, np.abs(v - target), np.inf)
            j = int(np.argmin(d))
            near = fin & (np.abs(v - v[j]) <= t)
            w = W[near, l]; w = w if np.isfinite(w).all() and w.sum() > 0 else np.ones(int(near.sum()))
            kappa[l] = float(np.sum(v[near] * w) / np.sum(w)); src[l] = "axial"; decided[l] = True
            for j2 in np.flatnonzero(near):
                inmaj[cids[j2]][l] = True
            continue
        still.append(l)
    dec_idx = np.flatnonzero(decided)
    for l in still:
        v = K[:, l]; fin = np.isfinite(v); t = tau[l]
        if dec_idx.size:
            j = int(dec_idx[int(np.argmin(np.abs(dec_idx - l)))])
            kappa[l] = kappa[j]; src[l] = "continuity"
            _mark_near(l, kappa[l], t)
            continue
        if i_ref >= 0 and fin[i_ref]:
            kappa[l] = v[i_ref]; src[l] = "reference"
            inmaj[ref_cid][l] = True
            _mark_near(l, kappa[l], t)
            continue
        kappa[l] = float(np.median(v[fin])); src[l] = "median"
        _mark_near(l, kappa[l], t)
    return {"kappa": kappa, "source": src, "n_votes": nv, "tau": tau, "in_majority": inmaj, "voters": cids,
            "toricity": {"rho": rho, "n_laterals": n_rho, "calibrated": rho is not None, "scale_trusted": bool(scale_trusted),
                         "min_laterals": int(rho_min_laterals)}}


# ── placement (c1, c0) from the tissue-placed lines ───────────────────────────────────────────────────────────
def placement_fit(lines: dict, c2: np.ndarray, fc: float, *, min_frames: int = MIN_FRAMES) -> dict:
    """Per canvas lateral with c2 pinned: robust line fit of (z − c2 u²) ≈ c0 + c1 u through every member's placed
    line, u = canvas frame − fc. Returns {c1, c0, rms, n (Lc,)}."""
    cids = list(lines)
    Lc, Fc = next(iter(lines.values())).shape if cids else (c2.size, 0)
    u_all = np.arange(Fc, dtype=float) - fc
    c1 = np.full(Lc, np.nan); c0 = np.full(Lc, np.nan); rms = np.full(Lc, np.nan); n = np.zeros(Lc, int)
    for l in range(Lc):
        if not np.isfinite(c2[l]):
            continue
        us = []; zs = []
        for c in cids:
            row = lines[c][l]; ok = np.isfinite(row)
            if ok.any():
                us.append(u_all[ok]); zs.append(row[ok] - c2[l] * u_all[ok] ** 2)
        if not us:
            continue
        u = np.concatenate(us); z = np.concatenate(zs)
        if np.unique(u).size < min_frames:
            continue
        c, r, k, _ = robust_polyfit(u, z, 1, min_n=12)
        if c is None or not np.all(np.isfinite(c)):
            continue
        c1[l], c0[l] = c; rms[l] = r; n[l] = k
    return {"c1": c1, "c0": c0, "rms": rms, "n": n}


def refit_c0(lines: dict, c2: np.ndarray, c1: np.ndarray, fc: float, *, min_frames: int = MIN_FRAMES) -> np.ndarray:
    """Per canvas lateral with c2 and c1 pinned: the robust mean (3×MAD) of z − c2 u² − c1 u over every member's placed
    line — the intercept that puts the smoothed quadratic through the tissue-placed lines."""
    cids = list(lines)
    Lc, Fc = next(iter(lines.values())).shape
    u_all = np.arange(Fc, dtype=float) - fc
    c0 = np.full(Lc, np.nan)
    for l in range(Lc):
        if not (np.isfinite(c2[l]) and np.isfinite(c1[l])):
            continue
        us = []; zs = []
        for c in cids:
            row = lines[c][l]; ok = np.isfinite(row)
            if ok.any():
                us.append(u_all[ok]); zs.append(row[ok])
        if not us:
            continue
        u = np.concatenate(us); z = np.concatenate(zs)
        if np.unique(u).size < min_frames:
            continue
        r = z - c2[l] * u ** 2 - c1[l] * u
        keep = np.ones(r.size, bool)
        for _ in range(4):
            med = float(np.median(r[keep])); mad = float(np.median(np.abs(r[keep] - med)))
            new = np.abs(r - med) <= max(3.0 * 1.4826 * mad, 1.5)
            if new.sum() < 6 or np.array_equal(new, keep):
                break
            keep = new
        c0[l] = float(np.mean(r[keep]))
    return c0


# ── the consensus v2 ──────────────────────────────────────────────────────────────────────────────────────────
def consensus_v2(ref, members: list, transforms: dict, lines: dict, colmask: np.ndarray, canvas: dict, *,
                 vote_only: list | None = None, pair_info: dict | None = None, log=None) -> dict:
    """The consensus v2 on the union canvas. `members` = the contributing MemberData (reference first, all on the
    common lateral grid); `transforms` cid → {df, dx, a, b}; `lines` cid → the member's tissue-PLACED served line (Lc,
    Fc) (group_job.place_on_canvas); `colmask` (Lc, Fc) covered cells; `vote_only` = [(member, {df, dx})] refused
    members that vote on the dome only; `pair_info` cid → {ok, rel, ncc_coarse} for the record. Returns a dict with
    curve (Lc, Fc) NaN off coverage, coef (Lc, 3) [c0, c1, c2] smooth (u = canvas frame − fc, px/frame units) and raw,
    fitted (Lc,), kappa_star (Lc,), dome_source (Lc,), tau, n_votes, votes, axial, bands, voters, flags, verdicts…"""
    t0 = time.time()
    Lc, Fc = colmask.shape
    l0 = int(canvas["origin"][0])
    fc = (Fc - 1) / 2.0
    sp_ref = np.asarray(ref.spacing, float)
    voters = [(m, transforms[m.cid], "reference" if m.cid == ref.cid else "contributing") for m in members]
    for m, T in (vote_only or []):
        voters.append((m, T, "vote_only"))
    L_ref = int(ref.served.shape[0])
    bands = lateral_bands(L_ref, l0)
    # spacing sanity (physical units drive every comparison)
    flags: list = []
    sp_l = {m.cid: float(np.asarray(m.spacing, float)[0]) for m, _T, _r in voters}
    sp_f = {m.cid: float(np.asarray(m.spacing, float)[2]) for m, _T, _r in voters}
    if sp_l and (max(sp_l.values()) / max(min(sp_l.values()), 1e-12) - 1.0) > SPACING_MISMATCH_REL:
        flags.append("lateral_scale_mismatch")
    if sp_f and (max(sp_f.values()) / max(min(sp_f.values()), 1e-12) - 1.0) > FRAME_SPACING_MISMATCH_REL:
        flags.append("frame_spacing_mismatch")
    # (a) own-dome votes, (c) the axial witness (needed by the tie-break)
    votes = own_dome_votes(voters, canvas)
    axial = axial_witness(voters, votes, bands, canvas)
    if not axial["scale_trusted"]:
        flags.append("lateral_scale_unverified")
    # (b) the majority per lateral → c2* in the CANVAS units (the reference's frame / depth spacing)
    maj = majority_kappa(votes, ref.cid, Lc, axial["lateral"], axial["scale_trusted"])
    voted = np.isfinite(maj["kappa"])
    c2_raw = c2_from_kappa(maj["kappa"], sp_ref)
    c2s = smooth_across_laterals(c2_raw, SMOOTH_MED, SMOOTH_SG)
    # placement from the tissue-placed lines: c1 (smoothed like c2), then c0 refitted and lightly smoothed
    pl = placement_fit(lines, c2s, fc)
    c1s = smooth_across_laterals(pl["c1"], SMOOTH_MED, SMOOTH_SG)
    c0_raw = refit_c0(lines, c2s, c1s, fc)
    c0s = smooth_across_laterals(c0_raw, C0_MED, C0_SG)
    fitted = np.isfinite(c2s) & np.isfinite(c1s) & np.isfinite(c0s) & colmask.any(axis=1)
    u = np.arange(Fc, dtype=float) - fc
    curve = np.full((Lc, Fc), np.nan)
    for l in np.flatnonzero(fitted):
        row = c0s[l] + c1s[l] * u + c2s[l] * u * u
        row[~colmask[l]] = np.nan
        curve[l] = row
    coef = np.stack([c0s, c1s, c2s], 1); coef_raw = np.stack([c0_raw, pl["c1"], c2_raw], 1)
    # dome sources
    src = maj["source"]
    counts = {s: int((src == s).sum()) for s in DOME_SOURCES}
    n_voted = int(voted.sum())
    undecided = (counts["reference"] + counts["median"]) / max(1, n_voted)
    n_decided = sum(counts[s] for s in DECIDED_SOURCES)
    if n_voted and undecided > UNDECIDED_FRAC:
        flags.append("dome_majority_failed")
    # per band: κ_frames (from the smoothed c2*), κ_axial, the ratio, the verdict, every voter's own κ / dissent
    kap_s = kappa_frames(c2s, sp_ref)
    apex_band = None
    if np.isfinite(c0s).any():
        lr = np.arange(L_ref) - l0
        lr = lr[(lr >= 0) & (lr < Lc)]
        cand = np.where(np.isfinite(c0s[lr]), c0s[lr], np.inf)
        if np.isfinite(cand).any():
            l_apex = int(lr[int(np.argmin(cand))]) + l0
            apex_band = next((b["band"] for b in bands if b["laterals_ref"][0] <= l_apex < b["laterals_ref"][1]), None)
    lo_env, hi_env = RATIO_ENVELOPE
    band_recs = []
    for b in bands:
        a0, a1 = b["laterals_canvas"]; a0 = max(0, a0); a1 = min(Lc, a1)
        sl = slice(a0, a1)
        kf = float(np.nanmedian(kap_s[sl])) if np.isfinite(kap_s[sl]).any() else float("nan")
        ka = float(axial["group"][b["band"]])
        tau_b = float(np.nanmedian(maj["tau"][sl])) if np.isfinite(maj["tau"][sl]).any() else float("nan")
        ratio = kf / ka if (np.isfinite(kf) and np.isfinite(ka) and ka != 0) else float("nan")
        if not (np.isfinite(kf) and np.isfinite(ka)):
            verdict = "no_data"
        elif ka <= 0 or kf <= 0:
            verdict = "sign"
        elif lo_env <= ratio <= hi_env:
            verdict = "witnessed"
        else:
            verdict = "mismatch"
        vb = {}
        for c in maj["voters"]:
            kl = votes[c]["kappa_lat"][sl]; okl = np.isfinite(kl) & voted[sl]
            own_k = float(np.nanmedian(kl)) if np.isfinite(kl).any() else float("nan")
            vb[c] = {"kappa": own_k, "dissent": (own_k - kf if np.isfinite(own_k) and np.isfinite(kf) else float("nan")),
                     "in_majority_frac": (float(np.mean(maj["in_majority"][c][sl][okl])) if okl.any() else None),
                     "n_laterals": int(np.isfinite(kl).sum()),
                     "kappa_axial": float(axial["per_member"][c]["kappa"][b["band"]]),
                     "kappa_axial_frames": int(axial["per_member"][c]["n_frames"][b["band"]])}
        rec = {**{k: v for k, v in b.items() if k != "centre_ref"}, "centre_ref": float(b["centre_ref"]), "apex": (b["band"] == apex_band),
               "kappa_frames": kf, "kappa_axial": ka, "ratio": ratio, "verdict": verdict, "tau": tau_b,
               "sources": {s: int((src[sl] == s).sum()) for s in DOME_SOURCES}, "votes": vb}
        if not axial["scale_trusted"] and np.isfinite(ratio):
            rec["ratio_at_scan_size_mm"] = {f"{s:.1f}": float(ratio * (s / 4.0) ** 2) for s in SCAN_SIZES_MM}
        band_recs.append(rec)
    verdicts = [b["verdict"] for b in band_recs]
    if any(v in ("mismatch", "sign") for v in verdicts):
        flags.append("dome_axial_mismatch")
        axial_verdict = "mismatch" if axial["scale_trusted"] else "mismatch_scale_unverified"
    elif any(v == "witnessed" for v in verdicts):
        axial_verdict = "witnessed" if axial["scale_trusted"] else "witnessed_scale_unverified"
    else:
        axial_verdict = "no_data"
    n_voters = len(maj["voters"])
    maj_bands = sum(1 for b in band_recs if (b["sources"]["majority"] + b["sources"]["median3"])
                    > 0.5 * max(1, sum(b["sources"][s] for s in DOME_SOURCES if s != "none")))
    if n_voters >= 4 and maj_bands >= 4:
        dome_verdict = "witnessed_majority"           # a single wrong replicate is out-voted
    elif n_voters == 3 and maj_bands >= 4:
        dome_verdict = "median_of_three"              # accurate to the honest replicates' own error; a wrong one is NOT out-voted
    elif n_voters >= 2 and (counts["majority"] + counts["median3"]) > 0:
        dome_verdict = "majority_partial" if maj_bands < 4 else "majority_two_voters"
    elif n_voters == 1:
        dome_verdict = "single_voter"
    else:
        dome_verdict = "undecided"
    for b in band_recs:                               # the band's own toricity (κ*/κ_axial) next to the calibrated ρ
        b["rho"] = b["ratio"]
    if len(members) < 2:
        flags.append("reference_only")
    # per-voter records + flags (WARNINGS only: the numbers carry the corrections)
    hf = (Fc - 1) / 2.0
    voter_recs = []
    for c in maj["voters"]:
        v = votes[c]
        dis = [b["votes"][c]["dissent"] for b in band_recs]
        taus = [b["tau"] for b in band_recs]
        n_dis = sum(1 for d, t in zip(dis, taus) if np.isfinite(d) and np.isfinite(t) and abs(d) > t)
        # the correction the majority implies for this voter at the frame ends (px): Δc2 · hf²
        kl = v["kappa_lat"]; both = np.isfinite(kl) & np.isfinite(maj["kappa"])
        dc2 = c2_from_kappa(np.nanmedian(kl[both] - maj["kappa"][both]) if both.any() else np.nan, sp_ref)
        corr_px = float(abs(dc2) * hf ** 2) if np.isfinite(dc2) else None
        # witnessed = in every dissenting band ≥ 2 OTHER voters sit within τ of κ*, or the axial witness confirms κ*
        others_agree = []
        for b in band_recs:
            if not (np.isfinite(b["votes"][c]["dissent"]) and np.isfinite(b["tau"]) and abs(b["votes"][c]["dissent"]) > b["tau"]):
                continue
            n_oth = sum(1 for o in maj["voters"] if o != c and np.isfinite(b["votes"][o]["dissent"]) and abs(b["votes"][o]["dissent"]) <= b["tau"])
            others_agree.append(n_oth >= 2 or (axial["scale_trusted"] and b["verdict"] == "witnessed"))
        witnessed = all(others_agree) if others_agree else True
        fl: list = []
        info = pair_info.get(c, {}) if pair_info else {}
        if v["role"] == "vote_only":
            fl.append("vote_only")
        if n_dis >= DISSENT_BANDS and corr_px is not None and corr_px > CORRECTION_PX and not witnessed:
            fl.append("dome_unwitnessed")
        inm = [b["votes"][c]["in_majority_frac"] for b in band_recs]
        note = None
        if n_dis >= DISSENT_BANDS:
            note = (f"own dome off the majority in {n_dis}/{len(band_recs)} bands (≈ {corr_px:.1f} px at the frame ends); "
                    + ("corrected by the smooth dome move, witnessed by the other voters" if witnessed and v["role"] != "vote_only"
                       else ("votes only (refused transform)" if v["role"] == "vote_only" else "NOT witnessed")))
        voter_recs.append({"cid": c, "role": v["role"], "spacing_mm": [float(x) for x in v["spacing"]], "dx_median": v["dx_median"],
                           "n_laterals_voting": v["n_voting"], "line_rms_median_px": v["line_rms_median_px"],
                           "kappa_by_band": [b["votes"][c]["kappa"] for b in band_recs],
                           "dissent_by_band": dis, "in_majority_by_band": inm, "dissent_bands": n_dis,
                           "correction_at_frame_ends_px": corr_px, "witnessed": witnessed,
                           "kappa_axial_by_band": [b["votes"][c]["kappa_axial"] for b in band_recs],
                           "kappa_own_median": (float(np.nanmedian(kl)) if np.isfinite(kl).any() else None),
                           "pair_ok": info.get("ok"), "rel_struct": info.get("rel"), "ncc_coarse": info.get("ncc_coarse"),
                           "flags": fl, "note": note})
    if log:
        log(f"    consensus v2: voters {maj['voters']} sources {counts} axial {axial_verdict} dome {dome_verdict} flags {flags} {time.time() - t0:.1f}s")
    return {"curve": curve, "coef": coef, "coef_raw": coef_raw, "fc": fc, "scale": 1.0, "fitted": fitted,
            "n_pts": pl["n"], "lateral_range": ([int(np.flatnonzero(fitted)[0]), int(np.flatnonzero(fitted)[-1])] if fitted.any() else None),
            "kappa_star": maj["kappa"], "dome_source": src, "tau": maj["tau"], "n_votes": maj["n_votes"], "in_majority": maj["in_majority"],
            "votes": votes, "axial": axial, "bands": band_recs, "voters": voter_recs, "voter_ids": maj["voters"],
            "source_counts": counts, "voted_laterals": n_voted, "undecided_fraction": float(undecided),
            "decided_laterals": int(n_decided), "toricity": maj["toricity"], "n_voters": int(n_voters),
            "apex_band": apex_band, "axial_verdict": axial_verdict, "dome_verdict": dome_verdict, "flags": flags,
            "placement_rms": pl["rms"], "seconds": time.time() - t0}


def summary_of(cons: dict) -> dict:
    """The JSON-able record of a consensus_v2 result (no per-lateral arrays)."""
    fin = lambda v: (None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v)  # noqa: E731
    tor = dict(cons.get("toricity") or {})
    return {"version": CONSENSUS_VERSION, "revision": CONSENSUS_REVISION,
            "dome": {"rule": ("per lateral, by the number of votes n: n = 1 single; n = 3 the MEDIAN of the three scans' OWN along-frame "
                              "domes (median3: error <= the largest honest error; a wrong scan is out-voted only with >= 4 scans or an axial "
                              "witness); n = 2 / n >= 4 the tightest floor(n/2)+1 subset within 2 tau (majority, se^-2 weighted); no majority "
                              "-> the vote nearest kappa_axial x rho (axial; rho = the group's own median kappa*/kappa_axial over the decided "
                              "laterals, a calibrated toricity), else the nearest decided lateral's value (continuity), else the reference's "
                              "own vote; c2* smoothed across laterals"),
                     "guarantee": ("with n honest-error replicates the decided dome is accurate to the honest replicates' own error (n = 3: "
                                   "bounded by max honest error, the wrong replicate never decides alone); a single wrong replicate is "
                                   "out-voted only with n >= 4 or the calibrated axial witness"),
                     "n_voters": cons.get("n_voters"), "decided_laterals": cons.get("decided_laterals"),
                     "toricity": {"rho": fin(tor.get("rho")), "n_laterals": tor.get("n_laterals"), "calibrated": tor.get("calibrated"),
                                  "min_laterals": tor.get("min_laterals"),
                                  "note": "rho = median kappa*/kappa_axial over the decided laterals; the tie-break compares each vote with "
                                          "kappa_axial(l) x rho (rho cancels an unverified lateral scale)"},
                     "tau_rel": TAU_REL, "tau_abs_per_mm": TAU_ABS, "sources": cons["source_counts"], "voted_laterals": cons["voted_laterals"],
                     "undecided_fraction": cons["undecided_fraction"], "verdict": cons["dome_verdict"],
                     "smoothing": {"c2_c1": f"running median {SMOOTH_MED} + Savitzky-Golay {SMOOTH_SG} / order 2 across laterals",
                                   "c0": f"light: running median {C0_MED} + Savitzky-Golay {C0_SG} / order 2 (divots / limbus are real)"}},
            "axial_witness": {"envelope": list(RATIO_ENVELOPE), "scale_trusted": bool(cons["axial"]["scale_trusted"]),
                              "legacy_members": list(cons["axial"]["legacy"]), "legacy_lateral_mm": LEGACY_LATERAL_MM,
                              "lateral_spacing_mm": cons["axial"]["lateral_spacing_mm"], "half_laterals": cons["axial"]["half_laterals"],
                              "verdict": cons["axial_verdict"], "apex_band": cons["apex_band"],
                              "rule": (f"per band (5 fifths of the reference's laterals): kappa_frames = 2 c2* dz/df^2 (median over the band) vs "
                                       f"kappa_axial = median over voters and frames of 2 q2 dz/dl^2 (within-B-scan quadratic across +-{AXIAL_HALF} "
                                       f"laterals about the band centre, each member's OWN lateral spacing); witnessed when the ratio sits in "
                                       f"[{RATIO_ENVELOPE[0]}, {RATIO_ENVELOPE[1]}]; a legacy 4.0/513 mm header spacing is not trusted (ratio also "
                                       f"given for {', '.join(f'{s:.1f}' for s in SCAN_SIZES_MM)} mm scan sizes)")},
            "bands": [{k: (fin(v) if not isinstance(v, dict) else {kk: ({kkk: fin(vvv) for kkk, vvv in vv.items()} if isinstance(vv, dict) else fin(vv))
                                                                     for kk, vv in v.items()}) for k, v in b.items()} for b in cons["bands"]],
            "voters": cons["voters"], "voter_ids": cons["voter_ids"], "flags": list(cons["flags"]), "seconds": cons["seconds"]}


# ── (d) the spread of the consensus when another member is the anchor ─────────────────────────────────────────
def frame_ok(T: dict, f: int) -> bool:
    return 0 <= f < T["dx"].size and bool(np.isfinite(T["dx"][f]) and np.isfinite(T["a"][f]) and np.isfinite(T["b"][f]))


def consensus_spread(curve_X: np.ndarray, canvas_X: dict, curve_R: np.ndarray, canvas_R: dict, T_XR: dict, L_X: int, F_X: int,
                     L_R: int | None = None, spacing_R=None) -> dict:
    """The consensus computed with member X as the anchor (curve_X on canvas_X, X's coordinates) mapped into the
    served canvas (canvas_R) by the served rigid pair X → R (df, dx(f), a(f) + b(f)·x) and compared with the served
    consensus curve_R on the cells both cover: spread_px = RMS of Δ, median / p90 |Δ|, mean, n.

    What the raw spread contains: the consensus is a surface in its ANCHOR's coordinates, and the pair engine bends
    every other member's tissue onto the anchor's tissue — so the anchor's OWN smooth dome error is in the coordinate
    system itself (the apply step's dome move removes it by bending the anchor: consensus − its own dome). Mapping the
    X-anchored consensus into the served coordinates by the served pair therefore differs from the served consensus by
    (the anchors' own-dome errors' difference)·u² even when both consensus domes are the same κ* — plus the pair's own
    placement error. So three numbers: spread_px (raw), spread_beyond_pose_px (a whole-volume pose removed: shift +
    tilt in x + linear trend along frames), shape_spread_px (also a smooth dome move removed: + u² and x·u terms —
    exactly the family a per-frame rigid smooth dome move absorbs; what remains is the anchor dependence the final
    result keeps: non-quadratic per-frame errors, lateral-profile and placement differences), and dome_part_kappa
    (the removed u² term in 1/mm — compare with the anchors' own-dome dissent). Cells beyond X's own frames are not
    mapped. `spacing_R` (lateral, depth, frame mm) converts the dome part to 1/mm."""
    l0X, z0X, f0X = [int(v) for v in canvas_X["origin"]]; LcX, _DcX, FcX = [int(v) for v in canvas_X["shape"]]
    l0R, z0R, f0R = [int(v) for v in canvas_R["origin"]]; LcR, _DcR, FcR = [int(v) for v in canvas_R["shape"]]
    hs = max(1.0, (L_X - 1) / 2.0)
    T = {"df": int(T_XR["df"]), "dx": np.asarray(T_XR["dx"], float), "a": np.asarray(T_XR["a"], float), "b": np.asarray(T_XR["b"], float)}
    diffs = []; xs = []; us = []
    lat_R = np.arange(LcR, dtype=float)
    L_R = int(L_R) if L_R is not None else int(L_X)
    hsR = max(1.0, (L_R - 1) / 2.0); fcR0 = (FcR - 1) / 2.0
    for fcX in range(FcX):
        f = fcX + f0X
        if not (0 <= f < F_X) or not frame_ok(T, f):
            continue
        fcR = f + T["df"] - f0R
        if not (0 <= fcR < FcR):
            continue
        col = curve_X[:, fcX]
        okc = np.isfinite(col)
        if not okc.any():
            continue
        lcX = np.flatnonzero(okc)
        l = lcX + l0X
        zR = (col[okc] + z0X) + T["a"][f] + T["b"][f] * (l - (L_X - 1) / 2.0) / hs
        lcR = l + T["dx"][f] - l0R
        rowR = curve_R[:, fcR]
        ref = np.interp(lcR, lat_R, np.where(np.isfinite(rowR), rowR, np.nan), left=np.nan, right=np.nan)
        # both interpolation neighbours must be finite: np.interp propagates NaN from either neighbour
        d = (zR - z0R) - ref
        okd = np.isfinite(d)
        if okd.any():
            diffs.append(d[okd]); xs.append((lcR[okd] + l0R - (L_R - 1) / 2.0) / hsR); us.append(np.full(int(okd.sum()), fcR - fcR0))
    if not diffs:
        return {"spread_px": None, "spread_beyond_pose_px": None, "shape_spread_px": None, "dome_part_kappa": None, "median_abs_px": None,
                "p90_abs_px": None, "mean_px": None, "n_cells": 0, "pose": None}
    d = np.concatenate(diffs); x = np.concatenate(xs); u = np.concatenate(us)
    A1 = np.stack([np.ones(d.size), x, u], 1)
    p1, *_ = np.linalg.lstsq(A1, d, rcond=None)
    r1 = d - A1 @ p1
    A2 = np.stack([np.ones(d.size), x, u, u * u, x * u], 1)
    p2, *_ = np.linalg.lstsq(A2, d, rcond=None)
    r2 = d - A2 @ p2
    dome_k = (float(kappa_frames(p2[3], spacing_R)) if spacing_R is not None else None)
    return {"spread_px": float(np.sqrt(np.mean(d * d))), "spread_beyond_pose_px": float(np.sqrt(np.mean(r1 * r1))),
            "shape_spread_px": float(np.sqrt(np.mean(r2 * r2))), "dome_part_kappa": dome_k,
            "median_abs_px": float(np.median(np.abs(d))), "p90_abs_px": float(np.percentile(np.abs(d), 90)), "mean_px": float(np.mean(d)),
            "n_cells": int(d.size),
            "pose": {"shift_px": float(p2[0]), "tilt_px_half_span": float(p2[1]), "trend_px_per_frame": float(p2[2]),
                     "dome_px_per_frame2": float(p2[3]), "tilt_trend_px_per_frame": float(p2[4])}}


def roundtrip_error(T_XR: dict, T_RX: dict, L: int) -> dict:
    """The served pair X → R composed with the independently registered R → X (the sensitivity pair) should be the
    identity: per X frame the leftover dx / a / b (px) and the df mismatch — the pair engine's own inconsistency, which
    the spread inherits through the mapping."""
    I = compose_transforms(T_XR, T_RX, L)
    ok = np.isfinite(I["dx"]) & np.isfinite(I["a"]) & np.isfinite(I["b"])
    rms = lambda v: (float(np.sqrt(np.mean(v[ok] ** 2))) if ok.any() else None)  # noqa: E731
    return {"df_error": int(I["df"]), "dx_rms": rms(I["dx"]), "a_rms_px": rms(I["a"]), "b_rms_px": rms(I["b"]),
            "a_peak_px": (float(np.max(np.abs(I["a"][ok]))) if ok.any() else None), "n_frames": int(ok.sum())}


# ── rigid per-frame transform algebra (PairResult's convention: moving index + shift = reference index) ───────
def invert_transform(T: dict, L: int, F_ref: int) -> dict:
    """The inverse of X → R (indexed by X frames) as R → X (indexed by R frames): df' = −df; dx'(fR) = −dx(f);
    a'(fR) = −a(f) + b(f)·dx(f)/hs; b'(fR) = −b(f), f = fR − df. NaN where R frame f has no X partner."""
    hs = max(1.0, (L - 1) / 2.0)
    df = int(T["df"]); dx = np.asarray(T["dx"], float); a = np.asarray(T["a"], float); b = np.asarray(T["b"], float)
    F = dx.size
    out = {"df": -df, "dx": np.full(F_ref, np.nan), "a": np.full(F_ref, np.nan), "b": np.full(F_ref, np.nan)}
    for fR in range(F_ref):
        f = fR - df
        if 0 <= f < F and np.isfinite(dx[f]) and np.isfinite(a[f]) and np.isfinite(b[f]):
            out["dx"][fR] = -dx[f]; out["a"][fR] = -a[f] + b[f] * dx[f] / hs; out["b"][fR] = -b[f]
    return out


def compose_transforms(T1: dict, T2: dict, L: int) -> dict:
    """T2 ∘ T1: m → R (T1, indexed by m frames) then R → X (T2, indexed by R frames) → m → X indexed by m frames:
    df = df1 + df2; dx(f) = dx1(f) + dx2(fR); a(f) = a1(f) + a2(fR) + b2(fR)·dx1(f)/hs; b(f) = b1(f) + b2(fR)."""
    hs = max(1.0, (L - 1) / 2.0)
    df1 = int(T1["df"]); dx1 = np.asarray(T1["dx"], float); a1 = np.asarray(T1["a"], float); b1 = np.asarray(T1["b"], float)
    df2 = int(T2["df"]); dx2 = np.asarray(T2["dx"], float); a2 = np.asarray(T2["a"], float); b2 = np.asarray(T2["b"], float)
    F = dx1.size
    out = {"df": df1 + df2, "dx": np.full(F, np.nan), "a": np.full(F, np.nan), "b": np.full(F, np.nan)}
    for f in range(F):
        fR = f + df1
        if 0 <= fR < dx2.size and np.isfinite(dx1[f]) and np.isfinite(dx2[fR]) and np.isfinite(a1[f]) and np.isfinite(a2[fR]):
            out["dx"][f] = dx1[f] + dx2[fR]; out["a"][f] = a1[f] + a2[fR] + b2[fR] * dx1[f] / hs; out["b"][f] = b1[f] + b2[fR]
    return out


# ── the pair cache (engine md5 + params hash + ref + mov) ─────────────────────────────────────────────────────
PAIR_CACHE_FORMAT = "cornea-pair-v2"   # v2: arrays inside dict fields are type-tagged (exact round trip)


def params_hash(params_dict: dict) -> str:
    return hashlib.md5(json.dumps(params_dict, sort_keys=True, default=str).encode()).hexdigest()[:12]


class PairCache:
    """PairResults under <dir>/<mov>__to__<ref>.npz, keyed by the engine md5 and the PairParams hash: arrays as npz
    entries, everything else as one JSON string. `ctor` rebuilds the result (group_align.PairResult)."""

    def __init__(self, directory: Path, engine_md5: str, params_hash_: str, ctor=None):
        self.dir = Path(directory); self.engine_md5 = str(engine_md5); self.params_hash = str(params_hash_); self.ctor = ctor
        self.hits = 0; self.misses = 0

    def path(self, ref: str, mov: str) -> Path:
        return self.dir / f"{mov}__to__{ref}.npz"

    def get(self, ref: str, mov: str):
        p = self.path(ref, mov)
        if not p.exists() or self.ctor is None:
            self.misses += 1
            return None
        try:
            z = np.load(p, allow_pickle=False)
            if str(z["__format__"]) != PAIR_CACHE_FORMAT or str(z["__engine_md5__"]) != self.engine_md5 or str(z["__params_hash__"]) != self.params_hash:
                return None
            d = json.loads(str(z["__json__"]))
            kw = {}
            for k in z.files:
                if not k.startswith("__"):
                    kw[k] = np.asarray(z[k])
            for k, v in d.items():
                if k == "dx_segments":
                    kw[k] = [(int(s[0]), int(s[1]), float(s[2])) for s in v]
                elif k == "shape":
                    kw[k] = tuple(int(x) for x in v)
                else:
                    kw[k] = _decode(v)                 # arrays inside quality / ceiling / coarse come back as arrays
            res = self.ctor(**kw)
            self.hits += 1
            return res
        except Exception:  # noqa: BLE001 — an unreadable / stale cache entry is a miss
            self.misses += 1
            return None

    def put(self, res) -> Path:
        p = self.path(res.ref_cid, res.mov_cid)
        p.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}; payload = {}
        for k, v in res.__dict__.items():
            if isinstance(v, np.ndarray):
                arrays[k] = v
            else:
                payload[k] = _encode(v)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}.npz")
        np.savez_compressed(tmp, __format__=PAIR_CACHE_FORMAT, __engine_md5__=self.engine_md5, __params_hash__=self.params_hash,
                            __json__=json.dumps(payload, allow_nan=True), **arrays)
        os.replace(tmp, p)
        return p


def _encode(v):
    """JSON-able copy with arrays TYPE-TAGGED ({"__nd__": list, "dt": dtype}) so a dict field (quality / ceiling /
    coarse) round-trips exactly: lists stay lists, arrays come back as arrays (tuples become lists)."""
    if isinstance(v, np.ndarray):
        return {"__nd__": v.tolist(), "dt": str(v.dtype)}
    if isinstance(v, dict):
        return {str(k): _encode(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_encode(x) for x in v]
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, Path):
        return str(v)
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return str(v)


def _decode(v):
    if isinstance(v, dict):
        if "__nd__" in v and "dt" in v and len(v) == 2:
            return np.asarray(v["__nd__"], dtype=np.dtype(v["dt"]))
        return {k: _decode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_decode(x) for x in v]
    return v
