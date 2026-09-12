"""group_job — the MINIMAL "Align group" job (step 4 "Aligned", reviewer ask 2026-09-11).

For one patient+eye group id: load every member from the store's case dirs (group_align.load_member, READ-ONLY on the
cases: write_cache=False), run group_align.register_group (the pose rule picks the reference; every member vs the
reference; transitivity opt-in), and persist under <groups root>/<gid>/align_min/:

    progress.json   {phase, members_done, members_total, pairs_done, pairs_total, started, updated, pid, running, error}
    result.json     {group, reference, reference_rule, members: [{cid, is_reference, df, dx_median, dx_range, tilt_median_px,
                     tilt_abs_median_px, lateral_scale, rel_struct, rel_speckle, matched_struct, coverage, measured_frames,
                     overlap_frames, ok, flags, reject_flags, non_contributing, pose_angle_deg, seconds, overlay, summary}],
                     engine_md5, engine_file, timestamp, seconds, timings, transitivity, pose, lateral_grid, ceilings}
    overlay_<cid>.png   per non-reference member: the engine-independent montage LAYOUT the reviewer liked
                     (wf_bs/q3_truth/bf.py) computed from the ENGINE's transform — top row the reference (3 B-scans at
                     frames near the start / middle / end of the overlap + 3 sagittal cuts at laterals near the left /
                     centre / right), middle row the member moved onto the reference grid by the served rigid transform
                     (df, per-frame dx, per-frame a + b·x — PairResult's convention), bottom row red = reference /
                     green = moved with the engine's per-frame structure NCC in the titles.
    consensus.png / consensus.json / aligned_rgb_pairs.nii.gz   the CONSENSUS v2 montage (fused union canvas + every
                     member's placed served line + the consensus curve), its record (group_consensus: majority own
                     dome per lateral, the axial witness per band, the reference sensitivity, flags) and the
                     half-resolution RGB(A) volume of the placed members (the PAIR placement) for the panel's 3-D
                     view (see build_consensus / reference_sensitivity_stage; result.json['consensus'] summarises them).
    pairs/<mov>__to__<ref>.npz   the pair cache (engine md5 + PairParams hash): the rule's pairs and the reference-
                     sensitivity pairs (every other member as the anchor), reused by a re-run on the same engine.
    transforms.json / aligned_<cid>.nii.gz / aligned_lines.npz / aligned.png / aligned_rgb.nii.gz   the APPLIED
                     result (reviewer ask 2026-09-11 #3: "are there small axial changes to any or all of the scans
                     such that smoothness … and consistency … is maximised?"): per member (reference included) a
                     FINAL per-frame rigid transform = the pair engine's (df, dx, a, b) plus a SMOOTH per-frame shift δa
                     and tilt δb·x = the shift + tilt of (provisional consensus − the member's OWN smooth dome Q_m),
                     never of the line (final_transforms; the line's residual is reported as line_residual /
                     line_off_consensus, never applied); every ok member moved by it onto the union canvas at full
                     resolution (aligned_<cid>.nii.gz, uint16, canvas affine), its final line (aligned_lines.npz),
                     the post-transform fused montage (aligned.png, same layout as consensus.png) and the FINAL
                     half-res RGB(A) volume (aligned_rgb.nii.gz; the 3-D tab's default). result.json['transforms']
                     summarises the sizes (see apply_transforms).
    aligned_pairs_<cid>.nii.gz / scrub/{before,after}_<cid>.npy / scrub/meta.json   the SCRUB data (reviewer ask
                     2026-09-11 #4: "allow the user to scrub through the sagittal views (before and after) of the
                     replicates after axial changes are applied"): per member the BEFORE placement (the pair engine's
                     df / dx / a / b only) on the SAME union canvas as the applied result (full-res uint16, canvas
                     affine, for download) and, for both stages, an uncompressed uint8 .npy copy (C order (Lc, Dc, Fc):
                     one canvas lateral = one contiguous (Dc, Fc) block, read by memory-map without loading the
                     volume; windowed 0 → window[1] like the montages, covered cells ≥ 1) — Lc·Dc·Fc bytes each
                     (CS001_OS 560×738×110: 45 MB per member per stage, 273 MB for 3 members × 2 stages; uint16 would
                     be twice that). scrub/meta.json holds the per-lateral line RMS to the consensus before / after
                     per member (all frames where both are defined), the default lateral (the reference's central
                     lateral) and the covered lateral range; scrub/<stage>_<lateral>.png caches the sidecar's
                     composites (GET /api/group/{gid}/align/sagittal → render_scrub_png). aligned_lines.npz also
                     carries line_pairs_<cid> / a_pair_<cid> / b_pair_<cid>.
    scrub/own_<cid>.npy   ANY NUMBER OF MEMBERS (reviewer 2026-09-12: "the sagittal scrub only seems to show 3 scans when
                     there are more than 3 scans in the subgroup"): a REFUSED member is not placed on the canvas (no transform
                     to place it by), so the composite shows its OWN middle sagittal (the scan's volume at its own lateral
                     L/2, unmoved, uint8 (D, F) windowed by its own p99) in a clearly labelled "not aligned — <reason>"
                     strip; scrub/meta.json lists EVERY subgroup member under 'roster' with a role ('reference' |
                     'contributing' | 'refused: <flags> (offset ≈ dx laterals, overlap f %)'), 'refused' (the strip's
                     records), 'n_members' / 'n_contributing'; 'members' stays the PLACED members (the memmaps). The
                     composite puts EVERY member of a stage on ONE row, the consensus blend first (the app scrolls horizontally); refused members follow in
                     the last row, so it stays ≤ 1800 px wide however many scans the subgroup holds (scrub_layout).
                     result.json['roster'] carries the same roster for the cards / 3-D legend.
    job.log         the run's log.

Run as a SUBPROCESS by api_server (POST /api/group/{gid}/align) so the engine's memory is released on exit and its
BLAS threads (3) never fight the sidecar's:  python3 group_job.py --group cs001_os --cases-root <cases> --out-dir <dir>
Nothing is written to any case (no manifest, no border_cache): manifest.group_aligned stays the consensus step's job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "3")

import numpy as np                      # noqa: E402
import scipy.ndimage as ndi             # noqa: E402

import group_consensus as gc_           # noqa: E402  (consensus v2: majority own dome, axial witness, spread, pair cache)

ENGINE_FILE = Path(__file__).resolve().parent / "group_align.py"
RESULT_NAME = "result.json"
PROGRESS_NAME = "progress.json"
LOG_NAME = "job.log"


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────────────────
def engine_md5() -> str:
    try:
        return hashlib.md5(ENGINE_FILE.read_bytes()).hexdigest()
    except OSError:
        return ""


def jsonable(v):
    """JSON-safe copy: arrays → lists, numpy scalars → python, non-finite floats → None, Paths → str."""
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return jsonable(v.tolist())
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return jsonable(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, Path):
        return str(v)
    return v


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def overlay_name(cid: str) -> str:
    return f"overlay_{cid}.png"


class Progress:
    def __init__(self, out_dir: Path, group: str, members: list[str]):
        self.path = out_dir / PROGRESS_NAME
        self.state = {"group": group, "members": list(members), "phase": "starting", "members_done": 0,
                      "members_total": len(members), "pairs_done": 0, "pairs_total": max(0, len(members) - 1),
                      "overlays_done": 0, "started": time.time(), "updated": time.time(), "pid": os.getpid(),
                      "running": True, "done": False, "error": None}
        self.flush()

    def update(self, **kw) -> None:
        self.state.update(kw); self.state["updated"] = time.time(); self.flush()

    def flush(self) -> None:
        try:
            _write_json(self.path, self.state)
        except OSError:
            pass


# ── the moved volume on the reference grid (the ENGINE's transform, PairResult's convention) ──────────────────
def moved_on_reference(ref, mov, r) -> tuple[np.ndarray, np.ndarray]:
    """The moving member's volume resampled onto the REFERENCE grid (L, D_ref, F_ref) by the served rigid transform:
    reference voxel (l_r, z_r, f_r) ← moving (l_r − dx_applied[f], z_r − a[f] − b[f]·x(l_m), f = f_r − df),
    x(l_m) = (l_m − (L−1)/2) / ((L−1)/2) — exactly group_align.warp_band's per-frame rule (a is the TOTAL axial move in
    corrected rows), linear interpolation, zero outside. `ref` / `mov` are the members on the group's common lateral
    grid. Returns (moved, mask) with mask True where the moved sample lies inside the moving canvas."""
    L, Dr, Fr = ref.volume.shape
    Lm, Dm, Fm = mov.volume.shape
    assert Lm == L, (Lm, L)
    out = np.zeros((L, Dr, Fr), np.float32)
    msk = np.zeros((L, Dr, Fr), bool)
    lat = np.arange(L, dtype=float)
    hs = max(1.0, (L - 1) / 2.0)
    zz = np.arange(Dr, dtype=float)
    a = np.asarray(r.a, float); b = np.asarray(r.b, float); dxa = np.asarray(r.dx_applied, float)
    for fr in range(Fr):
        f = fr - int(r.df)
        if not (0 <= f < Fm):
            continue
        if not (np.isfinite(dxa[f]) and np.isfinite(a[f]) and np.isfinite(b[f])):
            continue
        lm = lat - float(dxa[f])
        dzl = float(a[f]) + float(b[f]) * (lm - (L - 1) / 2.0) / hs
        ZZ = zz[None, :] - dzl[:, None]
        LL = np.broadcast_to(lm[:, None], (L, Dr))
        coords = [LL, ZZ]
        out[:, :, fr] = ndi.map_coordinates(mov.volume[:, :, f], coords, order=1, cval=0.0, mode="grid-constant")
        inside = (LL >= 0) & (LL <= Lm - 1) & (ZZ >= 0) & (ZZ <= Dm - 1)
        msk[:, :, fr] = inside
    return out, msk


def _served_range(ref, mov, r) -> tuple[int, int]:
    """Depth window [zlo, zhi) around the corneas: the reference's served line and the moving line carried."""
    Dr = ref.volume.shape[1]
    vals = [np.asarray(ref.served, float)[np.asarray(ref.valid, bool)]]
    try:
        L, Fm = mov.served.shape
        Fr = ref.volume.shape[2]
        lat = np.arange(L, dtype=float); hs = max(1.0, (L - 1) / 2.0)
        for fr in range(Fr):
            f = fr - int(r.df)
            if 0 <= f < Fm and np.isfinite(r.dx_applied[f]):
                lm = lat - float(r.dx_applied[f])
                sm = np.interp(lm, lat, np.where(np.isfinite(mov.served[:, f]), mov.served[:, f], np.nan), left=np.nan, right=np.nan)
                dzl = float(r.a[f]) + float(r.b[f]) * (lm - (L - 1) / 2.0) / hs
                carried = sm + dzl
                vals.append(carried[np.isfinite(carried)])
    except Exception:  # noqa: BLE001
        pass
    v = np.concatenate([np.asarray(x, float).ravel() for x in vals if np.asarray(x).size]) if vals else np.zeros(0)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0, Dr
    zlo = int(max(0, np.percentile(v, 1) - 60))
    zhi = int(min(Dr, np.percentile(v, 99) + 360))
    if zhi - zlo < 100:
        zlo, zhi = 0, Dr
    return zlo, zhi


def _short(cid: str) -> str:
    return cid[5:] if cid.startswith("case_") else cid


def render_overlay(ref, mov, r, path: Path, title: str, thr_frac: float = 0.35) -> dict:
    """The montage (see module docstring). Returns the frames / laterals shown and the per-panel NCCs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    P, PM = moved_on_reference(ref, mov, r)
    V = np.asarray(ref.volume, np.float32)
    L, Dr, Fr = V.shape
    zlo, zhi = _served_range(ref, mov, r)
    pos = V[V > 0]
    vmax = float(np.percentile(pos, 99.5)) if pos.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    thr = thr_frac * vmax
    tis_r = V[:, zlo:zhi, :] > thr
    tis_m = (P[:, zlo:zhi, :] > thr) & PM[:, zlo:zhi, :]
    both = tis_r & tis_m
    fr_cnt = both.sum(axis=(0, 1)); lat_cnt = both.sum(axis=(1, 2))
    part = np.array([0 <= fr - int(r.df) < mov.volume.shape[2] for fr in range(Fr)])
    fr_ok = np.flatnonzero((fr_cnt >= max(50, 0.02 * fr_cnt.max() if fr_cnt.max() > 0 else 50)) & part)
    if fr_ok.size == 0:
        fr_ok = np.flatnonzero(part) if part.any() else np.arange(Fr)
    lat_ok = np.flatnonzero(lat_cnt >= max(50, 0.02 * lat_cnt.max() if lat_cnt.max() > 0 else 50))
    if lat_ok.size == 0:
        lat_ok = np.arange(L)
    fsel = [int(fr_ok[int(round(q * (fr_ok.size - 1)))]) for q in (0.1, 0.5, 0.9)]
    lsel = [int(lat_ok[int(round(q * (lat_ok.size - 1)))]) for q in (0.15, 0.5, 0.85)]
    Fm = mov.volume.shape[2]
    pf_ncc = np.asarray(r.per_frame_ncc, float); pf_loc = np.asarray(r.per_frame_local_ncc, float)
    meas = np.asarray(r.measured, bool); live = np.asarray(r.live, bool)

    def status(f: int) -> str:
        return "measured" if meas[f] else ("interp" if live[f] else "dead/interp")

    # PHYSICAL aspect (reviewer 2026-09-12): columns 0-2 are B-scans (x = laterals, y = depth) → dz/dl; columns 3-5
    # are sagittal cuts (x = frames, y = depth) → dz/df. A pixel is then square in MILLIMETRES, not stretched.
    _sp = [float(v) for v in (getattr(ref, "spacing", None) or (1.0, 1.0, 1.0))]
    _dl, _dz, _df = (_sp + [1.0, 1.0, 1.0])[:3]
    asp_ax = (_dz / _dl) if (_dl > 0 and _dz > 0) else 1.0
    asp_sg = (_dz / _df) if (_df > 0 and _dz > 0) else 1.0
    fig, ax = plt.subplots(3, 6, figsize=(30, 13))
    ncc_shown = {}
    for k, fr in enumerate(fsel):
        f = fr - int(r.df)
        a = V[:, zlo:zhi, fr].T; bb = P[:, zlo:zhi, fr].T
        # the app's axial preview mirrors the laterals (display column = 512 − raw lateral): flip so left/right match
        a = a[:, ::-1]; bb = bb[:, ::-1]
        ax[0, k].imshow(a, cmap="gray", vmin=0, vmax=vmax, aspect=asp_ax)
        ax[0, k].set_title(f"REF {_short(ref.cid)}  frame {fr}  (app axial slice {Fr - 1 - fr})", fontsize=9)
        ax[1, k].imshow(bb, cmap="gray", vmin=0, vmax=vmax, aspect=asp_ax)
        if 0 <= f < Fm:
            ax[1, k].set_title(f"MOVED {_short(mov.cid)}  f {f}→{fr}  dx {r.dx_applied[f]:+.1f}  a {r.a[f]:+.1f}  b {r.b[f]:+.1f}  [{status(f)}]", fontsize=8)
            n1 = pf_ncc[f] if f < pf_ncc.size else float("nan"); n2 = pf_loc[f] if f < pf_loc.size else float("nan")
            ncc_shown[str(fr)] = {"mov_frame": int(f), "ncc_struct": (None if not np.isfinite(n1) else float(n1)),
                                  "local_ncc_struct": (None if not np.isfinite(n2) else float(n2)), "status": status(f)}
            ttl = f"overlay R=ref G=moved   frame NCC (struct) {n1:.2f}   local {n2:.2f}"
        else:
            ax[1, k].set_title(f"MOVED {_short(mov.cid)}  (no partner frame)", fontsize=9)
            ttl = "overlay R=ref G=moved   (no partner frame)"
        rgb = np.zeros(a.shape + (3,), np.float32)
        rgb[..., 0] = np.clip(a / vmax, 0, 1); rgb[..., 1] = np.clip(bb / vmax, 0, 1)
        ax[2, k].imshow(rgb, aspect=asp_ax); ax[2, k].set_title(ttl, fontsize=9)
        for rr in range(3):
            ax[rr, k].set_yticks(np.arange(0, zhi - zlo, 100)); ax[rr, k].set_yticklabels(np.arange(zlo, zhi, 100))
            ax[rr, k].set_xlabel("lateral (app orientation) →")
    for k, l in enumerate(lsel):
        a = V[l, zlo:zhi, :]; bb = P[l, zlo:zhi, :]
        # the app's sagittal preview shows HIGH frame indices on the LEFT (scaleX(-1)): flip the frame axis
        a = a[:, ::-1]; bb = bb[:, ::-1]
        ax[0, 3 + k].imshow(a, cmap="gray", vmin=0, vmax=vmax, aspect=asp_sg)
        ax[0, 3 + k].set_title(f"REF {_short(ref.cid)}  lateral {l}  (app sagittal slice ≈ {L - 1 - l})", fontsize=9)
        ax[1, 3 + k].imshow(bb, cmap="gray", vmin=0, vmax=vmax, aspect=asp_sg)
        ax[1, 3 + k].set_title(f"MOVED {_short(mov.cid)}  lateral {l}", fontsize=9)
        rgb = np.zeros(a.shape + (3,), np.float32)
        rgb[..., 0] = np.clip(a / vmax, 0, 1); rgb[..., 1] = np.clip(bb / vmax, 0, 1)
        ax[2, 3 + k].imshow(rgb, aspect=asp_sg); ax[2, 3 + k].set_title("overlay R=ref G=moved (x = frame)", fontsize=9)
        for rr in range(3):
            ax[rr, 3 + k].set_yticks(np.arange(0, zhi - zlo, 100)); ax[rr, 3 + k].set_yticklabels(np.arange(zlo, zhi, 100))
            ax[rr, 3 + k].set_xticks(np.arange(0, Fr, 20)); ax[rr, 3 + k].set_xticklabels((Fr - 1 - np.arange(0, Fr, 20)))
            ax[rr, 3 + k].set_xlabel("← frame (high on the left, as in the app)")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=60)
    plt.close(fig)
    return {"frames": fsel, "laterals": lsel, "depth_window": [zlo, zhi], "per_frame": ncc_shown}


# ── UNION canvas + PROVISIONAL consensus + RGB volume (reviewer ask 2026-09-11 #2: "I do not see a consensus
#    image" / "the overlaps are not seen in 3D") ─────────────────────────────────────────────────────────────────
# The reference and every ok member are placed on ONE union canvas by the engine's rigid transform (df, per-frame dx,
# a, b·x on the group's common lateral grid; the reference grid extended by the members' placements, no extrapolation
# — a cell is covered only where ≥ 1 member has data), the fused image is the masked mean of the placed members, and
# the consensus CURVE is consensus v2 (group_consensus.consensus_v2): the DOME (c2 per lateral) is the MAJORITY of the
# members' OWN along-frame domes (the reference is only the coordinate anchor — reviewer 2026-09-11: "technically no
# single replicate is a reference"), checked per lateral band against the within-B-scan (axial) curvature in physical
# units, the PLACEMENT (c1, c0) from the tissue-placed lines (c0 only lightly smoothed: divots / limbus are real).
# The PROVISIONAL curve (consensus_curve: every coefficient from the placed lines, which inherit the anchor's dome) is
# kept for the members' own domes (member_dome) and as a diagnostic.
CONSENSUS_PNG = "consensus.png"
CONSENSUS_JSON = "consensus.json"
VOLUME_NAME = "aligned_rgb.nii.gz"            # the FINAL placement (post-transform) once apply_transforms ran; until
                                              # then a copy of the pair placement so the 3-D tab always has a volume
VOLUME_PAIRS_NAME = "aligned_rgb_pairs.nii.gz"  # the pair engine's placement (pre-transform)
ALIGNED_PNG = "aligned.png"
ALIGNED_LINES = "aligned_lines.npz"
TRANSFORMS_JSON = "transforms.json"
PROFILE_BEYOND_TILT_PX = 2.0                  # a frame whose dome difference to the consensus is not a shift + tilt
LINE_OFF_CONSENSUS_PX = 2.0                   # a frame whose LINE (after the smooth move) is off the consensus by more (reported, never applied)
AFTER_RMS_BAR_PX = 1.0                        # the bar on the DOME PART of the residual after the move (see final_transforms)
SMOOTH_D2_A_PX = 0.1                          # bar: max |second difference along frames| of the APPLIED δa (px)
SMOOTH_D2_B_PX = 0.2                          # … and of the applied δb (px at the half-span)
FIXED_INLIER_FRAC = 0.8                       # a lateral is in the FIXED inlier set when inlier on ≥ this share of its fitted frames
DELTA_SG_WIN = 15                             # Savitzky-Golay window (order 2) that smooths δa(f) / δb(f) over the covered frames
# TISSUE GATE on the apply (2026-09-12, refutation R4): a scan whose mean tissue disagreement with the OTHER placed scans (in their
# overlaps, measured from the scrub images by the tissue-edge rule) would RISE by more than TISSUE_GATE_SCAN_PX after its dome move
# has that move HELD (a_pair / b_pair served, δa = δb = 0, 'dome_move_held_by_tissue'); the group mean after must stay within
# TISSUE_GATE_GROUP_PX of before (reported: transforms.json['tissue_gate'])
TISSUE_GATE_SCAN_PX = 0.3
TISSUE_GATE_GROUP_PX = 0.1
# TRANSITIVE PLACEMENT (2026-09-12): a refused member placed through a contributing member needs the pair engine's own round trip
# (X → C composed with the independently registered C → X) to close within these bounds
TRANSITIVE_ROUNDTRIP_DX = 3.0                 # laterals (RMS over the X frames)
TRANSITIVE_ROUNDTRIP_PX = 3.0                 # px (a, RMS)


def aligned_volume_name(cid: str) -> str:
    return f"aligned_{cid}.nii.gz"


def aligned_pairs_volume_name(cid: str) -> str:
    return f"aligned_pairs_{cid}.nii.gz"


SCRUB_DIR = "scrub"                           # under align_min/: the memmap copies, meta.json and the PNG cache
SCRUB_META = "meta.json"
SCRUB_STAGES = ("before", "after")            # before = the pair engine's placement, after = axial changes applied


def scrub_volume_name(stage: str, cid: str) -> str:
    return f"{stage}_{cid}.npy"


def scrub_png_name(stage: str, lateral: int) -> str:
    return f"{stage}_{int(lateral)}.png"
CONSENSUS_NOTE = ("consensus v2 (majority own dome, axial-checked): the dome per lateral is the majority of the scans' OWN "
                  "along-frame domes (the reference is only the coordinate anchor), checked per lateral band against the "
                  "within-B-scan curvature in physical units; c1 / c0 from the tissue-placed lines (c0 lightly smoothed)")
CONSENSUS_NOTE_PROVISIONAL = ("provisional: per-lateral quadratic along frames through the tissue-placed lines, coefficients "
                              "smoothed across laterals (inherits the anchor's dome; diagnostic only)")
PAIRS_DIR = "pairs"                            # under align_min/: the pair cache (group_consensus.PairCache)
# ONE distinct colour per replicate — used for the lines in consensus.png / aligned.png, the additive blend in the
# scrub "all" panel and the 3-D volume (reviewer 2026-09-12: "as many unique colours as there are replicates", not
# R / G / B cycled). The first colours are the familiar red / green / blue so a 3-scan group looks as before; beyond
# the fixed list, evenly spaced hues are generated (bright, saturated, additive-friendly).
MEMBER_COLOURS = ["#ff5050", "#40e040", "#5a9bff", "#ffc83c", "#e070ff", "#40e0e0", "#ff9a40", "#a0a0ff",
                  "#b4ff40", "#ff64b4", "#64ffb4", "#ffffff"]


def member_palette(n: int) -> list:
    """`n` distinct hex colours, the fixed list first, then evenly spaced hues (HSV, S 0.75, V 1.0)."""
    import colorsys
    out = list(MEMBER_COLOURS[:max(0, n)])
    k = len(out)
    while len(out) < n:
        i = len(out) - k
        h = ((i + 0.5) / max(1, n - k) + 0.08) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
        out.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return out


CHANNEL_NAMES = ["R", "G", "B"]        # legacy: older results named a member's 3-D channel R / G / B


def identity_transform(n_frames: int) -> dict:
    z = np.zeros(int(n_frames), float)
    return {"df": 0, "dx": z.copy(), "a": z.copy(), "b": z.copy()}


def pair_transform(r) -> dict:
    """The engine's served rigid transform of a PairResult (moving index + shift = reference index)."""
    return {"df": int(r.df), "dx": np.asarray(r.dx_applied, float), "a": np.asarray(r.a, float), "b": np.asarray(r.b, float)}


def _frame_ok(T: dict, f: int) -> bool:
    return 0 <= f < T["dx"].size and bool(np.isfinite(T["dx"][f]) and np.isfinite(T["a"][f]) and np.isfinite(T["b"][f]))


def union_canvas(members: list, transforms: dict) -> dict:
    """The union canvas in REFERENCE coordinates: origin (l0, z0, f0) ≤ 0 and shape (Lc, Dc, Fc) such that every
    placed member (its whole volume moved by its transform) fits; canvas index = reference index − origin."""
    lo = np.array([0.0, 0.0, 0.0]); hi = np.array([0.0, 0.0, 0.0])
    for m in members:
        T = transforms[m.cid]
        L, D, F = m.volume.shape
        fs = [f for f in range(F) if _frame_ok(T, f)]
        if not fs:
            continue
        fs = np.asarray(fs)
        dx = T["dx"][fs]; a = T["a"][fs]; b = np.abs(T["b"][fs])
        lo = np.minimum(lo, [dx.min(), (a - b).min(), fs.min() + T["df"]])
        hi = np.maximum(hi, [L - 1 + dx.max(), D - 1 + (a + b).max(), fs.max() + T["df"]])
    origin = [int(np.floor(v)) for v in lo]
    upper = [int(np.ceil(v)) + 1 for v in hi]
    shape = [int(u - o) for u, o in zip(upper, origin)]
    return {"origin": origin, "shape": shape}


def place_on_canvas(m, T: dict, canvas: dict, *, volume: bool = True) -> tuple:
    """Place one member on the union canvas by its rigid transform: canvas (lc, zc, fc) ← moving
    (lc + l0 − dx[f], zc + z0 − a[f] − b[f]·x(l_m), f = fc + f0 − df), x(l_m) = (l_m − (L−1)/2) / ((L−1)/2) — the
    same rule as moved_on_reference (group_align.warp_band's convention), linear interpolation, zero outside.
    Returns (vol float32 or None, mask bool (Lc, Dc, Fc) or None, line (Lc, Fc) = the member's served line placed on
    the canvas (NaN where the member has no valid line), colmask (Lc, Fc) = the member has data in that column)."""
    l0, z0, f0 = canvas["origin"]; Lc, Dc, Fc = canvas["shape"]
    L, D, F = m.volume.shape
    hs = max(1.0, (L - 1) / 2.0)
    lat_c = np.arange(Lc, dtype=float) + l0
    zz = np.arange(Dc, dtype=float) + z0
    sv = np.where(np.asarray(m.valid, bool), np.asarray(m.served, float), np.nan)
    out = np.zeros((Lc, Dc, Fc), np.float32) if volume else None
    msk = np.zeros((Lc, Dc, Fc), bool) if volume else None
    line = np.full((Lc, Fc), np.nan)
    col = np.zeros((Lc, Fc), bool)
    for fc in range(Fc):
        f = fc + f0 - int(T["df"])
        if not _frame_ok(T, f):
            continue
        lm = lat_c - float(T["dx"][f])
        dzl = float(T["a"][f]) + float(T["b"][f]) * (lm - (L - 1) / 2.0) / hs
        inside_l = (lm >= 0) & (lm <= L - 1)
        col[:, fc] = inside_l
        # the served line carried: both neighbours of the interpolation must be valid
        i0 = np.clip(np.floor(lm).astype(int), 0, L - 1); i1 = np.clip(i0 + 1, 0, L - 1); w = np.clip(lm - i0, 0, 1)
        s = (1 - w) * sv[i0, f] + w * sv[i1, f]
        s[~inside_l] = np.nan
        line[:, fc] = s + dzl - z0
        if volume:
            ZZ = zz[None, :] - dzl[:, None]
            LL = np.broadcast_to(lm[:, None], (Lc, Dc))
            out[:, :, fc] = ndi.map_coordinates(m.volume[:, :, f], [LL, ZZ], order=1, cval=0.0, mode="grid-constant")
            msk[:, :, fc] = (LL >= 0) & (LL <= L - 1) & (ZZ >= 0) & (ZZ <= D - 1)
    return out, msk, line, col


def _half_res(vol: np.ndarray, msk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """2×2 block mean over (lateral, depth) of the MASKED volume, full frames → (mean, covered)."""
    Lc, Dc, Fc = vol.shape
    Lh, Dh = (Lc + 1) // 2, (Dc + 1) // 2
    v = np.zeros((Lh * 2, Dh * 2, Fc), np.float32); v[:Lc, :Dc] = np.where(msk, vol, 0.0)
    k = np.zeros((Lh * 2, Dh * 2, Fc), np.float32); k[:Lc, :Dc] = msk
    v = v.reshape(Lh, 2, Dh, 2, Fc).sum(axis=(1, 3)); k = k.reshape(Lh, 2, Dh, 2, Fc).sum(axis=(1, 3))
    return np.where(k > 0, v / np.maximum(k, 1), 0.0).astype(np.float32), k > 0


def robust_quadratic(f: np.ndarray, z: np.ndarray, fc: float, scale: float, n_iter: int = 6):
    """z ≈ c0 + c1·u + c2·u², u = (f − fc)/scale, least squares with 3×MAD reweighting (a point beyond 3 × 1.4826 ×
    MAD of the residuals is dropped, MAD floored at 0.5 px). None when fewer than 6 points survive."""
    u = (np.asarray(f, float) - fc) / scale
    A = np.stack([np.ones_like(u), u, u * u], axis=1)
    w = np.ones(u.size, bool)
    c = None
    for _ in range(n_iter):
        if w.sum() < 6:
            return None
        c, *_ = np.linalg.lstsq(A[w], z[w], rcond=None)
        res = z - A @ c
        mad = float(np.median(np.abs(res[w] - np.median(res[w]))))
        thr = 3.0 * 1.4826 * max(mad, 0.5)
        w_new = np.abs(res) <= thr
        if np.array_equal(w_new, w):
            break
        w = w_new
    return c


def consensus_curve(lines: dict, colmask: np.ndarray, *, min_frames: int = 30, med: int = 31, sg: int = 151,
                    med_c0: int | None = None, sg_c0: int | None = None) -> dict:
    """PROVISIONAL consensus (now a diagnostic and the members' OWN domes, member_dome): per canvas lateral a robust
    quadratic along frames through the placed served lines of every member (≥ min_frames distinct covered frames
    with a line point), the coefficients smoothed across laterals (running median `med` + Savitzky-Golay `sg` /
    order 2 over the fitted laterals, gaps interpolated; c0 with `med_c0` / `sg_c0` when given — the LIGHT smoothing
    consensus v2 uses, so a member's own dome and the consensus share their lateral profile), evaluated only on
    covered cells. Returns {curve (Lc, Fc) NaN elsewhere, coef (Lc, 3) NaN where no fit, fc, scale, fitted (Lc,)
    bool, n_pts (Lc,)}."""
    from scipy.signal import savgol_filter
    cids = list(lines)
    Lc, Fc = colmask.shape
    fr = np.arange(Fc, dtype=float)
    fc = (Fc - 1) / 2.0; scale = max(1.0, Fc / 2.0)
    coef = np.full((Lc, 3), np.nan); n_pts = np.zeros(Lc, int); fitted = np.zeros(Lc, bool)
    for l in range(Lc):
        fs = []; zs = []
        for cid in cids:
            row = lines[cid][l]
            ok = np.isfinite(row)
            if ok.any():
                fs.append(fr[ok]); zs.append(row[ok])
        if not fs:
            continue
        f_all = np.concatenate(fs); z_all = np.concatenate(zs)
        n_pts[l] = int(f_all.size)
        if np.unique(f_all).size < min_frames:
            continue
        c = robust_quadratic(f_all, z_all, fc, scale)
        if c is not None and np.all(np.isfinite(c)):
            coef[l] = c; fitted[l] = True
    curve = np.full((Lc, Fc), np.nan)
    out = {"curve": curve, "coef": coef, "coef_raw": coef.copy(), "fc": fc, "scale": scale, "fitted": fitted, "n_pts": n_pts,
           "lateral_range": None}
    if fitted.sum() < 8:
        return out
    idx = np.flatnonzero(fitted); l_lo, l_hi = int(idx[0]), int(idx[-1])
    span = np.arange(l_lo, l_hi + 1)
    sm = np.full((Lc, 3), np.nan)
    n = span.size
    for k in range(3):
        v = np.interp(span, idx, coef[idx, k])                                # fill gaps between fitted laterals
        mk = med if (k > 0 or med_c0 is None) else med_c0
        sk = sg if (k > 0 or sg_c0 is None) else sg_c0
        wk = min(sk, n if n % 2 == 1 else n - 1)
        v = ndi.median_filter(v, size=max(1, min(mk, n)), mode="nearest")
        if wk >= 5:
            v = savgol_filter(v, wk, 2, mode="interp")
        sm[span, k] = v
    u = (fr - fc) / scale
    for l in span:
        c = sm[l]
        row = c[0] + c[1] * u + c[2] * u * u
        row[~colmask[l]] = np.nan
        curve[l] = row
    out.update(coef=sm, lateral_range=[l_lo, l_hi])
    return out


def _win_uint8(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((np.asarray(v, np.float32) - lo) / max(hi - lo, 1e-6) * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render_consensus(fused: np.ndarray, count: np.ndarray, lines: dict, curve: np.ndarray, colmask: np.ndarray,
                     canvas: dict, colours: dict, vmax: float, path: Path, title: str, curve_label: str = "consensus v2",
                     spacing_mm=None) -> dict:
    """consensus.png: 2 rows × 3 columns — three B-scans (frames near the start / middle / end of the covered range)
    and three sagittal cuts (laterals near the left / centre / right), the fused image with every member's placed
    served line (thin dashed, one colour per member) and the PROVISIONAL consensus curve (thick white). Same
    display orientation as the pair overlays (axial laterals mirrored; sagittal high frames on the left)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Lc, Dc, Fc = fused.shape
    l0, z0, f0 = canvas["origin"]
    fr_cov = np.flatnonzero(colmask.any(axis=0)); lat_cov = np.flatnonzero(colmask.any(axis=1))
    if fr_cov.size == 0:
        fr_cov = np.arange(Fc)
    if lat_cov.size == 0:
        lat_cov = np.arange(Lc)
    fsel = [int(fr_cov[int(round(q * (fr_cov.size - 1)))]) for q in (0.1, 0.5, 0.9)]
    lsel = [int(lat_cov[int(round(q * (lat_cov.size - 1)))]) for q in (0.15, 0.5, 0.85)]
    allz = np.concatenate([v[np.isfinite(v)] for v in lines.values()] + [curve[np.isfinite(curve)]])
    if allz.size:
        zlo = int(max(0, np.percentile(allz, 1) - 60)); zhi = int(min(Dc, np.percentile(allz, 99) + 360))
        if zhi - zlo < 100:
            zlo, zhi = 0, Dc
    else:
        zlo, zhi = 0, Dc
    # PHYSICAL aspect (reviewer 2026-09-12: "the scans must be represented dimensionally accurately in all the views"):
    # one depth row is dz mm, one lateral dl mm, one frame df mm — imshow(aspect=dz/dl) / (dz/df) makes a pixel square
    # in MILLIMETRES instead of stretching the slice to the axes box.
    sp = [float(v) for v in (spacing_mm if spacing_mm is not None else (1.0, 1.0, 1.0))]
    dl, dz, df_ = (sp + [1.0, 1.0, 1.0])[:3]
    asp_axial = (dz / dl) if (dl > 0 and dz > 0) else 1.0            # B-scan panel: x = laterals, y = depth
    asp_sag = (dz / df_) if (df_ > 0 and dz > 0) else 1.0            # sagittal panel: x = frames, y = depth
    fig, ax = plt.subplots(2, 3, figsize=(30, 12))
    for k, fc_ in enumerate(fsel):
        img = fused[:, zlo:zhi, fc_].T[:, ::-1]                       # (depth, lateral) mirrored like the app
        ax[0, k].imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect=asp_axial)
        xd = Lc - 1 - np.arange(Lc)
        for cid, ln in lines.items():
            ax[0, k].plot(xd, ln[:, fc_] - zlo, color=colours[cid], lw=1.4, ls="--", label=_short(cid))
        ax[0, k].plot(xd, curve[:, fc_] - zlo, color="white", lw=3.0, label=curve_label)
        nmem = int(count[:, zlo:zhi, fc_].max()) if count.size else 0
        ax[0, k].set_title(f"FUSED  canvas frame {fc_} (ref frame {fc_ + f0}; app axial slice ≈ {Fc - 1 - fc_})  members ≤ {nmem}", fontsize=9)
        ax[0, k].set_xlim(0, Lc - 1); ax[0, k].set_ylim(zhi - zlo, 0)
        ax[0, k].set_yticks(np.arange(0, zhi - zlo, 100)); ax[0, k].set_yticklabels(np.arange(zlo, zhi, 100) + z0)
        ax[0, k].set_xlabel("lateral (app orientation) →")
    for k, l in enumerate(lsel):
        img = fused[l, zlo:zhi, :][:, ::-1]                             # (depth, frame) high frames on the left
        ax[1, k].imshow(img, cmap="gray", vmin=0, vmax=vmax, aspect=asp_sag)
        xd = Fc - 1 - np.arange(Fc)
        for cid, ln in lines.items():
            ax[1, k].plot(xd, ln[l, :] - zlo, color=colours[cid], lw=1.4, ls="--", label=_short(cid))
        ax[1, k].plot(xd, curve[l, :] - zlo, color="white", lw=3.0, label=curve_label)
        ax[1, k].set_title(f"FUSED  canvas lateral {l} (ref lateral {l + l0})", fontsize=9)
        ax[1, k].set_xlim(0, Fc - 1); ax[1, k].set_ylim(zhi - zlo, 0)
        ax[1, k].set_yticks(np.arange(0, zhi - zlo, 100)); ax[1, k].set_yticklabels(np.arange(zlo, zhi, 100) + z0)
        ax[1, k].set_xlabel("← canvas frame (high on the left, as in the app)")
    ax[0, 0].legend(loc="lower left", fontsize=8, framealpha=0.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=60)
    plt.close(fig)
    return {"frames": fsel, "laterals": lsel, "depth_window": [zlo, zhi]}


def write_rgb_volume(chans: list, spacing_mm, path: Path, colours: dict | None = None) -> dict:
    """aligned_rgb.nii.gz: the placed members at half resolution laterally + in depth (full frames) as ONE colour
    volume — each member painted in ITS OWN colour (member_palette; reviewer 2026-09-12) and added, clipped at 255,
    uncovered = 0. Written as
    RGBA32 (niivue renders DT_RGB24 with alpha = 0.21R + 0.72G + 0.07B, which would make a blue member nearly
    invisible in 3-D; RGBA32 with A = max(R, G, B) shows every channel equally and is the same texture path the
    app's debug consensus already renders). Affine: lateral 2×, depth 2×, frame 1× the physical spacing (mm),
    oriented like the app's main viewer (lat → X, frames → Y, depth → Z)."""
    import nibabel as nib
    Lh, Dh, Fc = chans[0][1].shape
    pal = member_palette(len(chans))
    acc = np.zeros((Lh, Dh, Fc, 3), np.float32)
    gain = min(1.0, 3.0 / max(1, len(chans)))          # many members: the plain sum would clip to white
    used: dict = {}
    for i, (cid, u8) in enumerate(chans):
        hexc = str((colours or {}).get(cid) or pal[i])
        used[cid] = hexc
        rgb_i = np.asarray(_hex_rgb(hexc), np.float32) / 255.0
        f = u8.astype(np.float32) * gain
        for c in range(3):
            if rgb_i[c] > 0:
                acc[..., c] += f * rgb_i[c]
    rgb = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
    dt = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1"), ("A", "u1")])
    out = np.zeros((Lh, Dh, Fc), dtype=dt)
    out["R"] = rgb[..., 0]; out["G"] = rgb[..., 1]; out["B"] = rgb[..., 2]; out["A"] = rgb.max(axis=-1)
    sp = np.asarray(spacing_mm, float)
    sl, sd, sf = 2.0 * float(sp[0]), 2.0 * float(sp[1]), float(sp[2])
    aff = np.array([[-sl, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -sf, 0.0],
                    [0.0, -sd, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]], float)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(out), aff), str(path))
    return {"shape": [int(Lh), int(Dh), int(Fc)], "spacing_mm": [sl, sd, sf], "datatype": "RGBA32 (A = max channel)",
            "colours": used}


def vote_only_members(gmd: dict, pairs: dict, ok_ids: set) -> list:
    """[(member, {df, dx})] — refused pairs that still VOTE on the dome (group_consensus.VOTE_ONLY_REL / _NCC): their
    own line is independent of the refused transform; only df and the per-frame dx place the vote. A pair whose df /
    dx are not finite cannot place its vote and is skipped."""
    out = []
    for cid, r in pairs.items():
        if cid in ok_ids or r.ok:
            continue
        rel = float(r.relative_match) if r.relative_match is not None else float("nan")
        ncc = float(r.ncc_coarse) if r.ncc_coarse is not None else float("nan")
        if not (np.isfinite(rel) and rel >= gc_.VOTE_ONLY_REL and np.isfinite(ncc) and ncc >= gc_.VOTE_ONLY_NCC):
            continue
        dx = np.asarray(r.dx_applied, float)
        if cid not in gmd or not np.isfinite(dx).any():
            continue
        out.append((gmd[cid], {"df": int(r.df), "dx": dx, "a": np.zeros_like(dx), "b": np.zeros_like(dx)}))
    return out


def pair_info_of(pairs: dict) -> dict:
    return {cid: {"ok": bool(r.ok), "rel": (float(r.relative_match) if np.isfinite(float(r.relative_match)) else None),
                  "ncc_coarse": (float(r.ncc_coarse) if np.isfinite(float(r.ncc_coarse)) else None), "flags": list(r.flags)}
            for cid, r in pairs.items()}


def build_consensus(group: str, ref, members: list, transforms: dict, out_dir: Path, log=print, ctx: dict | None = None, *,
                    vote_only: list | None = None, pair_info: dict | None = None) -> dict:
    """The consensus stage: union canvas → placements → consensus v2 (group_consensus.consensus_v2) → fused image +
    consensus.png → aligned_rgb_pairs.nii.gz (+ a copy as aligned_rgb.nii.gz until apply_transforms replaces it) →
    consensus.json. `members` = the reference first, then the ok members (all on the common lateral grid);
    `transforms` = cid → {df, dx, a, b} (identity for the reference); `vote_only` = [(member, {df, dx})] refused
    members that vote on the dome only (vote_only_members); `pair_info` cid → {ok, rel, ncc_coarse} for the record.
    `ctx` (optional dict) receives the arrays apply_transforms / the sensitivity stage need: canvas, lines (cid →
    placed served line (Lc, Fc)), curve (Lc, Fc), colmask, colours, windows, cons (the consensus_v2 dict)."""
    t0 = time.time()
    canvas = union_canvas(members, transforms)
    Lc, Dc, Fc = canvas["shape"]
    log(f"  consensus: union canvas origin {canvas['origin']} shape {canvas['shape']} ({len(members)} members)")
    if Lc * Dc * Fc > 400_000_000:
        raise RuntimeError(f"union canvas too large ({canvas['shape']}) — a member's transform is wild")
    V = np.asarray(ref.volume, np.float32)
    pos = V[V > 0]
    lo, hi = (float(np.percentile(pos, 1)), float(np.percentile(pos, 99))) if pos.size else (0.0, 1.0)
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = 0.0, max(1.0, float(V.max()))
    # the 3-D volume's channels: TISSUE only — render_overlay's tissue threshold (0.35 × p99.5) maps to 0 so the speckle
    # floor carries no colour / alpha (an opaque block would hide the overlap); p99.5 maps to 255
    vtop = float(np.percentile(pos, 99.5)) if pos.size else hi
    if not np.isfinite(vtop) or vtop <= 0:
        vtop = hi
    v_lo, v_hi = 0.35 * vtop, vtop
    ssum = np.zeros((Lc, Dc, Fc), np.float32); cnt = np.zeros((Lc, Dc, Fc), np.uint8)
    colmask = np.zeros((Lc, Fc), bool)
    lines: dict = {}; chans: list = []; colours: dict = {}
    _pal = member_palette(len(members))          # one distinct colour per replicate (reviewer 2026-09-12)
    for i, m in enumerate(members):
        t1 = time.time()
        vol, msk, line, col = place_on_canvas(m, transforms[m.cid], canvas)
        ssum += np.where(msk, vol, 0.0); cnt += msk
        colmask |= col
        lines[m.cid] = line
        colours[m.cid] = _pal[i]
        h, hk = _half_res(vol, msk)
        chans.append((m.cid, np.where(hk, _win_uint8(h, v_lo, v_hi), 0).astype(np.uint8)))
        del vol, msk
        log(f"    placed {m.cid}: line cells {int(np.isfinite(line).sum())} columns {int(col.sum())} {time.time() - t1:.1f}s")
    fused = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0).astype(np.float32)
    del ssum
    cons = gc_.consensus_v2(ref, members, transforms, lines, colmask, canvas, vote_only=vote_only, pair_info=pair_info, log=log)
    curve = cons["curve"]
    prov = consensus_curve(lines, colmask)                                   # the provisional curve: diagnostic only
    rms: dict = {}; rms_prov: dict = {}
    for cid, ln in lines.items():
        both = np.isfinite(ln) & np.isfinite(curve)
        rms[cid] = (float(np.sqrt(np.mean((ln[both] - curve[both]) ** 2))) if both.any() else None)
        bothp = np.isfinite(ln) & np.isfinite(prov["curve"])
        rms_prov[cid] = (float(np.sqrt(np.mean((ln[bothp] - prov["curve"][bothp]) ** 2))) if bothp.any() else None)
    bothc = np.isfinite(curve) & np.isfinite(prov["curve"])
    v2_minus_prov = _sizes(curve[bothc] - prov["curve"][bothc]) if bothc.any() else {"peak": None, "rms": None, "n": 0}
    fr_cov = np.flatnonzero(colmask.any(axis=0))
    covered = [int(fr_cov[0]), int(fr_cov[-1])] if fr_cov.size else [None, None]
    bands_txt = "  ".join(f"b{b['band']}{'*' if b['apex'] else ''} κf {b['kappa_frames']:.3f}/κa {b['kappa_axial']:.3f}={b['ratio']:.2f} {b['verdict']}"
                          if np.isfinite(b["ratio"]) else f"b{b['band']} {b['verdict']}" for b in cons["bands"])
    title = (f"{group}: consensus v2 (majority own dome, axial-checked) on the union canvas — fused = masked mean of {len(members)} placed members "
             f"(anchor {_short(ref.cid)}); dashed = each member's served line placed by the engine's rigid transform; thick white = consensus v2 "
             f"(dome = majority of the scans' OWN domes: {cons['source_counts']['majority']} majority / {cons['source_counts']['axial']} axial / "
             f"{cons['source_counts']['reference']} reference / {cons['source_counts']['single']} single laterals; c1 / c0 from the tissue-placed lines)\n"
             f"axial witness per band [{gc_.RATIO_ENVELOPE[0]}, {gc_.RATIO_ENVELOPE[1]}] (κ in 1/mm{'; lateral scale UNVERIFIED (legacy 4.0/513 mm header)' if not cons['axial']['scale_trusted'] else ''}): "
             f"{bands_txt}   flags {cons['flags'] or 'none'}\n"
             f"RMS to consensus: " + "  ".join(f"{_short(c)} {v:.1f}px" if v is not None else f"{_short(c)} —" for c, v in rms.items())
             + f"   (v2 − provisional: peak {v2_minus_prov['peak'] if v2_minus_prov['peak'] is None else round(v2_minus_prov['peak'], 1)} rms "
               f"{v2_minus_prov['rms'] if v2_minus_prov['rms'] is None else round(v2_minus_prov['rms'], 2)} px)")
    info = render_consensus(fused, cnt, lines, curve, colmask, canvas, colours, hi, out_dir / CONSENSUS_PNG, title,
                            curve_label="consensus v2 (majority own dome, axial-checked)", spacing_mm=ref.spacing)
    log(f"    consensus.png frames {info['frames']} laterals {info['laterals']}  fitted laterals {int(cons['fitted'].sum())}/{Lc}  "
        f"v2 − provisional peak {v2_minus_prov['peak']} rms {v2_minus_prov['rms']} px")
    vinfo = write_rgb_volume(chans, ref.spacing, out_dir / VOLUME_PAIRS_NAME, colours=colours)
    import shutil
    shutil.copyfile(out_dir / VOLUME_PAIRS_NAME, out_dir / VOLUME_NAME)     # replaced by the FINAL placement later
    log(f"    {VOLUME_PAIRS_NAME} {vinfo} (copied to {VOLUME_NAME} until the transforms are applied)")
    channels = {cid: CHANNEL_NAMES[i % 3] for i, (cid, _u) in enumerate(chans)}
    if ctx is not None:
        ctx.update(canvas=canvas, lines=lines, curve=curve, colmask=colmask, colours=colours, window=(lo, hi),
                   volume_window=(v_lo, v_hi), rms_before=rms, cons=cons, prov=prov)
    fl = lambda v: (None if not np.isfinite(v) else float(v))  # noqa: E731
    per_lateral = [{"lateral": int(l), "ref_lateral": int(l + canvas["origin"][0]),
                    "c0": fl(cons["coef"][l, 0]), "c1": fl(cons["coef"][l, 1]), "c2": fl(cons["coef"][l, 2]),
                    "c2_raw": fl(cons["coef_raw"][l, 2]), "kappa_star": fl(cons["kappa_star"][l]), "dome_source": str(cons["dome_source"][l]),
                    "n_votes": int(cons["n_votes"][l]), "tau": fl(cons["tau"][l]),
                    "votes": {c: fl(cons["votes"][c]["kappa_lat"][l]) for c in cons["voter_ids"]},
                    "c2_provisional": (fl(prov["coef"][l, 2] / prov["scale"] ** 2) if np.isfinite(prov["coef"][l, 2]) else None),
                    "fitted": bool(cons["fitted"][l]), "n_points": int(cons["n_pts"][l])} for l in range(Lc)]
    v2 = gc_.summary_of(cons)
    cj = {"group": group, "reference": ref.cid, "note": CONSENSUS_NOTE, "provisional": False, **v2,
          "n_members": len(members), "members": [m.cid for m in members],
          "canvas": {**canvas, "coordinates": "reference grid; canvas index = reference index - origin"},
          "covered_frame_range": covered, "covered_frames": int(fr_cov.size),
          "curve_model": {"form": "z = c0 + c1*u + c2*u^2, u = canvas_frame - fc (px, frames)", "fc": cons["fc"], "scale": 1.0,
                          "robust": "3xMAD reweighting, >= 30 evidence frames per lateral spanning >= 40",
                          "smoothing": v2["dome"]["smoothing"],
                          "fitted_laterals": int(cons["fitted"].sum()), "lateral_range": cons.get("lateral_range")},
          "provisional_curve": {"note": CONSENSUS_NOTE_PROVISIONAL, "rms_to_curve_px": rms_prov, "v2_minus_provisional_px": v2_minus_prov,
                                "fitted_laterals": int(prov["fitted"].sum())},
          "per_lateral": per_lateral, "rms_to_consensus_px": rms, "colours": colours, "channels": channels,
          "window": [lo, hi], "volume_window": [v_lo, v_hi], "png": CONSENSUS_PNG, "png_info": info, "volume": VOLUME_NAME,
          "volume_pairs": VOLUME_PAIRS_NAME, "volume_info": vinfo, "seconds": time.time() - t0}
    _write_json(out_dir / CONSENSUS_JSON, cj)
    # the result.json summary (small: no per-lateral table)
    summ = {k: v for k, v in cj.items() if k != "per_lateral"}
    summ["json"] = CONSENSUS_JSON
    return summ


# ── (d) REFERENCE SENSITIVITY: every other member as the coordinate anchor ────────────────────────────────────
SENSITIVITY_NOTE = ("the reference is only the coordinate anchor; this is how much the consensus would move if another scan were the "
                    "anchor: the whole consensus re-run with that scan as the anchor (its own pairs, cached) and mapped into the served "
                    "coordinates by the served pair. LEAD NUMBERS: shape_spread_px (what is left beyond a whole-volume pose and a smooth "
                    "dome move — the anchor dependence the final result keeps) and the dome κ spread per band. The RAW spread_px (RMS "
                    "over the covered cells) is dominated by each anchor's own smooth dome error (the consensus lives in its anchor's "
                    "coordinates; the apply step's dome move removes that error by bending the anchor) plus the pair engine's placement "
                    "error, so it is shown second, with its dome part in κ")


def reference_sensitivity_stage(group: str, ref, gmd: dict, member_ids: list, rule_transforms: dict, ctx: dict, pair_provider,
                                out_dir: Path | None = None, log=print, progress=None) -> dict:
    """For every member X other than the served anchor: pairs m → X for every other member (`pair_provider(anchor, mov)` →
    {ok, df, dx, a, b, rel, ncc_coarse, flags} or None; the job wraps register_pair + the pair cache, tests a known
    composition), the union canvas of X + its ok members, consensus v2 with X as the anchor, mapped into the served
    canvas by the served pair X → reference (group_consensus.consensus_spread). An X with no accepted pair to the served
    anchor is skipped (its consensus cannot be mapped). Returns {served_anchor, note, anchors: {cid: record},
    spread_max_px, kappa_by_band_by_anchor, kappa_band_spread}. Updates consensus.json when `out_dir` is given."""
    t_all = time.time()
    rule_curve = ctx["curve"]; rule_canvas = ctx["canvas"]; rule_cons = ctx["cons"]
    out: dict = {"served_anchor": ref.cid, "note": SENSITIVITY_NOTE, "anchors": {}, "seconds": None,
                 "grid": "every anchor on the served run's common lateral grid"}
    out["anchors"][ref.cid] = {"anchor": ref.cid, "served": True, "spread_px": 0.0, "shape_spread_px": 0.0, "median_abs_px": 0.0, "n_cells": int(np.isfinite(rule_curve).sum()),
                               "contributing": [m for m in rule_transforms if m != ref.cid], "refused": [], "skipped": None,
                               "kappa_by_band": [b["kappa_frames"] for b in rule_cons["bands"]], "sources": rule_cons["source_counts"],
                               "flags": list(rule_cons["flags"]), "seconds": 0.0}
    others = [c for c in member_ids if c != ref.cid]
    for i, X in enumerate(others):
        t0 = time.time()
        rec: dict = {"anchor": X, "served": False, "contributing": [], "refused": [], "vote_only": [], "skipped": None,
                     "spread_px": None, "median_abs_px": None, "n_cells": 0, "kappa_by_band": None, "sources": None, "flags": None}
        out["anchors"][X] = rec
        if X not in rule_transforms or X not in gmd:
            rec["skipped"] = f"no accepted pair {X} → {ref.cid}: a consensus anchored on {X} cannot be mapped into the served coordinates"
            log(f"    sensitivity anchor {X}: skipped ({rec['skipped']})")
            rec["seconds"] = time.time() - t0
            continue
        try:
            mX = gmd[X]; L_X, _D, F_X = mX.volume.shape
            trs = {X: identity_transform(F_X)}; vote_only: list = []; pinfo: dict = {}
            movs = [m for m in member_ids if m != X]
            for j, m in enumerate(movs):
                if progress:
                    progress(f"reference sensitivity: anchor {_short(X)} ({i + 1}/{len(others)}) pair {j + 1}/{len(movs)}")
                r = pair_provider(X, m)
                if r is None:
                    rec["refused"].append({"cid": m, "reason": "no pair"}); continue
                pinfo[m] = {"ok": bool(r["ok"]), "rel": r.get("rel"), "ncc_coarse": r.get("ncc_coarse"), "flags": list(r.get("flags") or [])}
                dx = np.asarray(r["dx"], float)
                if r["ok"]:
                    trs[m] = {"df": int(r["df"]), "dx": dx, "a": np.asarray(r["a"], float), "b": np.asarray(r["b"], float)}
                    rec["contributing"].append(m)
                    if m == ref.cid:               # the pair engine's round trip X → R → X (its own inconsistency)
                        rec["pair_roundtrip"] = gc_.roundtrip_error(rule_transforms[X], trs[m], L_X)
                else:
                    rec["refused"].append({"cid": m, "flags": pinfo[m]["flags"], "rel": r.get("rel"), "ncc_coarse": r.get("ncc_coarse"),
                                           "reason": r.get("reason"), "df": r.get("df"), "overlap_fraction": r.get("overlap_fraction")})
                    rel = r.get("rel"); ncc = r.get("ncc_coarse")
                    if (rel is not None and np.isfinite(rel) and rel >= gc_.VOTE_ONLY_REL and ncc is not None and np.isfinite(ncc)
                            and ncc >= gc_.VOTE_ONLY_NCC and np.isfinite(dx).any() and m in gmd):
                        vote_only.append((gmd[m], {"df": int(r["df"]), "dx": dx, "a": np.zeros_like(dx), "b": np.zeros_like(dx)}))
                        rec["vote_only"].append(m)
            members_X = [mX] + [gmd[m] for m in rec["contributing"]]
            canvas_X = union_canvas(members_X, trs)
            LcX, _DcX, FcX = canvas_X["shape"]
            if LcX * FcX > 20_000_000:
                raise RuntimeError(f"union canvas too large ({canvas_X['shape']})")
            lines_X: dict = {}; colmask_X = np.zeros((LcX, FcX), bool)
            for m in members_X:
                _v, _k, line, col = place_on_canvas(m, trs[m.cid], canvas_X, volume=False)
                lines_X[m.cid] = line; colmask_X |= col
            cons_X = gc_.consensus_v2(mX, members_X, trs, lines_X, colmask_X, canvas_X, vote_only=vote_only, pair_info=pinfo)
            sp = gc_.consensus_spread(cons_X["curve"], canvas_X, rule_curve, rule_canvas, rule_transforms[X], L_X, F_X,
                                      L_R=int(ref.served.shape[0]), spacing_R=np.asarray(ref.spacing, float))
            rec.update(sp)
            rec.update(kappa_by_band=[b["kappa_frames"] for b in cons_X["bands"]], sources=cons_X["source_counts"], flags=list(cons_X["flags"]),
                       axial_verdict=cons_X["axial_verdict"], dome_verdict=cons_X["dome_verdict"], canvas=canvas_X,
                       covered_cells=int(np.isfinite(cons_X["curve"]).sum()))
            log(f"    sensitivity anchor {X}: contributing {rec['contributing']} refused {[r_['cid'] for r_ in rec['refused']]} "
                f"spread {sp['spread_px'] if sp['spread_px'] is None else round(sp['spread_px'], 2)} px (beyond a whole-volume pose "
                f"{sp['spread_beyond_pose_px'] if sp['spread_beyond_pose_px'] is None else round(sp['spread_beyond_pose_px'], 2)}; shape beyond pose + smooth dome move "
                f"{sp['shape_spread_px'] if sp['shape_spread_px'] is None else round(sp['shape_spread_px'], 2)}, dome part κ "
                f"{sp['dome_part_kappa'] if sp['dome_part_kappa'] is None else round(sp['dome_part_kappa'], 4)}; median |Δ| "
                f"{sp['median_abs_px'] if sp['median_abs_px'] is None else round(sp['median_abs_px'], 2)}, n {sp['n_cells']}; "
                f"pair round trip {json.dumps(jsonable(rec.get('pair_roundtrip')))}) "
                f"κ bands {[round(k, 4) if k is not None and np.isfinite(k) else None for k in rec['kappa_by_band']]} {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001 — a diagnostic never sinks the job
            rec["skipped"] = f"failed: {type(e).__name__}: {e}"
            log(f"    sensitivity anchor {X} FAILED: {e}\n{traceback.format_exc()}")
        rec["seconds"] = time.time() - t0
    spreads = [r["spread_px"] for r in out["anchors"].values() if r.get("spread_px") is not None and not r.get("served")]
    out["spread_max_px"] = (max(spreads) if spreads else None)
    shapes = [r["shape_spread_px"] for r in out["anchors"].values() if r.get("shape_spread_px") is not None and not r.get("served")]
    out["shape_spread_max_px"] = (max(shapes) if shapes else None)
    out["anchors_evaluated"] = int(len(spreads)); out["anchors_skipped"] = [c for c, r in out["anchors"].items() if r.get("skipped")]
    kb = {c: r["kappa_by_band"] for c, r in out["anchors"].items() if r.get("kappa_by_band")}
    out["kappa_by_band_by_anchor"] = kb
    if len(kb) >= 2:
        nb = len(next(iter(kb.values())))
        spread_b = []
        for k in range(nb):
            vals = [v[k] for v in kb.values() if v[k] is not None and np.isfinite(v[k])]
            spread_b.append((max(vals) - min(vals)) if len(vals) >= 2 else None)
        out["kappa_band_spread"] = spread_b
    out["seconds"] = time.time() - t_all
    if out_dir is not None:
        cj = read_json(Path(out_dir) / CONSENSUS_JSON)
        if cj is not None:
            cj["reference_sensitivity"] = {k: v for k, v in out.items()}
            _write_json(Path(out_dir) / CONSENSUS_JSON, cj)
    return out


# ── FINAL per-member transforms + APPLY (reviewer ask 2026-09-11 #3) ──────────────────────────────────────────
# "now that alignment is performed on CS001_OS are there small axial changes to any or all of the scans such that
# smoothness is maximised and consistency between each scan is maximised?" — yes, and the change must be LOW-ORDER
# along frames (defect 2026-09-11, cs001_os_v1 lateral 149: the first version fitted δa / δb per frame to the LINE's
# residual, so a served line that dipped into the tissue over a few frames became a −12 px tissue bump; LINE ERROR
# MUST NEVER MOVE TISSUE). Now, per member (the reference included): Q_m = the member's OWN smooth dome (member_dome:
# the consensus model fitted to its placed served line alone); δ(l, f) = C(l, f) − Q_m(l, f) is smooth in f by
# construction; per frame δ ≈ δa(f) + δb(f)·x over the member's valid laterals (robust_line; a smooth-to-smooth dome
# correction) and a_final = a_pair + δa, b_final = b_pair + δb (the RIGID RULE: a B-scan only ever moves by a
# per-frame depth shift and a tilt, never a per-column warp). Frames the consensus does not cover keep the pair
# transform (hold, no extrapolation). The residual of δ beyond the tilt (RMS over the valid laterals) is the true
# within-B-scan shape mismatch; above PROFILE_BEYOND_TILT_PX the frame is flagged 'profile_beyond_tilt'. The LINE's
# residual after the move (placed line + δa + δb·x − C, RMS over laterals) is REPORTED per frame as 'line_residual'
# and frames beyond LINE_OFF_CONSENSUS_PX are flagged 'line_off_consensus' — a line-quality flag (candidate GT for
# the edge tools), never applied.
# REVISION 3 (verifier 2026-09-11, cs001_os_v1): the per-frame 3×MAD inlier set of the shift + tilt fit flipped frame to
# frame (507 → 462 → 476 → 512 laterals) and STEPPED the applied field (|d² δa| 0.43 px, δb 1.47 → 2.78 px between two
# frames) although C − Q_m is smooth. Now the fit uses ONE FIXED inlier set per member (laterals inlier on ≥
# FIXED_INLIER_FRAC of their fitted frames, else the member's own column mask), plain least squares per frame, and δa(f),
# δb(f) are smoothed along frames (Savitzky-Golay DELTA_SG_WIN / order 2 over the covered span; frames outside it stay
# held). The APPLIED field is the smoothed one (delta_a / delta_b; the pre-smoothing fit is delta_a_raw / delta_b_raw)
# and must meet |d² δa| ≤ SMOOTH_D2_A_PX, |d² δb| ≤ SMOOTH_D2_B_PX (smooth_ok). The residual beyond the move is split
# into a LATERAL-PROFILE part (its per-lateral mean over frames — the frame-independent difference between the
# consensus's and the member's own lateral profile with c0 free; a rigid per-frame move can never remove it and it is
# not a dome error) and the DOME part (what varies along frames). AFTER_RMS_BAR_PX applies to the DOME part: 1.0 px,
# because with three replicates each own dome carries its own error (2–4 px at the frame ends on CS001) and the part
# of C − Q_m a shift + tilt cannot absorb — the across-lateral variation of that dome difference — sits at 0.5–1 px for
# honest replicates; the former 0.5 px bar was met only while c0 was tied to the consensus's own smoothing.
def robust_line(x: np.ndarray, r: np.ndarray, n_iter: int = 6, min_pts: int = 12):
    """r ≈ c0 + c1·x, least squares with 3×MAD reweighting (a point beyond 3 × 1.4826 × MAD of the residuals is
    dropped, MAD floored at 0.5 px). Returns (c0, c1, inlier mask) or None when fewer than min_pts points survive."""
    x = np.asarray(x, float); r = np.asarray(r, float)
    A = np.stack([np.ones_like(x), x], axis=1)
    w = np.ones(x.size, bool)
    c = None
    for _ in range(n_iter):
        if w.sum() < min_pts:
            return None
        c, *_ = np.linalg.lstsq(A[w], r[w], rcond=None)
        res = r - A @ c
        mad = float(np.median(np.abs(res[w] - np.median(res[w]))))
        thr = 3.0 * 1.4826 * max(mad, 0.5)
        w_new = np.abs(res) <= thr
        if np.array_equal(w_new, w):
            break
        w = w_new
    if c is None or not np.all(np.isfinite(c)):
        return None
    return float(c[0]), float(c[1]), w


def _sizes(v: np.ndarray, sel: np.ndarray | None = None) -> dict:
    v = np.asarray(v, float)
    if sel is not None:
        v = v[np.asarray(sel, bool)]
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"peak": None, "rms": None, "n": 0}
    return {"peak": float(np.max(np.abs(v))), "rms": float(np.sqrt(np.mean(v * v))), "n": int(v.size)}


def smooth_delta(v: np.ndarray, ok: np.ndarray, win: int = DELTA_SG_WIN) -> np.ndarray:
    """δ(f) smoothed along frames: Savitzky-Golay `win` / order 2 over the span of `ok` frames (gaps interpolated for
    the filter, the edge windows fitted by the polynomial — no running median: a median would step). NaN outside the
    span and on frames not ok (those hold the pair transform)."""
    from scipy.signal import savgol_filter
    out = np.full(v.shape, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    span = np.arange(idx[0], idx[-1] + 1)
    x = np.interp(span, idx, v[idx])
    w = min(win, span.size if span.size % 2 == 1 else span.size - 1)
    if w >= 5:
        x = savgol_filter(x, w, 2, mode="interp")
    out[span] = x
    out[~ok] = np.nan
    return out


def d2_max(v: np.ndarray, ok: np.ndarray) -> float | None:
    """max |second difference| of a per-frame series over runs of consecutive ok frames (None without 3 such frames)."""
    v = np.asarray(v, float); ok = np.asarray(ok, bool) & np.isfinite(v)
    d2 = v[2:] - 2.0 * v[1:-1] + v[:-2]
    m = ok[2:] & ok[1:-1] & ok[:-2]
    return float(np.max(np.abs(d2[m]))) if m.any() else None


def smooth_over_frames(v: np.ndarray, ok: np.ndarray, win: int = 11) -> np.ndarray:
    """The smooth (dome) part of a per-frame series: running median 5 + Savitzky-Golay `win` / order 2 over the
    frames where `ok`, NaN elsewhere (gaps between ok frames interpolated for the filter only)."""
    from scipy.signal import savgol_filter
    out = np.full(v.shape, np.nan)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    span = np.arange(idx[0], idx[-1] + 1)
    x = np.interp(span, idx, v[idx])
    if span.size >= 5:
        x = ndi.median_filter(x, size=5, mode="nearest")
    w = min(win, span.size if span.size % 2 == 1 else span.size - 1)
    if w >= 5:
        x = savgol_filter(x, w, 2, mode="interp")
    out[span] = x
    out[~ok] = np.nan
    return out


def member_dome(line: np.ndarray, colmask_m: np.ndarray, *, min_frames: int = 30) -> dict:
    """Q_m — the member's OWN smooth dome: consensus_curve run on this member's placed served line alone (per canvas
    lateral a robust quadratic along frames over ≥ min_frames covered frames, 3×MAD reweighting, c2 / c1 smoothed
    across laterals like the consensus and c0 only LIGHTLY — consensus v2's profile, so C − Q_m is the smooth dome
    difference and the shared lateral profile (divots, limbus) cancels). Returns consensus_curve's dict."""
    return consensus_curve({"m": line}, np.asarray(colmask_m, bool), min_frames=min_frames,
                           med_c0=gc_.C0_MED, sg_c0=gc_.C0_SG)


def final_transforms(members: list, transforms: dict, canvas: dict, lines: dict, curve: np.ndarray, *,
                     min_laterals: int = 30, fixed_inlier_frac: float = FIXED_INLIER_FRAC, sg_win: int = DELTA_SG_WIN) -> dict:
    """Per member the FINAL rigid transform (see the section comment): a_final = a_pair + δa(f), b_final = b_pair +
    δb(f) where (δa, δb) is the per-frame shift + tilt fit of the SMOOTH difference C − Q_m (consensus minus the
    member's own dome) over ONE FIXED inlier set of the member's laterals (revision 3), smoothed along frames — never
    of the line itself. Returns cid → {df, dx, a, b (final), a_pair, b_pair, delta_a, delta_b (the APPLIED, smoothed
    fields; NaN where held), delta_a_raw, delta_b_raw (the per-frame fits), delta_a_smooth (= delta_a), covered (F,)
    bool, residual_rms (F,) = RMS of (C − Q_m) − (δa + δb·x) over the valid laterals (the within-B-scan shape mismatch
    beyond the rigid move), dome_part_rms (F,) = the same with the lateral profile p(l) removed, profile (Lc,) = p(l)
    (the per-lateral mean over frames of that residual), dome_rms_before (F,) = RMS of C − Q_m, rms_before (F,) = RMS
    of the LINE − C before, line_residual (F,) = RMS of the placed line after the move − C (REPORTED, never applied),
    inliers (F,) laterals in the fit, fixed_inliers (Lc,) bool, n_points, profile_beyond_tilt (F,) bool,
    line_off_consensus (F,) bool, dome_coef (Lc, 3), dome_curve (Lc, Fc), sizes}."""
    l0, z0, f0 = canvas["origin"]; Lc, Fc = curve.shape
    lat_c = np.arange(Lc, dtype=float) + l0
    out: dict = {}
    for m in members:
        T = transforms[m.cid]
        L, D, F = m.volume.shape
        hs = max(1.0, (L - 1) / 2.0)
        a_f = np.asarray(T["a"], float).copy(); b_f = np.asarray(T["b"], float).copy()
        da_raw = np.full(F, np.nan); db_raw = np.full(F, np.nan)
        rr = np.full(F, np.nan); rb = np.full(F, np.nan); dr = np.full(F, np.nan); lr = np.full(F, np.nan); dp = np.full(F, np.nan)
        nin = np.zeros(F, int); npts = np.zeros(F, int)
        covered = np.zeros(F, bool)
        ln = lines[m.cid]
        _v, _k, _ln, colm = place_on_canvas(m, T, canvas, volume=False)          # the member's own column mask
        dome = member_dome(ln, colm)
        Q = dome["curve"]
        # the smooth difference r(l, f) = C − Q_m and the tilt coordinate x(l, f) on the member's evidence cells
        R = np.full((Lc, F), np.nan); X = np.full((Lc, F), np.nan); fcs = np.full(F, -1, int)
        for f in range(F):
            fc = f + int(T["df"]) - f0
            if not (0 <= fc < Fc) or not _frame_ok(T, f):
                continue
            lm = lat_c - float(T["dx"][f])
            x = (lm - (L - 1) / 2.0) / hs
            sel = colm[:, fc] & np.isfinite(curve[:, fc]) & np.isfinite(Q[:, fc])
            npts[f] = int(sel.sum())
            if sel.sum() < min_laterals:
                continue
            fcs[f] = fc
            R[sel, f] = curve[sel, fc] - Q[sel, fc]                                # smooth − smooth: the dome correction
            X[sel, f] = x[sel]
        cand = np.flatnonzero(fcs >= 0)
        # pass 1: per-frame robust fits vote for the FIXED inlier set (a lateral inlier on ≥ fixed_inlier_frac of its frames)
        vote = np.zeros(Lc); cnt = np.zeros(Lc)
        for f in cand:
            sel = np.isfinite(R[:, f])
            fit = robust_line(X[sel, f], R[sel, f])
            if fit is None:
                continue
            idx = np.flatnonzero(sel)
            cnt[idx] += 1; vote[idx[fit[2]]] += 1
        fixed = (cnt > 0) & (vote >= fixed_inlier_frac * cnt)
        fixed_source = "inlier_vote"
        if int(fixed.sum()) < min_laterals:
            fixed = cnt > 0; fixed_source = "column_mask"                          # the member's own column mask
        # pass 2: plain least squares per frame on the fixed set (the SAME laterals frame after frame)
        for f in cand:
            sel = np.isfinite(R[:, f]) & fixed
            if sel.sum() < min_laterals:
                sel = np.isfinite(R[:, f])
                if sel.sum() < min_laterals:
                    continue
            A = np.stack([np.ones(int(sel.sum())), X[sel, f]], 1)
            c, *_ = np.linalg.lstsq(A, R[sel, f], rcond=None)
            if not np.all(np.isfinite(c)):
                continue
            covered[f] = True
            da_raw[f] = float(c[0]); db_raw[f] = float(c[1]); nin[f] = int(sel.sum())
        # the APPLIED field: δa / δb smoothed along the covered frames
        da = smooth_delta(da_raw, covered, sg_win); db = smooth_delta(db_raw, covered, sg_win)
        covered &= np.isfinite(da) & np.isfinite(db)
        a_f[covered] = a_f[covered] + da[covered]; b_f[covered] = b_f[covered] + db[covered]
        # residuals of the SMOOTH difference beyond the applied move, split into a lateral profile and a dome part
        RES = np.full((Lc, F), np.nan)
        for f in np.flatnonzero(covered):
            sel = np.isfinite(R[:, f])
            RES[sel, f] = R[sel, f] - (da[f] + db[f] * X[sel, f])
        with np.errstate(invalid="ignore"):
            prof = np.nanmean(RES[:, covered], axis=1) if covered.any() else np.full(Lc, np.nan)
        for f in np.flatnonzero(covered):
            fc = fcs[f]
            sel = np.isfinite(RES[:, f])
            res = RES[sel, f]
            rr[f] = float(np.sqrt(np.mean(res * res)))
            dres = res - np.where(np.isfinite(prof[sel]), prof[sel], 0.0)
            dp[f] = float(np.sqrt(np.mean(dres * dres)))
            r = R[sel, f]
            dr[f] = float(np.sqrt(np.mean(r * r)))
            # the LINE's residual to the consensus before / after the move — a line-quality number, reported only
            both = np.isfinite(ln[:, fc]) & np.isfinite(curve[:, fc])
            if both.any():
                lm = lat_c - float(T["dx"][f]); x = (lm - (L - 1) / 2.0) / hs
                d0 = curve[both, fc] - ln[both, fc]
                rb[f] = float(np.sqrt(np.mean(d0 * d0)))
                d1 = ln[both, fc] + da[f] + db[f] * x[both] - curve[both, fc]
                lr[f] = float(np.sqrt(np.mean(d1 * d1)))
        flag = covered & (rr > PROFILE_BEYOND_TILT_PX)
        line_off = covered & (lr > LINE_OFF_CONSENSUS_PX)
        a_pair = np.asarray(T["a"], float); b_pair = np.asarray(T["b"], float)

        def _rms(v: np.ndarray, sel: np.ndarray):
            v = v[sel]; v = v[np.isfinite(v)]
            return float(np.sqrt(np.mean(v * v))) if v.size else None
        dome_after = _rms(dp, covered)                                              # the DOME part (the bar)
        pf = prof[np.isfinite(prof) & (cnt > 0)]
        profile_rms = float(np.sqrt(np.mean(pf * pf))) if pf.size else None
        d2a = d2_max(da, covered); d2b = d2_max(db, covered)
        smooth_ok = ((d2a is None or d2a <= SMOOTH_D2_A_PX) and (d2b is None or d2b <= SMOOTH_D2_B_PX))
        sizes = {"delta_a": _sizes(da, covered), "delta_b": _sizes(db, covered),
                 "delta_a_raw": _sizes(da_raw, covered), "delta_b_raw": _sizes(db_raw, covered),
                 "delta_a_smooth": _sizes(da, covered), "delta_a_jitter": _sizes(da_raw - da, covered),
                 "delta_b_jitter": _sizes(db_raw - db, covered),
                 "delta_a_d2_max": d2a, "delta_b_d2_max": d2b, "smooth_bar_d2_a_px": SMOOTH_D2_A_PX, "smooth_bar_d2_b_px": SMOOTH_D2_B_PX,
                 "smooth_ok": bool(smooth_ok), "smoothing": f"Savitzky-Golay {sg_win} / order 2 over the covered frames (frames outside held)",
                 "fixed_inliers": int(fixed.sum()), "fixed_inlier_source": fixed_source, "fixed_inlier_frac": fixed_inlier_frac,
                 "tissue_a": _sizes(a_pair, covered), "tissue_b": _sizes(b_pair, covered),
                 "rms_before_px": _rms(rb, covered),                     # the LINE to the consensus, pair placement
                 "rms_after_px": _rms(lr, covered),                      # the LINE to the consensus after the smooth move (reported, not applied)
                 "rms_after_all_covered_px": _rms(lr, covered),
                 "line_residual_rms_px": _rms(lr, covered), "line_residual": _sizes(lr, covered),
                 "line_off_consensus_frames": int(line_off.sum()), "line_off_consensus_px": LINE_OFF_CONSENSUS_PX,
                 "dome_rms_before_px": _rms(dr, covered),                # C − Q_m before the move
                 "dome_rms_after_px": dome_after,                        # the DOME part of C − Q_m beyond the shift + tilt (varies along frames)
                 "residual_rms_after_px": _rms(rr, covered),             # the whole residual beyond the shift + tilt (dome part + lateral profile)
                 "profile_rms_px": profile_rms,                          # the LATERAL-PROFILE part (frame-independent; a rigid move cannot remove it)
                 "dome_fitted_laterals": int(np.asarray(dome["fitted"], bool).sum()),
                 "frames": int(F), "covered_frames": int(covered.sum()), "held_frames": int(F - covered.sum()),
                 "profile_beyond_tilt_frames": int(flag.sum()),
                 "after_rms_bar_px": AFTER_RMS_BAR_PX, "after_rms_bar_of": "dome part",
                 "after_rms_ok": (bool(dome_after <= AFTER_RMS_BAR_PX) if dome_after is not None else None)}
        out[m.cid] = {"df": int(T["df"]), "dx": np.asarray(T["dx"], float), "a": a_f, "b": b_f, "a_pair": a_pair, "b_pair": b_pair,
                      "delta_a": da, "delta_b": db, "delta_a_raw": da_raw, "delta_b_raw": db_raw, "delta_a_smooth": da, "covered": covered,
                      "residual_rms": rr, "dome_part_rms": dp, "profile": prof, "fixed_inliers": fixed,
                      "dome_rms_before": dr, "rms_before": rb, "line_residual": lr, "inliers": nin, "n_points": npts,
                      "profile_beyond_tilt": flag, "line_off_consensus": line_off,
                      "dome_coef": np.asarray(dome["coef"], float), "dome_curve": Q, "dome_fc": dome["fc"], "dome_scale": dome["scale"],
                      "sizes": sizes}
    return out


def canvas_affine(spacing_mm, origin) -> np.ndarray:
    """The union canvas's NIfTI affine: the case volumes' orientation (lat → −X, frames → −Y, depth → −Z, see
    write_rgb_volume) at full spacing, translated so canvas index (0, 0, 0) sits at REFERENCE index `origin`."""
    sp = np.asarray(spacing_mm, float)
    aff = np.array([[-sp[0], 0.0, 0.0, 0.0],
                    [0.0, 0.0, -sp[2], 0.0],
                    [0.0, -sp[1], 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]], float)
    aff[:3, 3] = aff[:3, :3] @ np.asarray(origin, float)
    return aff


def write_member_volume(vol: np.ndarray, msk: np.ndarray, spacing_mm, canvas: dict, path: Path) -> dict:
    """aligned_<cid>.nii.gz: the member moved by its FINAL transform on the union canvas, full resolution, uint16
    (the case volumes are uint16), uncovered = 0, canvas affine."""
    import nibabel as nib
    u16 = np.where(msk, np.clip(np.rint(vol), 0, 65535), 0).astype(np.uint16)
    img = nib.Nifti1Image(np.ascontiguousarray(u16), canvas_affine(spacing_mm, canvas["origin"]))
    img.header.set_xyzt_units("mm")
    nib.save(img, str(path))
    return {"shape": [int(v) for v in u16.shape], "dtype": "uint16", "covered_voxels": int(msk.sum())}


# ── SCRUB data (reviewer ask 2026-09-11 #4: "allow the user to scrub through the sagittal views (before and after)
#    of the replicates after axial changes are applied") ──────────────────────────────────────────────────────────
def reset_scrub_dir(scrub_dir: Path) -> None:
    """Empty align_min/scrub/ (memmaps, meta.json and the PNG cache of the previous run) — meta.json goes first so a
    reader never pairs a new meta with old volumes."""
    scrub_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(scrub_dir.iterdir(), key=lambda q: 0 if q.name == SCRUB_META else 1):
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass


def write_scrub_volume(vol: np.ndarray, msk: np.ndarray, vmax: float, path: Path) -> dict:
    """<stage>_<cid>.npy: the placed member as an uncompressed uint8 volume in C order (Lc, Dc, Fc) — one canvas
    lateral is one contiguous (Dc, Fc) block, so a sagittal cut is a memory-mapped read of Dc·Fc bytes. Windowed
    0 → vmax like the montages (vmin 0, vmax = the reference's p99); covered cells are ≥ 1 (uncovered = 0) so the
    coverage survives the windowing. Written under a temp name then renamed."""
    u8 = _win_uint8(vol, 0.0, float(vmax))
    u8 = np.where(np.asarray(msk, bool), np.maximum(u8, 1), 0).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, u8)
    os.replace(tmp, path)
    return {"file": path.name, "shape": [int(v) for v in u8.shape], "dtype": "uint8", "bytes": int(u8.nbytes), "window": [0.0, float(vmax)]}


def per_lateral_rms(line: np.ndarray, curve: np.ndarray) -> np.ndarray:
    """RMS of line − curve along frames per lateral over the frames where both are finite (NaN where none)."""
    both = np.isfinite(line) & np.isfinite(curve)
    d = np.where(both, line - curve, 0.0)
    n = both.sum(axis=1)
    out = np.full(line.shape[0], np.nan)
    ok = n > 0
    out[ok] = np.sqrt((d[ok] ** 2).sum(axis=1) / n[ok])
    return out


# ── TISSUE-edge metrics (reviewer ask 2026-09-11: "maybe have a RMS read out for the axially corrected beneath the
#    current RMS") — measured from the IMAGES, not the lines, so the benefit of the axial changes is a number even
#    when the eye cannot tell. ─────────────────────────────────────────────────────────────────────────────────────
TISSUE_HALF = 40        # rows searched either side of the consensus
TISSUE_RUN = 5          # running-mean length (rows); the edge is the run's CENTRE (first + 2)
TISSUE_FLOOR = 15       # the speckle floor = mean of the first 15 rows of the window (above the consensus)
TISSUE_PCT = 90.0       # the tissue level = p90 of the LOWER half of the window (consensus and below)
TISSUE_KEYS = ("tissue_rms", "tissue_disagreement", "tissue_summary")


def tissue_edge_rows(vol, curve: np.ndarray) -> np.ndarray:
    """The anterior TISSUE edge per (canvas lateral, frame) of one scrub memmap (Lc, Dc, Fc) uint8: within ±TISSUE_HALF
    rows of the consensus, the first row whose TISSUE_RUN-row running mean exceeds the midpoint between the speckle
    floor (mean of the window's first TISSUE_FLOOR rows) and the tissue p90 (lower half of the window). NaN where
    undefined: no consensus, window outside the canvas, cell uncovered (all zero), no contrast, or no crossing.
    Vectorised over frames; one contiguous (Dc, Fc) read per lateral (~0.2 s per volume on CS001)."""
    Lc, Dc, Fc = vol.shape
    W = 2 * TISSUE_HALF + 1
    c = np.rint(np.asarray(curve, float))
    ok = np.isfinite(c)
    c0 = np.where(ok, c, 0).astype(int) - TISSUE_HALF
    ok &= (c0 >= 0) & (c0 + W <= Dc)
    c0 = np.clip(c0, 0, max(0, Dc - W))
    off = np.arange(W)[:, None]
    fidx = np.arange(Fc)[None, :]
    out = np.full((Lc, Fc), np.nan)
    if W > Dc:
        return out
    for l in range(Lc):
        if not ok[l].any():
            continue
        win = np.asarray(vol[l])[c0[l][None, :] + off, fidx].astype(np.float32)    # (W, Fc)
        floor = win[:TISSUE_FLOOR].mean(axis=0)
        p90 = np.percentile(win[TISSUE_HALF:], TISSUE_PCT, axis=0)
        thr = 0.5 * (floor + p90)
        cs = np.cumsum(np.vstack([np.zeros((1, Fc), np.float32), win]), axis=0)
        rm = (cs[TISSUE_RUN:] - cs[:-TISSUE_RUN]) / TISSUE_RUN
        above = rm > thr[None, :]
        good = ok[l] & (win > 0).any(axis=0) & above.any(axis=0) & (p90 > floor + 1.0)
        edge = c0[l] + above.argmax(axis=0) + (TISSUE_RUN - 1) / 2.0
        out[l, good] = edge[good]
    return out


def pairwise_tissue_disagreement(edges: dict) -> tuple[dict, dict]:
    """TISSUE GATE (R4): per scan the mean |tissue edge − the OTHER scans' tissue edges| over every (lateral, frame) cell where both
    are defined, pooled over the other scans (NaN when the scan overlaps no other scan), and the pairwise matrix {cid: {other:
    {mean_abs_px, n}}}. `edges` cid → (Lc, Fc) tissue-edge rows (tissue_edge_rows)."""
    cids = list(edges)
    per_scan: dict = {}; matrix: dict = {c: {} for c in cids}
    for i, c in enumerate(cids):
        e = np.asarray(edges[c], float)
        tot = 0.0; n_tot = 0
        for o in cids:
            if o == c:
                continue
            eo = np.asarray(edges[o], float)
            both = np.isfinite(e) & np.isfinite(eo)
            n = int(both.sum())
            d = float(np.mean(np.abs(e[both] - eo[both]))) if n else float("nan")
            matrix[c][o] = {"mean_abs_px": (d if n else None), "n": n}
            if n:
                tot += d * n; n_tot += n
        per_scan[c] = (tot / n_tot) if n_tot else float("nan")
    return per_scan, matrix


def tissue_gate_decide(edges_before: dict, edges_after: dict, *, thr_scan: float = TISSUE_GATE_SCAN_PX, thr_group: float = TISSUE_GATE_GROUP_PX,
                       max_iter: int | None = None) -> dict:
    """TISSUE GATE (R4): decide which scans' dome moves are HELD. Iteratively: per scan the pairwise tissue disagreement before (pair
    placement) and after (dome move applied, held scans at their before placement); a scan whose after exceeds its before by more than
    thr_scan is held (its after edges become its before edges) and the rest are re-judged, until nothing new is held. Returns
    {held: [cid…], held_for_group, per_scan: {cid: {before, after_initial, after, held, reason (rule 'scan' | 'group')}}, group_before,
    group_after_initial, group_after, ok (group_after ≤ group_before + thr_group), iterations, group_holds, thresholds}. The GROUP bar is
    enforced too: while the mean over scans after exceeds before + thr_group, the scan whose value rose the most is held next."""
    cids = list(edges_before)
    cur = {c: np.asarray(edges_after[c], float) for c in cids}
    before, _mb = pairwise_tissue_disagreement({c: edges_before[c] for c in cids})
    after0, _ma = pairwise_tissue_disagreement(cur)
    held: list = []; reasons: dict = {}
    after = dict(after0)
    n_iter = 0
    limit = int(max_iter) if max_iter is not None else len(cids) + 1
    while n_iter < limit:
        n_iter += 1
        new = [c for c in cids if c not in held and np.isfinite(after.get(c, np.nan)) and np.isfinite(before.get(c, np.nan))
               and after[c] > before[c] + float(thr_scan)]
        if not new:
            break
        # hold the worst offender first (one at a time: a hold changes every other scan's after value)
        worst = max(new, key=lambda c: after[c] - before[c])
        held.append(worst)
        reasons[worst] = {"before_px": float(before[worst]), "after_px": float(after[worst]), "rise_px": float(after[worst] - before[worst]),
                          "threshold_px": float(thr_scan), "iteration": n_iter, "rule": "scan"}
        cur[worst] = np.asarray(edges_before[worst], float)
        after, _ma = pairwise_tissue_disagreement(cur)
    # the GROUP bar: the mean over scans after must stay within thr_group of before — while it does not, the scan whose disagreement
    # rose the most is held next (largest rise first; a scan whose value did not rise is never held for the group), so a delivered
    # result never leaves the scans' tissue agreement worse than the pair placement by more than the bar
    def _group(vals: dict) -> float | None:
        v = [vals[c] for c in cids if np.isfinite(vals.get(c, np.nan))]
        return float(np.mean(v)) if v else None
    gb = _group(before); ga0 = _group(after0); ga_ = _group(after)
    n_group = 0
    while gb is not None and ga_ is not None and ga_ > gb + float(thr_group) and len(held) < len(cids):
        cand = [c for c in cids if c not in held and np.isfinite(after.get(c, np.nan)) and np.isfinite(before.get(c, np.nan)) and after[c] > before[c]]
        if not cand:
            break
        n_iter += 1; n_group += 1
        worst = max(cand, key=lambda c: after[c] - before[c])
        held.append(worst)
        reasons[worst] = {"before_px": float(before[worst]), "after_px": float(after[worst]), "rise_px": float(after[worst] - before[worst]),
                          "threshold_px": float(thr_scan), "iteration": n_iter, "rule": "group",
                          "group_before_px": gb, "group_after_px": ga_, "group_threshold_px": float(thr_group)}
        cur[worst] = np.asarray(edges_before[worst], float)
        after, _ma = pairwise_tissue_disagreement(cur)
        ga_ = _group(after)
    fin = lambda v: (float(v) if v is not None and np.isfinite(v) else None)  # noqa: E731
    return {"held": list(held), "held_for_group": [c for c in held if (reasons.get(c) or {}).get("rule") == "group"],
            "per_scan": {c: {"before_px": fin(before[c]), "after_initial_px": fin(after0[c]), "after_px": fin(after[c]), "held": c in held,
                             "reason": reasons.get(c)} for c in cids},
            "group_before_px": gb, "group_after_initial_px": ga0, "group_after_px": ga_,
            "ok": (gb is None or ga_ is None or ga_ <= gb + float(thr_group)), "iterations": int(n_iter), "group_holds": int(n_group),
            "threshold_scan_px": float(thr_scan), "threshold_group_px": float(thr_group),
            "rule": ("per scan: the mean |tissue edge − the other placed scans' tissue edges| over the cells of their overlaps (the scrub "
                     "tissue-edge rule, measured from the images), before (pair placement) and after (dome move applied); a scan whose "
                     f"after value exceeds its before value by more than {thr_scan} px has its dome move HELD (a_pair / b_pair served, δa = δb "
                     f"= 0) and the rest are re-judged; then, while the group mean after exceeds before + {thr_group} px, the scan whose value "
                     "rose the most is held as well")}


def tissue_edge_metrics(scrub_dir: Path, lines_npz: Path, meta: dict | None = None) -> dict:
    """Per member × stage × canvas lateral: the tissue-edge RMS to the consensus (frames where both are defined) and,
    per stage × lateral, the scan-to-scan DISAGREEMENT (mean over frames of max − min of the members' tissue edges
    where every member's is defined) + overall summaries. Keys: tissue_rms[cid][stage] (Lc, None where undefined),
    tissue_disagreement[stage] (Lc), tissue_summary {disagreement_mean: {stage}, rms: {cid: {stage}}, rule, seconds}.
    Reads scrub/meta.json (members / files) unless given; the memmaps are never loaded whole."""
    t0 = time.time()
    scrub_dir = Path(scrub_dir)
    meta = meta or read_json(scrub_dir / SCRUB_META) or {}
    members = list(meta.get("members") or [])
    z = np.load(Path(lines_npz))
    curve = np.asarray(z["consensus"], float)
    Lc = curve.shape[0]
    edges: dict = {s: {} for s in SCRUB_STAGES}
    rms: dict = {cid: {} for cid in members}
    for s in SCRUB_STAGES:
        for cid in members:
            fn = (meta.get("files") or {}).get(cid, {}).get(s) or scrub_volume_name(s, cid)
            vol = np.load(scrub_dir / fn, mmap_mode="r")
            e = tissue_edge_rows(vol, curve)
            edges[s][cid] = e
            rms[cid][s] = per_lateral_rms(e, curve)
            del vol
    dis: dict = {}
    for s in SCRUB_STAGES:
        if members:
            st = np.stack([edges[s][cid] for cid in members])
            alld = np.isfinite(st).all(axis=0)
            d = np.where(alld, st.max(axis=0) - st.min(axis=0), np.nan)
            n = alld.sum(axis=1)
            v = np.full(Lc, np.nan)
            v[n > 0] = np.nansum(np.where(alld, d, 0.0), axis=1)[n > 0] / n[n > 0]
            dis[s] = v
        else:
            dis[s] = np.full(Lc, np.nan)
    tolist = lambda a: [None if not np.isfinite(x) else float(x) for x in np.asarray(a, float)]
    fin = lambda a: (float(np.sqrt(np.nanmean(np.asarray(a, float) ** 2))) if np.isfinite(a).any() else None)
    # R4: the PAIRWISE disagreement (per scan against the others in their overlaps) — the tissue gate's number
    pair_scan: dict = {}; pair_mat: dict = {}
    for s in SCRUB_STAGES:
        ps, pm = pairwise_tissue_disagreement({cid: edges[s][cid] for cid in members})
        pair_scan[s] = {cid: (float(v) if np.isfinite(v) else None) for cid, v in ps.items()}; pair_mat[s] = pm
    summary = {"disagreement_mean": {s: (float(np.nanmean(dis[s])) if np.isfinite(dis[s]).any() else None) for s in SCRUB_STAGES},
               "rms": {cid: {s: fin(rms[cid][s]) for s in SCRUB_STAGES} for cid in members},
               "pairwise_scan": {cid: {s: pair_scan[s][cid] for s in SCRUB_STAGES} for cid in members},
               "pairwise_group": {s: (float(np.mean([v for v in pair_scan[s].values() if v is not None])) if any(v is not None for v in pair_scan[s].values()) else None)
                                  for s in SCRUB_STAGES},
               "pairwise_rule": "per scan: mean |tissue edge − the other placed scans' tissue edges| over the cells of their overlaps (pooled)",
               "rule": (f"anterior tissue edge per (lateral, frame) = first row within ±{TISSUE_HALF} rows of the consensus where the "
                        f"{TISSUE_RUN}-row running mean exceeds the midpoint between the speckle floor (first {TISSUE_FLOOR} rows of the window) "
                        f"and the tissue p{TISSUE_PCT:.0f}; measured from the uint8 scrub images, not the lines"),
               "seconds": round(time.time() - t0, 2)}
    return {"tissue_rms": {cid: {s: tolist(rms[cid][s]) for s in SCRUB_STAGES} for cid in members},
            "tissue_disagreement": {s: tolist(dis[s]) for s in SCRUB_STAGES},
            "tissue_summary": summary, "tissue_pairwise": pair_mat}


def write_scrub_meta(group: str, ref, members: list, canvas: dict, lines_pairs: dict, lines: dict, curve: np.ndarray,
                     colmask: np.ndarray, colours: dict, channels: dict, vmax: float, files: dict, scrub_dir: Path, *,
                     roster: list | None = None, refused: list | None = None) -> dict:
    """scrub/meta.json: what the sidecar's sagittal endpoint and the panel's SCRUB tab need — canvas, members, colours
    / 3-D channels, per-lateral line RMS to the consensus BEFORE (pair placement) and AFTER (applied) per member
    (all frames where the line and the consensus are both defined — the LINE's residual, reported only), the default lateral (the reference's central lateral) and the covered
    lateral range. Returns the small summary stored in transforms.json (no per-lateral arrays)."""
    Lc, Dc, Fc = canvas["shape"]; l0, z0, f0 = canvas["origin"]
    rms = {cid: {"before": per_lateral_rms(lines_pairs[cid], curve), "after": per_lateral_rms(lines[cid], curve)} for cid in lines}
    lat_cov = np.flatnonzero(np.asarray(colmask, bool).any(axis=1))
    covered = [int(lat_cov[0]), int(lat_cov[-1])] if lat_cov.size else [0, Lc - 1]
    default_lat = int(np.clip(int(round((ref.volume.shape[0] - 1) / 2.0)) - l0, covered[0], covered[1]))
    total = sum(int(files[cid]["scrub"][s]["bytes"]) for cid in lines for s in SCRUB_STAGES)
    summ = {"dir": SCRUB_DIR, "meta": SCRUB_META, "stages": list(SCRUB_STAGES), "laterals": int(Lc), "depth": int(Dc), "frames": int(Fc),
            "covered_range": covered, "default_lateral": default_lat, "bytes": total, "dtype": "uint8", "window": [0.0, float(vmax)],
            "files": {cid: {s: files[cid]["scrub"][s]["file"] for s in SCRUB_STAGES} for cid in lines},
            "rms_frames": "all frames where the member's line and the consensus are both defined",
            "rms_summary": {cid: {s: (float(np.sqrt(np.nanmean(rms[cid][s] ** 2))) if np.isfinite(rms[cid][s]).any() else None)
                                  for s in SCRUB_STAGES} for cid in lines}}
    roster = list(roster or [])
    if not roster:                                    # a caller without a roster: every placed member, reference first
        roster = [{"cid": cid, "role": ("reference" if cid == ref.cid else "contributing"), "is_reference": cid == ref.cid, "ok": True, "placed": True}
                  for cid in lines]
    refused = list(refused or [])
    summ["n_members"] = len(roster); summ["n_contributing"] = sum(1 for r_ in roster if r_.get("placed"))
    # the summary (transforms.json / result.json) names the refused scans; meta.json keeps the FULL records — the summary is
    # merged into meta below (**summ), so the two must not share a key
    summ["refused_ids"] = [{"cid": r_["cid"], "role": r_.get("role")} for r_ in refused]
    meta = {"group": group, "reference": ref.cid, "members": list(lines), "colours": {cid: colours[cid] for cid in lines},
            "spacing_mm": [float(v) for v in ref.spacing],   # lateral, depth, frame (mm) — the panels keep this aspect
            "channels": {cid: channels[cid] for cid in lines}, "canvas": {"origin": [int(v) for v in canvas["origin"]], "shape": [int(v) for v in canvas["shape"]]},
            # ANY NUMBER OF MEMBERS (2026-09-12): every subgroup member with its role; the refused ones with their own sagittal
            "roster": roster, "roles": {r_["cid"]: r_.get("role") for r_ in roster}, "all_members": [r_["cid"] for r_ in roster],
            "n_members": len(roster), "n_contributing": sum(1 for r_ in roster if r_.get("placed")), "refused": refused,
            "members_note": "'members' = the PLACED scans (reference + contributing: the memmaps / RMS); 'roster' = every scan of the subgroup with its role",
            "stage_labels": {"before": "BEFORE — the pair engine's placement (df, dx, a + b·x per frame)",
                             "after": "AFTER — axial changes applied (+ a SMOOTH per-frame shift δa and tilt δb·x: consensus v2 − the member's own dome)"},
            "rms": {cid: {s: rms[cid][s] for s in SCRUB_STAGES} for cid in lines},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **summ}
    try:                                   # the TISSUE-edge metrics (measured from the memmaps just written)
        meta.update(tissue_edge_metrics(scrub_dir, scrub_dir.parent / ALIGNED_LINES, meta))
        summ["tissue_summary"] = meta["tissue_summary"]
    except Exception as e:                 # never let a metric sink the job
        meta["tissue_error"] = f"{type(e).__name__}: {e}"
        summ["tissue_error"] = meta["tissue_error"]
    _write_json(scrub_dir / SCRUB_META, meta)
    return summ


def load_scrub(out_dir: Path) -> dict | None:
    """The scrub data of a finished job: meta.json, one memmap per stage × member (never loaded whole) and the
    placed lines (aligned_lines.npz: line_pairs_<cid> before / line_<cid> after / consensus). None when absent."""
    d = Path(out_dir) / SCRUB_DIR
    meta = read_json(d / SCRUB_META)
    if not meta or not (Path(out_dir) / ALIGNED_LINES).exists():
        return None
    vols: dict = {}
    for stage in SCRUB_STAGES:
        vols[stage] = {}
        for cid in meta["members"]:
            p = d / scrub_volume_name(stage, cid)
            if not p.exists():
                return None
            vols[stage][cid] = np.load(p, mmap_mode="r")
    z = np.load(Path(out_dir) / ALIGNED_LINES)
    try:
        lines = {"after": {cid: np.asarray(z[f"line_{cid}"], float) for cid in meta["members"]},
                 "before": {cid: np.asarray(z[f"line_pairs_{cid}"], float) for cid in meta["members"]}}
        curve = np.asarray(z["consensus"], float)
    except KeyError:
        return None
    own: dict = {}                                   # the refused members' OWN middle sagittals (small; absent on older results)
    for r_ in (meta.get("refused") or []):
        o = r_.get("own") or {}
        p = d / str(o.get("file") or f"own_{r_['cid']}.npy")
        if o and p.exists():
            try:
                own[r_["cid"]] = np.load(p)
            except (OSError, ValueError):
                pass
    return {"meta": meta, "vols": vols, "lines": lines, "curve": curve, "own": own, "mtime": (d / SCRUB_META).stat().st_mtime}


def _hex_rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def _distinct_labels(cids: list) -> dict:
    """Short per-member labels for a legend: the short ids with their common prefix stripped when every remainder is
    non-empty and distinct (case_cs001_os_v1 / _v2 / _v3 → v1 / v2 / v3), else the short ids."""
    short = {cid: _short(cid) for cid in cids}
    vals = list(short.values())
    if len(vals) < 2:
        return short
    pre = os.path.commonprefix(vals)
    pre = pre[:pre.rfind("_") + 1] if "_" in pre else ""
    rem = {cid: s[len(pre):] for cid, s in short.items()}
    if pre and all(rem.values()) and len(set(rem.values())) == len(rem):
        return rem
    return short


def scrub_depth_window(data: dict, lateral: int, stages=SCRUB_STAGES, margin_above: int = 60, margin_below: int = 360) -> tuple[int, int]:
    """The depth rows shown for one canvas lateral: around the lines at that lateral (every member, both stages, the
    consensus) — margin_above rows above the highest, margin_below below the lowest (the corneal thickness) — or,
    where no line exists, the rows any member covers there (± 10), or the whole canvas."""
    meta = data["meta"]; Dc = int(meta["canvas"]["shape"][1])
    vals = [data["curve"][lateral]] + [data["lines"][s][cid][lateral] for s in stages for cid in meta["members"]]
    v = np.concatenate([np.asarray(x, float) for x in vals]); v = v[np.isfinite(v)]
    if v.size:
        zlo = max(0, int(np.floor(v.min())) - margin_above); zhi = min(Dc, int(np.ceil(v.max())) + margin_below)
    else:
        rows = np.zeros(Dc, bool)
        for s in stages:
            for cid in meta["members"]:
                rows |= (np.asarray(data["vols"][s][cid][lateral]) > 0).any(axis=1)
        idx = np.flatnonzero(rows)
        zlo, zhi = (max(0, int(idx[0]) - 10), min(Dc, int(idx[-1]) + 10)) if idx.size else (0, Dc)
    if zhi - zlo < 100:
        zlo, zhi = 0, Dc
    return int(zlo), int(zhi)


SCRUB_MAX_COLS = 4          # member columns per row of the composite (+ 'all' at the end of the last row): ≤ 1800 px wide
SCRUB_STRIP_TT = 56         # title band of the 'not aligned' strip (the strip header + per panel: label, 2 reason lines)


def scrub_layout(n_members: int, n_refused: int, stages: tuple, panel_w: int = 340, panel_h: int = 440,
                 max_cols: int = SCRUB_MAX_COLS, *, only_all: bool = False, with_all: bool = True) -> dict:
    """The composite's geometry for ANY number of members. Reviewer 2026-09-12: ONE row per stage — every BEFORE panel
    on one row and every AFTER panel on the next — with the CONSENSUS ('all', every member blended + the consensus
    curve) as the FIRST (leftmost) column; the panel is as wide as it needs to be and the app scrolls horizontally.
    Refused members follow in a 'not aligned' strip (wrapped, they are not on the canvas). Returns {rows: [{stage |
    'refused', row, cols: ['all' | cid index | ('refused', k)], x0, y0}], W, H, ML, HT, TT, G, BT, panel_w, panel_h,
    strip_y0}."""
    ML, HT, TT, G, BT = 46, 34, 34, 6, 16       # left margin, header (2 lines), per-row title band (stage + panel), gap, bottom
    rows: list = []
    y = HT
    chunks = [list(range(n_members))]                       # one row per stage: every member on it
    for s in stages:
        cols = (["all"] if with_all else []) + list(range(n_members))   # the consensus blend first, then the scans
        if only_all:
            cols = ["all"]
        rows.append({"stage": s, "row": 0, "cols": cols, "x0": ML, "y0": y + TT, "title_y": y + 2, "first": True, "last": True})
        y += TT + panel_h + G
    strip_y0 = None
    if n_refused > 0:
        y += 8
        strip_y0 = y
        rchunks = [list(range(i, min(i + max_cols + 1, n_refused))) for i in range(0, n_refused, max_cols + 1)]
        for k, ch in enumerate(rchunks):
            rows.append({"stage": "refused", "row": k, "cols": [("refused", j) for j in ch], "x0": ML, "y0": y + SCRUB_STRIP_TT,
                         "title_y": y + 2, "first": k == 0, "last": k == len(rchunks) - 1})
            y += SCRUB_STRIP_TT + panel_h + G
    ncol = max(len(r["cols"]) for r in rows) if rows else 1
    W = ML + ncol * (panel_w + G); H = y + BT
    return {"rows": rows, "W": int(W), "H": int(H), "ML": ML, "HT": HT, "TT": TT, "G": G, "BT": BT, "panel_w": panel_w, "panel_h": panel_h,
            "strip_y0": strip_y0, "n_cols": int(ncol), "member_rows_per_stage": len(chunks), "refused_rows": (len(rows) - len(stages) * len(chunks))}


def _wrap_text(draw, text: str, font, width: int, max_lines: int = 2) -> list:
    """Greedy word wrap of `text` into ≤ max_lines lines of ≤ width px (the last line ellipsised)."""
    words = str(text).split()
    lines: list = []; cur = ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= width or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines and draw.textlength(lines[-1] + "…", font=font) > width and " " in lines[-1]:
            lines[-1] = lines[-1].rsplit(" ", 1)[0]
        lines[-1] = lines[-1] + "…"
    return lines


def scrub_columns(meta: dict, panel_w: int = 340, ml: int = 46, gap: int = 6) -> dict:
    """Where each column sits in the composite (the app maps a click to the scan under it, reviewer 2026-09-12):
    {cols: ['all', <cid>…], ml, gap, panel_w, x0: [px…]} — the consensus blend first, then the placed members."""
    cids = list(meta.get("members") or [])
    cols = ["all"] + cids
    return {"cols": cols, "ml": int(ml), "gap": int(gap), "panel_w": int(panel_w),
            "x0": [int(ml + i * (panel_w + gap)) for i in range(len(cols))]}


def render_scrub_png(data: dict, lateral: int, stage: str = "both", panel_w: int = 340, panel_h: int = 440,
                     only: str | None = None) -> bytes:
    """One PNG composite for a canvas lateral: rows = stages (BEFORE = pair placement, AFTER = axial changes applied)
    × columns = every PLACED member in grey + "all" (the members blended additively, each in its own colour);
    x = canvas frames with HIGH frames on the LEFT (the app's sagittal orientation), y = depth rows (the
    scrub_depth_window crop); every panel carries the member's line (thin, its colour) and the consensus (thick white;
    on a member panel only over that member's line span), the title the member and its line RMS to the consensus at
    this lateral. ANY NUMBER OF MEMBERS (2026-09-12, scrub_layout): the member columns wrap into rows of at most
    SCRUB_MAX_COLS (4) per stage with 'all' at the end of the last row (≤ 1800 px wide), and the REFUSED members of the
    subgroup — not placed, no transform — get a panel each in a labelled "NOT ALIGNED — <reason>" strip showing their
    OWN middle sagittal (scrub/own_<cid>.npy, the scan's own lateral L/2, unmoved) so the reviewer sees every scan.
    PIL only (no matplotlib): a composite renders in tens of ms so the panel can scrub at ~6 fps."""
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    meta = data["meta"]; cids = list(meta["members"])
    # `only` = one scan (or "all") shown on its own, large — the click-to-zoom of a single panel (reviewer 2026-09-12)
    zoom_one = None
    if only:
        zoom_one = str(only)
        if zoom_one != "all" and zoom_one not in cids:
            raise ValueError(f"{zoom_one} is not a placed member of this group")
        cids = [] if zoom_one == "all" else [zoom_one]
    Lc, Dc, Fc = [int(v) for v in meta["canvas"]["shape"]]; l0, z0, f0 = [int(v) for v in meta["canvas"]["origin"]]
    if not (0 <= int(lateral) < Lc):
        raise ValueError(f"lateral {lateral} outside 0..{Lc - 1}")
    l = int(lateral)
    stages = tuple(SCRUB_STAGES) if stage == "both" else (stage,)
    for s in stages:
        if s not in SCRUB_STAGES:
            raise ValueError(f"stage must be one of {SCRUB_STAGES} or 'both'")
    zlo, zhi = scrub_depth_window(data, l, stages)
    rows = zhi - zlo
    # DIMENSIONAL ACCURACY: a sagittal cut is (Fc · frame spacing) mm wide and (rows · depth spacing) mm deep — keep
    # that ratio instead of stretching the slice into a fixed box (the panel height follows the width).
    sp_m = [float(v) for v in (meta.get("spacing_mm") or [0.0078, 0.0031, 0.04])]
    dl_m, dz_m, df_m = (sp_m + [1.0, 1.0, 1.0])[:3]
    if dz_m > 0 and df_m > 0 and rows > 0 and Fc > 0:
        panel_h = int(max(120, min(900, round(panel_w * (rows * dz_m) / (Fc * df_m)))))
    sx = panel_w / float(Fc); sy = panel_h / float(rows)
    refused = list(meta.get("refused") or [])
    own = data.get("own") or {}
    if zoom_one:
        refused = []                                   # a single-panel zoom shows just that scan
    lay = scrub_layout(len(cids), len(refused), stages, panel_w, panel_h, only_all=(zoom_one == "all"),
                       with_all=(zoom_one is None or zoom_one == "all"))
    ML, HT, TT, G, BT = lay["ML"], lay["HT"], lay["TT"], lay["G"], lay["BT"]
    W, H = lay["W"], lay["H"]
    img = Image.new("RGB", (W, H), (14, 14, 16))
    draw = ImageDraw.Draw(img)
    try:                                            # DejaVu (matplotlib ships it) has the δ / · glyphs
        font = ImageFont.truetype("DejaVuSans.ttf", 12); font_s = ImageFont.truetype("DejaVuSans.ttf", 10)
    except OSError:
        try:
            font = ImageFont.load_default(size=12); font_s = ImageFont.load_default(size=10)
        except TypeError:                           # Pillow < 10.1
            font = font_s = ImageFont.load_default()
    _pal_r = member_palette(len(cids))
    colours = {cid: _hex_rgb(meta["colours"].get(cid, _pal_r[i])) for i, cid in enumerate(cids)}
    labels = meta.get("stage_labels") or {}
    lab_short = _distinct_labels(cids)
    n_all = int(meta.get("n_members") or (len(cids) + len(refused)))

    def fx(f: float) -> float:                      # canvas frame → panel x (high frames on the left)
        return (Fc - 1 - f + 0.5) * sx

    def fy(z: float) -> float:                      # canvas row → panel y
        return (z - zlo + 0.5) * sy

    def polyline(x0: int, y0: int, series: np.ndarray, colour, width: int, span=None) -> None:
        """Draw z(frame) as connected segments between consecutive finite frames (a gap breaks the line)."""
        pts: list = []
        lo_f, hi_f = (0, Fc - 1) if span is None else span
        for f in range(Fc):
            z = series[f]
            if lo_f <= f <= hi_f and np.isfinite(z) and zlo - 5 <= z <= zhi + 5:
                pts.append((x0 + fx(f), y0 + fy(z)))
            else:
                if len(pts) >= 2:
                    draw.line(pts, fill=colour, width=width)
                pts = []
        if len(pts) >= 2:
            draw.line(pts, fill=colour, width=width)

    curve = data["curve"][l]
    cov_idx = np.flatnonzero(np.isfinite(curve))
    contrib_txt = (f"{len(cids)} of {n_all} scans contribute" + (f"; not aligned: {', '.join(_short(r_['cid']) for r_ in refused)}" if refused else ""))
    draw.text((6, 3), f"{meta.get('group', '')} · canvas lateral {l} / {Lc - 1} (reference lateral {l + l0}) · x = canvas frames, HIGH on the LEFT "
                      f"(as in the app) · y = depth rows {zlo + z0}…{zhi + z0} (reference rows) · {contrib_txt}", fill=(200, 200, 205), font=font)
    draw.text((6, 18), "thin colour = the member's placed line · thick white = consensus v2 (majority own dome) · RMS = that line's RMS to the "
                       "consensus along the frames of this lateral · 'all' = every member blended in ITS OWN colour (additive)",
              fill=(150, 150, 156), font=font)
    last_member_row = [r_ for r_ in lay["rows"] if r_["stage"] != "refused"][-1]
    for row in lay["rows"]:
        s = row["stage"]; y0 = row["y0"]
        if s == "refused":
            # the NOT ALIGNED strip: each refused member's OWN middle sagittal, unmoved — no line RMS, no consensus
            if row["first"]:
                draw.text((ML, row["title_y"]), f"NOT ALIGNED — {len(refused)} of {n_all} scans are not placed on the canvas (their pair with the reference was "
                                                f"refused); shown = each scan's OWN middle sagittal (its own lateral L/2, unmoved), x = its own frames (high on the left)",
                          fill=(255, 120, 120), font=font)
            for jx, (_tag, k) in enumerate(row["cols"]):
                x0 = ML + jx * (panel_w + G)
                r_ = refused[k]; cid = r_["cid"]
                col = _hex_rgb(str(r_.get("colour") or (r_.get("own") or {}).get("colour") or member_palette(len(cids) + k + 1)[len(cids) + k]))
                o = r_.get("own") or {}
                arr = own.get(cid)
                if arr is not None and o:
                    D_o, F_o = arr.shape
                    dw = o.get("depth_window") or [0, D_o]
                    zl, zh = int(max(0, dw[0])), int(min(D_o, dw[1]))
                    if zh - zl < 50:
                        zl, zh = 0, D_o
                    cut = np.asarray(arr)[zl:zh, ::-1]
                    pan = Image.fromarray(np.ascontiguousarray(cut), "L").resize((panel_w, panel_h), Image.BILINEAR).convert("RGB")
                    img.paste(pan, (x0, y0))
                    ln = np.array([np.nan if v is None else float(v) for v in (o.get("line") or [])], float)
                    if ln.size == F_o:
                        sxo = panel_w / float(F_o); syo = panel_h / float(zh - zl)
                        pts: list = []
                        for f in range(F_o):
                            zv = ln[f]
                            if np.isfinite(zv) and zl - 5 <= zv <= zh + 5:
                                pts.append((x0 + (F_o - 1 - f + 0.5) * sxo, y0 + (zv - zl + 0.5) * syo))
                            else:
                                if len(pts) >= 2:
                                    draw.line(pts, fill=col, width=1)
                                pts = []
                        if len(pts) >= 2:
                            draw.line(pts, fill=col, width=1)
                    draw.text((x0 + 4, y0 + panel_h - 14), f"own lateral {o.get('lateral')} · own rows {zl}…{zh}", fill=(170, 170, 176), font=font_s)
                else:
                    draw.rectangle([x0, y0, x0 + panel_w, y0 + panel_h], fill=(26, 26, 30))
                    draw.text((x0 + 8, y0 + 8), "no image (older result — re-run the alignment)", fill=(170, 170, 176), font=font_s)
                draw.text((x0, y0 - SCRUB_STRIP_TT + 16), f"{_short(cid)} · NOT ALIGNED", fill=col, font=font)   # 12 px: rows +16…+28
                for li, txt in enumerate(_wrap_text(draw, str(r_.get("role") or r_.get("reason") or "refused"), font_s, panel_w, 2)):
                    draw.text((x0, y0 - SCRUB_STRIP_TT + 31 + 11 * li), txt, fill=(255, 150, 150), font=font_s)   # 10 px: rows +31…+52 (< the panel at +56)
                draw.rectangle([x0 - 1, y0 - 1, x0 + panel_w, y0 + panel_h], outline=(140, 60, 60), width=1)
            continue
        # depth ticks left of the first column (reference rows every 100)
        for z in range(zlo, zhi):
            if (z + z0) % 100 == 0:
                yy = y0 + fy(z)
                draw.line([(ML - 5, yy), (ML - 1, yy)], fill=(160, 160, 165), width=1)
                draw.text((2, yy - 6), str(z + z0), fill=(160, 160, 165), font=font_s)
        for jx, colspec in enumerate(row["cols"]):
            x0 = ML + jx * (panel_w + G)
            if colspec != "all":
                cid = cids[int(colspec)]
                cut = np.asarray(data["vols"][s][cid][l])[zlo:zhi, ::-1]              # (rows, frames) high frames left
                pan = Image.fromarray(np.ascontiguousarray(cut), "L").resize((panel_w, panel_h), Image.BILINEAR).convert("RGB")
                img.paste(pan, (x0, y0))
                r = meta["rms"][cid][s][l] if l < len(meta["rms"][cid][s]) else None
                rtxt = f"RMS {r:.2f} px" if (r is not None and np.isfinite(r)) else "RMS —"
                title = f"{_short(cid)}{' (ref)' if cid == meta.get('reference') else ''} · {s.upper()} · {rtxt}"
                ln = data["lines"][s][cid][l]
                fin = np.flatnonzero(np.isfinite(ln))
                if fin.size:
                    polyline(x0, y0, curve, (255, 255, 255), 3, span=(int(fin[0]), int(fin[-1])))
                polyline(x0, y0, ln, colours[cid], 1)
                draw.text((x0, y0 - 15), title, fill=colours[cid], font=font)
            else:
                acc = np.zeros((rows, Fc, 3), np.float32)
                gain = min(1.0, 3.0 / max(1, len(cids)))      # with many members the plain sum clips to white
                for cid in cids:
                    u8 = np.asarray(data["vols"][s][cid][l])[zlo:zhi, ::-1].astype(np.float32)
                    cr, cg, cb = colours[cid]
                    for ci, cv in enumerate((cr, cg, cb)):
                        if cv:
                            acc[..., ci] += u8 * (cv / 255.0) * gain
                rgb = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
                pan = Image.fromarray(np.ascontiguousarray(rgb), "RGB").resize((panel_w, panel_h), Image.BILINEAR)
                img.paste(pan, (x0, y0))
                if cov_idx.size:
                    polyline(x0, y0, curve, (255, 255, 255), 3, span=(int(cov_idx[0]), int(cov_idx[-1])))
                for cid in cids:
                    polyline(x0, y0, data["lines"][s][cid][l], colours[cid], 1)
                head = f"all · {s.upper()} · "
                draw.text((x0, y0 - 15), head, fill=(220, 220, 225), font=font)
                xx_ = x0 + draw.textlength(head, font=font)
                for cid in cids:                                  # each name in its own colour
                    t_ = f"{lab_short[cid]}  "
                    draw.text((xx_, y0 - 15), t_, fill=colours[cid], font=font)
                    xx_ += draw.textlength(t_, font=font)
            draw.rectangle([x0 - 1, y0 - 1, x0 + panel_w, y0 + panel_h], outline=(70, 70, 76), width=1)
            if row is last_member_row:              # frame ticks under the bottom member row (canvas frames every 20)
                for f in range(0, Fc, 20):
                    xx = x0 + fx(f)
                    draw.line([(xx, y0 + panel_h + 1), (xx, y0 + panel_h + 4)], fill=(160, 160, 165), width=1)
                    draw.text((xx - 6, y0 + panel_h + 5), str(f), fill=(160, 160, 165), font=font_s)
        if row["first"]:
            lab = labels.get(s, s.upper())
            draw.text((ML, row["title_y"]), lab + (f"   (row {row['row'] + 1} of {lay['member_rows_per_stage']})" if lay["member_rows_per_stage"] > 1 else ""),
                      fill=(235, 200, 90), font=font)
        else:
            draw.text((ML, row["title_y"]), f"{s.upper()} — continued (row {row['row'] + 1} of {lay['member_rows_per_stage']})", fill=(235, 200, 90), font=font)
    bio = BytesIO()
    img.save(bio, "PNG", compress_level=3)
    return bio.getvalue()


def own_sagittal_record(m, scrub_dir: Path, colour: str) -> dict:
    """A REFUSED member's OWN middle sagittal for the 'not aligned' strip (2026-09-12): the scan's volume at its own lateral
    L/2 (unmoved, NOT placed on the canvas), uint8 (D, F) windowed 0 → its own p99 (covered cells ≥ 1), written to
    scrub/own_<cid>.npy; its own served line at that lateral (rows, NaN = none) and the depth window around it."""
    L, D, F = m.volume.shape
    lm = int(round((L - 1) / 2.0))
    cut = np.asarray(m.volume[lm], np.float32)                                  # (D, F)
    pos = np.asarray(m.volume, np.float32)[::4, ::2, ::2]
    pos = pos[pos > 0]
    vmax = float(np.percentile(pos, 99)) if pos.size else float(max(1.0, cut.max()))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = float(max(1.0, cut.max()))
    u8 = _win_uint8(cut, 0.0, vmax)
    u8 = np.where(cut > 0, np.maximum(u8, 1), 0).astype(np.uint8)
    path = scrub_dir / f"own_{m.cid}.npy"
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, np.ascontiguousarray(u8)); os.replace(tmp, path)
    sv = np.where(np.asarray(m.valid, bool)[lm], np.asarray(m.served, float)[lm], np.nan)
    fin = sv[np.isfinite(sv)]
    if fin.size:
        zlo, zhi = max(0, int(np.floor(fin.min())) - 60), min(D, int(np.ceil(fin.max())) + 360)
    else:
        zlo, zhi = 0, D
    if zhi - zlo < 100:
        zlo, zhi = 0, D
    return {"file": path.name, "lateral": lm, "shape": [int(D), int(F)], "window": [0.0, vmax], "colour": colour,
            "line": [None if not np.isfinite(v) else float(v) for v in sv], "depth_window": [int(zlo), int(zhi)],
            "note": "the scan's OWN middle sagittal (its own lateral L/2, unmoved) — this scan is NOT placed on the canvas"}


def apply_transforms(group: str, ref, members: list, transforms: dict, ctx: dict, out_dir: Path, log=print, *,
                     roster: list | None = None, all_members: dict | None = None) -> dict:
    """The APPLY stage: final_transforms → per-member aligned volumes on the (re-)union canvas + aligned_lines.npz →
    aligned.png (fused post-transform montage, final lines dashed, consensus thick white) → aligned_rgb.nii.gz
    rebuilt from the FINAL placement → transforms.json. Returns the result.json['transforms'] summary."""
    t0 = time.time()
    canvas0 = ctx["canvas"]; lines0 = ctx["lines"]; curve0 = ctx["curve"]; colours = ctx["colours"]
    lo, hi = ctx["window"]; v_lo, v_hi = ctx["volume_window"]
    fin = final_transforms(members, transforms, canvas0, lines0, curve0)
    for cid, t in fin.items():
        sz = t["sizes"]
        log(f"    final {cid}: covered {sz['covered_frames']}/{sz['frames']} held {sz['held_frames']} beyond-tilt {sz['profile_beyond_tilt_frames']}  "
            f"line-off-consensus {sz['line_off_consensus_frames']}  "
            f"δa peak {sz['delta_a']['peak']} rms {sz['delta_a']['rms']} (jitter {sz['delta_a_jitter']['rms']}; d2 max {sz['delta_a_d2_max']})  "
            f"δb peak {sz['delta_b']['peak']} rms {sz['delta_b']['rms']} (d2 max {sz['delta_b_d2_max']}) smooth_ok {sz['smooth_ok']}  "
            f"fixed inliers {sz['fixed_inliers']} ({sz['fixed_inlier_source']})  "
            f"tissue a peak {sz['tissue_a']['peak']} rms {sz['tissue_a']['rms']}  dome C−Q {sz['dome_rms_before_px']} → dome part {sz['dome_rms_after_px']} "
            f"(profile part {sz['profile_rms_px']}, whole {sz['residual_rms_after_px']})  "
            f"LINE RMS to consensus {sz['rms_before_px']} → {sz['rms_after_px']} (reported)")
    # the union canvas must hold BOTH placements (the consensus curve lives on canvas0; the final moves may grow it)
    T_final = {cid: {"df": t["df"], "dx": t["dx"], "a": t["a"], "b": t["b"]} for cid, t in fin.items()}
    c1 = union_canvas(members, T_final)
    origin = [min(a, b) for a, b in zip(canvas0["origin"], c1["origin"])]
    upper = [max(o0 + s0, o1 + s1) for o0, s0, o1, s1 in zip(canvas0["origin"], canvas0["shape"], c1["origin"], c1["shape"])]
    canvas = {"origin": origin, "shape": [u - o for u, o in zip(upper, origin)]}
    Lc, Dc, Fc = canvas["shape"]
    if Lc * Dc * Fc > 400_000_000:
        raise RuntimeError(f"union canvas too large ({canvas['shape']}) — a final transform is wild")
    # the consensus curve on the (possibly grown) canvas
    dl, _dz, dfr = (o0 - o for o0, o in zip(canvas0["origin"], origin))
    dz = canvas0["origin"][1] - origin[1]
    L0, F0 = curve0.shape

    def _on_canvas(arr0: np.ndarray, shape: tuple, dl: int, dfr: int, dz: float) -> np.ndarray:
        arr = np.full(shape, np.nan)
        arr[dl:dl + L0, dfr:dfr + F0] = arr0 + dz
        return arr
    curve = _on_canvas(curve0, (Lc, Fc), dl, dfr, dz)
    log(f"  apply: canvas origin {canvas['origin']} shape {canvas['shape']} (pairs canvas {canvas0['origin']} {canvas0['shape']})")
    ssum = np.zeros((Lc, Dc, Fc), np.float32); cnt = np.zeros((Lc, Dc, Fc), np.uint8)
    colmask = np.zeros((Lc, Fc), bool)
    lines: dict = {}; chans: list = []; files: dict = {}
    rms_after: dict = {}
    # the SCRUB data (reviewer ask #4): both placements of every member on THIS canvas as uint8 memmaps
    scrub_dir = out_dir / SCRUB_DIR
    reset_scrub_dir(scrub_dir)
    lines_pairs: dict = {}; colmask_pairs = np.zeros((Lc, Fc), bool)
    chans_d: dict = {}

    def _place_after(m, T: dict, *, subtract_first: dict | None = None) -> dict:
        """Place member m by T as its FINAL placement: the fused accumulators, its half-res channel, aligned_<cid>.nii.gz, the AFTER
        scrub memmap and its placed line. `subtract_first` (a previous final transform) removes that placement from the fused
        accumulators before adding the new one (the tissue gate's hold)."""
        nonlocal ssum, cnt, colmask
        if subtract_first is not None:
            vol0, msk0, _l0, _c0 = place_on_canvas(m, subtract_first, canvas)
            ssum -= np.where(msk0, vol0, 0.0); cnt -= msk0
            del vol0, msk0
        vol, msk, line, col = place_on_canvas(m, T, canvas)
        ssum += np.where(msk, vol, 0.0); cnt += msk
        colmask |= col
        lines[m.cid] = line
        h, hk = _half_res(vol, msk)
        chans_d[m.cid] = np.where(hk, _win_uint8(h, v_lo, v_hi), 0).astype(np.uint8)
        vinfo = write_member_volume(vol, msk, ref.spacing, canvas, out_dir / aligned_volume_name(m.cid))
        s_after = write_scrub_volume(vol, msk, hi, scrub_dir / scrub_volume_name("after", m.cid))
        del vol, msk
        return {"vinfo": vinfo, "scrub_after": s_after}

    def _rms_after_of(m) -> float | None:
        """The LINE's after-RMS measured on the PLACED final line over the covered frames (the same number final_transforms
        reports as line_residual, re-derived from the placement so the montage and the record agree)."""
        t = fin[m.cid]; F = m.volume.shape[2]; line = lines[m.cid]
        good_fc = np.zeros(Fc, bool)
        for f in range(F):
            fc = f + t["df"] - origin[2]
            if 0 <= fc < Fc and t["covered"][f]:
                good_fc[fc] = True
        both = np.isfinite(line) & np.isfinite(curve) & good_fc[None, :]
        return (float(np.sqrt(np.mean((line[both] - curve[both]) ** 2))) if both.any() else None)
    for m in members:
        t1 = time.time()
        pa = _place_after(m, T_final[m.cid])
        vinfo = pa["vinfo"]
        files[m.cid] = {"volume": aligned_volume_name(m.cid), **vinfo}
        # the BEFORE placement (the pair engine's df / dx / a / b only) on the SAME canvas: the scrub tab's top row
        volp, mskp, linep, colp = place_on_canvas(m, transforms[m.cid], canvas)
        lines_pairs[m.cid] = linep; colmask_pairs |= colp
        pinfo = write_member_volume(volp, mskp, ref.spacing, canvas, out_dir / aligned_pairs_volume_name(m.cid))
        s_before = write_scrub_volume(volp, mskp, hi, scrub_dir / scrub_volume_name("before", m.cid))
        del volp, mskp
        files[m.cid]["volume_pairs"] = aligned_pairs_volume_name(m.cid); files[m.cid]["volume_pairs_info"] = pinfo
        files[m.cid]["scrub"] = {"before": s_before, "after": pa["scrub_after"]}
        rms_after[m.cid] = _rms_after_of(m)
        log(f"    moved {m.cid}: {files[m.cid]['volume']} {vinfo} RMS after (placed) {rms_after[m.cid]} {time.time() - t1:.1f}s")
    # TISSUE GATE (2026-09-12, R4): measured on the scrub memmaps just written, BEFORE the final files — a scan whose dome move
    # would raise its tissue disagreement with the other placed scans by more than TISSUE_GATE_SCAN_PX has the move HELD
    gate: dict | None = None
    try:
        t1 = time.time()
        edges: dict = {s: {} for s in SCRUB_STAGES}
        for s in SCRUB_STAGES:
            for m in members:
                vol_mm = np.load(scrub_dir / scrub_volume_name(s, m.cid), mmap_mode="r")
                edges[s][m.cid] = tissue_edge_rows(vol_mm, curve)
                del vol_mm
        gate = tissue_gate_decide(edges["before"], edges["after"])
        gate["seconds"] = round(time.time() - t1, 2)
        for cid in gate["held"]:
            m = next(mm_ for mm_ in members if mm_.cid == cid)
            t = fin[cid]; rsn = gate["per_scan"][cid]["reason"]
            log(f"    TISSUE GATE: holding the dome move of {cid} — its tissue disagreement with the other scans would rise "
                f"{rsn['before_px']:.2f} → {rsn['after_px']:.2f} px "
                + (f"(> +{rsn['threshold_px']} px)" if rsn.get("rule") == "scan" else
                   f"(the group mean {rsn.get('group_before_px', float('nan')):.2f} → {rsn.get('group_after_px', float('nan')):.2f} px was beyond +{rsn.get('group_threshold_px')} px)")
                + "; serving a_pair / b_pair only")
            prev_T = dict(T_final[cid])
            cov = np.asarray(t["covered"], bool)
            zero = np.where(cov, 0.0, np.nan)
            t["a"] = np.asarray(t["a_pair"], float).copy(); t["b"] = np.asarray(t["b_pair"], float).copy()
            t["delta_a"] = zero.copy(); t["delta_b"] = zero.copy(); t["delta_a_smooth"] = zero.copy()
            t["line_residual"] = np.asarray(t["rms_before"], float).copy()
            t["line_off_consensus"] = cov & (np.asarray(t["rms_before"], float) > LINE_OFF_CONSENSUS_PX)
            sz = t["sizes"]
            z_ = _sizes(zero, cov)
            sz.update({"delta_a": z_, "delta_b": dict(z_), "delta_a_smooth": dict(z_), "delta_a_jitter": _sizes(np.asarray(t["delta_a_raw"], float) - 0.0, cov),
                       "delta_b_jitter": _sizes(np.asarray(t["delta_b_raw"], float) - 0.0, cov),
                       "delta_a_d2_max": 0.0, "delta_b_d2_max": 0.0, "smooth_ok": True,
                       "rms_after_px": sz.get("rms_before_px"), "rms_after_all_covered_px": sz.get("rms_before_px"),
                       "line_residual_rms_px": sz.get("rms_before_px"), "line_residual": _sizes(np.asarray(t["rms_before"], float), cov),
                       "line_off_consensus_frames": int(t["line_off_consensus"].sum()),
                       "dome_rms_after_px": sz.get("dome_rms_before_px"), "residual_rms_after_px": sz.get("dome_rms_before_px"),
                       "after_rms_ok": (bool(sz["dome_rms_before_px"] <= AFTER_RMS_BAR_PX) if sz.get("dome_rms_before_px") is not None else None),
                       "dome_move_held": True, "dome_move_held_by_tissue": dict(rsn),
                       "hold_note": "dome move HELD by the tissue gate: the pair transform is served as final (δa = δb = 0)"})
            T_final[cid] = {"df": t["df"], "dx": t["dx"], "a": t["a"], "b": t["b"]}
            pa = _place_after(m, T_final[cid], subtract_first=prev_T)
            files[cid].update(pa["vinfo"]); files[cid]["scrub"]["after"] = pa["scrub_after"]
            rms_after[cid] = _rms_after_of(m)
        for cid in fin:
            fin[cid]["sizes"].setdefault("dome_move_held", False)
        log(f"    tissue gate: group disagreement before {gate['group_before_px']} → after {gate['group_after_initial_px']} (initial) → "
            f"{gate['group_after_px']} px (held {gate['held'] or 'none'}; ok {gate['ok']}) "
            f"per scan {json.dumps({c: [round(v['before_px'], 2) if v['before_px'] is not None else None, round(v['after_px'], 2) if v['after_px'] is not None else None] for c, v in gate['per_scan'].items()})} {gate['seconds']}s")
    except Exception as e:  # noqa: BLE001 — the gate never sinks the apply; its absence is reported
        gate = {"error": f"{type(e).__name__}: {e}", "held": [], "ok": None}
        log(f"    tissue gate FAILED: {e}\n{traceback.format_exc()}")
    chans = [(m.cid, chans_d[m.cid]) for m in members]
    fused = np.where(cnt > 0, ssum / np.maximum(cnt, 1), 0.0).astype(np.float32)
    del ssum
    np.savez_compressed(out_dir / ALIGNED_LINES, members=np.asarray(list(lines)), origin=np.asarray(origin), shape=np.asarray(canvas["shape"]),
                        consensus=curve, **{f"line_{cid}": ln for cid, ln in lines.items()},
                        **{f"a_{cid}": fin[cid]["a"] for cid in lines}, **{f"b_{cid}": fin[cid]["b"] for cid in lines},
                        **{f"dx_{cid}": fin[cid]["dx"] for cid in lines}, **{f"df_{cid}": np.asarray(fin[cid]["df"]) for cid in lines},
                        **{f"line_pairs_{cid}": ln for cid, ln in lines_pairs.items()},
                        **{f"a_pair_{cid}": fin[cid]["a_pair"] for cid in lines}, **{f"b_pair_{cid}": fin[cid]["b_pair"] for cid in lines},
                        **{f"dome_{cid}": _on_canvas(fin[cid]["dome_curve"], curve.shape, dl, dfr, dz) for cid in lines},
                        **{f"line_residual_{cid}": fin[cid]["line_residual"] for cid in lines})
    # ANY NUMBER OF MEMBERS (2026-09-12): the refused members' OWN middle sagittals for the 'not aligned' strip
    refused_recs: list = []
    placed_ids = {m.cid for m in members}
    for k, r_ in enumerate(roster or []):
        if r_.get("placed") or r_["cid"] in placed_ids:
            continue
        rec = dict(r_)
        col = member_palette(len(members) + len(refused_recs) + 1)[len(members) + len(refused_recs)]
        mm_ = (all_members or {}).get(r_["cid"])
        if mm_ is not None:
            try:
                rec["own"] = own_sagittal_record(mm_, scrub_dir, col)
            except Exception as e:  # noqa: BLE001 — a missing thumbnail never sinks the job
                rec["own"] = None; rec["own_error"] = f"{type(e).__name__}: {e}"
        else:
            rec["own"] = None
        rec["colour"] = col
        refused_recs.append(rec)
    scrub = write_scrub_meta(group, ref, members, canvas, lines_pairs, lines, curve, colmask | colmask_pairs, colours,
                             {m.cid: CHANNEL_NAMES[i % 3] for i, m in enumerate(members)}, hi, files, scrub_dir,
                             roster=roster, refused=refused_recs)
    log(f"    scrub: {scrub['laterals']} laterals, covered {scrub['covered_range']}, default {scrub['default_lateral']}, "
        f"{scrub['bytes'] / 1e6:.0f} MB of memmaps + {ALIGNED_LINES} line_pairs; roster {len(roster or [])} members, "
        f"{len(members)} placed, {len(refused_recs)} refused (own sagittals {[r_['cid'] for r_ in refused_recs if r_.get('own')]})")
    rb = ctx.get("rms_before") or {}
    title = (f"{group}: APPLIED axial changes (consensus v2) — fused = masked mean of {len(members)} members moved by their FINAL "
             f"rigid transform (pair engine df / dx / a / b + a SMOOTH per-frame δa shift + δb·x tilt = consensus − the member's own dome); dashed = each member's FINAL line; "
             f"thick white = consensus v2 (majority own dome, axial-checked)\nLINE RMS to consensus before → after: " +
             "  ".join(f"{_short(c)} {rb.get(c, float('nan')):.1f} → {v:.2f} px" if v is not None else f"{_short(c)} —" for c, v in rms_after.items()) +
             "   (the line's residual is reported, never applied: line error must not move tissue)")
    info = render_consensus(fused, cnt, lines, curve, colmask, canvas, colours, hi, out_dir / ALIGNED_PNG, title,
                            curve_label="consensus v2 (majority own dome, axial-checked)", spacing_mm=ref.spacing)
    del fused, cnt
    vinfo = write_rgb_volume(chans, ref.spacing, out_dir / VOLUME_NAME, colours=colours)
    log(f"    {ALIGNED_PNG} frames {info['frames']} laterals {info['laterals']}; {VOLUME_NAME} (final) {vinfo}")
    per_member = []
    tj_members = []
    for m in members:
        t = fin[m.cid]; sz = dict(t["sizes"]); sz["rms_after_placed_px"] = rms_after[m.cid]
        per_member.append({"cid": m.cid, "is_reference": m.cid == ref.cid, "df": t["df"], **sz,
                           "profile_beyond_tilt": [int(f) for f in np.flatnonzero(t["profile_beyond_tilt"])],
                           "line_off_consensus": [int(f) for f in np.flatnonzero(t["line_off_consensus"])],
                           "files": files[m.cid]})
        tj_members.append({"cid": m.cid, "is_reference": m.cid == ref.cid, "df": t["df"], "dx": t["dx"], "a_final": t["a"], "b_final": t["b"],
                           "a_pair": t["a_pair"], "b_pair": t["b_pair"], "delta_a": t["delta_a"], "delta_b": t["delta_b"],
                           "delta_a_raw": t["delta_a_raw"], "delta_b_raw": t["delta_b_raw"],
                           "delta_a_smooth": t["delta_a_smooth"], "covered": t["covered"], "residual_rms": t["residual_rms"],
                           "dome_part_rms": t["dome_part_rms"], "profile": t["profile"], "fixed_inliers": t["fixed_inliers"],
                           "rms_before": t["rms_before"], "dome_rms_before": t["dome_rms_before"], "line_residual": t["line_residual"],
                           "inliers": t["inliers"], "n_points": t["n_points"],
                           "profile_beyond_tilt": t["profile_beyond_tilt"], "line_off_consensus": t["line_off_consensus"],
                           "dome_coef": t["dome_coef"], "dome_model": {"form": "z = c0 + c1*u + c2*u^2, u = (pairs-canvas frame - fc) / scale",
                                                                       "fc": t["dome_fc"], "scale": t["dome_scale"], "canvas": "pairs_canvas"},
                           "flags": {"profile_beyond_tilt": [int(f) for f in np.flatnonzero(t["profile_beyond_tilt"])],
                                     "line_off_consensus": [int(f) for f in np.flatnonzero(t["line_off_consensus"])]},
                           "sizes": sz, "files": files[m.cid]})
    note = ("per-frame moves are RIGID (a depth shift + a tilt across the B-scan; never a per-column warp) and SMOOTH along frames: "
            "δa / δb are the shift + tilt of consensus − the member's own dome (smooth-to-smooth), never of the line — fitted on one "
            "FIXED inlier set of laterals and Savitzky-Golay smoothed along frames (the applied field must meet |d2 δa| ≤ "
            f"{SMOOTH_D2_A_PX} px, |d2 δb| ≤ {SMOOTH_D2_B_PX} px); line error is reported (line_residual / line_off_consensus), never "
            "applied; the consensus is v2 (majority own dome, axial-checked)")
    tj = {"group": group, "reference": ref.cid, "note": note, "provisional": False, "consensus_version": gc_.CONSENSUS_VERSION, "rigid": True,
          "tissue_gate": gate,
          "convention": {"a": "total axial move in corrected rows (moving row + a + b·x = reference row)", "b": "tilt px at x = ±1 (half-span)",
                         "x": "(l_m - (L-1)/2) / ((L-1)/2)", "dx": "per-frame lateral shift (moving lateral + dx = reference lateral)",
                         "df": "moving frame + df = reference frame", "canvas": "reference index = canvas index + origin"},
          "profile_beyond_tilt_px": PROFILE_BEYOND_TILT_PX, "line_off_consensus_px": LINE_OFF_CONSENSUS_PX, "after_rms_bar_px": AFTER_RMS_BAR_PX,
          "after_rms_bar_of": ("dome_rms_after_px = the DOME part of (consensus − the member's dome) beyond the shift + tilt, i.e. with the "
                               "frame-independent lateral profile removed (profile_rms_px, reported separately: a rigid per-frame move cannot "
                               "remove it and it is not a dome error); 1.0 px because with three replicates each own dome carries its own "
                               "error and the across-lateral variation of that dome difference sits at 0.5-1 px for honest replicates"),
          "smooth_bar_d2_a_px": SMOOTH_D2_A_PX, "smooth_bar_d2_b_px": SMOOTH_D2_B_PX,
          "smooth_bar_of": "max |second difference along frames| of the APPLIED δa / δb over the covered frames",
          "apply_revision": gc_.CONSENSUS_REVISION,
          "canvas": canvas, "pairs_canvas": canvas0, "members": tj_members, "lines": ALIGNED_LINES, "png": ALIGNED_PNG, "png_info": info,
          "volume": VOLUME_NAME, "volume_pairs": VOLUME_PAIRS_NAME, "volume_info": vinfo, "scrub": scrub, "seconds": time.time() - t0}
    _write_json(out_dir / TRANSFORMS_JSON, tj)
    summ = {k: v for k, v in tj.items() if k != "members"}
    summ["members"] = per_member
    summ["json"] = TRANSFORMS_JSON
    summ["after_rms_ok_all"] = all((pm["after_rms_ok"] is not False) for pm in per_member)
    summ["smooth_ok_all"] = all(bool(pm.get("smooth_ok", True)) for pm in per_member)
    summ["held_scans"] = list((gate or {}).get("held") or [])
    return summ


# ── the member record ─────────────────────────────────────────────────────────────────────────────────────────
def member_record(ga, cid: str, r, reference: str, non_contributing: dict) -> dict:
    q = r.quality if isinstance(r.quality, dict) else {}
    msp = q.get("match_speckle") or {}
    m = np.asarray(r.measured, bool)
    part = np.array([r.frame_partner(f) is not None for f in range(r.n_frames)])
    dxa = np.asarray(r.dx_applied, float)
    b = np.asarray(r.b, float)
    sel = part & m
    dx_med = float(np.nanmedian(dxa[sel])) if sel.any() and np.isfinite(dxa[sel]).any() else float("nan")
    dx_rng = ([float(np.nanmin(dxa[part])), float(np.nanmax(dxa[part]))] if part.any() and np.isfinite(dxa[part]).any() else [None, None])
    tilt_med = float(np.nanmedian(b[m])) if m.any() and np.isfinite(b[m]).any() else float("nan")
    tilt_abs = float(np.nanmedian(np.abs(b[m]))) if m.any() and np.isfinite(b[m]).any() else float("nan")
    rej = [fl for fl in r.flags if fl in ga.REJECT_FLAGS]
    rel_sp = msp.get("relative_match", float("nan"))
    return {"cid": cid, "is_reference": False, "reference": reference, "df": int(r.df),
            "dx_median": dx_med, "dx_range": dx_rng, "dx_segments": [(int(f0), int(f1), float(dx)) for f0, f1, dx in r.dx_segments],
            "tilt_median_px": tilt_med, "tilt_abs_median_px": tilt_abs,
            "a_median_px": (float(np.nanmedian(np.asarray(r.a, float)[m])) if m.any() else float("nan")),
            "lateral_scale": float(r.lateral_scale), "rel_struct": float(r.relative_match),
            "rel_speckle": (float(rel_sp) if rel_sp is not None else float("nan")),
            "matched_struct": float(r.matched_frac_0_5), "matched_speckle": msp.get("matched_frac_0.5"),
            "coverage": float(r.coverage), "measured_frames": int(m.sum()), "overlap_frames": int(part.sum()),
            "ok": bool(r.ok), "flags": list(r.flags), "reject_flags": rej,
            "non_contributing": non_contributing.get(cid), "pose_angle_deg": float(r.pose_angle_deg),
            "ncc_coarse": float(r.ncc_coarse), "seconds": float(r.timings.get("total", float("nan"))),
            # PARTIAL OVERLAP (2026-09-12): the laterals the two scans share under the served shift, as a fraction of the moving laterals
            "overlap_laterals": (float(r.overlap_laterals) if np.isfinite(float(getattr(r, "overlap_laterals", float("nan")))) else None),
            "overlap_fraction": (float(r.overlap_fraction) if np.isfinite(float(getattr(r, "overlap_fraction", float("nan")))) else None),
            "overlap": q.get("overlap"),
            "overlay": overlay_name(cid), "summary": r.summary()}


def member_role(rec: dict) -> str:
    """The roster role of a member record: 'reference' | 'contributing' | 'contributing (via <cid>)' (TRANSITIVE PLACEMENT) |
    'refused: <reject flags> (<reason>)' — the reason from group_align.overlap_reason (R2, 2026-09-12: it names the correspondence
    BEYOND the bar when the record holds one — 'best correspondence at ≈ −449 laterals (12% overlap, below the 19% bar)' — and the
    plain 'offset ≈ dx laterals, overlap f%' otherwise)."""
    if rec.get("is_reference"):
        return "reference"
    if rec.get("ok"):
        return "contributing"
    if rec.get("via"):
        return f"contributing (via {rec['via']})"
    import group_align as ga
    rej = list(rec.get("reject_flags") or []) or ["refused"]
    if rec.get("non_contributing") and str(rec["non_contributing"]).startswith("pose_beyond_frame_rigid"):
        rej = ["pose_beyond_frame_rigid"]
    reason = ga.overlap_reason(rec.get("overlap"), rej, dx_median=rec.get("dx_median"), overlap_fraction=rec.get("overlap_fraction"))
    return f"refused: {', '.join(rej)}" + (f" ({reason})" if reason else "")


def build_roster(recs: list) -> list:
    """EVERY subgroup member with its role (member_role) — the scrub meta / result.json 'roster' (2026-09-12). A member placed
    TRANSITIVELY (rec['via']) is placed with role 'contributing (via <cid>)' and carries its route."""
    out = []
    for rec in recs:
        via = rec.get("via")
        out.append({"cid": rec["cid"], "role": member_role(rec), "is_reference": bool(rec.get("is_reference")), "ok": bool(rec.get("ok")),
                    "placed": bool(rec.get("is_reference") or rec.get("ok") or via), "reject_flags": list(rec.get("reject_flags") or []),
                    "reason": rec.get("non_contributing"), "df": rec.get("df"), "dx_median": rec.get("dx_median"),
                    "overlap_laterals": rec.get("overlap_laterals"), "overlap_fraction": rec.get("overlap_fraction"),
                    "rel_struct": rec.get("rel_struct"), "ncc_coarse": rec.get("ncc_coarse"),
                    "via": via, "route": rec.get("route")})
    return out


def transitive_placement(refused: list, contributing: list, transforms: dict, provider, L: int, *, min_rel: float, log=print,
                         progress=None, roundtrip_dx: float = TRANSITIVE_ROUNDTRIP_DX, roundtrip_px: float = TRANSITIVE_ROUNDTRIP_PX) -> dict:
    """TRANSITIVE PLACEMENT (2026-09-12): every member X whose direct pair with the reference was refused is registered to each
    CONTRIBUTING member C (`provider(C, X)` → the pair X → C: {ok, df, dx, a, b, rel, ncc_coarse, flags}; the job's provider
    caches). A route is ADMISSIBLE when that pair is ok, its relative match ≥ min_rel, it is not at the bar (neither
    'dx_at_search_edge' nor the judged 'near_search_edge'), and the pair engine's own ROUND TRIP — X → C composed with the
    independently registered reverse pair C → X (`provider(X, C)`) — closes with df exact, dx RMS ≤ roundtrip_dx laterals and
    a RMS ≤ roundtrip_px px (group_consensus.roundtrip_error). The member is then placed by the COMPOSED transform X → C → R
    (compose_transforms; C's served transform) with role 'contributing (via C)'; several routes → the best relative match.
    Returns cid → {via, transform, rel, ncc_coarse, roundtrip, df, dx_median, overlap_fraction, routes}."""
    out: dict = {}
    for X in refused:
        routes: list = []
        for C in contributing:
            if C not in transforms:
                continue
            if progress:
                progress(f"transitive placement: {_short(X)} via {_short(C)}")
            rec: dict = {"via": C, "admissible": False, "why": None}
            r = provider(C, X)
            if r is None:
                rec["why"] = "no pair"; routes.append(rec); continue
            rec.update(ok=bool(r["ok"]), rel=r.get("rel"), ncc_coarse=r.get("ncc_coarse"), flags=list(r.get("flags") or []),
                       df=int(r["df"]), overlap_fraction=r.get("overlap_fraction"), reason=r.get("reason"))
            if not r["ok"]:
                rec["why"] = "pair refused" + (f": {r.get('reason')}" if r.get("reason") else ""); routes.append(rec); continue
            rel = r.get("rel")
            if rel is None or not np.isfinite(float(rel)) or float(rel) < float(min_rel):
                rec["why"] = f"relative match {rel} below {min_rel}"; routes.append(rec); continue
            if "dx_at_search_edge" in rec["flags"] or "near_search_edge" in rec["flags"]:
                rec["why"] = "at the search edge"; routes.append(rec); continue
            T_XC = {"df": int(r["df"]), "dx": np.asarray(r["dx"], float), "a": np.asarray(r["a"], float), "b": np.asarray(r["b"], float)}
            rb = provider(X, C)
            if rb is None or not rb["ok"]:
                rec["why"] = "reverse pair refused (no round trip)" + (f": {rb.get('reason')}" if rb and rb.get("reason") else ""); routes.append(rec); continue
            T_CX = {"df": int(rb["df"]), "dx": np.asarray(rb["dx"], float), "a": np.asarray(rb["a"], float), "b": np.asarray(rb["b"], float)}
            rt = gc_.roundtrip_error(T_XC, T_CX, int(L))
            rec["roundtrip"] = rt
            if rt["df_error"] != 0 or rt["dx_rms"] is None or rt["a_rms_px"] is None or rt["dx_rms"] > float(roundtrip_dx) or rt["a_rms_px"] > float(roundtrip_px):
                rec["why"] = (f"round trip does not close (df error {rt['df_error']}, dx {rt['dx_rms']} laterals, a {rt['a_rms_px']} px; "
                              f"bounds 0 / {roundtrip_dx} / {roundtrip_px})"); routes.append(rec); continue
            T_XR = gc_.compose_transforms(T_XC, transforms[C], int(L))
            okf = np.isfinite(T_XR["dx"]) & np.isfinite(T_XR["a"]) & np.isfinite(T_XR["b"])
            if not okf.any():
                rec["why"] = "the composition covers no frame"; routes.append(rec); continue
            rec.update(admissible=True, transform=T_XR, df_composed=int(T_XR["df"]), dx_median_composed=float(np.nanmedian(T_XR["dx"][okf])),
                       frames_composed=int(okf.sum()))
            routes.append(rec)
        adm = [rt for rt in routes if rt["admissible"]]
        if adm:
            best = max(adm, key=lambda rt: float(rt["rel"]))
            out[X] = {"via": best["via"], "transform": best["transform"], "rel": float(best["rel"]), "ncc_coarse": best.get("ncc_coarse"),
                      "roundtrip": best["roundtrip"], "df": int(best["df_composed"]), "dx_median": float(best["dx_median_composed"]),
                      "df_via": int(best["df"]), "frames": int(best["frames_composed"]), "overlap_fraction_via": best.get("overlap_fraction"),
                      "routes": [{k: v for k, v in rt.items() if k != "transform"} for rt in routes]}
            log(f"  TRANSITIVE {X}: placed via {best['via']} (rel {best['rel']:.3f}, df {best['df']:+d} ∘ → composed df {out[X]['df']:+d} "
                f"dx {out[X]['dx_median']:+.1f}, round trip df {best['roundtrip']['df_error']} dx {best['roundtrip']['dx_rms']:.2f} a {best['roundtrip']['a_rms_px']:.2f}); "
                f"routes {[(rt['via'], 'ok' if rt['admissible'] else rt['why']) for rt in routes]}")
        else:
            log(f"  TRANSITIVE {X}: no admissible route — {[(rt['via'], rt['why']) for rt in routes]}")
            out[X] = {"via": None, "routes": [{k: v for k, v in rt.items() if k != "transform"} for rt in routes]}
    return out


# ── the job ───────────────────────────────────────────────────────────────────────────────────────────────────
def run_group_job(group: str, cases_root: Path, out_dir: Path, *, transitivity: bool = False, workers: int = 3,
                  reference: str | None = None, log=print, no_sensitivity: bool = False,
                  members: list[str] | None = None) -> dict:
    import group_align as ga
    out_dir.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if members:
        # an EXPLICIT member list (the sidecar's subgroup-aware resolution, api_server._group_members): the group id may
        # be a subgroup id (<patient>_<eye>_s<k>) that ga.group_members (patient+eye only) does not know
        members = [str(c) for c in members]
        missing = [c for c in members if not (cases_root / c / "manifest.json").exists()]
        if missing:
            raise FileNotFoundError(f"members not found under {cases_root}: {missing}")
        gk = ga.group_key(ga.read_manifest(cases_root / members[0]))
        key = {"patient": (gk[1] if gk else None), "eye": (gk[2] if gk else None)}
    else:
        members, key = ga.group_members(group, cases_root)
    if not members:
        raise FileNotFoundError(f"no members for group {group!r} under {cases_root}")
    prog = Progress(out_dir, group, members)
    log(f"=== group {group} ({key}) members {members}  engine {engine_md5()}  transitivity={transitivity}")
    loaded = []
    for i, cid in enumerate(members):
        prog.update(phase=f"loading {cid}", members_done=i)
        t0 = time.time()
        m = ga.load_member(cases_root / cid, posterior="auto", scar=True, workers=workers, write_cache=False)
        loaded.append(m)
        log(f"  loaded {cid}: shape {m.shape} spacing {np.round(np.asarray(m.spacing) * 1000, 4).tolist()} um "
            f"served={m.served_source} move={m.move_source} pad {m.canvas_pad} valid {m.valid_area} {time.time() - t0:.1f}s")
    prog.update(phase="registering", members_done=len(members))
    # per-pair progress: register_group is monolithic, so count its register_pair / coarse_register calls (this is a
    # dedicated subprocess: the patch is local to it)
    n_ordered = len(members) * (len(members) - 1)
    counters = {"coarse": 0, "pairs": 0}
    _orig_pair = ga.register_pair; _orig_coarse = ga.coarse_register

    def _pair(*a, **kw):
        prog.update(phase=f"registering pair {min(counters['pairs'] + 1, len(members) - 1)}/{len(members) - 1}", pairs_done=counters["pairs"])
        res = _orig_pair(*a, **kw)
        counters["pairs"] += 1
        prog.update(pairs_done=min(counters["pairs"], len(members) - 1))
        return res

    def _coarse(*a, **kw):
        counters["coarse"] += 1
        prog.update(phase=f"pose matrix {min(counters['coarse'], n_ordered)}/{n_ordered}")
        return _orig_coarse(*a, **kw)
    ga.register_pair = _pair; ga.coarse_register = _coarse
    try:
        t0 = time.time()
        g = ga.register_group(loaded, reference=reference, params=None, transitivity=bool(transitivity), max_triples=3, quality=True)
        dt_reg = time.time() - t0
    finally:
        ga.register_pair = _orig_pair; ga.coarse_register = _orig_coarse
    log(f"  reference {g.reference} rule {json.dumps(jsonable(g.reference_rule))} grid {json.dumps(jsonable(g.lateral_grid))} {dt_reg:.0f}s")
    log("  pose matrix: " + ", ".join(f"{m_}->{r_}: {v:.1f}" for (m_, r_), v in g.pose.items()))
    recs: list[dict] = []
    ref_m = next(m for m in loaded if m.cid == g.reference)
    recs.append({"cid": g.reference, "is_reference": True, "reference": g.reference, "df": 0, "dx_median": 0.0, "dx_range": [0.0, 0.0],
                 "tilt_median_px": 0.0, "tilt_abs_median_px": 0.0, "lateral_scale": 1.0, "rel_struct": None, "rel_speckle": None,
                 "matched_struct": None, "coverage": 1.0, "measured_frames": None, "overlap_frames": int(ref_m.shape[2]),
                 "ok": True, "flags": [], "reject_flags": [], "non_contributing": None, "pose_angle_deg": None, "seconds": None,
                 "overlay": None, "shape": list(ref_m.shape), "valid_area": int(ref_m.valid_area), "served_source": ref_m.served_source})
    for cid in members:
        if cid == g.reference:
            continue
        r = g.pairs[cid]
        rec = member_record(ga, cid, r, g.reference, g.non_contributing)
        mm = next(m for m in loaded if m.cid == cid)
        rec.update(shape=list(mm.shape), valid_area=int(mm.valid_area), served_source=mm.served_source)
        recs.append(rec)
        log(f"  PAIR {cid} -> {g.reference}: df {rec['df']:+d} dx {rec['dx_median']:+.1f} {rec['dx_range']} tilt {rec['tilt_median_px']:+.1f} "
            f"scale {rec['lateral_scale']:.4f} rel_struct {rec['rel_struct']:.3f} rel_speckle {rec['rel_speckle']:.3f} cov {rec['coverage']:.3f} "
            f"overlap {rec['overlap_fraction'] if rec['overlap_fraction'] is None else round(rec['overlap_fraction'], 3)} "
            f"meas {rec['measured_frames']}/{rec['overlap_frames']} ok {rec['ok']} reject {rec['reject_flags']} {rec['seconds']:.0f}s")
    # overlays from the ENGINE's transform on the group's common lateral grid (the grid register_group used)
    prog.update(phase="rendering overlays")
    gm, _grid = ga.common_lateral_grid(loaded, g.reference, ga.PairParams().lateral_scale_tol)
    gmd = {m.cid: m for m in gm}
    n_ov = 0
    for rec in recs:
        if rec["is_reference"]:
            continue
        cid = rec["cid"]; r = g.pairs[cid]
        t0 = time.time()
        try:
            ttl = (f"{group}: {cid} → {g.reference}   df {rec['df']:+d}  dx {rec['dx_median']:+.1f}  tilt {rec['tilt_median_px']:+.1f} px  "
                   f"scale {rec['lateral_scale']:.4f}  structure {rec['rel_struct']:.2f}×  speckle {rec['rel_speckle']:.2f}×  cov {rec['coverage']:.2f}  "
                   f"ok {rec['ok']}  {('REJECT ' + ','.join(rec['reject_flags'])) if rec['reject_flags'] else ''}")
            rec["overlay_info"] = render_overlay(gmd[g.reference], gmd[cid], r, out_dir / overlay_name(cid), ttl)
            log(f"  overlay {cid}: {rec['overlay_info']['frames']} / {rec['overlay_info']['laterals']} {time.time() - t0:.1f}s")
        except Exception as e:  # noqa: BLE001
            rec["overlay"] = None; rec["overlay_error"] = f"{type(e).__name__}: {e}"
            log(f"  overlay {cid} FAILED: {e}\n{traceback.format_exc()}")
        n_ov += 1
        prog.update(overlays_done=n_ov)
    # the pair cache (the rule's pairs now; the sensitivity pairs below) — engine md5 + PairParams hash keyed
    pparams = ga.PairParams()
    cache = gc_.PairCache(out_dir / PAIRS_DIR, engine_md5(), gc_.params_hash(pparams.as_dict()), ctor=ga.PairResult)
    for cid_, r_ in g.pairs.items():
        try:
            cache.put(r_)
        except Exception as e:  # noqa: BLE001
            log(f"  pair cache: could not store {cid_} -> {g.reference}: {e}")
    # the PAIR PROVIDER (the transitive placement and the reference-sensitivity anchors): register_pair on the common grid's
    # bands, EXACTLY the rules of the direct pairs (the bar-edge judge included), cached under align_min/pairs
    bands_c: dict = {}; ceilings: dict = {}

    def band_of(cid: str):
        if cid not in bands_c:
            bands_c[cid] = ga._as_band(gmd[cid], ga.BAND_ROWS_DEFAULT, pparams)
        return bands_c[cid]

    def provider(anchor: str, mov: str):
        r = cache.get(anchor, mov)
        src = "cached"
        if r is None:
            t1 = time.time()
            if anchor not in ceilings:
                ceilings[anchor] = ga._feature_ceiling(band_of(anchor), tuple(pparams.local_win), "struct")
            r = ga.register_pair(band_of(anchor), band_of(mov), pparams, ceiling=ceilings[anchor], quality=True)
            cache.put(r)
            src = f"registered in {time.time() - t1:.0f}s"
        rej_ = [f_ for f_ in r.flags if f_ in ga.REJECT_FLAGS]
        reason = (", ".join(rej_) + ga.overlap_note(r)) if rej_ else None
        log(f"    pair {mov} -> {anchor}: {src}; ok {r.ok} rel {float(r.relative_match):.3f} ncc_coarse {float(r.ncc_coarse):.3f} "
            f"df {int(r.df):+d} flags {rej_}{(' ' + ga.overlap_note(r)) if (rej_ or 'near_search_edge' in r.flags) else ''}")
        return {"ok": bool(r.ok), "df": int(r.df), "dx": np.asarray(r.dx_applied, float), "a": np.asarray(r.a, float),
                "b": np.asarray(r.b, float), "rel": (float(r.relative_match) if np.isfinite(float(r.relative_match)) else None),
                "ncc_coarse": (float(r.ncc_coarse) if np.isfinite(float(r.ncc_coarse)) else None), "flags": list(r.flags),
                "overlap_fraction": (float(r.overlap_fraction) if np.isfinite(float(getattr(r, "overlap_fraction", float("nan")))) else None),
                "reason": reason}
    # the DIRECT transforms (reference + ok members)
    transforms: dict = {g.reference: identity_transform(gmd[g.reference].volume.shape[2])}
    for rec in recs:
        if not rec["is_reference"] and rec["ok"]:
            transforms[rec["cid"]] = pair_transform(g.pairs[rec["cid"]])
    # TRANSITIVE PLACEMENT (2026-09-12): a refused member registered to each contributing member; an admissible route
    # (ok, rel ≥ min_relative_match, not at the bar, the engine's round trip closing) places it by the composed transform
    transitive: dict = {}
    refused_ids = [rec["cid"] for rec in recs if not rec["is_reference"] and not rec["ok"]]
    contrib_ids = [rec["cid"] for rec in recs if not rec["is_reference"] and rec["ok"]]
    if refused_ids and contrib_ids:
        prog.update(phase="transitive placement")
        try:
            t0 = time.time()
            transitive = transitive_placement(refused_ids, contrib_ids, transforms, provider, int(gmd[g.reference].volume.shape[0]),
                                              min_rel=float(pparams.min_relative_match), log=log, progress=lambda ph: prog.update(phase=ph))
            log(f"  transitive placement done in {time.time() - t0:.1f}s: placed {[c for c, tp in transitive.items() if tp.get('via')]}")
        except Exception as e:  # noqa: BLE001 — the placement never sinks the job
            log(f"  transitive placement FAILED: {e}\n{traceback.format_exc()}")
            transitive = {}
    for cid, tp in transitive.items():
        rec = next(r_ for r_ in recs if r_["cid"] == cid)
        rec["route"] = jsonable({k: v for k, v in tp.items() if k != "transform"})
        if tp.get("via"):
            rec["via"] = tp["via"]
            transforms[cid] = tp["transform"]
    roster = build_roster(recs)
    log("  roster: " + "; ".join(f"{_short(r_['cid'])} = {r_['role']}" for r_ in roster))
    # CONSENSUS v2 + the RGB volume on the union canvas (reviewer ask 2026-09-11 #2; majority own dome 2026-09-11):
    # reference + ok members (+ the transitively placed ones) place; refused pairs with a strong match VOTE on the dome only
    prog.update(phase="consensus v2 + 3-D volume")
    consensus = None; consensus_error = None
    cons_members: list = []; cons_ctx: dict = {}
    try:
        placed_ids = [r_["cid"] for r_ in roster if r_["placed"] and not r_["is_reference"]]
        cons_members = [gmd[g.reference]] + [gmd[cid] for cid in placed_ids]
        vote_only = vote_only_members(gmd, g.pairs, set(transforms))
        if vote_only:
            log(f"  vote-only members (refused pair, strong match): {[m.cid for m, _t in vote_only]}")
        pinfo_all = pair_info_of(g.pairs)
        for cid, tp in transitive.items():
            if tp.get("via"):
                pinfo_all[cid] = {"ok": True, "rel": tp["rel"], "ncc_coarse": tp.get("ncc_coarse"), "flags": ["transitive"], "via": tp["via"]}
        t0 = time.time()
        consensus = build_consensus(group, gmd[g.reference], cons_members, transforms, out_dir, log=log, ctx=cons_ctx,
                                    vote_only=vote_only, pair_info=pinfo_all)
        log(f"  consensus v2 + volume done in {time.time() - t0:.1f}s: RMS {json.dumps(jsonable(consensus['rms_to_consensus_px']))} "
            f"sources {json.dumps(consensus['dome']['sources'])} axial {consensus['axial_witness']['verdict']} flags {consensus['flags']}")
    except Exception as e:  # noqa: BLE001
        consensus_error = f"{type(e).__name__}: {e}"
        log(f"  consensus FAILED: {e}\n{traceback.format_exc()}")
    # the FINAL per-member transforms + the APPLIED result (reviewer ask 2026-09-11 #3)
    transforms_summary = None; transforms_error = None
    if consensus is not None:
        prog.update(phase="applying axial changes")
        try:
            t0 = time.time()
            transforms_summary = apply_transforms(group, gmd[g.reference], cons_members, transforms, cons_ctx, out_dir, log=log,
                                                  roster=roster, all_members=gmd)
            log(f"  transforms applied in {time.time() - t0:.1f}s: RMS after "
                f"{json.dumps(jsonable({pm['cid']: pm['rms_after_px'] for pm in transforms_summary['members']}))}")
        except Exception as e:  # noqa: BLE001
            transforms_error = f"{type(e).__name__}: {e}"
            log(f"  apply FAILED: {e}\n{traceback.format_exc()}")
    # REFERENCE SENSITIVITY (reviewer 2026-09-11: "technically no single replicate is a reference"): every other member
    # as the anchor, its pairs registered (cached under align_min/pairs), its consensus mapped into the served coordinates
    sensitivity = None; sensitivity_error = None
    if consensus is not None and len(members) >= 2 and not no_sensitivity:
        prog.update(phase="reference sensitivity")
        try:
            t0 = time.time()
            sensitivity = reference_sensitivity_stage(group, gmd[g.reference], gmd, list(members), transforms, cons_ctx, provider,
                                                      out_dir=out_dir, log=log, progress=lambda ph: prog.update(phase=ph))
            consensus["reference_sensitivity"] = sensitivity
            log(f"  reference sensitivity done in {time.time() - t0:.1f}s: spread per anchor "
                f"{json.dumps({c: (None if r_.get('spread_px') is None else round(r_['spread_px'], 2)) for c, r_ in sensitivity['anchors'].items()})} "
                f"(pair cache hits {cache.hits} misses {cache.misses})")
        except Exception as e:  # noqa: BLE001
            sensitivity_error = f"{type(e).__name__}: {e}"
            log(f"  reference sensitivity FAILED: {e}\n{traceback.format_exc()}")
    if consensus is not None:
        consensus["reference_sensitivity_error"] = sensitivity_error
        consensus["roster"] = roster; consensus["n_subgroup_members"] = len(recs)
        consensus["n_contributing"] = sum(1 for r_ in roster if r_["placed"]); consensus["refused"] = [r_ for r_ in roster if not r_["placed"]]
    result = {"group": group, "patient": key.get("patient"), "eye": key.get("eye"), "reference": g.reference,
              "consensus": consensus, "consensus_error": consensus_error,
              "reference_sensitivity_error": sensitivity_error, "consensus_version": gc_.CONSENSUS_VERSION,
              "transforms": transforms_summary, "transforms_error": transforms_error,
              "reference_rule": g.reference_rule, "members": recs, "member_ids": members,
              "roster": roster, "n_members": len(recs), "n_contributing": sum(1 for r_ in roster if r_["placed"]),
              "refused": [r_ for r_ in roster if not r_["placed"]],
              "transitive": {cid: jsonable({k: v for k, v in tp.items() if k != "transform"}) for cid, tp in transitive.items()},
              "engine_md5": engine_md5(), "engine_file": str(ENGINE_FILE), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "seconds": time.time() - t_all, "seconds_register": dt_reg, "timings": g.timings,
              "transitivity": g.transitivity, "transitivity_requested": bool(transitivity), "pose": {f"{m_}->{r_}": v for (m_, r_), v in g.pose.items()},
              "lateral_grid": g.lateral_grid, "non_contributing": g.non_contributing,
              "ceilings": {"struct": {k: v for k, v in g.ceiling.items() if not isinstance(v, np.ndarray)},
                           "speckle": ({k: v for k, v in g.ceiling_speckle.items() if not isinstance(v, np.ndarray)} if g.ceiling_speckle else None)},
              "params": ga.PairParams().as_dict(), "workers": workers}
    _write_json(out_dir / RESULT_NAME, result)
    prog.update(phase="done", running=False, done=True)
    log(f"=== done {group} in {time.time() - t_all:.0f}s → {out_dir / RESULT_NAME}")
    return result


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group", required=True)
    ap.add_argument("--cases-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--transitivity", action="store_true")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-sensitivity", action="store_true", help="skip the reference-sensitivity stage (extra pairs)")
    ap.add_argument("--members", default=None, help="comma-separated case ids (the sidecar's subgroup-aware member list)")
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / LOG_NAME, "a", encoding="utf-8")

    def log(msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        logf.write(line + "\n"); logf.flush()
        print(line, flush=True)
    try:
        run_group_job(a.group, Path(a.cases_root), out_dir, transitivity=a.transitivity, workers=a.workers,
                      reference=a.reference, log=log, no_sensitivity=a.no_sensitivity,
                      members=([c for c in a.members.split(",") if c.strip()] if a.members else None))
        return 0
    except Exception as e:  # noqa: BLE001
        log(f"FAILED: {type(e).__name__}: {e}\n{traceback.format_exc()}")
        p = read_json(out_dir / PROGRESS_NAME) or {}
        p.update(phase="failed", running=False, done=False, error=f"{type(e).__name__}: {e}", updated=time.time())
        try:
            _write_json(out_dir / PROGRESS_NAME, p)
        except OSError:
            pass
        return 1
    finally:
        logf.close()


if __name__ == "__main__":
    sys.exit(main())
