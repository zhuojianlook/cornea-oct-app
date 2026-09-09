"""Tests for the EDGE-GAIN queue — `oct-edge-suggest` / `oct_preprocess._edge_gain_map`.

The feature answers the reviewer's 2026-09-06 ask ("the program should highlight slices were if the edges
were corrected, a much smoother result would occur") and it has ONE semantic that must never drift: since
2026-09-05 the per-frame rigid move is measured from the TISSUE, so correcting a line changes the SERVED EDGE
and the exported GT, NOT the image geometry. The tests below therefore only ever assert things about the
delivered LINE.

Three layers:
  A. pure-metric unit tests on synthetic (L, D, F) arrays — no case store, no I/O, no detector;
  B. endpoint shape / early-out / auth tests through the conftest TestClient;
  C. an opt-in real-data acceptance run against the motivating scan (case_cs009_os_v3), skipped when the
     reviewer's store is not present so CI stays green.
"""
from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

import oct_preprocess as oct_mod


# ── synthetic fixtures ──────────────────────────────────────────────────────
L_SYN, D_SYN, F_SYN = 60, 120, 40
FLAT_ROW = 40.0


def _flat_edge(L=L_SYN, F=F_SYN, row=FLAT_ROW):
    return np.full((L, F), float(row), dtype=np.float64)


def _dome_edge(L=L_SYN, F=F_SYN, row=FLAT_ROW):
    """A smooth edge with a real dome across frames AND a big frame-common axial excursion — i.e. what a
    healthy scan's served edge actually looks like before step 1 of the metric."""
    f = np.arange(F, dtype=np.float64)
    dome = 0.02 * (f - F / 2.0) ** 2
    motion = 12.0 * np.sin(2 * np.pi * f / max(1.0, F / 2.0))     # frame-common: identical at every lateral
    return np.tile(row + dome + motion, (L, 1))


def _vol(L=L_SYN, D=D_SYN, F=F_SYN, air=200.0):
    """A raw-like volume that is uniformly DIM above the edge (so the air-above witness stays silent)."""
    return np.full((L, D, F), float(air), dtype=np.float32)


def _conf(L=L_SYN, F=F_SYN, value=1.0):
    return np.full((L, F), float(value), dtype=np.float64)


def _run(S, arr=None, conf=None, elig=None, **over):
    return oct_mod._edge_gain_map(S, arr, conf, elig, over)


def _gain_picks(mp, need_px=None):
    g = np.asarray(mp["gain"], dtype=np.float64)
    need = mp["need_px"] if need_px is None else need_px
    return np.flatnonzero(np.isfinite(g) & (g >= need))


# ══ A. pure metric ══════════════════════════════════════════════════════════
def test_frame_common_is_removed():
    """A big axial excursion applied identically to EVERY lateral is motion, not a line error. Without
    step 1 every lateral scores equally rough and the metric is blind."""
    S = _dome_edge()
    mp = _run(S, _vol(), _conf(value=0.0))          # conf 0 everywhere: only step 1 can keep this quiet
    g = np.asarray(mp["gain"])
    assert np.nanmax(np.abs(g)) < 0.5, f"frame-common motion leaked into the gain: max {np.nanmax(g)}"
    assert _gain_picks(mp).size == 0


def test_one_lateral_spike_is_found():
    S = _flat_edge()
    S[50, :] += 40.0
    conf = _conf()
    conf[50, :] = 0.0                                # no boundary contrast under the displaced line
    mp = _run(S, _vol(), conf)
    picks = _gain_picks(mp)
    assert picks.tolist() == [50], picks.tolist()
    g = float(mp["gain"][50])
    assert 30.0 <= g <= 45.0, g


def test_leave_one_out_guard():
    """The guard keeps a block NARROWER than the reference window from defending itself. A 9-lateral block
    is a majority of the 17 laterals a guard-free window would see, so its own displacement becomes the
    'reference'; with GUARD=2 only 6 of 14 window members are contaminated and the median survives."""
    S = _flat_edge()
    S[46:55, :] += 40.0
    conf = _conf(); conf[46:55, :] = 0.0
    guarded = _run(S, _vol(), conf, edge_guard=2)
    unguarded = _run(S, _vol(), conf, edge_guard=0)
    assert float(guarded["gain"][50]) >= 0.9 * 40.0, float(guarded["gain"][50])
    assert float(unguarded["gain"][50]) <= 0.6 * float(guarded["gain"][50]), (
        float(unguarded["gain"][50]), float(guarded["gain"][50]))


def test_smooth_broad_shift_is_not_flagged():
    """A broad shift moves the cross-lateral reference with it — plausible anatomy, not a line error.
    Shouldered with a 20-lateral cosine ramp because real anatomy has no 40 px cliff between two adjacent
    laterals (measured median |S[l+1]-S[l]| across the four target scans is 0.05-0.80 px); a genuine cliff
    IS a line error and the queue is right to flag one."""
    Lb = 260
    S = _flat_edge(L=Lb)
    prof = np.zeros(Lb)
    prof[60:180] = 40.0
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 20)))
    prof[40:60] = 40.0 * ramp
    prof[180:200] = 40.0 * ramp[::-1]
    S += prof[:, None]
    conf = _conf(L=Lb); conf[40:200, :] = 0.0
    mp = _run(S, _vol(L=Lb), conf)
    assert _gain_picks(mp).size == 0, _gain_picks(mp).tolist()


def test_high_conf_disagreement_is_not_flagged():
    """W1 alone fires on the legitimate steep limbus; the AND with W2 (no boundary contrast) is what keeps
    a correctly-traced but unusual slice quiet."""
    S = _flat_edge()
    S[50, :] += 40.0
    mp = _run(S, _vol(), _conf(value=1.0))
    assert _gain_picks(mp).size == 0
    assert not np.asarray(mp["flag_gain"])[50].any()


def test_air_run_requires_min_run():
    """A bloom is a CONTIGUOUS frame run; isolated hot frames are speckle."""
    S = _flat_edge()
    base = _vol()
    short = base.copy()
    short[30, int(FLAT_ROW) - 30:int(FLAT_ROW) - 9, 10:14] = 5000.0     # 4 frames
    mp4 = _run(S, short, _conf())
    assert not np.asarray(mp4["flag_unv"]).any()

    long = base.copy()
    long[30, int(FLAT_ROW) - 30:int(FLAT_ROW) - 9, 10:15] = 5000.0      # 5 frames
    mp5 = _run(S, long, _conf())
    unv = np.asarray(mp5["flag_unv"])
    assert unv.any()
    assert unv[30].sum() >= 5
    assert not unv[np.arange(L_SYN) != 30].any()


def test_air_window_off_the_top_is_unscorable():
    """M4 — the air window is rows [S-30, S-10]. Where the line sits SHALLOWER than 30 that window falls off
    the top of the B-scan; the old clamp then sampled fixed rows 0..20, which are BELOW the line, and a bright
    band there produced a flare finding about a place the witness never looked. Such cells are unscorable:
    never flagged, never part of air_ref, and COUNTED."""
    S = _flat_edge(row=10.0)                       # 10 < edge_air_hi (30): no window fits anywhere
    arr = _vol()
    arr[30, 0:21, 10:24] = 5000.0                  # bright rows 0..20 — what the clamp used to sample
    mp = _run(S, arr, _conf())
    assert not np.asarray(mp["flag_unv"]).any(), "flagged a cell whose air window does not fit"
    assert mp["air_ref"] is None
    assert mp["air_skipped"] == L_SYN * F_SYN


def test_air_skipped_counts_only_the_unfitted_cells():
    """Mixed depths: the deep half is scored normally (its flare is still found) and only the shallow half is
    counted as skipped."""
    S = _flat_edge()
    S[:20, :] = 10.0                               # shallow half — window does not fit
    arr = _vol()
    arr[30, int(FLAT_ROW) - 30:int(FLAT_ROW) - 9, 10:24] = 5000.0   # flare on a DEEP lateral
    arr[5, 0:21, 10:24] = 5000.0                                    # bright below a SHALLOW line: unscorable
    mp = _run(S, arr, _conf())
    unv = np.asarray(mp["flag_unv"])
    assert unv[30].any(), "the flare on a lateral whose window fits was lost"
    assert not unv[:20].any(), "flagged a shallow lateral whose air window does not fit"
    assert mp["air_skipped"] == 20 * F_SYN


def test_air_skipped_zero_when_every_window_fits():
    mp = _run(_flat_edge(), _vol(), _conf())
    assert mp["air_skipped"] == 0


def test_unverifiable_carries_no_px():
    """An `unverifiable` finding cannot deliver a quoted px (at a flare the served edge is already smooth
    across laterals), so it must carry none — the determinism report's UNSPANNED precedent."""
    S = _flat_edge()
    arr = _vol()
    arr[30, int(FLAT_ROW) - 30:int(FLAT_ROW) - 9, 10:20] = 5000.0
    mp = _run(S, arr, _conf())
    assert np.asarray(mp["flag_unv"])[30].any()
    # the endpoint's own wording, exercised end-to-end in test_edge_suggest_unverifiable_reason_has_no_px
    frames = oct_mod._mask_runs(np.asarray(mp["flag_unv"])[30], 1)
    reason = ("the line crosses a specular flare - no dark background above it at frames "
              + ", ".join(f"{a}-{b}" for a, b in frames) + "; the image cannot confirm the line there")
    assert not re.search(r"\d+(\.\d+)?\s*px", reason), reason


def test_crop_band_frames_excluded():
    """crop_bands are the reviewer's ⊟ artifact bands; the applied move over them is extrapolated, so a
    60 px excursion inside the band is not a line error to be corrected."""
    p = {"crop_bands": {"0": [30, 39], "59": [30, 39]}}
    elig, excl = oct_mod._edge_eligibility(p, L_SYN, F_SYN)
    assert excl["crop_bands"] == list(range(30, 40))
    assert not elig[:, 30:40].any() and elig[:, :30].all()
    S = _flat_edge()
    S[50, 30:40] += 60.0
    conf = _conf(); conf[50, 30:40] = 0.0
    mp = _run(S, _vol(), conf, elig=elig)
    assert _gain_picks(mp).size == 0


def test_surface_crop_frames_excluded():
    """surface_crop_frames have NO anterior surface at all — opposite polarity from oct-bottom-suggest,
    which requires that band for the posterior."""
    p = {"surface_crop_frames": list(range(0, 12))}
    elig, excl = oct_mod._edge_eligibility(p, L_SYN, F_SYN)
    assert excl["surface_crop"] == list(range(0, 12))
    assert not elig[:, :12].any()
    S = _flat_edge()
    S[50, 0:12] += 60.0
    conf = _conf(); conf[50, 0:12] = 0.0
    mp = _run(S, _vol(), conf, elig=elig)
    assert _gain_picks(mp).size == 0


def test_all_nan_lateral_returns_nan_not_raise():
    S = _flat_edge()
    S[7, :] = np.nan
    mp = _run(S, _vol(), _conf())
    assert not np.isfinite(mp["gain"][7])
    assert _gain_picks(mp).size == 0


def test_too_small_volume_early_outs():
    assert _run(np.zeros((2, 40)))["reason"] == "fewer than 3 laterals"
    assert _run(np.zeros((40,)))["reason"] == "no served edge"


def test_degrades_without_volume_or_conf():
    """Guidance must never block editing — but degrading must never mean INVENTING findings.

    No raw volume -> no air witness. No conf -> W2 is UNEVALUABLE, so there are NO gain findings at all:
    treating conf=None as 0 made W2 true at every cell, the AND collapsed onto the cross-lateral witness
    (which this module's own docstring says fires on the legitimate steep limbus), and a clean scan came back
    as a list of false "redraw" findings."""
    S = _flat_edge(); S[50, :] += 40.0
    mp = _run(S, None, None)
    assert mp["air_ref"] is None and not np.asarray(mp["flag_unv"]).any()
    assert mp["conf_ok"] is False
    assert not np.asarray(mp["flag_gain"]).any()
    assert _gain_picks(mp).size == 0
    # and no `gain` number may be quoted anywhere: a 0.0 would read as "measured, and it is clean"
    assert not np.isfinite(np.asarray(mp["gain"])).any()
    # the departure itself is still measured (it is what the endpoint reports as the withheld quantity)
    assert np.isfinite(mp["dev"][50]) and mp["dev"][50] > 30.0


def test_conf_wrong_shape_is_unevaluable():
    """A conf map of the wrong shape is as unevaluable as a missing one — it must not be padded with zeros
    (zeros mean 'no boundary contrast', i.e. the witness silently reads as SATISFIED)."""
    S = _flat_edge(); S[50, :] += 40.0
    mp = _run(S, _vol(), np.zeros((L_SYN, F_SYN + 3)))
    assert mp["conf_ok"] is False
    assert _gain_picks(mp).size == 0 and not np.asarray(mp["flag_gain"]).any()


def test_conf_present_is_evaluable():
    """The counterpart: a usable conf map means conf_ok True and the AND does its job."""
    S = _flat_edge(); S[50, :] += 40.0
    conf = _conf(); conf[50, :] = 0.0
    mp = _run(S, _vol(), conf)
    assert mp["conf_ok"] is True
    assert _gain_picks(mp).tolist() == [50]


# ══ B. endpoint ═════════════════════════════════════════════════════════════
def _border_case(make_case, cases_root, write_nifti, cid="case_edge_syn", *, spike=True, manifest=None):
    """A case with a real `input/_raw_border.nii.gz` so the endpoint's raw path is exercised. The volume is
    built so the FAST/robust detector finds a flat bright boundary; the served edge is then whatever
    _served_edge_map returns (we assert on shape, not on injected px, for the endpoint tests)."""
    L, D, F = 40, 80, 30
    vol = np.full((L, D, F), 30, np.uint16)
    vol[:, 30:, :] = 900                                   # bright tissue below row 30 -> a clean edge
    if spike:
        vol[20, 30:60, :] = 30                             # lateral 20: boundary pushed 30 px deeper
        vol[20, 60:, :] = 900
    cid = make_case(cid, manifest=manifest)
    write_nifti(vol, cases_root / cid / "input" / "_raw_border.nii.gz")
    return cid


def test_edge_suggest_shape(client, cases_root, make_case, write_nifti):
    cid = _border_case(make_case, cases_root, write_nifti)
    r = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}})
    assert r.status_code == 200, r.text
    b = r.json()
    for k in ("slices", "n_frames", "witness", "move_source", "need_px", "dev_need_px", "sigma_lat_px",
              "tau_lat_px", "air_ref", "air_skipped", "conf_ok", "order", "k_air", "min_run",
              "median_dev_px", "p90_dev_px", "excluded_frames", "drawn",
              "picks", "regions", "n_left", "reason", "reason_kind"):
        assert k in b, f"missing response key {k!r}"
    assert b["witness"] == "cross-lateral + contrast + air-above"
    assert isinstance(b["excluded_frames"], dict)
    assert set(b["excluded_frames"]) == {"crop_bands", "surface_crop"}
    assert len(b["picks"]) <= 8
    assert b["n_left"] == len(b["regions"])
    for q in b["picks"] + b["regions"]:
        assert q["kind"] in ("gain", "unverifiable")
        assert isinstance(q["lateral"], int) and 0 <= q["lateral"] < b["slices"]
        assert q["lo"] <= q["lateral"] <= q["hi"]
        assert isinstance(q["drawn"], bool) and isinstance(q["reason"], str)
        for fr in q["frames"]:
            assert len(fr) == 2 and fr[0] <= fr[1] and 0 <= fr[0] < b["n_frames"]
        assert q["qualified"] in ("gain", "departure", "flare")
        if q["kind"] == "unverifiable":
            assert q["gain_px"] is None and q["off_px"] is None and q["qualified"] == "flare"
        else:
            # every `gain`-kind sentence talks about a DISTANCE, so the distance must be on the record
            assert isinstance(q["off_px"], (int, float))


def test_edge_suggest_picks_are_gain_first(client, cases_root, make_case, write_nifti):
    cid = _border_case(make_case, cases_root, write_nifti, cid="case_edge_order")
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    kinds = [q["kind"] for q in b["picks"]]
    assert kinds == sorted(kinds, key=lambda k: 0 if k == "gain" else 1)
    gains = [q["gain_px"] for q in b["picks"] if q["kind"] == "gain"]
    assert gains == sorted(gains, reverse=True)


def test_edge_suggest_unverifiable_reason_has_no_px(client, cases_root, make_case, write_nifti):
    """A specular flare above the edge must produce an `unverifiable` finding with NO px number."""
    L, D, F = 40, 80, 30
    vol = np.full((L, D, F), 30, np.uint16)
    vol[:, 30:, :] = 900
    vol[18:23, 0:29, 8:20] = 4000                     # bloom: bright ABOVE the edge over 12 frames
    cid = make_case("case_edge_flare")
    write_nifti(vol, cases_root / "case_edge_flare" / "input" / "_raw_border.nii.gz")
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    unv = [q for q in b["regions"] if q["kind"] == "unverifiable"]
    assert unv, b
    for q in unv:
        assert q["gain_px"] is None
        assert not re.search(r"\d+(\.\d+)?\s*px", q["reason"]), q["reason"]
        assert "specular flare" in q["reason"]
        # The frame numbers in the reason are ARRAY indices, but the sagittal border panel is scaleX(-1) —
        # so the reason must also say WHICH SIDE of the B-scan to look at, the same hint the
        # under-determination prompt already ships (SliceGallery.tsx sideOf, COV_EDGE_BAND = 12). Without it
        # the queue names a place the reviewer cannot find on screen.
        assert "of the B-scan)" in q["reason"], q["reason"]
        assert any(s in q["reason"] for s in
                   ("display-LEFT edge", "display-RIGHT edge", "middle")), q["reason"]


def test_edge_suggest_clean_scan_is_silent(client, cases_root, make_case, write_nifti):
    """A scan with no line error returns no gain picks and quotes the measured floor rather than implying
    the deviation is zero."""
    cid = _border_case(make_case, cases_root, write_nifti, cid="case_edge_clean", spike=False)
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    assert [q for q in b["picks"] if q["kind"] == "gain"] == []
    assert b["reason"] is not None and b["reason"].startswith("already at the measured floor")


def test_edge_suggest_excluded_frames_reported(client, cases_root, make_case, write_nifti):
    cid = _border_case(make_case, cases_root, write_nifti, cid="case_edge_excl", manifest={
        "oct_params": {"crop_bands": {"0": [25, 29], "39": [25, 29]}, "surface_crop_frames": [0, 1, 2]}})
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    assert b["excluded_frames"]["crop_bands"] == [25, 26, 27, 28, 29]
    assert b["excluded_frames"]["surface_crop"] == [0, 1, 2]


# ── a SERVED EDGE we control, so the endpoint's gating layer can be tested without real data ──
# The synthetic detector output is flat (dev 0 on every lateral), so a test that needs a real departure has to
# supply the served edge itself. Everything downstream — _edge_gain_map, the two bars, the sentences — is the
# production code path; only the curve the queue reasons about is injected.
def _spiked_case(monkeypatch, cases_root, make_case, write_nifti, cid, *, spike_frames=None, amp=40.0,
                 lat=20, extra=()):
    import api_server
    cid = _border_case(make_case, cases_root, write_nifti, cid=cid, spike=False)
    L, F = 40, 30
    S = np.full((L, F), 30.0)
    fr = slice(None) if spike_frames is None else slice(*spike_frames)
    S[lat, fr] += amp
    for l2, a2, fr2 in extra:
        S[l2, (slice(None) if fr2 is None else slice(*fr2))] += a2
    monkeypatch.setattr(api_server, "_served_edge_map", lambda *a, **k: (S, S, None))
    return cid, S, L, F


def _conf_zeros(L, F, holes=None):
    """A conf map the AND is satisfied by. `holes` = {lateral: (f0, f1)} keeps contrast HIGH outside that
    frame window, so only part of a displaced line is doubly-flagged — the partial-flagging case that made
    `gain` collapse below the floor."""
    def _f(arr, S, p):
        c = np.zeros((L, F), np.float64)
        if holes:
            for l2, (f0, f1) in holes.items():
                c[l2, :] = 1.0
                c[l2, f0:f1] = 0.0
        return c
    return _f


def test_edge_suggest_withholds_gain_when_conf_witness_raises(client, cases_root, make_case, write_nifti,
                                                              monkeypatch):
    """H1 — surface_confidence_map failing must WITHHOLD the smoothing findings, not silently satisfy W2.

    The control half of the test matters: with a working witness this exact scan yields a gain pick, so the
    silence below is caused by the failure and nothing else."""
    import api_server
    cid, S, L, F = _spiked_case(monkeypatch, cases_root, make_case, write_nifti, "case_edge_confok")
    monkeypatch.setattr(oct_mod, "surface_confidence_map", _conf_zeros(L, F))
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    assert b["conf_ok"] is True
    assert [q["lateral"] for q in b["picks"] if q["kind"] == "gain"] == [20], b["picks"]

    def _boom(*a, **k):
        raise RuntimeError("confidence map unavailable")
    monkeypatch.setattr(oct_mod, "surface_confidence_map", _boom)
    b2 = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    assert b2["conf_ok"] is False
    assert [q for q in b2["picks"] if q["kind"] == "gain"] == [], b2["picks"]
    assert [q for q in b2["regions"] if q["kind"] == "gain"] == []
    # the AIR witness is independent of the contrast map, so `unverifiable` findings legitimately survive —
    # what must not survive is a smoothing claim resting on a witness that could not be evaluated
    assert all(q["kind"] == "unverifiable" for q in b2["regions"]), b2["regions"]
    # …and it SAYS so. Silence with no explanation would read as "this scan is clean".
    assert b2["reason_kind"] == "no_contrast_witness", b2["reason"]
    assert "contrast witness" in (b2["reason"] or "")
    assert not (b2["reason"] or "").startswith("already at the measured floor"), b2["reason"]
    assert "contrast witness unavailable" in b2["witness"]
    # no gain figure may be quoted while the witness that defines it could not be evaluated
    assert not re.search(r"max gain", b2["reason"] or ""), b2["reason"]


def test_edge_suggest_departure_below_the_gain_floor_is_surfaced(client, cases_root, make_case, write_nifti,
                                                                 monkeypatch):
    """H2 — a line uniformly 40 px off its neighbours, of which only 8 frames also lack contrast: the gain
    (RMS removed by zeroing ONLY those cells) is ~5.8 px, below the 8 px floor. Before the second bar this was
    dropped and the reviewer was told "already at the measured floor (max gain 5.8 px < 8)" — the miss
    reported as reassurance."""
    cid, S, L, F = _spiked_case(monkeypatch, cases_root, make_case, write_nifti, "case_edge_devbar")
    monkeypatch.setattr(oct_mod, "surface_confidence_map", _conf_zeros(L, F, holes={20: (4, 12)}))
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    hit = [q for q in b["picks"] if q["lateral"] == 20]
    assert hit, b["reason"]
    q = hit[0]
    assert q["kind"] == "gain" and q["qualified"] == "departure", q
    assert q["gain_px"] < b["need_px"], q                 # the old bar would have dropped it
    assert q["dev_px"] >= b["dev_need_px"], q             # the new bar is why it is here
    assert 38.0 <= q["off_px"] <= 42.0, q                 # and the departure is the ~40 px actually present


def test_edge_suggest_floor_reason_quotes_both_bars(client, cases_root, make_case, write_nifti, monkeypatch):
    """The honest-silence sentence must quote the quantities it actually gated on — both of them, with the
    measured maxima, so a departure that fell under its bar is visible rather than hidden behind the gain."""
    # a small departure (10 px over 6 of 30 frames) of which only half also lacks contrast, so the two
    # quantities differ: gain ~1.3 px, departure ~4.5 px. Both are under their bars, so the scan is silent —
    # and the sentence has to name both, because quoting the gain alone hides the larger number.
    cid, S, L, F = _spiked_case(monkeypatch, cases_root, make_case, write_nifti, "case_edge_bothbars",
                                amp=10.0, spike_frames=(0, 6))
    monkeypatch.setattr(oct_mod, "surface_confidence_map", _conf_zeros(L, F, holes={20: (0, 3)}))
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    assert [q for q in b["picks"] if q["kind"] == "gain"] == [], b["picks"]
    assert b["reason_kind"] == "floor"
    m = re.search(r"max gain ([\d.]+) px < ([\d.]+) px, max departure ([\d.]+) px < ([\d.]+) px",
                  b["reason"] or "")
    assert m, b["reason"]
    g, gbar, d, dbar = (float(x) for x in m.groups())
    assert gbar == round(b["need_px"], 1) and dbar == round(b["dev_need_px"], 1)
    assert g < gbar and d < dbar
    # the two numbers are genuinely different quantities: the departure is the larger one, and it is the one
    # the old sentence hid behind "max gain N px"
    assert d > g, (g, d)


def test_edge_suggest_reason_numbers_match_the_field_they_name(client, cases_root, make_case, write_nifti,
                                                               monkeypatch):
    """M3 — "sits ~N px off" must quote the DEPARTURE (off_px), never the gain. Here they differ by 15 px:
    the line is 40 px off over 12 of 30 frames, so the achievable RMS reduction is ~25 px."""
    cid, S, L, F = _spiked_case(monkeypatch, cases_root, make_case, write_nifti, "case_edge_words",
                                spike_frames=(0, 12))
    monkeypatch.setattr(oct_mod, "surface_confidence_map", _conf_zeros(L, F))
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    q = next(x for x in b["picks"] if x["lateral"] == 20)
    assert q["qualified"] == "gain"
    off = re.search(r"sits ~([\d.]+) px off", q["reason"])
    assert off, q["reason"]
    assert abs(float(off.group(1)) - q["off_px"]) <= 0.05, (off.group(1), q)
    rec = re.search(r"recovers about ([\d.]+) px", q["reason"])
    assert rec, q["reason"]
    assert abs(float(rec.group(1)) - q["gain_px"]) <= 0.05, (rec.group(1), q)
    # the two really are different quantities — this is the defect the test guards
    assert abs(q["off_px"] - q["gain_px"]) > 5.0, q
    assert "departs from its neighbours by" not in q["reason"], q["reason"]


def test_edge_suggest_every_gain_reason_matches_its_fields(client, cases_root, make_case, write_nifti,
                                                           monkeypatch):
    """The same rule over a mixed queue: gain-qualified, departure-qualified and whole-line wordings."""
    cid, S, L, F = _spiked_case(monkeypatch, cases_root, make_case, write_nifti, "case_edge_mixed",
                                lat=8, amp=40.0, extra=[(20, 20.0, None), (32, 40.0, None)])
    monkeypatch.setattr(oct_mod, "surface_confidence_map", _conf_zeros(L, F, holes={32: (4, 12)}))
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    for q in [x for x in b["regions"] if x["kind"] == "gain"]:
        off = re.search(r"sits ~([\d.]+) px off", q["reason"])
        assert off and abs(float(off.group(1)) - q["off_px"]) <= 0.05, q
        rec = re.search(r"recover[sy] about ([\d.]+) px|\(~([\d.]+) px\)", q["reason"])
        if q["gain_px"] is not None:
            assert rec, q["reason"]
            val = float(next(g for g in rec.groups() if g))
            assert abs(val - q["gain_px"]) <= 0.05, q


def test_edge_suggest_order_is_most_correctable_first(client, cases_root, make_case, write_nifti, monkeypatch):
    """ORDER (decision recorded 2026-09-07): keep the gain ranking — the alternative (rank by the reduction in
    each lateral's own off-deg-2) measured NEGATIVE on four of seven real picks. The `order` STRING must name
    that key. It once said "worst departure from the neighbouring slices first", which is false of a
    gain-descending row: on cs009_os_v3 that order puts l303 (departure 18.71, gain 14.52) ahead of l183
    (departure 20.80, gain 10.46). Departure-qualified regions promise no removal, so they follow."""
    cid, S, L, F = _spiked_case(monkeypatch, cases_root, make_case, write_nifti, "case_edge_order2",
                                lat=8, amp=40.0, extra=[(20, 20.0, None), (32, 40.0, None)])
    monkeypatch.setattr(oct_mod, "surface_confidence_map", _conf_zeros(L, F, holes={32: (4, 12)}))
    b = client.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    assert "most correctable first" in b["order"], b["order"]
    assert "worst departure" not in b["order"], b["order"]      # the claim the ranking does not support
    lats = [q["lateral"] for q in b["picks"] if q["kind"] == "gain"]
    assert lats[:3] == [8, 20, 32], b["picks"]
    quals = [q["qualified"] for q in b["picks"] if q["kind"] == "gain"]
    assert quals == sorted(quals, key=lambda k: 0 if k == "gain" else 1), quals
    # The key IS the gain, so the shipped row is gain-descending among gain-qualified regions — assert the
    # property the string now promises, rather than the departure order it used to promise.
    gains = [q["gain_px"] for q in b["regions"] if q["qualified"] == "gain"]
    assert gains == sorted(gains, reverse=True), gains


def test_edge_suggest_not_preprocessed(client, cases_root):
    """A guidance feature must degrade, never 500."""
    import orchestration as orch
    orch.ensure_case_dirs("case_edge_bare")
    orch.write_manifest_value("case_edge_bare", {"patient_id": "x"})
    r = client.post("/api/case/case_edge_bare/oct-edge-suggest", json={"params": {}})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["reason"] == "not preprocessed"
    assert b["picks"] == [] and b["regions"] == [] and b["n_left"] == 0


def test_edge_suggest_token(cases_root, monkeypatch):
    """POST needs x-cornea-token when CORNEA_API_TOKEN is set (a dev sidecar without it needs none)."""
    from fastapi.testclient import TestClient
    import api_server
    monkeypatch.setattr(api_server, "_API_TOKEN", "sekrit", raising=False)
    with TestClient(api_server.app) as c:
        r = c.post("/api/case/case_edge_bare/oct-edge-suggest", json={"params": {}})
        assert r.status_code == 401, r.status_code
        r2 = c.post("/api/case/case_edge_bare/oct-edge-suggest", json={"params": {}},
                    headers={"x-cornea-token": "sekrit"})
        assert r2.status_code == 200


def test_served_edge_map_matches_curves(client, cases_root, make_case, write_nifti):
    """THE refactor's regression test: the queue and the pane must read the same curve. If these ever
    diverge the banner is reasoning about a line the reviewer cannot see."""
    import api_server
    import orchestration as orch
    cid = _border_case(make_case, cases_root, write_nifti, cid="case_edge_same")
    curves = client.post(f"/api/case/{cid}/oct-border-curves-all", json={}).json()
    m = orch.read_manifest(cid)
    p = {**oct_mod.DEFAULT_PARAMS, **(m.get("oct_params") or {})}
    arr = api_server._load_border_vol(api_server._ensure_raw_border_nifti(cid))
    edges, fits, _ = api_server._served_edge_map(cid, m, arr, p, 1)
    np.testing.assert_allclose(np.round(edges, 1), np.asarray(curves["edges"]), atol=1e-9)
    np.testing.assert_allclose(np.round(fits, 1), np.asarray(curves["fits"]), atol=1e-9)


# ══ C. real-data acceptance (opt-in) ════════════════════════════════════════
_STORE = Path(os.environ.get("CORNEA_REVIEW_STORE",
                             "/home/zhuojian/Desktop/Integration/review_cases/cases"))


def _have(cid: str) -> bool:
    return (_STORE / cid / "manifest.json").exists()


@pytest.fixture(scope="module")
def real_client():
    """A TestClient over a COPY of the reviewer's store (never a symlink: a sidecar can write through one).
    Copies are made per requested case by `real_case`."""
    from fastapi.testclient import TestClient
    import api_server
    import settings
    tmp = Path(tempfile.mkdtemp(prefix="cornea_edgesuggest_"))
    root = tmp / "cases"
    root.mkdir(parents=True, exist_ok=True)
    old = settings.CASES_ROOT
    settings.CASES_ROOT = root
    try:
        with TestClient(api_server.app) as c:
            yield c, root
    finally:
        settings.CASES_ROOT = old
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture(scope="module")
def real_case(real_client):
    c, root = real_client
    def _get(cid: str):
        if not _have(cid):
            pytest.skip(f"reviewer store case {cid} not present under {_STORE} — real-data test skipped")
        dst = root / cid
        if not dst.exists():
            shutil.copytree(_STORE / cid, dst, symlinks=False)
        return c, cid
    return _get


@pytest.mark.realdata
def test_cs009_os_v3_surfaces_the_flare_band(real_case):
    c, cid = real_case("case_cs009_os_v3")
    b = c.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    unv = [q for q in b["regions"] if q["kind"] == "unverifiable"]
    assert unv, "the apex specular flare was not surfaced"
    hit = [q for q in unv if q["lo"] <= 223 and q["hi"] >= 203]
    assert hit, [(q["lo"], q["hi"]) for q in unv]
    frames = [f for q in hit for fr in q["frames"] for f in range(fr[0], fr[1] + 1)]
    assert set(frames) & set(range(34, 46)), frames


@pytest.mark.realdata
def test_cs009_os_v3_quiet_slices_stay_quiet(real_case):
    c, cid = real_case("case_cs009_os_v3")
    b = c.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    for lat in (446, 389):                       # display 67 and 124 — measured gain 0.42 and 0.00
        assert all(q["lateral"] != lat for q in b["picks"]), lat
        assert all(not (q["lo"] <= lat <= q["hi"]) for q in b["regions"]), lat


@pytest.mark.realdata
def test_cs009_os_v3_drawn_spikes_are_surfaced(real_case):
    """The one-lateral spikes are all AT reviewer-drawn laterals — proof that a `near`-radius exclusion would
    hide the worst findings in the volume. Seven before 2026-09-09; the chord witness (chord_witness_refusals)
    then stopped bridging the stroke gaps on laterals 103, 145, 183 and 303 with chords through the tissue
    (24 → 0.8 px off the raw crossing at those frames), so those four spikes no longer exist to be surfaced."""
    c, cid = real_case("case_cs009_os_v3")
    b = c.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {"n": 24}}).json()
    gains = [q for q in b["picks"] if q["kind"] == "gain"]
    assert {q["lateral"] for q in gains} == {23, 44, 450}, [q["lateral"] for q in gains]
    assert all(q["drawn"] for q in gains)
    assert gains[0]["gain_px"] > 0, gains[0]


@pytest.mark.realdata
@pytest.mark.parametrize("cid", ["case_cs009_os_v1", "case_cs002_os_v1", "case_cs008_od_v2"])
def test_floor_scans_return_no_gain_picks(real_case, cid):
    c, cid = real_case(cid)
    b = c.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}}).json()
    assert [q for q in b["picks"] if q["kind"] == "gain"] == []
    assert (b["reason"] or "").startswith("already at the measured floor"), b["reason"]


@pytest.mark.realdata
def test_runtime_budget(real_case):
    c, cid = real_case("case_cs009_os_v3")
    c.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}})      # warm the caches
    t0 = time.time()
    c.post(f"/api/case/{cid}/oct-edge-suggest", json={"params": {}})
    assert time.time() - t0 < 8.0
