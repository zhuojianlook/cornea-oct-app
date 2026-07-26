"""Regression guard for the surface-crop "geom" frame rule (_sc_geom_frames).

WHY THESE TESTS EXIST. The rule's headline numbers (micro-F1 0.931 / precision 0.937 / recall 0.925 over 37
ground-truth scans, 0 false frames on the CS010 peripheral-limbus trap, 0 of 129 vetted non-clipped scans
firing) are measured against a surface corpus that lives OUTSIDE the repo (.work/persist/, gitignored, ~1 GB).
Nothing in-tree would notice if a refactor silently changed the rule. So a 2 MB micro-fixture of six scans'
anterior surfaces is committed here, with their frame sets frozen in sc_geom_golden.json:

  cs010_os_v3   the peripheral-limbus TRAP    → MUST stay empty (the legacy rule reported 7 frames here)
  cs014_os_v1   user-adjudicated mild clip    → the relabelled case
  cs014_os_v2   sibling replicate, GT clip
  cs008_od_v3   TILT: raw geometry mislocates the apex, so the mc flanks carry the signal
  cs029_os_v2   a long tapering tail
  cs001_od_v1   a clean vetted non-clipped scan → MUST stay empty

The fixture holds SURFACES, not volumes (a few hundred KB per scan instead of ~350 MB), which is why it can
live in-tree at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import oct_preprocess as ocp

DATA = Path(__file__).resolve().parent / "data"
FIXTURE = DATA / "sc_geom_fixture.npz"
GOLDEN = DATA / "sc_geom_golden.json"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(), reason="surface-crop fixture not present")


def _fixture():
    z = np.load(FIXTURE)
    return z, sorted({k.split("__")[0] for k in z.files})


def _reference_medf(a, k=7):
    """Literal transcription of the validated reference smoother (.work/wf/surface_crop_detect.py `_medf`).
    Kept here verbatim so the fast vectorised production version is checked against the thing that actually
    produced the published numbers, not against itself."""
    n = len(a)
    h = k // 2
    r = np.empty(n)
    for i in range(n):
        r[i] = np.median(a[max(0, i - h):min(n, i + h + 1)])
    return r


def test_lateral_median_is_bit_identical_to_the_reference():
    """The production smoother is vectorised for speed (~260x). It must be BIT-identical, not merely close:
    the thresholds it feeds are exact comparisons, so a last-bit difference can flip a frame."""
    z, cids = _fixture()
    for cid in cids:
        for name in ("S_mc", "S_raw"):
            A = z[f"{cid}__{name}"]
            fast = ocp._sc_lateral_median(A)
            slow = np.stack([_reference_medf(A[:, i]) for i in range(A.shape[1])], axis=1)
            assert np.array_equal(fast, slow), f"{cid}/{name} smoother diverged from the reference"


@pytest.mark.parametrize("L", [1, 2, 3, 6, 7, 8, 13])
def test_lateral_median_handles_profiles_shorter_than_the_window(L):
    """The window (k=7) does not fit in a short profile, so every row is a truncated EVEN-length window whose
    median is a mean of two elements — the case where a float64 upcast would break exactness."""
    A = (np.arange(L * 3, dtype=np.float32) % 5).reshape(L, 3)
    fast = ocp._sc_lateral_median(A)
    slow = np.stack([_reference_medf(A[:, i]) for i in range(3)], axis=1)
    assert np.array_equal(fast, slow)


def test_geom_frames_match_the_frozen_golden_sets():
    z, cids = _fixture()
    golden = json.loads(GOLDEN.read_text())
    for cid in cids:
        got = ocp._sc_geom_frames(z[f"{cid}__S_mc"], z[f"{cid}__S_raw"], z[f"{cid}__M"])
        assert got == golden[cid], f"{cid}: frame set changed"


def test_the_limbus_trap_and_a_clean_scan_stay_empty():
    """The single most important property: the rule must emit NOTHING when the apex is genuinely in-frame.
    cs010_os_v3 grazes the top only at the peripheral limbus (the legacy rule auto-cropped 7 frames there);
    cs001_od_v1 is a vetted normal scan."""
    z, _ = _fixture()
    for cid in ("case_cs010_os_v3", "case_cs001_od_v1"):
        assert ocp._sc_geom_frames(z[f"{cid}__S_mc"], z[f"{cid}__S_raw"], z[f"{cid}__M"]) == []


def test_the_motion_shift_sign_is_load_bearing():
    """A mutation test for the orientation of M. The rule recovers the raw apex as (mc apex + M), so feeding
    -M must change the answer. If this passes trivially, the shift is not actually being used and the rule has
    silently degenerated to a single-geometry test."""
    z, cids = _fixture()
    changed = 0
    for cid in cids:
        a = ocp._sc_geom_frames(z[f"{cid}__S_mc"], z[f"{cid}__S_raw"], z[f"{cid}__M"])
        b = ocp._sc_geom_frames(z[f"{cid}__S_mc"], z[f"{cid}__S_raw"], -z[f"{cid}__M"])
        changed += int(a != b)
    assert changed > 0, "negating the motion shift changed nothing — M is not being used"


def _stub_call(z, cid, **kw):
    """Drive the real detect_surface_crop_frames with a volume stub. `edge`=1e9 makes _clip_mask all-False
    (its criterion needs 0 <= edge < clip_edge_floor), so the legacy count rule is a PROVEN no-op and any
    frames in the result can only have come from the geom rule."""
    S_mc = z[f"{cid}__S_mc"]
    F = int(S_mc.shape[1])
    sag = np.zeros((1, 64, F), np.float32)
    edge = np.full((1, F), 1e9, np.float32)
    return ocp.detect_surface_crop_frames(sag, dict(ocp.DEFAULT_PARAMS), detect=edge, **kw)


def test_evidence_reaches_the_rule_through_the_public_function():
    """End-to-end wiring: the frames the public entry point returns must equal the rule's own output, and it
    must report which rule produced them. Catches a transposed or mis-ordered evidence hand-off."""
    z, cids = _fixture()
    golden = json.loads(GOLDEN.read_text())
    for cid in cids:
        res = _stub_call(z, cid, sc_s_mc=z[f"{cid}__S_mc"], sc_s_raw=z[f"{cid}__S_raw"],
                         sc_shift=z[f"{cid}__M"])
        assert res["rule"] == "geom"
        assert sorted(res["frames"]) == golden[cid], cid


def test_omitting_any_evidence_falls_back_to_the_legacy_rule():
    """THE REVERT GUARANTEE. Every pre-existing caller passes detect= only, so all of them must keep taking the
    count path untouched. Each evidence argument is individually required."""
    z, cids = _fixture()
    cid = cids[0]
    full = dict(sc_s_mc=z[f"{cid}__S_mc"], sc_s_raw=z[f"{cid}__S_raw"], sc_shift=z[f"{cid}__M"])
    assert _stub_call(z, cid)["rule"] == "count"                       # no evidence at all
    for drop in full:
        kw = {k: (None if k == drop else v) for k, v in full.items()}
        res = _stub_call(z, cid, **kw)
        assert res["rule"] == "count", f"dropping {drop} did not fall back"
        assert res["frames"] == []


def test_crop_detect_count_ignores_supplied_evidence():
    """crop_detect='count' is the no-migration revert switch: it must win even when evidence is present."""
    z, cids = _fixture()
    cid = "case_cs014_os_v2"
    S_mc = z[f"{cid}__S_mc"]
    F = int(S_mc.shape[1])
    sag = np.zeros((1, 64, F), np.float32)
    edge = np.full((1, F), 1e9, np.float32)
    res = ocp.detect_surface_crop_frames(sag, {**ocp.DEFAULT_PARAMS, "crop_detect": "count"}, detect=edge,
                                         sc_s_mc=S_mc, sc_s_raw=z[f"{cid}__S_raw"], sc_shift=z[f"{cid}__M"])
    assert res["rule"] == "count" and res["frames"] == []


def test_every_selected_frame_gets_laterals_for_the_axial_overlay():
    """The axial view draws lateral_by_frame. The geom frame set is NOT derived from the clip mask, and on ~22%
    of selected frames the mask column is empty — so without the apex-derived fallback the overlay renders
    nothing on exactly the frames this rule added. Here the mask is all-False by construction, which is the
    worst case."""
    z, cids = _fixture()
    for cid in cids:
        res = _stub_call(z, cid, sc_s_mc=z[f"{cid}__S_mc"], sc_s_raw=z[f"{cid}__S_raw"],
                         sc_shift=z[f"{cid}__M"])
        for f in res["frames"]:
            assert res["lateral_by_frame"].get(f), f"{cid} frame {f} has no laterals to draw"


def test_a_geom_failure_degrades_to_the_count_rule(monkeypatch):
    """A bug in the new rule must never abort a preprocessing run — it must fall back and say so."""
    z, cids = _fixture()
    cid = cids[0]

    def boom(*a, **k):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(ocp, "_sc_geom_frames", boom)
    res = _stub_call(z, cid, sc_s_mc=z[f"{cid}__S_mc"], sc_s_raw=z[f"{cid}__S_raw"], sc_shift=z[f"{cid}__M"])
    assert res["rule"] == "count-fallback"
    assert res["frames"] == []


def test_shared_detection_is_still_reused_by_axial_motion_correct():
    """axial_motion_correct grew a detect= passthrough so one detection pass can serve both it and the rule.
    Supplying it must produce the same result as letting it detect internally."""
    vol = np.zeros((16, 48, 24), np.float32)
    vol[:, 20:26, :] = 900.0                     # a flat slab: motion correction is a no-op either way
    p = {**ocp.DEFAULT_PARAMS, "axial_motion_correct": True}
    S = ocp.detect_surface_all(ocp.reformat_to_sagittal(vol), p)
    a, ia = ocp.axial_motion_correct(vol.copy(), p)
    b, ib = ocp.axial_motion_correct(vol.copy(), p, detect=S)
    assert np.array_equal(a, b)
    assert ia.get("applied") == ib.get("applied")


def test_manual_crop_path_does_not_touch_the_auto_only_cap():
    """REGRESSION (caught in the live app, not by the gates): the rule-aware frac cap read the detector result
    to pick its threshold, but on the MANUAL path the user supplies surface_crop_frames directly, the detector
    never runs, and that variable does not exist — every manual-GT preprocess died with UnboundLocalError. The
    auto gates must be unreachable when a human chose the frames.

    Asserted structurally (a full preprocess needs a real .OCT): the cap must be resolved INSIDE an `_auto_crop`
    guard, never in a statement that evaluates unconditionally."""
    import inspect
    src = inspect.getsource(ocp.preprocess_oct_to_nifti)
    lines = [ln.strip() for ln in src.splitlines()]
    cap_reads = [i for i, ln in enumerate(lines) if "_ci.get(" in ln and "rule" in ln]
    assert cap_reads, "cap no longer consults the detector rule - update this test"
    for i in cap_reads:
        guard = " ".join(lines[max(0, i - 3):i + 1])
        assert "_auto_crop" in guard, f"line {i} reads the detector result without an _auto_crop guard: {lines[i]}"
