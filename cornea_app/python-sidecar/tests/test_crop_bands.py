"""EXPLICIT crop-band contract (2026-09-09) — backend pins.

The reviewer: "specify the start of one crop, e.g. start band on one slice and end band on another, which is the
normal function, and also other bands with their own start and end". A BAND is an explicit object with an id and
ITS OWN marks; bands are independent (no linking, no merging). tests/data/crop_band_fixtures.json is the SHARED
pin between oct_preprocess.resolve_crop_bands and the frontend (src/store/cropBands.ts bandsAt) — every probe in
it must be reproduced exactly here. The legacy single-band map must resolve BIT-IDENTICALLY to the pre-contract
resolver, whose text is embedded below verbatim (git show HEAD:cornea_app/python-sidecar/oct_preprocess.py, the
_artifact_bands function only).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

import oct_preprocess as oct_mod
import orchestration as orch

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "data" / "crop_band_fixtures.json"


def _cases():
    return json.loads(FIXTURE.read_text())["cases"]


# ── 1. the shared fixture: every probe of every case ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["name"])
def test_resolve_crop_bands_reproduces_every_fixture_probe(case):
    got = oct_mod.resolve_crop_bands(case["crop_bands"], case["n_lateral"], case["n_frames"])
    assert sorted(got) == list(range(case["n_lateral"]))           # every lateral is a key
    assert case["probes"], case["name"]
    for lat_s, want in case["probes"].items():
        have = [[int(lo), int(hi), int(bid)] for lo, hi, bid in got[int(lat_s)]]
        assert have == want, (case["name"], lat_s, have, want)


def test_fixture_spec_names_the_explicit_model():
    j = json.loads(FIXTURE.read_text())
    assert "EXPLICIT" in j["spec"] and "no linking" in j["spec"]
    names = [c["name"] for c in j["cases"]]
    assert len(names) == len(set(names))
    for c in j["cases"]:
        assert set(c) >= {"name", "n_lateral", "n_frames", "crop_bands", "probes"}


# ── 2. legacy single-band map == the pre-contract resolver, bit for bit ──────────────────────────────────────────
_OLD_ARTIFACT_BANDS_SRC = r'''
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
'''


def _old_artifact_bands():
    ns = {"np": np, "math": math}
    exec(_OLD_ARTIFACT_BANDS_SRC, ns)
    return ns["_artifact_bands"]


def _same(old, new):
    if old is None or new is None:
        assert old is None and new is None, (old, new)
        return
    assert len(old) == len(new)
    for o, n in zip(old, new):
        assert o.dtype == n.dtype, (o.dtype, n.dtype)
        assert np.array_equal(o, n), (o, n)


_LEGACY_CASES = [
    ({"0": [30, 39], "59": [30, 39]}, 60, 40),
    ({"10": [20, 30], "20": [25, 35]}, 64, 50),
    ({"5": [35, 45], "3": [9, 2]}, 8, 40),
    ({"x": [1, 2], "2": [1], "3": "junk", "9": [1, 2], "4": [5, 7], "6": [[1, 2], [3, 4]]}, 8, 40),
    ({"7": [3, 9]}, 30, 40),
    ({"0": [10, 20], "10": [20, 30], "30": [0, 10]}, 64, 50),
    ({"0": [20, 30], "200": [20, 30]}, 512, 100),      # the old resolver had NO gap limit either: one band
    ({"0": [20, 30], "511": [20, 30]}, 512, 100),      # the common store shape: marks at both extreme laterals
    ({"3": ["1", "4.9"], "5": [True, 7.2]}, 8, 40),    # int()-able scalars of any type
    ({}, 8, 40), (None, 8, 40), ({"x": 1}, 8, 40),
]


@pytest.mark.parametrize("raw,n_lat,n_frames", _LEGACY_CASES)
def test_legacy_map_is_bit_identical_to_the_old_resolver(raw, n_lat, n_frames):
    old = _old_artifact_bands()
    _same(old({"crop_bands": raw}, n_frames, n_lat), oct_mod._artifact_bands({"crop_bands": raw}, n_frames, n_lat))


def test_legacy_map_bit_identical_random_sweep():
    old = _old_artifact_bands()
    rng = np.random.default_rng(20260909)
    for _ in range(400):
        n_lat = int(rng.integers(1, 513)); n_frames = int(rng.integers(1, 101))
        k = int(rng.integers(1, 7))
        raw = {}
        for _j in range(k):
            lat = int(rng.integers(-3, n_lat + 3))                     # some out of range
            lo = int(rng.integers(-5, n_frames + 5)); hi = int(rng.integers(-5, n_frames + 5))
            raw[str(lat)] = [lo, hi]
        _same(old({"crop_bands": raw}, n_frames, n_lat), oct_mod._artifact_bands({"crop_bands": raw}, n_frames, n_lat))


# ── 3. explicit-model semantics ──────────────────────────────────────────────────────────────────────────────────
def test_one_band_start_on_one_slice_end_on_another_is_the_normal_function():
    r = oct_mod.resolve_crop_bands({"bands": [{"id": 1, "marks": {"0": [20, 30], "511": [20, 30]}}]}, 512, 100)
    assert all(r[lat] == [(20, 30, 1)] for lat in range(512))


def test_two_single_mark_bands_never_link():
    """The SAME marks as above but as two bands → nothing between them (no auto-linking by frame overlap)."""
    r = oct_mod.resolve_crop_bands({"bands": [{"id": 1, "marks": {"0": [20, 30]}}, {"id": 2, "marks": {"511": [20, 30]}}]}, 512, 100)
    assert r[0] == [(20, 30, 1)] and r[511] == [(20, 30, 2)]
    assert all(r[lat] == [] for lat in range(1, 511))


def test_bands_are_independent_even_when_frames_overlap():
    r = oct_mod.resolve_crop_bands({"bands": [{"id": 1, "marks": {"0": [0, 10], "20": [0, 10]}},
                                              {"id": 2, "marks": {"5": [5, 15]}}]}, 30, 40)
    assert r[5] == [(0, 10, 1), (5, 15, 2)]
    assert r[6] == [(0, 10, 1)]                       # band 2 is single-mark: its slice only
    assert r[4] == [(0, 10, 1)]


def test_a_new_mark_on_a_slice_replaces_that_bands_mark_there():
    """One mark per lateral per band: a dict can only hold one — the last write wins, as the frontend does."""
    m = {"3": [1, 2]}; m["3"] = [5, 9]
    r = oct_mod.resolve_crop_bands({"bands": [{"id": 1, "marks": m}]}, 8, 40)
    assert r[3] == [(5, 9, 1)]


def test_resolve_params_argument_is_ignored():
    raw = {"bands": [{"id": 1, "marks": {"0": [1, 2]}}, {"id": 2, "marks": {"3": [1, 2]}}]}
    assert oct_mod.resolve_crop_bands(raw, 8, 10) == oct_mod.resolve_crop_bands(raw, 8, 10, {"crop_band_link_frames": 99, "crop_band_max_gap": 999})


def test_resolve_degenerate_dims():
    assert oct_mod.resolve_crop_bands({"0": [1, 2]}, 0, 10) == {}
    assert oct_mod.resolve_crop_bands({"0": [1, 2]}, 4, 0) == {0: [], 1: [], 2: [], 3: []}


# ── 4. the union consumers ───────────────────────────────────────────────────────────────────────────────────────
_OVERLAP = {"bands": [{"id": 1, "marks": {"0": [0, 10], "20": [0, 10]}}, {"id": 2, "marks": {"10": [8, 15], "30": [8, 15]}}]}


def test_artifact_bands_is_the_union_unique_ascending():
    ab = oct_mod._artifact_bands({"crop_bands": _OVERLAP}, 30, 40)
    assert ab is not None and len(ab) == 40
    assert np.array_equal(ab[15], np.arange(0, 16)) and ab[15].dtype == np.array([], dtype=int).dtype
    assert np.array_equal(ab[5], np.arange(0, 11))
    assert np.array_equal(ab[25], np.arange(8, 16))
    assert ab[31].size == 0 and ab[39].size == 0


def test_artifact_bands_disjoint_bands_on_one_lateral_keep_the_gap():
    ab = oct_mod._artifact_bands({"crop_bands": {"bands": [{"id": 1, "marks": {"0": [2, 4]}}, {"id": 2, "marks": {"0": [8, 9]}}]}}, 20, 4)
    assert np.array_equal(ab[0], np.array([2, 3, 4, 8, 9]))


def test_artifact_bands_none_without_a_valid_band():
    assert oct_mod._artifact_bands({"crop_bands": {"bands": []}}, 30, 40) is None
    assert oct_mod._artifact_bands({"crop_bands": {}}, 30, 40) is None
    assert oct_mod._artifact_bands({}, 30, 40) is None
    assert oct_mod._artifact_bands({"crop_bands": {"bands": [{"id": 1, "marks": {"99": [1, 2]}}]}}, 30, 40) is None


def test_artifact_mask_follows_the_union():
    m = oct_mod._artifact_mask({"crop_bands": _OVERLAP}, 40, 30)
    assert m.shape == (40, 30) and m.dtype == bool
    assert m[15, :16].all() and not m[15, 16:].any()
    assert m[5, :11].all() and not m[5, 11:].any()
    assert m[25, 8:16].all() and not m[25, :8].any()
    assert not m[31].any()
    ab = oct_mod._artifact_bands({"crop_bands": _OVERLAP}, 30, 40)
    for lat in range(40):
        assert np.array_equal(np.flatnonzero(m[lat]), ab[lat])


def test_reconstruct_surface_walks_each_run_and_keeps_the_cornea_between_two_bands():
    nf = 40
    surf = np.tile(np.linspace(100.0, 140.0, nf, dtype=np.float32), (2, 1))
    surf[0, 5:10] = 300.0; surf[0, 20:25] = 300.0                   # the detector dived into two artifacts
    p = {"crop_bands": {"bands": [{"id": 1, "marks": {"0": [5, 9]}}, {"id": 2, "marks": {"0": [20, 24]}}]}}
    out = oct_mod._reconstruct_surface_over_bands(surf, p)
    assert out.shape == surf.shape and out.dtype == surf.dtype
    assert np.array_equal(out[1], surf[1])                           # untouched lateral
    assert np.array_equal(out[0, 10:20], surf[0, 10:20])             # cornea BETWEEN the bands is kept
    assert (out[0, 5:10] < 200).all() and (out[0, 20:25] < 200).all()
    # each run is a linear ramp between its OWN flank medians (5 frames a side, non-band frames only)
    assert np.all(np.diff(out[0, 5:10]) > 0) and np.all(np.diff(out[0, 20:25]) > 0)
    assert (out[0, 5:10] > 100).all() and (out[0, 5:10] < 120).all()
    assert (out[0, 20:25] > 115).all() and (out[0, 20:25] < 135).all()


# ── 5. persisted form / normaliser ───────────────────────────────────────────────────────────────────────────────
def test_normalize_legacy_becomes_explicit_band_1():
    assert oct_mod.normalize_crop_bands({"59": [39, 30], "0": [30, 39]}) == \
        {"bands": [{"id": 1, "marks": {"0": [30, 39], "59": [30, 39]}}]}


def test_normalize_explicit_is_sorted_and_idempotent():
    raw = {"bands": [{"id": 7, "marks": {"19": [12, 10], "0": [10, 12]}}, {"id": 3, "marks": {"4": [2, 4]}}]}
    n = oct_mod.normalize_crop_bands(raw)
    assert n == {"bands": [{"id": 3, "marks": {"4": [2, 4]}}, {"id": 7, "marks": {"0": [10, 12], "19": [10, 12]}}]}
    assert oct_mod.normalize_crop_bands(n) == n
    assert oct_mod.normalize_crop_bands(json.loads(json.dumps(n))) == n


def test_normalize_empty_and_unsupported_forms():
    assert oct_mod.normalize_crop_bands({}) == {}
    assert oct_mod.normalize_crop_bands(None) == {}
    assert oct_mod.normalize_crop_bands({"bands": []}) == {}
    assert oct_mod.normalize_crop_bands({"bands": [{"id": 1, "marks": {}}]}) == {}      # empty band dropped
    assert oct_mod.normalize_crop_bands({"3": [[1, 2], [5, 6]]}) == {}                   # intermediate form: not supported
    assert oct_mod.normalize_crop_bands({"bands": "junk"}) == {}


def test_normalize_ids_missing_or_duplicated_get_fresh_ids_never_merged():
    n = oct_mod.normalize_crop_bands({"bands": [{"id": 2, "marks": {"0": [1, 2]}}, {"marks": {"1": [1, 2]}},
                                                 {"id": 2, "marks": {"2": [1, 2]}}, {"id": 0, "marks": {"3": [1, 2]}}]})
    assert [b["id"] for b in n["bands"]] == [1, 2, 3, 4]
    assert [list(b["marks"]) for b in n["bands"]] == [["1"], ["0"], ["2"], ["3"]]


def test_grouping_knobs_and_auto_linking_are_gone():
    assert "crop_band_link_frames" not in oct_mod.DEFAULT_PARAMS
    assert "crop_band_max_gap" not in oct_mod.DEFAULT_PARAMS
    assert not hasattr(oct_mod, "_group_crop_marks")
    src = Path(oct_mod.__file__).read_text()
    for needle in ("crop_band_link_frames", "crop_band_max_gap", "_group_crop_marks", "_interp_crop_group"):
        assert needle not in src, needle
    import api_server
    api_src = Path(api_server.__file__).read_text()
    for needle in ("crop_band_link_frames", "crop_band_max_gap"):
        assert needle not in api_src, needle


# ── 6. api: request normalisation + persistence, cache signature ────────────────────────────────────────────────
def test_oct_marks_persists_explicit_form_and_accepts_legacy(client, make_case):
    cid = make_case("case_crop_bands_marks")
    legacy = {"59": [39, 30], "0": [30, 39]}
    want = {"bands": [{"id": 1, "marks": {"0": [30, 39], "59": [30, 39]}}]}
    r = client.post(f"/api/case/{cid}/oct-marks", json={"crop_bands": legacy})
    assert r.status_code == 200 and r.json()["changed"]["crop_bands"] == want
    assert orch.read_manifest(cid)["oct_params"]["crop_bands"] == want

    two = {"bands": [{"id": 2, "marks": {"300": [20, 30], "400": [20, 30]}}, {"id": 1, "marks": {"0": [20, 30], "100": [20, 30]}}]}
    r = client.post(f"/api/case/{cid}/oct-marks", json={"crop_bands": two})
    got = r.json()["changed"]["crop_bands"]
    assert [b["id"] for b in got["bands"]] == [1, 2]
    assert got["bands"][1]["marks"] == {"300": [20, 30], "400": [20, 30]}
    assert orch.read_manifest(cid)["oct_params"]["crop_bands"] == got

    r = client.post(f"/api/case/{cid}/oct-marks", json={"crop_bands": {"bands": []}})
    assert r.status_code == 200 and r.json()["changed"]["crop_bands"] is None
    assert "crop_bands" not in orch.read_manifest(cid)["oct_params"]

    client.post(f"/api/case/{cid}/oct-marks", json={"crop_bands": legacy})
    r = client.post(f"/api/case/{cid}/oct-marks", json={"crop_bands": {}})
    assert r.json()["changed"]["crop_bands"] is None
    assert "crop_bands" not in orch.read_manifest(cid)["oct_params"]


def test_oct_marks_none_leaves_crop_bands_untouched(client, make_case):
    cid = make_case("case_crop_bands_none")
    client.post(f"/api/case/{cid}/oct-marks", json={"crop_bands": {"0": [1, 2]}})
    client.post(f"/api/case/{cid}/oct-marks", json={"surface_crop_frames": [0]})
    assert orch.read_manifest(cid)["oct_params"]["crop_bands"] == {"bands": [{"id": 1, "marks": {"0": [1, 2]}}]}


def test_detect_params_sig_is_canonical_over_both_forms():
    import api_server
    legacy = {"59": [39, 30], "0": [30, 39]}
    explicit = {"bands": [{"id": 1, "marks": {"0": [30, 39], "59": [30, 39]}}]}
    s1 = api_server._detect_params_sig({"crop_bands": legacy})
    s2 = api_server._detect_params_sig({"crop_bands": explicit})
    assert s1 == s2 and ";crop_bands=" in s1
    assert "crop_band_link_frames" not in s1 and "crop_band_max_gap" not in s1
    s3 = api_server._detect_params_sig({"crop_bands": {"bands": [{"id": 2, "marks": {"0": [30, 39], "59": [30, 39]}}]}})
    assert s3 != s1                                                   # a different band id is a different geometry
    assert ";crop_bands=" not in api_server._detect_params_sig({})
    assert ";crop_bands=" not in api_server._detect_params_sig({"crop_bands": {}})


def test_edge_eligibility_excluded_frames_union_over_bands():
    p = {"crop_bands": {"bands": [{"id": 1, "marks": {"0": [30, 34], "59": [30, 34]}}, {"id": 2, "marks": {"0": [36, 39], "59": [36, 39]}}]}}
    elig, excl = oct_mod._edge_eligibility(p, 60, 40)
    assert excl["crop_bands"] == [30, 31, 32, 33, 34, 36, 37, 38, 39]
    assert elig[:, 35].all() and not elig[:, 30:35].any() and not elig[:, 36:40].any()


def test_run_provenance_brief_reads_both_forms():
    import run_provenance as rp
    assert rp._crop_bands_brief({"3": [1, 2]}).startswith("1 band(s), 1 mark(s): band 1 (1 marks) @ 3:[1,2]")
    two = {"bands": [{"id": 1, "marks": {"0": [1, 2], "9": [1, 2]}}, {"id": 2, "marks": {"4": [5, 6]}}]}
    assert rp._crop_bands_brief(two).startswith("2 band(s), 3 mark(s): band 1 (2 marks) @ 0:[1,2] 9:[1,2]; band 2 (1 marks) @ 4:[5,6]")
    assert rp._crop_bands_brief({"bands": []}) == "absent"
