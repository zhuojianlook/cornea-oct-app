"""PIPELINE_VERSION stamp + run_is_current + the pipeline_* fields the app reads on case open.

Why: the reviewer opened case_cs011_od_v3 through the approve queue, saw a result from an OLDER flatten and
reported it as a regression. The app now re-runs an un-approved scan on open when its last run was not made
by the pipeline that is running; these tests pin the backend half of that contract:
  * oct_preprocess.PIPELINE_VERSION is a non-empty string and preprocess_oct_to_nifti stamps it into the
    run record (checked by grepping the return sites — a full run needs a real .OCT).
  * run_is_current(manifest) -> (bool, reason): stamped-current / unstamped / mismatched / no oct_iter /
    not preprocessed / kept-raw.
  * POST /api/case and GET /api/case/{id} carry pipeline_current, pipeline_version_run,
    pipeline_version_now, pipeline_reason at top level.
  * GET /api/cases/pipeline-status lists every OCT case with {current, vetted}.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import oct_preprocess as oct_mod

_SIDE = Path(__file__).resolve().parent.parent


def _api():
    import api_server
    return api_server


# ── the constant + the stamp ────────────────────────────────────────────────
def test_pipeline_version_is_a_nonempty_string():
    assert isinstance(oct_mod.PIPELINE_VERSION, str)
    assert oct_mod.PIPELINE_VERSION.strip()
    assert oct_mod.PIPELINE_VERSION == "2026-09-07.tissue-v3"


def test_every_return_of_preprocess_oct_to_nifti_stamps_the_record():
    """Static guard: each `return info` inside preprocess_oct_to_nifti is immediately preceded by the stamp,
    and the worker's ITER print setdefaults it. Running the real function needs a .OCT file."""
    src = (_SIDE / "oct_preprocess.py").read_text()
    start = src.index("def preprocess_oct_to_nifti(")
    # the function ends where the next top-level def/class starts
    m = re.search(r"\n(def |class )", src[start + 10:])
    body = src[start:start + 10 + (m.start() if m else len(src))]
    returns = [mm.start() for mm in re.finditer(r"\n\s+return info\n", body)]
    assert len(returns) >= 3, "expected the surface-crop, corrections and auto return sites"
    for pos in returns:
        window = body[max(0, pos - 200):pos]
        assert 'info["pipeline_version"] = PIPELINE_VERSION' in window, body[pos - 300:pos + 20]
    assert '_info.setdefault("pipeline_version", PIPELINE_VERSION)' in src


def test_json_safe_keeps_the_stamp():
    rec = oct_mod._json_safe({"passes": 1, "pipeline_version": oct_mod.PIPELINE_VERSION, "_private": object()})
    assert rec["pipeline_version"] == oct_mod.PIPELINE_VERSION
    assert "_private" not in rec


# ── run_is_current (pure) ───────────────────────────────────────────────────
def test_run_is_current_stamped_with_running_version():
    api = _api()
    ok, why = api.run_is_current({"oct_preprocessed": True,
                                  "oct_iter": {"passes": 1, "stopped": "redetect",
                                               "pipeline_version": oct_mod.PIPELINE_VERSION}})
    assert ok is True and why == "current"


def test_run_is_current_unstamped_old_run_is_stale():
    api = _api()
    # a real pre-2026-09-07 record shape: tissue flatten, judged second pass, but NO pipeline_version
    ok, why = api.run_is_current({"oct_preprocessed": True,
                                  "oct_iter": {"passes": 1, "stopped": "redetect",
                                               "flatten": {"mode": "tissue"}, "tissue_motion": {"passes": 2}}})
    assert ok is False and why == "unstamped"


def test_run_is_current_mismatched_version_is_stale():
    api = _api()
    ok, why = api.run_is_current({"oct_preprocessed": True,
                                  "oct_iter": {"passes": 1, "pipeline_version": "2026-01-01.older"}})
    assert ok is False and why == "mismatch"


def test_run_is_current_no_oct_iter_at_all():
    api = _api()
    ok, why = api.run_is_current({"oct_preprocessed": True})          # cohort load-dir import shape
    assert ok is False and why == "no_oct_iter"
    ok, why = api.run_is_current({"oct_preprocessed": True, "oct_iter": None})
    assert ok is False and why == "no_oct_iter"
    ok, why = api.run_is_current({})                                    # never preprocessed
    assert ok is False and why == "not_preprocessed"
    ok, why = api.run_is_current(None)
    assert ok is False and why == "not_preprocessed"


def test_run_is_current_kept_raw_is_not_stale():
    """'Use original' is a reviewer decision, not a pipeline output — an auto re-run must never undo it."""
    api = _api()
    ok, why = api.run_is_current({"oct_preprocessed": True, "oct_kept_raw": True,
                                  "oct_iter": {"passes": 0, "best_pass": 0, "metrics": [], "stopped": "kept_raw"}})
    assert ok is True and why == "kept_raw"


def test_parse_iter_info_fallback_is_deliberately_stale():
    """A worker that printed no ITER line yields a record with no stamp → the run reads as stale
    (the app's once-per-session guard bounds the re-run; a silent 'current' would hide a broken worker)."""
    api = _api()
    rec = api._parse_iter_info("OK /tmp/x.nii.gz\n")
    assert "pipeline_version" not in rec
    assert api.run_is_current({"oct_preprocessed": True, "oct_iter": rec}) == (False, "unstamped")
    rec2 = api._parse_iter_info('ITER {"passes": 1, "pipeline_version": "%s"}\nOK x\n' % oct_mod.PIPELINE_VERSION)
    assert api.run_is_current({"oct_preprocessed": True, "oct_iter": rec2}) == (True, "current")


# ── the case-open response (POST /api/case = what caseStore.openCase reads; GET mirrors it) ──
def test_case_open_response_carries_pipeline_fields_current(client, make_case):
    cid = make_case("case_pv_current", manifest={
        "oct_source": "/data/CS011_OD_3D Cornea.OCT",
        "oct_iter": {"passes": 1, "stopped": "redetect", "flatten": {"mode": "tissue"},
                     "tissue_motion": {"passes": 2}, "pipeline_version": oct_mod.PIPELINE_VERSION},
        "preproc_vetted": True,
    })
    r = client.post("/api/case", json={"case_id": cid})
    assert r.status_code == 200
    body = r.json()
    assert body["case_id"] == cid
    assert body["pipeline_current"] is True
    assert body["pipeline_version_run"] == oct_mod.PIPELINE_VERSION
    assert body["pipeline_version_now"] == oct_mod.PIPELINE_VERSION
    assert body["pipeline_reason"] == "current"
    # the manifest is still nested unchanged (the app reads manifest.preproc_vetted next to these)
    assert body["manifest"]["preproc_vetted"] is True

    g = client.get(f"/api/case/{cid}")
    assert g.status_code == 200
    for k in ("pipeline_current", "pipeline_version_run", "pipeline_version_now", "pipeline_reason"):
        assert g.json()[k] == body[k]


def test_case_open_response_stale_unstamped_and_unvetted(client, make_case):
    # textbook auto re-run target (case_cs010_os_v3 shape): old auto run, no stamp, not approved
    cid = make_case("case_pv_stale", manifest={
        "oct_source": "/data/CS010_OS_3D Cornea.OCT",
        "oct_iter": {"passes": 3, "stopped": "diminishing", "flatten": None},
        "preproc_vetted": False,
    })
    body = client.post("/api/case", json={"case_id": cid}).json()
    assert body["pipeline_current"] is False
    assert body["pipeline_version_run"] is None
    assert body["pipeline_version_now"] == oct_mod.PIPELINE_VERSION
    assert body["pipeline_reason"] == "unstamped"
    assert body["manifest"]["preproc_vetted"] is False


def test_case_open_response_mismatch_and_no_iter(client, make_case):
    c1 = make_case("case_pv_mismatch", manifest={
        "oct_source": "/data/a.OCT", "oct_iter": {"passes": 1, "pipeline_version": "2000-01-01.v0"}})
    b1 = client.post("/api/case", json={"case_id": c1}).json()
    assert (b1["pipeline_current"], b1["pipeline_reason"], b1["pipeline_version_run"]) == (False, "mismatch", "2000-01-01.v0")

    c2 = make_case("case_pv_noiter", manifest={"oct_source": "/data/b.OCT"})   # make_case sets oct_preprocessed True
    b2 = client.post("/api/case", json={"case_id": c2}).json()
    assert (b2["pipeline_current"], b2["pipeline_reason"], b2["pipeline_version_run"]) == (False, "no_oct_iter", None)


def test_case_open_brand_new_case_is_not_preprocessed(client):
    body = client.post("/api/case", json={"case_id": "case_pv_brand_new"}).json()
    assert body["pipeline_current"] is False
    assert body["pipeline_reason"] == "not_preprocessed"
    assert body["pipeline_version_run"] is None


# ── GET /api/cases/pipeline-status ──────────────────────────────────────────
def test_cases_pipeline_status_lists_every_oct_case(client, make_case):
    make_case("case_ps_cur_vet", manifest={
        "oct_source": "/data/a.OCT", "preproc_vetted": True,
        "oct_iter": {"passes": 1, "pipeline_version": oct_mod.PIPELINE_VERSION}})
    make_case("case_ps_cur_unvet", manifest={
        "oct_source": "/data/b.OCT", "preproc_vetted": False,
        "oct_iter": {"passes": 1, "pipeline_version": oct_mod.PIPELINE_VERSION}})
    make_case("case_ps_stale_vet", manifest={
        "oct_source": "/data/c.OCT", "preproc_vetted": True, "oct_iter": {"passes": 2, "stopped": "grew"}})
    make_case("case_ps_stale_unvet", manifest={
        "oct_source": "/data/d.OCT", "preproc_vetted": False, "oct_iter": {"passes": 2, "stopped": "diminishing"}})
    make_case("case_ps_kept_raw", manifest={
        "oct_source": "/data/e.OCT", "preproc_vetted": False, "oct_kept_raw": True,
        "oct_iter": {"passes": 0, "best_pass": 0, "metrics": [], "stopped": "kept_raw"}})
    make_case("case_ps_not_oct")      # no oct_source → not listed (same rule as /api/cases/list)

    r = client.get("/api/cases/pipeline-status")
    assert r.status_code == 200
    body = r.json()
    assert body["pipeline_version_now"] == oct_mod.PIPELINE_VERSION
    rows = {c["case_id"]: c for c in body["cases"]}
    assert "case_ps_not_oct" not in rows
    assert set(rows) == {"case_ps_cur_vet", "case_ps_cur_unvet", "case_ps_stale_vet", "case_ps_stale_unvet",
                         "case_ps_kept_raw"}
    for cid, cur, vet in [("case_ps_cur_vet", True, True), ("case_ps_cur_unvet", True, False),
                          ("case_ps_stale_vet", False, True), ("case_ps_stale_unvet", False, False),
                          ("case_ps_kept_raw", True, False)]:
        assert rows[cid]["current"] is cur, cid
        assert rows[cid]["vetted"] is vet, cid
        assert rows[cid]["preprocessed"] is True
    assert rows["case_ps_cur_vet"]["version_run"] == oct_mod.PIPELINE_VERSION
    assert rows["case_ps_stale_vet"]["version_run"] is None
    assert rows["case_ps_stale_unvet"]["reason"] == "unstamped"
    assert rows["case_ps_kept_raw"]["reason"] == "kept_raw"
    # exactly the scans the app will re-run on open
    assert body["n_stale_unvetted"] == 1


def test_cases_pipeline_status_empty_root(client, cases_root):
    r = client.get("/api/cases/pipeline-status")
    assert r.status_code == 200
    assert r.json()["cases"] == []
    assert r.json()["n_stale_unvetted"] == 0
