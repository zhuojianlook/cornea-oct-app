"""Lifecycle step-table + endpoint tests for the 13-step timeline.

Step 4 "Aligned" (manifest.group_aligned) = the group-wise 3D alignment of a patient+eye group's replicate
scans (regularising their sagittal curvatures); it sits BETWEEN Vetted (3) and Cornea/SAM2 (5). The old scar
consensus step is now 10 "Scar-aligned". Mirrors cornea_app/src/api/lifecycle.ts (LIFECYCLE_STEPS / scanStep)
and its node:test twin tests/unit/lifecycle.test.ts.

Covers: _STEP_RESET_FLAGS shape (mirrors scanStep), reset-step around the new step, the STUB
POST /api/group/{gid}/align (validates the group, reports members, writes NOTHING) and the manifest-only
POST /api/case/{id}/group-aligned, plus cases/list carrying `life.group_aligned`.
All state lives in the conftest tempdir (client/make_case fixtures) — never the real store.
"""
from __future__ import annotations

import orchestration as orch
import labels


# ── the step table mirrors api/lifecycle.ts ─────────────────────────────────
def test_step_table_mirrors_lifecycle_ts(client):
    import api_server
    t = api_server._STEP_RESET_FLAGS
    assert sorted(t) == list(range(2, 14))
    assert api_server._MAX_STEP == 13
    assert t[2] == ["oct_preprocessed", "oct_iter"]
    assert t[3] == ["preproc_vetted"]
    # Aligned (group curvature) + its step-4 state: the reviewer's approval of the applied axial changes and the
    # subgroup confirmation the alignment was started under (2026-09-11), all cleared together
    assert t[4] == ["group_aligned", "aligned_approved", "aligned_approved_at", "align_subgroup_confirmed"]
    assert "sam2_meta" in t[5]                            # Cornea (SAM2) — still the labelmap boundary
    assert t[6] == ["cornea_vetted"]
    assert "scar_classification" in t[7]                  # Classified AFTER cornea vet (as scanStep orders it)
    assert t[8] == ["subgroup_confirmed"]
    assert "scar_done" in t[9]
    assert "consensus_case" in t[10]                      # Scar-aligned
    assert "normalized" in t[11]
    assert t[12] == ["corrected_labelmap"]
    assert t[13] == ["training_scheduled"]
    # Every flag belongs to exactly one step.
    flags = [f for keys in t.values() for f in keys]
    assert len(flags) == len(set(flags))


# ── reset-step around the new step ──────────────────────────────────────────
def test_reset_to_aligned_keeps_group_aligned_clears_sam2_and_classification(client, make_case):
    cid = make_case("case_reset_aligned", manifest={
        "preproc_vetted": True,                                   # 3
        "group_aligned": {"ts": "t", "group": "cs001_od"},        # 4
        "sam2_meta": {"vote": 2},                                 # 5
        "cornea_vetted": True,                                    # 6
        "scar_classification": "scar",                            # 7
    })
    r = client.post(f"/api/case/{cid}/reset-step", json={"step": 4})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True and body["step"] == 4
    m = orch.read_manifest(cid)
    assert m.get("preproc_vetted") is True
    assert m.get("group_aligned") == {"ts": "t", "group": "cs001_od"}   # step 4 itself is kept
    assert m.get("sam2_meta") is None
    assert m.get("cornea_vetted") is None
    assert m.get("scar_classification") is None
    assert "group_aligned" not in body["cleared"]
    assert "sam2_meta" in body["cleared"]
    # Below SAM2 (target < 5) → the on-disk labelmap goes too.
    assert not labels.corrected_path(cid).exists()


def test_reset_to_vetted_clears_group_aligned(client, make_case):
    cid = make_case("case_reset_vetted", manifest={"preproc_vetted": True, "group_aligned": {"ts": "t"}})
    r = client.post(f"/api/case/{cid}/reset-step", json={"step": 3})
    assert r.status_code == 200
    assert "group_aligned" in r.json()["cleared"]
    m = orch.read_manifest(cid)
    assert m.get("group_aligned") is None
    assert m.get("preproc_vetted") is True


def test_reset_to_cornea_vet_clears_classification_keeps_labelmap(client, make_case):
    cid = make_case("case_reset_cv", manifest={
        "sam2_meta": {"vote": 2}, "cornea_vetted": True, "scar_classification": "scar", "subgroup_confirmed": True,
    })
    r = client.post(f"/api/case/{cid}/reset-step", json={"step": 6})
    assert r.status_code == 200
    m = orch.read_manifest(cid)
    assert m.get("sam2_meta") == {"vote": 2}
    assert m.get("cornea_vetted") is True
    assert m.get("scar_classification") is None
    assert m.get("subgroup_confirmed") is None
    assert labels.corrected_path(cid).exists()


def test_reset_step_range_is_1_to_13(client, make_case):
    cid = make_case("case_reset_13", manifest={"training_scheduled": True})
    assert client.post(f"/api/case/{cid}/reset-step", json={"step": 13}).status_code == 200
    assert orch.read_manifest(cid).get("training_scheduled") is True   # 13 is the last step: nothing after it
    assert client.post(f"/api/case/{cid}/reset-step", json={"step": 14}).status_code == 400
    assert client.post(f"/api/case/{cid}/reset-step", json={"step": 0}).status_code == 400


# ── POST /api/group/{gid}/align — now the minimal job (tests/test_group_align_job.py); here: membership + 404s ──
def _make_group(make_case):
    """Two OD replicates + one OS scan of the same patient, a consensus case, and a non-OCT case."""
    make_case("case_cs001_od_v1", manifest={"oct_source": "/data/CS001_OD_3D Cornea.OCT",
                                            "patient_id": "CS001", "eye": "OD", "preproc_vetted": True})
    make_case("case_cs001_od_v2", manifest={"oct_source": "/data/CS001_OD_3D Cornea (1).OCT",
                                            "patient_id": "CS001", "eye": "OD"})
    make_case("case_cs001_os_v1", manifest={"oct_source": "/data/CS001_OS_3D Cornea.OCT",
                                            "patient_id": "CS001", "eye": "OS"})
    make_case("case_cs001_od_consensus", manifest={"oct_source": "/data/CS001_OD_3D Cornea.OCT",
                                                   "patient_id": "CS001", "eye": "OD",
                                                   "consensus_cases": ["case_cs001_od_v1", "case_cs001_od_v2"]})
    make_case("case_registered_only")   # no oct_source → not an OCT-loader case, never a member


def test_group_align_status_reports_members_and_writes_nothing(client, make_case):
    _make_group(make_case)
    ids = ("case_cs001_od_v1", "case_cs001_od_v2", "case_cs001_os_v1", "case_cs001_od_consensus", "case_registered_only")
    before = {c: orch.read_manifest(c) for c in ids}
    r = client.get("/api/group/CS001_OD/align/status")
    assert r.status_code == 200
    body = r.json()
    assert body["members"] == ["case_cs001_od_v1", "case_cs001_od_v2"]
    assert body["patient"] == "CS001" and body["eye"] == "OD"
    assert body["group"] == "cs001_od"
    assert body["running"] is False and body["result_exists"] is False
    # NOTHING written: every manifest is byte-for-byte what it was, and no group_aligned appeared.
    for c in ids:
        assert orch.read_manifest(c) == before[c], c
        assert orch.read_manifest(c).get("group_aligned") is None, c


def test_group_align_id_is_case_insensitive_and_separator_tolerant(client, make_case):
    _make_group(make_case)
    for gid in ("cs001_od", "CS001|OD", "cs001%20od"):   # a "/" cannot be a path segment; never in an id
        r = client.get(f"/api/group/{gid}/align/status")
        assert r.status_code == 200, gid
        assert r.json()["members"] == ["case_cs001_od_v1", "case_cs001_od_v2"], gid
    r = client.get("/api/group/CS001_OS/align/status")
    assert r.status_code == 200
    assert r.json()["members"] == ["case_cs001_os_v1"]


def test_group_align_uses_the_cases_list_grouping_rule(client, make_case):
    _make_group(make_case)
    rows = client.get("/api/cases/list").json()["cases"]
    expect = sorted(c["case_id"] for c in rows
                    if (c["patient"] or "").lower() == "cs001" and (c["eye"] or "").upper() == "OD")
    assert expect == ["case_cs001_od_v1", "case_cs001_od_v2"]   # the consensus + non-OCT cases are not listed
    assert client.get("/api/group/CS001_OD/align/status").json()["members"] == expect


def test_group_align_filename_parsed_identity_matches_cases_list(client, make_case):
    # No patient_id/eye in the manifest → the source filename is parsed, exactly as cases/list does.
    cid = make_case("case_parsed", manifest={"oct_source": "/data/CS007_12345_3D Cornea_OS_2026-01-02.OCT"})
    row = next(c for c in client.get("/api/cases/list").json()["cases"] if c["case_id"] == cid)
    assert row["patient"] == "CS007" and row["eye"] == "OS"
    r = client.get(f"/api/group/{row['patient']}_{row['eye']}/align/status")
    assert r.status_code == 200
    assert r.json()["members"] == [cid]


def test_group_align_unknown_group_404(client, make_case):
    _make_group(make_case)
    assert client.post("/api/group/CS999_OD/align").status_code == 404
    assert client.post("/api/group/_/align").status_code == 404
    # A consensus id / a bare patient id is not a patient+eye group either.
    assert client.post("/api/group/case_cs001_od_consensus/align").status_code == 404
    assert client.post("/api/group/CS001/align").status_code == 404


# ── POST /api/case/{id}/group-aligned (manifest-only) ───────────────────────
def _life(client, cid):
    return next(c for c in client.get("/api/cases/list").json()["cases"] if c["case_id"] == cid)["life"]


def test_group_aligned_set_and_clear(client, make_case):
    cid = make_case("case_ga", manifest={"oct_source": "/data/CS001_OD_3D Cornea.OCT",
                                         "patient_id": "CS001", "eye": "OD", "preproc_vetted": True})
    assert _life(client, cid)["group_aligned"] is False
    r = client.post(f"/api/case/{cid}/group-aligned", json={"aligned": True, "note": "manual check"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    ga = body["group_aligned"]
    assert ga["group"] == "cs001_od"
    assert ga["note"] == "manual check"
    assert ga["source"] == "manual"
    assert ga["ts"]
    assert orch.read_manifest(cid).get("group_aligned") == ga
    # cases/list carries the step-4 flag (as a boolean) so the sidebar colours the row "Aligned".
    assert _life(client, cid)["group_aligned"] is True
    # Nothing else moved.
    m = orch.read_manifest(cid)
    assert m.get("preproc_vetted") is True and m.get("sam2_meta") is None
    # Clear.
    r2 = client.post(f"/api/case/{cid}/group-aligned", json={"aligned": False})
    assert r2.status_code == 200
    assert r2.json() == {"ok": True, "group_aligned": None}
    assert orch.read_manifest(cid).get("group_aligned") is None
    assert _life(client, cid)["group_aligned"] is False


def test_group_aligned_defaults_and_404(client, make_case):
    cid = make_case("case_ga_default")
    r = client.post(f"/api/case/{cid}/group-aligned", json={})
    assert r.status_code == 200
    ga = r.json()["group_aligned"]
    assert ga is not None and ga["note"] is None and ga["group"] is None   # no patient/eye resolvable here
    # Unknown case → 404 and NO ghost case dir materialised.
    assert client.post("/api/case/case_ga_missing/group-aligned", json={"aligned": True}).status_code == 404
    assert not orch.case_root("case_ga_missing").exists()
