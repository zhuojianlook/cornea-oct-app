"""Step 4 "Aligned" subgroups + approval (reviewer spec 2026-09-11): GET/POST /api/group/{gid}/subgroups, the per-subgroup
align jobs (`<patient>_<eye>_s<k>`), the group_aligned stamp on completion, POST /api/case/{id}/aligned-approve and the
step-4 reset flags. The job subprocess is monkeypatched (as in test_group_align_job.py); everything lives in the conftest
tempdir."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import api_server
import group_job
import orchestration as orch
import settings

ODS = ("case_cs001_od_v1", "case_cs001_od_v2", "case_cs001_od_v3")


def _make_eye(make_case):
    for i in (1, 2, 3):
        make_case(f"case_cs001_od_v{i}", manifest={"oct_source": f"/data/CS001_OD_3D Cornea ({i}).OCT",
                                                   "patient_id": "CS001", "eye": "OD", "preproc_vetted": True})
    make_case("case_cs001_os_v1", manifest={"oct_source": "/data/CS001_OS_3D Cornea.OCT", "patient_id": "CS001", "eye": "OS"})


class _FakeProc:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc


@pytest.fixture
def ws(tmp_path, monkeypatch):
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setattr(settings, "WORKSPACE_ROOT", root, raising=False)
    monkeypatch.setattr(api_server, "_ALIGN_PROCS", {}, raising=False)
    monkeypatch.setattr(api_server, "_ALIGN_QUEUE", [], raising=False)
    return root


def _fake_spawn(record, monkeypatch, rc=0, ts="2026-09-11T12:00:00"):
    """A job that 'finishes' at once: result.json lists exactly req.members (the sidecar's subgroup-aware list)."""
    def spawn(gid, d, req):
        record.append({"gid": gid, "members": list(req.members or []), "force": req.force})
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        members = list(req.members or [])
        res = {"group": gid, "reference": members[0], "member_ids": members, "engine_md5": group_job.engine_md5(),
               "timestamp": ts, "seconds": 1.0,
               "members": [{"cid": c, "is_reference": i == 0, "ok": True, "flags": [], "reject_flags": [], "overlay": None}
                           for i, c in enumerate(members)]}
        (d / group_job.RESULT_NAME).write_text(json.dumps(res))
        (d / group_job.PROGRESS_NAME).write_text(json.dumps({"phase": "done", "running": False, "done": True, "pid": None, "error": None}))
        return _FakeProc(rc)
    monkeypatch.setattr(api_server, "_align_spawn", spawn)
    return spawn


def _wait_stamp(client, gid, ts=None, n=60):
    """The stamp lands from the watcher thread or lazily on a status call — poll the status until it is on."""
    st = None
    for _ in range(n):
        st = client.get(f"/api/group/{gid}/align/status").json()
        s = st.get("stamp") or {}
        if s.get("stamped_timestamp") and (ts is None or s["stamped_timestamp"] == ts):
            return st
        time.sleep(0.05)
    return st


def test_subgroups_get_lists_scans_with_default_subgroup(client, make_case, ws):
    _make_eye(make_case)
    r = client.get("/api/group/CS001_OD/subgroups")
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["group"] == "cs001_od" and b["patient"] == "CS001" and b["eye"] == "OD"
    assert [s["case_id"] for s in b["scans"]] == list(ODS)
    assert all(s["subgroup"] == "1" and s["align_group"] == "cs001_od_s1" and not s["group_aligned"] and not s["aligned_approved"] for s in b["scans"])
    assert set(b["subgroups"]) == {"1"} and b["subgroups"]["1"]["members"] == list(ODS)
    assert b["subgroups"]["1"]["status"] is None          # no job dir yet
    assert client.get("/api/group/CS001_OD_s1/subgroups").json()["group"] == "cs001_od"   # a subgroup id resolves to its eye
    assert client.get("/api/group/nobody_od/subgroups").status_code == 404


def test_group_id_subgroup_suffix_parsing():
    assert api_server._align_base_and_sub("CS001_OD") == ("cs001_od", None)
    assert api_server._align_base_and_sub("cs001_od_s2") == ("cs001_od", "2")
    assert api_server._align_base_and_sub("p5_os") == ("p5_os", None)          # an eye 'OS' is not a subgroup suffix
    assert api_server._align_base_and_sub("x_s2_od_s1") == ("x_s2_od", "1")


def test_confirm_one_subgroup_aligns_and_stamps_group_aligned(client, make_case, ws, monkeypatch):
    _make_eye(make_case)
    rec = []
    _fake_spawn(rec, monkeypatch)
    before_os = orch.read_manifest("case_cs001_os_v1")
    r = client.post("/api/group/CS001_OD/subgroups", json={"assignments": {"case_cs001_od_v1": 1, "case_cs001_od_v2": "1", "case_cs001_od_v3": 1, "case_cs001_os_v1": 1}, "confirm": True})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["rejected"] == ["case_cs001_os_v1"]             # not this eye → never written
    assert orch.read_manifest("case_cs001_os_v1") == before_os
    assert b["subgroups"] == {c: "1" for c in ODS}
    assert len(b["jobs"]) == 1 and b["jobs"][0]["group"] == "cs001_od_s1" and b["jobs"][0]["started"] is True and b["skipped"] == []
    assert rec[0]["gid"] == "cs001_od_s1" and rec[0]["members"] == list(ODS)
    for c in ODS:
        m = orch.read_manifest(c)
        assert m["scar_subgroup"] == "1" and m["align_subgroup_confirmed"] is True
        assert m.get("subgroup_confirmed") is None            # the step-8 flag is NOT touched
    st = _wait_stamp(client, "cs001_od_s1")
    assert st["done"] is True and st["stamp"]["stamped_timestamp"] == "2026-09-11T12:00:00"
    for c in ODS:
        ga = orch.read_manifest(c)["group_aligned"]
        assert ga and ga["group"] == "cs001_od_s1" and ga["source"] == "align_job" and ga["result_timestamp"] == "2026-09-11T12:00:00"
    # the result's member records carry the live flags; the cases list carries the booleans
    res = client.get("/api/group/cs001_od_s1/align/result").json()
    assert all(m["group_aligned"] and m["aligned_approved"] is None for m in res["members"])
    assert res["approval"] == {"members": {c: False for c in ODS}, "all": False, "any": False}
    life = {c["case_id"]: c["life"] for c in client.get("/api/cases/list").json()["cases"]}
    assert life["case_cs001_od_v1"]["group_aligned"] is True and life["case_cs001_od_v1"]["aligned_approved"] is False
    assert life["case_cs001_od_v1"]["align_subgroup_confirmed"] is True
    # a plain re-POST with the result cached: no new job, still stamped, nothing re-run
    r2 = client.post("/api/group/CS001_OD/subgroups", json={"assignments": {}, "confirm": True}).json()
    assert r2["jobs"][0]["cached"] is True and len(rec) == 1


def test_two_subgroups_run_one_job_each_and_a_singleton_is_skipped(client, make_case, ws, monkeypatch):
    _make_eye(make_case)
    rec = []
    _fake_spawn(rec, monkeypatch)
    r = client.post("/api/group/cs001_od/subgroups", json={"assignments": {"case_cs001_od_v1": 1, "case_cs001_od_v2": 1, "case_cs001_od_v3": 2}})
    assert r.status_code == 200, r.text
    b = r.json()
    assert [j["group"] for j in b["jobs"]] == ["cs001_od_s1"] and b["jobs"][0]["started"] is True
    assert len(b["skipped"]) == 1 and b["skipped"][0]["subgroup"] == "2" and b["skipped"][0]["members"] == ["case_cs001_od_v3"]
    assert "single scan" in b["skipped"][0]["reason"]
    assert rec[0]["members"] == ["case_cs001_od_v1", "case_cs001_od_v2"]
    _wait_stamp(client, "cs001_od_s1")
    assert orch.read_manifest("case_cs001_od_v1")["group_aligned"]["group"] == "cs001_od_s1"
    assert orch.read_manifest("case_cs001_od_v3").get("group_aligned") is None      # the singleton stays at Vetted
    assert orch.read_manifest("case_cs001_od_v3")["scar_subgroup"] == "2"
    # subgroup-aware member resolution
    assert api_server._group_members("cs001_od_s1")[0] == ["case_cs001_od_v1", "case_cs001_od_v2"]
    assert api_server._group_members("cs001_od_s2")[0] == ["case_cs001_od_v3"]
    assert api_server._group_members("cs001_od")[0] == list(ODS)
    g = client.get("/api/group/cs001_od/subgroups").json()
    assert set(g["subgroups"]) == {"1", "2"} and g["subgroups"]["2"]["alignable"] is False
    # two alignable subgroups: the second is QUEUED behind the first when the first still runs
    api_server._ALIGN_PROCS.clear()
    rec2 = []
    def spawn_slow(gid, d, req):
        rec2.append(gid); Path(d).mkdir(parents=True, exist_ok=True)
        return _FakeProc(None)                                   # never finishes during the test
    monkeypatch.setattr(api_server, "_align_spawn", spawn_slow)
    orch.write_manifest_value("case_cs001_od_v3", {"scar_subgroup": "2"})
    make_case("case_cs001_od_v4", manifest={"oct_source": "/data/CS001_OD_3D Cornea (4).OCT", "patient_id": "CS001", "eye": "OD", "scar_subgroup": "2"})
    r = client.post("/api/group/cs001_od/subgroups", json={"assignments": {}, "force": True}).json()
    kinds = {j["group"]: ("started" if j.get("started") else "queued" if j.get("queued") else "?") for j in r["jobs"]}
    assert kinds == {"cs001_od_s1": "started", "cs001_od_s2": "queued"} and rec2 == ["cs001_od_s1"]
    assert client.get("/api/group/cs001_od/subgroups").json()["queued"] == ["cs001_od_s2"]


def test_aligned_approve_requires_alignment_then_records_and_reset_clears(client, make_case, ws, monkeypatch):
    _make_eye(make_case)
    r = client.post("/api/case/case_cs001_od_v1/aligned-approve", json={"approve": True})
    assert r.status_code == 400
    rec = []
    _fake_spawn(rec, monkeypatch)
    client.post("/api/group/cs001_od/subgroups", json={"assignments": {}})
    _wait_stamp(client, "cs001_od_s1")
    r = client.post("/api/case/case_cs001_od_v1/aligned-approve", json={"approve": True, "note": "meets consensus"})
    assert r.status_code == 200, r.text
    ap = r.json()["aligned_approved"]
    assert ap["note"] == "meets consensus" and ap["group"] == "cs001_od_s1" and ap["result_timestamp"] == "2026-09-11T12:00:00"
    assert orch.read_manifest("case_cs001_od_v1")["aligned_approved_at"] == ap["ts"]
    res = client.get("/api/group/cs001_od_s1/align/result").json()
    assert res["approval"]["members"]["case_cs001_od_v1"] is True and res["approval"]["any"] is True and res["approval"]["all"] is False
    # withdraw
    r = client.post("/api/case/case_cs001_od_v1/aligned-approve", json={"approve": False})
    assert r.json()["aligned_approved"] is None and orch.read_manifest("case_cs001_od_v1").get("aligned_approved_at") is None
    # approve again, then rolling back to Vetted (step 3) clears the step-4 flags together
    client.post("/api/case/case_cs001_od_v1/aligned-approve", json={"approve": True})
    assert api_server._STEP_RESET_FLAGS[4] == ["group_aligned", "aligned_approved", "aligned_approved_at", "align_subgroup_confirmed"]
    r = client.post("/api/case/case_cs001_od_v1/reset-step", json={"step": 3})
    assert r.status_code == 200
    m = orch.read_manifest("case_cs001_od_v1")
    assert m.get("group_aligned") is None and m.get("aligned_approved") is None and m.get("align_subgroup_confirmed") is None
    assert api_server._case_subgroup(m) == "1"              # the label itself is data (default 1), not a step flag


def test_rerun_with_a_new_result_voids_the_approval(client, make_case, ws, monkeypatch):
    _make_eye(make_case)
    rec = []
    _fake_spawn(rec, monkeypatch, ts="2026-09-11T12:00:00")
    client.post("/api/group/cs001_od/subgroups", json={"assignments": {}})
    _wait_stamp(client, "cs001_od_s1")
    for c in ODS:
        client.post(f"/api/case/{c}/aligned-approve", json={"approve": True})
    assert client.get("/api/group/cs001_od_s1/align/result").json()["approval"]["all"] is True
    # the panel's force re-run of the SUBGROUP id → a new result timestamp → stamped again, approval voided
    _fake_spawn(rec, monkeypatch, ts="2026-09-11T13:00:00")
    r = client.post("/api/group/cs001_od_s1/align", json={"force": True})
    assert r.status_code == 200 and r.json()["started"] is True
    st = _wait_stamp(client, "cs001_od_s1", ts="2026-09-11T13:00:00")
    assert st["stamp"]["stamped_timestamp"] == "2026-09-11T13:00:00"
    for c in ODS:
        m = orch.read_manifest(c)
        assert m["group_aligned"]["result_timestamp"] == "2026-09-11T13:00:00" and m.get("aligned_approved") is None
    # manual clear of group_aligned clears the approval too
    client.post("/api/case/case_cs001_od_v1/aligned-approve", json={"approve": True})
    client.post("/api/case/case_cs001_od_v1/group-aligned", json={"aligned": False})
    m = orch.read_manifest("case_cs001_od_v1")
    assert m.get("group_aligned") is None and m.get("aligned_approved") is None


def test_plain_align_of_the_eye_still_touches_no_manifest(client, make_case, ws, monkeypatch):
    """The legacy path (POST /align on the plain eye id, never set up through /subgroups) stamps nothing."""
    _make_eye(make_case)
    rec = []
    _fake_spawn(rec, monkeypatch)
    before = {c: orch.read_manifest(c) for c in ODS}
    r = client.post("/api/group/CS001_OD/align", json={})
    assert r.status_code == 200 and r.json()["started"] is True
    time.sleep(0.2)
    st = client.get("/api/group/cs001_od/align/status").json()
    assert st["done"] is True and st["stamp"] is None
    for c, m in before.items():
        assert orch.read_manifest(c) == m
