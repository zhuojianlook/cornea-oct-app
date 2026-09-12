"""The MINIMAL Align-group job endpoints (api_server ↔ group_job.py): POST /api/group/{gid}/align starts a subprocess
(monkeypatched here — the engine itself is covered by test_group_align.py), GET …/align/status, …/align/result,
…/align/overlay/{member}. Outputs live under settings.WORKSPACE_ROOT/groups/<gid>/align_min; NO case manifest is
touched (group_aligned stays the consensus step's job). All state lives in the conftest tempdir."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import api_server
import group_job
import orchestration as orch
import settings

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


def _make_group(make_case):
    make_case("case_cs001_od_v1", manifest={"oct_source": "/data/CS001_OD_3D Cornea.OCT",
                                            "patient_id": "CS001", "eye": "OD", "preproc_vetted": True})
    make_case("case_cs001_od_v2", manifest={"oct_source": "/data/CS001_OD_3D Cornea (1).OCT",
                                            "patient_id": "CS001", "eye": "OD"})
    make_case("case_cs001_os_v1", manifest={"oct_source": "/data/CS001_OS_3D Cornea.OCT",
                                            "patient_id": "CS001", "eye": "OS"})
    make_case("case_cs001_od_consensus", manifest={"oct_source": "/data/CS001_OD_3D Cornea.OCT",
                                                   "patient_id": "CS001", "eye": "OD",
                                                   "consensus_cases": ["case_cs001_od_v1", "case_cs001_od_v2"]})
    make_case("case_registered_only")


class _FakeProc:
    def __init__(self, rc):
        self.rc = rc

    def poll(self):
        return self.rc


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A fresh workspace root (groups/ lands under it) and no leftover job handles."""
    root = tmp_path / "ws"
    root.mkdir()
    monkeypatch.setattr(settings, "WORKSPACE_ROOT", root, raising=False)
    monkeypatch.setattr(api_server, "_ALIGN_PROCS", {}, raising=False)
    return root


def _fake_spawn(record, rc=0, write_result=True, monkeypatch=None):
    def spawn(gid, d, req):
        record.append({"gid": gid, "dir": Path(d), "force": req.force, "transitivity": req.transitivity})
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        if write_result:
            members = ["case_cs001_od_v1", "case_cs001_od_v2"]
            res = {"group": gid, "reference": members[0], "member_ids": members, "engine_md5": group_job.engine_md5(),
                   "timestamp": "2026-09-11T12:00:00", "seconds": 1.5,
                   "members": [{"cid": members[0], "is_reference": True, "ok": True, "flags": [], "reject_flags": [], "overlay": None},
                               {"cid": members[1], "is_reference": False, "df": 3, "dx_median": -2.5, "rel_struct": 1.01,
                                "rel_speckle": 1.2, "coverage": 0.8, "ok": True, "flags": ["arbitrated"], "reject_flags": [],
                                "overlay": group_job.overlay_name(members[1])}]}
            (d / group_job.RESULT_NAME).write_text(json.dumps(res))
            (d / group_job.overlay_name(members[1])).write_bytes(PNG)
            (d / group_job.PROGRESS_NAME).write_text(json.dumps({"phase": "done", "running": False, "done": True, "pid": None, "error": None}))
        return _FakeProc(rc)
    monkeypatch.setattr(api_server, "_align_spawn", spawn)
    return spawn


def test_align_starts_job_writes_under_groups_and_touches_no_manifest(client, make_case, ws, monkeypatch):
    _make_group(make_case)
    ids = ("case_cs001_od_v1", "case_cs001_od_v2", "case_cs001_os_v1", "case_cs001_od_consensus", "case_registered_only")
    before = {c: orch.read_manifest(c) for c in ids}
    rec = []
    _fake_spawn(rec, monkeypatch=monkeypatch)
    r = client.post("/api/group/CS001_OD/align", json={})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["ok"] is True and b["started"] is True
    assert b["members"] == ["case_cs001_od_v1", "case_cs001_od_v2"]
    assert b["group"] == "cs001_od" and b["patient"] == "CS001" and b["eye"] == "OD"
    assert rec and rec[0]["gid"] == "cs001_od" and rec[0]["force"] is False and rec[0]["transitivity"] is False
    assert rec[0]["dir"] == ws / "groups" / "cs001_od" / "align_min"
    # the job "finished" (fake proc rc 0, result written): status + result + overlay all served
    st = client.get("/api/group/cs001_od/align/status").json()
    assert st["running"] is False and st["done"] is True and st["result_exists"] is True and st["error"] is None
    res = client.get("/api/group/CS001|OD/align/result").json()
    assert res["reference"] == "case_cs001_od_v1" and res["overlay_url_base"] == "/api/group/cs001_od/align/overlay/"
    assert [m["cid"] for m in res["members"]] == ["case_cs001_od_v1", "case_cs001_od_v2"]
    ov = client.get("/api/group/cs001_od/align/overlay/case_cs001_od_v2")
    assert ov.status_code == 200 and ov.headers["content-type"].startswith("image/png") and ov.content == PNG
    assert client.get("/api/group/cs001_od/align/overlay/case_cs001_od_v1").status_code == 404
    # NOTHING written to any case: manifests byte-for-byte as before, no group_aligned
    for c in ids:
        assert orch.read_manifest(c) == before[c], c
        assert orch.read_manifest(c).get("group_aligned") is None, c
    assert not (settings.CASES_ROOT / "groups").exists()


def test_align_cached_unless_forced_and_previous_result_kept(client, make_case, ws, monkeypatch):
    _make_group(make_case)
    rec = []
    _fake_spawn(rec, monkeypatch=monkeypatch)
    assert client.post("/api/group/cs001_od/align", json={}).json()["started"] is True
    r = client.post("/api/group/cs001_od/align", json={}).json()
    assert r["started"] is False and r["cached"] is True and len(rec) == 1
    r = client.post("/api/group/cs001_od/align", json={"force": True, "transitivity": True}).json()
    assert r["started"] is True and len(rec) == 2 and rec[1]["transitivity"] is True
    assert (ws / "groups" / "cs001_od" / "align_min" / "result.prev.json").exists()


def test_align_running_job_is_reported_and_blocks_other_groups(client, make_case, ws, monkeypatch):
    _make_group(make_case)
    make_case("case_cs002_od_v1", manifest={"oct_source": "/data/CS002_OD_3D Cornea.OCT", "patient_id": "CS002", "eye": "OD"})
    make_case("case_cs002_od_v2", manifest={"oct_source": "/data/CS002_OD_3D Cornea (1).OCT", "patient_id": "CS002", "eye": "OD"})
    rec = []
    _fake_spawn(rec, rc=None, write_result=False, monkeypatch=monkeypatch)   # never finishes
    r = client.post("/api/group/cs001_od/align", json={}).json()
    assert r["started"] is True and r["running"] is True
    r = client.post("/api/group/cs001_od/align", json={"force": True}).json()
    assert r["started"] is False and r["already_running"] is True and len(rec) == 1
    st = client.get("/api/group/cs001_od/align/status").json()
    assert st["running"] is True and st["done"] is False and st["result_exists"] is False
    assert st["progress"]["phase"] == "starting"
    assert client.get("/api/group/cs001_od/align/result").status_code == 404
    r2 = client.post("/api/group/cs002_od/align", json={})
    assert r2.status_code == 409
    assert client.get("/api/group/cs002_od/align/status").json()["running_elsewhere"] == "cs001_od"


def test_align_failed_job_reports_error(client, make_case, ws, monkeypatch):
    _make_group(make_case)
    rec = []
    _fake_spawn(rec, rc=1, write_result=False, monkeypatch=monkeypatch)
    client.post("/api/group/cs001_od/align", json={})
    st = client.get("/api/group/cs001_od/align/status").json()
    assert st["running"] is False and st["done"] is False and "exited with code 1" in st["error"]


def test_align_status_members_follow_the_cases_list_rule_without_starting(client, make_case, ws, monkeypatch):
    _make_group(make_case)
    rec = []
    _fake_spawn(rec, monkeypatch=monkeypatch)
    for gid in ("cs001_od", "CS001|OD", "cs001%20od"):
        st = client.get(f"/api/group/{gid}/align/status")
        assert st.status_code == 200, gid
        assert st.json()["members"] == ["case_cs001_od_v1", "case_cs001_od_v2"], gid
    assert client.get("/api/group/CS001_OS/align/status").json()["members"] == ["case_cs001_os_v1"]
    rows = client.get("/api/cases/list").json()["cases"]
    expect = sorted(c["case_id"] for c in rows if (c["patient"] or "").lower() == "cs001" and (c["eye"] or "").upper() == "OD")
    assert client.get("/api/group/CS001_OD/align/status").json()["members"] == expect
    assert rec == []


def test_align_single_scan_group_400_and_unknown_404(client, make_case, ws, monkeypatch):
    _make_group(make_case)
    rec = []
    _fake_spawn(rec, monkeypatch=monkeypatch)
    assert client.post("/api/group/CS001_OS/align", json={}).status_code == 400
    for gid in ("CS999_OD", "_", "case_cs001_od_consensus", "CS001"):
        assert client.post(f"/api/group/{gid}/align", json={}).status_code == 404, gid
        assert client.get(f"/api/group/{gid}/align/status").status_code == 404, gid
        assert client.get(f"/api/group/{gid}/align/result").status_code == 404, gid
    assert client.get("/api/group/cs001_od/align/overlay/case_cs001_od_v2").status_code == 404
    assert rec == []


def test_align_gid_is_a_safe_path_segment():
    assert api_server._align_gid("CS001 OD") == "cs001_od"
    for bad in ("", "..", "."):
        with pytest.raises(Exception):
            api_server._align_gid(bad)
