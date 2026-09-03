"""One preprocessing run per scan at a time.

The re-run button's only guard was React state, so a page reloaded mid-run — or a second client — could
start a second run while the first was still writing previews/volume.nii.gz, the border caches and
manifest.json for the same case. Observed on a real scan: two full runs back to back from one review
session. These tests pin the server-side guard that makes that impossible.
"""
from __future__ import annotations

import threading
import time

import pytest

import api_server


def test_second_concurrent_run_is_refused(client, make_case, monkeypatch):
    cid = make_case("case_lock")
    started = threading.Event()
    release = threading.Event()

    def _slow_impl(case_id, req):
        started.set()
        release.wait(timeout=10)
        return {"ok": True, "case_id": case_id}

    monkeypatch.setattr(api_server, "_oct_preprocess_case_impl", _slow_impl)

    out = {}

    def _first():
        out["first"] = client.post(f"/api/case/{cid}/oct-preprocess", json={}).status_code

    t = threading.Thread(target=_first, daemon=True)
    t.start()
    assert started.wait(timeout=10), "the first run never entered the implementation"

    second = client.post(f"/api/case/{cid}/oct-preprocess", json={})
    assert second.status_code == 409
    assert "already in progress" in second.json().get("detail", "")

    release.set()
    t.join(timeout=10)
    assert out["first"] == 200


def test_lock_is_released_even_when_the_run_raises(client, make_case, monkeypatch):
    """A failed run must not wedge the scan — otherwise one crash makes it unprocessable until restart."""
    cid = make_case("case_lock_raise")

    def _boom(case_id, req):
        raise RuntimeError("boom")

    monkeypatch.setattr(api_server, "_oct_preprocess_case_impl", _boom)
    with pytest.raises(RuntimeError):
        client.post(f"/api/case/{cid}/oct-preprocess", json={})
    assert not api_server.case_run_in_progress(cid)

    monkeypatch.setattr(api_server, "_oct_preprocess_case_impl", lambda case_id, req: {"ok": True})
    assert client.post(f"/api/case/{cid}/oct-preprocess", json={}).status_code == 200


def test_the_lock_is_per_case_not_global(client, make_case, monkeypatch):
    """Two DIFFERENT scans may legitimately run at once — the guard must not serialise the whole store."""
    a = make_case("case_lock_a")
    b = make_case("case_lock_b")
    started = threading.Event()
    release = threading.Event()

    def _slow_impl(case_id, req):
        started.set()
        release.wait(timeout=10)
        return {"ok": True}

    monkeypatch.setattr(api_server, "_oct_preprocess_case_impl", _slow_impl)
    t = threading.Thread(target=lambda: client.post(f"/api/case/{a}/oct-preprocess", json={}), daemon=True)
    t.start()
    assert started.wait(timeout=10)

    monkeypatch.setattr(api_server, "_oct_preprocess_case_impl", lambda case_id, req: {"ok": True})
    assert client.post(f"/api/case/{b}/oct-preprocess", json={}).status_code == 200
    release.set(); t.join(timeout=10)


def test_running_endpoint_reports_state(client, make_case, monkeypatch):
    """A page reloaded mid-run needs to be able to ask whether it is still going."""
    cid = make_case("case_lock_status")
    assert client.get(f"/api/case/{cid}/oct-preprocess-running").json() == {"running": False}

    started = threading.Event(); release = threading.Event()

    def _slow_impl(case_id, req):
        started.set(); release.wait(timeout=10); return {"ok": True}

    monkeypatch.setattr(api_server, "_oct_preprocess_case_impl", _slow_impl)
    t = threading.Thread(target=lambda: client.post(f"/api/case/{cid}/oct-preprocess", json={}), daemon=True)
    t.start()
    assert started.wait(timeout=10)
    time.sleep(0.05)
    assert client.get(f"/api/case/{cid}/oct-preprocess-running").json() == {"running": True}
    release.set(); t.join(timeout=10)
    assert client.get(f"/api/case/{cid}/oct-preprocess-running").json() == {"running": False}
