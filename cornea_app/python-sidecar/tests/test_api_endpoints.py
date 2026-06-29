"""Unit tests for the FAST FastAPI endpoints of the cornea OCT sidecar.

Scope: only the cheap, deterministic handlers — health/capabilities/config,
case-info reads, and the manifest-flag mutators that drive the timeline
(vet-preprocessing, classification, subgroup(+confirm), scar/skip,
training/schedule, reset-step) plus cornea.nii.gz and the cases list/stat
enumerators. NO SAM2 / preprocess / consensus-build (those need torch/CUDA or
real data) are exercised here.

All state is built with the conftest `make_case`/`client` fixtures against an
isolated cases_root tempdir. Request bodies + response field names were read
from api_server.py before asserting.
"""
from __future__ import annotations

import numpy as np

import orchestration as orch
import labels


# ── GET /api/health ─────────────────────────────────────────────────────────
def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    # shell_version is present (empty string in dev — no Tauri env)
    assert "shell_version" in body


# ── GET /api/system/capabilities ────────────────────────────────────────────
def test_system_capabilities(client):
    r = client.get("/api/system/capabilities")
    assert r.status_code == 200
    body = r.json()
    # Resource fields the frontend uses to size batch preprocessing.
    for key in ("cpu_count", "cpu_budget", "ram_gb", "gpu", "max_concurrency"):
        assert key in body, f"missing capability field {key!r}"
    assert body["cpu_count"] >= 1
    assert body["cpu_budget"] >= 2
    assert body["max_concurrency"] >= 1
    # gpu sub-object always reports a cuda flag (False when no nvidia-smi).
    assert "cuda" in body["gpu"]
    assert isinstance(body["gpu"]["cuda"], bool)


# ── GET + PUT /api/config round-trip ────────────────────────────────────────
def test_config_get_and_put_roundtrip(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    before = r.json()
    assert isinstance(before, dict)

    # PUT a default_case_id and confirm it round-trips through public_config().
    r2 = client.put("/api/config", json={"default_case_id": "case_roundtrip_xyz"})
    assert r2.status_code == 200
    after = r2.json()
    assert after.get("default_case_id") == "case_roundtrip_xyz"

    # A fresh GET reflects the persisted value.
    r3 = client.get("/api/config")
    assert r3.status_code == 200
    assert r3.json().get("default_case_id") == "case_roundtrip_xyz"


# ── GET /api/case/{id} ──────────────────────────────────────────────────────
def test_get_case_returns_manifest_for_made_case(client, make_case):
    cid = make_case("case_get_made")
    r = client.get(f"/api/case/{cid}")
    assert r.status_code == 200
    info = r.json()
    assert info["case_id"] == cid
    # current_case_info nests the manifest written by make_case.
    man = info["manifest"]
    assert man["case_id"] == cid
    assert man.get("input_volume")
    assert man.get("oct_preprocessed") is True


def test_get_case_unknown_returns_empty_manifest(client):
    # current_case_info reads an empty manifest for an unmade case (200, no flags).
    r = client.get("/api/case/case_does_not_exist_404")
    assert r.status_code == 200
    info = r.json()
    assert info["case_id"] == "case_does_not_exist_404"
    assert info["manifest"] == {}


# ── POST /api/case/{id}/vet-preprocessing ───────────────────────────────────
def test_vet_preprocessing_sets_flag(client, make_case):
    cid = make_case("case_vet")
    r = client.post(f"/api/case/{cid}/vet-preprocessing")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "preproc_vetted": True}
    # Persisted in the manifest.
    assert orch.read_manifest(cid).get("preproc_vetted") is True


# ── POST /api/case/{id}/classification ──────────────────────────────────────
def test_classification_sets_scar(client, make_case):
    cid = make_case("case_cls_scar")
    r = client.post(f"/api/case/{cid}/classification",
                    json={"classification": "scar", "scar_range": [3, 7]})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["scar_classification"] == "scar"
    assert body["scar_range"] == [3, 7]
    m = orch.read_manifest(cid)
    assert m.get("scar_classification") == "scar"
    assert m.get("scar_range") == [3, 7]


def test_classification_control_clears_scar_range(client, make_case):
    cid = make_case("case_cls_ctrl")
    # First tag as scar with a range...
    client.post(f"/api/case/{cid}/classification",
                json={"classification": "scar", "scar_range": [2, 5]})
    # ...then demote to control: the stale range must be cleared.
    r = client.post(f"/api/case/{cid}/classification", json={"classification": "control"})
    assert r.status_code == 200
    body = r.json()
    assert body["scar_classification"] == "control"
    assert body["scar_range"] is None
    assert orch.read_manifest(cid).get("scar_range") is None


def test_classification_invalid_rejected(client, make_case):
    cid = make_case("case_cls_bad")
    r = client.post(f"/api/case/{cid}/classification", json={"classification": "tumour"})
    assert r.status_code == 400


# ── POST /api/case/{id}/subgroup  + /subgroup/confirm ───────────────────────
def test_subgroup_set_and_confirm(client, make_case):
    cid = make_case("case_subgrp")
    r = client.post(f"/api/case/{cid}/subgroup", json={"subgroup": "posterior"})
    assert r.status_code == 200
    assert r.json() == {"ok": True, "scar_subgroup": "posterior"}
    assert orch.read_manifest(cid).get("scar_subgroup") == "posterior"

    r2 = client.post(f"/api/case/{cid}/subgroup/confirm")
    assert r2.status_code == 200
    body = r2.json()
    assert body["ok"] is True
    # confirm preserves the previously-set subgroup and sets the gate flag.
    assert body["scar_subgroup"] == "posterior"
    assert body["subgroup_confirmed"] is True
    m = orch.read_manifest(cid)
    assert m.get("scar_subgroup") == "posterior"
    assert m.get("subgroup_confirmed") is True


def test_subgroup_defaults_to_one(client, make_case):
    cid = make_case("case_subgrp_default")
    # No subgroup body field → defaults to "1".
    r = client.post(f"/api/case/{cid}/subgroup", json={})
    assert r.status_code == 200
    assert r.json()["scar_subgroup"] == "1"
    # confirm on a never-set subgroup also defaults to "1".
    cid2 = make_case("case_subgrp_confirm_default")
    r2 = client.post(f"/api/case/{cid2}/subgroup/confirm")
    assert r2.status_code == 200
    assert r2.json()["scar_subgroup"] == "1"
    assert orch.read_manifest(cid2).get("subgroup_confirmed") is True


# ── POST /api/case/{id}/scar/skip ───────────────────────────────────────────
def test_scar_skip_sets_done(client, make_case):
    cid = make_case("case_skip", manifest={"scar_classification": "control"})
    r = client.post(f"/api/case/{cid}/scar/skip")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "scar_done": True}
    assert orch.read_manifest(cid).get("scar_done") is True


# ── POST /api/case/{id}/training/schedule ───────────────────────────────────
def test_training_schedule_toggle(client, make_case):
    cid = make_case("case_sched")
    r_on = client.post(f"/api/case/{cid}/training/schedule", json={"scheduled": True})
    assert r_on.status_code == 200
    assert r_on.json() == {"ok": True, "training_scheduled": True}
    assert orch.read_manifest(cid).get("training_scheduled") is True

    r_off = client.post(f"/api/case/{cid}/training/schedule", json={"scheduled": False})
    assert r_off.status_code == 200
    assert r_off.json() == {"ok": True, "training_scheduled": False}
    assert orch.read_manifest(cid).get("training_scheduled") is False


# ── POST /api/case/{id}/reset-step ──────────────────────────────────────────
def test_reset_step_clears_later_flags_keeps_earlier(client, make_case):
    # A scan that has progressed past SAM2 (step 5): cornea segmented + scar done +
    # aligned + scheduled. Reset back to step 5 should clear steps 6+ but keep step-5
    # artifacts (sam2_meta).
    cid = make_case("case_reset", manifest={
        "sam2_meta": {"vote": 2},          # step 5
        "cornea_vetted": True,             # step 6
        "subgroup_confirmed": True,        # step 7
        "scar_done": True,                 # step 8
        "scar_metrics": {"volume_mm3": 1.0},
        "consensus_case": "some_consensus",  # step 9
        "training_scheduled": True,        # step 12
    })
    r = client.post(f"/api/case/{cid}/reset-step", json={"step": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["step"] == 5

    m = orch.read_manifest(cid)
    # Step 5 itself is kept.
    assert m.get("sam2_meta") == {"vote": 2}
    # Later steps cleared (set to None per _STEP_RESET_FLAGS).
    assert m.get("cornea_vetted") is None
    assert m.get("subgroup_confirmed") is None
    assert m.get("scar_done") is None
    assert m.get("scar_metrics") is None
    assert m.get("consensus_case") is None
    assert m.get("training_scheduled") is None
    # The cleared keys are reported back.
    assert "scar_done" in body["cleared"]
    assert "consensus_case" in body["cleared"]
    assert "sam2_meta" not in body["cleared"]
    # The corrected labelmap (step-5 artifact) survives a reset to step 5 (>=5).
    assert labels.corrected_path(cid).exists()


def test_reset_step_below_sam2_removes_labelmap(client, make_case):
    # Rolling back below SAM2 (target < 5) must delete the on-disk labelmap so a
    # rolled-back scan can't linger in training/overlays.
    cid = make_case("case_reset_low", manifest={
        "sam2_meta": {"vote": 2},
        "scar_metrics": {"volume_mm3": 1.0},
    })
    assert labels.corrected_path(cid).exists()
    r = client.post(f"/api/case/{cid}/reset-step", json={"step": 3})
    assert r.status_code == 200
    m = orch.read_manifest(cid)
    assert m.get("sam2_meta") is None
    assert m.get("scar_metrics") is None
    # On-disk labelmap removed.
    assert not labels.corrected_path(cid).exists()


def test_reset_step_unknown_case_404(client):
    r = client.post("/api/case/case_reset_missing/reset-step", json={"step": 5})
    assert r.status_code == 404


def test_reset_step_out_of_range_rejected(client, make_case):
    cid = make_case("case_reset_oob")
    r = client.post(f"/api/case/{cid}/reset-step", json={"step": 99})
    assert r.status_code == 400
    r0 = client.post(f"/api/case/{cid}/reset-step", json={"step": 0})
    assert r0.status_code == 400


def test_reset_step_refuses_consensus_case(client, make_case):
    # A built consensus case carries consensus_cases (its members) → reset is refused.
    cid = make_case("case_reset_cons", manifest={"consensus_cases": ["a", "b"]})
    r = client.post(f"/api/case/{cid}/reset-step", json={"step": 5})
    assert r.status_code == 400


# ── GET /api/case/{id}/cornea.nii.gz ────────────────────────────────────────
def test_cornea_nifti_for_segmented_case(client, make_case, tmp_path):
    import nibabel as nib

    cid = make_case("case_cornea")  # make_case writes the corrected labelmap
    r = client.get(f"/api/case/{cid}/cornea.nii.gz")
    assert r.status_code == 200
    assert r.headers["content-type"] in ("application/gzip", "application/x-gzip")
    # Gzip magic header — the response really is a gzipped NIfTI.
    assert r.content[:2] == b"\x1f\x8b"

    # The served bytes decode to a NIfTI binary cornea mask (1 where cornea or scar).
    out = tmp_path / "cornea.nii.gz"
    out.write_bytes(r.content)
    arr = np.asarray(nib.load(str(out)).dataobj)
    uniq = set(np.unique(np.rint(arr).astype(np.int64)).tolist())
    assert uniq.issubset({0, 1})
    assert 1 in uniq  # the made case has a cornea band


def test_cornea_nifti_404_without_segmentation(client, make_case):
    # A case with no corrected labelmap → 404. Build the case dirs but remove the label.
    cid = make_case("case_no_seg")
    labels.corrected_path(cid).unlink()
    r = client.get(f"/api/case/{cid}/cornea.nii.gz")
    assert r.status_code == 404


# ── GET /api/cases/stat ─────────────────────────────────────────────────────
def test_cases_stat_counts_dirs(client, make_case):
    # Fresh isolated root: zero cases until we make some.
    r0 = client.get("/api/cases/stat")
    assert r0.status_code == 200
    assert r0.json()["count"] == 0

    make_case("case_stat_a")
    make_case("case_stat_b")
    r = client.get("/api/cases/stat")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert "cases_root" in body


# ── GET /api/cases/list ─────────────────────────────────────────────────────
def test_cases_list_enumerates_oct_cases(client, make_case):
    # Only OCT-loader cases (those with oct_source/companion_txt) are listed.
    make_case("case_oct_one", manifest={
        "oct_source": "/data/CS001_OD_3D Cornea.OCT",
        "patient_id": "CS001",
        "eye": "OD",
        "preproc_vetted": True,
        "scar_classification": "scar",
        "scar_subgroup": "1",
    })
    # A directly-registered (non-OCT) case has no oct_source → skipped.
    make_case("case_registered_only")

    r = client.get("/api/cases/list")
    assert r.status_code == 200
    cases = r.json()["cases"]
    ids = {c["case_id"] for c in cases}
    assert "case_oct_one" in ids
    assert "case_registered_only" not in ids

    entry = next(c for c in cases if c["case_id"] == "case_oct_one")
    assert entry["filename"] == "CS001_OD_3D Cornea.OCT"
    assert entry["patient"] == "CS001"
    assert entry["eye"] == "OD"
    # Per-scan lifecycle flags mirror the manifest.
    life = entry["life"]
    assert life["preproc_vetted"] is True
    assert life["scar_classification"] == "scar"
    assert life["scar_subgroup"] == "1"
    assert life["sam2_meta"] is False  # no segmentation meta set


def test_cases_list_skips_consensus_member(client, make_case):
    # A consensus case (consensus_cases set) is excluded from the loader list.
    make_case("case_real_oct", manifest={"oct_source": "/data/X.OCT"})
    make_case("case_cons_synthetic", manifest={
        "oct_source": "/data/Y.OCT",
        "consensus_cases": ["case_real_oct", "case_other"],
    })
    r = client.get("/api/cases/list")
    assert r.status_code == 200
    ids = {c["case_id"] for c in r.json()["cases"]}
    assert "case_real_oct" in ids
    assert "case_cons_synthetic" not in ids


def test_flag_endpoints_404_on_unknown_case_no_ghost_dir(client, cases_root):
    # The flag-only manifest endpoints must 404 on a typo'd/unknown id rather than silently
    # materialize a ghost case dir (write_manifest_value mkdirs the case dir). Regression guard
    # for the v0.0.93 _require_case() check on vet/classify/subgroup/confirm/schedule/skip.
    ghost = "case_does_not_exist_zzz"
    calls = [
        ("/api/case/%s/vet-preprocessing" % ghost, {}),
        ("/api/case/%s/classification" % ghost, {"classification": "scar"}),
        ("/api/case/%s/subgroup" % ghost, {"subgroup": "1"}),
        ("/api/case/%s/subgroup/confirm" % ghost, {}),
        ("/api/case/%s/training/schedule" % ghost, {"scheduled": True}),
        ("/api/case/%s/scar/skip" % ghost, {}),
    ]
    for path, body in calls:
        r = client.post(path, json=body)
        assert r.status_code == 404, "%s -> %s (expected 404)" % (path, r.status_code)
    # no manifest / case dir was created for the ghost id
    assert not (cases_root / ghost).exists()


def test_flag_endpoints_still_work_for_existing_case(client, make_case):
    # The guard must NOT break the normal path: an existing case still flips its flag.
    cid = make_case("case_guard_ok", manifest={"oct_preprocessed": True})
    r = client.post("/api/case/%s/vet-preprocessing" % cid, json={})
    assert r.status_code == 200 and r.json()["preproc_vetted"] is True
