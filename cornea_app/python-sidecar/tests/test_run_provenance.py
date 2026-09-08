"""Steps v2 — run_provenance.py (run-provenance tree + publication export).

Synthetic cases (tiny volumes + a cs008-shaped manifest) exercise every node builder, the layout / SVG / methods
text, the export folder and the four endpoints. `test_real_case_copy` COPIES case_cs008_od_v2 (manifest, raw,
border_cache, corrected volume) into the test's own cases_root and never writes to review_cases.
Run: cd python-sidecar && python3 -m pytest tests/test_run_provenance.py -q -p no:cacheprovider
"""
from __future__ import annotations

import io
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import orchestration as orch
import oct_preprocess as oct_mod
import run_provenance as rp

pytestmark = pytest.mark.filterwarnings("ignore::pytest.PytestUnknownMarkWarning")

L, D, F, PAD = 24, 64, 12, 6
REAL_CASE = Path(os.environ.get("CORNEA_REAL_CASE") or "/home/zhuojian/Desktop/Integration/review_cases/cases/case_cs008_od_v2")

# tissue-motion params that let the measurement run on a 24-lateral / 12-frame toy volume
_TM_PARAMS = {"tissue_motion_lat_step": 1, "tissue_motion_band": 1, "tissue_motion_min_bands": 4,
              "tissue_motion_cut_guard": 0, "tissue_motion_min_segment": 6}


def _synth_raw() -> np.ndarray:
    """(L, D, F) uint16 raw volume: a bright textured band (rows 20–34) shifted per frame by a sinusoid."""
    rng = np.random.default_rng(7)
    tex = rng.integers(150, 400, size=(L, D), dtype=np.int64)
    base = np.full((L, D), 30, np.int64)
    base[:, 20:35] = 0
    vol = np.zeros((L, D, F), np.uint16)
    for f in range(F):
        sh = int(round(4.0 * np.sin(2 * np.pi * f / F)))
        sl = base.copy()
        sl[:, 20:35] = tex[:, 20:35]
        vol[:, :, f] = np.roll(sl, sh, axis=1).clip(0, 4000).astype(np.uint16)
    return vol


def _surface(row=20.0) -> np.ndarray:
    fr = np.arange(F, dtype=np.float32)
    return (row + 4.0 * np.sin(2 * np.pi * fr / F))[None, :].repeat(L, axis=0).astype(np.float32)


def _write_nifti(arr, path: Path):
    import nibabel as nib
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr), np.diag([-0.01, -0.003, 0.04, 1.0])), str(path))


def _anchors_sig(anchors):
    return rp._anchors_sig(anchors)


def _corrections_manifest(cid, root: Path, raw: np.ndarray, tm_info: dict) -> dict:
    ba = {"3": {str(f): 21.0 + f for f in range(2, 7)}, "12": {str(f): 22.0 + 0.5 * f for f in range(1, 7)}}
    cpa = {"12": {str(f): 40.0 for f in range(0, 4)}, "3": {"5": 63.0}}   # one ABSENT sentinel (>= D-1)
    return {
        "case_id": cid, "oct_source": "/nonexistent/CS999_1_3D_Cornea_OD_2024-01-01.OCT", "oct_spacing": [0.01, 0.003, 0.04],
        "input_volume": str(root / "input" / "corrected.nii.gz"), "corrected_volume": str(root / "input" / "corrected.nii.gz"),
        "oct_preprocessed": True, "review_flags": ["weak-edge"], "difficult_scan": True, "preproc_vetted": False,
        "difficult_reason": {"text": "faint periphery", "at": 1788600000},          # dict-valued manifest field (L12)
        "oct_params": {"dp_sigma_depth": 4.0, "dp_sigma_frame": 2.0, "dp_below": 16, "dp_max_jump": 6,
                       "border_anchors": ba, "crop_post_anchors": cpa, "surface_crop_frames": [0, 1, 2], "surface_crop_mode": "manual",
                       "border_generalize": True, "border_guided": False, "redetect_seed_window": 2.0,
                       "corrected_edge_anchors": {"23": {"4": 30.0, "5": 31.0, "6": 30.5}}, "flatten_exclude_laterals": [5],
                       **_TM_PARAMS},
        "oct_iter": {
            "passes": 1, "best_pass": 1, "metrics": [], "stopped": "redetect",
            "flatten": {"mode": "tissue", "drawn_frames": 0, "cropped_frames": 3, "shift_range": tm_info.get("shift_range"),
                        "tilt_max_px": tm_info.get("tilt_max_px"), "min_target_row": -3.0, "max_target_row": 60.0},
            "tissue_motion": {**tm_info, "band_bottom_guide": {"applied": False, "frames": F,
                                                                "reason": "crop band is 3 of 12 frames: guide disabled in this fixture"}},
            "crop_reconstruction": {"n_frames": 3, "n_columns_rebuilt": 60, "n_fallback_cells": 4, "n_placed_drawn_laterals": 6,
                                    "from": "bottom − interpolated thickness (drawn lines first, served fallback)",
                                    "thickness": {"model": "linear", "source": "drawn", "thickness": {"median": 15.0, "p10": 14.0, "p90": 16.0},
                                                  "boundary": {"left_median": None, "right_median": 15.2},
                                                  "n_placed": 60, "n_estimate_fallback": 57, "n_slices_served": 3, "n_slices_fallback": 20,
                                                  "n_slices_drawn_left": 0, "n_slices_drawn_right": 2, "n_top_present_kept": 5, "n_ceiling_clamped": 1}},
            "canvas_extend": {"pad": PAD, "reason": "corrected apex above the window", "depth_before": D, "depth_after": D + PAD, "move_allowance": 0},
            "detector_stages": {"skipped": ["rigid_height_refine", "rigid_frame_derotate", "rigid_frame_refine"], "reason": "surface-cropped run: fixture"},
            "roughness_veto": {"waived": True, "why": "this run flattens to the edge you drew, so roughness is your call in step 2 (cs048)",
                               "stages": ["rigid_height_refine", "sagittal_quad_align"],
                               "step_guard": "on — a stage that introduces an across-frame tissue step is still declined"},
            "sagittal_quad_align": {"applied": False, "reason": "skipped (sag_quad_align=False): fixture — the flatten already fitted your drawn edge; "
                                                                "measured 2.48 -> 5.47 px on cs048_od_v1_3, so it is off by default — set sag_quad_align=True per case to compare"},
            "edit_transform": {"applied": False, "rounds": 1, "reason": "corrected_edit_mode='line': stored transform kept on record, not applied"},
            "final_qa": {"dev": 1.5, "axial": 0.5, "score": 1.8, "coverage": 1.0, "path": "redetect", "max_jitter": None,
                         "needs_review": False, "review_reasons": []},
            "determinism": {"route": "dense", "n_slices_drawn": 2, "n_points_drawn": 11, "sigma_px": 0.0, "sigma_eff_px": 1.2, "T_px": 4.0,
                            "n_findings": 0, "per_frame": {"n": [2] * F, "w": [1.0] * F, "E_px": [0.0] * F},
                            "floor": {"off_quad_px": 2.0, "rms_px": 1.5}, "blind_frames": {"driven": F, "too_sparse": 0}, "interp_min_slices": 12},
        },
    }


@pytest.fixture
def synth_corrections_case(cases_root):
    cid = "case_zz_corr"
    root = orch.case_root(cid)
    orch.ensure_case_dirs(cid)
    raw = _synth_raw()
    _write_nifti(raw, root / "input" / "_raw_border.nii.gz")
    cor = np.zeros((L, D + PAD, F), np.uint16); cor[:, PAD:, :] = raw
    _write_nifti(cor, root / "input" / "corrected.nii.gz")
    params = {**oct_mod.DEFAULT_PARAMS, **_TM_PARAMS}
    a, b, info = oct_mod.tissue_motion_move(raw.transpose(2, 1, 0), params)
    assert info.get("applied"), info
    m = _corrections_manifest(cid, root, raw, info)
    bc = root / "border_cache"; bc.mkdir(exist_ok=True)
    sig = _anchors_sig(m["oct_params"]["border_anchors"])
    rm = os.path.getmtime(root / "input" / "_raw_border.nii.gz")
    surf = _surface(20.0)
    np.savez_compressed(bc / "baseline.npz", surface=surf, raw_mtime=rm, params_sig="algo=dp-v7-pure-interp;fixture")
    np.savez_compressed(bc / "redetect.npz", surface=surf + 1.0, anchors_sig=sig, raw_mtime=rm, params_sig="algo=dp-v7-pure-interp;fixture")
    np.savez_compressed(bc / "provided_edges.npz", surface=surf + 1.0)
    np.savez_compressed(bc / "placed_edges.npz", surface=surf + 0.5)
    post = _surface(40.0); post[3, 5] = np.nan
    np.savez_compressed(bc / "posterior_edges.npz", surface=post)
    orch.write_manifest_value(cid, m)
    return cid


@pytest.fixture
def synth_automatic_case(cases_root):
    cid = "case_zz_auto"
    root = orch.case_root(cid)
    orch.ensure_case_dirs(cid)
    raw = _synth_raw()
    _write_nifti(raw, root / "input" / "_raw_border.nii.gz")
    _write_nifti(raw, root / "input" / "corrected.nii.gz")
    m = {"case_id": cid, "oct_spacing": [0.01, 0.003, 0.04], "input_volume": str(root / "input" / "corrected.nii.gz"),
         "corrected_volume": str(root / "input" / "corrected.nii.gz"), "oct_preprocessed": True,
         "oct_params": {"dp_sigma_depth": 4.0, "dp_sigma_frame": 2.0, "dp_below": 32, "dp_max_jump": 10, "surface_crop_frames": [0, 1], "surface_crop_mode": "manual"},
         "oct_iter": {"passes": 3, "best_pass": 3, "metrics": [3.1, 2.0, 1.6], "axial_metrics": [0.6, 0.6, 0.6], "scores": [3.5, 2.3, 1.9],
                      "stopped": "diminishing", "auto_tune": {"cached": True},
                      "axial_motion_correct": {"applied": True, "frames_adjusted": 11, "motion_std": 2.0, "max_shift": 4.0, "shift": [float(x) for x in range(F)]},
                      "rigid_height_refine": {"applied": True, "frames_adjusted": 9, "max_jitter": 1.1, "rough_before": 1.2, "rough_after": 1.1},
                      "rigid_frame_derotate": {"applied": True, "frames_rotated": 11, "max_deg": 1.0, "iters": 2, "rough_before": 1.1, "rough_after": 1.0},
                      "rigid_frame_refine": {"applied": False, "reason": "measure unreliable", "frames_over_25px": 2, "dev_rms": 9.0},
                      "surface_crop": {"n_frames": 2, "pad": 4, "rule": "manual", "auto": False, "clamped": False},
                      "final_qa": {"dev": 1.4, "axial": 0.5, "score": 1.7, "coverage": 1.0, "path": "diminishing", "max_jitter": 1.1,
                                   "needs_review": True, "review_reasons": ["motion"]}}}
    orch.write_manifest_value(cid, m)
    return cid


def _node(graph, nid):
    for n in graph["nodes"]:
        if n["id"] == nid:
            return n
    raise AssertionError(f"node {nid} missing")


def _decode_png(data_url: str):
    from PIL import Image
    assert data_url.startswith("data:image/png;base64,")
    import base64
    return Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1])))


# ── 1 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_mode_detection(synth_corrections_case, synth_automatic_case, make_case):
    assert rp.build_run_graph(synth_corrections_case, want_images=False)["run"]["mode"] == "corrections"
    assert rp.build_run_graph(synth_automatic_case, want_images=False)["run"]["mode"] == "automatic"
    cid = make_case("case_zz_plain")
    assert rp.build_run_graph(cid, want_images=False)["run"]["mode"] == "none"


# ── 2 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def _check_graph_consistency(g):
    ids = {n["id"] for n in g["nodes"]}
    for e in g["edges"]:
        assert e["from"] in ids and e["to"] in ids
    by = {n["id"]: n for n in g["nodes"]}
    for e in g["edges"]:
        assert e["to"] in by[e["from"]]["children"]
        assert e["from"] in by[e["to"]]["parents"]
    for n in g["nodes"]:
        for c in n["children"]:
            assert any(e["from"] == n["id"] and e["to"] == c for e in g["edges"])
        if n["status"] != "applied":
            assert n["status_reason"], n["id"]


def test_canonical_nodes_present(synth_corrections_case, synth_automatic_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=False)
    ids = {n["id"] for n in g["nodes"]}
    assert set(rp.CANONICAL_CORRECTIONS) <= ids
    assert g["run"]["stages"] == rp._SPINE_CORRECTIONS          # nothing of the automatic chain ran in this fixture
    assert [n["id"] for n in g["nodes"] if n["role"] == "spine"] == g["run"]["stages"]
    assert [i for i in rp._RUN_ORDER if i in g["run"]["stages"]] == g["run"]["stages"]   # run order
    _check_graph_consistency(g)
    ga = rp.build_run_graph(synth_automatic_case, want_images=False)
    ida = {n["id"] for n in ga["nodes"]}
    assert set(rp.CANONICAL_AUTOMATIC) <= ida and set(rp.CANONICAL_CORRECTIONS) <= ida
    assert ga["run"]["stages"] == rp._SPINE_AUTOMATIC
    for nid in ("n04_served_surface", "n06_placement", "n07_tissue_motion", "n08_band_guide", "n09_canvas_extend", "n10_flatten", "n11_pane_edits"):
        n = _node(ga, nid)
        assert n["status"] == "not_run" and n["status_reason"]
    assert _node(ga, "n03_crop_marks")["status"] == "applied"          # its oct_params ARE present
    assert "a08_surface_crop" in _node(ga, "n03_crop_marks")["children"]
    _check_graph_consistency(ga)


# ── 3 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_statuses_and_reasons(synth_corrections_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=False)
    bg = _node(g, "n08_band_guide")
    assert bg["status"] == "declined" and bg["status_reason"] == "crop band is 3 of 12 frames: guide disabled in this fixture"
    assert _node(g, "n11b_edit_transform")["status"] == "declined"
    ce = _node(g, "n09_canvas_extend")
    assert ce["status"] == "applied" and any(x["label"] == "pad" and x["value"] == PAD for x in ce["numbers"])
    fl = _node(g, "n10_flatten")
    assert fl["status"] == "applied" and any("truncat" in w for w in fl["warnings"])
    assert any("truncat" in w for w in g["run"]["warnings"])
    assert _node(g, "n10a_detector_stages")["status"] == "declined"
    assert _node(g, "n07_tissue_motion")["status"] == "applied"
    # missing posterior → missing_data, no exception
    (orch.case_root(synth_corrections_case) / "border_cache" / "posterior_edges.npz").unlink()
    rp._GRAPH_CACHE.clear()
    g2 = rp.build_run_graph(synth_corrections_case, want_images=True)
    n5 = _node(g2, "n05_bottom_lines")
    assert n5["status"] == "missing_data" and "posterior_edges.npz" in n5["status_reason"]


# ── 4 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_images(synth_corrections_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=True)
    for nid in ("n00_raw", "n01_detect", "n02_top_lines", "n04_served_surface", "n07_tissue_motion", "n12_output"):
        n = _node(g, nid)
        assert n["images"], (nid, n["image_error"])
        for im in n["images"]:
            assert im["width"] > 0 and im["height"] > 0
            pil = _decode_png(im["data_url"])
            assert pil.size == (im["width"], im["height"])
            assert im["file"].endswith(".png") and im["caption"]
    for nid in ("n10a_detector_stages", "n10b_roughness_veto", "n10c_sag_quad_align"):
        assert _node(g, nid)["images"] == []
    assert g["run"]["slices"]["most_edited"] == 12
    assert g["run"]["slices"]["rendered"] == sorted({L // 2, 12})   # unique [central, most_edited] (both 12 here)
    g0 = rp.build_run_graph(synth_corrections_case, want_images=False)
    assert all(n["images"] == [] for n in g0["nodes"])
    g3 = rp.build_run_graph(synth_corrections_case, slice_index=3, want_images=True)
    assert g3["run"]["slices"]["rendered"] == [3]
    for n in g3["nodes"]:
        for im in n["images"]:
            if im["orientation"] == "sagittal":
                assert im["slice_index"] == 3, (n["id"], im["file"])


# ── 5 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_tissue_motion_recompute(synth_corrections_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=False)
    n7 = _node(g, "n07_tissue_motion")
    nums = {x["label"]: x["value"] for x in n7["numbers"]}
    import nibabel as nib
    raw = np.asanyarray(nib.load(str(orch.case_root(synth_corrections_case) / "input" / "_raw_border.nii.gz")).dataobj)
    _, _, info = oct_mod.tissue_motion_move(raw.transpose(2, 1, 0), {**oct_mod.DEFAULT_PARAMS, **_TM_PARAMS})
    assert np.allclose(nums["recomputed shift range"], info["shift_range"], atol=1e-6)
    assert isinstance(nums["recompute_matches"], bool) and nums["recompute_matches"] is True


# ── 6 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_json_safe(synth_corrections_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=True)
    txt = json.dumps(g, allow_nan=False)      # raises on NaN/inf; numpy scalars would raise TypeError
    assert json.loads(txt)["run"]["mode"] == "corrections"


# ── 7 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_layout_and_svg(synth_corrections_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=False)
    lay = rp.layout_graph(g)
    by = {n["id"]: n for n in g["nodes"]}
    assert set(lay["nodes"]) <= set(by)
    for nid, n in by.items():                      # M10: every spine node and every input that did something is drawn …
        assert (nid in lay["nodes"]) == (n["role"] == "spine" or n["status"] in ("applied", "declined", "missing_data", "error")), nid
    assert set(lay["omitted"]) == {nid for nid in by if nid not in lay["nodes"]}
    assert "a03_flatten_passes" in lay["omitted"] and "n11_pane_edits" in lay["omitted"]   # … other-chain / pending stay in nodes.json
    ys = [lay["nodes"][s]["y"] for s in g["run"]["stages"]]
    # no far column here (the pending pane edits are omitted) → two-column footprint: inputs at FAR_X, spine at INPUT_X
    assert not lay["has_far_column"] and lay["spine_x"] == rp.INPUT_X and lay["input_x"] == rp.FAR_X
    assert all(lay["nodes"][s]["x"] == lay["spine_x"] for s in g["run"]["stages"])
    assert ys == sorted(ys) and len(set(ys)) == len(ys)
    for n in g["nodes"]:
        if n["role"] == "input" and n["id"] in lay["nodes"]:
            assert lay["nodes"][n["id"]]["x"] == lay["input_x"]
    assert all(b["w"] == rp.NODE_W and b["h"] == rp.NODE_H for b in lay["nodes"].values())
    svg = rp.render_diagram_svg(g, lay)
    assert svg.startswith("<svg")
    import textwrap
    for nid in lay["nodes"]:
        for ln in textwrap.wrap(by[nid]["title"], rp._TITLE_WRAP)[:2]:      # wrapped, never truncated
            assert ln.rstrip("…") in svg or ln[:20] in svg, (nid, ln)
    for nid in lay["omitted"]:
        assert f'>{by[nid]["title"]}<' not in svg
    assert "declined" in svg and "Legend" in svg and "reviewer input" in svg and "not run" in svg
    assert "folded (line mode)" not in svg          # the pending pane edits are not drawn
    ET.fromstring(svg)


# ── 8 ────────────────────────────────────────────────────────────────────────────────────────────────────────
def test_methods_paragraph(synth_corrections_case, synth_automatic_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=False)
    txt = rp.methods_paragraph(g)
    sr = g["run"]  # noqa: F841
    tm = _node(g, "n07_tissue_motion")
    rng = [x["value"] for x in tm["numbers"] if x["label"] == "shift range (applied)"][0]
    assert f"extended by {PAD} rows" in txt
    assert f"{rng[0]:.1f}..{rng[1]:.1f}" in txt
    assert "depth σ 4.0" in txt
    assert "guide disabled in this fixture" in txt
    assert "{" not in txt
    ta = rp.methods_paragraph(rp.build_run_graph(synth_automatic_case, want_images=False))
    assert "passes" in ta and "{" not in ta


# ── 9 ────────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not rp._HAVE_MPL, reason="matplotlib not installed")
def test_export_writes_all_files(synth_corrections_case, tmp_path):
    from PIL import Image
    out = tmp_path / "rep"
    info = rp.export_run_report(synth_corrections_case, out, dpi=300)
    for f in ("figure.png", "figure.svg", "diagram.svg", "methods.md", "nodes.json", "report.html", "figure_caption.md", "README.txt"):
        assert (out / f).exists(), f
    w, h = Image.open(out / "figure.png").size
    assert abs(w - 7.2 * 300) <= 0.05 * 7.2 * 300 and abs(h - 5.6 * 300) <= 0.05 * 5.6 * 300
    g = json.loads((out / "nodes.json").read_text())
    assert len(g["nodes"]) == info["n_nodes"]
    n_img = sum(len(n["images"]) for n in g["nodes"])
    assert len(list((out / "nodes").glob("*.png"))) == n_img > 0
    html = (out / "report.html").read_text()
    assert "<svg" in html and "src='http" not in html and 'src="http' not in html
    for n in g["nodes"]:
        import html as _h
        assert _h.escape(n["title"]) in html
    zp = rp.zip_report(out)
    import zipfile
    assert any(name.endswith("report.html") for name in zipfile.ZipFile(zp).namelist())


# ── 10 ───────────────────────────────────────────────────────────────────────────────────────────────────────
def test_endpoints(client, synth_corrections_case, make_case, tmp_path, cases_root):
    cid = synth_corrections_case
    r = client.get(f"/api/case/{cid}/oct-run-graph?want_images=0")
    assert r.status_code == 200 and r.json()["run"]["mode"] == "corrections"
    r = client.post(f"/api/case/{cid}/oct-run-graph", json={"slice_index": 3})
    assert r.status_code == 200 and r.json()["run"]["slices"]["rendered"] == [3]
    assert client.get("/api/case/case_zz_nope/oct-run-graph").status_code == 404
    plain = make_case("case_zz_plain2")
    r = client.get(f"/api/case/{plain}/oct-run-graph?want_images=0")
    assert r.status_code == 200 and r.json()["run"]["mode"] == "none"
    assert client.post(f"/api/case/{plain}/export-run-report", json={}).status_code == 409
    if not rp._HAVE_MPL:
        pytest.skip("matplotlib not installed")
    r = client.post(f"/api/case/{cid}/export-run-report", json={"dpi": 100})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["download_url"] == f"/api/case/{cid}/run-report.zip" and body["zip_name"].endswith(".zip") and body["bytes"] > 0
    assert Path(body["zip"]).exists() and Path(body["folder"]).is_dir()
    r = client.get(f"/api/case/{cid}/run-report.zip")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/zip")
    assert r.headers["content-disposition"].rstrip('"').endswith(".zip")
    r = client.post(f"/api/case/{cid}/run-report-save", json={"dest": str(cases_root / "x.zip")})
    assert 400 <= r.status_code < 500
    dest = tmp_path / "outside" / "report.zip"
    r = client.post(f"/api/case/{cid}/run-report-save", json={"dest": str(dest)})
    assert r.status_code == 200 and dest.exists()
    for _ in range(3):
        time.sleep(1.05)   # distinct timestamps
        assert client.post(f"/api/case/{cid}/export-run-report", json={"dpi": 72}).status_code == 200
    exp = orch.case_root(cid) / "exports"
    assert len([p for p in exp.glob("run_report_*") if p.is_dir()]) <= 3
    assert len(list(exp.glob("run_report_*.zip"))) <= 3


# ── 12 H1 / M6: pending vs consumed corrected-pane edits ────────────────────────────────────────────────────
def _all_text(g):
    out = [g["run"].get("mode") or ""]
    for n in g["nodes"]:
        out += [str(n.get(k) or "") for k in ("method", "description", "summary", "status_reason", "title")]
        out += [str(im.get("caption") or "") for im in n.get("images", [])]
        out += [str(w) for w in n.get("warnings", [])]
        for x in n.get("numbers", []):
            v = x.get("value")
            out += [v] if isinstance(v, str) else [str(y) for y in v if isinstance(y, str)] if isinstance(v, list) else []
    out += [str(w) for w in g["run"].get("warnings", [])]
    return out


def test_pane_edits_pending_vs_consumed(synth_corrections_case):
    cid = synth_corrections_case
    g = rp.build_run_graph(cid, want_images=True)
    n11 = _node(g, "n11_pane_edits")
    assert n11["status"] == "not_run" and "pending" in n11["status_reason"] and "3 edge points on 1 slices" in n11["status_reason"]
    assert n11["pending"]["n_points"] == 3 and n11["pending"]["laterals"] == [23] and n11["consumed"] is None
    assert n11["method"] == ""                                            # never claimed consumed
    assert any(x["label"] == "consumed on the last run" and x["value"] is False for x in n11["numbers"])
    assert any("PENDING" in w for w in g["run"]["warnings"])
    txt = rp.methods_paragraph(g)
    assert "Corrected-pane edits" not in txt and "consumed" not in txt
    assert n11["images"] and "PENDING" in n11["images"][0]["caption"]
    # consumed: the fold record is on oct_iter and the pane keys were cleared by the fold
    m = orch.read_manifest(cid)
    m["oct_params"].pop("corrected_edge_anchors"); m["oct_params"].pop("corrected_post_anchors", None)
    m["oct_params"]["border_anchors"]["23"] = {"4": 24.0, "5": 25.0, "6": 24.5}
    m["oct_iter"]["corrected_fold"] = {"folded": True, "mode": "line", "n_points": 3, "laterals": [23], "pinned_laterals": [],
                                       "n_bottom_points": 0, "bottom_folded_slices": [], "excluded_laterals": [5, 23], "backup": "pre.json"}
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g2 = rp.build_run_graph(cid, want_images=True)
    n11 = _node(g2, "n11_pane_edits")
    assert n11["status"] == "applied" and n11["consumed"]["laterals"] == [23] and n11["consumed"]["excluded_laterals"] == [5, 23]
    assert n11["pending"]["n_points"] == 0 and not n11["warnings"]
    assert "folded into the original-pane lines" in n11["method"] and "3 points on 1 slices" in n11["method"]
    assert "5, 23" in n11["method"] or "5–5, 23–23" in n11["method"] or "kept out of the per-frame fit" in n11["method"]
    assert "Corrected-pane edits" in rp.methods_paragraph(g2)
    # M6 wiring: pane edits feed the original-pane lines, not the flatten
    e = {(x["from"], x["to"]): x["label"] for x in g2["edges"]}
    assert e[("n11_pane_edits", "n02_top_lines")] == "folded (line mode)" and e[("n11_pane_edits", "n05_bottom_lines")] == "folded (line mode)"
    assert ("n11_pane_edits", "n10_flatten") not in e
    ids = [n["id"] for n in g2["nodes"]]
    assert ids.index("n02_top_lines") < ids.index("n11_pane_edits") < ids.index("n04_served_surface")
    lay = rp.layout_graph(g2)
    assert "n11_pane_edits" in lay["nodes"] and any(x["from"] == "n11_pane_edits" and x["to"] == "n02_top_lines" for x in lay["edges"])
    # transform-mode legacy: wired to the edit transform
    m["oct_iter"]["corrected_fold"]["mode"] = "transform"
    m["oct_iter"]["corrected_fold"]["transform"] = {"rounds": 1, "shift_range": [-2.0, 1.0], "tilt_max_px": 0.5}
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g3 = rp.build_run_graph(cid, want_images=False)
    e3 = {(x["from"], x["to"]): x["label"] for x in g3["edges"]}
    assert e3[("n11_pane_edits", "n11b_edit_transform")] == "fitted transform" and ("n11_pane_edits", "n05_bottom_lines") not in e3
    assert "fitted to a per-frame rigid shift and tilt" in _node(g3, "n11_pane_edits")["method"]


# ── 13 H2: stages that ran on either chain sit on the spine; not-run text matches the mode ──────────────────
def test_other_chain_stage_that_ran_is_on_the_spine(synth_corrections_case, synth_automatic_case):
    cid = synth_corrections_case
    m = orch.read_manifest(cid)
    m["oct_iter"]["crop"] = {"n_voxels": 1234}
    m["oct_params"]["crop_bands"] = {"3": [1, 2]}
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g = rp.build_run_graph(cid, want_images=False)
    crop = _node(g, "a09_crop")
    assert crop["status"] == "applied" and crop["role"] == "spine" and "a09_crop" in g["run"]["stages"]
    st = g["run"]["stages"]
    assert st.index("n11b_edit_transform") < st.index("a09_crop") < st.index("n12_output")
    assert set(rp._SPINE_CORRECTIONS) <= set(st) and [i for i in rp._RUN_ORDER if i in st] == st
    assert "1234 voxels" in rp.methods_paragraph(g)
    assert ("n03_crop_marks", "a09_crop") in {(e["from"], e["to"]) for e in g["edges"]}
    _check_graph_consistency(g)
    # the automatic chain's stages that did not run say so with the RIGHT wording (the keys used to be inverted)
    for nid in ("a03_flatten_passes", "a04_rigid_height_refine", "a07_manual_patch"):
        n = _node(g, nid)
        assert n["status"] == "not_run" and n["role"] == "input" and n["chain"] == "other"
        assert "no reviewer corrections" not in n["status_reason"], (nid, n["status_reason"])
    assert "corrections run" in _node(g, "a03_flatten_passes")["status_reason"]
    assert _node(g, "a04_rigid_height_refine")["status_reason"] == "surface-cropped run: fixture"   # the run's own skip reason wins
    ga = rp.build_run_graph(synth_automatic_case, want_images=False)
    for nid in ("n07_tissue_motion", "n09_canvas_extend", "n10_flatten", "n11b_edit_transform"):
        n = _node(ga, nid)
        assert n["status"] == "not_run" and "automatic run" in n["status_reason"], (nid, n["status_reason"])
    assert _node(ga, "a08_surface_crop")["role"] == "spine"


# ── 14 M3: mode detection beyond flatten.mode == 'tissue' ───────────────────────────────────────────────────
def test_mode_detection_variants(synth_corrections_case, tmp_path):
    it = {"stopped": "redetect", "flatten": {"mode": "drawn+minimax"}}
    assert rp._mode(it) == "corrections"
    assert rp._mode({"stopped": "redetect", "passes": 1}) == "corrections"
    assert rp._mode({"stopped": "diminishing", "final_qa": {"path": "redetect"}}) == "corrections"
    assert rp._mode({"stopped": "diminishing", "flatten": {"mode": "minimax"}}) == "corrections"
    assert rp._mode({"stopped": "diminishing", "passes": 2}) == "automatic"
    root = tmp_path / "c"; (root / "border_cache").mkdir(parents=True)
    np.savez(root / "border_cache" / "provided_edges.npz", surface=np.zeros((2, 2), np.float32))
    assert rp._mode({"stopped": "diminishing"}, {"border_anchors": {"1": {"2": 3.0}}}, root) == "corrections"
    assert rp._mode({"stopped": "diminishing"}, {"border_anchors": {}}, root) == "automatic"
    assert rp._mode({}) == "none" and rp._mode(None) == "none"
    # the flatten node is labelled with the actual move source
    cid = synth_corrections_case
    m = orch.read_manifest(cid)
    m["oct_iter"]["flatten"]["mode"] = "drawn+minimax"
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g = rp.build_run_graph(cid, want_images=False)
    assert g["run"]["mode"] == "corrections" and g["run"]["flatten_kind"] == "drawn"
    n10 = _node(g, "n10_flatten")
    assert "drawn-line move" in n10["title"] and "drawn-line move" in n10["method"]
    m["oct_iter"]["flatten"]["mode"] = "tissue"
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    assert "tissue-measured move" in _node(rp.build_run_graph(cid, want_images=False), "n10_flatten")["title"]


# ── 15 M7 / L19: third person, no foreign case names, in every sentence ─────────────────────────────────────
def test_methods_are_third_person(synth_corrections_case, synth_automatic_case):
    import re
    for cid in (synth_corrections_case, synth_automatic_case):
        g = rp.build_run_graph(cid, want_images=True)
        texts = _all_text(g) + [rp.methods_paragraph(g)]
        for t in texts:
            assert not re.search(r"\b(you|your|yours)\b", t, re.I), t
            assert not re.search(r"\bcs\d{3}", t, re.I), t
        rv = _node(g, "n10b_roughness_veto")
        if rv["status"] == "declined":
            assert "reviewer-drawn edge" in rv["method"] and "in step" not in rv["method"]
            assert "stayed on" in rv["method"]
            marked = [x["value"] for x in rv["numbers"] if x["label"] == "stages evaluated under the waiver"][0]
            assert "rigid_height_refine (skipped)" in marked and "sagittal_quad_align (declined)" in marked    # L20
        assert "cs048" not in _node(g, "n10c_sag_quad_align")["description"]
    assert rp._neutral("measured on cs008 and cs042", "case_cs008_od_v2") == "measured on cs008 and another scan"
    # recorded reasons are reduced to their factual clause: status prefixes and UI advice never reach a Methods sentence
    assert rp._reason_clause("skipped (sag_quad_align=False): the flatten already fitted the drawn edge; this re-flatten measured 2.5 px on cs048, "
                             "so it is off by default — set sag_quad_align=True per case to compare") == "the flatten already fitted the drawn edge"
    assert rp._reason_clause("declined: introduced a TISSUE step at frames 60->61 (worst 17 px) — a step is never allowed; this is the guard") \
        == "introduced a TISSUE step at frames 60->61 (worst 17 px)"
    assert rp._reason_clause("") == "not recorded" and rp._reason_clause(None) == "not recorded"
    g = rp.build_run_graph(synth_corrections_case, want_images=False)
    assert "skipped because surface-cropped run: fixture." in _node(g, "n10a_detector_stages")["method"]
    # the quoted 'reason' number of n10c carries the factual clause only (no second person, no other scan, no UI advice)
    reason = [x["value"] for x in _node(g, "n10c_sag_quad_align")["numbers"] if x["label"] == "reason"][0]
    assert reason == "fixture" or reason.startswith("fixture"), reason
    assert "cs048" not in reason and "set sag_quad_align" not in reason


# ── 16 L12: numbers[].value is never a dict ──────────────────────────────────────────────────────────────────
def test_numbers_are_flat(synth_corrections_case, synth_automatic_case):
    def _flat(v):
        return v is None or isinstance(v, (str, int, float, bool)) or (isinstance(v, list) and all(x is None or isinstance(x, (str, int, float, bool)) for x in v))
    for cid in (synth_corrections_case, synth_automatic_case):
        g = rp.build_run_graph(cid, want_images=False)
        for n in g["nodes"]:
            for x in n["numbers"]:
                assert _flat(x["value"]), (n["id"], x)
    n0 = {x["label"]: x["value"] for x in _node(rp.build_run_graph(synth_corrections_case, want_images=False), "n00_raw")["numbers"]}
    assert n0["difficult reason"] == "faint periphery"


# ── 17 L16: malformed anchors degrade, never raise ───────────────────────────────────────────────────────────
def test_malformed_anchors_degrade(client, synth_corrections_case):
    cid = synth_corrections_case
    m = orch.read_manifest(cid)
    m["oct_params"]["border_anchors"] = {"abc": {"1": 2.0}, "3": "oops", "4": {"x": 1, "2": "nan", "5": None, "6": "inf"}, "7": {"1": 9.0}, "8": None}
    m["oct_params"]["crop_post_anchors"] = {"q": 1, "3": {"z": "w"}}
    m["oct_params"]["corrected_edge_anchors"] = {"1": [1, 2]}
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g = rp.build_run_graph(cid, want_images=True)
    assert g["run"]["mode"] == "corrections" and _node(g, "n02_top_lines")["status"] == "applied"
    assert {x["label"]: x["value"] for x in _node(g, "n02_top_lines")["numbers"]}["slices drawn"] == 1
    assert not any(n["status"] == "error" for n in g["nodes"]), [(n["id"], n["status_reason"]) for n in g["nodes"] if n["status"] == "error"]
    r = client.get(f"/api/case/{cid}/oct-run-graph?want_images=0")
    assert r.status_code == 200
    assert rp._anchors_sig(m["oct_params"]["border_anchors"]) == "7:1=9"


# ── 18 L17: dpi clamp ───────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not rp._HAVE_MPL, reason="matplotlib not installed")
def test_dpi_clamped(client, synth_corrections_case):
    from PIL import Image
    assert rp.clamp_dpi(5000) == 600 and rp.clamp_dpi(10) == 72 and rp.clamp_dpi(None) == 300 and rp.clamp_dpi("x") == 300
    r = client.post(f"/api/case/{synth_corrections_case}/export-run-report", json={"dpi": 10})
    assert r.status_code == 200, r.text
    w, _ = Image.open(Path(r.json()["folder"]) / "figure.png").size
    assert abs(w - 7.2 * 72) <= 0.05 * 7.2 * 72


# ── 19 M8 / M11 / L13 / L15 / L22 / M9 / M5 ─────────────────────────────────────────────────────────────────
def test_placement_split_and_sigma_eff(synth_corrections_case):
    g = rp.build_run_graph(synth_corrections_case, want_images=True)
    n6 = _node(g, "n06_placement")
    assert "57 (95 %) fell back to the fixed thickness estimate" in n6["method"] and "3 (5 %) took the thickness" in n6["method"]
    nums = {x["label"]: x["value"] for x in n6["numbers"]}
    assert nums["cells from the fixed estimate fallback"] == 57 and nums["cells from the thickness model"] == 3
    n4 = _node(g, "n04_served_surface")
    nums = {x["label"]: x["value"] for x in n4["numbers"]}
    assert nums["σ effective (tolerance basis)"] == 1.2 and "σ_eff = 1.20 px" in n4["method"] and "tolerance T = 4.0 px" in n4["method"]
    plots = [im for im in n4["images"] if im["orientation"] == "plot"]
    assert plots and "identically zero" in plots[0]["caption"]
    n2 = _node(g, "n02_top_lines")
    assert any("densify_anchor_polylines" in im["caption"] for im in n2["images"])       # L13
    assert rp._move_arrays(rp._ctx(synth_corrections_case)) is not None and rp._move_arrays(rp._ctx(synth_corrections_case))[2] is False


@pytest.mark.skipif(not rp._HAVE_MPL, reason="matplotlib not installed")
def test_export_rc_and_captions_and_params(synth_corrections_case, tmp_path):
    import matplotlib.pyplot as plt
    before = dict(plt.rcParams)
    out = tmp_path / "rep3"
    rp.export_run_report(synth_corrections_case, out, slice_index=3, dpi=72)
    assert dict(plt.rcParams) == before                                                   # L15
    cap = (out / "figure_caption.md").read_text()
    assert "(a) Raw sagittal slice 3" in cap and "bottom-line points" not in cap and "top-edge points" in cap   # L22 (slice 3 has only a sentinel)
    out2 = tmp_path / "rep12"
    rp.export_run_report(synth_corrections_case, out2, slice_index=12, dpi=72)
    assert "4 bottom-line points" in (out2 / "figure_caption.md").read_text()
    md = (out / "methods.md").read_text()
    assert "Parameters (stages that ran" in md and "Automatic corneal-surface detection (DP): dp_sigma_depth = 4" in md
    assert "Served surface" in md and "route = dense" in md and "{" not in md                 # M9: per-stage, route params included
    assert "Flatten passes" not in md                                                          # a stage that did not run lists nothing
    # M5: the graph cache key sees every border_cache file
    ctx = rp._ctx(synth_corrections_case); sl = rp._slices(ctx, None)
    k1 = rp._cache_key(ctx, sl, False, 300)
    pe = orch.case_root(synth_corrections_case) / "border_cache" / "placed_edges.npz"
    os.utime(pe, (time.time() + 5, time.time() + 5))
    k2 = rp._cache_key(rp._ctx(synth_corrections_case), sl, False, 300)
    assert k1 != k2


# ── 20 L21 / L23: automatic surface-crop branch, new reviewer-input nodes ───────────────────────────────────
def test_surface_crop_branch_and_new_input_nodes(synth_automatic_case):
    cid = synth_automatic_case
    m = orch.read_manifest(cid)
    m["oct_iter"]["stopped"] = "surface_crop"; m["oct_iter"]["metrics"] = []; m["oct_iter"]["passes"] = 1; m["oct_iter"]["scores"] = None
    m["oct_iter"]["surface_crop"]["frames"] = [0, 1]
    m["oct_iter"]["axial_anchors"] = {"applied": True, "frames_adjusted": 2}
    m["oct_params"]["axial_anchors"] = {"4": {"3": 21.0, "10": 22.0, "17": 21.5}, "5": {"3": 21.0}}
    m["defect_marks"] = [{"orient": "sagittal", "slice": 3, "cols": [1, 2, 3], "tag": "edge_detection"}]
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g = rp.build_run_graph(cid, want_images=True)
    a3 = _node(g, "a03_flatten_passes")
    assert a3["status"] == "not_run" and "surface-crop branch" in a3["status_reason"] and a3["method"] == ""
    assert "1 passes" not in rp.methods_paragraph(g)
    a8 = _node(g, "a08_surface_crop")
    assert a8["status"] == "applied" and a8["images"] and "surface-crop extend" in a8["images"][0]["caption"].lower()
    ax = _node(g, "n02b_axial_gt")
    assert ax["status"] == "applied" and ax["kind"] == "user" and ax["images"][0]["orientation"] == "axial"
    assert "2 frames (4 points)" in ax["method"] and ("n02b_axial_gt", "n12_output") in {(e["from"], e["to"]) for e in g["edges"]}
    mk = _node(g, "n02c_marks")
    assert mk["status"] == "applied" and mk["kind"] == "user"
    assert {x["label"]: x["value"] for x in mk["numbers"]}["marks on the raw / original view"] == 1
    lay = rp.layout_graph(g)
    assert "n02b_axial_gt" in lay["nodes"] and "n02c_marks" in lay["nodes"] and "a03_flatten_passes" in lay["nodes"]   # a03 is canonical spine
    txt = rp.methods_paragraph(g)
    assert "B-scan plane" in txt and "Mark tool" in txt
    # absent → not_run and off the diagram
    m["oct_params"].pop("axial_anchors"); m["oct_iter"].pop("axial_anchors"); m["defect_marks"] = []
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g2 = rp.build_run_graph(cid, want_images=False)
    assert _node(g2, "n02b_axial_gt")["status"] == "not_run" and _node(g2, "n02c_marks")["status"] == "not_run"
    lay2 = rp.layout_graph(g2)
    assert "n02b_axial_gt" in lay2["omitted"] and "n02c_marks" in lay2["omitted"]
    # the automatic-run per-pass plot (L23) on the plain automatic fixture
    m["oct_iter"].update({"stopped": "diminishing", "metrics": [3.1, 2.0, 1.6], "passes": 3, "scores": [3.5, 2.3, 1.9]})
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    a3 = _node(rp.build_run_graph(cid, want_images=True), "a03_flatten_passes")
    assert a3["status"] == "applied" and a3["images"] and a3["images"][0]["orientation"] == "plot"


# ── 11 ───────────────────────────────────────────────────────────────────────────────────────────────────────
@pytest.mark.real_data
@pytest.mark.skipif(not (REAL_CASE / "manifest.json").exists(), reason="real cs008 case not available")
def test_real_case_copy(cases_root):
    t0 = time.time()
    src_mtime = (REAL_CASE / "manifest.json").stat().st_mtime
    cid = REAL_CASE.name
    dst = cases_root / cid
    (dst / "input").mkdir(parents=True)
    shutil.copy2(REAL_CASE / "manifest.json", dst / "manifest.json")
    shutil.copy2(REAL_CASE / "input" / "_raw_border.nii.gz", dst / "input" / "_raw_border.nii.gz")
    shutil.copytree(REAL_CASE / "border_cache", dst / "border_cache")
    m = json.loads((dst / "manifest.json").read_text())
    vol = Path(m["input_volume"])
    shutil.copy2(vol, dst / "input" / vol.name)
    m["input_volume"] = m["corrected_volume"] = str(dst / "input" / vol.name)
    (dst / "manifest.json").write_text(json.dumps(m))
    g = rp.build_run_graph(cid, want_images=True)
    assert g["run"]["mode"] == "corrections"
    ba = m["oct_params"]["border_anchors"]
    n2 = {x["label"]: x["value"] for x in _node(g, "n02_top_lines")["numbers"]}
    assert n2["slices drawn"] == len(ba) and n2["points"] == sum(len(v) for v in ba.values())
    it = m["oct_iter"]
    n7 = {x["label"]: x["value"] for x in _node(g, "n07_tissue_motion")["numbers"]}
    ref = (it["tissue_motion"].get("pre_guide") or it["tissue_motion"])["shift_range"]
    assert n7["recompute_matches"] is True
    assert max(abs(a - b) for a, b in zip(n7["recomputed shift range"], ref)) < 0.05
    guide = it["tissue_motion"].get("band_bottom_guide") or {}
    assert (_node(g, "n08_band_guide")["status"] == "declined") == (not guide.get("applied"))
    has_warn = any("truncat" in w for w in g["run"]["warnings"])
    assert has_warn == (float(it["flatten"]["min_target_row"]) < 0)
    if rp._HAVE_MPL:
        out = cases_root / "_report"
        info = rp.export_run_report(cid, out)
        assert (out / "figure.png").exists() and info["n_nodes"] == len(g["nodes"])
    assert time.time() - t0 < 60
    assert (REAL_CASE / "manifest.json").stat().st_mtime == src_mtime      # read-only guard


# ── 22 M10 client/server consistency: layout constants mirror runTreeLayout.ts; absent inputs carry no edges ────
def test_layout_mirrors_frontend_and_absent_inputs_have_no_edges(synth_corrections_case, synth_automatic_case):
    import re
    ts = Path(__file__).resolve().parents[2] / "src" / "components" / "viewer" / "runTreeLayout.ts"
    if ts.exists():
        src = ts.read_text()
        for name in ("NODE_W", "NODE_H", "GAP", "COL_GAP", "FAR_X"):
            m = re.search(rf"export const {name} = (\d+);", src)
            assert m and int(m.group(1)) == getattr(rp, name), (name, m and m.group(1), getattr(rp, name))
        assert "INPUT_X = FAR_X + NODE_W + COL_GAP" in src and rp.INPUT_X == rp.FAR_X + rp.NODE_W + rp.COL_GAP
        assert "SPINE_X = INPUT_X + NODE_W + COL_GAP" in src and rp.SPINE_X == rp.INPUT_X + rp.NODE_W + rp.COL_GAP
    # automatic run: the reviewer-input slots with nothing on record feed nothing (the frontend hides edge-less not-run
    # nodes exactly like diagram.svg omits them)
    ga = rp.build_run_graph(synth_automatic_case, want_images=False)
    for nid in ("n02_top_lines", "n05_bottom_lines", "n11_pane_edits", "n02b_axial_gt"):
        n = _node(ga, nid)
        assert n["status"] == "not_run"
        assert not [e for e in ga["edges"] if e["kind"] != "note" and nid in (e["from"], e["to"])], nid
    lay = rp.layout_graph(ga)
    assert "n11_pane_edits" in lay["omitted"] and not lay["has_far_column"]
    # corrections run with pending pane edits: n11 keeps its edges (labelled pending) so the pending edits stay visible
    g = rp.build_run_graph(synth_corrections_case, want_images=False)
    e = {(x["from"], x["to"]): x["label"] for x in g["edges"]}
    assert e[("n11_pane_edits", "n02_top_lines")].startswith("pending") and "not consumed" in e[("n11_pane_edits", "n05_bottom_lines")]
    # consumed (line mode) → far column in use: n11 at FAR_X, inputs at INPUT_X, spine at SPINE_X, edge left→right
    cid = synth_corrections_case
    m = orch.read_manifest(cid)
    m["oct_params"].pop("corrected_edge_anchors"); m["oct_params"].pop("corrected_post_anchors", None)
    m["oct_iter"]["corrected_fold"] = {"folded": True, "mode": "line", "n_points": 3, "laterals": [23], "n_bottom_points": 0,
                                       "bottom_folded_slices": [], "excluded_laterals": [23]}
    orch.write_manifest_value(cid, m); rp._GRAPH_CACHE.clear()
    g2 = rp.build_run_graph(cid, want_images=False)
    lay2 = rp.layout_graph(g2)
    assert lay2["has_far_column"] and lay2["nodes"]["n11_pane_edits"]["x"] == rp.FAR_X
    assert lay2["nodes"]["n02_top_lines"]["x"] == rp.INPUT_X and lay2["nodes"]["n04_served_surface"]["x"] == rp.SPINE_X
    ed = [x for x in lay2["edges"] if x["from"] == "n11_pane_edits" and x["to"] == "n02_top_lines"][0]
    assert ed["label"] == "folded (line mode)" and ed["points"][0][0] < ed["points"][1][0]
    svg = rp.render_diagram_svg(g2, lay2)
    assert "folded (line mode)" in svg and f'width="{lay2["width"]}"' in svg
    ET.fromstring(svg)
