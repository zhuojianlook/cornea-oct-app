"""The SCRUB composite for ANY number of members (reviewer 2026-09-12: "the sagittal scrub only seems to show 3 scans when
there are more than 3 scans in the subgroup, maybe some horizontal scaling is required"): group_job.scrub_layout wraps the
placed member columns into rows of ≤ SCRUB_MAX_COLS (4) plus 'all' at the end of the last row (≤ 1800 px wide), the
refused members of the subgroup get a panel each in a "NOT ALIGNED — <reason>" strip showing their OWN middle sagittal
(scrub/own_<cid>.npy), scrub/meta.json lists every member under 'roster' with a role, load_scrub reads the own images, and
render_scrub_png produces a PNG of the layout's size with the strip painted. Synthetic memmaps in a tempdir, no engine."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest

import group_job as gj


def _write_scrub(tmp: Path, placed: list, refused: list, Lc=6, Dc=140, Fc=30, own_D=120, own_F=30) -> Path:
    out = tmp / "align_min"
    sd = out / gj.SCRUB_DIR
    sd.mkdir(parents=True)
    rng = np.random.default_rng(3)
    lines = {}; lines_pairs = {}
    for k, cid in enumerate(placed):
        for st in gj.SCRUB_STAGES:
            v = (40 + 60 * rng.random((Lc, Dc, Fc))).astype(np.uint8)
            v[:, :20, :] = 0
            np.save(sd / gj.scrub_volume_name(st, cid), np.ascontiguousarray(v))
        lines[cid] = 60.0 + 2.0 * k + 0.02 * (np.arange(Fc) - Fc / 2) ** 2 + np.zeros((Lc, 1))
        lines_pairs[cid] = lines[cid] + 1.0
    curve = 61.0 + 0.02 * (np.arange(Fc) - Fc / 2) ** 2 + np.zeros((Lc, 1))
    np.savez_compressed(out / gj.ALIGNED_LINES, consensus=curve, **{f"line_{c}": lines[c] for c in placed},
                        **{f"line_pairs_{c}": lines_pairs[c] for c in placed})
    roster = [{"cid": c, "role": ("reference" if k == 0 else "contributing"), "is_reference": k == 0, "ok": True, "placed": True}
              for k, c in enumerate(placed)]
    ref_recs = []
    for k, (cid, reason, with_image) in enumerate(refused):
        rec = {"cid": cid, "role": reason, "is_reference": False, "ok": False, "placed": False, "reject_flags": ["dx_beyond_max"],
               "reason": reason, "colour": gj.MEMBER_COLOURS[(len(placed) + k) % len(gj.MEMBER_COLOURS)]}
        if with_image:
            img = (50 + 100 * rng.random((own_D, own_F))).astype(np.uint8)
            np.save(sd / f"own_{cid}.npy", img)
            rec["own"] = {"file": f"own_{cid}.npy", "lateral": 256, "shape": [own_D, own_F], "window": [0.0, 1000.0], "colour": rec["colour"],
                          "line": [50.0 + 0.01 * (f - own_F / 2) ** 2 for f in range(own_F)], "depth_window": [0, own_D]}
        else:
            rec["own"] = None
        roster.append({k2: v2 for k2, v2 in rec.items() if k2 not in ("own", "colour")})
        ref_recs.append(rec)
    rms = {c: {st: [1.0 + 0.1 * k] * Lc for st in gj.SCRUB_STAGES} for k, c in enumerate(placed)}
    meta = {"group": "g", "reference": placed[0], "members": list(placed),
            "colours": {c: gj.MEMBER_COLOURS[k % len(gj.MEMBER_COLOURS)] for k, c in enumerate(placed)},
            "channels": {c: gj.CHANNEL_NAMES[k % 3] for k, c in enumerate(placed)},
            "canvas": {"origin": [0, -10, 0], "shape": [Lc, Dc, Fc]}, "laterals": Lc, "depth": Dc, "frames": Fc,
            "covered_range": [0, Lc - 1], "default_lateral": 2, "stages": list(gj.SCRUB_STAGES), "rms": rms,
            "roster": roster, "roles": {r["cid"]: r["role"] for r in roster}, "all_members": [r["cid"] for r in roster],
            "n_members": len(roster), "n_contributing": len(placed), "refused": ref_recs,
            "files": {c: {st: gj.scrub_volume_name(st, c) for st in gj.SCRUB_STAGES} for c in placed}}
    (sd / gj.SCRUB_META).write_text(json.dumps(meta))
    return out


def _png_size(png: bytes) -> tuple:
    from PIL import Image
    im = Image.open(BytesIO(png)); im.load()
    return im.size, np.asarray(im.convert("L"))


def test_scrub_layout_wraps_columns_and_stays_under_1800_px():
    # 3 placed + 'all' = one row per stage (the CS001_OS layout, unchanged), no strip
    lay = gj.scrub_layout(3, 0, tuple(gj.SCRUB_STAGES))
    assert lay["W"] == 46 + 4 * 346 == 1430 and lay["n_cols"] == 4 and lay["strip_y0"] is None
    assert [r["cols"] for r in lay["rows"]] == [[0, 1, 2, "all"], [0, 1, 2, "all"]]
    # 6 placed: rows of 4 + [2, all] per stage
    lay = gj.scrub_layout(6, 0, tuple(gj.SCRUB_STAGES))
    assert [(r["stage"], r["cols"]) for r in lay["rows"]] == [("before", [0, 1, 2, 3]), ("before", [4, 5, "all"]), ("after", [0, 1, 2, 3]), ("after", [4, 5, "all"])]
    assert lay["W"] == 1430 and lay["member_rows_per_stage"] == 2 and lay["refused_rows"] == 0
    # 4 placed + 'all' = 5 columns (the widest a row gets) ≤ 1800; 5 refused wrap into one strip row of 5
    lay = gj.scrub_layout(4, 5, tuple(gj.SCRUB_STAGES))
    assert lay["W"] == 46 + 5 * 346 == 1776 <= 1800 and lay["n_cols"] == 5
    assert lay["rows"][-1]["stage"] == "refused" and len(lay["rows"][-1]["cols"]) == 5 and lay["strip_y0"] is not None
    # 8 placed + 6 refused: never wider than 1800, two strip rows
    lay = gj.scrub_layout(8, 6, tuple(gj.SCRUB_STAGES))
    assert lay["W"] <= 1800 and lay["refused_rows"] == 2 and sum(len(r["cols"]) for r in lay["rows"] if r["stage"] == "refused") == 6
    # a single stage
    lay1 = gj.scrub_layout(3, 0, ("after",))
    assert len(lay1["rows"]) == 1 and lay1["H"] < gj.scrub_layout(3, 0, tuple(gj.SCRUB_STAGES))["H"]
    # every row's y0 increases and the strip sits below the member rows
    lay = gj.scrub_layout(5, 2, tuple(gj.SCRUB_STAGES))
    ys = [r["y0"] for r in lay["rows"]]
    assert ys == sorted(ys) and lay["strip_y0"] > max(r["y0"] for r in lay["rows"] if r["stage"] != "refused")


def test_render_six_members_three_refused_paints_the_not_aligned_strip(tmp_path):
    placed = ["case_x_v2", "case_x_v1", "case_x_v3"]
    refused = [("case_x_v4", "refused: no_correspondence, dx_beyond_max (offset ≈ -383 laterals, overlap 25%)", True),
               ("case_x_v5", "refused: dx_beyond_max (offset ≈ -403 laterals, overlap 21%)", True),
               ("case_x_v6", "refused: dx_beyond_max (offset ≈ -392 laterals, overlap 24%)", False)]
    out = _write_scrub(tmp_path, placed, refused)
    data = gj.load_scrub(out)
    assert data is not None and data["meta"]["members"] == placed and set(data["own"]) == {"case_x_v4", "case_x_v5"}
    assert data["meta"]["n_members"] == 6 and data["meta"]["n_contributing"] == 3 and len(data["meta"]["refused"]) == 3
    assert [r["role"] for r in data["meta"]["roster"]][:3] == ["reference", "contributing", "contributing"]
    lay = gj.scrub_layout(3, 3, tuple(gj.SCRUB_STAGES))
    png = gj.render_scrub_png(data, 2)
    (W, H), g = _png_size(png)
    assert (W, H) == (lay["W"], lay["H"]) and W <= 1800
    # the strip: the two refused members WITH an image are painted (random texture → high variance), the third is a flat
    # "no image" panel; every strip panel sits below the last member row
    strip = [r for r in lay["rows"] if r["stage"] == "refused"][0]
    pw, ph = lay["panel_w"], lay["panel_h"]
    for j, (_t, k) in enumerate(strip["cols"]):
        x0 = strip["x0"] + j * (pw + lay["G"]); y0 = strip["y0"]
        block = g[y0 + 20:y0 + ph - 20, x0 + 20:x0 + pw - 20]
        if refused[k][2]:
            assert block.std() > 15 and block.mean() > 60, (k, block.std(), block.mean())
        else:
            assert block.std() < 8 and block.mean() < 60, (k, block.std(), block.mean())
    # the member rows are painted too (the memmaps' texture), and the 'all' panel of each stage
    for row in [r for r in lay["rows"] if r["stage"] != "refused"]:
        for j, c in enumerate(row["cols"]):
            x0 = row["x0"] + j * (pw + lay["G"]); y0 = row["y0"]
            block = g[y0 + 60:y0 + ph - 20, x0 + 20:x0 + pw - 20]
            assert block.std() > 5, (row["stage"], c)
    # a single stage renders the strip as well, half the member rows
    png1 = gj.render_scrub_png(data, 2, stage="after")
    (W1, H1), _ = _png_size(png1)
    assert W1 == W and H1 == gj.scrub_layout(3, 3, ("after",))["H"] < H
    with pytest.raises(ValueError):
        gj.render_scrub_png(data, 99)


def test_render_seven_placed_members_wraps_rows_and_no_strip(tmp_path):
    placed = [f"case_y_v{k}" for k in range(1, 8)]
    out = _write_scrub(tmp_path, placed, [])
    data = gj.load_scrub(out)
    assert data is not None and data["own"] == {} and data["meta"]["refused"] == []
    lay = gj.scrub_layout(7, 0, tuple(gj.SCRUB_STAGES))
    assert [(r["stage"], r["cols"]) for r in lay["rows"]] == [("before", [0, 1, 2, 3]), ("before", [4, 5, 6, "all"]),
                                                              ("after", [0, 1, 2, 3]), ("after", [4, 5, 6, "all"])]
    png = gj.render_scrub_png(data, 1)
    (W, H), g = _png_size(png)
    assert (W, H) == (lay["W"], lay["H"]) and W == 1430 <= 1800          # [4] + [3, all] → 4 columns
    # the second row of each stage is painted (the wrapped members), including the 'all' panel at its end
    for row in [r for r in lay["rows"] if r["row"] == 1]:
        for j, c in enumerate(row["cols"]):
            x0 = row["x0"] + j * (lay["panel_w"] + lay["G"]); y0 = row["y0"]
            assert g[y0 + 60:y0 + lay["panel_h"] - 20, x0 + 20:x0 + lay["panel_w"] - 20].std() > 5, (row["stage"], c)


def test_member_role_and_roster():
    recs = [{"cid": "a", "is_reference": True, "ok": True, "reject_flags": []},
            {"cid": "b", "is_reference": False, "ok": True, "reject_flags": [], "dx_median": 26.1, "overlap_fraction": 0.95},
            {"cid": "c", "is_reference": False, "ok": False, "reject_flags": ["dx_beyond_max"], "dx_median": -402.7, "overlap_fraction": 0.214,
             "non_contributing": "refused: dx_beyond_max (no_overlap: offset ≈ -403 laterals, overlap 21%)", "df": 5},
            {"cid": "d", "is_reference": False, "ok": False, "reject_flags": ["no_correspondence"], "dx_median": float("nan"), "overlap_fraction": None,
             "overlap": {"offset": {"dx": -390.0, "fraction": 0.24}}, "non_contributing": "refused: no_correspondence"},
            {"cid": "e", "is_reference": False, "ok": False, "reject_flags": ["pose_beyond_frame_rigid"], "dx_median": 10.0, "overlap_fraction": 0.9,
             "non_contributing": "pose_beyond_frame_rigid (15.3 deg > 8.0 deg)"}]
    assert gj.member_role(recs[0]) == "reference" and gj.member_role(recs[1]) == "contributing"
    assert gj.member_role(recs[2]) == "refused: dx_beyond_max (offset ≈ -403 laterals, overlap 21%)"
    assert gj.member_role(recs[3]) == "refused: no_correspondence (offset ≈ -390 laterals, overlap 24%)"
    assert gj.member_role(recs[4]) == "refused: pose_beyond_frame_rigid (offset ≈ +10 laterals, overlap 90%)"
    ros = gj.build_roster(recs)
    assert [r["placed"] for r in ros] == [True, True, False, False, False] and ros[2]["df"] == 5 and ros[2]["overlap_fraction"] == 0.214
    assert json.dumps(gj.jsonable(ros))


def test_write_scrub_meta_keeps_the_refused_records_and_own_sagittals(tmp_path):
    """write_scrub_meta merges its summary into meta.json (**summ): the refused records with their OWN-sagittal files must survive
    (a shared 'refused' key once replaced them with the cid/role summary) and load_scrub must read the own images back."""
    import types
    placed = ["case_z_v2", "case_z_v1"]
    out = _write_scrub(tmp_path, placed, [], Lc=6, Dc=140, Fc=30)
    sd = out / gj.SCRUB_DIR
    Lc, Dc, Fc = 6, 140, 30
    rng = np.random.default_rng(7)
    vol = (30 + 400 * rng.random((41, 96, Fc))).astype(np.float32)
    served = np.full((41, Fc), 20.0); valid = np.ones((41, Fc), bool)
    m4 = types.SimpleNamespace(cid="case_z_v4", volume=vol, served=served, valid=valid, spacing=np.array([0.0078, 0.0031, 0.04]))
    own = gj.own_sagittal_record(m4, sd, "#ffc83c")
    assert own["file"] == "own_case_z_v4.npy" and (sd / own["file"]).exists() and own["lateral"] == 20 and own["shape"] == [96, Fc]
    assert len(own["line"]) == Fc and own["depth_window"][0] == 0
    ref = types.SimpleNamespace(cid=placed[0], volume=np.zeros((6, 140, Fc), np.float32), spacing=np.array([0.0078, 0.0031, 0.04]))
    z = np.load(out / gj.ALIGNED_LINES)
    lines = {c: np.asarray(z[f"line_{c}"], float) for c in placed}; lines_pairs = {c: np.asarray(z[f"line_pairs_{c}"], float) for c in placed}
    curve = np.asarray(z["consensus"], float)
    files = {c: {"scrub": {st: {"file": gj.scrub_volume_name(st, c), "bytes": Lc * Dc * Fc} for st in gj.SCRUB_STAGES}} for c in placed}
    roster = [{"cid": placed[0], "role": "reference", "is_reference": True, "ok": True, "placed": True},
              {"cid": placed[1], "role": "contributing", "is_reference": False, "ok": True, "placed": True},
              {"cid": "case_z_v4", "role": "refused: dx_beyond_max (offset ≈ -403 laterals, overlap 21%)", "is_reference": False, "ok": False, "placed": False,
               "reject_flags": ["dx_beyond_max"], "df": 5, "dx_median": -402.7, "overlap_fraction": 0.21}]
    refused = [dict(roster[2], own=own, colour="#ffc83c")]
    summ = gj.write_scrub_meta("g", ref, [ref], {"origin": [0, -10, 0], "shape": [Lc, Dc, Fc]}, lines_pairs, lines, curve, np.ones((Lc, Fc), bool),
                               {c: gj.MEMBER_COLOURS[i] for i, c in enumerate(placed)}, {c: gj.CHANNEL_NAMES[i] for i, c in enumerate(placed)}, 255.0, files, sd,
                               roster=roster, refused=refused)
    assert summ["n_members"] == 3 and summ["n_contributing"] == 2 and summ["refused_ids"] == [{"cid": "case_z_v4", "role": roster[2]["role"]}]
    meta = json.loads((sd / gj.SCRUB_META).read_text())
    assert meta["members"] == placed and meta["n_members"] == 3 and meta["n_contributing"] == 2 and meta["all_members"] == [placed[0], placed[1], "case_z_v4"]
    assert [r["role"] for r in meta["roster"]] == ["reference", "contributing", roster[2]["role"]]
    r4 = meta["refused"][0]
    assert r4["cid"] == "case_z_v4" and r4["own"]["file"] == "own_case_z_v4.npy" and r4["colour"] == "#ffc83c" and r4["dx_median"] == -402.7
    assert r4["own"]["lateral"] == 20 and len(r4["own"]["line"]) == Fc
    data = gj.load_scrub(out)
    assert data is not None and set(data["own"]) == {"case_z_v4"} and data["own"]["case_z_v4"].shape == (96, Fc)
    png = gj.render_scrub_png(data, 2)
    (W, H), _ = _png_size(png)
    assert (W, H) == (gj.scrub_layout(2, 1, tuple(gj.SCRUB_STAGES))["W"], gj.scrub_layout(2, 1, tuple(gj.SCRUB_STAGES))["H"])
