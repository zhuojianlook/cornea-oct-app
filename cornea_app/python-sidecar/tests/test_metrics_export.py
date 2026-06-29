"""Unit tests for metrics_export.py — the cross-case scar quantification export.

Covers the pure / deterministic helpers:
  * parse_case_meta   — filename -> patient/eye/date/variant regex parse
  * resolve_case_meta — manifest-aware identity (consensus inheritance, oct_source vs
                        input_volume precedence, user override of patient/eye)
  * build_row         — recompute scar/cornea metrics from the corrected labelmap + base
                        volume (exact numbers on known synthetic labelmaps)
  * build_summary     — aggregate rows (auto-discovery + explicit id list, None-skipping)
  * write_summary     — CSV + JSON deliverables (columns, ordering, missing-key fill)

All arrays are tiny (<=24 voxels per axis). No network/GPU/SAM2/torch/real data.

Geometry note: the conftest write_nifti uses affine diag([0.02, 0.02, 0.04, 1.0]),
so nibabel get_zooms() == (0.02, 0.02, 0.04) on the (frames, depth, lateral) axes.
With the default make_case labelmap (cornea band along depth axis 1, depth_axis==1),
voxel volume = 0.02*0.02*0.04 = 1.6e-5 mm3 and the en-face pixel area (axes 0 & 2)
= 0.02*0.04 = 8e-4 mm2. These constants pin the expected numbers below.
"""
from __future__ import annotations

import csv
import json

import numpy as np
import pytest

import metrics_export as me
import orchestration as orch
import labels


# Geometry constants implied by the conftest default affine.
SP = (0.02, 0.02, 0.04)
VOXEL_MM3 = SP[0] * SP[1] * SP[2]          # 1.6e-5
ENFACE_PIXEL_MM2 = SP[0] * SP[2]           # 8e-4  (plane axes 0 & 2, depth axis 1)


# ───────────────────────── parse_case_meta ─────────────────────────

def test_parse_case_meta_full_filename_with_variant():
    meta = me.parse_case_meta(
        "preprocessed_CS001_14145_3D Cornea_OD_2024-07-11 (2)_0.dcm")
    assert meta == {"patient_id": "CS001", "eye": "OD",
                    "date": "2024-07-11", "variant": "2"}


def test_parse_case_meta_no_variant_defaults_empty_string():
    meta = me.parse_case_meta("preprocessed_CS021_99_foo_OS_2023-01-02.dcm")
    assert meta == {"patient_id": "CS021", "eye": "OS",
                    "date": "2023-01-02", "variant": ""}


def test_parse_case_meta_uppercases_patient_and_eye():
    # lowercase pid + lowercase eye in a case-insensitive match must be normalised.
    meta = me.parse_case_meta("preprocessed_cs042_12_z_os_2021-12-31.dcm")
    assert meta["patient_id"] == "CS042"
    assert meta["eye"] == "OS"


def test_parse_case_meta_none_returns_blank_dict():
    assert me.parse_case_meta(None) == {
        "patient_id": "", "eye": "", "date": "", "variant": ""}


def test_parse_case_meta_unparseable_name_returns_blanks():
    # No O[DS]/date pattern -> regex misses -> all fields blank (never raises).
    assert me.parse_case_meta("random_volume_file.nii.gz") == {
        "patient_id": "", "eye": "", "date": "", "variant": ""}


def test_parse_case_meta_uses_basename_not_directory():
    # A misleading parent directory must not be parsed; only the file name.
    meta = me.parse_case_meta(
        "/some/OS/2099-09-09/preprocessed_CS010_7_a_OD_2024-01-02.dcm")
    assert meta["eye"] == "OD"
    assert meta["date"] == "2024-01-02"
    assert meta["patient_id"] == "CS010"


# ───────────────────────── resolve_case_meta ─────────────────────────

def test_resolve_case_meta_from_oct_source(cases_root):
    cid = "wk_cs005_od"
    orch.write_manifest_value(
        cid, {"oct_source": "preprocessed_CS005_3_x_OD_2024-03-04 (1)_0.dcm"})
    meta = me.resolve_case_meta(cid)
    assert meta == {"patient_id": "CS005", "eye": "OD",
                    "date": "2024-03-04", "variant": "1"}


def test_resolve_case_meta_oct_source_wins_over_input_volume(cases_root):
    # oct_source preserves the (N) replicate suffix; it is tried first.
    cid = "wk_pref"
    orch.write_manifest_value(cid, {
        "oct_source": "preprocessed_CS001_1_x_OD_2024-07-11 (3)_0.dcm",
        "input_volume": "preprocessed_CS999_9_x_OS_2000-01-01.dcm",
    })
    meta = me.resolve_case_meta(cid)
    assert meta["patient_id"] == "CS001"
    assert meta["variant"] == "3"


def test_resolve_case_meta_falls_back_to_input_volume(cases_root):
    cid = "wk_inputonly"
    orch.write_manifest_value(
        cid, {"input_volume": "preprocessed_CS012_4_x_OS_2022-02-02.dcm"})
    meta = me.resolve_case_meta(cid)
    assert meta["patient_id"] == "CS012"
    assert meta["eye"] == "OS"


def test_resolve_case_meta_consensus_inherits_reference_identity(cases_root):
    ref = "memberA"
    orch.write_manifest_value(
        ref, {"oct_source": "preprocessed_CS007_55_x_OS_2022-05-06.dcm"})
    con = "consensusA"
    orch.write_manifest_value(con, {"consensus_report": {"reference": ref}})
    meta = me.resolve_case_meta(con)
    assert meta["patient_id"] == "CS007"
    assert meta["eye"] == "OS"
    assert meta["date"] == "2022-05-06"
    # consensus row is always tagged variant=consensus (the deliverable biomarker key).
    assert meta["variant"] == "consensus"


def test_resolve_case_meta_consensus_self_reference_does_not_recurse(cases_root):
    # reference == case_id must NOT be treated as a consensus inherit (no infinite loop);
    # it parses its own oct_source instead.
    cid = "selfref"
    orch.write_manifest_value(cid, {
        "consensus_report": {"reference": cid},
        "oct_source": "preprocessed_CS002_2_x_OD_2021-01-01.dcm",
    })
    meta = me.resolve_case_meta(cid)
    assert meta["patient_id"] == "CS002"
    assert meta["variant"] == ""  # NOT "consensus"


def test_resolve_case_meta_manifest_override_wins_for_id_and_eye(cases_root):
    # A persisted group-header edit of patient_id/eye overrides the filename parse,
    # but date/variant still come from the filename.
    cid = "wk_override"
    orch.write_manifest_value(cid, {
        "input_volume": "preprocessed_CS003_11_x_OD_2020-01-01.dcm",
        "patient_id": "cs999",  # lowercase -> must be uppercased
        "eye": "os",
    })
    meta = me.resolve_case_meta(cid)
    assert meta["patient_id"] == "CS999"
    assert meta["eye"] == "OS"
    assert meta["date"] == "2020-01-01"   # from filename, untouched by override


def test_resolve_case_meta_missing_manifest_is_all_blank(cases_root):
    assert me.resolve_case_meta("never_created") == {
        "patient_id": "", "eye": "", "date": "", "variant": ""}


# ───────────────────────── build_row (exact numbers) ─────────────────────────

def test_build_row_exact_volumes_and_area(make_case):
    """Default make_case labelmap: cornea band [:,10:14,:], scar [2:6,11:13,6:12].
      scar voxels   = 4*2*6  = 48
      cornea tissue = 8*4*20 = 640  (cornea + scar union)
      depth_axis    = 1 -> en-face area projects frames(4) x lateral(6) = 24 px.
    """
    cid = make_case("wk_default")
    row = me.build_row(cid)
    assert row is not None

    scar_vox, cornea_vox, footprint_px = 48, 640, 24
    assert row["scar_present"] is True
    assert row["scar_volume_mm3"] == pytest.approx(scar_vox * VOXEL_MM3)
    assert row["cornea_volume_mm3"] == pytest.approx(cornea_vox * VOXEL_MM3)
    assert row["scar_area_mm2"] == pytest.approx(footprint_px * ENFACE_PIXEL_MM2)
    assert row["scar_fraction_of_cornea"] == pytest.approx(
        round(scar_vox / cornea_vox, 4))
    assert row["label_source"] == "corrected"
    assert row["case"] == cid


def test_build_row_density_mean_and_weighted_volume(make_case):
    """The base volume's raw reflectivity feeds densitometry. Default make_case
    paints every scar voxel at 360, so mean == 360 and the density-weighted volume
    == sum(reflectivity) * voxel_mm3 == 48*360*VOXEL_MM3."""
    cid = make_case("wk_density")
    row = me.build_row(cid)
    assert row["scar_density_mean"] == pytest.approx(360.0)
    assert row["scar_density_weighted_mm3u"] == pytest.approx(
        round(48 * 360 * VOXEL_MM3, 4))


def test_build_row_carries_parsed_meta_from_manifest(make_case):
    cid = make_case(
        "wk_meta",
        manifest={"oct_source": "preprocessed_CS088_5_x_OS_2023-08-09 (4)_0.dcm"})
    row = me.build_row(cid)
    assert row["patient_id"] == "CS088"
    assert row["eye"] == "OS"
    assert row["date"] == "2023-08-09"
    assert row["variant"] == "4"


def test_build_row_no_scar_reports_absent(make_case):
    """A pure-cornea labelmap (no SCAR class): scar metrics collapse to 0/absent and
    NO density fields are produced (quantify only computes density when scar present)."""
    vol = np.full((8, 24, 20), 20, np.uint16)
    vol[:, 10:14, :] = 200
    lab = np.zeros(vol.shape, np.uint8)
    lab[:, 10:14, :] = 1   # cornea only
    cid = make_case("wk_noscar", vol=vol, lab=lab)
    row = me.build_row(cid)
    assert row["scar_present"] is False
    assert row["scar_volume_mm3"] == 0.0
    assert row["scar_area_mm2"] == 0.0
    # cornea still measured: 8*4*20 = 640 voxels.
    assert row["cornea_volume_mm3"] == pytest.approx(640 * VOXEL_MM3)
    assert row["scar_fraction_of_cornea"] == 0.0
    # density helpers return "" sentinel when scar absent.
    assert row["scar_density_mean"] == ""
    assert row["scar_density_weighted_mm3u"] == ""


def test_build_row_returns_none_without_labelmap(cases_root):
    """A case with a base volume but NO corrected labelmap yields None (skipped row)."""
    cid = "wk_nolabel"
    orch.ensure_case_dirs(cid)
    base = orch.case_root(cid) / "previews" / "volume.nii.gz"
    import nibabel as nib
    nib.save(nib.Nifti1Image(np.zeros((4, 4, 4), np.uint16),
                             np.diag([0.02, 0.02, 0.04, 1.0])), str(base))
    assert not labels.corrected_path(cid).exists()
    assert me.build_row(cid) is None


def test_build_row_returns_none_without_base_volume(cases_root, write_nifti):
    """A corrected labelmap with no previews/volume.nii.gz base yields None."""
    cid = "wk_nobase"
    orch.ensure_case_dirs(cid)
    # write_label_nifti needs a base to stamp the affine; use a scratch file outside
    # the previews/ path so _base_volume() can't find it.
    scratch = orch.case_root(cid) / "segmentation" / "scratch_base.nii.gz"
    lab = np.zeros((4, 4, 4), np.uint8)
    lab[:, 1:3, :] = 1
    write_nifti(np.zeros((4, 4, 4), np.uint16), scratch)
    labels.write_label_nifti(lab, scratch, labels.corrected_path(cid))
    assert labels.corrected_path(cid).exists()
    assert me._base_volume(cid) is None
    assert me.build_row(cid) is None


def test_build_row_scales_with_spacing(cases_root, write_nifti, make_case):
    """Volume scales exactly with voxel size: doubling every spacing axis -> 8x volume."""
    cid = make_case("wk_smallsp")
    row_small = me.build_row(cid)

    cid2 = "wk_bigsp"
    orch.ensure_case_dirs(cid2)
    vol = np.full((8, 24, 20), 20, np.uint16)
    vol[:, 10:14, :] = 200
    vol[2:6, 11:13, 6:12] = 360
    lab = np.zeros(vol.shape, np.uint8)
    lab[:, 10:14, :] = 1
    lab[2:6, 11:13, 6:12] = 2
    pv = orch.case_root(cid2) / "previews" / "volume.nii.gz"
    big_affine = np.diag([0.04, 0.04, 0.08, 1.0])   # 2x each axis
    write_nifti(vol, pv, big_affine)
    labels.write_label_nifti(lab, pv, labels.corrected_path(cid2))
    orch.write_manifest_value(cid2, {"input_volume": str(pv)})
    row_big = me.build_row(cid2)

    assert row_big["scar_volume_mm3"] == pytest.approx(
        row_small["scar_volume_mm3"] * 8, rel=1e-6)
    # en-face area scales with the two in-plane axes -> 4x.
    assert row_big["scar_area_mm2"] == pytest.approx(
        row_small["scar_area_mm2"] * 4, rel=1e-6)


# ───────────────────────── build_summary ─────────────────────────

def test_build_summary_autodiscovers_cases_with_labelmaps(make_case, cases_root):
    a = make_case("aa_case", manifest={
        "oct_source": "preprocessed_CS001_1_x_OD_2024-01-01.dcm"})
    b = make_case("bb_case", manifest={
        "oct_source": "preprocessed_CS002_1_x_OS_2024-02-02.dcm"})
    # a directory with NO corrected labelmap must be ignored by auto-discovery.
    orch.ensure_case_dirs("cc_no_label")
    rows = me.build_summary()
    cases = {r["case"] for r in rows}
    assert cases == {a, b}
    assert "cc_no_label" not in cases


def test_build_summary_explicit_ids_skips_none_rows(make_case, cases_root):
    good = make_case("good_case")
    rows = me.build_summary([good, "missing_case"])
    # missing_case produces None (no labelmap) and is dropped, not raised.
    assert [r["case"] for r in rows] == [good]


def test_build_summary_empty_when_no_cases(cases_root):
    assert me.build_summary() == []


# ───────────────────────── write_summary ─────────────────────────

def test_write_summary_writes_csv_and_json(make_case, tmp_path):
    cid = make_case("wk_export", manifest={
        "oct_source": "preprocessed_CS001_1_x_OD_2024-07-11 (2)_0.dcm"})
    rows = me.build_summary([cid])
    out_dir = tmp_path / "out"
    res = me.write_summary(rows, out_dir)

    assert res["n_cases"] == 1
    assert res["csv"] == str(out_dir / "scar_summary.csv")
    assert res["json"] == str(out_dir / "scar_summary.json")
    assert (out_dir / "scar_summary.csv").exists()
    assert (out_dir / "scar_summary.json").exists()

    # JSON round-trips the exact row dicts.
    loaded = json.loads((out_dir / "scar_summary.json").read_text())
    assert loaded == rows

    # CSV header == the canonical column order; one data row with matching values.
    with (out_dir / "scar_summary.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == me._COLUMNS
        csv_rows = list(reader)
    assert len(csv_rows) == 1
    assert csv_rows[0]["case"] == cid
    assert csv_rows[0]["patient_id"] == "CS001"
    assert csv_rows[0]["variant"] == "2"


def test_write_summary_fills_missing_keys_and_ignores_extra(tmp_path):
    """DictWriter is fed only _COLUMNS: absent keys become "" and unexpected keys
    (e.g. an internal field) are dropped, never raising."""
    rows = [{"case": "x1", "patient_id": "CS9",
             "scar_volume_mm3": 1.25, "unexpected_field": "drop_me"}]
    out_dir = tmp_path / "out2"
    res = me.write_summary(rows, out_dir)
    assert res["n_cases"] == 1
    with (out_dir / "scar_summary.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == me._COLUMNS
        row = next(reader)
    assert row["case"] == "x1"
    assert row["scar_volume_mm3"] == "1.25"
    assert row["eye"] == ""          # missing key -> blank
    assert "unexpected_field" not in row


def test_write_summary_empty_rows_writes_header_only(tmp_path):
    out_dir = tmp_path / "empty_out"
    res = me.write_summary([], out_dir)
    assert res["n_cases"] == 0
    text = (out_dir / "scar_summary.csv").read_text()
    # header present, no data rows.
    assert text.strip() == ",".join(me._COLUMNS)
    assert json.loads((out_dir / "scar_summary.json").read_text()) == []


def test_write_summary_creates_nested_out_dir(make_case, tmp_path):
    cid = make_case("wk_nested")
    rows = me.build_summary([cid])
    out_dir = tmp_path / "deep" / "nested" / "out"
    assert not out_dir.exists()
    res = me.write_summary(rows, out_dir)
    assert out_dir.exists()
    assert res["n_cases"] == 1
