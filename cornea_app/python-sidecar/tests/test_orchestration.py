"""Unit tests for orchestration.py — case-id sanitization, manifest read/merge/
write (incl. corruption recovery), scheduled-case filtering, and preview listing.

All tests use tiny synthetic on-disk fixtures under the isolated cases_root; no
network, GPU, SAM2/torch, or real data.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import orchestration as orch
import settings


# ── safe_case_id ────────────────────────────────────────────────────────────
class TestSafeCaseId:
    def test_alnum_and_allowed_punct_preserved(self):
        # letters, digits, and the allowed set _-. survive untouched
        assert orch.safe_case_id("CS001_OD-v2.1") == "CS001_OD-v2.1"

    def test_slashes_spaces_become_underscore(self):
        # a path-like id: slashes and spaces -> _, but leading/trailing _ stripped
        assert orch.safe_case_id("foo/bar baz") == "foo_bar_baz"

    def test_unicode_becomes_underscore(self):
        # non-ASCII letters are NOT isalnum-allowed? Actually str.isalnum() is True for
        # many unicode letters, so they ARE preserved. Punctuation/symbols become _.
        # Use an explicit symbol to prove the mapping.
        assert orch.safe_case_id("a@b#c") == "a_b_c"

    def test_unicode_letters_preserved_by_isalnum(self):
        # documents the real behavior: unicode *letters* pass isalnum() and survive.
        assert orch.safe_case_id("café") == "café"

    def test_leading_trailing_dot_dash_underscore_stripped(self):
        assert orch.safe_case_id("...__--case--__...") == "case"

    def test_interior_dots_dashes_preserved(self):
        assert orch.safe_case_id("a.b-c_d") == "a.b-c_d"

    def test_empty_string_defaults(self):
        assert orch.safe_case_id("") == "case_001"

    def test_none_defaults(self):
        assert orch.safe_case_id(None) == "case_001"

    def test_only_strippable_chars_defaults(self):
        # a string made entirely of leading/trailing-strip chars collapses to empty -> default
        assert orch.safe_case_id("._-._-") == "case_001"

    def test_whitespace_only_defaults(self):
        # outer whitespace is stripped first, leaving nothing
        assert orch.safe_case_id("   ") == "case_001"

    def test_outer_whitespace_stripped_before_mapping(self):
        # leading/trailing spaces are removed by .strip() so they do NOT become _
        assert orch.safe_case_id("  abc  ") == "abc"

    def test_interior_space_maps_then_stripped(self):
        # interior space -> _, but a trailing space (after a real char) maps to _ and is
        # then stripped by .strip("._-"). " a b " -> strip -> "a b" -> "a_b".
        assert orch.safe_case_id(" a b ") == "a_b"


# ── read_manifest ───────────────────────────────────────────────────────────
class TestReadManifest:
    def test_missing_file_returns_empty_dict(self, cases_root):
        result = orch.read_manifest("nope_case")
        assert result == {}
        assert isinstance(result, dict)

    def test_non_dict_json_returns_empty_dict(self, cases_root):
        # a manifest holding a JSON list is coerced to {} (read_manifest guards isinstance)
        path = orch.manifest_path("listy")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([1, 2, 3]))
        assert orch.read_manifest("listy") == {}

    def test_reads_back_written_value(self, cases_root):
        orch.write_manifest_value("rt", {"k": "v"})
        assert orch.read_manifest("rt")["k"] == "v"


# ── write_manifest_value ────────────────────────────────────────────────────
class TestWriteManifestValue:
    def test_stamps_case_id(self, cases_root):
        out = orch.write_manifest_value("Stamp Me", {"a": 1})
        # case_id is the *sanitized* id, not the raw input
        assert out["case_id"] == orch.safe_case_id("Stamp Me") == "Stamp_Me"
        assert out["a"] == 1

    def test_returns_value_matches_disk(self, cases_root):
        out = orch.write_manifest_value("ondisk", {"oct_source": "/x.OCT"})
        on_disk = json.loads(orch.manifest_path("ondisk").read_text())
        assert on_disk == out
        assert on_disk["oct_source"] == "/x.OCT"

    def test_second_write_preserves_prior_keys(self, cases_root):
        orch.write_manifest_value("merge", {"first": 1, "shared": "old"})
        out = orch.write_manifest_value("merge", {"second": 2, "shared": "new"})
        # prior key kept, new key added, overlapping key overwritten by the update
        assert out["first"] == 1
        assert out["second"] == 2
        assert out["shared"] == "new"
        assert out["case_id"] == "merge"

    def test_updates_can_override_case_id_field(self, cases_root):
        # case_id is stamped THEN updates applied, so an explicit case_id in updates wins
        out = orch.write_manifest_value("orig", {"case_id": "forced"})
        assert out["case_id"] == "forced"

    def test_writes_to_sanitized_path(self, cases_root):
        orch.write_manifest_value("a b/c", {"x": 1})
        # the file lands under the sanitized id, not the raw one
        sanitized = orch.safe_case_id("a b/c")
        assert (settings.CASES_ROOT / sanitized / "manifest.json").exists()


# ── manifest corruption recovery ────────────────────────────────────────────
class TestManifestCorruptionRecovery:
    def _write_raw(self, case_id: str, raw_bytes: bytes) -> Path:
        path = orch.manifest_path(case_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw_bytes)
        return path

    def test_corrupt_manifest_backed_up_and_new_value_saved(self, cases_root):
        # Decodable text that is NOT valid JSON -> the recovery path backs it up.
        # (Non-UTF-8 garbage hits read_text's except -> treated as empty, no backup;
        #  that behavior is covered separately in test_undecodable_bytes_no_backup.)
        cid = "corrupt"
        garbage = b"{this is not valid json!!! oops"
        path = self._write_raw(cid, garbage)
        backup = path.with_suffix(path.suffix + ".corrupt")
        assert not backup.exists()

        out = orch.write_manifest_value(cid, {"fresh": True})

        # a .corrupt backup of the garbage was created
        assert backup.exists()
        # the backup holds the original garbage bytes (not the new manifest)
        assert backup.read_bytes() == garbage
        # the new manifest is valid JSON with the new value + stamped id
        assert out == {"case_id": cid, "fresh": True}
        assert json.loads(path.read_text()) == out

    def test_undecodable_bytes_backed_up_then_starts_fresh(self, cases_root):
        # Non-UTF-8 bytes make read_text() raise. We can't merge them, but the file may hold the
        # only copy of oct_source + flags, so it must be BACKED UP to .corrupt (not silently
        # destroyed) before a fresh manifest is written — symmetric with the text-corrupt path.
        cid = "undecodable"
        garbage = b"\xff\xfe\x00\x01 not utf8"
        path = self._write_raw(cid, garbage)
        backup = path.with_suffix(path.suffix + ".corrupt")
        out = orch.write_manifest_value(cid, {"fresh": True})
        assert backup.exists()                      # the binary-corrupt bytes were preserved
        assert backup.read_bytes() == garbage
        assert out == {"case_id": cid, "fresh": True}
        assert json.loads(path.read_text()) == out  # new manifest written cleanly

    def test_prior_keys_not_resurrected_from_garbage(self, cases_root):
        cid = "noresurrect"
        # garbage that *mentions* a plausible prior key — it must NOT leak into the new manifest
        self._write_raw(cid, b'{"oct_source": "/old.OCT", BROKEN')
        out = orch.write_manifest_value(cid, {"new_flag": 1})
        assert "oct_source" not in out
        assert out == {"case_id": cid, "new_flag": 1}

    def test_valid_existing_manifest_not_backed_up(self, cases_root):
        cid = "valid"
        orch.write_manifest_value(cid, {"keep": "me"})
        path = orch.manifest_path(cid)
        backup = path.with_suffix(path.suffix + ".corrupt")
        # a normal merge must NOT create a .corrupt backup
        orch.write_manifest_value(cid, {"also": "here"})
        assert not backup.exists()
        merged = orch.read_manifest(cid)
        assert merged["keep"] == "me" and merged["also"] == "here"

    def test_empty_file_treated_as_no_prior(self, cases_root):
        # an empty/whitespace file is not "corruption": no backup, just start fresh
        cid = "emptyfile"
        path = self._write_raw(cid, b"   \n  ")
        backup = path.with_suffix(path.suffix + ".corrupt")
        out = orch.write_manifest_value(cid, {"a": 1})
        assert not backup.exists()
        assert out == {"case_id": cid, "a": 1}

    def test_non_dict_json_overwritten_without_backup(self, cases_root):
        # valid-but-not-an-object JSON (a list) parses fine, so no .corrupt backup; it is
        # simply replaced because `current` falls back to {} when parsed is not a dict.
        cid = "jsonlist"
        path = self._write_raw(cid, b"[1, 2, 3]")
        backup = path.with_suffix(path.suffix + ".corrupt")
        out = orch.write_manifest_value(cid, {"a": 1})
        assert not backup.exists()
        assert out == {"case_id": cid, "a": 1}


# ── filter_scheduled ────────────────────────────────────────────────────────
class TestFilterScheduled:
    def test_none_scheduled_returns_all_in_order(self, cases_root):
        ids = ["c3", "c1", "c2"]
        for c in ids:
            orch.write_manifest_value(c, {"oct_preprocessed": True})
        out = orch.filter_scheduled(ids)
        assert out == ids  # order preserved, every case returned

    def test_some_scheduled_returns_only_those_in_order(self, cases_root):
        ids = ["a", "b", "c", "d"]
        for c in ids:
            orch.write_manifest_value(c, {})
        # flag b and d (out of input order on purpose)
        orch.write_manifest_value("d", {"training_scheduled": True})
        orch.write_manifest_value("b", {"training_scheduled": True})
        out = orch.filter_scheduled(ids)
        # only scheduled ones, in the ORIGINAL input order (b before d)
        assert out == ["b", "d"]

    def test_all_scheduled_returns_all(self, cases_root):
        ids = ["x", "y"]
        for c in ids:
            orch.write_manifest_value(c, {"training_scheduled": True})
        assert orch.filter_scheduled(ids) == ids

    def test_empty_input_returns_empty(self, cases_root):
        assert orch.filter_scheduled([]) == []

    def test_falsy_schedule_flag_not_counted(self, cases_root):
        # training_scheduled present but falsy -> treated as not scheduled
        orch.write_manifest_value("p", {"training_scheduled": False})
        orch.write_manifest_value("q", {"training_scheduled": 0})
        out = orch.filter_scheduled(["p", "q"])
        # none truly scheduled -> backward-compatible "return all"
        assert out == ["p", "q"]

    def test_returns_new_list_not_input(self, cases_root):
        ids = ["m", "n"]
        for c in ids:
            orch.write_manifest_value(c, {})
        out = orch.filter_scheduled(ids)
        assert out == ids and out is not ids  # the none-scheduled branch returns list(...)


# ── preview_images_from_dir / preview_listing_from_dir ──────────────────────
_PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000d49444154789c63f8cfc0f01f0005ff02fedccc59e70000000049454e44ae426082"
)


def _write_png(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_PNG_1x1)
    return path


def _write_preview_manifest(directory: Path, images: list[dict]) -> None:
    (directory / "preview_manifest.json").write_text(json.dumps({"images": images}))


class TestPreviewImagesFromDir:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert orch.preview_images_from_dir("g", tmp_path / "absent") == []

    def test_lists_pngs_sorted_with_data_url(self, tmp_path):
        d = tmp_path / "prev"
        _write_png(d / "b.png")
        _write_png(d / "a.png")
        # a non-png file must be ignored
        (d / "ignore.txt").write_text("nope")
        out = orch.preview_images_from_dir("ctx", d)
        assert [i["file_name"] for i in out] == ["a.png", "b.png"]  # sorted
        first = out[0]
        assert first["group"] == "ctx"
        assert first["label"] == "ctx / a.png"
        assert first["path"] == str(d / "a.png")
        assert first["data_url"].startswith("data:image/png;base64,")
        # the base64 payload decodes back to the original PNG bytes
        import base64
        payload = first["data_url"].split(",", 1)[1]
        assert base64.b64decode(payload) == _PNG_1x1

    def test_metadata_merged_from_manifest(self, tmp_path):
        d = tmp_path / "prev"
        _write_png(d / "s.png")
        _write_preview_manifest(d, [{
            "file_name": "s.png", "orientation": "sagittal", "slice_index": 7,
            "source_width": 640, "source_height": 480,
            "image_width": 320, "image_height": 240,
        }])
        out = orch.preview_images_from_dir("seg", d)
        assert len(out) == 1
        m = out[0]
        assert m["orientation"] == "sagittal"
        assert m["slice_index"] == 7
        assert m["source_width"] == 640 and m["source_height"] == 480
        assert m["image_width"] == 320 and m["image_height"] == 240

    def test_missing_metadata_fields_are_none(self, tmp_path):
        d = tmp_path / "prev"
        _write_png(d / "x.png")  # no preview_manifest.json
        out = orch.preview_images_from_dir("g", d)
        assert out[0]["orientation"] is None
        assert out[0]["slice_index"] is None


class TestPreviewListingFromDir:
    def test_missing_dir_returns_empty(self, tmp_path):
        assert orch.preview_listing_from_dir("g", tmp_path / "absent", "/static") == []

    def test_src_uses_base_and_cache_bust_suffix(self, tmp_path):
        d = tmp_path / "prev"
        png = _write_png(d / "frame_001.png")
        out = orch.preview_listing_from_dir("dense", d, "/cases/c/previews/x")
        assert len(out) == 1
        item = out[0]
        # lazy listing: no inline base64
        assert item["data_url"] == ""
        # src = base + "/" + filename + "?v=<mtime>"
        assert item["src"].startswith("/cases/c/previews/x/frame_001.png?v=")
        ver = item["src"].rsplit("?v=", 1)[1]
        assert ver == str(int(png.stat().st_mtime))
        assert item["file_name"] == "frame_001.png"
        assert item["group"] == "dense"

    def test_cache_bust_changes_when_file_rewritten(self, tmp_path):
        d = tmp_path / "prev"
        png = _write_png(d / "f.png")
        # force an explicit, different mtime and confirm it flows into ?v=
        import os
        os.utime(png, (1000, 1000))
        out = orch.preview_listing_from_dir("g", d, "/base")
        assert out[0]["src"] == "/base/f.png?v=1000"

    def test_sorted_and_metadata_merged(self, tmp_path):
        d = tmp_path / "prev"
        _write_png(d / "z.png")
        _write_png(d / "a.png")
        _write_preview_manifest(d, [{"file_name": "a.png", "slice_index": 3}])
        out = orch.preview_listing_from_dir("g", d, "/b")
        assert [i["file_name"] for i in out] == ["a.png", "z.png"]
        assert out[0]["slice_index"] == 3
        assert out[1]["slice_index"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
