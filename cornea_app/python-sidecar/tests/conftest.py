"""Shared pytest fixtures for the cornea sidecar unit suite.

ISOLATION CONTRACT (never touch real user data):
  * CORNEA_DATA_DIR is forced to a throwaway tempdir at import time, BEFORE any app
    module imports `settings` (settings reads it once at import to compute CASES_ROOT).
  * CORNEA_API_TOKEN is forced empty (dev mode) so TestClient mutating calls pass the
    origin/token guard (TestClient sends no Origin header).
  * The `cases_root` fixture monkeypatches settings.CASES_ROOT to a per-test tmp dir;
    orchestration.case_root() reads settings.CASES_ROOT at call time, so every case
    write lands under the test's own tmp dir.

Fixtures provided:
  cases_root  -> Path to a fresh per-test cases/ dir (settings.CASES_ROOT patched to it)
  make_volume -> factory(shape=(F,D,L), fill=int, affine=None) -> np.ndarray  (small synthetic OCT)
  write_nifti -> factory(arr, path, affine=None) -> Path
  make_case   -> factory(cid, vol=None, lab=None, manifest=None, affine=None) -> case_id
                 (writes previews/volume.nii.gz + the corrected labelmap + manifest)
  client      -> fastapi.testclient.TestClient bound to api_server.app (isolated cases_root)
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ── isolate BEFORE importing any app module ─────────────────────────────────
_SESSION_DATA_DIR = tempfile.mkdtemp(prefix="cornea_pytest_")
os.environ["CORNEA_DATA_DIR"] = _SESSION_DATA_DIR
os.environ["CORNEA_API_TOKEN"] = ""

_SIDE = Path(__file__).resolve().parent.parent          # python-sidecar/
if str(_SIDE) not in sys.path:
    sys.path.insert(0, str(_SIDE))

import settings           # noqa: E402  (import after env is set)
import orchestration as orch  # noqa: E402
import labels             # noqa: E402

# A simple, well-conditioned anisotropic affine resembling OCT voxel geometry.
_DEFAULT_AFFINE = np.diag([0.02, 0.02, 0.04, 1.0]).astype(float)


@pytest.fixture
def cases_root(tmp_path, monkeypatch):
    """A fresh cases/ root for the test; settings.CASES_ROOT points at it."""
    root = tmp_path / "cases"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "CASES_ROOT", root, raising=False)
    return root


@pytest.fixture
def make_volume():
    def _make(shape=(8, 24, 20), fill=20, cornea_band=None, scar_box=None, dtype=np.uint16):
        """Small synthetic OCT-like volume (frames, depth, lateral).
        cornea_band=(d0,d1) paints a bright stromal band; scar_box=(f0,f1,d0,d1,l0,l1) a brighter blob."""
        vol = np.full(shape, fill, dtype)
        if cornea_band is not None:
            d0, d1 = cornea_band
            vol[:, d0:d1, :] = 200
        if scar_box is not None:
            f0, f1, d0, d1, l0, l1 = scar_box
            vol[f0:f1, d0:d1, l0:l1] = 360
        return vol
    return _make


@pytest.fixture
def write_nifti():
    import nibabel as nib

    def _write(arr, path, affine=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(np.asarray(arr), _DEFAULT_AFFINE if affine is None else affine), str(path))
        return path
    return _write


@pytest.fixture
def make_case(cases_root, write_nifti):
    """Build a minimal segmented case on disk: previews/volume.nii.gz, the corrected
    labelmap (0/1/2), and a manifest. Returns the case_id."""
    def _make(cid="case_zz_test", vol=None, lab=None, manifest=None, affine=None):
        orch.ensure_case_dirs(cid)
        if vol is None:
            vol = np.full((8, 24, 20), 20, np.uint16)
            vol[:, 10:14, :] = 200
            vol[2:6, 11:13, 6:12] = 360
        pv = orch.case_root(cid) / "previews" / "volume.nii.gz"
        write_nifti(vol, pv, affine)
        if lab is None:
            lab = np.zeros(np.asarray(vol).shape, np.uint8)
            lab[:, 10:14, :] = 1
            lab[2:6, 11:13, 6:12] = 2
        labels.write_label_nifti(np.asarray(lab).astype(np.uint8), pv, labels.corrected_path(cid))
        base = {"input_volume": str(pv), "corrected_volume": str(pv), "oct_preprocessed": True}
        if manifest:
            base.update(manifest)
        orch.write_manifest_value(cid, base)
        return cid
    return _make


@pytest.fixture
def client(cases_root):
    """TestClient bound to the FastAPI app, with an isolated cases_root already patched.
    Importing api_server is deferred to here so the env isolation above is in effect."""
    from fastapi.testclient import TestClient
    import api_server
    with TestClient(api_server.app) as c:
        yield c
