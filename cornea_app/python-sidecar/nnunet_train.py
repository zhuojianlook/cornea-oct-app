"""nnU-Net v2 training proof-of-concept.

Builds an nnU-Net raw dataset from the PER-SCAN segmentations — each scan's OWN corrected
labelmap (NOT the consensus) — across ALL subgroups, then runs the standard nnU-Net workflow
(plan_and_preprocess → train) inside an ISOLATED venv (cornea_app/.venv-nnunet) that reuses the
system torch/CUDA but keeps nnU-Net's own dependencies away from the sidecar (so SAM2 is untouched).

Two modeling modes (selectable in the UI):
  single3 — ONE dataset, labels {background:0, cornea:1, scar:2}. If NO scan has any scar the
            scar class is dropped automatically (a 2-class background/cornea model).
  cascade — TWO models:
            (A) cornea: labels {background:0, cornea:1} (scar merged into cornea, since scar is a
                sub-region of cornea) — "first cornea segmentation".
            (B) scar : 2 input channels (OCT + the cornea mask) with labels {background:0, scar:1}
                — "then scar segmentation inside cornea only" (the cornea-prior channel constrains
                it). Skipped automatically if no scan has scar.

Training labels are read through labels.best_labelmap_nnunet so they match the viewer/export.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import nibabel as nib

import settings
import orchestration as orch
import labels as label_mod

CORNEA_APP_DIR = Path(__file__).resolve().parents[1]
VENV_DIR = CORNEA_APP_DIR / ".venv-nnunet"
VENV_PY = VENV_DIR / "bin" / "python"

# nnU-Net's three env-var trees (kept separate from the manual export under output/nnunet/).
NN_BASE = settings.WORKSPACE_ROOT / "output" / "nnunet"
NN_RAW = NN_BASE / "nnUNet_raw"
NN_PRE = NN_BASE / "nnUNet_preprocessed"
NN_RES = NN_BASE / "nnUNet_results"
LOG_PATH = NN_BASE / "train.log"

# Dataset ids/names per mode (stable so re-runs overwrite, not pile up).
DS_SINGLE3 = (701, "Dataset701_CorneaOCT3cls")
DS_CORNEA = (711, "Dataset711_CorneaOCTcornea")
DS_SCAR = (712, "Dataset712_CorneaOCTscar")

# Short-epoch trainer variants nnU-Net ships (resolved against what's installed at runtime).
_SHORT_TRAINERS = ("nnUNetTrainer_10epochs", "nnUNetTrainer_5epochs", "nnUNetTrainer_1epoch")

# ── live job state (one training job at a time) ─────────────────────────────
_STATE: dict = {
    "running": False, "done": False, "error": None, "mode": None, "config": None,
    "length": None, "stage": None, "steps": [], "datasets": [], "started_at": None,
    "finished_at": None, "n_cases": 0, "scar_present": None,
}
_LOCK = threading.Lock()
_VENV_LOCK = threading.Lock()
_VENV_READY = False   # cache: once the venv imports nnunetv2 it stays ready (avoids a subprocess per status poll)


def venv_ready() -> bool:
    """Authoritative readiness — the venv exists AND can import nnunetv2. Cached once true."""
    global _VENV_READY
    if _VENV_READY:
        return True
    if not VENV_PY.exists():
        return False
    try:
        r = subprocess.run([str(VENV_PY), "-c", "import nnunetv2"], capture_output=True, timeout=60)
        if r.returncode == 0:
            _VENV_READY = True
        return _VENV_READY
    except Exception:  # noqa: BLE001
        return False


def ensure_venv(log: Path | None = None) -> None:
    """Create .venv-nnunet if absent and install nnU-Net into it. Reuses the system torch/CUDA by
    pointing a .pth at the user site-packages (where torch lives) and installing nnU-Net's own deps
    INTO the venv via the base pip's --python (so the sidecar's env / SAM2 stay untouched). Idempotent
    and serialized (a second concurrent setup waits, then no-ops once the first finishes)."""
    import site
    import sys as _sys
    with _VENV_LOCK:
        if venv_ready():
            return
        _ensure_venv_locked(log, site, _sys)


def _ensure_venv_locked(log, site, _sys) -> None:
    log = log or (NN_BASE / "venv_setup.log")
    log.parent.mkdir(parents=True, exist_ok=True)

    def sh(cmd):
        with open(log, "a") as fh:
            fh.write(f"\n$ {' '.join(str(c) for c in cmd)}\n"); fh.flush()
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line); fh.flush()
            if p.wait() != 0:
                raise RuntimeError(f"command failed: {' '.join(str(c) for c in cmd)} (see {log})")

    if not VENV_PY.exists():
        sh([_sys.executable, "-m", "venv", "--without-pip", str(VENV_DIR)])
        # let the venv see the shared user site (torch/numpy/SimpleITK/cv2/nibabel live there)
        pth = VENV_DIR / "lib" / f"python{_sys.version_info.major}.{_sys.version_info.minor}" / "site-packages" / "_reuse_usersite.pth"
        pth.write_text(site.getusersitepackages() + "\n")
    sh([_sys.executable, "-m", "pip", "--python", str(VENV_PY), "install", "nnunetv2", "numpy<2"])
    if not venv_ready():
        raise RuntimeError(f"nnU-Net venv install did not complete — see {log}")


def per_scan_segmented_cases() -> list[str]:
    """Every PER-SCAN OCT case (across all subgroups) that has a corrected labelmap — the
    consensus cases themselves are excluded (we train on the individual scans, not the vote)."""
    if not settings.CASES_ROOT.exists():
        return []
    out = []
    for d in sorted(settings.CASES_ROOT.iterdir()):
        if not d.is_dir():
            continue
        cid = d.name
        m = orch.read_manifest(cid)
        if m.get("consensus_cases") or cid.endswith("_consensus"):
            continue                       # this IS a consensus case → skip
        if not m.get("oct_source"):
            continue                       # per-scan OCT scans only
        if label_mod.corrected_path(cid).exists():
            out.append(cid)
    return out


def _eye_group_key(src: str | None, cid: str) -> str:
    """Stable per-eye grouping key from the SOURCE filename, used ONLY when the patient id is
    unresolved. Replicates of one physical eye differ only by a '(N)' suffix (and a trailing
    channel/index like '_0'); stripping those + the file extension collapses them to one key, so
    sibling repeats share a group instead of each becoming a leak-prone singleton 'patient'. Falls
    back to the case id when there is no source filename at all."""
    if not src:
        return cid.upper()
    stem = Path(src).name
    for _ in range(2):                          # peel double extensions (.nii.gz) too
        new = Path(stem).stem
        if new == stem:
            break
        stem = new
    stem = re.sub(r"\s*\(\d+\)", "", stem)       # drop the '(N)' replicate marker
    stem = re.sub(r"_\d+$", "", stem)            # drop a trailing channel/index suffix (e.g. '_0')
    stem = re.sub(r"\s+", " ", stem).strip()
    return ("STEM:" + stem).upper() if stem else cid.upper()


def case_meta(cid: str) -> dict:
    """Patient / eye / subgroup for a scan (manifest first, then filename parse) — used for
    patient-grouped splits (no repeat-of-the-same-eye leakage) and subgroup analyses.

    `group` is the deterministic key the splits group on: the resolved patient id when available,
    else a per-eye key from the source filename stem (replicate '(N)' stripped) so replicates of
    one eye stay together. `patient_resolved` is False when we had to fall back to the filename."""
    m = orch.read_manifest(cid)
    patient = m.get("patient_id")
    eye = m.get("eye")
    if not patient or not eye:
        try:
            import metrics_export
            meta = metrics_export.parse_case_meta(m.get("oct_source") or m.get("input_volume") or cid)
            patient = patient or meta.get("patient_id") or meta.get("patient")
            eye = eye or meta.get("eye") or meta.get("laterality")
        except Exception:  # noqa: BLE001
            pass
    patient_resolved = bool(patient)
    group = patient.upper() if patient_resolved else _eye_group_key(
        m.get("oct_source") or m.get("input_volume"), cid)
    return {"case": cid, "patient": (patient or cid).upper(), "eye": (eye or "?"),
            "group": group, "patient_resolved": patient_resolved,
            "subgroup": str(m.get("scar_subgroup") or "1")}


def split_patient_grouped(cases: list[str], test_frac: float = 0.2):
    """Hold out WHOLE patients for the test set (so repeat scans of one eye never straddle the
    train/test boundary — that would leak). Deterministic. Returns (trainval, test, meta_by_case)."""
    meta = {c: case_meta(c) for c in cases}
    by_pat: dict[str, list[str]] = {}
    for c in cases:
        by_pat.setdefault(meta[c]["group"], []).append(c)
    patients = sorted(by_pat)
    n_test_pat = 0 if len(patients) < 3 else max(1, round(len(patients) * test_frac))
    test_patients = set(patients[:n_test_pat])         # deterministic slice
    test = [c for p in test_patients for c in by_pat[p]]
    trainval = [c for c in cases if c not in set(test)]
    return sorted(trainval), sorted(test), meta


def _resolve_short_trainer() -> str:
    """Pick the shortest-epoch trainer variant that's actually installed."""
    try:
        r = subprocess.run(
            [str(VENV_PY), "-c",
             "import nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_Xepochs as m;"
             "print('\\n'.join(n for n in dir(m) if n.startswith('nnUNetTrainer_')))"],
            capture_output=True, text=True, timeout=60)
        avail = set(r.stdout.split())
    except Exception:  # noqa: BLE001
        avail = set()
    for t in _SHORT_TRAINERS:
        if t in avail:
            return t
    return _SHORT_TRAINERS[0]   # best-effort; nnU-Net errors clearly if truly absent


def _reset_dataset(name: str) -> Path:
    """Fresh raw dataset dir (imagesTr/labelsTr) — wipe a prior build so renamed/removed scans
    don't linger as orphan training pairs."""
    d = NN_RAW / name
    shutil.rmtree(d, ignore_errors=True)
    (d / "imagesTr").mkdir(parents=True, exist_ok=True)
    (d / "labelsTr").mkdir(parents=True, exist_ok=True)
    return d


def _write_uint8(arr_ijk: np.ndarray, base_nifti: Path, dst: Path) -> None:
    label_mod.write_label_nifti(arr_ijk, base_nifti, dst)


def _drop_case_files(d: Path, cid: str) -> None:
    """Remove any partial image/label files for a case (so a failed write leaves no orphan
    image without its label, which would later fail nnU-Net's dataset-integrity check)."""
    for p in (d / "imagesTr").glob(f"{cid}_*.nii.gz"):
        p.unlink(missing_ok=True)
    (d / "labelsTr" / f"{cid}.nii.gz").unlink(missing_ok=True)


def _dataset_json(d: Path, channels: dict, labels: dict, n: int, desc: str) -> None:
    (d / "dataset.json").write_text(json.dumps({
        "channel_names": channels,
        "labels": labels,
        "numTraining": int(n),
        "file_ending": ".nii.gz",
        "description": desc,
    }, indent=2))


def build_single3(cases: list[str], image_resolver) -> dict:
    """One dataset, labels {bg,cornea,scar} (scar dropped if no scan has any)."""
    d = _reset_dataset(DS_SINGLE3[1])
    n, scar_any = 0, False
    skipped = []
    for cid in cases:
        arr, _ = label_mod.best_labelmap_nnunet(cid)
        if arr is None:
            skipped.append({"case": cid, "reason": "no labelmap"})
            continue
        try:
            base = image_resolver(cid)
            _write_uint8(arr.astype(np.uint8), base, d / "labelsTr" / f"{cid}.nii.gz")
            shutil.copyfile(base, d / "imagesTr" / f"{cid}_0000.nii.gz")
        except Exception as exc:  # noqa: BLE001 — skip a bad case, don't abort the whole job
            _drop_case_files(d, cid)
            skipped.append({"case": cid, "reason": f"resolve/write: {exc}"})
            continue
        scar_any = scar_any or bool((arr == 2).any())
        n += 1
    labels = {"background": 0, "cornea": 1, "scar": 2} if scar_any else {"background": 0, "cornea": 1}
    _dataset_json(d, {"0": "OCT"}, labels, n, "Cornea OCT — single 3-class (bg/cornea/scar)")
    return {"id": DS_SINGLE3[0], "name": DS_SINGLE3[1], "n": n, "scar_present": scar_any,
            "labels": labels, "skipped": skipped}


def build_cornea(cases: list[str], image_resolver) -> dict:
    """Cascade stage A — cornea (cornea∪scar) vs background."""
    d = _reset_dataset(DS_CORNEA[1])
    n, skipped = 0, []
    for cid in cases:
        arr, _ = label_mod.best_labelmap_nnunet(cid)
        if arr is None:
            skipped.append({"case": cid, "reason": "no labelmap"})
            continue
        try:
            base = image_resolver(cid)
            _write_uint8((arr > 0).astype(np.uint8), base, d / "labelsTr" / f"{cid}.nii.gz")
            shutil.copyfile(base, d / "imagesTr" / f"{cid}_0000.nii.gz")
        except Exception as exc:  # noqa: BLE001 — skip a bad case, don't abort the whole job
            _drop_case_files(d, cid)
            skipped.append({"case": cid, "reason": f"resolve/write: {exc}"})
            continue
        n += 1
    _dataset_json(d, {"0": "OCT"}, {"background": 0, "cornea": 1}, n,
                  "Cornea OCT cascade A — cornea vs background")
    return {"id": DS_CORNEA[0], "name": DS_CORNEA[1], "n": n, "skipped": skipped}


def build_scar(cases: list[str], image_resolver) -> dict:
    """Cascade stage B — scar vs background WITHIN cornea, given the cornea mask as channel 1.
    Only cases with scar contribute foreground, but all cases train the in-cornea negative."""
    d = _reset_dataset(DS_SCAR[1])
    n, scar_any, skipped = 0, False, []
    for cid in cases:
        arr, _ = label_mod.best_labelmap_nnunet(cid)
        if arr is None:
            skipped.append({"case": cid, "reason": "no labelmap"})
            continue
        try:
            base = image_resolver(cid)
            _write_uint8((arr > 0).astype(np.uint8), base, d / "imagesTr" / f"{cid}_0001.nii.gz")  # ch1 = cornea prior
            _write_uint8((arr == 2).astype(np.uint8), base, d / "labelsTr" / f"{cid}.nii.gz")      # scar → 1
            shutil.copyfile(base, d / "imagesTr" / f"{cid}_0000.nii.gz")          # channel 0 = OCT (last)
        except Exception as exc:  # noqa: BLE001 — skip a bad case, don't abort the whole job
            _drop_case_files(d, cid)
            skipped.append({"case": cid, "reason": f"resolve/write: {exc}"})
            continue
        scar_any = scar_any or bool((arr == 2).any())
        n += 1
    _dataset_json(d, {"0": "OCT", "1": "corneaprior"}, {"background": 0, "scar": 1}, n,
                  "Cornea OCT cascade B — scar within cornea (OCT + cornea-prior channel). "
                  "NOTE: the stage-B cornea-prior channel is the GROUND-TRUTH cornea here (it "
                  "exactly encloses the scar), whereas inference uses the PREDICTED cornea — so "
                  "reported scar metrics from this cascade are OPTIMISTIC (upper-bound) estimates.")
    return {"id": DS_SCAR[0], "name": DS_SCAR[1], "n": n, "scar_present": scar_any, "skipped": skipped}


def _nn_env() -> dict:
    e = dict(os.environ)
    e["nnUNet_raw"] = str(NN_RAW)
    e["nnUNet_preprocessed"] = str(NN_PRE)
    e["nnUNet_results"] = str(NN_RES)
    e["nnUNet_n_proc_DA"] = e.get("nnUNet_n_proc_DA", "4")   # modest DA workers for a PoC box
    return e


def _bin(name: str) -> str:
    return str(VENV_DIR / "bin" / name)


def _run(cmd: list[str], log: "os.PathLike | str") -> int:
    """Run a venv nnU-Net command, streaming combined output to the log."""
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, env=_nn_env(), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:        # stream
            fh.write(line)
            fh.flush()
        return proc.wait()


def _write_splits(name: str) -> dict:
    """Write a PATIENT-GROUPED splits_final.json so repeat scans of one eye never straddle the
    train/val boundary (random 5-fold would leak them) — and so training runs with as few as 1–2
    scans (nnU-Net's default 5-fold needs ≥5). Each fold holds out one patient for validation."""
    ids = sorted(p.name[:-7] for p in (NN_RAW / name / "labelsTr").glob("*.nii.gz"))  # strip ".nii.gz"
    meta = {c: case_meta(c) for c in ids}
    by_pat: dict[str, list[str]] = {}
    for c in ids:
        by_pat.setdefault(meta[c]["group"], []).append(c)
    pats = sorted(by_pat)
    folds = []
    for i in range(min(5, len(pats))):
        va = by_pat[pats[i]]
        tr = [c for c in ids if c not in set(va)]
        if tr and va:
            folds.append({"train": tr, "val": va})
    if not folds:                              # single patient / 1 case → degenerate train==val (PoC only)
        folds = [{"train": ids, "val": ids}]
    pre = NN_PRE / name
    pre.mkdir(parents=True, exist_ok=True)
    (pre / "splits_final.json").write_text(json.dumps(folds, indent=2))
    return {"n": len(ids), "n_patients": len(pats), "patient_grouped": len(pats) >= 2}


def _plan_and_train(did: int, name: str, config: str, trainer: str, log: Path) -> None:
    rc = _run([_bin("nnUNetv2_plan_and_preprocess"), "-d", str(did), "-c", config,
               "--verify_dataset_integrity"], log)
    if rc != 0:
        raise RuntimeError(f"plan_and_preprocess failed (dataset {did}, exit {rc}) — see {log}")
    sp = _write_splits(name)   # patient-grouped split (also makes small-N feasible; nnU-Net needs ≥5 for 5-fold)
    with open(log, "a") as fh:
        fh.write(f"[splits] {name}: {sp['n']} case(s), {sp['n_patients']} patient(s)"
                 f"{'' if sp['patient_grouped'] else ' (single-patient degenerate split)'}\n")
    rc = _run([_bin("nnUNetv2_train"), str(did), config, "0", "-tr", trainer], log)
    if rc != 0:
        raise RuntimeError(f"train failed (dataset {did}, exit {rc}) — see {log}")


def _set(**kw) -> None:
    with _LOCK:
        _STATE.update(kw)


def _push_step(text: str) -> None:
    with _LOCK:
        _STATE["steps"] = _STATE["steps"] + [text]
        _STATE["stage"] = text


def start_training(mode: str, config: str, length: str, image_resolver,
                   subset: list[str] | None = None) -> dict:
    """Kick off a background training job. Returns immediately with the resolved plan; progress
    is polled via status(). One job at a time.

    `subset` (optional) restricts training to the chosen candidate cases (the user's first-run
    training subset). It is intersected with the discovered per-scan segmentations, so unknown /
    stale ids are dropped silently. When None/empty, ALL per-scan segmentations are used (the
    prior behaviour — backward compatible)."""
    # Claim the slot ATOMICALLY (check + set under one lock) so two near-simultaneous requests
    # can't both pass and corrupt the shared dataset/state. Released on any validation/launch failure.
    with _LOCK:
        if _STATE["running"]:
            raise RuntimeError("A training job is already running.")
        _STATE["running"] = True
    try:
        if not venv_ready():
            raise RuntimeError("nnU-Net venv is not ready (.venv-nnunet). Install it first.")
        cases = per_scan_segmented_cases()
        if not cases:
            raise RuntimeError("No per-scan segmentations found. Segment some scans (SAM2) first.")
        if subset:                              # honor the user's chosen first-run training subset
            chosen = set(subset)
            cases = [c for c in cases if c in chosen]   # preserve discovery order; drop stale ids
            if not cases:
                raise RuntimeError("None of the selected cases have a per-scan segmentation. "
                                   "Pick at least one segmented scan to include in training.")
        else:                                   # no explicit subset → respect "Schedule for training"
            cases = orch.filter_scheduled(cases)
        config = config if config in ("2d", "3d_fullres") else "2d"
        trainer = _resolve_short_trainer() if length == "short" else "nnUNetTrainer"

        for d in (NN_RAW, NN_PRE, NN_RES):
            d.mkdir(parents=True, exist_ok=True)
        LOG_PATH.write_text(f"nnU-Net PoC — mode={mode} config={config} length={length} "
                            f"trainer={trainer} cases={len(cases)}\n")

        with _LOCK:
            _STATE.update({"running": True, "done": False, "error": None, "mode": mode,
                           "config": config, "length": length, "trainer": trainer,
                           "stage": "starting", "steps": [], "datasets": [],
                           "started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "finished_at": None,
                           "n_cases": len(cases), "scar_present": None})

        threading.Thread(target=_worker, args=(mode, config, trainer, cases, image_resolver),
                         daemon=True).start()
    except Exception:                       # validation OR thread launch failed → release the slot
        _set(running=False)
        raise
    return {"started": True, "mode": mode, "config": config, "trainer": trainer,
            "n_cases": len(cases), "cases": cases}


def flow_counts() -> dict:
    """Case-selection tally for the study-flow figure: total scan folders → exclusions → included."""
    c = {"total": 0, "excluded_consensus": 0, "excluded_non_oct": 0, "excluded_no_label": 0, "included": 0}
    if not settings.CASES_ROOT.exists():
        return c
    for d in sorted(settings.CASES_ROOT.iterdir()):
        if not d.is_dir():
            continue
        cid = d.name
        c["total"] += 1
        m = orch.read_manifest(cid)
        if m.get("consensus_cases") or cid.endswith("_consensus"):
            c["excluded_consensus"] += 1
        elif not m.get("oct_source"):
            c["excluded_non_oct"] += 1
        elif not label_mod.corrected_path(cid).exists():
            c["excluded_no_label"] += 1
        else:
            c["included"] += 1
    return c


def list_runs() -> list[dict]:
    """Every completed/attempted First-Run Folder (first_run_v*), newest first, with the summary the UI
    shows (version, timestamp, mode, config, counts) read from each run_spec.json."""
    out: list[dict] = []
    if not NN_BASE.exists():
        return out
    for d in sorted(NN_BASE.glob("first_run_v*"), reverse=True):
        if not d.is_dir():
            continue
        spec = {}
        try:
            spec = json.loads((d / "run_spec.json").read_text())
        except Exception:  # noqa: BLE001
            pass
        # Has the report (the publication artifact) been produced?
        has_report = any(d.glob("*.pdf")) or (d / "report.json").exists() or any(d.glob("report*.html"))
        out.append({
            "name": d.name,
            "version": spec.get("version"),
            "timestamp": spec.get("timestamp"),
            "mode": spec.get("mode"),
            "config": spec.get("config"),
            "counts": spec.get("counts") or {},
            "has_report": bool(has_report),
        })
    return out


def delete_run(name: str) -> bool:
    """Delete ONE First-Run Folder by name. Guarded: the name must be a plain first_run_v* folder that
    lives directly under NN_BASE (no path traversal, never the raw/preprocessed/results trees)."""
    if not name or "/" in name or "\\" in name or ".." in name or not name.startswith("first_run_v"):
        raise ValueError("invalid run name")
    d = (NN_BASE / name).resolve()
    if d.parent != NN_BASE.resolve() or not d.is_dir():
        raise ValueError("run not found")
    shutil.rmtree(d, ignore_errors=True)
    return not d.exists()


def _next_run_version() -> int:
    """Next run version number — scans existing first_run_v* folders (survives restarts), so every
    Start Training lands in its OWN versioned folder and never overwrites a prior run."""
    import re
    mx = 0
    if NN_BASE.exists():
        for d in NN_BASE.glob("first_run_v*"):
            m = re.match(r"first_run_v(\d+)", d.name)
            if m:
                mx = max(mx, int(m.group(1)))
    return mx + 1


def _first_run_folder(mode, config, trainer, models, datasets, trainval, test, meta,
                      image_resolver) -> Path:
    """Write the run spec, then run nnunet_report.py IN THE VENV (which has matplotlib/seaborn/pandas)
    to predict the held-out test set + assemble the publication First-Run Folder. Returns its path."""
    version = _next_run_version()
    run_dir = NN_BASE / f"first_run_v{version:03d}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _set(run_version=version)

    def _case_entry(cid):
        try:
            img = image_resolver(cid)
        except Exception:  # noqa: BLE001
            img = None
        return {**meta.get(cid, {"case": cid}),
                "image": str(img) if img else None,
                "label": str(label_mod.corrected_path(cid))}

    rep_oct_params = orch.read_manifest(trainval[0]).get("oct_params") if trainval else None
    test_entries = [_case_entry(c) for c in test]
    resolved_test = [e for e in test_entries if e["image"]]
    dropped = [e["case"] for e in test_entries if not e["image"]]
    if dropped:
        with open(LOG_PATH, "a") as fh:
            fh.write(f"[report] {len(dropped)} test case(s) dropped (no resolvable image): {dropped}\n")
    spec = {
        "run_dir": str(run_dir), "version": version, "mode": mode, "config": config, "trainer": trainer,
        "nn_raw": str(NN_RAW), "nn_pre": str(NN_PRE), "nn_res": str(NN_RES),
        "models": models,
        "datasets": [{k: d.get(k) for k in ("id", "name", "n", "scar_present", "labels", "skipped")} for d in datasets],
        "test_cases": resolved_test,
        "trainval_cases": [meta.get(c, {"case": c}) for c in trainval],
        # counts.test reflects the cases the report can actually score (resolved images), not the raw split.
        # included_used = the cases that actually fed the split (= len(trainval)+len(test) = len(cases)
        # = trainval + counts.test + test_dropped_no_image); flow_counts()['included'] stays as the
        # separate ELIGIBLE-cohort number. The CONSORT figure's split-feeding box must use included_used.
        "counts": {**flow_counts(), "trainval": len(trainval), "test": len(resolved_test),
                   "test_dropped_no_image": len(dropped),
                   "included_used": len(trainval) + len(test)},
        "oct_params": rep_oct_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    n_unresolved = sum(1 for c in (trainval + test) if not meta.get(c, {}).get("patient_resolved", True))
    if n_unresolved:
        spec["unresolved_patient_warning"] = (
            f"{n_unresolved} scan(s) had an unresolved patient id and were grouped by a per-eye "
            f"filename key (replicate '(N)' stripped) to avoid same-eye train/test leakage. Verify "
            f"the grouping is correct for these scans.")
    (run_dir / "run_spec.json").write_text(json.dumps(spec, indent=2, default=str))

    report_py = Path(__file__).resolve().parent / "nnunet_report.py"
    rc = _run([str(VENV_PY), str(report_py), "--spec", str(run_dir / "run_spec.json")], run_dir / "report.log")
    if rc != 0:
        with open(LOG_PATH, "a") as fh:
            fh.write(f"[report] generation returned {rc} — see {run_dir/'report.log'} (partial folder kept)\n")
    return run_dir


def _worker(mode: str, config: str, trainer: str, cases: list[str], image_resolver) -> None:
    try:
        trainval, test, meta = split_patient_grouped(cases)
        # A scan whose patient id couldn't be resolved is grouped by a per-eye filename key (so
        # sibling repeats of one eye stay together) rather than its own id. Surface that count.
        unident = sum(1 for c in cases if not meta[c].get("patient_resolved", True))
        _push_step(f"Patient-grouped split: {len(trainval)} train/val · {len(test)} test"
                   + (f" · ⚠ {unident} scan(s) with unresolved patient id (check grouping)" if unident else ""))
        _set(n_trainval=len(trainval), n_test=len(test), n_unidentified_patient=unident)
        if not trainval:
            raise RuntimeError("No train/val cases after the split.")
        datasets, models = [], {}
        if mode == "cascade":
            _push_step("Building cornea dataset (cascade A)")
            a = build_cornea(trainval, image_resolver)
            datasets.append(a); _set(datasets=datasets)
            _push_step(f"Train cornea model — dataset {a['id']} ({a['n']} cases)")
            _plan_and_train(a["id"], a["name"], config, trainer, LOG_PATH)
            models["cornea"] = {"id": a["id"], "name": a["name"]}

            _push_step("Building scar dataset (cascade B — scar within cornea)")
            b = build_scar(trainval, image_resolver)
            datasets.append(b); _set(datasets=datasets, scar_present=b.get("scar_present"))
            if b["n"] >= 1 and b.get("scar_present"):
                _push_step(f"Train scar model — dataset {b['id']} ({b['n']} cases)")
                _plan_and_train(b["id"], b["name"], config, trainer, LOG_PATH)
                models["scar"] = {"id": b["id"], "name": b["name"]}
            else:
                _push_step("No scar in any scan — skipping the scar model (cornea-only cascade)")
        else:  # single3
            _push_step("Building single 3-class dataset")
            s = build_single3(trainval, image_resolver)
            datasets.append(s); _set(datasets=datasets, scar_present=s.get("scar_present"))
            _push_step(f"Train 3-class model — dataset {s['id']} ({s['n']} cases)")
            _plan_and_train(s["id"], s["name"], config, trainer, LOG_PATH)
            models["single3"] = {"id": s["id"], "name": s["name"], "labels": s.get("labels")}

        _push_step("Assembling First-Run Folder (predict test · metrics · figures · tables)")
        run_dir = _first_run_folder(mode, config, trainer, models, datasets, trainval, test, meta, image_resolver)
        _set(first_run_dir=str(run_dir))
        _push_step("Done")
        _set(running=False, done=True, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as exc:  # noqa: BLE001
        with open(LOG_PATH, "a") as fh:
            fh.write(f"\n[ERROR] {exc}\n")
        _set(running=False, done=False, error=str(exc),
             finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))


def status(log_tail_lines: int = 40) -> dict:
    with _LOCK:
        st = dict(_STATE)
    tail = ""
    if LOG_PATH.exists():
        try:
            lines = LOG_PATH.read_text(errors="replace").splitlines()
            tail = "\n".join(lines[-log_tail_lines:])
        except Exception:  # noqa: BLE001
            tail = ""
    st["log_tail"] = tail
    st["venv_ready"] = venv_ready()   # authoritative (venv imports nnunetv2), cached once true
    return st
