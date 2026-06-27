#!/usr/bin/env python
"""Assemble the publication "First-Run Folder" for an nnU-Net training run.

Runs INSIDE the nnU-Net venv (has nnU-Net, matplotlib, seaborn, pandas, scipy). Invoked by
nnunet_train._first_run_folder with --spec <run_spec.json>. It:
  1. predicts the HELD-OUT test set (single 3-class, or the two-stage cascade using the PREDICTED
     cornea as the stage-B prior — the honest inference path),
  2. computes per-case + aggregate metrics (Dice, HD95) and quantification (scar volume / en-face
     area / density) vs the expert label,
  3. writes every requested artifact, publication figure, table, and reporting checklist.

Items that need data we don't have here (clinical severity/VA, scanner metadata, multi-rater) are
emitted as clearly-labelled SCAFFOLDS (template CSVs + placeholder figures) rather than fabricated.
A generation_report.json records what was produced vs skipped and why.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

GEN = {"produced": [], "skipped": []}


def ok(item):
    GEN["produced"].append(item)


def skip(item, why):
    GEN["skipped"].append({"item": item, "why": why})


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ── IO + metric helpers ─────────────────────────────────────────────────────
def load(path):
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj)
    return arr, tuple(float(z) for z in img.header.get_zooms()[:3])


def dice(a, b):
    a, b = a.astype(bool), b.astype(bool)
    s = a.sum() + b.sum()
    if s == 0:
        return float("nan")          # both empty → undefined (e.g. no scar in GT or pred)
    return float(2.0 * (a & b).sum() / s)


def hd95(a, b, spacing):
    """Symmetric 95th-percentile Hausdorff (mm)."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    a, b = a.astype(bool), b.astype(bool)
    if not a.any() or not b.any():
        return float("nan")
    a_surf = a & ~binary_erosion(a)
    b_surf = b & ~binary_erosion(b)
    dt_b = distance_transform_edt(~b_surf, sampling=spacing)
    dt_a = distance_transform_edt(~a_surf, sampling=spacing)
    d = np.concatenate([dt_b[a_surf], dt_a[b_surf]])
    return float(np.percentile(d, 95)) if d.size else float("nan")


def _depth_axis(cornea_mask):
    """The A-scan/depth axis = the one whose face-on projection fills the cornea
    into the densest disc (the cornea is a thin curved shell — collapsing its thin
    direction yields the largest footprint). Returns 0, 1, or 2.

    Copied VERBATIM from scar._depth_axis so the report's en-face area uses the SAME
    depth-axis rule as the canonical quantifier (scar.quantify -> scar_summary.csv).
    A plain argmin(spacing) disagrees on anisotropic OCT geometry where depth is not
    the smallest-spacing axis. nnunet_report runs in a separate venv, so this is a
    copy rather than an import (no sys.path / cross-venv fragility)."""
    shape = cornea_mask.shape
    best_axis, best_score = 0, -1.0
    for a in range(3):
        footprint = int(cornea_mask.any(axis=a).sum())
        plane_area = shape[(a + 1) % 3] * shape[(a + 2) % 3]
        score = footprint / max(plane_area, 1)      # how fully the en-face disc fills
        if score > best_score:
            best_axis, best_score = a, score
    return best_axis


def quantify(mask, img, spacing, cornea=None):
    """Scar volume (mm³), en-face area (mm², projecting out the morphological depth axis — the same
    rule scar.quantify uses so report numbers match scar_summary.csv), and density (mean OCT
    intensity in the mask). `cornea` (cornea∪scar) defines the depth axis when given; falls back to
    the scar mask itself for empty/legacy calls."""
    mask = mask.astype(bool)
    vol = float(mask.sum() * float(np.prod(spacing)))
    ref = cornea.astype(bool) if cornea is not None else mask
    depth_axis = _depth_axis(ref) if ref.any() else int(np.argmin(spacing))
    enface = mask.max(axis=depth_axis)
    others = [s for i, s in enumerate(spacing) if i != depth_axis]
    area = float(enface.sum() * others[0] * others[1])
    density = float(img[mask].mean()) if mask.any() else 0.0
    return {"volume_mm3": round(vol, 4), "enface_area_mm2": round(area, 4), "density": round(density, 3)}


# ── prediction ──────────────────────────────────────────────────────────────
def _predict(did, config, trainer, in_dir, out_dir, save_prob=True):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(Path(sys.executable).parent / "nnUNetv2_predict"),
           "-i", str(in_dir), "-o", str(out_dir), "-d", str(did), "-c", config,
           "-f", "0", "-tr", trainer, "-p", "nnUNetPlans"]
    if save_prob:
        cmd.append("--save_probabilities")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"nnUNetv2_predict failed (d{did}): {r.stderr[-600:]}")


def _copy_probs(src_dir, prob_dir, prefix=""):
    """Copy nnU-Net probability outputs — BOTH the .npz and its .pkl sidecar (the .pkl carries the
    properties needed to interpret/reload the .npz; without it the saved probabilities are unusable)."""
    for npz in src_dir.glob("*.npz"):
        shutil.copyfile(npz, prob_dir / f"{prefix}{npz.name}")
        pkl = npz.with_suffix(".pkl")
        if pkl.exists():
            shutil.copyfile(pkl, prob_dir / f"{prefix}{pkl.name}")


def predict_test(spec, work, pred_dir, prob_dir):
    """Predict every test case; return {cid: combined_3class_pred_array}. Also stages prob maps."""
    mode, config, trainer = spec["mode"], spec["config"], spec["trainer"]
    models = spec["models"]
    test = spec["test_cases"]
    preds = {}
    if not test:
        skip("predictions", "no held-out test cases (cohort too small for a patient-grouped test split)")
        return preds

    if mode == "single3":
        ins = work / "imagesTs_single3"; ins.mkdir(parents=True, exist_ok=True)
        for tc in test:
            shutil.copyfile(tc["image"], ins / f"{tc['case']}_0000.nii.gz")
        outd = work / "pred_single3"
        _predict(models["single3"]["id"], config, trainer, ins, outd)
        for tc in test:
            p = outd / f"{tc['case']}.nii.gz"
            if p.exists():
                arr, _ = load(p)
                preds[tc["case"]] = np.rint(arr).astype(np.uint8)
                shutil.copyfile(p, pred_dir / f"{tc['case']}.nii.gz")
        _copy_probs(outd, prob_dir)
        return preds

    # cascade: cornea first, then scar within the PREDICTED cornea
    ins_c = work / "imagesTs_cornea"; ins_c.mkdir(parents=True, exist_ok=True)
    for tc in test:
        shutil.copyfile(tc["image"], ins_c / f"{tc['case']}_0000.nii.gz")
    out_c = work / "pred_cornea"
    _predict(models["cornea"]["id"], config, trainer, ins_c, out_c)
    cornea = {}
    for tc in test:
        p = out_c / f"{tc['case']}.nii.gz"
        if p.exists():
            arr, _ = load(p)
            cornea[tc["case"]] = (np.rint(arr) > 0).astype(np.uint8)

    scar = {}
    if "scar" in models:
        ins_s = work / "imagesTs_scar"; ins_s.mkdir(parents=True, exist_ok=True)
        for tc in test:
            if tc["case"] not in cornea:
                continue
            shutil.copyfile(tc["image"], ins_s / f"{tc['case']}_0000.nii.gz")
            base = nib.load(tc["image"])
            nib.save(nib.Nifti1Image(cornea[tc["case"]].astype(np.uint8), base.affine),
                     str(ins_s / f"{tc['case']}_0001.nii.gz"))   # PREDICTED cornea as the prior channel
        out_s = work / "pred_scar"
        _predict(models["scar"]["id"], config, trainer, ins_s, out_s)
        for tc in test:
            p = out_s / f"{tc['case']}.nii.gz"
            if p.exists():
                arr, _ = load(p)
                scar[tc["case"]] = (np.rint(arr) > 0).astype(np.uint8)
        _copy_probs(out_s, prob_dir, "scar_")
    _copy_probs(out_c, prob_dir, "cornea_")

    for tc in test:
        cid = tc["case"]
        if cid not in cornea:
            continue
        comb = cornea[cid].astype(np.uint8).copy()           # 0/1
        if cid in scar:
            comb[(scar[cid] == 1) & (cornea[cid] == 1)] = 2   # scar only inside predicted cornea
        preds[cid] = comb
        base = nib.load(tc["image"])
        nib.save(nib.Nifti1Image(comb, base.affine), str(pred_dir / f"{cid}.nii.gz"))
    return preds


# ── metrics + quantification table ──────────────────────────────────────────
def compute_metrics(spec, preds):
    rows = []
    for tc in spec["test_cases"]:
        cid = tc["case"]
        if cid not in preds:
            continue
        gt, sp = load(tc["label"])
        gt = np.rint(gt).astype(np.uint8)
        img, _ = load(tc["image"])
        pr = preds[cid]
        gt_cor, pr_cor = gt > 0, pr > 0
        gt_sc, pr_sc = gt == 2, pr == 2
        # Depth axis from the CORNEA mask (cornea∪scar) so en-face area matches scar.quantify.
        gq, pq = quantify(gt_sc, img, sp, cornea=gt_cor), quantify(pr_sc, img, sp, cornea=pr_cor)
        rows.append({
            "case": cid, "patient": tc.get("patient"), "eye": tc.get("eye"), "subgroup": tc.get("subgroup"),
            "dice_cornea": dice(pr_cor, gt_cor), "dice_scar": dice(pr_sc, gt_sc),
            "hd95_cornea_mm": hd95(pr_cor, gt_cor, sp), "hd95_scar_mm": hd95(pr_sc, gt_sc, sp),
            "gt_has_scar": bool(gt_sc.any()),
            "scar_vol_mm3_gt": gq["volume_mm3"], "scar_vol_mm3_pred": pq["volume_mm3"],
            "enface_area_mm2_gt": gq["enface_area_mm2"], "enface_area_mm2_pred": pq["enface_area_mm2"],
            "density_gt": gq["density"], "density_pred": pq["density"],
        })
    return rows


# ── figures ─────────────────────────────────────────────────────────────────
def _overlay(ax, img2d, mask2d, color, title):
    ax.imshow(img2d.T, cmap="gray", origin="lower", aspect="auto")
    if mask2d is not None and mask2d.any():
        m = np.ma.masked_where(~mask2d.T.astype(bool), mask2d.T)
        ax.imshow(m, cmap=color, alpha=0.45, origin="lower", aspect="auto")
    ax.set_title(title, fontsize=8); ax.axis("off")


def _mid_scar_slice(gt, axis):
    """Index along `axis` (frames) with the most scar, else the middle."""
    sc = gt == 2
    if sc.any():
        sums = sc.sum(axis=tuple(i for i in range(3) if i != axis))
        return int(np.argmax(sums))
    return gt.shape[axis] // 2


def fig_study_flow(spec, path):
    plt = _plt()
    c = spec["counts"]
    fig, ax = plt.subplots(figsize=(6, 6)); ax.axis("off")
    boxes = [
        (f"All scan folders\nn = {c['total']}", 0.9),
        (f"Excluded\n consensus cases: {c['excluded_consensus']}\n non-OCT: {c['excluded_non_oct']}\n no expert label: {c['excluded_no_label']}", 0.66),
        (f"Included scans\nn = {c.get('included_used', c['included'])}", 0.42),
        (f"Train/val: {c['trainval']}   ·   Test (held-out patients): {c['test']}", 0.18),
    ]
    for txt, y in boxes:
        ax.text(0.5, y, txt, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="#eef", ec="#557"))
        if y > 0.2:
            ax.annotate("", xy=(0.5, y - 0.12), xytext=(0.5, y - 0.04), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Figure 1. Study flow (case selection, exclusions, split)", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_pipeline(path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 2.2)); ax.axis("off"); ax.set_xlim(0, 5); ax.set_ylim(0, 1)
    steps = ["OCT volume\n(preprocessed)", "Cornea\nsegmentation", "Scar\nsegmentation\n(within cornea)", "Quantification\n(vol · area · density)"]
    for i, s in enumerate(steps):
        ax.text(i + 0.5, 0.5, s, ha="center", va="center", fontsize=9,
                bbox=dict(boxstyle="round", fc="#efe", ec="#575"))
        if i < len(steps) - 1:
            ax.annotate("", xy=(i + 1.05, 0.5), xytext=(i + 0.95, 0.5), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Figure 2. Pipeline schematic", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_representative(spec, preds, path, failures=False, df=None):
    plt = _plt()
    test = spec["test_cases"]
    cids = [t["case"] for t in test if t["case"] in preds]
    if not cids:
        return False
    if failures and df is not None and len(df):
        order = df.sort_values("dice_cornea")["case"].tolist()
        cids = [c for c in order if c in preds][:3]
    else:
        cids = cids[:3]
    fig, axes = plt.subplots(len(cids), 3, figsize=(9, 3 * len(cids)), squeeze=False)
    for r, cid in enumerate(cids):
        tc = next(t for t in test if t["case"] == cid)
        gt, sp = load(tc["label"]); gt = np.rint(gt).astype(np.uint8)
        img, _ = load(tc["image"]); pr = preds[cid]
        # Frame (B-scan) axis = the COARSEST-spacing axis (slice/frames ≈0.04mm), so each panel is a
        # true depth×lateral B-scan; depth=argmin(spacing), lateral=the middle one.
        frame_axis = int(np.argmax(sp))
        k = _mid_scar_slice(gt, frame_axis)
        sl = [slice(None)] * 3; sl[frame_axis] = k
        im2 = img[tuple(sl)]; gt2 = gt[tuple(sl)]; pr2 = pr[tuple(sl)]
        _overlay(axes[r][0], im2, None, "gray", f"{cid}\nOCT")
        _overlay(axes[r][1], im2, gt2 == 2, "autumn", "expert scar")
        _overlay(axes[r][2], im2, pr2 == 2, "winter", "predicted scar")
    fig.suptitle("Figure 9. Failure cases (lowest Dice)" if failures else "Figure 3. Representative slices — expert vs predicted", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True


def fig_enface(spec, preds, path):
    plt = _plt()
    test = [t for t in spec["test_cases"] if t["case"] in preds]
    if not test:
        return False
    tc = test[0]
    gt, sp = load(tc["label"]); gt = np.rint(gt).astype(np.uint8); pr = preds[tc["case"]]
    # En-face depth axis via the SAME morphological rule as scar.quantify (not argmin(spacing)), from
    # the cornea (label>0); fall back to argmin only if the case has no cornea.
    cornea_ref = (gt > 0) | (pr > 0)
    da = _depth_axis(cornea_ref) if cornea_ref.any() else int(np.argmin(sp))
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow((gt == 2).max(axis=da).T, cmap="autumn", origin="lower", aspect="auto"); axes[0].set_title("expert scar (en-face)", fontsize=9); axes[0].axis("off")
    axes[1].imshow((pr == 2).max(axis=da).T, cmap="winter", origin="lower", aspect="auto"); axes[1].set_title("predicted scar (en-face)", fontsize=9); axes[1].axis("off")
    fig.suptitle(f"Figure 4. En-face scar projection — {tc['case']}", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return True


def fig_box(df, path):
    plt = _plt()
    import pandas as pd
    long = []
    for _, r in df.iterrows():
        for cls in ("cornea", "scar"):
            long.append({"class": cls, "metric": "Dice", "value": r[f"dice_{cls}"]})
            long.append({"class": cls, "metric": "HD95 (mm)", "value": r[f"hd95_{cls}_mm"]})
    ld = pd.DataFrame(long).dropna()
    if ld.empty:
        return False
    try:
        import seaborn as sns
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, met in zip(axes, ("Dice", "HD95 (mm)")):
            sub = ld[ld["metric"] == met]
            if len(sub):
                sns.boxplot(data=sub, x="class", y="value", ax=ax)
                sns.stripplot(data=sub, x="class", y="value", ax=ax, color="black", size=4)
            ax.set_title(met, fontsize=9); ax.set_xlabel(""); ax.set_ylabel("")
        fig.suptitle("Figure 5. Dice / HD95 by class (test set)", fontsize=10)
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        return True
    except Exception:
        return False


def fig_bland_altman(df, path):
    plt = _plt()
    pairs = [("scar volume (mm³)", "scar_vol_mm3_gt", "scar_vol_mm3_pred"),
             ("en-face area (mm²)", "enface_area_mm2_gt", "enface_area_mm2_pred"),
             ("density", "density_gt", "density_pred")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    any_pts = False
    for ax, (title, gc, pc) in zip(axes, pairs):
        d = df[[gc, pc]].dropna()
        if len(d) >= 1:
            mean = (d[gc] + d[pc]) / 2.0; diff = d[pc] - d[gc]
            ax.scatter(mean, diff, s=20)
            bias = float(diff.mean()); sd = float(diff.std(ddof=1)) if len(diff) > 1 else 0.0
            ax.axhline(bias, color="k"); ax.axhline(bias + 1.96 * sd, color="r", ls="--"); ax.axhline(bias - 1.96 * sd, color="r", ls="--")
            any_pts = True
        ax.set_title(title, fontsize=9); ax.set_xlabel("mean(expert, pred)", fontsize=8); ax.set_ylabel("pred − expert", fontsize=8)
    fig.suptitle("Figure 6. Bland–Altman: quantification agreement (test set)", fontsize=10)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return any_pts


def fig_subgroup(df, path):
    plt = _plt()
    try:
        import seaborn as sns
        fig, axes = plt.subplots(1, 2, figsize=(9, 4))
        for ax, key in zip(axes, ("subgroup", "eye")):
            sub = df.dropna(subset=["dice_cornea"])
            if len(sub):
                sns.boxplot(data=sub, x=key, y="dice_cornea", ax=ax)
                sns.stripplot(data=sub, x=key, y="dice_cornea", ax=ax, color="black", size=4)
            ax.set_title(f"cornea Dice by {key}", fontsize=9); ax.set_xlabel(""); ax.set_ylabel("Dice")
        fig.suptitle("Figure 8. Subgroup performance (scar type / eye)\n(scanner · severity · image-quality need clinical metadata)", fontsize=9)
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        return True
    except Exception:
        return False


def fig_placeholder(path, title, need):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 4)); ax.axis("off")
    ax.text(0.5, 0.5, f"{title}\n\nRequires: {need}\nProvide clinical_metadata.csv to populate.",
            ha="center", va="center", fontsize=10, bbox=dict(boxstyle="round", fc="#fee", ec="#a55"))
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


# ── environment + protocol docs ─────────────────────────────────────────────
def env_versions():
    import importlib.metadata as md
    lines = [f"python {platform.python_version()}", f"platform {platform.platform()}"]
    try:
        import torch
        lines.append(f"torch {torch.__version__} (cuda_available={torch.cuda.is_available()}, "
                     f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")
    except Exception:
        pass
    pkgs = sorted((d.metadata["Name"], d.version) for d in md.distributions() if d.metadata.get("Name"))
    lines += [f"{n}=={v}" for n, v in pkgs]
    return "\n".join(lines)


ANNOTATION_PROTOCOL = """# Annotation protocol

**Target labels (per voxel):** 0 = background, 1 = cornea, 2 = scar (a sub-region of cornea).

**How labels were produced**
1. Each Optovue Avanti `.OCT` is preprocessed (corneal-edge + column + 3D-active correction;
   faithful port of the lab's DICOMSmootherSteps) into a geometry-correct NIfTI.
2. The cornea is segmented with SAM2 (per scan).
3. Scar is delineated within the cornea (hyper-reflective constraint + click-guided SAM2),
   then **expert-corrected** on the slice viewer (paint/erase) — the corrected labelmap
   (`<case>_corrected.nii.gz`) is the training target.

**Important:** training here uses each scan's OWN corrected labelmap (the per-scan expert
segmentation), NOT the multi-scan consensus. Consensus cases are excluded from the dataset.

**Notes for a publication-grade protocol (fill in):** annotator(s) and experience, software
versions, time per case, adjudication/consensus rule for disagreements, and the definition of
scar used by annotators.
"""


def reporting_docs(rep_dir, spec):
    (rep_dir / "CLAIM_checklist.md").write_text(f"""# CLAIM checklist (Checklist for AI in Medical Imaging)

Use CLAIM as the PRIMARY reporting guideline. Fill each item for the manuscript.

## Title / Abstract
- [ ] Identify as a study of AI methodology for medical imaging segmentation.

## Introduction
- [ ] Scientific/clinical background; study objectives.

## Methods
- [ ] Study design (retrospective); eligibility criteria → see figures/fig1_study_flow.png + tables/table2_split_annotation.csv.
- [ ] Data source / scanner (Optovue Avanti); acquisition parameters → voxel_spacing.csv (per-case mm).
- [ ] Ground-truth annotation → annotation_protocol.md (annotators, definitions — FILL IN).
- [ ] Data splits at the PATIENT level (no repeat-scan leakage) → splits.json (train/val/test).
- [ ] Model: nnU-Net v2, config={spec.get('config')}, mode={spec.get('mode')}, trainer={spec.get('trainer')}.
- [ ] Preprocessing → preprocessing_params.json + nnUNet plans (in models/).
- [ ] Metrics: Dice, HD95 (segmentation); volume/area/density agreement (quantification).

## Results
- [ ] Cohort/flow → fig1; demographics → table1 (NEEDS clinical metadata).
- [ ] Performance → table3_model_performance.csv, figures/fig5_dice_hd95.png.
- [ ] Agreement vs expert → table4 + fig6 (Bland–Altman).
- [ ] Subgroup analysis → fig8 (scanner/severity/quality NEED metadata).
- [ ] Failure analysis → fig9.

## Discussion / Reproducibility
- [ ] Limitations (small PoC cohort; cascade uses predicted-cornea prior at inference).
- [ ] Code/model availability → models/ weights, run_spec.json, environment.txt.

---
## When to ADD other guidelines
- **STARD-AI** — add ONLY if you make DIAGNOSTIC ACCURACY claims (e.g. scar present/absent as a test
  vs a clinical reference standard). This PoC reports segmentation/quantification, not diagnosis.
- **TRIPOD+AI** — add if you build/validate a model PREDICTING A CLINICAL OUTCOME (e.g. visual acuity,
  progression) from the imaging biomarkers.
- **DECIDE-AI** — add if you evaluate the tool PROSPECTIVELY in the clinical workflow (early live use).
""")
    ok("reporting/CLAIM_checklist.md")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    spec = json.loads(Path(ap.parse_args().spec).read_text())
    run = Path(spec["run_dir"])

    # folder skeleton
    dirs = {k: run / k for k in ("models", "predictions", "probability_maps", "metrics", "figures", "tables", "reporting")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    work = run / "_work"; work.mkdir(exist_ok=True)

    # Always leave a generation report + (at least a minimal) README, even if generation aborts.
    try:
        _generate(spec, run, dirs, work)
    except Exception as exc:  # noqa: BLE001
        skip(f"fatal:{type(exc).__name__}", str(exc))
    finally:
        if not (run / "README.md").exists():
            (run / "README.md").write_text("# First-Run Folder (incomplete)\n\n"
                                           "Generation failed partway — see generation_report.json.\n")
        (run / "generation_report.json").write_text(json.dumps(GEN, indent=2))
        shutil.rmtree(work, ignore_errors=True)
        print(f"First-run folder: {run} (produced {len(GEN['produced'])}, skipped {len(GEN['skipped'])})")


def _generate(spec, run, dirs, work):
    import pandas as pd

    # 1) trained weights → models/<dataset>/...
    for m in spec["models"].values():
        src = Path(spec["nn_res"]) / m["name"]
        if src.exists():
            shutil.copytree(src, dirs["models"] / m["name"], dirs_exist_ok=True)
            ok(f"models/{m['name']}")
    # preprocessing params: nnUNet plans + the OCT preprocess params
    plans = {}
    for m in spec["models"].values():
        pf = Path(spec["nn_pre"]) / m["name"] / "nnUNetPlans.json"
        if pf.exists():
            plans[m["name"]] = json.loads(pf.read_text())
    (run / "preprocessing_params.json").write_text(json.dumps(
        {"oct_preprocess_params": spec.get("oct_params"), "nnunet_plans": plans}, indent=2, default=str))
    ok("preprocessing_params.json")

    # 2) predict the held-out test set
    preds = {}
    try:
        preds = predict_test(spec, work, dirs["predictions"], dirs["probability_maps"])
        if preds:
            ok(f"predictions ({len(preds)} test case(s)) + probability_maps")
    except Exception as exc:  # noqa: BLE001
        skip("predictions", f"{exc}")

    # 3) metrics + quantification
    rows = compute_metrics(spec, preds) if preds else []
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(dirs["metrics"] / "per_case_metrics.csv", index=False); ok("metrics/per_case_metrics.csv")
        agg = {}
        for col in ("dice_cornea", "dice_scar", "hd95_cornea_mm", "hd95_scar_mm"):
            v = df[col].dropna()
            agg[col] = {"mean": round(float(v.mean()), 4) if len(v) else None,
                        "std": round(float(v.std(ddof=1)), 4) if len(v) > 1 else None, "n": int(len(v))}
        pd.DataFrame(agg).T.to_csv(dirs["metrics"] / "aggregate_metrics.csv"); ok("metrics/aggregate_metrics.csv")
    else:
        skip("metrics", "no test predictions to score")

    # dataset manifest + voxel spacing
    man = []
    for grp, items in (("trainval", spec.get("trainval_cases", [])), ("test", spec.get("test_cases", []))):
        for it in items:
            row = {"case": it.get("case"), "split": grp, "patient": it.get("patient"),
                   "eye": it.get("eye"), "subgroup": it.get("subgroup")}
            lab = it.get("label")
            if lab and Path(lab).exists():
                arr, sp = load(lab); arr = np.rint(arr).astype(np.uint8)
                row.update({"spacing_x_mm": sp[0], "spacing_y_mm": sp[1], "spacing_z_mm": sp[2],
                            "cornea_voxels": int((arr >= 1).sum()), "scar_voxels": int((arr == 2).sum()),
                            "has_scar": bool((arr == 2).any())})
            man.append(row)
    mdf = pd.DataFrame(man)
    mdf.to_csv(run / "dataset_manifest.csv", index=False); ok("dataset_manifest.csv")
    spcols = [c for c in ("case", "split", "spacing_x_mm", "spacing_y_mm", "spacing_z_mm") if c in mdf.columns]
    mdf[spcols].to_csv(run / "voxel_spacing.csv", index=False); ok("voxel_spacing.csv")

    # splits.json (train/val from the nnUNet split file + held-out test)
    splits = {"test": [t.get("case") for t in spec.get("test_cases", [])]}
    if spec.get("unresolved_patient_warning"):
        splits["unresolved_patient_warning"] = spec["unresolved_patient_warning"]
    for m in spec["models"].values():
        sf = Path(spec["nn_pre"]) / m["name"] / "splits_final.json"
        if sf.exists():
            try:
                splits["train_val_fold0"] = json.loads(sf.read_text())[0]
                break
            except Exception:
                pass
    (run / "splits.json").write_text(json.dumps(splits, indent=2)); ok("splits.json")

    # protocol + environment
    (run / "annotation_protocol.md").write_text(ANNOTATION_PROTOCOL); ok("annotation_protocol.md")
    (run / "environment.txt").write_text(env_versions()); ok("environment.txt")

    # 4) figures
    def fig(name, fn, *a):
        try:
            r = fn(*a)
            ok(f"figures/{name}") if r is not False else skip(f"figures/{name}", "no data")
        except Exception as exc:  # noqa: BLE001
            skip(f"figures/{name}", str(exc))

    F = dirs["figures"]
    fig("fig1_study_flow.png", fig_study_flow, spec, F / "fig1_study_flow.png")
    fig("fig2_pipeline.png", fig_pipeline, F / "fig2_pipeline.png")
    if preds:
        fig("fig3_representative.png", fig_representative, spec, preds, F / "fig3_representative.png")
        fig("fig4_enface.png", fig_enface, spec, preds, F / "fig4_enface.png")
    if len(df):
        fig("fig5_dice_hd95.png", fig_box, df, F / "fig5_dice_hd95.png")
        fig("fig6_bland_altman.png", fig_bland_altman, df, F / "fig6_bland_altman.png")
        fig("fig8_subgroup.png", fig_subgroup, df, F / "fig8_subgroup.png")
        fig("fig9_failures.png", fig_representative, spec, preds, F / "fig9_failures.png", True, df)
    fig_placeholder(F / "fig7_severity_scatter.png", "Figure 7. Correlation with clinical severity / visual acuity",
                    "per-case clinical severity or visual-acuity values"); ok("figures/fig7_severity_scatter.png (scaffold)")
    # Figure 10. nnU-Net training curves vs epoch (loss · pseudo-Dice · LR) — surface nnU-Net's
    # progress.png from each trained model into figures/ so it sits with the publication set.
    for role, m in spec["models"].items():
        prog = next((dirs["models"] / m["name"]).rglob("progress.png"), None)
        if prog and prog.exists():
            dst = F / (f"fig10_training_curve_{role}.png" if len(spec["models"]) > 1 else "fig10_training_curve.png")
            shutil.copyfile(prog, dst); ok(f"figures/{dst.name}")
        else:
            skip(f"figures/fig10_training_curve_{role}", "nnU-Net progress.png not found")

    # 5) tables
    T = dirs["tables"]
    # table2 split + annotation summary
    c = spec["counts"]
    pd.DataFrame([{"total_scans": c["total"], "excluded_consensus": c["excluded_consensus"],
                   "excluded_non_oct": c["excluded_non_oct"], "excluded_no_label": c["excluded_no_label"],
                   "included": c.get("included_used", c["included"]), "included_eligible": c["included"],
                   "train_val": c["trainval"], "test": c["test"],
                   "annotation": "per-scan SAM2 + expert correction (0/1/2)"}]).to_csv(T / "table2_split_annotation.csv", index=False)
    ok("tables/table2_split_annotation.csv")
    # table3 model performance + table4 agreement
    if len(df):
        perf = []
        cascade = spec.get("mode") == "cascade"
        for cls in ("cornea", "scar"):
            d_ = df[f"dice_{cls}"].dropna(); h_ = df[f"hd95_{cls}_mm"].dropna()
            perf.append({"class": cls, "n": int(len(d_)),
                         "dice_mean": round(float(d_.mean()), 4) if len(d_) else None,
                         "dice_std": round(float(d_.std(ddof=1)), 4) if len(d_) > 1 else None,
                         "hd95_mm_mean": round(float(h_.mean()), 4) if len(h_) else None,
                         "hd95_mm_std": round(float(h_.std(ddof=1)), 4) if len(h_) > 1 else None,
                         "note": ("OPTIMISTIC/upper-bound: stage-B scar model was TRAINED on the "
                                  "ground-truth cornea prior but INFERRED on the predicted cornea")
                                 if (cascade and cls == "scar") else ""})
        pd.DataFrame(perf).to_csv(T / "table3_model_performance.csv", index=False); ok("tables/table3_model_performance.csv")
        agree = []
        for title, gc, pc in (("scar_volume_mm3", "scar_vol_mm3_gt", "scar_vol_mm3_pred"),
                              ("enface_area_mm2", "enface_area_mm2_gt", "enface_area_mm2_pred"),
                              ("density", "density_gt", "density_pred")):
            dd = df[[gc, pc]].dropna()
            if len(dd):
                diff = dd[pc] - dd[gc]
                r = dd[gc].corr(dd[pc]) if len(dd) > 1 else None
                agree.append({"measure": title, "n": int(len(dd)), "bias_mean_diff": round(float(diff.mean()), 4),
                              "loa_lower": round(float(diff.mean() - 1.96 * diff.std(ddof=1)), 4) if len(diff) > 1 else None,
                              "loa_upper": round(float(diff.mean() + 1.96 * diff.std(ddof=1)), 4) if len(diff) > 1 else None,
                              "pearson_r": round(float(r), 4) if r is not None else None})
        pd.DataFrame(agree).to_csv(T / "table4_quantification_agreement.csv", index=False); ok("tables/table4_quantification_agreement.csv")
        # table6 ablation (this run's row; train the other mode to compare)
        pd.DataFrame([{"model": spec["mode"], "config": spec["config"],
                       "dice_cornea_mean": (round(float(df["dice_cornea"].dropna().mean()), 4) if df["dice_cornea"].notna().any() else None),
                       "dice_scar_mean": (round(float(df["dice_scar"].dropna().mean()), 4) if df["dice_scar"].notna().any() else None),
                       "note": "run the other mode (single3 / cascade) to fill the comparison row"}]).to_csv(
            T / "table6_ablation_single_vs_cascade.csv", index=False); ok("tables/table6_ablation_single_vs_cascade.csv")
    # table1 demographics scaffold + table5 inter-rater scaffold + clinical metadata template
    pats = sorted({r.get("patient") for r in man if r.get("patient")})
    pd.DataFrame([{"patient": p, "eye": "", "age": "", "sex": "", "disease": "", "severity": "", "visual_acuity": "", "scanner": ""} for p in pats]).to_csv(
        T / "table1_demographics_TEMPLATE.csv", index=False); ok("tables/table1_demographics_TEMPLATE.csv (fill clinical fields)")
    pd.DataFrame([{"case": "", "rater": "", "dice_vs_reference": "", "scar_vol_mm3": ""}]).to_csv(
        T / "table5_inter_rater_TEMPLATE.csv", index=False); ok("tables/table5_inter_rater_TEMPLATE.csv (needs multi-rater)")
    pd.DataFrame([{"patient": p, "severity": "", "visual_acuity_logmar": "", "scanner": "", "image_quality": ""} for p in pats]).to_csv(
        run / "clinical_metadata_TEMPLATE.csv", index=False); ok("clinical_metadata_TEMPLATE.csv")

    # 6) reporting docs
    reporting_docs(dirs["reporting"], spec)

    # README + generation report
    (run / "README.md").write_text(f"""# First-Run Folder v{spec.get('version', '?')} — {spec['timestamp']}

nnU-Net **{spec['mode']}** model, config **{spec['config']}**, trainer **{spec['trainer']}**.
Trained on PER-SCAN expert labels (not consensus), patient-grouped split.
{('> ⚠ ' + spec['unresolved_patient_warning'] + chr(10)) if spec.get('unresolved_patient_warning') else ''}

## Contents
- `models/` trained weights + nnU-Net plans (per dataset)
- `predictions/`, `probability_maps/` — held-out test set
- `metrics/` per_case_metrics.csv, aggregate_metrics.csv
- `dataset_manifest.csv`, `voxel_spacing.csv`, `splits.json`
- `preprocessing_params.json`, `annotation_protocol.md`, `environment.txt`
- `figures/` (1 flow · 2 pipeline · 3 representative · 4 en-face · 5 Dice/HD95 · 6 Bland–Altman ·
  7 severity[scaffold] · 8 subgroup · 9 failures · 10 training curve [loss/Dice/LR vs epoch])
- `tables/` (1 demographics[template] · 2 split/annotation · 3 performance · 4 agreement ·
  5 inter-rater[template] · 6 ablation)
- `reporting/CLAIM_checklist.md` — primary guideline; STARD-AI/TRIPOD+AI/DECIDE-AI notes inside.

## To complete for publication
Fill `clinical_metadata_TEMPLATE.csv` (severity, visual acuity, scanner, image quality) and
`tables/table1_demographics_TEMPLATE.csv` / `table5_inter_rater_TEMPLATE.csv`, then regenerate the
severity/subgroup/demographics/inter-rater artifacts. Cascade inference uses the PREDICTED cornea as
the stage-B prior, but stage-B was TRAINED on the GROUND-TRUTH cornea prior (which exactly encloses
the scar) — so the reported SCAR Dice/HD95 (table3_model_performance.csv, aggregate_metrics.csv) are
OPTIMISTIC, upper-bound estimates and must be reported as such (note this train/inference prior
asymmetry as a limitation).

See `generation_report.json` for exactly what was produced vs skipped.
""")
    ok("README.md")


if __name__ == "__main__":
    main()
