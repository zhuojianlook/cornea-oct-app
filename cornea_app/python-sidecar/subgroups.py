"""Automatic subgroup assignment within an eye.

Replicate scans of one eye should image the SAME scar. If some scans mismatch — a different lesion, or
a partial FOV of a different region — they are a different SUBGROUP and must not be averaged together.
We align every scan of an eye to a representative reference, warp the scars into that frame, build a
pairwise FOV-restricted scar-Dice similarity graph, and split into subgroups by connected components
(edge when Dice ≥ threshold). Eyes with <2 scarred scans (e.g. normal controls) stay one subgroup.
"""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

import orchestration as orch
import labels as L
import registration as reg

REF_SCAR, REF_CORNEA = 2, 1


def _vol(cid):
    return orch.case_root(cid) / "previews" / "volume.nii.gz"


def _scar_mm3(cid) -> float:
    img = sitk.ReadImage(str(L.corrected_path(cid)))
    sp = img.GetSpacing()
    return float(int((sitk.GetArrayFromImage(img) == REF_SCAR).sum()) * sp[0] * sp[1] * sp[2])


def _fov_dice(sa, da, sb, db) -> float:
    """Scar Dice on the shared field-of-view (partial-cut-aware): 0 if neither images scar there."""
    common = da & db
    a, b = sa & common, sb & common
    s = int(a.sum()) + int(b.sum())
    return float(2 * int((a & b).sum()) / s) if s else 0.0


def assign_subgroups(eye_cids, dice_threshold: float = 0.35, min_scar_mm3: float = 0.2) -> dict:
    """Return {case_id: subgroup_int} (1-based; subgroup 1 = the largest cluster). The reference is the
    median-scar-volume scan so an outlier can't anchor the frame. Single-scan / low-scar eyes → all 1."""
    cids = [c for c in eye_cids if L.corrected_path(c).exists() and _vol(c).exists()]
    if len(cids) <= 1:
        return {c: 1 for c in cids}
    vols = {c: _scar_mm3(c) for c in cids}
    scarred = [c for c in cids if vols[c] >= min_scar_mm3]
    if len(scarred) <= 1:                      # nothing (or one thing) to split on → one subgroup
        return {c: 1 for c in cids}

    ref = sorted(scarred, key=lambda c: vols[c])[len(scarred) // 2]   # median-volume reference
    ref_vol = _vol(ref)
    scars, datas = {}, {}
    for c in cids:
        if c == ref:
            lab = reg.resample_label(L.corrected_path(c), ref_vol, reg.identity())
            wv = reg.resample_volume(ref_vol, ref_vol, reg.identity())
        else:
            tx, _ = reg.align_transform(ref_vol, L.corrected_path(ref), _vol(c), L.corrected_path(c))
            lab = reg.resample_label(L.corrected_path(c), ref_vol, tx)
            wv = reg.resample_volume(_vol(c), ref_vol, tx)
        scars[c] = lab == REF_SCAR
        datas[c] = sitk.GetArrayFromImage(wv) > 0

    # similarity graph over the SCARRED scans; connected components → clusters
    n = len(scarred)
    adj = {c: set() for c in scarred}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = scarred[i], scarred[j]
            if _fov_dice(scars[a], datas[a], scars[b], datas[b]) >= dice_threshold:
                adj[a].add(b); adj[b].add(a)
    seen, clusters = set(), []
    for c in scarred:
        if c in seen:
            continue
        stack, comp = [c], []
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x)
            stack.extend(adj[x] - seen)
        clusters.append(comp)
    clusters.sort(key=len, reverse=True)       # subgroup 1 = largest
    out = {}
    for k, comp in enumerate(clusters, start=1):
        for c in comp:
            out[c] = k
    # low-scar / unscarred scans of a SPLIT eye → attach to the nearest scarred scan's subgroup by FOV
    for c in cids:
        if c in out:
            continue
        if len(clusters) == 1:
            out[c] = 1
            continue
        best, bestd = 1, -1.0
        for comp_k, comp in enumerate(clusters, start=1):
            d = max(_fov_dice(scars[c], datas[c], scars[s], datas[s]) for s in comp)
            if d > bestd:
                bestd, best = d, comp_k
        out[c] = best
    return out


def assign_for_eye_dict(eye_to_cids: dict, **kw) -> dict:
    """{eye_key: [cids]} → {cid: subgroup_int}, run per eye."""
    res = {}
    for eye, cids in eye_to_cids.items():
        try:
            res.update(assign_subgroups(cids, **kw))
        except Exception as exc:  # noqa: BLE001
            print(f"[subgroups] {eye} failed: {exc}")
            for c in cids:
                res[c] = 1
    return res
