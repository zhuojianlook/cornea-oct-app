"""Auto-subgroup DRY-RUN over the cohort, from the consensus's already-aligned warped labels.

For each per-eye consensus, build the pairwise scar-FOV-Dice graph among its scans and split into
subgroups by connected components (edge when Dice ≥ threshold). Scans imaging a DIFFERENT scar/region
fall into a separate component → a different subgroup. Reports which eyes split (and how). Also shows the
within-subgroup cornea Dice for split eyes, to confirm alignment is robust WITHIN a true subgroup.
READ-ONLY (no manifest writes).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import settings

THRESH = 0.45


def _dice(a, b):
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else float("nan")


def _fov_dice(sa, ca, sb, cb, da=None, db=None):
    # FOV = image-data overlap (consensus.py / subgroups.py), AND-ed with the cornea masks.
    c = ca & cb
    if da is not None and db is not None:
        c = c & da & db
    a, b = sa & c, sb & c
    s = int(a.sum()) + int(b.sum())
    return 2.0 * int((a & b).sum()) / s if s else 0.0


def _components(cids, sim, thresh):
    adj = {c: set() for c in cids}
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            if sim[(cids[i], cids[j])] >= thresh:
                adj[cids[i]].add(cids[j]); adj[cids[j]].add(cids[i])
    seen, comps = set(), []
    for c in cids:
        if c in seen:
            continue
        st, comp = [c], []
        while st:
            x = st.pop()
            if x in seen:
                continue
            seen.add(x); comp.append(x); st.extend(adj[x] - seen)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def main():
    root = settings.CASES_ROOT
    n_split = 0
    for d in sorted(root.glob("case_*_consensus")):
        ccid = d.name
        members = orch.read_manifest(ccid).get("consensus_cases") or []
        cls = "control" if any(str(orch.read_manifest(c).get("scar_classification")) == "control" for c in members) else "scar"
        labs = {}
        data = {}
        for c in members:
            p = d / "scans" / c / "label.nii.gz"
            if p.exists():
                labs[c] = np.rint(np.asarray(nib.load(str(p)).dataobj)).astype(np.uint8)
                vp = d / "scans" / c / "volume.nii.gz"   # warped per-scan image-data FOV (consensus.py)
                if vp.exists():
                    data[c] = np.asarray(nib.load(str(vp)).dataobj) > 0
        cids = list(labs)
        if len(cids) < 2:
            continue
        cornea = {c: labs[c] >= 1 for c in cids}
        scar = {c: labs[c] == 2 for c in cids}
        scarred = [c for c in cids if scar[c].sum() > 0]
        eye = ccid.replace("case_", "").replace("_consensus", "")
        if len(scarred) < 2:
            continue  # control / no-scar eye → single subgroup
        sim = {}
        for i in range(len(scarred)):
            for j in range(i + 1, len(scarred)):
                sim[(scarred[i], scarred[j])] = _fov_dice(scar[scarred[i]], cornea[scarred[i]],
                                                          scar[scarred[j]], cornea[scarred[j]],
                                                          data.get(scarred[i]), data.get(scarred[j]))
        comps = _components(scarred, sim, THRESH)
        if len(comps) > 1:
            n_split += 1
            tag = lambda c: c.split("_")[-1]
            print(f"{eye:11} ({cls}) SPLITS into {len(comps)} subgroups: " +
                  " | ".join("{" + ",".join(tag(c) for c in comp) + "}" for comp in comps))
            # within-subgroup cornea Dice for the largest subgroup (alignment should be high there)
            big = comps[0]
            if len(big) >= 2:
                cd = [_dice(cornea[big[i]], cornea[big[j]]) for i in range(len(big)) for j in range(i + 1, len(big))]
                print(f"            largest-subgroup cornea Dice {np.nanmean(cd):.3f} (vs whole-eye lower)")
    print(f"\n{n_split} eye(s) split into multiple subgroups (threshold scar-FOV-Dice {THRESH}).")


if __name__ == "__main__":
    main()
