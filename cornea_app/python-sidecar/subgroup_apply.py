"""Apply auto-subgroups to the cohort: write scar_subgroup to each scar-eye member, and rebuild
consensus PER SUBGROUP (so each consensus is a clean same-scar replicate set).

Clusters each scar eye's scans by pairwise scar-FOV-Dice (from the consensus's already-aligned warped
labels). Largest cluster = subgroup "1" (keeps the default consensus id); others get "2","3",… Controls
are left as one subgroup (normal — nothing to split). Rebuild uses the live /consensus/build endpoint.
"""
from __future__ import annotations

import sys
import json
import urllib.request
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import orchestration as orch
import settings

THRESH = 0.45


def _post(path, payload):
    req = urllib.request.Request("http://127.0.0.1:8765" + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)


def _fov_dice(sa, ca, sb, cb):
    c = ca & cb
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
        comps.append(sorted(comp))
    comps.sort(key=len, reverse=True)
    return comps


def main():
    root = settings.CASES_ROOT
    applied = []
    for d in sorted(root.glob("case_*_consensus")):
        ccid = d.name
        m = orch.read_manifest(ccid)
        members = m.get("consensus_cases") or []
        if not members:
            continue
        cls = "control" if any(str(orch.read_manifest(c).get("scar_classification")) == "control" for c in members) else "scar"
        if cls != "scar":
            for c in members:                       # controls → single subgroup
                orch.write_manifest_value(c, {"scar_subgroup": "1"})
            continue
        labs = {}
        for c in members:
            p = d / "scans" / c / "label.nii.gz"
            if p.exists():
                labs[c] = np.rint(np.asarray(nib.load(str(p)).dataobj)).astype(np.uint8)
        scarred = [c for c in labs if (labs[c] == 2).sum() > 0]
        if len(scarred) < 2:
            for c in members:
                orch.write_manifest_value(c, {"scar_subgroup": "1"})
            continue
        cornea = {c: labs[c] >= 1 for c in labs}
        scar = {c: labs[c] == 2 for c in labs}
        sim = {}
        for i in range(len(scarred)):
            for j in range(i + 1, len(scarred)):
                sim[(scarred[i], scarred[j])] = _fov_dice(scar[scarred[i]], cornea[scarred[i]],
                                                          scar[scarred[j]], cornea[scarred[j]])
        comps = _components(scarred, sim, THRESH)
        # map member → subgroup; non-scarred members attach to subgroup 1
        sg = {}
        for k, comp in enumerate(comps, 1):
            for c in comp:
                sg[c] = k
        for c in members:
            sg.setdefault(c, 1)
            orch.write_manifest_value(c, {"scar_subgroup": str(sg[c])})
        eye = ccid.replace("case_", "").replace("_consensus", "")
        if len(comps) == 1:
            continue  # no real split → existing consensus already correct
        # rebuild consensus per subgroup (only multi-scan subgroups get a consensus case)
        tag = lambda c: c.split("_")[-1]
        for k, comp in enumerate(comps, 1):
            members_k = [c for c in members if sg[c] == k]
            if len(members_k) >= 2:
                try:
                    _post("/api/consensus/build", {"cases": members_k, "subgroup": str(k)})
                    applied.append(f"{eye} sg{k} {{{','.join(tag(c) for c in members_k)}}}")
                except Exception as exc:  # noqa: BLE001
                    applied.append(f"{eye} sg{k} REBUILD FAILED: {str(exc)[:80]}")
            else:
                applied.append(f"{eye} sg{k} {{{tag(members_k[0])}}} (single scan — no consensus)")
    print("Applied subgroups + rebuilt consensus for split eyes:")
    for a in applied:
        print("  " + a)
    if not applied:
        print("  (no scar eyes split)")


if __name__ == "__main__":
    main()
