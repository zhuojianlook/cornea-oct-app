"""Export finished cases to an nnU-Net v2 raw dataset.

Layout produced (under output/nnunet/<DatasetName>/):
  imagesTr/<case>_0000.nii.gz   the OCT volume (channel 0)
  labelsTr/<case>.nii.gz        integer labelmap, remapped to 0/1/2
  dataset.json

Final label convention (the training target): 0=background, 1=cornea, 2=scar.
The grown .seg.nrrd uses label = (seeds.json segment index + 1) with 0 for
unpainted voxels; we remap by segment NAME so order changes can't corrupt it.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import nibabel as nib

import settings
import orchestration as orch

NNUNET_LABELS = {"background": 0, "cornea": 1, "scar": 2}
DATASET_ROOT = settings.WORKSPACE_ROOT / "output" / "nnunet"


def _remap_label_nifti(case_id: str, base_nifti: Path, dst: Path) -> bool:
    """Write the nnU-Net labelmap (0/1/2) for one case. False if no seg."""
    import nrrd

    seg_path = orch.case_output_seg(case_id)
    if not seg_path.exists():
        return False
    data, _ = nrrd.read(str(seg_path))
    data = np.asarray(data)
    if data.ndim == 4:
        data = data.max(axis=int(np.argmin(data.shape)))

    seed_spec = orch.read_json(orch.case_seed_json(case_id))
    # grown label value (index+1) -> segment name
    name_of_value = {i + 1: s.get("name") for i, s in enumerate(seed_spec.get("segments", []))}

    out = np.zeros_like(data, dtype=np.uint8)  # unpainted (0) stays background
    for value, name in name_of_value.items():
        target = NNUNET_LABELS.get(name)
        if target:  # background→0 is already the default
            out[data == value] = target

    base = nib.load(str(base_nifti))
    dst.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.ascontiguousarray(out), base.affine), str(dst))
    return True


def export_case(case_id: str, dataset_dir: Path, base_nifti: Path) -> dict:
    cid = orch.safe_case_id(case_id)
    images = dataset_dir / "imagesTr"
    labels = dataset_dir / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)

    if not _remap_label_nifti(cid, base_nifti, labels / f"{cid}.nii.gz"):
        return {"case_id": cid, "exported": False, "reason": "no segmentation"}
    shutil.copyfile(base_nifti, images / f"{cid}_0000.nii.gz")
    return {"case_id": cid, "exported": True}


def write_dataset_json(dataset_dir: Path, num_training: int) -> None:
    (dataset_dir / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "OCT"},
        "labels": NNUNET_LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "description": "Cornea OCT background/cornea/scar segmentation",
    }, indent=2))


def cases_with_segmentation() -> list[str]:
    if not settings.CASES_ROOT.exists():
        return []
    out = []
    for case_dir in sorted(settings.CASES_ROOT.iterdir()):
        if (case_dir / "segmentation" / f"{case_dir.name}.seg.nrrd").exists():
            out.append(case_dir.name)
    return out
