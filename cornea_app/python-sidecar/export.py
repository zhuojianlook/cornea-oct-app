"""Export finished cases to an nnU-Net v2 raw dataset.

Layout produced (under output/nnunet/<DatasetName>/):
  imagesTr/<case>_0000.nii.gz   the OCT volume (channel 0)
  labelsTr/<case>.nii.gz        integer labelmap, remapped to 0/1/2
  dataset.json

Final label convention (the training target): 0=background, 1=cornea, 2=scar.
The training labels are the expert-corrected labelmaps (<case>_corrected.nii.gz).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import settings
import orchestration as orch
import labels as label_mod

NNUNET_LABELS = label_mod.NNUNET_LABELS
DATASET_ROOT = settings.WORKSPACE_ROOT / "output" / "nnunet"


def clean_dataset(dataset_dir: Path) -> None:
    """Wipe imagesTr/labelsTr before a re-export so deleted/renamed cases (e.g. the old
    "_2" duplicates) don't linger as orphan training pairs that silently inflate the set.
    Only the two image/label subdirs are removed; any hand-added metadata is preserved."""
    for sub in ("imagesTr", "labelsTr"):
        shutil.rmtree(dataset_dir / sub, ignore_errors=True)


def export_case(case_id: str, dataset_dir: Path, base_nifti: Path) -> dict:
    cid = orch.safe_case_id(case_id)
    images = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    images.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    arr, source = label_mod.best_labelmap_nnunet(cid)
    if arr is None:
        return {"case_id": cid, "exported": False, "reason": "no segmentation"}
    label_mod.write_label_nifti(arr, base_nifti, labels_dir / f"{cid}.nii.gz")
    shutil.copyfile(base_nifti, images / f"{cid}_0000.nii.gz")
    return {"case_id": cid, "exported": True, "source": source}


def write_dataset_json(dataset_dir: Path, num_training: int) -> None:
    (dataset_dir / "dataset.json").write_text(json.dumps({
        "channel_names": {"0": "OCT"},
        "labels": NNUNET_LABELS,
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "description": "Cornea OCT background/cornea/scar segmentation",
    }, indent=2))


def cases_with_segmentation() -> list[str]:
    """Cases that have an expert-corrected labelmap to export."""
    if not settings.CASES_ROOT.exists():
        return []
    out = []
    for case_dir in sorted(settings.CASES_ROOT.iterdir()):
        if label_mod.corrected_path(case_dir.name).exists():
            out.append(case_dir.name)
    # Respect the "Schedule for training" gate: if any corrected case is scheduled, export only those.
    return orch.filter_scheduled(out)
