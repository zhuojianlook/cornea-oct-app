"""Input volume loading helpers for Slicer bridge scripts."""

import os

import slicer


def _first_scalar_volume_from_node_ids(node_ids):
    for node_id in node_ids:
        node = slicer.mrmlScene.GetNodeByID(node_id)
        if node and node.IsA("vtkMRMLScalarVolumeNode"):
            return node
    return None


def load_dicom_volume(path):
    """Load a DICOM file by importing its containing folder into a temporary database."""
    from DICOMLib import DICOMUtils

    dicom_dir = os.path.dirname(os.path.abspath(path))
    if not dicom_dir:
        raise RuntimeError(f"Cannot determine DICOM folder for {path}")

    with DICOMUtils.TemporaryDICOMDatabase() as db:
        if not DICOMUtils.importDicom(dicom_dir, db, copyFiles=False):
            raise RuntimeError(f"Failed to import DICOM folder: {dicom_dir}")

        long_path = slicer.util.longPath(os.path.abspath(path))
        series_uid = slicer.dicomDatabase.seriesForFile(long_path)
        if series_uid:
            loaded_node_ids = DICOMUtils.loadSeriesByUID([series_uid])
            volume_node = _first_scalar_volume_from_node_ids(loaded_node_ids)
            if volume_node:
                return volume_node

        loaded_node_ids = []
        for patient_uid in slicer.dicomDatabase.patients():
            loaded_node_ids.extend(DICOMUtils.loadPatientByUID(patient_uid))
        volume_node = _first_scalar_volume_from_node_ids(loaded_node_ids)
        if volume_node:
            return volume_node

    raise RuntimeError(f"Failed to load DICOM volume from: {path}")


def load_input_volume(path):
    """Load standard medical image files, with DICOM fallback for .dcm/.dicom."""
    if not path:
        raise ValueError("Input volume path is required")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    volume_node = slicer.util.loadVolume(path)
    if volume_node is not None:
        return volume_node

    extension = os.path.splitext(path)[1].lower()
    if extension in [".dcm", ".dicom"]:
        return load_dicom_volume(path)

    raise RuntimeError(f"Failed to load volume: {path}")
