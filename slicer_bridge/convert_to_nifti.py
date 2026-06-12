"""Convert any Slicer-loadable volume (incl. DICOM .dcm/.dicom) to a NIfTI file.

Run under the Slicer executable:
    Slicer --no-main-window --python-script convert_to_nifti.py \
        --input-volume <path> --output <out.nii.gz>

Used by the sidecar to make a niivue-ready NIfTI from DICOM input (niivue can't
read DICOM directly). Reuses slicer_volume_io.load_input_volume, which handles
the DICOM temporary-database import.
"""
import argparse
import os

import slicer
import slicer_volume_io

LAUNCH_CWD = os.environ.get("PWD") or os.getcwd()


def resolve(path):
    return os.path.abspath(os.path.join(LAUNCH_CWD, path)) if not os.path.isabs(path) else path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-volume", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    node = slicer_volume_io.load_input_volume(resolve(args.input_volume))
    out = resolve(args.output)
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if not slicer.util.saveNode(node, out):
        raise RuntimeError(f"Failed to save volume to {out}")
    print(f"SAVED:{out}")


main()
