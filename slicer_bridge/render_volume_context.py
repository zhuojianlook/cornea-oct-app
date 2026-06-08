#!/usr/bin/env python3
"""Render unpainted OCT slice previews for vision-agent paint planning."""

import argparse
import os
import sys
import traceback

import numpy as np
import slicer

from preview_io import save_previews
from slicer_volume_io import load_input_volume


LAUNCH_CWD = os.environ.get("PWD") or os.getcwd()


def resolve_launch_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(LAUNCH_CWD, path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render unpainted context PNGs from an OCT volume.")
    parser.add_argument("--input-volume", required=True)
    parser.add_argument("--preview-dir", required=True)
    parser.add_argument("--max-slices-per-orientation", type=int, default=9)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    input_volume = resolve_launch_path(args.input_volume)
    preview_dir = resolve_launch_path(args.preview_dir)

    slicer.mrmlScene.Clear()
    volume_node = load_input_volume(input_volume)
    array = slicer.util.arrayFromVolume(volume_node).astype(np.float32)

    saved = save_previews(
        array,
        {},
        preview_dir,
        "context",
        volume_node.GetSpacing(),
        max_slices_per_orientation=args.max_slices_per_orientation,
    )
    print({"context_preview_images": saved})


def exit_slicer(status):
    if hasattr(slicer.util, "exit"):
        slicer.util.exit(status)
    else:
        slicer.app.exit(status)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
        exit_slicer(0)
    except Exception:
        traceback.print_exc()
        exit_slicer(1)
