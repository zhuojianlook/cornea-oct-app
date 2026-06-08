#!/usr/bin/env python3
"""Render visible background/cornea seed paint previews for an existing seed JSON."""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import slicer

from preview_io import save_previews, seed_masks_from_spec
from slicer_volume_io import load_input_volume


LAUNCH_CWD = os.environ.get("PWD") or os.getcwd()


def resolve_launch_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(LAUNCH_CWD, path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Render PNG previews from seed JSON paint.")
    parser.add_argument("--input-volume", required=True)
    parser.add_argument("--seed-json", required=True)
    parser.add_argument("--preview-dir", required=True)
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    input_volume = resolve_launch_path(args.input_volume)
    seed_json = resolve_launch_path(args.seed_json)
    preview_dir = resolve_launch_path(args.preview_dir)

    slicer.mrmlScene.Clear()
    volume_node = load_input_volume(input_volume)
    array = slicer.util.arrayFromVolume(volume_node).astype(np.float32)

    with open(seed_json, "r", encoding="utf-8") as fp:
        seed_spec = json.load(fp)
    masks = seed_masks_from_spec(array.shape, seed_spec)
    saved = save_previews(array, masks, preview_dir, "seeds", volume_node.GetSpacing())
    print(json.dumps({"preview_images": saved}, indent=2))


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
