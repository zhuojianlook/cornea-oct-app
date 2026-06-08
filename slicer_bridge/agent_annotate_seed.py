#!/usr/bin/env python3
"""Append Grow-from-Seeds seed annotations from agent or preview coordinates."""

import argparse
import json
import os


SEGMENT_COLORS = {
    "background": [0.05, 0.05, 0.05],
    "cornea": [0.1, 0.7, 1.0],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Append a background/cornea seed to seeds.json.")
    parser.add_argument("--seed-json", required=True)
    parser.add_argument("--segment", choices=["background", "cornea"], required=True)
    parser.add_argument("--radius", nargs=3, type=int, default=[5, 5, 2])
    parser.add_argument("--ijk", nargs=3, type=int)
    parser.add_argument("--preview-manifest")
    parser.add_argument("--file-name")
    parser.add_argument("--x-fraction", type=float)
    parser.add_argument("--y-fraction", type=float)
    return parser.parse_args()


def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fp:
            return json.load(fp)
    return default


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def seed_from_preview(args):
    if not args.preview_manifest or not args.file_name:
        return None
    if args.x_fraction is None or args.y_fraction is None:
        return None

    manifest = load_json(args.preview_manifest, {"images": []})
    item = next(
        (image for image in manifest.get("images", []) if image.get("file_name") == args.file_name),
        None,
    )
    if item is None:
        raise ValueError(f"Preview file not found in manifest: {args.file_name}")

    source_width = int(item["source_width"])
    source_height = int(item["source_height"])
    column = clamp(round(args.x_fraction * (source_width - 1)), 0, source_width - 1)
    row_from_top = clamp(round(args.y_fraction * (source_height - 1)), 0, source_height - 1)
    row = source_height - 1 - row_from_top
    slice_index = int(item["slice_index"])
    orientation = item["orientation"]

    if orientation == "axial":
        return [column, row, slice_index]
    if orientation == "coronal":
        return [column, slice_index, row]
    if orientation == "sagittal":
        return [slice_index, column, row]
    raise ValueError(f"Unsupported preview orientation: {orientation}")


def append_seed(seed_spec, segment_name, ijk, radius):
    segments = seed_spec.setdefault("segments", [])
    segment = next((item for item in segments if item.get("name") == segment_name), None)
    if segment is None:
        segment = {"name": segment_name, "color": SEGMENT_COLORS[segment_name], "seeds": []}
        segments.append(segment)
    segment.setdefault("seeds", []).append(
        {
            "ijk": [int(value) for value in ijk],
            "radius_voxels": [max(1, int(value)) for value in radius],
        }
    )
    seed_spec["segments"] = [
        item for name in ("background", "cornea") for item in segments if item.get("name") == name
    ]
    return seed_spec


def main():
    args = parse_args()
    ijk = args.ijk or seed_from_preview(args)
    if ijk is None:
        raise ValueError("Provide --ijk or preview coordinates via --preview-manifest/--file-name/--x-fraction/--y-fraction")

    seed_spec = load_json(args.seed_json, {"segments": []})
    seed_spec = append_seed(seed_spec, args.segment, ijk, args.radius)
    with open(args.seed_json, "w", encoding="utf-8") as fp:
        json.dump(seed_spec, fp, indent=2)

    print(json.dumps({"segment": args.segment, "ijk": ijk, "radius_voxels": args.radius}, indent=2))


if __name__ == "__main__":
    main()
