#!/usr/bin/env python3
"""Run a seeded Grow from Seeds segmentation inside 3D Slicer.

This script is meant to be launched by Slicer, not system Python:

    Slicer --no-main-window --python-script seeded_grow_from_seeds.py --self-test

Seed coordinates use Slicer's IJK order: [i, j, k]. Numpy arrays internally use
[k, j, i], so all coordinate conversion is kept in this file.
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np

import slicer
import vtk

from slicer_volume_io import load_input_volume


LAUNCH_CWD = os.environ.get("PWD") or os.getcwd()


DEFAULT_SEGMENT_SPECS = [
    {
        "name": "background",
        "color": [0.05, 0.05, 0.05],
        "seeds": [
            {"ijk": [8, 8, 8], "radius_voxels": [5, 5, 4]},
            {"ijk": [86, 86, 70], "radius_voxels": [5, 5, 4]},
        ],
    },
    {
        "name": "cornea",
        "color": [0.1, 0.7, 1.0],
        "seeds": [
            {"ijk": [48, 48, 40], "radius_voxels": [7, 7, 4]},
            {"ijk": [39, 52, 39], "radius_voxels": [5, 5, 3]},
        ],
    },
]


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Create background/cornea/scar seeds and run Grow from Seeds."
    )
    parser.add_argument("--input-volume", help="Input OCT volume, for example .nrrd or .nii.gz")
    parser.add_argument("--seed-json", help="JSON file containing segment seed definitions")
    parser.add_argument("--output-seg", default="output/segmentation.seg.nrrd")
    parser.add_argument("--qa-json", default="output/segmentation_qa.json")
    parser.add_argument("--scene", help="Optional .mrb scene output for debugging/review")
    parser.add_argument("--preview-dir", help="Optional directory for PNG segmentation previews")
    parser.add_argument("--seed-locality-factor", type=float, default=0.0)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Create a synthetic OCT-like volume and run the bridge on it.",
    )
    return parser.parse_args(argv)


def resolve_launch_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(LAUNCH_CWD, path))


def resolve_paths(args):
    args.input_volume = resolve_launch_path(args.input_volume)
    args.seed_json = resolve_launch_path(args.seed_json)
    args.output_seg = resolve_launch_path(args.output_seg)
    args.qa_json = resolve_launch_path(args.qa_json)
    args.scene = resolve_launch_path(args.scene)
    args.preview_dir = resolve_launch_path(args.preview_dir)
    return args


def ensure_parent_dir(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def create_self_test_volume():
    shape_kji = (80, 96, 96)
    array = np.full(shape_kji, 20.0, dtype=np.float32)
    zz, yy, xx = np.indices(shape_kji)

    cornea = (
        ((xx - 48.0) / 32.0) ** 2
        + ((yy - 48.0) / 24.0) ** 2
        + ((zz - 40.0) / 14.0) ** 2
    ) < 1.0
    scar_candidate = (
        ((xx - 58.0) / 9.0) ** 2
        + ((yy - 44.0) / 8.0) ** 2
        + ((zz - 43.0) / 5.0) ** 2
    ) < 1.0
    scar = cornea & scar_candidate

    array[cornea] = 95.0
    array[scar] = 155.0

    volume_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", "self_test_oct")
    slicer.util.updateVolumeFromArray(volume_node, array)
    volume_node.SetSpacing(0.03, 0.03, 0.05)
    volume_node.SetOrigin(0.0, 0.0, 0.0)
    volume_node.CreateDefaultDisplayNodes()
    if volume_node.GetDisplayNode():
        volume_node.GetDisplayNode().AutoWindowLevelOn()
    return volume_node


def load_volume(path):
    if not path:
        raise ValueError("--input-volume is required unless --self-test is used")
    return load_input_volume(path)


def load_seed_spec(path, use_self_test_defaults):
    if path:
        with open(path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        if isinstance(data, dict):
            return data.get("segments", [])
        return data
    if use_self_test_defaults:
        return DEFAULT_SEGMENT_SPECS
    raise ValueError("--seed-json is required unless --self-test is used")


def normalize_radius(radius):
    if radius is None:
        return np.array([3, 3, 1], dtype=np.float64)
    if isinstance(radius, (int, float)):
        return np.array([radius, radius, radius], dtype=np.float64)
    if len(radius) != 3:
        raise ValueError("radius_voxels must be a number or a 3-value list")
    return np.array(radius, dtype=np.float64)


def paint_ellipsoid(mask_kji, ijk, radius_voxels):
    center_ijk = np.array(ijk, dtype=np.float64)
    radius_ijk = normalize_radius(radius_voxels)
    if np.any(radius_ijk <= 0):
        raise ValueError("radius_voxels values must be positive")

    shape_kji = np.array(mask_kji.shape)
    center_kji = center_ijk[[2, 1, 0]]
    radius_kji = radius_ijk[[2, 1, 0]]

    min_kji = np.maximum(np.floor(center_kji - radius_kji).astype(int), 0)
    max_kji = np.minimum(np.ceil(center_kji + radius_kji).astype(int), shape_kji - 1)

    slices = tuple(slice(min_kji[axis], max_kji[axis] + 1) for axis in range(3))
    local_shape = tuple(max_kji - min_kji + 1)
    local_indices = np.indices(local_shape)
    for axis in range(3):
        local_indices[axis] = local_indices[axis] + min_kji[axis]

    distance = np.zeros(local_shape, dtype=np.float64)
    for axis in range(3):
        distance += ((local_indices[axis] - center_kji[axis]) / radius_kji[axis]) ** 2
    mask_kji[slices][distance <= 1.0] = 1


def paint_line(mask_kji, start_ijk, end_ijk, radius_voxels):
    start = np.array(start_ijk, dtype=np.float64)
    end = np.array(end_ijk, dtype=np.float64)
    radius_ijk = normalize_radius(radius_voxels)
    distance = float(np.linalg.norm(end - start))
    step = max(1.0, float(np.min(radius_ijk)) * 0.45)
    sample_count = max(2, int(np.ceil(distance / step)) + 1)
    for t in np.linspace(0.0, 1.0, sample_count):
        paint_ellipsoid(mask_kji, start * (1.0 - t) + end * t, radius_ijk)


def paint_polyline(mask_kji, points_ijk, radius_voxels):
    points = [point for point in points_ijk if point is not None]
    if len(points) == 1:
        paint_ellipsoid(mask_kji, points[0], radius_voxels)
        return
    for start, end in zip(points[:-1], points[1:]):
        paint_line(mask_kji, start, end, radius_voxels)


def create_segmentation_from_seeds(volume_node, segment_specs):
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass(
        "vtkMRMLSegmentationNode", "SeededGrowFromSeeds"
    )
    segmentation_node.SetReferenceImageGeometryParameterFromVolumeNode(volume_node)
    segmentation_node.CreateDefaultDisplayNodes()

    volume_array = slicer.util.arrayFromVolume(volume_node)
    segment_ids_by_name = {}

    for spec in segment_specs:
        name = spec["name"]
        color = spec.get("color", [0.5, 0.5, 0.5])
        segment_id = segmentation_node.GetSegmentation().AddEmptySegment(name)
        segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
        segment.SetColor(float(color[0]), float(color[1]), float(color[2]))

        seed_mask = np.zeros(volume_array.shape, dtype=np.uint8)
        for stroke in spec.get("strokes", []):
            points = stroke.get("points_ijk") or stroke.get("ijk_points") or []
            paint_polyline(seed_mask, points, stroke.get("radius_voxels"))
        for seed in spec.get("seeds", []):
            paint_ellipsoid(seed_mask, seed["ijk"], seed.get("radius_voxels"))

        voxel_count = int(np.count_nonzero(seed_mask))
        if voxel_count == 0:
            raise ValueError(f"Segment '{name}' has no painted seed voxels")

        slicer.util.updateSegmentBinaryLabelmapFromArray(
            seed_mask, segmentation_node, segment_id, volume_node
        )
        segment_ids_by_name[name] = segment_id

    return segmentation_node, segment_ids_by_name


def set_source_volume(segment_editor_widget, volume_node):
    if hasattr(segment_editor_widget, "setSourceVolumeNode"):
        segment_editor_widget.setSourceVolumeNode(volume_node)
    else:
        segment_editor_widget.setMasterVolumeNode(volume_node)


def run_grow_from_seeds(volume_node, segmentation_node, seed_locality_factor):
    segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
    segment_editor_widget.setMRMLScene(slicer.mrmlScene)
    segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segment_editor_widget.setMRMLSegmentEditorNode(segment_editor_node)
    segment_editor_widget.setSegmentationNode(segmentation_node)
    set_source_volume(segment_editor_widget, volume_node)

    segment_editor_widget.setActiveEffectByName("Grow from seeds")
    effect = segment_editor_widget.activeEffect()
    if effect is None:
        raise RuntimeError("Could not activate Slicer's 'Grow from seeds' effect")

    effect.setParameter("SeedLocalityFactor", str(seed_locality_factor))
    effect.setParameter("AutoUpdate", "1")
    effect.self().onPreview()
    slicer.app.processEvents()
    effect.self().onApply()
    slicer.app.processEvents()

    slicer.mrmlScene.RemoveNode(segment_editor_node)
    segment_editor_widget = None


def segment_stats(volume_node, segmentation_node, segment_ids_by_name):
    spacing = volume_node.GetSpacing()
    voxel_volume = float(spacing[0] * spacing[1] * spacing[2])
    stats = {}
    for name, segment_id in segment_ids_by_name.items():
        array = slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id, volume_node)
        nonzero = np.argwhere(array > 0)
        voxel_count = int(nonzero.shape[0])
        item = {
            "segment_id": segment_id,
            "voxel_count": voxel_count,
            "volume_mm3": voxel_count * voxel_volume,
        }
        if voxel_count:
            min_kji = nonzero.min(axis=0).astype(int).tolist()
            max_kji = nonzero.max(axis=0).astype(int).tolist()
            item["bounds_ijk"] = {
                "min": [min_kji[2], min_kji[1], min_kji[0]],
                "max": [max_kji[2], max_kji[1], max_kji[0]],
            }
        stats[name] = item
    return stats


def save_outputs(volume_node, segmentation_node, segment_ids_by_name, args):
    ensure_parent_dir(args.output_seg)
    if not slicer.util.saveNode(segmentation_node, args.output_seg):
        raise RuntimeError(f"Failed to save segmentation: {args.output_seg}")

    preview_images = []
    if args.preview_dir:
        from preview_io import save_previews

        volume_array = slicer.util.arrayFromVolume(volume_node).astype(np.float32)
        masks_by_name = {
            name: slicer.util.arrayFromSegmentBinaryLabelmap(segmentation_node, segment_id, volume_node)
            for name, segment_id in segment_ids_by_name.items()
        }
        preview_images = save_previews(
            volume_array,
            masks_by_name,
            args.preview_dir,
            "segmentation",
            volume_node.GetSpacing(),
        )

    qa = {
        "slicer_version": slicer.app.applicationVersion,
        "volume_name": volume_node.GetName(),
        "volume_spacing": list(volume_node.GetSpacing()),
        "segments": segment_stats(volume_node, segmentation_node, segment_ids_by_name),
        "preview_images": preview_images,
    }
    ensure_parent_dir(args.qa_json)
    with open(args.qa_json, "w", encoding="utf-8") as fp:
        json.dump(qa, fp, indent=2)

    if args.scene:
        ensure_parent_dir(args.scene)
        if not slicer.util.saveScene(args.scene):
            raise RuntimeError(f"Failed to save scene: {args.scene}")


def main(argv):
    args = resolve_paths(parse_args(argv))
    slicer.mrmlScene.Clear()

    volume_node = create_self_test_volume() if args.self_test else load_volume(args.input_volume)
    segment_specs = load_seed_spec(args.seed_json, args.self_test)
    segmentation_node, segment_ids_by_name = create_segmentation_from_seeds(volume_node, segment_specs)
    run_grow_from_seeds(volume_node, segmentation_node, args.seed_locality_factor)
    save_outputs(volume_node, segmentation_node, segment_ids_by_name, args)

    print(f"Saved segmentation: {os.path.abspath(args.output_seg)}")
    print(f"Saved QA: {os.path.abspath(args.qa_json)}")
    if args.scene:
        print(f"Saved scene: {os.path.abspath(args.scene)}")


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
