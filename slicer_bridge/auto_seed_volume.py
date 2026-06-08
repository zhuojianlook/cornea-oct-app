#!/usr/bin/env python3
"""Generate initial background/cornea seeds for one volume in 3D Slicer.

This is a heuristic initializer. It does not claim a final segmentation; it
creates case-specific starting seeds for Grow from Seeds.
"""

import argparse
import json
import os
import sys
import traceback

import numpy as np
import slicer

from slicer_volume_io import load_input_volume
from preview_io import save_previews, seed_masks_from_spec


LAUNCH_CWD = os.environ.get("PWD") or os.getcwd()


def resolve_launch_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(LAUNCH_CWD, path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Generate initial Grow from Seeds seed JSON.")
    parser.add_argument("--input-volume", required=True)
    parser.add_argument("--output-seed-json", required=True)
    parser.add_argument("--qa-json")
    parser.add_argument("--preview-dir")
    parser.add_argument("--axial-slice-count", type=int, default=17)
    parser.add_argument("--axial-column-count", type=int, default=49)
    parser.add_argument("--through-plane-column-count", type=int, default=7)
    parser.add_argument("--background-through-plane", action="store_true")
    parser.add_argument("--air-gap-scale", type=float, default=0.32)
    parser.add_argument("--cornea-band-position", type=float, default=0.52)
    return parser.parse_args(argv)


def ensure_parent_dir(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def clamp_ijk(ijk, dims_ijk):
    return [
        int(max(0, min(dims_ijk[axis] - 1, round(ijk[axis]))))
        for axis in range(3)
    ]


def radius(dims_ijk, divisors, minimums):
    return [
        int(max(minimums[axis], round(dims_ijk[axis] / divisors[axis])))
        for axis in range(3)
    ]


def sample_evenly(values, count):
    values = sorted(set(int(value) for value in values))
    if not values or int(count) <= 0:
        return []
    count = max(1, min(int(count), len(values)))
    if count == len(values):
        return values
    positions = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[int(position)] for position in sorted(set(positions))]


def moving_average(values, window):
    window = int(max(1, window))
    if window <= 1:
        return np.asarray(values, dtype=np.float32)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    values = np.asarray(values, dtype=np.float32)
    before = window // 2
    after = window - 1 - before
    padded = np.pad(values, (before, after), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def smooth_stroke(points, sort_axis, smooth_window=5):
    if len(points) < 3:
        return points
    ordered = sorted(([int(value) for value in point] for point in points), key=lambda point: point[sort_axis])
    coordinates = np.array(ordered, dtype=np.float32)
    x = coordinates[:, sort_axis]
    x_span = float(np.max(x) - np.min(x))
    if x_span <= 0:
        return ordered
    normalized_x = (x - np.mean(x)) / x_span
    for axis in range(3):
        if axis == sort_axis:
            continue
        y = coordinates[:, axis]
        if float(np.max(y) - np.min(y)) < 1.0:
            continue
        degree = min(2, len(ordered) - 1)
        coeffs = np.polyfit(normalized_x, y, degree)
        fitted = np.polyval(coeffs, normalized_x)
        residual = np.abs(y - fitted)
        median_residual = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median_residual)))
        cutoff = median_residual + max(4.0, 3.0 * mad)
        keep = residual <= cutoff
        if np.count_nonzero(keep) >= degree + 1 and np.count_nonzero(~keep):
            coeffs = np.polyfit(normalized_x[keep], y[keep], degree)
            fitted = np.polyval(coeffs, normalized_x)
        coordinates[:, axis] = fitted
    return [[int(round(value)) for value in point] for point in coordinates]


def connected_runs(mask):
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, breaks)
    return [(int(group[0]), int(group[-1])) for group in groups if group.size]


def nearest_candidate(coords_kji, target_kji, max_points=200000):
    if coords_kji.size == 0:
        return None
    coords = coords_kji
    if coords.shape[0] > max_points:
        step = int(np.ceil(coords.shape[0] / max_points))
        coords = coords[::step]
    target = np.array(target_kji, dtype=np.float64)
    distances = np.sum((coords.astype(np.float64) - target) ** 2, axis=1)
    return coords[int(np.argmin(distances))]


def kji_to_ijk(kji):
    return [int(kji[2]), int(kji[1]), int(kji[0])]


def dedupe_points(points):
    seen = set()
    unique = []
    for point in points:
        key = tuple(int(value) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(key))
    return unique


def mask_coords(mask):
    coords = np.argwhere(mask)
    if coords.shape[0] == 0:
        return coords
    return coords


def robust_percentiles(array):
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError("Volume has no finite voxels")
    return {
        f"p{percentile}": float(np.percentile(finite, percentile))
        for percentile in (1, 5, 50, 65, 75, 85, 90, 95, 98, 99)
    }


def central_mask(shape_kji):
    k, j, i = np.indices(shape_kji)
    dims = np.array([shape_kji[2], shape_kji[1], shape_kji[0]], dtype=np.float64)
    return (
        (i >= dims[0] * 0.12)
        & (i <= dims[0] * 0.88)
        & (j >= dims[1] * 0.10)
        & (j <= dims[1] * 0.90)
        & (k >= dims[2] * 0.08)
        & (k <= dims[2] * 0.92)
    )


def seed_from_mask(mask, fallback_ijk, dims_ijk, target_ijk=None):
    coords = mask_coords(mask)
    if coords.shape[0] == 0:
        return clamp_ijk(fallback_ijk, dims_ijk)
    if target_ijk is None:
        center_kji = np.median(coords, axis=0)
    else:
        target_kji = [target_ijk[2], target_ijk[1], target_ijk[0]]
        nearest = nearest_candidate(coords, target_kji)
        center_kji = nearest if nearest is not None else np.median(coords, axis=0)
    return clamp_ijk(kji_to_ijk(center_kji), dims_ijk)


def sample_seed_points_from_mask(mask, dims_ijk, samples_by_axis):
    coords = mask_coords(mask)
    if coords.shape[0] == 0:
        return []

    global_median = np.median(coords, axis=0)
    points = []
    for axis, count in samples_by_axis.items():
        lower = int(coords[:, axis].min())
        upper = int(coords[:, axis].max())
        if upper <= lower:
            indices = [lower]
        else:
            indices = np.linspace(lower, upper, max(1, count)).round().astype(int).tolist()
        for index in sorted(set(indices)):
            slice_coords = coords[coords[:, axis] == index]
            if slice_coords.shape[0] == 0:
                target = np.array(global_median, dtype=np.float64)
                target[axis] = index
                nearest = nearest_candidate(coords, target)
                center_kji = nearest if nearest is not None else target
            else:
                center_kji = np.median(slice_coords, axis=0)
            points.append(clamp_ijk(kji_to_ijk(center_kji), dims_ijk))
    return dedupe_points(points)


def paired_background_points(cornea_points, dims_ijk, margin_ijk):
    points = []
    for i, j, k in cornea_points:
        points.extend(
            [
                [margin_ijk[0], j, k],
                [dims_ijk[0] - 1 - margin_ijk[0], j, k],
                [i, margin_ijk[1], k],
                [i, dims_ijk[1] - 1 - margin_ijk[1], k],
                [i, j, margin_ijk[2]],
                [i, j, dims_ijk[2] - 1 - margin_ijk[2]],
            ]
        )
    return dedupe_points(clamp_ijk(point, dims_ijk) for point in points)


def axial_band_for_column(image, column, percentiles):
    height, width = image.shape
    half_width = max(2, int(round(width / 160)))
    left = max(0, int(column) - half_width)
    right = min(width, int(column) + half_width + 1)
    profile = np.median(image[:, left:right], axis=1)
    profile = moving_average(profile, max(7, int(round(height / 55))))

    search_start = int(round(height * 0.12))
    search_end = int(round(height * 0.96))
    search = profile[search_start:search_end]
    if search.size == 0:
        return None

    threshold = max(float(np.percentile(search, 68)), float(percentiles["p75"]))
    mask = profile >= threshold
    mask[:search_start] = False
    mask[search_end:] = False

    min_length = max(10, int(round(height * 0.025)))
    max_length = max(min_length + 1, int(round(height * 0.45)))
    best = None
    for start, end in connected_runs(mask):
        length = end - start + 1
        if length < min_length or length > max_length:
            continue
        band = profile[start : end + 1]
        center = (start + end) / 2.0
        center_penalty = abs(center - height * 0.58) / float(height)
        score = float(np.mean(band)) + length * 0.03 - center_penalty * 20.0
        if best is None or score > best[0]:
            best = (score, start, end)
    if best is None:
        return None
    _, start, end = best
    return int(start), int(end)


def axial_band_quality(image, percentiles):
    height, width = image.shape
    columns = np.linspace(width * 0.12, width * 0.88, 31).round().astype(int)
    bands = [axial_band_for_column(image, column, percentiles) for column in columns]
    valid = [band for band in bands if band is not None]
    if len(valid) < 6:
        return None
    tops = np.array([band[0] for band in valid], dtype=np.float32)
    bottoms = np.array([band[1] for band in valid], dtype=np.float32)
    thickness = bottoms - tops
    return {
        "valid_columns": len(valid),
        "median_top": float(np.median(tops)),
        "median_bottom": float(np.median(bottoms)),
        "median_thickness": float(np.median(thickness)),
    }


def semantic_seed_strokes(
    array,
    percentiles,
    dims_ijk,
    axial_slice_count=17,
    axial_column_count=49,
    through_plane_column_count=7,
    background_through_plane=False,
    air_gap_scale=0.32,
    cornea_band_position=0.52,
):
    depth, height, width = array.shape
    k_quality = []
    for k in range(depth):
        quality = axial_band_quality(array[k], percentiles)
        if quality and quality["median_thickness"] >= max(12, height * 0.025):
            k_quality.append((k, quality))

    if k_quality:
        selected_k = sample_evenly([k for k, _quality in k_quality], axial_slice_count)
    else:
        selected_k = sample_evenly(
            range(int(depth * 0.08), max(int(depth * 0.92), 1)),
            axial_slice_count,
        )

    columns = sample_evenly(
        range(int(width * 0.10), max(int(width * 0.90), 1)),
        axial_column_count,
    )
    cornea_strokes = []
    background_strokes = []
    through_plane = {
        "cornea": {int(column): [] for column in columns},
        "air": {int(column): [] for column in columns},
        "posterior": {int(column): [] for column in columns},
    }
    band_records = []

    for k in selected_k:
        image = array[k]
        detected_in_slice = 0
        cornea_line = []
        air_line = []
        posterior_line = []
        for i in columns:
            band = axial_band_for_column(image, i, percentiles)
            if band is None:
                continue
            top, bottom = band
            thickness = max(1, bottom - top + 1)
            if thickness < max(10, height * 0.025):
                continue

            detected_in_slice += 1
            cornea_j = int(round(top + thickness * cornea_band_position))
            gap = max(14, int(round(thickness * air_gap_scale)))
            air_j = int(round(top - gap))
            posterior_j = int(round(bottom + gap))

            if 0 <= cornea_j < height:
                point = [int(i), cornea_j, int(k)]
                cornea_line.append(point)
                through_plane["cornea"][int(i)].append(point)
            if air_j >= int(height * 0.04):
                point = [int(i), air_j, int(k)]
                air_line.append(point)
                through_plane["air"][int(i)].append(point)
            if posterior_j <= height - 1:
                point = [int(i), posterior_j, int(k)]
                posterior_line.append(point)
                through_plane["posterior"][int(i)].append(point)

        if len(cornea_line) >= 2:
            cornea_strokes.append(smooth_stroke(cornea_line, sort_axis=0, smooth_window=7))
        if len(air_line) >= 2:
            background_strokes.append(smooth_stroke(air_line, sort_axis=0, smooth_window=7))
        if len(posterior_line) >= 2:
            background_strokes.append(smooth_stroke(posterior_line, sort_axis=0, smooth_window=7))
        if detected_in_slice:
            band_records.append({"k": int(k), "sampled_columns": int(detected_in_slice)})

    through_columns = sample_evenly(through_plane["cornea"].keys(), through_plane_column_count)
    for column in through_columns:
        points = through_plane["cornea"][column]
        if len(points) >= 2:
            cornea_strokes.append(smooth_stroke(points, sort_axis=2, smooth_window=5))
    if background_through_plane:
        for label in ("air", "posterior"):
            for column in through_columns:
                points = through_plane[label][column]
                if len(points) >= 2:
                    background_strokes.append(smooth_stroke(points, sort_axis=2, smooth_window=5))

    return cornea_strokes, background_strokes, band_records


def generate_seed_spec(
    array,
    axial_slice_count=17,
    axial_column_count=49,
    through_plane_column_count=7,
    background_through_plane=False,
    air_gap_scale=0.32,
    cornea_band_position=0.52,
):
    shape_kji = array.shape
    dims_ijk = [shape_kji[2], shape_kji[1], shape_kji[0]]
    percentiles = robust_percentiles(array)
    cornea_strokes, background_strokes, band_records = semantic_seed_strokes(
        array,
        percentiles,
        dims_ijk,
        axial_slice_count=axial_slice_count,
        axial_column_count=axial_column_count,
        through_plane_column_count=through_plane_column_count,
        background_through_plane=background_through_plane,
        air_gap_scale=air_gap_scale,
        cornea_band_position=cornea_band_position,
    )

    if not cornea_strokes or not background_strokes:
        raise ValueError("Agent could not detect enough cornea/background paint candidates")

    background_radius = radius(dims_ijk, [70, 95, 58], [5, 5, 2])
    cornea_radius = radius(dims_ijk, [80, 110, 58], [4, 4, 2])
    cornea_control_points = sum(len(stroke) for stroke in cornea_strokes)
    background_control_points = sum(len(stroke) for stroke in background_strokes)

    spec = {
        "segments": [
            {
                "name": "background",
                "color": [0.05, 0.05, 0.05],
                "seeds": [],
                "strokes": [
                    {
                        "points_ijk": [clamp_ijk(point, dims_ijk) for point in stroke],
                        "radius_voxels": background_radius,
                    }
                    for stroke in background_strokes
                ],
            },
            {
                "name": "cornea",
                "color": [0.1, 0.7, 1.0],
                "seeds": [],
                "strokes": [
                    {
                        "points_ijk": [clamp_ijk(point, dims_ijk) for point in stroke],
                        "radius_voxels": cornea_radius,
                    }
                    for stroke in cornea_strokes
                ],
            },
        ]
    }
    qa = {
        "shape_kji": list(shape_kji),
        "dims_ijk": dims_ijk,
        "percentiles": percentiles,
        "agent_marking": {
            "background_seed_count": background_control_points,
            "cornea_seed_count": cornea_control_points,
            "background_stroke_count": len(background_strokes),
            "cornea_stroke_count": len(cornea_strokes),
            "painted_axial_slices": len(band_records),
            "strategy": "Detected the bright corneal band in axial B-scans column-by-column; painted connected cornea/background strokes in axial slices and through-plane stacks.",
            "parameters": {
                "axial_slice_count": axial_slice_count,
                "axial_column_count": axial_column_count,
                "through_plane_column_count": through_plane_column_count,
                "background_through_plane": background_through_plane,
                "air_gap_scale": air_gap_scale,
                "cornea_band_position": cornea_band_position,
            },
            "band_records": band_records,
        },
        "generated_seed_spec": spec,
    }
    return spec, qa


def main(argv):
    args = parse_args(argv)
    input_volume = resolve_launch_path(args.input_volume)
    output_seed_json = resolve_launch_path(args.output_seed_json)
    qa_json = resolve_launch_path(args.qa_json)
    preview_dir = resolve_launch_path(args.preview_dir)

    volume_node = load_input_volume(input_volume)
    array = slicer.util.arrayFromVolume(volume_node).astype(np.float32)
    spec, qa = generate_seed_spec(
        array,
        axial_slice_count=args.axial_slice_count,
        axial_column_count=args.axial_column_count,
        through_plane_column_count=args.through_plane_column_count,
        background_through_plane=args.background_through_plane,
        air_gap_scale=args.air_gap_scale,
        cornea_band_position=args.cornea_band_position,
    )

    ensure_parent_dir(output_seed_json)
    with open(output_seed_json, "w", encoding="utf-8") as fp:
        json.dump(spec, fp, indent=2)
    print(f"Saved seed JSON: {output_seed_json}")

    if qa_json:
        ensure_parent_dir(qa_json)
        with open(qa_json, "w", encoding="utf-8") as fp:
            json.dump(qa, fp, indent=2)
        print(f"Saved auto-seed QA: {qa_json}")

    if preview_dir:
        masks = seed_masks_from_spec(array.shape, spec)
        saved = save_previews(array, masks, preview_dir, "seeds", volume_node.GetSpacing())
        print(f"Saved seed previews: {saved}")


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
