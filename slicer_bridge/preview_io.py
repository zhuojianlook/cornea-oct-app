"""Small PNG preview helpers for Slicer bridge scripts.

These functions intentionally avoid GUI widgets so previews can be generated in
batch mode. Arrays are in Slicer/numpy KJI order unless a function says IJK.
"""

import os
import struct
import zlib

import numpy as np


SEGMENT_COLORS = {
    "background": (255, 145, 0),
    "cornea": (0, 190, 255),
    "scar_diffuse": (255, 180, 120),   # low-density scar (diffuse haze)
    "scar_mod": (255, 110, 80),        # moderate density
    "scar": (255, 30, 70),             # dense scar core
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def write_png_rgb(path, rgb, compress_level=6):
    """Write an RGB uint8 numpy array to a PNG file using only stdlib."""
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("write_png_rgb expects an HxWx3 uint8 array")
    height, width, _ = rgb.shape
    # PNG scanlines = a filter byte (0) prepended to each row. Build it vectorised (no Python
    # per-row loop, which held the GIL and dominated dense renders) so it parallelises across
    # threads and is simply much faster.
    scanlines = np.empty((height, 1 + width * 3), dtype=np.uint8)
    scanlines[:, 0] = 0
    scanlines[:, 1:] = np.ascontiguousarray(rgb).reshape(height, width * 3)
    raw = scanlines.tobytes()

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, compress_level))
        + chunk(b"IEND", b"")
    )
    with open(path, "wb") as fp:
        fp.write(png)


def resize_rgb_nearest(rgb, target_height, target_width):
    rgb = np.asarray(rgb, dtype=np.uint8)
    height, width, _ = rgb.shape
    target_height = max(1, int(target_height))
    target_width = max(1, int(target_width))
    if target_height == height and target_width == width:
        return rgb
    row_indices = np.linspace(0, height - 1, target_height).round().astype(int)
    col_indices = np.linspace(0, width - 1, target_width).round().astype(int)
    return rgb[row_indices][:, col_indices]


def normalize_gray(image):
    image = np.asarray(image, dtype=np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = lo + 1.0
    normalized = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def gray_to_rgb(image):
    gray = normalize_gray(image)
    return np.stack([gray, gray, gray], axis=-1)


def overlay_mask(rgb, mask, color, alpha=0.45):
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return rgb
    out = rgb.astype(np.float32)
    color_array = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + color_array * alpha
    return out.astype(np.uint8)


def paint_ellipsoid(mask_kji, ijk, radius_voxels):
    center_ijk = np.array(ijk, dtype=np.float64)
    radius_ijk = np.array(radius_voxels, dtype=np.float64)
    radius_ijk = np.maximum(radius_ijk, 1.0)

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
    radius = np.maximum(np.array(radius_voxels, dtype=np.float64), 1.0)
    distance = float(np.linalg.norm(end - start))
    step = max(1.0, float(np.min(radius)) * 0.45)
    sample_count = max(2, int(np.ceil(distance / step)) + 1)
    for t in np.linspace(0.0, 1.0, sample_count):
        paint_ellipsoid(mask_kji, start * (1.0 - t) + end * t, radius)


def paint_polyline(mask_kji, points_ijk, radius_voxels):
    points = [point for point in points_ijk if point is not None]
    if len(points) == 1:
        paint_ellipsoid(mask_kji, points[0], radius_voxels)
        return
    for start, end in zip(points[:-1], points[1:]):
        paint_line(mask_kji, start, end, radius_voxels)


def seed_masks_from_spec(shape_kji, seed_spec):
    masks = {}
    for segment in seed_spec.get("segments", []):
        name = segment.get("name")
        if not name:
            continue
        mask = np.zeros(shape_kji, dtype=np.uint8)
        for stroke in segment.get("strokes", []):
            points = stroke.get("points_ijk") or stroke.get("ijk_points") or []
            paint_polyline(mask, points, stroke.get("radius_voxels", [1, 1, 1]))
        for seed in segment.get("seeds", []):
            paint_ellipsoid(mask, seed.get("ijk", [0, 0, 0]), seed.get("radius_voxels", [1, 1, 1]))
        masks[name] = mask
    return masks


def center_slices(volume_shape_kji, masks_by_name):
    cornea = masks_by_name.get("cornea")
    if cornea is not None and np.count_nonzero(cornea):
        coords = np.argwhere(cornea > 0)
        center_kji = np.median(coords, axis=0).astype(int)
        return {
            "axial": int(center_kji[0]),
            "coronal": int(center_kji[1]),
            "sagittal": int(center_kji[2]),
        }
    return {
        "axial": volume_shape_kji[0] // 2,
        "coronal": volume_shape_kji[1] // 2,
        "sagittal": volume_shape_kji[2] // 2,
    }


def slice_volume(volume_array, orientation, index):
    if orientation == "axial":
        return volume_array[index, :, :]
    if orientation == "coronal":
        return volume_array[:, index, :]
    if orientation == "sagittal":
        return volume_array[:, :, index]
    raise ValueError(f"Unknown orientation: {orientation}")


def slice_mask(mask, orientation, index):
    return slice_volume(mask, orientation, index)


def orientation_axis(orientation):
    if orientation == "axial":
        return 0
    if orientation == "coronal":
        return 1
    if orientation == "sagittal":
        return 2
    raise ValueError(f"Unknown orientation: {orientation}")


def orientation_axes(orientation):
    if orientation == "axial":
        return {"fixed_axis": "k", "row_axis": "j", "column_axis": "i"}
    if orientation == "coronal":
        return {"fixed_axis": "j", "row_axis": "k", "column_axis": "i"}
    if orientation == "sagittal":
        return {"fixed_axis": "i", "row_axis": "k", "column_axis": "j"}
    raise ValueError(f"Unknown orientation: {orientation}")


def orientation_spacing(orientation, spacing_ijk):
    spacing_i, spacing_j, spacing_k = [float(value) for value in spacing_ijk]
    if orientation == "axial":
        return spacing_j, spacing_i
    if orientation == "coronal":
        return spacing_k, spacing_i
    if orientation == "sagittal":
        return spacing_k, spacing_j
    raise ValueError(f"Unknown orientation: {orientation}")


def scale_to_physical_aspect(rgb, orientation, spacing_ijk):
    row_spacing, column_spacing = orientation_spacing(orientation, spacing_ijk)
    smallest_spacing = max(min(row_spacing, column_spacing), 1e-9)
    target_height = round(rgb.shape[0] * row_spacing / smallest_spacing)
    target_width = round(rgb.shape[1] * column_spacing / smallest_spacing)

    max_dimension = 1400
    largest = max(target_height, target_width)
    if largest > max_dimension:
        scale = max_dimension / float(largest)
        target_height = round(target_height * scale)
        target_width = round(target_width * scale)

    return resize_rgb_nearest(rgb, target_height, target_width)


def save_manifest(output_dir, items):
    path = os.path.join(output_dir, "preview_manifest.json")
    with open(path, "w", encoding="utf-8") as fp:
        import json

        json.dump({"images": items}, fp, indent=2)


def cleanup_prefix(output_dir, prefix):
    if not os.path.isdir(output_dir):
        return
    for file_name in os.listdir(output_dir):
        if file_name.startswith(f"{prefix}_") and file_name.endswith(".png"):
            os.remove(os.path.join(output_dir, file_name))


def mask_for_slice_selection(masks_by_name):
    cornea = masks_by_name.get("cornea")
    if cornea is not None and np.count_nonzero(cornea):
        return cornea > 0
    union = None
    for mask in masks_by_name.values():
        if mask is None:
            continue
        current = mask > 0
        union = current if union is None else (union | current)
    return union


def paint_indices(mask, axis):
    if mask is None:
        return set()
    other_axes = tuple(candidate for candidate in range(3) if candidate != axis)
    counts = np.sum(mask > 0, axis=other_axes)
    return set(int(index) for index in np.argwhere(counts > 0).ravel())


def sample_from_candidates(candidates, max_slices):
    candidates = sorted(set(int(index) for index in candidates))
    if not candidates:
        return []
    count = max(1, min(int(max_slices), len(candidates)))
    if count == len(candidates):
        return candidates
    positions = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [candidates[int(position)] for position in sorted(set(positions))]


def sampled_indices_for_orientation(volume_shape_kji, masks_by_name, orientation, max_slices):
    axis = orientation_axis(orientation)
    cornea_indices = paint_indices(masks_by_name.get("cornea"), axis)
    background_indices = paint_indices(masks_by_name.get("background"), axis)
    both_indices = cornea_indices & background_indices
    sampled_both = sample_from_candidates(both_indices, max_slices)
    if sampled_both:
        return sampled_both

    selection_mask = mask_for_slice_selection(masks_by_name)
    if selection_mask is not None and np.count_nonzero(selection_mask):
        coords = np.argwhere(selection_mask)
        lower = int(coords[:, axis].min())
        upper = int(coords[:, axis].max())
    else:
        lower = 0
        upper = int(volume_shape_kji[axis] - 1)

    if upper <= lower:
        return [lower]

    count = max(1, min(int(max_slices), upper - lower + 1))
    indices = np.linspace(lower, upper, count).round().astype(int).tolist()
    return sorted(set(int(index) for index in indices))


def save_previews(volume_array, masks_by_name, output_dir, prefix, spacing_ijk=None,
                  max_slices_per_orientation=9, max_dim=None, rotate=None, compress_level=None):
    """`max_slices_per_orientation` may be an int (all orientations) OR a dict
    {orientation: count} for per-axis density (e.g. dense axial B-scans for OCT scrub).
    `max_dim` caps the longest PNG side (keeps payload small when rendering many slices).
    `rotate` is an optional {orientation: k} of 90° CCW turns (np.rot90 k; negative = CW)
    baked into the saved PNG so the gallery shows the slice the right way up (cornea on top).
    image_width/height in the manifest reflect the rotated PNG."""
    ensure_dir(output_dir)
    cleanup_prefix(output_dir, prefix)
    spacing_ijk = spacing_ijk or [1.0, 1.0, 1.0]
    # Dense renders (plain context scrubs OR dense overlay panels) are served lazily one slice
    # at a time, so trade a little file size for a much faster render (zlib level 1). Small
    # overlay sets (the 9-slice segmentation previews) keep level 6.
    if compress_level is None:
        compress_level = 1 if not masks_by_name else 6

    def _render_slice(orientation, stack_position, index):
        rgb = gray_to_rgb(slice_volume(volume_array, orientation, index))
        source_height, source_width = rgb.shape[:2]
        for segment_name in ("background", "cornea", "scar_diffuse", "scar_mod", "scar"):
            mask = masks_by_name.get(segment_name)
            if mask is None:
                continue
            if segment_name.startswith("scar"):
                alpha = 0.62          # paint scar tiers last (diffuse→dense), on top
            elif segment_name == "cornea":
                alpha = 0.56
            elif prefix == "seeds":
                alpha = 0.58
            else:
                alpha = 0.20
            rgb = overlay_mask(
                rgb,
                slice_mask(mask, orientation, index) > 0,
                SEGMENT_COLORS.get(segment_name, (255, 255, 255)),
                alpha=alpha,
            )
        rgb = scale_to_physical_aspect(rgb, orientation, spacing_ijk)
        if max_dim:
            h, w = rgb.shape[:2]
            longest = max(h, w)
            if longest > max_dim:
                s = max_dim / longest
                rgb = resize_rgb_nearest(rgb, round(h * s), round(w * s))
        path = os.path.join(output_dir, f"{prefix}_{orientation}_{index:04d}.png")
        final = np.flipud(rgb)
        k = (rotate or {}).get(orientation, 0)
        if k:
            final = np.ascontiguousarray(np.rot90(final, k))
        write_png_rgb(path, final, compress_level=compress_level)
        axes = orientation_axes(orientation)
        return path, {
            "file_name": os.path.basename(path),
            "prefix": prefix,
            "orientation": orientation,
            "slice_index": int(index),
            "stack_position": int(stack_position),
            "fixed_axis": axes["fixed_axis"],
            "row_axis": axes["row_axis"],
            "column_axis": axes["column_axis"],
            "source_width": int(source_width),
            "source_height": int(source_height),
            "image_width": int(final.shape[1]),
            "image_height": int(final.shape[0]),
            "rotate_k": int(k),   # 90° CCW turns baked into the PNG (so clicks can undo it)
            "spacing_ijk": [float(value) for value in spacing_ijk],
            "paint_pixels": {
                name: int(np.count_nonzero(slice_mask(mask, orientation, index) > 0))
                for name, mask in masks_by_name.items()
                if mask is not None
            },
        }

    tasks = []
    for orientation in ("axial", "coronal", "sagittal"):
        mx = (max_slices_per_orientation.get(orientation, 9)
              if isinstance(max_slices_per_orientation, dict) else max_slices_per_orientation)
        indices = sampled_indices_for_orientation(volume_array.shape, masks_by_name, orientation, mx)
        for stack_position, index in enumerate(indices):
            tasks.append((orientation, stack_position, index))

    # Render slices in parallel across threads. The per-slice work (numpy resize + zlib PNG
    # encode) releases the GIL, so threads scale well — and, unlike a fork/spawn pool, they
    # never touch the sidecar's CUDA/torch state, so this is safe to run in-process.
    results = [None] * len(tasks)
    if len(tasks) <= 1:
        for i, t in enumerate(tasks):
            results[i] = _render_slice(*t)
    else:
        import concurrent.futures
        workers = max(1, min(16, (os.cpu_count() or 2)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_render_slice, *t): i for i, t in enumerate(tasks)}
            for fut in concurrent.futures.as_completed(futs):
                results[futs[fut]] = fut.result()

    saved = [r[0] for r in results]
    manifest_items = [r[1] for r in results]
    save_manifest(output_dir, manifest_items)
    return saved
