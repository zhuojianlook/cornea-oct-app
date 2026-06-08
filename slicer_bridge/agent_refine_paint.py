#!/usr/bin/env python3
"""Generate, score, and refine curved cornea/background paint proposals."""

import argparse
import json
import os
import shutil
import sys
import traceback

import numpy as np
import slicer

from auto_seed_volume import generate_seed_spec
from preview_io import save_previews, seed_masks_from_spec, slice_mask, sampled_indices_for_orientation
from slicer_volume_io import load_input_volume


LAUNCH_CWD = os.environ.get("PWD") or os.getcwd()


def resolve_launch_path(path):
    if not path or os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(LAUNCH_CWD, path))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Closed-loop agent paint refinement.")
    parser.add_argument("--input-volume", required=True)
    parser.add_argument("--output-seed-json", required=True)
    parser.add_argument("--qa-json")
    parser.add_argument("--preview-dir", required=True)
    parser.add_argument("--feedback-json")
    parser.add_argument("--keep-candidates", action="store_true")
    return parser.parse_args(argv)


def ensure_parent_dir(path):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def component_count(mask, stride=4, max_components=80):
    reduced = np.asarray(mask, dtype=bool)[::stride, ::stride]
    visited = np.zeros(reduced.shape, dtype=bool)
    height, width = reduced.shape
    components = 0
    for y in range(height):
        for x in range(width):
            if not reduced[y, x] or visited[y, x]:
                continue
            components += 1
            if components >= max_components:
                return components
            stack = [(y, x)]
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not reduced[ny, nx]:
                        continue
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return components


def slice_metrics(volume_shape, masks_by_name):
    items = []
    for orientation in ("axial", "coronal", "sagittal"):
        indices = sampled_indices_for_orientation(volume_shape, masks_by_name, orientation, 9)
        for index in indices:
            item = {"orientation": orientation, "slice_index": int(index), "segments": {}}
            for name, mask in masks_by_name.items():
                plane = slice_mask(mask, orientation, index) > 0
                item["segments"][name] = {
                    "pixels": int(np.count_nonzero(plane)),
                    "components": int(component_count(plane)),
                }
            items.append(item)
    return items


def score_metrics(metrics):
    score = 0.0
    penalties = []
    for item in metrics:
        orientation = item["orientation"]
        cornea = item["segments"].get("cornea", {})
        background = item["segments"].get("background", {})
        c_pixels = float(cornea.get("pixels", 0))
        b_pixels = float(background.get("pixels", 0))
        c_components = float(cornea.get("components", 0))
        b_components = float(background.get("components", 0))

        if orientation == "axial":
            if c_pixels < 900:
                penalty = (900 - c_pixels) / 180.0
                score -= penalty
                penalties.append(f"low axial cornea paint slice {item['slice_index']}: {c_pixels:.0f}")
            score -= max(0.0, c_components - 3.0) * 2.0
            score -= max(0.0, b_components - 7.0) * 1.2
            score -= max(0.0, b_pixels / max(c_pixels, 1.0) - 5.0) * 0.8
        else:
            score -= max(0.0, c_components - 8.0) * 2.5
            score -= max(0.0, b_components - 10.0) * 2.0
            score -= max(0.0, c_pixels - 2800.0) / 700.0
            score -= max(0.0, b_pixels - 3500.0) / 700.0
            if c_components > 16 or b_components > 18:
                penalties.append(
                    f"{orientation} clutter slice {item['slice_index']}: cornea components {c_components:.0f}, background components {b_components:.0f}"
                )
    return score, penalties


def feedback_rejected_candidates(feedback_json):
    if not feedback_json:
        return set()
    feedback_json = resolve_launch_path(feedback_json)
    if not os.path.exists(feedback_json):
        return set()
    with open(feedback_json, "r", encoding="utf-8") as fp:
        try:
            entries = json.load(fp)
        except json.JSONDecodeError:
            return set()
    if not isinstance(entries, list):
        return set()
    rejected = set()
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("decision") != "Rejected":
            continue
        candidate = entry.get("candidate")
        if isinstance(candidate, str) and candidate:
            rejected.add(candidate)
    return rejected


def candidate_configs():
    return [
        {
            "name": "axial_curves_only_dense",
            "axial_slice_count": 17,
            "axial_column_count": 49,
            "through_plane_column_count": 0,
            "background_through_plane": False,
            "air_gap_scale": 0.32,
            "cornea_band_position": 0.52,
        },
        {
            "name": "axial_curves_with_sparse_cornea_depth",
            "axial_slice_count": 17,
            "axial_column_count": 49,
            "through_plane_column_count": 3,
            "background_through_plane": False,
            "air_gap_scale": 0.32,
            "cornea_band_position": 0.52,
        },
        {
            "name": "axial_curves_medium",
            "axial_slice_count": 13,
            "axial_column_count": 41,
            "through_plane_column_count": 0,
            "background_through_plane": False,
            "air_gap_scale": 0.34,
            "cornea_band_position": 0.52,
        },
        {
            "name": "axial_curves_wider_background",
            "axial_slice_count": 17,
            "axial_column_count": 49,
            "through_plane_column_count": 0,
            "background_through_plane": False,
            "air_gap_scale": 0.42,
            "cornea_band_position": 0.52,
        },
        {
            "name": "sparse_cornea_depth_control",
            "axial_slice_count": 13,
            "axial_column_count": 33,
            "through_plane_column_count": 3,
            "background_through_plane": False,
            "air_gap_scale": 0.36,
            "cornea_band_position": 0.50,
        },
        {
            "name": "penalty_control_background_depth",
            "axial_slice_count": 17,
            "axial_column_count": 49,
            "through_plane_column_count": 5,
            "background_through_plane": True,
            "air_gap_scale": 0.32,
            "cornea_band_position": 0.52,
        },
    ]


def copy_best_previews(candidate_dir, preview_dir):
    os.makedirs(preview_dir, exist_ok=True)
    for file_name in os.listdir(preview_dir):
        path = os.path.join(preview_dir, file_name)
        if file_name.endswith(".png") or file_name == "preview_manifest.json":
            os.remove(path)
    for file_name in os.listdir(candidate_dir):
        if file_name.endswith(".png") or file_name == "preview_manifest.json":
            shutil.copy2(os.path.join(candidate_dir, file_name), os.path.join(preview_dir, file_name))


def main(argv):
    args = parse_args(argv)
    input_volume = resolve_launch_path(args.input_volume)
    output_seed_json = resolve_launch_path(args.output_seed_json)
    qa_json = resolve_launch_path(args.qa_json)
    preview_dir = resolve_launch_path(args.preview_dir)
    rejected_candidates = feedback_rejected_candidates(args.feedback_json)

    slicer.mrmlScene.Clear()
    volume_node = load_input_volume(input_volume)
    array = slicer.util.arrayFromVolume(volume_node).astype(np.float32)
    spacing = volume_node.GetSpacing()

    candidate_root = os.path.join(preview_dir, "_candidates")
    shutil.rmtree(candidate_root, ignore_errors=True)
    os.makedirs(candidate_root, exist_ok=True)

    results = []
    best = None
    for index, config in enumerate(candidate_configs(), start=1):
        config_dir = os.path.join(candidate_root, f"{index:02d}_{config['name']}")
        spec, qa = generate_seed_spec(array, **{key: value for key, value in config.items() if key != "name"})
        masks = seed_masks_from_spec(array.shape, spec)
        save_previews(array, masks, config_dir, "seeds", spacing)
        metrics = slice_metrics(array.shape, masks)
        raw_score, penalties = score_metrics(metrics)
        feedback_penalty = 1000.0 if config["name"] in rejected_candidates else 0.0
        score = raw_score - feedback_penalty
        if feedback_penalty:
            penalties = [*penalties, "candidate rejected by user feedback"]
        result = {
            "candidate": config["name"],
            "score": score,
            "raw_score": raw_score,
            "feedback_penalty": feedback_penalty,
            "parameters": {key: value for key, value in config.items() if key != "name"},
            "metrics": metrics,
            "penalties": penalties,
            "agent_marking": qa["agent_marking"],
            "preview_dir": config_dir,
        }
        results.append(result)
        if best is None or score > best["score"]:
            best = {**result, "spec": spec, "qa": qa}

    ensure_parent_dir(output_seed_json)
    with open(output_seed_json, "w", encoding="utf-8") as fp:
        json.dump(best["spec"], fp, indent=2)
    copy_best_previews(best["preview_dir"], preview_dir)

    final_qa = best["qa"]
    final_qa["agent_refinement"] = {
        "selected_candidate": best["candidate"],
        "selected_score": best["score"],
        "selected_raw_score": best["raw_score"],
        "selected_feedback_penalty": best["feedback_penalty"],
        "selected_parameters": best["parameters"],
        "selected_penalties": best["penalties"],
        "feedback_rejected_candidates": sorted(rejected_candidates),
        "candidate_scores": [
            {
                "candidate": result["candidate"],
                "score": result["score"],
                "raw_score": result["raw_score"],
                "feedback_penalty": result["feedback_penalty"],
                "penalty_count": len(result["penalties"]),
                "parameters": result["parameters"],
            }
            for result in results
        ],
    }
    if qa_json:
        ensure_parent_dir(qa_json)
        with open(qa_json, "w", encoding="utf-8") as fp:
            json.dump(final_qa, fp, indent=2)

    if not args.keep_candidates:
        shutil.rmtree(candidate_root, ignore_errors=True)

    print(json.dumps(final_qa["agent_refinement"], indent=2))


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
