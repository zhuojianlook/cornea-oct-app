#!/usr/bin/env python3
"""Start a live Slicer bridge for agent-assisted seeded segmentation.

Launch with normal Slicer so the GUI stays open:

    Slicer --python-script live_bridge.py --self-test

Then call commands through WebServer /slicer/exec, for example:

    curl -X POST localhost:2016/slicer/exec --data \
      "slicer.agent.add_seed('scar', [58, 44, 43], [3, 3, 2]); __execResult = slicer.agent.preview()"
"""

import argparse
import json
import os
import sys

import numpy as np

import qt
import slicer

BRIDGE_DIR = os.path.dirname(os.path.abspath(__file__))
if BRIDGE_DIR not in sys.path:
    sys.path.insert(0, BRIDGE_DIR)

from seeded_grow_from_seeds import (  # noqa: E402
    DEFAULT_SEGMENT_SPECS,
    create_self_test_volume,
    create_segmentation_from_seeds,
    load_seed_spec,
    load_volume,
    normalize_radius,
    paint_ellipsoid,
    resolve_launch_path,
    run_grow_from_seeds,
    save_outputs,
    segment_stats,
    set_source_volume,
)


SEGMENT_DEFAULTS = {
    "background": [0.05, 0.05, 0.05],
    "cornea": [0.1, 0.7, 1.0],
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Start a live Slicer segmentation agent bridge.")
    parser.add_argument("--input-volume")
    parser.add_argument("--seed-json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--port", type=int, default=2016)
    parser.add_argument("--seed-locality-factor", type=float, default=0.0)
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--quit-after-test", action="store_true")
    return parser.parse_args(argv)


def paint_ellipsoid_value(mask_kji, ijk, radius_voxels, value):
    before = int(np.count_nonzero(mask_kji))
    if value:
        paint_ellipsoid(mask_kji, ijk, radius_voxels)
    else:
        center_ijk = np.array(ijk, dtype=np.float64)
        radius_ijk = normalize_radius(radius_voxels)
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
        local = mask_kji[slices]
        local[distance <= 1.0] = 0
    after = int(np.count_nonzero(mask_kji))
    return {"before": before, "after": after}


class AgentSegmentationBridge:
    def __init__(self, seed_locality_factor=0.0):
        self.seed_locality_factor = seed_locality_factor
        self.volume_node = None
        self.segmentation_node = None
        self.segment_ids_by_name = {}
        self.segment_editor_widget = None
        self.segment_editor_node = None
        self.grow_effect = None

    def load_case(self, input_volume=None, seed_json=None, self_test=False):
        slicer.mrmlScene.Clear()
        self.volume_node = create_self_test_volume() if self_test else load_volume(resolve_launch_path(input_volume))
        seed_json = resolve_launch_path(seed_json)
        if seed_json:
            segment_specs = load_seed_spec(seed_json, False)
        elif self_test:
            segment_specs = DEFAULT_SEGMENT_SPECS
        else:
            segment_specs = [
                {"name": name, "color": color, "seeds": [{"ijk": [0, 0, 0], "radius_voxels": [1, 1, 1]}]}
                for name, color in SEGMENT_DEFAULTS.items()
            ]
        self.segmentation_node, self.segment_ids_by_name = create_segmentation_from_seeds(
            self.volume_node, segment_specs
        )
        self._show_loaded_nodes()
        return self.summary()

    def _show_loaded_nodes(self):
        if self.segmentation_node:
            self.segmentation_node.CreateClosedSurfaceRepresentation()
            display_node = self.segmentation_node.GetDisplayNode()
            if display_node:
                display_node.SetVisibility(True)
        layout_manager = slicer.app.layoutManager()
        if not layout_manager:
            return
        if self.volume_node:
            slicer.util.setSliceViewerLayers(background=self.volume_node)
        slicer.util.resetSliceViews()
        slicer.app.processEvents()

    def _ensure_editor(self):
        if self.segment_editor_widget:
            return
        self.segment_editor_widget = slicer.qMRMLSegmentEditorWidget()
        self.segment_editor_widget.setMRMLScene(slicer.mrmlScene)
        self.segment_editor_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
        self.segment_editor_widget.setMRMLSegmentEditorNode(self.segment_editor_node)
        self.segment_editor_widget.setSegmentationNode(self.segmentation_node)
        set_source_volume(self.segment_editor_widget, self.volume_node)
        self.segment_editor_widget.setActiveEffectByName("Grow from seeds")
        self.grow_effect = self.segment_editor_widget.activeEffect()
        if self.grow_effect is None:
            raise RuntimeError("Could not activate Grow from seeds")
        self.grow_effect.setParameter("SeedLocalityFactor", str(self.seed_locality_factor))
        self.grow_effect.setParameter("AutoUpdate", "1")

    def add_seed(self, segment_name, ijk, radius_voxels=(3, 3, 1)):
        return self._edit_seed(segment_name, ijk, radius_voxels, value=1)

    def erase_seed(self, segment_name, ijk, radius_voxels=(3, 3, 1)):
        return self._edit_seed(segment_name, ijk, radius_voxels, value=0)

    def _edit_seed(self, segment_name, ijk, radius_voxels, value):
        if segment_name not in self.segment_ids_by_name:
            raise ValueError(f"Unknown segment: {segment_name}")
        segment_id = self.segment_ids_by_name[segment_name]
        mask = slicer.util.arrayFromSegmentBinaryLabelmap(
            self.segmentation_node, segment_id, self.volume_node
        ).copy()
        result = paint_ellipsoid_value(mask, ijk, radius_voxels, value)
        slicer.util.updateSegmentBinaryLabelmapFromArray(
            mask, self.segmentation_node, segment_id, self.volume_node
        )
        result["segment"] = segment_name
        result["ijk"] = list(ijk)
        result["radius_voxels"] = list(radius_voxels)
        return result

    def preview(self):
        self._ensure_editor()
        self.grow_effect.self().onPreview()
        slicer.app.processEvents()
        return self.summary()

    def apply(self):
        self._ensure_editor()
        self.grow_effect.self().onApply()
        slicer.app.processEvents()
        return self.summary()

    def run_apply_once(self):
        run_grow_from_seeds(self.volume_node, self.segmentation_node, self.seed_locality_factor)
        return self.summary()

    def save(self, output_seg, qa_json, scene=None):
        class Args:
            pass

        args = Args()
        args.output_seg = resolve_launch_path(output_seg)
        args.qa_json = resolve_launch_path(qa_json)
        args.scene = resolve_launch_path(scene)
        save_outputs(self.volume_node, self.segmentation_node, self.segment_ids_by_name, args)
        return {"output_seg": args.output_seg, "qa_json": args.qa_json, "scene": args.scene}

    def capture_three_views(self, output_dir):
        output_dir = resolve_launch_path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        layout_manager = slicer.app.layoutManager()
        if not layout_manager:
            raise RuntimeError("capture_three_views requires Slicer to run with the GUI visible")
        layout_manager.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)
        self._show_loaded_nodes()
        slicer.util.forceRenderAllViews()
        slicer.app.processEvents()

        saved = {}
        for view_name in ("Red", "Yellow", "Green"):
            widget = layout_manager.sliceWidget(view_name)
            path = os.path.join(output_dir, f"{view_name.lower()}.png")
            widget.sliceView().grab().save(path)
            saved[view_name.lower()] = path

        three_d_widget = layout_manager.threeDWidget(0)
        if three_d_widget:
            path = os.path.join(output_dir, "three_d.png")
            three_d_widget.threeDView().grab().save(path)
            saved["three_d"] = path
        return saved

    def summary(self):
        if not self.volume_node or not self.segmentation_node:
            return {"loaded": False}
        return {
            "loaded": True,
            "volume": self.volume_node.GetName(),
            "segments": segment_stats(
                self.volume_node, self.segmentation_node, self.segment_ids_by_name
            ),
        }


def start_webserver(port):
    from WebServer import WebServerLogic

    logic = WebServerLogic(
        port=port,
        enableSlicer=True,
        enableExec=True,
        enableDICOM=False,
        enableStaticPages=False,
        enableCORS=False,
    )
    logic.start()
    slicer.agentWebServerLogic = logic
    return logic.port


def exit_slicer(status):
    if hasattr(slicer.util, "exit"):
        slicer.util.exit(status)
    else:
        slicer.app.exit(status)


def main(argv):
    args = parse_args(argv)
    slicer.agent = AgentSegmentationBridge(seed_locality_factor=args.seed_locality_factor)
    if args.input_volume or args.self_test:
        slicer.agent.load_case(args.input_volume, args.seed_json, self_test=args.self_test)
        if args.quit_after_test:
            slicer.agent.run_apply_once()
        else:
            slicer.agent.preview()
    if not args.no_server:
        port = start_webserver(args.port)
        print(f"Agent bridge ready: http://localhost:{port}/slicer/exec")
    print("Bridge object: slicer.agent")
    print("Example: slicer.agent.add_seed('cornea', [58, 44, 43], [3, 3, 2])")
    if args.quit_after_test:
        if args.input_volume or args.self_test:
            print(json.dumps(slicer.agent.summary(), indent=2))
        exit_slicer(0)


if __name__ == "__main__":
    main(sys.argv[1:])
