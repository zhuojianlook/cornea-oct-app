# Slicer Seeded Grow Bridge

This is a first-cut bridge for agent-assisted cornea/scar segmentation in 3D Slicer.
It creates three seed segments, paints seed ellipsoids by IJK coordinates, runs
Slicer's `Grow from seeds` effect, then saves the segmentation and QA metadata.

Run the synthetic smoke test:

```bash
./slicer_bridge/run_seeded_grow.sh \
  --self-test \
  --output-seg output/self_test.seg.nrrd \
  --qa-json output/self_test_qa.json \
  --scene output/self_test.mrb
```

Run on a real corrected OCT volume:

```bash
./slicer_bridge/run_seeded_grow.sh \
  --input-volume /path/to/corrected_oct.nrrd \
  --seed-json slicer_bridge/seed_template.json \
  --output-seg output/case_001.seg.nrrd \
  --qa-json output/case_001_qa.json \
  --scene output/case_001.mrb
```

Seed coordinates are in Slicer IJK order: `[i, j, k]`. The correction loop is
currently file-based: add or adjust seed ellipsoids in the JSON file, rerun the
script, and review the saved `.mrb` or `.seg.nrrd` in Slicer.

Start the live bridge in the Slicer GUI:

```bash
/home/zhuojian/Applications/Slicer-5.10.0-linux-amd64/Slicer \
  --python-script slicer_bridge/live_bridge.py --self-test
```

Then commands can be sent through Slicer's local WebServer:

```bash
curl -X POST localhost:2016/slicer/exec --data \
  "slicer.agent.add_seed('scar', [58, 44, 43], [3, 3, 2]); __execResult = slicer.agent.preview()"
```

Useful live commands:

```python
slicer.agent.load_case("/path/to/corrected_oct.nrrd", "slicer_bridge/seed_template.json")
slicer.agent.add_seed("scar", [145, 118, 63], [4, 4, 2])
slicer.agent.erase_seed("cornea", [140, 118, 63], [3, 3, 1])
slicer.agent.preview()
slicer.agent.apply()
slicer.agent.capture_three_views("output/case_001_views")
slicer.agent.save("output/case_001.seg.nrrd", "output/case_001_qa.json", "output/case_001.mrb")
```
