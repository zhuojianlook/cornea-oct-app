# Cornea OCT Segmentation

A desktop app for producing **background / cornea / scar** voxel labels from 3D
OCT volumes, to train a UNet (e.g. nnU-Net) for the same segmentation.

The app lives in [`cornea_app/`](cornea_app/) and mirrors the architecture of the
multipanelfigure app: **React + TypeScript + Vite + Tailwind + MUI** frontend with
the **niivue** medical volume viewer, a thin **Tauri v2** Rust shell, and a
**FastAPI Python sidecar** that drives 3D Slicer and the vision models.

## Workflow

1. **Load** a 3D OCT volume (NRRD / NIfTI; DICOM via Slicer).
2. **Seed paint** — an AI (OpenAI / local OpenAI-compatible / MedGemma) or a
   deterministic heuristic paints cornea + background seeds. Edit them
   interactively in niivue with a **cornea / background / scar** pen.
3. **Verdict** — Accept, or Reject with notes → the model **repaints using your
   feedback** (active-learning loop).
4. **Grow from Seeds** — 3D Slicer grows the full segmentation; it overlays the
   volume with voxel/volume QA.
5. **Scar detection** (optional) — an AI or heuristic marks scar *inside* the
   cornea, re-grows into a 3-class labelmap, and reports metrics.
6. **Export → nnU-Net** — writes `output/nnunet/Dataset501_CorneaOCT/` with
   `imagesTr/`, `labelsTr/`, and `dataset.json`. Labels: `0=background,
   1=cornea, 2=scar` (scar optional per case).

## Running (browser-dev-first)

```bash
cd cornea_app
pip install -r python-sidecar/requirements.txt   # first time
npm install                                       # first time
./dev-launch.sh                                   # sidecar :8765 + Vite :1420
```

Then open <http://localhost:1420>. The app talks to the sidecar via the browser
fetch fallback — no native window required.

### Native window (optional, deferred)

The native Tauri v2 window needs WebKitGTK 4.1, which isn't installed here (only
4.0). Install it once, then `./dev-launch.sh --native`:

```bash
sudo apt install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev patchelf
```

## Layout

- `cornea_app/src/` — React frontend (niivue viewer, Zustand stores, API client)
- `cornea_app/python-sidecar/` — FastAPI sidecar (orchestration, Slicer runner,
  vision providers, volume/seed/scar/export modules)
- `cornea_app/src-tauri/` — thin Tauri v2 Rust shell (scaffolded; build deferred)
- `slicer_bridge/` — 3D Slicer scripts (rendering, Grow from Seeds, live bridge)
- `local_vision/` — local MedGemma vision bridge + 2D helpers

The 3D Slicer executable path and vision-provider settings are configurable in
the app's Settings panel.
