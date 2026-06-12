# Cornea OCT Scar Quantification

A desktop app that turns a 3D corneal OCT volume into **objective scar metrics**
(volume, en-face area, density) and a **3-class voxel labelmap** (`0=background,
1=cornea, 2=scar`) — the research deliverable, and the eventual training set for an
automatic model (nnU-Net).

The app lives in [`cornea_app/`](cornea_app/): a **React + TypeScript + Vite +
Tailwind + MUI** frontend with the **niivue** medical viewer (2D slice-gallery
fallback when WebGL2 is unavailable), a thin **Tauri v2** Rust shell, and a
**FastAPI Python sidecar**. Segmentation is **SAM2** (in-process, GPU); the only
3D Slicer dependency is DICOM → NIfTI conversion.

## Workflow (3 stages)

1. **Segment** — load a 3D OCT volume (NIfTI/NRRD; DICOM via Slicer), then
   **SAM2** segments the cornea by treating each of the axial / coronal / sagittal
   planes as a movie and fusing the three passes into one 3D cornea mask.
2. **Correct** — load the segmentation as an editable niivue drawing and fix the
   cornea boundary with the pen (`cornea=1`, `background=2` erases). Saved as the
   canonical corrected labelmap.
3. **Scar** — **Detect scar (auto)** flags the hyper-reflective stroma inside the
   cornea as scar *candidates* (a sensitivity slider controls how much), shown in
   density tiers; correct them with the scar pen (`3`). The corrected scar is
   quantified — **volume (mm³), en-face area (mm²), densitometry** — and
   **Export scar metrics** writes `output/scar_summary.csv` across all cases for
   outcome correlation.

**Export → nnU-Net** (sidebar) writes `output/nnunet/Dataset501_CorneaOCT/`
(`imagesTr/`, `labelsTr/`, `dataset.json`) from the corrected labelmaps — the
training set for automating this pipeline once enough patients are labelled.

## Running (browser-dev-first)

```bash
cd cornea_app
pip install -r python-sidecar/requirements.txt   # first time
npm install                                       # first time
./dev-launch.sh                                   # sidecar :8765 + Vite :1420
```

Then open <http://localhost:1420>. SAM2 needs the checkpoint at
`cornea_app/sam2_ckpt/sam2.1_hiera_small.pt` and a CUDA GPU.

Batch a cohort from the CLI: `python python-sidecar/process_cohort.py`
(ingest → SAM2 → scar/auto → summary; then correct each case in the app).

### Native window (optional, deferred)

The native Tauri v2 window needs WebKitGTK 4.1:

```bash
sudo apt install libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev patchelf
```

## Layout

- `cornea_app/src/` — React frontend (niivue viewer, Zustand `workflowStore`, API client)
- `cornea_app/python-sidecar/` — FastAPI sidecar: `sam2_segment` (cornea), `scar`
  (detection + quantification), `masks` (correction round-trip), `metrics_export`
  (scar_summary), `export` (nnU-Net), `postprocess` (in-process preview rendering)
- `slicer_bridge/` — `convert_to_nifti.py` (DICOM→NIfTI) + pure-numpy `preview_io.py`
- `cornea_app/sam2_ckpt/` — SAM2 checkpoint (downloaded, gitignored)

Label convention everywhere: `0=background, 1=cornea, 2=scar` (scar optional per case).
