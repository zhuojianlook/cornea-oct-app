# Legacy code — retained for reference, NOT used by either app

Nothing in this directory is imported, executed, or packaged. It is kept so the
original numerical pipeline stays inspectable (and citable) after the production
code moved on. Do not add it to the import path.

## `oct_preprocess_streamlit_port.py`

The original headless port of the `OCT_Extraction` preprocessing pipeline, taken
from the user's Streamlit scripts:

1. read `.OCT` (`oct_converter` POCT) → raw B-scan z-stack
2. `oct_to_dicom` (`DICOMGeneratorlossless.py`) → uint16 multi-frame DICOM + geometry
3. `smooth_volume` (`DICOMSmootherSteps.py`) → corneal-edge + column correction,
   3-D active correction across slices

760 lines. Byte-identical to the last committed revision of the file at its old
path `cornea_app/python-sidecar/oct_preprocess.py` (blob `936f729`, branch
`cornea-wip`, commit `1919ad9`).

### What replaced it

The production detector now lives at the same path,
`cornea_app/python-sidecar/oct_preprocess.py` (~5,608 lines), and shares only the
filename. The substantive differences:

- **Dynamic-programming corneal-surface detector** is the default, replacing the
  gradient + RANSAC-quadratic edge fit here. Auto-tuned per scan.
- **Rigid-only B-scan constraint.** An axial B-scan is captured near-instantaneously,
  so its internal geometry is truth. Production applies a rigid per-frame depth
  shift/rotation and never per-column-deforms a B-scan. The `smooth_volume`
  column-displacement warp in this file violates that invariant.
- **Inter-frame motion correction** (`axial_motion_correct`), height refinement
  (`rigid_height_refine`) and de-rotation (`rigid_frame_derotate`) — none of which
  exist here.
- Clipped-apex surface-crop reconstruction, de-tilt, auto crop-region, and the
  iterate-to-convergence loop with per-pass quality metrics.

`smooth_volume` here is still the faithful reference for the *original* behaviour,
including its three documented deviations from the Streamlit source (fixed
`read_oct_volume()[0].volume` read contract, O(N) cached-edge 3-D active
correction, and `corr_factor` scaling the column displacement).
