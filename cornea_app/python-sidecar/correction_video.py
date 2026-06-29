"""#10 — export a scan's preprocessing CORRECTION as an MP4 grid video.

Grid: ROWS = orientation (axial / coronal / sagittal), COLUMNS = the correction passes laid out
"after (final) ← … passes … → before (raw)" (per the request: after on the left, before on the right,
intermediate iterative passes between). Each video frame scrubs one slice (all rows advance together
through their normalised slice fraction), so the whole volume's correction is reviewable as a movie.

Source = the already-rendered grayscale preview PNGs (context_raw = before, context_iter{k} = each pass,
context = the final/after), so the video matches exactly what the viewer shows. Encoded H.264 / yuv420p
via imageio's bundled ffmpeg, so it plays in any browser/player. Pure CPU, read-only on the case data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2
import imageio.v2 as imageio

import orchestration as orch

_ORIENTS = ["axial", "coronal", "sagittal"]
_CELL_W, _CELL_H = 300, 220          # per-cell letterbox size
_LABEL_COL = 70                      # left gutter for the orientation labels
_HEADER_H = 26                       # top strip for the column labels
_FPS = 15
_MAX_FRAMES = 200                    # cap so a 513-slice scrub stays a reasonable-length clip


def _group_images(case_id: str, group: str) -> dict[str, list[tuple[int, str]]]:
    """{orientation: [(slice_index, png_path), …]} for a preview group, sorted by slice index. Empty
    when the group doesn't exist (e.g. context_iter{k} on a single-pass scan)."""
    from api_server import _preview_group_dir   # local import avoids a circular import at module load
    out: dict[str, list[tuple[int, str]]] = {o: [] for o in _ORIENTS}
    for im in orch.preview_images_from_dir(group, _preview_group_dir(case_id, group)):
        o = im.get("orientation"); si = im.get("slice_index"); p = im.get("path")
        if o in out and si is not None and p:
            out[o].append((int(si), str(p)))
    for o in out:
        out[o].sort(key=lambda t: t[0])
    return out


def _columns(case_id: str, manifest: dict) -> list[tuple[str, str]]:
    """Ordered (group, label) columns: after (final) → iterative passes → before (raw). Only groups that
    actually have previews are included (so a single-pass scan is just after | before)."""
    passes = int(((manifest.get("oct_iter") or {}).get("passes") or 0) or 0)
    cols: list[tuple[str, str]] = [("context", "after (final)")]
    for k in range(1, passes + 1):
        cols.append((f"context_iter{k}", f"pass {k}"))
    cols.append(("context_raw", "before (raw)"))
    # keep only groups that have any preview image
    return [(grp, lab) for grp, lab in cols
            if any(_group_images(case_id, grp)[o] for o in _ORIENTS)]


def _letterbox(path: str | None, w: int, h: int) -> np.ndarray:
    """Read a grayscale PNG and fit it into a w×h BGR cell preserving aspect (black padding)."""
    cell = np.zeros((h, w, 3), np.uint8)
    if not path:
        return cell
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return cell
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    rs = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_NEAREST)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    cell[y0:y0 + nh, x0:x0 + nw] = cv2.cvtColor(rs, cv2.COLOR_GRAY2BGR)
    return cell


def _pick(lst: list[tuple[int, str]], frac: float) -> str | None:
    """The slice path at the given 0..1 fraction through a group/orientation's sorted slices."""
    if not lst:
        return None
    j = int(round(frac * (len(lst) - 1)))
    return lst[max(0, min(len(lst) - 1, j))][1]


def export_correction_mp4(case_id: str, out_path: Path) -> dict:
    """Build the grid MP4 for `case_id` at `out_path`. Returns {out, frames, columns, orientations}.
    Raises ValueError if there are no correction previews to render."""
    manifest = orch.read_manifest(case_id)
    cols = _columns(case_id, manifest)
    if not cols:
        raise ValueError("No preprocessing previews to export — preprocess the scan first.")
    # cache each column's per-orientation slice lists
    col_imgs = {grp: _group_images(case_id, grp) for grp, _ in cols}
    rows = [o for o in _ORIENTS if any(col_imgs[grp][o] for grp, _ in cols)]
    if not rows:
        raise ValueError("No slices to render.")

    # frame count = the densest orientation across columns (capped)
    n_slices = max((len(col_imgs[grp][o]) for grp, _ in cols for o in rows), default=0)
    frames = max(1, min(_MAX_FRAMES, n_slices))

    n_cols, n_rows = len(cols), len(rows)
    grid_w = _LABEL_COL + n_cols * _CELL_W
    grid_h = _HEADER_H + n_rows * _CELL_H
    grid_w += grid_w % 2; grid_h += grid_h % 2          # even dims for yuv420p

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=_FPS, codec="libx264",
                                format="FFMPEG", pixelformat="yuv420p", macro_block_size=None,
                                output_params=["-crf", "20", "-preset", "medium"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    try:
        for t in range(frames):
            frac = 0.0 if frames == 1 else t / (frames - 1)
            canvas = np.zeros((grid_h, grid_w, 3), np.uint8)
            # column headers
            for ci, (_grp, lab) in enumerate(cols):
                x = _LABEL_COL + ci * _CELL_W + 8
                cv2.putText(canvas, lab, (x, 18), font, 0.5, (210, 210, 210), 1, cv2.LINE_AA)
            for ri, o in enumerate(rows):
                y = _HEADER_H + ri * _CELL_H
                cv2.putText(canvas, o, (6, y + _CELL_H // 2), font, 0.45, (140, 200, 255), 1, cv2.LINE_AA)
                for ci, (grp, _lab) in enumerate(cols):
                    cell = _letterbox(_pick(col_imgs[grp][o], frac), _CELL_W - 4, _CELL_H - 4)
                    x = _LABEL_COL + ci * _CELL_W + 2
                    canvas[y + 2:y + 2 + cell.shape[0], x:x + cell.shape[1]] = cell
            # slice readout
            cv2.putText(canvas, f"slice {t + 1}/{frames}", (grid_w - 150, 18), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
            writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    finally:
        writer.close()
    return {"out": str(out_path), "frames": frames,
            "columns": [l for _g, l in cols], "orientations": rows}
