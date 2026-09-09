"""run_provenance — the run-provenance tree behind the "⚙ Steps" panel (Steps v2, 2026-09-06).

A READ of what the LAST preprocessing run recorded — manifest.oct_params (every reviewer input), manifest.oct_iter
(what the run did), border_cache/*.npz (the served / placed / posterior surfaces), input/_raw_border.nii.gz (raw)
and manifest.input_volume (corrected) — rendered as a graph of pipeline stages (spine) with the reviewer's inputs
branching in from the left and decisions marked applied/declined with the recorded reason. Nothing is re-run: the
only computation is `oct_preprocess.tissue_motion_move` (deterministic, ~1.6 s) so the per-frame move can be plotted.

Conventions (see the data map in the Steps v2 spec):
  * every border_cache surface is float32 (laterals, frames) in RAW-canvas rows (negatives allowed) except
    applied_move.npz (padded rows); corrected row = raw row + oct_iter.canvas_extend.pad (+ applied move);
  * raw NIfTI is (lateral, depth, frame); the worker's (frames, depth, laterals) = arr.transpose(2, 1, 0);
  * the app shows sagittal panels scaleX(-1): frame 0 on the RIGHT, display-left = HIGH frame indices. Every image
    here is drawn in that orientation (x axis inverted) and its caption says so;
  * oct_params keys are strings; posterior_edges may hold NaN (ABSENT sentinel >= depth-1) — masked before drawing.

Public API: build_run_graph, export_run_report, zip_report, layout_graph, render_diagram_svg, methods_paragraph.
No FastAPI here; never raises for missing data (a missing file degrades into a node status).
"""
from __future__ import annotations

import base64
import datetime as _dt
import html as _html
import io
import json
import math
import os
import re
import shutil
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import orchestration as orch
import oct_preprocess as oct_mod

try:  # matplotlib is the renderer; PIL is the fallback so the graph endpoint never 500s without it
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle  # noqa: F401
    _HAVE_MPL = True
except Exception:  # noqa: BLE001
    plt = None
    _HAVE_MPL = False

SCHEMA_VERSION = 1
GENERATOR = "run_provenance v1"

# ── layout constants (MIRROR of the frontend's runTreeLayout.ts — keep the two in step, M10) ───────────────────
NODE_W = 340      # fits "Automatic corneal-surface detection (DP)" on one line at the card font
NODE_H = 84       # chips row + a two-line title + a two-line summary
GAP = 28
COL_GAP = 40
FAR_X = 40                              # far-left column: a reviewer input that feeds ONLY other inputs (n11 → n02/n05)
INPUT_X = FAR_X + NODE_W + COL_GAP      # 420
SPINE_X = INPUT_X + NODE_W + COL_GAP    # 800
TOP = 16
# without a far column the diagram keeps a two-column footprint: inputs at FAR_X, spine at INPUT_X (as the frontend)

# ── fixed colours ──────────────────────────────────────────────────────────────────────────────────────────────
C_SERVED = "#ff4040"
C_BASELINE = "#5a96ff"
C_BOTTOM = "#ffa500"
C_PLACED = "#ff40ff"
C_TARGET = "#40dc60"
C_AUTO = "#ffffff"
C_BAND = (91 / 255, 192 / 255, 1.0, 0.25)
C_CROP = (1.0, 69 / 255, 58 / 255, 0.25)
C_SHIFT = "#ff4040"
C_TILT = "#5a96ff"

THUMB_W_IN, THUMB_H_IN = 4.8, 3.0
PLOT_W_IN, PLOT_H_IN = 4.8, 2.2
EXPORT_W_IN, EXPORT_H_IN = 6.4, 4.0

_GRAPH_CACHE: dict = {}
_GRAPH_CACHE_MAX = 8
_TM_CACHE: dict = {}


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# small utilities
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _json_safe(obj):
    """numpy → python, NaN/inf → None, Path → str; recursive."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _r(v, nd=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return round(f, nd)


def _fmt(v, nd=1) -> str:
    if v is None:
        return "?"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if not math.isfinite(f):
            return "?"
        return f"{f:.{nd}f}"
    if isinstance(v, (list, tuple)):
        return "..".join(_fmt(x, nd) for x in v) if len(v) == 2 else ", ".join(_fmt(x, nd) for x in v)
    return str(v)


def _flat_value(v):
    """numbers[].value must be a scalar, None or a list of scalars (L12): dicts are flattened to 'k: v; k: v' text
    (a dict with a 'text' field — e.g. difficult_reason — becomes that text); nested lists become 'a, b' text."""
    v = _json_safe(v)
    if isinstance(v, dict):
        if isinstance(v.get("text"), str):
            return v["text"]
        return "; ".join(f"{k}: {_flat_value(x)}" for k, x in v.items()) or "—"
    if isinstance(v, list):
        if any(isinstance(x, (dict, list)) for x in v):
            return [str(_flat_value(x)) if isinstance(x, dict) else (", ".join(str(_flat_value(y)) for y in x) if isinstance(x, list) else x) for x in v]
        return v
    return v


def _n(label, value, unit=None) -> dict:
    d = {"label": str(label), "value": _flat_value(value)}
    if unit:
        d["unit"] = unit
    return d


def _int_keys(d) -> dict:
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        try:
            out[int(k)] = v
        except (TypeError, ValueError):
            continue
    return out


def _anchor_points(anchors) -> tuple[dict, int]:
    """{int lateral: {int frame: float row}} + total points, from a str-keyed oct_params anchor dict."""
    out: dict = {}
    n = 0
    for lat, fr in _int_keys(anchors).items():
        if not isinstance(fr, dict):
            continue
        inner = {}
        for f, r in fr.items():
            try:
                fi = int(f); fv = float(r)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fv):
                continue
            inner[fi] = fv
        if inner:
            out[lat] = inner
            n += len(inner)
    return out, n


def _anchors_sig(anchors) -> str:
    """Canonical anchor signature — same string api_server._border_anchors_sig writes into redetect/generalize/
    guided.npz (reimplemented here so this module never imports api_server). Malformed slices / frames / rows are
    skipped, never raised (L16: a bad manifest must degrade the graph, not 500 the endpoint)."""
    if not isinstance(anchors, dict):
        return ""
    slices = []
    for s, fr in anchors.items():
        try:
            slices.append((int(s), fr))
        except (TypeError, ValueError):
            continue
    parts = []
    for s, fr in sorted(slices, key=lambda x: x[0]):
        if not isinstance(fr, dict):
            continue
        cells = []
        for f, r in fr.items():
            try:
                fi = int(f); rv = float(r)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(rv):
                continue
            cells.append((fi, int(round(rv))))
        if cells:
            parts.append(f"{s}:" + ",".join(f"{fi}={ri}" for fi, ri in sorted(cells)))
    return ";".join(parts)


def _runs(xs) -> str:
    """[0,1,2,5,6] → '0–2, 5–6'."""
    xs = sorted({int(x) for x in xs})
    if not xs:
        return "—"
    out, s0, prev = [], xs[0], xs[0]
    for c in xs[1:] + [None]:
        if c is None or c != prev + 1:
            out.append(f"{s0}–{prev}" if prev > s0 else f"{s0}")
            s0 = c
        prev = c if c is not None else prev
    return ", ".join(out)


def _iso(ts) -> str | None:
    try:
        return _dt.datetime.fromtimestamp(float(ts)).replace(microsecond=0).isoformat()
    except Exception:  # noqa: BLE001
        return None


_CORR_FLATTEN_MODES = ("tissue", "drawn+minimax", "minimax")


def _mode(it, p=None, root=None) -> str:
    """'corrections' | 'automatic' | 'none'. A corrections run is recognised by ANY of: oct_iter.stopped == 'redetect',
    final_qa.path == 'redetect', flatten.mode in {tissue, drawn+minimax, minimax} (M3), or — when the case root is
    given — border_cache/provided_edges.npz present together with border_anchors on record."""
    if not isinstance(it, dict) or not it:
        return "none"
    fl = it.get("flatten") if isinstance(it.get("flatten"), dict) else {}
    fq = it.get("final_qa") if isinstance(it.get("final_qa"), dict) else {}
    if str(it.get("stopped") or "") == "redetect" or str(fq.get("path") or "") == "redetect" \
            or str(fl.get("mode") or "") in _CORR_FLATTEN_MODES:
        return "corrections"
    if root is not None and isinstance(p, dict) and isinstance(p.get("border_anchors"), dict) and p.get("border_anchors"):
        try:
            if (Path(root) / "border_cache" / "provided_edges.npz").exists():
                return "corrections"
        except OSError:
            pass
    return "automatic"


def _flatten_kind(it) -> str:
    """'tissue' (tissue-measured move), 'drawn' (drawn-line move: 'drawn+minimax' / 'minimax') or '' (no record)."""
    fl = it.get("flatten") if isinstance(it, dict) and isinstance(it.get("flatten"), dict) else {}
    mode = str(fl.get("mode") or "")
    if mode == "tissue":
        return "tissue"
    if mode in ("drawn+minimax", "minimax"):
        return "drawn"
    return ""


_FLATTEN_LABEL = {"tissue": "tissue-measured move", "drawn": "drawn-line move", "": "move source not recorded"}

# second-person UI phrases → third person (M7). Applied to every method / description / summary / status reason.
_SECOND_PERSON = [
    (r"\bthe edge you drew\b", "the reviewer-drawn edge"), (r"\byou drew\b", "the reviewer drew"),
    (r"\byour call\b", "the reviewer's decision"), (r"\byou\b", "the reviewer"), (r"\byours\b", "the reviewer's"),
    (r"\byour\b", "the reviewer's"),
]
_CASE_TOKEN = re.compile(r"\b(?:case_)?cs\d{3}(?:_[a-z0-9]+)*\b", re.I)
_STEP_HINT = re.compile(r"\s+in step \d\b", re.I)


def _neutral(text, own_case: str = "") -> str:
    """Neutral, third-person wording for text that may come from UI-facing record fields: second person removed,
    foreign case names replaced by 'another scan' (the case's own id is kept), UI step hints dropped."""
    s = str(text or "")
    if not s:
        return s
    for pat, rep in _SECOND_PERSON:
        s = re.sub(pat, rep, s, flags=re.I)
    own = (own_case or "").lower()
    s = _CASE_TOKEN.sub(lambda m: m.group(0) if own and m.group(0).lower().lstrip("case_") in own else "another scan", s)
    s = _STEP_HINT.sub("", s)
    return s.strip()


_REASON_PREFIX = re.compile(r"^\s*(?:declined|skipped|waived|not applied)\s*(?:\([^)]*\))?\s*[:—-]\s*", re.I)
_REASON_CUT = (" — ", " -- ", "; this ", "; set ", ", so it is", " so it is off", "; see ", " (see ")
_REASON_UI = re.compile(r"(?:^|\.\s+)(?:set|open|run|use|click|press|try|re-run)\b[^.]*\.?", re.I)


def _reason_clause(text, own_case: str = "") -> str:
    """A recorded reason/why field reduced to its FACTUAL clause for a Methods sentence: status prefixes
    ('declined:', 'skipped (…):') dropped, UI advice ('— set X per case to compare', '; this is the … guard') cut,
    imperative UI sentences removed, third person (M7). Empty input → 'not recorded'."""
    sraw = str(text or "").strip()
    if not sraw:
        return "not recorded"
    sraw = _REASON_PREFIX.sub("", sraw)
    for cut in _REASON_CUT:
        i = sraw.find(cut)
        if i > 0:
            sraw = sraw[:i]
    sraw = _REASON_UI.sub("", sraw)
    sraw = _neutral(sraw, own_case).rstrip(" .;:,")
    return sraw or "not recorded"


DPI_MIN, DPI_MAX = 72, 600


def clamp_dpi(dpi) -> int:
    """Export dpi clamped to 72..600 (L17); None / garbage → 300."""
    try:
        d = int(float(dpi))
    except (TypeError, ValueError):
        d = 300
    return int(min(DPI_MAX, max(DPI_MIN, d)))


def _load_npz(path: Path):
    try:
        if not path.exists():
            return None
        z = np.load(str(path), allow_pickle=False)
        out = {}
        for k in z.files:
            v = z[k]
            out[k] = v if v.ndim else v.item()
        return out
    except Exception:  # noqa: BLE001
        return None


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# run context
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
_CACHE_NAMES = ("baseline", "redetect", "generalize", "guided", "provided_edges", "placed_edges",
                "posterior_edges", "applied_move")


@dataclass
class RunCtx:
    cid: str
    root: Path
    m: dict
    p: dict
    it: dict
    mode: str
    raw_path: Path | None
    cor_path: Path | None
    spacing: list
    pad: int
    caches: dict = field(default_factory=dict)
    cache_mtimes: dict = field(default_factory=dict)
    bc_listing: list = field(default_factory=list)   # (name, mtime_ns, size) of EVERY border_cache/*.npz (M5)
    _raw: object = None
    _cor: object = None
    _raw_tried: bool = False
    _cor_tried: bool = False
    _tm: object = None
    _tm_tried: bool = False
    L: int | None = None
    D: int | None = None
    F: int | None = None
    run_time: str | None = None

    # lazy volumes ------------------------------------------------------------------------------------------
    @property
    def raw(self):
        if not self._raw_tried:
            self._raw_tried = True
            self._raw = _load_vol(self.raw_path)
        return self._raw

    @property
    def cor(self):
        if not self._cor_tried:
            self._cor_tried = True
            self._cor = _load_vol(self.cor_path)
        return self._cor

    def cache(self, name: str):
        return self.caches.get(name)

    def surf(self, name: str):
        c = self.caches.get(name)
        if c is None:
            return None
        s = c.get("surface") if "surface" in c else c.get("move")
        if s is None:
            return None
        s = np.asarray(s, dtype=np.float64)
        return s if s.ndim == 2 else None

    @property
    def anchors(self) -> dict:
        return self.p.get("border_anchors") if isinstance(self.p.get("border_anchors"), dict) else {}

    @property
    def crop_frames(self) -> list[int]:
        cf = self.p.get("surface_crop_frames") or []
        out = []
        for f in cf:
            try:
                out.append(int(f))
            except (TypeError, ValueError):
                continue
        return sorted(set(out))

    def served_for_flatten(self) -> tuple[object, str]:
        """placed_edges supersedes provided_edges only when its mtime >= provided's (the corrected pane's rule)."""
        pl, pr = self.surf("placed_edges"), self.surf("provided_edges")
        if pl is not None and (pr is None or self.cache_mtimes.get("placed_edges", 0) >= self.cache_mtimes.get("provided_edges", 0)):
            return pl, "placed_edges"
        if pr is not None:
            return pr, "provided_edges"
        return None, ""

    def raw_mtime(self) -> float | None:
        try:
            return os.path.getmtime(self.raw_path) if self.raw_path else None
        except OSError:
            return None


def _load_vol(path: Path | None):
    if not path or not Path(path).exists():
        return None
    try:
        import nibabel as nib
        return np.asanyarray(nib.load(str(path)).dataobj)
    except Exception:  # noqa: BLE001
        return None


def _ctx(case_id: str) -> RunCtx:
    cid = orch.safe_case_id(case_id)
    root = orch.case_root(cid)
    m = orch.read_manifest(cid)
    op = m.get("oct_params") if isinstance(m.get("oct_params"), dict) else {}
    p = {**oct_mod.DEFAULT_PARAMS, **op}
    it = m.get("oct_iter") if isinstance(m.get("oct_iter"), dict) else {}
    mode = _mode(it, op, root)
    raw_path = root / "input" / "_raw_border.nii.gz"
    if not raw_path.exists():
        raw_path = None
    cor = m.get("input_volume") or m.get("corrected_volume")
    cor_path = Path(cor) if cor and Path(cor).exists() else None
    ce = it.get("canvas_extend") if isinstance(it.get("canvas_extend"), dict) else {}
    try:
        pad = int(ce.get("pad") or 0)
    except (TypeError, ValueError):
        pad = 0
    spacing = m.get("oct_spacing") if isinstance(m.get("oct_spacing"), list) else []
    ctx = RunCtx(cid=cid, root=root, m=m, p=p, it=it, mode=mode, raw_path=raw_path, cor_path=cor_path,
                 spacing=[_r(s, 6) for s in spacing], pad=pad)
    bc = root / "border_cache"
    for name in _CACHE_NAMES:
        fp = bc / f"{name}.npz"
        ctx.caches[name] = _load_npz(fp)
        try:
            ctx.cache_mtimes[name] = os.path.getmtime(fp) if fp.exists() else 0.0
        except OSError:
            ctx.cache_mtimes[name] = 0.0
    try:   # every cache file, not just provided_edges: the graph cache key must see redetect/guided/placed changes (M5)
        for fp in sorted(bc.glob("*.npz")) if bc.is_dir() else []:
            st = fp.stat()
            ctx.bc_listing.append((fp.name, int(st.st_mtime_ns), int(st.st_size)))
    except OSError:
        pass
    # dims: raw header first (no data load), then a surface, then the corrected volume
    L = D = F = None
    try:
        if raw_path:
            import nibabel as nib
            shp = nib.load(str(raw_path)).shape
            L, D, F = int(shp[0]), int(shp[1]), int(shp[2])
    except Exception:  # noqa: BLE001
        pass
    if L is None:
        for name in ("provided_edges", "baseline", "placed_edges", "posterior_edges"):
            s = ctx.surf(name)
            if s is not None:
                L, F = int(s.shape[0]), int(s.shape[1])
                break
    if D is None:
        try:
            D = int(ce.get("depth_before")) if ce.get("depth_before") else None
        except (TypeError, ValueError):
            D = None
    if (L is None or D is None) and cor_path is not None:
        try:
            import nibabel as nib
            shp = nib.load(str(cor_path)).shape
            L = L if L is not None else int(shp[0])
            F = F if F is not None else int(shp[2])
            D = D if D is not None else int(shp[1]) - pad
        except Exception:  # noqa: BLE001
            pass
    ctx.L, ctx.D, ctx.F = L, D, F
    # run time: the manifest has no timestamp → mtime of the corrected volume, else provided_edges.npz
    rt = None
    for cand in (cor_path, bc / "provided_edges.npz"):
        try:
            if cand and Path(cand).exists():
                rt = _iso(os.path.getmtime(cand))
                break
        except OSError:
            continue
    ctx.run_time = rt
    return ctx


def _tissue_motion_arrays(ctx: RunCtx):
    """(a[F], b[F], info) recomputed from the raw volume by oct_preprocess.tissue_motion_move — deterministic, so it
    reproduces the run's record; None when there is no raw volume. Cached per (raw path, mtime, params)."""
    if ctx._tm_tried:
        return ctx._tm
    ctx._tm_tried = True
    if ctx.raw_path is None:
        return None
    keys = sorted(k for k in ctx.p if k.startswith("tissue_motion"))
    key = (str(ctx.raw_path), ctx.raw_mtime(), json.dumps({k: ctx.p[k] for k in keys}, sort_keys=True, default=str))
    hit = _TM_CACHE.get(key)
    if hit is not None:
        ctx._tm = hit
        return hit
    raw = ctx.raw
    if raw is None:
        return None
    try:
        a, b, info = oct_mod.tissue_motion_move(np.asarray(raw).transpose(2, 1, 0), ctx.p)
        res = (np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), dict(info))
    except Exception as exc:  # noqa: BLE001
        res = (None, None, {"applied": False, "error": type(exc).__name__})
    _TM_CACHE.clear()
    _TM_CACHE[key] = res
    ctx._tm = res
    return res


def _band_guide_arrays(ctx: RunCtx):
    """(da, db, info) from oct_preprocess.band_bottom_guide over the served posterior + drawn bottom lines; None
    when inputs are missing. Never raises."""
    tm = _tissue_motion_arrays(ctx)
    post = ctx.surf("posterior_edges")
    if tm is None or tm[0] is None or post is None or not ctx.crop_frames or not ctx.D:
        return None
    try:
        dens = oct_mod.densify_anchor_polylines(ctx.p.get("crop_post_anchors"), int(ctx.D),
                                                bool(ctx.p.get("densify_drawn_polylines", True)))
        dens = {l: {f: v for f, v in r.items() if v < ctx.D - 1} for l, r in dens.items()}
        da, db, info = oct_mod.band_bottom_guide(post, dens, tm[0], tm[1], ctx.crop_frames, int(post.shape[0]), ctx.p)
        return np.asarray(da, np.float64), np.asarray(db, np.float64), dict(info)
    except Exception as exc:  # noqa: BLE001
        return None, None, {"applied": False, "error": type(exc).__name__}


def _move_arrays(ctx: RunCtx):
    """(a[F], b[F], guided) — the per-frame move the run APPLIED: the tissue-motion a/b plus the band-guide delta
    whenever the record says the guide was applied (L14: the post-guide move is used everywhere a target is drawn).
    None when the tissue motion is not recomputable."""
    tm = _tissue_motion_arrays(ctx)
    if tm is None or tm[0] is None or not tm[2].get("applied"):
        return None
    a = np.asarray(tm[0], np.float64).copy(); b = np.asarray(tm[1], np.float64).copy()
    rec = ctx.it.get("tissue_motion") if isinstance(ctx.it.get("tissue_motion"), dict) else {}
    bgrec = rec.get("band_bottom_guide") if isinstance(rec.get("band_bottom_guide"), dict) else {}
    guided = False
    if bgrec.get("applied"):
        bg = _band_guide_arrays(ctx)
        if bg is not None and bg[0] is not None:
            da = np.asarray(bg[0], np.float64); db = np.asarray(bg[1], np.float64) if bg[1] is not None else np.zeros_like(da)
            if da.shape == a.shape:
                a = a + da
                if db.shape == b.shape:
                    b = b + db
                guided = True
    return a, b, guided


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# slices + header
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _slices(ctx: RunCtx, slice_index) -> dict:
    L = int(ctx.L or 1)
    central = L // 2
    pts, _ = _anchor_points(ctx.anchors)
    if pts:
        most = max(sorted(pts.keys()), key=lambda k: len(pts[k]))
        most_edited = int(min(L - 1, max(0, most)))
    else:
        most_edited = central
    if slice_index is not None:
        try:
            si = int(min(L - 1, max(0, int(slice_index))))
        except (TypeError, ValueError):
            si = central
        rendered = [si]
    else:
        rendered = [central] if central == most_edited else [central, most_edited]
    bpts, _ = _anchor_points(ctx.p.get("crop_post_anchors"))
    return {"central": central, "most_edited": most_edited, "rendered": rendered,
            "drawn_top": sorted(pts.keys()), "drawn_bottom": sorted(bpts.keys()),
            "requested": None if slice_index is None else int(slice_index)}


def _identity(ctx: RunCtx) -> tuple[str, str]:
    pid = str(ctx.m.get("patient_id") or "").strip().lower()
    eye = str(ctx.m.get("eye") or "").strip().lower()
    if not (pid and eye):
        try:
            import metrics_export
            meta = metrics_export.parse_case_meta(ctx.m.get("oct_source") or ctx.m.get("input_volume"))
            pid = pid or str(meta.get("patient_id", "")).strip().lower()
            eye = eye or str(meta.get("eye", "")).strip().lower()
        except Exception:  # noqa: BLE001
            pass
    return pid, eye


def _warnings(ctx: RunCtx) -> list[str]:
    w = []
    try:
        fl = ctx.it.get("flatten") if isinstance(ctx.it.get("flatten"), dict) else {}
        try:
            mtr = fl.get("min_target_row")
            if mtr is not None and float(mtr) < 0:
                w.append(f"flatten target above the padded canvas (min_target_row = {float(mtr):.1f}): "
                         f"the deepest placed cells were truncated")
        except (TypeError, ValueError):
            pass
        post = ctx.surf("posterior_edges")
        if post is not None:
            nn = int((~np.isfinite(post)).sum())
            if nn > 0:
                w.append(f"served posterior has {nn} absent cell(s) where the reviewer marked the bottom as absent (drawn as gaps)")
        sig = _anchors_sig(ctx.anchors)
        for name in ("redetect", "generalize", "guided"):
            c = ctx.cache(name)
            if c is not None and "anchors_sig" in c and str(c.get("anchors_sig")) != sig:
                w.append(f"border_cache/{name}.npz was computed for a different anchor set (stale)")
        am = ctx.cache("applied_move")
        if am is not None and not _applied_move_fresh(ctx):
            w.append("border_cache/applied_move.npz key does not match the current raw/corrected volumes (stale)")
        pend = _pending_pane_edits(ctx)
        if pend["n_points"] or pend["n_bottom_points"] or pend["n_marks"]:
            w.append(f"corrected-pane edits are PENDING (not consumed by the last run): {pend['n_points']} edge points on "
                     f"{pend['n_slices']} slices, {pend['n_bottom_points']} bottom points, {pend['n_marks']} defect marks")
    except Exception as exc:  # noqa: BLE001 — warnings are advisory; a malformed record must not take the header down
        w.append(f"warning scan incomplete: {type(exc).__name__}")
    return w


def _pending_pane_edits(ctx: RunCtx) -> dict:
    """Corrected-pane edits still in oct_params = NOT consumed by the last run (the fold clears them on consumption;
    the consumed set lives in oct_iter.corrected_fold). H1."""
    p = ctx.p
    ea, nea = _anchor_points(p.get("corrected_edge_anchors"))
    pa, npa = _anchor_points(p.get("corrected_post_anchors"))
    dm = p.get("corrected_defect_marks") if isinstance(p.get("corrected_defect_marks"), dict) else {}
    ndm = sum(len(v) for v in dm.values() if isinstance(v, list))
    acc = p.get("corrected_accurate") if isinstance(p.get("corrected_accurate"), dict) else {}
    return {"n_slices": len(ea), "n_points": nea, "laterals": sorted(ea.keys()), "n_bottom_slices": len(pa),
            "n_bottom_points": npa, "bottom_laterals": sorted(pa.keys()), "n_marks": ndm,
            "accurate_laterals": sorted(_int_keys(acc).keys())}


def _applied_move_fresh(ctx: RunCtx) -> bool:
    am = ctx.cache("applied_move")
    if am is None or "key" not in am:
        return False
    try:
        key = str(am["key"]).split(":")
        rk = str(os.stat(ctx.raw_path).st_mtime_ns) if ctx.raw_path else None
        wk = str(os.stat(ctx.cor_path).st_mtime_ns) if ctx.cor_path else None
        return len(key) >= 2 and key[0] == rk and key[1] == wk
    except OSError:
        return False


def _run_header(ctx: RunCtx, slices: dict) -> dict:
    pid, eye = _identity(ctx)
    dims_raw = [ctx.L, ctx.D, ctx.F] if ctx.L is not None and ctx.D is not None else None
    dims_cor = None
    if ctx.cor_path is not None:
        try:
            import nibabel as nib
            dims_cor = [int(x) for x in nib.load(str(ctx.cor_path)).shape[:3]]
        except Exception:  # noqa: BLE001
            dims_cor = None
    return {
        "case_id": ctx.cid, "patient_id": pid or None, "eye": eye or None, "mode": ctx.mode,
        "run_time": ctx.run_time,
        "input_volume": str(ctx.cor_path) if ctx.cor_path else None,
        "raw_volume": str(ctx.raw_path) if ctx.raw_path else None,
        "dims_raw": dims_raw, "dims_corrected": dims_cor, "spacing_mm": ctx.spacing or None,
        "slices": slices, "stages": [], "warnings": _warnings(ctx),
        "generated_at": _dt.datetime.now().replace(microsecond=0).isoformat(), "generator": GENERATOR,
    }


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# rendering
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _stretch(sl: np.ndarray) -> np.ndarray:
    """1–99 percentile stretch to [0,1] — identical to oct_border_slice_png."""
    sl = np.asarray(sl, dtype=np.float32)
    finite = sl[np.isfinite(sl)]
    if finite.size == 0:
        return np.zeros(sl.shape, np.float32)
    lo = float(np.percentile(finite, 1)); hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((sl - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _gray(ctx: RunCtx, lateral: int):
    """(depth, frames) stretched raw sagittal slice, or None."""
    raw = ctx.raw
    if raw is None:
        return None
    idx = int(min(raw.shape[0] - 1, max(0, lateral)))
    return _stretch(np.asarray(raw[idx], dtype=np.float32))


def _gray_cor(ctx: RunCtx, lateral: int):
    cor = ctx.cor
    if cor is None:
        return None
    idx = int(min(cor.shape[0] - 1, max(0, lateral)))
    return _stretch(np.asarray(cor[idx], dtype=np.float32))


def _gray_axial(vol, frame: int):
    if vol is None:
        return None
    f = int(min(vol.shape[2] - 1, max(0, frame)))
    return _stretch(np.asarray(vol[:, :, f], dtype=np.float32).T)


_LS = {"solid": "-", "dashed": "--", "dotted": ":"}


def _style_axes(ax, xlabel, ylabel, dark=True):
    fg = "#dddddd" if dark else "#222222"
    ax.set_xlabel(xlabel, fontsize=8, color=fg, labelpad=2)
    ax.set_ylabel(ylabel, fontsize=8, color=fg, labelpad=2)
    ax.tick_params(labelsize=7, colors=fg, length=2, pad=1)
    for sp in ax.spines.values():
        sp.set_color("#666666" if dark else "#888888")
        sp.set_linewidth(0.6)
    ax.set_facecolor("black" if dark else "white")


def _draw_overlays(ax, img_shape, curves, points, bands, dark=True):
    D, X = img_shape
    ymax = D - 0.5
    for (f0, f1, rgba, label) in bands or ():
        ax.axvspan(float(f0) - 0.5, float(f1) + 0.5, color=rgba, lw=0, label=label)
    for c in curves or ():
        y, color, style, label = c[0], c[1], c[2], c[3]
        lw = c[4] if len(c) > 4 else 1.2
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(y)
        if not ok.any():
            continue
        yy = np.where(ok, y, np.nan)
        ax.plot(np.arange(yy.size), yy, color=color, lw=lw, ls=_LS.get(style, "-"), label=label, clip_on=True)
        above = ok & (y < 0)
        if above.any():   # dotted marker at the canvas edge where the curve sits ABOVE the imaging window
            xs = np.arange(yy.size)[above]
            ax.plot(xs, np.full(xs.size, 0.5), color=color, lw=0, marker="v", ms=2.2, alpha=0.9, clip_on=True)
    for pnt in points or ():
        xs, ys, color, label = pnt[0], pnt[1], pnt[2], pnt[3]
        ms = pnt[4] if len(pnt) > 4 else 3.0
        xs = np.asarray(xs, dtype=np.float64); ys = np.asarray(ys, dtype=np.float64)
        ok = np.isfinite(ys)
        ax.plot(xs[ok], np.clip(ys[ok], 0.5, ymax), color=color, lw=0, marker="o", ms=ms, label=label, clip_on=True)


def _fig_image(img, curves=(), points=(), bands=(), size=(THUMB_W_IN, THUMB_H_IN), xlabel="frame (B-scan)",
               ylabel="depth (rows)", legend=True, dark=True, ax=None, fig=None):
    """A (depth, X) image in the app's DISPLAY orientation (x inverted: index 0 on the RIGHT) with overlays drawn
    in array coordinates. Returns the figure (or draws into `ax` when given)."""
    own = ax is None
    if own:
        fig = plt.figure(figsize=size, dpi=100, facecolor="black" if dark else "white")
        ax = fig.add_axes([0.085, 0.13, 0.905, 0.855])
    D, X = img.shape
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0, aspect="auto", interpolation="nearest",
              extent=(-0.5, X - 0.5, D - 0.5, -0.5))
    _draw_overlays(ax, img.shape, curves, points, bands, dark)
    ax.set_xlim(X - 0.5, -0.5)          # display orientation: index 0 on the right
    ax.set_ylim(D - 0.5, -0.5)
    _style_axes(ax, xlabel + "  ◀ display-left = high indices", ylabel, dark)
    if legend and (curves or points or bands):
        leg = ax.legend(fontsize=6, loc="lower left", frameon=True, framealpha=0.55,
                        facecolor="black" if dark else "white", edgecolor="#444444", labelcolor="#eeeeee" if dark else "#222222")
        leg.get_frame().set_linewidth(0.4)
    return fig


def _fig_pair(left, right, size=(THUMB_W_IN, THUMB_H_IN), dark=True):
    """Two (depth, X) panels side by side; each = dict(img, curves, points, bands, xlabel)."""
    fig = plt.figure(figsize=size, dpi=100, facecolor="black" if dark else "white")
    axl = fig.add_axes([0.075, 0.13, 0.43, 0.855]); axr = fig.add_axes([0.565, 0.13, 0.43, 0.855])
    for ax, pnl in ((axl, left), (axr, right)):
        _fig_image(pnl["img"], pnl.get("curves", ()), pnl.get("points", ()), pnl.get("bands", ()),
                   xlabel=pnl.get("xlabel", "frame"), ylabel=pnl.get("ylabel", "depth (rows)"), legend=pnl.get("legend", True),
                   dark=dark, ax=ax, fig=fig)
    return fig


def _fig_plot(series, size=(PLOT_W_IN, PLOT_H_IN), xlabel="frame (B-scan)", ylabel="px", hlines=(0.0,), bands=(),
              dark=True, invert_x=True, ax=None, fig=None, annotate=None):
    """Line plot over frames: series = [(y, color, label, style)]. x inverted like the images (frame 0 right)."""
    own = ax is None
    if own:
        fig = plt.figure(figsize=size, dpi=100, facecolor="black" if dark else "white")
        ax = fig.add_axes([0.11, 0.19, 0.87, 0.78])
    n = 0
    for (f0, f1, rgba, label) in bands or ():
        ax.axvspan(float(f0) - 0.5, float(f1) + 0.5, color=rgba, lw=0, label=label)
    for s in series:
        y, color, label = s[0], s[1], s[2]
        style = s[3] if len(s) > 3 else "solid"
        y = np.asarray(y, dtype=np.float64)
        n = max(n, y.size)
        ax.plot(np.arange(y.size), y, color=color, lw=1.2, ls=_LS.get(style, "-"), label=label)
    for h in hlines or ():
        ax.axhline(h, color="#888888", lw=0.6, ls="--")
    if invert_x and n:
        ax.set_xlim(n - 0.5, -0.5)
    _style_axes(ax, xlabel + ("  ◀ display-left = high indices" if invert_x else ""), ylabel, dark)
    ax.grid(True, color="#333333" if dark else "#dddddd", lw=0.4)
    if annotate:
        ax.text(0.01, 0.03, annotate, transform=ax.transAxes, fontsize=6.5, va="bottom", ha="left",
                color="#eeeeee" if dark else "#222222")
    if series:
        leg = ax.legend(fontsize=6, loc="upper left", frameon=True, framealpha=0.55,
                        facecolor="black" if dark else "white", edgecolor="#444444", labelcolor="#eeeeee" if dark else "#222222")
        leg.get_frame().set_linewidth(0.4)
    return fig


def _fig_to_png_bytes(fig, dpi=100) -> tuple[bytes, int, int]:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    finally:
        plt.close(fig)
    data = buf.getvalue()
    w, h = _png_size(data)
    return data, w, h


def _png_size(data: bytes) -> tuple[int, int]:
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w = int.from_bytes(data[16:20], "big"); h = int.from_bytes(data[20:24], "big")
            return w, h
    except Exception:  # noqa: BLE001
        pass
    return 0, 0


def _data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _fig_to_data_url(fig, dpi=100) -> tuple[str, int, int]:
    data, w, h = _fig_to_png_bytes(fig, dpi)
    return _data_url(data), w, h


# ── PIL fallback (no matplotlib) ───────────────────────────────────────────────────────────────────────────────
def _hex_rgb(color) -> tuple:
    if isinstance(color, tuple):
        return tuple(int(255 * c) for c in color[:3])
    c = str(color).lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _pil_image(img, curves=(), points=(), bands=(), size_px=(480, 300)) -> tuple[bytes, int, int]:
    from PIL import Image, ImageDraw
    D, X = img.shape
    g = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(g[:, :, None], 3, axis=2).astype(np.float32)
    for (f0, f1, rgba, _lab) in bands or ():
        c = np.array(_hex_rgb(rgba), np.float32); a = rgba[3] if isinstance(rgba, tuple) and len(rgba) > 3 else 0.25
        lo, hi = max(0, int(f0)), min(X - 1, int(f1))
        rgb[:, lo:hi + 1] = rgb[:, lo:hi + 1] * (1 - a) + c[None, None, :] * a
    im = Image.fromarray(rgb.astype(np.uint8), "RGB")
    dr = ImageDraw.Draw(im)
    for c in curves or ():
        y = np.asarray(c[0], np.float64); col = _hex_rgb(c[1])
        pts = [(int(x), int(round(v))) for x, v in enumerate(y) if math.isfinite(v) and 0 <= v < D]
        if len(pts) > 1:
            dr.line(pts, fill=col, width=2)
    for pnt in points or ():
        col = _hex_rgb(pnt[2])
        for x, v in zip(pnt[0], pnt[1]):
            if math.isfinite(v):
                v = min(D - 1, max(0, v))
                dr.ellipse([x - 2, v - 2, x + 2, v + 2], fill=col)
    im = im.transpose(Image.FLIP_LEFT_RIGHT).resize(size_px, Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue(), size_px[0], size_px[1]


def _pil_plot(series, size_px=(480, 220)) -> tuple[bytes, int, int]:
    from PIL import Image, ImageDraw
    W, H = size_px
    im = Image.new("RGB", (W, H), (0, 0, 0)); dr = ImageDraw.Draw(im)
    ys = [np.asarray(s[0], np.float64) for s in series if len(s)]
    allv = np.concatenate([y[np.isfinite(y)] for y in ys]) if ys else np.zeros(0)
    lo = float(min(allv.min(), 0.0)) if allv.size else -1.0; hi = float(max(allv.max(), 0.0)) if allv.size else 1.0
    if hi <= lo:
        hi = lo + 1.0
    def _py(v):
        return int(H - 12 - (v - lo) / (hi - lo) * (H - 24))
    dr.line([(0, _py(0.0)), (W, _py(0.0))], fill=(120, 120, 120), width=1)
    for s in series:
        y = np.asarray(s[0], np.float64); n = y.size
        pts = [(int(W - 1 - i * (W - 1) / max(1, n - 1)), _py(v)) for i, v in enumerate(y) if math.isfinite(v)]
        if len(pts) > 1:
            dr.line(pts, fill=_hex_rgb(s[1]), width=2)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    return buf.getvalue(), W, H


# ── image record helpers ────────────────────────────────────────────────────────────────────────────────────
class _Renderer:
    """Renders node images either as thumbnails (data URLs) or at export size (PNG bytes per file)."""

    def __init__(self, thumb_h: int = 300, export: bool = False, dpi: int = 100):
        self.thumb_h = int(min(600, max(120, thumb_h)))
        self.export = export
        self.dpi = dpi
        self.scale = self.thumb_h / 300.0
        self.files: dict[str, bytes] = {}

    def _size(self, base_w, base_h):
        if self.export:
            return (EXPORT_W_IN, EXPORT_W_IN * base_h / base_w)
        return (base_w * self.scale, base_h * self.scale)

    def image(self, img, role, slice_index, orientation, caption, file, curves=(), points=(), bands=(), xlabel="frame (B-scan)") -> dict:
        if _HAVE_MPL:
            fig = _fig_image(img, curves, points, bands, size=self._size(THUMB_W_IN, THUMB_H_IN), xlabel=xlabel)
            png, w, h = _fig_to_png_bytes(fig, self.dpi)
        else:
            png, w, h = _pil_image(img, curves, points, bands, size_px=(int(480 * self.scale), int(300 * self.scale)))
        return self._rec(png, w, h, role, slice_index, orientation, caption, file)

    def pair(self, left, right, role, slice_index, orientation, caption, file) -> dict:
        if _HAVE_MPL:
            fig = _fig_pair(left, right, size=self._size(THUMB_W_IN, THUMB_H_IN))
            png, w, h = _fig_to_png_bytes(fig, self.dpi)
        else:
            png, w, h = _pil_image(right["img"], right.get("curves", ()), right.get("points", ()), right.get("bands", ()),
                                   size_px=(int(480 * self.scale), int(300 * self.scale)))
        return self._rec(png, w, h, role, slice_index, orientation, caption, file)

    def plot(self, series, role, caption, file, bands=(), ylabel="px", annotate=None, xlabel="frame (B-scan)", invert_x=True) -> dict:
        if _HAVE_MPL:
            fig = _fig_plot(series, size=self._size(PLOT_W_IN, PLOT_H_IN), bands=bands, ylabel=ylabel, annotate=annotate,
                            xlabel=xlabel, invert_x=invert_x)
            png, w, h = _fig_to_png_bytes(fig, self.dpi)
        else:
            png, w, h = _pil_plot(series, size_px=(int(480 * self.scale), int(220 * self.scale)))
        return self._rec(png, w, h, role, None, "plot", caption, file)

    def _rec(self, png, w, h, role, slice_index, orientation, caption, file) -> dict:
        self.files[file] = png
        return {"role": role, "slice_index": slice_index, "orientation": orientation, "data_url": _data_url(png),
                "width": int(w), "height": int(h), "caption": caption, "file": file}


_DISP = "display-left = high frame indices, frame 0 at right"


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# node construction
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _mk(nid, kind, role, title, status, reason=None, summary="", description="", method="", numbers=None, params=None,
        images=None, image_error=None, warnings=None, source=None) -> dict:
    if status != "applied" and not reason:
        reason = "not recorded"
    return {"id": nid, "kind": kind, "role": role, "status": status, "status_reason": reason, "title": title,
            "summary": summary, "description": description, "method": method, "numbers": numbers or [],
            "params": _json_safe(params or {}), "images": images or [], "image_error": image_error,
            "parents": [], "children": [], "warnings": warnings or [], "source": source or {"manifest": [], "files": []}}


def _safe_images(fn):
    """Run an image builder; on failure return ([], error string) so the node degrades instead of the graph."""
    try:
        return list(fn() or []), None
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def _drawn_pts(pts: dict, lateral: int):
    d = pts.get(int(lateral)) or {}
    fr = np.array(sorted(d.keys()), dtype=np.float64)
    rows = np.array([d[int(f)] for f in fr], dtype=np.float64)
    return fr, rows


def _densified_line(ctx: RunCtx, lateral: int, F: int):
    """(y[F], guarded) — the drawn line of one lateral rendered through the SAME densify_anchor_polylines call the run
    used (reference = served surface, chord guard on the folded laterals), so stroke gaps the chord guard refused stay
    gaps (L13). Falls back to the plain polyline when the densify is unavailable."""
    guarded = set()
    try:
        guarded = set(oct_mod.folded_laterals(ctx.p))
        served = ctx.surf("provided_edges")
        dens = oct_mod.densify_anchor_polylines(ctx.anchors, int(ctx.D) if ctx.D else None,
                                                bool(ctx.p.get("densify_drawn_polylines", True)),
                                                reference=served, guarded_slices=guarded)
        d = dens.get(int(lateral)) or {}
        y = np.full(F, np.nan)
        for f, r in d.items():
            fi = int(f)
            if 0 <= fi < F:
                y[fi] = float(r)
        return y, (int(lateral) in guarded), True
    except Exception:  # noqa: BLE001
        pts, _ = _anchor_points(ctx.anchors)
        return _polyline(pts.get(int(lateral)) or {}, F), (int(lateral) in guarded), False


def _polyline(pts_lat: dict, F: int):
    """Dense polyline over the drawn frames of one lateral (straight segments like the editor)."""
    y = np.full(F, np.nan)
    if not pts_lat:
        return y
    fr = np.array(sorted(pts_lat.keys())); rows = np.array([pts_lat[f] for f in fr], dtype=np.float64)
    if fr.size == 1:
        y[int(fr[0])] = rows[0]
        return y
    lo, hi = int(fr.min()), int(fr.max())
    xs = np.arange(lo, hi + 1)
    y[lo:hi + 1] = np.interp(xs, fr, rows)
    return y


def _above_note(y, D) -> str:
    y = np.asarray(y, np.float64); ok = np.isfinite(y)
    n = int((ok & (y < 0)).sum())
    return f" {n} column(s) sit above the imaging window (clipped, ▾ markers)." if n else ""


def _band_list(ctx: RunCtx) -> list:
    cf = ctx.crop_frames
    if not cf:
        return []
    return [(cf[0], cf[-1], C_BAND, f"surface-crop band {cf[0]}–{cf[-1]}")]


# ── n00 raw ────────────────────────────────────────────────────────────────────────────────────────────────
def _node_raw(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    m = ctx.m
    pid, eye = _identity(ctx)
    bg = m.get("border_gt") if isinstance(m.get("border_gt"), dict) else None
    numbers = [_n("case", ctx.cid), _n("patient / eye", f"{(pid or '?').upper()} / {(eye or '?').upper()}"),
               _n("dims raw (laterals, depth, frames)", [ctx.L, ctx.D, ctx.F]),
               _n("spacing", ctx.spacing, "mm"), _n(".OCT source", os.path.basename(str(m.get("oct_source") or "")) or None),
               _n("review flags", m.get("review_flags") or []), _n("difficult scan", bool(m.get("difficult_scan"))),
               _n("difficult reason", m.get("difficult_reason") or None), _n("preprocessing vetted", bool(m.get("preproc_vetted"))),
               _n("border GT", (f"confirmed, {bg.get('n_slices')} slices / {bg.get('n_points')} points" if bg and bg.get("confirmed") else "none"))]
    ok = ctx.raw_path is not None
    status = "applied" if ok else "missing_data"
    reason = None if ok else "input/_raw_border.nii.gz absent — open the scan (Fix-columns) once to create it"
    sx = ctx.spacing if len(ctx.spacing) == 3 else [None, None, None]
    summary = f"{ctx.L or '?'} laterals × {ctx.D or '?'} depth × {ctx.F or '?'} frames" + (f" · {', '.join(m.get('review_flags'))}" if m.get("review_flags") else "")
    description = (f"The raw Avanti volume as reformatted to sagittal slices: {ctx.F or '?'} B-scans (frames), each {ctx.L or '?'} "
                   f"A-scans (laterals) deep by {ctx.D or '?'} rows. This is the input every stage below reads; the tree never re-reads the .OCT.")
    method = (f"Volumes were acquired as {ctx.F or '?'} B-scans of {ctx.L or '?'} A-scans × {ctx.D or '?'} depth samples "
              f"({_fmt(sx[0], 4)}×{_fmt(sx[1], 5)}×{_fmt(sx[2], 3)} mm).")
    images, err = [], None
    if R is not None and ok:
        def _b():
            out = []
            for s in sl["rendered"]:
                g = _gray(ctx, s)
                if g is None:
                    continue
                out.append(R.image(g, "central" if s == sl["central"] else ("most_edited" if s == sl["most_edited"] else "requested"), s, "sagittal",
                                   f"Raw sagittal slice {s} ({_DISP}).", f"n00_raw_s{s}.png"))
            fa = int((ctx.F or 1) // 2)
            ga = _gray_axial(ctx.raw, fa)
            if ga is not None:
                out.append(R.image(ga, "primary", fa, "axial", f"Raw axial B-scan (frame {fa}); lateral 0 at right as in the app.",
                                   f"n00_raw_axial_f{fa}.png", xlabel="lateral (A-scan)"))
            return out
        images, err = _safe_images(_b)
    return _mk("n00_raw", "input", "spine", "Raw scan", status, reason, summary, description, method, numbers,
               {"oct_volume_index": m.get("oct_volume_index", 0)}, images, err,
               source={"manifest": ["case_id", "oct_source", "oct_spacing", "review_flags", "difficult_scan", "preproc_vetted", "border_gt"],
                       "files": ["input/_raw_border.nii.gz"]})


# ── n01 detect ─────────────────────────────────────────────────────────────────────────────────────────────
def _node_detect(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    base = ctx.cache("baseline")
    surf = ctx.surf("baseline")
    p = ctx.p
    params = {k: p.get(k) for k in ("dp_sigma_depth", "dp_sigma_frame", "dp_below", "dp_max_jump")}
    at = ctx.it.get("auto_tune") if isinstance(ctx.it.get("auto_tune"), dict) else None
    if at:
        params["auto_tune"] = at
    numbers = [_n(k, v) for k, v in params.items() if k != "auto_tune"]
    ok = surf is not None
    if ok:
        sig = str(base.get("params_sig", ""))
        rm = base.get("raw_mtime")
        try:
            rm_match = (ctx.raw_mtime() is not None and abs(float(rm) - float(ctx.raw_mtime())) < 1e-3)
        except (TypeError, ValueError):
            rm_match = False
        numbers += [_n("cache params_sig", sig[:60] + ("…" if len(sig) > 60 else "")), _n("raw mtime matches", bool(rm_match)),
                    _n("surface row range", [_r(np.nanmin(surf), 1), _r(np.nanmax(surf), 1)], "px")]
    status = "applied" if ok else "missing_data"
    reason = None if ok else "no cached baseline (border_cache/baseline.npz) — open Fix-columns once or re-run"
    summary = "DP edge tracker" + (f" · rows {_fmt(np.nanmin(surf))}..{_fmt(np.nanmax(surf))}" if ok else " · no cache")
    description = ("The anterior corneal surface detected automatically on every sagittal slice of the raw volume by the "
                   "dynamic-programming edge tracker (despeckle → tissue-below-gated gradient → smooth DP path). It is the "
                   "starting surface the reviewer corrects; it is shown in blue on every image below.")
    method = (f"The anterior corneal surface was detected automatically per sagittal slice by a dynamic-programming edge tracker "
              f"(depth σ {_fmt(p.get('dp_sigma_depth'))}, frame σ {_fmt(p.get('dp_sigma_frame'))}, tissue-below gate "
              f"{_fmt(p.get('dp_below'), 0)} px, max jump {_fmt(p.get('dp_max_jump'), 0)} px).")
    images, err = [], None
    if R is not None and ok:
        def _b():
            out = []
            for s in sl["rendered"]:
                g = _gray(ctx, s)
                if g is None:
                    continue
                y = surf[min(surf.shape[0] - 1, s)]
                out.append(R.image(g, "central" if s == sl["central"] else "most_edited", s, "sagittal",
                                   f"Automatic (baseline) surface in blue on raw slice {s} ({_DISP}).{_above_note(y, ctx.D)}",
                                   f"n01_detect_s{s}.png", curves=[(y, C_BASELINE, "solid", "auto surface")]))
            return out
        images, err = _safe_images(_b)
    return _mk("n01_detect", "auto", "spine", "Automatic corneal-surface detection (DP)", status, reason, summary, description,
               method, numbers, params, images, err,
               source={"manifest": ["oct_params.dp_*", "oct_iter.auto_tune"], "files": ["border_cache/baseline.npz"]})


# ── n02 top lines ──────────────────────────────────────────────────────────────────────────────────────────
def _node_top_lines(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    pts, npts = _anchor_points(ctx.anchors)
    p = ctx.p
    params = {k: p.get(k) for k in ("border_generalize", "border_guided", "redetect_seed_window", "densify_drawn_polylines")}
    folded = sorted(oct_mod.folded_laterals(p))
    if folded:
        params["flatten_exclude_laterals"] = folded
    ok = bool(pts)
    numbers = [_n("slices drawn", len(pts)), _n("points", npts)]
    if folded:
        numbers.append(_n("folded laterals (chord-guarded, out of the per-frame fit)", _runs(folded)))
    if ok:
        rows = np.array([r for d in pts.values() for r in d.values()], dtype=np.float64)
        per = [len(d) for d in pts.values()]
        numbers += [_n("row range", f"{rows.min():.0f}..{rows.max():.0f}", "px"), _n("points above window (row < 0)", int((rows < 0).sum())),
                    _n("frames per slice (min..max)", f"{min(per)}..{max(per)}"), _n("laterals", _runs(pts.keys()))]
    status = "applied" if ok else "not_run"
    reason = None if ok else "no top-edge lines on record (oct_params.border_anchors empty)"
    summary = f"{len(pts)} slices, {npts} points" + (f"; rows {rows.min():.0f}..{rows.max():.0f}" if ok else "")
    description = ("The reviewer's top-edge corrections on the ORIGINAL pane: on each drawn sagittal slice the dragged points "
                   "(and the straight segments between them) replace the automatic surface. They are the ground truth the "
                   "served surface is pinned to; points above the window (negative rows) mark an apex outside the scan.")
    method = (f"A reviewer corrected the anterior surface on {len(pts)} sagittal slices ({npts} points"
              + (f"; {int((rows < 0).sum())} points above the imaging window" if ok else "") + ").") if ok else ""
    images, err = [], None
    if R is not None and ok:
        def _b():
            out = []
            targets = [s for s in sl["rendered"]]
            if sl["most_edited"] not in targets and sl.get("requested") is None:
                targets.append(sl["most_edited"])
            for s in targets:
                g = _gray(ctx, s)
                if g is None:
                    continue
                fr, rows_s = _drawn_pts(pts, s)
                poly, guarded, via_run = _densified_line(ctx, s, g.shape[1])
                if fr.size:
                    cap = (f"Raw sagittal slice {s} ({_DISP}) with the reviewer's top-edge points (red) and the line between them "
                           + ("rendered by the run's own densify_anchor_polylines (reference = served surface"
                              + (", chord guard on this folded lateral: stroke gaps the guard refused are left open" if guarded else "")
                              + ")." if via_run else "as straight segments."))
                else:
                    cap = f"Raw sagittal slice {s} ({_DISP}) — no top-edge line drawn on this slice."
                out.append(R.image(g, "most_edited" if s == sl["most_edited"] else "central", s, "sagittal", cap + _above_note(rows_s, ctx.D),
                                   f"n02_top_lines_s{s}.png", curves=[(poly, C_SERVED, "solid", "drawn line", 0.8)] if fr.size else [],
                                   points=[(fr, rows_s, C_SERVED, "drawn points", 2.5)] if fr.size else []))
            return out
        images, err = _safe_images(_b)
    return _mk("n02_top_lines", "user", "input", "Reviewer top-edge lines (original pane)", status, reason, summary, description,
               method, numbers, params, images, err,
               source={"manifest": ["oct_params.border_anchors", "oct_params.border_generalize", "oct_params.border_guided"], "files": []})


def _crop_bands_brief(cb) -> str:
    """One line for oct_params.crop_bands in EITHER persisted form (explicit {"bands": [{"id", "marks"}, …]} since
    2026-09-09; legacy {"<lateral>": [lo, hi]} = one band): 'N band(s), M mark(s): band 1 (2 marks) @ 0:[30,39]
    511:[30,39]; …'. Bands are independent (each interpolated across its own marks), so they are listed per band."""
    try:
        bands = oct_mod.parse_crop_bands(cb)
    except Exception:  # noqa: BLE001 — provenance must never fail on a malformed mark
        return "unparseable"
    if not bands:
        return "absent"
    parts = []
    for bid, marks in bands[:4]:
        items = list(marks.items())
        txt = " ".join(f"{lat}:[{lo},{hi}]" for lat, (lo, hi) in items[:4]) + (" …" if len(items) > 4 else "")
        parts.append(f"band {bid} ({len(items)} marks) @ {txt}")
    n_marks = sum(len(m) for _, m in bands)
    return f"{len(bands)} band(s), {n_marks} mark(s): " + "; ".join(parts) + (" …" if len(bands) > 4 else "")


# ── n03 crop marks ─────────────────────────────────────────────────────────────────────────────────────────
def _node_crop_marks(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    p, it = ctx.p, ctx.it
    cf = ctx.crop_frames
    mode = str(p.get("surface_crop_mode") or ("manual" if cf else "auto"))
    cb = p.get("crop_bands") if isinstance(p.get("crop_bands"), dict) else {}
    cr = p.get("crop_region") if isinstance(p.get("crop_region"), dict) else {}
    numbers = [_n("surface-crop band", f"{mode}: frames {_runs(cf)} ({len(cf)} of {ctx.F or '?'})" if cf else "absent"),
               _n("artifact bands (crop_bands)", _crop_bands_brief(cb) if cb else "absent"),
               _n("crop region", (f"laterals {cr.get('lateral')} × {len(cr.get('frames') or [])} frames" + (" (auto)" if cr.get("auto") else "")) if cr and cr.get("frames") else "absent")]
    scv = it.get("surface_crop") if isinstance(it.get("surface_crop"), dict) else None
    if scv:
        numbers.append(_n("run record (surface_crop)", {k: scv.get(k) for k in ("n_frames", "pad", "rule", "auto", "clamped") if k in scv}))
    crop_rec = it.get("crop") if isinstance(it.get("crop"), dict) else None
    if crop_rec:
        numbers.append(_n("zeroed voxels (crop)", crop_rec.get("n_voxels")))
    present = bool(cf or cb or (cr and cr.get("frames")))
    status = "applied" if present else "not_run"
    reason = None if present else "no crop marks (no surface-crop band, artifact bands or crop region on record)"
    summary = (f"band {cf[0]}–{cf[-1]} ({len(cf)} frames, {mode})" if cf else "no band") + (f" · {len(cb)} artifact bands" if cb else "") + (" · crop region" if cr and cr.get("frames") else "")
    description = ("Frame columns the reviewer marked: the SURFACE-CROP band (B-scans whose apex is cut off at the top of the window, "
                   "so no anterior surface exists there and the top is placed from the posterior instead), per-lateral ARTIFACT bands "
                   "(time-domain artifact columns excluded from the fit and zeroed) and a rectangular CROP region (zeroed before SAM2).")
    method = (f"B-scans {cf[0]}–{cf[-1]} ({len(cf)} of {ctx.F or '?'}) were marked as apex-cropped ({mode} mode)." if cf else "")
    if cb:
        method += f" {len(cb)} per-lateral artifact bands were excluded from the surface fit and zeroed."
    if cr and cr.get("frames"):
        method += f" A crop region of {len(cr.get('frames'))} frames over laterals {cr.get('lateral')} was zeroed."
    images, err = [], None
    if R is not None and present:
        def _b():
            out = []
            s = sl["rendered"][0]
            g = _gray(ctx, s)
            if g is None:
                return out
            bands = _band_list(ctx)
            for k, v in cb.items():
                try:
                    if int(k) == int(s) and len(v) == 2:
                        bands.append((int(v[0]), int(v[1]), C_CROP, "artifact band"))
                except (TypeError, ValueError):
                    continue
            if cr and cr.get("frames") and cr.get("lateral") and len(cr["lateral"]) == 2 and min(cr["lateral"]) <= s <= max(cr["lateral"]):
                fr = sorted(int(f) for f in cr["frames"])
                bands.append((fr[0], fr[-1], C_CROP, "crop region"))
            out.append(R.image(g, "primary", s, "sagittal",
                               f"Raw slice {s} ({_DISP}); blue = surface-crop band, red = artifact/crop columns on this lateral.",
                               f"n03_crop_marks_s{s}.png", bands=bands))
            return out
        images, err = _safe_images(_b)
    return _mk("n03_crop_marks", "user", "input", "Surface-crop band / artifact bands / crop region", status, reason, summary,
               description, method, numbers, {"surface_crop_mode": mode, "auto_surface_crop": p.get("auto_surface_crop")}, images, err,
               source={"manifest": ["oct_params.surface_crop_frames", "oct_params.surface_crop_mode", "oct_params.crop_bands",
                                    "oct_params.crop_region", "oct_iter.surface_crop", "oct_iter.crop"], "files": []})


# ── n04 served surface ─────────────────────────────────────────────────────────────────────────────────────
def _served_source(ctx: RunCtx) -> tuple[str, bool]:
    """Which cache the API rule served (redetect for a dense route; guided/generalize otherwise) + its sig freshness."""
    det = ctx.it.get("determinism") if isinstance(ctx.it.get("determinism"), dict) else {}
    route = str(det.get("route") or "")
    sig = _anchors_sig(ctx.anchors)
    if route == "dense":
        name = "redetect"
    elif ctx.p.get("border_guided") and ctx.cache("guided") is not None:
        name = "guided"
    elif ctx.p.get("border_generalize"):
        name = "generalize"
    else:
        name = "redetect"
    c = ctx.cache(name)
    fresh = bool(c is not None and str(c.get("anchors_sig", "")) == sig)
    return name, fresh


def _node_served_surface(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    surf = ctx.surf("provided_edges")
    det = ctx.it.get("determinism") if isinstance(ctx.it.get("determinism"), dict) else {}
    pts, npts = _anchor_points(ctx.anchors)
    route = str(det.get("route") or "")
    src, fresh = _served_source(ctx)
    floor = det.get("floor") if isinstance(det.get("floor"), dict) else {}
    blind = det.get("blind_frames") if isinstance(det.get("blind_frames"), dict) else {}
    numbers = [_n("route", route or None), _n("served from cache", f"{src}.npz" + (" (fresh)" if fresh else " (stale/absent sig)")),
               _n("slices drawn", det.get("n_slices_drawn", len(pts))), _n("points drawn", det.get("n_points_drawn", npts)),
               _n("σ effective (tolerance basis)", det.get("sigma_eff_px"), "px"), _n("tolerance T", det.get("T_px"), "px"),
               _n("reviewer scatter σ (measured; 0 when every drawn frame is pinned exactly)", _r(det.get("sigma_px"), 3), "px"),
               _n("findings", det.get("n_findings")),
               _n("floor off-quadratic", _r(floor.get("off_quad_px")), "px"), _n("floor rms", _r(floor.get("rms_px")), "px"),
               _n("blind frames driven / too sparse", f"{blind.get('driven', '?')} / {blind.get('too_sparse', '?')}")]
    ok = surf is not None and (ctx.mode == "corrections" or bool(pts))   # a leftover cache on an automatic run is not a served surface
    if ok:
        numbers.append(_n("surface row range", [_r(np.nanmin(surf), 1), _r(np.nanmax(surf), 1)], "px"))
    status = "applied" if ok else ("missing_data" if ctx.mode == "corrections" else "not_run")
    reason = None if ok else ("border_cache/provided_edges.npz absent — run 'Re-run with corrections' to regenerate" if ctx.mode == "corrections"
                              else "automatic run — no reviewer corrections were on record")
    clause = {"dense": "per-frame interpolation across the drawn slices"}.get(route, None) or (
        "guided re-detection within ±1 px of the drawn surface" if src == "guided" else
        "a generalised per-frame correction field" if src == "generalize" else "local re-detection around the drawn slices")
    summary = f"route {route or '?'} · {src}.npz + pinned lines" + (f" · σ_eff {_fmt(det.get('sigma_eff_px'), 2)} px, T {_fmt(det.get('T_px'))} px" if det else "")
    description = ("The surface the run actually flattened to (border_cache/provided_edges.npz): the reviewer's lines pinned exactly on "
                   f"the drawn slices and {clause} between them, then a gentle limbus-band smoothing. Red on every image; the automatic "
                   "baseline stays blue for comparison. The determinism report measures how much of it the drawn lines determine.")
    method = ((f"The served surface was obtained by {clause} with drawn slices pinned exactly (tolerance T = "
               f"{_fmt(det.get('T_px'))} px derived from an effective reviewer scatter σ_eff = {_fmt(det.get('sigma_eff_px'), 2)} px"
               + (f"; measured scatter on the drawn frames {_fmt(det.get('sigma_px'), 2)} px" if det.get("sigma_px") is not None else "") + ").") if ok and det else
              ("The served surface was obtained by " + clause + " with drawn slices pinned exactly." if ok else ""))
    images, err = [], None
    if R is not None and ok:
        def _b():
            out = []
            base = ctx.surf("baseline")
            for s in sl["rendered"]:
                g = _gray(ctx, s)
                if g is None:
                    continue
                y = surf[min(surf.shape[0] - 1, s)]
                curves = [(y, C_SERVED, "solid", "served surface")]
                if base is not None:
                    curves.insert(0, (base[min(base.shape[0] - 1, s)], C_BASELINE, "solid", "auto baseline", 0.8))
                fr, rows = _drawn_pts(pts, s)
                out.append(R.image(g, "central" if s == sl["central"] else "most_edited", s, "sagittal",
                                   f"Served surface (red) vs automatic baseline (blue) on raw slice {s} ({_DISP}); red dots = drawn points.{_above_note(y, ctx.D)}",
                                   f"n04_served_surface_s{s}.png", curves=curves, points=[(fr, rows, C_SERVED, "drawn points", 2.5)] if fr.size else []))
            pf = det.get("per_frame") if isinstance(det.get("per_frame"), dict) else {}
            if pf.get("E_px") or pf.get("n"):
                e = np.asarray(pf.get("E_px") or [], np.float64)
                nn = np.asarray(pf.get("n") or [], np.float64)
                e_zero = (e.size == 0) or (not np.any(np.abs(e[np.isfinite(e)]) > 1e-6))
                if e_zero:   # M11: an all-zero E_px strip says nothing — plot the drawn-slice count and say why
                    out.append(R.plot([(nn, C_BASELINE, "drawn slices per frame")], "plot",
                                      "Determinism per frame: drawn-slice count (blue). E_px (the upper-bound effect of reviewer scatter) is "
                                      f"identically zero on this run because every drawn frame is pinned exactly; the tolerance T = {_fmt(det.get('T_px'))} px "
                                      f"derives from σ_eff = {_fmt(det.get('sigma_eff_px'), 2)} px instead.",
                                      "n04_served_surface_determinism.png", bands=_band_list(ctx), ylabel="slices"))
                else:
                    out.append(R.plot([(e, C_SERVED, "E_px (upper-bound effect of reviewer scatter)"), (nn, C_BASELINE, "drawn slices per frame")],
                                      "plot", "Determinism per frame: upper-bound effect of the reviewer's scatter (red) and drawn-slice count (blue).",
                                      "n04_served_surface_determinism.png", bands=_band_list(ctx)))
            return out
        images, err = _safe_images(_b)
    return _mk("n04_served_surface", "auto", "spine", "Served surface (lines pinned + interpolated / re-detected)", status, reason,
               summary, description, method, numbers,
               {"route": route, "served_cache": src, "interp_min_slices": det.get("interp_min_slices", ctx.p.get("interp_min_slices")),
                **{k: ctx.p.get(k) for k in ("redetect_interp_window", "redetect_seed_window", "guided_window", "guided_window_min",
                                             "dense_anchor_redetect", "border_generalize", "border_guided", "densify_drawn_polylines")}},
               images, err, source={"manifest": ["oct_iter.determinism"], "files": ["border_cache/provided_edges.npz", f"border_cache/{src}.npz"]})


# ── n05 bottom lines ───────────────────────────────────────────────────────────────────────────────────────
def _node_bottom_lines(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    pts, npts = _anchor_points(ctx.p.get("crop_post_anchors"))
    post = ctx.surf("posterior_edges")
    D = ctx.D or 0
    n_sent = sum(1 for d in pts.values() for v in d.values() if D and v >= D - 1)
    cpa, ncpa = _anchor_points(ctx.p.get("corrected_post_anchors"))
    numbers = [_n("slices drawn", len(pts)), _n("points", npts), _n("absent-bottom sentinels", n_sent),
               _n("pending corrected-pane bottom lines", f"{len(cpa)} slices / {ncpa} points" if cpa else "none"),
               _n("served posterior", "present" if post is not None else "absent")]
    if pts:
        rows = np.array([r for d in pts.values() for r in d.values()], dtype=np.float64)
        numbers.append(_n("row range", f"{rows.min():.0f}..{rows.max():.0f}", "px"))
    if post is not None:
        numbers.append(_n("posterior NaN cells", int((~np.isfinite(post)).sum())))
    have = bool(pts) or post is not None
    if not have:
        status, reason = "not_run", "no bottom lines and no served posterior on record"
    elif post is None and ctx.mode == "corrections":
        status, reason = "missing_data", "border_cache/posterior_edges.npz absent (deleted by the last bottom-line fold; re-run to regenerate)"
    else:
        status, reason = "applied", None
    summary = f"{len(pts)} slices, {npts} points" + (" · posterior served" if post is not None else " · no served posterior")
    description = ("Inside the surface-crop band the anterior does not exist, so the only edge is the POSTERIOR: the detector's bottom "
                   "edge plus the reviewer's bottom lines, interpolated across slices into the served posterior (orange). A point at the "
                   "canvas bottom is the ABSENT sentinel (no bottom here) and is left out.")
    method = (f"The posterior surface was served from the detector's bottom edge and {len(pts)} reviewer bottom lines ({npts} points)."
              if pts else ("The posterior surface was served from the detector's bottom edge." if post is not None else ""))
    images, err = [], None
    if R is not None and have and ctx.raw_path is not None:
        def _b():
            out = []
            targets = list(sl["rendered"])
            if sl.get("requested") is None and pts:
                best = max(sorted(pts.keys()), key=lambda k: len(pts[k]))
                if best not in targets:
                    targets.append(best)
            for s in targets:
                g = _gray(ctx, s)
                if g is None:
                    continue
                curves = []
                if post is not None:
                    curves.append((post[min(post.shape[0] - 1, s)], C_BOTTOM, "solid", "served posterior"))
                fr, rows = _drawn_pts(pts, s)
                ok = rows < (D - 1) if D else np.ones(rows.size, bool)
                out.append(R.image(g, "primary" if s == targets[-1] else "central", s, "sagittal",
                                   f"Served posterior (orange) on raw slice {s} ({_DISP}); orange dots = reviewer bottom points.",
                                   f"n05_bottom_lines_s{s}.png", curves=curves, bands=_band_list(ctx),
                                   points=[(fr[ok], rows[ok], C_BOTTOM, "bottom points", 2.5)] if fr.size else []))
            return out
        images, err = _safe_images(_b)
    return _mk("n05_bottom_lines", "user", "input", "Reviewer bottom lines + served posterior", status, reason, summary, description,
               method, numbers, {}, images, err,
               source={"manifest": ["oct_params.crop_post_anchors", "oct_params.corrected_post_anchors"], "files": ["border_cache/posterior_edges.npz"]})


# ── n06 placement ──────────────────────────────────────────────────────────────────────────────────────────
def _placed_split(th: dict) -> tuple:
    """(cells from the thickness model, cells from the estimate fallback, fallback %) from crop_reconstruction.thickness;
    (None, n_est, None) when n_placed is not on record."""
    try:
        n_pl = int(th.get("n_placed")); n_est = int(th.get("n_estimate_fallback") or 0)
    except (TypeError, ValueError):
        return None, th.get("n_estimate_fallback"), None
    if n_pl <= 0:
        return None, n_est, None
    return max(0, n_pl - n_est), n_est, 100.0 * n_est / n_pl


def _node_placement(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    rec = ctx.it.get("crop_reconstruction") if isinstance(ctx.it.get("crop_reconstruction"), dict) else None
    placed = ctx.surf("placed_edges")
    th = rec.get("thickness") if rec and isinstance(rec.get("thickness"), dict) else {}
    tt = th.get("thickness") if isinstance(th.get("thickness"), dict) else {}
    bd = th.get("boundary") if isinstance(th.get("boundary"), dict) else {}
    numbers = []
    if rec:
        numbers += [_n("band frames", rec.get("n_frames")), _n("columns rebuilt", rec.get("n_columns_rebuilt")),
                    _n("fallback cells", rec.get("n_fallback_cells")), _n("placed drawn laterals", rec.get("n_placed_drawn_laterals")),
                    _n("from", rec.get("from")), _n("thickness model / source", f"{th.get('model', '?')} / {th.get('source', '?')}"),
                    _n("thickness median (p10..p90)", f"{_fmt(tt.get('median'))} ({_fmt(tt.get('p10'))}..{_fmt(tt.get('p90'))})", "px"),
                    _n("boundary thickness left / right", f"{_fmt(bd.get('left_median'))} / {_fmt(bd.get('right_median'))}", "px"),
                    _n("cells placed (total)", th.get("n_placed")), _n("cells from the thickness model", _placed_split(th)[0]),
                    _n("cells from the fixed estimate fallback", th.get("n_estimate_fallback")),
                    _n("fallback share", f"{_placed_split(th)[2]:.0f} %" if _placed_split(th)[2] is not None else None),
                    _n("slices served from lines / fully on the estimate", f"{th.get('n_slices_served', '?')} / {th.get('n_slices_fallback', '?')}"),
                    _n("slices with drawn thickness left / right", f"{th.get('n_slices_drawn_left', '?')} / {th.get('n_slices_drawn_right', '?')}"),
                    _n("top-present cells kept", th.get("n_top_present_kept")), _n("ceiling-clamped cells", th.get("n_ceiling_clamped"))]
    if rec:
        status, reason = ("applied", None) if placed is not None else ("missing_data", "border_cache/placed_edges.npz absent — re-run to regenerate")
    else:
        status = "not_run"
        reason = "no surface-crop band — nothing to place" if ctx.mode == "corrections" else "automatic run — no reviewer corrections were on record"
    summary = (f"{rec.get('n_frames')} band frames · {rec.get('n_columns_rebuilt')} columns · T median {_fmt(tt.get('median'))} px" if rec else "no band")
    description = ("Where the apex is cropped the anterior surface is PLACED rather than detected: bottom − corneal thickness, with the "
                   "thickness interpolated from the reviewer's drawn lines first and the served surface as fallback (a fixed estimate where "
                   "neither exists). The placed top (magenta, dashed) replaces the served surface inside the band.")
    method = ""
    if rec:
        n_model, n_est, pct = _placed_split(th)
        if th.get("n_placed"):
            method = (f"Within this band the anterior surface was placed as the posterior surface minus a corneal thickness: of "
                      f"{int(th.get('n_placed'))} placed cells, {n_model} ({100 - pct:.0f} %) took the thickness interpolated from the "
                      f"{th.get('source', 'drawn')} lines ({th.get('model', 'linear')} model, median {_fmt(tt.get('median'))} px) and {n_est} ({pct:.0f} %) "
                      f"fell back to the fixed thickness estimate ({th.get('n_slices_served', '?')} slices served from lines, "
                      f"{th.get('n_slices_fallback', '?')} slices entirely on the estimate), yielding {rec.get('n_columns_rebuilt')} rebuilt columns.")
        else:
            method = (f"Within this band the anterior surface was placed from the posterior surface minus a {th.get('model', 'linear')} "
                      f"corneal-thickness model (median {_fmt(tt.get('median'))} px), yielding {rec.get('n_columns_rebuilt')} rebuilt columns.")
    images, err = [], None
    if R is not None and rec and placed is not None:
        def _b():
            out = []
            post = ctx.surf("posterior_edges"); served = ctx.surf("provided_edges")
            cf = set(ctx.crop_frames)
            for s in sl["rendered"]:
                g = _gray(ctx, s)
                if g is None:
                    continue
                F = g.shape[1]
                yp = placed[min(placed.shape[0] - 1, s)].copy()
                inb = np.array([f in cf for f in range(F)])
                y_in = np.where(inb, yp, np.nan)
                curves = [(y_in, C_PLACED, "dashed", "placed top (band)")]
                if served is not None:
                    curves.insert(0, (np.where(inb, np.nan, served[min(served.shape[0] - 1, s)]), C_SERVED, "solid", "served top"))
                if post is not None:
                    curves.append((post[min(post.shape[0] - 1, s)], C_BOTTOM, "solid", "served posterior", 0.9))
                out.append(R.image(g, "central" if s == sl["central"] else "most_edited", s, "sagittal",
                                   f"Placed top (magenta, dashed) inside the band, served top (red) outside, posterior (orange); raw slice {s} ({_DISP}).{_above_note(y_in, ctx.D)}",
                                   f"n06_placement_s{s}.png", curves=curves, bands=_band_list(ctx)))
            return out
        images, err = _safe_images(_b)
    return _mk("n06_placement", "auto", "spine", "Top placement inside the crop band", status, reason, summary, description, method,
               numbers, {"crop_max_pad": ctx.p.get("crop_max_pad"), "crop_pad_margin": ctx.p.get("crop_pad_margin")}, images, err,
               source={"manifest": ["oct_iter.crop_reconstruction"], "files": ["border_cache/placed_edges.npz"]})


# ── n07 tissue motion ──────────────────────────────────────────────────────────────────────────────────────
def _node_tissue_motion(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    rec = ctx.it.get("tissue_motion") if isinstance(ctx.it.get("tissue_motion"), dict) else None
    applied = bool(rec and rec.get("applied"))
    pre = rec.get("pre_guide") if rec and isinstance(rec.get("pre_guide"), dict) else None
    numbers = []
    tm = None
    if rec:
        numbers += [_n("applied", applied)]
        if applied:
            numbers += [_n("frame pairs measured", rec.get("pairs_measured")), _n("bands", rec.get("bands")), _n("segments", rec.get("segments")),
                        _n("shift range (applied)", rec.get("shift_range"), "px"), _n("tilt max (applied)", rec.get("tilt_max_px"), "px"),
                        _n("trajectory range", rec.get("trajectory_range"), "px"), _n("residual rms", rec.get("residual_rms_px"), "px")]
            if pre:
                numbers += [_n("pre-guide shift range", pre.get("shift_range"), "px"), _n("pre-guide tilt max", pre.get("tilt_max_px"), "px")]
        else:
            numbers.append(_n("reason", rec.get("reason") or rec.get("error")))
    if ctx.mode == "corrections" and rec is not None:
        tm = _tissue_motion_arrays(ctx)
        if tm is not None and tm[0] is not None and tm[2].get("applied"):
            ref = (pre or rec).get("shift_range")
            try:
                match = bool(ref is not None and max(abs(float(tm[2]["shift_range"][0]) - float(ref[0])),
                                                     abs(float(tm[2]["shift_range"][1]) - float(ref[1]))) < 0.05)
            except (TypeError, ValueError, IndexError):
                match = False
            numbers += [_n("recomputed shift range", tm[2].get("shift_range"), "px"), _n("recompute_matches", match)]
        elif tm is not None:
            numbers.append(_n("recompute", "failed: " + str(tm[2].get("reason") or tm[2].get("error"))))
        else:
            numbers.append(_n("recompute", "raw volume unavailable"))
    if rec is None:
        status = "not_run"
        reason = "automatic run — no reviewer corrections were on record" if ctx.mode != "corrections" else "tissue-motion measurement not recorded"
    elif applied:
        status, reason = "applied", None
    else:
        status, reason = "declined", str(rec.get("reason") or rec.get("error") or "not applied")
    summary = (f"shift {_fmt(rec.get('shift_range'))} px, tilt ≤ {_fmt(rec.get('tilt_max_px'))} px over {rec.get('pairs_measured')} pairs" if applied else "not applied")
    description = ("Shape from lines, motion from tissue: the per-frame rigid move (depth shift a[f] and half-span tilt b[f]) is measured "
                   "from the tissue itself by cross-correlating adjacent B-scans in lateral bands, integrating the lags into a trajectory and "
                   "referencing it to its own best-fit parabola (the dome). No surface is used, so the drawn lines set the SHAPE and the "
                   "tissue sets the MOTION. The plot is recomputed deterministically from the raw volume.")
    method = ((f"Per-frame rigid axial motion (shift a_f, tilt b_f) was measured from the tissue itself by adjacent-frame A-scan "
               f"cross-correlation over {rec.get('bands')} lateral bands ({rec.get('pairs_measured')} frame pairs), integrated to a trajectory "
               f"and referenced to its own best-fit parabola (shift range {_fmt((pre or rec).get('shift_range'))} px, max tilt "
               f"{_fmt((pre or rec).get('tilt_max_px'))} px; residual r.m.s. {_fmt(rec.get('residual_rms_px'), 2)} px).") if applied else
              (f"Tissue-motion measurement was evaluated but declined because {_reason_clause(reason, ctx.cid)}." if rec else ""))
    images, err = [], None
    if R is not None and tm is not None and tm[0] is not None and tm[2].get("applied"):
        def _b():
            a, b = tm[0], tm[1]
            return [R.plot([(a, C_SHIFT, "a[f] shift (px, + = deeper)"), (b, C_TILT, "b[f] tilt (px half-span)")], "plot",
                           "Per-frame rigid move measured from the tissue: shift a[f] (red) and tilt b[f] (blue); shaded = surface-crop band.",
                           "n07_tissue_motion_ab.png", bands=_band_list(ctx),
                           annotate=f"shift {_fmt(tm[2].get('shift_range'))} px · tilt max {_fmt(tm[2].get('tilt_max_px'))} px")]
        images, err = _safe_images(_b)
    params = {k: ctx.p.get(k) for k in ("tissue_motion_lat_step", "tissue_motion_band", "tissue_motion_max_lag", "tissue_motion_min_bands",
                                        "tissue_motion_cut_guard", "tissue_motion_depth_smooth")}
    return _mk("n07_tissue_motion", "auto", "spine", "Tissue-motion measurement (rigid per-frame move)", status, reason, summary,
               description, method, numbers, params, images, err,
               source={"manifest": ["oct_iter.tissue_motion"], "files": ["input/_raw_border.nii.gz (recompute)"]})


# ── n08 band guide ─────────────────────────────────────────────────────────────────────────────────────────
def _node_band_guide(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    tmrec = ctx.it.get("tissue_motion") if isinstance(ctx.it.get("tissue_motion"), dict) else {}
    rec = tmrec.get("band_bottom_guide") if isinstance(tmrec.get("band_bottom_guide"), dict) else None
    applied = bool(rec and rec.get("applied"))
    numbers = []
    if rec:
        numbers.append(_n("applied", applied))
        if applied:
            po = rec.get("posterior_off_parabola_px") if isinstance(rec.get("posterior_off_parabola_px"), dict) else {}
            numbers += [_n("refined frames", rec.get("refined_frames")), _n("drawn-driven frames", rec.get("drawn_driven_frames")),
                        _n("faded frames", rec.get("faded_frames")), _n("rejected laterals", len(rec.get("rejected_laterals") or [])),
                        _n("shift range (delta)", rec.get("shift_range"), "px"), _n("tilt max (delta)", rec.get("tilt_max_px"), "px"),
                        _n("posterior off-parabola before → after", f"{_fmt(po.get('before'), 2)} → {_fmt(po.get('after'), 2)}", "px")]
        else:
            numbers.append(_n("reason", rec.get("reason") or rec.get("error")))
    if rec is None:
        status = "not_run"
        reason = ("automatic run — no reviewer corrections were on record" if ctx.mode != "corrections" else
                  "no band guide evaluated (no surface-crop band or tissue motion not applied)")
    elif applied:
        status, reason = "applied", None
    elif rec.get("error"):
        status, reason = "error", f"band guide raised {rec.get('error')}"
    else:
        status, reason = "declined", str(rec.get("reason") or "not applied")
    summary = (f"refined frames {_runs(rec.get('refined_frames') or [])}, shift {_fmt(rec.get('shift_range'))} px" if applied else
               (f"declined — {reason}" if rec else "not evaluated"))
    description = ("In the crop band the only edge is the posterior, so its smoothness is allowed to refine the band's move: the served "
                   "posterior is carried by the measured move, each lateral's posterior is fitted to its parabola over the un-cropped frames, "
                   "and the band's per-frame shift is refined (drawn bottom lines decide it where drawn, fading over neighbouring frames).")
    method = ((f"The band's move was refined by the posterior-surface guide on frames {_runs(rec.get('refined_frames') or [])} "
               f"(shift range {_fmt(rec.get('shift_range'))} px).") if applied else
              (f"The posterior-surface guide was evaluated but declined because {_reason_clause(reason, ctx.cid)}." if rec and status == "declined" else ""))
    images, err = [], None
    if R is not None and applied:
        def _b():
            tm = _tissue_motion_arrays(ctx)
            bg = _band_guide_arrays(ctx)
            if tm is None or tm[0] is None or bg is None or bg[0] is None:
                raise RuntimeError("band guide delta not recomputable (raw volume or served posterior missing)")
            a = tm[0]; da = bg[0]
            return [R.plot([(a, "#ff9090", "a[f] before guide", "dotted"), (a + da, C_SHIFT, "a[f] after guide"), (da, C_TILT, "guide delta")],
                           "plot", "Per-frame shift a[f] before (dotted) and after the posterior guide; shaded = surface-crop band.",
                           "n08_band_guide_ab.png", bands=_band_list(ctx))]
        images, err = _safe_images(_b)
    elif R is not None and status == "declined" and ctx.surf("posterior_edges") is not None and ctx.crop_frames:
        def _b2():   # what the guide would have judged: the served posterior vs its parabola over the un-cropped frames
            post = ctx.surf("posterior_edges"); cf = set(ctx.crop_frames)
            s = sl["rendered"][0]
            g = _gray(ctx, s)
            if g is None:
                raise RuntimeError("raw volume unavailable")
            y = post[min(post.shape[0] - 1, s)].astype(np.float64)
            F = y.size; fr = np.arange(F, dtype=np.float64)
            inb = np.array([f in cf for f in range(F)])
            ok = np.isfinite(y) & ~inb
            if ok.sum() < 3:
                raise RuntimeError("fewer than 3 un-cropped posterior frames on this slice")
            par = np.polyval(np.polyfit(fr[ok], y[ok], 2), fr)
            res = y - par; inres = res[inb & np.isfinite(y)]
            rms = float(np.sqrt(np.mean(inres ** 2))) if inres.size else float("nan")
            return [R.image(g, "primary", s, "sagittal",
                            f"Band guide declined ({reason}): served posterior (orange) vs its parabola fitted over the un-cropped frames "
                            f"(green dashed) on raw slice {s} ({_DISP}); off-parabola inside the band {_fmt(rms, 2)} px r.m.s.",
                            f"n08_band_guide_s{s}.png", curves=[(y, C_BOTTOM, "solid", "served posterior"), (par, C_TARGET, "dashed", "parabola (un-cropped frames)")],
                            bands=_band_list(ctx))]
        images, err = _safe_images(_b2)
    return _mk("n08_band_guide", "decision", "spine", "Band bottom guide (posterior-driven refinement)", status, reason, summary, description,
               method, numbers, {"band_bottom_guide": ctx.p.get("band_bottom_guide")}, images, err,
               source={"manifest": ["oct_iter.tissue_motion.band_bottom_guide"], "files": ["border_cache/posterior_edges.npz (recompute)"]})


# ── n09 canvas extend ──────────────────────────────────────────────────────────────────────────────────────
def _node_canvas_extend(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    rec = ctx.it.get("canvas_extend") if isinstance(ctx.it.get("canvas_extend"), dict) else None
    pad = ctx.pad
    numbers = []
    if rec:
        numbers += [_n("pad", pad, "rows"), _n("reason", rec.get("reason")), _n("depth before → after", f"{rec.get('depth_before')} → {rec.get('depth_after')}"),
                    _n("move allowance", rec.get("move_allowance", 0), "rows"), _n("fill", ctx.p.get("corrections_pad_fill", "zeros"))]
    if rec is None:
        status = "not_run"
        reason = "automatic run — no reviewer corrections were on record" if ctx.mode != "corrections" else "canvas extension not recorded"
    elif pad > 0:
        status, reason = "applied", None
    else:
        status, reason = "declined", "no extension needed (corrected apex stays inside the window)"
    summary = f"+{pad} rows ({rec.get('depth_before')} → {rec.get('depth_after')})" if rec and pad > 0 else "no extension"
    description = ("When the corrected apex would sit above the top of the window (a drawn or placed point at a negative row, or the "
                   "measured move lifting it there), zero rows are added ABOVE the volume so nothing is truncated. Every row number in the "
                   "corrected volume is the raw row plus this pad.")
    method = (f"The canvas was extended by {pad} rows ({rec.get('depth_before')}→{rec.get('depth_after')}) because {_reason_clause(rec.get('reason'), ctx.cid)}, "
              f"filled with {ctx.p.get('corrections_pad_fill', 'zeros')}." if rec and pad > 0 else
              ("No canvas extension was needed." if rec else ""))
    images, err = [], None
    if R is not None and status == "applied":
        def _b():
            s = sl["rendered"][0]
            g = _gray(ctx, s); gc = _gray_cor(ctx, s)
            if g is None or gc is None:
                raise RuntimeError("raw or corrected volume unavailable")
            padded = np.zeros((g.shape[0] + pad, g.shape[1]), np.float32); padded[pad:] = g
            served = ctx.surf("provided_edges")
            curves_l = [(served[min(served.shape[0] - 1, s)] + pad, C_SERVED, "solid", "served + pad")] if served is not None else []
            return [R.pair({"img": padded, "curves": curves_l, "xlabel": "raw + pad"},
                           {"img": gc, "xlabel": "corrected"}, "primary", s, "sagittal",
                           f"Left: raw slice {s} with {pad} zero rows added on top (served surface + pad in red); right: the corrected slice ({_DISP}).",
                           f"n09_canvas_extend_s{s}.png")]
        images, err = _safe_images(_b)
    return _mk("n09_canvas_extend", "decision", "spine", "Canvas extension (zero rows above the window)", status, reason, summary,
               description, method, numbers, {"crop_max_pad": ctx.p.get("crop_max_pad"), "crop_pad_margin": ctx.p.get("crop_pad_margin"),
                                              "corrections_pad_fill": ctx.p.get("corrections_pad_fill")}, images, err,
               source={"manifest": ["oct_iter.canvas_extend"], "files": []})


# ── n10 flatten ────────────────────────────────────────────────────────────────────────────────────────────
def _node_flatten(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    rec = ctx.it.get("flatten") if isinstance(ctx.it.get("flatten"), dict) else None
    ds = ctx.it.get("detector_stages") if isinstance(ctx.it.get("detector_stages"), dict) else {}
    rv = ctx.it.get("roughness_veto") if isinstance(ctx.it.get("roughness_veto"), dict) else {}
    warnings = []
    numbers = []
    if rec:
        numbers += [_n("mode", rec.get("mode")), _n("drawn frames", rec.get("drawn_frames")), _n("cropped frames", rec.get("cropped_frames")),
                    _n("shift range", rec.get("shift_range"), "px"), _n("tilt max", rec.get("tilt_max_px"), "px"),
                    _n("target row min / max (padded)", f"{_fmt(rec.get('min_target_row'))} / {_fmt(rec.get('max_target_row'))}", "px")]
        try:
            if rec.get("min_target_row") is not None and float(rec["min_target_row"]) < 0:
                warnings.append(f"flatten target above the padded canvas (min_target_row = {float(rec['min_target_row']):.1f}): the deepest placed cells were truncated")
        except (TypeError, ValueError):
            pass
    kind = _flatten_kind(ctx.it)
    label = _FLATTEN_LABEL[kind]
    if rec is None:
        status = "not_run"
        reason = ("automatic run — the rigid flatten passes apply instead" if ctx.mode != "corrections"
                  else "the run recorded no flatten block (older pipeline version)")
    else:
        status, reason = "applied", None
    summary = (f"{label} · mode '{rec.get('mode')}' · shift {_fmt(rec.get('shift_range'))} px · tilt ≤ {_fmt(rec.get('tilt_max_px'))} px" if rec else "not run")
    description = ("Each B-scan is moved RIGIDLY (one depth shift and one tilt per frame — never a per-column deformation, because a B-scan is "
                   "captured instantaneously) so that the served/placed surface lands on its target: target row = surface + pad + a[f] + b[f]·x. "
                   + ("On this run a[f], b[f] are the TISSUE-MEASURED move (tissue-motion stage above, band guide included when applied)."
                      if kind == "tissue" else
                      "On this run a[f], b[f] are the DRAWN-LINE move: a per-frame minimax fit of the served/drawn surface to its quadratic."
                      if kind == "drawn" else "The move source was not recorded.")
                   + " The green curve is that target on the corrected slice.")
    skipped = ", ".join(ds.get("skipped") or []) if ds else ""
    method = ((f"Each B-scan was then shifted and tilted rigidly to its target ({label}, mode '{rec.get('mode')}'; shift {_fmt(rec.get('shift_range'))} px, "
               f"tilt ≤ {_fmt(rec.get('tilt_max_px'))} px; target row = surface + pad + a[f] + b[f]·x).") if rec else "")
    images, err = [], None
    if R is not None and rec is not None:
        def _b():
            out = []
            surf, src = ctx.served_for_flatten()
            mv = _move_arrays(ctx) if kind != "drawn" else None
            fresh = _applied_move_fresh(ctx)
            move = ctx.surf("applied_move") if fresh else None
            served = ctx.surf("provided_edges")
            for s in sl["rendered"]:
                gc = _gray_cor(ctx, s)
                if gc is None:
                    continue
                Dc, F = gc.shape
                curves = []
                if surf is not None and mv is not None:
                    a, b, guided = mv
                    if a.size == F:
                        L = surf.shape[0]
                        x = (s - (L - 1) / 2.0) / max(1e-9, (L - 1) / 2.0)
                        tgt = surf[min(L - 1, s)] + ctx.pad + a + b * x
                        curves.append((tgt, C_TARGET, "solid", f"flatten target ({src}" + (", post-guide move)" if guided else ")")))
                if served is not None and move is not None and move.shape == served.shape:
                    carried = np.clip(served[min(served.shape[0] - 1, s)] + ctx.pad + move[min(move.shape[0] - 1, s)], 0, Dc - 1)
                    curves.append((carried, C_AUTO, "dashed", "served surface carried", 0.9))
                cap = (f"Corrected slice {s} ({_DISP}) with the flatten target (green) = surface + pad + a[f] + b[f]·x"
                       + (" and the served surface carried by the measured move (white dashed)." if move is not None else "; applied_move.npz is stale or absent, so no carried surface is drawn."))
                out.append(R.image(gc, "central" if s == sl["central"] else "most_edited", s, "sagittal", cap, f"n10_flatten_s{s}.png", curves=curves, bands=_band_list(ctx)))
            return out
        images, err = _safe_images(_b)
    return _mk("n10_flatten", "auto", "spine", f"Rigid per-frame flatten ({label})", status, reason, summary, description, method, numbers,
               {"flatten_motion_source": ctx.p.get("flatten_motion_source", "tissue"), "flatten_rigid_quad": ctx.p.get("flatten_rigid_quad", True),
                "tissue_motion_max_tilt_px": ctx.p.get("tissue_motion_max_tilt_px")}, images, err, warnings,
               source={"manifest": ["oct_iter.flatten"], "files": ["border_cache/placed_edges.npz", "border_cache/applied_move.npz"]})


def _node_detector_stages(ctx: RunCtx, sl: dict, R) -> dict:
    ds = ctx.it.get("detector_stages") if isinstance(ctx.it.get("detector_stages"), dict) else None
    numbers = [_n("skipped stages", ds.get("skipped")), _n("reason", ds.get("reason"))] if ds else []
    if ds is None:
        status, reason = "not_run", ("automatic run — the detector stages ran as the rigid chain" if ctx.mode != "corrections" else "not recorded")
    else:
        status, reason = "declined", str(ds.get("reason") or "skipped")
    summary = ("skipped: " + ", ".join(ds.get("skipped") or [])) if ds else "—"
    return _mk("n10a_detector_stages", "decision", "spine", "Detector-driven refinement stages", status, reason, summary,
               "The detector-driven rigid stages (height refine, frame derotate, frame refine) re-detect the surface on the corrected volume "
               "and would move frames on that basis; on a corrections run the measured move already carries the correction, so they are skipped.",
               (f"The detector-driven refinement stages ({', '.join(ds.get('skipped') or [])}) were skipped because {_reason_clause(ds.get('reason'), ctx.cid)}." if ds else ""),
               numbers, {"tissue_motion_skip_detector_stages": ctx.p.get("tissue_motion_skip_detector_stages")}, [], None,
               source={"manifest": ["oct_iter.detector_stages"], "files": []})


def _node_roughness_veto(ctx: RunCtx, sl: dict, R) -> dict:
    rv = ctx.it.get("roughness_veto") if isinstance(ctx.it.get("roughness_veto"), dict) else None
    waived = bool(rv and rv.get("waived"))
    step = str(rv.get("step_guard") or "") if rv else ""
    step_on = step.lower().startswith("on")
    ds = ctx.it.get("detector_stages") if isinstance(ctx.it.get("detector_stages"), dict) else {}
    skipped = set(ds.get("skipped") or [])
    marked = []
    for st in (rv.get("stages") or []) if rv else []:
        r = ctx.it.get(st) if isinstance(ctx.it.get(st), dict) else None
        if st in skipped or r is None:
            marked.append(f"{st} (skipped)")
        elif r.get("applied"):
            marked.append(f"{st} (applied)")
        else:
            marked.append(f"{st} (declined)")
    why = ("waived because the run flattened to a reviewer-drawn edge — the drawn line is the target, so across-frame roughness is not a veto criterion"
           if waived else "enforced on every rigid stage")
    numbers = [_n("waived", waived), _n("stages evaluated under the waiver", marked), _n("tissue-step guard", "on" if step_on else ("off" if rv else None))] if rv else []
    if rv is None:
        status, reason = "not_run", ("automatic run — roughness vetoes were enforced per stage" if ctx.mode != "corrections" else "not recorded")
    elif waived:
        status, reason = "declined", why
    else:
        status, reason = "applied", None
    summary = ("waived" if waived else "enforced") if rv else "—"
    return _mk("n10b_roughness_veto", "decision", "spine", "Roughness veto", status, reason, summary,
               "Every rigid stage normally declines a move that makes the across-frame surface rougher. When the run flattens to a drawn edge the "
               "reviewer's line is the truth, so that veto is waived; the trace-free tissue-step guard stays on regardless.",
               (f"Roughness vetoes were {why}; the trace-free tissue-step guard stayed {'on' if step_on else 'off'}"
                + (f" (stages evaluated under the waiver: {', '.join(marked)})" if marked else "") + "." if rv else ""),
               numbers, {}, [], None, source={"manifest": ["oct_iter.roughness_veto", "oct_iter.detector_stages"], "files": []})


def _node_sag_quad_align(ctx: RunCtx, sl: dict, R) -> dict:
    rec = ctx.it.get("sagittal_quad_align") if isinstance(ctx.it.get("sagittal_quad_align"), dict) else None
    numbers = [_n("applied", bool(rec.get("applied"))), _n("reason", rec.get("reason"))] if rec else []
    if rec is None:
        status, reason = "not_run", ("automatic run — no reviewer corrections were on record" if ctx.mode != "corrections" else "not recorded")
    elif rec.get("applied"):
        status, reason = "applied", None
    else:
        status, reason = "declined", str(rec.get("reason") or "not applied")
    return _mk("n10c_sag_quad_align", "decision", "spine", "Sagittal quadratic align (detector-driven re-flatten)", status, reason,
               ("applied" if rec and rec.get("applied") else "off by default") if rec else "—",
               "A second, detector-driven re-flatten of every sagittal slice to its best-fit quadratic. Off by default on the corrections path "
               "because it can undo the drawn-line flatten; it can be enabled per case (sag_quad_align).",
               (f"A detector-driven sagittal quadratic re-alignment was {'applied' if rec.get('applied') else 'evaluated but declined because ' + _reason_clause(rec.get('reason'), ctx.cid)}." if rec else ""),
               numbers, {"sag_quad_align": ctx.p.get("sag_quad_align")}, [], None, source={"manifest": ["oct_iter.sagittal_quad_align"], "files": []})


# ── n11 pane edits ─────────────────────────────────────────────────────────────────────────────────────────
def _node_pane_edits(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    """Corrected-pane edits. CONSUMED edits are those in oct_iter.corrected_fold (folded=True) — the fold moves them into
    the original-pane inputs before the worker runs and clears them from oct_params. corrected_* keys still in
    oct_params are therefore PENDING: drawn after the last run, not consumed by it (H1)."""
    p = ctx.p
    pend = _pending_pane_edits(ctx)
    fold = ctx.it.get("corrected_fold") if isinstance(ctx.it.get("corrected_fold"), dict) else None
    consumed = bool(fold and fold.get("folded"))
    fold_mode = str(fold.get("mode") or "line") if fold else None
    fel = sorted(oct_mod.folded_laterals(p))
    f_lats = sorted(int(x) for x in (fold.get("laterals") or []) if str(x).lstrip("-").isdigit()) if fold else []
    f_bottom = sorted(int(x) for x in (fold.get("bottom_folded_slices") or []) if str(x).lstrip("-").isdigit()) if fold else []
    f_excl = sorted(int(x) for x in (fold.get("excluded_laterals") or []) if str(x).lstrip("-").isdigit()) if fold else []
    f_tr = fold.get("transform") if fold and isinstance(fold.get("transform"), dict) else {}
    numbers = []
    if consumed:
        numbers += [_n("consumed on the last run", True), _n("fold mode", fold_mode), _n("folded points", fold.get("n_points")),
                    _n("folded laterals", _runs(f_lats) if f_lats else "—"), _n("bottom points folded", fold.get("n_bottom_points", 0)),
                    _n("bottom folded slices", _runs(f_bottom) if f_bottom else "—"),
                    _n("laterals excluded from the per-frame fit", _runs(f_excl or fel) if (f_excl or fel) else "—"),
                    _n("pinned (accurate) laterals", _runs(fold.get("pinned_laterals") or []) if fold.get("pinned_laterals") else "—"),
                    _n("backup", (fold.get("fold") or {}).get("backup") if isinstance(fold.get("fold"), dict) else fold.get("backup"))]
        if f_tr:
            numbers += [_n("fitted transform rounds", f_tr.get("rounds")), _n("fitted shift range", f_tr.get("shift_range"), "px"),
                        _n("fitted tilt max", f_tr.get("tilt_max_px"), "px")]
    else:
        numbers.append(_n("consumed on the last run", False))
    n_pend = pend["n_points"] + pend["n_bottom_points"] + pend["n_marks"]
    numbers += [_n("PENDING edge laterals / points (not yet consumed)", f"{pend['n_slices']} / {pend['n_points']}"),
                _n("PENDING bottom laterals / points", f"{pend['n_bottom_slices']} / {pend['n_bottom_points']}"),
                _n("PENDING defect marks", pend["n_marks"]), _n("PENDING accurate laterals", pend["accurate_laterals"] or "—")]
    warnings = []
    if ctx.mode != "corrections":
        status, reason = "not_run", ("automatic run — no reviewer corrections were on record" if not n_pend else
                                     f"pending: {pend['n_points']} edge points on {pend['n_slices']} slices drawn on the corrected pane have not been consumed by a run yet")
    elif consumed:
        status, reason = "applied", None
        if n_pend:
            warnings.append(f"{pend['n_points']} edge points on {pend['n_slices']} slices (+ {pend['n_bottom_points']} bottom points, {pend['n_marks']} marks) "
                            "drawn since the last run are PENDING — not consumed yet")
    elif n_pend:
        status, reason = "not_run", (f"pending: {pend['n_points']} edge points on {pend['n_slices']} slices, {pend['n_bottom_points']} bottom points and "
                                     f"{pend['n_marks']} defect marks on the corrected pane have not been consumed by a run yet")
    else:
        status, reason = "not_run", "no corrected-pane edits on record"
    summary = ((f"consumed: {fold.get('n_points')} pts on {len(f_lats)} laterals ({fold_mode})" if consumed else "none consumed")
               + (f" · PENDING {pend['n_points']} edge / {pend['n_bottom_points']} bottom pts" if n_pend else ""))
    description = ("Edits the reviewer made on the CORRECTED pane (rows in the padded corrected canvas). In line mode they are consumed as LINE "
                   "ground truth: folded back into the original-pane top / bottom lines (chord-guarded, folded laterals kept out of the per-frame "
                   "fit) before the worker runs, then cleared. Edits still on record were drawn after the last run and are pending.")
    method = ""
    if consumed and ctx.mode == "corrections":
        if fold_mode == "transform":
            method = (f"Corrected-pane edits ({fold.get('n_points')} points on {len(f_lats)} slices) were fitted to a per-frame rigid shift and tilt "
                      f"({f_tr.get('rounds', '?')} round(s); shift range {_fmt(f_tr.get('shift_range'))} px, max tilt {_fmt(f_tr.get('tilt_max_px'))} px) "
                      "and the verified lines were folded into the original-pane anchors.")
        else:
            method = (f"Corrected-pane edits ({fold.get('n_points')} points on {len(f_lats)} slices"
                      + (f"; {fold.get('n_bottom_points')} bottom points on {len(f_bottom)} slices" if f_bottom else "")
                      + ") were folded into the original-pane lines as line ground truth before the run"
                      + (f", with laterals {_runs(f_excl or fel)} kept out of the per-frame fit" if (f_excl or fel) else "") + ".")
    images, err = [], None
    if R is not None and ctx.cor_path is not None and (consumed or n_pend):
        def _b():
            ea, _ = _anchor_points(p.get("corrected_edge_anchors")); pa, _ = _anchor_points(p.get("corrected_post_anchors"))
            dm = p.get("corrected_defect_marks") if isinstance(p.get("corrected_defect_marks"), dict) else {}
            if sl.get("requested") is not None:
                s = int(sl["rendered"][0])
            elif n_pend:
                s = int(sorted(ea.keys())[0]) if ea else (int(sorted(pa.keys())[0]) if pa else sl["most_edited"])
            else:
                s = int(f_lats[0]) if f_lats else sl["most_edited"]
            gc = _gray_cor(ctx, s)
            if gc is None:
                raise RuntimeError("corrected volume unavailable")
            Dc = gc.shape[0]
            curves = []; pts = []
            served = ctx.surf("provided_edges"); move = ctx.surf("applied_move") if _applied_move_fresh(ctx) else None
            if served is not None and move is not None and move.shape == served.shape:
                curves.append((np.clip(served[min(served.shape[0] - 1, s)] + ctx.pad + move[min(move.shape[0] - 1, s)], 0, Dc - 1), C_AUTO, "dashed", "carried served surface", 0.9))
                if consumed and s in f_lats:   # the folded line now lives in border_anchors: show it carried by the applied move
                    apts, _ = _anchor_points(ctx.anchors)
                    fr, rows = _drawn_pts(apts, s)
                    if fr.size:
                        mv = move[min(move.shape[0] - 1, s)]
                        pts.append((fr, rows + ctx.pad + mv[fr.astype(int)], C_SERVED, "folded line (carried)", 2.0))
            fr, rows = _drawn_pts(ea, s); fb, rb = _drawn_pts(pa, s)
            if fr.size:
                pts.append((fr, rows, C_SERVED, "PENDING edge points", 2.5))
            if fb.size:
                pts.append((fb, rb, C_BOTTOM, "PENDING bottom points", 2.5))
            bands = []
            for k, segs in dm.items():
                try:
                    if int(k) == s:
                        for sg in segs:
                            bands.append((int(sg[0]), int(sg[1]), C_CROP, "pending defect mark"))
                except (TypeError, ValueError, IndexError):
                    continue
            cap = (f"Corrected slice {s} ({_DISP}): "
                   + ("the folded (consumed) line carried by the applied move (red dots)" if (consumed and s in f_lats and any(x[3].startswith('folded') for x in pts)) else "")
                   + ("; " if (consumed and s in f_lats and fr.size) else "")
                   + (f"PENDING corrected-pane points not yet consumed (red top, orange bottom; padded rows)" if (fr.size or fb.size) else "")
                   + (" and the carried served surface (white dashed)." if curves else "."))
            return [R.image(gc, "most_edited", s, "sagittal", cap, f"n11_pane_edits_s{s}.png", curves=curves, points=pts, bands=bands)]
        images, err = _safe_images(_b)
    node = _mk("n11_pane_edits", "user", "input", "Corrected-pane edits (line ground truth)", status, reason, summary, description,
               method, numbers, {"corrected_edit_mode": p.get("corrected_edit_mode", "line")}, images, err, warnings,
               source={"manifest": ["oct_iter.corrected_fold", "oct_params.corrected_edge_anchors", "oct_params.corrected_post_anchors",
                                    "oct_params.corrected_defect_marks", "oct_params.flatten_exclude_laterals"], "files": []})
    node["pending"] = pend
    node["consumed"] = ({"mode": fold_mode, "n_points": fold.get("n_points"), "laterals": f_lats, "bottom_folded_slices": f_bottom,
                         "excluded_laterals": f_excl or fel} if consumed else None)
    return node


def _node_edit_transform(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    rec = ctx.it.get("edit_transform") if isinstance(ctx.it.get("edit_transform"), dict) else None
    et = ctx.p.get("edit_transform") if isinstance(ctx.p.get("edit_transform"), dict) else None
    numbers = []
    if rec:
        numbers += [_n("applied", bool(rec.get("applied"))), _n("rounds", rec.get("rounds"))]
        if rec.get("applied"):
            numbers += [_n("max shift", rec.get("max_shift_px"), "px"), _n("max tilt", rec.get("max_tilt_px"), "px"), _n("frames moved", rec.get("frames_moved"))]
        else:
            numbers.append(_n("reason", rec.get("reason")))
    if et:
        numbers.append(_n("history rounds on record", len(et.get("history") or [])))
    if rec is None and et is None:
        status, reason = "not_run", ("automatic run — no reviewer corrections were on record" if ctx.mode != "corrections" else "no legacy edit transform on record")
    elif rec and rec.get("applied"):
        status, reason = "applied", None
    elif rec:
        status, reason = "declined", str(rec.get("reason") or "not applied")
    else:
        status, reason = "declined", "a stored transform is on record but the last run did not evaluate it"
    summary = (f"{rec.get('rounds')} round(s) · {'applied' if rec.get('applied') else 'kept on record, not applied'}" if rec else ("on record" if et else "—"))
    description = ("The LEGACY corrected-pane mechanism: pane edits fitted to a per-frame rigid shift+tilt accumulated in oct_params.edit_transform. "
                   "Demoted (2026-09-04): in line mode the edits are folded as ground truth instead and any stored transform stays on record unapplied.")
    method = ((f"A legacy edit transform of {rec.get('rounds')} rounds remained on record but was not applied." if rec and not rec.get("applied") else
               f"A legacy edit transform of {rec.get('rounds')} rounds was applied (max shift {_fmt(rec.get('max_shift_px'))} px).") if rec else "")
    images, err = [], None
    if R is not None and et and et.get("shift"):
        def _b():
            return [R.plot([(np.asarray(et.get("shift"), np.float64), C_SHIFT, "stored shift[f]"), (np.asarray(et.get("tilt") or [], np.float64), C_TILT, "stored tilt[f]")],
                           "plot", "Legacy edit transform on record (not applied in line mode): per-frame shift (red) and tilt (blue).", "n11b_edit_transform.png")]
        images, err = _safe_images(_b)
    return _mk("n11b_edit_transform", "decision", "spine", "Legacy edit transform", status, reason, summary, description, method, numbers,
               {"corrected_edit_mode": ctx.p.get("corrected_edit_mode", "line")}, images, err,
               source={"manifest": ["oct_params.edit_transform", "oct_iter.edit_transform"], "files": []})


# ── n12 output ─────────────────────────────────────────────────────────────────────────────────────────────
def _node_output(ctx: RunCtx, sl: dict, R: _Renderer | None) -> dict:
    it = ctx.it
    fq = it.get("final_qa") if isinstance(it.get("final_qa"), dict) else {}
    dims = None; dtype = None
    if ctx.cor_path is not None:
        try:
            import nibabel as nib
            im = nib.load(str(ctx.cor_path)); dims = [int(x) for x in im.shape[:3]]; dtype = str(im.get_data_dtype())
        except Exception:  # noqa: BLE001
            pass
    numbers = [_n("corrected volume", str(ctx.cor_path) if ctx.cor_path else None), _n("dims corrected", dims), _n("dtype", dtype)]
    if fq:
        numbers += [_n("boundary deviation (coverage-corrected)", _r(fq.get("dev")), "px"), _n("axial", _r(fq.get("axial")), "px"), _n("score", _r(fq.get("score"))),
                    _n("coverage", fq.get("coverage")), _n("path", fq.get("path")), _n("max jitter", "unknown" if fq.get("max_jitter") is None else fq.get("max_jitter")),
                    _n("needs review", bool(fq.get("needs_review"))), _n("review reasons", fq.get("review_reasons") or [])]
    if isinstance(it.get("crop"), dict):
        numbers.append(_n("zeroed voxels (crop)", it["crop"].get("n_voxels")))
    numbers += [_n("stopped", it.get("stopped")), _n("passes", it.get("passes"))]
    if ctx.mode == "none":
        status, reason = "missing_data", "not preprocessed yet"
    elif ctx.cor_path is None:
        status, reason = "missing_data", "manifest.input_volume missing on disk"
    else:
        status, reason = "applied", None
    summary = (f"{dims} · dev {_fmt(fq.get('dev'), 2)} px · " + ("flagged: " + ", ".join(fq.get("review_reasons") or []) if fq.get("needs_review") else "not flagged")) if dims and fq else (str(dims) if dims else "no output")
    description = ("The corrected volume that becomes the case's working volume for segmentation, with the final QA: coverage-corrected "
                   "boundary deviation of the re-detected surface from its quadratic (dev), the axial metric and the review flags.")
    method = ((f"The corrected volume ({'×'.join(str(d) for d in dims) if dims else '?'}) had a coverage-corrected boundary deviation of "
               f"{_fmt(fq.get('dev'), 2)} px (axial {_fmt(fq.get('axial'), 2)} px; score {_fmt(fq.get('score'), 2)}; "
               + ("flagged for review: " + ", ".join(fq.get("review_reasons") or []) if fq.get("needs_review") else "not flagged") + ").") if fq and status == "applied" else "")
    images, err = [], None
    if R is not None and status == "applied":
        def _b():
            out = []
            for s in sl["rendered"]:
                gc = _gray_cor(ctx, s)
                if gc is None:
                    continue
                out.append(R.image(gc, "central" if s == sl["central"] else "most_edited", s, "sagittal", f"Corrected sagittal slice {s} ({_DISP}).", f"n12_output_s{s}.png"))
            fa = int((ctx.F or 1) // 2)
            ga = _gray_axial(ctx.cor, fa)
            if ga is not None:
                out.append(R.image(ga, "primary", fa, "axial", f"Corrected axial B-scan (frame {fa}); lateral 0 at right as in the app.", f"n12_output_axial_f{fa}.png", xlabel="lateral (A-scan)"))
            return out
        images, err = _safe_images(_b)
    return _mk("n12_output", "output", "spine", "Corrected volume + final QA", status, reason, summary, description, method, numbers,
               {"qa_dev_flag": ctx.p.get("qa_dev_flag")}, images, err, source={"manifest": ["input_volume", "oct_iter.final_qa", "oct_iter.crop"], "files": ["input_volume"]})


# ── automatic-chain nodes ───────────────────────────────────────────────────────────────────────────────────
def _stage_status(rec, ctx: RunCtx, auto_missing_reason="not recorded on this run"):
    if rec is None:
        if ctx.mode == "corrections":
            ds = ctx.it.get("detector_stages") if isinstance(ctx.it.get("detector_stages"), dict) else {}
            return "not_run", str(ds.get("reason") or "corrections run — this stage of the automatic chain did not run")
        return "not_run", auto_missing_reason
    if rec.get("applied"):
        return "applied", None
    return "declined", str(rec.get("reason") or "not applied")


def _rec_numbers(rec, keys) -> list:
    return [_n(k.replace("_", " "), rec.get(k)) for k in keys if rec is not None and k in rec]


def _node_axial_motion_correct(ctx: RunCtx, sl: dict, R) -> dict:
    rec = ctx.it.get("axial_motion_correct") if isinstance(ctx.it.get("axial_motion_correct"), dict) else None
    status, reason = _stage_status(rec, ctx)
    numbers = _rec_numbers(rec, ("applied", "frames_adjusted", "motion_std", "max_shift", "reason"))
    images, err = [], None
    if R is not None and rec and rec.get("shift"):
        images, err = _safe_images(lambda: [R.plot([(np.asarray(rec["shift"], np.float64), C_SHIFT, "AMC shift[f]")], "plot",
                                                   "Axial motion correction: per-frame depth shift.", "a02_axial_motion_correct_shift.png")])
    return _mk("a02_axial_motion_correct", "auto", "spine", "Axial motion correction (rigid per-frame shift)", status, reason,
               (f"{rec.get('frames_adjusted')} frames, max {_fmt(rec.get('max_shift'))} px" if rec and rec.get("applied") else (reason or "")),
               "Per-frame rigid depth shift that removes the axial motion between B-scans before flattening (the surface's frame-to-frame jitter).",
               (f"Axial motion was corrected by a rigid per-frame shift ({rec.get('frames_adjusted')} frames adjusted, max {_fmt(rec.get('max_shift'))} px)." if rec and rec.get("applied") else
                (f"Axial motion correction was evaluated but declined because {_reason_clause(reason, ctx.cid)}." if rec else "")),
               numbers, {"axial_motion_correct": ctx.p.get("axial_motion_correct")}, images, err, source={"manifest": ["oct_iter.axial_motion_correct"], "files": []})


def _node_flatten_passes(ctx: RunCtx, sl: dict, R) -> dict:
    it = ctx.it
    stopped = str(it.get("stopped") or "")
    scored = bool(it.get("metrics"))
    has = ctx.mode == "automatic" and it.get("passes") is not None and stopped != "surface_crop" and scored
    numbers = [_n("passes", it.get("passes")), _n("best pass", it.get("best_pass")), _n("stopped", stopped),
               _n("boundary deviation per pass", [_r(x) for x in (it.get("metrics") or [])], "px"),
               _n("axial per pass", [_r(x) for x in (it.get("axial_metrics") or [])], "px"), _n("scores", [_r(x) for x in (it.get("scores") or [])])] if has else []
    if has:
        status, reason = "applied", None
    elif ctx.mode == "automatic" and stopped == "surface_crop":
        status, reason = "not_run", "surface-crop branch taken — the clipped apex was reconstructed instead and no keep-best pass was scored (see the surface-crop stage)"
    elif ctx.mode == "automatic" and it.get("passes") is not None:
        status, reason = "not_run", f"no scored pass on record (stopped '{stopped}')"
    else:
        status, reason = "not_run", ("corrections run — a single rigid flatten to the served surface replaces the keep-best passes" if ctx.mode == "corrections" else "scan has not been preprocessed")
    images, err = [], None
    if R is not None and has:
        def _b():
            series = [(np.asarray([float(x) for x in it.get("metrics") or []], np.float64), C_SERVED, "boundary deviation (px)")]
            if it.get("axial_metrics"):
                series.append((np.asarray([float(x) for x in it["axial_metrics"]], np.float64), C_BASELINE, "axial (px)"))
            if it.get("scores"):
                series.append((np.asarray([float(x) for x in it["scores"]], np.float64), C_TARGET, "score", "dashed"))
            return [R.plot(series, "plot", f"Keep-best iteration: boundary deviation (red), axial (blue) and score (green) per pass; index 0 = before "
                           f"the first pass, best pass {it.get('best_pass')}, stopped '{stopped}'.", "a03_flatten_passes_metrics.png",
                           xlabel="pass", invert_x=False, ylabel="px / score")]
        images, err = _safe_images(_b)
    return _mk("a03_flatten_passes", "auto", "spine", "Flatten passes (keep-best iteration)", status, reason,
               (f"{it.get('passes')} pass(es), kept {it.get('best_pass')}, stop '{stopped}'" if has else ("surface-crop branch" if stopped == "surface_crop" else "—")),
               "Iterative flatten of every sagittal slice to its best-fit quadratic by rigid axial shifts; the pass with the lowest boundary deviation is kept.",
               (f"The volume was flattened in {it.get('passes')} passes (best pass {it.get('best_pass')}, stopped '{stopped}')." if has else ""),
               numbers, {"oct_max_iterations": ctx.m.get("oct_max_iterations")}, images, err, source={"manifest": ["oct_iter.passes", "oct_iter.metrics", "oct_iter.best_pass"], "files": []})


def _rigid_stage(nid, title, key, ctx: RunCtx, R, plot_key=None, method_name="") -> dict:
    rec = ctx.it.get(key) if isinstance(ctx.it.get(key), dict) else None
    status, reason = _stage_status(rec, ctx)
    numbers = _rec_numbers(rec, ("applied", "frames_adjusted", "frames_rotated", "frames_moved", "max_jitter", "max_deg", "max_shift", "iters",
                                 "rough_before", "rough_after", "rougher_worst_px", "frames_over_25px", "dev_rms", "reason"))
    images, err = [], None
    if R is not None and rec and plot_key and rec.get(plot_key):
        images, err = _safe_images(lambda: [R.plot([(np.asarray(rec[plot_key], np.float64), C_SHIFT, f"{key} {plot_key}[f]")], "plot",
                                                   f"{title}: per-frame {plot_key}.", f"{nid}_{plot_key}.png")])
    summ = (f"rough {_fmt(rec.get('rough_before'), 3)} → {_fmt(rec.get('rough_after'), 3)}" if rec and rec.get("applied") else (reason or ""))
    return _mk(nid, "decision", "spine", title, status, reason, summ,
               f"{title}: a detector-driven rigid stage of the automatic chain, declined whenever it would make the across-frame surface rougher or introduce a tissue step.",
               (f"{method_name or title} was applied (roughness {_fmt(rec.get('rough_before'), 3)} → {_fmt(rec.get('rough_after'), 3)} px)." if rec and rec.get("applied") else
                (f"{method_name or title} was evaluated but declined because {_reason_clause(reason, ctx.cid)}." if rec else "")),
               numbers, {}, images, err, source={"manifest": [f"oct_iter.{key}"], "files": []})


def _node_rigid_height_refine(ctx, sl, R):
    return _rigid_stage("a04_rigid_height_refine", "Rigid height refine", "rigid_height_refine", ctx, R, method_name="Rigid height refinement")


def _node_rigid_frame_derotate(ctx, sl, R):
    return _rigid_stage("a05_rigid_frame_derotate", "Rigid frame derotate", "rigid_frame_derotate", ctx, R, method_name="Rigid frame derotation")


def _node_rigid_frame_refine(ctx, sl, R):
    return _rigid_stage("a06_rigid_frame_refine", "Rigid frame refine", "rigid_frame_refine", ctx, R, plot_key="shift", method_name="Rigid frame refinement")


def _node_manual_patch(ctx: RunCtx, sl: dict, R) -> dict:
    mp = ctx.p.get("manual_patch") if isinstance(ctx.p.get("manual_patch"), dict) else {}
    rec = ctx.it.get("manual_patch") if isinstance(ctx.it.get("manual_patch"), dict) else None
    if ctx.mode == "corrections":
        status, reason = "not_run", "legacy per-frame patch applies only on the automatic path; ignored on a corrections run"
    elif rec:
        status, reason = "applied", None
    elif mp:
        status, reason = "declined", "a manual patch is on record but the last run did not record applying it"
    else:
        status, reason = "not_run", "no manual per-frame patch on record"
    numbers = [_n("frames on record", len(mp)), _n("frames applied", rec.get("n_frames") if rec else None)]
    return _mk("a07_manual_patch", "user", "input", "Manual per-frame patch (legacy)", status, reason, f"{len(mp)} frames on record",
               "Legacy reviewer nudges (per-frame depth + tilt) applied last on the automatic path as ground truth.",
               (f"A manual per-frame patch was applied to {rec.get('n_frames')} frames." if rec else ""), numbers, {}, [], None,
               source={"manifest": ["oct_params.manual_patch", "oct_iter.manual_patch"], "files": []})


def _node_surface_crop_auto(ctx: RunCtx, sl: dict, R) -> dict:
    rec = ctx.it.get("surface_crop") if isinstance(ctx.it.get("surface_crop"), dict) else None
    prop = (ctx.m.get("oct_proposals") or {}).get("surface_crop") if isinstance(ctx.m.get("oct_proposals"), dict) else None
    if rec:
        status, reason = "applied", None
    elif prop:
        status, reason = "declined", "proposed by the detector but not applied (awaiting review)"
    elif ctx.mode == "corrections":
        status, reason = "not_run", "corrections run — the surface-crop band is placed from the posterior (see placement)"
    else:
        status, reason = "not_run", "no surface crop detected or marked"
    numbers = _rec_numbers(rec, ("n_frames", "pad", "rule", "auto", "clamped", "algo", "frac_frames", "pb_span")) if rec else ([_n("proposal", prop)] if prop else [])
    frames = sorted(int(f) for f in (rec.get("frames") or []) if str(f).lstrip("-").isdigit()) if rec else []
    if frames:
        numbers.append(_n("frames", _runs(frames)))
    images, err = [], None
    if R is not None and rec:
        def _b():
            s = sl["rendered"][0]
            g = _gray(ctx, s); gc = _gray_cor(ctx, s)
            if g is None or gc is None:
                raise RuntimeError("raw or corrected volume unavailable")
            fr = frames or ctx.crop_frames
            bands = [(fr[0], fr[-1], C_BAND, f"surface-crop frames {fr[0]}–{fr[-1]}")] if fr else []
            base = ctx.surf("baseline")
            curves = [(base[min(base.shape[0] - 1, s)], C_BASELINE, "solid", "auto surface", 0.8)] if base is not None else []
            return [R.pair({"img": g, "curves": curves, "bands": bands, "xlabel": "raw"}, {"img": gc, "bands": bands, "xlabel": f"corrected (+{rec.get('pad')} rows)"},
                           "primary", s, "sagittal",
                           f"Automatic surface-crop extend on slice {s} ({_DISP}): left the raw slice with the clipped-apex frames shaded, right the "
                           f"corrected slice after the posterior-parabola reconstruction and the {rec.get('pad')}-row canvas extension.",
                           f"a08_surface_crop_s{s}.png")]
        images, err = _safe_images(_b)
    return _mk("a08_surface_crop", "decision", "spine", "Surface-crop extend (automatic)", status, reason,
               (f"{rec.get('n_frames')} frames, pad {rec.get('pad')}, rule {rec.get('rule')}" if rec else (reason or "")),
               "Automatic clipped-apex correction: posterior parabola + canvas extended upward on the automatic path (this branch replaces the "
               "keep-best flatten passes when taken).",
               (f"The clipped apex was reconstructed on {rec.get('n_frames')} B-scans ({_runs(frames) if frames else '?'}) from the posterior parabola with a "
                f"{rec.get('pad')}-row canvas extension (rule '{rec.get('rule')}'" + (f", algorithm {rec.get('algo')}" if rec.get("algo") else "") + ")." if rec else ""),
               numbers, {"surface_crop_mode": ctx.p.get("surface_crop_mode"), "auto_surface_crop": ctx.p.get("auto_surface_crop")}, images, err,
               source={"manifest": ["oct_iter.surface_crop", "oct_proposals.surface_crop"], "files": []})


def _node_crop_auto(ctx: RunCtx, sl: dict, R) -> dict:
    rec = ctx.it.get("crop") if isinstance(ctx.it.get("crop"), dict) else None
    cr = ctx.p.get("crop_region") if isinstance(ctx.p.get("crop_region"), dict) else None
    cb = ctx.p.get("crop_bands") if isinstance(ctx.p.get("crop_bands"), dict) else None
    if rec:
        status, reason = "applied", None
    elif cr or cb:
        status, reason = "declined", "crop marks on record but the last run zeroed nothing"
    else:
        status, reason = "not_run", "no crop region / artifact bands on record"
    numbers = [_n("zeroed voxels", rec.get("n_voxels") if rec else None), _n("crop region", bool(cr)),
               _n("artifact bands", len(oct_mod.parse_crop_bands(cb)) if cb else 0),
               _n("guard-removed frames", ctx.it.get("crop_guard_removed_frames"))]
    return _mk("a09_crop", "decision", "spine", "Crop (zero artifact columns / region)", status, reason,
               (f"{rec.get('n_voxels')} voxels zeroed" if rec else (reason or "")),
               "Zeroes the reviewer's crop region and per-lateral artifact bands before segmentation.",
               (f"{rec.get('n_voxels')} voxels were zeroed by the reviewer's crop marks." if rec else ""), numbers, {}, [], None,
               source={"manifest": ["oct_iter.crop", "oct_params.crop_region", "oct_params.crop_bands"], "files": []})


# ── n02b axial fix-tool ground truth ────────────────────────────────────────────────────────────────────────
def _node_axial_gt(ctx: RunCtx, sl: dict, R) -> dict:
    aa = ctx.p.get("axial_anchors") if isinstance(ctx.p.get("axial_anchors"), dict) else {}
    rec = ctx.it.get("axial_anchors") if isinstance(ctx.it.get("axial_anchors"), dict) else None
    frames: dict = {}
    for f, lm in aa.items():
        try:
            fi = int(f)
        except (TypeError, ValueError):
            continue
        if not isinstance(lm, dict):
            continue
        inner = {}
        for l, d in lm.items():
            try:
                li = int(l); dv = float(d)
            except (TypeError, ValueError):
                continue
            if math.isfinite(dv):
                inner[li] = dv
        if inner:
            frames[fi] = inner
    npts = sum(len(v) for v in frames.values())
    numbers = [_n("frames drawn", len(frames)), _n("points", npts), _n("frames", _runs(frames.keys()) if frames else "—"),
               _n("applied on the last run", bool(rec and rec.get("applied"))), _n("frames adjusted", rec.get("frames_adjusted") if rec else None)]
    if not frames and not rec:
        status, reason = "not_run", "no axial fix-tool ground truth on record (oct_params.axial_anchors empty)"
    elif rec and rec.get("applied"):
        status, reason = "applied", None
    elif rec:
        status, reason = "declined", "axial anchors on record but the last run adjusted no frames"
    else:
        status, reason = "declined", "axial anchors on record but the last run recorded no application"
    images, err = [], None
    if R is not None and frames and ctx.raw_path is not None:
        def _b():
            f0 = sorted(frames.keys())[0]
            ga = _gray_axial(ctx.raw, f0)
            if ga is None:
                raise RuntimeError("raw volume unavailable")
            xs = np.array(sorted(frames[f0].keys()), np.float64); ys = np.array([frames[f0][int(x)] for x in xs], np.float64)
            return [R.image(ga, "primary", f0, "axial", f"Raw axial B-scan (frame {f0}) with the reviewer's axial fix-tool points (red); lateral 0 at right as in the app.",
                            f"n02b_axial_gt_f{f0}.png", points=[(xs, ys, C_SERVED, "axial GT points", 2.5)], xlabel="lateral (A-scan)")]
        images, err = _safe_images(_b)
    return _mk("n02b_axial_gt", "user", "input", "Reviewer axial fix-tool ground truth", status, reason,
               f"{len(frames)} frames, {npts} points" + (" · applied" if rec and rec.get("applied") else ""),
               "Corrections the reviewer made in the B-scan (axial) plane with the Fix-axial tool: sticky per-frame ground truth that a post-hoc "
               "rigid per-frame warp re-applies at the end of every run (feathered back to the detected surface over the undrawn laterals).",
               (f"The reviewer corrected the anterior surface in the B-scan plane on {len(frames)} frames ({npts} points); a post-hoc rigid per-frame "
                f"warp aligned those frames to the drawn surface ({rec.get('frames_adjusted')} frames adjusted)." if rec and rec.get("applied") else ""),
               numbers, {"axial_gt_feather": ctx.p.get("axial_gt_feather")}, images, err,
               source={"manifest": ["oct_params.axial_anchors", "oct_iter.axial_anchors"], "files": []})


# ── n02c Mark-tool marks ────────────────────────────────────────────────────────────────────────────────────
def _node_marks(ctx: RunCtx, sl: dict, R) -> dict:
    raw_marks = [x for x in (ctx.m.get("defect_marks") if isinstance(ctx.m.get("defect_marks"), list) else []) if isinstance(x, dict)]
    raw_marks += [x for x in (ctx.p.get("defect_marks") if isinstance(ctx.p.get("defect_marks"), list) else []) if isinstance(x, dict)]
    cdm = ctx.p.get("corrected_defect_marks") if isinstance(ctx.p.get("corrected_defect_marks"), dict) else {}
    n_c = sum(len(v) for v in cdm.values() if isinstance(v, list))
    by_orient: dict = {}
    for mk in raw_marks:
        by_orient[str(mk.get("orient") or "?")] = by_orient.get(str(mk.get("orient") or "?"), 0) + 1
    tags = sorted({str(mk.get("tag")) for mk in raw_marks if mk.get("tag")})
    numbers = [_n("marks on the raw / original view", len(raw_marks)), _n("by orientation", by_orient or "—"), _n("tags", tags or "—"),
               _n("marks on the corrected pane (laterals / ranges)", f"{len(cdm)} / {n_c}")]
    present = bool(raw_marks or n_c)
    status, reason = ("applied", None) if present else ("not_run", "no Mark-tool marks on record")
    images, err = [], None
    if R is not None and n_c and ctx.cor_path is not None:
        def _b():
            s = int(sorted(_int_keys(cdm).keys())[0])
            gc = _gray_cor(ctx, s)
            if gc is None:
                raise RuntimeError("corrected volume unavailable")
            bands = []
            for sg in cdm.get(str(s)) or cdm.get(s) or []:
                try:
                    bands.append((int(sg[0]), int(sg[1]), C_CROP, "defect mark"))
                except (TypeError, ValueError, IndexError):
                    continue
            return [R.image(gc, "primary", s, "sagittal", f"Corrected slice {s} ({_DISP}) with the reviewer's Mark-tool ranges (red); marks are review "
                            "annotations in corrected-frame coordinates and gate no stage.", f"n02c_marks_s{s}.png", bands=bands)]
        images, err = _safe_images(_b)
    return _mk("n02c_marks", "user", "input", "Reviewer Mark-tool marks", status, reason,
               f"{len(raw_marks)} raw-view marks · {n_c} corrected-pane ranges",
               "Columns the reviewer flagged with the Mark tool as wrongly detected (raw view: viewer-canonical indices, cleared by every re-preprocess; "
               "corrected pane: corrected-frame coordinates). They annotate the review and gate no processing stage.",
               (f"The reviewer flagged {len(raw_marks) + n_c} column ranges with the Mark tool as review annotations; no stage was gated by them." if present else ""),
               numbers, {}, images, err, source={"manifest": ["defect_marks", "oct_params.defect_marks", "oct_params.corrected_defect_marks"], "files": []})


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# assembly
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Stage ids in PIPELINE ORDER across both chains. The spine of a run = the canonical stages of its mode plus every
# other-chain stage whose own record says it ran (applied / declined), in this order (H2: a09_crop / _apply_crop runs
# on corrections runs too; a08 surface-crop auto is its own branch on the automatic path).
_RUN_ORDER = ["n00_raw", "n01_detect", "n04_served_surface", "n06_placement", "n07_tissue_motion", "n08_band_guide", "n09_canvas_extend",
              "a02_axial_motion_correct", "a03_flatten_passes", "n10_flatten", "a04_rigid_height_refine", "a05_rigid_frame_derotate",
              "a06_rigid_frame_refine", "n10a_detector_stages", "n10b_roughness_veto", "n10c_sag_quad_align", "a07_manual_patch",
              "a08_surface_crop", "n11b_edit_transform", "a09_crop", "n12_output"]
_SPINE_CORRECTIONS = ["n00_raw", "n01_detect", "n04_served_surface", "n06_placement", "n07_tissue_motion", "n08_band_guide",
                      "n09_canvas_extend", "n10_flatten", "n10a_detector_stages", "n10b_roughness_veto", "n10c_sag_quad_align",
                      "n11b_edit_transform", "n12_output"]
_SPINE_AUTOMATIC = ["n00_raw", "n01_detect", "a02_axial_motion_correct", "a03_flatten_passes", "a04_rigid_height_refine",
                    "a05_rigid_frame_derotate", "a06_rigid_frame_refine", "a07_manual_patch", "a08_surface_crop", "a09_crop", "n12_output"]
# reviewer-input nodes → the spine / input nodes they feed (first existing child = the row they sit level with)
_INPUT_CHILDREN = {"n02_top_lines": ["n04_served_surface"], "n03_crop_marks": ["n04_served_surface", "a08_surface_crop", "a09_crop"],
                   "n05_bottom_lines": ["n06_placement"], "n11_pane_edits": ["n02_top_lines", "n05_bottom_lines"],
                   "n02b_axial_gt": ["n12_output"], "n02c_marks": ["n12_output"]}
_INPUT_LABELS = {"n11_pane_edits": "folded (line mode)", "n02b_axial_gt": "post-hoc per-frame warp", "n02c_marks": "review annotation"}
_INPUT_ORDER = ["n02_top_lines", "n03_crop_marks", "n05_bottom_lines", "n11_pane_edits", "n02b_axial_gt", "n02c_marks"]
# kept for callers / tests: the canonical node sets per mode (inputs included)
_INPUT_CORRECTIONS = {"n02_top_lines": "n04_served_surface", "n03_crop_marks": "n04_served_surface",
                      "n05_bottom_lines": "n06_placement", "n11_pane_edits": "n02_top_lines"}
_INPUT_AUTOMATIC = {"n03_crop_marks": "a08_surface_crop", "n02_top_lines": None, "n05_bottom_lines": None, "n11_pane_edits": None}

CANONICAL_CORRECTIONS = ["n00_raw", "n01_detect", "n02_top_lines", "n03_crop_marks", "n04_served_surface", "n05_bottom_lines", "n06_placement",
                         "n07_tissue_motion", "n08_band_guide", "n09_canvas_extend", "n10_flatten", "n10a_detector_stages", "n10b_roughness_veto",
                         "n10c_sag_quad_align", "n11_pane_edits", "n11b_edit_transform", "n12_output"]
CANONICAL_AUTOMATIC = ["n00_raw", "n01_detect", "a02_axial_motion_correct", "a03_flatten_passes", "a04_rigid_height_refine", "a05_rigid_frame_derotate",
                       "a06_rigid_frame_refine", "a07_manual_patch", "a08_surface_crop", "a09_crop", "n12_output"]

_BUILDERS = {
    "n00_raw": _node_raw, "n01_detect": _node_detect, "n02_top_lines": _node_top_lines, "n03_crop_marks": _node_crop_marks,
    "n04_served_surface": _node_served_surface, "n05_bottom_lines": _node_bottom_lines, "n06_placement": _node_placement,
    "n07_tissue_motion": _node_tissue_motion, "n08_band_guide": _node_band_guide, "n09_canvas_extend": _node_canvas_extend,
    "n10_flatten": _node_flatten, "n10a_detector_stages": _node_detector_stages, "n10b_roughness_veto": _node_roughness_veto,
    "n10c_sag_quad_align": _node_sag_quad_align, "n11_pane_edits": _node_pane_edits, "n11b_edit_transform": _node_edit_transform,
    "n12_output": _node_output, "a02_axial_motion_correct": _node_axial_motion_correct, "a03_flatten_passes": _node_flatten_passes,
    "a04_rigid_height_refine": _node_rigid_height_refine, "a05_rigid_frame_derotate": _node_rigid_frame_derotate,
    "a06_rigid_frame_refine": _node_rigid_frame_refine, "a07_manual_patch": _node_manual_patch, "a08_surface_crop": _node_surface_crop_auto,
    "a09_crop": _node_crop_auto, "n02b_axial_gt": _node_axial_gt, "n02c_marks": _node_marks,
}

# Why a stage of the OTHER chain did not run, keyed by THIS run's mode (H2: the keys used to be inverted).
_NOT_RUN_TEXT = {
    "corrections": "corrections run — this stage of the automatic chain did not run",
    "automatic": "automatic run — no reviewer corrections were on record",
    "none": "scan has not been preprocessed",
}
_RAN = ("applied", "declined")


def _build_node(nid, ctx, sl, R):
    try:
        n = _BUILDERS[nid](ctx, sl, R)
    except Exception as exc:  # noqa: BLE001 — one broken record must not take the tree down
        n = _mk(nid, "auto", "spine", nid, "error", f"{type(exc).__name__}: {exc}", "", "", "", [], {}, [], None)
    for k in ("method", "description", "summary", "status_reason"):   # M7: third person, no foreign case names
        if n.get(k):
            n[k] = _neutral(n[k], ctx.cid)
    for im in n.get("images") or []:
        if im.get("caption"):
            im["caption"] = _neutral(im["caption"], ctx.cid)
    n["warnings"] = [_neutral(w, ctx.cid) for w in (n.get("warnings") or [])]
    for num in n.get("numbers") or []:               # recorded reason fields quoted as numbers get the same treatment
        v = num.get("value")
        if isinstance(v, str):
            num["value"] = _reason_clause(v, ctx.cid) if "reason" in str(num.get("label", "")).lower() and v.strip() else _neutral(v, ctx.cid)
        elif isinstance(v, list):
            num["value"] = [_neutral(x, ctx.cid) if isinstance(x, str) else x for x in v]
    return n


def _assemble(ctx: RunCtx, sl: dict, want_images: bool, thumb_h: int = 300, R: _Renderer | None = None) -> dict:
    if want_images and R is None:
        R = _Renderer(thumb_h=thumb_h)
    RR = R if want_images else None
    mode = ctx.mode
    canon = _SPINE_CORRECTIONS if mode == "corrections" else (_SPINE_AUTOMATIC if mode == "automatic" else ["n00_raw", "n12_output"])
    nodes: dict[str, dict] = {}
    for nid in _RUN_ORDER:                       # every stage of both chains is built from its own record …
        nodes[nid] = _build_node(nid, ctx, sl, RR)
    for nid in _INPUT_ORDER:
        nodes[nid] = _build_node(nid, ctx, sl, RR)
    if mode == "none":
        for nid, n in nodes.items():
            if nid not in ("n00_raw", "n12_output"):
                n["status"] = "not_run"; n["status_reason"] = _NOT_RUN_TEXT["none"]; n["images"] = []; n["method"] = ""
    # … and the spine is the canonical chain plus every other-chain stage whose record says it ran (never overridden)
    spine = [nid for nid in _RUN_ORDER if nid in canon or nodes[nid]["status"] in _RAN]
    for nid in _RUN_ORDER:
        n = nodes[nid]
        if nid in spine:
            n["role"] = "spine"; n["chain"] = "spine"
        else:
            n["role"] = "input"; n["chain"] = "other"
            if n["status"] not in _RAN and not n.get("status_reason"):
                n["status_reason"] = _NOT_RUN_TEXT[mode]
    for nid in _INPUT_ORDER:
        nodes[nid]["role"] = "input"; nodes[nid]["chain"] = "input"
    # edges
    edges = []
    for a, b in zip(spine[:-1], spine[1:]):
        edges.append({"from": a, "to": b, "kind": "spine", "label": None})
    fold = ctx.it.get("corrected_fold") if isinstance(ctx.it.get("corrected_fold"), dict) else {}
    for nid in _INPUT_ORDER:
        children = list(_INPUT_CHILDREN.get(nid, []))
        label = _INPUT_LABELS.get(nid, "reviewer input")
        pending = bool((nodes[nid].get("pending") or {}).get("n_points") or (nodes[nid].get("pending") or {}).get("n_bottom_points")
                       or (nodes[nid].get("pending") or {}).get("n_marks")) if nid == "n11_pane_edits" else False
        if nodes[nid]["status"] == "not_run" and not pending:
            continue                              # nothing on record feeds nothing (M10: absent inputs stay off the tree)
        if nid == "n11_pane_edits" and str(fold.get("mode") or "") == "transform" and nodes[nid].get("consumed"):
            children = ["n11b_edit_transform", "n02_top_lines"]
            label = "fitted transform"
        elif nid == "n11_pane_edits" and not nodes[nid].get("consumed"):
            label = "pending fold (line mode) — not consumed yet"
        for c in children:
            if c not in nodes:
                continue
            if nodes[c]["chain"] == "other":     # a stage that did not run on this chain is no target
                continue
            lab = label
            if nid == "n11_pane_edits" and c == "n02_top_lines" and label == "fitted transform":
                lab = "folded (transform mode)"
            edges.append({"from": nid, "to": c, "kind": "input", "label": lab})
        if nid in ("n02_top_lines", "n03_crop_marks", "n05_bottom_lines", "n02b_axial_gt") and "n00_raw" in nodes:
            edges.append({"from": "n00_raw", "to": nid, "kind": "note", "label": "drawn on the raw scan"})
    for e in edges:
        nodes[e["from"]]["children"].append(e["to"]); nodes[e["to"]]["parents"].append(e["from"])
    # node list order: spine in stage order, each preceded by the inputs that feed it — an input is listed BEFORE the
    # inputs feeding it (n02 before n11), so a single-pass layout (runTreeLayout.ts) finds the child already placed
    feeds: dict[str, list[str]] = {}
    for nid in _INPUT_ORDER:
        kids = [e["to"] for e in edges if e["from"] == nid and e["kind"] == "input"]
        if kids:
            feeds.setdefault(kids[0], []).append(nid)
    listed: list[str] = []
    def _add(target):
        for iid in feeds.get(target, []):
            if iid not in listed:
                listed.append(iid)
                _add(iid)
    for nid in spine:
        _add(nid)
        listed.append(nid)
    listed += [i for i in _INPUT_ORDER if i not in listed]
    listed += [i for i in _RUN_ORDER if i not in listed]
    run = _run_header(ctx, sl)
    run["stages"] = list(spine)
    run["flatten_kind"] = _flatten_kind(ctx.it)
    run["warnings"] = list(run["warnings"]) + [w for n in nodes.values() for w in n.get("warnings", []) if w not in run["warnings"]]
    graph = {"$schema": "run-graph/v1", "schema_version": SCHEMA_VERSION, "run": run, "nodes": [nodes[i] for i in listed], "edges": edges}
    return _json_safe(graph)


def _cache_key(ctx: RunCtx, sl: dict, want_images: bool, thumb_h: int):
    """Keyed on EVERY border_cache/*.npz (name, mtime_ns, size), the manifest, the raw and the corrected volume (M5)."""
    def _mt(p):
        try:
            return os.stat(p).st_mtime_ns if p and Path(p).exists() else 0
        except OSError:
            return 0
    return (ctx.cid, _mt(ctx.cor_path), _mt(ctx.raw_path), tuple(ctx.bc_listing), _mt(orch.manifest_path(ctx.cid)),
            tuple(sl["rendered"]), int(thumb_h), bool(want_images))


def build_run_graph(case_id: str, slice_index: int | None = None, want_images: bool = True, thumb_h: int = 300) -> dict:
    """The whole run-graph response for a case (see the schema in the Steps v2 spec)."""
    ctx = _ctx(case_id)
    sl = _slices(ctx, slice_index)
    key = _cache_key(ctx, sl, want_images, thumb_h)
    hit = _GRAPH_CACHE.get(key)
    if hit is not None:
        return hit
    graph = _assemble(ctx, sl, want_images, thumb_h)
    if len(_GRAPH_CACHE) >= _GRAPH_CACHE_MAX:
        _GRAPH_CACHE.pop(next(iter(_GRAPH_CACHE)))
    _GRAPH_CACHE[key] = graph
    return graph


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# layout + diagram svg
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _in_diagram(n: dict) -> bool:
    """Spine nodes always; a non-spine node only when its record says it did something (M10: not-run other-chain
    stages and absent reviewer inputs stay in nodes.json but off the diagram)."""
    return n.get("role") == "spine" or n.get("status") in ("applied", "declined", "missing_data", "error")


def layout_graph(graph: dict) -> dict:
    """Mirror of runTreeLayout.ts: spine top-down in run.stages order; reviewer inputs in a left column level with the
    first placed node they feed; an input that feeds ONLY other inputs (n11 → n02/n05) one column further left so its
    labelled edge is drawn left→right; not-run non-spine nodes omitted (M10, returned as `omitted`). Without a far
    column the tree keeps a two-column footprint (inputs at FAR_X, spine at INPUT_X)."""
    stages = list(graph.get("run", {}).get("stages") or [])
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    order = [n["id"] for n in graph.get("nodes", [])]
    spine = [sid for sid in stages if sid in nodes and nodes[sid].get("role") == "spine"]
    spine += [nid for nid in order if nodes[nid].get("role") == "spine" and nid not in spine]
    spine_set = set(spine)
    omitted = [nid for nid in order if nid not in spine_set and not _in_diagram(nodes[nid])]
    inputs = [nid for nid in order if nid not in spine_set and nid not in omitted]
    def _kids(nid):
        return [c for c in nodes[nid].get("children", []) if c in nodes and c not in omitted]
    def _feeds_spine(nid):
        return any(c in spine_set for c in _kids(nid))
    def _feeds_only_inputs(nid):
        return bool(_kids(nid)) and not _feeds_spine(nid)
    near = [nid for nid in inputs if not _feeds_only_inputs(nid)]
    far = [nid for nid in inputs if _feeds_only_inputs(nid)]
    has_far = bool(far)
    input_x = INPUT_X if has_far else FAR_X
    spine_x = SPINE_X if has_far else INPUT_X
    step = NODE_H + GAP
    boxes: dict[str, dict] = {}
    for i, sid in enumerate(spine):
        boxes[sid] = {"x": spine_x, "y": TOP + i * step, "w": NODE_W, "h": NODE_H}
    def _place(col, x):
        used: set = set()
        for nid in col:
            child = next((c for c in nodes[nid].get("children", []) if c in boxes), None)
            y = boxes[child]["y"] if child else TOP
            while y in used:
                y += step
            used.add(y)
            boxes[nid] = {"x": x, "y": y, "w": NODE_W, "h": NODE_H}
    _place(near, input_x)
    _place(far, FAR_X)
    edges = []
    for e in graph.get("edges", []):
        a, b = boxes.get(e["from"]), boxes.get(e["to"])
        if not a or not b or e.get("kind") == "note":      # note edges are not drawn (as in the frontend)
            continue
        if e["from"] in spine_set and e["to"] in spine_set:
            pts = [[a["x"] + a["w"] / 2, a["y"] + a["h"]], [b["x"] + b["w"] / 2, b["y"]]]
        elif a["x"] < b["x"]:                                # left column → right column: right edge into the left edge
            pts = [[a["x"] + a["w"], a["y"] + a["h"] / 2], [b["x"], b["y"] + b["h"] / 2]]
        elif a["x"] == b["x"]:                               # same column: vertical
            if a["y"] <= b["y"]:
                pts = [[a["x"] + a["w"] / 2, a["y"] + a["h"]], [b["x"] + b["w"] / 2, b["y"]]]
            else:
                pts = [[a["x"] + a["w"] / 2, a["y"]], [b["x"] + b["w"] / 2, b["y"] + b["h"]]]
        else:                                                # backwards: right edge to right edge
            pts = [[a["x"] + a["w"], a["y"] + a["h"] / 2], [b["x"] + b["w"], b["y"] + b["h"] / 2]]
        edges.append({"from": e["from"], "to": e["to"], "kind": e.get("kind"), "points": pts, "label": e.get("label")})
    max_y = max([b["y"] for b in boxes.values()] + [TOP])
    height = (max_y if spine else TOP) + NODE_H + TOP
    return {"nodes": boxes, "edges": edges, "width": spine_x + NODE_W + COL_GAP, "height": height, "omitted": omitted,
            "spine_x": spine_x, "input_x": input_x, "far_x": FAR_X if has_far else None, "has_far_column": has_far,
            "constants": {"SPINE_X": SPINE_X, "INPUT_X": INPUT_X, "FAR_X": FAR_X, "NODE_W": NODE_W, "NODE_H": NODE_H, "GAP": GAP,
                          "COL_GAP": COL_GAP, "TOP": TOP}}


_FILL_LIGHT = {"input": "#eef2f7", "user": "#e6f4ff", "auto": "#ffffff", "decision": "#fff7e6", "output": "#eaf7ec"}
_FILL_DARK = {"input": "#2a2f36", "user": "#1f3a4d", "auto": "#23262b", "decision": "#3d3320", "output": "#1f3a29"}


def _status_glyph(n: dict) -> str:
    st = n.get("status")
    if st == "applied":
        return "✓ applied"
    if st == "declined":
        return "✗ declined — " + textwrap.shorten(str(n.get("status_reason") or ""), 60, placeholder="…")
    if st == "not_run":
        return "○ not run"
    if st == "missing_data":
        return "⚠ missing data — " + textwrap.shorten(str(n.get("status_reason") or ""), 50, placeholder="…")
    return "⚠ error — " + textwrap.shorten(str(n.get("status_reason") or ""), 50, placeholder="…")


_TITLE_WRAP = 50      # characters per title line at 10 px bold inside a 340 px card
_SUMMARY_WRAP = 64    # characters per summary line at 8.5 px
_LEGEND_H = 46


def render_diagram_svg(graph: dict, layout: dict, palette: str = "light") -> str:
    dark = palette == "dark"
    fills = _FILL_DARK if dark else _FILL_LIGHT
    fg = "#e8e8e8" if dark else "#111111"; sub = "#bbbbbb" if dark else "#444444"; edge = "#999999" if dark else "#555555"
    bg = "#15171a" if dark else "#ffffff"; border = "#777777" if dark else "#333333"
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    W, H = int(layout["width"]), int(layout["height"])
    run = graph.get("run", {})
    total_h = H + _LEGEND_H + 22
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{total_h}" viewBox="0 0 {W} {total_h}" font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="100%" height="100%" fill="{bg}"/>',
           f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{edge}"/></marker></defs>']
    labels = []     # edge labels are painted LAST (after the cards) so a card never covers them
    for e in layout.get("edges", []):
        pts = e["points"]
        dash = ' stroke-dasharray="5,4"' if e.get("kind") == "input" else ""
        out.append(f'<line x1="{pts[0][0]:.1f}" y1="{pts[0][1]:.1f}" x2="{pts[1][0]:.1f}" y2="{pts[1][1]:.1f}" stroke="{edge}" stroke-width="1.2"{dash} marker-end="url(#arrow)"/>')
        if e.get("kind") == "input" and e.get("label"):
            # label at the edge's mid-point (as the frontend); a background halo keeps it legible where it overlaps a card
            mx = (pts[0][0] + pts[1][0]) / 2; my = (pts[0][1] + pts[1][1]) / 2 - 4
            labels.append(f'<text x="{mx:.1f}" y="{my:.1f}" font-size="7" fill="{sub}" text-anchor="middle" paint-order="stroke" stroke="{bg}" '
                          f'stroke-width="3" stroke-linejoin="round">{_html.escape(str(e["label"]))}</text>')
    for nid, b in layout["nodes"].items():
        n = nodes.get(nid)
        if not n:
            continue
        fill = fills.get(n.get("kind"), "#ffffff")
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="{fill}" stroke="{border}" stroke-width="1"/>')
        tlines = textwrap.wrap(str(n.get("title", nid)), _TITLE_WRAP)[:2] or [nid]
        if len(tlines) == 2 and len(str(n.get("title", ""))) > 2 * _TITLE_WRAP:
            tlines[1] = textwrap.shorten(tlines[1], _TITLE_WRAP, placeholder="…")
        for i, ln in enumerate(tlines):
            out.append(f'<text x="{x + 8}" y="{y + 13 + 12 * i}" font-size="10" font-weight="bold" fill="{fg}">{_html.escape(ln)}</text>')
        sy = y + 13 + 12 * len(tlines) + 2
        slines = textwrap.wrap(str(n.get("summary") or ""), _SUMMARY_WRAP)[:2]
        for i, ln in enumerate(slines):
            out.append(f'<text x="{x + 8}" y="{sy + 11 * i}" font-size="8.5" fill="{fg}">{_html.escape(ln)}</text>')
        out.append(f'<text x="{x + 8}" y="{y + h - 7}" font-size="8" fill="{sub}">{_html.escape(_status_glyph(n))}</text>')
    out += labels
    # legend: kind colours + status markers
    ly = H + 4
    out.append(f'<text x="{GAP}" y="{ly + 10}" font-size="8" font-weight="bold" fill="{fg}">Legend</text>')
    lx = GAP + 44
    for kind, label in (("input", "input"), ("user", "reviewer input"), ("auto", "automatic stage"), ("decision", "decision"), ("output", "output")):
        out.append(f'<rect x="{lx}" y="{ly + 2}" width="12" height="10" rx="2" fill="{fills.get(kind)}" stroke="{border}" stroke-width="0.8"/>')
        out.append(f'<text x="{lx + 16}" y="{ly + 10}" font-size="8" fill="{fg}">{_html.escape(label)}</text>')
        lx += 22 + 6 * len(label) + 12
    out.append(f'<text x="{GAP}" y="{ly + 26}" font-size="8" fill="{fg}">✓ applied   ✗ declined (with the recorded reason)   ○ not run   ⚠ missing data / error   '
               f'solid arrow = pipeline order, dashed arrow = reviewer input feeding a stage</text>')
    om = layout.get("omitted") or []
    out.append(f'<text x="{GAP}" y="{ly + 40}" font-size="7.5" fill="{sub}">{_html.escape(str(run.get("case_id", "")))} · {_html.escape(str(run.get("mode", "")))} run · '
               f'{_html.escape(str(run.get("run_time") or ""))} · generated by {GENERATOR}'
               + (f' · {len(om)} stage(s) that did not run on this chain are listed in nodes.json only' if om else "") + '</text>')
    out.append("</svg>")
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# methods paragraph
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _pval(v) -> str:
    """Parameter value as brace-free text for methods.md."""
    v = _json_safe(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, dict):
        return "(" + "; ".join(f"{k}: {_pval(x)}" for k, x in v.items()) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_pval(x) for x in v) + "]"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def methods_paragraph(graph: dict) -> str:
    run = graph.get("run", {})
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    stages = run.get("stages") or []
    ran = [nodes[i] for i in stages if i in nodes and nodes[i].get("status") not in ("not_run", "missing_data", "error")]
    spine_sent = [n["method"].strip() for n in ran if n.get("method")]
    user_nodes = [n for n in graph.get("nodes", []) if n.get("role") == "input" and n.get("kind") == "user" and n.get("status") == "applied"]
    user_sent = [n["method"].strip() for n in user_nodes if n.get("method")]
    head = f"# Methods — corneal OCT volume correction ({run.get('case_id')}, {run.get('run_time') or 'run time unknown'})"
    para1 = " ".join(s for s in spine_sent if s)
    para2 = " ".join(s for s in user_sent if s)
    out = [head, "", para1 or "(no completed stages on record)", ""]
    if para2:
        out += [para2, ""]
    out.append("Parameters (stages that ran, in run order; reviewer inputs that were applied)")
    seen = False
    for n in ran + user_nodes:
        params = {k: v for k, v in (n.get("params") or {}).items() if v is not None and v != {} and v != []}
        if not params:
            continue
        seen = True
        out.append(f"- {n.get('title')}: " + ", ".join(f"{k} = {_pval(v)}" for k, v in params.items()))
    if not seen:
        out.append("- (no parameters recorded for the stages that ran)")
    out.append("")
    out.append("Quality")
    for num in nodes.get("n12_output", {}).get("numbers", []):
        if num["label"] in ("boundary deviation (coverage-corrected)", "axial", "score", "coverage", "needs review", "review reasons", "max jitter"):
            out.append(f"- {num['label']}: {num['value']}{(' ' + num['unit']) if num.get('unit') else ''}")
    return "\n".join(out) + "\n"


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# publication export
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
def _panel_letter(ax, letter):
    ax.text(-0.08, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left", color="#111111")


_EXPORT_RC = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8, "svg.fonttype": "none"}


def _export_figure(ctx: RunCtx, graph: dict, S: int, dpi: int) -> tuple[bytes, str, str]:
    """figure.png/svg — 2×2 double-column panel; returns (png bytes, svg text, caption markdown). The matplotlib rc
    is scoped with rc_context (L15) so the sidecar's global rcParams are never mutated."""
    with plt.rc_context(_EXPORT_RC):
        return _export_figure_inner(ctx, graph, S, dpi)


def _export_figure_inner(ctx: RunCtx, graph: dict, S: int, dpi: int) -> tuple[bytes, str, str]:
    fig = plt.figure(figsize=(7.2, 5.6), dpi=100, facecolor="white")
    axs = [fig.add_axes(r) for r in ([0.08, 0.57, 0.38, 0.36], [0.58, 0.57, 0.38, 0.36], [0.08, 0.08, 0.38, 0.36], [0.58, 0.08, 0.38, 0.36])]
    cap = []
    g = _gray(ctx, S); gc = _gray_cor(ctx, S)
    pts, _ = _anchor_points(ctx.anchors); bpts, _ = _anchor_points(ctx.p.get("crop_post_anchors"))
    bands = _band_list(ctx)
    if ctx.mode == "corrections":
        # (a) raw with reviewer inputs
        fr, rows = _drawn_pts(pts, S); fb, rb = _drawn_pts(bpts, S)
        okb = rb < (ctx.D - 1) if ctx.D else np.ones(rb.size, bool)
        n_top = int(fr.size); n_bot = int(fb[okb].size) if fb.size else 0
        if g is not None:
            pnts = ([(fr, rows, C_SERVED, "top-edge points", 2.0)] if fr.size else []) + ([(fb[okb], rb[okb], C_BOTTOM, "bottom-line points", 2.0)] if n_bot else [])
            _fig_image(g, [], pnts, bands, xlabel="frame", dark=False, ax=axs[0], fig=fig, legend=True)
        drawn = ([f"{n_top} top-edge points (red)"] if n_top else []) + ([f"{n_bot} bottom-line points (orange)"] if n_bot else []) \
            + ([f"surface-crop band {ctx.crop_frames[0]}–{ctx.crop_frames[-1]} (blue)"] if bands else [])
        cap.append(f"(a) Raw sagittal slice {S} with the reviewer's inputs on this slice: " + (", ".join(drawn) if drawn else "none drawn on this slice") + ".")
        # (b) served surfaces
        served = ctx.surf("provided_edges"); base = ctx.surf("baseline"); post = ctx.surf("posterior_edges"); placed = ctx.surf("placed_edges")
        curves = []
        if base is not None:
            curves.append((base[min(base.shape[0] - 1, S)], C_BASELINE, "solid", "automatic baseline", 0.9))
        if served is not None:
            curves.append((served[min(served.shape[0] - 1, S)], C_SERVED, "solid", "served top"))
        if post is not None:
            curves.append((post[min(post.shape[0] - 1, S)], C_BOTTOM, "solid", "served posterior"))
        if placed is not None and ctx.crop_frames:
            cf = set(ctx.crop_frames); yp = placed[min(placed.shape[0] - 1, S)]
            curves.append((np.where([f in cf for f in range(yp.size)], yp, np.nan), C_PLACED, "dashed", "placed top (band)"))
        if g is not None:
            _fig_image(g, curves, [], bands, xlabel="frame", dark=False, ax=axs[1], fig=fig)
        cap.append("(b) Served surfaces on the same slice: baseline (blue), served top (red), served posterior (orange), placed top inside the band (magenta, dashed).")
        # (c) the per-frame move the run applied: tissue-measured (post-guide, pre-guide shift dotted when the guide was
        #     applied — L14) or, on a drawn-line-move run, the applied move recorded in applied_move.npz
        tm = _tissue_motion_arrays(ctx); rec = ctx.it.get("tissue_motion") or {}
        kind = _flatten_kind(ctx.it)
        flrec = ctx.it.get("flatten") if isinstance(ctx.it.get("flatten"), dict) else {}
        series = []
        mv = None
        if kind == "drawn":
            move = ctx.surf("applied_move") if _applied_move_fresh(ctx) else None
            if move is not None:
                med = np.nanmedian(move, axis=0); rng = np.nanmax(move, axis=0) - np.nanmin(move, axis=0)
                series = [(med, C_SHIFT, "applied move, median over laterals"), (rng, C_TILT, "lateral range of the move (tilt)")]
            _fig_plot(series, bands=bands, dark=False, ax=axs[2], fig=fig, xlabel="frame", ylabel="px",
                      annotate=f"shift range {_fmt(flrec.get('shift_range'))} px, tilt max {_fmt(flrec.get('tilt_max_px'))} px (drawn-line move)")
            cap.append(f"(c) Drawn-line move (mode '{flrec.get('mode')}'): per-frame applied depth shift (red, median over laterals) and its lateral range "
                       f"(blue); shift range {_fmt(flrec.get('shift_range'))} px, tilt max {_fmt(flrec.get('tilt_max_px'))} px"
                       + ("." if series else "; applied_move.npz is stale or absent, so no curve is drawn."))
        else:
            mv = _move_arrays(ctx)
            if mv is not None:
                a, b, guided = mv
                if guided and tm is not None and tm[0] is not None:
                    series.append((tm[0], "#ff9090", "shift a[f] before guide", "dotted"))
                series += [(a, C_SHIFT, "shift a[f]" + (" (post-guide)" if guided else "")), (b, C_TILT, "tilt b[f]")]
            _fig_plot(series, bands=bands, dark=False, ax=axs[2], fig=fig, xlabel="frame", ylabel="px",
                      annotate=f"shift range {_fmt(rec.get('shift_range'))} px, tilt max {_fmt(rec.get('tilt_max_px'))} px")
            cap.append(f"(c) Tissue-motion measurement: per-frame shift a[f] (red) and tilt b[f] (blue); shift range {_fmt(rec.get('shift_range'))} px, "
                       f"tilt max {_fmt(rec.get('tilt_max_px'))} px" + ("; the dotted curve is the shift before the band guide." if series and mv is not None and mv[2] else "."))
        # (d) corrected with target
        carried_drawn = False
        if gc is not None:
            curves = []
            surf, src = ctx.served_for_flatten()
            if surf is not None and mv is not None and mv[0].size == gc.shape[1] and kind != "drawn":
                L = surf.shape[0]; x = (S - (L - 1) / 2.0) / max(1e-9, (L - 1) / 2.0)
                curves.append((surf[min(L - 1, S)] + ctx.pad + mv[0] + mv[1] * x, C_TARGET, "solid", "flatten target" + (" (post-guide)" if mv[2] else "")))
            move = ctx.surf("applied_move") if _applied_move_fresh(ctx) else None
            if served is not None and move is not None and move.shape == served.shape:
                curves.append((np.clip(served[min(served.shape[0] - 1, S)] + ctx.pad + move[min(move.shape[0] - 1, S)], 0, gc.shape[0] - 1), "#333333", "dashed", "served surface carried", 0.9))
                carried_drawn = True
            _fig_image(gc, curves, [], bands, xlabel="frame", dark=False, ax=axs[3], fig=fig)
            if ctx.pad > 0:
                axs[3].annotate("", xy=(gc.shape[1] - 1, ctx.pad), xytext=(gc.shape[1] - 1, 0), arrowprops={"arrowstyle": "|-|", "color": C_TARGET, "lw": 0.8})
                axs[3].text(gc.shape[1] - 3, ctx.pad / 2, f"pad {ctx.pad}", fontsize=6, color=C_TARGET, va="center", ha="right")
        cap.append(f"(d) Corrected slice {S} with the flatten target (green)"
                   + (" and the carried served surface (dashed)" if carried_drawn else "")
                   + (f"; canvas pad {ctx.pad} rows bracketed at the top." if ctx.pad > 0 else "."))
    else:
        base = ctx.surf("baseline")
        if g is not None:
            _fig_image(g, [(base[min(base.shape[0] - 1, S)], C_BASELINE, "solid", "automatic baseline")] if base is not None else [], [], [], xlabel="frame", dark=False, ax=axs[0], fig=fig)
        cap.append(f"(a) Raw sagittal slice {S} with the automatic baseline surface (blue).")
        amc = ctx.it.get("axial_motion_correct") or {}
        _fig_plot([(np.asarray(amc.get("shift") or [], np.float64), C_SHIFT, "AMC shift[f]")] if amc.get("shift") else [], dark=False, ax=axs[1], fig=fig, xlabel="frame")
        cap.append("(b) Axial motion correction: per-frame shift.")
        names = ["rigid_height_refine", "rigid_frame_derotate", "rigid_frame_refine"]
        bef = [float((ctx.it.get(k) or {}).get("rough_before") or np.nan) for k in names]; aft = [float((ctx.it.get(k) or {}).get("rough_after") or np.nan) for k in names]
        xs = np.arange(3)
        axs[2].bar(xs - 0.18, np.nan_to_num(bef), 0.36, color="#bbbbbb", label="rough before"); axs[2].bar(xs + 0.18, np.nan_to_num(aft), 0.36, color=C_TILT, label="rough after")
        axs[2].set_xticks(xs); axs[2].set_xticklabels(["height", "derotate", "refine"], fontsize=7); _style_axes(axs[2], "rigid stage", "roughness (px)", dark=False)
        axs[2].legend(fontsize=6)
        cap.append("(c) Rigid stage summary: surface roughness before/after each detector-driven stage.")
        if gc is not None:
            _fig_image(gc, [], [], [], xlabel="frame", dark=False, ax=axs[3], fig=fig)
        cap.append(f"(d) Corrected slice {S}.")
    for ax, letter in zip(axs, "abcd"):
        _panel_letter(ax, letter)
    png = io.BytesIO(); fig.savefig(png, format="png", dpi=dpi, facecolor="white")
    svg = io.StringIO(); fig.savefig(svg, format="svg", facecolor="white")
    plt.close(fig)
    caption = f"**Figure.** Run provenance for {ctx.cid} ({ctx.mode} run, {ctx.run_time or 'time unknown'}). " + " ".join(cap) + f" All panels are display-oriented ({_DISP})."
    return png.getvalue(), svg.getvalue(), caption + "\n"


def _report_html(graph: dict, diagram_svg: str, figure_png: bytes, methods_md: str, caption_md: str) -> str:
    run = graph["run"]
    esc = _html.escape
    def _chip(n):
        st = n.get("status"); col = {"applied": "#2e7d32", "declined": "#b26a00", "not_run": "#777", "missing_data": "#b71c1c", "error": "#b71c1c"}.get(st, "#555")
        return f'<span class="chip" style="background:{col}">{esc(str(st))}</span>' + (f' <span class="reason">{esc(str(n.get("status_reason")))}</span>' if n.get("status_reason") else "")
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Run report — " + esc(str(run.get("case_id"))) + "</title>",
             "<style>body{font-family:Helvetica,Arial,sans-serif;font-size:11pt;color:#111;background:#fff;max-width:1100px;margin:0 auto;padding:24px}"
             "h1{font-size:18pt}h2{font-size:13pt;border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:28px}table{border-collapse:collapse;font-size:9.5pt}"
             "td,th{border:1px solid #ddd;padding:3px 7px;text-align:left;vertical-align:top}section{page-break-inside:avoid;margin:14px 0;padding:10px 12px;border:1px solid #e3e3e3;border-radius:6px}"
             "section.input{margin-left:36px;background:#f3f9ff}.badge{font-size:8pt;background:#1976d2;color:#fff;padding:1px 6px;border-radius:8px;margin-left:6px}"
             ".chip{font-size:8pt;color:#fff;padding:1px 6px;border-radius:8px}.reason{font-size:9pt;color:#555}.method{font-style:italic;color:#333}"
             "figure{margin:8px 0}figcaption{font-size:9pt;color:#444}img{max-width:100%;height:auto;border:1px solid #ddd}.imgs{display:flex;flex-wrap:wrap;gap:10px}.imgs figure{max-width:480px}"
             ".warn{background:#fff3e0;border:1px solid #ffb74d;padding:6px 10px;border-radius:4px;font-size:9.5pt}pre{white-space:pre-wrap;font-size:9.5pt}"
             "@page{margin:15mm}@media print{section{page-break-inside:avoid}}</style></head><body>",
             f"<h1>Run report — {esc(str(run.get('case_id')))}</h1>",
             "<table><tbody>" + "".join(f"<tr><th>{esc(k)}</th><td>{esc(str(run.get(k)))}</td></tr>" for k in
                                        ("case_id", "patient_id", "eye", "mode", "run_time", "input_volume", "raw_volume", "dims_raw", "dims_corrected", "spacing_mm", "generated_at", "generator")) + "</tbody></table>"]
    if run.get("warnings"):
        parts.append("<div class='warn'><b>Warnings</b><ul>" + "".join(f"<li>{esc(str(w))}</li>" for w in run["warnings"]) + "</ul></div>")
    parts.append("<h2>Workflow diagram</h2><div style='overflow:auto'>" + diagram_svg + "</div>")
    if figure_png:
        parts.append("<h2>Figure</h2><figure><img src='" + _data_url(figure_png) + "' alt='figure'/><figcaption>" + esc(caption_md.strip()) + "</figcaption></figure>")
    parts.append("<h2>Methods</h2><pre>" + esc(methods_md) + "</pre>")
    parts.append("<h2>Steps</h2>")
    for n in graph["nodes"]:
        cls = "input" if n.get("role") == "input" else "spine"
        parts.append(f"<section class='{cls}'><h3 id='{esc(n['id'])}'>{esc(str(n.get('title')))}" + ("<span class='badge'>Reviewer input</span>" if n.get("kind") == "user" else "") + "</h3>")
        parts.append("<p>" + _chip(n) + "</p>")
        if n.get("description"):
            parts.append(f"<p>{esc(str(n['description']))}</p>")
        if n.get("method"):
            parts.append(f"<p class='method'>{esc(str(n['method']))}</p>")
        if n.get("numbers"):
            parts.append("<table><tbody>" + "".join(f"<tr><th>{esc(str(x['label']))}</th><td>{esc(str(x['value']))}{(' ' + esc(str(x['unit']))) if x.get('unit') else ''}</td></tr>" for x in n["numbers"]) + "</tbody></table>")
        if n.get("warnings"):
            parts.append("<div class='warn'>" + "<br/>".join(esc(str(w)) for w in n["warnings"]) + "</div>")
        if n.get("images"):
            parts.append("<div class='imgs'>" + "".join(f"<figure><img src='{im['data_url']}' width='{im['width']}' alt='{esc(im['file'])}'/><figcaption>{esc(str(im.get('caption') or ''))}</figcaption></figure>" for im in n["images"]) + "</div>")
        elif n.get("image_error"):
            parts.append(f"<p class='reason'>image unavailable: {esc(str(n['image_error']))}</p>")
        parts.append("</section>")
    parts.append("</body></html>")
    return "\n".join(parts)


def export_run_report(case_id: str, out_dir: Path, slice_index: int | None = None, dpi: int = 300, include_filmstrip: bool = False) -> dict:
    """Write the publication folder (see export_plan). Returns {folder, files, n_nodes, mode}. Requires matplotlib."""
    if not _HAVE_MPL:
        raise RuntimeError("matplotlib is required for the publication export (pip install matplotlib)")
    out_dir = Path(out_dir)
    dpi = clamp_dpi(dpi)
    ctx = _ctx(case_id)
    sl = _slices(ctx, slice_index)
    R = _Renderer(thumb_h=300, export=True, dpi=200)
    graph = _assemble(ctx, sl, True, 300, R)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "nodes").mkdir(exist_ok=True)
    files: list[str] = []
    for name, data in R.files.items():
        (out_dir / "nodes" / name).write_bytes(data); files.append(f"nodes/{name}")
    S = int(sl["rendered"][0]) if slice_index is not None else int(sl["most_edited"])
    png, svg, caption = _export_figure(ctx, graph, S, dpi)
    (out_dir / "figure.png").write_bytes(png); (out_dir / "figure.svg").write_text(svg); (out_dir / "figure_caption.md").write_text(caption)
    lay = layout_graph(graph)
    diagram = render_diagram_svg(graph, lay)
    (out_dir / "diagram.svg").write_text(diagram); (out_dir / "diagram_dark.svg").write_text(render_diagram_svg(graph, lay, "dark"))
    methods = methods_paragraph(graph)
    (out_dir / "methods.md").write_text(methods)
    (out_dir / "nodes.json").write_text(json.dumps(graph, indent=1))
    (out_dir / "report.html").write_text(_report_html(graph, diagram, png, methods, caption))
    readme = (f"Run report for {ctx.cid} ({ctx.mode} run)\n\n"
              "figure.png / figure.svg  — the four-panel publication figure (300 dpi / vector)\n"
              "figure_caption.md        — its caption with this run's numbers\n"
              "diagram.svg              — the workflow/provenance tree (diagram_dark.svg for slides)\n"
              "methods.md               — Methods paragraph composed from the stages that ran\n"
              "nodes.json               — the full run graph (every node, numbers, captions, inline images)\n"
              "nodes/*.png              — every node image at export size\n"
              "report.html              — self-contained report (open in a browser; print to PDF)\n"
              + ("filmstrip/*.png          — the detector filmstrip previews copied from the case\n" if include_filmstrip else "") +
              f"\ngenerated by Cornea {GENERATOR} from manifest.json (oct_params / oct_iter) and border_cache on {graph['run']['generated_at']}; nothing was re-run.\n")
    (out_dir / "README.txt").write_text(readme)
    files = ["figure.png", "figure.svg", "figure_caption.md", "diagram.svg", "diagram_dark.svg", "methods.md", "nodes.json", "report.html", "README.txt"] + files
    if include_filmstrip:
        src = ctx.root / "previews" / "oct_steps"
        if src.is_dir():
            (out_dir / "filmstrip").mkdir(exist_ok=True)
            for fp in sorted(src.glob("*.png")):
                shutil.copyfile(fp, out_dir / "filmstrip" / fp.name); files.append(f"filmstrip/{fp.name}")
    return {"folder": str(out_dir), "files": files, "n_nodes": len(graph["nodes"]), "mode": ctx.mode}


def zip_report(out_dir: Path, zip_path: Path | None = None) -> Path:
    out_dir = Path(out_dir)
    zp = Path(zip_path) if zip_path else out_dir.with_suffix(".zip")
    tmp = zp.with_name(zp.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(out_dir.rglob("*")):
            if fp.is_file():
                z.write(fp, f"{out_dir.name}/{fp.relative_to(out_dir)}")
    os.replace(tmp, zp)
    return zp
