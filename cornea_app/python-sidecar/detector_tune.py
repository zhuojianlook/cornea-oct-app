"""GLOBAL DETECTOR TUNING from the reviewer's own corrections.

THE LOOP THIS SERVES. The reviewer opens each scan at its worst slice, corrects the detected corneal surface
and rejects, or approves. Those corrections are ground truth about where the cornea actually is. Re-running
preprocessing applies each scan's corrections to THAT scan and nothing else — it cannot make the next scan
better, so on its own the loop never converges. This module closes it: take every correction drawn so far,
search the detector's parameters for a setting that reproduces them better, and adopt it globally — but only
if it does not disturb the scans already approved.

WHY PARAMETER SEARCH AND NOT FITTING. "Improve the algorithm" has to mean something that generalises to the
300-odd scans nobody has corrected. A per-scan correction field (generalize_surface) does not: it is a
displacement learned from one eye. The detector's parameters do — they are the algorithm.

TWO RULES THE SEARCH OBEYS.

  1. CORRECTIONS ARE APPROXIMATE. The reviewer's own words: "they can be slightly off and not pixel perfect."
     So the objective is a HINGE — error inside the tolerance band scores zero, and nothing is gained by
     fitting an anchor exactly. A plain mean-squared fit would chase the reviewer's hand jitter and would
     rank a detector that traced their wobble above one that found the true smooth surface.

  2. NO REGRESSIONS. Approved scans are the reviewer's signed-off output. A candidate that improves the
     corrected scans while shifting the surface on approved ones is not an improvement, it is a trade the
     reviewer never agreed to. Every round-winner is checked against a sample of approved scans and dropped
     if the surface moves more than the guard allows.

COST. The detected surface goes through whole-volume post-passes that mix neighbouring slices
(_robust_dome_smooth, _lateral_smooth_by_confidence, the dip/spike rejects), so scoring one anchored slice in
isolation is NOT the same function the pipeline runs. Every evaluation is therefore a full-volume detection.
The search is coordinate descent (one parameter at a time, keep the winner, move on) rather than a grid,
because a grid over six parameters is thousands of volume detections. Volumes are loaded once per round and
reused across that round's candidates, which is what keeps this to hours rather than days.

RUNS IN ITS OWN PROCESS, ALWAYS. detect_surface_all parallelises over slices with a FORK pool, and
_map_slices spells out the condition for that being safe: "the heavy smoother runs in an isolated subprocess
(oct_preprocess CLI), never directly inside the CUDA-bearing sidecar." Forking a server process copies only
the calling thread, so any lock another thread happens to hold is copied LOCKED and never released — the
children wedge, and the parent waits on them forever. Running the search in a sidecar thread hit exactly
that: three defunct children and a progress counter frozen mid-round, immune to cancellation because a
deadlock raises nothing for the serial fallback to catch. So the search is invoked as a CLI subprocess
(see main), reports through a status file, and is cancelled through a sentinel file or by killing its group.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import oct_preprocess as oct_mod

# The parameters worth searching, in the order they are searched, with the candidate values tried for each.
# DELIBERATELY SMALL. Every extra value costs a full volume detection per corpus scan, and these six are the
# ones that actually move the anterior surface: the DP detector's smoothness (sigma_depth/frame), how far
# below the surface it looks for tissue (dp_below), how far it may jump between columns (dp_max_jump), the
# search window, and the faint-onset snap that corrects the detector's known shallow bias.
# Values bracket the current default rather than replacing it, so "no change" is always in the running.
SEARCH_SPACE: list[tuple[str, list]] = [
    ("dp_sigma_depth", [2.0, 3.0, 4.5, 6.0]),
    ("dp_sigma_frame", [2.0, 3.0, 4.5, 6.0]),
    ("dp_below", [16, 24, 32, 40]),
    ("dp_max_jump", [6, 10, 16]),
    ("detect_window", [6.0, 10.0, 16.0]),
    ("faint_snap_frac", [0.0, 0.3, 0.45, 0.6]),
]

TOL_PX = 3.0          # inside this, a correction counts as reproduced (see rule 1)
GUARD_MEDIAN_PX = 0.75  # an approved scan's surface may move this much (median) and no more
GUARD_P95_PX = 3.0      # ...and this much at the 95th percentile, so a local blow-up is caught too
MIN_GAIN = 0.02         # relative improvement needed to adopt at all — below this it is noise, not a fix
HOLDOUT_FRAC = 0.3      # share of corrected scans withheld from the search and used only to judge it
MIN_FOR_HOLDOUT = 4     # below this there is no split worth making — the run reports fit only, and says so


class Cancelled(Exception):
    """Raised inside the worker when the reviewer cancels, so a long run unwinds at the next checkpoint."""


def _anchor_points(manifest: dict) -> list[tuple[int, int, float]]:
    """(lateral_slice, frame, true_depth) for every correction on this scan.

    Skips the ABSENT sentinel: a frame the reviewer marked as having no anterior surface is a statement that
    there is nothing to detect there, so scoring the detector against it would penalise it for not finding a
    surface the reviewer said does not exist."""
    anc = ((manifest.get("oct_params") or {}).get("border_anchors")) or {}
    out: list[tuple[int, int, float]] = []
    for s_key, frames in anc.items():
        try:
            s = int(s_key)
        except (TypeError, ValueError):
            continue
        if not isinstance(frames, dict):
            continue
        for f_key, d in frames.items():
            try:
                f = int(f_key); dv = float(d)
            except (TypeError, ValueError):
                continue
            if np.isfinite(dv):
                out.append((s, f, dv))
    return out


def score_surface(surf: np.ndarray, pts: Iterable[tuple[int, int, float]], tol: float = TOL_PX) -> dict:
    """How well a detected surface reproduces the reviewer's corrections.

    `hinge` is the objective: mean of max(0, |err| - tol), so being inside the band is free and the search
    cannot be rewarded for tracing hand jitter. `within` is the human-readable companion — the fraction of
    past corrections the detector now gets right on its own, which is the convergence number.
    Depths at or above the canvas floor are excluded on both sides: the reviewer's absent-sentinel and the
    detector's failure-to-find are both "no surface", not a large error."""
    L, F = surf.shape
    errs: list[float] = []
    for s, f, gt in pts:
        if not (0 <= s < L and 0 <= f < F):
            continue
        a = float(surf[s, f])
        if not np.isfinite(a):
            continue
        errs.append(abs(a - gt))
    if not errs:
        return {"n": 0, "hinge": float("inf"), "within": 0.0, "mean": float("inf")}
    e = np.asarray(errs, dtype=np.float64)
    return {"n": int(e.size),
            "hinge": float(np.maximum(0.0, e - tol).mean()),
            "within": float((e <= tol).mean()),
            "mean": float(e.mean())}


class TuneRun:
    """One tuning pass. Owns its own cancel flag and progress dict so the endpoint can poll and stop it."""

    def __init__(self, corpus: list[dict], guard: list[dict], workers: int = 4,
                 log: Callable[[str], None] | None = None,
                 status_path: str | None = None, cancel_path: str | None = None):
        self.corpus = corpus          # [{case_id, src, volume_index, pts, params}]
        self.guard = guard            # [{case_id, src, volume_index, params}]
        self.workers = int(workers)
        self.cancel = threading.Event()
        self._log = log or (lambda _m: None)
        # Cross-PROCESS progress and cancellation. The run lives in its own process (see the module note), so
        # a threading.Event cannot reach it and its state cannot be read directly — both travel as files.
        self.status_path = status_path
        self.cancel_path = cancel_path
        self.train: list[dict] = list(corpus)   # set in _run once the split is known
        self.hold: list[dict] = []
        self.state: dict = {
            "running": True, "phase": "starting", "done": 0, "total": 0,
            "started": round(time.time(), 1), "baseline": None, "best": None,
            "adopted": None, "tried": [], "note": "", "n_corpus": len(corpus), "n_guard": len(guard),
        }

    # ---- plumbing -------------------------------------------------------------------------------------
    def _check(self) -> None:
        if self.cancel.is_set() or (self.cancel_path and os.path.exists(self.cancel_path)):
            raise Cancelled()

    def _publish(self) -> None:
        """Write the status file atomically. Atomically because the sidecar polls it on a timer: a reader that
        catches a half-written file would show a parse error as if the run had failed."""
        if not self.status_path:
            return
        try:
            tmp = f"{self.status_path}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(self.state, fh)
            os.replace(tmp, self.status_path)
        except OSError:
            pass

    def _load(self, entry: dict) -> np.ndarray:
        self._check()
        vol = oct_mod.read_oct_zstack(entry["src"], int(entry.get("volume_index", 0) or 0))
        return oct_mod.reformat_to_sagittal(vol).astype("float32")

    def _detect(self, sag: np.ndarray, entry: dict, overrides: dict) -> np.ndarray:
        """The auto surface under `overrides`. The scan's OWN oct_params are applied first and the candidate
        on top, so a scan carrying a per-scan detector setting is scored the way the pipeline would actually
        run it — otherwise the search optimises a detector that never executes."""
        self._check()
        p = {**oct_mod.DEFAULT_PARAMS, **(entry.get("params") or {}), **overrides}
        return oct_mod.detect_surface_all(sag, p, workers=self.workers)

    # ---- the pass -------------------------------------------------------------------------------------
    def run(self) -> dict:
        try:
            out = self._run()
            self._publish()
            return out
        except Cancelled:
            self.state.update({"running": False, "phase": "cancelled",
                               "note": "Cancelled — nothing was changed."})
            self._publish()
            return self.state
        except Exception as exc:  # noqa: BLE001 — a tuning crash must not take the sidecar with it
            self.state.update({"running": False, "phase": "failed", "note": f"{type(exc).__name__}: {exc}"[:300]})
            self._publish()
            return self.state

    def _run(self) -> dict:
        if not self.corpus:
            self.state.update({"running": False, "phase": "done",
                               "note": "No corrections to learn from yet."})
            return self.state

        # ---- TRAIN / HELD-OUT SPLIT ------------------------------------------------------------------
        # THE QUESTION THIS ANSWERS: does a parameter set found from these corrections help scans it has
        # never seen? Tuning and scoring on the same points measures FIT, and fit always improves — a search
        # over six parameters can shave the objective on any fixed set of points without the detector being
        # any better on the 300 scans nobody has corrected. Adopting on that number would quietly bake in
        # whatever suits the handful of eyes that happen to have been reviewed.
        # So a slice of the corpus is withheld from the search entirely and only ever scored. Improvement
        # there is evidence of generalisation; improvement only on the training half is evidence of nothing.
        # Every OTHER scan is trained on — with few corrections, holding back more than a third costs more in
        # search signal than it buys in confidence.
        # SPLIT BY EYE, never by scan. Replicates of one eye are near-identical volumes, so holding one back
        # while training on its siblings tests nothing — the search has already seen that cornea. Whole eyes
        # move together, and the count that matters is therefore EYES, not scans.
        groups: dict[str, list[dict]] = {}
        for c in self.corpus:
            groups.setdefault(str(c.get("group") or c.get("case_id")), []).append(c)
        keys = sorted(groups)
        n_hold = max(1, round(len(keys) * HOLDOUT_FRAC)) if len(keys) >= MIN_FOR_HOLDOUT else 0
        # every k-th eye rather than the tail, so the held-out set is not one patient or one session
        hold_keys = set(keys[:: max(1, len(keys) // n_hold)][:n_hold]) if n_hold else set()
        self.train = [c for k in keys if k not in hold_keys for c in groups[k]]
        self.hold = [c for k in keys if k in hold_keys for c in groups[k]]
        self.state["n_eyes"] = len(keys)
        self.state["n_eyes_holdout"] = len(hold_keys)
        self.state["n_train"] = len(self.train)
        self.state["n_holdout"] = len(self.hold)
        if not self.hold:
            self.state["holdout_note"] = (
                f"Only {self.state['n_eyes']} corrected EYE(S) — too few to hold any back, so this run can "
                f"show fit but not generalisation. Corrections on at least {MIN_FOR_HOLDOUT} different eyes "
                f"are needed before the held-out check means anything (replicates of one eye do not count "
                f"separately: the search has already seen that cornea).")

        # One volume detection per (corpus scan × candidate), plus baseline and held-out passes. Counting
        # ROUNDS instead of candidates made the bar read "23/7" — a progress number that goes past its own
        # total is worse than no progress number, because it reads as a fault in the thing you are waiting on.
        self.state["total"] = (len(self.train) * (1 + sum(len(v) for _k, v in SEARCH_SPACE))
                               + len(self.hold) * 2)
        best: dict = {}                      # the winning overrides so far, applied cumulatively

        # ---- baseline: what the CURRENT detector scores on the corrections ----------------------------
        self.state["phase"] = "baseline"
        self._publish()
        base_scores = self._score_corpus({}, self.train)
        self.state["baseline"] = base_scores
        self._log(f"baseline hinge={base_scores['hinge']:.3f} within={base_scores['within']:.2f} "
                  f"on {base_scores['n']} points")
        cur = base_scores

        # ---- coordinate descent -----------------------------------------------------------------------
        for key, values in SEARCH_SPACE:
            self._check()
            self.state["phase"] = f"tuning {key}"
            self._publish()
            trials: list[dict] = []
            for v in values:
                cand = {**best, key: v}
                if cand == {**best}:         # the current value — already measured as `cur`
                    continue
                sc = self._score_corpus(cand, self.train)
                trials.append({"param": key, "value": v, **sc})
                self._log(f"  {key}={v}: hinge={sc['hinge']:.3f} within={sc['within']:.2f}")
            if not trials:
                continue
            trials.sort(key=lambda t: (t["hinge"], -t["within"]))
            win = trials[0]
            self.state["tried"].append(win)
            # Accept the round only on a real margin, so the search does not wander on noise.
            if win["hinge"] < cur["hinge"] * (1.0 - MIN_GAIN):
                best = {**best, key: win["value"]}
                cur = {k: win[k] for k in ("n", "hinge", "within", "mean")}
                self._log(f"  -> {key}={win['value']} accepted (hinge {cur['hinge']:.3f})")
            self.state["best"] = {"params": dict(best), **cur}

        if not best:
            self.state.update({"running": False, "phase": "done", "adopted": None,
                               "note": "No parameter change beat the current detector — nothing adopted."})
            return self.state

        # ---- held-out: did it generalise, or only fit? ------------------------------------------------
        if self.hold:
            self.state["phase"] = "checking held-out scans"
            self._publish()
            hb = self._score_corpus({}, self.hold)
            ha = self._score_corpus(best, self.hold)
            self.state["holdout"] = {"before": hb, "after": ha,
                                     "improved": bool(ha["hinge"] < hb["hinge"] * (1.0 - MIN_GAIN))}
            self._log(f"held-out: hinge {hb['hinge']:.3f} -> {ha['hinge']:.3f}, "
                      f"within {hb['within']:.2f} -> {ha['within']:.2f}")
            if not self.state["holdout"]["improved"]:
                self.state.update({"running": False, "phase": "done", "adopted": None,
                                   "note": (f"Fitted the corrected scans but did NOT generalise: on "
                                            f"{len(self.hold)} held-out scan(s) the error went "
                                            f"{hb['hinge']:.2f} -> {ha['hinge']:.2f}. Nothing adopted — this "
                                            f"is what over-fitting a small corpus looks like, and it means "
                                            f"more corrected scans, not different parameters.")})
                return self.state

        # ---- regression guard: approved scans must not move ------------------------------------------
        self.state["phase"] = "checking approved scans"
        self._publish()
        moved = self._guard_shift(best)
        self.state["guard"] = moved
        if moved and (moved["median"] > GUARD_MEDIAN_PX or moved["p95"] > GUARD_P95_PX):
            self.state.update({"running": False, "phase": "done", "adopted": None,
                               "note": (f"Rejected: it would move the surface on approved scans by "
                                        f"{moved['median']:.2f} px median / {moved['p95']:.2f} px p95, "
                                        f"over the {GUARD_MEDIAN_PX}/{GUARD_P95_PX} px guard.")})
            return self.state

        self.state.update({"running": False, "phase": "done", "adopted": dict(best),
                           "note": (f"Adopted {len(best)} parameter change(s): corrections reproduced within "
                                    f"{TOL_PX:g} px went from {base_scores['within']*100:.0f}% to "
                                    f"{cur['within']*100:.0f}%.")})
        return self.state

    # ---- scoring --------------------------------------------------------------------------------------
    def _score_corpus(self, overrides: dict, scans: list[dict] | None = None) -> dict:
        """Pooled score over the given scans (default: the training half). Volumes are loaded per scan and
        released immediately — holding them would be over a gigabyte for a modest corpus."""
        errs_n = 0; hinge_sum = 0.0; within_sum = 0.0; mean_sum = 0.0
        for entry in (self.corpus if scans is None else scans):
            self._check()
            sag = self._load(entry)
            try:
                surf = self._detect(sag, entry, overrides)
            finally:
                del sag
            sc = score_surface(surf, entry["pts"])
            if sc["n"]:
                errs_n += sc["n"]
                hinge_sum += sc["hinge"] * sc["n"]
                within_sum += sc["within"] * sc["n"]
                mean_sum += sc["mean"] * sc["n"]
            self.state["done"] += 1
            self._publish()
        if not errs_n:
            return {"n": 0, "hinge": float("inf"), "within": 0.0, "mean": float("inf")}
        return {"n": errs_n, "hinge": hinge_sum / errs_n,
                "within": within_sum / errs_n, "mean": mean_sum / errs_n}

    def _guard_shift(self, overrides: dict) -> dict | None:
        """How far the candidate moves the detected surface on scans the reviewer already approved.
        Measured as |candidate - current| over the whole surface, pooled across the guard sample."""
        diffs: list[np.ndarray] = []
        for entry in self.guard:
            self._check()
            sag = self._load(entry)
            try:
                a = self._detect(sag, entry, {})
                b = self._detect(sag, entry, overrides)
            finally:
                del sag
            if a.shape == b.shape:
                diffs.append(np.abs(a - b).ravel())
        if not diffs:
            return None
        d = np.concatenate(diffs)
        return {"n": int(d.size), "median": float(np.median(d)), "p95": float(np.percentile(d, 95)),
                "max": float(d.max())}


def main() -> int:
    """Run one tuning pass in THIS process and report through files.

    Invoked by the sidecar as a subprocess, for the reason in the module note: the search forks a worker pool
    per detection, and forking the server process deadlocks. A fresh interpreter has no server threads to
    inherit, so the pool behaves exactly as it does under the oct_preprocess CLI.

    --job     JSON: {"corpus": [...], "guard": [...], "workers": N}
    --status  written continuously; the sidecar polls it
    --cancel  sentinel path; the sidecar creates it to stop the run at the next checkpoint
    """
    ap = argparse.ArgumentParser(description="Global detector tuning from reviewer corrections")
    ap.add_argument("--job", required=True)
    ap.add_argument("--status", required=True)
    ap.add_argument("--cancel", default=None)
    args = ap.parse_args()

    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)
    # anchors survive JSON as lists; the scorer wants tuples
    corpus = [{**c, "pts": [tuple(p) for p in (c.get("pts") or [])]} for c in (job.get("corpus") or [])]
    run = TuneRun(corpus, job.get("guard") or [], workers=int(job.get("workers") or 4),
                  log=lambda m: print(m, flush=True),
                  status_path=args.status, cancel_path=args.cancel)
    st = run.run()
    return 0 if st.get("phase") in ("done", "cancelled") else 1


if __name__ == "__main__":
    sys.exit(main())
