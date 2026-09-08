/* Steps v2 — the "⚙ Steps" overlay (mounted by VolumeCanvas as `absolute inset-0 z-20`). Two tabs:
     • Run tree  — every step the LAST run actually took (reviewer inputs included) as a workflow tree,
                   read from the run record (GET/POST oct-run-graph; nothing is re-run) + the
                   "Export for publication" button (figure/diagram/methods/zip).            → RunTree
     • Detector filmstrip — the former StepsViewer body, unchanged: replays the DEFAULT detector path on
                   the raw .OCT (POST oct-preprocess-steps, heavy) — mounted lazily, only when picked.
                                                                                         → DetectorFilmstrip
   Export name + `{ onClose }` prop are unchanged so VolumeCanvas needs no change; the "← 3D view" button
   text and the phrase "preprocessing decision tree" are pinned by tests/e2e/step2.spec.ts. */

import { useState } from "react";
import { ToggleButton, ToggleButtonGroup } from "@mui/material";
import { RunTree } from "./RunTree";
import { DetectorFilmstrip } from "./DetectorFilmstrip";

type Tab = "tree" | "filmstrip";

export function StepsViewer({ onClose }: { onClose: () => void }) {
  const [tab, setTab] = useState<Tab>("tree");

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      <div className="flex items-center gap-2 px-3 py-1 border-b flex-wrap"
        style={{ borderColor: "var(--c-border)", background: "var(--c-surface)" }}>
        <button onClick={onClose}
          style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-accent)", cursor: "pointer", fontSize: 12, padding: "2px 8px" }}>
          ← 3D view
        </button>
        <ToggleButtonGroup exclusive size="small" value={tab}
          onChange={(_, v: Tab | null) => { if (v) setTab(v); }}>
          <ToggleButton value="tree" sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}
            title="What the last run did, step by step, including your inputs (read from the run record — nothing re-runs)">
            🌳 Run tree
          </ToggleButton>
          <ToggleButton value="filmstrip" sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}
            title="Replay the default detector path on the raw scan for one slice (re-runs the worker; slow)">
            🎞 Detector filmstrip
          </ToggleButton>
        </ToggleButtonGroup>
        <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
          {tab === "tree"
            ? "Preprocessing decision tree — every step of the LAST run, including your inputs. Click a node for details."
            : "Preprocessing decision tree — the default detector path replayed on the raw scan (not the corrections run)."}
        </span>
      </div>
      {tab === "tree" ? <RunTree /> : <DetectorFilmstrip />}
    </div>
  );
}
