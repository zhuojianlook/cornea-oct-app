/* ──────────────────────────────────────────────────────────
   AppShell — top-level layout, mirrors the multipanelfigure shell.
   DocumentTabs (top) | Sidebar (260px) | Toolbar + workspace.
   ────────────────────────────────────────────────────────── */

import { useEffect } from "react";
import { Alert } from "@mui/material";
import { useCaseStore } from "../../store/caseStore";
import { useUpdater } from "../../store/updaterStore";
import { DocumentTabs } from "./DocumentTabs";
import { Sidebar } from "./Sidebar";
import { TimelineBar } from "./TimelineBar";
import { VolumeCanvas } from "../viewer/VolumeCanvas";
import { UpdateBanner } from "../UpdateBanner";
import { AlignDebugPanel } from "../debug/AlignDebugPanel";
import { useWorkflowStore } from "../../store/workflowStore";

export function AppShell() {
  const config = useCaseStore((s) => s.config);
  const apiError = useCaseStore((s) => s.apiError);
  const fetchConfig = useCaseStore((s) => s.fetchConfig);
  const checkUpdates = useUpdater((s) => s.check);
  const debugOpen = useWorkflowStore((s) => s.debugOpen);

  useEffect(() => {
    // Connect to the sidecar but start blank — don't auto-open the last case or show
    // its segmentation on refresh.
    fetchConfig();
    checkUpdates(false); // silent launch check; the banner only shows if an update exists
  }, [fetchConfig, checkUpdates]);

  if (!config) {
    return (
      <div
        className="flex h-screen w-screen items-center justify-center flex-col gap-4"
        style={{ backgroundColor: "var(--c-bg)", color: "var(--c-text-dim)" }}
      >
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 16 }}>
          <div
            style={{
              width: 40,
              height: 40,
              border: "3px solid rgba(255,255,255,0.15)",
              borderTopColor: "rgba(255,255,255,0.6)",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
            }}
          />
          <span style={{ fontSize: 16, fontWeight: 500, letterSpacing: 0.5 }}>
            Cornea OCT Segmentation
          </span>
          <span style={{ fontSize: 12, opacity: 0.6, maxWidth: 360, textAlign: "center" }}>
            {apiError ?? "Connecting to the segmentation sidecar…"}
          </span>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden select-none">
      {apiError && (
        <Alert
          severity="error"
          sx={{ position: "fixed", top: 0, left: 0, right: 0, zIndex: 9999, borderRadius: 0 }}
        >
          {apiError}
        </Alert>
      )}

      <UpdateBanner />
      <DocumentTabs />

      {/* Fluid: the workspace fills whatever width Chrome allocates (w-full). A modest min-width is
          only a floor so the 3D viewer stays usable on very narrow windows — below it the workspace
          SCROLLS instead of clipping (toolbars also scroll internally). */}
      <div className="flex-1 min-h-0 overflow-x-auto overflow-y-hidden">
        <div className="flex h-full w-full" style={{ minWidth: 760 }}>
          {/* The Debug overlay covers the WORKSPACE but not the sidebar, which would otherwise stay live
              behind it — including "🗑 Wipe all saved cases", an unrecoverable delete of every real case,
              one stray click from a panel the user is only reading. Debug is a read-only view: nothing in
              the sidebar is reachable from it by design, so the whole aside goes inert while it is open.
              `inert` (not just pointer-events) also removes it from tab order and the a11y tree, so a
              keyboard user can't reach the wipe either. DocumentTabs stays live — that's the way out. */}
          <aside
            inert={debugOpen}
            aria-hidden={debugOpen}
            className="flex-none overflow-y-auto overflow-x-hidden border-r"
            style={{
              width: 280,
              minWidth: 280,
              backgroundColor: "var(--c-surface)",
              borderColor: "var(--c-border)",
              opacity: debugOpen ? 0.35 : 1,           // reads as "not part of this view", not as broken
              filter: debugOpen ? "grayscale(1)" : undefined,
              transition: "opacity 120ms linear",
            }}
          >
            <Sidebar />
          </aside>

          {/* The Debug view COVERS the workspace rather than replacing it: unmounting VolumeCanvas would
              leave the niivue singleton bound to a dead canvas (see VolumeCanvas.tsx — a remount re-binds a
              new canvas to the old instance and the 3-D view goes blank). Same overlay pattern VolumeCanvas
              already uses for its own 2-D panels. AlignDebugPanel only mounts when debugOpen, so nothing in
              the Debug tab fetches until it is actually opened. */}
          <main className="relative flex flex-1 flex-col min-w-0 min-h-0">
            <TimelineBar />
            <div className="flex flex-1 min-h-0 min-w-0">
              <VolumeCanvas />
            </div>
            {debugOpen && (
              <div className="absolute inset-0 z-40 flex flex-col" style={{ backgroundColor: "var(--c-bg)" }}>
                <AlignDebugPanel />
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
}
