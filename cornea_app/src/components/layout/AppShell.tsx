/* ──────────────────────────────────────────────────────────
   AppShell — top-level layout, mirrors the multipanelfigure shell.
   DocumentTabs (top) | Sidebar (260px) | Toolbar + workspace.
   ────────────────────────────────────────────────────────── */

import { useEffect } from "react";
import { Alert } from "@mui/material";
import { useCaseStore } from "../../store/caseStore";
import { DocumentTabs } from "./DocumentTabs";
import { Sidebar } from "./Sidebar";
import { Toolbar } from "./Toolbar";
import { StageStepper } from "../stepper/StageStepper";
import { VolumeCanvas } from "../viewer/VolumeCanvas";
import { InspectorPanel } from "../panels/InspectorPanel";

export function AppShell() {
  const config = useCaseStore((s) => s.config);
  const apiError = useCaseStore((s) => s.apiError);
  const fetchConfig = useCaseStore((s) => s.fetchConfig);
  const openCase = useCaseStore((s) => s.openCase);

  useEffect(() => {
    (async () => {
      await fetchConfig();
      await openCase();
    })();
  }, [fetchConfig, openCase]);

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

      <DocumentTabs />

      <div className="flex flex-1 min-h-0 min-w-0">
        <aside
          className="flex-none overflow-y-auto overflow-x-hidden border-r"
          style={{
            width: 280,
            minWidth: 240,
            backgroundColor: "var(--c-surface)",
            borderColor: "var(--c-border)",
          }}
        >
          <Sidebar />
        </aside>

        <main className="flex flex-1 flex-col min-w-0 min-h-0">
          <Toolbar />
          <StageStepper />
          <div className="flex flex-1 min-h-0 min-w-0">
            <VolumeCanvas />
            <aside
              className="flex-none border-l overflow-hidden"
              style={{ width: 340, backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)" }}
            >
              <InspectorPanel />
            </aside>
          </div>
        </main>
      </div>
    </div>
  );
}
