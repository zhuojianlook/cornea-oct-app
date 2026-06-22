import { useEffect } from "react";
import { useStore } from "./store/annotatorStore";
import { LoginGate } from "./components/LoginGate";
import { VolumeBrowser } from "./components/VolumeBrowser";
import { AnnotatorCanvas } from "./components/AnnotatorCanvas";
import { PaintToolbar } from "./components/PaintToolbar";
import { SaveBar } from "./components/SaveBar";
import { StatusBar } from "./components/StatusBar";
import { UpdateBanner } from "./components/UpdateBanner";

export default function App() {
  const activeUser = useStore((s) => s.activeUser);
  const init = useStore((s) => s.init);
  const checkUpdates = useStore((s) => s.checkUpdates);
  useEffect(() => { init(); checkUpdates(false); }, [init, checkUpdates]);

  // Keyboard shortcuts (active once a user is in the annotator; ignored while typing).
  useEffect(() => {
    if (!activeUser) return;
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable)) return;
      const s = useStore.getState();
      if (!s.loaded) return;
      const meta = e.ctrlKey || e.metaKey;
      if (meta) {
        const k = e.key.toLowerCase();
        if (k === "z") { e.preventDefault(); if (e.shiftKey) s.redo(); else s.undo(); }
        else if (k === "y") { e.preventDefault(); s.redo(); }
        else if (k === "s") { e.preventDefault(); void s.save(); }
        return;
      }
      let hit = true;
      switch (e.key) {
        case "1": s.setTool("paint"); s.setPenLabel(1); break;
        case "2": s.setTool("paint"); s.setPenLabel(2); break;
        case "3": s.setTool("paint"); s.setPenLabel(3); break;
        case "e": case "E": s.setTool("paint"); s.setPenLabel(0); break;
        case "b": case "B": s.setTool("paint"); break;
        case "w": case "W": s.setTool("wand"); break;
        case "n": case "N": s.setTool("navigate"); break;
        case "[": s.setPenSize(Math.max(1, s.penSize - 2)); break;
        case "]": s.setPenSize(Math.min(60, s.penSize + 2)); break;
        case "f": case "F": void s.smartFill(); break;
        case "Enter": if (s.canConfirm) s.confirmFill(); break;
        case "+": case "=": s.zoomIn(); break;
        case "-": case "_": s.zoomOut(); break;
        case "0": s.resetView(); break;
        default: hit = false;
      }
      if (hit) e.preventDefault();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [activeUser]);

  if (!activeUser) {
    return (
      <div className="flex flex-col h-screen w-screen">
        <UpdateBanner />
        <div className="flex-1 min-h-0"><LoginGate /></div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen w-screen select-none" style={{ backgroundColor: "var(--c-bg)", color: "var(--c-text)" }}>
      <UpdateBanner />
      <SaveBar />
      <PaintToolbar />
      <div className="flex flex-1 min-h-0">
        <aside className="flex-none border-r" style={{ width: 292, backgroundColor: "var(--c-surface)", borderColor: "var(--c-border)" }}>
          <VolumeBrowser />
        </aside>
        <main className="flex flex-1 min-h-0 min-w-0">
          <AnnotatorCanvas />
        </main>
      </div>
      <StatusBar />
    </div>
  );
}
