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
  useEffect(() => { init(); }, [init]);

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
