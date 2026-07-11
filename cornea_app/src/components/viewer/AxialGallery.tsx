import { useEffect, useMemo, useRef, useState } from "react";
import { CircularProgress } from "@mui/material";
import { api, resourceUrl } from "../../api/client";
import { useCaseStore } from "../../store/caseStore";
import { useWorkflowStore } from "../../store/workflowStore";
import type { CaseInfo } from "../../api/types";

// AXIAL fix-tool: correct the anterior corneal surface in the AXIAL (B-scan) plane — a FIXED FRAME shown as
// lateral×depth, the surface dragged ACROSS LATERALS onto the true band. This reaches apex/limbus notches on the
// low-signal first/last frames that the SAGITTAL fix-columns tool (which corrects along the FRAME axis) cannot.
//
// COORDINATES. The niivue viewer is what the user scrubs; we OPEN on (and number by) the niivue AXIAL slice they
// selected. niivue mirrors both axes vs the array (verified via manifest.defect_marks): niivue axial slice s ↔
// array frame (n-1-s); niivue column c ↔ array lateral (n-1-c). So: navigate + label in niivue-slice space
// (matches the niivue toolbar's "axial slice N/M"); convert to the ARRAY frame `f = nFrames-1-niiSlice` for the
// endpoints + anchors; and scaleX(-1)-flip the panel + invert the drag's screen-x so array lateral 0 is on the
// VISUAL RIGHT. Backend contract (all in ARRAY indices):
//   GET  oct-axial-slice?frame=f   → corrected-output B-scan PNG (depth rows × lateral cols, depth 0 = TOP)
//   POST oct-axial-curve {slice_index:f} → {edge[lateral], fit, n_lateral, depth_vox, n_frames}
//   POST oct-axial-redetect {axial_anchors:{f:{lateral:depth}}} → persist sticky GT (Confirm); Run = oct-preprocess
type AxMap = Map<number, Map<number, number>>;   // ARRAY frame -> lateral -> depth (corrected-output depth space, 0 = TOP)

const LAT_SP = 0.0078, DEP_SP = 0.0031;          // Avanti physical spacing (mm) → correct B-scan aspect

function cloneMap(m: AxMap): AxMap { const o: AxMap = new Map(); m.forEach((inner, f) => o.set(f, new Map(inner))); return o; }
function sig(m: AxMap): string {
  return [...m.keys()].sort((a, b) => a - b).map((f) =>
    `${f}:${[...(m.get(f) as Map<number, number>).entries()].sort((a, b) => a[0] - b[0])
      .map(([l, d]) => `${l}=${Math.round(d)}`).join(",")}`).join(";");
}
function ocParams(ci: CaseInfo | null): Record<string, unknown> {
  return ((ci?.manifest as Record<string, unknown> | undefined)?.oct_params as Record<string, unknown> | undefined) ?? {};
}

export function AxialGallery({ filterCss, readOnly = false, initialSlice = 0, sliceTotal = 0 }:
  { filterCss?: string; readOnly?: boolean; initialSlice?: number; sliceTotal?: number }) {
  const caseId = useCaseStore((s) => s.caseId);
  const caseInfo = useCaseStore((s) => s.caseInfo);
  const openCase = useCaseStore((s) => s.openCase);
  const segVersion = useWorkflowStore((s) => s.segVersion);
  const wfSet = useWorkflowStore((s) => s.set);

  // persisted axial anchors (keyed by ARRAY frame) → seed the editable set
  const persistedSig = JSON.stringify(ocParams(caseInfo).axial_anchors ?? {});
  const persisted = useMemo(() => {
    const o: AxMap = new Map();
    try {
      for (const [f, lats] of Object.entries(JSON.parse(persistedSig) as Record<string, Record<string, number>>)) {
        const inner = new Map<number, number>();
        for (const [l, d] of Object.entries(lats)) inner.set(Number(l), Number(d));
        if (inner.size) o.set(Number(f), inner);
      }
    } catch { /* ignore malformed */ }
    return o;
  }, [persistedSig]);

  // navigate in NIIVUE-slice space (matches the viewer); nFrames from the curve response (hint = niivue slice count).
  const [niiSlice, setNiiSlice] = useState(initialSlice);
  const [nFrames, setNFrames] = useState(sliceTotal > 1 ? sliceTotal : 0);
  const [nLateral, setNLateral] = useState(0);
  const [depthVox, setDepthVox] = useState(0);
  const [edge, setEdge] = useState<number[] | null>(null);
  const [anchors, setAnchors] = useState<AxMap>(new Map());
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  const nTot = nFrames > 1 ? nFrames : (sliceTotal > 1 ? sliceTotal : 0);
  const arrayFrame = nTot > 1 ? Math.max(0, Math.min(nTot - 1, (nTot - 1) - niiSlice)) : 0;   // niivue slice → array frame

  useEffect(() => { setAnchors(cloneMap(persisted)); }, [persistedSig]);   // re-seed on case load / after Confirm

  // detected surface for the current ARRAY frame (edge over laterals). Re-fetches after a Run (segVersion).
  useEffect(() => {
    if (!caseId) { setEdge(null); return; }
    let cancel = false;
    api.json<{ n_lateral: number; depth_vox: number; n_frames: number; frame: number; edge: number[]; fit: number[] }>(
      `/api/case/${caseId}/oct-axial-curve`, "POST", JSON.stringify({ slice_index: arrayFrame }))
      .then((r) => { if (cancel) return; setEdge(r.edge); setNLateral(r.n_lateral); setDepthVox(r.depth_vox); setNFrames(r.n_frames); })
      .catch(() => { if (!cancel) { setEdge(null); setMsg("Preprocess this scan first."); } });
    return () => { cancel = true; };
  }, [caseId, arrayFrame, segVersion]);

  const imgSrc = caseId && nTot > 1 ? resourceUrl(`/api/case/${caseId}/oct-axial-slice?frame=${arrayFrame}&t=${segVersion}`) : null;
  const curAnchors = anchors.get(arrayFrame) ?? null;
  const anchorCount = useMemo(() => { let n = 0; anchors.forEach((m) => { n += m.size; }); return n; }, [anchors]);
  // edited frames, shown as niivue-slice numbers (1-based) so they match what the user scrubs
  const editedSlices = useMemo(() => [...anchors.keys()].filter((f) => (anchors.get(f)?.size ?? 0) > 0)
    .map((f) => (nTot > 1 ? (nTot - 1 - f) : f) + 1).sort((a, b) => a - b), [anchors, nTot]);
  const dirty = sig(anchors) !== sig(persisted);

  const edgeY = (l: number): number => {
    const a = curAnchors?.get(l);
    if (a != null) return a;                       // un-confirmed drag → WYSIWYG (the line you're moving is visible)
    return edge ? edge[l] : 0;
  };
  const spanPts = (): string => {
    if (!edge || nLateral < 2) return "";
    const pts: string[] = [`0,${edgeY(0)}`];
    for (let l = 0; l < nLateral; l++) pts.push(`${l + 0.5},${edgeY(l)}`);
    pts.push(`${nLateral},${edgeY(nLateral - 1)}`);
    return pts.join(" ");
  };

  // drag: screen → (lateral, depth). The panel is scaleX(-1)-flipped so lateral 0 is on the VISUAL RIGHT;
  // invert the screen-x fraction (1 - fx) to recover the array lateral. A drag within 1px of the detected
  // edge clears that anchor (snap-back), so you can undo by dragging back onto the line.
  const dragRef = useRef(false);
  const applyDrag = (clientX: number, clientY: number, svg: SVGSVGElement) => {
    if (readOnly || !edge || nLateral < 2 || depthVox < 2) return;
    const r = svg.getBoundingClientRect();
    const lateral = Math.round((1 - (clientX - r.left) / r.width) * nLateral - 0.5);
    if (lateral < 0 || lateral >= nLateral || lateral >= edge.length) return;
    const depth = Math.round(Math.max(0, Math.min(depthVox - 1, ((clientY - r.top) / r.height) * depthVox)));
    setAnchors((prev) => {
      const o = cloneMap(prev);
      let inner = o.get(arrayFrame); if (!inner) { inner = new Map(); o.set(arrayFrame, inner); }
      if (Math.abs(depth - (edge[lateral] ?? depth)) <= 1) inner.delete(lateral);   // back on the detected line → clear
      else inner.set(lateral, depth);
      if (inner.size === 0) o.delete(arrayFrame);
      return o;
    });
  };
  const onDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (readOnly) return;
    dragRef.current = true; e.currentTarget.setPointerCapture(e.pointerId); applyDrag(e.clientX, e.clientY, e.currentTarget);
  };
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => { if (dragRef.current) applyDrag(e.clientX, e.clientY, e.currentTarget); };
  const onUp = () => { dragRef.current = false; };

  const anchorsToApi = (m: AxMap) => {
    const o: Record<string, Record<string, number>> = {};
    m.forEach((inner, f) => {
      const io: Record<string, number> = {};
      inner.forEach((d, l) => { io[String(l)] = Math.round(d); });
      if (Object.keys(io).length) o[String(f)] = io;
    });
    return o;
  };

  const confirm = async () => {
    if (!caseId || busy) return;
    setBusy(true); setMsg("");
    try {
      await api.json(`/api/case/${caseId}/oct-axial-redetect`, "POST", JSON.stringify({ axial_anchors: anchorsToApi(anchors) }));
      await openCase();                            // refresh oct_params (persisted anchors → enables Run)
      setMsg(anchorCount ? `Saved ${anchorCount} point(s) on ${editedSlices.length} slice(s) — press Run to apply.` : "Cleared.");
    } catch { setMsg("Confirm failed."); } finally { setBusy(false); }
  };
  const run = async () => {
    if (!caseId || busy) return;
    setBusy(true); setMsg("Running preprocessing…");
    try {
      const hasBorder = Object.keys((ocParams(caseInfo).border_anchors as Record<string, unknown>) ?? {}).length > 0;
      await api.json(`/api/case/${caseId}/oct-preprocess`, "POST", JSON.stringify(hasBorder ? { use_redetect: true } : {}));
      await openCase();
      wfSet("segVersion", segVersion + 1);         // re-fetch the corrected B-scan + surface (warp applied)
      setMsg("Done — slice(s) warped onto your curve.");
    } catch { setMsg("Run failed."); } finally { setBusy(false); }
  };
  const clearFrame = () => setAnchors((prev) => { const o = cloneMap(prev); o.delete(arrayFrame); return o; });

  // Concrete pixel box for the B-scan (aspect-ratio alone collapses to 0 without a base dimension) — measure the
  // editor host and fit the physical aspect inside it (mirrors SliceGallery's bDispW/bDispH).
  const hostRef = useRef<HTMLDivElement>(null);
  const [host, setHost] = useState({ w: 0, h: 0 });
  useEffect(() => {
    const el = hostRef.current; if (!el) return;
    const upd = () => setHost({ w: Math.max(0, el.clientWidth - 16), h: Math.max(0, el.clientHeight - 16) });
    const ro = new ResizeObserver(upd); ro.observe(el); upd();
    return () => ro.disconnect();
  }, []);

  const aspect = nLateral > 0 && depthVox > 0 ? (nLateral * LAT_SP) / (depthVox * DEP_SP) : 2;
  let dispW = 0, dispH = 0;
  if (host.w > 1 && host.h > 1 && aspect > 0) {
    dispW = host.w; dispH = dispW / aspect;
    if (dispH > host.h) { dispH = host.h; dispW = dispH * aspect; }
  }
  const btn = (extra?: React.CSSProperties): React.CSSProperties => ({
    padding: "3px 10px", borderRadius: 6, fontSize: 12, lineHeight: 1.4, cursor: "pointer",
    border: "1px solid var(--c-border)", background: "var(--c-surface)", color: "var(--c-text)", ...extra });
  const maxSlice = Math.max(0, nTot - 1);

  return (
    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column",
                  background: "var(--c-bg)", color: "var(--c-text)", zIndex: 20 }}>
      {/* toolbar — navigate in niivue-slice space so it matches the viewer */}
      <div style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 10px", flexWrap: "wrap",
                    borderBottom: "1px solid var(--c-border)", flex: "none" }}>
        <span style={{ fontSize: 12, fontWeight: 600 }}>Fix axial</span>
        <span style={{ fontSize: 12, opacity: 0.75 }}>axial slice {niiSlice + 1} / {nTot || "…"}</span>
        <button style={btn()} disabled={busy} onClick={() => setNiiSlice(0)}>⏮ first</button>
        <button style={btn()} disabled={busy || niiSlice <= 0} onClick={() => setNiiSlice((s) => Math.max(0, s - 1))}>◀</button>
        <input type="range" min={0} max={maxSlice} value={Math.min(niiSlice, maxSlice)} disabled={busy || nTot < 2}
               onChange={(e) => setNiiSlice(Number(e.target.value))} style={{ width: 160 }} />
        <button style={btn()} disabled={busy || niiSlice >= maxSlice} onClick={() => setNiiSlice((s) => Math.min(maxSlice, s + 1))}>▶</button>
        <button style={btn()} disabled={busy} onClick={() => setNiiSlice(maxSlice)}>last ⏭</button>
        <span style={{ flex: 1 }} />
        {editedSlices.length > 0 && <span style={{ fontSize: 11, opacity: 0.7 }}>edited: {editedSlices.join(", ")}</span>}
        <button style={btn()} disabled={busy || readOnly || !(curAnchors?.size)} onClick={clearFrame}>Clear slice</button>
        <button style={btn({ borderColor: dirty ? "var(--c-accent)" : "var(--c-border)" })} disabled={busy || readOnly || !dirty} onClick={confirm}>Confirm</button>
        <button style={btn({ background: "var(--c-accent)", color: "#fff", borderColor: "var(--c-accent)" })}
                disabled={busy || readOnly} onClick={run}>Run preprocessing</button>
        {busy && <CircularProgress size={16} />}
      </div>
      {msg && <div style={{ fontSize: 11, padding: "2px 10px", opacity: 0.8, flex: "none" }}>{msg}</div>}
      {/* editor */}
      <div ref={hostRef} style={{ flex: 1, minHeight: 0, position: "relative", display: "flex", alignItems: "center", justifyContent: "center", padding: 8, overflow: "hidden" }}>
        {!edge || nLateral < 2 ? (
          <div style={{ opacity: 0.6, fontSize: 13 }}>{busy ? "Loading…" : "No axial surface — preprocess this scan first."}</div>
        ) : dispW > 1 && dispH > 1 ? (
          <div style={{ position: "relative", width: dispW, height: dispH }}>
            <div style={{ position: "absolute", inset: 0, transform: "scaleX(-1)" }}>
              {imgSrc && <img src={imgSrc} alt="axial B-scan" draggable={false}
                style={{ display: "block", width: "100%", height: "100%", objectFit: "fill",
                         imageRendering: "pixelated", filter: filterCss }} />}
              <svg viewBox={`0 0 ${nLateral} ${depthVox}`} preserveAspectRatio="none"
                   onPointerDown={onDown} onPointerMove={onMove} onPointerUp={onUp} onPointerLeave={onUp}
                   style={{ position: "absolute", inset: 0, width: "100%", height: "100%",
                            cursor: readOnly ? "default" : "row-resize", touchAction: "none" }}>
                {/* detected / dragged anterior surface (red) — the line you drag onto the true band */}
                <polyline fill="none" stroke="#ff4d4d" vectorEffect="non-scaling-stroke"
                          strokeWidth={dirty ? 1.3 : 0.9} opacity={dirty ? 0.95 : 0.75} points={spanPts()} />
                {/* anchored laterals → pink vertical ticks (circles squash under the stretched viewBox) */}
                {curAnchors && [...curAnchors.entries()].map(([l, d], i) => (
                  <line key={`a${i}`} x1={l + 0.5} y1={d - depthVox / 50} x2={l + 0.5} y2={d + depthVox / 50}
                        stroke="#ff5db0" strokeWidth={1.6} vectorEffect="non-scaling-stroke" opacity={0.95} />
                ))}
              </svg>
            </div>
          </div>
        ) : <div style={{ opacity: 0.6, fontSize: 13 }}>Sizing…</div>}
      </div>
      <div style={{ fontSize: 11, padding: "3px 10px", opacity: 0.6, flex: "none", borderTop: "1px solid var(--c-border)" }}>
        Drag the red surface line onto the true corneal band where the auto-detector is off, then Confirm → Run. Corrections are saved per scan and re-applied on every re-run.
      </div>
    </div>
  );
}
