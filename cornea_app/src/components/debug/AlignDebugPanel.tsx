/* Debug tab — REPLICATE-ALIGNMENT COMPARISON.

   Two repeat scans of the same eye, several alignment methods, one magenta/green composite each:
   magenta = fixed, green = moving, white/grey = the two agreeing. Where a method has aligned the pair
   the cornea reads as one grey structure; where it hasn't, you get coloured fringes. That picture is
   the point — the score row is corroboration, not the verdict.

   Deliberately NOT declared in this UI: a winner. The leading methods differ by less than the metric's
   own noise floor (~0.01–0.02), so ordering them by score would manufacture a result the numbers don't
   support. Methods render in a FIXED order with identity first, and the user judges the pixels.

   TWO YARDSTICKS, and they disagree — the reason this panel leads with the residual:
   `primary` (NCC over a dilated tissue mask) is an INTENSITY PROXY. It is dominated by bulk tissue
   overlap, so a residual TILT hardly moves it. On the anchor pair it scored the 2-constant fix (0.8547)
   and brute-force translation (0.8432) as a tie inside the noise floor — and the pictures showed that was
   wrong: the v2→v3 offset is tilted (67 vox at frame 5 → 26 at frame 95), a translation structurally
   cannot remove a tilt, and brute-force's sagittal has a green fringe at one end flipping to magenta at
   the other. `resid_um`/`resid_vox` (mean |Δ| between the two detected anterior surfaces) called it
   correctly: 1.6 vox / ~5 µm for the rigid methods vs 9.6 vox / ~30 µm for brute-force — 6×. That is the
   GEOMETRIC truth, and it is the number that matters: ~30 µm of boundary error lands straight in scar
   Dice when a scar label is propagated onto a replicate, which is the whole point of aligning. */

import { useEffect } from "react";
import { CircularProgress, LinearProgress, MenuItem, Select, ToggleButton, ToggleButtonGroup, Tooltip } from "@mui/material";
import {
  ALIGN_METHODS,
  ALIGN_VIEWS,
  useDebugStore,
  viewUrl,
  type AlignResult,
  type MethodId,
} from "../../store/debugStore";

// case_cs001_os_v2 → "v2"; case_cs030_od_v1_2 → "v1_2" (scheme B replicates keep their full suffix).
const repLabel = (cid: string, eye: string | null): string =>
  eye && cid.startsWith(`case_${eye}_`) ? cid.slice(`case_${eye}_`.length) : cid;

const fmt = (v: number | null | undefined, dp = 4): string =>
  typeof v === "number" && Number.isFinite(v) ? v.toFixed(dp) : "—";
const fmtSigned = (v: number | null | undefined, dp = 4): string =>
  typeof v === "number" && Number.isFinite(v) ? `${v >= 0 ? "+" : ""}${v.toFixed(dp)}` : "—";
const fmtT = (t: number[] | null | undefined): string =>
  Array.isArray(t) && t.length === 3 ? t.map((x) => x.toFixed(3)).join(", ") : "—";

const num = (v: number | null | undefined): v is number => typeof v === "number" && Number.isFinite(v);

// "5 µm (1.6 vox)" — microns first: the residual is a physical boundary error, and µm is the unit the
// scar-Dice consequence is felt in.
const fmtResid = (r: AlignResult): string =>
  num(r.resid_um) ? `${r.resid_um.toFixed(1)} µm${num(r.resid_vox) ? ` (${r.resid_vox.toFixed(1)} vox)` : ""}`
    : num(r.resid_vox) ? `${r.resid_vox.toFixed(1)} vox`
    : "—";

const RESID_TIP =
  "SURFACE RESIDUAL — the geometric truth. Mean |Δ| between the two scans' detected anterior corneal " +
  "surfaces after this method's transform. This is a physical boundary error: it is what lands in scar " +
  "Dice when a scar label is propagated from one replicate onto another, which is what the alignment is " +
  "FOR. Lower is closer, and unlike primary it is not fooled by a residual tilt.";

const TILT_TIP =
  "Residual surface tilt across the frame stack (first frames → last), in voxels. A pure translation " +
  "cannot remove a tilt no matter what it finds — so a method can leave a large tilt here while its " +
  "primary score still looks competitive. Near zero = the two surfaces are parallel, not just overlapping.";

const PRIMARY_TIP =
  "PRIMARY (NCC) — an intensity proxy, not geometry. Computed over a dilated tissue mask, so it is " +
  "dominated by bulk tissue overlap and is nearly BLIND to a tilt: on the anchor pair it scored a " +
  "translation-only method as tied with a rigid one whose surface residual was 6× better. Its own noise " +
  "floor is ~0.01–0.02. Use it to catch gross failures; judge geometry on the residual and the pixels.";

// Don't hijack arrow keys while the user is inside a Select/menu/text field.
const isTypingTarget = (t: EventTarget | null): boolean => {
  const el = t as HTMLElement | null;
  if (!el || !el.closest) return false;
  return (
    /^(INPUT|TEXTAREA|SELECT)$/.test(el.tagName) ||
    !!el.closest('[role="combobox"], [role="listbox"], [role="option"], .MuiSlider-root')
  );
};

const selSx = { fontSize: 12, minWidth: 140, "& .MuiSelect-select": { py: 0.4 } };

export function AlignDebugPanel() {
  const groups = useDebugStore((s) => s.groups);
  const groupsBusy = useDebugStore((s) => s.groupsBusy);
  const groupsLoaded = useDebugStore((s) => s.groupsLoaded);
  const groupsError = useDebugStore((s) => s.groupsError);
  const eye = useDebugStore((s) => s.eye);
  const fixedCase = useDebugStore((s) => s.fixedCase);
  const movingCase = useDebugStore((s) => s.movingCase);
  const methods = useDebugStore((s) => s.methods);
  const running = useDebugStore((s) => s.running);
  const progress = useDebugStore((s) => s.progress);
  const error = useDebugStore((s) => s.error);
  const results = useDebugStore((s) => s.results);
  const jobNote = useDebugStore((s) => s.note);
  const geometry = useDebugStore((s) => s.geometry);
  const view = useDebugStore((s) => s.view);
  const layout = useDebugStore((s) => s.layout);
  const focus = useDebugStore((s) => s.focus);
  const loadGroups = useDebugStore((s) => s.loadGroups);
  const selectEye = useDebugStore((s) => s.selectEye);
  const setFixed = useDebugStore((s) => s.setFixed);
  const setMoving = useDebugStore((s) => s.setMoving);
  const swapPair = useDebugStore((s) => s.swapPair);
  const toggleMethod = useDebugStore((s) => s.toggleMethod);
  const setView = useDebugStore((s) => s.setView);
  const setLayout = useDebugStore((s) => s.setLayout);
  const setFocus = useDebugStore((s) => s.setFocus);
  const cycleFocus = useDebugStore((s) => s.cycleFocus);
  const run = useDebugStore((s) => s.run);

  // Lazy: the eye list is only fetched once this panel actually mounts (i.e. the Debug tab is opened).
  useEffect(() => {
    void loadGroups();
  }, [loadGroups]);

  // ←/→ cycle the method WITHOUT changing the view — the single most useful move in this panel, since
  // flipping the same slice between methods makes a fringe appear/disappear in place.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      if (isTypingTarget(e.target) || results.length === 0) return;
      e.preventDefault();
      cycleFocus(e.key === "ArrowRight" ? 1 : -1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [cycleFocus, results.length]);

  const group = groups.find((g) => g.eye === eye);
  const canRun = !!fixedCase && !!movingCase && fixedCase !== movingCase && !running;
  const identity = results.find((r) => r.method === "identity");
  const focused = results.find((r) => r.method === focus) ?? results[0];
  // Bar scale for the residual column. Identity is included on purpose: it is the honest top of the
  // scale (the unaligned offset), so every other method is drawn as a fraction of "did nothing".
  const maxResid = Math.max(0, ...results.map((r) => (num(r.resid_vox) ? r.resid_vox : 0)));

  const label = (r: AlignResult): string =>
    r.label || ALIGN_METHODS.find((m) => m.id === r.method)?.label || r.method;
  // The STATIC per-method explainer — not the backend's job-level `note` (rendered as a banner below).
  const blurb = (r: AlignResult): string => ALIGN_METHODS.find((m) => m.id === r.method)?.blurb ?? "";
  const wobble = (m: string): string | undefined => ALIGN_METHODS.find((x) => x.id === m)?.wobble;

  /* ── one method's composite + its scores ── */
  const card = (r: AlignResult, big: boolean) => {
    const src = viewUrl(r.views?.[view]);
    const isFocus = r.method === focus;
    const failed = r.raised || r.ok === false;
    return (
      <div
        key={r.method}
        onClick={() => setFocus(r.method as MethodId)}
        style={{
          border: `1px solid ${isFocus ? "var(--c-accent)" : "var(--c-border)"}`,
          borderRadius: 8,
          overflow: "hidden",
          background: "var(--c-surface)",
          cursor: "pointer",
          display: "flex",
          flexDirection: "column",
          minWidth: 0,
        }}
      >
        <div className="px-2 py-1 flex items-center gap-2" style={{ background: "var(--c-surface2)", borderBottom: "1px solid var(--c-border)" }}>
          <Tooltip title={blurb(r)} arrow>
            <span style={{ fontSize: 12, color: "var(--c-text)" }}>{label(r)}</span>
          </Tooltip>
          {r.method === "identity" && (
            <span style={{ fontSize: 9, color: "var(--c-text-dim)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "0 4px" }}>
              REFERENCE
            </span>
          )}
          {r.method === "asis" && (
            <Tooltip title="This is the alignment the shipped app performs today — the baseline, not a proposal." arrow>
              <span style={{ fontSize: 9, color: "var(--c-text-dim)", border: "1px solid var(--c-border)", borderRadius: 4, padding: "0 4px" }}>
                SHIPPING TODAY
              </span>
            </Tooltip>
          )}
          {/* Honest badge: this row's numbers move between identical runs. Surfaced, not silently smoothed. */}
          {wobble(r.method) && (
            <Tooltip title={wobble(r.method)} arrow>
              <span style={{ fontSize: 9, color: "var(--c-text-dim)", border: "1px dashed var(--c-border)", borderRadius: 4, padding: "0 4px", cursor: "help" }}>
                ~±0.01 RUN-TO-RUN
              </span>
            </Tooltip>
          )}
          {failed && (
            <Tooltip
              title="The registration raised (ITK: all samples map outside the moving image buffer). In production this is CAUGHT: align_transform keeps whichever of identity/rigid wins a cornea-Dice check, so the pair simply stays unaligned. Designed fallback — safe, not a crash."
              arrow
            >
              <span style={{ fontSize: 9, color: "#f59e0b", border: "1px solid #f59e0b", borderRadius: 4, padding: "0 4px" }}>
                RAISED → IDENTITY FALLBACK
              </span>
            </Tooltip>
          )}
          <div className="flex-1" />
          {typeof r.runtime_s === "number" && (
            <span style={{ fontSize: 10, color: "var(--c-text-dim)" }}>{r.runtime_s.toFixed(1)}s</span>
          )}
        </div>

        <div style={{ background: "#000", display: "flex", alignItems: "center", justifyContent: "center", minHeight: big ? 0 : 150, flex: big ? 1 : undefined }}>
          {src ? (
            <img
              src={src}
              alt={`${label(r)} — ${view}`}
              draggable={false}
              style={{
                width: big ? "100%" : "100%",
                height: big ? "100%" : undefined,
                objectFit: "contain",
                display: "block",
                imageRendering: "pixelated",
              }}
            />
          ) : (
            <span style={{ fontSize: 11, color: "var(--c-text-dim)", padding: 24 }}>
              {r.error ? r.error : `no ${view} view`}
            </span>
          )}
        </div>

        <div className="px-2 py-1" style={{ fontSize: 10, color: "var(--c-text-dim)", lineHeight: 1.6, borderTop: "1px solid var(--c-border)" }}>
          {/* GEOMETRY FIRST, and emphasised: the residual is the number that agreed with the pictures.
              The intensity proxy is deliberately subordinate below it — it is the one that was wrong. */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0 8px", alignItems: "baseline" }}>
            <span style={{ fontSize: 8, letterSpacing: 0.5, opacity: 0.7 }}>GEOMETRY</span>
            <Tooltip title={RESID_TIP} arrow>
              <span style={{ color: "var(--c-text)", fontSize: 12, cursor: "help" }}>
                residual <b>{fmtResid(r)}</b>
              </span>
            </Tooltip>
            <Tooltip title={TILT_TIP} arrow>
              <span style={{ cursor: "help" }}>tilt {num(r.tilt_vox) ? `${fmtSigned(r.tilt_vox, 1)} vox` : "—"}</span>
            </Tooltip>
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: "0 8px", alignItems: "baseline", opacity: 0.85 }}>
            <Tooltip title={PRIMARY_TIP} arrow>
              <span style={{ fontSize: 8, letterSpacing: 0.5, opacity: 0.7, cursor: "help" }}>PROXY</span>
            </Tooltip>
            <Tooltip title={PRIMARY_TIP} arrow>
              <span style={{ cursor: "help" }}>primary {fmt(r.primary)}</span>
            </Tooltip>
            <span>Δ {fmtSigned(r.delta)}</span>
            <span>rot {fmt(r.rot_deg, 3)}°</span>
            <span>t [{fmtT(r.t_mm)}] mm</span>
            <span>frac_out {fmt(r.frac_out, 4)}</span>
          </div>
          {blurb(r) && (
            <div style={{ fontSize: 9, opacity: 0.75, marginTop: 2, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
              {blurb(r)}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-1 flex-col min-h-0 min-w-0" style={{ backgroundColor: "var(--c-bg)" }}>
      {/* ── controls ── */}
      <div className="flex items-center gap-2 px-3 py-1 border-b flex-wrap" style={{ borderColor: "var(--c-border)", background: "var(--c-surface)" }}>
        <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
          Replicate alignment
        </span>

        <Tooltip title="Eyes with ≥2 repeat scans" arrow>
          <span>
            <Select size="small" value={group ? eye ?? "" : ""} disabled={groupsBusy || running || groups.length === 0}
              onChange={(e) => selectEye(e.target.value as string)} sx={selSx} displayEmpty>
              {groups.length === 0 && <MenuItem value="" sx={{ fontSize: 12 }}>{groupsBusy ? "loading…" : "no eyes"}</MenuItem>}
              {groups.map((g) => (
                <MenuItem key={g.eye} value={g.eye} sx={{ fontSize: 12 }}>
                  {g.eye} ({g.cases.length})
                </MenuItem>
              ))}
            </Select>
          </span>
        </Tooltip>

        <span className="text-[11px]" style={{ color: "#e879f9" }}>fixed</span>
        <Select size="small" value={group && fixedCase ? fixedCase : ""} disabled={!group || running}
          onChange={(e) => setFixed(e.target.value as string)} sx={{ ...selSx, minWidth: 90 }} displayEmpty>
          {(group?.cases ?? []).map((c) => (
            <MenuItem key={c} value={c} sx={{ fontSize: 12 }}>{repLabel(c, eye)}</MenuItem>
          ))}
        </Select>

        <Tooltip title="Swap fixed/moving (swaps which scan is magenta and which is green)" arrow>
          <span>
            <button onClick={swapPair} disabled={running}
              className="text-[11px]"
              style={{ background: "none", border: "1px solid var(--c-border)", borderRadius: 4, color: "var(--c-text-dim)", cursor: running ? "default" : "pointer", padding: "2px 6px" }}>
              ⇄
            </button>
          </span>
        </Tooltip>

        <span className="text-[11px]" style={{ color: "#4ade80" }}>moving</span>
        <Select size="small" value={group && movingCase ? movingCase : ""} disabled={!group || running}
          onChange={(e) => setMoving(e.target.value as string)} sx={{ ...selSx, minWidth: 90 }} displayEmpty>
          {(group?.cases ?? []).map((c) => (
            <MenuItem key={c} value={c} sx={{ fontSize: 12 }}>{repLabel(c, eye)}</MenuItem>
          ))}
        </Select>

        <span style={{ width: 1, height: 18, background: "var(--c-border)" }} />

        {ALIGN_METHODS.map((m) => (
          <Tooltip key={m.id} title={m.wobble ? `${m.blurb} — ${m.wobble}` : m.blurb} arrow>
            <span>
              <label
                className="text-[11px] flex items-center gap-1"
                style={{ color: m.locked ? "var(--c-text-dim)" : "var(--c-text)", cursor: m.locked || running ? "default" : "pointer" }}
              >
                <input type="checkbox" checked={m.locked || !!methods[m.id]} disabled={m.locked || running}
                  onChange={() => toggleMethod(m.id)} style={{ accentColor: "var(--c-accent)", cursor: m.locked || running ? "default" : "pointer" }} />
                {m.label}
              </label>
            </span>
          </Tooltip>
        ))}

        <button onClick={() => void run()} disabled={!canRun}
          className="text-[11px] flex items-center gap-1"
          style={{
            background: canRun ? "var(--c-accent)" : "var(--c-surface2)",
            border: "1px solid var(--c-border)", borderRadius: 4,
            color: canRun ? "#fff" : "var(--c-text-dim)",
            cursor: canRun ? "pointer" : "default", padding: "2px 10px",
          }}>
          {running && <CircularProgress size={11} color="inherit" />}
          {running ? "Running…" : "Run"}
        </button>
      </div>

      {running && <LinearProgress variant={progress > 0 ? "determinate" : "indeterminate"} value={Math.round(progress * 100)} sx={{ height: 2 }} />}

      {/* ── the backend's job-level finding about THIS PAIR ──
          Without this the expert sees a low score and ugly composites and reads METHOD FAILURE, when the
          truth can be DATA MISMATCH: cs005_od v1 vs v9 have different lateral spacing and cover different
          FOVs, so no method can make them agree everywhere. That inverts the purpose of the panel, so the
          note gets a banner — not a tooltip. It lands before the first method finishes. */}
      {jobNote && (
        <div
          className="px-3 py-1 flex items-start gap-2"
          style={{ background: "rgba(245,158,11,0.12)", borderBottom: "1px solid #f59e0b", fontSize: 11, color: "var(--c-text)" }}
        >
          <span style={{ color: "#f59e0b", flex: "none", fontSize: 12, lineHeight: 1.5 }}>⚠</span>
          <div style={{ minWidth: 0 }}>
            <span style={{ color: "#f59e0b", fontWeight: 600 }}>This pair, not the methods: </span>
            {jobNote}
            <div style={{ fontSize: 10, color: "var(--c-text-dim)", marginTop: 1 }}>
              Read the scores below with that in mind — where the two scans do not overlap there is nothing
              for any method to align, so a poor score here is not evidence that the method is at fault.
              {Array.isArray(geometry?.window) && geometry.window.length === 2 && (
                <> Composites are windowed [{geometry.window.map((w) => Math.round(Number(w))).join(", ")}] from the fixed scan.</>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── view switcher + legend ── */}
      <div className="flex items-center gap-3 px-3 py-1 border-b flex-wrap" style={{ borderColor: "var(--c-border)", background: "var(--c-surface2)" }}>
        <ToggleButtonGroup size="small" exclusive value={view} onChange={(_, v) => v && setView(v)}>
          {ALIGN_VIEWS.map((v) => (
            <Tooltip key={v.id} title={v.hint} arrow>
              <ToggleButton value={v.id} sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}>
                {v.label}
              </ToggleButton>
            </Tooltip>
          ))}
        </ToggleButtonGroup>

        <ToggleButtonGroup size="small" exclusive value={layout} onChange={(_, v) => v && setLayout(v)}>
          <Tooltip title="All methods at once, same view — scan across for the odd one out" arrow>
            <ToggleButton value="grid" sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}>Side by side</ToggleButton>
          </Tooltip>
          <Tooltip title="One method large, identity kept alongside — ←/→ flips methods in place" arrow>
            <ToggleButton value="flip" sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}>Flip</ToggleButton>
          </Tooltip>
        </ToggleButtonGroup>

        {/* The legend is not decoration: without it the composite is unreadable. */}
        <span className="flex items-center gap-2 text-[11px]" style={{ color: "var(--c-text-dim)" }}>
          <span style={{ width: 9, height: 9, background: "#e879f9", borderRadius: 2 }} /> fixed
          <span style={{ width: 9, height: 9, background: "#4ade80", borderRadius: 2 }} /> moving
          <span style={{ width: 9, height: 9, background: "#d4d4d8", borderRadius: 2 }} /> aligned (white/grey)
          <span style={{ opacity: 0.8 }}>— colour fringes = disagreement</span>
        </span>

        {results.length > 0 && (
          <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
            ←/→ flips methods
          </span>
        )}
      </div>

      {/* ── cross-method scoreboard ──
          The per-card numbers can't be compared across methods without scanning the whole grid. This strip
          puts them in one aligned column, which is the only way to SEE the finding that motivated it: the
          residual column separates methods that the primary column calls a tie. Deliberately in the same
          fixed order as everything else — no sorting, no winner, no colour-coding of "good"/"bad". */}
      {results.length > 0 && (
        <div className="px-3 py-1 border-b overflow-x-auto" style={{ borderColor: "var(--c-border)", background: "var(--c-surface)" }}>
          <table style={{ fontSize: 10, borderCollapse: "collapse", minWidth: 560 }}>
            <thead>
              <tr style={{ color: "var(--c-text-dim)" }}>
                <th style={{ textAlign: "left", fontWeight: 400, padding: "0 10px 1px 0" }}>method</th>
                <th style={{ textAlign: "right", fontWeight: 400, padding: "0 10px 1px 0" }}>
                  <Tooltip title={RESID_TIP} arrow>
                    <span style={{ color: "var(--c-text)", cursor: "help" }}>surface residual ↓geometry</span>
                  </Tooltip>
                </th>
                <th style={{ textAlign: "right", fontWeight: 400, padding: "0 10px 1px 0" }}>
                  <Tooltip title={TILT_TIP} arrow><span style={{ cursor: "help" }}>tilt</span></Tooltip>
                </th>
                <th style={{ textAlign: "right", fontWeight: 400, padding: "0 10px 1px 0", opacity: 0.8 }}>
                  <Tooltip title={PRIMARY_TIP} arrow><span style={{ cursor: "help" }}>primary ↓proxy</span></Tooltip>
                </th>
                <th style={{ textAlign: "right", fontWeight: 400, padding: "0 10px 1px 0", opacity: 0.8 }}>rot</th>
                <th style={{ textAlign: "right", fontWeight: 400, padding: "0 0 1px 0", opacity: 0.8 }}>frac_out</th>
              </tr>
            </thead>
            <tbody style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}>
              {results.map((r) => (
                <tr
                  key={r.method}
                  onClick={() => setFocus(r.method as MethodId)}
                  style={{ cursor: "pointer", background: r.method === focus ? "var(--c-surface2)" : undefined }}
                >
                  <td style={{ padding: "0 10px 0 0", color: "var(--c-text)", whiteSpace: "nowrap" }}>{label(r)}</td>
                  <td style={{ padding: "0 10px 0 0", textAlign: "right", color: "var(--c-text)", whiteSpace: "nowrap" }}>
                    <span style={{ display: "inline-flex", alignItems: "center", gap: 5, justifyContent: "flex-end" }}>
                      {/* Magnitude bar: shows the SPREAD, which is the whole point — it is not a verdict. */}
                      {num(r.resid_vox) && maxResid > 0 && (
                        <span style={{ width: 44, height: 4, background: "var(--c-surface2)", borderRadius: 2, overflow: "hidden", flex: "none" }}>
                          <span style={{ display: "block", height: "100%", width: `${Math.max(2, (r.resid_vox / maxResid) * 100)}%`, background: "var(--c-text-dim)" }} />
                        </span>
                      )}
                      {fmtResid(r)}
                    </span>
                  </td>
                  <td style={{ padding: "0 10px 0 0", textAlign: "right", color: "var(--c-text-dim)", whiteSpace: "nowrap" }}>
                    {num(r.tilt_vox) ? `${fmtSigned(r.tilt_vox, 1)} vox` : "—"}
                  </td>
                  <td style={{ padding: "0 10px 0 0", textAlign: "right", color: "var(--c-text-dim)", whiteSpace: "nowrap" }}>
                    {fmt(r.primary)}
                    {wobble(r.method) && (
                      <Tooltip title={wobble(r.method)} arrow>
                        <span style={{ cursor: "help", color: "#f59e0b" }}>&nbsp;~</span>
                      </Tooltip>
                    )}
                  </td>
                  <td style={{ padding: "0 10px 0 0", textAlign: "right", color: "var(--c-text-dim)" }}>{fmt(r.rot_deg, 2)}°</td>
                  <td style={{ padding: 0, textAlign: "right", color: "var(--c-text-dim)" }}>{fmt(r.frac_out, 4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ── results ── */}
      <div className="flex-1 min-h-0 overflow-auto p-3">
        {groupsError ? (
          <div className="text-center" style={{ color: "var(--c-red)", fontSize: 12 }}>Couldn't load the replicate list: {groupsError}</div>
        ) : error ? (
          <div className="text-center" style={{ color: "var(--c-red)", fontSize: 12 }}>{error}</div>
        ) : !groupsLoaded && groupsBusy ? (
          <div className="text-center" style={{ color: "var(--c-text-dim)", fontSize: 13 }}>Finding eyes with repeat scans…</div>
        ) : results.length === 0 ? (
          <div className="text-center" style={{ color: "var(--c-text-dim)", fontSize: 13, marginTop: 24 }}>
            {running ? "Aligning — the first methods appear as they finish…" : "Pick a pair and press Run."}
            <div style={{ fontSize: 11, opacity: 0.75, marginTop: 6, maxWidth: 620, marginLeft: "auto", marginRight: "auto" }}>
              Each method's result is drawn as a magenta/green composite of the two scans. Both are windowed from the
              FIXED scan and resampled to a true-mm aspect, so what you see is geometry — not a brightness or
              aspect-ratio difference between the scans.
            </div>
          </div>
        ) : layout === "grid" ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 10, alignItems: "start" }}>
            {results.map((r) => card(r, false))}
          </div>
        ) : (
          <div style={{ display: "flex", gap: 10, height: "100%", minHeight: 0 }}>
            {/* Identity stays pinned in flip layout: the comparison is meaningless without the reference in view. */}
            {identity && (
              <div style={{ width: 240, flex: "none", display: "flex", flexDirection: "column" }}>
                {card(identity, false)}
              </div>
            )}
            <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", gap: 6 }}>
              <ToggleButtonGroup size="small" exclusive value={focus} onChange={(_, v) => v && setFocus(v as MethodId)}>
                {results.map((r) => (
                  <ToggleButton key={r.method} value={r.method} sx={{ py: 0.1, px: 1, fontSize: 11, textTransform: "none" }}>
                    {label(r)}
                  </ToggleButton>
                ))}
              </ToggleButtonGroup>
              <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
                {focused && <div style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column" }}>{card(focused, true)}</div>}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ── the caveat that keeps this honest ── */}
      {results.length > 0 && (
        <div className="px-3 py-1 border-t" style={{ borderColor: "var(--c-border)", background: "var(--c-surface)", fontSize: 10, color: "var(--c-text-dim)" }}>
          <b style={{ color: "var(--c-text)" }}>Two yardsticks, and they can disagree.</b>{" "}
          <b>Surface residual</b> is the geometric one: the mean gap between the two scans' detected anterior surfaces,
          i.e. a physical boundary error — and boundary error is exactly what lands in scar Dice when a scar label is
          propagated onto a replicate, which is what this alignment is for. <b>primary</b> (NCC over a dilated tissue
          mask) is an intensity proxy: it is dominated by bulk tissue overlap and is <b>nearly blind to a tilt</b> — we
          found that the hard way, when it scored a translation-only method as tied with a rigid one whose residual was
          6× better, and the composites showed the translation had left a visible tilt fringe that a translation
          structurally cannot remove. Its noise floor is ~0.01–0.02, so leading methods sit inside it and their primary
          ranking is not a result. No winner is marked here on purpose: use primary to spot gross failures, and judge
          the geometry on the residual and the pixels.
          {identity && typeof identity.primary === "number" && (
            <> Identity scores {fmt(identity.primary)} on this pair; Δ is measured against it.</>
          )}
        </div>
      )}
    </div>
  );
}
