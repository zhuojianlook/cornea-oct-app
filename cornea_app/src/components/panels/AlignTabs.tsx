/* ──────────────────────────────────────────────────────────
   AlignTabs — the alignment CONTENT (Consensus + pairs / 3-D view / Scrub) shared by the "⧉ Alignment (pairs)…"
   dialog (GroupAlignPanel.tsx) and the step-4 "Aligned" pane in the main area (VolumeCanvas; reviewer spec 2026-09-11
   #3: "the 4. Aligned tab displays what the panel has"). Everything below the next line is the panel as it was.
   "Alignment (pairs)" — the MINIMAL Align-group panel (step 4 "Aligned", reviewer ask 2026-09-11).
   Click "⧉ Align group" → the sidecar runs group_align.register_group for the patient+eye group in a
   subprocess (POST /api/group/{gid}/align) → one card per member: the engine's red/green overlay in both
   planes (reference = red, member moved by the engine's rigid transform = green) with the structure /
   speckle match percentages (relative to the reference's own adjacent-frame ceiling) and every flag.
   Reviewer ask #2 (same day): the FIRST card is the consensus (the fused union canvas with every member's placed
   served line and the consensus curve, group_job.build_consensus) and a "3-D view" tab renders the placed members
   as one RGB volume (reference = red, members = green / blue) in niivue's render mode. Reviewer's point 2026-09-11
   ("technically no single replicate is a reference"): the consensus is now v2 (group_consensus) — the DOME per
   lateral is the MAJORITY of the scans' OWN along-frame domes (the reference is only the coordinate anchor), checked
   per lateral band against the within-B-scan (axial) curvature in physical units, and the REFERENCE SENSITIVITY
   (the consensus re-anchored on every other scan, mapped into the served coordinates) is reported per anchor; the
   card shows the votes per band, the dome-source histogram, the spread per anchor and the group flags.
   Reviewer ask #3 (same day): "are there small axial changes to any or all of the scans such that smoothness …
   and consistency … is maximised?" — the SECOND card ("Applied axial changes") shows, per member, the per-frame
   RIGID shift + tilt (δa, δb·x) group_job.apply_transforms applied — since the 2026-09-11 smooth-dome fix the shift +
   tilt of (consensus − the scan's OWN smooth dome), smooth along frames by construction; the LINE's residual to the
   consensus after the move is reported per scan (RMS + frames off consensus), never applied (line error must never
   move tissue) — the
   residual to the consensus before → after, and the post-transform fused montage (aligned.png); the 3-D tab
   toggles between the pair placement and the applied result (default applied).
   Reviewer ask #4 (same day): "allow the user to scrub through the sagittal views (before and after) of the
   replicates after axial changes are applied" — the "Scrub" tab (AlignScrub): a lateral slider / keys / play over
   per-lateral before-vs-after sagittal composites the sidecar renders from the job's memmapped placements.
   Everything shown comes from the engine's result.json (group_job.py); nothing is written to any case.
   ────────────────────────────────────────────────────────── */
import { useCallback, useEffect, useRef, useState } from "react";
import { Button, Checkbox, CircularProgress, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Slider, Tab, Tabs, ToggleButton, ToggleButtonGroup } from "@mui/material";
import { Niivue, SLICE_TYPE, DRAG_MODE } from "@niivue/niivue";
import { api, resourceUrl } from "../../api/client";
import { destroyNiivue, releaseVolumes, withVolumeNames } from "../../niivue/nvRelease";

export interface AlignProgress {
  phase?: string; members_done?: number; members_total?: number; pairs_done?: number; pairs_total?: number;
  overlays_done?: number; started?: number; error?: string | null;
}
export interface AlignStatus {
  group: string; patient?: string | null; eye?: string | null; members: string[];
  running: boolean; done: boolean; error: string | null; progress: AlignProgress | null;
  result_exists: boolean; result_timestamp?: string | null; reference?: string | null;
  engine_md5_now?: string; engine_md5_result?: string | null; engine_stale?: boolean;
  seconds?: number | null; running_elsewhere?: string | null;
  started?: boolean; already_running?: boolean; cached?: boolean;
  // step-4 additions (2026-09-11): the subgroup of a <patient>_<eye>_s<k> id; the group_aligned stamp state of a job
  // started through /subgroups (stamped_timestamp = the result its scans were stamped for); queue fields.
  subgroup?: string | null;
  stamp?: { requested?: string; members?: string[]; stamped_timestamp?: string | null; stamped?: string[]; stamped_at?: string; unapproved?: string[] } | null;
  queued?: boolean; queued_behind?: string | null;
}
export interface AlignMember {
  cid: string; is_reference: boolean; reference: string; df: number;
  dx_median: number | null; dx_range: (number | null)[]; tilt_median_px: number | null; tilt_abs_median_px: number | null;
  lateral_scale: number; rel_struct: number | null; rel_speckle: number | null; matched_struct: number | null;
  coverage: number | null; measured_frames: number | null; overlap_frames: number | null;
  ok: boolean; flags: string[]; reject_flags: string[]; non_contributing: string | null;
  pose_angle_deg: number | null; seconds: number | null; overlay: string | null; overlay_error?: string;
  shape?: number[]; valid_area?: number; served_source?: string | null;
  // PARTIAL OVERLAP (2026-09-12): the laterals the scan shares with the reference under the served shift, as a fraction of its laterals
  overlap_laterals?: number | null; overlap_fraction?: number | null;
  overlap?: { verdict?: string; offset?: { dx?: number; fraction?: number; source?: string } | null } | null;
  // TRANSITIVE PLACEMENT (2026-09-12): a refused member placed through a contributing member (its route: the pair to that member,
  // the pair engine's round trip through the reverse pair, the composed transform to the reference)
  via?: string | null; route?: AlignRoute | null;
}
export interface AlignRoundtrip { df_error: number; dx_rms: number | null; a_rms_px: number | null; b_rms_px: number | null; a_peak_px: number | null; n_frames: number }
export interface AlignRoute {
  via?: string | null; rel?: number | null; ncc_coarse?: number | null; df?: number | null; dx_median?: number | null; df_via?: number | null;
  frames?: number | null; overlap_fraction_via?: number | null; roundtrip?: AlignRoundtrip | null;
  routes?: { via: string; admissible: boolean; why?: string | null; rel?: number | null; ok?: boolean; reason?: string | null }[];
}
/** result.json['roster'] / scrub meta 'roster' (2026-09-12): EVERY scan of the subgroup with its role — 'reference' | 'contributing' |
    'contributing (via <cid>)' (placed transitively) | 'refused: <flags> (<reason>)' — the reason names the correspondence BEYOND the bar
    when one is known ('best correspondence at ≈ −449 laterals (12% overlap, below the 19% bar)'). placed = it is on the union canvas. */
export interface AlignRosterEntry {
  cid: string; role: string; is_reference: boolean; ok: boolean; placed: boolean; reject_flags?: string[]; reason?: string | null;
  df?: number | null; dx_median?: number | null; overlap_laterals?: number | null; overlap_fraction?: number | null;
  rel_struct?: number | null; ncc_coarse?: number | null;
  via?: string | null; route?: AlignRoute | null;
}
/** consensus v2 (group_consensus.consensus_v2 → summary_of): one lateral band of the axial witness with every voter's vote. */
export interface AlignBandVote { kappa: number | null; dissent: number | null; in_majority_frac: number | null; n_laterals: number; kappa_axial: number | null; kappa_axial_frames: number }
export interface AlignBand {
  band: number; laterals_ref: number[]; laterals_canvas?: number[]; apex: boolean;
  kappa_frames: number | null; kappa_axial: number | null; ratio: number | null; verdict: string; tau: number | null;
  sources: Record<string, number>; votes: Record<string, AlignBandVote>; ratio_at_scan_size_mm?: Record<string, number>;
}
export interface AlignVoter {
  cid: string; role: string; dx_median: number; n_laterals_voting: number; line_rms_median_px?: number | null;
  kappa_by_band: (number | null)[]; dissent_by_band: (number | null)[]; in_majority_by_band: (number | null)[]; dissent_bands: number;
  correction_at_frame_ends_px: number | null; witnessed: boolean; kappa_axial_by_band: (number | null)[]; kappa_own_median?: number | null;
  pair_ok?: boolean | null; rel_struct?: number | null; ncc_coarse?: number | null; flags: string[]; note: string | null;
}
export interface AlignAnchor {
  anchor: string; served: boolean; spread_px: number | null; spread_beyond_pose_px?: number | null; shape_spread_px?: number | null; dome_part_kappa?: number | null;
  median_abs_px: number | null; p90_abs_px?: number | null; n_cells: number;
  pose?: { shift_px: number; tilt_px_half_span: number; trend_px_per_frame: number; dome_px_per_frame2?: number; tilt_trend_px_per_frame?: number } | null;
  pair_roundtrip?: { df_error: number; dx_rms: number | null; a_rms_px: number | null; b_rms_px: number | null; a_peak_px: number | null; n_frames: number } | null;
  contributing: string[]; refused: { cid: string; flags?: string[]; reason?: string | null; rel?: number | null; ncc_coarse?: number | null; df?: number | null; overlap_fraction?: number | null }[];
  vote_only?: string[]; skipped: string | null; kappa_by_band: (number | null)[] | null; sources?: Record<string, number> | null;
  flags?: string[] | null; seconds?: number | null;
}
export interface AlignSensitivity {
  served_anchor: string; note: string; anchors: Record<string, AlignAnchor>; spread_max_px: number | null; shape_spread_max_px?: number | null;
  anchors_evaluated?: number; anchors_skipped: string[]; kappa_band_spread?: (number | null)[]; seconds?: number | null;
}
/** result.json['consensus'] — group_job.build_consensus's summary (consensus.json without the per-lateral table); the v2 fields
    (version 2, group_consensus) are absent on results of the older (provisional) job. */
export interface AlignConsensus {
  note: string; provisional: boolean; n_members: number; members: string[]; reference: string;
  canvas: { origin: number[]; shape: number[] };
  covered_frame_range: (number | null)[]; covered_frames: number;
  curve_model?: { fitted_laterals?: number; lateral_range?: number[] | null };
  rms_to_consensus_px: Record<string, number | null>;
  colours: Record<string, string>; channels: Record<string, string>;
  png: string; volume: string; volume_info?: { shape: number[]; spacing_mm: number[]; datatype: string }; seconds?: number;
  version?: number; revision?: number;
  dome?: { rule: string; tau_rel: number; tau_abs_per_mm: number; sources: Record<string, number>; voted_laterals: number; undecided_fraction: number; verdict: string; smoothing?: Record<string, string>;
    guarantee?: string; n_voters?: number | null; decided_laterals?: number | null;
    toricity?: { rho: number | null; n_laterals: number | null; calibrated: boolean | null; min_laterals?: number | null; note?: string } | null };
  axial_witness?: { envelope: number[]; scale_trusted: boolean; legacy_members: string[]; lateral_spacing_mm: Record<string, number>; verdict: string; apex_band: number | null; rule?: string; half_laterals?: number };
  bands?: AlignBand[]; voters?: AlignVoter[]; voter_ids?: string[]; flags?: string[];
  reference_sensitivity?: AlignSensitivity | null; reference_sensitivity_error?: string | null;
  provisional_curve?: { rms_to_curve_px: Record<string, number | null>; v2_minus_provisional_px: AlignSize; fitted_laterals?: number };
}
/** peak / RMS of a per-frame series over the covered frames (px). */
export interface AlignSize { peak: number | null; rms: number | null; n: number }
/** result.json['transforms'].members[i] — group_job.apply_transforms's per-member summary. */
export interface AlignTransformMember {
  cid: string; is_reference: boolean; df: number;
  delta_a: AlignSize; delta_b: AlignSize; delta_a_smooth: AlignSize; delta_a_jitter: AlignSize; tissue_a: AlignSize; tissue_b: AlignSize;
  rms_before_px: number | null; rms_after_px: number | null; rms_after_all_covered_px: number | null; rms_after_placed_px?: number | null;
  frames: number; covered_frames: number; held_frames: number; profile_beyond_tilt_frames: number;
  after_rms_bar_px: number; after_rms_ok: boolean | null; profile_beyond_tilt: number[];
  /* revision 3 (2026-09-11): the APPLIED δa / δb are fitted on one FIXED inlier set of laterals and smoothed along frames;
     delta_*_raw = the per-frame fits; delta_*_d2_max = max |second difference| of the applied field (bars smooth_bar_d2_*);
     dome_rms_after_px = the DOME part of the residual beyond the shift + tilt (bar after_rms_bar_px = 1.0), profile_rms_px =
     the frame-independent lateral-profile part (reported separately: a rigid move cannot remove it). */
  delta_a_raw?: AlignSize; delta_b_raw?: AlignSize; delta_b_jitter?: AlignSize;
  delta_a_d2_max?: number | null; delta_b_d2_max?: number | null; smooth_bar_d2_a_px?: number; smooth_bar_d2_b_px?: number; smooth_ok?: boolean;
  fixed_inliers?: number; fixed_inlier_source?: string; profile_rms_px?: number | null; residual_rms_after_px?: number | null; after_rms_bar_of?: string;
  /* 2026-09-11 smooth-dome fix: δa / δb are now the shift + tilt of (consensus − the scan's OWN smooth dome); the LINE's
     residual after the move is REPORTED (line_residual_rms_px, frames > line_off_consensus_px flagged), never applied.
     dome_rms_* = consensus − the scan's dome before / beyond the shift + tilt (the after-RMS bar applies to the latter).
     Absent on results of the older job. */
  line_residual_rms_px?: number | null; line_off_consensus_frames?: number; line_off_consensus_px?: number; line_off_consensus?: number[];
  dome_rms_before_px?: number | null; dome_rms_after_px?: number | null; dome_fitted_laterals?: number;
  files?: { volume: string; shape: number[] };
  /* TISSUE GATE (2026-09-12, R4): the dome move was HELD (a_pair / b_pair served, δa = δb = 0) because the scan's tissue disagreement
     with the other placed scans would have risen by more than the threshold */
  dome_move_held?: boolean; dome_move_held_by_tissue?: { before_px: number; after_px: number; rise_px: number; threshold_px: number; iteration?: number } | null;
  hold_note?: string;
}
export interface AlignTissueGate {
  held: string[]; per_scan: Record<string, { before_px: number | null; after_initial_px: number | null; after_px: number | null; held: boolean;
    reason?: { before_px: number; after_px: number; rise_px: number; threshold_px: number; rule?: "scan" | "group"; group_before_px?: number; group_after_px?: number } | null }>;
  group_before_px: number | null; group_after_initial_px: number | null; group_after_px: number | null; ok: boolean | null;
  iterations?: number; group_holds?: number; held_for_group?: string[]; threshold_scan_px?: number; threshold_group_px?: number; rule?: string; seconds?: number; error?: string;
}
export interface AlignTransforms {
  note: string; provisional: boolean; rigid: boolean; reference: string; members: AlignTransformMember[];
  canvas: { origin: number[]; shape: number[] }; png: string; volume: string; volume_pairs: string;
  profile_beyond_tilt_px: number; after_rms_bar_px: number; after_rms_ok_all: boolean; seconds?: number;
  line_off_consensus_px?: number; after_rms_bar_of?: string;
  smooth_ok_all?: boolean; smooth_bar_d2_a_px?: number; smooth_bar_d2_b_px?: number; smooth_bar_of?: string; apply_revision?: number;
  scrub?: { laterals: number; covered_range: number[]; default_lateral: number; bytes: number } | null;
  tissue_gate?: AlignTissueGate | null; held_scans?: string[];
}
export interface AlignResult {
  group: string; patient?: string | null; eye?: string | null; reference: string; members: AlignMember[];
  engine_md5: string; timestamp: string; seconds: number; overlay_url_base: string;
  consensus?: AlignConsensus | null; consensus_error?: string | null; consensus_url?: string; volume_url?: string;
  volume_pairs_url?: string; aligned_url?: string; transforms_url?: string; scrub_url?: string; scrub_meta_url?: string;
  transforms?: AlignTransforms | null; transforms_error?: string | null;
  non_contributing?: Record<string, string>; reference_rule?: Record<string, unknown>;
  transitivity?: unknown[]; status?: AlignStatus;
  roster?: AlignRosterEntry[]; n_members?: number; n_contributing?: number; refused?: AlignRosterEntry[];
}

/** The roster of a result: the job's (2026-09-12) or, for an older result, derived from the member records. */
function rosterOf(result: AlignResult | null): AlignRosterEntry[] {
  if (!result) return [];
  if (result.roster && result.roster.length) return result.roster;
  return (result.members ?? []).map((m) => {
    const placed = m.is_reference || m.ok || !!m.via;
    const parts: string[] = [];
    if (!placed && m.dx_median != null && Number.isFinite(m.dx_median)) parts.push(`offset ≈ ${m.dx_median > 0 ? "+" : ""}${Math.round(m.dx_median)} laterals`);
    if (!placed && m.overlap_fraction != null && Number.isFinite(m.overlap_fraction)) parts.push(`overlap ${Math.round(m.overlap_fraction * 100)}%`);
    const role = m.is_reference ? "reference" : m.ok ? "contributing" : m.via ? `contributing (via ${m.via})`
      : `refused: ${(m.reject_flags ?? []).join(", ") || "refused"}${parts.length ? ` (${parts.join(", ")})` : ""}`;
    return { cid: m.cid, role, is_reference: m.is_reference, ok: m.ok, placed, reject_flags: m.reject_flags, reason: m.non_contributing,
             df: m.df, dx_median: m.dx_median, overlap_laterals: m.overlap_laterals ?? null, overlap_fraction: m.overlap_fraction ?? null,
             rel_struct: m.rel_struct, ncc_coarse: null, via: m.via ?? null, route: m.route ?? null };
  });
}

/** "N of M scans contribute (reference v2; contributing v1, v3); refused: v4 — …; v5 — …" — the line every view leads with
    (2026-09-12: refused scans must never vanish silently from the scrub / 3-D / cards). */
function ContributionLine({ roster, colours, testid }: { roster: AlignRosterEntry[]; colours?: Record<string, string>; testid: string }) {
  if (!roster.length) return null;
  const placed = roster.filter((r) => r.placed);
  const refused = roster.filter((r) => !r.placed);
  const ref = roster.find((r) => r.is_reference);
  const contributing = placed.filter((r) => !r.is_reference);
  const sw = (cid: string) => colours?.[cid]
    ? <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: colours[cid], marginRight: 3, verticalAlign: "middle" }} />
    : null;
  return (
    <div className="text-xs" data-testid={testid} data-contributing={placed.length} data-members={roster.length}
      style={{ padding: "3px 6px", borderRadius: 4, background: refused.length ? "rgba(255,170,40,0.10)" : "rgba(61,220,132,0.08)",
               border: `1px solid ${refused.length ? "var(--c-amber, #ffaa28)" : "var(--c-ok, #3ddc84)"}` }}>
      <b>{placed.length} of {roster.length} scans contribute</b>
      {ref ? <> (reference {sw(ref.cid)}<b>{short(ref.cid)}</b>{contributing.length ? <>; contributing {contributing.map((r, i) => (
        <span key={r.cid} title={r.via ? routeTitle(r) : undefined} data-via={r.via ?? undefined}>{i ? ", " : ""}{sw(r.cid)}{short(r.cid)}{r.via ? <span style={{ color: "var(--c-amber, #ffaa28)" }}> (via {short(r.via)})</span> : null}</span>
      ))}</> : null})</> : null}
      {refused.length ? (
        <>
          {"; "}<b style={{ color: "var(--c-danger, #ff5252)" }}>refused ({refused.length}):</b>{" "}
          {refused.map((r, i) => (
            <span key={r.cid} title={r.reason ?? r.role}>{i ? "; " : ""}<b>{short(r.cid)}</b> — {r.role}</span>
          ))}
          <span style={{ color: "var(--c-text-dim)" }}> · a refused scan is NOT placed on the canvas (no transform) — it is shown unmoved in the Scrub tab's "not aligned" strip and its card below</span>
        </>
      ) : <span style={{ color: "var(--c-text-dim)" }}> · every scan of the subgroup is placed</span>}
    </div>
  );
}

function short(cid: string): string { return cid.startsWith("case_") ? cid.slice(5) : cid; }
/** The route of a transitively placed scan, for a tooltip / the member card (2026-09-12). */
function routeTitle(r: { cid: string; via?: string | null; route?: AlignRoute | null }): string {
  const rt = r.route;
  if (!r.via) return "";
  const rtrip = rt?.roundtrip;
  return `${short(r.cid)} is placed THROUGH ${short(r.via)}: its direct pair with the reference was refused, but it registers to ${short(r.via)}`
    + (rt?.rel != null ? ` (structure match ${Math.round(rt.rel * 100)}% of ceiling` : "(")
    + (rt?.df_via != null ? `, df ${rt.df_via > 0 ? "+" : ""}${rt.df_via}` : "")
    + (rt?.overlap_fraction_via != null ? `, overlap ${Math.round(rt.overlap_fraction_via * 100)}%` : "") + ")"
    + (rtrip ? `; the pair engine's round trip ${short(r.cid)} → ${short(r.via)} → ${short(r.cid)} closes to df ${rtrip.df_error}, dx ${rtrip.dx_rms == null ? "—" : rtrip.dx_rms.toFixed(1)} laterals, a ${rtrip.a_rms_px == null ? "—" : rtrip.a_rms_px.toFixed(1)} px` : "")
    + (rt?.df != null && rt?.dx_median != null ? `; composed transform to the reference: df ${rt.df > 0 ? "+" : ""}${rt.df}, dx ${rt.dx_median > 0 ? "+" : ""}${rt.dx_median.toFixed(1)} laterals` : "")
    + (rt?.routes && rt.routes.length > 1 ? `; other routes: ${rt.routes.filter((x) => x.via !== r.via).map((x) => `${short(x.via)} ${x.admissible ? "admissible" : (x.why ?? "not admissible")}`).join("; ")}` : "");
}
const fmt = (v: number | null | undefined, d = 1, sign = false): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : `${sign && v > 0 ? "+" : ""}${v.toFixed(d)}`;
const pct = (v: number | null | undefined): string =>
  v === null || v === undefined || !Number.isFinite(v) ? "—" : `${Math.round(v * 100)}%`;

// Reviewer 2026-09-12: every replicate has its OWN colour (the job's member_palette) in the 3-D volume, the
// consensus / aligned montages and the scrub blend. CHANNEL_RGB is only the fallback for OLDER results, whose
// volumes were painted R / G / B by member order.
const CHANNEL_RGB: Record<string, string> = { R: "rgb(255,70,70)", G: "rgb(70,230,70)", B: "rgb(90,150,255)" };
const DEFAULT_CAM = { azimuth: 120, elevation: 15 };

/* ── the 3-D view: ONE niivue instance in render mode showing aligned_rgb.nii.gz (RGBA32: colours baked by the
   backend, A = max channel → uncovered cells are transparent). Left-drag rotates, wheel zooms, the slider moves a
   clip plane through the volume. Created when the tab is shown and destroyed when it is left, so at most one extra
   WebGL context is alive (the main viewer's is the other). ───────────────────────────────────────────────────── */
function Align3D({ url, legend, stamp, roster }: { url: string; legend: { cid: string; channel: string; colour?: string | null; rms: number | null; is_reference?: boolean }[]; stamp: string; roster?: AlignRosterEntry[] }) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [clipOn, setClipOn] = useState(false);
  const [clipDepth, setClipDepth] = useState(0);
  const [clipAzi, setClipAzi] = useState(0);
  const [clipElev, setClipElev] = useState(0);
  const [opacity, setOpacity] = useState(1);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    setError(null); setLoading(true);
    if (!canvas.getContext("webgl2")) {
      setError("This window can't provide a WebGL2 context, so the 3-D view is disabled (the cards still work).");
      setLoading(false);
      return;
    }
    let cancelled = false;
    const onLost = (e: Event) => { e.preventDefault(); setError("The WebGL context was lost — close and reopen the 3-D view."); };
    canvas.addEventListener("webglcontextlost", onLost, false);
    let nv: Niivue;
    try {
      nv = new Niivue({
        backColor: [0.05, 0.05, 0.06, 1], show3Dcrosshair: false, isColorbar: false, dragAndDropEnabled: false,
        isNearestInterpolation: false, yoke3Dto2DZoom: true,
        // left-drag = rotate the render (crosshair drag mode is niivue's native 3-D rotate), right/centre = pan.
        mouseEventConfig: { leftButton: { primary: DRAG_MODE.crosshair }, rightButton: DRAG_MODE.pan, centerButton: DRAG_MODE.pan },
      });
      nv.attachToCanvas(canvas);
      nvRef.current = nv;
      if (typeof window !== "undefined") (window as unknown as { alignNv: Niivue }).alignNv = nv; // test hook
    } catch (e) {
      setError(`niivue failed to initialise WebGL: ${e instanceof Error ? e.message : String(e)}`);
      setLoading(false);
      return;
    }
    (async () => {
      // RGBA32 volume: colours + alpha are baked in → no colormap / cal_min / cal_max (they don't apply).
      await nv.loadVolumes(withVolumeNames([{ url: `${url}${url.includes("?") ? "&" : "?"}t=${encodeURIComponent(stamp)}`, opacity: 1 }]));
      if (cancelled) return;
      nv.setSliceType(SLICE_TYPE.RENDER);
      nv.setRenderAzimuthElevation(DEFAULT_CAM.azimuth, DEFAULT_CAM.elevation);
      nv.setClipPlane([2, 0, 0]);   // depth > 1 ⇒ no clipping
      nv.drawScene();
    })()
      .catch((e) => !cancelled && setError(`3-D volume failed to load: ${e instanceof Error ? e.message : String(e)}`))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
      canvas.removeEventListener("webglcontextlost", onLost, false);
      releaseVolumes(nv);
      destroyNiivue(nv);
      nvRef.current = null;
    };
  }, [url, stamp]);

  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || nv.volumes.length === 0) return;
    try { nv.setClipPlane(clipOn ? [clipDepth, clipAzi, clipElev] : [2, 0, 0]); } catch { /* not ready */ }
  }, [clipOn, clipDepth, clipAzi, clipElev, loading]);
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || nv.volumes.length === 0) return;
    try { nv.setOpacity(0, opacity); nv.drawScene(); } catch { /* not ready */ }
  }, [opacity, loading]);

  const reset = () => {
    const nv = nvRef.current;
    setClipOn(false); setClipDepth(0); setClipAzi(0); setClipElev(0); setOpacity(1);
    if (!nv) return;
    try {
      nv.setRenderAzimuthElevation(DEFAULT_CAM.azimuth, DEFAULT_CAM.elevation);
      nv.volScaleMultiplier = 1;
      nv.setClipPlane([2, 0, 0]);
      nv.drawScene();
    } catch { /* best-effort */ }
  };

  return (
    <div className="flex flex-col gap-1" style={{ height: "100%" }} data-testid="align-3d">
      {roster && roster.length > 0 && <ContributionLine roster={roster} testid="align-3d-contribution" />}
      <div className="flex items-center gap-3 flex-wrap text-xs" style={{ color: "var(--c-text-dim)" }}>
        <span>drag = rotate · wheel = zoom · right-drag = pan</span>
        <Button size="small" variant="outlined" onClick={reset} data-testid="align-3d-reset">⟲ Reset view</Button>
        <FormControlLabel sx={{ m: 0 }} label={<span className="text-xs">clip plane</span>}
          control={<Checkbox size="small" checked={clipOn} onChange={(e) => setClipOn(e.target.checked)} />} />
        <span className="flex items-center gap-1" style={{ width: 200 }}>depth
          <Slider size="small" min={-1} max={1} step={0.01} value={clipDepth} disabled={!clipOn} onChange={(_, v) => setClipDepth(v as number)} />
        </span>
        <span className="flex items-center gap-1" style={{ width: 170 }}>azimuth
          <Slider size="small" min={0} max={360} step={5} value={clipAzi} disabled={!clipOn} onChange={(_, v) => setClipAzi(v as number)} />
        </span>
        <span className="flex items-center gap-1" style={{ width: 170 }}>elevation
          <Slider size="small" min={-90} max={90} step={5} value={clipElev} disabled={!clipOn} onChange={(_, v) => setClipElev(v as number)} />
        </span>
        <span className="flex items-center gap-1" style={{ width: 150 }}>opacity
          <Slider size="small" min={0.1} max={1} step={0.05} value={opacity} onChange={(_, v) => setOpacity(v as number)} />
        </span>
        <span className="flex items-center gap-2 flex-wrap" data-testid="align-3d-legend">
          {legend.map((l) => (
            <span key={l.cid} className="flex items-center gap-1">
              <span style={{ width: 11, height: 11, borderRadius: 2, background: l.colour || CHANNEL_RGB[l.channel] || "#aaa", flex: "none" }} />
              <b style={{ color: "var(--c-text)" }}>{l.cid}</b>{l.is_reference ? " (reference)" : ""}{l.rms != null ? ` · RMS ${fmt(l.rms, 1)} px` : ""}
            </span>
          ))}
          <span>· overlap = additive colour{roster && roster.some((r) => !r.placed) ? ` · ${roster.filter((r) => !r.placed).length} refused scan(s) not in this volume` : ""}</span>
        </span>
      </div>
      {error && <div className="text-xs py-1" style={{ color: "var(--c-danger, #ff5252)" }}>⚠ {error}</div>}
      <div style={{ position: "relative", flex: 1, minHeight: 420, background: "#0d0d0f", borderRadius: 4, overflow: "hidden" }}>
        <canvas ref={canvasRef} style={{ width: "100%", height: "100%", display: "block" }} />
        {loading && !error && (
          <div className="flex items-center gap-2 text-xs" style={{ position: "absolute", top: 8, left: 10, color: "#ddd" }}>
            <CircularProgress size={12} color="inherit" /> loading the aligned volume…
          </div>
        )}
      </div>
    </div>
  );
}

/* ── the SCRUB tab (reviewer ask #4, 2026-09-11: "allow the user to scrub through the sagittal views (before and after)
   of the replicates after axial changes are applied"): one composite PNG per canvas lateral from the sidecar
   (GET …/align/sagittal?lateral=L&stage=both — rows BEFORE = pair placement / AFTER = axial changes applied, columns =
   every member in grey + "all" blended with each scan in its own colour; x = frames high-on-the-left, y = depth; member line thin
   in its colour, consensus thick white, line RMS in the titles). A slider / ← → keys (shift = 10) / a ~6 fps play
   loop pick the lateral; the two neighbouring laterals are prefetched as Image objects; a strip under the slider plots
   each member's line RMS to the consensus AFTER (solid) and BEFORE (faint) vs lateral so the user sees where the
   changes helped. ──────────────────────────────────────────────────────────────────────────────────────────────── */
export interface ScrubMeta {
  group: string; reference: string; members: string[]; colours: Record<string, string>; channels: Record<string, string>;
  canvas: { origin: number[]; shape: number[] }; laterals: number; depth: number; frames: number;
  covered_range: number[]; default_lateral: number; stages: string[];
  rms: Record<string, { before: (number | null)[]; after: (number | null)[] }>;
  rms_summary?: Record<string, { before: number | null; after: number | null }>;
  stage_labels?: Record<string, string>; png_url: string; timestamp?: string; bytes?: number; rms_frames?: string;
  // TISSUE-edge metrics (reviewer ask 2026-09-11: "a RMS read out for the axially corrected beneath the current RMS"):
  // measured from the scrub IMAGES (anterior tissue edge per lateral × frame), not the lines — see group_job.tissue_edge_metrics
  tissue_rms?: Record<string, { before: (number | null)[]; after: (number | null)[] }>;
  tissue_disagreement?: { before: (number | null)[]; after: (number | null)[] };
  tissue_summary?: { disagreement_mean: { before: number | null; after: number | null }; rms: Record<string, { before: number | null; after: number | null }>; rule?: string; seconds?: number;
    // R4 (2026-09-12): per scan the mean disagreement with the OTHER placed scans in their overlaps (the tissue gate's number)
    pairwise_scan?: Record<string, { before: number | null; after: number | null }>; pairwise_group?: { before: number | null; after: number | null }; pairwise_rule?: string };
  tissue_error?: string;
  // ANY NUMBER OF MEMBERS (2026-09-12): every scan of the subgroup with its role; 'members' stays the PLACED scans (memmaps / RMS)
  roster?: AlignRosterEntry[]; roles?: Record<string, string>; all_members?: string[]; n_members?: number; n_contributing?: number;
  refused?: { cid: string; role?: string; reason?: string | null; colour?: string; own?: { lateral?: number; note?: string } | null }[];
  /** where each column sits in the composite, so a click picks the scan under it (reviewer 2026-09-12) */
  columns?: { cols: string[]; ml: number; gap: number; panel_w: number; x0: number[] };
}
const SCRUB_FPS = 6;
type StripView = "line" | "tissue";
const hasTissue = (m: ScrubMeta | null | undefined): boolean => !!(m && m.tissue_rms && m.tissue_disagreement && m.tissue_summary);

function RmsStrip({ meta, lateral, onPick, view, onView }: { meta: ScrubMeta; lateral: number; onPick: (l: number) => void; view: StripView; onView: (v: StripView) => void }) {
  // Reviewer spec 2026-09-11 #5: TWO separate graphs, BEFORE and AFTER, stacked, on the SAME y-scale (so a change in
  // height between them is a change in RMS, not in scale). The line|tissue toggle and click-to-jump apply to both.
  const Lc = Math.max(1, meta.laterals);
  const H = 56, W = 1000;                                   // viewBox units; stretched to the container width
  const tissue = view === "tissue" && hasTissue(meta);
  const series = tissue ? meta.tissue_rms! : meta.rms;      // per member {before, after} per lateral
  const dis = tissue ? meta.tissue_disagreement : undefined; // scan-to-scan disagreement (grey), tissue view only
  let vmax = 0.5;
  for (const cid of meta.members) for (const st of ["before", "after"] as const)
    for (const v of series[cid]?.[st] ?? []) if (v != null && Number.isFinite(v)) vmax = Math.max(vmax, v);
  if (dis) for (const st of ["before", "after"] as const) for (const v of dis[st] ?? []) if (v != null && Number.isFinite(v)) vmax = Math.max(vmax, v);
  vmax = Math.min(vmax, 40);                                // a wild lateral must not flatten the strip
  const x = (l: number) => ((l + 0.5) / Lc) * W;
  const y = (v: number) => H - 3 - (Math.min(v, vmax) / vmax) * (H - 8);
  const path = (arr: (number | null)[]) => {
    let d = ""; let pen = false;
    arr.forEach((v, l) => {
      if (v == null || !Number.isFinite(v)) { pen = false; return; }
      d += `${pen ? "L" : "M"}${x(l).toFixed(1)},${y(v).toFixed(1)} `; pen = true;
    });
    return d;
  };
  const [c0, c1] = meta.covered_range ?? [0, Lc - 1];
  const pick = (e: React.MouseEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const l = Math.round(((e.clientX - r.left) / Math.max(1, r.width)) * Lc - 0.5);
    onPick(Math.max(0, Math.min(Lc - 1, l)));
  };
  const tog = (v: StripView, label: string, title: string, disabled = false) => (
    <button type="button" onClick={() => !disabled && onView(v)} disabled={disabled} title={title} data-testid={`align-scrub-strip-${v}`}
      style={{ fontSize: 10, padding: "1px 6px", cursor: disabled ? "default" : "pointer", borderRadius: 3, border: "1px solid var(--c-border, #555)",
               background: view === v ? "var(--c-accent, #4ea1ff)" : "transparent", color: view === v ? "#000" : (disabled ? "var(--c-text-dim)" : "inherit"), opacity: disabled ? 0.5 : 1 }}>
      {label}
    </button>
  );
  const scale = `0–${vmax.toFixed(vmax < 5 ? 1 : 0)} px`;
  const graph = (stage: "before" | "after", title: string, sub: string) => (
    <div className="flex items-center gap-2" style={{ width: "100%" }} data-testid={`align-scrub-graph-${stage}`}>
      <span className="text-[10px]" style={{ color: "var(--c-text-dim)", width: 74, flex: "none", textAlign: "right", lineHeight: 1.15 }} title={sub}>
        <b style={{ color: "var(--c-text)" }}>{title}</b><br />RMS {scale}
      </span>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" onClick={pick} data-testid={stage === "after" ? "align-scrub-strip" : "align-scrub-strip-before"}
        style={{ width: "100%", height: H, display: "block", background: "#101012", borderRadius: 3, cursor: "crosshair", flex: 1 }}>
        <rect x={x(c0) - 0.5 * W / Lc} y={0} width={(c1 - c0 + 1) * W / Lc} height={H} fill="#1d1d22" />
        {meta.members.map((cid) => (
          <path key={cid} d={path(series[cid]?.[stage] ?? [])} stroke={meta.colours[cid] ?? "#aaa"} strokeWidth={1.5} fill="none" vectorEffect="non-scaling-stroke" />
        ))}
        {dis && (
          <path data-testid={`align-scrub-strip-disagreement-${stage}`} d={path(dis[stage] ?? [])} stroke="#bbb" strokeWidth={1.5} fill="none" vectorEffect="non-scaling-stroke" />
        )}
        <line x1={x(lateral)} x2={x(lateral)} y1={0} y2={H} stroke="#fff" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      </svg>
    </div>
  );
  return (
    <div className="flex flex-col gap-1" style={{ width: "100%" }} data-testid="align-scrub-strip-wrap" data-view={tissue ? "tissue" : "line"} data-scale={vmax.toFixed(2)}>
      <div className="flex items-center gap-2 text-[10px]" style={{ color: "var(--c-text-dim)" }}>
        <span className="flex items-center gap-1" style={{ flex: "none" }}>
          {tog("line", "line", "line RMS to the consensus per lateral (the detected line's residual)")}
          {tog("tissue", "tissue", hasTissue(meta) ? "tissue RMS to the consensus per lateral, measured from the images; grey = scan-to-scan tissue disagreement" : "no tissue metrics in this result", !hasTissue(meta))}
        </span>
        <span>{tissue ? "tissue" : "line"} RMS to the consensus per lateral · BEFORE (pair placement) above, AFTER (axial changes applied) below · same scale {scale} · click either graph to jump</span>
      </div>
      {graph("before", "BEFORE", "the pair engine's placement, before the axial changes")}
      {graph("after", "AFTER", "after the axial changes were applied")}
    </div>
  );
}

function AlignScrub({ metaUrl, stamp, onZoom }: { metaUrl: string; stamp: string; onZoom: (url: string, label?: string) => void }) {
  const [meta, setMeta] = useState<ScrubMeta | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [lat, setLat] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fitWidth, setFitWidth] = useState(false);   // off: natural size + horizontal scroll (reviewer 2026-09-12)
  const [view, setView] = useState<StripView>("tissue");     // strip: line | tissue (tissue when the result has it)
  const cache = useRef<Map<string, HTMLImageElement>>(new Map());
  const latRef = useRef(0);
  latRef.current = lat;

  useEffect(() => {
    let cancelled = false;
    setMeta(null); setErr(null); setPlaying(false); cache.current.clear();
    api.json<ScrubMeta>(metaUrl).then((m) => {
      if (cancelled) return;
      setMeta(m); setLat(Math.max(0, Math.min(m.laterals - 1, m.default_lateral ?? 0))); setView(hasTissue(m) ? "tissue" : "line");
    }).catch((e) => !cancelled && setErr(e instanceof Error ? e.message : String(e)));
    return () => { cancelled = true; };
  }, [metaUrl, stamp]);

  const url = useCallback((l: number) => meta ? resourceUrl(`${meta.png_url}?lateral=${l}&stage=both&t=${encodeURIComponent(stamp)}`) : "", [meta, stamp]);
  const Lc = meta?.laterals ?? 1;
  const [c0, c1] = meta?.covered_range ?? [0, Lc - 1];
  const clamp = useCallback((l: number) => Math.max(0, Math.min(Lc - 1, l)), [Lc]);
  const step = useCallback((d: number) => setLat((l) => clamp(l + d)), [clamp]);

  // prefetch the two neighbouring laterals (cached by URL; the sidecar caches the PNG on disk too)
  useEffect(() => {
    if (!meta) return;
    for (const l of [lat - 1, lat + 1, lat - 2, lat + 2]) {
      if (l < 0 || l >= Lc) continue;
      const u = url(l);
      if (cache.current.has(u)) continue;
      const im = new Image(); im.src = u; cache.current.set(u, im);
      if (cache.current.size > 64) { const k = cache.current.keys().next().value; if (k !== undefined) cache.current.delete(k); }
    }
  }, [lat, meta, Lc, url]);

  // ← / → step 1, shift = 10, space = play/pause (ignored while typing in a field or on the slider, which steps itself)
  useEffect(() => {
    const h = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable)) return;
      if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); step((e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 10 : 1)); }
      else if (e.key === " ") { e.preventDefault(); setPlaying((p) => !p); }
    };
    window.addEventListener("keydown", h);
    return () => window.removeEventListener("keydown", h);
  }, [step]);

  // play: step through the covered laterals at ~6 fps, wrapping, until pressed again
  useEffect(() => {
    if (!playing || !meta) return;
    const id = window.setInterval(() => {
      setLat((l) => { const n = l + 1; return n > c1 || n < c0 ? c0 : n; });
    }, Math.round(1000 / SCRUB_FPS));
    return () => window.clearInterval(id);
  }, [playing, meta, c0, c1]);

  const cur = meta ? url(lat) : "";
  useEffect(() => { if (cur) setLoading(true); }, [cur]);
  const fmtR = (v: number | null | undefined) => v == null || !Number.isFinite(v) ? "—" : v.toFixed(2);
  const l0 = meta?.canvas.origin[0] ?? 0;

  if (err) return <div className="text-xs py-1" style={{ color: "var(--c-danger, #ff5252)" }}>⚠ {err}</div>;
  if (!meta) return <div className="flex items-center gap-2 text-xs py-2"><CircularProgress size={13} /> loading the scrub data…</div>;
  return (
    <div className="flex flex-col gap-1" style={{ flex: 1, minHeight: 0 }} data-testid="align-scrub">
      {/* SCRUB CONTROLS — one prominent row: Play, ±10 / ±1 buttons, the wide lateral slider, a typed lateral. */}
      <div className="flex items-center gap-3 text-sm" data-testid="align-scrub-controls"
        style={{ padding: "8px 12px", border: "1px solid var(--c-border, #444)", borderRadius: 8, background: "rgba(255,255,255,0.04)" }}>
        <b style={{ whiteSpace: "nowrap" }}>Sagittal scrub · lateral</b>
        <Button size="medium" variant={playing ? "contained" : "outlined"} onClick={() => setPlaying((p) => !p)} data-testid="align-scrub-play"
          title={`Step through the covered laterals ${c0}–${c1} at ~${SCRUB_FPS} fps (space)`}>{playing ? "⏸ Pause" : "▶ Play"}</Button>
        <Button size="medium" variant="outlined" onClick={() => step(-10)} title="10 laterals left (shift+←)" data-testid="align-scrub-prev10">◀◀ 10</Button>
        <Button size="medium" variant="outlined" onClick={() => step(-1)} title="previous lateral (←)" data-testid="align-scrub-prev">◀ 1</Button>
        <span style={{ flex: 1, minWidth: 260, display: "flex", alignItems: "center", padding: "0 10px" }}>
          <Slider size="medium" min={0} max={Lc - 1} step={1} value={lat} onChange={(_, v) => { setPlaying(false); setLat(v as number); }}
            valueLabelDisplay="on" data-testid="align-scrub-slider" />
        </span>
        <Button size="medium" variant="outlined" onClick={() => step(1)} title="next lateral (→)" data-testid="align-scrub-next">1 ▶</Button>
        <Button size="medium" variant="outlined" onClick={() => step(10)} title="10 laterals right (shift+→)" data-testid="align-scrub-next10">10 ▶▶</Button>
        <input type="number" min={0} max={Lc - 1} step={1} value={lat} data-testid="align-scrub-input"
          onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) { setPlaying(false); setLat(Math.max(0, Math.min(Lc - 1, Math.round(v)))); } }}
          style={{ width: 72, padding: "4px 6px", fontSize: 14, background: "var(--c-bg, #111)", color: "inherit", border: "1px solid var(--c-border, #555)", borderRadius: 4 }}
          title="type a canvas lateral" />
        <span style={{ whiteSpace: "nowrap", color: "var(--c-text-dim)" }}>/ {Lc - 1}</span>
        <FormControlLabel sx={{ ml: 0.5 }} data-testid="align-scrub-fit"
          control={<Checkbox size="small" checked={fitWidth} onChange={(e) => setFitWidth(e.target.checked)} />}
          label={<span className="text-xs">fit width</span>} />
      </div>
      {meta.roster && meta.roster.length > 0 && <ContributionLine roster={meta.roster} colours={meta.colours} testid="align-scrub-contribution" />}
      <div className="flex items-center gap-2 flex-wrap text-xs">
        <span data-testid="align-scrub-readout">
          canvas lateral <b>{lat}</b> / {Lc - 1} (reference lateral {lat + l0}) · covered {c0}–{c1}
          {meta.n_members != null ? <> · <b>{meta.members.length} of {meta.n_members} scans placed</b>{meta.roster?.some((r) => r.via) ? ` (${meta.roster.filter((r) => r.via).map((r) => `${short(r.cid)} via ${short(r.via!)}`).join(", ")})` : ""}{meta.refused?.length ? ` (not aligned: ${meta.refused.map((r) => short(r.cid)).join(", ")})` : ""}</> : null}
          {" · "}line RMS to consensus before → after (placed scans):{" "}
          {meta.members.map((cid) => (
            <span key={cid} style={{ marginRight: 8 }}>
              <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: meta.colours[cid], marginRight: 3, verticalAlign: "middle" }} />
              <b>{cid === meta.reference ? `${cid} (ref)` : cid}</b> {fmtR(meta.rms[cid]?.before?.[lat])} → <b>{fmtR(meta.rms[cid]?.after?.[lat])}</b> px
            </span>
          ))}
        </span>
        {loading && <CircularProgress size={12} />}
        <span style={{ color: "var(--c-text-dim)", marginLeft: "auto" }}>← → step 1 · shift = 10 · space = play · click the image to zoom</span>
      </div>
      {/* the TISSUE readout beneath the line readout (reviewer ask 2026-09-11): the same numbers measured from the images */}
      <div className="flex items-center gap-2 flex-wrap text-xs" data-testid="align-scrub-tissue-readout">
        {hasTissue(meta) ? (
          <span title={meta.tissue_summary?.rule ?? ""}>
            tissue RMS to consensus (measured from the images) before → after:{" "}
            {meta.members.map((cid) => (
              <span key={cid} style={{ marginRight: 8 }}>
                <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: meta.colours[cid], marginRight: 3, verticalAlign: "middle" }} />
                <b>{cid === meta.reference ? `${cid} (ref)` : cid}</b> {fmtR(meta.tissue_rms![cid]?.before?.[lat])} → <b>{fmtR(meta.tissue_rms![cid]?.after?.[lat])}</b> px
              </span>
            ))}
            {" · "}scan-to-scan tissue disagreement before → after:{" "}
            <b data-testid="align-scrub-tissue-disagreement">{fmtR(meta.tissue_disagreement!.before?.[lat])} → {fmtR(meta.tissue_disagreement!.after?.[lat])} px</b>
          </span>
        ) : (
          <span style={{ color: "var(--c-text-dim)" }}>tissue RMS (measured from the images): {meta.tissue_error ? `unavailable (${meta.tissue_error})` : "not in this result — re-run the alignment"}</span>
        )}
      </div>
      <RmsStrip meta={meta} lateral={lat} onPick={(l) => { setPlaying(false); setLat(l); }} view={view} onView={setView} />
      <div className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
        graphs: each scan's {view === "tissue" && hasTissue(meta) ? "TISSUE RMS to the consensus (measured from the images)" : "line RMS to the consensus"} per lateral — top graph = BEFORE (pair placement), bottom graph = AFTER the axial changes, same y-scale;
        {view === "tissue" && hasTissue(meta) ? " grey line = scan-to-scan tissue disagreement (mean over frames of max − min of the scans' tissue edges);" : ""} grey band = covered laterals; click to jump.
        Top rows = BEFORE (the pair engine's placement), bottom rows = AFTER (axial changes applied); columns = each PLACED scan + "all" blended with each scan in its own colour
        (more than 4 scans wrap onto a second row per stage); x = frames (high on the left, as in the app), y = depth; thin colour = the scan's line, thick white = the consensus.
        {meta.refused?.length ? <> A red-labelled <b>NOT ALIGNED</b> strip at the bottom shows each refused scan's OWN middle sagittal (its own lateral L/2, unmoved — it has no transform, so it is not on the canvas) with the reason.</> : null}
      </div>
      {/* Reviewer 2026-09-12: every BEFORE on one row and every AFTER on the next, consensus first — so the composite
          is as wide as the group needs and this box SCROLLS horizontally instead of shrinking the panels. */}
      <div style={{ flex: 1, minHeight: 0, overflowX: "auto", overflowY: "auto" }}>
        {/* Clicking a panel zooms THAT scan (reviewer 2026-09-12: the magnifier opens the scan under the pointer,
            not the whole composite): the click's x in the image's own pixels picks the column from meta.columns and
            the zoom dialog loads …&member=<cid> — that scan's BEFORE / AFTER rendered large. */}
        <img src={cur} alt={`sagittal before/after at lateral ${lat}`} data-testid="align-scrub-img"
          onLoad={() => setLoading(false)} onError={() => setLoading(false)}
          onClick={(e) => {
            const im = e.currentTarget, r = im.getBoundingClientRect();
            const xNat = (e.clientX - r.left) * (im.naturalWidth / Math.max(1, r.width));
            const cols = meta.columns;
            let pick: string | null = null;
            if (cols) {
              for (let i = 0; i < cols.cols.length; i++) {
                if (xNat >= cols.x0[i] - cols.gap && xNat < cols.x0[i] + cols.panel_w + cols.gap) { pick = cols.cols[i]; break; }
              }
            }
            const base = url(lat);
            if (pick) onZoom(`${base}&member=${encodeURIComponent(pick)}`, pick === "all" ? `all scans blended · lateral ${lat}` : `${pick} · lateral ${lat}`);
            else onZoom(base, `sagittal scrub · lateral ${lat}`);
          }}
          title="click a panel to zoom that scan (before / after)"
          style={fitWidth ? { width: "100%", display: "block", cursor: "zoom-in" } : { display: "block", cursor: "zoom-in", maxWidth: "none" }} />
      </div>
    </div>
  );
}

const SOURCE_ORDER = ["majority", "median3", "axial", "continuity", "reference", "single", "median", "none"];
const SOURCE_TITLE: Record<string, string> = {
  majority: "n = 2 or n ≥ 4: the tightest floor(n/2)+1 subset of the scans' own domes within 2 tau — se^-2 weighted mean (n ≥ 4 out-votes one wrong scan)",
  median3: "n = 3: the MEDIAN of the three scans' own domes — accurate to the honest scans' own error; a wrong scan is out-voted only with ≥ 4 scans or the axial witness",
  axial: "no majority: the vote nearest κ_axial × ρ (ρ = the group's own κ*/κ_axial over the decided laterals, a calibrated toricity), plus every vote within tau of it",
  continuity: "no majority and no calibrated axial witness: the nearest decided lateral's value (never the anchor by default)",
  reference: "nothing decided anywhere: the reference's own dome kept (nothing licenses moving it)",
  single: "one voter covers the lateral: its own dome",
  median: "nothing decided and the reference does not vote here: the median vote",
  none: "no vote",
};
const SOURCE_COLOR: Record<string, string> = { majority: "var(--c-ok, #3ddc84)", median3: "#2fb8a8", axial: "var(--c-accent, #4ea1ff)", continuity: "#8a9bb8", reference: "var(--c-amber, #ffaa28)" };
const VERDICT_COLOR: Record<string, string> = { witnessed: "var(--c-ok, #3ddc84)", mismatch: "var(--c-danger, #ff5252)", sign: "var(--c-danger, #ff5252)", no_data: "var(--c-text-dim)" };
const WARN_FLAGS = new Set(["dome_axial_mismatch", "dome_majority_failed", "dome_unwitnessed", "lateral_scale_mismatch", "frame_spacing_mismatch", "reference_only"]);
const flagTitle = (f: string): string => ({
  dome_axial_mismatch: "the along-frame dome of the consensus and the within-B-scan curvature disagree beyond the toricity envelope in ≥ 1 band",
  dome_majority_failed: "more than 20 % of the voted laterals had no majority and fell back to the reference's own dome",
  lateral_scale_unverified: "the header lateral spacing is the legacy 4.0/513 mm cube — the physical lateral scale (and so the axial ratio) is unverified; the ratio is also given for 4.6 / 6.0 mm scan sizes",
  lateral_scale_mismatch: "the scans' lateral spacings differ by more than 2 %",
  frame_spacing_mismatch: "the scans' frame spacings differ by more than 1 %",
  reference_only: "no other scan contributes: the consensus is the reference alone",
  vote_only: "this scan's pair was refused; its own dome votes but it gets no transform",
  dome_unwitnessed: "this scan's own dome disagrees with the majority and nothing (neither two other voters nor a trusted axial check) witnesses the majority",
}[f] ?? f);
const k3 = (v: number | null | undefined): string => v == null || !Number.isFinite(v) ? "—" : v.toFixed(3);

/** The consensus v2 details (majority own dome, axial witness, reference sensitivity, flags) — group_consensus.summary_of. */
function ConsensusV2Details({ cons }: { cons: AlignConsensus }) {
  const bands = cons.bands ?? [];
  const voters = cons.voters ?? [];
  const ids = cons.voter_ids ?? voters.map((v) => v.cid);
  const aw = cons.axial_witness;
  const dome = cons.dome;
  const sens = cons.reference_sensitivity ?? null;
  const flags = cons.flags ?? [];
  const sources = dome?.sources ?? {};
  const voted = dome?.voted_laterals ?? 0;
  const col = (cid: string) => cons.colours[cid] ?? "#aaa";
  return (
    <div className="flex flex-col gap-1 text-[11px]" data-testid="align-consensus-v2">
      {/* group flags */}
      <div className="flex items-center gap-2 flex-wrap" data-testid="align-consensus-flags">
        <span style={{ color: "var(--c-text-dim)" }}>group flags:</span>
        {flags.length === 0 && <b style={{ color: "var(--c-ok, #3ddc84)" }}>none</b>}
        {flags.map((f) => (
          <span key={f} title={flagTitle(f)} style={{ padding: "0 6px", borderRadius: 3, border: "1px solid", fontWeight: 600,
            color: WARN_FLAGS.has(f) ? "var(--c-danger, #ff5252)" : "var(--c-amber, #ffaa28)", borderColor: WARN_FLAGS.has(f) ? "var(--c-danger, #ff5252)" : "var(--c-amber, #ffaa28)" }}>{f}</span>
        ))}
        {dome && <span style={{ color: "var(--c-text-dim)" }} title={dome.guarantee ?? ""}>· dome verdict <b style={{ color: "var(--c-text)" }}>{dome.verdict}</b>{dome.n_voters != null ? ` (${dome.n_voters} voters${dome.decided_laterals != null ? `, ${dome.decided_laterals} laterals decided` : ""})` : ""}</span>}
        {dome?.toricity && <span style={{ color: "var(--c-text-dim)" }} title={dome.toricity.note ?? ""} data-testid="align-consensus-toricity">· calibrated toricity ρ = κ*/κ_axial <b style={{ color: "var(--c-text)" }}>{dome.toricity.calibrated ? fmt(dome.toricity.rho, 3) : "not calibrated"}</b>{dome.toricity.calibrated ? ` (${dome.toricity.n_laterals} laterals)` : ` (< ${dome.toricity.min_laterals ?? 30} decided laterals)`}</span>}
        {dome?.guarantee && dome.n_voters === 3 && <span style={{ color: "var(--c-amber, #ffaa28)" }}>· three scans: the dome is accurate to the honest scans' own error — a wrong scan is out-voted only with ≥ 4 scans or the axial witness</span>}
        {aw && <span style={{ color: "var(--c-text-dim)" }}>· axial witness <b style={{ color: VERDICT_COLOR[aw.verdict.replace("_scale_unverified", "")] ?? "var(--c-text)" }}>{aw.verdict}</b>
          {" "}(envelope {aw.envelope[0]}–{aw.envelope[1]}; lateral scale {aw.scale_trusted ? "trusted" : "UNVERIFIED — legacy 4.0/513 mm header"})</span>}
      </div>
      {/* dome-source histogram */}
      {dome && (
        <div className="flex items-center gap-2 flex-wrap" data-testid="align-consensus-sources" title={dome.rule}>
          <span style={{ color: "var(--c-text-dim)" }}>dome source per lateral ({voted} voted; tau = max({dome.tau_rel} × median |κ|, {dome.tau_abs_per_mm} /mm)):</span>
          {SOURCE_ORDER.filter((k) => (sources[k] ?? 0) > 0).map((k) => (
            <span key={k} title={SOURCE_TITLE[k]} style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
              <span style={{ display: "inline-block", height: 8, width: Math.max(2, Math.round(160 * (sources[k] ?? 0) / Math.max(1, voted))), background: SOURCE_COLOR[k] ?? "#999", borderRadius: 2 }} />
              <b>{k}</b> {sources[k]}
            </span>
          ))}
          {dome.undecided_fraction > 0 && <span style={{ color: "var(--c-text-dim)" }}>· undecided {Math.round(dome.undecided_fraction * 100)}%</span>}
        </div>
      )}
      {/* the votes per band */}
      {bands.length > 0 && (
        <div style={{ overflowX: "auto" }}>
          <table data-testid="align-consensus-bands" style={{ borderCollapse: "collapse", whiteSpace: "nowrap" }}>
            <thead>
              <tr style={{ color: "var(--c-text-dim)" }}>
                <th style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }}>band (ref laterals)</th>
                {ids.map((cid) => (
                  <th key={cid} style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }} title="this scan's OWN along-frame curvature κ (1/mm) in the band and the share of its laterals inside the majority">
                    <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: col(cid), marginRight: 3, verticalAlign: "middle" }} />
                    own κ {short(cid)}{voters.find((v) => v.cid === cid)?.role === "vote_only" ? " (vote only)" : cid === cons.reference ? " (anchor)" : ""}
                  </th>
                ))}
                <th style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }} title="the majority's along-frame curvature (median over the band of the smoothed c2*)">majority κ*</th>
                <th style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }} title="within-B-scan curvature: median over voters and frames of the quadratic across ±100 laterals about the band centre, each scan's own lateral spacing">axial κ</th>
                <th style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }}>ratio κ*/κ_axial</th>
                <th style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }}>verdict</th>
                <th style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }}>sources</th>
              </tr>
            </thead>
            <tbody>
              {bands.map((b) => (
                <tr key={b.band} data-testid={`align-consensus-band-${b.band}`}>
                  <td style={{ padding: "1px 10px 1px 0" }}><b>{b.band}</b>{b.apex ? " ★ apex" : ""} · {b.laterals_ref[0]}–{b.laterals_ref[1] - 1}</td>
                  {ids.map((cid) => {
                    const v = b.votes?.[cid];
                    const inMaj = v?.in_majority_frac;
                    return (
                      <td key={cid} style={{ padding: "1px 10px 1px 0" }} title={v ? `dissent ${k3(v.dissent)} /mm vs tau ${k3(b.tau)}; own axial κ ${k3(v.kappa_axial)} (${v.kappa_axial_frames} frames)` : ""}>
                        {k3(v?.kappa)}{inMaj != null ? <span style={{ color: inMaj >= 0.5 ? "var(--c-ok, #3ddc84)" : "var(--c-amber, #ffaa28)" }}> ({Math.round(inMaj * 100)}% in)</span> : ""}
                      </td>
                    );
                  })}
                  <td style={{ padding: "1px 10px 1px 0" }}><b>{k3(b.kappa_frames)}</b></td>
                  <td style={{ padding: "1px 10px 1px 0" }}>{k3(b.kappa_axial)}</td>
                  <td style={{ padding: "1px 10px 1px 0" }}>
                    <b>{fmt(b.ratio, 2)}</b>
                    {b.ratio_at_scan_size_mm ? <span style={{ color: "var(--c-text-dim)" }}> ({Object.entries(b.ratio_at_scan_size_mm).map(([k, v]) => `${k} mm: ${v.toFixed(2)}`).join(" · ")})</span> : null}
                  </td>
                  <td style={{ padding: "1px 10px 1px 0", fontWeight: 600, color: VERDICT_COLOR[b.verdict] ?? "inherit" }}>{b.verdict}</td>
                  <td style={{ padding: "1px 10px 1px 0", color: "var(--c-text-dim)" }}>{SOURCE_ORDER.filter((k) => (b.sources?.[k] ?? 0) > 0).map((k) => `${k} ${b.sources[k]}`).join(" · ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {/* per-voter notes */}
      {voters.some((v) => v.note || v.flags.length) && (
        <div className="flex flex-col" data-testid="align-consensus-voters">
          {voters.filter((v) => v.note || v.flags.length).map((v) => (
            <span key={v.cid}>
              <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: col(v.cid), marginRight: 3, verticalAlign: "middle" }} />
              <b>{short(v.cid)}</b>: {v.note ?? ""}
              {v.flags.map((f) => <span key={f} title={flagTitle(f)} style={{ marginLeft: 6, color: WARN_FLAGS.has(f) ? "var(--c-danger, #ff5252)" : "var(--c-amber, #ffaa28)", fontWeight: 600 }}>{f}</span>)}
            </span>
          ))}
        </div>
      )}
      {/* reference sensitivity */}
      <div className="flex flex-col gap-0.5" data-testid="align-consensus-sensitivity" title={sens?.note ?? ""}>
        <div className="flex items-center gap-2 flex-wrap">
          <span style={{ color: "var(--c-text-dim)" }}>reference sensitivity — the reference is only the coordinate anchor; with another scan as the anchor the consensus's <b>SHAPE</b> (beyond a whole-volume pose + a smooth dome move — the anchor dependence the final result keeps) moves by:</span>
          {sens ? (
            <>
              {Object.values(sens.anchors).map((a) => (
                <span key={a.anchor} data-testid={`align-consensus-anchor-${a.anchor}`}
                  title={a.skipped ?? `contributing: ${a.contributing.join(", ") || "none"}${a.refused?.length ? `; refused: ${a.refused.map((r) => `${short(r.cid)}${r.reason ? ` (${r.reason})` : (r.flags?.length ? ` (${r.flags.join(", ")})` : "")}`).join("; ")}` : ""}${a.vote_only?.length ? `; vote-only: ${a.vote_only.join(", ")}` : ""}; ${a.n_cells} cells${a.kappa_by_band ? `; κ per band ${a.kappa_by_band.map((k) => k3(k)).join(" / ")}` : ""}${a.pose ? `; whole-volume pose removed for the shape spread: shift ${a.pose.shift_px.toFixed(1)} px, tilt ${a.pose.tilt_px_half_span.toFixed(1)} px, trend ${a.pose.trend_px_per_frame.toFixed(2)} px/frame` : ""}${a.pair_roundtrip ? `; pair engine round trip ${short(a.anchor)} → served → ${short(a.anchor)}: df error ${a.pair_roundtrip.df_error}, a ${fmt(a.pair_roundtrip.a_rms_px, 1)} px RMS (peak ${fmt(a.pair_roundtrip.a_peak_px, 1)}), tilt ${fmt(a.pair_roundtrip.b_rms_px, 1)} px, dx ${fmt(a.pair_roundtrip.dx_rms, 1)} laterals` : ""}`}>
                  <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: col(a.anchor), marginRight: 3, verticalAlign: "middle" }} />
                  <b>{short(a.anchor)}</b>{a.served ? " (served)" : ""}{" "}
                  {a.skipped ? <span style={{ color: "var(--c-amber, #ffaa28)" }}>skipped — {a.skipped}</span>
                    : a.served ? <>0 px</>
                    : <>shape <b>{fmt(a.shape_spread_px ?? a.spread_px, 2)}</b> px RMS{a.kappa_by_band ? <span style={{ color: "var(--c-text-dim)" }}> (κ per band {a.kappa_by_band.map((k) => k3(k)).join(" / ")})</span> : null}
                        {a.refused?.length ? <span style={{ color: "var(--c-text-dim)" }}> [refused {a.refused.map((r) => short(r.cid)).join(", ")}]</span> : null}</>}
                </span>
              ))}
              {sens.kappa_band_spread && <span style={{ color: "var(--c-text-dim)" }}>· dome κ spread across anchors per band <b style={{ color: "var(--c-text)" }}>{sens.kappa_band_spread.map((k) => k == null ? "—" : k.toFixed(4)).join(" / ")}</b> /mm</span>}
            </>
          ) : (
            <span style={{ color: "var(--c-amber, #ffaa28)" }}>{cons.reference_sensitivity_error ? `failed (${cons.reference_sensitivity_error})` : "not in this result"}</span>
          )}
        </div>
        {sens && (
          <div className="flex items-center gap-2 flex-wrap" data-testid="align-consensus-sensitivity-raw" style={{ color: "var(--c-text-dim)" }}>
            <span>raw spread (the whole re-anchored consensus mapped into the served coordinates — dominated by that anchor's own smooth dome error, which the apply step's dome move removes; shown for completeness):</span>
            {Object.values(sens.anchors).filter((a) => !a.served && !a.skipped).map((a) => (
              <span key={a.anchor}>
                <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: col(a.anchor), marginRight: 3, verticalAlign: "middle" }} />
                {short(a.anchor)} {fmt(a.spread_px, 2)} px{a.dome_part_kappa != null ? ` (dome part κ ${fmt(a.dome_part_kappa, 3)} /mm ≈ its own dome error` : ""}{a.pair_roundtrip?.a_rms_px != null ? `${a.dome_part_kappa != null ? "; " : " ("}pair round-trip ${fmt(a.pair_roundtrip.a_rms_px, 1)} px` : ""}{a.dome_part_kappa != null || a.pair_roundtrip?.a_rms_px != null ? ")" : ""}
              </span>
            ))}
          </div>
        )}
      </div>
      {cons.provisional_curve && (
        <div style={{ color: "var(--c-text-dim)" }}>
          v2 − provisional curve (the old consensus that inherited the anchor's dome): peak {fmt(cons.provisional_curve.v2_minus_provisional_px.peak, 1)} / RMS {fmt(cons.provisional_curve.v2_minus_provisional_px.rms, 2)} px
        </div>
      )}
    </div>
  );
}

/** The alignment content for ONE group id (header: re-run / refresh / summary / tabs; body: Consensus + pairs, 3-D view,
 *  Scrub). Used by the "⧉ Alignment (pairs)…" dialog and the step-4 "Aligned" pane. Loads the result on mount, polls
 *  while a job runs; autoStart (re-)runs the job on mount. Must sit in a definite-height flex column (it fills it). */
export function AlignTabs({ gid, autoStart, title, onReAlign }:
  { gid: string | null; autoStart?: boolean; title?: React.ReactNode; onReAlign?: () => void }) {
  const [status, setStatus] = useState<AlignStatus | null>(null);
  const [result, setResult] = useState<AlignResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [zoom, setZoom] = useState<{ cid: string; url: string } | null>(null);
  const [tab, setTab] = useState<"cards" | "3d" | "scrub">("cards");
  const [stage3d, setStage3d] = useState<"applied" | "pairs">("applied");   // 3-D tab: with the axial changes applied / pairs only
  const pollRef = useRef<number | null>(null);
  const seen = useRef<string | null>(null);

  const path = useCallback((tail: string) => `/api/group/${encodeURIComponent(gid ?? "")}/align${tail}`, [gid]);

  const loadResult = useCallback(async () => {
    if (!gid) return;
    try {
      const r = await api.json<AlignResult>(path("/result"));
      setResult(r);
    } catch (e) {
      // no result yet is not an error worth showing
      const msg = e instanceof Error ? e.message : String(e);
      if (!/No alignment result/.test(msg)) setErr(msg);
      setResult(null);
    }
  }, [gid, path]);

  const refresh = useCallback(async () => {
    if (!gid) return;
    try {
      const s = await api.json<AlignStatus>(path("/status"));
      setStatus(s);
      setErr(s.error ?? null);
      const stamp = s.result_timestamp ?? null;
      if (s.result_exists && !s.running && stamp !== seen.current) { seen.current = stamp; await loadResult(); }
      if (!s.result_exists) setResult(null);
      return s;
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      return null;
    }
  }, [gid, path, loadResult]);

  const stopPoll = useCallback(() => { if (pollRef.current) { window.clearInterval(pollRef.current); pollRef.current = null; } }, []);
  const startPoll = useCallback(() => {
    stopPoll();
    pollRef.current = window.setInterval(() => {
      void refresh().then((s) => { if (s && !s.running) stopPoll(); });
    }, 3000);
  }, [refresh, stopPoll]);

  const start = useCallback(async (force: boolean) => {
    if (!gid) return;
    setStarting(true); setErr(null);
    try {
      const s = await api.json<AlignStatus>(path(""), "POST", JSON.stringify({ force }));
      setStatus(s);
      if (s.running || s.started) startPoll();
      else if (s.result_exists) await loadResult();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }, [gid, path, startPoll, loadResult]);

  // Load the result on open (if it exists); poll while a job runs; optionally start the job on open.
  useEffect(() => {
    if (!gid) { stopPoll(); return; }
    seen.current = null; setResult(null); setStatus(null); setErr(null); setTab("cards");
    void (async () => {
      const s = await refresh();
      if (autoStart && s && !s.running) await start(true);
      else if (s?.running) startPoll();
    })();
    return () => stopPoll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gid]);

  const running = !!status?.running;
  const prog = status?.progress ?? null;
  const elapsed = status?.seconds != null ? `${Math.round(status.seconds)} s` : "";
  const members = result?.members ?? [];
  const stamp = result?.timestamp ?? "";
  const cons = result?.consensus ?? null;
  const consUrl = cons && result?.consensus_url ? resourceUrl(`${result.consensus_url}?t=${encodeURIComponent(stamp)}`) : null;
  const tr = result?.transforms ?? null;
  const alignedUrl = tr && result?.aligned_url ? resourceUrl(`${result.aligned_url}?t=${encodeURIComponent(stamp)}`) : null;
  const volFinalUrl = cons && result?.volume_url ? resourceUrl(result.volume_url) : null;
  const volPairsUrl = cons ? resourceUrl(result?.volume_pairs_url ?? `${result?.volume_url ?? ""}?stage=pairs`) : null;
  // the applied volume exists only when the transforms stage ran; an older result has just the pair placement
  const stageShown: "applied" | "pairs" = tr && stage3d === "applied" ? "applied" : "pairs";
  const volUrl = stageShown === "applied" ? volFinalUrl : (tr ? volPairsUrl : volFinalUrl);
  const trBy = (cid: string) => tr?.members.find((m) => m.cid === cid) ?? null;
  const legend = cons ? cons.members.map((cid) => ({
    cid, channel: cons.channels?.[cid] ?? "?", colour: cons.colours?.[cid] ?? null, is_reference: cid === cons.reference,
    rms: stageShown === "applied" ? (trBy(cid)?.rms_after_px ?? null) : (cons.rms_to_consensus_px[cid] ?? null),
  })) : [];
  const show3d = !!gid && tab === "3d" && !!volUrl && !running;
  const roster = rosterOf(result);
  const scrubMetaUrl = tr?.scrub && result?.scrub_meta_url ? result.scrub_meta_url : null;   // the scrub data exists only for results of the new job
  // the Applied card's tissue summary line (reviewer ask 2026-09-11) comes from scrub/meta.json (lazily completed by the sidecar for older results)
  const [scrubMeta, setScrubMeta] = useState<ScrubMeta | null>(null);
  const resultStamp = result?.timestamp ?? "";
  useEffect(() => {
    let cancelled = false;
    setScrubMeta(null);
    if (!scrubMetaUrl || running) return;
    api.json<ScrubMeta>(scrubMetaUrl).then((m) => { if (!cancelled) setScrubMeta(m); }).catch(() => { /* the Scrub tab reports its own error */ });
    return () => { cancelled = true; };
  }, [scrubMetaUrl, resultStamp, running]);

  return (
    <div className="flex flex-col" style={{ flex: 1, minHeight: 0, minWidth: 0 }} data-testid="align-tabs" data-gid={gid ?? ""}>
      <div style={{ fontSize: 15, padding: "4px 8px", flex: "none" }}>
        <div className="flex items-center gap-3 flex-wrap">
          {title ?? <span>group <b>{gid ?? "?"}</b></span>}
          <Button size="small" variant="contained" color="primary" disabled={!gid || running || starting}
            startIcon={(running || starting) ? <CircularProgress size={13} color="inherit" /> : undefined}
            onClick={onReAlign ? () => onReAlign() : () => void start(true)}
            title={onReAlign
              ? "Align this group again: asks FIRST which scans belong to which subgroup (reviewer 2026-09-12), then runs the pair registration engine on each subgroup from scratch (a few minutes per pair)."
              : "Run the pair registration engine (group_align.register_group) on every scan of this eye against the reference; re-runs from scratch (a few minutes per pair)."}>
            {running ? "Aligning group…" : "⧉ Align group"}
          </Button>
          <Button size="small" variant="outlined" disabled={!gid} onClick={() => void refresh()}>↻ Refresh</Button>
          {result && (
            <span className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
              reference <b>{result.reference}</b> · {members.length} scans · engine {result.engine_md5?.slice(0, 8)} · {stamp} · {Math.round(result.seconds)} s
              {status?.engine_stale && <span style={{ color: "var(--c-amber, #ffaa28)" }}> · ⚠ engine changed since this result</span>}
            </span>
          )}
          {result && (
            <Tabs value={tab} onChange={(_, v) => setTab(v as "cards" | "3d" | "scrub")} sx={{ minHeight: 30, ml: "auto" }} data-testid="align-tabs">
              <Tab value="cards" label="Consensus + pairs" sx={{ minHeight: 30, py: 0, fontSize: 12 }} data-testid="align-tab-cards" />
              <Tab value="3d" label="3-D view" disabled={!volUrl} sx={{ minHeight: 30, py: 0, fontSize: 12 }} data-testid="align-tab-3d"
                title={volUrl ? "The placed members as one RGB volume (niivue render mode)" : "No aligned volume in this result — re-run the alignment"} />
              <Tab value="scrub" label="Scrub" sx={{ minHeight: 30, py: 0, fontSize: 12 }} data-testid="align-tab-scrub"
                title={scrubMetaUrl ? "Scrub through the sagittal views before / after the axial changes, lateral by lateral" : "No applied transforms (scrub data) in this result — re-run the alignment"} />
            </Tabs>
          )}
        </div>
      </div>
      <div className="flex flex-col" style={{ fontSize: 13, padding: 8, flex: 1, minHeight: 0, overflow: "auto", borderTop: "1px solid var(--c-border)" }} data-testid="align-tabs-body">
        {running && (
          <div className="flex items-center gap-2 py-1 text-xs" data-testid="align-progress">
            <CircularProgress size={14} />
            <span>
              {prog?.phase ?? "running"} · members {prog?.members_done ?? 0}/{prog?.members_total ?? status?.members.length ?? 0}
              {" · "}pairs {prog?.pairs_done ?? 0}/{prog?.pairs_total ?? Math.max(0, (status?.members.length ?? 1) - 1)}
              {prog?.overlays_done ? ` · overlays ${prog.overlays_done}` : ""}{elapsed ? ` · ${elapsed}` : ""}
            </span>
          </div>
        )}
        {err && <div className="text-xs py-1" style={{ color: "var(--c-danger, #ff5252)" }}>⚠ {err}</div>}
        {status?.running_elsewhere && !running && (
          <div className="text-xs py-1" style={{ color: "var(--c-amber, #ffaa28)" }}>Another group's alignment is running ({status.running_elsewhere}); wait for it before starting this one.</div>
        )}
        {!running && !result && !err && (
          <div className="text-xs py-2" style={{ color: "var(--c-text-dim)" }}>
            No alignment yet for this group{status ? ` (${status.members.length} scans: ${status.members.join(", ")})` : ""}. Click "⧉ Align group (re-run)".
          </div>
        )}
        {show3d && (
          <div style={{ flex: 1, minHeight: 420, display: "flex", flexDirection: "column", gap: 4 }}>
            <div className="flex items-center gap-2 flex-wrap text-xs" data-testid="align-3d-stage">
              <span style={{ color: "var(--c-text-dim)" }}>volume:</span>
              <ToggleButtonGroup size="small" exclusive value={stageShown} onChange={(_, v) => { if (v) setStage3d(v as "applied" | "pairs"); }}>
                <ToggleButton value="pairs" sx={{ py: 0, fontSize: 11, textTransform: "none" }} data-testid="align-3d-stage-pairs"
                  title="The pair engine's placement only (df, dx, a, b·x per frame) — nothing applied">pairs only</ToggleButton>
                <ToggleButton value="applied" disabled={!tr} sx={{ py: 0, fontSize: 11, textTransform: "none" }} data-testid="align-3d-stage-applied"
                  title={tr ? "Every member moved by its FINAL rigid transform (pair transform + per-frame δa shift + δb·x tilt to the consensus)" : "No applied result in this run — re-run the alignment"}>
                  with axial changes applied</ToggleButton>
              </ToggleButtonGroup>
              <span style={{ color: "var(--c-text-dim)" }}>
                {stageShown === "applied" ? "legend RMS = residual to the consensus AFTER the move" : "legend RMS = residual to the consensus of the pair placement"}
              </span>
            </div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <Align3D key={stageShown} url={volUrl!} legend={legend} stamp={stamp} roster={roster} />
            </div>
          </div>
        )}
        {result && tab === "scrub" && !running && (
          scrubMetaUrl ? (
            <AlignScrub metaUrl={scrubMetaUrl} stamp={stamp} onZoom={(u, label) => setZoom({ cid: label || "sagittal scrub", url: u })} />
          ) : (
            <div className="text-xs py-2" style={{ color: "var(--c-amber, #ffaa28)" }} data-testid="align-scrub-none">
              This group has no applied transforms yet{result.transforms_error ? ` (${result.transforms_error})` : ""} — the scrub view needs the before / after
              placements the job writes when the axial changes are applied. Click "⧉ Align group (re-run)".
            </div>
          )
        )}
        {result && tab === "cards" && (
          <div className="flex flex-col gap-2">
            {/* FIRST card: the PROVISIONAL consensus (reviewer ask #2) */}
            <div className="flex flex-col gap-1 rounded border p-2" data-testid="align-card-consensus"
              style={{ borderColor: "var(--c-accent, #4ea1ff)", backgroundColor: "var(--c-surface)" }}>
              <ContributionLine roster={roster} colours={cons?.colours} testid="align-consensus-contribution" />
              <div className="flex items-center gap-2 flex-wrap text-xs">
                <b style={{ fontSize: 13 }} data-testid="align-consensus-title">{cons?.version === 2 ? `Consensus v2 (majority own dome, axial-checked)${cons.revision ? ` · rev ${cons.revision}` : ""}` : "Consensus (provisional)"}</b>
                {cons ? (
                  <span style={{ color: "var(--c-text-dim)" }}>
                    fused union canvas of {cons.n_members} placed scans (reference <b>{cons.reference}</b>) · canvas {cons.canvas.shape.join("×")} (origin {cons.canvas.origin.join(", ")})
                    {" · "}covered frames {cons.covered_frame_range[0]}–{cons.covered_frame_range[1]} ({cons.covered_frames})
                    {cons.curve_model?.fitted_laterals != null ? ` · curve fitted on ${cons.curve_model.fitted_laterals} laterals` : ""}
                    {" · "}RMS to consensus:{" "}
                    {cons.members.map((cid) => (
                      <span key={cid} style={{ marginRight: 8 }}>
                        <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: cons.colours[cid], marginRight: 3, verticalAlign: "middle" }} />
                        <b>{cid}</b> {fmt(cons.rms_to_consensus_px[cid], 1)} px
                      </span>
                    ))}
                  </span>
                ) : (
                  <span style={{ color: "var(--c-amber, #ffaa28)" }}>
                    no consensus in this result{result.consensus_error ? ` (${result.consensus_error})` : " — re-run the alignment (older result)"}
                  </span>
                )}
              </div>
              {cons && <div className="text-[11px]" style={{ color: cons.version === 2 ? "var(--c-text-dim)" : "var(--c-amber, #ffaa28)" }}>{cons.note}</div>}
              {cons?.version === 2 && <ConsensusV2Details cons={cons} />}
              {cons && (
                <div className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                  top row: three B-scans (start / middle / end of the covered frames); bottom row: three sagittal cuts (left / centre / right);
                  dashed = each scan's served line placed by the engine's rigid transform (df, dx, a + b·x); thick white = the consensus curve.
                  {volUrl ? " The same placement is the \"3-D view\" tab." : ""}
                </div>
              )}
              {consUrl && (
                <a href={consUrl} target="_blank" rel="noreferrer" title="Open the full-size consensus image in a new tab"
                  onClick={(e) => { e.preventDefault(); setZoom({ cid: cons?.version === 2 ? "consensus v2" : "consensus (provisional)", url: consUrl }); }}>
                  <img src={consUrl} alt="consensus" data-testid="align-consensus-img"
                    style={{ width: "100%", maxWidth: 1600, display: "block", cursor: "zoom-in" }} />
                </a>
              )}
            </div>
            {/* SECOND card: the APPLIED axial changes (reviewer ask #3) */}
            <div className="flex flex-col gap-1 rounded border p-2" data-testid="align-card-applied"
              style={{ borderColor: "var(--c-ok, #3ddc84)", backgroundColor: "var(--c-surface)" }}>
              <ContributionLine roster={roster} colours={cons?.colours} testid="align-applied-contribution" />
              <div className="flex items-center gap-2 flex-wrap text-xs">
                <b style={{ fontSize: 13 }}>Applied axial changes ({cons?.version === 2 ? "consensus v2" : "provisional consensus"})</b>
                {tr ? (
                  <span style={{ color: "var(--c-text-dim)" }}>
                    every scan (reference <b>{tr.reference}</b> included) moved by its FINAL rigid transform = the pair engine's df / dx / a / b·x
                    {" "}+ a <b>smooth</b> per-frame shift <b>δa</b> and tilt <b>δb·x</b> = the shift + tilt of (consensus − the scan's <b>own smooth dome</b>), fitted 3×MAD robust
                    over the scan's laterals — a smooth-to-smooth dome correction, never a fit to the line (line error must never move tissue) · canvas {tr.canvas.shape.join("×")}
                    {" · "}frames the consensus does not cover keep the pair transform (held) · a frame whose dome difference beyond the tilt exceeds {tr.profile_beyond_tilt_px} px RMS is flagged <i>profile_beyond_tilt</i>
                    {tr.line_off_consensus_px != null ? <>{" · "}a frame whose <b>line</b> (after the move) is off the consensus by more than {tr.line_off_consensus_px} px RMS is flagged <i>line_off_consensus</i> (line quality — reported, never applied; candidate GT for the edge tools)</> : null}
                    {" · "}<span title={tr.after_rms_bar_of ?? ""}>dome-residual bar {tr.after_rms_bar_px} px{(tr.apply_revision ?? 0) >= 3 ? <> on the <b>dome part</b> (the frame-independent lateral-profile part is reported separately — a rigid move cannot remove it)</> : " (older result: on the whole residual beyond the shift + tilt — re-run for revision 3)"}: <b style={{ color: tr.after_rms_ok_all ? "var(--c-ok, #3ddc84)" : "var(--c-amber, #ffaa28)" }}>{tr.after_rms_ok_all ? "met by every scan" : "not met by every scan"}</b></span>
                    {tr.smooth_bar_d2_a_px != null ? <>{" · "}<span title={tr.smooth_bar_of ?? ""} data-testid="align-applied-smooth">smooth along frames (max |d²| of the applied δa ≤ {tr.smooth_bar_d2_a_px} px, δb ≤ {tr.smooth_bar_d2_b_px} px; fitted on one FIXED inlier set of laterals, Savitzky-Golay smoothed): <b style={{ color: tr.smooth_ok_all ? "var(--c-ok, #3ddc84)" : "var(--c-amber, #ffaa28)" }}>{tr.smooth_ok_all ? "met by every scan" : "NOT met by every scan"}</b></span></> : null}
                  </span>
                ) : (
                  <span style={{ color: "var(--c-amber, #ffaa28)" }}>
                    nothing applied in this result{result.transforms_error ? ` (${result.transforms_error})` : (cons ? " — re-run the alignment (older result)" : " — no consensus to apply")}
                  </span>
                )}
              </div>
              {tr && <div className="text-[11px]" style={{ color: "var(--c-amber, #ffaa28)" }}>{tr.note}</div>}
              {tr && scrubMetaUrl && (
                <div className="text-xs" data-testid="align-applied-tissue" title={scrubMeta?.tissue_summary?.rule ?? ""}>
                  {hasTissue(scrubMeta) ? (
                    <>
                      <b>tissue</b> (measured from the images, not the lines): scan-to-scan disagreement mean{" "}
                      <b>{fmt(scrubMeta!.tissue_summary!.disagreement_mean.before, 2)} → {fmt(scrubMeta!.tissue_summary!.disagreement_mean.after, 2)} px</b>
                      {" · "}per-scan tissue RMS to consensus before → after:{" "}
                      {scrubMeta!.members.map((cid) => (
                        <span key={cid} style={{ marginRight: 8 }}>
                          <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: scrubMeta!.colours[cid] ?? "#aaa", marginRight: 3, verticalAlign: "middle" }} />
                          {cid} {fmt(scrubMeta!.tissue_summary!.rms[cid]?.before, 2)} → <b>{fmt(scrubMeta!.tissue_summary!.rms[cid]?.after, 2)}</b> px
                        </span>
                      ))}
                    </>
                  ) : (
                    <span style={{ color: "var(--c-text-dim)" }}>tissue RMS (measured from the images): {scrubMeta ? (scrubMeta.tissue_error ? `unavailable (${scrubMeta.tissue_error})` : "not in this result") : "loading…"}</span>
                  )}
                </div>
              )}
              {tr && (tr.tissue_gate || tr.held_scans) && (
                <div className="text-xs" data-testid="align-applied-tissue-gate" title={tr.tissue_gate?.rule ?? ""}
                  style={{ padding: "3px 6px", borderRadius: 4, background: (tr.tissue_gate?.held?.length ? "rgba(255,170,40,0.10)" : "rgba(61,220,132,0.08)"),
                           border: `1px solid ${tr.tissue_gate?.held?.length ? "var(--c-amber, #ffaa28)" : "var(--c-ok, #3ddc84)"}` }}>
                  <b>tissue gate</b> (each scan's mean tissue disagreement with the OTHER placed scans in their overlaps, measured from the images before the final files are written):
                  {tr.tissue_gate && !tr.tissue_gate.error ? (
                    <>
                      {" "}group mean <b>{fmt(tr.tissue_gate.group_before_px, 2)} → {fmt(tr.tissue_gate.group_after_px, 2)} px</b>
                      {tr.tissue_gate.group_after_initial_px != null && tr.tissue_gate.held?.length ? <span style={{ color: "var(--c-text-dim)" }}> (would have been {fmt(tr.tissue_gate.group_after_initial_px, 2)} px before the holds)</span> : null}
                      {" · "}<b style={{ color: tr.tissue_gate.ok ? "var(--c-ok, #3ddc84)" : "var(--c-danger, #ff5252)" }}>{tr.tissue_gate.ok ? `within +${tr.tissue_gate.threshold_group_px ?? 0.1} px of before` : `NOT within +${tr.tissue_gate.threshold_group_px ?? 0.1} px of before`}</b>
                      {" · "}per scan before → after:{" "}
                      {Object.entries(tr.tissue_gate.per_scan).map(([cid, v]) => (
                        <span key={cid} style={{ marginRight: 8 }} data-testid={`align-tissue-gate-${cid}`}>
                          <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: cons?.colours[cid] ?? "#aaa", marginRight: 3, verticalAlign: "middle" }} />
                          {short(cid)} {fmt(v.before_px, 2)} → <b>{fmt(v.after_px, 2)}</b>{v.held ? <b style={{ color: "var(--c-amber, #ffaa28)" }}> HELD</b> : ""}
                        </span>
                      ))}
                      {tr.tissue_gate.held?.length ? (
                        <span style={{ color: "var(--c-amber, #ffaa28)" }}>
                          {" · "}<b>dome move held for {tr.tissue_gate.held.map(short).join(", ")}</b>: {tr.tissue_gate.held.map((cid) => {
                            const rs = tr.tissue_gate!.per_scan[cid]?.reason;
                            return rs ? `${short(cid)} would have risen ${rs.before_px.toFixed(2)} → ${rs.after_px.toFixed(2)} px (${rs.rule === "group" ? `held for the group bar +${tr.tissue_gate!.threshold_group_px ?? 0.1} px` : `> +${rs.threshold_px} px`})` : short(cid);
                          }).join("; ")} — the pair transform is served as final (δa = δb = 0); line error never moves tissue
                        </span>
                      ) : <span style={{ color: "var(--c-text-dim)" }}> · no scan held (every dome move left its tissue agreement within +{tr.tissue_gate.threshold_scan_px ?? 0.3} px)</span>}
                    </>
                  ) : <span style={{ color: "var(--c-amber, #ffaa28)" }}> not measured{tr.tissue_gate?.error ? ` (${tr.tissue_gate.error})` : ""}</span>}
                </div>
              )}
              {tr && (
                <div style={{ overflowX: "auto" }}>
                  <table className="text-[11px]" data-testid="align-applied-table" style={{ borderCollapse: "collapse", whiteSpace: "nowrap" }}>
                    <thead>
                      <tr style={{ color: "var(--c-text-dim)" }}>
                        {["scan", "applied δa (smooth dome correction) peak / RMS · max |d²|", "applied δb peak / RMS · max |d²|", "tissue-motion part (pair a) peak / RMS",
                          "dome residual before → dome part after (bar) · lateral-profile part", "line residual to consensus before → after (RMS, reported)",
                          "frames off consensus (line)", "profile_beyond_tilt", "frames covered / held"].map((h) => (
                          <th key={h} style={{ textAlign: "left", padding: "1px 10px 1px 0", fontWeight: 500 }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {tr.members.map((t) => (
                        <tr key={t.cid} data-testid={`align-applied-${t.cid}`}>
                          <td style={{ padding: "1px 10px 1px 0" }}>
                            <span style={{ display: "inline-block", width: 9, height: 9, borderRadius: 2, background: cons?.colours[t.cid] ?? "#aaa", marginRight: 3, verticalAlign: "middle" }} />
                            <b>{t.cid}</b>{t.is_reference ? " (reference)" : ""}
                          </td>
                          <td style={{ padding: "1px 10px 1px 0" }} data-testid={`align-applied-da-${t.cid}`}>
                            {t.dome_move_held ? <b style={{ color: "var(--c-amber, #ffaa28)" }} title={t.hold_note ?? "dome move HELD by the tissue gate"}>HELD (tissue gate) · </b> : null}
                            <b>{fmt(t.delta_a.peak, 1)}</b> / {fmt(t.delta_a.rms, 2)} px
                            {t.delta_a_d2_max != null ? <> · max |d²| <b style={{ color: (t.smooth_bar_d2_a_px != null && t.delta_a_d2_max <= t.smooth_bar_d2_a_px) ? "var(--c-ok, #3ddc84)" : "var(--c-amber, #ffaa28)" }}>{fmt(t.delta_a_d2_max, 3)}</b>{t.smooth_bar_d2_a_px != null ? ` (bar ${t.smooth_bar_d2_a_px})` : ""}</> : <> (frame-to-frame jitter RMS {fmt(t.delta_a_jitter.rms, 2)})</>}
                            {t.fixed_inliers != null ? <span style={{ color: "var(--c-text-dim)" }}> · {t.fixed_inliers} fixed laterals</span> : null}</td>
                          <td style={{ padding: "1px 10px 1px 0" }} data-testid={`align-applied-db-${t.cid}`}><b>{fmt(t.delta_b.peak, 1)}</b> / {fmt(t.delta_b.rms, 2)} px half-span
                            {t.delta_b_d2_max != null ? <> · max |d²| <b style={{ color: (t.smooth_bar_d2_b_px != null && t.delta_b_d2_max <= t.smooth_bar_d2_b_px) ? "var(--c-ok, #3ddc84)" : "var(--c-amber, #ffaa28)" }}>{fmt(t.delta_b_d2_max, 3)}</b>{t.smooth_bar_d2_b_px != null ? ` (bar ${t.smooth_bar_d2_b_px})` : ""}</> : null}</td>
                          <td style={{ padding: "1px 10px 1px 0" }}>{t.is_reference ? "0 (reference)" : <>{fmt(t.tissue_a.peak, 1)} / {fmt(t.tissue_a.rms, 2)} px</>}</td>
                          <td style={{ padding: "1px 10px 1px 0" }}>
                            {t.dome_rms_after_px !== undefined ? (
                              <>{fmt(t.dome_rms_before_px ?? null, 2)} → <b style={{ color: t.after_rms_ok ? "var(--c-ok, #3ddc84)" : "var(--c-amber, #ffaa28)" }}>{fmt(t.dome_rms_after_px, 2)}</b> px
                                {t.after_rms_ok === false ? ` (> ${t.after_rms_bar_px} bar)` : ""}
                                {t.profile_rms_px !== undefined ? <span style={{ color: "var(--c-text-dim)" }}> · profile part {fmt(t.profile_rms_px, 2)} px (whole {fmt(t.residual_rms_after_px ?? null, 2)})</span> : null}</>
                            ) : <span style={{ color: "var(--c-text-dim)" }}>— (older result: re-run)</span>}
                          </td>
                          <td style={{ padding: "1px 10px 1px 0" }} data-testid={`align-line-residual-${t.cid}`}>
                            {fmt(t.rms_before_px, 2)} → <b>{fmt(t.line_residual_rms_px ?? t.rms_after_px, 2)}</b> px
                          </td>
                          <td style={{ padding: "1px 10px 1px 0", color: t.line_off_consensus_frames ? "var(--c-amber, #ffaa28)" : undefined }}>
                            {t.line_off_consensus_frames !== undefined ? `${t.line_off_consensus_frames} frames` : "—"}
                          </td>
                          <td style={{ padding: "1px 10px 1px 0", color: t.profile_beyond_tilt_frames ? "var(--c-amber, #ffaa28)" : undefined }}>{t.profile_beyond_tilt_frames} frames</td>
                          <td style={{ padding: "1px 10px 1px 0" }}>{t.covered_frames} / {t.held_frames} of {t.frames}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {tr && (
                <div className="text-[11px]" style={{ color: "var(--c-text-dim)" }}>
                  same layout as the consensus montage; fused = the moved scans; dashed = each scan's FINAL line; thick white = the consensus.
                  The move is the smooth dome correction only, so after it each scan's DOME sits on the consensus (dome residual); the line's own
                  residual (its jitter, dips into the tissue) is reported per scan — "frames off consensus" are candidate edge corrections, and the
                  Scrub tab's titles show that line RMS per lateral.
                  {volFinalUrl ? " The \"3-D view\" tab shows this applied result (toggle to \"pairs only\" to compare)." : ""}
                </div>
              )}
              {alignedUrl && (
                <a href={alignedUrl} target="_blank" rel="noreferrer" title="Open the full-size applied montage in a new tab"
                  onClick={(e) => { e.preventDefault(); setZoom({ cid: "applied axial changes", url: alignedUrl }); }}>
                  <img src={alignedUrl} alt="applied axial changes" data-testid="align-applied-img"
                    style={{ width: "100%", maxWidth: 1600, display: "block", cursor: "zoom-in" }} />
                </a>
              )}
            </div>
            {members.map((m) => {
              const url = m.overlay ? resourceUrl(`${result.overlay_url_base}${encodeURIComponent(m.cid)}?t=${encodeURIComponent(stamp)}`) : null;
              const info = m.non_contributing ? [m.non_contributing] : [];
              const viaEntry = roster.find((r) => r.cid === m.cid);
              const via = m.via ?? viaEntry?.via ?? null;
              const verdict = m.is_reference ? "reference" : (m.ok ? "ok" : via ? `REFUSED against the reference — placed via ${short(via)}` : "REFUSED");
              const color = m.is_reference ? "var(--c-accent, #4ea1ff)" : (m.ok ? "var(--c-ok, #3ddc84)" : via ? "var(--c-amber, #ffaa28)" : "var(--c-danger, #ff5252)");
              return (
                <div key={m.cid} className="flex flex-col gap-1 rounded border p-2" data-testid={`align-card-${m.cid}`}
                  style={{ borderColor: "var(--c-border)", backgroundColor: "var(--c-surface)" }}>
                  <div className="flex items-center gap-2 flex-wrap text-xs">
                    <b style={{ fontSize: 13 }}>{m.cid}</b>
                    <span style={{ color, fontWeight: 600 }}>{verdict}</span>
                    {m.is_reference ? (
                      <span style={{ color: "var(--c-text-dim)" }}>
                        the other scans are moved onto this scan's grid · {m.shape ? `${m.shape[0]}×${m.shape[1]}×${m.shape[2]}` : ""} · valid {m.valid_area ?? "—"} cells
                      </span>
                    ) : (
                      <span style={{ color: "var(--c-text-dim)" }}>
                        vs <b>{m.reference}</b> · df <b>{fmt(m.df, 0, true)}</b> frames · dx <b>{fmt(m.dx_median, 1, true)}</b> laterals [{fmt(m.dx_range?.[0], 0, true)}, {fmt(m.dx_range?.[1], 0, true)}]
                        {" · "}tilt <b>{fmt(m.tilt_median_px, 1, true)}</b> px half-span (|b| {fmt(m.tilt_abs_median_px, 1)}) · scale <b>{fmt(m.lateral_scale, 4)}</b>
                        {" · "}structure match <b>{pct(m.rel_struct)}</b> of ceiling (matched {pct(m.matched_struct)}) · speckle match <b>{pct(m.rel_speckle)}</b>
                        {" · "}coverage <b>{pct(m.coverage)}</b> · measured {m.measured_frames ?? "—"}/{m.overlap_frames ?? "—"} frames
                        {m.overlap_fraction != null && Number.isFinite(m.overlap_fraction) ? <>{" · "}<span title="the laterals this scan shares with the reference under the served shift (PARTIAL OVERLAP: a pair is registered on the overlap and reported as a fraction of the scan's laterals)" data-testid={`align-overlap-${m.cid}`}>overlap <b>{Math.round(m.overlap_laterals ?? 0)} laterals ({pct(m.overlap_fraction)})</b></span></> : null}
                        {m.pose_angle_deg != null && Number.isFinite(m.pose_angle_deg) ? ` · pose ${fmt(m.pose_angle_deg, 1)}°` : ""}
                        {m.seconds != null ? ` · ${Math.round(m.seconds)} s` : ""}
                      </span>
                    )}
                  </div>
                  {via && (
                    <div className="text-[11px]" style={{ color: "var(--c-amber, #ffaa28)" }} data-testid={`align-route-${m.cid}`}>
                      ⤳ {routeTitle({ cid: m.cid, via, route: m.route ?? viaEntry?.route ?? null })}
                    </div>
                  )}
                  {!m.is_reference && (
                    <div className="text-[11px] flex flex-wrap gap-1" style={{ color: "var(--c-text-dim)" }}>
                      {m.reject_flags.map((f) => <span key={f} style={{ color: "var(--c-danger, #ff5252)", fontWeight: 600 }}>{f}</span>)}
                      {info.map((f) => <span key={f} style={{ color: "var(--c-amber, #ffaa28)" }}>{f}</span>)}
                      {m.flags.filter((f) => !m.reject_flags.includes(f)).map((f) => <span key={f}>{f}</span>)}
                      {m.flags.length === 0 && <span>no flags</span>}
                    </div>
                  )}
                  {url && (
                    <a href={url} target="_blank" rel="noreferrer" title="Open the full-size overlay in a new tab"
                      onClick={(e) => { e.preventDefault(); setZoom({ cid: m.cid, url }); }}>
                      <img src={url} alt={`overlay ${m.cid}`} style={{ width: "100%", maxWidth: 1600, display: "block", cursor: "zoom-in" }} />
                    </a>
                  )}
                  {!m.is_reference && !url && <div className="text-xs" style={{ color: "var(--c-amber, #ffaa28)" }}>no overlay{m.overlay_error ? ` (${m.overlay_error})` : ""}</div>}
                </div>
              );
            })}
          </div>
        )}
      </div>
      {/* full-size view: the PNG at its native size, scrollable */}
      <Dialog open={!!zoom} onClose={() => setZoom(null)} maxWidth={false} PaperProps={{ sx: { width: "98vw", maxWidth: "98vw", height: "96vh" } }}>
        <DialogTitle sx={{ fontSize: 13, py: 0.5 }}>
          {zoom?.cid} — {zoom?.cid.startsWith("consensus") ? "fused canvas (dashed = placed served lines, white = consensus)"
            : zoom?.cid.startsWith("applied") ? "fused canvas after the move (dashed = FINAL lines, white = consensus)"
            : zoom?.cid.startsWith("sagittal") ? "top = BEFORE (pair placement), bottom = AFTER (axial changes applied); thin colour = the scan's line, white = consensus"
            : "overlay (red = reference, green = moved)"}
          {zoom && <a href={zoom.url} target="_blank" rel="noreferrer" style={{ marginLeft: 12, color: "var(--c-accent)", fontSize: 12 }}>open in a tab</a>}
        </DialogTitle>
        <DialogContent sx={{ p: 0.5, overflow: "auto" }}>
          {zoom && <img src={zoom.url} alt={`overlay ${zoom.cid}`} style={{ display: "block", maxWidth: "none" }} />}
        </DialogContent>
        <DialogActions sx={{ py: 0.5 }}><Button size="small" onClick={() => setZoom(null)}>Close</Button></DialogActions>
      </Dialog>
    </div>
  );
}
