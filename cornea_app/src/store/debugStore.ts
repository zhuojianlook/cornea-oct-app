/* Debug tab — replicate-alignment comparison.

   Picks two repeat scans of ONE eye, runs several alignment methods on the pair, and shows each
   method's result as a magenta/green composite so the alignment can be judged by eye rather than
   by score table. Read-only: nothing here writes to a case.

   Job lifecycle (the API contract): POST /api/debug/align/compare → { job_id }, then poll
   GET /api/debug/align/job/{job_id} until status leaves "running". Partial results are rendered
   as they arrive so the first methods are visible while the slow ones finish. */

import { create } from "zustand";
import { immer } from "zustand/middleware/immer";
import { api, resourceUrl } from "../api/client";

export type MethodId = "identity" | "asis" | "fixed" | "bruteforce";
export type ViewId = "bscan" | "sagittal" | "zoom";
/** The two 3-D turntable modes. Keys match the backend `turntable` dict verbatim (no mapping layer). */
export type Mode3d = "overlap" | "disagreement";
export type Layout = "grid" | "flip";

/** The methods offered, in a FIXED display order — identity first, because it is the reference every
    other method is judged against. Order is never sorted by score: the top methods differ by less than
    the metric's noise floor, so ranking them in the UI would be a lie dressed as a result.

    `blurb` is a STATIC per-method explainer written here. It is NOT the backend's job-level `note`
    (which reports real per-pair findings, e.g. a geometry mismatch) — the two were previously both
    called "note", and the backend's was silently dropped as a result. Keep them distinct. */
export const ALIGN_METHODS: {
  id: MethodId;
  label: string;
  blurb: string;
  locked?: boolean;
  /** Set when the method's numbers are NOT reproducible run-to-run; rendered as an honest badge. */
  wobble?: string;
}[] = [
  {
    id: "identity",
    label: "Identity (no alignment)",
    blurb: "The two scans exactly as acquired. The reference every other method is measured against — always shown.",
    locked: true, // always run: without it there is nothing to compare against
  },
  {
    id: "asis",
    label: "Production as-is",
    blurb:
      "registration._rigid_intensity exactly as the app ships it TODAY. When this raises or misses, production " +
      "falls back to identity by design (align_transform keeps whichever wins a cornea-Dice check) — that fallback " +
      "is the safety net working, not a crash.",
    // Measured, not theoretical: three identical UI runs gave primary 0.7671 / 0.7817 / 0.7860 and rot
    // 4.42–4.94°. SetMetricSamplingPercentage(0.05, seed=1) pins the sample SET, not the result — ITK's
    // multithreaded Mattes reduction is run-order dependent. Surfaced, deliberately NOT "fixed": changing
    // it would change shipped behaviour.
    wobble:
      "This row is NOT deterministic: run-to-run variation ~±0.01 primary, ~±0.25°. ITK's multithreaded Mattes " +
      "reduction is run-order dependent — the seed pins the samples, not the result. Re-run and this number moves; " +
      "don't read a difference this small as a difference.",
  },
  {
    id: "fixed",
    label: "2-constant fix",
    blurb:
      "The same function with two constants changed: smoothing sigmas [2.0, 1.0, 0.0] → [0.04, 0.02, 0.0] mm and " +
      "learning rate 0.8 → 0.03 (no mask). The sigmas are in mm, and this volume is ~13× anisotropic — the legacy " +
      "values blur the cornea away at the coarse levels.",
  },
  {
    id: "bruteforce",
    label: "Brute-force translation",
    blurb:
      "Exhaustive full-resolution 3-DOF translation search by FFT cross-correlation. Deterministic, but translation " +
      "only — it cannot correct rotation OR a TILT, so a tilted pair will not close no matter what it finds. Watch " +
      "the surface residual rather than primary: a leftover tilt barely moves an NCC score.",
  },
];

export const ALIGN_VIEWS: { id: ViewId; label: string; hint: string }[] = [
  { id: "bscan", label: "B-scan", hint: "One B-scan (the plane the OCT actually captures instantaneously)." },
  { id: "sagittal", label: "Sagittal", hint: "A sagittal cut through the frame stack — where inter-frame drift shows." },
  { id: "zoom", label: "Zoom", hint: "Zoomed on the cornea, where a sub-voxel misalignment is actually visible." },
];

/** The two 3-D turntable modes, offered ALONGSIDE the 2-D views. Selecting one is what makes the run
    request the (heavier) 3-D render — see `run()`. Same magenta/green convention as the 2-D overlap for
    OVERLAP; a hot |diff| MIP for DISAGREEMENT (the novel view). */
export const ALIGN_MODES_3D: { id: Mode3d; label: string; hint: string }[] = [
  {
    id: "overlap",
    label: "3D Overlap",
    hint:
      "3-D magenta/green MIP of the aligned pair, rotating about the en-face axis. White/grey where the two " +
      "replicates agree, coloured fringes where they don't — the round-view twin of the 2-D overlay.",
  },
  {
    id: "disagreement",
    label: "3D Disagreement",
    hint:
      "3-D MIP of |fixed − moving| inside the tissue, hot where the replicates differ. A residual TILT reads " +
      "as a band along one edge; a genuine per-replicate SCAR difference reads as a localized blob — both " +
      "invisible on a single slice, obvious in the round.",
  },
];

export interface AlignGroup {
  eye: string;
  cases: string[];
}

/** The per-method 3-D turntable payload (present only when the run was launched with render_3d=true and the
    method did not error). Each mode is an ORDERED list of frame PNG URLs — the SAME N frames, camera, window
    and crop across every method and both modes, so flipping methods or scrubbing the angle keeps the pose
    locked and only the fringes/hotspots move. `overlap` = magenta(fixed)/green(moving)/white(agree) MIP;
    `disagreement` = hot |fixed−moving| MIP over dim anatomy. URLs are GET-able by an <img> (token-exempt). */
export interface Turntable {
  overlap?: string[] | null;
  disagreement?: string[] | null;
  n_frames?: number;
  axis?: string;
}

export interface AlignResult {
  method: string;
  label?: string;
  ok?: boolean;
  raised?: boolean;
  rot_deg?: number | null;
  t_mm?: number[] | null;
  primary?: number | null;
  identity_primary?: number | null;
  delta?: number | null;
  frac_out?: number | null;
  runtime_s?: number | null;
  error?: string | null;
  views?: Partial<Record<ViewId, string>> | null;

  /* GEOMETRIC truth — mean |Δ| between the two scans' detected anterior surfaces after alignment, and
     the residual tilt across the frame stack. These are the numbers that agree with the pictures:
     a pure translation cannot remove a tilt, and NCC (`primary`, an intensity proxy over a dilated
     tissue mask) is dominated by bulk overlap so it barely registers one. On the anchor pair the rigid
     methods leave 1.6 vox (~5 µm) and brute-force leaves 9.6 vox (~30 µm) — 6× — while their primary
     scores tie inside the noise floor. ~30 µm of boundary error lands straight in scar Dice when a scar
     label is propagated onto a replicate, which is what this alignment exists to do. */
  resid_um?: number | null;
  resid_vox?: number | null;
  tilt_vox?: number | null;

  /** 3-D turntable frames for this method (see Turntable). null when the run was 2-D-only or the method
      errored — the 3-D card then shows a spinner (still rendering) or the method's error. */
  turntable?: Turntable | null;
}

/** Backend job-level geometry detail (shapes/spacings of the pair + the render window). Also carries
    the view_geometry slice indices/crops, hence the index signature. */
export interface AlignGeometry {
  fixed_shape?: number[];
  moving_shape?: number[];
  fixed_spacing_mm?: number[];
  moving_spacing_mm?: number[];
  window?: number[];
  [k: string]: unknown;
}

interface AlignJob {
  status: "running" | "done" | "error";
  progress?: number;
  error?: string | null;
  results?: AlignResult[] | null;
  /** A real per-pair finding from the backend — e.g. "Geometry differs: … the two scans do not cover
      the same FOV." MUST reach the DOM: without it the expert reads a low score + ugly composites as a
      METHOD failure when the truth is a DATA mismatch (cs005_od v1 vs v9 differ in lateral spacing). */
  note?: string | null;
  geometry?: AlignGeometry | null;
}

/** A view URL from the backend may be sidecar-relative ("/api/debug/…") or already absolute. Both are
    GET-able by an <img> without a token; only the relative form needs the sidecar base prepended. */
export const viewUrl = (u?: string | null): string =>
  !u ? "" : /^(https?:|data:|blob:)/.test(u) ? u : resourceUrl(u);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// The pair the user asked to see first, when this eye is present in the case store.
const DEFAULT_EYE = "cs001_os";
const DEFAULT_FIXED = "case_cs001_os_v2";
const DEFAULT_MOVING = "case_cs001_os_v3";

const POLL_MS = 900;
const MAX_POLL_FAILURES = 5; // tolerate a blip; give up before spinning forever on a dead sidecar

// Respect the OS "reduce motion" setting for the turntable's DEFAULT: the auto-rotate spinner is a moving
// animation, so if the user has asked for less motion we start paused (they can still opt in). Read once at
// module load; the component also re-checks live so a mid-session OS change is honoured.
const prefersReducedMotion =
  typeof window !== "undefined" &&
  typeof window.matchMedia === "function" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

interface DebugState {
  groups: AlignGroup[];
  groupsBusy: boolean;
  groupsLoaded: boolean;
  groupsError: string | null;

  eye: string | null;
  fixedCase: string | null;
  movingCase: string | null;
  methods: Record<MethodId, boolean>;

  running: boolean;
  progress: number;
  error: string | null;
  results: AlignResult[];
  note: string | null;          // job-level finding from the backend (geometry mismatch, …)
  geometry: AlignGeometry | null;
  runToken: number; // guards a superseded run from writing over a newer one's results

  view: ViewId;
  layout: Layout;
  focus: MethodId; // the method shown large in flip layout

  /* 3-D turntable axis. `mode3d` null = a 2-D view is active; a value = a 3-D turntable is active (and the
     next run renders 3-D). `rotIdx` is the SHARED frame index every method's turntable rotates to, so the
     row stays in one pose. `render3d` tracks whether the CURRENT results carry 3-D turntables — it gates the
     lazy auto-render (don't re-run if the data is already there) and resets whenever the pair changes. */
  mode3d: Mode3d | null;
  rotIdx: number;
  autoRotate: boolean;
  render3d: boolean;

  loadGroups: () => Promise<void>;
  selectEye: (eye: string) => void;
  setFixed: (c: string) => void;
  setMoving: (c: string) => void;
  swapPair: () => void;
  toggleMethod: (m: MethodId) => void;
  setView: (v: ViewId) => void;
  setMode3d: (m: Mode3d | null) => void;
  setRotIdx: (i: number) => void;
  stepRot: (n: number) => void;
  setAutoRotate: (on: boolean) => void;
  setLayout: (l: Layout) => void;
  setFocus: (m: MethodId) => void;
  cycleFocus: (dir: 1 | -1) => void;
  run: () => Promise<void>;
}

export const useDebugStore = create<DebugState>()(
  immer((set, get) => ({
    groups: [],
    groupsBusy: false,
    groupsLoaded: false,
    groupsError: null,

    eye: null,
    fixedCase: null,
    movingCase: null,
    methods: { identity: true, asis: true, fixed: true, bruteforce: true },

    running: false,
    progress: 0,
    error: null,
    results: [],
    note: null,
    geometry: null,
    runToken: 0,

    view: "bscan",
    layout: "grid",
    focus: "identity",

    mode3d: null,
    rotIdx: 0,
    autoRotate: !prefersReducedMotion, // start paused when the OS asks for reduced motion
    render3d: false,

    // Lazy by design: nothing fetches until the Debug tab mounts the panel and calls this.
    loadGroups: async () => {
      if (get().groupsBusy || get().groupsLoaded) return;
      set((s) => {
        s.groupsBusy = true;
        s.groupsError = null;
      });
      try {
        const r = await api.json<{ groups: AlignGroup[] }>("/api/debug/align/groups");
        const groups = (r.groups || []).filter((g) => (g.cases?.length ?? 0) >= 2);
        set((s) => {
          s.groups = groups;
          s.groupsLoaded = true;
          // Default to the pair the user asked about; otherwise the first eye's first two replicates.
          const wanted = groups.find((g) => g.eye === DEFAULT_EYE);
          const g =
            wanted && wanted.cases.includes(DEFAULT_FIXED) && wanted.cases.includes(DEFAULT_MOVING)
              ? wanted
              : groups[0];
          if (!g) return;
          s.eye = g.eye;
          const isDefault = g === wanted && g.cases.includes(DEFAULT_FIXED) && g.cases.includes(DEFAULT_MOVING);
          s.fixedCase = isDefault ? DEFAULT_FIXED : g.cases[0];
          s.movingCase = isDefault ? DEFAULT_MOVING : g.cases[1];
        });
      } catch (e) {
        set((s) => {
          s.groupsError = e instanceof Error ? e.message : String(e);
          s.groupsBusy = false;
        });
        return;
      }
      set((s) => {
        s.groupsBusy = false;
      });
    },

    selectEye: (eye) =>
      set((s) => {
        if (s.eye === eye) return;
        s.eye = eye;
        const g = s.groups.find((x) => x.eye === eye);
        s.fixedCase = g?.cases[0] ?? null;
        s.movingCase = g?.cases[1] ?? null;
        s.results = []; // results belong to the old pair — never show them under a new one
        s.note = null;  // ...and so does the geometry note: showing it under a new pair would be a lie
        s.geometry = null;
        s.render3d = false; // the new pair has no 3-D turntables yet — a 3-D mode re-renders them lazily
        s.error = null;
        s.progress = 0;
      }),

    setFixed: (c) =>
      set((s) => {
        if (s.fixedCase === c) return;
        s.fixedCase = c;
        if (s.movingCase === c) {
          // A scan can't be compared with itself — push moving to any other replicate.
          const g = s.groups.find((x) => x.eye === s.eye);
          s.movingCase = g?.cases.find((x) => x !== c) ?? null;
        }
        s.results = [];
        s.note = null;
        s.geometry = null;
        s.render3d = false; // the new pair has no 3-D turntables yet — a 3-D mode re-renders them lazily
        s.error = null;
      }),

    setMoving: (c) =>
      set((s) => {
        if (s.movingCase === c) return;
        s.movingCase = c;
        if (s.fixedCase === c) {
          const g = s.groups.find((x) => x.eye === s.eye);
          s.fixedCase = g?.cases.find((x) => x !== c) ?? null;
        }
        s.results = [];
        s.note = null;
        s.geometry = null;
        s.render3d = false; // the new pair has no 3-D turntables yet — a 3-D mode re-renders them lazily
        s.error = null;
      }),

    swapPair: () =>
      set((s) => {
        const f = s.fixedCase;
        s.fixedCase = s.movingCase;
        s.movingCase = f;
        s.results = []; // magenta/green swap meaning with the pair — stale images would mislead
        s.note = null;  // the note names fixed/moving — it is wrong the moment they swap
        s.geometry = null;
        s.render3d = false; // the new pair has no 3-D turntables yet — a 3-D mode re-renders them lazily
        s.error = null;
      }),

    toggleMethod: (m) =>
      set((s) => {
        if (ALIGN_METHODS.find((x) => x.id === m)?.locked) return; // identity is the reference; not optional
        s.methods[m] = !s.methods[m];
      }),

    setView: (v) => set((s) => { s.view = v; }),
    // Selecting a 3-D mode only flips the axis; the component's lazy effect notices `render3d` is false and
    // fires the (heavier) 3-D render — so 2-D runs stay light and 3-D is never requested until asked for.
    setMode3d: (m) => set((s) => { s.mode3d = m; }),
    setRotIdx: (i) => set((s) => { s.rotIdx = i < 0 ? 0 : Math.round(i); }),
    // Advance one frame with wraparound — driven by the auto-rotate timer. Takes n from the component so the
    // timer callback never closes over a stale frame count.
    stepRot: (n) => set((s) => { if (n > 0) s.rotIdx = (s.rotIdx + 1) % n; }),
    setAutoRotate: (on) => set((s) => { s.autoRotate = on; }),
    setLayout: (l) => set((s) => { s.layout = l; }),
    setFocus: (m) => set((s) => { s.focus = m; }),

    // Left/right cycling through whatever methods actually returned — the core interaction: the same
    // view, the same window, one method swapped for the next, so the fringes are the only thing moving.
    cycleFocus: (dir) =>
      set((s) => {
        const ids = s.results.map((r) => r.method as MethodId);
        if (ids.length === 0) return;
        const i = ids.indexOf(s.focus);
        s.focus = ids[(((i < 0 ? 0 : i) + dir) % ids.length + ids.length) % ids.length];
      }),

    run: async () => {
      const { fixedCase, movingCase, methods } = get();
      if (!fixedCase || !movingCase) return;
      if (fixedCase === movingCase) {
        set((s) => { s.error = "Pick two DIFFERENT replicates — a scan aligned to itself is trivially identity."; });
        return;
      }
      const list = ALIGN_METHODS.filter((m) => m.locked || methods[m.id]).map((m) => m.id);
      // 3-D is opt-in: only render the (heavier) turntable when a 3-D mode is the active axis. The flag is
      // set NOW (not on completion) so the lazy render effect sees the render in-flight and never double-fires
      // — and it stays set even if the run errors, so a failing sidecar can't spin the effect into a loop.
      const wants3d = get().mode3d != null;
      const token = get().runToken + 1;
      set((s) => {
        s.runToken = token;
        s.running = true;
        s.progress = 0;
        s.error = null;
        s.results = [];
        s.note = null;
        s.geometry = null;
        s.render3d = wants3d;
      });
      // A newer run (or a pair change) supersedes this one — drop its writes instead of racing them.
      const current = () => get().runToken === token;
      try {
        const { job_id } = await api.json<{ job_id: string }>(
          "/api/debug/align/compare",
          "POST",
          JSON.stringify({ fixed_case: fixedCase, moving_case: movingCase, methods: list, render_3d: wants3d }),
        );
        if (!current()) return;
        if (!job_id) throw new Error("The sidecar did not return a job id.");

        let fails = 0;
        for (;;) {
          await sleep(POLL_MS);
          if (!current()) return;
          let job: AlignJob;
          try {
            job = await api.json<AlignJob>(`/api/debug/align/job/${job_id}`);
            fails = 0;
          } catch (e) {
            if (++fails >= MAX_POLL_FAILURES) throw e;
            continue;
          }
          if (!current()) return;
          set((s) => {
            s.progress = typeof job.progress === "number" ? job.progress : s.progress;
            // The backend fills these in BEFORE the first method finishes, so a geometry mismatch is on
            // screen while the run is still going — which is exactly when the user needs to know that the
            // ugly composites about to appear are a data mismatch, not a method failure.
            s.note = job.note ?? null;
            s.geometry = job.geometry ?? null;
            if (job.results) {
              s.results = job.results; // partial results render as they land
              if (!job.results.some((r) => r.method === s.focus) && job.results[0])
                s.focus = job.results[0].method as MethodId;
            }
          });
          if (job.status === "done") break;
          if (job.status === "error") throw new Error(job.error || "The alignment run failed.");
        }
        if (!current()) return;
        set((s) => { s.progress = 1; });
      } catch (e) {
        if (!current()) return;
        set((s) => { s.error = e instanceof Error ? e.message : String(e); });
      } finally {
        if (current()) set((s) => { s.running = false; });
      }
    },
  })),
);
