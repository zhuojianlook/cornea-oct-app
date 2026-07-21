/* Review-flag vocabulary — the single source of truth for the per-scan attention flags written to
   manifest.review_flags, shared by the sidebar row BADGE and the case-type FILTER (both in OctLoader) so
   a flag's label/colour can never drift between them. The backend (api_server.py set_review_flags) does
   NOT know these names: it accepts any slug matching ^[a-z][a-z0-9-]{1,31}$, so this table is purely
   presentational and an UNKNOWN slug — a new backend flag, or a retired A/B/C cybernetic-loop value still
   sitting in an old manifest — must still render readably (see reviewFlagMeta). */

export interface ReviewFlagMeta {
  slug: string;         // the value stored in manifest.review_flags
  label: string;        // full name (filter option, tooltip heading)
  short: string;        // compact chip text for the DENSE sidebar row — keep to ~7 chars
  description: string;  // what the flag actually means (tooltip body)
  color: string;        // chip colour
}

// Ordered most- to least-serious, which is also the order the filter options appear in.
// Colours reuse the app's existing attention palette: red/orange = a real defect in the output,
// sky = informational (matches the ⬚ surface-crop badge, which is the same subsystem), warm amber/
// yellow + violet = statistical outliers from a round's distribution, not necessarily wrong.
export const REVIEW_FLAGS: ReviewFlagMeta[] = [
  { slug: "zeroed-frames", label: "Zeroed frames", short: "Zeroed", color: "#ef4444",       // red (worst)
    description: "The automatic off-cornea noise-crop blanked a contiguous block of B-scans." },
  { slug: "still-clipped", label: "Still clipped", short: "Clipped", color: "#fb923c",      // orange
    description: "Cornea surface is still at the canvas top after correction." },
  { slug: "crop-clamped", label: "Crop clamped", short: "Clamped", color: "#38bdf8",        // sky (informational)
    description: "Surface-crop upward extension hit the 160-row safety cap." },
  { slug: "residual-motion", label: "Residual motion", short: "Motion", color: "#f59e0b",   // amber (outlier)
    description: "Inter-frame surface motion in the round's worst 5%." },
  { slug: "jagged-surface", label: "Jagged surface", short: "Jagged", color: "#facc15",     // yellow (outlier)
    description: "In-B-scan surface jaggedness in the round's worst 5%." },
  { slug: "weak-edge", label: "Weak edge", short: "Weak", color: "#a78bfa",                 // violet (outlier)
    description: "Low detector confidence — faint or ambiguous corneal edge." },
];

const BY_SLUG = new Map(REVIEW_FLAGS.map((f) => [f.slug, f]));

/** Presentation for one flag slug. An unrecognised slug is NEVER dropped — it renders as itself in the
 *  neutral flag amber, so a flag the backend gained before this table did (or a legacy A/B/C value) is
 *  still visible and searchable rather than silently missing. Its chip text is truncated because an
 *  arbitrary slug is unbounded and the sidebar row is narrow. */
export function reviewFlagMeta(slug: string): ReviewFlagMeta {
  const known = BY_SLUG.get(slug);
  if (known) return known;
  return {
    slug,
    label: slug,
    short: slug.length > 12 ? `${slug.slice(0, 11)}…` : slug,
    description: "Unrecognised review flag — no description available.",
    color: "#f59e0b",
  };
}

/** The flags on a scan's `life`, as a clean string list. `life` is optional and DUAL-SHAPE (the
 *  cases/list payload vs the full mirrored manifest of the open scan), and its review_flags is typed
 *  `unknown` — so filter to strings rather than trusting the contents: a non-string element would
 *  otherwise reach the badge and render as "[object Object]". */
export function reviewFlagsOf(life?: Record<string, unknown>): string[] {
  const raw = life?.review_flags;
  return Array.isArray(raw) ? raw.filter((f): f is string => typeof f === "string" && f.length > 0) : [];
}
