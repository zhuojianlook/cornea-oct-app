/* Shared API types — mirror the sidecar's response shapes. */

export interface AppConfig {
  workspace_root: string;
  cases_root: string;
  slicer_executable: string;
  default_case_id: string;
}

export interface CaseInfo {
  case_id: string;
  root: string;
  input_dir: string;
  segmentation_dir: string;
  qa_json: string;
  manifest: Record<string, unknown>;
}

export interface PreviewImage {
  label: string;
  path: string;
  group: string;
  file_name: string;
  data_url: string;
  /** Lazy preview URL (relative to the sidecar). When set, the gallery loads this slice on
      demand via resourceUrl(src) instead of an inline base64 data_url — used for dense scrub
      groups (every slice) so the listing stays small. */
  src?: string | null;
  orientation: string | null;
  slice_index: number | null;
  source_width: number | null;
  source_height: number | null;
  image_width: number | null;
  image_height: number | null;
  /** 90° CCW turns (np.rot90 k) baked into the PNG for display; clicks undo it (coords.ts). */
  rotate_k?: number | null;
}

export type Stage = 1 | 2 | 3;

export interface ConsensusScan {
  case: string;
  role: string; // "reference" | "scar" | "cornea" (alignment anchor used)
  scar_volume_mm3: number;
  scar_dice_to_ref: number;
  scar_dice_to_ref_fov?: number; // Dice vs ref restricted to the shared field-of-view (partial-cut-aware)
  fov_overlap_fraction?: number; // how much of the two scans' coverage is shared
  matched_fraction: number; // fraction of this scan's scar that falls in the consensus
  low_correspondence: boolean; // likely a different FOV patch — only partly comparable
}

export interface ConsensusReport {
  n_scans: number;
  reference: string;
  agreement_threshold: number;
  scar_volume_mm3: { mean: number; std: number; cv_percent: number; per_scan: number[] };
  consensus_scar_mm3: number;
  core_full_agreement_mm3: number;
  union_mm3: number;
  mean_pairwise_scar_dice: number | null;
  mean_pairwise_scar_dice_fov?: number | null; // agreement within the shared FOV (partial-cut-aware)
  per_scan: ConsensusScan[];
  scans: string[];
  segmentation_errors?: Record<string, string>;
  subgroup?: string;
}

export interface ScarMetrics {
  scar_present: boolean;
  scar_voxels?: number;
  scar_volume_mm3?: number;
  scar_area_mm2?: number;
  cornea_voxels?: number;
  cornea_volume_mm3?: number;
  scar_fraction_of_cornea?: number;
  scar_components?: number;
  largest_component_fraction?: number;
  scar_density?: {
    mean?: number; median?: number; std?: number; p10?: number; p90?: number;
    weighted_volume_mm3u?: number; tier_volume_mm3?: number[];
  };
  scar_bounds_ijk?: { min: number[]; max: number[] };
  note?: string;
}
