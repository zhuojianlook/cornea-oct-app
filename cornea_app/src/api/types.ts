/* Shared API types — mirror the sidecar's response shapes. */

export interface AppConfig {
  workspace_root: string;
  cases_root: string;
  slicer_executable: string;
  default_case_id: string;
  vision_provider: "local" | "openai" | "medgemma" | string;
  openai_model: string;
  local_vision_base_url: string;
  has_openai_api_key: boolean;
}

export interface CaseInfo {
  case_id: string;
  root: string;
  input_dir: string;
  segmentation_dir: string;
  review_dir: string;
  seed_json: string;
  output_seg: string;
  qa_json: string;
  manifest: Record<string, unknown>;
}

export interface SeedStroke {
  points_ijk: number[][];
  radius_voxels: number[];
}

export interface SeedSeedPoint {
  ijk: number[];
  radius_voxels: number[];
}

export interface SeedSegment {
  name: string;
  color: number[];
  seeds: SeedSeedPoint[];
  strokes?: SeedStroke[];
}

export interface SeedSpec {
  segments: SeedSegment[];
}

export interface GrowQa {
  segments?: Record<string, { voxel_count?: number; volume_mm3?: number }>;
  [key: string]: unknown;
}

export interface PreviewImage {
  label: string;
  path: string;
  group: string;
  file_name: string;
  data_url: string;
  orientation: string | null;
  slice_index: number | null;
  source_width: number | null;
  source_height: number | null;
  image_width: number | null;
  image_height: number | null;
}

export interface AiPaintResult {
  case_info: CaseInfo;
  seed_spec: SeedSpec;
  qa: {
    agent_marking?: { cornea_stroke_count?: number; background_stroke_count?: number };
    confidence?: number | null;
    issues?: string[];
    paint_agent?: { provider?: string; model?: string; summary?: string };
    [k: string]: unknown;
  } | null;
  mode?: string;
}

export type Stage = 1 | 2 | 3 | 4;

export interface ScarMetrics {
  scar_present: boolean;
  scar_voxels?: number;
  cornea_voxels?: number;
  scar_fraction_of_cornea?: number;
  scar_bounds_ijk?: { min: number[]; max: number[] };
  note?: string;
}
