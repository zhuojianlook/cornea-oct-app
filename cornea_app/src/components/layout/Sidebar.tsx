import {
  TextField,
  MenuItem,
  Button,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Typography,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import { useRef, useState } from "react";
import { useCaseStore } from "../../store/caseStore";
import { usePaintStore } from "../../store/paintStore";
import { api } from "../../api/client";

const SYNTHETIC_VOLUME = "/home/zhuojian/Desktop/Integration/output/synthetic_oct.nrrd";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="px-3 py-3">
      <div
        className="text-[11px] uppercase tracking-wide mb-2"
        style={{ color: "var(--c-text-dim)" }}
      >
        {title}
      </div>
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

export function Sidebar() {
  const { config, caseId, caseInfo, busy, exportInfo, setCaseId, fetchConfig, openCase, registerVolume, uploadVolume, exportNnunet } =
    useCaseStore();
  const paint = usePaintStore();
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [volPath, setVolPath] = useState(SYNTHETIC_VOLUME);

  const inputVolume =
    (caseInfo?.manifest?.["corrected_volume"] as string) ||
    (caseInfo?.manifest?.["input_volume"] as string) ||
    "";

  const saveSettings = async () => {
    await api.putConfig({
      vision_provider: paint.provider,
      openai_model: paint.model,
      local_vision_base_url: paint.baseUrl,
      openai_api_key: paint.apiKey || undefined,
    });
    await fetchConfig();
  };

  return (
    <div className="flex flex-col">
      <Section title="Case">
        <div className="flex gap-2">
          <TextField
            label="Case ID"
            value={caseId ?? ""}
            onChange={(e) => setCaseId(e.target.value)}
            fullWidth
          />
          <Button variant="outlined" onClick={openCase} disabled={busy}>
            Open
          </Button>
        </div>
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="Volume">
        <TextField
          label="Volume path"
          value={volPath}
          onChange={(e) => setVolPath(e.target.value)}
          fullWidth
        />
        <div className="flex gap-2">
          <Button
            variant="contained"
            onClick={() => registerVolume(volPath)}
            disabled={busy || !caseId}
            fullWidth
          >
            Register
          </Button>
          <Button variant="outlined" onClick={() => fileRef.current?.click()} disabled={busy || !caseId}>
            Upload
          </Button>
          <input
            ref={fileRef}
            type="file"
            accept=".nrrd,.nii,.gz,.mha,.mhd"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) uploadVolume(f);
              e.target.value = "";
            }}
          />
        </div>
        {inputVolume && (
          <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-all" }}>
            {inputVolume}
          </Typography>
        )}
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="Training export">
        <Button variant="outlined" onClick={exportNnunet} disabled={busy}>
          Export all → nnU-Net
        </Button>
        {exportInfo && (
          <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-all" }}>
            {exportInfo}
          </Typography>
        )}
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Accordion
        disableGutters
        sx={{ backgroundColor: "transparent", boxShadow: "none", "&:before": { display: "none" } }}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon fontSize="small" />}>
          <span className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
            Model &amp; Settings
          </span>
        </AccordionSummary>
        <AccordionDetails>
          <div className="flex flex-col gap-2">
            <TextField
              select
              label="Vision provider"
              value={paint.provider}
              onChange={(e) => paint.set("provider", e.target.value as typeof paint.provider)}
              fullWidth
            >
              <MenuItem value="local">Local (OpenAI-compatible)</MenuItem>
              <MenuItem value="openai">OpenAI cloud</MenuItem>
              <MenuItem value="medgemma">MedGemma bridge</MenuItem>
            </TextField>
            <TextField
              label="Model"
              value={paint.model}
              onChange={(e) => paint.set("model", e.target.value)}
              fullWidth
            />
            <TextField
              label="Local base URL"
              value={paint.baseUrl}
              onChange={(e) => paint.set("baseUrl", e.target.value)}
              fullWidth
            />
            <TextField
              label="OpenAI API key"
              type="password"
              placeholder={config?.has_openai_api_key ? "set (from env)" : "blank uses env"}
              value={paint.apiKey}
              onChange={(e) => paint.set("apiKey", e.target.value)}
              fullWidth
            />
            <TextField
              label="Paint instruction (optional)"
              value={paint.reviewerPrompt}
              onChange={(e) => paint.set("reviewerPrompt", e.target.value)}
              multiline
              minRows={2}
              fullWidth
            />
            <Button variant="outlined" onClick={saveSettings}>
              Save settings
            </Button>
            <Typography variant="caption" sx={{ color: "var(--c-text-dim)" }}>
              Slicer: {config?.slicer_executable}
            </Typography>
          </div>
        </AccordionDetails>
      </Accordion>
    </div>
  );
}
