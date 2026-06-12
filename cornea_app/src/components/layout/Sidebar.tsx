import { TextField, Button, Divider, Typography } from "@mui/material";
import { useCaseStore } from "../../store/caseStore";
import { ScanLoader } from "../panels/ScanLoader";
import { OctLoader } from "../panels/OctLoader";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="px-3 py-3">
      <div className="text-[11px] uppercase tracking-wide mb-2" style={{ color: "var(--c-text-dim)" }}>
        {title}
      </div>
      <div className="flex flex-col gap-2">{children}</div>
    </div>
  );
}

export function Sidebar() {
  const { config, caseId, caseInfo, busy, exportInfo, setCaseId, openCase, exportNnunet } = useCaseStore();

  const inputVolume =
    (caseInfo?.manifest?.["corrected_volume"] as string) ||
    (caseInfo?.manifest?.["input_volume"] as string) ||
    "";

  return (
    <div className="flex flex-col">
      <Section title="OCT preprocessing">
        <OctLoader />
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="Load volume(s) directly">
        <ScanLoader />
      </Section>

      <Divider sx={{ borderColor: "var(--c-border)" }} />

      <Section title="Open existing case">
        <div className="flex gap-2">
          <TextField label="Case ID" value={caseId ?? ""} onChange={(e) => setCaseId(e.target.value)} fullWidth />
          <Button variant="outlined" onClick={openCase} disabled={busy}>
            Open
          </Button>
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
        {config?.slicer_executable && (
          <Typography variant="caption" sx={{ color: "var(--c-text-dim)", wordBreak: "break-all" }}>
            Slicer: {config.slicer_executable}
          </Typography>
        )}
      </Section>
    </div>
  );
}
