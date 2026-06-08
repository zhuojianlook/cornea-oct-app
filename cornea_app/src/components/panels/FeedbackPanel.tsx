/* Active-learning verdict: accept the paint, or reject with notes → repaint. */

import { useState } from "react";
import { Button, TextField } from "@mui/material";
import { usePaintStore } from "../../store/paintStore";

export function FeedbackPanel() {
  const { submitFeedback, aiBusy, seedImages } = usePaintStore();
  const [notes, setNotes] = useState("");
  if (seedImages.length === 0) return null;

  return (
    <div className="rounded p-2 flex flex-col gap-2" style={{ backgroundColor: "var(--c-surface2)" }}>
      <div className="text-[11px] uppercase tracking-wide" style={{ color: "var(--c-text-dim)" }}>
        Verdict
      </div>
      <TextField
        placeholder="What's wrong (used to repaint), or why it's acceptable?"
        value={notes}
        onChange={(e) => setNotes(e.target.value)}
        multiline
        minRows={2}
        fullWidth
      />
      <div className="flex gap-2">
        <Button
          variant="contained"
          color="secondary"
          fullWidth
          disabled={aiBusy}
          onClick={() => submitFeedback("accept", notes)}
        >
          Accept
        </Button>
        <Button
          variant="contained"
          color="error"
          fullWidth
          disabled={aiBusy}
          onClick={() => {
            submitFeedback("reject", notes);
            setNotes("");
          }}
        >
          Reject → Repaint
        </Button>
      </div>
    </div>
  );
}
