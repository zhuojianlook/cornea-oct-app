import { test as base, expect, type Page, type Locator } from "@playwright/test";

/* Deterministic fixture case ids produced by _seed.py (one per lifecycle step + a consensus). */
export const FIX = {
  raw: "case_zz_raw",                 // step 1
  auto: "case_zz_auto",               // step 2 (preprocessed, unvetted)
  classified: "case_zz_classified",   // step 4
  cornea: "case_zz_cornea",           // step 5 (cornea segmented, awaiting vet)
  corneavet: "case_zz_corneavet",     // step 6 (cornea vetted, awaiting subgroup)
  control: "case_zz_control",         // step 6 control (no scar) — steps 7-11 N/A
  subgroup: "case_zz_subgroup",       // step 7 (subgroup assigned)
  scar: "case_zz_scar",               // step 8 (scar segmented)
  vet: "case_zz_vet",                 // step 2 (MUTABLE — Approve preprocessing -> classify)
  corrected: "case_zz_corrected",     // step 11 (MUTABLE — Schedule / Unschedule)
  members: ["case_zz_od_v1", "case_zz_od_v2", "case_zz_od_v3"],
  consensus: "case_zzod_od_consensus", // step 9 (replicates aligned)
} as const;

/** A console error we tolerate everywhere (the app serves no favicon). */
const BENIGN = [/favicon\.ico/i];

/* Extended test that auto-collects console errors + page errors + >=500 responses, and exposes them
 * as `consoleErrors`. Tests can assert `expect(consoleErrors).toEqual([])` once they've settled. */
export const test = base.extend<{ consoleErrors: string[] }>({
  consoleErrors: async ({ page }, use) => {
    const errs: string[] = [];
    page.on("console", (m) => {
      if (m.type() === "error" && !BENIGN.some((re) => re.test(m.text()))) errs.push("CONSOLE: " + m.text());
    });
    page.on("pageerror", (e) => errs.push("PAGEERROR: " + e.message));
    page.on("response", (r) => { if (r.status() >= 500) errs.push(`HTTP ${r.status()} ${r.url()}`); });
    await use(errs);
  },
});
export { expect };

/** Load the app shell and wait for the sidebar + timeline to render. */
export async function gotoApp(page: Page) {
  await page.goto("/");
  await expect(page.getByText("OCT preprocessing")).toBeVisible();
  // the initial (no-case) action message confirms the shell + timeline rendered
  await expect(page.getByText("Open or preprocess a scan to begin.")).toBeVisible();
}

/** Open a case via the "Open existing case" box and wait for the viewer/timeline to reflect it. */
export async function openCase(page: Page, caseId: string) {
  const box = page.getByPlaceholder(/case id/i);
  await box.fill(caseId);
  await page.getByRole("button", { name: "Open", exact: true }).click();
  // openCase reloads the manifest -> the action bar stops saying "Open or preprocess a scan to begin".
  await expect(page.getByText("Open or preprocess a scan to begin.")).toHaveCount(0, { timeout: 15_000 });
  await page.waitForTimeout(400); // let previews/overlay settle
}

/** The visible interactive button labels inside the central <main> action area + viewer toolbar, with any
 *  LEADING decorative glyph stripped (buttons are labelled e.g. "▶ Run SAM2 (cornea)", "✓ Approve
 *  preprocessing") so callers can assert on the plain label. A trailing glyph (e.g. "Correct ✎") is kept. */
export async function mainButtons(page: Page): Promise<string[]> {
  return page.locator("main button").evaluateAll((bs) =>
    [...new Set(bs.map((b) => (b.textContent || "").trim().replace(/^[^\p{L}\p{N}(]+/u, "").trim()).filter(Boolean))]);
}

/** A button in <main> by accessible name (substring), scoped to avoid sidebar matches. */
export function mainBtn(page: Page, name: string | RegExp): Locator {
  return page.locator("main").getByRole("button", { name }).first();
}

/** Count of niivue/overlay <canvas> elements in the viewer (a rendered WebGL viewer has >=1). */
export async function canvasCount(page: Page): Promise<number> {
  return page.locator("main canvas").count();
}
