import { test, expect, gotoApp, openCase, mainButtons, FIX } from "./helpers";

/* READ-ONLY lifecycle coverage: for each seeded fixture case, open it and assert the
 * step-specific action-bar buttons are present (or, for the raw step, the prompt message).
 * Do NOT click any backend-mutating action button here — fixtures are seeded once and the
 * suite runs serially (see progression.spec.ts for the mutating walk). */

test.describe("lifecycle", () => {
  test("step 1 (raw): prompts to preprocess, no step buttons", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.raw);
    await expect(page.getByText("Preprocess this scan in the sidebar")).toBeVisible();
    expect(consoleErrors).toEqual([]);
  });

  test("step 2 (auto): preprocessing approval + display-mode buttons", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.auto);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining([
        "Approve preprocessing",
        "Re-run preprocessing",
        "Use original (raw)",
        "Before/after",
        "Fix columns",
        "Steps",
      ]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 4 (classified): offers SAM2 cornea segmentation", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.classified);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Run SAM2 (cornea)"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 5 (cornea): paint + confirm cornea/background", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.cornea);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Paint cornea/background", "Confirm cornea/background"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 6 (corneavet): confirm/auto subgroup", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.corneavet);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Confirm subgroup", "Auto subgroups"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 7 (subgroup): scar detection methods", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.subgroup);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Detect scar (threshold)", "Scar via SAM2"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 8 (scar): scar detect/correct/align", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.scar);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining([
        "Detect scar (threshold)",
        "Scar via SAM2",
        "Correct ✎",
        "Align replicates",
      ]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 11 (corrected): schedule + correct + export metrics", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.corrected);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Schedule for training", "Correct ✎", "Export metrics"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 9 (consensus): consensus choices + grid modes", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.consensus);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining([
        "Use consensus (all)",
        "Keep each replicate's",
        "Normalize against controls",
        "Skip normalization",
        "Correct ✎",
        "Consensus",
        "Scans grid",
        "Volume align",
        "Scar overlap",
      ]),
    );
    expect(consoleErrors).toEqual([]);
  });
});
