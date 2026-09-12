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

  test("step 3 (vetted): offers the group alignment, not SAM2 yet", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.vetted);
    const btns = await mainButtons(page);
    expect(btns).toEqual(expect.arrayContaining(["Align group"]));
    expect(btns).not.toContain("Run SAM2 (cornea)");
    expect(consoleErrors).toEqual([]);
  });

  test("step 4 (aligned): approve the axial changes, then the cornea detection (SAM) surfaces", async ({ page, consoleErrors }) => {
    // Reviewer spec 2026-09-11 #4: step 4 offers "✓ Approve axial changes" + a disabled "Rectify" placeholder; the
    // cornea-detection button is HIDDEN until the approval, then surfaces as "Cornea detection (SAM)".
    await gotoApp(page);
    await openCase(page, FIX.classified);
    const btns = await mainButtons(page);
    expect(btns).toEqual(expect.arrayContaining(["✓ Approve axial changes", "Rectify"]));
    expect(btns).not.toContain("Cornea detection (SAM)");
    expect(btns).not.toContain("Run SAM2 (cornea)");
    expect(btns).not.toContain("Align group");
    await expect(page.getByTestId("aligned-rectify")).toBeDisabled();
    await page.getByTestId("aligned-approve").click();
    await expect(page.getByTestId("aligned-approved")).toBeVisible({ timeout: 15_000 });
    await expect(page.getByTestId("cornea-detect-sam")).toBeVisible();
    // withdraw again so the seeded fixture is left as it was for the other tests
    await page.getByTestId("aligned-unapprove").click();
    await expect(page.getByTestId("aligned-approve")).toBeVisible({ timeout: 15_000 });
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

  test("step 7 (corneavet → classified): confirm/auto subgroup", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.corneavet);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Confirm subgroup", "Auto subgroups"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 8 (subgroup): scar detection methods", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.subgroup);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Detect scar (threshold)", "Scar via SAM2"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 9 (scar): scar detect/correct/align", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.scar);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining([
        "Detect scar (threshold)",
        "Scar via SAM2",
        "Correct ✎",
        "Align scar replicates",
      ]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 12 (corrected): schedule + correct + export metrics", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.corrected);
    expect(await mainButtons(page)).toEqual(
      expect.arrayContaining(["Schedule for training", "Correct ✎", "Export metrics"]),
    );
    expect(consoleErrors).toEqual([]);
  });

  test("step 10 (consensus, scar-aligned): consensus choices + grid modes", async ({ page, consoleErrors }) => {
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

  test("step 7 control (no scar): offers Schedule + marks scar steps 8-12 N/A", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.control);
    const btns = await mainButtons(page);
    // A control is READY after cornea vet → Schedule (+ Correct), NOT the scar/subgroup controls.
    expect(btns).toEqual(expect.arrayContaining(["Schedule for training"]));
    expect(btns).not.toContain("Confirm subgroup");
    expect(btns).not.toContain("Detect scar (threshold)");
    // Steps 8-12 (Subgroup/Scar/Scar-aligned/Normalized/Corrected) render struck-through (not applicable);
    // the group alignment (step 4 "Aligned") applies to a control too, so it is NOT struck.
    const struck = (await page.locator('main span[style*="line-through"]').allInnerTexts()).join(" ");
    for (const s of ["Subgroup", "Scar", "Scar-aligned", "Normalized", "Corrected"]) expect(struck).toContain(s);
    expect(struck).not.toContain("Aligned");   // case-sensitive: "Scar-aligned" does not match
    expect(consoleErrors).toEqual([]);
  });
});
