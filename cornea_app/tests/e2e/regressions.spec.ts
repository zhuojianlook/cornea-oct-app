import { test, expect, gotoApp, openCase, mainButtons, mainBtn, canvasCount, FIX } from "./helpers";

test.describe("regressions (cycle 1-6 fixes)", () => {
  test("(1) FIX.scar scar-method select has no 'Morph + largest component' option (morph_lcc removed v0.0.92)", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.scar);
    // the scar-method <select> is on the step-8 surface; the retired option must be gone
    await expect(page.locator("option", { hasText: "Morph + largest component" })).toHaveCount(0);
    expect(consoleErrors).toEqual([]);
  });

  test("(2) FIX.auto opens with no console errors (no segmentation-display 404 on step-2 open, v0.0.94)", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.auto);
    // step-2 action bar should be present; assert it rendered before settling on the error check
    const btns = await mainButtons(page);
    expect(btns).toEqual(expect.arrayContaining(["Approve preprocessing", "Re-run preprocessing", "Use original (raw)"]));
    // a rendered niivue viewer (the segmentation-display path that used to 404) has >=1 canvas
    expect(await canvasCount(page)).toBeGreaterThanOrEqual(1);
    expect(consoleErrors).toEqual([]);
  });

  test("(3) FIX.corrected exposes the 'Export metrics' button (step 11)", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.corrected);
    await expect(mainBtn(page, /Export metrics/)).toBeVisible();
    expect(consoleErrors).toEqual([]);
  });

  test("(4) deleted dead panels are gone (no 'Cohort batch') but 'OCT preprocessing' remains", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await expect(page.getByText("Cohort batch")).toHaveCount(0);
    await expect(page.getByText("OCT preprocessing")).toBeVisible();
    expect(consoleErrors).toEqual([]);
  });
});
