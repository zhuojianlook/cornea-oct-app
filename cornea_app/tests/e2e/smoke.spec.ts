import { test, expect, gotoApp, openCase, mainButtons, FIX } from "./helpers";

test.describe("smoke", () => {
  test("app shell loads with the sidebar + 12-step timeline and no console errors", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    // every lifecycle step label is present in the strip (pills render as "<n>Label", so substring match)
    for (const label of ["Auto", "Vetted", "Classified", "Subgroup", "Aligned", "Normalized", "Corrected", "Scheduled"]) {
      await expect(page.getByText(label).first()).toBeVisible();
    }
    expect(consoleErrors).toEqual([]);
  });

  test("opening the consensus case reaches the step-9 surface", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.consensus);
    const btns = await mainButtons(page);
    // step-9 action bar + subgroup-grid modes
    expect(btns).toEqual(expect.arrayContaining(["Use consensus (all)", "Keep each replicate's", "Consensus", "Scans grid", "Volume align", "Scar overlap"]));
    expect(consoleErrors).toEqual([]);
  });
});
