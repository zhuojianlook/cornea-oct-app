import { test, expect, gotoApp, openCase, mainButtons, mainBtn, canvasCount, FIX } from "./helpers";

/* Step 9 (eye-consensus) surface for an aligned-replicates consensus case (FIX.consensus).
 * READ-ONLY: we assert the step-9 action bar WITHOUT clicking its backend-mutating buttons, then
 * exercise the four client-side subgroup-grid MODES (Consensus / Scans grid / Volume align /
 * Scar overlap), which only swap the viewer and never persist. */
test.describe("consensus (step 9)", () => {
  test("step-9 action buttons are present without being clicked", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.consensus);

    // The mutating step-9 actions must be OFFERED (presence only — clicking them is reserved for
    // progression.spec.ts so the once-seeded fixtures stay at step 9).
    const btns = await mainButtons(page);
    expect(btns).toEqual(
      expect.arrayContaining(["Consensus", "Scans grid", "Volume align", "Scar overlap"]),
    );
    // Glyph-prefixed action-bar labels → match by substring/RegExp, never click.
    await expect(mainBtn(page, /Use consensus \(all\)/)).toBeVisible();
    await expect(mainBtn(page, /Keep each replicate's/)).toBeVisible();
    await expect(mainBtn(page, /Normalize against controls/)).toBeVisible();
    await expect(mainBtn(page, /Skip normalization/)).toBeVisible();
    await expect(mainBtn(page, /Correct/)).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });

  test("Consensus mode renders the voted-consensus image", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.consensus);

    // "Consensus" is the default grid mode; click it explicitly to be deterministic (client-side).
    await mainBtn(page, "Consensus").click();
    await expect(page.locator("main img").first()).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });

  test("Scans grid renders multiple images with a Per-scan / Consensus scar toggle", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.consensus);

    await mainBtn(page, "Scans grid").click();

    // The grid lays out before/corrected/scar previews per replicate → several <img>. Wait for the grid
    // to populate (lazy preview listings land a beat after the mode switch) before counting.
    const imgs = page.locator("main img");
    await expect(imgs.first()).toBeVisible();
    await expect.poll(() => imgs.count()).toBeGreaterThan(1);

    // The scar-source toggle (unique to grid mode = SubgroupGrid's overlay toggle) offers "Per scan" +
    // "Consensus". "Consensus" is the grid's DEFAULT scar source, so the group renders both buttons.
    const perScan = mainBtn(page, "Per scan");
    await expect(perScan).toBeVisible();

    // Flip the scar source: Per scan -> Consensus (both client-side, non-persisting). There are TWO
    // "Consensus" buttons in <main> (the grid-MODE toggle and this scar-SOURCE toggle), so scope the
    // click to the scar-source ToggleButtonGroup. Build the `has` filter from a fresh, NON-.first()
    // locator — a `.first()`-terminated locator (like mainBtn's) is not valid inside `has`.
    await perScan.click();
    const scarToggleGroup = page
      .locator("main .MuiToggleButtonGroup-root")
      .filter({ has: page.getByRole("button", { name: "Per scan", exact: true }) });
    await scarToggleGroup.getByRole("button", { name: "Consensus", exact: true }).click();

    // Still in the grid (the scar-source toggle is grid-only) and still showing images.
    await expect(mainBtn(page, "Per scan")).toBeVisible();
    await expect(imgs.first()).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });

  test("Volume align renders a niivue canvas with per-replicate opacity sliders", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.consensus);

    await mainBtn(page, "Volume align").click();

    // The pairwise-overlap viewer owns its own niivue canvas.
    await expect.poll(() => canvasCount(page)).toBeGreaterThanOrEqual(1);

    // Three opacity sliders (replicate A / replicate B / overlap) → at least two.
    const sliders = page.locator("main").getByRole("slider");
    expect(await sliders.count()).toBeGreaterThanOrEqual(2);

    // The viewer is framed in terms of replicates.
    await expect(page.locator("main").getByText(/replicate/i).first()).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });

  test("Scar overlap renders a niivue canvas with a tolerance control", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.consensus);

    await mainBtn(page, "Scar overlap").click();

    // The 3D agreement viewer owns its own niivue canvas.
    await expect.poll(() => canvasCount(page)).toBeGreaterThanOrEqual(1);

    // The boundary-tolerance control (label + slider) is the defining affordance of this mode.
    await expect(page.locator("main").getByText(/tolerance/i).first()).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });
});
