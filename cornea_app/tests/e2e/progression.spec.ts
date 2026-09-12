import { test, expect, gotoApp, openCase, mainButtons, mainBtn, canvasCount, FIX } from "./helpers";

/* The ONE mutating spec. Runs serially against dedicated mutable fixtures (FIX.vet, FIX.corrected)
 * so it never perturbs the read-only specs that share the seeded backend. Every other spec must
 * avoid clicking backend-mutating buttons; here we drive two real lifecycle transitions. */
test.describe.serial("progression (mutating)", () => {
  test("A: Approve preprocessing advances step 2 -> Vetted (step 3: Align group)", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.vet);

    // step 2: the Approve/Re-run/Use-original action bar is present (label text carries a leading ✓ glyph).
    const before = await mainButtons(page);
    expect(before).toEqual(expect.arrayContaining([expect.stringContaining("Approve preprocessing")]));
    // a segmented/preprocessed viewer is rendered.
    expect(await canvasCount(page)).toBeGreaterThanOrEqual(1);

    // MUTATE: approve the preprocessing -> Vetted (step 3) — the next action is the group-wise 3D alignment
    // ("⧉ Align group"); SAM2 follows at Aligned (4) and classification only after the cornea vet.
    await mainBtn(page, /Approve preprocessing/).click();

    // step 3 (vetted): the action bar now offers the group alignment (read-only here: never click it).
    await expect(mainBtn(page, /Align group/)).toBeVisible({ timeout: 15_000 });

    const after = await mainButtons(page);
    expect(after).toEqual(expect.arrayContaining(["Align group"]));
    expect(after).not.toContain("Run SAM2 (cornea)");
    // the step-2 Approve button is gone now that we've moved past it.
    expect(after).not.toEqual(expect.arrayContaining([expect.stringContaining("Approve preprocessing")]));

    expect(consoleErrors).toEqual([]);
  });

  test("B: Schedule for training toggles Scheduled <-> Unschedule (step 11 <-> 12)", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.corrected);

    // step 11: the Schedule-for-training action is present and not yet scheduled.
    await expect(mainBtn(page, /Schedule for training/)).toBeVisible();
    expect(await canvasCount(page)).toBeGreaterThanOrEqual(1);

    // MUTATE: schedule this scan for training -> step 12 (green).
    await mainBtn(page, /Schedule for training/).click();

    // step 12: "Scheduled for training." confirmation + an Unschedule button appear.
    await expect(page.locator("main").getByText(/Scheduled/).first()).toBeVisible({ timeout: 15_000 });
    await expect(mainBtn(page, /Unschedule/)).toBeVisible();

    // MUTATE BACK: unschedule -> returns to step 11 with the Schedule button restored (leaves the fixture as found).
    await mainBtn(page, /Unschedule/).click();
    await expect(mainBtn(page, /Schedule for training/)).toBeVisible({ timeout: 15_000 });
    await expect(mainBtn(page, /Unschedule/)).toHaveCount(0);

    expect(consoleErrors).toEqual([]);
  });
});
