import { test, expect, gotoApp, openCase, mainButtons, mainBtn, canvasCount, FIX } from "./helpers";

test.describe("step 2 — preprocessing approval + display modes", () => {
  test("auto case shows the step-2 action bar + a rendered viewer with no console errors", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.auto);

    // step-2 action bar buttons
    const btns = await mainButtons(page);
    expect(btns).toEqual(
      expect.arrayContaining([
        "Approve preprocessing",
        "Re-run preprocessing",
        "Use original (raw)",
        "Before/after",
        "Fix columns",
        "Steps",
      ])
    );

    // a niivue viewer is rendered
    expect(await canvasCount(page)).toBeGreaterThanOrEqual(1);

    // CRITICAL regression (v0.0.94): opening an unsegmented step-2 case must NOT
    // log a segmentation-display.nii.gz 404 (or any other console error).
    expect(consoleErrors).toEqual([]);
  });

  test('"Before/after" mode shows Raw + Corrected labels and a refinement-pass control', async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.auto);

    await mainBtn(page, /Before\/after/).click();

    // Both before/after panel labels appear once BeforeAfterViewer renders (the previews load a beat
    // after the toggle). The component labels the left/right panels "original (raw)" and the corrected
    // pass label ("preprocessed" for a single-pass scan) — NOT the bare "Raw"/"Corrected" of the
    // timeline steps — so match those panel labels by substring.
    const main = page.locator("main");
    await expect(main.getByText(/original \(raw\)/i).first()).toBeVisible();
    await expect(main.getByText(/preprocessed/i).first()).toBeVisible();
    // the refinement-pass selector appears
    await expect(main.getByText(/refinement pass/i).first()).toBeVisible();

    expect(consoleErrors).toEqual([]);
  });

  test('"Fix columns" mode exposes the border-edit controls', async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.auto);

    // Track any failing (>=400) responses so we can prove the only failure is the EXPECTED fix-columns
    // border request below (the consoleErrors fixture only records >=500, so 400s are tracked here).
    const failed: string[] = [];
    page.on("response", (r) => { if (r.status() >= 400) failed.push(`${r.status()} ${r.url()}`); });

    await mainBtn(page, /Fix columns/).click();

    // border-edit controls: the mode toggles (Edge / Parabola / ✂ Cut / ✛ Surface crop / ⊟ Crop region)
    // + the local re-detect actions (Confirm border, Run preprocessing). The Confirm/Run buttons render
    // in the toolbar from the start but are DISABLED until anchors are dragged + confirmed, so assert they
    // are PRESENT (toBeVisible passes for a rendered-but-disabled button) — entering fix-columns mode is
    // what we're verifying, not running a re-detect.
    const main = page.locator("main");
    await expect(main.getByRole("button", { name: "Edge", exact: true })).toBeVisible();
    await expect(main.getByRole("button", { name: "Parabola", exact: true })).toBeVisible();
    await expect(mainBtn(page, /Cut/)).toBeVisible();
    await expect(mainBtn(page, /Confirm border/)).toBeVisible();
    await expect(mainBtn(page, /Run preprocessing/)).toBeVisible();

    // The synthetic fixture has no real .OCT working volume, so entering fix-columns fires
    // POST oct-border-curves-all → the app CORRECTLY returns 400 ("Case … has no working volume"),
    // which surfaces as a benign browser "Failed to load resource … 400" console line (the message text
    // carries no URL). Prove via the network that EVERY >=400 response is that one expected request, then
    // tolerate the matching console line; any other console error still fails the test.
    const benign400 = /Failed to load resource: the server responded with a status of 400/;
    expect(failed.every((f) => /oct-border-curves-all/.test(f))).toBe(true);
    expect(failed.length).toBeGreaterThan(0); // the expected 400 really did happen (tolerance is live)
    const unexpected = consoleErrors.filter((e) => !benign400.test(e));
    expect(unexpected).toEqual([]);
  });

  test('"Steps" opens an inline preprocessing-steps panel that can be closed', async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.auto);

    // Track failing (>=400) responses (the consoleErrors fixture only records >=500) so the expected
    // Steps 400 below is verified, not blindly tolerated.
    const failed: string[] = [];
    page.on("response", (r) => { if (r.status() >= 400) failed.push(`${r.status()} ${r.url()}`); });

    await mainBtn(page, /Steps/).click();

    // The Steps surface is an INLINE panel (StepsViewer mounted in VolumeCanvas), NOT a role=dialog/modal.
    // Its defining header is "Preprocessing decision tree …" plus a "← 3D view" close control.
    const main = page.locator("main");
    await expect(main.getByText(/preprocessing decision tree/i).first()).toBeVisible();
    const close = main.getByRole("button", { name: /3D view/i });
    await expect(close).toBeVisible();

    // Close it → the panel tears down (its header is gone).
    await close.click();
    await expect(main.getByText(/preprocessing decision tree/i)).toHaveCount(0, { timeout: 5_000 });

    // The synthetic fixture has no real .OCT source, so the panel's POST oct-preprocess-steps CORRECTLY
    // returns 400 (the panel shows "Couldn't render steps … no .OCT source") → a benign "Failed to load
    // resource … 400" console line. Confirm via the network that every >=400 is that one request, then
    // tolerate the matching console line; any other console error still fails the test.
    const benign400 = /Failed to load resource: the server responded with a status of 400/;
    expect(failed.every((f) => /oct-preprocess-steps/.test(f))).toBe(true);
    const unexpected = consoleErrors.filter((e) => !benign400.test(e));
    expect(unexpected).toEqual([]);
  });
});
