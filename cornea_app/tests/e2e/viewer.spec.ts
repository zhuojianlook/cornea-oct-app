import { test, expect, gotoApp, openCase, mainButtons, mainBtn, canvasCount, FIX } from "./helpers";

test.describe("viewer toolbar", () => {
  test("segmented case: views, Slices/Segmentation toggle, and display sliders are client-side and error-free", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.scar);

    // a rendered niivue viewer (Slices/Segmentation/overlay) has at least one <canvas>
    expect(await canvasCount(page)).toBeGreaterThanOrEqual(1);

    // the full viewer toolbar is present on a segmented case
    const btns = await mainButtons(page);
    expect(btns).toEqual(
      expect.arrayContaining(["Slices", "Segmentation", "Multi", "Axial", "Coronal", "Sagittal", "3D", "reset"]),
    );

    // Segmentation toggle is ENABLED for a case that has a segmentation; toggle it then back to Slices.
    const segToggle = mainBtn(page, "Segmentation");
    await expect(segToggle).toBeEnabled();
    await segToggle.click();
    await expect(mainBtn(page, "Slices")).toBeEnabled();
    await mainBtn(page, "Slices").click();

    // Cycle every view button — all client-side, safe to click anywhere.
    for (const view of ["Axial", "Coronal", "Sagittal", "Multi", "3D"]) {
      await mainBtn(page, view).click();
      // viewer should still hold its canvas after a view switch
      expect(await canvasCount(page)).toBeGreaterThanOrEqual(1);
    }

    // The three display sliders, located by their toolbar group titles (robust to other sliders).
    const contrast = page.locator('[title="Contrast (display only)"] input[type="range"]');
    const brightness = page.locator('[title="Brightness (display only)"] input[type="range"]');
    const blur = page.locator('[title="Gaussian blur (display only)"] input[type="range"]');
    await expect(contrast).toBeVisible();
    await expect(brightness).toBeVisible();
    await expect(blur).toBeVisible();

    // Drive Contrast/Brightness via fill and Blur via keyboard (display-only CSS filter, non-persisting).
    await contrast.fill("150");
    await expect(contrast).toHaveValue("150");
    await brightness.fill("80");
    await expect(brightness).toHaveValue("80");
    await blur.focus();
    await blur.press("ArrowRight");
    await expect(blur).not.toHaveValue("0");

    // reset restores the display adjustments to their defaults.
    await mainBtn(page, "reset").click();
    await expect(contrast).toHaveValue("100");
    await expect(brightness).toHaveValue("100");
    await expect(blur).toHaveValue("0");

    expect(consoleErrors).toEqual([]);
  });

  test("raw (un-preprocessed) case: the Segmentation toggle is disabled", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.raw);

    // Viewer toolbar is present even on a raw scan; the Segmentation overlay has nothing to show yet.
    await expect(mainBtn(page, "Slices")).toBeVisible();
    const segToggle = mainBtn(page, "Segmentation");
    await expect(segToggle).toBeVisible();
    await expect(segToggle).toBeDisabled();

    expect(consoleErrors).toEqual([]);
  });
});
