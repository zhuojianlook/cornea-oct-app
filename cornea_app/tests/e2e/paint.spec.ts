import { test, expect, gotoApp, openCase, mainButtons, mainBtn, FIX } from "./helpers";

/* The correction PAINT layer for FIX.scar: Correct -> paint toolbar -> Navigate disables the brush -> Cancel.
   Correct only loads the draw layer client-side (it does NOT persist until "Save correction"), so opening it
   and cancelling is read-only/safe in this serial suite.

   NOTE: the PaintToolbar (✏ Paint / ✋ Navigate / Cornea … ✨ Smart fill) renders INSIDE <main> (it's part
   of VolumeCanvas), but mainButtons() STRIPS a leading decorative glyph — so "✏ Paint" comes back as
   "Paint", "▣ Fill region" as "Fill region", etc. We therefore assert the pen tools with page-level
   getByRole/name (substring/RegExp) which match the full accessible label glyph-and-all. The Cornea /
   Background / Scar / Erase pens carry no leading glyph, so those DO survive mainButtons() unchanged. */

// Glyph-prefixed pen tools → assert by accessible-name substring (mainButtons strips the glyph).
const GLYPH_TOOLS: RegExp[] = [/Paint/, /Navigate/, /Fill region/, /Undo/, /Smart fill/];
// Glyphless pen labels → survive mainButtons() verbatim.
const PEN_LABELS = ["Cornea", "Background", "Scar", "Erase"];

test.describe("paint correction layer", () => {
  test("Correct opens the paint toolbar; Navigate disables the brush; Cancel tears it down", async ({ page, consoleErrors }) => {
    await gotoApp(page);
    await openCase(page, FIX.scar);

    // Enter correction mode (loads the draw layer; non-persisting until Save). Correct shows "Loading…"
    // while the paint layer loads, so wait for the toolbar's first button instead of the click resolving.
    await mainBtn(page, /Correct/).click();

    // The full pen toolbar appears (PaintToolbar mounts once `correcting` is on + the layer loaded).
    for (const re of GLYPH_TOOLS) {
      await expect(page.getByRole("button", { name: re }).first()).toBeVisible();
    }
    // The glyphless pens (Cornea/Background/Scar/Erase) survive mainButtons() verbatim.
    const btns = await mainButtons(page);
    expect(btns).toEqual(expect.arrayContaining(PEN_LABELS));
    // Save correction + Cancel show up in the action bar while correcting.
    expect(btns).toEqual(expect.arrayContaining(["Cancel"]));
    await expect(mainBtn(page, /Save correction/)).toBeVisible();

    // Regression (v0.0.92): switching to Navigate must disable the brush controls.
    await page.getByRole("button", { name: /Navigate/ }).first().click();

    // The "▣ Fill region" toggle is disabled in Navigate mode.
    const fillToggle = page.getByRole("button", { name: /Fill region/ }).first();
    await expect(fillToggle).toBeDisabled();

    // The brush Size slider is disabled in Navigate mode. The toolbar lays out the brush control as a flex
    // group <div>…<span>Size</span><MuiSlider/></div>; grab that group via its "Size" label, then the slider.
    const sizeGroup = page
      .locator("main")
      .locator("div:has(> span)", { has: page.getByText("Size", { exact: true }) })
      .first();
    await expect(sizeGroup).toContainText("Size");
    // MUI renders a hidden <input type="range"> that carries the real `disabled` attribute, plus a
    // role="slider" thumb; the root span picks up the Mui-disabled class. Assert disabled across these.
    const rangeInput = sizeGroup.locator('input[type="range"]');
    const sliderRoot = sizeGroup.locator('[class*="MuiSlider-root"]').first();
    if (await rangeInput.count()) {
      await expect(rangeInput.first()).toBeDisabled();
    } else {
      await expect(sliderRoot).toHaveClass(/Mui-disabled/);
    }

    // Cancel discards the correction and removes the paint toolbar.
    await mainBtn(page, /Cancel/).click();
    await expect(page.getByRole("button", { name: /Paint/ })).toHaveCount(0);
    await expect(mainBtn(page, /Save correction/)).toHaveCount(0);

    expect(consoleErrors).toEqual([]);
  });
});
