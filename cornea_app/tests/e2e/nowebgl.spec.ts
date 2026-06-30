import { test, expect, gotoApp, openCase, mainBtn, FIX } from "./helpers";

/* The deployed/VS-Code path may have no WebGL2; VolumeCanvas then falls back to the 2-D SliceGallery.
 * Force WebGL unavailable and confirm the shell + a segmented case still render (no crash) via that path.
 * (No console-error assertion here: niivue legitimately logs a WebGL-unavailable error on this path.) */
test.describe("no-WebGL fallback", () => {
  test("app + a segmented case render via the 2-D gallery when WebGL is unavailable", async ({ page }) => {
    await page.addInitScript(() => {
      const orig = HTMLCanvasElement.prototype.getContext;
      // @ts-expect-error — narrow override to knock out only WebGL contexts
      HTMLCanvasElement.prototype.getContext = function (type: string, ...args: unknown[]) {
        return type === "webgl" || type === "webgl2" || type === "experimental-webgl"
          ? null
          : (orig as (t: string, ...a: unknown[]) => unknown).call(this, type, ...args);
      };
    });

    await gotoApp(page);
    await openCase(page, FIX.scar);

    // The 2-D fallback shows preview images (no niivue <canvas>), and the viewer toolbar still works.
    await expect(page.locator("main img").first()).toBeVisible({ timeout: 15_000 });
    await expect(mainBtn(page, "Slices")).toBeVisible();
    // No hard page crash: the app shell is still mounted.
    await expect(page.getByText("OCT preprocessing")).toBeVisible();
  });
});
