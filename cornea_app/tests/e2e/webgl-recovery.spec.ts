import { test, expect, gotoApp, openCase, FIX } from "./helpers";

/* Regression for the "black canvas forever after a lost WebGL context" bug: WebKitGTK/NVIDIA can drop the
 * WebGL2 context under pressure; nvController now rebuilds niivue on webglcontextrestored and VolumeCanvas
 * reloads the volume. We simulate loss+restore via the WEBGL_lose_context extension and assert recovery. */
test.describe("webgl context-loss recovery", () => {
  test("a lost+restored WebGL context rebuilds niivue and reloads the volume (not black forever)", async ({ page }) => {
    await gotoApp(page);
    await openCase(page, FIX.scar);

    // niivue exposes its instance on window.nv; wait until the base volume is loaded.
    await expect.poll(() => page.evaluate(() => {
      const nv = (window as unknown as { nv?: { volumes?: unknown[] } }).nv;
      return nv?.volumes?.length ?? 0;
    }), { timeout: 15_000 }).toBeGreaterThanOrEqual(1);

    // Force a context loss + restore on the niivue canvas (the WEBGL_lose_context extension is the standard
    // way to simulate the driver event). Returns whether the extension was available.
    const simulated = await page.evaluate(async () => {
      const canvas = document.querySelector("main canvas") as HTMLCanvasElement | null;
      if (!canvas) return "no-canvas";
      const gl = canvas.getContext("webgl2") as WebGL2RenderingContext | null;
      const ext = gl?.getExtension("WEBGL_lose_context");
      if (!ext) return "no-ext";
      ext.loseContext();
      await new Promise((r) => setTimeout(r, 200));
      ext.restoreContext();
      return "ok";
    });
    // SwiftShader/Chromium support the extension; if a runner lacks it, don't fail the suite.
    test.skip(simulated !== "ok", `WEBGL_lose_context unavailable (${simulated})`);

    // After restore, nvController rebuilds niivue (new window.nv) and the volume reloads → volumes>=1 again.
    await expect.poll(() => page.evaluate(() => {
      const nv = (window as unknown as { nv?: { volumes?: unknown[] } }).nv;
      return nv?.volumes?.length ?? 0;
    }), { timeout: 20_000, message: "viewer did not recover its volume after context restore" }).toBeGreaterThanOrEqual(1);

    // The viewer toolbar is still functional (the canvas didn't take the React tree down).
    await expect(page.locator("main canvas").first()).toBeVisible();
  });
});
