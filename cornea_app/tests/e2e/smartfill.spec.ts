import { test, expect, gotoApp, openCase, mainBtn, FIX } from "./helpers";

/* Regression: user-painted BACKGROUND must be an absolute barrier that smart fill never re-grows.
 * Cornea-vet floods unlabelled → AUTO_BG (label 4, re-growable on the rim); the user's background pen writes
 * label 2 which must survive smart fill. We enter cornea-vet on a segmented fixture, paint a band of cornea
 * as background (label 2) via drawBitmap, run smart fill, and assert none of it reverted to cornea. */
test.describe("smart fill — user background is an absolute barrier", () => {
  test("painting a band as background sticks through smart fill (no re-grow)", async ({ page }) => {
    await gotoApp(page);
    await openCase(page, FIX.cornea);   // step 5: cornea segmented, awaiting vet

    await mainBtn(page, /Paint cornea\/background/).click();
    // wait for the draw layer + the auto-background flood (label 4)
    await expect.poll(() => page.evaluate(() => {
      const d = (window as any).nv?.drawBitmap; if (!d) return -1;
      let four = 0; for (let i = 0; i < d.length; i++) if (d[i] === 4) four++; return four;
    }), { timeout: 15_000 }).toBeGreaterThan(0);

    // paint a band of current cornea (label 1) → user background (label 2); record those voxels.
    const painted = await page.evaluate(() => {
      const nv = (window as any).nv; const d = nv.drawBitmap;
      const dr = nv.volumes[0].dimsRAS; const nx = dr[1], nxny = nx * dr[2];
      (window as any).__p = new Uint8Array(d.length); let n = 0;
      const lo = Math.floor(nx * 0.45), hi = Math.floor(nx * 0.55);
      for (let i = 0; i < d.length; i++) if (d[i] === 1) { const x = (i % nxny) % nx; if (x >= lo && x <= hi) { d[i] = 2; (window as any).__p[i] = 1; n++; } }
      try { nv.refreshDrawing(true); } catch { /* */ }
      return n;
    });
    expect(painted).toBeGreaterThan(0);

    await mainBtn(page, /Smart fill/).click();
    await expect.poll(() => page.evaluate(() =>
      /Filling…|Growing labels/.test(document.body.innerText) ? 1 : 0), { timeout: 60_000 }).toBe(0);

    // none of the user-painted-background voxels may have reverted to cornea (label 1).
    const reverted = await page.evaluate(() => {
      const d = (window as any).nv.drawBitmap, p = (window as any).__p; let r = 0;
      for (let i = 0; i < d.length; i++) if (p[i] && d[i] === 1) r++; return r;
    });
    expect(reverted).toBe(0);
  });
});
