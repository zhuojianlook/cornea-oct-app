import { test, expect, FIX, gotoApp, openCase } from "./helpers";
import type { Page, Route } from "@playwright/test";

/* STALE-PIPELINE AUTO RE-RUN ON OPEN (reviewer 2026-09-07: "make sure that scans that are open through the
 * approve queue open through the new pipeline").
 *
 * The sidecar stamps every run with oct_preprocess.PIPELINE_VERSION and reports `pipeline_current` on the
 * case-open response. When a scan opens with pipeline_current:false AND it is not approved, caseStore.openCase
 * dispatches the exact call the "↻ Re-run with corrections" button makes — once per open — with a visible notice.
 *
 * Every open path (launch, sidebar row, Approve/Skip → next, the open-by-id box) funnels through openCase, so the
 * open-by-id box exercises the same code the queue does. The run itself is STUBBED at the network layer: the
 * synthetic fixtures have no .OCT source, so the real endpoint would 400, and what this spec asserts is the
 * dispatch (which endpoint, which body, how many times), not the pipeline.
 */

const RERUN_BODY = { use_redetect: true, corrected_edit_feedback: false, corrected_smooth_align: false,
                     corrected_trusted_laterals: null };

/** Stub the re-run's endpoints for `cid` and collect every oct-preprocess POST body. */
async function stubRerun(page: Page, cid: string): Promise<string[]> {
  const posts: string[] = [];
  const stub = async (route: Route) => {
    const req = route.request();
    if (req.method() !== "POST") return route.continue();
    if (/\/oct-preprocess$/.test(req.url())) posts.push(req.postData() ?? "");
    await route.fulfill({ status: 200, contentType: "application/json", body: "{}" });
  };
  for (const ep of ["oct-preprocess", "oct-border-generalize", "oct-border-guided", "oct-corrected-curve"]) {
    await page.route(`**/api/case/${cid}/${ep}`, stub);
  }
  return posts;
}

/** Every oct-preprocess POST the page makes, for any case (the negative tests). */
function trackPreprocessPosts(page: Page): string[] {
  const urls: string[] = [];
  page.on("request", (r) => { if (r.method() === "POST" && /\/api\/case\/[^/]+\/oct-preprocess$/.test(r.url())) urls.push(r.url()); });
  return urls;
}

test.describe("stale-pipeline auto re-run on open", () => {
  test("an unapproved scan whose run is not current is re-run once, with the same call as ↻ Re-run with corrections",
       async ({ page }) => {
    // Routes BEFORE the app loads: the launch reopen of default_case_id may itself be this case.
    const posts = await stubRerun(page, FIX.stale);
    const all = trackPreprocessPosts(page);
    await gotoApp(page);
    await openCase(page, FIX.stale);

    // The notice explains why the scan is running (rendered in the review row; the global status line carries it too).
    const note = page.getByTestId("auto-rerun-note");
    await expect(note).toBeVisible({ timeout: 15_000 });
    await expect(note).toHaveAttribute("title", /older pipeline/i);

    // Exactly the button's request, exactly once.
    await expect.poll(() => posts.length, { timeout: 15_000 }).toBe(1);
    expect(JSON.parse(posts[0])).toEqual(RERUN_BODY);

    // The run reloads the case (still unstamped on the fixture, so the sidecar still says stale) — the once-per-open
    // guard must keep it from firing again, and the note settles on the outcome.
    await expect(note).toHaveAttribute("title", /re-ran this scan through the current pipeline/i, { timeout: 15_000 });
    await page.waitForTimeout(1500);
    expect(posts.length).toBe(1);
    expect(all.filter((u) => !u.includes(`/${FIX.stale}/`))).toEqual([]);   // no other case was touched

    // Leave the persisted default_case_id on a quiet (current) case so the next spec's launch reopen is silent.
    await openCase(page, FIX.auto);
  });

  test("approved scans and current runs are never auto re-run", async ({ page, consoleErrors }) => {
    const all = trackPreprocessPosts(page);
    await gotoApp(page);
    // Approved + no run record at all (stale by definition) → untouched: approval wins.
    await openCase(page, FIX.classified);
    await page.waitForTimeout(800);
    expect(all).toEqual([]);
    await expect(page.getByTestId("auto-rerun-note")).toHaveCount(0);
    // Unapproved but stamped with the current PIPELINE_VERSION by _seed.py → nothing to do.
    await openCase(page, FIX.auto);
    await page.waitForTimeout(800);
    expect(all).toEqual([]);
    await expect(page.getByTestId("auto-rerun-note")).toHaveCount(0);
    expect(consoleErrors).toEqual([]);
  });
});
