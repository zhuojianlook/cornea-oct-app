/* Headless UI smoke test (no user data needed): loads the app in the no-WebGL path
   (forces the 2D SliceGallery, matching the VS Code Simple Browser), and asserts the
   shell renders blank with the loader panels present and no console/server errors.
   Run the dev servers first (dev-launch.sh), then: node test/ui-smoke.cjs            */
const puppeteer = require("puppeteer-core");
const CHROME = process.env.CHROME || "/usr/bin/google-chrome";
const URL = process.env.APP_URL || "http://localhost:1420/";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const pass = [], fail = [];
const check = (n, ok, d) => (ok ? pass : fail).push(n + (d ? ` — ${d}` : ""));

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
    args: ["--no-sandbox", "--disable-setuid-sandbox", "--disable-gpu"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 950 });
  // Force no-WebGL2 so the 2D gallery path is exercised (the deployed environment).
  await page.evaluateOnNewDocument(() => {
    const o = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (t, ...a) { return (t === "webgl2" || t === "webgl") ? null : o.call(this, t, ...a); };
  });
  const errs = [];
  page.on("pageerror", (e) => errs.push("PAGEERROR: " + e.message));
  page.on("response", (r) => { if (r.status() >= 500) errs.push(`HTTP ${r.status()} ${r.url()}`); });

  await page.goto(URL, { waitUntil: "networkidle2", timeout: 30000 });
  await sleep(1500);
  const body = await page.evaluate(() => document.body.innerText);
  const imgLen = await page.evaluate(() => { const a = [...document.querySelectorAll("img")].filter((i) => i.src.startsWith("data:image")); return a[0] ? a[0].src.length : 0; });

  check("app-loads", /Cornea OCT/i.test(body));
  check("blank-on-load", imgLen === 0, "no slice image until a case is loaded");
  check("oct-loader-present", /OCT preprocessing/i.test(body));
  check("cohort-panel-present", /Cohort batch/i.test(body));
  check("no-page-or-server-errors", errs.length === 0, errs.slice(0, 3).join(" | "));

  console.log("PASS:", pass.length); pass.forEach((p) => console.log("  ✓ " + p));
  console.log("FAIL:", fail.length); fail.forEach((f) => console.log("  ✗ " + f));
  await browser.close();
  process.exit(fail.length ? 1 : 0);
})().catch((e) => { console.error("CRASH:", e.message); process.exit(2); });
