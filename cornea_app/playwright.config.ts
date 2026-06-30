import { defineConfig, devices } from "@playwright/test";

/* E2E suite for the Cornea OCT app. A single FastAPI sidecar serves both the built UI and the API on
 * a dedicated test port (8799) against an ISOLATED data dir (/tmp/cornea_pw_e2e) seeded with synthetic,
 * GPU-free fixtures (tests/e2e/_seed.py via global-setup). Never touches the user's real cases/ or their
 * running app. The app is a single stateful backend, so tests run SERIALLY (workers:1) for determinism.
 */
const PORT = Number(process.env.CORNEA_PW_PORT || 8799);
const DATA_DIR = process.env.CORNEA_PW_DATA || "/tmp/cornea_pw_e2e";

export default defineConfig({
  testDir: "./tests/e2e",
  globalSetup: "./tests/e2e/global-setup.ts",
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]],
  timeout: 60_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    headless: true,
    viewport: { width: 1600, height: 1000 },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    launchOptions: {
      // Software WebGL2 so the niivue viewers render headlessly without depending on a GPU.
      args: ["--enable-unsafe-swiftshader", "--use-gl=angle", "--use-angle=swiftshader", "--no-sandbox"],
    },
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    // Build the UI, then run ONLY the sidecar (it serves API + dist) on the test port + isolated data dir.
    command: `npm run build && cd python-sidecar && CORNEA_DATA_DIR=${DATA_DIR} CORNEA_API_TOKEN= python3 api_server.py --port ${PORT}`,
    url: `http://127.0.0.1:${PORT}/api/health`,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    stdout: "pipe",
    stderr: "pipe",
  },
});
