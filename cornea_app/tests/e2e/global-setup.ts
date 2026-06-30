import { execFileSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));

/* Seed the isolated test data dir with synthetic, GPU-free fixtures before the suite runs.
 * Idempotent (wipes + rebuilds the zz* cases). The sidecar reads case folders live per request,
 * so this is safe regardless of whether it runs before or after the webServer starts. */
export default function globalSetup() {
  const dataDir = process.env.CORNEA_PW_DATA || "/tmp/cornea_pw_e2e";
  const seed = path.join(HERE, "_seed.py");
  const out = execFileSync("python3", [seed], {
    env: { ...process.env, CORNEA_DATA_DIR: dataDir, CORNEA_API_TOKEN: "" },
    encoding: "utf8",
  });
  process.stdout.write(out);
}
