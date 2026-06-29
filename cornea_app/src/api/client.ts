/* ──────────────────────────────────────────────────────────
   API client for the FastAPI sidecar (python-sidecar/api_server.py).
   Uses the Tauri IPC proxy (invoke "proxy_request") when running inside
   the native shell; falls back to browser fetch in dev / browser-first mode.
   Mirrors the multipanelfigure app's client.ts.
   ────────────────────────────────────────────────────────── */

import type { AppConfig } from "./types";

// The sidecar base URL. In the Tauri shell the shell picks the port at startup (8765 if free, else a
// free one — so it always owns its OWN sidecar, never a stale/foreign one) and exposes it via
// get_sidecar_base. We resolve it once here so DIRECT resource fetches (resourceUrl → niivue) and the
// browser-fetch fallback hit the right sidecar. JSON/upload calls in the shell go via proxy_request,
// which reads the same port shell-side.
let _base = "http://127.0.0.1:8765";
let _basePromise: Promise<void> | null = null;
export function initSidecarBase(): Promise<void> {
  if (!_basePromise) {
    _basePromise = (async () => {
      const invoke = await getInvoke();
      if (invoke) {
        try {
          const b = (await invoke("get_sidecar_base", {})) as string;
          if (b) _base = b;
        } catch { /* older shell without the command → keep the default :8765 */ }
      }
    })();
  }
  return _basePromise;
}

let _invoke: ((cmd: string, args: Record<string, unknown>) => Promise<unknown>) | null = null;
let _invokeReady = false;

async function getInvoke() {
  if (_invokeReady) return _invoke;
  const inTauri =
    typeof window !== "undefined" &&
    ("__TAURI_INTERNALS__" in window || "__TAURI__" in window || "__TAURI_IPC__" in window);
  if (!inTauri) {
    _invoke = null;
    _invokeReady = true;
    return _invoke;
  }
  try {
    const mod = await import("@tauri-apps/api/core");
    _invoke = mod.invoke;
  } catch {
    _invoke = null;
  }
  _invokeReady = true;
  return _invoke;
}

async function apiRequest(path: string, method = "GET", body?: string): Promise<string> {
  const invoke = await getInvoke();
  if (invoke) {
    return invoke("proxy_request", { method, path, body: body ?? null }) as Promise<string>;
  }
  const res = await fetch(`${_base}${path}`, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body,
  });
  return res.text();
}

async function apiJson<T>(path: string, method = "GET", body?: string): Promise<T> {
  const text = await apiRequest(path, method, body);
  const parsed = JSON.parse(text);
  if (parsed && parsed.detail) {
    throw new Error(`API error: ${JSON.stringify(parsed.detail)}`);
  }
  return parsed as T;
}

// Base64-encode bytes in CHUNKS. `btoa(String.fromCharCode(...bytes))` spreads the WHOLE array as function
// args, which overflows the call stack ("Maximum call stack size exceeded") for anything more than ~100 KB —
// e.g. an edited segmentation drawing (a full-volume NIfTI). That silently broke every drawing upload through
// the Tauri proxy (Confirm cornea/background, Save correction) for large/dense drawings — the encode threw
// before the request was sent, so no flag was ever written. Chunking keeps each spread small.
function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  const CHUNK = 0x8000; // 32 KB per spread
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK) as unknown as number[]);
  }
  return btoa(binary);
}

/** Upload one or more files to a sidecar endpoint (multipart, field "files"). */
async function apiUpload<T>(path: string, files: File[]): Promise<T> {
  const invoke = await getInvoke();
  let text: string;
  if (invoke) {
    const payload = await Promise.all(
      files.map(async (f) => ({
        name: f.name,
        data: bytesToBase64(new Uint8Array(await f.arrayBuffer())),
      })),
    );
    text = (await invoke("proxy_upload", {
      path,
      files: payload,
      fieldName: "files",
    })) as string;
  } else {
    const form = new FormData();
    for (const f of files) form.append("files", f);
    const res = await fetch(`${_base}${path}`, { method: "POST", body: form });
    text = await res.text();
    if (!res.ok) {
      let detail = text;
      try {
        detail = JSON.stringify(JSON.parse(text).detail ?? text);
      } catch {
        /* non-JSON error body — keep the raw text */
      }
      throw new Error(`API error (${res.status}): ${detail}`);
    }
  }
  // Surface a FastAPI {detail: …} error instead of returning it as if it were data.
  const parsed = JSON.parse(text);
  if (parsed && parsed.detail) {
    throw new Error(`API error: ${JSON.stringify(parsed.detail)}`);
  }
  return parsed as T;
}

/** Absolute URL for a binary sidecar resource (volume / segmentation NIfTI). */
export function resourceUrl(path: string): string {
  return `${_base}${path}`;
}

export let lastHealthError = "";

export async function checkHealth(): Promise<boolean> {
  try {
    await initSidecarBase(); // resolve the real sidecar port before any direct resource fetch
    const text = await apiRequest("/api/health");
    const data = JSON.parse(text);
    if (data.status === "ok") {
      lastHealthError = "";
      return true;
    }
    lastHealthError = `Health response: ${text.substring(0, 200)}`;
    return false;
  } catch (e) {
    lastHealthError = e instanceof Error ? e.message : String(e);
    return false;
  }
}

export const api = {
  getConfig: () => apiJson<AppConfig>("/api/config"),
  putConfig: (patch: Partial<AppConfig>) =>
    apiJson<AppConfig>("/api/config", "PUT", JSON.stringify(patch)),

  request: apiRequest,
  json: apiJson,
  upload: apiUpload,
};

// Resolve the sidecar base as early as possible (Tauri injects its internals before app JS runs),
// so resourceUrl is correct by the time the first volume/preview loads.
void initSidecarBase();
