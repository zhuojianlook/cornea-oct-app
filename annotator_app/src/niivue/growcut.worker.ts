/* CPU "Grow from seeds" (Slicer-style) — replaces niivue's GPU drawGrowCut, which is unusably slow and
   broken in WebKitGTK / software-WebGL (and even failed on a discrete GPU here). Runs in a Web Worker so
   the UI stays responsive and can show progress.

   Algorithm: multi-source geodesic flood. Every unlabelled voxel is assigned the label of the seed it is
   geodesically closest to, where the step cost between adjacent voxels is |Δintensity| (+ a small spatial
   term so flat regions don't let one label run away). This is a Dijkstra shortest-path from all seeds at
   once. Because step costs are small bounded integers, we use a circular bucket priority queue (Dial's
   algorithm) → O(N) time, bounded memory, processing voxels in non-decreasing cost order. 6-connectivity.

   Message in:  { intensity: Uint8Array, seeds: Uint8Array, nx, ny, nz, spatial }
   Message out: { type:'progress', pct } …  then  { type:'done', label: Uint8Array } (label buffer transferred) */

interface InMsg { intensity: Uint8Array; seeds: Uint8Array; nx: number; ny: number; nz: number; spatial: number; }

self.onmessage = (e: MessageEvent<InMsg>) => {
  const { intensity, seeds, nx, ny, nz, spatial } = e.data;
  const N = nx * ny * nz;
  const nxny = nx * ny;
  const INF = 0x7fffffff;
  const dist = new Int32Array(N).fill(INF);
  const label = new Uint8Array(N);
  const sp = Math.max(0, spatial | 0);
  const maxEdge = 255 + sp;        // max cost of a single step (|Δ(uint8)| ≤ 255)
  const nb = maxEdge + 1;          // circular bucket window size
  const buckets: Int32Array[] = []; // grow-on-demand stacks per bucket
  const bcount = new Int32Array(nb);
  const bcap = new Int32Array(nb);
  for (let i = 0; i < nb; i++) { buckets.push(new Int32Array(1024)); bcap[i] = 1024; }
  const push = (b: number, v: number) => {
    if (bcount[b] === bcap[b]) { const n = new Int32Array(bcap[b] * 2); n.set(buckets[b]); buckets[b] = n; bcap[b] *= 2; }
    buckets[b][bcount[b]++] = v;
  };

  let queueSize = 0;
  for (let i = 0; i < N; i++) {
    const s = seeds[i];
    if (s !== 0) { dist[i] = 0; label[i] = s; push(0, i); queueSize++; }
  }
  if (queueSize === 0) { (self as unknown as Worker).postMessage({ type: "done", label }, [label.buffer]); return; }

  let cur = 0, processed = 0, lastPct = -1;
  while (queueSize > 0) {
    let b = cur % nb;
    while (bcount[b] === 0) { cur++; b = cur % nb; }  // advance to next non-empty cost bucket
    const v = buckets[b][--bcount[b]];
    queueSize--;
    if (dist[v] !== cur) continue; // stale (superseded) entry
    processed++;
    if ((processed & 0x3ffff) === 0) {                 // ~every 260k voxels
      const pct = (processed * 100 / N) | 0;
      if (pct !== lastPct) { lastPct = pct; (self as unknown as Worker).postMessage({ type: "progress", pct }); }
    }
    const iv = intensity[v], lv = label[v], dv = cur;
    const z = (v / nxny) | 0, rem = v - z * nxny, y = (rem / nx) | 0, x = rem - y * nx;
    // 6-connected neighbours
    if (x > 0)      { const w = v - 1;    const nd = dv + Math.abs(iv - intensity[w]) + sp; if (nd < dist[w]) { dist[w] = nd; label[w] = lv; push(nd % nb, w); queueSize++; } }
    if (x < nx - 1) { const w = v + 1;    const nd = dv + Math.abs(iv - intensity[w]) + sp; if (nd < dist[w]) { dist[w] = nd; label[w] = lv; push(nd % nb, w); queueSize++; } }
    if (y > 0)      { const w = v - nx;   const nd = dv + Math.abs(iv - intensity[w]) + sp; if (nd < dist[w]) { dist[w] = nd; label[w] = lv; push(nd % nb, w); queueSize++; } }
    if (y < ny - 1) { const w = v + nx;   const nd = dv + Math.abs(iv - intensity[w]) + sp; if (nd < dist[w]) { dist[w] = nd; label[w] = lv; push(nd % nb, w); queueSize++; } }
    if (z > 0)      { const w = v - nxny; const nd = dv + Math.abs(iv - intensity[w]) + sp; if (nd < dist[w]) { dist[w] = nd; label[w] = lv; push(nd % nb, w); queueSize++; } }
    if (z < nz - 1) { const w = v + nxny; const nd = dv + Math.abs(iv - intensity[w]) + sp; if (nd < dist[w]) { dist[w] = nd; label[w] = lv; push(nd % nb, w); queueSize++; } }
  }
  (self as unknown as Worker).postMessage({ type: "progress", pct: 100 });
  (self as unknown as Worker).postMessage({ type: "done", label }, [label.buffer]);
};
