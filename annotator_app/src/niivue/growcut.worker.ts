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

interface InMsg { intensity: Uint8Array; seeds: Uint8Array; nx: number; ny: number; nz: number; spatial: number;
                  close?: boolean; closeLabel?: number; }

/* Morphological CLOSE (26-conn dilate then erode) + 3-D hole fill of one label, bounded to its bbox — so a
   magic-wand-derived scar comes back as a solid, closed shape (small gaps bridged, interior holes filled).
   Closing only ADDS voxels (close(A) ⊇ A), so it never deletes scar; padding the bbox by 2 keeps the erosion
   off the real shape. A guard skips the close when the scar bbox spans a large fraction of the volume (the
   degenerate no-background-seeds case where the grown scar isn't a shape to close) so cost stays bounded. */
function closeShapeInPlace(label: Uint8Array, nx: number, ny: number, nz: number, target: number): void {
  const nxny = nx * ny, N = nx * ny * nz;
  let x0 = nx, y0 = ny, z0 = nz, x1 = -1, y1 = -1, z1 = -1;
  for (let v = 0; v < N; v++) {
    if (label[v] !== target) continue;
    const z = (v / nxny) | 0, r = v - z * nxny, y = (r / nx) | 0, x = r - y * nx;
    if (x < x0) x0 = x; if (x > x1) x1 = x;
    if (y < y0) y0 = y; if (y > y1) y1 = y;
    if (z < z0) z0 = z; if (z > z1) z1 = z;
  }
  if (x1 < 0) return;                                   // no target voxels
  const pad = 2;
  x0 = Math.max(0, x0 - pad); y0 = Math.max(0, y0 - pad); z0 = Math.max(0, z0 - pad);
  x1 = Math.min(nx - 1, x1 + pad); y1 = Math.min(ny - 1, y1 + pad); z1 = Math.min(nz - 1, z1 + pad);
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, bd = z1 - z0 + 1, bwh = bw * bh, bn = bwh * bd;
  // GUARD: a real corneal scar is a compact sub-region, so its bbox is small. If the grown scar spans a
  // large fraction of the volume (the degenerate "wand a scar, then Smart Fill with NO background seeds" →
  // scar becomes the geodesically-nearest seed almost everywhere), the result isn't a shape to close and
  // closing it would allocate hundreds of MB. Skip in that case (closing whole-volume scar is meaningless).
  if (bn > Math.min(12_000_000, N * 0.45)) return;
  const li = (x: number, y: number, z: number) => (z - z0) * bwh + (y - y0) * bw + (x - x0);
  const m = new Uint8Array(bn);
  for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++)
    if (label[z * nxny + y * nx + x] === target) m[li(x, y, z)] = 1;

  const morph = (src: Uint8Array, dilate: boolean): Uint8Array => {
    const out = new Uint8Array(bn);
    for (let z = 0; z < bd; z++) for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++) {
      let any = false, all = true;
      for (let dz = -1; dz <= 1; dz++) { const zz = z + dz;
        for (let dy = -1; dy <= 1; dy++) { const yy = y + dy;
          for (let dx = -1; dx <= 1; dx++) { const xx = x + dx;
            const inside = xx >= 0 && xx < bw && yy >= 0 && yy < bh && zz >= 0 && zz < bd;
            const mv = inside ? src[zz * bwh + yy * bw + xx] : 0;
            if (mv) any = true; else all = false;
          } } }
      out[z * bwh + y * bw + x] = dilate ? (any ? 1 : 0) : (all ? 1 : 0);
    }
    return out;
  };
  const closed = morph(morph(m, true), false);          // dilate → erode = close

  // hole fill: flood background (closed===0) inward from the box border; any background NOT reached = hole.
  const bg = new Uint8Array(bn);
  let stack = new Int32Array(Math.max(1024, bn >> 4)), sp = 0;   // typed, grow-on-demand (not a number[])
  const tryPush = (i: number) => {
    if (closed[i] || bg[i]) return;
    bg[i] = 1;
    if (sp === stack.length) { const n = new Int32Array(stack.length * 2); n.set(stack); stack = n; }
    stack[sp++] = i;
  };
  for (let z = 0; z < bd; z++) for (let y = 0; y < bh; y++) for (let x = 0; x < bw; x++)
    if (x === 0 || x === bw - 1 || y === 0 || y === bh - 1 || z === 0 || z === bd - 1) tryPush(z * bwh + y * bw + x);
  while (sp > 0) {
    const i = stack[--sp]; const z = (i / bwh) | 0, r = i - z * bwh, y = (r / bw) | 0, x = r - y * bw;
    if (x > 0) tryPush(i - 1); if (x < bw - 1) tryPush(i + 1);
    if (y > 0) tryPush(i - bw); if (y < bh - 1) tryPush(i + bw);
    if (z > 0) tryPush(i - bwh); if (z < bd - 1) tryPush(i + bwh);
  }
  for (let z = z0; z <= z1; z++) for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    const b = li(x, y, z);
    if (closed[b] || !bg[b]) label[z * nxny + y * nx + x] = target;   // closed shape + interior holes → target
  }
}

self.onmessage = (e: MessageEvent<InMsg>) => {
  const { intensity, seeds, nx, ny, nz, spatial, close, closeLabel } = e.data;
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
  if (close && (closeLabel === 1 || closeLabel === 2)) closeShapeInPlace(label, nx, ny, nz, closeLabel);
  (self as unknown as Worker).postMessage({ type: "progress", pct: 100 });
  (self as unknown as Worker).postMessage({ type: "done", label }, [label.buffer]);
};
