"""Eye-motion analysis from the detected corneal surface of a 3-D OCT scan.

The 3-D "3D Cornea" scan acquires B-scans along the SLOW axis sequentially, so the slow (frame) axis is a
TIME axis (~136 Hz on the Avanti). The detected corneal surface depth at a fixed lateral position, as a
function of frame, is therefore a depth-vs-time trace; once the smooth corneal shape is removed, the residual
is the patient's eye/head MOTION during the ~0.74 s scan. This module extracts that motion(t), its power
spectrum, candidate saccade/microsaccade spikes, and a dominant motion direction relative to the cornea.

Validated on real Avanti data (CS015): the per-frame global motion agrees across two independent detectors
(corr 0.66-0.99), even/odd lateral split-half corr 0.994-0.999 (coherent across the whole surface), and
cross-replicate corr ~=0 (real per-scan motion, not an instrument artifact), SNR ~17 above the noise floor.

KEY method facts (empirically established — see the feasibility study):
  * sub-pixel interpolation is a NO-OP here (the noise is GROSS mis-detection, not quantisation), so we use
    the FAST integer-argmax detector and instead REJECT OUTLIERS — this is the whole signal-recovery trick.
  * what's trustworthy: the trace shape + the dominant-frequency SET (stable across detectors/split-halves)
    in the resolvable band ~[1/T, frame_rate/2]; the scalar amplitude in um is shape-model dependent (~2x).
  * not accessible: <~1.35 Hz (drift/cardiac/respiration) is under the frequency resolution of 101 frames;
    ocular microtremor (70-90 Hz) is above Nyquist and aliases — never reported as a resolved line.
"""
from __future__ import annotations

import numpy as np

import oct_preprocess as octp

# Optovue Avanti / AngioVue line (A-scan) rate, Hz — the device spec; the .OCT companion .txt has no timing,
# so the time axis (hence all Hz) is derived from this. Editable per request (calibration).
DEFAULT_ASCAN_RATE_HZ = 70000.0

# Physiological eye-movement bands for labelling spectral peaks (Hz). Ranges are deliberately broad; each
# label is advisory. Order matters (first match wins for a peak's centre frequency).
_BANDS = [
    ("respiration*", 0.0, 0.5),       # * = below the usual frequency resolution of a 0.74 s scan
    ("cardiac/pulse*", 0.5, 1.6),
    ("drift / low", 1.6, 4.0),
    ("low tremor", 4.0, 12.0),
    ("tremor", 12.0, 30.0),
    ("high tremor", 30.0, 1e9),       # mostly aliased (OMT 70-90 Hz folds below Nyquist)
]


def _band_label(hz: float) -> str:
    for name, lo, hi in _BANDS:
        if lo <= hz < hi:
            return name
    return "?"


def extract_surface(sag: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Detected corneal-surface depth per (lateral slice, frame) via the FAST detector.
    `sag` = sagittal volume (lateral, depth, frames). Returns S (lateral, frames), float32 depth rows."""
    p = {**octp.DEFAULT_PARAMS, **(params or {})}
    sigma = float(p["sigma"]); max_jump = float(p["max_jump"]); mfs = int(p["median_filter_size"])
    n, _depth, F = sag.shape
    S = np.empty((n, F), dtype=np.float32)
    for i in range(n):
        raw = octp._detect_surface_gradient(np.ascontiguousarray(sag[i]).astype(np.float32), sigma)
        S[i] = octp._smooth_median(octp._correct_surface(raw, max_jump), mfs)
    return S


def separate_motion(S: np.ndarray, detrend_order: int = 2, outlier_px: float = 15.0):
    """Remove the smooth corneal shape (per-lateral low-order polynomial over frames) and recover the global
    per-frame axial MOTION robustly. Returns (motion[F] centred, residual R[lateral,F], inlier_mask[lateral,F]).

    The per-lateral detrend removes each A-line's smooth corneal cross-section in the FRAME direction; what
    survives is motion + detection noise. The motion is GLOBAL (same axial shift across all lateral at a given
    frame), so masking gross mis-detections (|resid - per-frame-median| > outlier_px) then taking the per-frame
    MEDIAN over lateral averages the noise down (~1/sqrt(N)) and leaves the coherent motion (this masking is
    the empirically-decisive step: SNR 2 -> 17)."""
    S = np.asarray(S, dtype=np.float64)
    n, F = S.shape
    xf = np.arange(F, dtype=np.float64)
    R = np.empty_like(S)
    deg = max(1, int(detrend_order))
    for i in range(n):
        R[i] = S[i] - np.polyval(np.polyfit(xf, S[i], deg), xf)
    med = np.median(R, axis=0)                                   # robust per-frame centre
    mask = np.abs(R - med[None, :]) < float(outlier_px)
    motion = np.empty(F)
    for f in range(F):
        col = R[mask[:, f], f]
        motion[f] = np.median(col) if col.size else med[f]
    motion -= np.median(motion)                                  # zero-centre the trace
    return motion, R, mask


def motion_spectrum(motion: np.ndarray, frame_rate: float, sinc_correct: bool = False,
                    frame_time: float | None = None):
    """One-sided Hann-windowed power spectrum of the (linearly-detrended) motion(t).
    Returns (freqs_hz[K], power[K]) with power[0]=0. Optionally divide out the intra-frame motion-blur boxcar
    (|sinc(f*frame_time)|^2, Wiener-regularised) — each frame integrates over frame_time as the fast axis sweeps."""
    F = len(motion)
    x = np.arange(F, dtype=np.float64)
    detr = motion - np.polyval(np.polyfit(x, motion, 1), x)      # drop any residual linear ramp
    w = np.hanning(F)
    spec = np.fft.rfft(detr * w)
    P = (np.abs(spec) ** 2) / (np.sum(w ** 2) or 1.0)
    freqs = np.fft.rfftfreq(F, d=1.0 / frame_rate)
    if P.size:
        P[0] = 0.0                                               # DC carries no motion info after centring
    if sinc_correct and frame_time:
        s = np.sinc(freqs * float(frame_time))                  # np.sinc(x) = sin(pi x)/(pi x)
        P = P / (s ** 2 + 0.05)                                  # regularised inverse (blur correction)
    return freqs, P


def find_peaks(freqs: np.ndarray, P: np.ndarray, df: float, top: int = 5):
    """Top-`top` local maxima of the spectrum, each as {hz, period_ms, power_frac, label, resolved}.
    resolved=False when the peak sits within ~1 bin of DC (below the frequency resolution of the record)."""
    if P.size < 3:
        return []
    tot = float(P.sum()) or 1.0
    loc = [k for k in range(1, len(P) - 1) if P[k] > P[k - 1] and P[k] >= P[k + 1]]
    if len(P) >= 2 and P[-1] > P[-2]:
        loc.append(len(P) - 1)
    loc.sort(key=lambda k: P[k], reverse=True)
    out = []
    for k in loc[:top]:
        hz = float(freqs[k])
        out.append({
            "hz": round(hz, 2),
            "period_ms": round(1000.0 / hz, 1) if hz > 0 else None,
            "power_frac": round(float(P[k]) / tot, 3),
            "label": _band_label(hz),
            "resolved": bool(hz > 1.5 * df),
        })
    return out


def detect_spikes(motion: np.ndarray, frame_rate: float, um_per_px: float, k_mad: float = 5.0):
    """Candidate saccade/microsaccade events = frames whose axial velocity is a robust outlier. Saccades are
    brief (1-2 frame) ballistic jumps, NOT a spectral line, so they're surfaced as time-domain events. Returns
    a list of {frame, t_ms, velocity_um_per_s}. (Advisory — a single-frame mis-detection can masquerade as a
    spike; flagged as 'candidate'.)"""
    if len(motion) < 4:
        return []
    vel = np.diff(motion.astype(np.float64)) * frame_rate * um_per_px   # um/s between successive frames
    med = np.median(vel)
    mad = np.median(np.abs(vel - med)) * 1.4826 or 1e-6
    thr = float(k_mad) * mad
    out = []
    for j in np.where(np.abs(vel - med) > thr)[0]:
        out.append({"frame": int(j + 1), "t_ms": round((j + 1) / frame_rate * 1000.0, 1),
                    "velocity_um_per_s": round(float(vel[j]), 1)})
    return out


def motion_direction(R: np.ndarray, mask: np.ndarray, lateral_spacing_mm: float, depth_spacing_mm: float,
                     S: np.ndarray):
    """One dominant motion vector relative to the cornea, from the residual field.

    Per frame f, fit R[:,f] ~ A(f) + G(f)*x (x = centred lateral index, inliers only):
      A(f) = piston  -> AXIAL motion (along the OCT beam / surface normal at the apex).
      G(f) = lateral tilt -> apparent in-plane slip along the B-scan (lateral) axis: a curved cornea sliding
             laterally changes apparent depth in proportion to the local surface slope, so the physical
             lateral shift ~= -G(f) / mean_lateral_slope.
    The dominant direction = principal axis (PCA) of the 2-D [axial_um(t), inplane_lat_um(t)] cloud; the angle
    from the surface normal + the lateral (nasal/temporal) sign give an apex-frame vector. Confidence = the
    temporal coherence of the tilt (lag-1 autocorrelation; white noise ~ 0).

    LIMITATION (reported to the user): only the LATERAL in-plane axis is sampled within an instant; the
    elevational (slow-axis) in-plane component is confounded with scan progression, so the en-face azimuth is
    along the lateral axis only — a true 2-D en-face direction needs an orthogonal scan."""
    R = np.asarray(R, dtype=np.float64)
    n, F = R.shape
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    A = np.empty(F); G = np.empty(F)
    for f in range(F):
        m = mask[:, f]
        xf, yf = (x[m], R[m, f]) if m.sum() >= 8 else (x, R[:, f])
        c = np.polyfit(xf, yf, 1)                       # [slope, intercept]
        G[f] = c[0]; A[f] = c[1]
    A = np.nan_to_num(A); G = np.nan_to_num(G)
    # static dome slope (px per lateral index) from the mean surface across lateral — for the slip conversion
    dome = np.polyfit(x, np.median(S.astype(np.float64), axis=1), 2)
    slope = np.nan_to_num(np.polyval(np.polyder(dome), x))
    # The slip conversion physical_lateral = -G/slope is only meaningful where the surface is curved enough to
    # turn a lateral shift into an apparent depth change. A near-flat surface (apex region, low-curvature scan,
    # or a failed detection) gives a slope of ~0 (float-dust), and dividing by it blows the lateral estimate up
    # to physically impossible values (~1e15 µm). NOTE: `float(x) or 1e-3` does NOT guard this — Python `or`
    # only substitutes for an exact 0.0, and ~1e-15 dust is truthy. Use a real floor + a curvature gate.
    med_slope = float(np.median(np.abs(slope)))
    CURVATURE_MIN = 0.05            # px per lateral index (~ a few µm/index) below which slip is untrustworthy
    INPLANE_MAX_UM = 300.0          # physiological clip: corneal in-plane slip within one scan is well under this
    inplane_reliable = med_slope >= CURVATURE_MIN
    mean_abs_slope = max(med_slope, CURVATURE_MIN)
    axial_um = (A - np.median(A)) * depth_spacing_mm * 1000.0
    if inplane_reliable:
        inplane_px = -(G - np.median(G)) / mean_abs_slope           # apparent lateral shift in lateral indices
        inplane_um = np.clip(inplane_px * lateral_spacing_mm * 1000.0, -INPLANE_MAX_UM, INPLANE_MAX_UM)
    else:
        inplane_um = np.zeros_like(axial_um)                        # low curvature -> in-plane slip not resolvable
    axial_rms = float(np.std(axial_um)); inplane_rms = float(np.std(inplane_um))
    # PCA of the 2-D motion cloud -> dominant direction (axial vs lateral) + variance explained
    M = np.vstack([axial_um, inplane_um]).T
    M = M - M.mean(0)
    try:
        _u, sv, vt = np.linalg.svd(M, full_matrices=False)
        dom = vt[0]                                                  # [axial, lateral] unit components
        var_explained = float(sv[0] ** 2 / (np.sum(sv ** 2) or 1.0))
    except Exception:  # noqa: BLE001
        dom = np.array([1.0, 0.0]); var_explained = 0.0
    # orient the dominant axis so its axial component is +; tilt from the normal toward the lateral axis
    if dom[0] < 0:
        dom = -dom
    tilt_deg = float(np.degrees(np.arctan2(abs(dom[1]), abs(dom[0]))))
    # lag-1 autocorrelation of the tilt series = direction coherence (validated ~0.8 on real data)
    g = G - G.mean()
    coh = float(np.dot(g[:-1], g[1:]) / (np.dot(g, g) or 1e-9)) if F > 2 else 0.0
    lateral_sign = ("temporal(+x)" if (np.median(G) >= 0) else "nasal(-x)") if inplane_reliable else "n/a"
    return {
        "axial_um_rms": round(axial_rms, 2),
        # in-plane slip is None (not 0) when the surface is too flat to resolve it — keeps a nonsense huge value
        # or a misleading 0 out of the UI; the arrow then renders axial-only.
        "inplane_lateral_um_rms": round(inplane_rms, 2) if inplane_reliable else None,
        "inplane_reliable": bool(inplane_reliable),
        "axial_frac": round(float(abs(dom[0])), 3),
        "lateral_frac": round(float(abs(dom[1])), 3) if inplane_reliable else 0.0,
        "tilt_from_normal_deg": round(tilt_deg, 1) if inplane_reliable else 0.0,
        "lateral_azimuth": lateral_sign,
        "coherence": round(coh, 3),
        "variance_explained": round(var_explained, 3),
    }


def analyze_motion(sag: np.ndarray, ascan_rate_hz: float = DEFAULT_ASCAN_RATE_HZ,
                   ascans_per_frame: int | None = None, depth_spacing_mm: float | None = None,
                   lateral_spacing_mm: float | None = None, detrend_order: int = 2,
                   sinc_correct: bool = False, params: dict | None = None) -> dict:
    """Full per-scan motion analysis from a sagittal volume (lateral, depth, frames). Returns a JSON-able dict
    with the time-domain trace, power spectrum, dominant-frequency peaks, candidate spikes, and direction."""
    n, depth, F = sag.shape
    apf = int(ascans_per_frame or n)                                # A-scans per B-scan ~= lateral count
    frame_rate = float(ascan_rate_hz) / max(1, apf)                 # frames per second
    frame_time = apf / float(ascan_rate_hz)                         # seconds per B-scan (for sinc blur)
    dsp = float(depth_spacing_mm if depth_spacing_mm else octp.DEPTH_SPACING)
    lsp = float(lateral_spacing_mm if lateral_spacing_mm else octp.LATERAL_SPACING)
    um = dsp * 1000.0
    df = frame_rate / F

    S = extract_surface(sag, params)
    motion, R, mask = separate_motion(S, detrend_order=detrend_order)
    freqs, P = motion_spectrum(motion, frame_rate, sinc_correct=sinc_correct, frame_time=frame_time)
    peaks = find_peaks(freqs, P, df)
    spikes = detect_spikes(motion, frame_rate, um)
    direction = motion_direction(R, mask, lsp, dsp, S)

    # SNR: coherent motion amplitude vs the averaging noise floor (per-slice residual std / sqrt(N_inliers)).
    # Use nan-aware stats (masked-out samples are NaN) so all-masked columns don't poison the estimate.
    inl = int(max(1, np.median(mask.sum(0))))
    with np.errstate(all="ignore"):
        per_frame_std = np.nanstd(np.where(mask, R, np.nan), axis=0)
        noise_floor_px = float(np.nanmedian(per_frame_std)) / np.sqrt(inl)
    snr = round(float(np.std(motion) / noise_floor_px), 1) if noise_floor_px and np.isfinite(noise_floor_px) else None

    return {
        "n_frames": int(F), "frame_rate_hz": round(frame_rate, 2), "total_s": round(F / frame_rate, 3),
        "nyquist_hz": round(frame_rate / 2, 2), "df_hz": round(df, 3), "ascans_per_frame": apf,
        "ascan_rate_hz": float(ascan_rate_hz), "um_per_px": round(um, 4),
        "time_ms": [round(i / frame_rate * 1000.0, 2) for i in range(F)],
        "motion_um": [round(float(v) * um, 3) for v in motion],
        "freqs_hz": [round(float(v), 3) for v in freqs],
        "power": [round(float(v), 6) for v in (P / (P.max() or 1.0))],   # normalised 0..1
        "peaks": peaks,
        "spikes": spikes,
        "direction": direction,
        "snr": snr,
    }
