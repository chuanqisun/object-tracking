# Technical Plan: Classical CV Puck Tracker in JavaScript

## 1. Runtime & Technology Foundation

**Target environment:** Browser-first (portable, zero-install), with the architecture kept Node-compatible for headless/server use.

| Concern                                                         | Choice                                                                         | Rationale                                                            |
| :-------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------- |
| Frame acquisition                                               | `MediaStream` / `<video>` + `VideoFrame` (WebCodecs)                           | Zero-copy access to camera frames                                    |
| Pixel-heavy stages (color transforms, thresholding, morphology) | **WebGL2/WebGPU fragment shaders** or **WASM (OpenCV.js as optional backend)** | Per-pixel work is the bottleneck; must not run in plain JS loops     |
| Blob/contour analysis, ellipse fitting                          | WASM or typed-array JS                                                         | Moderate cost, runs on small masks                                   |
| Tracking core (Kalman, Hungarian, BYTE)                         | **Pure JS/TypeScript, written from scratch**                                   | Tiny matrices (4–10 objects), negligible cost, maximum debuggability |
| Threading                                                       | Web Workers + `OffscreenCanvas` + `SharedArrayBuffer`                          | Keep UI thread free; pipeline the stages                             |

Key principle: **the tracker core is pure, dependency-free TypeScript**; only the pixel-processing front-end is allowed to use GPU/WASM acceleration, hidden behind a backend interface so it's swappable.

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Application Layer                        │
│        (UI, visualization overlay, calibration wizard)          │
└──────────────────────────────┬──────────────────────────────────┘
                               │  TrackerFacade API
┌──────────────────────────────┴──────────────────────────────────┐
│                         Orchestrator                            │
│   frame scheduler · pipeline wiring · config store · telemetry  │
└───────┬──────────────────┬──────────────────┬───────────────────┘
        │                  │                  │
┌───────┴───────┐  ┌───────┴────────┐  ┌──────┴──────────┐
│  Acquisition  │  │  Detection     │  │  Tracking Core  │
│  Module       │→ │  Pipeline      │→ │  (BYTE, JS)     │
│  (camera,     │  │  (Worker +     │  │  (main or       │
│   homography) │  │   GPU/WASM)    │  │   worker)       │
└───────────────┘  └────────────────┘  └──────┬──────────┘
                                              │
                                     ┌────────┴────────┐
                                     │  Output Bus     │
                                     │ (events, state  │
                                     │  snapshots)     │
                                     └─────────────────┘
```

### Module boundaries (each an independent package/folder)

1. **`acquisition`** — camera setup, frame pump, timestamping, exposure metadata if available.
2. **`calibration`** — homography estimation (click 4+ ground points), color palette capture (sample each puck's chromaticity prototype), threshold auto-tuning. Persists to a serializable `CalibrationProfile`.
3. **`detection`** — the classical CV pipeline producing scored candidate detections.
4. **`tracking`** — from-scratch BYTE implementation: Kalman filter, cost matrices, Hungarian solver, track lifecycle, color Re-ID rebirth.
5. **`core/types`** — shared plain-data contracts (`Detection`, `Track`, `FrameResult`) as flat typed structures.
6. **`viz` / `debug`** — overlay renderer, per-stage mask inspector, cost-matrix visualizer.

**Contract between detection and tracking** is a plain array of:

```
Detection {
  x, y            // metric ground-plane cm (post-homography)
  radiusPx, ellipseRatio
  chroma          // (a*, b*) or (r, g) vector
  score           // composite confidence C(d) ∈ [0,1]
  sourceFlags     // which detectors fired (MSER, bgsub, color)
}
```

This decoupling means the tracking core can be unit-tested with synthetic detection streams, replayed logs, or a totally different detector later.

---

## 3. Detection Pipeline (Worker-side)

Runs as a linear pass per frame; every stage is a pluggable node implementing a common `Stage` interface (`process(FrameContext): FrameContext`), so stages can be reordered, disabled, or swapped from config.

```
VideoFrame
   │
   ▼
[1] Downscale + Rectify (single GPU pass; homography baked into sampling)
   │
   ▼
[2] Color transform pass (GPU): RGB → normalized (r,g) + Lab a*/b* channels
   │
   ├──▶ [3a] Color-distance maps: per-puck ΔE against calibrated palette
   ├──▶ [3b] Background subtraction (MOG2-style, shadow-tagging) — WASM or GPU
   └──▶ [3c] MSER / stable-region extraction on chroma channels — WASM
   │
   ▼
[4] Mask fusion + morphology (open/close), connected components
   │
   ▼
[5] Per-blob analysis (JS/WASM on small data):
      contour → ellipse fit → expected-size check at (u,v)
      → composite score C(d) = w·geometry + w·color + w·contrast
   │
   ▼
[6] Project centroids through H → metric detections → post to tracking
```

**Design notes:**

- **Fusion, not voting-out:** a blob found by _any_ branch becomes a candidate; agreement between branches raises its score. This is exactly what feeds BYTE well — shadowed pucks found only by background subtraction arrive as _low-score_ detections instead of being dropped.
- **Adaptive exposure handling:** stage 2 maintains a running global luminance statistic; color-distance thresholds are expressed in chromaticity space so they need no per-frame retuning, but the contrast score normalizes against the current frame statistics.
- **Region-of-interest fast path:** once tracks exist, the detector can optionally restrict expensive branches (MSER) to dilated windows around predicted track positions + a low-frequency full-frame sweep (e.g., every Nth frame) to catch reappearing pucks.

---

## 4. Tracking Core: From-Scratch BYTE in JS

Pure TypeScript, fully synchronous, deterministic, no allocation in the hot path (pre-allocated pools for tracks and cost matrices — trivial at N ≤ 10).

### 4.1 Components

| Component           | Responsibility                                                                           |
| :------------------ | :--------------------------------------------------------------------------------------- |
| `KalmanFilter2D`    | Constant-velocity model in metric cm; state `[x, y, vx, vy]`; hand-rolled 4×4 matrix ops |
| `CostMatrixBuilder` | Pluggable similarity functions producing an `N_tracks × N_dets` matrix                   |
| `HungarianSolver`   | Classic O(n³) assignment — trivially fast at this scale                                  |
| `TrackStore`        | Track lifecycle state machine + ID management                                            |
| `ByteAssociator`    | The two-stage BYTE orchestration logic                                                   |
| `ReIdMatcher`       | Chromaticity-prototype matching for lost-track rebirth                                   |

### 4.2 Track lifecycle state machine

```
            confirmed for M frames
 TENTATIVE ─────────────────────────▶ ACTIVE
     │  unmatched                       │ unmatched
     ▼                                  ▼
  DELETED                            LOST ──▶ (color Re-ID match) ──▶ ACTIVE
                                        │
                                        └─ age > maxLostFrames ──▶ DELETED
```

- New IDs are only born from **high-score** detections (BYTE rule).
- Since identity = color, `TrackStore` enforces a **palette constraint**: at most one active track per calibrated puck color; a new detection matching a lost track's color _always_ resurrects that ID rather than minting a new one.

### 4.3 Per-frame association flow

```
predict all tracks (Kalman)
   │
   ▼
split detections: D_high (C ≥ τ_high), D_low (τ_low ≤ C < τ_high)
   │
   ▼
Stage 1: ACTIVE+LOST tracks × D_high
   cost = λ · gated Mahalanobis distance + (1−λ) · ΔE(chroma)
   Hungarian assign; gate rejects impossible pairs (distance, color)
   │
   ▼
Stage 2: remaining tracks × D_low
   cost = pure metric distance (color deliberately ignored — shadow-corrupted)
   Hungarian assign with tighter distance gate
   │
   ▼
update matched (Kalman correct + EMA-update color prototype
                only when score is high — never learn colors from shadows)
demote unmatched ACTIVE → LOST
resurrect LOST via ReIdMatcher against leftover D_high
birth TENTATIVE tracks from unclaimed D_high (only if color is unclaimed)
discard unclaimed D_low as noise
   │
   ▼
emit FrameResult snapshot
```

All thresholds (`τ_high`, `τ_low`, gates, λ, lost-track TTL) live in a single typed config object — nothing hard-coded.

---

## 5. Threading & Data-Flow Model

```
Main thread            Detection Worker              Tracking
────────────           ─────────────────             ────────
frame pump ──frame──▶  GPU/WASM pipeline ──dets──▶   BYTE core ──▶ state
(UI, overlay ◀────────────── FrameResult snapshots ◀──────────────┘
 rendering)
```

- **Back-pressure, not queuing:** if the detector is busy, the frame pump _drops_ frames (latest-wins). Tracking handles variable Δt because the Kalman predict step is time-parameterized (`dt` from frame timestamps), so dropped frames don't corrupt velocity estimates.
- **Zero-copy transfers:** frames move to the worker as `VideoFrame`/`ImageBitmap` transferables; detection arrays come back as small structured-clone payloads (tens of numbers).
- Tracking is cheap enough to run either in the detection worker (lowest latency, one hop) or on the main thread (easier debugging) — decided by a config flag, since the core is transport-agnostic.

---

## 6. Customizability Strategy

1. **Everything behind interfaces:** `DetectorBackend` (GPU vs. WASM vs. pure-JS reference), `SimilarityFn`, `MotionModel`, `Stage`. Default wiring is declarative — a pipeline description object, not code.
2. **Single config tree**, validated and hot-reloadable at runtime: detection weights (`w_geom`, `w_color`, `w_contrast`), BYTE thresholds, gates, lifecycle timings. A debug UI can expose sliders bound directly to it for live tuning.
3. **Calibration as data:** homography, color palette, expected puck size per image region — all serialized in `CalibrationProfile`, so switching tables/cameras is a data swap, not a code change.
4. **Record & replay:** the orchestrator can log raw detection streams (tiny JSON per frame). The tracking core, being pure and deterministic, replays these offline for regression tests and threshold tuning without a camera — this is the single most valuable testing investment for the from-scratch BYTE code.

---

## 7. Performance Budget & Milestones

**Budget (target 30 FPS = 33 ms):** rectify+color GPU passes ~2–4 ms · bg-sub/MSER ~5–10 ms on a downscaled frame (e.g., 640-wide) · blob analysis <2 ms · BYTE core <0.5 ms · overlay render ~1 ms. Comfortable headroom; degrade gracefully by downscaling further or reducing MSER frequency.

**Milestones:**

1. **M1 — Tracking core first:** pure-JS BYTE + Kalman + Hungarian, tested entirely against synthetic/replayed detection streams. (De-risks the novel from-scratch part before any CV work.)
2. **M2 — Minimal detector:** color-distance thresholding + connected components only, main thread, no GPU. End-to-end tracking on easy footage.
3. **M3 — Robustness layer:** homography rectification, background subtraction with shadow tagging, MSER branch, composite scoring → real `D_low` population for BYTE stage 2.
4. **M4 — Performance pass:** worker + GPU/WASM backends, frame dropping, ROI fast path.
5. **M5 — Tooling:** calibration wizard, live-tuning debug panel, record/replay harness.
