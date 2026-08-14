To track 4–10 solid-color pucks without deep learning while leveraging the core principles of **ByteTrack** (two-stage association of high- and low-confidence detections, motion modeling, and track recovery), you need a pipeline combining **illumination-invariant classical vision detectors**, a **continuous scoring mechanism**, and a **customized BYTE association engine**.

---

### 1. Geometric Normalization: Ground Plane Homography (BEV)

Since the camera is static with slight perspective/angle distortion, calibrate a **Planar Homography matrix ($H$)** using 4+ known points on the surface:

- **Metric Space Tracking:** Maps pixel coordinates $(u, v)$ to real-world ground-plane coordinates $(x, y)$ in centimeters.
- **Invariant Size Filtering:** A puck of radius $r = 1.5\text{ cm}$ has an invariant circular footprint on the ground plane, eliminating perspective scale differences.
- **Linear Motion:** The Kalman Filter operates linearly in real physical units ($\text{cm}, \text{cm/s}$) rather than non-linear perspective space.

---

### 2. Illumination- and Shadow-Robust Detection Techniques

Given low color saturation, varying exposure, and shadows, standard RGB thresholding will fail. Use the following complementary techniques:

```
[ Input Frame ] ──> [ Homography Rectification / Multi-space Transform ]
                          │
         ┌────────────────┴─────────────────┐
         ▼                                  ▼
[ Illumination-Invariant ]       [ Contrast / Geometry-Based ]
- Normalized Chromaticity (r,g)  - MSER (Maximally Stable Extremal Regions)
- CIELAB (L*a*b*) / Opponent Color - Direct Least Squares Ellipse Fitting
- MOG2 Background Subtraction     - Gradient-based Edge Saliency
  with Shadow Discard
         │                                  │
         └────────────────┬─────────────────┘
                          ▼
            [ Blob / Contour Fusion ]
                          ▼
       [ Continuous Detection Scoring C(d) ] ──> [ High: D_high (C > τ) ]
                                            ──> [ Low:  D_low  (C ≤ τ) ]
```

#### A. Normalized Chromaticity & Opponent Color Spaces

Separate chromaticity (pure color) from luminance (lighting/shadow changes):

- **Normalized $(r, g)$ Space:**
  $$r = \frac{R}{R + G + B + \epsilon}, \quad g = \frac{G}{R + G + B + \epsilon}$$
  Normalized chromaticity is largely invariant to overall exposure scaling.
- **CIELAB ($L^*a^*b^*$):**
  Use the chrominance channels ($a^*, b^*$) and Euclidean color difference ($\Delta E^*_{ab}$) against calibrated puck color priors. Drop the luminance channel $L^*$ to remain robust under shadows.

#### B. MSER (Maximally Stable Extremal Regions)

- MSER treats intensity levels as water thresholds and finds regions that remain stable across multiple thresholds.
- Because pucks are uniform, solid-colored patches, MSER reliably extracts puck blobs even under low saturation, low contrast, or varying exposure.

#### C. Background Modeling with Shadow Suppression (MOG2 / KNN)

- Since the camera is static, Gaussian Mixture Models (e.g., OpenCV's `createBackgroundSubtractorMOG2`) can model the background.
- Enable shadow detection (`detectShadows=True`). Shadows cause a drop in intensity without significantly altering chromaticity; MOG2 tags shadow pixels separately so they can be filtered out rather than creating false bounding box offsets.

#### D. Direct Least Squares Ellipse Fitting

- Project the camera-view contours onto the image and fit ellipses via the Fitzgibbon Direct Least Squares algorithm.
- Projected puck cylinders appear as truncated ellipses. By computing the ratio of semi-major to semi-minor axes ($\frac{a}{b}$), you can verify whether the contour matches the expected projection geometry of a $3.0\text{ cm} \times 2.5\text{ cm}$ cylinder at that image coordinate.

---

### 3. Continuous Detection Scoring for BYTE

ByteTrack requires a continuous confidence score $C(d) \in [0, 1]$ to separate detections into **high-confidence ($D_{high}$)** and **low-confidence ($D_{low}$)** sets. Compute $C(d)$ as a composite quality score:

$$C(d) = w_{\text{geom}} \cdot S_{\text{geom}}(d) + w_{\text{color}} \cdot S_{\text{color}}(d) + w_{\text{contrast}} \cdot S_{\text{contrast}}(d)$$

1. **Geometric Fit Score ($S_{\text{geom}}$):**
   $$S_{\text{geom}} = 1 - \frac{|\text{Area}_{\text{contour}} - \text{Area}_{\text{expected}}(u,v)|}{\text{Area}_{\text{expected}}(u,v)}$$
   (Penalizes partial occlusions, debris, or background noise).
2. **Color Likelihood Score ($S_{\text{color}}$):**
   $$S_{\text{color}} = \exp\left( -\frac{\min_i \Delta E^*_{ab}(d, \text{TargetColor}_i)}{\sigma_{\text{color}}} \right)$$
   (High when the puck matches one of the target palette colors; drops in deep shadows).
3. **Local Contrast Score ($S_{\text{contrast}}$):**
   Gradient magnitude around the contour boundary normalized by local background variance.

- **High Score ($D_{high}$, e.g., $C(d) \ge 0.6$):** Clearly visible, well-lit pucks.
- **Low Score ($D_{low}$, e.g., $0.2 \le C(d) < 0.6$):** Pucks in heavy shadow, pucks touching/partially occluded, or pucks with motion blur.

---

### 4. Customizing the BYTE Association Algorithm

Adapt Algorithm 1 from the paper directly to your puck tracking constraints:

```
                  ┌───────────────────────────────┐
                  │ Frame Detections D_k with C(d)│
                  └───────────────┬───────────────┘
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
          [ D_high (C > 0.6) ]            [ D_low (C ≤ 0.6) ]
                  │                               │
                  ▼                               │
        ┌───────────────────┐                     │
        │ First Association │ ◄── [ Active Tracks T (Kalman Filter) ]
        │ Metric: IoU/Dist  │
        │       + Color Sim │
        └─────────┬─────────┘
                  ├──────────────────────┐
                  ▼                      ▼
           [ Matched Tracks ]    [ Unmatched Tracks T_remain ]
                                         │
                                         ▼
                               ┌────────────────────┐
                               │ Second Association │ ◄── [ D_low ]
                               │ Metric: Pure BEV   │
                               │  Distance/IoU Only │
                               └─────────┬──────────┘
                                         ├─────────────────────┐
                                         ▼                     ▼
                                  [ Matched Low-C ]     [ Lost Tracks T_lost ]
                                  (Shadow/Occlusion           │
                                    recovered)          [ Color Re-ID Rebirth ]
```

#### Step 1: Track State & Motion Prediction (Kalman Filter)

- **State Vector:** $\mathbf{x} = [x, y, \dot{x}, \dot{y}, r]^T$ defined on the metric ground plane ($x, y$ in cm, puck radius $r = 1.5\text{ cm}$).
- Predict track positions at frame $k$: $\hat{\mathbf{x}}_k = F \mathbf{x}_{k-1}$.

#### Step 2: First Association ($T$ with $D_{high}$)

- **Similarity Metric #1:** Combined spatial and color affinity:
  $$\text{Cost}_1(t, d) = \lambda \cdot \mathcal{D}_{\text{Mahalanobis}}(t_{\text{pos}}, d_{\text{pos}}) + (1 - \lambda) \cdot \Delta E^*_{ab}(t_{\text{color\_prior}}, d_{\text{color}})$$
- Solved via the Hungarian (Munkres) Algorithm.
- Unmatched detections form candidate new tracks; unmatched tracks pass to $T_{remain}$.

#### Step 3: Second Association ($T_{remain}$ with $D_{low}$)

- _Key ByteTrack principle:_ In low-score detections (shadows, partial occlusions), color features are heavily degraded and unreliable.
- **Similarity Metric #2:** **Pure Spatial / BEV Euclidean Distance / IoU** (ignoring color):
  $$\text{Cost}_2(t, d) = \mathcal{D}_{\text{Euclidean}}(t_{\text{pred}}, d_{\text{low}})$$
- This matches tracklets across shadows and boundary occlusions without being thrown off by skewed color values.
- Detections in $D_{low}$ with no match are rejected as background artifacts/glare.

#### Step 4: Long-Range Occlusion Recovery (Track Rebirth)

- When a puck is fully occluded or leaves the frame, move its track into $T_{lost}$ and hold its identity for $N_{\text{lost}}$ frames (e.g., 60–150 frames).
- When an unmatched high-confidence detection $d \in D_{\text{remain}}$ appears, compare its chromaticity signature against tracks in $T_{lost}$. Because each puck has a unique solid color:
  $$\text{Match if } \Delta E^*_{ab}(d_{\text{color}}, t_{\text{lost\_color}}) < \text{Threshold}_{\text{ReID}}$$
- If matched, **re-activate the lost track ID** rather than creating a new identity.

---

### 5. Summary of Recommended Strategy

| Challenge                            | Classical Technique                       | Role in BYTE Framework                                                |
| :----------------------------------- | :---------------------------------------- | :-------------------------------------------------------------------- |
| **Perspective & Scale Variance**     | Ground-Plane Homography ($H$)             | Metric Kalman filtering & fixed circular radius gating                |
| **Exposure Shifts & Low Saturation** | Normalized $(r, g)$ & CIELAB $(a^*, b^*)$ | Stable chromaticity extraction independent of $L^*$                   |
| **Varying Shadow Regions**           | MSER + MOG2 (shadow mode)                 | Recovers low-contrast blobs; assigns to $D_{low}$                     |
| **Occlusion & Shadow Continuity**    | Two-stage BYTE matching                   | Stage 2 associates low-score shadowed blobs using purely spatial cues |
| **Identity Persistence**             | Color Prototype Re-ID on $T_{lost}$       | Re-links recovered pucks to their original IDs after full occlusions  |
