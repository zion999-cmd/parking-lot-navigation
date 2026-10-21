# red_arrow heading: what the code actually does, and where my spike went wrong

## 1. `red_arrow.detect` — the real chain

### Red region
`_mask()` (`red_arrow.py:50-56`) is a pixel test, not `cv2.inRange`:

    red = ((h <= 12) | (h >= 168)) & (s >= 70) & (v >= 50)

Then `MORPH_OPEN` with a **2x2** kernel (`red_arrow.py:78`) — the smallest possible opening,
which removes single-pixel noise without eating a 32x19 arrow.

Note the hue arcs: **0-12 and 168-180**. The production channel that now owns *position*
(`hsv_marker`) measured the arrow at **H 24-46** and uses **0-50**. These two disagree about
what colour the arrow is, yet both find it. They are finding different blobs.
`red_arrow` finds the redder edge of the arrow; `hsv_marker` finds the orange body.
This matters for the archaeology: **`red_arrow`'s heading is computed from a different,
smaller blob than the one `hsv_marker` reports as the position.**

### Which contour
No contour. `detect()` uses `connectedComponentsWithStats` (`:79`), filters components by
`area in [60, 800]` and `aspect in [1.2, 3.5]` (`:84-90`), then picks:
- with `prefer_near`: the candidate **nearest the last position** (`:96-98`)
- otherwise: the **largest area** (`:100`)

No shape matching at all in `detect()`. `matchShapes` appears only in `detect_in_box()`
(`:160-171`), which production no longer calls for heading.

### Centre
`centroids[i]` from `connectedComponentsWithStats` (`:104`) — the **area-weighted centroid of
the blob**, not the bbox centre.

### PCA axis
`_heading()` (`:58-73`), on **all mask pixels** (`ys, xs = np.where(blob > 0)`), not on contour
points:

    pts = [xs - mean(xs), ys - mean(ys)]        # image coordinates, y DOWN
    _, evec = cv2.PCACompute(pts, maxComponents=2)
    u = evec[0]                                  # largest eigenvalue
    axis = degrees(atan2(u[1], u[0])) % 360      # y down => clockwise-positive

**The angle convention is `atan2(y, x)` in image coordinates**, i.e. clockwise on screen.
Eigenvector sign is whatever LAPACK returns, so `axis` is an axis, not a direction.

### 180 head/tail
`:66-73`:

    proj = pts @ u
    perp = abs(pts @ [-u[1], u[0]])
    lo, hi = proj.min(), proj.max()
    band = max((hi - lo) * 0.30, 1.0)
    narrow = percentile(perp[proj < lo + band], 85)
    wide   = percentile(perp[proj > hi - band], 85)
    return axis if wide > narrow else (axis + 180) % 360

So: **project the filled mask onto the axis, take the outer 30% at each end, and compare the
85th percentile of |perpendicular distance|.** The wider end is the head. Two details that
matter and that my spike did not copy: it uses the **filled mask**, and it uses the **85th
percentile**, not the maximum.

### Temporal filtering
None in `red_arrow.py`. The stabiliser is in `vehicle.py`: `_last_heading` is held when a frame
fails, and `hsv_marker`'s `prefer_near` biases blob selection. `hero_hog` has its own
`_LAST_HEADING`. **The 180 decision itself is single-frame.**

### Which branches are reachable
- `_heading`'s `+180` branch: **reachable**, it fires whenever the far end is wider. Measured
  in production: 355-358 deg over 12 consecutive samples, consistent.
- `detect_in_box`: **not called by production any more** (I replaced it with `detect` when
  wiring `hsv_marker`). It returns `None` when handed an arrow-sized box, which is why it was
  the wrong call.
- The `prefer_near` branch: reachable and used.

## 2. Why my spike reported width == axis, and why 180 never fired

Two separate mistakes, both mine.

**Mistake 1 - flipped angle convention.** `pca_heading.py:pca_axis` returns
`degrees(atan2(-axis[1], axis[0]))`. The minus sign was there to get a counter-clockwise
"maths" angle. `red_arrow` uses `atan2(u[1], u[0])`, clockwise. So the spike's 4.9 deg and
production's 355 deg are **the same direction**, mirrored: 360 - 4.9 = 355.1. The "7.9 deg
disagreement" I reported between `offset` (357.0) and the axis was an artefact of this: in
production's convention the axis is at 355.1, and offset agrees to **1.9 deg**.

**Mistake 2 - the 180 branch could not fire.** Both `front_by_width` and `front_by_area`
compared *contour points*, which `contour_of()` returns from `findContours(...RETR_EXTERNAL,
CHAIN_APPROX_NONE)` — a thin outline. For a hollow outline of a roughly symmetric shape:

    front_by_width: widths.append(perp.max() - perp.min())     # full spread of the half
    front_by_area:  (t > 0).sum() >= (t < 0).sum()             # counts contour points

Both halves span the same full width and carry nearly the same number of outline points, so
the comparison is a near-tie and `>=` picks `base` every time. That is exactly what the table
showed: identical median and identical spread to the bare axis, for 190/190 frames. The 180
branch was never executed — **not a tuning problem, the branch was unreachable.**

`red_arrow` does not have this bug because it projects **filled mask pixels**, where the head
genuinely has more of them, and it uses the **85th percentile** rather than max-min, which
reacts to the ribbon being wider at one end instead of to a single stray corner.

## 3. What the fix has to be

Same axis (reduce on filled mask pixels, `atan2(y, x)`, production convention), then a
**signed** score instead of a boolean:

    band      outermost 25% of the projected extent at each end
    width_e   mean |perp| of the pixels in that band        (mean, not a percentile)
    area_e    pixel count in that band
    score     (width_hi - width_lo) + lambda * (area_hi - area_lo) / scale
    heading   axis if score > 0 else axis + 180

Signed, so the table can report `head_tail_score` and its distance from zero, not just which
way a coin landed. `far` and `defect` stay as experimental controls only; the spike already
showed 8 and 2 flips respectively.

## 4. Results, 220 cached frames, every method anchored to the same blob

The first run of the evaluation reported `prod` flipping 76 times and the width score
straddling zero. **That was the evaluation's fault, not the code's**: it called
`red_arrow.detect(frame)` without `prefer_near`, so `detect` fell back to "largest red blob"
and every method was handed a different object than production hands it. With the anchor
production actually uses, `prefer_near=(240,485)`:

| method | valid | median | jitter | maxdev | flips | temporal | ms/frame |
|---|---|---|---|---|---|---|---|
| prod (red_arrow) | 220 | 356.3 | **1.1** | 4.4 | **0** | **0** | 4.47 |
| axis | 220 | 356.3 | 1.1 | 4.4 | 0 | 0 | 1.48 |
| width (fixed) | 220 | 356.3 | 1.1 | 4.4 | 0 | 0 | 1.34 |
| area (fixed) | 220 | 356.3 | 1.1 | 4.4 | 0 | 0 | 1.32 |
| combined | 220 | 356.3 | 1.1 | 4.4 | 0 | 0 | 1.31 |
| width85 (red_arrow's own) | 220 | 356.3 | 1.1 | 4.4 | 0 | 0 | 1.50 |
| offset | 220 | 357.5 | 0.8 | 9.2 | 0 | 0 | 1.46 |
| hog | **53** | 355.7 | 0.0 | 0.0 | 0 | 0 | **75.73** |

Head/tail scores, which is what the task asked for. Positive means the hi end is the head:

| method | min | median | max |
|---|---|---|---|
| width | **+1.05** | +1.60 | +2.14 |
| area | **+26** | +40 | +57 |
| combined | **+1.57** | +2.41 | +3.28 |
| width85 | **+1.58** | +2.73 | +4.04 |

**The scores never approach zero.** The head end is consistently 1.6 to 2.7 px wider and 40
px heavier. The 180 decision is not a coin flip - it has a margin of roughly 1.6 px against a
1.1 degree angular jitter.

## 5. Answers

**A. What is production's head/tail decision?** The 85th percentile of |perpendicular
distance| in the outer 30% band at each end of the axis, computed on the **filled mask**, with
the wider end declared the head (`red_arrow.py:66-73`). It is a single-frame decision; the
only temporal stabiliser is `vehicle.py` holding `_last_heading` across failures.

**B. Does the fixed width/area resolve the 180?** Yes, and so did the original - once the
evaluation stopped feeding it the wrong blob. All four end-comparison variants agree with each
other and with production to within 0.0 degrees over 220 frames, 0 flips, and all of their
scores stay on one side of zero. The spike's failure was two bugs of mine (flipped angle
convention, and contour points instead of filled pixels), not a weakness in the method.

**C. Can the heading be stable without HOG?** Yes, and it already is. `red_arrow` gives
220/220 valid frames at 1.1 degrees of jitter in 4.47 ms. `hero_hog` produced a heading in
53/220 frames at 75.73 ms - 17x slower, and silent three quarters of the time.

## 6. Should production change?

No. The task's own rule: only replace if clearly better. The heading path is already at
0 flips, 1.1 deg jitter and 0% failures, which meets every acceptance threshold. The
alternative variants are 3x cheaper per call (1.3 vs 4.5 ms) but that compares a whole
`detect()` against a single heading computation - `blob_of` plus the fixed width measure is
about 2.8 ms, not 1.3. A ~1.7 ms saving on a loop that already runs at 10 ms per frame is not
worth touching a heading path that measures this well.

**What should change is the fallback.** `hero_hog` is currently reached whenever `red_arrow`
returns no heading - but it answers on only 24% of frames and costs 76 ms when it does. On the
evidence here it is a 76 ms gamble with a one-in-four payoff. That is worth a separate
decision, not a change made in passing.
