# Angle sweep: the head/tail decision is anisotropic, and 4x upsampling fixes it

Two questions: is the direction worse at the diagonals, and was there a stabilisation step in
the old path that is now gone? Both are answered below, the second without needing frames.

    python angle_sweep.py --frames 60 --step 15 --out out/angle_sweep

Method: real arrows from real frames, rotated in place by rotating a 64 px patch about the
arrow and compositing it back through a blurred circular alpha so the paste leaves no seam.
Each arm is scored against **its own reading at 0 degrees** on the same frame, which cancels
the per-frame base angle and leaves only the rotation effect. 60 frames x 24 angles.

## Result

| arm | 0/90/180/270 | 45/135/225/315 |
|---|---|---|
| **detect** (native) | jitter **0.0 deg**, flips 0/228, margin **2.72 px** | jitter **77.7 deg**, flips **114/228**, margin **0.51 px** |
| **up4x** (4x cubic) | jitter 0.0 deg, flips 0/228, margin 11.12 px | jitter **0.7 deg**, flips **0/228**, margin 10.62 px |

At the diagonals the width margin collapses from 2.72 px to 0.51 px - **sub-pixel** - and the
decision becomes a coin flip: 114 of 228 diagonal frames are wrong by ~180 degrees. The user's
diagnosis was exact. A 32x19 px arrow rasterised at 45 degrees has no margin left, and no
threshold on that margin can recover it.

Upsampling the ROI 4x before the mask restores the margin to 10.62 px and removes every flip,
while staying 4x cheaper than the current call:

| | median | p95 |
|---|---|---|
| detect (now) | 7.39 ms | 10.98 ms |
| up4x | **1.87 ms** | **2.35 ms** |

(The cost difference is because `detect` scans the whole frame for candidates and `up4x` is
handed the arrow's position by `hsv_marker`.)

## What the git record says about "it used to be better"

`_heading` is **byte-identical to its first commit** (`3c3d361`), comments included. No version
of `red_arrow.py` has ever resized, rotated or blurred a ROI — that hypothesis is refuted.

What did exist and is now gone is **temporal**: commit `585a327` added a 30 degree gate with
12-frame persistence, and its own message describes exactly this symptom - "flipped by exactly
180 degrees between frames that differ only in blur - measured 185, 5, 182, 4, 186, 5 over
twenty frames". That gate lives in **`hero_hog`** (`HEADING_GATE`, `HEADING_PERSIST`), so it
left the chain when HOG did. `red_arrow` + `hold-last-on-failure` has no equivalent: a
successful-but-flipped frame is accepted immediately.

So the memory is real, and it was a temporal filter, not an image step.

## `detect_in_box` is dead code, not a lost advantage

It cannot ever succeed. `_reference_shape()` reads `data/arrow/design/white_arrow2.png`, the
white-paper photograph, and `max(contours, key=area)` returns the **400x400 paper rectangle**:
4 points, area 159201. Every candidate is compared against that rectangle, so
`matchShapes > 0.30` always holds and the function returns `None` for every frame - which is
why the `in_box` arm of this sweep is empty. It was the right call to stop calling it, and it
should not be revived as a comparator.

## What this implies for production

Nothing has been changed. The evidence says the diagonals can be fixed by upsampling the ROI
4x before the mask, which is both the stabilisation and a speedup, and does not touch position,
thresholds, PCA, or the head/tail rule. That is a separate change with its own regression.

Caveat worth stating: this rotates a patch synthetically. It reproduces the rasterisation
effect faithfully but not a physically rotated car (lighting, shadow, motion blur). A real-turn
test is still wanted before shipping.

# Gate sequence: what a turn costs, and what a flip costs

`gate_sequence.py` feeds the production path (heading_in_roi + HeadingState) a scripted
sequence on the same synthetic rotation, and reports how many frames a change takes to commit.

| case | behaviour | frames to commit |
|---|---|---|
| slew, +5 deg/frame | every frame inside the 30 deg gate | **0** - accepted 37/37, lag 0.0 deg |
| jump, 45 deg in one frame | exceeds the gate | **12** (pending x11, then committed) |
| single 180 deg flip | pending | **never**, heading does not move |

The important line is the first one: a car being pushed round in an arc never leaves the gate,
so the persistence adds **no latency at all** to a real turn. The 12-frame cost only applies to
a direction that appears between two frames with nothing in between, which is what a flip looks
like. Being slow to believe that is the point.

The trade-off, unchanged from 585a327 and stated rather than hidden: a *sustained* 180 error -
someone physically turning the marker round, or lighting that systematically inverts the
head/tail test - is believed after 12 frames. Twelve frames of agreement is the best available
evidence that it is not noise.
