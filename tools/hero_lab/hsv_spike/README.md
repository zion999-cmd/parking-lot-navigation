# HSV marker spike — Phase 1

A separate, revocable experiment: can an HSV red mask compress each frame into a few
candidates that contain the car's arrow? Nothing here is wired into production, and nothing
here reads `parking_spots.json`, `vision/data/labels` or `tools/hero_lab/data`.

    python hsv_candidates.py --frames 240 --config config.json --out out/phase1

## What Phase 1 found

**The arrow is not "red" in the OpenCV sense.** Sampling the pixels at the arrow (272,472)
in the live stream gives H 24-46, median 31 - the OrangeRed side of the H axis. The two red
arcs that the task started from (0-10 and 170-180) contain **none** of it.

Measured, 240 frames, config as shipped (H 0-10 / 170-180, S>=90, V>=60):

| | |
|---|---|
| candidates/frame | mean 1.89, median 2, max 4 |
| frames with zero | 3 / 240 (1.2%) |
| where they were | x 160-239, y 240-319 - **100% in one cell, none at the arrow** |

That single blob is a different red object (H median 11). So the mask looked clean and was
measuring the wrong thing.

Measured again with the arc widened to cover the arrow (H 0-50 / 170-180, S>=60, V>=40,
max_area 1200), 200 frames:

| | |
|---|---|
| candidates/frame | mean 2.17, median 2, max 6 |
| frames with zero | **0 / 200** |
| candidates within 60 px of the arrow | **200 / 200 frames** |
| that candidate's centre | x **240.0..241.3** (jitter 1.3 px), y **485.4..486.1** (jitter 0.7 px) |
| that candidate's area | 234..279 |

The extra 1-4 candidates sit at the frame edges (x 0-79 and x 720-799, low in the frame),
which is where the red bay lights and the LED strip are.

## The three questions the spike was asked

**A. Can the arrow be found stably?** Yes - 200/200 frames with 1.3 px of centre jitter,
but only after the H range was fixed. The arrow is OrangeRed, not Red.

**B. Do red background / LED / other objects cause many false positives?** Some, 1-4 per
frame, and they are spatially separated from the arrow (frame edges, y > 480). Filtering
them is a Phase 2 job (shape), not a threshold job - they are redder than the arrow.

**C. Is it steadier / lighter / easier to tune than HOG?** Steadier centre, yes: 1.3 px vs
the HOG box. Lighter: one mask plus connectedComponents, no model file. Easier to tune:
arguably - one JSON of thresholds, at the cost of having to know that the arrow is H 25-45.

## What this does not show

- No ground truth per frame, so "the candidate is the arrow" is inferred from position and
  from the fact that it is the only blob within 60 px of where HOG puts the car.
- The "largest candidate" statistic is misleading: the edge blobs are bigger than the arrow
  (area median 262 for the arrow, up to ~740 for the edge ones), so largest-first picks the
  wrong one. Selecting by shape or by compactness is Phase 2.
- Single-frame only. No tracker, by design.

## Phase 1 config as shipped vs as needed

`config.json` keeps the task's original arcs (0-10 / 170-180). `config.orangered.json` is the
measured one (0-50, S>=60, V>=40, max_area 1200). Compare both before changing the default.

# Phase 2 — shape matching

One arrow contour as the reference (picked from a frame at the position Phase 1 measured, so
it is the arrow itself, not a training set), then `cv2.matchShapes` with `CONTOURS_MATCH_I1`
against every candidate. Hu moments are invariant to position, scale and rotation, which is
what a candidate filter needs - and which is also why these scores cannot give a heading.

    python shape_match.py --frames 200 --out out/phase2

200 frames, 442 candidates (200 near the arrow, 242 elsewhere):

| | n | min | median | max |
|---|---|---|---|---|
| near the arrow | 200 | 0.016 | **0.102** | 0.415 |
| elsewhere | 242 | 0.105 | **0.873** | 5.971 |

**Cut at the arrow's own worst score (0.415): keeps 200/200 arrow candidates and 15/242 of
the others.** 94% of the Phase 1 clutter - the bay lights and LED strip at the frame edges -
is removed by shape alone, without touching a single colour threshold.

Precision 200/215 = 0.93, recall 200/200 = 1.00, F1 0.96 (against the position-based
definition of "the arrow" below - no per-frame ground truth exists yet).

## The caveat that matters

"Near the arrow" is a position test (within 60 px of 240,485), so this is not a ground-truth
evaluation. What is not circular is the separation itself: the score is computed from the
contour alone, and the two groups land at 0.102 and 0.873 without knowing where they are.

## What Phase 2 does not give

Heading. `matchShapes` is rotation-invariant by design, so an arrow pointing left and an
arrow pointing right score the same. If heading is wanted, it has to come from the template
angle sweep (Phase 3), not from here.
