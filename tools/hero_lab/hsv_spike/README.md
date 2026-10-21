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
