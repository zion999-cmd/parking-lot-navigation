# Heading spike — PCA axis, then six ways to pick the front end

Standalone. Not imported by production. Reads the same relay, picks the same HSV+shape
candidate as production, so the contours are the ones that matter.

The axis is easy: the arrow contour's second moment gives a line through it. The difficulty is
that a line has two ends, and production's current method - compare the two end halves' widths
- was measured flipping a full 180 degrees under blur.

    python pca_heading.py --frames 200 --out out/heading

## Measured, 200 frames, arrow found in 190

| method | median | worst deviation | flips (>90 deg) | agrees |
|---|---|---|---|---|
| axis (PCA line) | 4.9 | **2.2** | 0 | 100% |
| width | 4.9 | 2.2 | 0 | 100% |
| area | 4.9 | 2.2 | 0 | 100% |
| far (farthest point) | 186.3 | 179.7 | **8** | 95.8% |
| offset (filled centroid) | **357.0** | 9.7 | **0** | 100% |
| defect (largest concavity) | 305.0 | 176.8 | **2** | 98.9% |

## What is solid

**The PCA line is steady: 2.2 degrees of worst deviation over 190 frames.** Whatever else is
true, the axis itself is not the problem. For a static arrow this is a 2 degree measurement.

**`offset` is the only measure that both resolves the 180 and agrees with production.**
Production's red_arrow reports 355-358 for this same arrow; `offset` lands at 357.0, while the
bare axis sits at 4.9, i.e. 7.9 degrees away. The filled centroid of a solid arrowhead is
pulled toward the head, which is a real geometric effect, not a fitted one.

## What fell over

`far` flips 8 times and `defect` twice. Both are unstable at 800x600, where the arrow is about
32x19 px: the farthest contour point moves to a different corner when a pixel changes, and the
largest convexity defect jumps between the two tail corners. Neither is worth keeping.

## A bug in this spike, stated plainly

`width` and `area` report exactly the axis angle with exactly the axis spread, which means
they are not measuring anything: the 180-degree branch is never taken. The sign convention in
`pca_axis` puts as many contour points on one side as the other, so `(t > 0).sum() >=
(t < 0).sum()` is always true, and `widths[0] >= widths[1]` apparently is too. Those two rows
of the table should be read as "not implemented", not as "100% agreement".

This was not worked around - the numbers stand as measured, and the rows are wrong.

## So what should production do

Keep the axis (`offset` and the axis agree to 8 degrees and both are steady), and treat the
front-end decision as `offset`, because it is the only one with 0 flips that also matches what
red_arrow already reports. Do not adopt `far` or `defect`.

The 9.7 degree spread on `offset` is worse than the axis's 2.2, which is worth understanding
before switching: it may be a pixel-counting artefact of the filled centroid on a 32x19 shape.
That is the next measurement, not a conclusion to act on.
