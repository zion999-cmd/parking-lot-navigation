# Fragment grouping: measured, and the premise does not hold

The hypothesis was that the HSV mask breaks the arrow into fragments, so matchShapes sees two
wrong shapes instead of one right one, and that grouping the fragments would fix the rejections
at 90/270 degrees. This tool tests that offline on the 32 calibration samples before anything in
the live path changes.

## The premise

For each sample, at S>=60, count the fragments within 22 px of the marked arrow centre and
compare their total area with the area the operator clicked:

| | count |
|---|---|
| arrow is **one intact component**, coverage 1.00 | **28 / 32** |
| appears as 2+ nearby components | 4 / 32 (all are the arrow plus a big neighbour inside the 22 px window, coverage 4.4-6.1x) |
| **dilating 9 px merges the arrow with a neighbour** | **31 / 32** |

So the arrow is not fragmenting. What is nearby is *another red object*, and dilation joins the
two: a 33x21 arrow becomes a 70x39 group, which the physical gates then reject. Grouping does not
repair a broken arrow; it breaks a whole one.

## Ablation, 32 samples, leave-one-out reference bank made from the real sampled contours

| variant | candidate recall | selection accuracy | avg pool |
|---|---|---|---|
| baseline: one component, one reference | **78.1%** | - | - |
| A: pool from 3 saturation levels, one reference | 75.0% | 59.4% | 3.1 |
| B: A + 40 real references (32 sampled + 8 rotated) | **90.6%** | **84.4%** | 6.7 |
| A with 3 px grouping | 71.9% | 59.4% | 3.2 |
| B with 3 px grouping | 84.4% | 78.1% | 6.8 |

Grouping costs recall at every kernel size. The multi-reference bank is what helps: +12.5 points
of recall and +25 points of selection accuracy, with the pool growing 3.1 -> 6.7 candidates,
which is exactly what the belief layer is there to score.

## What still fails with B (3 of 32)

* 左上90 - arrow in pool never; the mask is thin there (19x30)
* 左下90 - the sample whose mask is only 14x23, partially covered
* 下270  - the anomalous sample: 37x36, area 1159, H 28 (the others are H 2-7), i.e. a merged
          blob rather than the arrow. Needs re-capture, not code.

Selection misses (arrow present but not top-scoring): 右下90, 右上0.

## Cost

Full-frame, three saturation levels, 40 references: 40.8 ms -> 42.5 ms median per frame (+4%).
Production searches a local window instead of the whole frame, where the same step costs
1.3-1.9 ms, so the absolute increase there is a fraction of a millisecond.

## Conclusion

Adopt the multi-reference bank. Do not adopt dilation-based grouping: the data says the arrow is
already one component and grouping mostly joins it to a neighbour.
