# Why these recordings were moved aside (2026-10-17)

The hero car was swapped for a yellow one and the front/rear profile was re-recorded
with `capture_vehicle_profile.py --record --label right|left --finalize`.

The recordings came out useless, and the reason is the *object*, not the tool:

| channel        | corr(right, left) | corr(right, reversed(left)) | should be |
|----------------|-------------------|-----------------------------|-----------|
| gray           | **+0.927**        | +0.605                      | negative  |
| saturation     | +0.338            | +0.316                      | negative  |
| red_fraction   | 0.000             | 0.000                       | negative  |
| width          | +0.482            | +0.319                      | negative  |

`corr(right, left) = +0.927` means the two recordings are nearly *identical*, so the
thing being profiled looks the same in both orientations. A car cannot be symmetric
under a 180 degree rotation - its silhouette is about 2.4:1 (the previous car measured
72x30 px) - so what was tracked is a nearly square 30x30 blob whose four orientations
are the same. The axis reading agreed: 90.0 deg for `right`, 77.7 deg for `left`, when a
180 degree turn must leave the *axis* unchanged.

`red_fraction` being identically zero is expected and separate: that channel recorded
"where the red paint is", and a yellow car has none. Any future template for this car
has to use a different second channel.

Nothing downstream is currently hurt: `vehicle.py` refuses to claim a front for a blob
with elongation below 1.6, and this one measured 1.15-1.29. But a wrong template plus an
*elongated* object would be accepted silently, so it is parked here rather than left in
place. With no `config/vehicle_profile.json`, `vehicle_template()` returns None - the
documented "not finalized" state - and the front falls back to the silhouette taper.

Redo once the car has actually been identified in the frame: `--record --label right`,
rotate 180 degrees, `--record --label left`, `--finalize`. The check that this succeeded
is the table above: `corr(right, left)` must come out clearly **negative**.

## Second round (same day, after the above was explained)

Recorded again: `--record --label left`, `right`, `up`, `down`. Also useless, and the same way:

| channel      | corr(right, left) | needed |
|--------------|-------------------|--------|
| gray         | +0.703            | < 0    |
| saturation   | +0.235            | < 0    |
| red_fraction | 0.000 (constant)  | < 0    |
| width        | +0.249            | < 0    |

and the mask the profiles were sampled from was still the paint alone: bbox 29x34, elongation
1.35. Both recordings are of a nearly square patch of yellow bodywork, which looks the same
through a 180 degree turn by construction, so there is nothing for a template to record. The
axis readings agreed this time (90.0 and 94.4 deg), which only means the square's noise axis
was stable.

`red_fraction` was also still identically zero, for a second reason: `capture_vehicle_profile.py`
computed that channel from `vehicle.RED_HUE_RANGES` - a hardcoded red constant - just as
`vehicle.py`'s raw_red used to. Both now read the configured `hue_ranges`, but the recordings
above were made before that fix, so the channel was dead regardless of the car's colour.

Conclusion, unchanged and now measured twice: **the template cannot work while the mask covers
only the paint.** Either the car gets paint at both ends (the previous car did, which is why a
colour mask covered it), or the cabin has to come from something other than paint colour.
