# Parking Vision — Fixed-ROI Occupancy PoC (P0002)

Proves one chain, and only this chain:

```
ESP32-S3-CAM 192.168.3.118 /stream      (firmware untouched, a pure video sensor)
              |
              v
      Python / OpenCV
              |
      fixed bay ROI  (calibrated by hand, assisted by Auto Detect)
              |
              v
      Spot 38 EMPTY   Spot 39 OCCUPIED   Spot 40 EMPTY
```

**Not in this stage:** YOLO or any deep learning, automatic bay detection *as the
source of truth*, road detection, ArUco/AprilTag, vehicle identification,
navigation, path planning, a web front end, WS2812B, MCU communication,
perspective transform, OTA.

> **P0002.1** turned `calibration.py` into a *calibration assistant*: OpenCV
> proposes bay candidates from the regular divider lines, a human drags them into
> place. The occupancy algorithm was not touched. Jump to section 3.

---

## 1. Environment

`cv2` is needed. On this machine it is already installed in
`/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python` (cv2 4.13.0,
numpy 2.4.6). Everything below assumes that interpreter; substitute your own.

```zsh
cd vision
P=/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python
$P --version
```

`pip install -r requirements.txt` only if `cv2` is missing.

## 2. Video input — measured facts

| | result |
|---|---|
| `cv2.VideoCapture("http://192.168.3.118/stream")` | ❌ `isOpened() == False`, *"Couldn't read video stream from file"*. OpenCV's bundled FFmpeg will not read this `multipart/x-mixed-replace` stream. |
| `src/camera.py` `MjpegHttpReader` (urllib + boundary + `Content-Length` + `cv2.imdecode`) | ✅ **60 frames in 6 s ≈ 10 fps**, 800x600, 0 reconnects, 0 decode failures |

So the fallback the task anticipated is the path in use. `open_camera()` still
tries `cv2.VideoCapture` first, so if a future OpenCV starts working nothing here
has to change. Force a backend with `--backend opencv|mjpeg`.

The reader reconnects on its own, so a camera reboot or an OTA does not kill the
viewer.

Frame quality of the current view (measured, `800x600`):

```
brightness mean=146.9  contrast(std)=34.4  sharpness(Laplacian var)=63.5
overexposed=0.00%      underexposed=0.00%
```

Well exposed, nothing clipped — but soft: a Laplacian variance in the 60s is low
(a perfectly sharp textured scene is usually in the hundreds). That is JPEG q12
plus optics plus a mostly flat scene; whether it is *enough* is decided by the
acceptance cases, not by this number.

## 3. Calibrate the three bays

Coordinates are **not** hardcoded anywhere, and nothing has to be drawn from
scratch: Auto Detect proposes candidates, you correct them.

```zsh
cd vision
P=/usr/local/Caskroom/miniconda/base/envs/vision-dev/bin/python
$P src/calibration.py --source http://192.168.3.118/stream --auto
```

Or calibrate offline from a saved frame with no camera:

```zsh
$P src/calibration.py --frame data/debug/frame.jpg --auto
```

### Window

```
[Auto Detect] [Gen Row] [Save] [Undo] [Reset]      <- clickable, or use the keys

 left click a bay         select it
 left click a corner      drag that corner
 left drag inside a bay   move the whole bay
 right click a corner     snap it to the nearest strong edge
```

| key | action |
|---|---|
| `a` | Auto Detect — **only adds** candidates, never moves or deletes your work |
| `g` / `G` | generate the next bay in the row to the right / left of the selected one |
| `n` | add an empty bay at the mouse position |
| `d` | delete the selected bay |
| `r` | rename the selected bay (type the id in the terminal) |
| `i` | set the inner detection scale of the selected bay (terminal) |
| `s` / `o` | save / load `config/parking_spots.json` |
| `z` / `c` | undo / reset (clear all) |
| `l` | toggle the detected-line overlay (edges, dividers, bands) |
| `v` | toggle the inner detection polygons |
| `Tab` | cycle the selection |
| `2`/`4`/`6`/`8` | nudge the selected bay by 5 px |
| `q` | save and quit |

### The workflow that actually works

1. `a` — see what Auto Detect found. The status line reports how many candidates
   were added, how many were already present, the row count and the measured bay
   pitch.
2. Drag corners of the bays that are close. Bays it missed: `n` to add one, then
   `g` repeatedly to walk the row (this uses the selected bay's own top edge, so
   it follows a slanted row too).
3. `s` to save. Next time the tool loads the JSON and you continue from there.

Auto Detect is deliberately allowed to be imperfect: every corner is draggable,
and a bay it fails on costs you one `n` plus a few `g` presses, not a redraw of
the whole row. It also cannot damage anything - one `z` undoes an entire run.

### Measured defaults for this sandbox

Tuned against the real camera on 2026-10-05 (saved frame and live stream agree):

| setting | value | why |
|---|---|---|
| `--detect-method` | `profile` | the Hough/`segments` path reads the two edges of each white line as two dividers, so the lower row came back with a bogus 21 px pitch, and on a noise frame it chained 1367 segments into 204 "dividers" |
| `--detect-orientation` | `both` (current default) | `vertical` was the value tuned on the 2026-10-05 framing; the current scene has rows in both directions, so both are merged. `auto` is still avoided: it picked `horizontal` on one run and `vertical` on another for the same scene |
| `--band-threshold` | `0.05` | 0.30 missed the upper row of bays entirely |
| periodicity confidence | autocorr >= 0.25 | measured: **0.53-0.67** on the real scene, 0.85 on a clean synthetic row, **0.068-0.128** on a noise-only frame — so the threshold sits a factor of two from both sides. This is what stops a random image producing ~100 bogus bays. |

Result on this scene: two rows at y 35..133 and y 196..390, **bay pitch 46-47 px in
both rows**, 30-31 candidates, bay width median 48 px. The consistent pitch across
two independent rows is the strongest sign it is reading real bays.

### If Auto Detect finds the wrong thing

The detector looks for rows of parallel, evenly spaced lines. Run it once in
report mode and read the numbers before clicking anything:

```zsh
$P src/calibration.py --source http://192.168.3.118/stream --detect-only
```

That prints each row's divider count and measured pitch, the candidate count,
their width/height range, a suggested width filter, and **a text picture of the
edge profile with the fitted comb underneath it**:

```
band 0  y=36..133  pitch=46.2px  peaks=18  comb=17
  | =:     =:     =.     +      *.     # ...  :   .  @    . .     =    ... |
  | ^      ^      ^      ^      ^      ^      ^      ^      ^      ^     ... |
       ^ matched peak   + reconstructed   o ignored peak
```

If the `^` marks do not sit under the tall bars, the pitch or the phase is wrong -
and that is checkable from a log, without opening an image.

It also writes `data/debug/detect_only.png` with every candidate outlined and
numbered; open that next to the live image.

Then tune, and repeat:

| knob | why |
|---|---|
| `--detect-orientation vertical` / `horizontal` | **try this first.** `auto` picks whichever orientation yields more candidates, and on a busy scene that can flip between runs. The overlay makes the right answer obvious. |
| `--max-width` / `--min-width` | bay size filter. `--detect-only` prints a suggestion derived from the measured pitch (`try --min-width 40 --max-width 142`). This is the single most effective knob. |
| `--band-threshold` | 0.05 default. Raise it (0.3-0.5) if unrelated bands appear; the old 0.30 default is what missed the upper row entirely. |
| `--grid-tolerance` | 0.18 default. Raise if real bays are not perfectly evenly spaced. |
| `--min-width` | also the smallest pitch the periodicity search will consider. |

The same options work in the window, so you can iterate without restarting.

### Only calibrate three bays for this stage

They are the point of the exercise; the sandbox numbering wins over any
convention. Use the sandbox's real numbers.

### Why there are two polygons per spot

```json
{
  "spots": [
    { "id": 38,
      "polygon": [[120, 260], [230, 260], [230, 380], [120, 380]] }
  ],
  "image_size": [800, 600]
}
```

* `polygon` — the whole bay. Drawn, and used to crop the reference.
* `detect_polygon` — the area occupancy actually looks at. If you omit it (the
  normal case) it is `polygon` **shrunk towards its centre by `inner_scale`**
  (default `0.62`, so 38 % of the area is thrown away, mostly the rim).

That is the deliberate answer to *"don't detect the bay boundaries"*: white bay
lines, red/white markers, the painted bay number and the kerb edge all sit in the
discarded rim. `tests/test_occupancy.py` pins this down — a test blacks out both
bay lines and paints a fake bay number, and asserts `changed_ratio == 0.0`.

Per spot you may override `inner_scale` or give an explicit `detect_polygon` if a
bay has something peculiar inside it.

## 4. Capture the empty-bay reference

The reference is **one full frame**, not per-spot crops, so that adding bays later
costs nothing.

```zsh
$P src/main.py            # window opens
# with all three bays empty, press  r
```

That writes `data/reference/reference.jpg` and prints the frame diagnostics.
Re-capture it whenever the lighting or camera moves — the whole method is a
comparison against this one frame.

## 5. Run

```zsh
$P src/main.py                                   # live window
$P src/main.py --headless --duration 20          # no window, prints one line/s
```

Window keys: `r` capture reference · `g` inner ROI · `d` changed-pixel overlay ·
`+`/`-` threshold ±0.01 · `[`/`]` pixel threshold ±5 · `p` print · `s` snapshot ·
`q` quit.

The window shows the bay outline (yellow), the inner detection area (green,
toggleable), the changed pixels in red, and `Spot 38 EMPTY 0.031` per bay.

## 5.1 Camera drift: align instead of redrawing

The camera is meant to be fixed, but a nudge moves the picture by a few pixels and
then every ROI sits slightly off its bay. There are two ways to handle that, and
neither of them touches the bay list:

**Runtime compensation (preferred, changes nothing on disk).** `main.py` estimates
the drift against the reference frame and moves the *incoming frame* back onto it,
so the ROIs and the reference ROIs stay exactly where they were drawn:

```zsh
$P src/main.py                       # --align off is the default: the camera is bolted down
$P src/main.py --align translation   # compensate a nudge (translation only)
$P src/main.py --align euclidean     # if the camera also got rotated
```

It re-estimates every `--align-every` frames (default 3) and logs the estimate.
Each CSV row carries `align_ok, align_dx, align_dy, align_corr`.

**Re-align the stored polygons** (when you would rather fix the file):

```zsh
# dry report, no change
$P src/calibration.py --frame data/debug/frame.jpg --template data/reference/reference.jpg
#   ... in the window, press  A  (moves every bay onto the current frame; s saves)

# or headless: align and save in one go
$P src/calibration.py --frame <current.jpg> --template data/reference/reference.jpg \
    --auto-align --out config/parking_spots.json
```

`A` and `--auto-align` move every polygon by the estimated drift. They never add,
delete or renumber a bay, and `z` undoes it.

### What it refuses to do, and why

The scene is an evenly spaced grid, and that makes one failure mode fundamentally
undecidable: **a shift of exactly one bay looks identical to no shift at all.**
Measured on the real frame, "do nothing" scores *higher* than the correct one-bay
correction (0.956 against 0.925). So the aligner does not guess. It may answer
"aligned, no correction needed" and leave the ROIs a bay out - harmless, because it
applies nothing - but it will never invent a sub-bay correction it cannot justify.

Thresholds come from measurement, not taste. Shifting the real calibrated frame and
letting ECC correct it:

| applied shift | no-move score | score after | ECC's dx vs truth |
|---|---|---|---|
| 3, 2 | 0.522 | **0.973** | +3.0 correct |
| 7, -4 | 0.190 | **0.974** | +7.0 correct |
| 15, 8 | 0.050 | **0.957** | +15.3 correct |
| 40, 0 | 0.504 | 0.545 | +27.7 wrong |
| 70, 0 | 0.449 | 0.636 | +25.0 wrong |
| 150, 0 | 0.361 | 0.623 | +105.0 wrong |

A real alignment lands at 0.95-0.97 and every failed one at 0.54-0.64, so the gate
is 0.8 - 0.16 away from both. Practical range on this scene: a nudge up to about
15 px (a third of a bay). Anything larger is reported as "cannot align", not
applied. A rejected alignment means the ROIs are NOT compensated, so the run
warns loudly instead of quietly returning numbers that look fine.

Two details worth recording. A brightness correlation is useless here (mostly flat
floor: a 150 px displacement still scored 0.96), so the check compares *edge*
correlation. And the alignment is property-tested with "an accepted alignment must
never make the frame worse" as well as with exact synthetic shifts.

## 6. The algorithm, and its only three knobs

```
empty reference ROI  ->  current ROI
        |                    |
        +---- cv2.absdiff ---+
                 |
        grayscale + GaussianBlur (5x5)          suppresses JPEG noise
                 |
        cv2.threshold(pixel_threshold = 25)     a pixel counts as changed
                 |
        morphology open 3 / close 5             despeckle, then join blobs
                 |
        and with the inner polygon mask         ignore everything outside
                 |
        changed_ratio = changed_px / inner_px
                 |
        changed_ratio >= changed_ratio_threshold  ->  OCCUPIED
```

| knob | default | meaning |
|---|---|---|
| `--changed-ratio` | 0.30 | fraction of the inner area that must change (the measured range is in `occupancy.py`) |
| `--pixel-threshold` | 25 | per-pixel difference (0-255) that counts as changed |
| `--blur` | 5 | pre-filter size |

**The threshold is not meant to be guessed.** See below.

## 7. Acceptance cases — and turning them into a threshold

Record each case, then let the data pick the threshold.

```zsh
cd vision
# case 1: all three bays empty
$P src/main.py --headless --duration 20 --csv data/cases/case1_all_empty.csv
# case 2: car only in 38
$P src/main.py --headless --duration 20 --csv data/cases/case2_spot38.csv
# case 3: cars in 38 and 40
$P src/main.py --headless --duration 20 --csv data/cases/case3_38_40.csv
# case 4: all three occupied
$P src/main.py --headless --duration 20 --csv data/cases/case4_all_occupied.csv
# case 5: cars removed again (Proves recovery to EMPTY)
$P src/main.py --headless --duration 20 --csv data/cases/case5_removed.csv
```

Each CSV row is `t_seconds, spot_id, changed_ratio, state, align_ok, align_dx,
align_dy, align_corr` (one row per spot per frame).

Then describe what was physically true while each file was recorded:

```zsh
$P tests/fit_threshold.py --cases 'data/cases/*.csv' --write-template data/cases/labels.json
# edit labels.json so every spot says EMPTY or OCCUPIED as it really was, then:
$P tests/fit_threshold.py --cases 'data/cases/*.csv' --labels data/cases/labels.json
```

It prints the EMPTY and OCCUPIED `changed_ratio` distributions, the overlap (if
any — that is the "we cannot separate these" answer), the recommended threshold
with its false-positive/false-negative counts, and a per-spot margin so a single
badly calibrated bay is visible.

## 8. Tests

```zsh
$P tests/test_occupancy.py    # 13 checks - the occupancy maths (unchanged by P0002.1)
$P tests/test_detector.py     # 17 checks - the bay candidate detector
$P tests/test_editor.py       # 31 checks - editing model + GUI wiring
$P tests/test_alignment.py    # 15 checks - camera drift compensation
```

76 checks, no camera required.


* **occupancy** - inner-polygon shrink maths, identical frame → EMPTY, car inside
  → OCCUPIED, **bay markings ignored**, **bounding-box corner ignored (polygon,
  not rectangle)**, JPEG-like noise stays EMPTY, partial occupancy lands between,
  thresholds respected, resolution change refused rather than silently
  mis-measured, two bays independent.
* **detector** - synthetic car parks where the answer is known exactly: 11
  layouts (1x4, 1x8, 2x6, 3x8, narrow 2x10, narrow+tall, short, wide, slanted,
  noisy) must come back with the exact bay count, and the polygons must sit on the
  drawn lines. Plus: a missing divider is **reconstructed** from the measured
  pitch, a blank image and a noise-only image must not produce a screenful of
  false positives, and horizontal rows must be found through the same code path.
* **editor** - add / drag corner / drag body / nudge / delete / clear / rename
  (including refusing a duplicate id) / inner scale range check / generate row
  (pitch, slanted rows, stopping at the frame edge and on overlap) / undo (one
  undo reverses one drag, and one undo reverses a whole Auto Detect) / save-load
  round trip / Auto Detect never modifying hand-made bays. The GUI layer is driven
  with **synthetic mouse and key events** (`on_mouse`, `act`), so button clicks,
  corner dragging, `Tab` cycling and `q`-to-quit are covered without a display.

## 9. Layout

```
vision/
├── README.md
├── requirements.txt
├── config/parking_spots.json      calibrated bays (created by calibration.py)
├── data/
│   ├── reference/reference.jpg    the empty-bay frame ('r')
│   ├── cases/                     acceptance CSVs + labels.json
│   └── debug/                     frame.jpg, calibration_aid.png, snapshots
├── src/
│   ├── camera.py                  video input: cv2.VideoCapture + MJPEG reader
│   ├── zoom.py                    PC-side digital zoom (image only)
│   ├── spot_config.py             bay polygons, inner-ROI maths, JSON I/O
│   ├── occupancy.py               the difference algorithm
│   ├── alignment.py               camera drift estimation (ECC) + guards
│   ├── detector.py                geometric bay candidate detection (no AI)
│   ├── arrow.py                   roof-arrow template match (position + heading)
│   ├── vehicle.py                 hero-car pose, map location, tracker
│   ├── spot_editor.py             editing model: drag/add/delete/rename/row/undo
│   ├── road_editor.py             road drawing / editing model
│   ├── road_network.py            road graph, bay access derivation, validation
│   ├── map_document.py            two-layer map file, atomic read/write
│   ├── calibration.py             the window, mouse and key map (+ --detect-only)
│   ├── contract.py                parking_map/1 + parking_state/1 documents
│   ├── emit_map.py                config -> out/parking_map.json
│   ├── route.py                   Dijkstra planner + refusal codes
│   ├── plan.py                    CLI: plan a route and write route.json
│   ├── live_view.py               annotated / raw MJPEG relays
│   ├── serve.py                   static pages + GET /api/route
│   └── main.py                    live viewer / headless recorder
├── web/                           board.html, navigation.html, index.html
├── out/                           generated map/state/route documents
└── tests/
    ├── test_occupancy.py          synthetic occupancy tests
    ├── test_detector.py           synthetic car park detection tests
    ├── test_editor.py             editing + GUI wiring tests
    ├── test_api.py                real HTTP round-trips against serve.py
    ├── test_navigation.py         page JS (extracted, run with node)
    └── fit_threshold.py           threshold from recorded cases
```

## 10. Troubleshooting

| symptom | look at |
|---|---|
| `cv2.VideoCapture` cannot open the stream | expected; the fallback reader is used automatically |
| every bay reads OCCUPIED | reference captured with cars present, or the camera moved. Re-capture with `r` |
| a bay flickers EMPTY/OCCUPIED | `changed_ratio` sits near the threshold — check the per-spot margins from `fit_threshold.py`, then move that bay's ROI off a shadow/edge or lower `inner_scale` |
| bay lines or numbers cause changes | lower `inner_scale` for that spot (`i` in the calibrator) |
| everything drifts slowly | lighting change (sun/clouds), not a bug — re-capture the reference |
| Auto Detect finds nothing | run `--detect-only` and read the report; usually `--detect-orientation` or `--max-width`. Not fatal: `n` then `g` repeated builds the row by hand |
| Auto Detect boxes the wrong lines | same: the overlay PNG shows what it locked onto |
| `frame geometry changed` error | the firmware's stream resolution changed; re-capture the reference |
