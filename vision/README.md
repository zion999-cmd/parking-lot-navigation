# Parking Vision — Fixed-ROI Occupancy PoC (P0002)

Proves one chain, and only this chain:

```
ESP32-S3-CAM 192.168.3.118 /stream      (firmware untouched, a pure video sensor)
              |
              v
      Python / OpenCV
              |
      fixed bay ROI  (calibrated by hand)
              |
              v
      Spot 38 EMPTY   Spot 39 OCCUPIED   Spot 40 EMPTY
```

**Not in this stage:** YOLO or any deep learning, automatic bay detection, road
detection, ArUco/AprilTag, vehicle identification, navigation, path planning, a
web front end, WS2812B, MCU communication, perspective transform, OTA.

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

Coordinates are **not** hardcoded anywhere. Two ways to produce
`config/parking_spots.json`:

**A. Interactive (preferred)**

```zsh
$P src/calibration.py --source http://192.168.3.118/stream --start-id 38
```

Click the four corners of a bay; the 4th click commits it. Repeat for the next
bay. Keys:

| key | action |
|---|---|
| left click | add a corner (4th commits the bay) |
| `u` | undo the last corner |
| `d` | delete the last committed spot |
| `r` | rename the last spot (type the id in the terminal) |
| `i` | change the inner detection scale for the last spot |
| `s` | save, `l` reload |
| `g` | toggle the coordinate grid |
| `v` | toggle the inner detection polygon |
| `q` | save and quit |

`r` and `i` ask in the terminal, so the window disappears for a moment — that is
expected.

**B. From the ruler image, by hand**

```zsh
$P src/calibration.py --frame data/debug/frame.jpg --grid
open data/debug/calibration_aid.png            # 50 px grid with coordinates
```

Then read the corner coordinates off the image and write `config/parking_spots.json`
with the same schema (see below). Useful when a GUI is inconvenient.

**Only calibrate three bays** for this stage, chosen from the clearest, most
central ones. Use the sandbox's real numbering.

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
| `--changed-ratio` | 0.15 | fraction of the inner area that must change |
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

Each CSV is `t_seconds, spot_id, changed_ratio, state` per frame.

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
$P tests/test_occupancy.py         # 13 checks, no camera needed
```

Synthetic-image tests for: inner-polygon shrink maths, identical frame → EMPTY,
car inside → OCCUPIED, **bay markings ignored**, **bounding-box corner ignored
(polygon, not rectangle)**, JPEG-like noise stays EMPTY, partial occupancy lands
between, thresholds respected, pixel threshold sensitivity, inner-scale effect,
resolution change refused rather than silently mis-measured, and two bays
independent.

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
│   ├── spot_config.py             bay polygons, inner-ROI maths, JSON I/O
│   ├── occupancy.py               the difference algorithm and its parameters
│   ├── calibration.py             click-the-corners tool + grid ruler
│   └── main.py                    live viewer / headless recorder
└── tests/
    ├── test_occupancy.py          synthetic tests
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
| `frame geometry changed` error | the firmware's stream resolution changed; re-capture the reference |
