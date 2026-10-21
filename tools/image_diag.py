"""Where resolution is actually lost: print the frame size at every stage.

The question this answers is specific - what shape does the algorithm really receive - and it
answers it by decoding the bytes rather than trusting a label. It reads the JPEG's own SOF
header (the size the sensor/encoder produced), then decodes it and reports what OpenCV gives,
then reports the same for the annotated relay, and finally measures edge sharpness so that
"soft" is a number rather than an impression.

    python tools/image_diag.py                     # default: the PC relay on :8021
    python tools/image_diag.py --url http://192.168.3.118/stream

Read-only: it fetches a few frames and prints. Nothing is configured or written.
"""
from __future__ import annotations

import argparse
import urllib.request

import cv2
import numpy as np


def jpeg_size(data: bytes):
    """Width and height from the JPEG's SOF marker - what the encoder actually produced."""
    i = 2
    while i < len(data) - 9:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return width, height
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = (data[i + 2] << 8) | data[i + 3]
        i += 2 + length
    return None


def grab(url: str, timeout: float = 15.0):
    handle = urllib.request.urlopen(url, timeout=timeout)
    buffer = b""
    try:
        while True:
            buffer += handle.read(16384)
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9", start)
            if start >= 0 and end > start:
                payload = buffer[start:end + 2]
                return payload, cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    finally:
        handle.close()
    raise RuntimeError("no frame from %s" % url)


def report(label, url):
    payload, frame = grab(url)
    size = jpeg_size(payload)
    print("  %-28s %s" % (label, url))
    print("    jpeg bytes                 %d" % len(payload))
    print("    jpeg SOF (what was encoded) %s" % (("%dx%d" % size) if size else "unreadable"))
    if frame is None:
        print("    decoded                    FAILED")
        return None
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("    frame.shape (what code sees) %s" % (frame.shape,))
    print("    grey mean %.1f   laplacian var %.1f"
          % (grey.mean(), cv2.Laplacian(grey, cv2.CV_64F).var()))
    # Sharpness of the sharpest 64x64 tile, so one empty corner cannot decide the verdict.
    best = 0.0
    h, w = grey.shape
    for y in range(0, max(1, h - 64), 64):
        for x in range(0, max(1, w - 64), 64):
            v = cv2.Laplacian(grey[y:y + 64, x:x + 64], cv2.CV_64F).var()
            best = max(best, v)
    print("    sharpest 64x64 tile        laplacian var %.1f" % best)
    return frame


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8021/stream")
    ap.add_argument("--annotated", default="http://127.0.0.1:8020/stream")
    args = ap.parse_args()

    print("stage by stage:")
    frame = report("PC relay (raw)", args.url)
    try:
        report("PC relay (annotated)", args.annotated)
    except Exception as exc:
        print("  annotated relay unreachable: %s" % exc)

    print()
    print("what the detector would receive:")
    if frame is not None:
        print("  frame.shape = %s   (this is the number that matters)" % (frame.shape,))
    print()
    print("reference points:")
    print("  640x480    VGA            - a 35 mm marker is ~20 px wide")
    print("  1280x720  720p            - ~40 px")
    print("  1920x1080 1080p           - ~60 px")
    print("  2592x1944 the OV5640's 5MP - ~80 px")
    print("  A 35 mm marker at 1.3 mm/px (the measured lot scale) is about 27 px wide,")
    print("  so anything at or below 1280 px wide leaves the marker at 40 px or less.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
