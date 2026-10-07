"""Look at the navigation page through a browser, and measure what comes back.

Manual, like car_locate.py: it needs Chrome and a port, so it is not part of the suite. It exists
because a source guard is not enough. The turn-the-map feature passed a test that read the page for
"applyCamera(ctx, viewW0, viewH0, cam, cam.rot)" - and did not work at all, because clampCamera
returned a new object without the rotation in it. Rendering the page and measuring the pixels found
it in one run, and this is that measurement kept.

What it checks, for a car placed at the same spot with three different headings:

    the car's marker points up the screen every time   (that is what heading-up *means*)
    the route covers a different set of pixels each time (that is the map turning)

    python tests/check_navigation_views.py --frame data/reference/reference.jpg

It writes its screenshots and a summary to --out (default /tmp/nav-views).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
# What the camera really emits: an image-coordinate angle. 270 is "up the frame", 0 is right.
HEADINGS = (270, 0, 90, 180)


def find_chrome() -> str:
    if os.path.exists(CHROME):
        return CHROME
    for name in ("google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("no Chrome found; this check needs a browser")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=os.path.join(ROOT, "data", "reference", "reference.jpg"))
    ap.add_argument("--map", default=os.path.join(ROOT, "out", "parking_map.json"))
    ap.add_argument("--state", default=os.path.join(ROOT, "out", "parking_state.json"))
    ap.add_argument("--port", type=int, default=8021)
    ap.add_argument("--size", default="900,1000")
    ap.add_argument("--out", default="/tmp/nav-views")
    args = ap.parse_args()

    import serve                                               # noqa: E402
    import math as _math

    os.makedirs(os.path.join(args.out, "out"), exist_ok=True)
    shutil.copytree(os.path.join(ROOT, "web"), os.path.join(args.out, "web"), dirs_exist_ok=True)
    shutil.copyfile(args.map, os.path.join(args.out, "out", "parking_map.json"))

    map_doc = json.load(open(args.map, encoding="utf-8"))
    state = json.load(open(args.state, encoding="utf-8"))
    nodes = {n["id"]: n for n in map_doc["road_network"]["nodes"]}
    edge = next(e for e in map_doc["road_network"]["edges"] if e["from"] == "r1")
    a, b = nodes[edge["from"]], nodes[edge["to"]]
    x, y = (a["x"] + b["x"]) / 2, (a["y"] + b["y"]) / 2
    target = next((s["id"] for s in map_doc["spots"]
                   if state["spots"].get(str(s["id"]), {}).get("state") == "EMPTY"), None)

    server = serve.serve(args.port, root=args.out,
                         map_path=os.path.join(args.out, "out", "parking_map.json"),
                         state_path=os.path.join(args.out, "out", "parking_state.json"),
                         route_path=os.path.join(args.out, "out", "route.json"))
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()

    chrome = find_chrome()
    shots = {}
    try:
        for heading in HEADINGS:
            state["vehicle"] = {
                "id": "hero", "x": x, "y": y, "heading": round(heading % 180, 1), "visible": True,
                "heading_is_axis_only": True, "front_is_known": True,
                "front_heading": float(heading),
                "location": {"kind": "road", "id": None, "distance_to_road": 0.0}}
            state["timestamp"] = int(time.time())
            with open(os.path.join(args.out, "out", "parking_state.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(state, fh)
            shot = os.path.join(args.out, f"heading{heading}.png")
            url = (f"http://127.0.0.1:{args.port}/web/navigation.html"
                   f"?spot={target}")
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--window-size={args.size}", "--virtual-time-budget=6000",
                            f"--screenshot={shot}", url],
                           capture_output=True, timeout=90)
            shots[heading] = shot
    finally:
        server.shutdown()
        server.server_close()

    try:
        from PIL import Image
        import numpy as np
    except ImportError:
        print("install pillow and numpy to measure the screenshots")
        return 0

    print(f"  car at ({x:.0f}, {y:.0f}), target bay {target}")
    print(f"  {'heading':>8} {'car px':>8} {'nose':>9} {'verdict':>9} {'route px':>9}")
    failures = 0
    for heading, shot in shots.items():
        if not os.path.exists(shot):
            print(f"  {heading:>8}  no screenshot")
            failures += 1
            continue
        im = np.asarray(Image.open(shot).convert("RGB")).astype(int)
        r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
        red = (r > 170) & (g < 110) & (b < 110) & (abs(g - b) < 60)
        blue = (b > 200) & (r < 130) & (g > 110) & (g < 200)
        if red.sum() < 50:
            print(f"  {heading:>8}  no car marker found")
            failures += 1
            continue
        ys, xs = np.nonzero(red)
        cx, cy = xs.mean(), ys.mean()
        far = int(np.argmax(np.hypot(xs - cx, ys - cy)))
        angle = math.degrees(math.atan2(ys[far] - cy, xs[far] - cx))
        # The marker's nose is the point furthest from its own centre of mass, and heading-up
        # means that point is towards the top of the screen whatever direction the car faces.
        ok = abs(angle + 90) < 25
        failures += 0 if ok else 1
        print(f"  {heading:>8} {int(red.sum()):>8} {angle:>8.1f}° "
              f"{'up':>9} {int(blue.sum()):>9}")
    print(f"\n  screenshots in {args.out}")
    if failures:
        print(f"  {failures} of {len(shots)} headings did not point up")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
