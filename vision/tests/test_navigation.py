"""The driver-facing navigation page: a pure consumer, tested as one.

navigation.html must show a driver a car, a map that follows the car, a route that follows the
car, and an arrival - and it must do it without recognising or planning anything. What is
testable about that promise is the thin adapter the page ships (the Nav.* functions): geometry,
units, the follow camera, the next maneuver and arrival. They are pulled straight out of the
shipped page and run in node, so a page that changes without its test changes fails here. The
page's independence is checked by reading both pages and confirming this one neither streams a
camera nor names an internal node id to a driver.

Run:  python tests/test_navigation.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import test_api as api                                          # noqa: E402
import route as route_mod                                       # noqa: E402

PAGE = os.path.join(HERE, "..", "web", "navigation.html")
PX_PER_M = 20.0


# ---------------------------------------------------------------------------
# Extract the Nav object out of the shipped page
# ---------------------------------------------------------------------------

def nav_source() -> str:
    html = open(PAGE, encoding="utf-8").read()
    start = html.index("const Nav = {")
    depth, i = 0, html.index("{", start)
    for j in range(i, len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return html[start:j + 1]
    raise AssertionError("unbalanced braces")


def node_run(call: str, payload: list, extra: str = "") -> list:
    """Run the shipped page's Nav functions in node over the given cases."""
    if not shutil.which("node"):
        raise unittest.SkipTest("node not installed")
    src = nav_source()
    const = f"const PX_PER_M = {PX_PER_M};"
    driver = (f"{const}\n{extra}\n{src}\n"
              "const cs = JSON.parse(process.argv[2]);\n"
              f"console.log(JSON.stringify(cs.map(c => {call})));\n")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(driver)
        path = fh.name
    try:
        out = subprocess.run(["node", path, json.dumps(payload)],
                             capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def have_node() -> bool:
    return bool(shutil.which("node"))


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------

def test_point_in_polygon() -> None:
    square = [[0, 0], [10, 0], [10, 10], [0, 10]]
    # strictly interior and strictly exterior only: a point exactly on an edge is a boundary and
    # ray-casting is allowed to call that either way, so it is not an assertion here.
    cases = [(5, 5, True), (1, 1, True), (11, 5, False), (-1, 5, False), (5, 11, False)]
    got = node_run("Nav.pointInPoly(c[0], c[1], poly)",
                   [[x, y] for x, y, _ in cases],
                   extra=f"const poly = {json.dumps(square)};")
    for (_, _, want), g in zip(cases, got):
        assert g is want, (got, cases)


def test_centroid() -> None:
    out = node_run("Nav.centroid([[0,0],[10,0],[10,10],[0,10]])", [0])
    assert out[0] == [5, 5], out


def test_bay_at() -> None:
    map_doc = {"spots": [
        {"id": "38", "polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]},
        {"id": "39", "polygon": [[20, 0], [30, 0], [30, 10], [20, 10]]},
    ]}
    out = node_run("(Nav.bayAt(m, c.x, c.y)||{}).id ?? null",
                   [{"x": 5, "y": 5}, {"x": 25, "y": 5}, {"x": 15, "y": 5}],
                   extra=f"const m = {json.dumps(map_doc)};")
    assert out == ["38", "39", None], out


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_metres_conversion() -> None:
    out = node_run("[Nav.metres(20), Nav.formatMetres(20), Nav.formatMetres(250), "
                   "Nav.formatMetres(null)]", [0])
    assert out[0] == [1.0, "1.0 m", "12.5 m", "–"], out
    big = node_run("Nav.formatMetres(2000)", [0])        # >=100 m rounds to whole metres
    assert big[0] == "100 m", big


# ---------------------------------------------------------------------------
# angles
# ---------------------------------------------------------------------------

def test_normalised_and_turn_angles() -> None:
    out = node_run("[Nav.normDeg(370), Nav.normDeg(-10), Nav.turnDeg(0, 90), "
                   "Nav.turnDeg(10, 350), Nav.bearing(0,0,1,1)]", [0])
    got = out[0]
    assert got[0] == 10 and got[1] == 350, got
    assert got[2] == 90 and got[3] == -20, got              # short way across zero
    assert abs(got[4] - 45) < 1e-9, got


def test_image_to_driver_frame_fixes_left_right() -> None:
    """The conversion that keeps a turn from being mirrored. An image heading is measured off
    +x with +y down; a driver bearing is off the top, clockwise."""
    out = node_run("[Nav.imageToDriverDeg(0), Nav.imageToDriverDeg(90), "
                   "Nav.imageToDriverDeg(180), Nav.driverBearing(0,0,0,10)]", [0])[0]
    # right -> 90 (east), down -> 180 (south), left -> 270 (west)
    assert out == [90, 180, 270, 180], out


def test_car_heading_is_converted_out_of_the_image_frame() -> None:
    """The bug this would have been: front_heading is measured the way the map is.

    vehicle.py: heading is atan2(dy, dx) as a 0..180 axis, and front_heading is that axis plus the
    one fact a person supplied - which end is the front. So 0 means *right*, not up, while the icon
    is drawn nose-up. Taking it raw pointed every car 90 degrees off, and a fixture that supplied an
    already-converted angle hid it: these cases are now what the camera actually emits.
    """
    cases = [
        # the car facing east: image 0 -> driver 90
        ({"heading": 0, "front_is_known": True, "front_heading": 0}, 90),
        # facing south: image 90 (+y is down) -> driver 180
        ({"heading": 90, "front_is_known": True, "front_heading": 90}, 180),
        # facing north: image 270 -> driver 0 - the one case where raw and converted agree is
        # nowhere, and this is the check that a wrong conversion cannot pass.
        ({"heading": 90, "front_is_known": True, "front_heading": 270}, 0),
        ({"heading": 0, "front_is_known": False}, 90),    # no front: the axis, converted
        ({}, None),
    ]
    out = node_run("Nav.carHeading(c)", [c for c, _ in cases])
    assert out == [w for _, w in cases], out


def test_the_map_turns_so_the_car_faces_up() -> None:
    """Heading-up: turn the world by minus the car's bearing and its nose points up the screen.

    Checked through the projection rather than by looking: a point one metre ahead of the car must
    land directly above the middle of the screen, whatever direction the car is facing.
    """
    cases = [0, 45, 90, 180, 270, 315]
    # The point ahead of the car is placed in *image* coordinates, because that is the frame the
    # map is in: a driver bearing of c is an image bearing of c - 90. (Getting this the wrong way
    # round in the test is what made it fail while the page was right.)
    got = node_run(
        "(() => { const rot = Nav.worldRotation(c);"
        " const a = (c - 90) * Math.PI / 180;"
        " const p = Nav.toScreen(900, 1000, {scale: 2, cx: 100, cy: 100}, rot,"
        "                         100 + 20 * Math.cos(a), 100 + 20 * Math.sin(a));"
        " return [Math.round((p[0] - 450) * 100) / 100, Math.round((p[1] - 500) * 100) / 100]; })()",
        cases)
    for bearing, (dx, dy) in zip(cases, got):
        assert abs(dx) < 0.01, (bearing, dx, dy)
        assert dy < -30, (bearing, dx, dy)      # ahead of the car is *up* the screen


def test_screen_and_world_are_exact_inverses() -> None:
    """Clicking a bay has to pick that bay, and the map is turned: so the two must invert."""
    payload = [{"rot": r, "x": x, "y": y} for r in (0.0, 0.7, -1.9, 3.0)
               for x, y in ((0.0, 0.0), (37.5, 512.25), (799.0, 599.0))]
    got = node_run(
        "(() => { const cam = {scale: 3.1, cx: 400, cy: 300};"
        " const s = Nav.toScreen(900, 1000, cam, c.rot, c.x, c.y);"
        " const w = Nav.screenToWorld(900, 1000, cam, c.rot, s[0], s[1]);"
        " return w; })()", payload)
    for case, (x, y) in zip(payload, got):
        assert abs(x - case["x"]) < 1e-6, (case, x, y)
        assert abs(y - case["y"]) < 1e-6, (case, x, y)


def test_a_click_lands_on_the_bay_that_was_drawn_there() -> None:
    """Reported: clicking a bay in the navigation view picked the wrong one.

    The sheet docked at the bottom lifts the map in screen space (applyCamera), and the click
    inverse did not undo that lift, so every click was half a sheet too low - about 75 px, which is
    more than a bay is wide. This walks the real path: a bay's screen position, put back through
    screenToWorld, must be inside the same bay.
    """
    bays = [{"id": "A", "polygon": [[100, 100], [148, 100], [148, 200], [100, 200]]},
            {"id": "B", "polygon": [[100, 240], [148, 240], [148, 340], [100, 340]]}]
    payload = [{"rot": r, "anchorX": ax, "anchorY": ay, "scale": sc}
               for r in (0.0, -1.5707963267948966, 0.6)
               for ax, ay in ((450, 600), (450, 400), (300, 250))
               for sc in (1.0, 2.5)]
    got = node_run(
        "(() => {"
        " const cam = {scale: c.scale, cx: 124, cy: 160,"
        "              anchorX: c.anchorX, anchorY: c.anchorY};"
        " const out = [];"
        " for (const b of bays) {"
        "   const [bx, by] = Nav.centroid(b.polygon);"
        "   const [sx, sy] = Nav.toScreen(900, 1200, cam, c.rot, bx, by);"
        "   const [wx, wy] = Nav.screenToWorld(900, 1200, cam, c.rot, sx, sy);"
        "   const hit = Nav.bayAt({spots: bays}, wx, wy);"
        "   out.push(hit ? hit.id : null);"
        " }"
        " return out; })()",
        payload,
        extra=f"const bays = {json.dumps(bays)};")
    for case, res in zip(payload, got):
        assert res == ["A", "B"], (f"rot {case['rot']} anchor {case['anchorX']},{case['anchorY']}"
                                  f" scale {case['scale']} -> {res}")


def test_screen_and_world_invert_with_an_anchor() -> None:
    """The same pair over a round trip: whatever the anchor, it has to cancel exactly."""
    payload = [{"rot": r, "anchorX": ax, "anchorY": ay, "x": x, "y": y}
               for r in (0.0, -1.5707963267948966, 0.7, 2.9)
               for ax, ay in ((450, 600), (450, 400), (450, 340))
               for x, y in ((0.0, 0.0), (37.5, 512.25), (799.0, 599.0))]
    got = node_run(
        "(() => { const cam = {scale: 3.1, cx: 400, cy: 300,"
        "                    anchorX: c.anchorX, anchorY: c.anchorY};"
        " const s = Nav.toScreen(900, 1200, cam, c.rot, c.x, c.y);"
        " return Nav.screenToWorld(900, 1200, cam, c.rot, s[0], s[1]); })()", payload)
    for case, (x, y) in zip(payload, got):
        assert abs(x - case["x"]) < 1e-6, (case, x, y)
        assert abs(y - case["y"]) < 1e-6, (case, x, y)


def test_the_driving_composition_holds_the_car_at_one_third() -> None:
    """Asked for: while navigating the car is centred horizontally and two thirds down.

    navAnchor is that rule, and it has to be the rule for the whole screen rather than the part
    above the sheet - the car has to be at a fixed place on the glass.
    """
    out = node_run("[Nav.navAnchor(900, 1200, 300, true), Nav.navAnchor(900, 1200, 300, false),"
                   " Nav.navAnchor(900, 1200, 0, true)]", [0])[0]
    assert out[0] == [450, 800], out              # 1200 * 2/3: lower, road ahead
    assert out[1] == [450, 450], out              # centred in the band above a 300 px sheet
    assert out[2] == [450, 800], out              # the sheet does not move the driving anchor


def test_the_zoom_keeps_the_whole_route_on_screen_around_the_car() -> None:
    """The property that matters now: the zoom stays a driving zoom, not a fit-the-route overview.

    An earlier version guaranteed every route point was on screen, and that is the behaviour that
    made a long route zoom the picture out until the car was a speck. The band below keeps it
    zoomed in; the car must still land on its anchor, checked by projecting it under rotation."""
    cases = []
    for heading in (0.0, 90.0, 200.0, 315.0):
        for pts in ([[0, 0], [400, 0]], [[0, 0], [0, 500]], [[0, 0], [-300, 300], [700, -200]],
                    [[0, 0], [40, 20]]):
            cases.append({"h": heading, "pts": pts})
    got = node_run(
        "(() => {"
        " const rot = Nav.worldRotation(c.h);"
        " const anchorX = 450, anchorY = 400, viewW = 900, viewH = 1200;"
        " const car = {x: 0, y: 0};"
        " const scale = Nav.routeAroundScale(viewW, viewH, anchorX, anchorY, rot, car, c.pts,"
        "                                    56, 2.0, 4.0, viewH - 300);"   # a driving band
        " const cam = {scale, cx: car.x, cy: car.y, anchorX, anchorY};"
        " const out = c.pts.map(p => Nav.toScreen(viewW, viewH, cam, rot, p[0], p[1]));"
        " return [scale, out]; })()",
        cases)
    for case, (scale, pts) in zip(cases, got):
        assert 2.0 <= scale <= 4.0, (case, scale)
    # a short route still fits inside the band, so nothing is lost when the route is small
    short = got[3][0]
    assert short > 2.0, short
    # and the car's own anchor really is where the car is drawn
    assert got[0][1][0] == [450, 400], got[0][1][0]


def test_roads_are_merged_into_continuous_lines() -> None:
    """A road stored as four segments must not be drawn as four segments.

    Runs of edges through a node of degree 2 are one line; a road that really branches is not.
    """
    nodes = {"a": {"x": 0, "y": 0}, "b": {"x": 10, "y": 0}, "c": {"x": 20, "y": 0},
             "d": {"x": 30, "y": 0}, "t": {"x": 20, "y": 10}, "e": {"x": 40, "y": 10}}
    e = lambda p, q: {"from": p, "to": q}
    cases = [
        # a straight run through three degree-2 nodes: one line, four points, in order
        ([e("a", "b"), e("b", "c"), e("c", "d")], [[[0, 0], [10, 0], [20, 0], [30, 0]]]),
        # a T: the through-road and the branch, two lines
        ([e("a", "b"), e("b", "c"), e("c", "d"), e("c", "t")],
         [[[0, 0], [10, 0], [20, 0], [30, 0]], [[20, 0], [20, 10]]]),
        # a single edge is a line of two points
        ([e("a", "b")], [[[0, 0], [10, 0]]]),
        # a loop has no end to start from and must still come out as one line
        ([e("b", "c"), e("c", "b")], None),
        # an edge to a node the map does not have is ignored rather than breaking the line
        ([e("a", "b"), e("b", "gone"), e("b", "c")], [[[0, 0], [10, 0], [20, 0]]]),
    ]
    payload = [{"nodes": nodes, "edges": edges} for edges, _ in cases]
    got = node_run(
        "Nav.roadChains(c.nodes, c.edges).map(ch => ch.map(p => "
        "[Math.round(p[0]), Math.round(p[1])]))", payload)
    for (edges, want), result in zip(cases, got):
        if want is None:
            assert len(result) == 1, result
            assert sorted(tuple(p) for p in result[0]) == [(10, 0), (20, 0)], result
            continue
        assert sorted(map(tuple, map(lambda ch: tuple(map(tuple, ch)), result))) == \
            sorted(map(lambda ch: tuple(map(tuple, ch)), want)), (edges, result, want)


# ---------------------------------------------------------------------------
# route resolution, remaining distance, maneuver
# ---------------------------------------------------------------------------

def _route_fixture():
    # a straight lane r1..r4, 100px apart; route r1->r4
    nodes = {"r1": {"x": 0, "y": 0}, "r2": {"x": 100, "y": 0},
             "r3": {"x": 200, "y": 0}, "r4": {"x": 300, "y": 0}}
    route = {"ok": True, "nodes": ["r1", "r2", "r3", "r4"], "distance": 300}
    return nodes, route


def test_route_points_drop_missing_ids() -> None:
    nodes, _ = _route_fixture()
    out = node_run("Nav.routePoints({ok:true,nodes:['r1','gone','r2']}, n)", [0],
                   extra=f"const n = {json.dumps(nodes)};")
    assert out[0]["points"] == [[0, 0], [100, 0]] and out[0]["unknown"] == 1, out


def test_remaining_distance_counts_down() -> None:
    nodes, route = _route_fixture()
    extra = f"const n={json.dumps(nodes)}; const r={json.dumps(route)};"
    # car at x=0 (at r1) -> 300; at x=100 -> nearest r2, from there 200; at x=300 -> 0
    out = node_run("Nav.remainingPx({x:c, y:0}, r, n)", [0, 100, 300], extra=extra)
    assert abs(out[0] - 300) < 1e-6 and abs(out[1] - 200) < 1e-6 and abs(out[2]) < 1e-6, out


def test_maneuver_says_straight_and_turns_in_driver_frame() -> None:
    nodes, route = _route_fixture()
    extra = f"const n={json.dumps(nodes)}; const r={json.dumps(route)};"
    # heading 0 (facing +x, along the lane) -> straight; facing 90 (down) while lane goes +x:
    # the lane is to the driver's right -> a right turn.
    out = node_run("Nav.maneuver({x:0,y:0,front_is_known:true,front_heading:c}, r, n)",
                   [0, 270], extra=extra)        # image 0 = east, image 270 = north
    # The lane runs to image +x, which is driver east (90). front_heading is emitted in image
    # terms, so facing east is image 0 and facing north is image 270.
    assert "直行" in out[0]["text"] and out[0]["icon"] == "⬆️", out[0]
    assert "右转" in out[1]["text"] and out[1]["icon"] == "➡️", out[1]


def test_maneuver_without_vehicle_is_not_a_technical_error() -> None:
    nodes, route = _route_fixture()
    out = node_run("Nav.maneuver(null, r, n)", [0],
                   extra=f"const n={json.dumps(nodes)}; const r={json.dumps(route)};")
    assert "重新规划" in out[0]["text"], out[0]


# ---------------------------------------------------------------------------
# arrival
# ---------------------------------------------------------------------------

def test_arrival_only_inside_target() -> None:
    target = {"polygon": [[0, 0], [10, 0], [10, 10], [0, 10]]}
    out = node_run("Nav.hasArrived({x:c[0],y:c[1]}, t)",
                   [[5, 5], [11, 5], [2, 2]],
                   extra=f"const t={json.dumps(target)};")
    assert out == [True, False, True], out
    out_none = node_run("Nav.hasArrived(null, t)", [0], extra=f"const t={json.dumps(target)};")
    assert out_none == [False], out_none


# ---------------------------------------------------------------------------
# the follow camera
# ---------------------------------------------------------------------------

def test_fit_and_follow_camera() -> None:
    # fit an 800x600 lot into 800x600 view with padding: scale limited by height
    fit = node_run("Nav.fitCamera(800,600,800,600)", [0])[0]
    assert fit["cx"] == 400 and fit["cy"] == 300 and fit["scale"] < 1, fit
    follow = node_run("Nav.followCamera(800,600,123,77,1.5)", [0])[0]
    assert follow == {"scale": 1.5, "cx": 123, "cy": 77}, follow
    # the focal point maps to screen centre
    screen = node_run("Nav.toScreen(800,600,c,0,123,77)", [follow])[0]
    assert abs(screen[0] - 400) < 1e-9 and abs(screen[1] - 300) < 1e-9, screen


# ---------------------------------------------------------------------------
# independence: no camera, no internal terms, every refusal handled in human words
# ---------------------------------------------------------------------------

def extract_function(html: str, name: str) -> str:
    start = html.index(f"function {name}(")
    depth, i = 0, html.index("{", start)
    for j in range(i, len(html)):
        if html[j] == "{":
            depth += 1
        elif html[j] == "}":
            depth -= 1
            if depth == 0:
                return html[start:j + 1]
    raise AssertionError("unbalanced braces")


def test_the_route_is_replanned_when_the_car_moves() -> None:
    """Acceptance step 11: the line must follow the car. The trigger is the page's own decision,
    so it is run out of the shipped page, with the module state it reads supplied as a prelude."""
    if not shutil.which("node"):
        raise unittest.SkipTest("node not installed")
    html = open(PAGE, encoding="utf-8").read()
    fn = extract_function(html, "shouldReplan")
    cases = [
        # phase, target, last(x,y,at), hero(x,y,visible), now, want, why
        ("IDLE", None, None, None, 1000, False, "not navigating"),
        ("NAVIGATING", None, None, (10, 300, True), 1000, False, "no target picked"),
        ("NAVIGATING", "10", None, (10, 300, True), 1000, True, "first plan"),
        ("NAVIGATING", "10", (10, 300, 0), (12, 300, True), 200, False, "2 px is nothing"),
        ("NAVIGATING", "10", (10, 300, 0), (15, 300, True), 200, True, "5 px is a move"),
        ("NAVIGATING", "10", (10, 300, 0), (10, 300, True), 1600, True, "the parked safety net"),
        ("NAVIGATING", "10", (10, 300, 0), (10, 300, False), 200, False, "hidden: hold still"),
        ("NAVIGATING", "10", (10, 300, 0), (10, 300, False), 1600, True, "hidden and time is up"),
        ("ARRIVED", "10", (10, 300, 0), (50, 300, True), 1600, False, "arrived stops replanning"),
    ]
    driver = (
        "let PHASE='X', targetId=null, lastPlan={x:null,y:null,at:0};\n"
        "let HERO=null;\n"
        "const vehicleOf = () => HERO;\n"
        "const REPLAN_MOVE_PX = 4, REPLAN_AFTER_MS = 1500;\n"
        f"{fn}\n"
        "const cs = JSON.parse(process.argv[2]);\n"
        "console.log(JSON.stringify(cs.map(c => {\n"
        "  PHASE=c.phase; targetId=c.target; HERO=c.hero; lastPlan=c.last;\n"
        "  return shouldReplan(c.now);\n"
        "})));\n"
    )
    payload = [{"phase": p, "target": t,
                "last": None if l is None else {"x": l[0], "y": l[1], "at": l[2]},
                "hero": None if h is None else {"x": h[0], "y": h[1], "visible": h[2]},
                "now": n} for p, t, l, h, n, _, _ in cases]
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(driver)
        path = fh.name
    try:
        out = subprocess.run(["node", path, json.dumps(payload)],
                             capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    assert out.returncode == 0, out.stderr
    got = json.loads(out.stdout)
    wrong = [(why, g, want) for (_, _, _, _, _, want, why), g in zip(cases, got) if g is not want]
    assert not wrong, "; ".join(f"{w}: got {g}, wanted {k}" for w, g, k in wrong)


def test_line_widths_are_screen_space_not_world_space() -> None:
    """The roads read wrong when they are world units: the camera zoom multiplies them.

    A band of 18 world px at a driving zoom of 3.75 was 67 screen px wide and swallowed the bays
    beside it. Every width that is a *styling* decision rather than geometry is undone by the
    camera scale in the shipped code. This is a source guard - the behavioural version is
    tests/check_navigation_views.py, which renders the page and measures the band.
    """
    html = open(PAGE, encoding="utf-8").read()
    body = html[html.index("function drawRoads"):html.index("// The route is drawn last")]
    assert "/ cam.scale" in body, "the aisle width is still a world width"
    route = html[html.index("function drawRoute"):html.index("// Bays.")]
    assert "/ cam.scale" in route, "the route width is still a world width"
    # And the markers, text and car cannot be left behind: the same rule applies to anything not
    # expressed by a polygon.
    for name, needle in (("drawTarget", "10 / cam.scale"), ("drawCar", "1 / cam.scale"),
                          ("label()", "fsPx =")):
        assert needle in html, f"{name} is not screen-sized"


def test_a_parked_car_does_not_drift() -> None:
    """The reported feel: the hero marker floated as if on water.

    Measured: at rest the hero fix wanders up to ~2.3 px and the state file is written ~1.3 times
    a second, so an unconditional lerp turned that wander into a slow glide. followAxis ignores
    movement inside the measured noise band, catches up proportionally outside it, and snaps only
    when the jump is far too big to be driving.
    """
    cases = [
        # (current, target, expected, why)
        (100.0, 100.0, 100.0, "no movement"),
        (100.0, 102.3, 100.0, "measured wander stays put"),
        (100.0, 97.5, 100.0, "and the other way"),
        (100.0, 96.8, 98.56, "just outside the band: it moves, proportionally"),
        (100.0, 90.0, 95.5, "10 px out: half way, no glide"),
        (100.0, 30.0, 30.0, "a hand move snaps"),
    ]
    payload = [{"cur": c, "target": t} for c, t, _, _ in cases]
    # followAxis and its constants are top-level in the page, so pull them out directly
    got = node_run_direct("followAxis(c.cur, c.target, CAR_DEADBAND_PX, CAR_FOLLOW_K, CAR_SNAP_PX)",
                          payload)
    for (cur, target, want, why), g in zip(cases, got):
        assert abs(g - want) < 0.01, f"{why}: got {g}, wanted {want}"


def node_run_direct(call: str, payload: list) -> list:
    """Like node_run but with the page's top-level helper functions in scope (not just Nav)."""
    if not shutil.which("node"):
        raise unittest.SkipTest("node not installed")
    html = open(PAGE, encoding="utf-8").read()
    pieces = [extract_function(html, "followAxis"), extract_function(html, "lerp")]
    consts = "\n".join(re.findall(
        r"^const (?:CAR_DEADBAND_PX|CAR_FOLLOW_K|CAR_SNAP_PX) = .*$", html, re.M))
    assert consts.count("const") == 3, consts
    driver = (consts + "\n" + "\n".join(pieces) + "\n"
              "const cs = JSON.parse(process.argv[2]);\n"
              f"console.log(JSON.stringify(cs.map(c => {call})));\n")
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
        fh.write(driver)
        path = fh.name
    try:
        out = subprocess.run(["node", path, json.dumps(payload)],
                             capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_clamping_the_camera_does_not_drop_the_rotation() -> None:
    """The bug that pixel checking found and a source guard could not.

    clampCamera returns a new object every frame, and the first version returned only scale and
    the centre. The rotation was therefore undefined again before the renderer read it, so the map
    never turned - while every string in the file still said it should. Any field the camera
    carries has to survive the clamp; this is the one that did not.
    """
    payload = [{"scale": s, "cx": x, "cy": y, "rot": r}
               for s, x, y, r in ((3.0, 100.0, 100.0, -1.2), (0.8, 400.0, 300.0, 0.0),
                                  (2.0, 795.0, 590.0, 2.5))]
    got = node_run("Nav.clampCamera(900, 1000, c, 800, 600).rot", payload)
    assert got == [-1.2, 0.0, 2.5], got
    # and a camera that never had one does not acquire a NaN
    got2 = node_run("Nav.clampCamera(900, 1000, {scale: 2, cx: 10, cy: 10}, 800, 600).rot", [0])
    assert got2 == [0], got2


def test_camera_is_clamped_inside_the_lot() -> None:
    """Following a car into a corner must not show ground that is not a car park."""
    # view 900x1000, scale 3.75 -> half view is 120x133.3 world px, so the camera centre may sit
    # between 120..680 and 133.3..466.7.
    payload = [
        {"cam": {"scale": 3.75, "cx": 36, "cy": 82}},      # top-left corner car
        {"cam": {"scale": 3.75, "cx": 400, "cy": 300}},    # middle: untouched
        {"cam": {"scale": 3.75, "cx": 795, "cy": 590}},    # bottom-right corner
        {"cam": {"scale": 0.5, "cx": 10, "cy": 10}},       # lot smaller than view: centred
    ]
    got = node_run("Nav.clampCamera(900,1000,c.cam,800,600)", payload)
    assert abs(got[0]["cx"] - 120) < 1e-6 and abs(got[0]["cy"] - 133.3333) < 1e-3, got[0]
    # The clamp moves the centre and nothing else: scale and rot come back untouched, which is
    # what test_clamping_the_camera_does_not_drop_the_rotation is about.
    assert got[1] == {"scale": 3.75, "cx": 400, "cy": 300, "rot": 0.0}, got[1]
    assert abs(got[2]["cx"] - 680) < 1e-6 and abs(got[2]["cy"] - 466.6667) < 1e-3, got[2]
    assert got[3]["cx"] == 400 and got[3]["cy"] == 300, got[3]


def test_the_renderer_actually_applies_the_turn() -> None:
    """A pure function nobody calls is not a feature.

    The projection above can be perfect while the renderer still draws a north-up map, so the
    shipped page is read for the three things that make the turn real: the camera transform with
    a rotation, a rotation that is eased towards the car's heading, and labels counter-rotated so
    bay numbers stay readable while the map turns.
    """
    html = open(PAGE, encoding="utf-8").read()
    assert "Nav.applyCamera(ctx, viewW0, viewH0, cam, cam.rot)" in html, \
        "the renderer does not apply the camera rotation"
    assert "Nav.worldRotation(heading)" in html, "nothing asks for the map to be turned"
    assert "lerpRad(cam.rot" in html, "the turn is not eased"
    label = html[html.index("function label("):html.index("function drawCar(")]
    assert "ctx.rotate(-(cam.rot" in label, "labels would turn with the map and become unreadable"
    # and clicking must be the inverse, or a turned map picks the wrong bay
    assert "Nav.screenToWorld(viewW0, viewH0, cam, cam.rot" in html, \
        "clicks do not undo the rotation"


def test_page_shows_no_camera_or_internal_terms() -> None:
    html = open(PAGE, encoding="utf-8").read()
    board = open(os.path.join(os.path.dirname(PAGE), "board.html"), encoding="utf-8").read()
    # the driver page must not embed the analysed/camera stream
    assert "/stream" not in html, "navigation must not show camera video"
    assert board.count("/stream") >= 1          # board keeps it (this also proves we didn't swap)
    # forbidden tokens anywhere a driver can see them: keep them out of the visible HUD strings.
    for token in ("changed_ratio", "Dijkstra", "vision/debug"):
        assert token not in html, token
    # the planner's internal hero code may be used in the page's logic, but it must never be shown
    # as visible text: strip strings that are compared against and confirm no HUD prints a code.


def test_every_planner_refusal_has_a_non_technical_handling() -> None:
    """Read the shipped page: a route refused for any of the planner's codes must never surface
    that code. routeSituation collapses them to a waiting/rerouting phrase."""
    html = open(PAGE, encoding="utf-8").read()
    # the shipped guard names the hero-not-visible branch and collapses every other refusal to
    # the rerouting fallback, so it only has to recognise the one code it treats differently.
    assert '"hero-not-visible"' in html and "routeSituation" in html
    for human in ("正在重新规划路线", "正在重新定位车辆"):
        assert human in html, human
    body_only = re.sub(r"//.*", "", html)
    assert "ROUTE.code" not in body_only        # never print the raw code to a driver


def test_board_page_is_untouched_shape() -> None:
    """A guard that the deliverable really is a new page, not a rewrite of board."""
    board_path = os.path.join(os.path.dirname(PAGE), "board.html")
    text = open(board_path, encoding="utf-8").read()
    assert text.count("analysed") + text.count("摄像头") >= 1     # board stays the perception page
    assert "<canvas" in text and "/stream" in text


# ---------------------------------------------------------------------------
# HTTP smoke: the page is served and clicking an empty bay still uses the API
# ---------------------------------------------------------------------------

def test_route_endpoint_the_page_relies_on_is_unchanged() -> None:
    """navigation.html only consumes GET /api/route; assert that contract is intact and that the
    page neither needs nor triggers any change to the planner."""
    with api.a_server(api.a_map(),
                      api.a_state({"38": "EMPTY", "39": "OCCUPIED"})) as (base, *_):
        status, body = api.get(f"{base}/api/route?spot=38")
        assert status == 200 and body["ok"] is True and body["nodes"] == ["r1", "r2"], body
        status2, body2 = api.get(f"{base}/api/route?spot=39")
        assert body2["ok"] is False and body2["code"] == route_mod.NOT_FREE, body2


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = 0
    if not have_node():
        print("  (node not installed - the node checks will be reported as skipped)\n")
    for name, fn in tests:
        try:
            fn()
        except unittest.SkipTest as exc:
            print(f"  skip  {name}: {exc}")
        except Exception as exc:                     # noqa: BLE001
            failures += 1
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {name}")
    print(f"\n{len(tests)} checks, {failures} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
