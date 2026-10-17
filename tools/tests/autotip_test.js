/* The annotation convention and the drawing units, as executable checks.

   Two things this has to catch, both of which have already gone wrong once:

   1. The convention: corner 1 is the car's RIGHT-FRONT corner and the rest follow clockwise, so
      p0->p1 runs along the car's right side pointing rearward and the front is the other way.
      Every label in vision/data/labels rests on that.

   2. The drawing units: Math.cos and Math.sin take RADIANS, and this page's own angleTo() returns
      DEGREES. Feeding one to the other draws an arrow that is merely wrong-looking rather than
      throwing, which is why the first version shipped with an automatic arrow that was not parallel
      to the first edge and a manual arrow that did not reach the clicked point. The first test
      written for this file checked the ANGLE - and the angle was right. It was the conversion to a
      unit vector that was wrong, so that is what is checked now.

   The functions are extracted verbatim from tools/label_ui.html: a copy would test the copy.

       node tools/tests/autotip_test.js                      (default: tools/label_ui.html)
       node tools/tests/autotip_test.js <path-to-ui.html>
*/
const fs = require("fs");
const path = require("path");

const ui = process.argv[2] || path.join(__dirname, "..", "label_ui.html");
const script = fs.readFileSync(ui, "utf8").match(/<script>([\s\S]*?)<\/script>/)[1];

for (const name of ["centre", "angleTo", "dirRadians", "sides", "autoTip", "unit",
                    "perpendicular", "snapRect", "reflect"]) {
  const m = script.match(new RegExp("function " + name + "\\([^)]*\\) \\{[\\s\\S]*?\\n\\}"));
  if (!m) {
    console.log(`  cannot find ${name}() in ${ui}`);
    process.exit(1);
  }
  eval(m[0]);
}

let failed = 0;
const check = (ok, line) => { if (!ok) failed++; console.log(`  ${ok ? "ok  " : "FAIL"} ${line}`); };

/* ---- 1. the convention: heading from the quad, four orientations ---- */
let quad = [];
const cars = {
  "nose up":    [[430,240],[430,360],[370,360],[370,240]],
  "nose right": [[460,330],[340,330],[340,270],[460,270]],
  "nose down":  [[370,360],[370,240],[430,240],[430,360]],
  "nose left":  [[340,270],[460,270],[460,330],[340,330]],
};
const want = { "nose up": 270, "nose right": 0, "nose down": 90, "nose left": 180 };
for (const [name, q] of Object.entries(cars)) {
  quad = q;
  const got = angleTo(centre(), autoTip());
  const err = Math.abs((got - want[name] + 540) % 360 - 180);
  check(err < 0.5, `${name}: heading ${got.toFixed(2)} (want ${want[name]}), ${err.toFixed(2)} deg out`);
}

/* the arrow must also be parallel to the first edge, not merely point the right way */
quad = cars["nose up"];
const edge = [quad[0][0] - quad[1][0], quad[0][1] - quad[1][1]];
const c = centre(), tip = autoTip();
const arrow = [tip[0] - c[0], tip[1] - c[1]];
const cross = Math.abs(edge[0] * arrow[1] - edge[1] * arrow[0]) /
              (Math.hypot(...edge) * Math.hypot(...arrow) || 1);
check(cross < 1e-9, `auto arrow is parallel to edge 1 (|sin| = ${cross.toExponential(1)})`);

/* ---- 2. the drawing units ---- */
const dirs = [0, 45, 90, 135, 180, 225, 270, 315, 271.3, 3.7];
let worstNew = 0, worstOld = 0;
for (const deg of dirs) {
  const from = [400, 300];
  const rad = dirs.length && deg * Math.PI / 180;
  const to = [from[0] + Math.cos(rad) * 50, from[1] + Math.sin(rad) * 50];
  const good = [Math.cos(dirRadians(from, to)), Math.sin(dirRadians(from, to))];
  const bad = [Math.cos(angleTo(from, to)), Math.sin(angleTo(from, to))];   // 把角度当弧度
  const wantVec = [Math.cos(rad), Math.sin(rad)];
  worstNew = Math.max(worstNew, Math.hypot(good[0] - wantVec[0], good[1] - wantVec[1]));
  worstOld = Math.max(worstOld, Math.hypot(bad[0] - wantVec[0], bad[1] - wantVec[1]));
}
check(worstNew < 1e-9, `dirRadians(): worst unit-vector error ${worstNew.toExponential(1)} over ${dirs.length} directions`);
check(worstOld > 0.1, `the old degrees-into-radians call is measurably wrong (${worstOld.toFixed(3)}), so this test would have caught it`);

/* ---- 3. the mistake the page warns about ---- */
quad = [[430,240],[370,240],[370,360],[430,360]];            // corner 2 across the width
const off = Math.abs((angleTo(centre(), autoTip()) - 270 + 540) % 360 - 180);
check(Math.abs(off - 90) < 5, `corner 2 on the width edge tilts the heading by ${off.toFixed(0)} deg - what the page warns about`);

/* ---- 4. the rectangle snap: 90 degrees by construction, not by measurement ---- */
const noisy = [[430, 240], [430, 360], [370, 362], [362, 358]];   // corners 3 and 4 clicked sloppily
const before = JSON.stringify(noisy);
const r = snapRect(noisy);
const seg = (i) => [r[(i + 1) % 4][0] - r[i][0], r[(i + 1) % 4][1] - r[i][1]];
const len = (v) => Math.hypot(v[0], v[1]);
const cosBetween = (a, b) => (a[0] * b[0] + a[1] * b[1]) / (len(a) * len(b) || 1);
let worstCos = 0, worstLen = 0;
for (let i = 0; i < 4; i++) {
  worstCos = Math.max(worstCos, Math.abs(cosBetween(seg(i), seg((i + 1) % 4))));
  worstLen = Math.max(worstLen, Math.abs(len(seg(i)) - len(seg((i + 2) % 4))));
}
const worstDeg = Math.asin(Math.min(1, worstCos)) * 180 / Math.PI;
check(worstCos < 1e-12,
      `four corners are 90 deg by construction: worst |cos| ${worstCos.toExponential(2)} ` +
      `= ${worstDeg.toExponential(2)} deg off (float64 rounding only)`);
check(worstLen < 1e-9, `opposite sides equal: worst difference ${worstLen.toExponential(2)} px`);
check(JSON.stringify(noisy) === before, "snapRect leaves the raw clicks untouched");
quad = r;
const par = (() => {
  const c = centre(), t = autoTip();
  const a = [t[0] - c[0], t[1] - c[1]], e1 = [r[1][0] - r[0][0], r[1][1] - r[0][1]];
  return Math.abs(e1[0] * a[1] - e1[1] * a[0]) / (len(e1) * len(a) || 1);
})();
check(par < 1e-9, `heading stays parallel to the snapped first edge (|sin| ${par.toExponential(1)})`);

console.log(failed ? `\n  ${failed} check(s) FAILED` : "\n  all checks passed");
process.exit(failed ? 1 : 0);
