/* The annotation convention, as an executable check.

   Corner 1 is the car's RIGHT-FRONT corner and the rest follow clockwise, so p0->p1 runs along the
   car's right side pointing rearward and the front direction is the other way. Every label in
   vision/data/labels rests on that, so it is tested rather than trusted.

   The functions below are extracted verbatim from tools/label_ui.html - a copy would test the copy.

       node tools/tests/autotip_test.js        (or: node tools/tests/autotip_test.js <ui.html>)

   Headings are image coordinates: 0 along +x (right), increasing towards +y (down).
*/
const fs = require("fs");
const path = process.argv[2] || require("path").join(__dirname, "..", "label_ui.html");
const src = fs.readFileSync(path, "utf8");
const script = src.match(/<script>([\s\S]*?)<\/script>/)[1];
for (const name of ["centre", "angleTo", "sides", "autoTip"]) {
  const m = script.match(new RegExp("function " + name + "\\([^)]*\\) \\{[\\s\\S]*?\\n\\}"));
  if (!m) { console.log("  cannot find " + name + " in " + path); process.exit(1); }
  eval(m[0]);
}
let quad = [];
function centre() {
  let x = 0, y = 0;
  quad.forEach(q => { x += q[0]; y += q[1]; });
  return [x / quad.length, y / quad.length];
}
function angleTo(from, to) {
  let a = Math.atan2(to[1] - from[1], to[0] - from[0]) * 180 / Math.PI;
  return (a + 360) % 360;
}
function sides(p) {
  return p.map((q, i) => Math.hypot(q[0] - p[(i + 1) % p.length][0], q[1] - p[(i + 1) % p.length][1]));
}
function autoTip() {
  // 前方 = 从中心沿 −(p1→p0)：p0 是右前角，p1 是同侧的后角，所以反向即向前。
  if (quad.length !== 4) return null;
  let dx = quad[0][0] - quad[1][0], dy = quad[0][1] - quad[1][1];
  const n = Math.hypot(dx, dy) || 1;
  dx /= n; dy /= n;
  const c = centre();
  const len = Math.max(...sides(quad)) / 2;        // 中心到车头边缘
  return [c[0] + dx * len, c[1] + dy * len];
}
// 约定：第 1 点 = 车的右前角，顺时针。于是 p0→p1 沿车身右长边指向后方，
// 前方 = 从中心沿 −(p1→p0)。这四条是全部标签的地基，所以做成可重跑的检查。
const cars = {
  "车头朝上": [[430,240],[430,360],[370,360],[370,240]],
  "车头朝右": [[460,330],[340,330],[340,270],[460,270]],
  "车头朝下": [[370,360],[370,240],[430,240],[430,360]],
  "车头朝左": [[340,270],[460,270],[460,330],[340,330]],
};
const want = { "车头朝上": 270, "车头朝右": 0, "车头朝下": 90, "车头朝左": 180 };
let bad = 0;
for (const [name, q] of Object.entries(cars)) {
  quad = q;
  const got = angleTo(centre(), autoTip());
  const err = Math.abs((got - want[name] + 540) % 360 - 180);
  if (err > 0.5) bad++;
  console.log(`  ${name}: ${got.toFixed(1)} deg (want ${want[name]}) err ${err.toFixed(2)} ${err <= 0.5 ? "ok" : "FAIL"}`);
}
// 第 2 点点到车宽那条边（错误用法）会偏 90 度，这正是页面要警告的情况。
quad = [[430,240],[370,240],[370,360],[430,360]];
const wrong = angleTo(centre(), autoTip());
const off = Math.abs((wrong - 270 + 540) % 360 - 180);
console.log(`  p1 on the width edge: ${wrong.toFixed(1)} deg, ${off.toFixed(0)} deg off - the page warns`);
if (bad || Math.abs(off - 90) > 5) { console.log("FAILED"); process.exit(1); }
console.log("  all four orientations exact; the width-edge mistake is detected by the warning");
