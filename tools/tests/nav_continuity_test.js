// P0011: heading continuity + keeping the icon through a brief loss.
// Extracts the animation/heading logic out of navigation.html and runs it. No DOM, no canvas: the
// two things under test are pure decisions - which heading to believe, and where the icon stays when
// there is no fix. The rendering itself still needs a browser (see docs/hero-pipeline.md §13).
const fs = require("fs"), path = require("path"), os = require("os");
const html = path.join(__dirname, "..", "..", "vision", "web", "navigation.html");
const page = fs.readFileSync(html, "utf8");
const scripts = [...page.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1]);
let src = scripts.sort((a, b) => b.length - a.length)[0];
// 只取纯逻辑部分（不碰 DOM）
const start = src.indexOf("function lerp(");
const end = src.indexOf("// Height of whatever sheet");
const logic = src.slice(start, end)
  .replace(/const CAR_DEADBAND_PX[\s\S]*?const CAR_SNAP_PX = [0-9.]+;/, "const CAR_DEADBAND_PX=3,CAR_FOLLOW_K=0.45,CAR_SNAP_PX=60;");
const harness = `
const Nav = { normDeg: d => ((d % 360) + 360) % 360 };
${logic}
module.exports = { lerpAngle, acceptHeading, followAxis,
                   reset: () => { headingPending = null; headingPendingCount = 0; } };
`;
const tmp = path.join(os.tmpdir(), "nav_logic_" + process.pid + ".js");
fs.writeFileSync(tmp, harness);
const L = require(tmp);
let fail = 0;
function check(label, got, want) {
  const ok = Math.abs(((got - want + 540) % 360) - 180) < 1.5 || got === want;
  if (!ok) { console.log(`  ✗ ${label}: got ${got}, want ${want}`); fail++; }
  else console.log(`  ✓ ${label}: ${typeof got === "number" ? got.toFixed(1) : got}`);
}
// A: 小角度直接就接受
check("92° from 90° accepted", L.acceptHeading(90, 92), 92);
// A: 359→1 只有 2° ✓（环绕）
check("359° -> 1° treated as 2°", L.acceptHeading(359, 1), 1);
// A: 单次 180° 不接受
L.reset();
check("90° -> 270° once is refused", L.acceptHeading(90, 270), 90);
// A: 连续两次才接受
L.reset();
check("90° -> 270° twice is accepted", (L.acceptHeading(90,270), L.acceptHeading(90,270)), 270);
// A: 持续转向正常跟随
L.reset();
check("continuous turn 90->100", L.acceptHeading(90, 100), 100);
// A: 两次不同的坏读数不算"持续"
L.reset();
L.acceptHeading(90, 270); L.acceptHeading(90, 180);
check("two different bad readings do not add up", L.acceptHeading(90, 200), 90);
// lerpAngle: 最短路径
check("lerpAngle 359->1 stays near 0", L.lerpAngle(359, 1, 0.5) < 1 || L.lerpAngle(359, 1, 0.5) > 359, 0);
// followAxis: 死区内不动 / 远处吸附
if (L.followAxis(100, 102, 3, 0.45, 60) !== 100) { console.log("  ✗ deadband moves"); fail++; } else console.log("  ✓ deadband keeps still");
if (L.followAxis(100, 300, 3, 0.45, 60) !== 300) { console.log("  ✗ snap fails"); fail++; } else console.log("  ✓ far jump snaps");
console.log(fail === 0 ? "\n  全部通过 ✓" : `\n  ${fail} 项失败 ✗`);
process.exit(fail ? 1 : 0);
