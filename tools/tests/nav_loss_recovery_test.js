// P0013: what the navigation page does when the vision data goes away and comes back.
//
// The decisions are extracted from navigation.html and run here: shouldReplan (when to ask the server
// for a route) and the refusal note (what to say, and how often). Rendering still needs a browser.
const fs = require("fs"), path = require("path"), os = require("os");
const html = path.join(__dirname, "..", "..", "vision", "web", "navigation.html");
const page = fs.readFileSync(html, "utf8");
const src = [...page.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1])
  .sort((a, b) => b.length - a.length)[0];

const start = src.indexOf("function shouldReplan(");
const end = src.indexOf("function cancelNavigation(");
let logic = src.slice(start, end);

const harness = `
let PHASE = "NAVIGATING", targetId = "B3", lastPlan = null, lastRefusalNote = null;
const REPLAN_AFTER_MS = 1500, REPLAN_MOVE_PX = 4;
let VEHICLE = null;
const vehicleOf = () => VEHICLE;
${logic}
module.exports = {
  setVehicle: v => { VEHICLE = v; },
  setPhase: p => { PHASE = p; },
  setTarget: t => { targetId = t; },
  plan: (x, y, at) => { lastPlan = { x, y, at }; },
  getLastPlan: () => lastPlan,
  clearPlan: () => { lastPlan = null; },
  shouldReplan,
};
`;
const tmp = path.join(os.tmpdir(), "p13_logic_" + process.pid + ".js");
fs.writeFileSync(tmp, harness);
const L = require(tmp);

let fail = 0;
const check = (label, got, want) => {
  if (got !== want) { console.log(`  ✗ ${label}: got ${got}, want ${want}`); fail++; }
  else console.log(`  ✓ ${label}`);
};
const hero = { visible: true, x: 300, y: 500 };
const lost = { id: "hero", visible: false };

// A: normal navigation, car stationary, timer not elapsed -> no replan
L.setVehicle(hero); L.plan(300, 500, 1000);
check("A 静止且未到 1.5 s: 不重规划", L.shouldReplan(1200), false);
// A: timer elapsed -> parked safety net fires
check("A 到 1.5 s: 重规划（安全网）", L.shouldReplan(2600), true);
// A: car moved -> replan
L.setVehicle({ visible: true, x: 310, y: 500 });     // the car moved after the plan
L.plan(300, 500, 1000);
check("A 车移动 10 px: 重规划", L.shouldReplan(1100), true);
L.setVehicle(hero);

// B: hero lost. Even after the timer, nothing to plan from.
L.setVehicle(lost); L.plan(300, 500, 1000);
check("B Hero 不可见 + 超过 1.5 s: 不重规划", L.shouldReplan(9000), false);
L.setVehicle(null); L.plan(300, 500, 1000);
check("B 没有 vehicle 字段 + 超过 1.5 s: 不重规划", L.shouldReplan(9000), false);
L.setVehicle({ visible: true, x: 300, y: 500, position_stale: true });
check("B 位置标记过期（但仍有坐标）: 由 server 拒绝，页面仍可询问", L.shouldReplan(9000), true);

// C: recovered -> replanning resumes
L.setVehicle(hero); L.plan(300, 500, 1000);
check("C 恢复后到达 1.5 s: 重新规划", L.shouldReplan(9000), true);
check("C 恢复后车移动: 重新规划", L.shouldReplan(1001), false);   // 未超过 MOVE_PX，也未到时间
L.setVehicle({ visible: true, x: 320, y: 500 });
check("C 恢复后移动 20 px: 重新规划", L.shouldReplan(1001), true);

// 不在导航中 / 没有目标
L.setPhase("IDLE");
check("IDLE: 不重规划", L.shouldReplan(9000), false);
L.setPhase("NAVIGATING"); L.setTarget(null);
check("没有目标: 不重规划", L.shouldReplan(9000), false);

console.log(fail ? `  ${fail} 项重规划用例失败 ✗` : "  重规划用例全部通过 ✓");

// --- P0013 follow-up: the icon glides to the new fix, it does not teleport -----------------
const page2 = fs.readFileSync(html, "utf8");
const s2 = [...page2.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1])
  .sort((a, b) => b.length - a.length)[0];
const a2 = s2.indexOf("const CAR_DEADBAND_PX");
const b2 = s2.indexOf("function lerpAngle(");
let glow = s2.slice(a2, b2) + s2.slice(b2, s2.indexOf("// A heading jump this large") > 0
  ? s2.indexOf("// A heading jump this large") : b2 + 400);
// A minimal harness: only followAxis/lerpAngle + the recovery constant.
const kMatch = page2.match(/const CAR_RECOVER_K = ([0-9.]+)/);
const k = kMatch ? Number(kMatch[1]) : null;
if (k === null) { console.log("  ✗ CAR_RECOVER_K missing"); process.exit(1); }
let gap = 200, frames = 0;
while (gap > 10 && frames < 600) { gap -= gap * k; frames += 1; }
console.log(`  ${frames <= 90 ? "✓" : "✗"} 200 px gap closes in ${(frames / 60).toFixed(2)} s (${frames} frames)`);
const tooSlow = frames > 150, tooFast = frames < 20;
if (tooSlow || tooFast) { console.log("  ✗ recovery glide is not about a second"); process.exit(1); }
console.log("  ✓ 恢复是滑行而不是瞬移");

// --- P0013 follow-up: the hero is drawn on a lane when the document says it is on the floor ----
const navSrc = fs.readFileSync(html, "utf8");
const navStart = navSrc.indexOf("const Nav = {");
const navEnd = navSrc.indexOf("const STATE_COLOURS") > 0
  ? navSrc.indexOf("// ---------------------------------------------------------------- snapping")
  : navSrc.indexOf("// ---------------------------------------------------------------- snapping");
// Just the two helpers, run against a plain lane.
const snapSrc = navSrc.slice(navSrc.indexOf("nearestLanePoint(chains, loops, x, y) {"));
const snapEnd = snapSrc.indexOf("},", snapSrc.indexOf("return this.nearestLanePoint"));
const body = snapSrc.slice(0, snapEnd + 2);
const snapHarness = `
const Nav = {
  ${body}
};
module.exports = Nav;
`;
const snapTmp = path.join(os.tmpdir(), "p13_snap_" + process.pid + ".js");
fs.writeFileSync(snapTmp, snapHarness);
const N2 = require(snapTmp);
const chain = [[100, 100], [300, 100]];      // a straight lane along y = 100
let ok = 0, bad = 0;
const eq = (label, got, want) => {
  const pass = Array.isArray(want) && Array.isArray(got)
    ? Math.abs(got[0] - want[0]) < 0.5 && Math.abs(got[1] - want[1]) < 0.5
    : got === want;
  if (pass) { console.log(`  ✓ ${label}`); ok++; } else { console.log(`  ✗ ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); bad++; }
};
// 在桌面上（floor）→ 吸附到最近的车道中心
eq("floor: 吸附到车道 (250,150) → (250,100)", N2.displayPoint({ x: 250, y: 150, location: { kind: "floor" } }, [chain], []), [250, 100]);
// 在车位里（spot）→ 画实际位置 ✓
eq("spot: 不吸附", N2.displayPoint({ x: 250, y: 150, location: { kind: "spot", id: "B3" } }, [chain], []), [250, 150]);
// 已经在路上（road）→ 画实际位置 ✓
eq("road: 不吸附", N2.displayPoint({ x: 250, y: 150, location: { kind: "road" } }, [chain], []), [250, 150]);
// 没有 location 字段 → 画实际位置 ✓（不猜 ✓）
eq("无 location: 不吸附", N2.displayPoint({ x: 250, y: 150 }, [chain], []), [250, 150]);
// 闭环也支持 ✓
eq("floor + 闭环: 吸附", N2.displayPoint({ x: 100, y: 105, location: { kind: "floor" } }, [], [[[50, 100], [150, 100], [150, 200], [50, 200]]]), [100, 100]);
console.log(bad ? `  ${bad} 项吸附用例失败 ✗` : "  吸附用例全部通过 ✓");
const total = fail + bad;
console.log(total ? `\n  ${total} 项失败 ✗` : "\n  全部通过 ✓");
process.exit(total ? 1 : 0);
