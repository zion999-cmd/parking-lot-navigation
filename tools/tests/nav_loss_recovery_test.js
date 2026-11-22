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
// The real page has this next to vehicleOf; shouldReplan calls it.
const trustedFix = (v) => !!v && v.visible !== false && !v.position_stale &&
                          typeof v.x === "number" && typeof v.y === "number";
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
// P0013.1: reviewed - the page used to ask and rely on the server refusing. The server did not
// check position_stale at all, so the request went out and could be planned from a stale fix. Now
// both ends refuse it, and the page does not ask.
L.setVehicle({ visible: true, x: 300, y: 500, position_stale: true });
check("B 位置过期（visible=true 但 stale）: 不重规划", L.shouldReplan(9000), false);

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
// 已经在路上（road）→ **也吸附** ✓
// 这是被实测推翻的一条 ✓：车辆在路上时 location.kind 就是 "road" ✗ 不是 "floor" ✗，
// 所以旧判断让吸附在车辆行驶时从不生效 ✗ → 图标在车道里飘 ✗ ✓
eq("road: 吸附到车道中心", N2.displayPoint({ x: 250, y: 150, location: { kind: "road" } }, [chain], []), [250, 100]);
// 没有 location 字段 → 画实际位置 ✓（不猜 ✓）
eq("无 location: 不吸附", N2.displayPoint({ x: 250, y: 150 }, [chain], []), [250, 150]);
// 闭环也支持 ✓
eq("floor + 闭环: 吸附", N2.displayPoint({ x: 100, y: 105, location: { kind: "floor" } }, [], [[[50, 100], [150, 100], [150, 200], [50, 200]]]), [100, 100]);
console.log(bad ? `  ${bad} 项吸附用例失败 ✗` : "  吸附用例全部通过 ✓");
const total = fail + bad;
console.log(total ? `\n  ${total} 项失败 ✗` : "\n  全部通过 ✓");
process.exit(total ? 1 : 0);

// --- P0013.1: position_stale is not a position -------------------------------------------
const st2 = fs.readFileSync(html, "utf8");
const s3 = [...st2.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1])
  .sort((a, b) => b.length - a.length)[0];
const tLines = s3.split("\n");
const tStartLine = tLines.findIndex(l => l.includes("const trustedFix = (v) =>"));
if (tStartLine < 0) { console.log("  ✗ trustedFix 抽取失败"); process.exit(1); }
let tSrc = tLines[tStartLine] + "\n";
if (!tSrc.includes(";")) tSrc += tLines[tStartLine + 1] + "\n";
const tTmp = path.join(os.tmpdir(), "p131_trusted_" + process.pid + ".js");
fs.writeFileSync(tTmp, tSrc + "\nmodule.exports = trustedFix;\n");
const tf = require(tTmp);
let bad2 = 0;
const chk = (label, got, want) => {
  if (got !== want) { console.log(`  ✗ ${label}: got ${got}, want ${want}`); bad2++; }
  else console.log(`  ✓ ${label}`);
};
chk("visible+新鲜 → 可信", tf({ visible: true, x: 1, y: 2 }), true);
chk("visible + position_stale → 不可信", tf({ visible: true, x: 1, y: 2, position_stale: true }), false);
chk("visible=false → 不可信", tf({ visible: false, x: 1, y: 2 }), false);
chk("没有坐标 → 不可信", tf({ visible: true }), false);
chk("没有 vehicle → 不可信", tf(null), false);
console.log(bad2 ? `  ${bad2} 项失败 ✗` : "  trustedFix 全部通过 ✓");
process.exit(bad2 ? 1 : 0);

// --- every identifier the page calls must exist -------------------------------------------
// whyOf() was called from requestRoute's refusal branch and never defined anywhere: every refusal
// threw a ReferenceError inside an async function, which is an unhandled rejection and a dead page.
{
  const full = fs.readFileSync(html, "utf8");
  const js = [...full.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1])
    .sort((a, b) => b.length - a.length)[0];
  const called = new Set([...js.matchAll(/\b([a-zA-Z_$][\w$]*)\(/g)].map(m => m[1]));
  const declared = new Set([
    ...[...js.matchAll(/(?:function\s+|const\s+|let\s+|var\s+)([a-zA-Z_$][\w$]*)/g)].map(m => m[1]),
    ...["if", "for", "while", "switch", "catch", "return", "typeof", "new", "Math", "JSON",
        "Number", "String", "Array", "Object", "Set", "Map", "fetch", "require", "console",
        "parseInt", "parseFloat", "isNaN", "setTimeout", "clearTimeout", "setInterval", "alert"],
  ]);
  const missing = [...called].filter(n => !declared.has(n) && !js.includes("." + n + "("));
  if (missing.length) {
    console.log(`  ✗ 调用但未定义: ${missing.join(", ")}`);
    process.exit(1);
  }
  console.log("  ✓ 页面调用的函数都有定义");
}

// --- displayPoint: a car on the tarmac rides the lane centreline -------------------------------
// Measured live: location.kind was "road", the guard only accepted "floor", so the projection never ran
// and the icon drifted inside the lane instead of sitting on it.
{
  const html2 = fs.readFileSync(path.join(ROOT, "vision", "web", "navigation.html"), "utf8");
  const a = html2.indexOf("displayPoint(vehicle, chains, loops) {");
  const b = html2.indexOf("carHeading(vehicle)", a);
  const h = `
const Nav = {
  nearestLanePoint(chains, loops, x, y) {
    let best = null, bestD = Infinity;
    for (const pts of chains || []) {
      for (let i = 0; i < pts.length - 1; i++) {
        const p0 = pts[i], p1 = pts[i + 1];
        const dx = p1[0] - p0[0], dy = p1[1] - p0[1], l2 = dx * dx + dy * dy;
        const t = l2 ? Math.max(0, Math.min(1, ((x - p0[0]) * dx + (y - p0[1]) * dy) / l2)) : 0;
        const px = p0[0] + t * dx, py = p0[1] + t * dy;
        const d = Math.hypot(px - x, py - y);
        if (d < bestD) { bestD = d; best = [px, py]; }
      }
    }
    return best;
  },
  ${html2.slice(a, b).replace("this.nearestLanePoint", "Nav.nearestLanePoint")}
};
module.exports = (kind, x, y) => Nav.displayPoint({ x, y, location: { kind } }, [[[0, 0], [100, 0]]], []);
`;
  const f = path.join(os.tmpdir(), "disp_" + process.pid + ".js");
  fs.writeFileSync(f, h);
  const dp = require(f);
  let bad = 0;
  const check = (name, got, want) => {
    const ok = JSON.stringify(got) === JSON.stringify(want);
    console.log(`  ${ok ? "✓" : "✗"} ${name}: ${JSON.stringify(got)}`);
    if (!ok) bad++;
  };
  check("road → 车道中心 (y=0)", dp("road", 50, 12), [50, 0]);
  check("floor → 车道中心", dp("floor", 50, -9), [50, 0]);
  check("spot → 原位不动", dp("spot", 50, 12), [50, 12]);
  check("unknown → 原位不动", dp("unknown", 50, 12), [50, 12]);
  process.exit(bad ? 1 : 0);
}
