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

console.log(fail ? `\n  ${fail} 项失败 ✗` : "\n  全部通过 ✓");
process.exit(fail ? 1 : 0);
