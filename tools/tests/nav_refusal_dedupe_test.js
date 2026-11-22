// P0013.2: the refusal note must be said once per occurrence, not once per page.
//
// Runs the page's real requestRoute against a stubbed fetch, so the sequence is exercised rather than
// asserted from the source: refuse, plan successfully, refuse the same way again. The second refusal
// has to be announced, otherwise the note stays stuck from the first one for the rest of the session.
const fs = require("fs"), path = require("path"), os = require("os");
const html = path.join(__dirname, "..", "..", "vision", "web", "navigation.html");
const page = fs.readFileSync(html, "utf8");
const js = [...page.matchAll(/<script[^>]*>([\s\S]*?)<\/script>/g)].map(m => m[1])
  .sort((a, b) => b.length - a.length)[0];

const fnStart = js.indexOf("async function requestRoute(manual) {");
const fnEnd = js.indexOf("// When to ask for a new route", fnStart);
if (fnStart < 0 || fnEnd < 0) { console.log("  ✗ requestRoute 抽取失败"); process.exit(1); }
const requestRoute = js.slice(fnStart, fnEnd);

const harness = `
let targetId = "B3", ROUTE = null, replanSeq = 0;
let lastPlan = { x: null, y: null, at: 0 };
let lastRefusalNote = null;
const said = [];
const transient = (text) => said.push(text);
const whyOf = (doc) => doc.code;
const vehicleOf = () => ({ visible: true, x: 300, y: 500 });
let NEXT = null;
const fetch = async () => ({ json: async () => NEXT });
${requestRoute}
module.exports = {
  setNext: (doc) => { NEXT = doc; },
  run: async () => { await requestRoute(false); },
  said: () => said,
  note: () => lastRefusalNote,
};
`;
const tmp = path.join(os.tmpdir(), "p132_route_" + process.pid + ".js");
fs.writeFileSync(tmp, harness);
const M = require(tmp);

let fail = 0;
const check = (label, got, want) => {
  const pass = JSON.stringify(got) === JSON.stringify(want);
  if (pass) console.log(`  ✓ ${label}`);
  else { console.log(`  ✗ ${label}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`); fail++; }
};

const refusal = { ok: false, code: "spot-not-free", kept_target: "B3" };
const success = { ok: true, target_spot: "B3", nodes: [] };

(async () => {
  // ① 第一次拒绝 → 提示一次
  M.setNext(refusal); await M.run();
  check("① 首次拒绝提示一次", M.said().length, 1);

  // ② 同样的拒绝再来一次 → 不重复提示
  M.setNext(refusal); await M.run();
  check("② 相同原因不重复提示", M.said().length, 1);

  // ③ 规划成功 → 复位
  M.setNext(success); await M.run();
  check("③ 成功后去重状态复位", M.note(), null);

  // ④ 相同原因再次拒绝 → 必须重新提示 ✓（这就是被吞掉的那一次）
  M.setNext(refusal); await M.run();
  check("④ 成功后相同原因再拒绝：重新提示", M.said().length, 2);

  console.log(fail ? `\n  ${fail} 项失败 ✗` : "\n  全部通过 ✓");
  process.exit(fail ? 1 : 0);
})();

// --- P0013.3: a refusal must not be stored as the route ------------------------------------
// Storing it made routeSituation alternate between "navigating" and "relocating" on alternate polls,
// because poll() adopts the good route from route.json and requestRoute put the refusal back. The
// symptom was the map zoom flapping between the car and the whole lot.
{
  const js2 = fs.readFileSync(html, "utf8");
  const f0 = js2.indexOf("async function requestRoute(manual) {");
  const f1 = js2.indexOf("// When to ask for a new route", f0);
  const body = js2.slice(f0, f1);
  const harness2 = `
let targetId = "B3", ROUTE = { ok: true, target_spot: "B3", nodes: [[1, 1], [2, 2]] };
let replanSeq = 0, lastPlan = { x: null, y: null, at: 0 }, lastRefusalNote = null;
const said = [];
const transient = (t) => said.push(t);
const whyOf = (doc) => doc.code;
const vehicleOf = () => ({ visible: true, x: 1, y: 2 });
let NEXT = null;
const fetch = async () => ({ json: async () => NEXT });
${body}
module.exports = {
  setNext: (d) => { NEXT = d; },
  run: async () => { await requestRoute(false); },
  route: () => ROUTE,
  said: () => said,
};
`;
  const p2 = path.join(os.tmpdir(), "p133_route_" + process.pid + ".js");
  fs.writeFileSync(p2, harness2);
  const M2 = require(p2);
  (async () => {
    const good = { ok: true, target_spot: "B3", nodes: [[1, 1], [2, 2]] };
    M2.setNext({ ok: false, code: "spot-not-free", kept_target: "B3" });
    await M2.run();
    const afterRefusal = JSON.stringify(M2.route());
    let bad = 0;
    if (afterRefusal === JSON.stringify(good)) console.log("  ✓ 拒绝没有覆盖 ROUTE");
    else { console.log(`  ✗ 拒绝覆盖了 ROUTE: ${afterRefusal}`); bad++; }
    // 连续拒绝 + 成功交替时，ROUTE 应始终是好的那条 ✓
    M2.setNext(good); await M2.run();
    M2.setNext({ ok: false, code: "hero-position-stale", kept_target: "B3" }); await M2.run();
    if (JSON.stringify(M2.route()) === JSON.stringify(good)) console.log("  ✓ 拒绝/成功交替后 ROUTE 仍是好路线");
    else { console.log("  ✗ ROUTE 被换掉了"); bad++; }
    console.log(bad ? `  ${bad} 项失败 ✗` : "  ROUTE 稳定性通过 ✓");
    process.exit(bad ? 1 : 0);
  })();
}
