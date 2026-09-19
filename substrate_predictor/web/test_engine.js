// node web/test_engine.js  : compares the JavaScript engine with Python reference outputs
const fs = require("fs"), path = require("path");
const E = require("./engine.js");
const root = path.join(__dirname, "..");
E.init(JSON.parse(fs.readFileSync(path.join(__dirname, "tables.json"))));
const fx = JSON.parse(fs.readFileSync(path.join(__dirname, "fixture.json")));
const models = {};
for (const f of fs.readdirSync(path.join(root, "models"))) if (f.endsWith(".json")) { const j = JSON.parse(fs.readFileSync(path.join(root, "models", f))); models[j.protease] = E.makeModel(j); }
let worst = { P: 0, pctFull: 0, uFull: 0, pGap: 0, pctGap: 0, contrib: 0 };
for (const m of Object.keys(fx.full)) {
  fx.full_windows.forEach((w, i) => {
    const s = E.scoreModel(models[m], w);
    worst.P = Math.max(worst.P, Math.abs(s.prob - fx.full[m].P[i]));
    worst.pctFull = Math.max(worst.pctFull, Math.abs(s.pctLR - fx.full[m].pct_lr[i]), Math.abs(s.pctPSSM - fx.full[m].pct_pssm[i]));
    worst.uFull = Math.max(worst.uFull, Math.abs(s.u - fx.full[m].u[i]));
  });
  fx.gapped_windows.forEach((w, i) => {
    const s = E.scoreModel(models[m], w);
    worst.pGap = Math.max(worst.pGap, Math.abs(s.prob - fx.gapped[m].P[i]));
    worst.pctGap = Math.max(worst.pctGap, Math.abs(s.pctLR - fx.gapped[m].pct_lr[i]), Math.abs(s.pctPSSM - fx.gapped[m].pct_pssm[i]));
  });
  for (const w of Object.keys(fx.contrib[m])) {
    const c = E.contributions(models[m], w);
    fx.contrib[m][w].forEach((v, k) => worst.contrib = Math.max(worst.contrib, Math.abs(v - c[k])));
  }
}
console.log("max abs differences:", worst);
const ok = worst.P < 1e-6 && worst.pctFull < 1e-6 && worst.uFull < 1e-6 && worst.pGap < 1e-6 && worst.pctGap < 0.03 && worst.contrib < 1e-6;
// parser checks
const p1 = E.parseInput("SGKIPRT|LTA"), p2 = E.parseInput("iprtlt a a"), p3 = E.parseInput("SGK");
console.log("parse SGKIPRT|LTA ->", p1.window, "| 8-mer ->", p2.window, "| short ->", p3.error);
const parseOk = p1.window === "IPRTLTA-" && p2.window === "IPRTLTAA" && !!p3.error;
console.log(ok && parseOk ? "PASS" : "FAIL");
process.exit(ok && parseOk ? 0 : 1);
