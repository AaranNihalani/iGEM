/* Substrate Predictor scoring engine. Pure functions (no DOM), mirrors the Python package `spred`.
 *   window  = 8 characters P4..P4'; '-' (or 'X'/'?') marks an unknown position.
 *   model   = parsed models/<PROTEASE>.json plus derived caches.
 */
(function (root) {
  "use strict";
  const AA = "ACDEFGHIKLMNPQRSTVWY";
  const POSITIONS = ["P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'"];
  const GRID_N = 1001;
  const NULL_SIZE = 20000;

  let TABLES = null;        // {vhse: {A:[8],...}, background: {A:..}}
  function init(tables) { TABLES = tables; }

  // ---------------------------------------------------------------- windows
  function parseInput(text) {
    /* Accepts "IPRTLTAA" (8-mer / gapped window) or a peptide with the scissile bond marked: "SGKIPRT|LTA", "SGKIPRT↓LTA".
       Returns {window, missing, note} or {error}. */
    const raw = (text || "").replace(/\s+/g, "").toUpperCase();
    if (!raw) return { error: "Enter a peptide." };
    const m = raw.match(/[|/↓]/);
    let win;
    if (m) {
      const parts = raw.split(/[|/↓]/);
      if (parts.length !== 2) return { error: "Mark exactly one scissile bond." };
      const left = parts[0], right = parts[1];
      if (!left || !right) return { error: "Residues are needed on both sides of the bond." };
      const L = left.slice(-4).padStart(4, "-"), R = right.slice(0, 4).padEnd(4, "-");
      win = L + R;
      var note = left.length > 4 || right.length > 4 ? "Only the four residues on each side of the bond are used (P4 to P4')." : "";
    } else {
      if (raw.length !== 8) return { error: "Give 8 residues (P4 to P4') or mark the scissile bond with |, for example SGKIPRT|LTA." };
      win = raw.replace(/[?X]/g, "-");
    }
    win = win.replace(/[?X]/g, "-");
    for (const c of win) if (c !== "-" && AA.indexOf(c) < 0) return { error: "Unsupported character '" + c + "'. Use the 20 standard residues, or - for unknown." };
    const missing = [];
    for (let i = 0; i < 8; i++) if (win[i] === "-") missing.push(i);
    if (missing.length > 4) return { error: "At least four of the eight positions must be known." };
    return { window: win, missing: missing, note: note || "" };
  }

  // ---------------------------------------------------------------- model construction
  function makeModel(json) {
    const bg = AA.split("").map(a => TABLES.background[a]);
    const expectedBg = json.pssm_log_odds.map(row => row.reduce((s, v, j) => s + bg[j] * v, 0));
    return { id: json.protease, mean: json.feature_mean, scale: json.feature_scale, coef: json.coef, intercept: json.intercept,
             pssm: json.pssm_log_odds, nullLR: json.null_lr_quantiles, nullPSSM: json.null_pssm_quantiles, meta: json.meta || {},
             expectedBg: expectedBg, _nullCache: {} };
  }

  // ---------------------------------------------------------------- scoring
  const sigmoid = x => 1 / (1 + Math.exp(-x));

  function blockContrib(model, win, pos) {
    // contribution of position `pos` to the logit (0 if unknown)
    const c = win[pos];
    if (c === "-") return 0;
    const v = TABLES.vhse[c];
    let s = 0;
    for (let d = 0; d < 8; d++) s += model.coef[pos * 8 + d] * ((v[d] - model.mean[pos * 8 + d]) / model.scale[pos * 8 + d]);
    return s;
  }
  function contributions(model, win) { const out = []; for (let p = 0; p < 8; p++) out.push(blockContrib(model, win, p)); return out; }
  function logit(model, win) { let s = model.intercept; for (let p = 0; p < 8; p++) s += blockContrib(model, win, p); return s; }
  function pssmPositions(model, win) { return win.split("").map((c, p) => c === "-" ? model.expectedBg[p] : model.pssm[p][AA.indexOf(c)]); }
  function pssmScore(model, win) { return pssmPositions(model, win).reduce((a, b) => a + b, 0); }

  function interp(x, xp, fp) {
    // numpy.interp semantics for increasing xp
    const n = xp.length;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    let lo = 0, hi = n - 1;
    while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (xp[mid] <= x) lo = mid; else hi = mid; }
    const t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo] + t * (fp[hi] - fp[lo]);
  }
  const GRID = Array.from({ length: GRID_N }, (_, i) => i / (GRID_N - 1));

  // seeded null windows for gapped scoring
  function mulberry32(a) { return function () { a |= 0; a = (a + 0x6D2B79F5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  let NULL_WINDOWS = null;
  function nullWindows() {
    if (NULL_WINDOWS) return NULL_WINDOWS;
    const cum = []; let acc = 0; for (const a of AA) { acc += TABLES.background[a]; cum.push(acc); }
    const rnd = mulberry32(20260919), out = [];
    for (let n = 0; n < NULL_SIZE; n++) {
      let w = "";
      for (let p = 0; p < 8; p++) { const u = rnd() * acc; let k = 0; while (k < 19 && u > cum[k]) k++; w += AA[k]; }
      out.push(w);
    }
    return (NULL_WINDOWS = out);
  }
  function maskedNull(model, missing) {
    const key = missing.join(",");
    if (model._nullCache[key]) return model._nullCache[key];
    const miss = new Set(missing), lr = [], ps = [];
    for (const w0 of nullWindows()) {
      const w = w0.split("").map((c, i) => miss.has(i) ? "-" : c).join("");
      lr.push(sigmoid(logit(model, w))); ps.push(pssmScore(model, w));
    }
    lr.sort((a, b) => a - b); ps.sort((a, b) => a - b);
    return (model._nullCache[key] = { lr: lr, ps: ps });
  }
  function upperBound(arr, x) { let lo = 0, hi = arr.length; while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] <= x) lo = m + 1; else hi = m; } return lo; }

  function scoreModel(model, win) {
    const missing = []; for (let i = 0; i < 8; i++) if (win[i] === "-") missing.push(i);
    const lg = logit(model, win), p = sigmoid(lg), s = pssmScore(model, win);
    let a, b;
    if (missing.length === 0) { a = interp(p, model.nullLR, GRID); b = interp(s, model.nullPSSM, GRID); }
    else { const nl = maskedNull(model, missing); a = upperBound(nl.lr, p) / nl.lr.length; b = upperBound(nl.ps, s) / nl.ps.length; }
    return { logit: lg, prob: p, pssm: s, pctLR: a, pctPSSM: b, u: 2 * a * b / (a + b + 1e-12) };
  }
  function scorePanel(models, members, win) {
    const out = {};
    for (const m of members) out[m] = scoreModel(models[m], win);
    return out;
  }
  function margin(scores, members, target) {
    let best = -1, who = null;
    for (const m of members) if (m !== target && scores[m].u > best) { best = scores[m].u; who = m; }
    return { margin: scores[target].u - best, competitor: who, competitorU: best };
  }
  function mutationScan(models, members, win, target) {
    const base = scorePanel(models, members, win), baseM = margin(base, members, target).margin, baseLogit = base[target].logit;
    const cells = [];
    for (let p = 0; p < 8; p++) {
      if (win[p] === "-") continue;
      for (const a of AA) {
        if (a === win[p]) continue;
        const w = win.slice(0, p) + a + win.slice(p + 1);
        const s = scorePanel(models, members, w);
        cells.push({ pos: p, aa: a, window: w, dLogit: s[target].logit - baseLogit, dMargin: margin(s, members, target).margin - baseM, u: s[target].u });
      }
    }
    return { cells: cells, baseMargin: baseM, baseLogit: baseLogit };
  }

  const api = { AA, POSITIONS, init, parseInput, makeModel, scoreModel, scorePanel, margin, mutationScan, contributions, pssmPositions, interp, GRID };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.SPEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
