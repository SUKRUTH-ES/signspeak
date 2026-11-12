// Glass folders + generated liquid artwork, ported to plain JS from the
// "Glass Folder Cards" React component. No framework, no images.

// #region art
export const ART_PRESETS = {
  aurora: { style: "streak", colors: ["#f4efe9", "#e9c2b8", "#c2433f", "#2a1e2a", "#22395e", "#6f97c2", "#e5edf4"], scale: 0.7, warp: 1.4, angle: -0.62, gloss: 0.18, veins: 0 },
  ember: { style: "marble", colors: ["#07090c", "#16252c", "#3d5f68", "#1a1416", "#a3201a", "#ef4a2b", "#ff8a5c"], scale: 0.62, warp: 3.2, angle: -0.5, gloss: 0.55, veins: 0.6 },
  dusk: { style: "marble", colors: ["#120c18", "#3e2850", "#8f6c9c", "#2b1a24", "#b3502f", "#eab08a", "#fbe7d2"], scale: 0.75, warp: 3.6, angle: 0.8, gloss: 0.5, veins: 0.5 },
  glacier: { style: "streak", colors: ["#f7f8fa", "#d9e2ea", "#9fb4c7", "#e8edf2", "#bccbd8", "#ffffff"], scale: 0.7, warp: 1.6, angle: 0.35, gloss: 0.25, veins: 0 },
  verdant: { style: "marble", colors: ["#070b08", "#173024", "#4f7a5a", "#141812", "#a7843a", "#e9cf87", "#fbf3d6"], scale: 0.7, warp: 3.4, angle: 2.1, gloss: 0.5, veins: 0.5 },
};

function hexToRgb(hex) {
  let h = hex.trim().replace(/^#/, "");
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  const n = /^[0-9a-f]{6}$/i.test(h) ? parseInt(h, 16) : 0x808080;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
export function hashString(s) {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
  return h >>> 0;
}
function makeFbm(seed) {
  const r = rng(seed);
  const perm = new Uint8Array(512);
  const grad = new Float32Array(512);
  const p = [];
  for (let i = 0; i < 256; i++) p.push(i);
  for (let i = 255; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  for (let i = 0; i < 512; i++) perm[i] = p[i & 255];
  for (let i = 0; i < 256; i++) {
    const a = r() * Math.PI * 2;
    grad[i * 2] = Math.cos(a);
    grad[i * 2 + 1] = Math.sin(a);
  }
  const noise = (x, y) => {
    const xi = Math.floor(x), yi = Math.floor(y);
    const xf = x - xi, yf = y - yi;
    const X = xi & 255, Y = yi & 255;
    const h00 = perm[X + perm[Y]] * 2, h10 = perm[X + 1 + perm[Y]] * 2;
    const h01 = perm[X + perm[Y + 1]] * 2, h11 = perm[X + 1 + perm[Y + 1]] * 2;
    const d00 = grad[h00] * xf + grad[h00 + 1] * yf;
    const d10 = grad[h10] * (xf - 1) + grad[h10 + 1] * yf;
    const d01 = grad[h01] * xf + grad[h01 + 1] * (yf - 1);
    const d11 = grad[h11] * (xf - 1) + grad[h11 + 1] * (yf - 1);
    const u = xf * xf * xf * (xf * (xf * 6 - 15) + 10);
    const v = yf * yf * yf * (yf * (yf * 6 - 15) + 10);
    return d00 + u * (d10 - d00) + v * (d01 - d00 + u * (d11 - d01 - d10 + d00));
  };
  return (x, y, octaves) => {
    let sum = 0, amp = 0.5;
    for (let i = 0; i < octaves; i++) {
      sum += amp * noise(x, y);
      const nx = 1.6 * x + 1.2 * y + 7.1;
      y = -1.2 * x + 1.6 * y + 3.3;
      x = nx;
      amp *= 0.5;
    }
    return sum;
  };
}

// Domain-warped noise shaded as a liquid surface, painted a few rows at a time.
function createArtJob(width, height, recipe, seed) {
  const w = Math.max(1, Math.round(width)), h = Math.max(1, Math.round(height));
  const colors = recipe.colors.length >= 2 ? recipe.colors : ["#111111", "#eeeeee"];
  const stops = colors.map((c, i) => ({ t: i / (colors.length - 1), c: hexToRgb(c) }));
  const scale = recipe.scale ?? 0.7, warp = recipe.warp ?? 3, gloss = recipe.gloss ?? 0.4, veins = recipe.veins ?? 0;
  const streak = recipe.style === "streak";
  const fbm = makeFbm(seed * 977 + 13);
  const r = rng(seed + 1);
  const ox = r() * 50, oy = r() * 50;
  const angle = (recipe.angle ?? 0) + (r() - 0.5) * 0.5;
  const ca = Math.cos(angle), sa = Math.sin(angle);
  const m = Math.min(w, h);
  const gw = Math.ceil(w / 2) + 3, gh = Math.ceil(h / 2) + 3;
  const G = new Float32Array(gw * gh), GQ = new Float32Array(gw * gh);
  const F = new Float32Array(w * h), Q = new Float32Array(w * h);
  const pixels = new Uint8ClampedArray(w * h * 4);
  let row = 0;

  const field = (gy) => {
    for (let gx = 0; gx < gw; gx++) {
      const u = ((gx - 1) * 2 - w / 2) / m, v = ((gy - 1) * 2 - h / 2) / m;
      let px = (u * ca - v * sa) * scale, py = (u * sa + v * ca) * scale;
      if (streak) { px *= 0.14; py *= 2.2; }
      px += ox; py += oy;
      const qx = fbm(px, py, 3), qy = fbm(px + 5.2, py + 1.3, 3);
      const rx = fbm(px + warp * qx + 1.7, py + warp * qy + 9.2, 3);
      const ry = fbm(px + warp * qx + 8.3, py + warp * qy + 2.8, 3);
      G[gy * gw + gx] = fbm(px + warp * rx, py + warp * ry, 4);
      GQ[gy * gw + gx] = Math.sqrt(qx * qx + qy * qy);
    }
  };
  const upsample = () => {
    const cr = (a, b, c, d) => (-a + 9 * b + 9 * c - d) / 16;
    const T = new Float32Array(w * gh);
    for (let gy = 0; gy < gh; gy++) {
      const rr = gy * gw;
      for (let x = 0; x < w; x++) {
        const c = (x >> 1) + 1;
        T[gy * w + x] = x & 1 ? cr(G[rr + c - 1], G[rr + c], G[rr + c + 1], G[rr + c + 2]) : G[rr + c];
      }
    }
    for (let y = 0; y < h; y++) {
      const c = (y >> 1) + 1;
      for (let x = 0; x < w; x++) {
        F[y * w + x] = y & 1 ? cr(T[(c - 1) * w + x], T[c * w + x], T[(c + 1) * w + x], T[(c + 2) * w + x]) : T[c * w + x];
        const gx = (x >> 1) + 1;
        const q0 = GQ[c * gw + gx];
        const qx = x & 1 ? (q0 + GQ[c * gw + gx + 1]) / 2 : q0;
        const q1 = y & 1 ? (x & 1 ? (GQ[(c + 1) * gw + gx] + GQ[(c + 1) * gw + gx + 1]) / 2 : GQ[(c + 1) * gw + gx]) : qx;
        Q[y * w + x] = (qx + q1) / 2;
      }
    }
  };
  const colorize = () => {
    const sample = [];
    const stride = Math.max(1, Math.floor(F.length / 4096));
    for (let i = 0; i < F.length; i += stride) sample.push(F[i]);
    sample.sort((a, b) => a - b);
    const lo = sample[Math.floor(sample.length * 0.02)], hi = sample[Math.floor(sample.length * 0.98)];
    const k = 1 / (hi - lo || 1);
    const lx = -0.55, ly = -0.65, lz = 0.52, slope = k * m * 0.012;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = y * w + x;
        let t = (F[i] - lo) * k;
        t = t < 0 ? 0 : t > 1 ? 1 : t;
        let j = 1;
        while (j < stops.length - 1 && stops[j].t < t) j++;
        const a = stops[j - 1], b = stops[j];
        let f = (t - a.t) / (b.t - a.t || 1);
        f = f < 0 ? 0 : f > 1 ? 1 : f;
        f = f * f * (3 - 2 * f);
        const gx = (F[x < w - 1 ? i + 1 : i] - F[x > 0 ? i - 1 : i]) * slope;
        const gy = (F[y < h - 1 ? i + w : i] - F[y > 0 ? i - w : i]) * slope;
        const nd = (-gx * lx - gy * ly + lz) / Math.sqrt(gx * gx + gy * gy + 1);
        const spec = Math.pow(nd > 0 ? nd : 0, 24) * gloss * 255;
        const silk = 1 + 0.035 * Math.sin(t * 150 + Q[i] * 24);
        const vein = veins * Math.max(0, 1 - Math.abs(Q[i] - 0.42) * 14) * 0.6;
        const lit = (0.82 + 0.28 * nd) * silk * (1 - vein);
        pixels[i * 4] = (a.c[0] + (b.c[0] - a.c[0]) * f) * lit + spec;
        pixels[i * 4 + 1] = (a.c[1] + (b.c[1] - a.c[1]) * f) * lit + spec;
        pixels[i * 4 + 2] = (a.c[2] + (b.c[2] - a.c[2]) * f) * lit + spec;
        pixels[i * 4 + 3] = 255;
      }
    }
  };
  const step = (rows) => {
    if (row >= gh) return true;
    const end = Math.min(gh, row + Math.max(1, rows));
    while (row < end) field(row++);
    if (row < gh) return false;
    upsample();
    colorize();
    return true;
  };
  return { width: w, height: h, pixels, step };
}

// One shared painter that spends a few ms per frame, so the page never janks.
const artCache = new Map();
const queue = [];
let pumping = false;
function pump() {
  if (pumping) return;
  pumping = true;
  const tick = () => {
    const until = performance.now() + 7;
    while (queue.length && performance.now() < until) {
      const head = queue[0];
      if (head.job.step(4)) {
        queue.shift();
        const c = document.createElement("canvas");
        c.width = head.job.width;
        c.height = head.job.height;
        c.getContext("2d").putImageData(new ImageData(head.job.pixels, c.width, c.height), 0, 0);
        head.done(c);
      }
    }
    if (queue.length) requestAnimationFrame(tick);
    else pumping = false;
  };
  requestAnimationFrame(tick);
}
function paint(recipe, seed, w, h, priority) {
  const key = JSON.stringify([recipe, seed, w, h]);
  let p = artCache.get(key);
  if (!p) {
    p = new Promise((done) => {
      let at = queue.length;
      while (at > 1 && queue[at - 1].priority > priority) at--;
      queue.splice(at, 0, { job: createArtJob(w, h, recipe, seed), priority, done });
      pump();
    });
    artCache.set(key, p);
  }
  return p;
}
const recipeOf = (art) => (typeof art === "string" ? ART_PRESETS[art] : art);

/** A canvas that fades in once its artwork has been painted. */
export function artCanvas(art, seed, w, h, density, priority, className) {
  const c = document.createElement("canvas");
  c.className = className;
  c.setAttribute("aria-hidden", "true");
  c.style.width = w + "px";
  c.style.height = h + "px";
  paint(recipeOf(art), seed, Math.round(w * density), Math.round(h * density), priority).then((src) => {
    c.width = src.width;
    c.height = src.height;
    c.getContext("2d").drawImage(src, 0, 0);
    c.dataset.ready = "";
  });
  return c;
}
// #endregion

/* ---------- page background ---------- */
export function paintBackground(host, art, seed) {
  host.querySelectorAll("canvas").forEach((c) => c.remove());
  const vw = Math.min(window.innerWidth, 1600), vh = Math.min(window.innerHeight, 1000);
  const c = artCanvas(art, seed, vw, vh, 0.5, 0, "bg-art");
  c.style.width = "";
  c.style.height = "";
  host.prepend(c);
}

/* ---------- folder ---------- */
const W = 340, H = 272, HEADROOM = 54, FRONT_TOP = 74, FRONT_H = H - FRONT_TOP;
const SHEET_W = 272, SHEET_H = 184, MAX_VISIBLE = 4;
const BACK_PATH =
  "M0 56A22 22 0 0 1 22 34L146 34C170 34 172 0 198 0L318 0A22 22 0 0 1 340 22L340 250A22 22 0 0 1 318 272L22 272A22 22 0 0 1 0 250Z";
const FRONT_PATH =
  `M0 22A22 22 0 0 1 22 0L118 0C144 0 146 32 172 32L318 32A22 22 0 0 1 340 54L340 ${FRONT_H - 22}` +
  `A22 22 0 0 1 318 ${FRONT_H}L22 ${FRONT_H}A22 22 0 0 1 0 ${FRONT_H - 22}Z`;
const fanSlot = (i, n) => (n <= 1 ? 0 : (i / (n - 1)) * 2 - 1);
let uidCounter = 0;

const el = (tag, cls, attrs = {}) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};
const svg = (tag, attrs = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};

/**
 * createFolder({ label, tone, art, seed, width, action: "add"|"more"|"none",
 *   labelPlacement, onAction, onToggle }) → { el, setFiles, setMeta, setOpen, setTone }
 * files: [{ name, meta?, art? }]
 */
export function createFolder(opts) {
  const { label = "Folder", tone = "light", art = "aurora", seed = 3, width = 220, action = "add",
    labelPlacement = "bottom-left", onAction, onToggle, appearDelay = 0 } = opts;
  const uid = "gf" + uidCounter++;
  const k = width / W;
  const density = Math.min(1.5, Math.max(1, window.devicePixelRatio || 1)) * k;
  let open = false, files = [], names = new Set();

  const root = el("div", "gf-root", { "data-tone": tone, "data-place": labelPlacement });
  root.style.width = width + "px";
  root.style.height = (H + HEADROOM) * k + "px";
  const stage = el("div", "gf-stage");
  stage.style.transform = `translateY(${HEADROOM * k}px) scale(${k})`;
  const tilt = el("div", "gf-tilt");
  tilt.append(el("div", "gf-shadow", { "aria-hidden": "true" }));

  const back = el("div", "gf-back", { "aria-hidden": "true" });
  back.style.clipPath = `path('${BACK_PATH}')`;
  if (art !== "none") back.append(artCanvas(art, seed, W + 24, H + 24, density, 0, "gf-art"));
  back.append(el("div", "gf-back-sheen"));
  const backRim = svg("svg", { class: "gf-rim gf-back-rim", viewBox: `0 0 ${W} ${H}`, "aria-hidden": "true" });
  backRim.append(svg("path", { d: BACK_PATH }));
  tilt.append(back, backRim);

  const front = el("div", "gf-front");
  front.style.clipPath = `path('${FRONT_PATH}')`;
  const rim = svg("svg", { class: "gf-rim gf-front-rim", viewBox: `0 0 ${W} ${FRONT_H}`, "aria-hidden": "true" });
  const defs = svg("defs");
  const grad = svg("linearGradient", { id: "rim-" + uid, x1: 0, y1: 0, x2: 0, y2: 1 });
  [[0, 0.95], [0.35, 0.35], [1, 0.12]].forEach(([o, a]) =>
    grad.append(svg("stop", { offset: o, "stop-color": "#fff", "stop-opacity": a })));
  defs.append(grad);
  rim.append(defs, svg("path", { d: FRONT_PATH, stroke: `url(#rim-${uid})` }));
  const hit = el("button", "gf-hit", { type: "button", "aria-expanded": "false" });
  const lab = el("div", "gf-label");
  const title = el("span", "gf-title");
  title.textContent = label;
  let metaEl = el("span", "gf-meta");
  lab.append(title, metaEl);
  front.append(el("div", "gf-tint", { "aria-hidden": "true" }), el("div", "gf-spec", { "aria-hidden": "true" }), rim, hit, lab);
  tilt.append(front);

  const filesBox = el("div", "gf-files", { role: "list", "aria-label": label + " contents" });
  filesBox.inert = true;
  tilt.append(filesBox);

  let key = null;
  if (action !== "none") {
    key = el("button", "gf-key gf-key--" + action, { type: "button", "aria-label": (action === "add" ? "Add to " : "Options for ") + label });
    const g = action === "add"
      ? svg("svg", { class: "gf-glyph gf-glyph--plus", viewBox: "0 0 24 24", "aria-hidden": "true" })
      : svg("svg", { class: "gf-glyph gf-glyph--more", viewBox: "0 0 30 12", "aria-hidden": "true" });
    if (action === "add") g.append(svg("path", { d: "M12 4.5v15M4.5 12h15" }));
    else [1, 11, 21].forEach((x) => g.append(svg("rect", { x, y: 1.5, width: 8, height: 9, rx: 3.4 })));
    key.append(g);
    key.onclick = (e) => { e.stopPropagation(); onAction?.(); };
    tilt.append(key);
  }
  stage.append(tilt);
  root.append(stage);

  function setOpen(next, notify = true) {
    open = next;
    root.toggleAttribute("data-open", open);
    hit.setAttribute("aria-expanded", String(open));
    hit.setAttribute("aria-label", (open ? "Close " : "Open ") + label);
    filesBox.inert = !open;
    if (notify) onToggle?.(open);
  }
  hit.onclick = () => setOpen(!open);

  function setMeta(text) {
    if (metaEl.textContent === text) return;
    const fresh = el("span", "gf-meta");
    fresh.textContent = text;
    metaEl.replaceWith(fresh);
    metaEl = fresh;
  }

  function setFiles(next) {
    const added = next.find((f) => !names.has(f.name) && names.size > 0);
    files = next;
    names = new Set(next.map((f) => f.name));
    const start = Math.max(0, files.length - MAX_VISIBLE);
    const visible = files.slice(start);
    filesBox.textContent = "";
    visible.forEach((file, v) => {
      const jitter = rng(hashString(file.name));
      const sheet = el("button", "gf-sheet" + (file.art ? " gf-sheet--art" : ""), { type: "button", role: "listitem" });
      if (added && file.name === added.name) {
        sheet.classList.add("gf-sheet--drop");
        sheet.addEventListener("animationend", () => sheet.classList.remove("gf-sheet--drop"), { once: true });
      }
      sheet.style.setProperty("--gf-slot", fanSlot(v, visible.length));
      sheet.style.setProperty("--gf-depth", visible.length - 1 - v);
      sheet.style.setProperty("--gf-jx", ((jitter() - 0.5) * 10).toFixed(2) + "px");
      sheet.style.setProperty("--gf-jr", ((jitter() - 0.5) * 3.2).toFixed(2) + "deg");
      sheet.style.zIndex = v + 1;
      if (file.art) {
        sheet.append(artCanvas(file.art, hashString(file.name) % 997, SHEET_W, SHEET_H, Math.min(density, 1.25), 1, "gf-sheet-art"));
      } else {
        const lines = el("span", "gf-lines", { "aria-hidden": "true" });
        ["86%", "92%", "64%", "88%", "42%"].forEach((w) => { const i = el("i"); i.style.width = w; lines.append(i); });
        sheet.append(lines);
      }
      if (file.glyph) {
        const g = el("span", "gf-sheet-glyph");
        g.textContent = file.glyph;
        sheet.append(g);
      }
      const head = el("span", "gf-sheet-head");
      const n = el("span", "gf-sheet-name");
      n.textContent = file.name;
      head.append(n);
      if (file.meta) { const m = el("span", "gf-sheet-meta"); m.textContent = file.meta; head.append(m); }
      sheet.append(head);
      sheet.onclick = () => opts.onFileSelect?.(file);
      filesBox.append(sheet);
    });
    if (added) {
      front.classList.remove("gf-front--bump");
      void front.offsetWidth;
      front.classList.add("gf-front--bump");
    }
  }

  // pointer parallax, without re-rendering
  let raf = 0, px = 0, py = 0;
  const write = () => { raf = 0; root.style.setProperty("--gf-px", px.toFixed(3)); root.style.setProperty("--gf-py", py.toFixed(3)); };
  root.addEventListener("pointermove", (e) => {
    if (e.pointerType === "touch") return;
    const r = root.getBoundingClientRect();
    px = ((e.clientX - r.left) / r.width) * 2 - 1;
    py = ((e.clientY - r.top) / r.height) * 2 - 1;
    if (!raf) raf = requestAnimationFrame(write);
  });
  root.addEventListener("pointerleave", () => { px = 0; py = 0; if (!raf) raf = requestAnimationFrame(write); });

  // assemble the first time it scrolls into view
  if (typeof IntersectionObserver !== "undefined") {
    const io = new IntersectionObserver((entries) => {
      if (!entries.some((e) => e.isIntersecting)) return;
      io.disconnect();
      setTimeout(() => root.setAttribute("data-entered", ""), appearDelay);
    }, { threshold: 0.25 });
    io.observe(root);
  } else root.setAttribute("data-entered", "");

  return {
    el: root, setFiles, setMeta, setOpen,
    get open() { return open; },
    setTone: (t) => root.setAttribute("data-tone", t),
  };
}
