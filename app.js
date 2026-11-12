import {
  HandLandmarker,
  FilesetResolver,
  DrawingUtils,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import { features, aslFeatures } from "./features.js";
import "./ui.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const STORE_KEY = "signspeak.v1";
const DEFAULT_SIGNS = ["Hello", "Thank you", "Yes", "No", "Help"];
const RECORD_MS = 3000;

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), canvas: $("overlay"), status: $("status"), startBtn: $("startBtn"),
  placeholder: $("camPlaceholder"), countdown: $("countdown"),
  predLabel: $("predLabel"), confText: $("confText"), confBar: $("confBar"), holdBar: $("holdBar"),
  sentence: $("sentence"), autoSpeak: $("autoSpeak"),
  speakBtn: $("speakBtn"), undoBtn: $("undoBtn"), clearBtn: $("clearBtn"), spaceBtn: $("spaceBtn"),
  camBtn: $("camBtn"), stopOverlayBtn: $("stopOverlayBtn"),
  aslBtn: $("aslBtn"), aslInfo: $("aslInfo"), aslGrid: $("aslGrid"),
  addForm: $("addForm"), newLabel: $("newLabel"), signList: $("signList"), totalSamples: $("totalSamples"),
  kRange: $("kRange"), kVal: $("kVal"), distRange: $("distRange"), distVal: $("distVal"),
  holdRange: $("holdRange"), holdVal: $("holdVal"),
  exportBtn: $("exportBtn"), importFile: $("importFile"), resetBtn: $("resetBtn"),
};
const ctx = els.canvas.getContext("2d");

const state = {
  labels: [],          // ordered list of sign names
  samples: [],         // { label, v: number[] }
  recording: null,     // { label, until }
  words: [],
  candidate: null, stable: 0, committed: false,
  asl: null, aslLabels: [], aslMinConf: 0.7,  // pretrained ASL alphabet network
  spelling: false,         // last token is a word being fingerspelled
  noHandFrames: 0,
  k: 5, maxDist: 1.2, holdFrames: 12,
  landmarker: null, running: false, lastVideoTime: -1,
};

/* ---------- persistence ---------- */
function save() {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify({ labels: state.labels, samples: state.samples }));
  } catch { /* storage unavailable; data stays in memory */ }
}
function load() {
  try {
    const data = JSON.parse(localStorage.getItem(STORE_KEY) || "null");
    if (data && Array.isArray(data.labels)) {
      state.labels = data.labels;
      state.samples = Array.isArray(data.samples) ? data.samples : [];
      return;
    }
  } catch { /* ignore */ }
  state.labels = [...DEFAULT_SIGNS];
}

/* ---------- pretrained ASL alphabet ---------- */
async function loadAsl() {
  els.aslBtn.disabled = true;
  els.aslInfo.textContent = "Loading pretrained model…";
  try {
    const data = await (await fetch("asl-model.json")).json();
    state.asl = data;
    state.aslLabels = data.labels;
    els.aslInfo.textContent =
      `${data.labels.length} letters · neural net · ${(data.testAccuracy * 100).toFixed(1)}% test accuracy`;
    els.aslBtn.textContent = "Unload";
    els.aslGrid.innerHTML = data.labels.map((l) => `<span data-l="${l}">${l}</span>`).join("");
    els.aslGrid.classList.remove("hidden");
    try { localStorage.setItem(STORE_KEY + ".asl", "1"); } catch {}
  } catch (err) {
    console.error(err);
    els.aslInfo.textContent = "Couldn't load asl-model.json";
  }
  els.aslBtn.disabled = false;
  renderSigns();
}
function unloadAsl() {
  state.asl = null; state.aslLabels = [];
  els.aslInfo.textContent = "Pretrained on real ASL photos: fingerspell words";
  els.aslBtn.textContent = "Load";
  els.aslGrid.classList.add("hidden");
  try { localStorage.removeItem(STORE_KEY + ".asl"); } catch {}
  renderSigns();
}

/* ---------- k-NN classifier ---------- */
function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
  return Math.sqrt(s);
}
function classify(v) {
  const k = state.k;
  const nearest = []; // kept sorted, length <= k
  const scan = (list) => {
    for (const s of list) {
      const d = dist(v, s.v);
      if (nearest.length < k || d < nearest[nearest.length - 1].d) {
        let i = nearest.length;
        while (i > 0 && nearest[i - 1].d > d) i--;
        nearest.splice(i, 0, { label: s.label, d });
        if (nearest.length > k) nearest.pop();
      }
    }
  };
  scan(state.samples);
  if (nearest.length === 0) return null;
  const votes = {};
  for (const n of nearest) votes[n.label] = (votes[n.label] || 0) + 1;
  const [label, count] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  const avgDist = nearest.filter((n) => n.label === label).reduce((s, n) => s + n.d, 0) / count;
  if (avgDist > state.maxDist) return { label: null, conf: 0 };
  return { label, conf: count / nearest.length };
}

/* ---------- ASL neural net (273 → 128 ReLU → 26 softmax) ---------- */
function predictAsl(lm) {
  const m = state.asl;
  const x = aslFeatures(lm).map((v, i) => (v - m.mean[i]) / m.scale[i]);
  const h = m.b1.slice();
  for (let i = 0; i < x.length; i++) {
    const xi = x[i], row = m.W1[i];
    for (let j = 0; j < h.length; j++) h[j] += xi * row[j];
  }
  for (let j = 0; j < h.length; j++) if (h[j] < 0) h[j] = 0;
  const z = m.b2.slice();
  for (let j = 0; j < h.length; j++) {
    const hj = h[j], row = m.W2[j];
    if (hj) for (let c = 0; c < z.length; c++) z[c] += hj * row[c];
  }
  const max = Math.max(...z);
  const e = z.map((v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  let best = 0;
  for (let c = 1; c < e.length; c++) if (e[c] > e[best]) best = c;
  return { label: m.labels[best], conf: e[best] / sum };
}

// Your own recorded signs win when they match closely; otherwise, with a single
// hand in view, the pretrained ASL alphabet model has a go.
function predict(vec, result) {
  const custom = state.samples.length ? classify(vec) : null;
  if (custom && custom.label) return custom;
  if (state.asl && result.landmarks.length === 1) {
    const p = predictAsl(result.landmarks[0]);
    return p.conf >= state.aslMinConf ? p : { label: null, conf: p.conf };
  }
  return custom;
}

/* ---------- sentence ---------- */
function speak(text) {
  if (!text || !("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}
function renderSentence() {
  const empty = state.words.length === 0;
  els.sentence.classList.toggle("empty", empty);
  if (empty) { els.sentence.textContent = "Your signed words will appear here…"; return; }
  els.sentence.textContent = "";
  state.words.forEach((w, i) => {
    if (i) els.sentence.append(" ");
    const span = document.createElement("span");
    span.textContent = w;
    if (state.spelling && i === state.words.length - 1) span.className = "typing";
    els.sentence.append(span);
  });
}
// Single letters (fingerspelling) join into the current word; anything else is a whole word.
function commitWord(word) {
  if (word.length === 1) {
    if (state.spelling) state.words[state.words.length - 1] += word;
    else { state.words.push(word); state.spelling = true; }
  } else {
    finishSpelling(false);
    state.words.push(word);
  }
  renderSentence();
  if (els.autoSpeak.checked) speak(word);
}
function finishSpelling(sayIt = true) {
  if (!state.spelling) return;
  state.spelling = false;
  const w = state.words[state.words.length - 1];
  if (sayIt && els.autoSpeak.checked && w.length > 1) speak(w);
  renderSentence();
}

/* ---------- sign list UI ---------- */
function countFor(label) { return state.samples.filter((s) => s.label === label).length; }
function renderSigns(activeLabel = null) {
  els.signList.innerHTML = "";
  for (const label of state.labels) {
    const li = document.createElement("li");
    if (label === activeLabel) li.classList.add("active");
    const n = countFor(label);
    li.innerHTML = `
      <div><span class="name"></span><span class="count">${n} sample${n === 1 ? "" : "s"}</span></div>
      <div class="actions">
        <button class="btn small primary" data-act="rec">● Record</button>
        <button class="btn small" data-act="del" title="Delete sign">✕</button>
      </div>`;
    li.querySelector(".name").textContent = label;
    li.querySelector('[data-act="rec"]').disabled = !state.running || !!state.recording;
    li.querySelector('[data-act="rec"]').onclick = () => startRecording(label);
    li.querySelector('[data-act="del"]').onclick = () => {
      state.labels = state.labels.filter((l) => l !== label);
      state.samples = state.samples.filter((s) => s.label !== label);
      save(); renderSigns();
    };
    els.signList.appendChild(li);
  }
  els.totalSamples.textContent = `${state.samples.length} samples`;
  const counts = Object.fromEntries(state.labels.map((l) => [l, countFor(l)]));
  document.dispatchEvent(new CustomEvent("signs:update", {
    detail: { labels: state.labels, counts, aslLabels: state.aslLabels },
  }));
}
let lastActive = null;
function highlight(label) {
  if (label === lastActive) return;
  lastActive = label;
  els.signList.querySelectorAll("li").forEach((li) => {
    li.classList.toggle("active", li.querySelector(".name").textContent === label);
  });
  els.aslGrid.querySelectorAll("span").forEach((s) => {
    s.classList.toggle("active", s.dataset.l === label);
  });
}

/* ---------- recording ---------- */
function startRecording(label) {
  if (!state.running || state.recording) return;
  let n = 3;
  els.countdown.className = "countdown";
  els.countdown.textContent = n;
  renderSigns();
  const tick = setInterval(() => {
    n -= 1;
    if (n > 0) { els.countdown.textContent = n; return; }
    clearInterval(tick);
    state.recording = { label, until: performance.now() + RECORD_MS };
    els.countdown.className = "countdown recording";
    els.countdown.textContent = `● Recording "${label}"…`;
  }, 700);
}
function stopRecording() {
  state.recording = null;
  els.countdown.className = "countdown hidden";
  save();
  renderSigns();
}

/* ---------- camera + loop ---------- */
async function start() {
  els.startBtn.disabled = true;
  els.status.textContent = "Loading model…";
  try {
    if (!state.landmarker) {
      const vision = await FilesetResolver.forVisionTasks(WASM_URL);
      state.landmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
      });
    }
    els.status.textContent = "Starting camera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
    });
    els.video.srcObject = stream;
    await els.video.play();
    els.canvas.width = els.video.videoWidth;
    els.canvas.height = els.video.videoHeight;
    els.placeholder.classList.add("hidden");
    els.status.textContent = "Live";
    els.status.classList.add("live");
    state.running = true;
    setCamButtons(true);
    renderSigns();
    requestAnimationFrame(loop);
  } catch (err) {
    console.error(err);
    els.status.textContent = "Camera/model error";
    els.placeholder.querySelector("p").textContent =
      "Couldn't start: " + (err.message || err) + ". Allow camera access and serve over http://localhost.";
    els.startBtn.disabled = false;
    setCamButtons(false);
  }
}

// Turn the camera fully off (releases the webcam light) without reloading.
function stop() {
  state.running = false;
  const stream = els.video.srcObject;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  els.video.srcObject = null;
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  if (state.recording) stopRecording();
  els.countdown.className = "countdown hidden";
  state.candidate = null; state.stable = 0; state.committed = false; state.lastVideoTime = -1;
  finishSpelling(false);
  showPrediction(null, 0, 0);
  els.placeholder.querySelector("p").textContent = "Camera is off";
  els.placeholder.classList.remove("hidden");
  els.startBtn.disabled = false;
  els.status.textContent = "Camera off";
  els.status.classList.remove("live");
  setCamButtons(false);
  renderSigns();
}

function setCamButtons(on) {
  els.camBtn.textContent = on ? "■ Stop camera" : "● Start camera";
  els.camBtn.classList.toggle("primary", !on);
  els.camBtn.disabled = false;
  els.stopOverlayBtn.classList.toggle("hidden", !on);
}

const drawer = new DrawingUtils(ctx);
function loop() {
  if (!state.running) return;
  const v = els.video;
  if (v.currentTime !== state.lastVideoTime) {
    state.lastVideoTime = v.currentTime;
    const result = state.landmarker.detectForVideo(v, performance.now());
    ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
    for (const lm of result.landmarks) {
      drawer.drawConnectors(lm, HandLandmarker.HAND_CONNECTIONS, { color: "#2fd3bc", lineWidth: 4 });
      drawer.drawLandmarks(lm, { color: "#7e72ff", lineWidth: 1, radius: 4 });
    }
    handleFrame(features(result), result);
  }
  requestAnimationFrame(loop);
}

function handleFrame(vec, result) {
  // recording mode: collect samples
  if (state.recording) {
    if (vec) state.samples.push({ label: state.recording.label, v: vec.map((x) => +x.toFixed(4)) });
    if (performance.now() > state.recording.until) stopRecording();
    return showPrediction(null, 0, 0);
  }

  if (!vec) {
    state.candidate = null; state.stable = 0; state.committed = false;
    // hand left the frame for ~1s → the fingerspelled word is finished
    if (++state.noHandFrames === 30) finishSpelling();
    return showPrediction(null, 0, 0);
  }
  state.noHandFrames = 0;

  const pred = predict(vec, result);
  const label = pred && pred.conf >= 0.6 ? pred.label : null;
  if (label && label === state.candidate) {
    state.stable += 1;
  } else {
    state.candidate = label; state.stable = label ? 1 : 0; state.committed = false;
  }
  if (label && !state.committed && state.stable >= state.holdFrames) {
    commitWord(label);
    state.committed = true;
  }
  const hold = state.committed ? 1 : Math.min(1, state.stable / state.holdFrames);
  showPrediction(label, pred ? pred.conf : 0, hold);
}

function showPrediction(label, conf, hold) {
  els.predLabel.textContent = label || (state.samples.length || state.asl ? "—" : "Record a sign first");
  els.confText.textContent = Math.round(conf * 100) + "%";
  els.confBar.style.width = conf * 100 + "%";
  els.holdBar.style.width = hold * 100 + "%";
  highlight(label);
}

/* ---------- controls ---------- */
els.startBtn.onclick = start;
els.camBtn.onclick = () => {
  if (state.running) return stop();
  els.camBtn.disabled = true;
  start();
};
els.stopOverlayBtn.onclick = stop;
els.speakBtn.onclick = () => speak(state.words.join(" "));
els.spaceBtn.onclick = () => finishSpelling();
els.undoBtn.onclick = () => {
  const last = state.words[state.words.length - 1];
  if (state.spelling && last && last.length > 1) state.words[state.words.length - 1] = last.slice(0, -1);
  else { state.words.pop(); state.spelling = false; }
  renderSentence();
};
els.clearBtn.onclick = () => { state.words = []; state.spelling = false; renderSentence(); };
els.aslBtn.onclick = () => (state.asl ? unloadAsl() : loadAsl());

els.addForm.onsubmit = (e) => {
  e.preventDefault();
  const name = els.newLabel.value.trim();
  if (!name || state.labels.includes(name)) return;
  state.labels.push(name);
  els.newLabel.value = "";
  save(); renderSigns();
};

const bindRange = (range, out, key, parse) => {
  range.oninput = () => { state[key] = parse(range.value); out.textContent = range.value; };
};
bindRange(els.kRange, els.kVal, "k", (x) => parseInt(x, 10));
bindRange(els.distRange, els.distVal, "maxDist", parseFloat);
bindRange(els.holdRange, els.holdVal, "holdFrames", (x) => parseInt(x, 10));

els.exportBtn.onclick = () => {
  const blob = new Blob([JSON.stringify({ labels: state.labels, samples: state.samples })], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "signspeak-model.json";
  a.click();
  URL.revokeObjectURL(a.href);
};
els.importFile.onchange = async () => {
  const file = els.importFile.files[0];
  if (!file) return;
  try {
    const data = JSON.parse(await file.text());
    if (!Array.isArray(data.labels) || !Array.isArray(data.samples)) throw new Error("bad file");
    state.labels = data.labels; state.samples = data.samples;
    save(); renderSigns();
  } catch { alert("That file isn't a SignSpeak model."); }
  els.importFile.value = "";
};
els.resetBtn.onclick = () => {
  if (!confirm("Delete all signs and samples?")) return;
  state.labels = [...DEFAULT_SIGNS]; state.samples = [];
  save(); renderSigns();
};

load();
renderSigns();
renderSentence();
try { if (localStorage.getItem(STORE_KEY + ".asl") === "1") loadAsl(); } catch {}
