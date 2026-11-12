// Glass UI layer: page background + the two sign folders. app.js owns the data
// and announces changes with a "signs:update" event; this file only draws.
import { createFolder, paintBackground } from "./glass.js";

const $ = (id) => document.getElementById(id);
const darkQuery = matchMedia("(prefers-color-scheme: dark)");
const isDark = () => {
  const forced = document.documentElement.dataset.theme;
  return forced ? forced === "dark" : darkQuery.matches;
};
const tone = () => (isDark() ? "dark" : "light");

function paintBg() {
  paintBackground($("bgArt"), isDark() ? "ember" : "aurora", isDark() ? 11 : 3);
}

const panels = { asl: $("panelAsl"), mine: $("panelMine") };
let current = null;
function select(name, isOpen) {
  if (isOpen) {
    current = name;
    (name === "asl" ? mine : asl).setOpen(false, false);
  } else if (current === name) {
    current = null;
  }
  for (const [n, p] of Object.entries(panels)) p.classList.toggle("hidden", n !== current);
  $("shelfHint").classList.toggle("hidden", !!current);
}

// two folders side by side, sized to the card they sit in
const shelfW = $("shelf").clientWidth || 360;
const width = Math.max(120, Math.min(210, Math.floor((shelfW - 18) / 2) - 4));
const asl = createFolder({
  label: "ASL Alphabet", art: "ember", seed: 11, width, action: "none", tone: tone(),
  labelPlacement: "top-left", appearDelay: 0,
  onToggle: (o) => select("asl", o),
});
const mine = createFolder({
  label: "My Signs", art: "aurora", seed: 3, width, action: "add", tone: tone(), appearDelay: 140,
  onAction: () => { mine.setOpen(true); $("newLabel").focus(); },
  onToggle: (o) => select("mine", o),
});
$("shelf").append(asl.el, mine.el);

const sheetArt = ["dusk", "verdant", "glacier", "aurora"];
document.addEventListener("signs:update", (e) => {
  const { labels, counts, aslLabels } = e.detail;
  mine.setFiles(labels.map((l, i) => ({
    name: l,
    meta: counts[l] ? counts[l] + " samples" : "untrained",
    art: counts[l] ? sheetArt[i % sheetArt.length] : undefined,
  })));
  mine.setMeta(labels.length + (labels.length === 1 ? " sign" : " signs"));
  if (aslLabels.length) {
    asl.setFiles(aslLabels.map((l) => ({ name: "Letter " + l, glyph: l, meta: "ASL", art: "ember" })));
    asl.setMeta(aslLabels.length + " letters · ready");
  } else {
    asl.setFiles([{ name: "asl-model.json", meta: "not loaded" }]);
    asl.setMeta("Open to load");
  }
});

darkQuery.addEventListener("change", () => {
  asl.setTone(tone());
  mine.setTone(tone());
  paintBg();
});

paintBg();
asl.setOpen(true);
