"""Build the results/ folder: accuracy progression, per-letter accuracy,
confusion matrix and a text training report. Same split as train_asl.py.

    python tools/make_report.py
"""
import json
import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# palette (reference data-viz palette, light surface)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"
BLUE_SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 11, "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
})

IU = np.triu_indices(21, 1)


def canon(P):
    P = P.copy()
    a = P[:, 5, :2] - P[:, 0, :2]
    b = P[:, 17, :2] - P[:, 0, :2]
    P[(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) < 0, :, 0] *= -1
    return P


def dists(P):
    return np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)[:, IU[0], IU[1]]


data = json.loads((ROOT / "data" / "landmarks_raw.json").read_text())
P = np.array([d["v"] for d in data], dtype=np.float64).reshape(-1, 21, 3)
y = np.array([d["label"] for d in data])
idx_tr, idx_te = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)
ytr, yte = y[idx_tr], y[idx_te]

# ---- accuracy progression -------------------------------------------------
raw = P.reshape(len(P), -1)
can = canon(P).reshape(len(P), -1)
full = np.hstack([can, dists(canon(P))])
sc = StandardScaler().fit(full[idx_tr])
fs = sc.transform(full)

stages = []
def run(name, model, X):
    model.fit(X[idx_tr], ytr)
    acc = (model.predict(X[idx_te]) == yte).mean()
    stages.append((name, acc))
    return model

run("Raw keypoints\n(k-NN)", KNeighborsClassifier(5), raw)
run("+ left & right hands\nunified (k-NN)", KNeighborsClassifier(5), can)
run("+ joint distances,\nscaled (k-NN)", KNeighborsClassifier(5), fs)
mlp = run("Neural network\n273 → 128 → 26", MLPClassifier((128,), alpha=0.1, max_iter=1000, random_state=0), fs)
pred = mlp.predict(fs[idx_te])

fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
names = [s[0] for s in stages][::-1]
accs = [s[1] * 100 for s in stages][::-1]
bars = ax.barh(names, accs, height=0.42, color=BLUE)
for b, a in zip(bars, accs):
    ax.text(b.get_width() - 1.2, b.get_y() + b.get_height() / 2, f"{a:.1f}%",
            va="center", ha="right", color="white", fontweight="bold", fontsize=11)
ax.set_xlim(0, 100)
ax.set_xlabel("Test accuracy (%)")
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_title("ASL alphabet: how each step improved accuracy", pad=14)
fig.tight_layout()
fig.savefig(OUT / "accuracy_progression.png")
plt.close(fig)

# ---- per-letter accuracy --------------------------------------------------
labels = sorted(set(y))
per = [(l, (pred[yte == l] == l).mean() * 100) for l in labels]
fig, ax = plt.subplots(figsize=(10, 3.8), dpi=160)
ax.bar([l for l, _ in per], [a for _, a in per], width=0.62, color=BLUE)
ax.axhline(np.mean([a for _, a in per]), color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(len(labels) - 0.5, np.mean([a for _, a in per]) + 1.5, f"mean {np.mean([a for _, a in per]):.1f}%",
        ha="right", va="bottom", color=INK_2, fontsize=10)
worst = min(per, key=lambda t: t[1])
ax.annotate(f"{worst[0]}: {worst[1]:.0f}%", (labels.index(worst[0]), worst[1]),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=9.5, fontweight="bold")
ax.set_ylim(0, 105)
ax.set_ylabel("Test accuracy (%)")
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.set_title("Accuracy per letter (neural network, held-out test set)", pad=14)
fig.tight_layout()
fig.savefig(OUT / "per_letter_accuracy.png")
plt.close(fig)

# ---- confusion matrix -----------------------------------------------------
cm = confusion_matrix(yte, pred, labels=labels)
norm = cm / cm.sum(axis=1, keepdims=True)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seq_blue", BLUE_SEQ)
fig, ax = plt.subplots(figsize=(9, 8.4), dpi=160)
im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
ax.set_xticks(range(len(labels)), labels)
ax.set_yticks(range(len(labels)), labels)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
for i in range(len(labels)):
    for j in range(len(labels)):
        if cm[i, j] and (i == j or cm[i, j] >= 1):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5,
                    color="white" if norm[i, j] > 0.5 else INK)
ax.set_xlabel("Predicted letter")
ax.set_ylabel("True letter")
ax.set_title(f"Confusion matrix · {len(yte)} test images · {(pred == yte).mean():.1%} accuracy", pad=14)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
cb.set_label("Share of the true letter's test images", color=INK_2)
cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "confusion_matrix.png")
plt.close(fig)

# ---- text report ----------------------------------------------------------
pairs = sorted(((cm[i, j], labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i != j and cm[i, j]), reverse=True)
lines = [
    "SignSpeak: ASL alphabet model report",
    "=" * 40,
    f"Samples with a detected hand: {len(y)}  (train {len(ytr)} / test {len(yte)}, stratified 80/20, seed 42)",
    f"Features: 63 normalised keypoints (one chirality) + 210 pairwise joint distances = 273",
    "",
    "Accuracy progression (same test set):",
    *[f"  {n.replace(chr(10), ' '):45s} {a:.1%}" for n, a in stages],
    "",
    "Most common confusions (true -> predicted: count):",
    *[f"  {t} -> {p}: {c}" for c, t, p in pairs[:10]],
    "",
    "Per-letter report (neural network):",
    classification_report(yte, pred, digits=3),
]
(OUT / "training_report.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:16]).encode("ascii", "replace").decode())
print("wrote", [p.name for p in OUT.iterdir()])
