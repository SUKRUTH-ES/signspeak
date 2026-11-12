"""Train the ASL alphabet classifier and export it for the browser.

Input : data/landmarks_raw.json   [{label, v: 63 floats}] from tools/build.html
Output: asl-model.json            small MLP (weights + scaler), run in app.js

Features (must match aslFeatures() in features.js):
  1. wrist-centred, scale-normalised 21x3 keypoints
  2. mirrored so every hand has the same chirality (left == right hand)
  3. + all 210 pairwise keypoint distances
"""
import json
import pathlib
import warnings

import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "landmarks_raw.json"
OUT = ROOT / "asl-model.json"
IU = np.triu_indices(21, 1)


def features(v63):
    P = np.asarray(v63, dtype=np.float64).reshape(-1, 21, 3).copy()
    a = P[:, 5, :2] - P[:, 0, :2]
    b = P[:, 17, :2] - P[:, 0, :2]
    flip = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) < 0
    P[flip, :, 0] *= -1
    D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)[:, IU[0], IU[1]]
    return np.hstack([P.reshape(len(P), -1), D])


data = json.loads(RAW.read_text())
X = features([d["v"] for d in data])
y = np.array([d["label"] for d in data])
print(f"{len(X)} samples, {X.shape[1]} features, {len(set(y))} letters")

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler().fit(Xtr)
A, B = scaler.transform(Xtr), scaler.transform(Xte)

knn = KNeighborsClassifier(5).fit(A, ytr)
print(f"baseline k-NN accuracy: {(knn.predict(B) == yte).mean():.1%}")

mlp = MLPClassifier((128,), alpha=0.1, max_iter=1000, random_state=0).fit(A, ytr)
acc = (mlp.predict(B) == yte).mean()
print(f"MLP test accuracy:      {acc:.1%}\n")
print(classification_report(yte, mlp.predict(B), digits=2))

# final model on all the data
scaler = StandardScaler().fit(X)
final = MLPClassifier((128,), alpha=0.1, max_iter=1000, random_state=0).fit(scaler.transform(X), y)
r = lambda a: np.round(a, 5).tolist()
model = {
    "type": "mlp",
    "labels": [str(c) for c in final.classes_],
    "testAccuracy": round(float(acc), 4),
    "mean": r(scaler.mean_),
    "scale": r(scaler.scale_),
    "W1": r(final.coefs_[0]), "b1": r(final.intercepts_[0]),
    "W2": r(final.coefs_[1]), "b2": r(final.intercepts_[1]),
}
OUT.write_text(json.dumps(model, separators=(",", ":")))
print(f"wrote {OUT.name}: {OUT.stat().st_size / 1e6:.2f} MB")
