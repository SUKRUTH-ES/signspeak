"""Unpack the ASL parquet dataset into data/images/<LETTER>/<n>.jpg plus a manifest."""
import json
import pathlib

import pyarrow.parquet as pq

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "asl.parquet"
OUT = ROOT / "data" / "images"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

table = pq.read_table(SRC)
images = table.column("image").to_pylist()
labels = table.column("label").to_pylist()

manifest = []
counts = {}
for img, lab in zip(images, labels):
    letter = LETTERS[lab]
    n = counts.get(letter, 0)
    counts[letter] = n + 1
    rel = f"{letter}/{n}.jpg"
    path = OUT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(img["bytes"])
    manifest.append({"label": letter, "src": rel})

(OUT / "manifest.json").write_text(json.dumps(manifest))
print(f"wrote {len(manifest)} images to {OUT}")
