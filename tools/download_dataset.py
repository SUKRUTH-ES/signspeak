"""Download the ASL alphabet dataset (~71 MB) from Hugging Face into data/asl.parquet.

Dataset: https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03
Only needed to rebuild keypoints from the photos; data/landmarks_raw.json is already in the repo.
"""
import pathlib
import urllib.request

URL = ("https://huggingface.co/datasets/Marxulia/asl_sign_languages_alphabets_v03"
       "/resolve/main/data/train-00000-of-00001.parquet")
OUT = pathlib.Path(__file__).resolve().parent.parent / "data" / "asl.parquet"

if OUT.exists():
    print(f"already present: {OUT}")
else:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def progress(blocks, size, total):
        if total > 0:
            print(f"\r{min(100, blocks * size * 100 // total)}%", end="", flush=True)

    urllib.request.urlretrieve(URL, OUT, progress)
    print(f"\nsaved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
