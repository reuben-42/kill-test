"""Optional: save CoQA + calibration text locally, for GPU nodes without internet.

    python scripts/download_data.py --out data/      # run on a machine WITH internet

Then pass --coqa-file data/coqa_validation.json and --calib-text-file data/calib.txt.
Models: run `huggingface-cli download Qwen/Qwen3-4B` (etc.) on the same machine so they're cached.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from killtest.data import load_coqa  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data")
    ap.add_argument("--n-calib-docs", type=int, default=2000, help="FineWeb-Edu documents to save")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "coqa_validation.json"), "w") as f:
        json.dump(load_coqa("validation"), f)
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    with open(os.path.join(a.out, "calib.txt"), "w") as f:
        for i, r in enumerate(ds):
            if i == a.n_calib_docs:
                break
            f.write(r["text"].replace("\n\n", "\n").strip() + "\n\n")
    print(f"saved to {a.out}/")


if __name__ == "__main__":
    main()
