"""Loading model pairs and run folders."""
from __future__ import annotations

import json
import os

import torch

from .kv import check_matched_kv

# Every pair here is "matched-KV": same number of KV heads (8) and head size (128).
PAIRS = {
    "0.6b-1.7b": ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"),  # fits on ~16-24 GB GPUs
    "1.7b-4b": ("Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"),      # ~24 GB GPUs
    "4b-8b": ("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"),          # 40 GB+ GPUs (default)
    "8b-32b": ("Qwen/Qwen3-8B", "Qwen/Qwen3-32B"),        # 80 GB+ (pair tested by Heo et al.)
}


def pick_devices(spec: str | None):
    if spec:
        devs = [d.strip() for d in spec.split(",")]
        return (devs[0], devs[-1])
    d = "cuda" if torch.cuda.is_available() else "cpu"
    return (d, d)


def load_model(name: str, device: str):
    import transformers
    from packaging.version import Version
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    # the argument was renamed torch_dtype -> dtype in transformers 4.56
    kw = "dtype" if Version(transformers.__version__) >= Version("4.56") else "torch_dtype"
    m = AutoModelForCausalLM.from_pretrained(name, **{kw: dtype})
    assert next(m.parameters()).dtype == dtype, "model loaded in the wrong precision"
    return m.to(device).eval()


def load_pair(small: str, large: str, devices=None):
    """Returns ({'S': model, 'L': model}, tokenizer wrapper)."""
    from transformers import AutoTokenizer

    from .data import Tok

    ds, dl = pick_devices(devices)
    print(f"loading {small} on {ds} and {large} on {dl}")
    if "cpu" in (str(ds), str(dl)):
        print("\n[warning] running on CPU: PyTorch can't see a GPU. Fitting and the kill test will take HOURS.\n"
              "  Check: python -c \"import torch; print(torch.cuda.is_available())\"\n"
              "  On Windows, plain `pip install torch` gives a CPU-only build; get the CUDA build from\n"
              "  https://pytorch.org/get-started/locally/ (only helps if the machine has an NVIDIA GPU).\n")
    ms, ml = load_model(small, ds), load_model(large, dl)
    check_matched_kv(ms.config, ml.config)
    tok_s, tok_l = AutoTokenizer.from_pretrained(small), AutoTokenizer.from_pretrained(large)
    if tok_s.get_vocab() != tok_l.get_vocab():
        raise ValueError("the two models use different tokenizers; pages wouldn't line up token by token")
    return {"S": ms, "L": ml}, Tok(tok_l)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
