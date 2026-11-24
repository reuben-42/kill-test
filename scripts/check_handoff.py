"""Step 2: sanity check ONE handoff before running long chats.

    python scripts/check_handoff.py --run runs/4b-8b

For each CoQA story: the target model reads the first question itself, but gets the
story as translated notes from the other model. Compared with the target reading
everything itself. If this is bad, fix the translator before the kill test.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from killtest.data import HEADER, load_coqa  # noqa: E402
from killtest.load import load_json, load_pair, save_json  # noqa: E402
from killtest.mapper import Mapper  # noqa: E402
from killtest.runner import single_handoff  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n", type=int, default=50, help="number of stories")
    ap.add_argument("--devices")
    ap.add_argument("--coqa-file", help="local JSON copy of CoQA (see scripts/download_data.py)")
    ap.add_argument("--mapper-dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()

    cfg = load_json(os.path.join(a.run, "run_config.json"))
    models, tok = load_pair(cfg["small"], cfg["large"], a.devices or cfg.get("devices"))
    dt = getattr(torch, a.mapper_dtype)
    stories = load_coqa("validation", a.coqa_file)[-a.n:]  # the kill test draws sessions with a seed; overlap doesn't matter here
    report = {}
    for src, tgt in (("S", "L"), ("L", "S")):
        mp = Mapper.load(os.path.join(a.run, f"mapper_{src}2{tgt}.pt"), models[src].config, models[tgt].config)
        mp = mp.to(models[tgt].device, dt)
        rows = []
        for s in stories:
            ctx = tok.encode(HEADER + f"Story:\n{s['story'].strip()}\n\n")
            q = tok.encode(f"Q: {s['questions'][0].strip()}\nA:")
            rows.append(single_handoff(models[src], models[tgt], mp, ctx, q, tok, s["answers"][0].strip()))
        nf, tf = np.mean([r["native_f1"] for r in rows]), np.mean([r["translated_f1"] for r in rows])
        report[f"{src}2{tgt}"] = dict(native_f1=nf, translated_f1=tf, retention=tf / max(nf, 1e-9),
                                      agree_f1=np.mean([r["agree_f1"] for r in rows]),
                                      mean_kl=np.mean([r["kl"] for r in rows]),
                                      top1_agree=np.mean([r["top1_agree"] for r in rows]), examples=rows[:5])
        r = report[f"{src}2{tgt}"]
        print(f"\n{src} -> {tgt}: F1 native {nf:.3f} | translated {tf:.3f} | retention {r['retention']:.0%} | "
              f"same first token {r['top1_agree']:.0%} | KL {r['mean_kl']:.3f}")
    save_json(report, os.path.join(a.run, "handoff_report.json"))
    print("\nRough guide: Heo et al. kept 73-98% of accuracy on their good pairs (different benchmarks).\n"
          "If retention here is far below ~70%, check the translator before running the kill test.\n"
          f"Next: python scripts/run_kill_test.py --run {a.run}")


if __name__ == "__main__":
    main()
