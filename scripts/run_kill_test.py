"""Step 3: play long CoQA chats three ways (always re-read / segment-wise / naive).

    python scripts/run_kill_test.py --run runs/4b-8b --n-sessions 20

Appends one line per turn to runs/<pair>/results.jsonl. Safe to stop and restart:
finished chats are skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from killtest.data import load_coqa, make_sessions, tokenize_session  # noqa: E402
from killtest.load import load_json, load_pair, save_json  # noqa: E402
from killtest.mapper import Mapper  # noqa: E402
from killtest.runner import run_session  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--n-sessions", type=int, default=20)
    ap.add_argument("--stories-per-session", type=int, default=4)
    ap.add_argument("--max-q", type=int, default=15, help="questions per story (4 x 15 = 60 turns)")
    ap.add_argument("--first", default="L", choices=["S", "L"],
                    help="who answers turn 1 (default L, so the small model receives from the large one first)")
    ap.add_argument("--max-new", type=int, default=24, help="max answer tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--coqa-file", help="local JSON copy of CoQA (see scripts/download_data.py)")
    ap.add_argument("--devices")
    ap.add_argument("--mapper-dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()

    cfg = load_json(os.path.join(a.run, "run_config.json"))
    models, tok = load_pair(cfg["small"], cfg["large"], a.devices or cfg.get("devices"))
    dt = getattr(torch, a.mapper_dtype)
    mappers = {}
    for src, tgt in (("S", "L"), ("L", "S")):
        mp = Mapper.load(os.path.join(a.run, f"mapper_{src}2{tgt}.pt"), models[src].config, models[tgt].config)
        mappers[(src, tgt)] = mp.to(models[tgt].device, dt)

    sessions = make_sessions(load_coqa(a.split, a.coqa_file), a.n_sessions, a.stories_per_session, a.max_q, a.seed)
    save_json(dict(args=vars(a), sessions=sessions), os.path.join(a.run, "sessions.json"))

    out_path = os.path.join(a.run, "results.jsonl")
    done = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            recs = [json.loads(l) for l in f if l.strip()]
        per = {}
        for r in recs:
            per[r["session_id"]] = per.get(r["session_id"], 0) + 1
        done = {s["session_id"] for s in sessions if per.get(s["session_id"], 0) == len(s["turns"])}
        partial = set(per) - done
        if partial:  # drop half-finished chats so they rerun cleanly
            with open(out_path, "w") as f:
                for r in recs:
                    if r["session_id"] not in partial:
                        f.write(json.dumps(r) + "\n")
    other = {"S": "L", "L": "S"}
    for sess in sessions:
        if sess["session_id"] in done:
            continue
        t0 = time.time()
        segs = tokenize_session(sess, tok)
        order = [a.first if t % 2 == 0 else other[a.first] for t in range(len(segs))]
        recs = []
        with open(out_path, "a") as f:
            for rec in run_session(models, mappers, sess, segs, order, tok, max_new=a.max_new):
                f.write(json.dumps(rec) + "\n"); f.flush()
                recs.append(rec)
        late = recs[len(recs) * 3 // 4:]
        print(f"chat {sess['session_id'] + 1}/{len(sessions)}: {len(recs)} turns, "
              f"{sum(len(s['ids']) for s in segs)} tokens, {time.time() - t0:.0f}s | F1 prefill/sw/nv "
              f"{np.mean([r['prefill_f1'] for r in recs]):.2f}/{np.mean([r['sw_f1'] for r in recs]):.2f}/"
              f"{np.mean([r['nv_f1'] for r in recs]):.2f} | late KL sw {np.mean([r['sw_kl'] for r in late]):.3f} "
              f"nv {np.mean([r['nv_kl'] for r in late]):.3f}")
    print(f"\ndone. Next: python scripts/analyze.py --run {a.run}")


if __name__ == "__main__":
    main()
