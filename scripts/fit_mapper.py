"""Step 1: fit the translators (small -> large and large -> small).

    python scripts/fit_mapper.py --pair 4b-8b

Writes runs/<pair>/mapper_S2L.pt, mapper_L2S.pt, fit reports and run_config.json.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from killtest.data import load_calibration  # noqa: E402
from killtest.load import PAIRS, load_pair, save_json  # noqa: E402
from killtest.mapper import fit_mapper  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="4b-8b", choices=sorted(PAIRS))
    ap.add_argument("--small", help="override the small model (HF id)")
    ap.add_argument("--large", help="override the large model (HF id)")
    ap.add_argument("--out", help="run folder (default runs/<pair>)")
    ap.add_argument("--devices", help="e.g. 'cuda:0' or 'cuda:0,cuda:1' (small, large)")
    ap.add_argument("--n-calib", type=int, default=64, help="calibration sequences (Heo et al. used 500)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--k", type=int, default=8, help="source layers per target layer")
    ap.add_argument("--lam", type=float, default=0.01, help="ridge strength")
    ap.add_argument("--group-size", type=int, default=4, help="target layers fitted per pass (lower = less GPU memory)")
    ap.add_argument("--calib-text-file", help="use a local text file instead of streaming FineWeb-Edu")
    a = ap.parse_args()

    small, large = a.small or PAIRS[a.pair][0], a.large or PAIRS[a.pair][1]
    out = a.out or os.path.join("runs", a.pair)
    os.makedirs(out, exist_ok=True)
    models, tok = load_pair(small, large, a.devices)
    save_json(dict(small=small, large=large, pair=a.pair, devices=a.devices, fit_args=vars(a)),
              os.path.join(out, "run_config.json"))

    print(f"loading {a.n_calib} calibration sequences of {a.seq_len} tokens")
    calib = load_calibration(tok, a.n_calib, a.seq_len, text_file=a.calib_text_file)

    for src, tgt in (("S", "L"), ("L", "S")):
        t0 = time.time()
        print(f"\nfitting {src} -> {tgt}")
        ms, mt = models[src], models[tgt]
        mapper, report = fit_mapper((ms, ms.config), (mt, mt.config), calib, k=a.k, lam=a.lam,
                                    group_size=a.group_size, device=mt.device)
        mapper.save(os.path.join(out, f"mapper_{src}2{tgt}.pt"))
        save_json(report, os.path.join(out, f"fit_report_{src}2{tgt}.json"))
        print(f"  held-out R^2: keys {report['mean_val_r2_k']:.3f}, values {report['mean_val_r2_v']:.3f} "
              f"({time.time() - t0:.0f}s)")
    print(f"\ndone -> {out}. Next: python scripts/check_handoff.py --run {out}")


if __name__ == "__main__":
    main()
