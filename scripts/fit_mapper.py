"""Step 1: fit the translators (small -> large and large -> small).

    python scripts/fit_mapper.py --pair 4b-8b

Writes runs/<pair>/mapper_S2L.pt, mapper_L2S.pt, fit reports and run_config.json.
"""
import argparse
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from killtest.data import load_calibration  # noqa: E402
from killtest.load import PAIRS, load_pair, save_json  # noqa: E402
from killtest.mapper import FeatureStore, fit_mapper  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="4b-8b", choices=sorted(PAIRS))
    ap.add_argument("--small", help="override the small model (HF id)")
    ap.add_argument("--large", help="override the large model (HF id)")
    ap.add_argument("--out", help="run folder (default runs/<pair>)")
    ap.add_argument("--devices", help="where the models live: 'cuda', 'cuda:0,cuda:1' or 'cuda,cpu' (small, large)")
    ap.add_argument("--fit-device", help="where the fitting math runs (default: a GPU if any is used)")
    ap.add_argument("--keep-cache", action="store_true", help="keep the calibration notes on disk afterwards")
    ap.add_argument("--cache-dir", help="where the temporary calibration notes go (default <out>/calib_cache); "
                                        "point it at a big local disk, e.g. /tmp/$USER-cache")
    ap.add_argument("--n-calib", type=int, default=64, help="calibration sequences (Heo et al. used 500)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--k", type=int, default=8, help="source layers per target layer")
    ap.add_argument("--lam", type=float, default=0.01, help="ridge strength")
    ap.add_argument("--group-size", type=int, default=4,
                    help="target layers fitted at once (lower = less GPU memory; use 1 on a 4 GB GPU)")
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

    fit_device = a.fit_device or next((str(m.device) for m in models.values() if m.device.type == "cuda"), "cpu")
    cache = a.cache_dir or os.path.join(out, "calib_cache")
    per_tok = sum(2 * m.config.num_hidden_layers * m.config.num_key_value_heads * m.config.head_dim * 2
                  for m in models.values())
    cache_gb = per_tok * a.n_calib * a.seq_len / 1e9
    hd = models["S"].config.num_key_value_heads * models["S"].config.head_dim
    mapper_gb = sum(2 * m.config.num_hidden_layers * (min(a.k, o.config.num_hidden_layers) * hd + 1) * hd * 4
                    for m, o in ((models["L"], models["S"]), (models["S"], models["L"]))) / 1e9
    print(f"\ndisk space needed: ~{cache_gb:.1f} GB of temporary notes in {cache} (deleted at the end)\n"
          f"                   ~{mapper_gb:.1f} GB for the two translators in {out}")
    for path, need in {cache: cache_gb, out: mapper_gb}.items():
        os.makedirs(path, exist_ok=True)
        free = shutil.disk_usage(path).free / 1e9
        if free < need * 1.1:
            print(f"[warning] only {free:.1f} GB free at {path}. Use --cache-dir / --out on a bigger disk, "
                  f"or fewer texts (--n-calib).")
    print("(school computers often also limit each person's home folder below what 'free' shows: check `quota -s`)")
    print(f"\nstep A: run each model over the texts once and save its notes")
    store = FeatureStore(models, calib, cache_dir=cache)
    try:
        for src, tgt in (("S", "L"), ("L", "S")):
            t0 = time.time()
            print(f"\nfitting {src} -> {tgt}")
            ms, mt = models[src], models[tgt]
            mapper, report = fit_mapper((ms, ms.config), (mt, mt.config), k=a.k, lam=a.lam,
                                        group_size=a.group_size, device=fit_device, store=store, names=(src, tgt))
            try:
                mapper.save(os.path.join(out, f"mapper_{src}2{tgt}.pt"))
            except (RuntimeError, OSError) as e:
                raise SystemExit(f"\ncould not save the translator to {out}: {e}\n"
                                 "This usually means the disk (or your storage quota) is full. Rerun with --out "
                                 "and/or --cache-dir on a disk with more space.") from e
            save_json(report, os.path.join(out, f"fit_report_{src}2{tgt}.json"))
            print(f"  held-out R^2: keys {report['mean_val_r2_k']:.3f}, values {report['mean_val_r2_v']:.3f} "
                  f"({time.time() - t0:.0f}s)")
    finally:
        if not a.keep_cache:
            store.close()
    print(f"\ndone -> {out}. Next: python scripts/check_handoff.py --run {out}")


if __name__ == "__main__":
    main()
