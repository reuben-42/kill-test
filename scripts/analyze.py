"""Step 4: plots + drop / pivot / go.

    python scripts/analyze.py --run runs/4b-8b

Uses thresholds.json (agree on it BEFORE looking at results). Writes runs/<pair>/analysis/.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from killtest.analysis import analyze, load_records  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--thresholds", default=os.path.join(os.path.dirname(__file__), "..", "thresholds.json"))
    a = ap.parse_args()

    with open(a.thresholds) as f:
        thr = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    recs = load_records(os.path.join(a.run, "results.jsonl"))
    out = os.path.join(a.run, "analysis")
    res = analyze(recs, thr, outdir=out)

    print(f"{res['n_sessions']} chats, {min(res['turns_per_session'])}-{max(res['turns_per_session'])} turns each\n")
    for c, name in (("sw", "segment-wise"), ("nv", "naive re-translation")):
        if c not in res:
            continue
        r = res[c]
        print(f"{name}:")
        print(f"  F1 drift vs always re-reading: first 10 turns {r['early_f1_drift_pp']:+.1f} pp, "
              f"turn {thr['late_start_turn']}+ {r['late_f1_drift_pp']:+.1f} pp "
              f"(95% CI {r['late_f1_drift_ci'][0]:+.1f} to {r['late_f1_drift_ci'][1]:+.1f})")
        print(f"  KL at the answer: first 10 turns {r['early_kl']:.3f}, late {r['late_kl']:.3f} "
              f"(late, small model answering {r['late_kl_by_answerer']['S']:.3f} / large {r['late_kl_by_answerer']['L']:.3f})")
        print(f"  do chats differ? p = {r['sessions_differ_p']:.3f}")
        print(f"  warning sign vs upcoming drift: rho = {r['signal_spearman']:.2f} "
              f"(oracle version {r['oracle_signal_spearman']:.2f})\n")
    if "nv" in res and "sw" in res and res["nv"]["late_kl"] <= res["sw"]["late_kl"]:
        print("[check] naive re-translation is NOT worse than segment-wise. Prior work says it should be;"
              " double-check the setup before trusting the decision.\n")
    print(f"DECISION: {res['decision']}\n  {res['reason']}\n\nplots + summary.json in {out}")


if __name__ == "__main__":
    main()
