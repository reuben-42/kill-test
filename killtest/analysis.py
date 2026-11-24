"""Turn the per-turn results into 3 plots and a drop / pivot / go decision."""
from __future__ import annotations

import json
import os
from collections import defaultdict

import numpy as np

from .metrics import bootstrap_mean_ci, sessions_differ_pvalue, spearman

DEFAULT_THRESHOLDS = {
    "late_start_turn": 50,        # "late" = this turn onwards (1-indexed)
    "drop_below_pp": 3.0,         # segment-wise late drift below this (and CI touching 0) => DROP
    "sessions_differ_p": 0.05,    # permutation p-value below this => chats drift differently
    "signal_window": 5,           # baseline = mean surprise over this many earlier turns (same model)
    "signal_horizon": 3,          # signal at turn t is compared with drift over turns t+1..t+h
    "min_signal_spearman": 0.3,   # rank correlation needed to say the warning sign works
}


def load_records(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def by_session(records):
    s = defaultdict(list)
    for r in records:
        s[r["session_id"]].append(r)
    return [sorted(v, key=lambda r: r["turn"]) for _, v in sorted(s.items())]


def _signals(turns, cond, window):
    """Deployable signal: surprise on the new question minus this model's recent average."""
    out, hist = [], defaultdict(list)
    for r in turns:
        s = r.get(f"{cond}_surprise_q", np.nan)
        prev = hist[r["answerer"]][-window:]
        out.append(s - np.mean(prev) if prev else np.nan)
        if not np.isnan(s):
            hist[r["answerer"]].append(s)
    return np.array(out)


def analyze(records, thr=None, outdir=None, make_plots=True):
    thr = {**DEFAULT_THRESHOLDS, **(thr or {})}
    sessions = by_session(records)
    late0 = thr["late_start_turn"] - 1  # 0-indexed
    res = {"thresholds": thr, "n_sessions": len(sessions),
           "turns_per_session": [len(s) for s in sessions]}

    for cond in ("sw", "nv"):
        if f"{cond}_f1" not in records[0]:
            continue
        late_f1, late_kl, kl_late_lists = [], [], []
        for s in sessions:
            late = [r for r in s if r["turn"] >= late0]
            if not late:
                continue
            late_f1.append(np.mean([(r["prefill_f1"] - r[f"{cond}_f1"]) * 100 for r in late]))
            late_kl.append(np.mean([r[f"{cond}_kl"] for r in late]))
            kl_late_lists.append([r[f"{cond}_kl"] for r in late])
        if not late_f1:
            raise ValueError(f"no session reaches turn {thr['late_start_turn']}; lower late_start_turn")
        m, lo, hi = bootstrap_mean_ci(late_f1)
        early = [np.mean([(r["prefill_f1"] - r[f"{cond}_f1"]) * 100 for r in s[:10]]) for s in sessions]
        res[cond] = dict(late_f1_drift_pp=m, late_f1_drift_ci=[lo, hi],
                         early_f1_drift_pp=float(np.mean(early)),
                         late_kl=float(np.mean(late_kl)),
                         early_kl=float(np.mean([np.mean([r[f"{cond}_kl"] for r in s[:10]]) for s in sessions])),
                         sessions_differ_p=sessions_differ_pvalue(kl_late_lists))
        # warning-sign test: signal now vs drift (KL) over the next few turns
        sig, sig_or, fut = [], [], []
        h = thr["signal_horizon"]
        for s in sessions:
            dep = _signals(s, cond, thr["signal_window"])
            kl = np.array([r[f"{cond}_kl"] for r in s])
            for i, r in enumerate(s[:-h]):
                sig.append(dep[i])
                sig_or.append(r[f"{cond}_surprise_q"] - r["prefill_surprise_q"])
                fut.append(kl[i + 1:i + 1 + h].mean())
        res[cond]["signal_spearman"] = spearman(sig, fut)
        res[cond]["oracle_signal_spearman"] = spearman(sig_or, fut)
        # where does drift show up: turns answered by the small vs the large model
        res[cond]["late_kl_by_answerer"] = {
            a: float(np.mean([r[f"{cond}_kl"] for s in sessions for r in s if r["turn"] >= late0 and r["answerer"] == a] or [np.nan]))
            for a in ("S", "L")}

    res["decision"], res["reason"] = decide(res, thr)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "summary.json"), "w") as f:
            json.dump(res, f, indent=2)
        if make_plots:
            plot(sessions, res, thr, outdir)
    return res


def decide(res, thr):
    sw = res.get("sw")
    if sw is None:
        return "N/A", "segment-wise condition was not run"
    lo = sw["late_f1_drift_ci"][0]
    if sw["late_f1_drift_pp"] < thr["drop_below_pp"] and lo <= 0:
        return "DROP", (f"segment-wise drift late in the chat is {sw['late_f1_drift_pp']:.1f} pp "
                        f"(95% CI {lo:.1f} to {sw['late_f1_drift_ci'][1]:.1f}): nothing builds up to detect")
    if sw["sessions_differ_p"] >= thr["sessions_differ_p"]:
        return "PIVOT", (f"drift builds up but chats don't differ (p = {sw['sessions_differ_p']:.3f}): "
                         "a fixed refresh schedule is enough; publish the measurement")
    if not (sw["signal_spearman"] >= thr["min_signal_spearman"]):
        extra = (" The oracle version of the signal does work "
                 f"(rho = {sw['oracle_signal_spearman']:.2f}), so try a better baseline."
                 if sw["oracle_signal_spearman"] >= thr["min_signal_spearman"] else "")
        return "PIVOT (signal)", (f"drift builds up and differs by chat, but the warning sign doesn't track it "
                                  f"(rho = {sw['signal_spearman']:.2f}).{extra}")
    return "GO", (f"drift builds up ({sw['late_f1_drift_pp']:.1f} pp late), differs by chat "
                  f"(p = {sw['sessions_differ_p']:.3f}), and the warning sign tracks it "
                  f"(rho = {sw['signal_spearman']:.2f})")


def plot(sessions, res, thr, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sw": "tab:blue", "nv": "tab:red"}
    names = {"sw": "segment-wise", "nv": "naive re-translation"}
    conds = [c for c in ("sw", "nv") if c in res]
    max_t = max(len(s) for s in sessions)

    def mean_by_turn(cond, fn, who=None):
        out = []
        for t in range(max_t):
            v = [fn(s[t]) for s in sessions if t < len(s) and (who is None or s[t]["answerer"] == who)]
            out.append(np.mean(v) if v else np.nan)
        return np.array(out)

    # 1) drift vs turn: faint line per chat + bold mean
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, (label, fn) in zip(axes, [("F1 drift vs. always re-reading (pp)",
                                       lambda c: lambda r: (r["prefill_f1"] - r[f"{c}_f1"]) * 100),
                                      ("KL at the answer (nats)", lambda c: lambda r: r[f"{c}_kl"])]):
        for c in conds:
            f = fn(c)
            for s in sessions:
                ax.plot([r["turn"] + 1 for r in s], [f(r) for r in s], color=colors[c], alpha=0.12, lw=0.8)
            m = mean_by_turn(c, f)
            sm = np.array([np.nanmean(m[max(0, i - 2):i + 3]) for i in range(len(m))])  # 5-turn average
            ax.plot(np.arange(1, max_t + 1), sm, color=colors[c], lw=2.2, label=f"{names[c]} (5-turn avg)")
        ax.axvline(thr["late_start_turn"], color="gray", ls="--", lw=1)
        ax.set_xlabel("turn"); ax.set_title(label); ax.axhline(0, color="black", lw=0.6)
    axes[0].legend(frameon=False)
    fig.suptitle("Does translation error build up over a long chat?")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig1_drift_vs_turn.png"), dpi=150); plt.close(fig)

    # 2) KL by who answered
    fig, ax = plt.subplots(figsize=(7, 4))
    for who, ls in (("S", "-"), ("L", "--")):
        for c in conds:
            m = mean_by_turn(c, lambda r, c=c: r[f"{c}_kl"], who)
            ok = ~np.isnan(m)
            ax.plot(np.arange(1, max_t + 1)[ok], m[ok], ls=ls, color=colors[c],
                    label=f"{names[c]}, {'small' if who == 'S' else 'large'} model answering")
    ax.set_xlabel("turn"); ax.set_ylabel("KL at the answer (nats)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("Where does drift show up?")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig2_by_answerer.png"), dpi=150); plt.close(fig)

    # 3) warning sign vs upcoming drift (segment-wise)
    if "sw" in res:
        sig, fut = [], []
        h = thr["signal_horizon"]
        for s in sessions:
            dep = _signals(s, "sw", thr["signal_window"])
            kl = np.array([r["sw_kl"] for r in s])
            for i in range(len(s) - h):
                sig.append(dep[i]); fut.append(kl[i + 1:i + 1 + h].mean())
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(sig, fut, s=8, alpha=0.4, color=colors["sw"])
        ax.set_xlabel("surprise on the new question minus recent average (nats)")
        ax.set_ylabel(f"KL over the next {h} turns (nats)")
        ax.set_title(f"Does surprise warn us early? Spearman rho = {res['sw']['signal_spearman']:.2f}")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "fig3_signal.png"), dpi=150); plt.close(fig)
