"""Scoring and small statistics helpers (no scipy needed)."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


# CoQA's official answer normalisation + token F1 ----------------------------
def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


# Statistics -------------------------------------------------------------------
def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share their mean rank)."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx, ry = rankdata(x), rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_mean_ci(per_session_values, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Mean over sessions with a 95% CI from resampling whole sessions."""
    v = np.asarray(per_session_values, float)
    v = v[~np.isnan(v)]
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(1)
    return float(v.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def sessions_differ_pvalue(values_by_session: list, n_perm: int = 2000, seed: int = 0) -> float:
    """Do chats really drift by different amounts, or is it just turn-to-turn noise?

    Permutation test: shuffle per-turn values across sessions and see how often the
    spread of session means is as large as the real one. Small p => sessions differ.
    """
    groups = [np.asarray(v, float)[~np.isnan(np.asarray(v, float))] for v in values_by_session]
    groups = [g for g in groups if len(g)]
    sizes = [len(g) for g in groups]
    allv = np.concatenate(groups)
    stat = lambda gs: np.var([g.mean() for g in gs])
    real = stat(groups)
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        p = rng.permutation(allv)
        gs, i = [], 0
        for s in sizes:
            gs.append(p[i:i + s]); i += s
        hits += stat(gs) >= real
    return (hits + 1) / (n_perm + 1)
