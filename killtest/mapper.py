"""The translator: a per-head linear (ridge) map from one model's KV cache to another's.

Recipe follows Heo et al. (arXiv 2608.03893) as described in the paper:
  * keys have RoPE stripped before mapping, and the target's RoPE is re-applied after,
    so the map does not depend on position;
  * for each target layer, pick the top-k most predictive source layers (single-layer
    ridge R^2, averaged over heads) and concatenate all their KV heads as the input;
  * fit a closed-form ridge regression (lambda = 0.01) on calibration text.

Details the paper does not pin down (our choices, easy to change):
  * keys and values select their source layers separately;
  * ridge is solved on mean-centred, per-token-normalised statistics
    (W = (Cov_xx + lam*I)^-1 Cov_xy, plus a bias), so lambda doesn't scale with data size;
  * all target heads of a layer share one input, so the per-head fits are solved together.
"""
from __future__ import annotations

import os
import time

import torch

from .kv import KV, apply_rope, flat_features, rope_cos_sin, head_dim


class Mapper:
    def __init__(self, sel_k, sel_v, Wk, bk, Wv, bv, src_cfg, tgt_cfg, meta=None):
        self.sel_k, self.sel_v = sel_k, sel_v  # per target layer: list of source layer ids
        self.Wk, self.bk, self.Wv, self.bv = Wk, bk, Wv, bv  # per target layer tensors
        self.src_cfg, self.tgt_cfg = src_cfg, tgt_cfg
        self.meta = meta or {}

    # ------------------------------------------------------------------ #
    def to(self, device, dtype=torch.float32) -> "Mapper":
        mv = lambda xs: [x.to(device=device, dtype=dtype) for x in xs]
        return Mapper(self.sel_k, self.sel_v, mv(self.Wk), mv(self.bk), mv(self.Wv), mv(self.bv),
                      self.src_cfg, self.tgt_cfg, self.meta)

    @torch.no_grad()
    def translate(self, kv: KV, start_pos: int, out_dtype=None) -> KV:
        """Translate source notes for positions [start_pos, start_pos+len(kv)) into target notes."""
        T = len(kv)
        if T == 0:
            return KV()
        dev = self.Wk[0].device
        wdt = self.Wk[0].dtype
        out_dtype = out_dtype or kv.k[0].dtype
        fk, fv = flat_features(kv.to(dev), self.src_cfg, start_pos)
        H, D = self.tgt_cfg.num_key_value_heads, head_dim(self.tgt_cfg)
        cos, sin = rope_cos_sin(self.tgt_cfg, torch.arange(start_pos, start_pos + T), dev)
        ks, vs = [], []
        for l in range(len(self.Wk)):
            xk = torch.cat([fk[s] for s in self.sel_k[l]], dim=-1).to(wdt)
            yk = (xk @ self.Wk[l] + self.bk[l]).float().view(T, H, D).permute(1, 0, 2)[None]
            ks.append(apply_rope(yk, cos, sin).to(out_dtype))
            xv = torch.cat([fv[s] for s in self.sel_v[l]], dim=-1).to(wdt)
            yv = (xv @ self.Wv[l] + self.bv[l]).float().view(T, H, D).permute(1, 0, 2)[None]
            vs.append(yv.to(out_dtype))
        return KV(ks, vs)

    # ------------------------------------------------------------------ #
    def save(self, path) -> None:
        torch.save({"sel_k": self.sel_k, "sel_v": self.sel_v,
                    "Wk": [w.cpu() for w in self.Wk], "bk": [b.cpu() for b in self.bk],
                    "Wv": [w.cpu() for w in self.Wv], "bv": [b.cpu() for b in self.bv],
                    "src_cfg": self.src_cfg.to_dict(), "tgt_cfg": self.tgt_cfg.to_dict(),
                    "meta": self.meta}, path)

    @staticmethod
    def load(path, src_cfg, tgt_cfg) -> "Mapper":
        d = torch.load(path, map_location="cpu", weights_only=False)
        return Mapper(d["sel_k"], d["sel_v"], d["Wk"], d["bk"], d["Wv"], d["bv"], src_cfg, tgt_cfg, d["meta"])

    @staticmethod
    def identity(cfg, n_layers: int) -> "Mapper":
        """Maps a model's notes onto themselves. Used only in tests."""
        HD = cfg.num_key_value_heads * head_dim(cfg)
        eye = [torch.eye(HD) for _ in range(n_layers)]
        zero = [torch.zeros(HD) for _ in range(n_layers)]
        sel = [[l] for l in range(n_layers)]
        return Mapper(sel, sel, eye, zero, [e.clone() for e in eye], [z.clone() for z in zero], cfg, cfg)


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
class Progress:
    """One line that updates in place: 'label 12/60 | 1m 30s elapsed, ~6m left'."""

    def __init__(self, total, label, enabled=True):
        self.total, self.label, self.enabled = total, label, enabled
        self.i, self.t0 = 0, time.time()
        self._show()

    @staticmethod
    def _fmt(sec):
        sec = int(sec)
        return f"{sec // 3600}h {sec % 3600 // 60}m" if sec >= 3600 else f"{sec // 60}m {sec % 60:02d}s"

    def _show(self):
        if not self.enabled:
            return
        el = time.time() - self.t0
        eta = f", ~{self._fmt(el / self.i * (self.total - self.i))} left" if 0 < self.i < self.total else ""
        end = "\n" if self.i == self.total else ""
        print(f"\r    {self.label} {self.i}/{self.total} | {self._fmt(el)} elapsed{eta}   ", end=end, flush=True)

    def step(self):
        self.i += 1
        self._show()


def _r2_per_head(y, pred, n_heads):
    """Head-averaged R^2 for y, pred of shape [N, H*D]."""
    N = y.shape[0]
    sse = ((y - pred) ** 2).view(N, n_heads, -1).sum(dim=(0, 2))
    sst = ((y - y.mean(0, keepdim=True)) ** 2).view(N, n_heads, -1).sum(dim=(0, 2)).clamp_min(1e-12)
    return (1 - sse / sst).mean().item()


class FeatureStore:
    """Every model's position-free KV features on the calibration texts, computed ONCE.

    Layout per model and kind ('k'/'v'): [n_layers, n_tokens, H*D] in float16, kept in RAM,
    or in files on disk when `cache_dir` is given (needed for real models: several GB).
    Fitting then reads from here instead of re-running the models for every layer group.
    """

    def __init__(self, models: dict, calib_ids: list, cache_dir=None, progress=True, log=print):
        import numpy as np

        self.cache_dir, self.arrays = cache_dir, {}
        lens = [ids.shape[1] for ids in calib_ids]
        self.offsets = [sum(lens[:i]) for i in range(len(lens) + 1)]
        N = self.offsets[-1]
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        for name, model in models.items():
            cfg = model.config
            L, HD = cfg.num_hidden_layers, cfg.num_key_value_heads * head_dim(cfg)
            for kind in ("k", "v"):
                if cache_dir:
                    path = os.path.join(cache_dir, f"{name}_{kind}.f16")
                    self.arrays[(name, kind)] = np.memmap(path, dtype=np.float16, mode="w+", shape=(L, N, HD))
                else:
                    self.arrays[(name, kind)] = torch.empty((L, N, HD), dtype=torch.float16)
            log(f"  reading calibration texts with {name} ({model.device})")
            bar = Progress(len(calib_ids), "texts", progress)
            for i, ids in enumerate(calib_ids):
                fk, fv = _kv_features(model, cfg, ids)
                a, b = self.offsets[i], self.offsets[i + 1]
                for kind, feats in (("k", fk), ("v", fv)):
                    for l, f in enumerate(feats):
                        h = f.half()
                        if not torch.isfinite(h).all():
                            raise OverflowError(f"{name} layer {l} {kind}: values too large for float16")
                        if cache_dir:
                            self.arrays[(name, kind)][l, a:b] = h.cpu().numpy()
                        else:
                            self.arrays[(name, kind)][l, a:b] = h.cpu()
                bar.step()
        self.n_texts = len(calib_ids)

    def get(self, name, kind, layer, a, b, device) -> torch.Tensor:
        x = self.arrays[(name, kind)][layer, a:b]
        x = torch.from_numpy(x.copy()) if not isinstance(x, torch.Tensor) else x
        return x.to(device).float()

    def span(self, first_text, last_text):
        """Token range covering texts [first_text, last_text)."""
        return self.offsets[first_text], self.offsets[last_text]

    def close(self):
        import gc
        import shutil

        self.arrays.clear()
        gc.collect()  # Windows can't delete a file that is still memory-mapped
        if self.cache_dir:
            shutil.rmtree(self.cache_dir, ignore_errors=True)


@torch.no_grad()
def r2_matrix(get_src, n_src, get_tgt, n_tgt, lam, n_heads, device, val_frac=0.25,
              label="scoring layer pairs", progress=True):
    """R^2[s, t]: how well source layer s alone predicts target layer t (ridge, held-out).

    Products in float32 (fast on any GPU); the small 1024x1024 solves in float64.
    """
    tgt = [get_tgt(t).half() for t in range(n_tgt)]  # kept on `device` as float16
    N = tgt[0].shape[0]
    n_tr = int(N * (1 - val_frac))
    R = torch.zeros(n_src, n_tgt)
    bar = Progress(n_src, label, progress)
    for s in range(n_src):
        X = get_src(s)
        mu = X[:n_tr].mean(0, keepdim=True)
        Xtr, Xva = X[:n_tr] - mu, X[n_tr:] - mu
        A = (Xtr.T @ Xtr).double() / n_tr
        A += lam * torch.eye(A.shape[0], device=device, dtype=A.dtype)
        Lc = torch.linalg.cholesky(A)
        for t in range(n_tgt):
            Y = tgt[t].float()
            my = Y[:n_tr].mean(0, keepdim=True)
            W = torch.cholesky_solve((Xtr.T @ (Y[:n_tr] - my)).double() / n_tr, Lc).float()
            R[s, t] = _r2_per_head(Y[n_tr:], Xva @ W + my, n_heads)
        bar.step()
    del tgt
    return R


def select_layers(R: torch.Tensor, k: int) -> list:
    k = min(k, R.shape[0])
    return [sorted(torch.topk(R[:, t], k).indices.tolist()) for t in range(R.shape[1])]


def _kv_features(model, cfg, ids):
    from .kv import forward
    _, kv = forward(model, ids, KV(), want_logits=False)
    return flat_features(kv, cfg, 0)


@torch.no_grad()
def fit_mapper(src, tgt, calib_ids=None, k=8, lam=0.01, n_select_seqs=16, n_val_seqs=4,
               group_size=4, device=None, log=print, progress=True, store=None, names=("src", "tgt")):
    """Fit a src->tgt mapper.

    src, tgt: (model, config) tuples. Either pass `calib_ids` (list of LongTensor [1, T]) or a
    ready FeatureStore `store` plus the two models' `names` in it. The last `n_val_seqs` texts
    are held out to report validation R^2. `device` = where the math runs (use the GPU).
    """
    (ms, cs), (mt, ct) = src, tgt
    device = torch.device(device or mt.device)
    own_store = store is None
    if own_store:
        names = ("src", "tgt") if ms is not mt else ("m", "m")
        store = FeatureStore({names[0]: ms, names[1]: mt}, calib_ids, progress=progress, log=log)
    sn, tn = names
    H = ct.num_key_value_heads
    n_texts = store.n_texts
    n_val_seqs = min(n_val_seqs, max(1, n_texts // 4))
    n_train = n_texts - n_val_seqs
    Ls, Lt = cs.num_hidden_layers, ct.num_hidden_layers
    log(f"  math runs on {device}")

    # 1) choose source layers for every target layer, from a small subset ---------
    n_sel = min(n_select_seqs, n_train)
    a, b = store.span(0, n_sel)
    log(f"  choosing the top-{k} source layers for each target layer ({n_sel} texts)")
    sel = {}
    R = {}
    for kind in ("k", "v"):
        R[kind] = r2_matrix(lambda s: store.get(sn, kind, s, a, b, device), Ls,
                            lambda t: store.get(tn, kind, t, a, b, device), Lt, lam, H, device,
                            label=f"scoring {'key' if kind == 'k' else 'value'} layers", progress=progress)
        sel[kind] = select_layers(R[kind], k)
    sel_k, sel_v = sel["k"], sel["v"]

    # 2) closed-form ridge per target layer, streamed over the training texts -----
    Wk, bk, Wv, bv = [None] * Lt, [None] * Lt, [None] * Lt, [None] * Lt
    groups = [list(range(i, min(i + group_size, Lt))) for i in range(0, Lt, group_size)]
    for gi, group in enumerate(groups):
        log(f"  fitting target layers {group[0]}-{group[-1]} ({gi + 1}/{len(groups)})")
        acc = {}
        for t in group:
            for kind in ("k", "v"):
                din = len(sel[kind][t]) * cs.num_key_value_heads * head_dim(cs)
                dout = H * head_dim(ct)
                acc[(t, kind)] = dict(xx=torch.zeros(din, din, device=device, dtype=torch.float64),
                                      xy=torch.zeros(din, dout, device=device, dtype=torch.float64),
                                      sx=torch.zeros(din, device=device, dtype=torch.float64),
                                      sy=torch.zeros(dout, device=device, dtype=torch.float64), n=0)
        bar = Progress(n_train, "texts", progress)
        for i in range(n_train):
            a, b = store.span(i, i + 1)
            for t in group:
                for kind in ("k", "v"):
                    X = torch.cat([store.get(sn, kind, s, a, b, device) for s in sel[kind][t]], dim=-1)
                    Y = store.get(tn, kind, t, a, b, device)
                    ac = acc[(t, kind)]
                    ac["xx"].add_(X.T @ X); ac["xy"].add_(X.T @ Y)  # float32 product, float64 running sum
                    ac["sx"].add_(X.sum(0)); ac["sy"].add_(Y.sum(0)); ac["n"] += X.shape[0]
                    del X, Y
            bar.step()
        for t in group:
            for kind in ("k", "v"):
                ac = acc.pop((t, kind))
                n = ac["n"]; mx = ac["sx"] / n; my = ac["sy"] / n
                cov = ac.pop("xx").div_(n).sub_(torch.outer(mx, mx))
                cov.diagonal().add_(lam)
                cxy = ac.pop("xy").div_(n).sub_(torch.outer(mx, my))
                W = torch.linalg.solve(cov, cxy)
                del cov, cxy
                bias = my - mx @ W
                if kind == "k":
                    Wk[t], bk[t] = W.float().cpu(), bias.float().cpu()
                else:
                    Wv[t], bv[t] = W.float().cpu(), bias.float().cpu()
                del W
        del acc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mapper = Mapper(sel_k, sel_v, Wk, bk, Wv, bv, cs, ct,
                    meta=dict(k=k, lam=lam, n_train_seqs=n_train, n_val_seqs=n_val_seqs))

    # 3) held-out check, one target layer at a time (keeps GPU memory small) -------
    log("  checking on held-out texts")
    a, b = store.span(n_train, n_texts)
    r2 = {"k": [], "v": []}
    W_all = {"k": (Wk, bk), "v": (Wv, bv)}
    bar = Progress(Lt, "target layers", progress)
    for t in range(Lt):
        for kind in ("k", "v"):
            X = torch.cat([store.get(sn, kind, s, a, b, device) for s in sel[kind][t]], -1)
            W, bias = W_all[kind][0][t].to(device), W_all[kind][1][t].to(device)
            r2[kind].append(_r2_per_head(store.get(tn, kind, t, a, b, device), X @ W + bias, H))
        bar.step()
    if own_store:
        store.close()
    report = dict(val_r2_k=r2["k"], val_r2_v=r2["v"], sel_k=sel_k, sel_v=sel_v,
                  single_layer_r2_k=R["k"].tolist(), single_layer_r2_v=R["v"].tolist())
    report["mean_val_r2_k"] = sum(r2["k"]) / Lt
    report["mean_val_r2_v"] = sum(r2["v"]) / Lt
    mapper.meta["report"] = {k_: report[k_] for k_ in ("mean_val_r2_k", "mean_val_r2_v")}
    return mapper, report
