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


@torch.no_grad()
def r2_matrix(src_feats, tgt_feats, lam, n_heads, device, val_frac=0.25, label="scoring layer pairs",
              progress=True):
    """R^2[s, t]: how well source layer s alone predicts target layer t (ridge, held-out)."""
    N = src_feats[0].shape[0]
    n_tr = int(N * (1 - val_frac))
    R = torch.zeros(len(src_feats), len(tgt_feats))
    tgt = [t.to(device) for t in tgt_feats]
    bar = Progress(len(src_feats), label, progress)
    for s, X in enumerate(src_feats):
        X = X.to(device).double()
        Xtr, Xva = X[:n_tr], X[n_tr:]
        mu = Xtr.mean(0, keepdim=True)
        A = (Xtr - mu).T @ (Xtr - mu) / n_tr + lam * torch.eye(X.shape[1], device=device, dtype=X.dtype)
        L = torch.linalg.cholesky(A)
        for t, Y in enumerate(tgt):
            Y = Y.double()
            Ytr, Yva = Y[:n_tr], Y[n_tr:]
            my = Ytr.mean(0, keepdim=True)
            W = torch.cholesky_solve((Xtr - mu).T @ (Ytr - my) / n_tr, L)
            R[s, t] = _r2_per_head(Yva, (Xva - mu) @ W + my, n_heads)
        bar.step()
    return R


def select_layers(R: torch.Tensor, k: int) -> list:
    k = min(k, R.shape[0])
    return [sorted(torch.topk(R[:, t], k).indices.tolist()) for t in range(R.shape[1])]


def _kv_features(model, cfg, ids):
    from .kv import forward
    _, kv = forward(model, ids, KV(), want_logits=False)
    return flat_features(kv, cfg, 0)


@torch.no_grad()
def fit_mapper(src, tgt, calib_ids, k=8, lam=0.01, n_select_seqs=16, n_val_seqs=4,
               group_size=4, device=None, log=print, progress=True):
    """Fit a src->tgt mapper.

    src, tgt: (model, config) tuples.  calib_ids: list of LongTensor [1, T] token sequences.
    The last `n_val_seqs` sequences are held out to report validation R^2.
    """
    (ms, cs), (mt, ct) = src, tgt
    device = device or mt.device
    H = ct.num_key_value_heads
    n_val_seqs = min(n_val_seqs, max(1, len(calib_ids) // 4))
    train, val = calib_ids[:-n_val_seqs], calib_ids[-n_val_seqs:]
    Ls, Lt = cs.num_hidden_layers, ct.num_hidden_layers

    # 1) choose source layers for every target layer, from a small subset ---------
    log(f"  running on {device}")
    log(f"  selecting top-{k} source layers using {min(n_select_seqs, len(train))} sequences")
    bar = Progress(min(n_select_seqs, len(train)), "reading texts", progress)
    sk, sv, tk, tv = [[] for _ in range(Ls)], [[] for _ in range(Ls)], [[] for _ in range(Lt)], [[] for _ in range(Lt)]
    for ids in train[:n_select_seqs]:
        fsk, fsv = _kv_features(ms, cs, ids)
        ftk, ftv = _kv_features(mt, ct, ids)
        for l in range(Ls):
            sk[l].append(fsk[l].half().cpu()); sv[l].append(fsv[l].half().cpu())
        for l in range(Lt):
            tk[l].append(ftk[l].half().cpu()); tv[l].append(ftv[l].half().cpu())
        bar.step()
    cat = lambda xs: [torch.cat(x).float() for x in xs]
    Rk = r2_matrix(cat(sk), cat(tk), lam, H, device, label="scoring key layers", progress=progress)
    Rv = r2_matrix(cat(sv), cat(tv), lam, H, device, label="scoring value layers", progress=progress)
    del sk, sv, tk, tv
    sel_k, sel_v = select_layers(Rk, k), select_layers(Rv, k)

    # 2) closed-form ridge per target layer, streamed over calibration -----------
    Wk, bk, Wv, bv = [None] * Lt, [None] * Lt, [None] * Lt, [None] * Lt
    groups = [list(range(i, min(i + group_size, Lt))) for i in range(0, Lt, group_size)]
    for gi, group in enumerate(groups):
        log(f"  fitting target layers {group[0]}-{group[-1]} ({gi + 1}/{len(groups)})")
        acc = {}
        for t in group:
            for kind, sel in (("k", sel_k[t]), ("v", sel_v[t])):
                din = len(sel) * cs.num_key_value_heads * head_dim(cs)
                dout = H * head_dim(ct)
                acc[(t, kind)] = dict(xx=torch.zeros(din, din, device=device, dtype=torch.float64),
                                      xy=torch.zeros(din, dout, device=device, dtype=torch.float64),
                                      sx=torch.zeros(din, device=device, dtype=torch.float64),
                                      sy=torch.zeros(dout, device=device, dtype=torch.float64), n=0)
        bar = Progress(len(train), "texts", progress)
        for ids in train:
            fsk, fsv = _kv_features(ms, cs, ids)
            ftk, ftv = _kv_features(mt, ct, ids)
            for t in group:
                for kind, sel, fs, ft in (("k", sel_k[t], fsk, ftk), ("v", sel_v[t], fsv, ftv)):
                    X = torch.cat([fs[s].to(device) for s in sel], dim=-1)
                    Y = ft[t].to(device)
                    a = acc[(t, kind)]
                    a["xx"] += (X.T @ X).double(); a["xy"] += (X.T @ Y).double()
                    a["sx"] += X.sum(0).double(); a["sy"] += Y.sum(0).double(); a["n"] += X.shape[0]
            bar.step()
        for t in group:
            for kind in ("k", "v"):
                a = acc.pop((t, kind))
                n = a["n"]; mx = a["sx"] / n; my = a["sy"] / n
                cov = a["xx"] / n - torch.outer(mx, mx)
                cxy = a["xy"] / n - torch.outer(mx, my)
                cov += lam * torch.eye(cov.shape[0], device=device, dtype=cov.dtype)
                W = torch.linalg.solve(cov, cxy)
                b = my - mx @ W
                if kind == "k":
                    Wk[t], bk[t] = W.float().cpu(), b.float().cpu()
                else:
                    Wv[t], bv[t] = W.float().cpu(), b.float().cpu()
        del acc

    mapper = Mapper(sel_k, sel_v, Wk, bk, Wv, bv, cs, ct,
                    meta=dict(k=k, lam=lam, n_train_seqs=len(train), n_val_seqs=len(val)))

    # 3) held-out check -----------------------------------------------------------
    m = mapper.to(device)
    r2k, r2v = [[] for _ in range(Lt)], [[] for _ in range(Lt)]
    log("  checking on held-out texts")
    bar = Progress(len(val), "texts", progress)
    for ids in val:
        fsk, fsv = _kv_features(ms, cs, ids)
        ftk, ftv = _kv_features(mt, ct, ids)
        for t in range(Lt):
            xk = torch.cat([fsk[s].to(device) for s in sel_k[t]], -1)
            xv = torch.cat([fsv[s].to(device) for s in sel_v[t]], -1)
            r2k[t].append(_r2_per_head(ftk[t].to(device), xk @ m.Wk[t] + m.bk[t], H))
            r2v[t].append(_r2_per_head(ftv[t].to(device), xv @ m.Wv[t] + m.bv[t], H))
        bar.step()
    avg = lambda xs: [sum(x) / len(x) for x in xs]
    report = dict(val_r2_k=avg(r2k), val_r2_v=avg(r2v), sel_k=sel_k, sel_v=sel_v,
                  single_layer_r2_k=Rk.tolist(), single_layer_r2_v=Rv.tolist())
    report["mean_val_r2_k"] = sum(report["val_r2_k"]) / Lt
    report["mean_val_r2_v"] = sum(report["val_r2_v"]) / Lt
    mapper.meta["report"] = {k_: report[k_] for k_ in ("mean_val_r2_k", "mean_val_r2_v")}
    return mapper, report
