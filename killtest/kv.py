"""KV-cache plumbing: a plain-tensor KV container, RoPE helpers, and model calls.

We keep each model's "notes" (KV cache) as plain tensors instead of a
transformers Cache object, so slicing, appending and cropping are just tensor
ops and the code works across transformers versions (the Cache API changed a
lot between 4.x and 5.x).

Shapes: every layer holds keys and values of shape [1, n_kv_heads, T, head_dim].
Keys are stored the way the model stores them: AFTER RoPE (position rotation).
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache


# --------------------------------------------------------------------------- #
# KV container
# --------------------------------------------------------------------------- #
@dataclass
class KV:
    k: list = field(default_factory=list)  # per layer [1, H, T, D]
    v: list = field(default_factory=list)

    def __len__(self) -> int:
        return 0 if not self.k else self.k[0].shape[2]

    @property
    def n_layers(self) -> int:
        return len(self.k)

    def slice(self, a: int, b: int) -> "KV":
        return KV([x[:, :, a:b] for x in self.k], [x[:, :, a:b] for x in self.v])

    def cat(self, other: "KV") -> "KV":
        if not self.k:
            return other
        if not other.k:
            return self
        assert self.n_layers == other.n_layers, "can't join notes from different models"
        return KV(
            [torch.cat([a, b], dim=2) for a, b in zip(self.k, other.k)],
            [torch.cat([a, b], dim=2) for a, b in zip(self.v, other.v)],
        )

    def to(self, device) -> "KV":
        return KV([x.to(device) for x in self.k], [x.to(device) for x in self.v])


def to_cache(kv: KV) -> DynamicCache:
    cache = DynamicCache()
    for i, (k, v) in enumerate(zip(kv.k, kv.v)):
        cache.update(k, v, i)
    return cache


def from_cache(cache) -> KV:
    if hasattr(cache, "layers"):  # transformers >= 4.56 / 5.x
        return KV([l.keys for l in cache.layers], [l.values for l in cache.layers])
    return KV(list(cache.key_cache), list(cache.value_cache))  # older versions


def _cache_kwarg(module) -> str:
    params = inspect.signature(module.forward).parameters
    return "cache" if "cache" in params else "past_key_values"


@torch.no_grad()
def forward(model, ids: torch.Tensor, kv: KV, want_logits: bool = True):
    """Run `ids` ([1, n]) on top of the notes `kv`.

    Returns (logits [n, vocab] float32 or None, new KV covering old + new tokens).
    """
    cache = to_cache(kv)
    ids = ids.to(model.device)
    if want_logits:
        out = model(input_ids=ids, use_cache=True, **{_cache_kwarg(model): cache})
        logits = out.logits[0].float()
    else:  # skip the (big) vocabulary projection
        model.model(input_ids=ids, use_cache=True, **{_cache_kwarg(model.model): cache})
        logits = None
    return logits, from_cache(cache)


# --------------------------------------------------------------------------- #
# RoPE (rotary position embedding)
# --------------------------------------------------------------------------- #
def rope_theta(cfg) -> float:
    params = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None) or {}
    rtype = params.get("rope_type", params.get("type", "default")) if isinstance(params, dict) else "default"
    if rtype not in (None, "default"):
        raise NotImplementedError(f"rope_type={rtype!r} not supported (Qwen3 uses 'default')")
    theta = params.get("rope_theta") if isinstance(params, dict) else None
    return float(theta if theta is not None else cfg.rope_theta)


def head_dim(cfg) -> int:
    return getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads


def rope_cos_sin(cfg, positions: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin tables [T, D] (float32), same formula as transformers' default RoPE."""
    d = head_dim(cfg)
    inv = 1.0 / (rope_theta(cfg) ** (torch.arange(0, d, 2, dtype=torch.float64) / d))
    freqs = positions.to(torch.float64)[:, None] * inv[None, :]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().float().to(device), emb.sin().float().to(device)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    kf = k.float()
    return (kf * cos + _rotate_half(kf) * sin).to(k.dtype)


def remove_rope(k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Undo the rotation (rotate by the negative angle)."""
    kf = k.float()
    return (kf * cos - _rotate_half(kf) * sin).to(k.dtype)


def flat_features(kv: KV, cfg, start_pos: int = 0) -> tuple[list, list]:
    """Per layer, position-free features [T, H*D] float32: keys de-RoPE'd, values as-is."""
    T = len(kv)
    if T == 0:
        return [], []
    dev = kv.k[0].device
    cos, sin = rope_cos_sin(cfg, torch.arange(start_pos, start_pos + T), dev)
    ks, vs = [], []
    for k, v in zip(kv.k, kv.v):
        H, D = k.shape[1], k.shape[3]
        ks.append(remove_rope(k.float(), cos, sin)[0].permute(1, 0, 2).reshape(T, H * D))
        vs.append(v.float()[0].permute(1, 0, 2).reshape(T, H * D))
    return ks, vs


def check_matched_kv(cfg_a, cfg_b) -> None:
    """Heo et al.'s mapper needs both models' notes to have the same shape per token."""
    ha, hb = cfg_a.num_key_value_heads, cfg_b.num_key_value_heads
    da, db = head_dim(cfg_a), head_dim(cfg_b)
    if (ha, da) != (hb, db):
        raise ValueError(f"not a matched-KV pair: kv_heads {ha} vs {hb}, head_dim {da} vs {db}")
    if rope_theta(cfg_a) != rope_theta(cfg_b):
        print(f"[warn] different rope_theta ({rope_theta(cfg_a)} vs {rope_theta(cfg_b)}); handled, but unusual")
