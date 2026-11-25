"""CPU tests with tiny, randomly initialised Qwen3 models (no downloads needed).

They can't tell us anything about real results; they check the plumbing:
positions/RoPE, slicing, translating pages, the three conditions, fitting, analysis.
Run with:  pytest -q
"""
import random

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from killtest.analysis import analyze
from killtest.data import make_sessions, tokenize_session
from killtest.kv import KV, flat_features, forward
from killtest.mapper import FeatureStore, Mapper, fit_mapper
from killtest.runner import run_session, single_handoff


class FakeTok:
    """Character-level 'tokenizer' over a 128-symbol vocabulary."""
    eos_token_id = None

    def encode(self, text):
        return [min(ord(c), 127) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def tiny_model(layers=2, hidden=64, seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=128, hidden_size=hidden, intermediate_size=2 * hidden,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=4096, tie_word_embeddings=False)
    return Qwen3ForCausalLM(cfg).eval()


def fake_stories(n, n_q=4, seed=0):
    rng = random.Random(seed)
    words = "alpha beta gamma delta river stone light green north quiet".split()
    para = lambda k: " ".join(rng.choice(words) for _ in range(k)) + "."
    return [dict(story=para(30), questions=[para(4) for _ in range(n_q)], answers=[para(2) for _ in range(n_q)])
            for _ in range(n)]


def test_rope_matches_model():
    """At layer 0 a key before RoPE depends only on the token, so the same token at
    different positions must give identical de-RoPE'd keys if our RoPE matches the model's."""
    m = tiny_model()
    _, kv = forward(m, torch.full((1, 12), 7), KV(), want_logits=False)
    fk, _ = flat_features(kv, m.config, 0)
    assert torch.allclose(fk[0], fk[0][:1].expand_as(fk[0]), atol=1e-5)


def test_translate_identity_roundtrip():
    m = tiny_model()
    _, kv = forward(m, torch.randint(0, 128, (1, 20)), KV(), want_logits=False)
    mp = Mapper.identity(m.config, m.config.num_hidden_layers)
    out = mp.translate(kv.slice(5, 15), start_pos=5)
    ref = kv.slice(5, 15)
    for a, b in zip(out.k + out.v, ref.k + ref.v):
        assert torch.allclose(a, b, atol=1e-5)


def test_conditions_match_prefill_with_perfect_translator():
    """Same model on both sides + identity translator => every condition must equal prefill."""
    m = tiny_model()
    agents = {"S": m, "L": m}
    ident = Mapper.identity(m.config, m.config.num_hidden_layers)
    mappers = {("S", "L"): ident, ("L", "S"): ident}
    tok = FakeTok()
    sess = make_sessions(fake_stories(4), n_sessions=1, stories_per_session=2, max_q_per_story=4)[0]
    segs = tokenize_session(sess, tok)
    order = ["S" if t % 2 == 0 else "L" for t in range(len(segs))]
    recs = list(run_session(agents, mappers, sess, segs, order, tok, max_new=6))
    assert len(recs) == 8
    for r in recs:
        assert r["sw_answer"] == r["prefill_answer"] and r["nv_answer"] == r["prefill_answer"]
        assert r["sw_kl"] < 1e-5 and r["nv_kl"] < 1e-5
        assert abs(r["sw_surprise_q"] - r["prefill_surprise_q"]) < 1e-4


def test_single_handoff_identity():
    m = tiny_model()
    mp = Mapper.identity(m.config, m.config.num_hidden_layers)
    tok = FakeTok()
    r = single_handoff(m, m, mp, tok.encode("Story: alpha beta gamma.\n\n"), tok.encode("Q: what?\nA:"), tok, "x", max_new=5)
    assert r["translated_answer"] == r["native_answer"] and r["top1_agree"] and r["kl"] < 1e-5


def test_fit_recovers_identity():
    """Fitting a model onto itself should pick the same layer and reach R^2 ~ 1."""
    m = tiny_model(layers=3)
    torch.manual_seed(1)
    calib = [torch.randint(0, 128, (1, 64)) for _ in range(12)]
    mp, rep = fit_mapper((m, m.config), (m, m.config), calib, k=1, lam=1e-4, n_select_seqs=6,
                         n_val_seqs=3, device="cpu", log=lambda *_: None, progress=False)
    assert mp.sel_k == [[0], [1], [2]] and mp.sel_v == [[0], [1], [2]]
    assert rep["mean_val_r2_k"] > 0.99 and rep["mean_val_r2_v"] > 0.99


def test_end_to_end_two_different_models(tmp_path):
    """Different sizes (2 vs 3 layers): fit both directions, run sessions, analyze, plot."""
    S, L = tiny_model(layers=2, hidden=64, seed=0), tiny_model(layers=3, hidden=96, seed=1)
    torch.manual_seed(2)
    calib = [torch.randint(0, 128, (1, 48)) for _ in range(10)]
    q = lambda *_: None
    s2l, _ = fit_mapper((S, S.config), (L, L.config), calib, k=2, n_select_seqs=4, device="cpu", log=q, progress=False)
    l2s, _ = fit_mapper((L, L.config), (S, S.config), calib, k=2, n_select_seqs=4, device="cpu", log=q, progress=False)
    path = tmp_path / "s2l.pt"
    s2l.save(path)
    s2l = Mapper.load(path, S.config, L.config)
    tok = FakeTok()
    recs = []
    for sess in make_sessions(fake_stories(6), n_sessions=3, stories_per_session=2, max_q_per_story=3):
        segs = tokenize_session(sess, tok)
        order = ["S" if t % 2 == 0 else "L" for t in range(len(segs))]
        recs += list(run_session({"S": S, "L": L}, {("S", "L"): s2l, ("L", "S"): l2s}, sess, segs, order, tok,
                                 max_new=4))
    res = analyze(recs, {"late_start_turn": 3, "signal_window": 2, "signal_horizon": 1}, outdir=str(tmp_path))
    assert res["decision"].split()[0] in {"DROP", "PIVOT", "GO"}
    for f in ("fig1_drift_vs_turn.png", "fig2_by_answerer.png", "fig3_signal.png", "summary.json"):
        assert (tmp_path / f).exists()


def test_disk_cache_matches_memory(tmp_path):
    """The on-disk feature cache (used for real models) must give the same translator as RAM."""
    S, L = tiny_model(layers=2, hidden=64, seed=0), tiny_model(layers=3, hidden=96, seed=1)
    torch.manual_seed(3)
    calib = [torch.randint(0, 128, (1, 40)) for _ in range(8)]
    kw = dict(k=2, n_select_seqs=4, device="cpu", log=lambda *_: None, progress=False)
    m_ram, _ = fit_mapper((S, S.config), (L, L.config), calib, **kw)
    store = FeatureStore({"S": S, "L": L}, calib, cache_dir=str(tmp_path / "cache"), progress=False,
                         log=lambda *_: None)
    m_disk, _ = fit_mapper((S, S.config), (L, L.config), store=store, names=("S", "L"), group_size=1, **kw)
    store.close()
    assert not (tmp_path / "cache").exists()
    assert m_ram.sel_k == m_disk.sel_k
    for a, b in zip(m_ram.Wk + m_ram.Wv, m_disk.Wk + m_disk.Wv):
        assert torch.allclose(a, b, atol=1e-4)
