"""Play one long chat three ways, side by side, turn by turn.

  prefill : the answering model always has its OWN notes on the whole chat
         (same result as re-reading everything; we just catch it up incrementally).
  sw   : segment-wise. Each model keeps its own notes. Pages written by the other
         model are translated ONCE, from that model's own notes, then kept.
  nv   : naive re-translation. One set of notes; at every handoff the WHOLE thing is
         translated into the next model's format (including already-translated pages).

The answering model always reads the new message (story / question) itself.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .kv import KV, forward
from .metrics import f1

CONDITIONS = ("prefill", "sw", "nv")


def _surprise(logits, ids, start):
    """Mean negative log-prob of ids[j] for j >= max(start, 1), using in-segment logits."""
    js = list(range(max(start, 1), len(ids)))
    if not js:
        return float("nan")
    lp = F.log_softmax(logits[[j - 1 for j in js]], dim=-1)
    tgt = torch.tensor([ids[j] for j in js], device=lp.device)
    return float(-lp.gather(1, tgt[:, None]).mean())


@torch.no_grad()
def _greedy(model, kv, last_logits, tok, max_new):
    out, nxt = [], int(last_logits.argmax())
    for _ in range(max_new):
        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
            break
        out.append(nxt)
        if "\n" in tok.decode(out):
            break
        logits, kv = forward(model, torch.tensor([[nxt]]), kv)
        nxt = int(logits[-1].argmax())
    return tok.decode(out).split("\n")[0].strip()


@torch.no_grad()
def single_handoff(src_model, tgt_model, mapper, context_ids, question_ids, tok, true_answer, max_new=24):
    """Sanity check: target reads the question itself, but the story comes as translated notes."""
    ctx, q = torch.tensor([context_ids]), torch.tensor([question_ids])
    _, kv_native = forward(tgt_model, ctx, KV(), want_logits=False)
    lg_n, kv_n = forward(tgt_model, q, kv_native)
    _, kv_src = forward(src_model, ctx, KV(), want_logits=False)
    kv_tr = mapper.translate(kv_src, 0, out_dtype=tgt_model.dtype).to(tgt_model.device)
    lg_t, kv_t = forward(tgt_model, q, kv_tr)
    a_n, a_t = _greedy(tgt_model, kv_n, lg_n[-1], tok, max_new), _greedy(tgt_model, kv_t, lg_t[-1], tok, max_new)
    lp_n, lp_t = F.log_softmax(lg_n[-1], -1), F.log_softmax(lg_t[-1], -1)
    return dict(native_answer=a_n, translated_answer=a_t, true_answer=true_answer,
                native_f1=f1(a_n, true_answer), translated_f1=f1(a_t, true_answer), agree_f1=f1(a_t, a_n),
                kl=float((lp_n.exp() * (lp_n - lp_t)).sum()), top1_agree=bool(lp_n.argmax() == lp_t.argmax()))


@torch.no_grad()
def run_session(agents: dict, mappers: dict, session: dict, segs: list, order: list, tok,
                max_new: int = 24, conditions=CONDITIONS):
    """agents: name -> model.  mappers: (src_name, tgt_name) -> Mapper.  order[t]: who answers turn t.

    Yields one record per turn.
    """
    starts, pos = [], 0
    for s in segs:
        starts.append(pos); pos += len(s["ids"])
    ends = [s + len(g["ids"]) for s, g in zip(starts, segs)]

    native = {n: KV() for n in agents}; native_done = {n: 0 for n in agents}
    own = {n: KV() for n in agents}; own_done = {n: 0 for n in agents}
    shared, shared_fmt = KV(), None

    for t, turn in enumerate(session["turns"]):
        M = order[t]
        model = agents[M]
        seg = segs[t]["ids"]
        seg_ids = torch.tensor([seg])
        rec = dict(session_id=session["session_id"], turn=t, answerer=M, story_pos=turn["story_pos"],
                   new_story=turn["new_story"], true_answer=turn["gold"], hist_len=starts[t], seg_len=len(seg))
        results = {}

        # --- prefill: catch the model up on anything it hasn't read, natively --------
        if "prefill" in conditions:
            if native_done[M] < t:
                missing = torch.tensor([sum((segs[j]["ids"] for j in range(native_done[M], t)), [])])
                _, native[M] = forward(model, missing, native[M], want_logits=False)
            logits, native[M] = forward(model, seg_ids, native[M])
            native_done[M] = t + 1
            results["prefill"] = (logits, native[M])

        # --- segment-wise: translate each missing page once, from its writer's notes --
        if "sw" in conditions:
            for j in range(own_done[M], t):
                w = order[j]
                assert w != M, "segment-wise: a page this model wrote should already be in its notes"
                page = own[w].slice(starts[j], ends[j])
                own[M] = own[M].cat(mappers[(w, M)].translate(page, starts[j], out_dtype=model.dtype)
                                    .to(model.device))
            logits, own[M] = forward(model, seg_ids, own[M])
            own_done[M] = t + 1
            results["sw"] = (logits, own[M])

        # --- naive: re-translate the whole shared notebook at every handoff ----------
        if "nv" in conditions:
            if shared_fmt is not None and shared_fmt != M:
                shared = mappers[(shared_fmt, M)].translate(shared, 0, out_dtype=model.dtype).to(model.device)
            logits, shared = forward(model, seg_ids, shared)
            shared_fmt = M
            results["nv"] = (logits, shared)

        # --- answers + measurements (prefill first, so the others compare against it) --
        lp_ref = F.log_softmax(results["prefill"][0][-1], -1) if "prefill" in results else None
        for c, (logits, kv) in results.items():
            ans = _greedy(model, kv, logits[-1], tok, max_new)
            rec[f"{c}_answer"] = ans
            rec[f"{c}_f1"] = f1(ans, turn["gold"])
            rec[f"{c}_surprise_q"] = _surprise(logits, seg, segs[t]["q_start"])
            rec[f"{c}_surprise_seg"] = _surprise(logits, seg, 0)
            if c != "prefill" and lp_ref is not None:
                lp = F.log_softmax(logits[-1], -1)
                rec[f"{c}_kl"] = float((lp_ref.exp() * (lp_ref - lp)).sum())
                rec[f"{c}_top1_agree"] = bool(lp.argmax() == lp_ref.argmax())
                rec[f"{c}_agree_f1"] = f1(ans, rec["prefill_answer"])
        yield rec
