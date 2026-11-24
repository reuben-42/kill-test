"""Data: long CoQA chats (sessions), calibration text, and the tokenizer wrapper."""
from __future__ import annotations

import random

import torch

HEADER = "Read each story and answer the questions about it in a few words.\n\n"


class Tok:
    """Small wrapper so the pipeline can run with a real HF tokenizer or a fake one in tests."""

    def __init__(self, hf_tokenizer):
        self.t = hf_tokenizer
        self.eos_token_id = hf_tokenizer.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.t.encode(text, add_special_tokens=False)

    def decode(self, ids) -> str:
        return self.t.decode(ids, skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# CoQA -> long sessions
# --------------------------------------------------------------------------- #
def load_coqa(split: str = "validation", path: str | None = None) -> list[dict]:
    """CoQA stories. `path` = a local JSON list of {story, questions, answers} (for offline GPU nodes)."""
    if path:
        import json

        with open(path) as f:
            return json.load(f)
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/coqa", split=split)
    return [dict(story=r["story"], questions=list(r["questions"]), answers=list(r["answers"]["input_text"]))
            for r in ds]


def make_sessions(stories: list[dict], n_sessions: int, stories_per_session: int = 4,
                  max_q_per_story: int = 15, seed: int = 0) -> list[dict]:
    """Glue several CoQA conversations into one long chat. Each story is used at most once."""
    rng = random.Random(seed)
    order = list(range(len(stories)))
    rng.shuffle(order)
    need = n_sessions * stories_per_session
    if need > len(order):
        raise ValueError(f"need {need} stories, only {len(order)} available")
    sessions = []
    for i in range(n_sessions):
        turns = []
        for j, si in enumerate(order[i * stories_per_session:(i + 1) * stories_per_session]):
            s = stories[si]
            for qi, (q, a) in enumerate(list(zip(s["questions"], s["answers"]))[:max_q_per_story]):
                turns.append(dict(story_id=si, story_pos=j, new_story=(qi == 0),
                                  story=s["story"] if qi == 0 else None,
                                  question=q.strip(), gold=a.strip()))
        sessions.append(dict(session_id=i, turns=turns))
    return sessions


def segment_texts(turns: list[dict]) -> list[dict]:
    """Text each turn adds to the chat BEFORE the model answers.

    Teacher forcing: the history always contains the TRUE previous answer, not the
    model's, so every condition sees exactly the same text and only the notes differ.
    Returns per turn {"pre": text before the question, "question": "Q: ...\\nA:"}.
    """
    out = []
    for t, turn in enumerate(turns):
        pre = HEADER if t == 0 else f" {turns[t - 1]['gold']}\n"
        if turn["new_story"]:
            pre += ("" if t == 0 else "\n") + f"Story:\n{turn['story'].strip()}\n\n"
        out.append(dict(pre=pre, question=f"Q: {turn['question']}\nA:"))
    return out


def tokenize_session(session: dict, tok) -> list[dict]:
    """Token ids per turn: ids = pre + question, q_start = where the question begins."""
    segs = []
    for s in segment_texts(session["turns"]):
        pre, q = tok.encode(s["pre"]), tok.encode(s["question"])
        segs.append(dict(ids=pre + q, q_start=len(pre)))
    return segs


# --------------------------------------------------------------------------- #
# Calibration text for fitting the translator (FineWeb-Edu, like Heo et al.)
# --------------------------------------------------------------------------- #
def load_calibration(tok, n_seqs: int, seq_len: int = 1024, seed: int = 0,
                     dataset: str = "HuggingFaceFW/fineweb-edu", config: str = "sample-10BT",
                     text_file: str | None = None) -> list[torch.Tensor]:
    if text_file:
        with open(text_file) as f:
            docs = [d for d in f.read().split("\n\n") if d.strip()]
        stream = iter(docs)
    else:
        from datasets import load_dataset

        ds = load_dataset(dataset, config, split="train", streaming=True).shuffle(seed=seed, buffer_size=1000)
        stream = (r["text"] for r in ds)
    seqs = []
    for text in stream:
        ids = tok.encode(text)
        if len(ids) >= seq_len:
            seqs.append(torch.tensor(ids[:seq_len])[None])
            if len(seqs) == n_seqs:
                break
    if len(seqs) < n_seqs:
        raise RuntimeError(f"only found {len(seqs)} documents with >= {seq_len} tokens")
    return seqs
