# kill-test

**Question:** when two models take turns in a long chat and pass translated KV caches ("notes") to each other, does the translation error build up over time, and differently for different chats? If not, there's nothing for a smart refresh policy to fix, and we drop the project.

We play long CoQA chats three ways and compare them turn by turn:

| Version | What happens at each handoff |
|---|---|
| `prefill` | The answering model has its own notes on the whole chat (= always re-reading). The gold standard. |
| `sw` (segment-wise) | Each model keeps its own notes. Pages written by the other model are translated **once**, from that model's own notes. |
| `nv` (naive) | One set of notes, **fully re-translated** into the other model's format at every handoff (photocopy of a photocopy). |

The translator is the closed-form ridge mapper from Heo et al. ([arXiv 2608.03893](https://arxiv.org/abs/2608.03893)).

## Findings: first run (November 24, 2026)

**Decision by the pre-agreed rules: PIVOT.** Translation error is mostly a cost paid up front, it is about the same in every chat, and our cheap warning signal can't see it. A "smart refresh over long chats" policy has little room to beat a fixed schedule.

**Setup:** Qwen3-0.6B ↔ Qwen3-1.7B on one 12 GB GPU · 20 chats of 48–60 turns (4 CoQA stories each), models alternating every turn, large model first · translator fit on 64 FineWeb-Edu texts of 1,024 tokens (k = 8), stored in bfloat16 · thresholds as in `thresholds.json`.

| | Segment-wise | Naive re-translation |
|---|---|---|
| F1 drift vs. always re-reading, first 10 turns | +7.2 pp | +38.8 pp |
| F1 drift, turn 50+ (95% CI) | +12.2 pp (+6.7 to +18.0) | +47.2 pp (+37.0 to +57.4) |
| KL at the answer, first 10 turns → late | 0.53 → 0.65 | 2.91 → 10.27 |
| Late KL, small / large model answering | 0.60 / 0.73 | 8.03 / 14.21 |
| Do chats drift by different amounts? (permutation p) | 0.50 (no) | 0.59 (no) |
| Warning sign vs. next 3 turns' KL (Spearman ρ) | −0.01 | 0.04 |
| Same, oracle version of the signal | −0.04 | 0.46 |

*Drift* = how much worse than the model that re-reads everything (positive = worse). *pp* = percentage points of F1.

### What it means

1. **The setup checks out.** Naive re-translation collapses: its KL grows from 2.9 to 10.3 and it loses ~40–47 pp of F1, consistent with earlier reports that it collapses. So the code reproduces a known result, and the other numbers are worth taking seriously.

2. **Segment-wise error is mostly an up-front cost, not a slow build-up.** It is already +7.2 pp worse in the first 10 turns, and only ~5 pp more appears by turn 50. In picture form: the photocopier is blurry from the first copy and gets only slightly blurrier after that.
   **Not yet shown:** that the extra ~5 pp is real. The early value (7.2) lies inside the late 95% CI (6.7 to 18.0). The pre-agreed DROP rule tested whether late drift is *large* (> 3 pp), not whether it *grows* from early to late. Those are different questions, so the printout's "drift builds up" is stronger than the evidence. A paired test (each chat's late drift minus its own early drift) is still to do.

3. **Chats drift alike (p ≈ 0.5).** Every chat ends up about equally blurry, so knowing *which* chat to refresh early buys little over refreshing on a timer. This is what triggered PIVOT.

4. **The "surprise" warning sign doesn't work, for a structural reason.** The deployable signal compares the model's surprise now with its own recent average. If the notes are *steadily* blurry, the recent average is blurry too and the difference cancels out. A "compared with recently" signal sees *changes*, not a *constant* problem, and segment-wise drift is mostly constant. The oracle version (compared with the always-re-read model) does work once damage is large (ρ = 0.46 for naive), so surprise does carry information; the baseline is what hides it.

5. **"Large model answering" shows more KL**, but bigger models make sharper predictions, so KL isn't directly comparable across the two models. This does not contradict Heo et al.'s finding that large → small drift grows faster.

### Why a fixed refresh schedule is enough (and what that doesn't tell us)

Two findings are easy to mix up:

| Finding | What it tells us |
|---|---|
| **Chats drift alike** (p ≈ 0.5) | A smart refresh can't beat a timer: there is no "this chat is worse, refresh it early" to detect. **This is why a fixed schedule is enough.** |
| **Drift is mostly up front, then roughly flat** | The *shape* of drift over time. Not yet confirmed (paired growth test still to do). |

A fixed schedule would be enough even if drift grew a lot over time, as long as every chat grew at the same speed; we would just refresh more often. Analogy: if every car's oil gets dirty at exactly the same rate, a "change every 5,000 km" sticker works as well as an oil sensor.

What the kill test does **not** tell us: *how often* to refresh, or whether a refresh is worth its cost. That is a separate cost-vs-quality trade-off.

### Caveats

- One small model pair; larger pairs may behave differently.
- The translator saw 64 calibration texts (Heo et al. used 500). A better translator could shrink the up-front 7 pp.
- 20 chats is little data for the "do chats differ?" test: real but small differences between chats could be missed.

### Where this points

The long-chat refresh idea looks weak. The bigger effect is the **cost of each single handoff**, which is the group's original question (should *this* handoff trust the translated cache?). Open questions for that:

- Is the up-front 7 pp a **weak translator** (more calibration data, nonlinear mapper) or **the models genuinely representing things differently**? Re-fitting with `--n-calib 256` and comparing would start to separate the two.
- **Does the up-front cost come from one big page?** Each chat starts with a long story. In segment-wise, the model that didn't read the story gets it as translated notes. If most of the early ~7 pp comes from that one big translated page rather than the many short Q&A pages, then *what* was translated matters more than *how many turns* have passed, and refreshing right after a long document is translated may be what pays off. Test: compare drift on turns right after a new story arrives with turns mid-story, or translate the story with re-reading and only the Q&A pages with the translator.
- A signal for a *constant* problem needs a baseline that isn't itself blurry, e.g. the sending model's own surprise on the same text.
- Run the paired growth test above before claiming anything about build-up.

## Run it (4 commands)

```bash
pip install -r requirements.txt
python -m pytest -q                                  # ~10 s on CPU, checks the plumbing with tiny random models

# 0. Agree on thresholds.json FIRST (what counts as drop / pivot / go). Don't change it after.
python scripts/fit_mapper.py    --pair 4b-8b         # 1. fit translators: small->large and large->small
python scripts/check_handoff.py --run runs/4b-8b     # 2. sanity check: does ONE handoff work?
python scripts/run_kill_test.py --run runs/4b-8b     # 3. 20 chats x ~60 turns x 3 versions
python scripts/analyze.py       --run runs/4b-8b     # 4. plots + DROP / PIVOT / GO
```

Step 3 can be stopped and restarted; finished chats are skipped.

### Which model pair for your GPU

All pairs below are "matched-KV" (8 KV heads, head size 128), which the mapper needs. Same tokenizer, so pages line up token for token.

| `--pair` | Models | Rough GPU memory |
|---|---|---|
| `0.6b-1.7b` | Qwen3-0.6B ↔ 1.7B | ~16 GB, or 4–8 GB with the large model on the CPU (below) |
| `1.7b-4b` | Qwen3-1.7B ↔ 4B | ~24 GB (add `--mapper-dtype bfloat16` to steps 2–3 if tight) |
| `4b-8b` (default) | Qwen3-4B ↔ 8B | ~40 GB |
| `8b-32b` | Qwen3-8B ↔ 32B (a pair Heo et al. tested) | 80 GB+, or two GPUs: `--devices cuda:0,cuda:1` |

Out of memory while fitting? Use `--group-size 2` (or 1). These numbers are estimates: about 2 bytes per model parameter, plus the translators and fitting buffers.

### Small GPU (under ~8 GB): small model on the GPU, large model on the CPU

Results are the same; only speed changes. The 1.7B model on the CPU makes its turns slow, so expect hours (an overnight run).

```bash
python scripts/fit_mapper.py    --pair 0.6b-1.7b --devices cuda,cpu --group-size 1
python scripts/check_handoff.py --run runs/0.6b-1.7b --mapper-dtype bfloat16 --mapper-device cuda
python scripts/run_kill_test.py --run runs/0.6b-1.7b --mapper-dtype bfloat16 --mapper-device cuda
python scripts/analyze.py       --run runs/0.6b-1.7b
```

- Fitting runs each model over the calibration texts **once** and keeps their notes on disk while it works:
  about 15 GB for `0.6b-1.7b` with the default 64 texts, plus ~3.8 GB for the two saved translators.
- `--devices` is remembered in `run_config.json`, so steps 2–3 reuse it.
- Try `run_kill_test.py --n-sessions 2` first: it prints seconds per chat, so you can estimate the full run.
- The 1.7B model needs about 7 GB of normal RAM on the CPU (it runs in float32 there).

### Out of disk space / `basic_ios::clear: iostream error` when saving

School home folders usually have a storage quota (check `quota -s`). Put the big files on the machine's local disk:
`--cache-dir /tmp/$USER-cache --out /tmp/$USER-run`, then pass `--run /tmp/$USER-run` to the later steps.
Copy `results.jsonl` and `analysis/` back home at the end, because `/tmp` can be wiped. Or use fewer texts with `--n-calib 32`.

### pytest crashes with `substitute_in_graph expects a function ... ansible`

Another pytest add-on on that computer (Ansible's) breaks PyTorch. Our tests need no add-ons, so turn them off:
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q`

### GPU nodes without internet (common on university clusters)

On a machine with internet:
```bash
python scripts/download_data.py --out data/
huggingface-cli download Qwen/Qwen3-4B && huggingface-cli download Qwen/Qwen3-8B
```
Then add `--calib-text-file data/calib.txt` to step 1 and `--coqa-file data/coqa_validation.json` to steps 2–3.

## What comes out (`runs/<pair>/`)

| File | What it is |
|---|---|
| `fit_report_*.json` | Translator quality (held-out R², 1.0 = perfect) and which source layers feed each target layer |
| `handoff_report.json` | Single-handoff check: F1 with translated vs. own notes, plus example answers |
| `results.jsonl` | One line per turn: each version's answer, F1, KL vs. `prefill`, and "surprise" |
| `analysis/fig1_drift_vs_turn.png` | Does drift grow over the chat? (faint line per chat, bold average) |
| `analysis/fig2_by_answerer.png` | Is it worse when the small model receives notes from the large one? |
| `analysis/fig3_signal.png` | Does surprise warn us before drift? |
| `analysis/summary.json` | All numbers + the decision |

### How the decision is made (`thresholds.json`)

1. **DROP** if segment-wise F1 drift from turn 50 on is under 3 pp and its 95% CI touches 0 → nothing builds up.
2. **PIVOT** if chats don't drift by different amounts (permutation test on late KL, p ≥ 0.05) → a fixed refresh schedule is enough.
3. **PIVOT (signal)** if drift differs by chat but surprise doesn't track the next 3 turns' KL (Spearman < 0.3).
4. **GO** otherwise.

Naive re-translation should look clearly worse than segment-wise. Earlier papers report it collapsing, so if it doesn't, suspect the setup before trusting the decision.

## Measurements, in plain words

- **F1**: word overlap between an answer and the true CoQA answer (1 = perfect). Drift = `prefill` F1 − other F1, in percentage points.
- **KL**: how different the model's next-word probabilities are at the answer, compared with `prefill`. 0 = identical. Less noisy than F1.
- **Surprise**: how unexpected the new question's words were to the model (average negative log-probability). It's free, because the model reads the question anyway. The analysis subtracts the same model's recent average, so a hard question isn't mistaken for blurry notes.

## Choices to know about (things Heo et al. don't fully specify, or that we simplified)

- **Teacher forcing**: the chat history always contains the *true* previous answers, so all three versions read exactly the same text and only the notes differ.
- **Prompt format**: plain text (`Story: … Q: … A:`), not a chat template, so Qwen3's thinking mode never triggers. Answers are greedy and stop at a newline.
- **Mapper**: keys and values pick their top-k source layers separately; ridge is solved on mean-centred statistics with λ = 0.01 and a bias term. Calibration defaults to 64 × 1024-token FineWeb-Edu sequences (Heo et al. used 500), which is enough for a kill test. Raise it with `--n-calib`.
- **"Naive re-translation"** follows our reading of Kvedaras & Rasymas (arXiv 2611.12062): the whole cache is re-translated at every handoff. Check their Section 8 to confirm.
- **Sessions**: 4 CoQA validation stories glued together, up to 15 questions each (≤ 60 turns). Models alternate every turn, large model first.

## Code map

```
killtest/kv.py        notes as plain tensors, RoPE remove/re-apply, model calls
killtest/mapper.py    the translator: layer selection, closed-form ridge fit, translate()
killtest/runner.py    plays one chat three ways; single-handoff check
killtest/data.py      CoQA -> long chats, calibration text
killtest/analysis.py  statistics, plots, decision
killtest/metrics.py   CoQA F1, Spearman, bootstrap, permutation test
tests/                CPU tests with tiny random models (perfect translator => all versions match prefill)
```
