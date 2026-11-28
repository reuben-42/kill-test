# kill-test

**Question:** when two models take turns in a long chat and pass translated notes (KV caches) to each other, does the translation error build up over time, and differently for different chats? If not, a "smart refresh" has nothing to fix.

## What the numbers mean

### The cast
| Term | Meaning |
|---|---|
| **S / L** | **S**mall model (Qwen3-0.6B) and **L**arge model (Qwen3-1.7B). "0.6B" = 0.6 billion adjustable numbers inside the model |
| **KV cache** | The model's "notes" on everything said so far |
| **Prefill** | Re-reading the whole chat to make fresh notes. Accurate but expensive |
| **Translator (mapper)** | Math that converts one model's notes into the other's shorthand ([Heo et al.](https://arxiv.org/abs/2608.03893)) |
| **Turn** | One question + one answer. A **chat** (session) is ~60 turns |
| **CoQA** | Public dataset of stories, each with a chain of questions and true answers |

### The three versions
| Name | What happens at each handoff |
|---|---|
| **prefill** | The answering model re-reads everything itself. Gold standard |
| **sw** (segment-wise) | Each model keeps its own notes. Each new page is translated **once**, from the notes of the model that read it |
| **nv** (naive) | One shared set of notes, **fully re-translated** at every handoff, including pages that were already translations (a photocopy of a photocopy) |

### The scores
| Term | Meaning |
|---|---|
| **F1** | How right an answer is: word overlap with the true answer, 0 to 1. It combines *precision* (of the words said, how many were right) and *recall* (of the right words, how many were said). True *"Mark Twain"* vs. said *"The writer is Mark Twain"* → precision 2/4, recall 2/2 → F1 = 0.67 |
| **pp** | Percentage points: plain subtraction of percentages (70% vs. 60% = 10 pp) |
| **Drift** | prefill's F1 − the other version's F1, in pp. **Positive = worse than re-reading** |
| **KL** (KL divergence) | How differently the model is "thinking" right before it answers: compares its probabilities for every possible next word with prefill's. 0 = identical, ~0.1 = nearly the same, ~1 = noticeably different, ~10 = wildly different. Unit: nats. Less noisy than F1 |
| **Surprise** | How unexpected the new question's words were to the model (average −log probability). Our proposed warning sign |

### The statistics
| Term | Meaning |
|---|---|
| **95% CI** | Range that very likely holds the true average, given only 20 chats. **If it includes 0, "no effect" can't be ruled out.** Computed by bootstrapping (resampling the chats thousands of times) |
| **p** (permutation test) | For "do chats differ?": the chance of differences this big if all chats were really the same. Below 0.05 = they really differ |
| **Spearman ρ** | Does one thing rise when another rises? −1 to +1, 0 = no relationship. Our bar for "the warning sign works" was 0.3 |
| **Oracle version** | The signal computed with information you wouldn't have in real use (prefill's own surprise): the best it could do |
| **Paired growth** | Each chat's late drift minus its *own* early drift, so harder chats don't skew the result |

## The experiment

1. **Two models from the same family** whose notes have the same shape: Qwen3-0.6B and Qwen3-1.7B (one 12 GB GPU).
2. **Built the translator:** both models read 64 web texts (FineWeb-Edu), and we learned a linear "dictionary" from one model's notes to the other's, in both directions.
3. **Built 20 long chats:** 4 CoQA stories glued together, up to 15 questions each, giving 48–60 turns.
4. **The models alternate answering**, large model first.
5. **Every chat was played 3 ways side by side:** prefill, segment-wise, naive.
6. **Same history for all three:** after each turn the *true* answer goes into the chat (teacher forcing), so the only difference between versions is the notes.
7. **Measured every turn:** F1, KL vs. prefill, and surprise.
8. **Decision rules were written down before running** (`thresholds.json`), so the test was fair.

## Findings

**Decision: PIVOT.** Drop "when to refresh over long chats". The cost sits in single handoffs of important pages.

| | Segment-wise | Naive |
|---|---|---|
| F1 drift vs. prefill, turns 1–10 → turn 50+ | +7.2 → +12.2 pp | +38.8 → +47.2 pp |
| Paired growth, turns 41+ minus 1–10 (95% CI) | +4.3 pp (−2.2 to +10.6) | +17.8 pp (+9.0 to +26.5) |
| KL, turns 1–10 → turn 50+ | 0.53 → 0.65 | 2.9 → 10.3 |
| Do chats differ? (p) | 0.50 | 0.59 |
| Warning sign vs. next 3 turns (ρ) · oracle | −0.01 · −0.04 | 0.04 · 0.46 |

1. **The setup works.** Naive collapses and gets worse over time, as earlier papers report.
2. **Segment-wise is a flat, one-time cost** of roughly 9–10 pp. The growth CI includes 0, so there's no proven build-up.
3. **The cost comes from using a translated story:**

   | Segment-wise turns | F1 drift | KL |
   |---|---|---|
   | Answering model read the current story itself | −1.1 pp | 0.11 |
   | Got the story translated (1–2 turns later) | +11.4 pp | 0.64 |
   | Further into the same story | +9.9 pp | 0.63 |

   Translated pages from *earlier* stories barely matter.
4. **All chats drift alike** (p ≈ 0.5), so a fixed refresh schedule does as well as a smart one.
5. **The surprise signal fails** (ρ ≈ 0). It compares against its own recent average, which hides a steady problem. The oracle version works on naive (ρ = 0.46), so surprise does carry information.

**Caveats:** one small model pair · translator fit on 64 texts (Heo et al. used 500) · only 20 chats · the single-handoff check (`check_handoff.py`) is flawed: it asks each story's first question with no example answer before it, so the 1.7B model answers with blanks (`____`), and its numbers should be ignored.

**Next:** is the ~10 pp per-handoff cost a weak translator (refit with more texts) or the two models genuinely differing?

Data: `results.jsonl` · plots and numbers in `analysis/` (`fig1_drift_vs_turn.png`, right panel, is the main figure).

## How to run

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
| `0.6b-1.7b` | Qwen3-0.6B ↔ 1.7B | ~12–16 GB, or 4–8 GB with the large model on the CPU (below) |
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
