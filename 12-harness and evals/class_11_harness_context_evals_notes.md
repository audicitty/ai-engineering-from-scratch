# Class 11 — Harness, Context, Evals

### Engineering systems around model intelligence

> Three sides of the same problem — **what wraps the model · what goes in the window · how we know it works**

---

## 0. Primer (for anyone new to AI)

| Word | Plain meaning |
| --- | --- |
| **Model** | The trained AI brain. Reads text, predicts the next words. On its own it can only talk. |
| **Harness** | The code wrapped *around* the model that lets it actually do things — read files, run commands, retry, stop. |
| **Agent** | A model plus a harness, working on a task by itself over many steps. |
| **Context window** | The model's short-term memory — everything it can "see" at once: your prompt, past messages, tool outputs. |
| **Token** | A chunk of text (~¾ of a word). Context windows are measured in tokens. |
| **Tool call** | The model asking the harness to do something real (run a command, search a file) and getting the result back. |
| **Eval** | A structured test of whether your system works. Not a vibe check — a repeatable score. |
| **Trace / transcript** | The full recorded log of one agent run: every message, tool call and result. |

> **One-line summary:** The model is only part of the product. This class is about the other part — the code around it, the information you feed it, and the tests that tell you whether either is working.

---

## 1. Today's Arc

```
   01   Harness    what wraps the model
   02   Context    what goes in the window
   03   Evals      training data for the harness
```

> Three sides of the same problem.

---

## 2. The Motivating Fact

**Same model. Three harnesses. Why don't they get the same score?**

| Harness | Score | Model |
| --- | --- | --- |
| ForgeCode | **79.8%** | Claude Opus 4.6 |
| Capy | **75.3%** | Claude Opus 4.6 |
| Claude Code | lower | Claude Opus 4.6 |

*Terminal-Bench 2.0 — same model, different harness.*

**Why this matters:** the model column is identical. Every point of difference comes from the code wrapped around it. If you're building a product, most of your leverage is in that wrapper — not in swapping models.

---

# BEAT 1 — HARNESS

---

## 3. Agent = Model + Harness

```
   ┌─────────────────────────────────────┐
   │      Agent  =  Model  +  Harness    │
   └─────────────────────────────────────┘
```

> **If you're not the model, you're the harness.**

Meaning: unless you train foundation models, every improvement you can make lives in the harness. That's your job.

---

## 4. Harness Primitives

**Each one patches a specific model deficiency.**

| # | Primitive | Deficiency it patches |
| --- | --- | --- |
| 01 | **Filesystem** | no durable state |
| 02 | **Bash + code** | no general-purpose action |
| 03 | **Sandbox** | unsafe execution at scale |
| 04 | **Sub-agents** | context contamination |
| 05 | **Middleware** | no deterministic interventions |

**Read the right-hand column first.** The primitives weren't dreamed up — each is a patch for something the model genuinely cannot do:

- A model forgets everything between runs → give it a **filesystem** to write notes to.
- A model can only emit text → give it **bash** so text becomes action.
- Model-written code might be destructive → run it in a **sandbox**.
- One long conversation fills with junk that confuses later steps → spin off **sub-agents** with clean context.
- Some rules must *always* hold (never delete prod, always stop after 50 steps) → enforce them in **middleware** code, not by asking the model nicely.

That last one is the subtle one. **Middleware = deterministic guarantees.** A model instructed not to do something might still do it. Code that refuses cannot.

---

## 5. Model–Harness Fit

**The harness is part of the model's effective parameters.**

| Operation | Codex / OpenAI | Claude / Anthropic |
| --- | --- | --- |
| Editing a file | `apply_patch` (diff) | `Edit` (old/new string) |
| Injected notes | `<oai-mem-citation>` | `<system-reminder>` |
| Running commands | `exec_command_tool` | `Bash` + Monitor |

> **Same operations. Different vocabularies. Baked in by post-training.**

**Why this happens:** each lab fine-tuned its model on *its own* tool format. The model has effectively memorised that vocabulary. Give Claude an `apply_patch` diff tool and it will underperform — not because diffs are worse, but because it was never trained on that shape.

Hence: the harness isn't neutral plumbing. It behaves like an extension of the model's weights.

---

## 6. Three Consequences

```
   01   No model-agnostic agent.
        Honest version — per-model harness; you pick a product, not a model.

   02   Mid-chat model swaps break.
        Transcript OOD, cache miss, tool-surface mismatch — all at once.

   03   The matched pair shifts.
        Yesterday's load-bearing scaffold is today's dead weight.
```

**Unpacking each:**

- **01** — "Works with any model" is marketing. In practice each model needs its own tuned harness, so you are really choosing a model+harness product.
- **02** — Swapping models halfway through a conversation fails three ways at once: the transcript is full of the *other* model's tool format (**OOD** = out of distribution, i.e. unlike anything it saw in training), the provider's **cache** of your conversation is invalidated so cost and latency spike, and the **tool surface** no longer matches.
- **03** — Harness code that compensates for a model weakness becomes dead weight once the next model fixes that weakness — and dead weight actively hurts, because it burns context and adds steps. **So delete code as models improve.**

---

## 7. Beat 1 Takeaway

```
   —   Primitives are derived, not invented.
   —   Match the pair: harness ↔ model.
   —   Scaffolding goes stale — delete code.
```

---

## 8. Beat 1 Hands-On

**Add primitives. Watch failures fix.**

| Primitive added | Failure it fixes |
| --- | --- |
| **Tool-result clearing** | output bloat — old tool results crowd the window; clear them once used |
| **Loop detection** | doom loops — agent repeats the same failing action forever; detect and break out |
| **Sub-agent dispatch** | parent context pollution — hand messy exploration to a child, keep only its answer |

> *(No notebook was attached with this deck — these three are what the practical session builds. If you have the `.ipynb`, send it and I'll add a cell-by-cell walkthrough.)*

---

# BEAT 2 — CONTEXT

---

## 9. The Problem — Context Rot

```
   accuracy (%)
   100 ┤╲
       │ ╲╲╲___
    80 ┤   ╲╲   ‾‾‾───___
       │    ╲ ╲╲         ‾‾‾───____  Sonnet 4
    60 ┤     ╲  ╲╲___              ‾‾‾ GPT-4.1
       │      ╲      ‾‾‾───____
    40 ┤       ‾‾‾───__________‾‾‾‾   Gemini 2.5
       └──┬─────┬──────┬──────┬──────┬
          0     8      16     24     32
              input tokens (thousands)
```

*Chroma research, 2025 · trivial copy task*

> **Frontier models break on a trivial copy task as input grows.**

**Read that again — the task is trivial.** Just copy words back. All three frontier models start near 100% and fall to 50–65% by 32k tokens. Nothing got harder except the amount of surrounding text.

**The lesson:** a big context window is a *capacity* number, not a *performance* guarantee. "It fits in 200k tokens" does not mean "it works at 200k tokens."

---

## 10. Attention Budget

> **Context is finite — in performance, not just tokens.**

```
       n tokens  →  n²  pairwise relationships
```

*Every token costs attention budget.*

> **THE PRINCIPLE: Smallest set of high-signal tokens.**

**Why n²:** the attention mechanism compares every token against every other token. Double the input and the comparisons quadruple. The model's finite "attention" is spread thinner across more pairs, so each individual fact gets less of it.

**Practical consequence:** adding a document "just in case" is not free. It dilutes everything else. Curate ruthlessly.

---

## 11. Right Altitude (System Prompts)

```
   BRITTLE ────────────── GOLDILOCKS ────────────── VAGUE
      │                        ●                        │
   hardcoded              specific behavior,        "be helpful,
    if/else               flexible judgment           be safe"
```

> **Specific behavior, flexible judgment.**

- **Too brittle** — you've written a giant if/else tree in English. It handles every case you imagined and shatters on the first one you didn't.
- **Too vague** — "be helpful, be safe" gives the model no actual guidance, so behaviour is unpredictable.
- **Goldilocks** — state *what good looks like* and the constraints that matter, then let the model's judgment fill the gaps.

---

## 12. Pre-loaded vs Just-in-Time

| **Pre-loaded / RAG** | **Just-in-time / Claude Code** |
| --- | --- |
| Stuff context up front | Lightweight identifiers |
| Fast on small data | Agent navigates via tools |
| Drowns on large data | Progressive disclosure |
| Throws away metadata | Mirrors human cognition |

**Pre-loaded:** search for relevant chunks, paste them all in, then ask. Fine when the corpus is small; collides with context rot when it isn't.

**Just-in-time:** give the agent *pointers* (file paths, IDs) and tools to fetch things itself. It opens only what it needs, when it needs it.

**"Throws away metadata"** is the underrated line. Chunking a repo into text fragments discards folder structure, file names, timestamps — signals a human would use immediately. JIT keeps them, because the agent sees the real filesystem.

**"Mirrors human cognition"** — you don't memorise a codebase before fixing a bug. You open files as you form hypotheses. JIT works the same way.

---

## 13. Long-Horizon Strategies

| Strategy | Use when | Example |
| --- | --- | --- |
| **Compaction** | conversational flow | summary-then-continue |
| **Note-taking** | iterative milestones | `NOTES.md`, scratchpad |
| **Sub-agents** | parallel exploration | clean context per branch |

- **Compaction** — when the window fills, summarise the conversation so far and continue from the summary. Keeps one continuous thread. Cost: detail is lost in compression.
- **Note-taking** — the agent writes findings to a file. Memory now lives *outside* the window and survives compaction entirely.
- **Sub-agents** — spawn a child with a fresh window for one sub-task; it returns only the answer, not its mess.

These are the context-side counterparts of the Beat 1 primitives — same problem, attacked from the information side.

---

## 14. Beat 2 Hands-On

```
   01   Reproduce context rot live    — Chroma's repeat-words task
   02   Compare pre-load vs JIT       — on a research task
   03   Add compaction                — watch the saw-tooth
```

**"Watch the saw-tooth"** describes the token-count graph under compaction: context grows, hits the threshold, collapses to a summary, grows again — a repeating saw-tooth pattern.

---

# BEAT 3 — EVALS

---

## 15. Why Evals

> **Evals are training data for harness improvement.**

```
   harness  +  evals  +  harness engineering   →   better agent
```

*Same loop as supervised learning. Gradient flows into the harness.*

**The analogy, made precise:** in supervised learning, labelled data plus gradient descent updates the *weights*. Here, evals plus your engineering updates the *harness*. The eval failures are the loss signal; you are the optimiser.

Without evals you're changing code and guessing. With them, every change has a number attached.

---

## 16. Vocabulary

| Term | Meaning |
| --- | --- |
| **task** | single test, defined inputs + success criteria |
| **trial** | one attempt at a task (run multiple) |
| **grader** | scoring logic — code, LLM, or human |
| **transcript** | full record — outputs, tool calls, reasoning |
| **outcome** | final state in the environment |
| **eval harness vs agent harness** | different layers — **don't conflate** |

**Why "trial" is separate from "task":** agents are non-deterministic. One run tells you almost nothing. Run the same task many times and you learn about *reliability*, which is section 20's whole point.

**Transcript vs outcome:** the outcome says whether it worked. The transcript says *why*. You debug from transcripts.

**The conflation warning:** the **agent harness** is the product — the thing being tested. The **eval harness** is the test rig around it. Mixing them means your tests change whenever your product does, and you lose your baseline.

---

## 17. The Mirage of Generic Metrics

**MMLU, HELM, BERTScore — these don't tell you if your product works.**

| **Foundation eval** | **Product eval** |
| --- | --- |
| Is the model generally capable? | Does **YOUR** pipeline do its job? |
| Standardized | Domain-specific |
| Cross-model | Failure-mode-driven |
| Easy to game | Custom criteria |
| Weak signal for products | Strong signal for shipping |

Foundation evals answer a model-buyer's question ("is this model good?"). They cannot answer a builder's question ("does my thing work for my users?"), because your failures are specific to your domain, your prompts and your harness.

**"Failure-mode-driven"** is the operative phrase: good product evals are built backwards from the ways your system actually breaks.

---

## 18. The 5-Star Lie

**What does a 3.7 in "helpfulness" mean?**

```
                    Binary  >  Likert

   Forces clarity · Faster to label · More consistent · Actionable
```

> **Decompose nuance into multiple binary checks, not a fuzzier scale.**

**The problem with a 1–5 scale:** nobody agrees what 3 means versus 4. Two labellers drift, the same labeller drifts across a day, and a score of 3.7 gives you no idea what to fix.

**The fix:** replace one fuzzy question with several sharp yes/no ones —

```
   ✗  "Rate helpfulness 1–5"  →  3.7

   ✓  Did it answer the question asked?        yes / no
      Did it cite a real source?               yes / no
      Did it avoid inventing facts?            yes / no
      Was it within the length limit?          yes / no
```

Now a failure names itself. You don't lose nuance — you relocate it from a fuzzy scale into multiple crisp checks.

---

## 19. Critique Shadowing

**Hamel Husain's process for building product evals.**

```
   01   Find THE principal domain expert
   02   Build diverse dataset (features × scenarios × personas)
   03   Pass / fail + detailed critique
   04   Fix obvious agent errors
   05   Build LLM judge, iterate to alignment
   06   Error analysis at scale
```

**Step by step:**

1. **THE expert, singular.** One authoritative judge. A committee produces inconsistent labels, and inconsistent labels can't train a judge.
2. **Diverse dataset** — build it systematically as a grid: features × scenarios × personas. Prevents twenty near-identical test cases.
3. **Pass/fail + critique.** Binary verdict (section 18) *plus* a written reason. The critiques become the judge's instructions later.
4. **Fix obvious errors first.** Don't automate scoring of bugs you can already see and fix cheaply.
5. **Build the LLM judge**, then tune it until it agrees with the expert.
6. **Error analysis at scale** — now that the judge is aligned, run it over thousands of traces and cluster the failures.

The name: the LLM judge *shadows* the expert's critiques until it can stand in for them.

---

## 20. LLM Judge Alignment

**Different splits than ML — small train, big test.**

```
   TRAIN 20%          DEV 40%              TEST 40%
   few-shot examples  iterate prompt       touched ONCE
```

> **Measure TPR + TNR — not raw agreement.**

**Why the split is inverted:** in normal ML, most data goes to training because the model learns from volume. Here "training" just means picking a handful of few-shot examples for the judge's prompt — that needs very little data. Most of your precious hand-labelled data should go to *measuring* whether the judge is trustworthy.

**"Touched ONCE"** — if you look at the test set, tweak the judge, and look again, you've tuned to it and the number is no longer honest.

**Why TPR + TNR instead of accuracy:**

- **TPR** (true positive rate) = of the genuinely good outputs, how many did the judge pass?
- **TNR** (true negative rate) = of the genuinely bad outputs, how many did the judge catch?

If 90% of your examples pass, a judge that says "pass" to everything scores 90% raw agreement while catching zero failures. TPR/TNR expose that instantly — TNR would be 0%.

---

## 21. pass@k versus pass^k

**Two metrics. Opposite stories.**

| | **pass@k** | **pass^k** |
| --- | --- | --- |
| Question | Any path to success in k tries? | **All** k trials succeed? |
| As k grows | ↑ goes **up** | ↓ goes **down** |
| Use for | first-try problems | reliability-critical |

**The trap:** both are computed from the same trials, and they move in opposite directions. Report `pass@k` and your agent looks like it's improving as you allow more attempts. Report `pass^k` and the same agent looks worse.

**Which to use:** if a human reviews the output and can retry (code suggestions, drafts), `pass@k` is honest. If the agent acts unsupervised — sending emails, moving money, changing production — `pass^k` is the only number that matters, because one failure out of k is a real failure.

---

## 22. The Flywheel

```
                      Traces
                    ●───────►●
                  ↗            ↘
      Better ●                    ● Evals
      agent    ↖                ↙
                    ●◄───────●
                 Harness updates
```

> **Every trace is a potential eval. Every eval improves the harness.**

The loop: run the agent → it produces **traces** → interesting traces (especially failures) become **evals** → evals point to **harness updates** → you get a **better agent** → which produces new traces.

**Why it's a flywheel and not just a loop:** each turn makes the next easier. Your eval suite grows, your failure taxonomy sharpens, and improvements compound. The cost of running it is roughly constant; the value keeps rising.

**It never ends** — new models shift the matched pair (section 6, consequence 03), so the flywheel keeps finding new work.

---

## 23. Beat 3 Hands-On

```
   01   Synthesize 20 diverse queries for company-research agent
   02   Hand-label pass / fail with critiques
   03   Train / dev / test split — write judge v1 (no few-shot)
   04   Iterate to v2 with few-shot, lock and test
   05   Cluster failure modes → harness backlog
```

This is critique shadowing (section 19) compressed into one session. Note **step 03 deliberately builds v1 without few-shot examples** — that's your baseline, so step 04 can show what few-shot actually bought you. And step 05 closes the flywheel: failure clusters become the harness to-do list.

---

# CLOSING

## 24. Three Takeaways

```
   01   Agent = Model + Harness.
        harness is part of the model's effective parameters

   02   Context is finite — in performance.
        smallest set of high-signal tokens, JIT > pre-load

   03   Evals are training data.
        binary > Likert, aligned LLM judge, the flywheel never ends
```

---

## 25. Quick-Recall Table

| Question | Answer |
| --- | --- |
| Why do three harnesses score differently on one model? | The harness is part of the model's effective parameters — same model, different wrapper, different result. |
| The five harness primitives? | Filesystem, bash+code, sandbox, sub-agents, middleware. |
| What is middleware for? | Deterministic interventions — rules enforced in code, not requested in a prompt. |
| Why does a mid-chat model swap break? | Transcript OOD + cache miss + tool-surface mismatch, simultaneously. |
| What is context rot? | Accuracy falls as input length grows, even on trivial tasks. Capacity ≠ performance. |
| Why does every token cost? | n tokens → n² pairwise attention relationships. |
| The context principle? | Smallest set of high-signal tokens. |
| Right altitude? | Between brittle (hardcoded if/else) and vague ("be helpful") — specific behaviour, flexible judgment. |
| Pre-loaded vs JIT? | Pre-load stuffs context up front, drowns on large data. JIT gives identifiers + tools, progressive disclosure. |
| Three long-horizon strategies? | Compaction, note-taking, sub-agents. |
| Why are evals "training data"? | Same loop as supervised learning — the gradient flows into the harness instead of the weights. |
| Why binary over Likert? | Forces clarity, faster to label, more consistent, actionable. A 3.7 tells you nothing to fix. |
| Why is the judge split 20/40/40? | "Training" is only a few few-shot examples; most labelled data should go to measurement. Test is touched once. |
| Why TPR + TNR, not accuracy? | On an imbalanced set, a judge that always says "pass" scores high on raw agreement and catches nothing. |
| pass@k vs pass^k? | @k: any of k succeeds, rises with k, for retryable work. ^k: all k succeed, falls with k, for unsupervised/reliability-critical work. |
| The flywheel? | Traces → evals → harness updates → better agent → traces. |

---

> **The through-line:** the harness decides what the model can do, the context decides how well it does it, and the evals are the only way you find out. Change any one and the other two shift.
