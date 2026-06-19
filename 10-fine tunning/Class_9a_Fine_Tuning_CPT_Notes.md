# Class 9a — Fine-Tuning & Continued Pre-Training (CPT)

> **One-line summary:** Fine-tuning is how we change a model *itself* (not just talk to it). This class covers *why* we do it, the clever trick (LoRA) that makes it cheap, and a hands-on project where we teach a tiny model to "speak finance" using real company reports.

---

## How to read these notes

Each section matches one idea from the slides. The second half ("The Notebook, Cell by Cell") walks through the actual code so you understand what every block does. Everything is in plain English — you don't need a machine-learning background to follow it.

---

# PART 0 — The Big Picture

## The journey so far

There are five different ways to make a language model useful. They build on each other:

| Stage | What it does | Simple meaning |
|-------|-------------|----------------|
| **Pre-training** | The model learns language | It reads the whole internet and learns how words fit together |
| **Prompting** | We talk to the model | We give it instructions in plain text |
| **RAG** | We give it external knowledge | We hand it documents to look things up in |
| **Agents** | We give it tools | It can use a calculator, search, call APIs |
| **Fine-tuning** | **We change the model itself** | We actually rewire its internal "brain" |

The first four leave the model unchanged — we're working *around* it. **Fine-tuning is the only one that edits the model's internals.** That's what this class is about.

---

## What is fine-tuning?

Think of pre-training and fine-tuning as two very different scales of the same process.

| | Pre-training | Fine-tuning |
|---|---|---|
| **Goal** | Learn language from the internet | Adapt that knowledge for *your* specific need |
| **Data size** | Trillions of words | Thousands of words |
| **Time** | Months | Hours |
| **Cost** | Enormous ($$$$$) | Small ($) |

**The key idea:** it's the *same mechanism*, just a different scale and different data.

> **Analogy:** A university graduate already knows *how to learn*. You're not teaching them from scratch — you're training them for one specific job. Fine-tuning is exactly that: you take a model that already knows English and train it for your particular task.

---

## Why fine-tune?

Fine-tuning is the right tool only in specific situations. Here's when the *other* tools fall short:

**Prompting (just giving instructions) doesn't work when:**
- The model doesn't know your industry's special language.
- You need the same style or format *every single time*, reliably.
- You need a smaller, faster model for production (prompting can't shrink a model).
- You want behaviour that no prompt can force.

**RAG (handing it documents) doesn't work when:**
- The problem isn't *missing facts* — it's *wrong behaviour*.
- Speed matters (RAG adds a slow "look it up first" step).
- The model can't even understand the documents you give it.

> **The crucial distinction:**
> **RAG adds knowledge. Fine-tuning changes behaviour and style.**
> They solve different problems. The best real-world systems use *both* together.

---

## The decision framework

A simple way to pick the right tool based on what's actually wrong:

| The problem | The fix |
|-------------|---------|
| "Model doesn't **KNOW** something" | RAG or **CPT** (Continued Pre-Training) |
| "Model doesn't **DO** something well" | **SFT** (Supervised Fine-Tuning) |
| "Model does it, but not the way humans want" | **RLHF / DPO** (preference training) |
| "Model needs to **reason** better" | **RLVR** (reward-based reasoning training) |
| Best systems | Combine all of them |

Don't worry about memorising the acronyms yet — they're explained next.

---

## The full post-training pipeline

After a model is first built ("the base model"), there's a whole sequence of optional training steps you can apply. Here's the map:

```
Base Model (a plain "next-word predictor")
   │
   ├── CPT   (Continued Pre-Training)   ← feed it raw text, same goal
   │
   ├── SFT   (Supervised Fine-Tuning)   ← feed it question→answer pairs
   │
   ├── RLHF  (RL from Human Feedback)   ← teach it human preferences
   │     └── DPO (Direct Preference Optimisation) ← a simpler version of RLHF
   │
   └── RLVR  (RL with Verifiable Rewards) ← for math / code / logic
```

- **This class (9a):** CPT
- **Next (9b):** SFT
- **After (9c):** RLHF / DPO → RLVR

A handy three-word summary of the whole journey: **what the model knows → what it does → what it values.**

---

# PART 1 — What's Actually Happening Inside

## What's inside a model?

A modern language model (a "transformer") is just a tall stack of identical layers. Each layer is made of two blocks:

```
┌─────────────────────────┐     ┌─────────────────────────┐
│   ATTENTION BLOCK        │     │   FFN BLOCK              │
│                          │     │                          │
│   Q_proj  (W_q)          │     │   gate_proj (W_g)        │
│   K_proj  (W_k)          │     │   up_proj   (W_up)       │
│   V_proj  (W_v)          │     │   down_proj (W_dn)       │
│   O_proj  (W_o)          │     │                          │
│                          │     │  "What to DO with        │
│ "WHICH tokens to pay     │     │   the information"       │
│  attention to"           │     │                          │
└─────────────────────────┘     └─────────────────────────┘

       + embed_tokens (the input layer)
       + lm_head      (the output layer)
```

- **Attention block** decides *which words in the sentence matter* for understanding each word.
- **FFN block** ("feed-forward network") decides *what to do* with that information. This is where most factual knowledge is stored.
- **embed_tokens** turns a word into numbers (the input).
- **lm_head** turns numbers back into the next word (the output).

> **The one thing to remember:** every one of these is just a **weight matrix** — a giant grid of numbers. "Training" means nudging those numbers. That's all a model really is.

---

## Full fine-tuning — the naive approach

The most obvious way to fine-tune is to update *every single number* in *every matrix*. This is called **full fine-tuning**, and it works like this:

| Step | What happens |
|------|--------------|
| **Forward** | Feed input in → model makes a prediction |
| **Loss** | Measure how wrong the prediction was |
| **Backward** | Work out how every number should change to be less wrong |
| **Update** | Nudge every number slightly to reduce the error |

It's simple and effective — the "obvious" approach. But it has two big problems, covered next.

---

## Problem 1: The memory math (why full fine-tuning breaks)

To *train* a model, your GPU has to hold several copies of it in memory at once:

| Item | Memory needed | Why |
|------|---------------|-----|
| Model weights | 1× model size | The numbers themselves |
| Gradients | 1× model size | One "how to change me" value per number |
| Optimizer states | 2× model size | The optimiser (AdamW) keeps two extra values per number |
| **Total** | **~4× model size (minimum!)** | And with activations + data, often **6–8×** |

Now plug in real models (using fp16, where each number takes 2 bytes):

| Model | Size on disk | Memory needed to train |
|-------|--------------|------------------------|
| SmolLM-135M | 270 MB | ~1–2 GB ✓ (easy) |
| LLaMA-3 8B | 16 GB | ~64–128 GB ⚠️ (hard) |
| LLaMA-3 70B | 140 GB | ~560+ GB ✗ (impossible on normal hardware) |

For reference: a top-end **A100 GPU has 80 GB**. Your **laptop has 8–16 GB**. So full fine-tuning of large models is simply out of reach for most people.

---

## Problem 2: The quality problems

Even if you *had* the memory, full fine-tuning causes trouble:

- **Overfitting.** Imagine training 8 *billion* parameters on just 1,000 documents (~5 million words). With so few examples and so many knobs, the model *memorises* the documents instead of *learning* from them.
- **Catastrophic forgetting.** Because *every* weight changes, the model's general knowledge gets overwritten. There's no "safe zone" of preserved skills. (Much more on this in Part 2.)
- **Storage & serving.** Each fine-tuned version is a *full copy* of all the weights. Five custom versions of LLaMA-70B = 5 × 140 GB = **700 GB** of storage.

---

## We need a better approach

A wishlist for a smarter method:

- ✓ Train on a single GPU (or even a laptop)
- ✓ Don't overfit on small datasets
- ✓ Preserve the model's general knowledge
- ✓ Store just the *changes*, not the whole model
- ✓ No slowdown when actually using the model

> **The big question:** What if we could **freeze the original model** and only train a *tiny* set of brand-new parameters? That idea is called **LoRA**.

---

## LoRA — the key insight

LoRA stands for **Low-Rank Adaptation** (from a 2021 paper by Hu et al.). The insight that started it all:

> *"When you fine-tune a large model, the **changes** to the weights have very low intrinsic dimensionality."*

In plain English:
- When you adapt a model, the *adjustment* you make (call it **ΔW**, "delta W" = the change) turns out to be highly repetitive.
- Most of that adjustment is redundant.
- So you can represent the *same* adjustment with **far fewer numbers**.

> **Analogy:** A photo of a clear blue sky is millions of pixels, but they're almost all the same colour. You don't need to store every pixel uniquely. Weight updates are compressible in exactly the same way.

---

## What does "low-rank" mean? (intuition)

"Rank" is a measure of how much *unique* information is in a grid of numbers.

| Rank | What it means |
|------|---------------|
| Rank 4096 | A full, complex matrix — 4096 × 4096 = 16.7 million unique numbers |
| Rank 1 | Just one pattern, repeated and scaled |
| Rank 32 | 32 independent patterns combined |

The surprising discovery: **fine-tuning adjustments are usually about "rank 32"**, even for a huge 4096 × 4096 matrix. That means the change you actually need is *tiny* compared to the full matrix.

> **Analogy — the choir:** Picture a choir of 100 singers. But only ~4 are singing genuinely unique parts; the other 96 are just harmonising or doubling those 4. To capture the whole sound, you only need to record the 4 unique voices. LoRA records just those few "unique voices" of the update.

---

## LoRA — the mechanism

Here's the actual trick, step by step.

Take one weight matrix **W** of size 4096 × 4096 = **16.7 million numbers**.

Instead of learning the full change **ΔW** directly, LoRA splits it into two skinny matrices:

```
ΔW  =  B  ×  A

  A : (4096 × 32)  = 131K numbers   "squeeze down to rank 32"
  B : (32 × 4096)  = 131K numbers   "expand back up to full size"

  Total: 262K numbers  vs  16.7M   →  ~64× fewer!
```

- **W stays frozen** (never changes).
- Only the tiny **A** and **B** get trained.
- The final output is: `W_frozen · x  +  (α/r) · B · A · x`
  (where `x` is the input, and `α/r` is just a scaling dial, explained later).

> **In one line:** Freeze the original. Train only the tiny add-on matrices ("adapters").

---

## LoRA visually

```
                      Input  x
                         │
          ┌──────────────┴──────────────┐
          │                             │
   ┌──────▼───────┐            ┌─────────▼─────────┐
   │ W_frozen × x │            │  A × x → B × (Ax) │
   │ (NO learning)│            │ (this part learns)│
   └──────┬───────┘            └─────────┬─────────┘
          │                              │
   original output                  LoRA output
          │                              │
          └──────────────┬───────────────┘
                         │
                         ▼
        Final = original + (α/r) · LoRA output
```

Two paths run in parallel: the frozen original (left, learns nothing) and the small LoRA add-on (right, does all the learning). Their outputs are added together.

> **Bonus:** After training, you can *merge* the adapter back in: `W_new = W + (α/r)·BA`. The result is one normal matrix again — so there's **zero extra cost** when you actually use the model.

---

## The memory savings (concrete)

Same model, two methods, dramatically different memory:

| | Full Fine-Tuning (LLaMA-3 8B) | LoRA rank 32 (LLaMA-3 8B) |
|---|---|---|
| Frozen weights | 16 GB | 16 GB → squeezed to 4-bit → **4 GB** |
| Trainable params | 16 GB | ~40 MB |
| Gradients | 16 GB | ~40 MB |
| Optimizer | 32 GB | ~80 MB |
| **Total** | **~64+ GB** (needs an A100) | **~4.2 GB** (fits on a laptop!) |

And the saved file is tiny:
- A LoRA **adapter file is ~80 MB** vs **16 GB** for a full model copy.
- 5 customers, 5 fine-tunes? That's **400 MB of adapters** all sharing one base model — instead of **80 GB** of full copies.

---

## LoRA hyperparameters (the dials you set)

| Setting | What it controls | Typical value |
|---------|------------------|---------------|
| **r (rank)** | Size of the "bottleneck" — bigger = more capacity but more memory | 8, 16, 32, or 64 |
| **α (alpha)** | A scaling dial for how strongly LoRA affects the output | Rule of thumb: set α = r (so scaling = 1) |
| **target_modules** | *Which* weight matrices get a LoRA add-on | Minimum: `q_proj`, `v_proj`. Recommended: all attention + all FFN. **For CPT, also add `embed_tokens`, `lm_head`** |
| **dropout** | A regularisation trick | Usually 0 (Unsloth is optimised for this) |

---

## Why CPT needs embed_tokens and lm_head

This is a CPT-specific detail that trips people up. Recall:
- **embed_tokens**: turns a word-ID into numbers → answers *"what does this word mean?"*
- **lm_head**: turns numbers back into a word → answers *"what word should come next?"*

| SFT (instruction tuning) | CPT (learning a new domain) |
|--------------------------|-----------------------------|
| The model already knows the vocabulary | The model is learning **new domain words** |
| You're just teaching it to *use* words differently | e.g. `EBITDA`, `10-K`, `diluted EPS`, `subordinated debentures` |
| → No need to touch the word layers | → **Must** update `embed_tokens` + `lm_head` |

> **The most common CPT mistake:** copying an SFT LoRA config (which skips the word layers) and then wondering why the model never learns the new domain. For CPT, *include the word layers*.

---

## Quantization — loading in 4-bit

"Quantization" means storing each number using fewer bits, to save memory. Fewer bits = smaller model, with a small accuracy trade-off.

| Format | Bits per number | Size of SmolLM-135M |
|--------|-----------------|---------------------|
| Float32 | 32 bits | 540 MB |
| Float16 | 16 bits | 270 MB |
| Int8 | 8 bits | 135 MB |
| **NF4** | **4 bits** | **67 MB** |

> **QLoRA = Quantized base (4-bit) + LoRA adapters (16-bit).**
> The big *frozen* part is compressed down to 4-bit to save memory. The tiny LoRA adapters that actually learn are kept in full precision so the learning stays accurate.

**NF4** ("4-bit NormalFloat") is specially designed for the way neural-network numbers are distributed, so it loses almost no quality.

---

## What Unsloth actually does

**Unsloth** is the training library used in the notebook. Important to understand:

> Unsloth is **NOT a different training method**. It's the **same math, made faster.**

**What it does:**
- Custom low-level GPU code for attention + LoRA (≈2× faster)
- Combines operations together (fewer slow trips to memory)
- Smart "gradient checkpointing" (≈50% less memory — explained next)
- Automatically uses efficient number formats
- Packs data tightly (no wasted space)

**What it does NOT do:**
- Change the training goal
- Change the model's design
- Do anything you couldn't already do with the standard tools (HuggingFace + PEFT + TRL)

> **Analogy:** Unsloth is like NumPy vs. writing matrix multiplication by hand in plain Python. Identical math, wildly different speed. If Unsloth disappeared tomorrow, you'd just use the slower standard tools.

---

## Gradient checkpointing (how Unsloth saves ~50% memory)

During training, the model normally saves the output of *every* layer so it can use them later in the "backward" step. For a deep model, that eats a lot of memory.

```
NORMAL TRAINING (store everything)      GRADIENT CHECKPOINTING (store some)
  Layer 1 → [STORED]                       Layer 1 → [STORED]
  Layer 2 → [STORED]                       Layer 2 → [recompute later]
  Layer 3 → [STORED]                       Layer 3 → [STORED]
  ...                                      Layer 4 → [recompute later]
  Layer 24 → [STORED]                      ...
  (memory grows with depth!)               (store some, recompute the rest)
```

**The trade-off:** training is ~30% slower, but uses ~50% less memory. For most people, fitting on the GPU at all is worth a little slowness. Unsloth is just smarter about *which* layers to checkpoint.

> This is exactly what `use_gradient_checkpointing="unsloth"` turns on in the notebook config.

---

## The training loop internals

Every kind of training — pre-training, CPT, and SFT — runs the *same six steps* in a loop:

1. **Tokenize** — turn text into token IDs (numbers)
2. **Forward** — push the tokens through the model → get predictions
3. **Loss** — measure the gap between the predicted next word and the actual next word (called "cross-entropy")
4. **Backward** — compute how to change the weights (only the LoRA ones!)
5. **Update** — the optimiser (AdamW) nudges the LoRA weights
6. **Repeat**

> **The only difference between pre-training, CPT, and SFT is the DATA.** The loop itself is identical.

---

## Key training concepts (the supporting tricks)

| Concept | What it means |
|---------|---------------|
| **Packing** | Glue short texts together into one long sequence instead of wasting space with padding. `[text1 | text2 | text3]` instead of `[text1 + PAD PAD PAD]`. |
| **Gradient accumulation** | Fake a bigger batch size. Batch of 32 with accumulation 2 acts like a batch of 64 — useful when memory is tight. |
| **Warmup** | Start with a tiny learning rate and ramp it up. Stops the early, clumsy steps from wrecking the weights. |
| **Early stopping** | Watch the validation error; stop the moment it stops improving. Prevents overfitting on small datasets. |

---

# PART 2 — The Enemy: Catastrophic Forgetting

## What happens when you only train on domain data

This is the danger CPT must guard against. Suppose you train *only* on finance text:

```
BEFORE CPT:
  "The cat sat on the"       →  "mat"            ✓ good general English
  "Revenue increased by"     →  random garbage   ✗ no finance knowledge

AFTER CPT (trained on finance only):
  "Revenue increased by"     →  "12% year over year"  ✓ finance learned!
  "The cat sat on the"       →  "balance sheet"        ✗ forgot English!
```

The model got great at finance — but the weights that used to handle everyday English got *overwritten*. That's **catastrophic forgetting**.

---

## Why forgetting happens (the mechanics)

The cause is simple once you see it:

> **Gradient updates are GREEDY.** At each step, the optimiser *only* cares about doing better on the *current* batch of data.

So if every single batch is finance documents:
- The updates always push toward "be better at finance."
- The updates *never* push toward "stay good at everyday text."
- General knowledge slowly erodes, batch after batch.

> The model doesn't *decide* to forget. The optimiser simply never receives a signal telling it to *remember* the old stuff.

---

## Solutions to forgetting

Five practical fixes (you usually combine several):

| Fix | How it works |
|-----|--------------|
| **1. Data mixing** | Train on ~80% domain + ~20% general text, so the optimiser keeps getting a "stay good at English" signal |
| **2. Low learning rate** | Don't move the weights too far from the original (≈1e-5 for full tuning, ≈2e-4 for LoRA) |
| **3. LoRA itself** | Freezing the base weights *is* forgetting-prevention — the original knowledge literally cannot be overwritten |
| **4. Short training** | Watch the validation error; the moment it flattens, stop |
| **5. Lower embedding LR** | Give `embed_tokens`/`lm_head` a much smaller learning rate (≈10% of the main rate), because those layers affect *every* token, not just domain ones |

---

# PART 3 — Hands-On: CPT with SEC 10-K Filings

## The plan

The project teaches a tiny model (SmolLM-135M) to speak "finance." The steps:

1. Load SmolLM-135M → try financial text → **watch it fail**
2. Get real SEC 10-K filings from HuggingFace (the `PleIAs/SEC` dataset)
3. Clean & chunk the data
4. Run CPT with Unsloth + LoRA
5. Try financial text again → **see the improvement**
6. Measure: perplexity before vs after
7. Experiment: with vs without data mixing (the forgetting test)

---

## What are 10-K filings?

A **10-K** is the big annual report that every public US company must file with the SEC (the US financial regulator). Each one contains:
- Business description
- Risk factors
- Financial statements
- Management discussion & analysis (MD&A)

The language is dense, formal, and jargon-heavy. For example:

> *"The Company recognized impairment charges of $142 million related to goodwill associated with the North America reporting unit during the fiscal year ended March 31, 2024."*

This is perfect CPT material precisely because **SmolLM has never seen much text like this** — so there's lots for it to learn.

---

## Data preparation

Raw filings are messy and huge, so they get cleaned and cut into bite-sized pieces:

```
Raw 10-K filing
   │
   ├── Strip boilerplate (legal disclaimers, page numbers)
   ├── Remove reference / exhibit sections
   ├── Cut into chunks (256 words each, 20% overlap)
   │
   ▼
JSONL file:  {"text": "a chunk of financial text..."}

Then split: hold out ~50 filings for evaluation (validation set)
```

- **Why chunk?** Filings can be 10,000+ words — far too long for the model's context window. We cut them into 256-word pieces.
- **Why overlap?** So a sentence isn't sliced in half at a chunk boundary and lose its meaning.
- **JSONL** is just the file format the trainer expects: one text record per line.

---

## The training config (explained)

The most important settings used for this CPT run:

```python
# --- LoRA config ---
r = 32                          # higher rank, because CPT teaches NEW knowledge
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention matrices
    "gate_proj", "up_proj", "down_proj",      # FFN (where knowledge lives!)
    "embed_tokens", "lm_head"                 # CPT-specific: new vocabulary
]
lora_alpha = 32                 # = rank, so scaling factor = 1

# --- Training ---
learning_rate = 2e-4            # standard for LoRA
embedding_learning_rate = 2e-5  # 10× LOWER for the word layers (anti-forgetting)
packing = True                  # no wasted padding
warmup_steps = 100              # gentle start
```

> **Key point:** target *all* the linear layers, including the FFN — that's where factual knowledge is stored. And use a *much smaller* learning rate for the word-embedding layers, since they touch every token.

---

## Training — what to watch

Reading the training output like a pro:

**Good signs:**
- Training error steadily going down
- Validation error tracking the training error (not splitting away from it)
- Validation error flattening out → time to stop

**Bad signs:**
- Validation error going *up* while training error drops → **overfitting**
- Training error suddenly spiking → learning rate too high
- Error stuck and not moving → learning rate too low, or too little data

> **Perplexity = e^(loss).** Lower is better. It measures how "surprised" the model is by the text. Low perplexity on finance text means the model now finds finance language natural.
>
> **The validation error is your north star.**

---

## Results comparison

What actually happens after CPT, on the prompt *"The company reported total revenue of"*:

**Base SmolLM (before):**
> *"...the most important thing is to be able to get a good understanding of the world"* — generic, not financial at all.

**After CPT:**
> *"...approximately $4.2 billion for the fiscal year ended December 31, 2023, representing an increase of 12%"* — fluent financial language!

**Perplexity on the SEC test set:**

| Model | Perplexity (lower = better) |
|-------|------------------------------|
| Base | ~180 |
| After CPT | ~45 |

That's roughly **4× lower perplexity** — the model went from confused to fluent in financial language.

---

## NEFTune — a subtle bonus trick

**NEFTune** is an optional regularisation trick: during training, add a little random noise to the word embeddings.

```
input_embeds = input_embeds + α * random_noise
```

**Why it helps:**
- Acts like "dropout for embeddings" — a gentle randomisation that prevents over-memorising
- Stops the model from clinging to *exact* word sequences
- Reliably improves results, especially on small datasets

> **Note:** NEFTune is mainly for *full* training. With LoRA you usually skip it, because LoRA already regularises by limiting how much can change.

---

## The forgetting experiment

The notebook runs the same training twice to prove the forgetting point:

| | Run A | Run B |
|---|---|---|
| **Data** | 100% SEC filings (no general text) | 80% SEC + 20% general text |
| **Expected on finance** | Great | Slightly worse |
| **Expected on general English** | Degraded (it forgot!) | Maintained |

Both runs are evaluated on a finance test set *and* a general-English test set. This directly demonstrates the **mixing trade-off**: pure-domain training is best at the domain but forgets everything else; mixed training sacrifices a little domain skill to keep its general ability.

---

# PART 4 — The Limit of CPT

## CPT made it smarter, not helpful

Here's the catch that motivates the *next* class. Ask the CPT'd model a direct question:

> **Prompt:** *"What was Apple's revenue in fiscal year 2023?"*
>
> **After CPT:** *"The company's revenue was $383.3 billion for the fiscal year ended September 30, 2023, representing a decrease of 2.8% compared to the prior fiscal year revenue of $394.3..."*

Notice: it **continued the text** in fluent finance-speak — but it **didn't actually answer the question** as an assistant would. It just kept "autocompleting."

> **The core distinction:**
> **CPT** teaches a model to **SOUND** like a domain.
> **SFT** teaches a model to **RESPOND** to instructions.

CPT alone gives you a fluent autocompleter, not a helpful assistant.

---

## What's coming next

- **Next session (9b) — SFT (Supervised Fine-Tuning):**
  - Instruction → response data format (chat templates)
  - Training on financial Q&A pairs
  - The model goes from "autocompleter" to "assistant"
  - Merging adapters for deployment

- **After that (9c) — RLHF, DPO & RLVR:**
  - Aligning the model with human preferences
  - Direct Preference Optimisation
  - Verifiable rewards for reasoning

> **The arc:** Today = *what the model knows.* Next = *what it does.* Then = *what it values.*

---

# KEY TAKEAWAYS

1. **Fine-tuning modifies the model itself** — unlike prompting or RAG, which leave it unchanged.
2. **LoRA** = freeze the base model, train tiny add-on matrices (~64× fewer parameters).
3. **CPT** = the same training goal as pre-training, just on different (domain) data.
4. **For CPT specifically**, include `embed_tokens` + `lm_head` in your LoRA targets.
5. **Catastrophic forgetting is real** — mix in general data to prevent it.
6. **Unsloth** = the same math, just executed faster.
7. **CPT teaches domain language, not instruction-following** — that's SFT's job.

> **The whole class in one sentence:** Fine-tuning is not magic — it's just gradient descent (the same learning loop) run on different data.

---
---

# THE NOTEBOOK, CELL BY CELL

This section explains the actual code file (`class_9a_f.ipynb`). The whole notebook teaches SmolLM-135M to speak finance, then proves it worked and shows its limits. Each cell is explained in plain English.

---

## Part 0: Setup

**Cell 0 (intro):** A markdown title cell listing the plan — load the model, prepare SEC data, run CPT, compare before/after, test forgetting, then tease SFT. It notes the project runs on a free Google Colab T4 GPU.

**Cell 2 — Install libraries:**
```python
!pip install -q unsloth
!pip install -q datasets evaluate rouge_score
```
Installs **Unsloth** (the fast training library that handles LoRA + quantization for us) plus helper libraries for loading datasets and scoring. The rest of the cell silences noisy warning messages so the output stays readable.

**Cell 3 — Imports and GPU check:**
```python
import torch, json, random, math, re, os
print(f"CUDA available: {torch.cuda.is_available()}")
```
Loads the standard tools (`torch` is the deep-learning engine; `re` is for text cleaning; `math` for the perplexity formula). The print confirms a GPU is actually available — if this says `False`, training would be painfully slow.

---

## Part 1: Meet the base model

**Cell 5 — Load SmolLM-135M:**
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="HuggingFaceTB/SmolLM-135M",
    max_seq_length=512,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
```
Downloads the base model and its **tokenizer** (the part that converts text ↔ numbers). `load_in_4bit=True` loads it in compressed 4-bit form to save memory (the quantization idea from the slides). `for_inference` puts it in "just generate text" mode — no training yet. The print shows it has ~135 million parameters.

**Cell 6 — A text-generation helper:**
```python
def generate_text(model, tokenizer, prompt, max_new_tokens=100):
    ...
    outputs = model.generate(..., temperature=0.7, top_p=0.9, repetition_penalty=1.2)
    new_tokens = outputs[0, inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)
```
A reusable function that feeds a prompt to the model and returns its completion. The settings (`temperature`, `top_p`) add a little randomness so the output sounds natural rather than robotic; `repetition_penalty` discourages it from repeating itself. The slicing keeps only the *new* words the model wrote, dropping the original prompt.

**Cell 7 — Test the base model on finance:**
```python
financial_prompts = ["The company reported total revenue of", ...]
for prompt in financial_prompts:
    completion = generate_text(model, tokenizer, prompt, max_new_tokens=60)
```
Feeds five finance-flavoured prompts to the *untrained* model. This is the "watch it fail" step — the output will be generic English, not real financial language. It sets up the "before" picture.

**Cell 9 — Define perplexity:**
```python
def compute_perplexity(model, tokenizer, texts, max_length=512):
    # Perplexity = e^(average loss). Lower = less surprised = better fit.
    ...
    return math.exp(avg_loss)
```
Defines **the key measuring stick for CPT**. It feeds real financial text to the model and measures how "surprised" it is. A high number means "this text looks alien to me"; a low number means "this feels natural." We'll compare this before vs after training.

**Cell 10 — Baseline perplexity:**
```python
base_ppl = compute_perplexity(model, tokenizer, sample_financial)
print(f"Base model perplexity on financial text: {base_ppl:.1f}")
```
Runs that measure on three sample financial sentences. Expect a *high* number here — the base model doesn't know finance yet. This is the baseline we want to beat.

---

## Part 2: Prepare the data

**Cell 12 — Download SEC filings:**
```python
sec_dataset = load_dataset("PleIAs/SEC", split="train", streaming=True)
NUM_TRAIN, NUM_VAL = 80, 20
for example in sec_dataset:
    if text and len(text.split()) > 1000:
        all_texts.append(text)
    if len(all_texts) >= 100: break
```
Pulls real SEC 10-K filings from HuggingFace. `streaming=True` means it reads them one at a time instead of downloading the entire 100+ GB dataset. It collects 100 substantial filings (only keeping ones longer than 1,000 words, to skip junk).

**Cell 13 — Inspect a filing:**
```python
sample = all_texts[0]
print(" ".join(words[:200]))   # first 200 words
print(" ".join(words[-100:]))  # last 100 words
```
Just prints the start and end of one filing so you can see what the raw data looks like — and notice the boilerplate junk at the end (signatures, exhibits) that the next cell strips out.

**Cell 15 — Clean the filings:**
```python
def clean_sec_filing(text):
    end_markers = [r"(?i)\bEXHIBIT\s+INDEX\b", r"(?i)\bSIGNATURES\b", ...]
    # cut off trailing exhibits/signatures, collapse extra whitespace
    ...
```
Removes the non-useful tail sections (exhibits, signatures, legal boilerplate) and tidies up whitespace. It only chops these if they appear near the *end* (past 70% of the document), so it doesn't accidentally delete real content. The print shows what percentage was removed from the first few filings.

**Cell 17 — Chunk the text:**
```python
def chunk_texts(texts, chunk_size=256, overlap=0.2):
    step = int(chunk_size * (1 - overlap))
    ...
# Split into train/val at the FILING level (not chunk level) to avoid leakage
random.shuffle(cleaned_texts)
train_filings = cleaned_texts[:80]
val_filings   = cleaned_texts[80:100]
train_chunks = chunk_texts(train_filings, ...)
val_chunks   = chunk_texts(val_filings, ...)
```
Cuts each long filing into 256-word chunks with 20% overlap (so sentences aren't sliced apart). **Important detail:** it splits training vs validation data at the *filing* level *before* chunking. This prevents "data leakage" — you never want chunks from the same document to appear in both the training set and the test set, or your evaluation would be falsely optimistic.

**Cell 18 — Convert to Dataset objects:**
```python
train_dataset = Dataset.from_dict({"text": train_chunks})
val_dataset   = Dataset.from_dict({"text": val_chunks})
```
Wraps the chunk lists in the HuggingFace `Dataset` format that the trainer expects.

**Cell 19 — Measure baseline perplexity on the real validation chunks:**
```python
val_sample = val_chunks[:50]
base_ppl_val = compute_perplexity(model, tokenizer, val_sample)
```
Records the base model's perplexity on the actual validation data **before** training. This is the "before" number we'll compare against after CPT. It has to be measured now, while the original model is still loaded.

---

## Part 3: CPT with Unsloth + LoRA

**Cell 21 — Reload a fresh model for training:**
```python
del model
torch.cuda.empty_cache()
model, tokenizer = FastLanguageModel.from_pretrained(...)
```
Deletes the inference model to free GPU memory, then loads a clean copy for training. (The inference-mode model from earlier isn't set up for learning, so we start fresh.)

**Cell 22 — Add LoRA adapters:**
```python
model = FastLanguageModel.get_peft_model(
    model, r=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj",
                    "embed_tokens","lm_head"],   # ← CPT-specific!
    lora_alpha=32, lora_dropout=0,
    use_gradient_checkpointing="unsloth",
)
```
This is the heart of LoRA. It freezes the whole base model and attaches small trainable adapters to the listed matrices. Note it targets **all** attention + FFN layers **plus `embed_tokens` and `lm_head`** — exactly the CPT requirement from the slides (the model needs to learn new financial vocabulary). The print then shows that only a *tiny percentage* of parameters are actually trainable — the frozen rest is your forgetting protection.

**Cell 24 — Configure the trainer:**
```python
training_args = UnslothTrainingArguments(
    num_train_epochs=2,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,     # effective batch = 32
    learning_rate=2e-4,
    embedding_learning_rate=2e-5,      # 10× lower for word layers
    warmup_steps=50,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",                # memory-saving optimiser
    packing=True,                      # no wasted padding
    eval_strategy="steps", eval_steps=25,
    load_best_model_at_end=True, metric_for_best_model="eval_loss",
)
```
Sets every training dial. The standout choices (straight from the slides): a lower learning rate for the embeddings (anti-forgetting), `cosine` decay (smoothly winds the learning rate down), `packing` (efficient data), and evaluating every 25 steps so we can watch the validation error and keep the best version.

**Cell 25 — Train:**
```python
trainer_stats = trainer.train()
```
Runs the actual training loop (~8–12 minutes on a T4). While it runs, you watch the `eval_loss` — it should go down and then flatten out.

---

## Part 4: Compare before vs after

**Cell 27 — Generate finance text again:**
```python
FastLanguageModel.for_inference(model)
for prompt in financial_prompts:
    completion = generate_text(model, tokenizer, prompt, max_new_tokens=60)
```
Switches back to generation mode and runs the *exact same* five finance prompts from Cell 7. Now the outputs should sound genuinely financial — the visible "after" win.

**Cell 28 — Perplexity comparison:**
```python
cpt_ppl = compute_perplexity(model, tokenizer, val_sample)
print(f"Base: {base_ppl_val:.1f}   After CPT: {cpt_ppl:.1f}")
```
The hard, numeric proof. It compares perplexity before vs after on the *same* validation chunks. A big drop (e.g. ~180 → ~45) means the model genuinely learned the domain, not just got lucky on a couple of prompts.

---

## Part 5: The forgetting experiment

**Cell 30 — Generate general English:**
```python
general_prompts = ["The cat sat on the", "Once upon a time there was a", ...]
```
Tests the CPT'd model on *everyday* (non-finance) prompts. If it now produces weird, finance-flavoured nonsense, that's catastrophic forgetting showing up.

**Cell 31 — Perplexity on general text:**
```python
general_ppl_after_cpt = compute_perplexity(model, tokenizer, general_texts)
```
Measures how surprised the finance-trained model now is by ordinary English. If this number is high, the model has partially forgotten general English. (LoRA softens this because the base weights are frozen — but it's not a perfect cure.)

---

## Part 6: CPT with data mixing (anti-forgetting)

**Cell 33 — Fresh model again:**
```python
del model, trainer
torch.cuda.empty_cache()
model, tokenizer = FastLanguageModel.from_pretrained(...)
model = FastLanguageModel.get_peft_model(model, r=32, target_modules=[...], ...)
```
Starts over with a brand-new model and the same LoRA setup, so the mixing experiment is a fair comparison against the pure-finance run.

**Cell 34 — Load general text for mixing:**
```python
wiki_data = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
wiki_texts = [t for t in wiki_data["text"] if t.strip() and len(t.split()) > 50]
num_general_chunks = len(train_chunks) // 4   # ~20% of the mix
general_chunks = chunk_texts(wiki_texts[:500], ...)[:num_general_chunks]
```
Downloads general English (Wikipedia text), filters out junk, and prepares enough chunks to make up about 20% of the training mix — implementing the "80% domain + 20% general" rule from the slides.

**Cell 35 — Combine and shuffle:**
```python
mixed_train = Dataset.from_dict({"text": train_chunks + general_chunks}).shuffle(seed=SEED)
```
Merges finance + general chunks and shuffles them so the two types are interleaved. This is what keeps reminding the optimiser to "stay good at English."

**Cell 36 — Train on the mixed data:**
```python
trainer_mix = UnslothTrainer(model=model, train_dataset=mixed_train, ...)
trainer_mix_stats = trainer_mix.train()
```
Runs the identical training config, but on the mixed dataset this time.

**Cell 37 — Compare all three:**
```python
mixed_sec_ppl     = compute_perplexity(model, tokenizer, val_sample)
mixed_general_ppl = compute_perplexity(model, tokenizer, general_texts)
# prints a Base / Pure CPT / Mixed CPT table
```
The payoff table. It compares finance perplexity and general-English perplexity across **Base**, **Pure CPT**, and **Mixed CPT**. The expected story: pure CPT is best at finance but worst at general English; mixed CPT gives up a little finance skill to *keep* its general ability. That trade-off is exactly why production systems almost always use mixing.

---

## Part 7: The SFT tease

**Cell 39 — Ask it real questions:**
```python
question_prompts = [
    "Question: What was Apple's total revenue in fiscal year 2023?\nAnswer:",
    "Question: What is EBITDA and why do investors care about it?\nAnswer:",
    ...
]
```
Asks the finance-fluent model direct questions. It will **continue the text** rather than **answer** — proving that CPT taught it financial *language*, not question-answering *behaviour*. This is the cliffhanger that motivates SFT in the next class.

---

## Part 8: Save the model

**Cell 41 — Save the adapter:**
```python
model.save_pretrained("./cpt_sec_adapter")
tokenizer.save_pretrained("./cpt_sec_adapter")
print(f"Adapter size: {adapter_size/1e6:.1f} MB   Full model would be: ~270 MB")
```
Saves only the small LoRA **adapter** — not the whole model. The print drives home the storage win: the adapter is a fraction of the full model's size, demonstrating the "store just the changes" benefit from the slides.

---

## Notebook summary

| Step | What it did | Why it mattered |
|------|-------------|-----------------|
| Base model test | Generated finance text with raw SmolLM | See it fail — the motivation for CPT |
| Data prep | Downloaded, cleaned, and chunked SEC 10-Ks | A real-world data pipeline |
| CPT training | LoRA (rank 32) with Unsloth on domain text | Teach the model financial language |
| Before/after | Compared generations + perplexity | Prove it actually worked |
| Forgetting test | Checked general English after CPT | Understand the trade-off |
| Data mixing | 80% domain + 20% general | The fix for catastrophic forgetting |
| SFT tease | Asked questions — it couldn't answer | Motivation for the next session |

> **Final takeaway:** CPT teaches a model to **SOUND** like a domain. It does **not** make it **HELPFUL**. Making it actually answer questions is the job of **SFT** — coming up in Class 9b, using a financial Q&A dataset (`virattt/financial-qa-10K`).
