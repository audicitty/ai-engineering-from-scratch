# Class 13 — Memory

### How AI systems remember you across sessions

**13 parts:** the problem · human memory · files · failures · OS pattern · layer · graph · time · forgetting · evaluation · production · voice · horizon

---

## 0. Primer (for anyone new to AI)

| Word | Plain meaning |
| --- | --- |
| **Context window** | Everything the model can see right now. Wiped when the session ends. |
| **Memory** | Anything that survives that wipe — facts about you carried into the next session. |
| **RAG** | Retrieval-Augmented Generation. Search a pile of documents for text similar to the question, paste the matches in. |
| **Embedding / vector store** | Turning text into numbers so "similar meaning" becomes "nearby numbers". The standard search machinery. |
| **grep** | Plain keyword search. Finds exact strings, understands no meaning. |
| **Consolidation** | Turning raw experience into a compact durable fact — what your brain does while you sleep. |
| **TTL** | Time-to-live. Automatic expiry after N days. |

> **One-line summary:** Storing what a user said is easy. Deciding *what is still true* is the hard part — and that is what memory means here.

---

# PART 1 — THE PROBLEM WE CAN'T YET NAME

## 1.1 A Scenario

```
   MAR   started PyTorch
   APR   switched to JAX
   MAY   back to PyTorch
```

**User now asks: "What framework should I use?"**

All three statements were true when said. Only one is true now. What should the system answer?

## 1.2 RAG versus Memory

```
   RAG — retrieves by similarity        MEMORY — tracks state over time
   ─────────────────────────────        ──────────────────────────────
   "What framework should I use?"       "What framework should I use?"
              ↓                                     ↓
   ┌─────────────────────────┐          ┌─────────────────────────────┐
   │ Stored as flat similarity│          │ Stored as state over time  │
   │ "Started learning PyTorch"│         │ Mar: started PyTorch        │
   │ "Switching to JAX"        │         │ Apr: switched to JAX        │
   │ "Back to PyTorch"         │         │ May: back to PyTorch (now)  │
   │                           │         │                             │
   │ Closest semantic match wins│        │ Latest state wins           │
   └─────────────────────────┘          └─────────────────────────────┘
              ↓                                     ↓
      "Start with PyTorch"                  "Stick with PyTorch"
    most-similar memory returned          current state with history
```

> **RAG retrieves by similarity. Memory tracks state over time.**

The difference is subtle but total. RAG has no concept of "later" — all three sentences sit in the same pile with equal standing, and whichever *reads* most like the question wins. Memory knows May came after April.

## 1.3 Two Problems, One Word

```
   —  "What do I know."               ← knowledge / RAG
   —  "What do I remember about you." ← memory
   —  The boundary between them is fuzzy.
```

Both get called "memory" in product copy. They are different engineering problems.

---

# PART 2 — THREE TYPES OF MEMORY

## 2.1 The Three Types

```
   Episodic     — events    ("we met on Tuesday")
   Semantic     — facts     ("Paris is the capital of France")
   Procedural   — skills    ("how to ride a bike")
```

## 2.2 Human Memory Pipeline

```
              ┌──────────────────────────┐
              │  Continuous input stream │
              └────────────┬─────────────┘
                           ↓
              ┌──────────────────────────┐
              │      Sensory filter      │
              │  drop noise, keep salient│
              └────────────┬─────────────┘
                           ↓
   forgetting  ┌──────────────────────────────────┐
      ←────────│  Working / short-term memory     │←──────┐
              │  active reasoning, secs to minutes│       │
              └────────────┬─────────────────────┘        │
                           ↓ if reinforced         consolidation
              ┌───────────────────────────────────────────┴──┐
              │            Long-term memory                  │
              │  ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
              │  │ Episodic │ │ Semantic │ │  Procedural  │  │
              │  │ events   │ │ facts &  │ │ skills and   │  │
              │  │ w/ time  │ │ concepts │ │ habits       │  │
              │  │"what     │ │"what is  │ │"how to do"   │  │
              │  │ happened"│ │ true"    │ │              │  │
              │  └──────────┘ └──────────┘ └──────────────┘  │
              └──────────────────────────────────────────────┘
```

**Why this diagram matters:** every architecture in the rest of the class is a partial copy of it. The sensory filter, the working/long-term split, consolidation, forgetting — each reappears as a design choice later.

---

# PART 3 — SHIP MEMORY MONDAY

## 3.1 The Solution: A File

```
   — Plain markdown.
   — Load at session start.
   — Save at session end.
```

No database. No embeddings. That's the whole design — and it's what three major companies actually shipped.

## 3.2 Files on Disk — Same Primitive, Three Companies

```
   Claude Code            Codex                  Hermes
   Anthropic              OpenAI                 Nous Research
   ─────────────────      ─────────────────      ─────────────────
   CLAUDE.md              AGENTS.md              MEMORY.md
   user-written,          layered discovery      world facts
   project root
   ─────────────────      ─────────────────      ─────────────────
   ~/.claude/…/memory/    ~/.codex/memories/     USER.md
   auto-generated         async consolidation    user facts

   · four memory types    · global+project+      · read whole into
   · Sonnet side-call       override               prompt
   · 200-line index cap   · grep over MEMORY.md  · 2,200 char cap
   · freshness warnings   · 32 KiB cap             (world)
                          · 30-day TTL           · 1,375 char cap
                                                   (user)
                                                 · frozen snapshot
```

> **Plain markdown. Load at session start. No embeddings.**

## 3.3 What They Don't Do

```
   — No embeddings.
   — No vector store.
   — Relevance via grep, or an LLM side-call.
```

**The lesson:** the industry default is not a vector database. It's a text file. Start there.

---

# PART 4 — WHERE THE FILE BREAKS

Three failure modes.

## 4.1 Failure 01 — The Cap

```
   ┌──────────────────────────────────────────────┐
   │             MEMORY.md  (on disk)             │
   │                                              │
   │  [001] User prefers Unsloth over vanilla HF  │ ┐
   │  [002] Base model: SmolLM-135M               │ │ loaded into
   │  [003] LoRA rank 32 with embed_tokens        │ │ system prompt
   │  [...] 195 more entries accumulated          │ │
   │  [200] Class 12 ran 1:20 hours, plan tighter │ ┘
   │ ─ ─ ─ ─ ─ ─ ─ cap: line 200 ─ ─ ─ ─ ─ ─ ─ ─  │
   │  [201] Course teaches AI/ML from scratch     │ ┐
   │  [202] Students complain when slides are read│ │ on disk,
   │  [203] Use OpenRouter across all classes     │ ┘ not loaded
   └──────────────────────────────────────────────┘
```

> **The hard ceiling — anything after line 200 stays on disk, unread.**

The file keeps growing; the loaded portion doesn't. Entry 203 might be the most important thing you ever told it, and it is invisible.

## 4.2 Failure 02 — Semantic Miss

```
   stored in MEMORY.md:
   ┌────────────────────────────────────────────┐
   │ bash scripts/cpt-train.sh configs/sec.yaml │
   └────────────────────────────────────────────┘
                       ↓
   user asks: "how do I start training?"
                  ↙            ↘
   ┌────────────────────┐  ┌────────────────────┐
   │  substring / grep  │  │  semantic search   │
   │  no shared keywords│  │  matches by meaning│
   │ ("train"≠"training")│ │  intent ≈ intent   │
   │                    │  │                    │
   │     no match       │  │       match        │
   └────────────────────┘  └────────────────────┘
```

Exact-match retrieval fails on paraphrase. The memory is *there* and still unreachable.

## 4.3 Failure 03 — Temporal & Portability

```
   — "Delhi" and "Bangalore" stored with equal weight — both still true.
   — File is local to one machine.
   — No team sharing.
```

The file has no notion of time, so a stale fact and a current fact look identical.

---

# PART 5 — BORROW FROM THE OS

## 5.1 The Idea

**What if the LLM managed its own memory?**

```
   — RAM is the context window.
   — Disk is the external store.
   — The LLM pages memory in and out.
```

## 5.2 Virtual Memory, Ported

```
   Operating system                    LLM agent (MemGPT / Letta)
   manages program memory              manages context window
   ──────────────────────              ──────────────────────────
   ┌──────────────────────┐            ┌──────────────────────────┐
   │        RAM           │            │    Context window        │
   │ active, fast, small  │            │ active, fast, small      │
   │      (~16 GB)        │            │     (200K tokens)        │
   └──────────────────────┘            └──────────────────────────┘
      ↓ page out  ↑ page in               ↓ write     ↑ recall
   ┌──────────────────────┐            ┌──────────────────────────┐
   │        Disk          │            │  External memory store   │
   │ durable, slow, big   │            │  recall + archival tiers │
   │      (~500 GB)       │            │                          │
   └──────────────────────┘            └──────────────────────────┘

   kernel decides what pages           LLM decides what pages
   programs see unified address space  model sees unified memory
```

> **Same pattern. Different substrate.**

The one real difference is in the last row: in an OS the *kernel* decides what to page. Here the *model itself* decides, by calling memory functions.

## 5.3 Three Tiers

```
   Core       — always loaded
   Recall     — searchable history
   Archival   — cold storage
```

## 5.4 The Lineage — MemGPT (2023) → Letta

```
   — The LLM calls memory functions directly.
   — Agent runtime, not a service.
   — Opinionated by design.
```

"Agent runtime, not a service" is the tradeoff: you adopt their whole agent loop, not a library you sprinkle into yours.

---

# PART 6 — MEMORY AS A LAYER

## 6.1 The Idea

**What if a separate system extracted facts?**

```
   — Memory layer alongside the conversation.
   — Watches every turn.
   — Extracts facts. Stores them.
```

Unlike Part 5, the model doesn't manage its own memory — a separate system watches and decides.

## 6.2 The Delhi → Bangalore Problem

```
   — Append fails.
   — Two contradictory memories.
   — Need write-time logic.
```

If your only operation is "append", you eventually hold both "lives in Delhi" and "lives in Bangalore" with no way to choose.

## 6.3 Four Decisions at Write Time

```
                 new fact arrives
           ┌──────────────────────────┐
           │ "just moved to Bangalore"│
           └────────────┬─────────────┘
                        ↓
              ┌────────────────────┐
              │ compare to existing│
              └─┬────┬──────┬────┬─┘
          ┌─────┘    │      │    └─────┐
          ↓          ↓      ↓          ↓
   ┌──────────┐┌──────────┐┌──────────┐┌──────────┐
   │   ADD    ││  UPDATE  ││  DELETE  ││   NOOP   │
   │genuinely ││ modifies ││invalidates││ already  │
   │   new    ││ existing ││   old    ││  known   │
   │→store    ││→edit in  ││→drop the ││→do       │
   │ fresh    ││  place   ││  prior   ││ nothing  │
   │          ││          ││          ││          │
   │e.g. first││ Delhi →  ││past      ││ noise    │
   │ mention  ││Bangalore ││NeurIPS   ││reduction │
   └──────────┘└──────────┘└──────────┘└──────────┘
```

> **Write-time intelligence, not just write-time storage.**

**NOOP is the underrated one.** Most of what a user says is already known. Storing it again inflates the store and dilutes retrieval. Choosing to write *nothing* is a real decision.

## 6.4 One Published Approach

```
   — mem0 paper, ECAI 2025.
   — LoCoMo 91.6% — vendor-reported.
   — Other shapes exist: Letta, Zep.
```

*(Note "vendor-reported" — Part 10 explains why that caveat matters.)*

---

# PART 7 — MEMORY AS A GRAPH

## 7.1 Three Relationship Types

```
   updates    — new replaces old
   extends    — new adds to existing
   derives    — new inferred from existing
```

## 7.2 Graph Relationships

```
   UPDATES                  EXTENDS                  DERIVES
   new replaces old         new adds to existing     new inferred
   ───────────────────      ───────────────────      ───────────────────
   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
   │ "based in Delhi"│      │"teaches AI      │      │"teaches AI      │
   │   6 months ago  │      │      course"    │      │      course"    │
   └────────┬────────┘      └────────┬────────┘      ├─────────────────┤
            ↓                        ↓               │"prefers minimal"│
   ┌─────────────────┐      ┌─────────────────┐      └────────┬────────┘
   │"based in        │      │"prepares decks  │               ↓
   │  Bangalore"     │      │    weekly"      │      ┌─────────────────┐
   │     today       │      │                 │      │"prefers lean,   │
   └─────────────────┘      └─────────────────┘      │ visual slides"  │
                                                     └─────────────────┘
   old marked            both remain active,         never said outright,
   closed_state,         together more useful        inferred from
   retrievable for       than alone                  connections
   history
```

> **The graph stores connections, not just facts.**

**`derives` is the interesting one.** The user never said "I prefer lean visual slides." The graph inferred it from two things they *did* say. That's memory producing knowledge that was never stated.

## 7.3 In Practice — Supermemory's Graph

```
   — Custom engine, not a generic graph DB.
   — Traversal-based retrieval.
   — Earns its complexity over months — not days.
```

**Read that last line as a warning.** A graph is expensive to build and only pays off once there's enough history to traverse. Don't start here.

---

# PART 8 — MEMORIES WITH TIME

## 8.1 The Scenario

```
   JAN   joined Anthropic
   OCT   left, started own lab
```

**"Where did the user work last spring?"**

Neither "current employer" nor "deleted old employer" answers this. You need the *window*.

## 8.2 Time Signatures

```
   — event_start
   — event_end
   — Not metadata — structural.
```

"Not metadata — structural" means the time fields participate in retrieval logic, rather than being a tag you attach and ignore.

## 8.3 Temporal State Transition

```
   ┌────────────────────────────┬──────────────────────────┐
   │   "works at Anthropic"     │   "works at own lab"     │→
   │ closed_state · event_end=Oct│ active · event_end = null│
   └────────────────────────────┴──────────────────────────┘
   ├────────────────────────────┼──────────────────────────┤
   Jan                         Oct                      today

   "where does the user work?"          → active memory
   "where did the user work last spring?"→ window overlap
   "when did the user start the lab?"   → event_start lookup
```

> **The old memory isn't deleted — its window closes.**

One data model, three different question types answered. `event_end = null` is the marker for "still true".

## 8.4 Two Implementations

```
   — mem0 temporal layer (May 2026).
   — Zep / Graphiti — bi-temporal edges.
   — LongMemEval 94.8% — vendor-reported.
```

*(Bi-temporal = tracking both when something was true in the world **and** when the system learned it. Those differ, and conflating them causes bugs.)*

---

# PART 9 — FORGETTING

## 9.1 The Expiring Exam

```
   "Exam tomorrow on transformer attention."

   Six months later —

   "What do I usually study around finals?"
```

> First memory is now stale. Second still useful.

The same stored fact is worthless as a *fact* and valuable as *evidence of a pattern*. So deleting it is wrong, and keeping it at full weight is also wrong.

## 9.2 Delete or Down-weight?

```
   — Delete is lossy.
   — Down-weight is lossless.
   — Scaling band: 0.3× to 1.5×.
```

## 9.3 Memory Decay

```
   retrieval weight
   1.0× ┤╲╲
        │ ╲ ╲╲___
   0.75×┤  ╲     ‾‾───___
        │   ╲             ‾‾‾───____  down-weight · lossless
   0.5× ┤    ╲                       ‾‾‾‾‾───────────
        │     ╲___
   0.25×┤         ‾‾──___
        │                ‾‾───_____  delete · lossy
     0  ┤                          ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾
        └────┬──────────┬──────────┬──────────┬────
        0    5         10         20         30   time
```

> **Lower weight, not lost — recoverable when needed.**

The down-weight curve flattens around 0.3× — it never reaches zero. The delete curve hits zero and stays there. Note the band goes up to **1.5×**, so repeated access can also *boost* a memory above its starting weight.

## 9.4 Old Idea, New Home

```
   1885   Ebbinghaus forgetting curve
   2023   MemoryBank
   2026   mem0 memory decay
```

The maths is 140 years old. Only the substrate is new.

---

# PART 10 — EVALUATION

## 10.1 Three Benchmarks

```
   LoCoMo        — Snap
   LongMemEval   — vendor-published
   ConvoMem      — Salesforce
```

## 10.2 Vendor Numbers

```
   mem0            LoCoMo 91.6%
   Supermemory     #1 across three
   Zep             94.8% DMR
```

> **All numbers vendor-reported. Interpret accordingly.**

## 10.3 But What's Being Measured?

```
   — Transcript still available at query time.
   — The system can re-search.
   — That's retrieval — not memory consolidation.
```

**The critique, stated plainly:** if the full conversation is still sitting there when the question arrives, the system can just go read it again. A high score proves it can *search well*. It proves nothing about whether it *remembered*.

## 10.4 Standard Benchmark vs NoReplay Setup

```
   Standard memory benchmark          NoReplay (Agarwal et al.)
   transcript available at query      transcript discarded after ingest
   ─────────────────────────────      ────────────────────────────────
   ┌───────────────────────────┐      ┌───────────────────────────┐
   │         Ingest            │      │         Ingest            │
   │ Store full transcript     │      │ One chronological pass,   │
   │ verbatim, index by chunk  │      │ update a fixed scratchpad │
   └────────────┬──────────────┘      └────────────┬──────────────┘
                ↓                                  ↓
   ┌───────────────────────────┐      ┌───────────────────────────┐
   │     Question arrives      │      │   Freeze the scratchpad   │
   │ Search full transcript,   │      │ Transcript discarded.     │
   │ retrieve top-k, iterate   │      │ Question arrives now.     │
   └────────────┬──────────────┘      └────────────┬──────────────┘
                ↓                                  ↓
   ┌───────────────────────────┐      ┌───────────────────────────┐
   │ Answer from retrieved     │      │ Answer from scratchpad    │
   │        chunks             │      │          only             │
   │ measures retrieval AND    │      │ isolates memory           │
   │ memory together           │      │ consolidation alone       │
   └───────────────────────────┘      └───────────────────────────┘
```

> **Same task, different question. What does the score mean?**

## 10.5 NoReplay — A Cleaner Test

```
   — One-pass ingest.
   — Frozen scratchpad.
   — Transcript discarded.
```

This forces the system to decide *at write time* what matters — because it never gets a second look. That is the actual definition of memory.

---

# PART 11 — PRODUCTION DEEP-DIVES

## 11.1 Claude Code

```
   — Two layers: CLAUDE.md + auto-memory.
   — Four memory types.
   — Sonnet side-call, not embeddings.
```

```
   ┌──────────────────────┐
   │     CLAUDE.md        │──┐
   │ user-written,        │  │
   │ project root         │  │
   └──────────────────────┘  │
                             ↓
   ┌──────────────────────┐  ┌────────────────────┐   ┌──────────────────┐
   │ ~/.claude/…/memory/  │─→│  Sonnet side-call  │──→│  system prompt   │
   │ auto-generated       │  │ reads filenames +  │   │ CLAUDE.md +      │
   │  · user              │  │ one-line descrips  │   │ selected files   │
   │  · feedback          │  │                    │   │ + freshness      │
   │  · project           │  │ picks 1–5 to load  │   │   warnings       │
   │  · reference         │  └────────────────────┘   └──────────────────┘
   │  index cap: 200 lines│           ▲
   └──────────────────────┘           │
            ▲ ┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄│
   ┌────────┴───────────────────────────────────────────────────┐
   │ extract-memories agent · runs in background · writes files │
   └────────────────────────────────────────────────────────────┘
```

> **No embeddings. No vector store. LLM as relevance filter.**

**The key trick:** instead of embedding everything, a cheap model reads only the *filenames and one-line descriptions*, then picks 1–5 files to actually load. A small LLM call replaces the entire vector-search stack — and it handles paraphrase, which fixes the Part 4.2 semantic miss.

**"Freshness warnings"** is the Part 8 problem handled cheaply: flag old memories rather than modelling time properly.

## 11.2 Codex

```
   — AGENTS.md — layered discovery.
   — Memories — async between sessions.
   — Grep over MEMORY.md, not vectors.
```

```
   AGENTS.md (static)
   ┌───────────────────────┐
   │ global                │┐
   │ ~/.codex/AGENTS.md    │|
   ├───────────────────────┤|  concat   ┌──────────────────────┐
   │ project root          │├─────────→ │ loaded at session    │
   │ project/AGENTS.md     │|           │ start · 32 KiB cap   │
   ├───────────────────────┤|           └──────────────────────┘
   │ override              │┘
   │ subdir/AGENTS.md      │
   └───────────────────────┘

   Memories (async)
   ┌────────────┐   ┌────────────┐   ┌────────────┐   ┌────────────┐
   │session ends│──→│  extract   │──→│   merge    │──→│   store    │
   │idle thresh.│   │(model call1)│  │(model call2)│  │ MEMORY.md  │
   └────────────┘   └────────────┘   └────────────┘   └────────────┘

   recall: session reads summary, greps MEMORY.md when needed
   no embeddings · async writes only · 30-day TTL · EEA/UK/CH unavailable
```

> **Two pipelines: static AGENTS.md + async Memories.**

**Extract and merge are separate model calls.** Extract pulls candidate facts; merge decides ADD/UPDATE/DELETE/NOOP against what's stored. That's Part 6's write-time intelligence, shipped.

**"EEA/UK/CH unavailable"** — a reminder that memory is a privacy feature before it's a technical one.

## 11.3 Tradeoffs

```
   Letta         — most rigorous OS pattern
   mem0          — most published research
   Supermemory   — breadth across interfaces
```

---

# PART 12 — THE HARDEST CASE: VOICE

## 12.1 The Constraint

```
   — Total round trip: 500–800 ms.
   — Memory budget: 50–100 ms.
   — A network call alone takes about that.
```

## 12.2 Voice Latency Budget

```
   ├─── 100 ───┼─ 50 ─┼── 100 ──┼────── 300 ──────┼─ 50 ─┤
     VAD final   STT    memory        LLM TTFT      TTS
                        ▲
                        │
              memory has 50–100 ms
        a hosted vector DB round trip alone
             is 20–80 ms on the network
```

> **A 600 ms round trip leaves 50–100 ms for memory.**

*(VAD = voice activity detection, knowing the user stopped talking. STT = speech to text. TTFT = time to first token. TTS = text to speech.)*

**The squeeze:** your entire memory budget is roughly one network hop. Any remote lookup consumes it all before doing any thinking.

## 12.3 The Inversion

```
   — Nothing expensive between turns.
   — Pre-load before the call.
   — Async writes after.
```

## 12.4 Voice — Three-Tier Stack

```
   ┌──────────────────────────────────────────────────────────────┐
   │ Tier 1 — hot cache                                           │
   │ 1–5 ms · blocking · in-process memory                        │
   │ pre-loaded before the call: user profile, last-call summary,  │
   │ open issues                                                   │
   └──────────────────────────────────────────────────────────────┘
   ┌──────────────────────────────────────────────────────────────┐
   │ Tier 2 — background retrieval                                │
   │ 50–150 ms · async between turns · staged for next turn        │
   │ episodic search, fires every 3–5 turns or on topic shift      │
   └──────────────────────────────────────────────────────────────┘
   ┌──────────────────────────────────────────────────────────────┐
   │ Tier 3 — async writes                                        │
   │ latency irrelevant · runs after turns and after call          │
   │ fact extraction, summarization, consolidation, conflict res.  │
   └──────────────────────────────────────────────────────────────┘
```

> **Pre-loaded · background · async — defined by latency budget.**

Only Tier 1 is blocking. Tier 2 fetches for the *next* turn, not this one. Tier 3 happens when nobody's waiting.

## 12.5 Failure Modes

```
   — Cold start — anonymous caller.
   — Race condition — pre-load versus write-back.
   — Per-turn extraction cost.
```

A race condition here means Tier 3 writing while Tier 1 has already loaded a stale copy.

---

# PART 13 — HORIZON

## 13.1 What's Still Open

```
   — Cross-tool portability.
   — Privacy & consent.
   — Team memory.
   — A core-identity layer.
```

## 13.2 A Five-Layer Future Architecture

```
   ┌──────────────────────────────────────────────────────────────┐
   │ Sensory                                                      │
   │ filters incoming information across modalities —             │
   │ text, voice, files, events                                   │
   ├──────────────────────────────────────────────────────────────┤
   │ Short-term                                                   │
   │ active topics across reasoning, working scratchpad           │
   ├──────────────────────────────────────────────────────────────┤
   │ Long-term                                                    │
   │ semantic + episodic memories with relationships,             │
   │ graph + temporal                                             │
   ├──────────────────────────────────────────────────────────────┤
   │ Memory managers                                              │
   │ background processes during idle: consolidation, pruning,    │
   │ reflection                                                   │
   ├──────────────────────────────────────────────────────────────┤
   │ Core                                                         │
   │ stable identity, evolves slowly, shapes how everything       │
   │ else is interpreted                                          │
   └──────────────────────────────────────────────────────────────┘
```

> **One sketch — sensory · short-term · long-term · managers · core.**

Compare to Part 2's human pipeline: sensory filter, working memory, long-term with episodic+semantic, consolidation. The sketch is the brain diagram with one addition — **Core**, a stable identity layer with no clean human equivalent.

## 13.3 Takeaway

```
   — Memory is not solved.
   — Infrastructure is mature.
   — Semantics are still craft.
   — Speed is set by what you prepared, not what you fetch.
```

**"Infrastructure is mature, semantics are still craft"** is the honest summary. Storing and retrieving text is a solved problem. Deciding *what deserves to be stored* and *what is still true* is not, and there's no library for it.

---

## Quick-Recall Table

| Question | Answer |
| --- | --- |
| RAG vs memory? | RAG retrieves by similarity; memory tracks state over time. RAG has no concept of "later". |
| Three types of memory? | Episodic (events), semantic (facts), procedural (skills). |
| Human pipeline? | Sensory filter → working/short-term → long-term, with consolidation and forgetting. |
| The simplest memory system? | A markdown file loaded at session start and saved at session end. |
| What do Claude Code / Codex / Hermes have in common? | Plain markdown files, no embeddings, no vector store. |
| The three file failure modes? | The cap (size limit), semantic miss (exact-match retrieval), temporal & portability. |
| The OS analogy? | Context window = RAM, external store = disk, LLM pages memory in/out. MemGPT → Letta. |
| The three MemGPT tiers? | Core (always loaded), recall (searchable), archival (cold). |
| The four write-time decisions? | ADD, UPDATE, DELETE, NOOP. |
| Why does NOOP matter? | Most input is already known; re-storing it inflates the store and dilutes retrieval. |
| The three graph relationships? | updates (replaces), extends (adds to), derives (inferred from). |
| Why is `derives` interesting? | It produces facts the user never actually stated. |
| How does temporal memory handle a job change? | The old memory isn't deleted — `event_end` is set and its window closes. `event_end = null` means still true. |
| Delete or down-weight? | Down-weight — lossless and recoverable. Scaling band 0.3× to 1.5×. |
| Where does decay come from? | Ebbinghaus forgetting curve, 1885. |
| What's wrong with the memory benchmarks? | The transcript is still available at query time, so they measure retrieval, not consolidation. |
| What does NoReplay change? | One-pass ingest, frozen scratchpad, transcript discarded — isolates consolidation. |
| How does Claude Code do relevance without embeddings? | A Sonnet side-call reads filenames + one-line descriptions and picks 1–5 files to load. |
| Why is voice the hardest case? | ~600 ms round trip leaves 50–100 ms for memory — roughly one network hop. |
| The voice three tiers? | Hot cache (1–5 ms, blocking), background retrieval (50–150 ms, async), async writes (latency irrelevant). |
| The final takeaway? | Infrastructure is mature; semantics are still craft. Speed is set by what you prepared, not what you fetch. |

---

> **The through-line:** every design in this class is answering one question — *what is still true?* Files ignore it, layers decide it at write time, graphs encode how facts relate, temporal models give facts a window, and decay lets them fade instead of vanish.
