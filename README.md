<div align="center">

# 🧠 AI Engineering from Scratch

### The Complete Guide to Understanding and Building Modern AI Systems

**From "What is AI?" to building agents, fine-tuning models, and deploying RAG pipelines.**

[![Stars](https://img.shields.io/github/stars/audicitty/ai-engineering-from-scratch?style=social)](https://github.com/audicitty/ai-engineering-from-scratch)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[📖 Start Learning](#-roadmap) · [🗺️ Roadmap](roadmap/ROADMAP.md) · [💻 Code Examples](#-code-examples) · [📝 Cheatsheets](cheatsheets/)

---

*"I couldn't find a single resource that explained AI engineering from absolute zero to production-ready — so I built one."*

</div>

## 🤔 What Is This?

This is a **free, open-source curriculum** that takes you from knowing nothing about AI to understanding how modern LLMs like GPT, Claude, Llama, and DeepSeek actually work — and how to build with them.

Every concept is explained in **simple language with diagrams, code examples, and real-world analogies.** No PhD required.

### Who is this for?

- 🎓 **Students** who want to understand AI beyond the buzzwords
- 💻 **Developers** who want to transition into AI/ML engineering
- 🔄 **Career switchers** looking for a structured learning path
- 🧠 **Curious minds** who want to know what's actually happening inside ChatGPT

### What makes this different?

| Other Resources | This Repo |
|----------------|-----------|
| Jump straight to code | Builds intuition first, then code |
| Assume math background | Explains math from scratch with analogies |
| Cover theory OR practice | Both — every concept has runnable code |
| Outdated (pre-2023) | Covers 2024-2025 techniques (GQA, RoPE, SwiGLU) |
| Scattered across blogs | One structured path, start to finish |

---

## 🗺️ Roadmap

### Phase 1: Foundations — *"How does AI actually work?"*

| # | Topic | Notes | Code | Status |
|---|-------|-------|------|--------|
| 01 | [History of AI — What is AI, how did we get to transformers](01-history-of-ai/) | ✅ | — | Complete |
| 02 | [Neural Networks — Backprop, gradient descent, training](02-neural-networks/) | ✅ | ✅ | Complete |
| 03 | [Transformers — Attention, encoder-decoder, self-attention](03-transformers/) | ✅ | ✅ | Complete |
| 04 | [Tensors, Matrices & PyTorch — The math and tools](04-tensors-and-pytorch/) | ✅ | ✅ | Complete |

### Phase 2: Deep Dive — *"What's inside a modern LLM?"*

| # | Topic | Notes | Code | Status |
|---|-------|-------|------|--------|
| 05 | [Coding Attention — Simple attention, KV cache, GQA, MLA](05-coding-attention/) | ✅ | ✅ | Complete |
| 06 | [Modern LLM Architecture — RMSNorm, SwiGLU, RoPE, GQA](06-modern-llm/) | ✅ | ✅ | Complete |
| 07 | [Hugging Face End-to-End — Loading, using, and deploying models](07-huggingface/) | 🔄 | 🔄 | In Progress |

### Phase 3: Building with LLMs — *"How do I build real AI products?"*

| # | Topic | Notes | Code | Status |
|---|-------|-------|------|--------|
| 08 | [Vector DBs & RAG — Retrieval-augmented generation](08-vector-dbs-and-rag/) | 📋 | 📋 | Planned |
| 09 | [Context Engineering — Summarization, data collection](09-context-engineering/) | 📋 | 📋 | Planned |
| 10 | [Agents from First Principles — Building an agent framework](10-agents-from-scratch/) | 📋 | 📋 | Planned |
| 11 | [Agent Frameworks — LangChain, CrewAI, and more](11-agent-frameworks/) | 📋 | 📋 | Planned |
| 12 | [Memory — Giving agents persistent memory](12-memory/) | 📋 | 📋 | Planned |
| 13 | [Computer Use & Multimodal Agents](13-computer-use-agents/) | 📋 | 📋 | Planned |

### Phase 4: Training & Evaluation — *"How do I make models better?"*

| # | Topic | Notes | Code | Status |
|---|-------|-------|------|--------|
| 14 | [What is Fine-tuning — When and why to fine-tune](14-finetuning/) | 📋 | 📋 | Planned |
| 15 | [RL Fine-tuning — RLHF, DPO, reward models](15-rl-finetuning/) | 📋 | 📋 | Planned |
| 16 | [Evals — Testing agents and models systematically](16-evals/) | 📋 | 📋 | Planned |
| 17 | [Advanced Topics — Scaling laws, MoE, distillation](17-advanced-topics/) | 📋 | 📋 | Planned |

### 🏗️ Projects

| Project | Description | Difficulty |
|---------|------------|------------|
| [Agent Framework](projects/agent-framework/) | Build an AI agent framework from scratch | ⭐⭐⭐ |
| [RL Fine-tuning + Evals](projects/rl-finetuning/) | Fine-tune a model with RL and write evaluation tests | ⭐⭐⭐⭐ |
| [Devin Clone](projects/devin-clone/) | Build an AI coding assistant | ⭐⭐⭐⭐⭐ |
| [Memory Framework](projects/memory-framework/) | Build a persistent memory system for AI agents | ⭐⭐⭐ |

---

## 💻 Code Examples

Every concept comes with runnable Python code. Here's a taste:

<details>
<summary><b>🔥 Simple Attention in 5 Lines</b></summary>

```python
import torch
import torch.nn.functional as F

Q = torch.randn(3, 64)   # 3 tokens, 64 dimensions
K = torch.randn(3, 64)
V = torch.randn(3, 64)

scores = Q @ K.T / 8.0                    # dot product + scale
weights = F.softmax(scores, dim=-1)        # normalize to probabilities
output = weights @ V                       # weighted sum of values
```

</details>

<details>
<summary><b>🔥 RMSNorm in 6 Lines</b></summary>

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
```

</details>

<details>
<summary><b>🔥 The 5-Line Training Loop (Same from MNIST to GPT)</b></summary>

```python
for epoch in range(num_epochs):
    y_pred = model(x_batch)              # 1. Forward pass
    loss = criterion(y_pred, y_batch)     # 2. Compute loss
    optimizer.zero_grad()                 # 3. Clear old gradients
    loss.backward()                       # 4. Backpropagation
    optimizer.step()                      # 5. Update weights
```

</details>

---

## 📝 Cheatsheets

Quick-reference guides you can print or bookmark:

| Cheatsheet | Description |
|-----------|------------|
| [PyTorch Essentials](cheatsheets/pytorch-essentials.md) | Tensors, operations, GPU, autograd |
| [Transformer Architecture](cheatsheets/transformer-architecture.md) | Classic vs Modern, all components |
| [Attention Variants](cheatsheets/attention-variants.md) | MHA vs GQA vs MQA vs MLA |
| [Training Checklist](cheatsheets/training-checklist.md) | Everything you need for a training run |

---

## 🚀 Getting Started

### Option 1: Just Read the Notes
Click any topic in the [Roadmap](#-roadmap) above and start reading. Everything is in Markdown — GitHub renders it beautifully.

### Option 2: Run the Code Locally
```bash
# Clone the repo
git clone https://github.com/audicitty/ai-engineering-from-scratch.git
cd ai-engineering-from-scratch

# Install dependencies
pip install torch transformers datasets

# Run any example
python 04-tensors-and-pytorch/code/tensor_basics.py
```

---

## 📊 How This Repo Is Organized

```
ai-engineering-from-scratch/
│
├── README.md                     ← You are here
├── roadmap/ROADMAP.md            ← Detailed learning path with time estimates
│
├── 01-history-of-ai/
│   ├── notes.md                  ← Comprehensive lecture notes
│   └── resources.md              ← Extra reading, papers, videos
│
├── 04-tensors-and-pytorch/
│   ├── notes.md                  ← Full notes with diagrams
│   ├── code/
│   │   ├── tensor_basics.py      ← Runnable code examples
│   │   └── training_loop.py
│   └── exercises/
│       └── exercises.md          ← Practice problems with solutions
│
├── cheatsheets/                  ← Quick-reference guides
├── projects/                     ← Hands-on capstone projects
└── assets/                       ← Images and diagrams
```

---

## 🌟 Contributing

Found a typo? Have a better explanation? Want to add notes for a topic?

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## 📣 Support This Project

If this helped you, please consider:

- ⭐ **Starring this repo** — it helps others find it
- 🐦 **Sharing on Twitter/LinkedIn** — spread the knowledge
- 🍴 **Forking and contributing** — make it even better
- 💬 **Opening an issue** — ask questions or suggest improvements

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

You're free to use, modify, and distribute these notes. Attribution is appreciated but not required.

---

<div align="center">

**Built with ❤️ by [audicitty](https://github.com/audicitty)**

*Learning AI shouldn't cost a fortune or require a PhD.*

</div>
