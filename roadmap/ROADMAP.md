# 🗺️ Complete AI Engineering Roadmap

## How to Use This Roadmap

Each phase builds on the previous one. Don't skip ahead — the foundations matter.

**Time estimate:** 12-16 weeks at 10-15 hours/week

---

## Phase 1: Foundations (Weeks 1-3)

> *Goal: Understand what AI is, how neural networks learn, and the math behind it all.*

### Week 1: History & Context
- [ ] What is AI? The journey from rule-based systems to deep learning
- [ ] Key milestones: Perceptron → Backprop → CNNs → RNNs → Transformers
- [ ] Why transformers won: the attention mechanism breakthrough

### Week 2: Neural Networks & Training
- [ ] How a single neuron works (weights × inputs + bias)
- [ ] Activation functions: ReLU, Sigmoid, Softmax
- [ ] Loss functions: MSE, Cross-Entropy
- [ ] Backpropagation: how gradients flow backwards
- [ ] Gradient descent: how models actually learn
- [ ] **Code:** Train a simple neural network on MNIST

### Week 3: Math & Tools
- [ ] Scalars → Vectors → Matrices → Tensors
- [ ] Matrix multiplication: the engine of deep learning
- [ ] Dot product: measuring similarity
- [ ] PyTorch: tensors, autograd, nn.Module
- [ ] The 5-line training loop
- [ ] GPU acceleration
- [ ] **Code:** Build and train a model in PyTorch from scratch

---

## Phase 2: Deep Dive into Transformers (Weeks 4-6)

> *Goal: Understand every component inside a modern LLM.*

### Week 4: Transformers
- [ ] Encoder-decoder architecture
- [ ] Self-attention mechanism step by step
- [ ] Multi-head attention: why multiple heads?
- [ ] Positional encoding: why order matters
- [ ] The full transformer forward pass

### Week 5: Coding Attention
- [ ] Code simple attention from scratch
- [ ] Add masking (causal attention)
- [ ] Implement KV cache for efficient generation
- [ ] Code GQA (Grouped Query Attention)
- [ ] Understand MLA (Multi-Latent Attention)
- [ ] **Code:** Build a complete attention module

### Week 6: Modern LLM Architecture
- [ ] The 4 upgrades: RMSNorm, SwiGLU, RoPE, GQA
- [ ] Pre-Norm vs Post-Norm and why it matters
- [ ] The KV cache memory bottleneck
- [ ] Classic (2017) vs Modern (2024) transformer comparison
- [ ] **Code:** Build a modern transformer block

---

## Phase 3: Building with LLMs (Weeks 7-11)

> *Goal: Build real AI applications and products.*

### Week 7: Hugging Face
- [ ] Loading pre-trained models
- [ ] Tokenizers and pipelines
- [ ] Model inference and generation
- [ ] Deploying models
- [ ] **Code:** End-to-end Hugging Face project

### Week 8: Observability
- [ ] Instrumenting LLM calls
- [ ] Tracing and logging
- [ ] Monitoring costs and latency
- [ ] Debugging LLM applications

### Week 9: Vector DBs & RAG
- [ ] What are embeddings and vector spaces?
- [ ] Vector databases: Pinecone, Chroma, Weaviate
- [ ] RAG pipeline: Retrieve → Augment → Generate
- [ ] Chunking strategies
- [ ] **Code:** Build a RAG pipeline from scratch

### Week 10: Context Engineering
- [ ] Prompt engineering vs context engineering
- [ ] Summarization techniques
- [ ] Data collection and preprocessing
- [ ] Managing context windows effectively

### Week 11: Agents
- [ ] What is an AI agent? (ReAct pattern)
- [ ] Tool use and function calling
- [ ] Building an agent from first principles
- [ ] Agent frameworks: LangChain, CrewAI
- [ ] Memory systems for agents
- [ ] Computer use and multimodal agents
- [ ] **Project:** Build a complete agent framework

---

## Phase 4: Training & Evaluation (Weeks 12-15)

> *Goal: Fine-tune models and evaluate them properly.*

### Week 12-13: Fine-tuning
- [ ] When to fine-tune vs prompt engineer vs RAG
- [ ] Supervised fine-tuning (SFT)
- [ ] LoRA and QLoRA: efficient fine-tuning
- [ ] Dataset preparation
- [ ] **Code:** Fine-tune a model for a specific use case

### Week 14: RL Fine-tuning
- [ ] RLHF: Reinforcement Learning from Human Feedback
- [ ] Reward models
- [ ] DPO: Direct Preference Optimization
- [ ] **Project:** RL fine-tuning + writing evals

### Week 15: Evals
- [ ] Why evaluation matters
- [ ] Designing evaluation suites
- [ ] Automated vs human evaluation
- [ ] Testing agents systematically
- [ ] Benchmarks and leaderboards

---

## Phase 5: Advanced Topics (Week 16+)

> *Goal: Understand the cutting edge.*

- [ ] Scaling laws and compute-optimal training
- [ ] Mixture of Experts (MoE)
- [ ] Model distillation
- [ ] Quantization (INT8, INT4, GPTQ, AWQ)
- [ ] Speculative decoding
- [ ] Multi-modal models (vision + language)
- [ ] Latest research papers and trends

---

## Recommended Resources

### Books
- *Dive into Deep Learning* (free online) — d2l.ai
- *Understanding Deep Learning* by Simon Prince

### Courses (Complementary)
- Andrej Karpathy's Neural Networks: Zero to Hero (YouTube, free)
- fast.ai Practical Deep Learning

### Papers (Read When Ready)
- "Attention Is All You Need" (2017) — The original transformer
- "LLaMA: Open and Efficient Foundation Language Models" (2023)
- "GQA: Training Generalized Multi-Query Transformer Models" (2023)

### Tools to Master
- PyTorch
- Hugging Face Transformers
- Weights & Biases (experiment tracking)
- vLLM (fast inference)
