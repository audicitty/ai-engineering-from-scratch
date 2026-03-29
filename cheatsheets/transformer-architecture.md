# 🏗️ Transformer Architecture Cheatsheet

## Classic vs Modern at a Glance

| Component | Classic (2017) | Modern (2024) | Why Changed |
|-----------|---------------|---------------|-------------|
| Normalization | LayerNorm (Post) | RMSNorm (Pre) | Faster + stable for 80+ layers |
| Position | Sinusoidal (absolute) | RoPE (relative) | Handles any sequence length |
| Attention | Multi-Head (MHA) | Grouped Query (GQA) | 4x less KV cache memory |
| Activation | ReLU | SwiGLU | Preserves subtle information |
| Depth | 6-12 layers | 32-80 layers | Pre-Norm made deep stacking possible |
| Context | 512-2K tokens | 8K-128K+ tokens | RoPE + GQA removed bottlenecks |

## A Single Transformer Block

### Classic Flow
```
Input → Attention → Add + LayerNorm → FFN (ReLU) → Add + LayerNorm → Output
         ↑_skip_connection_↑                ↑_skip_connection_↑
```

### Modern Flow
```
Input → RMSNorm → GQA Attention (+ RoPE) → Add → RMSNorm → SwiGLU FFN → Add → Output
  ↑________skip_connection_______________↑    ↑_______skip_connection________↑
```

## Key Components Explained

### RMSNorm
```python
rms = sqrt(mean(x²) + eps)
output = (x / rms) * weight
# No mean subtraction, no bias — simpler than LayerNorm
```

### RoPE (Rotary Position Embedding)
- Rotates Q and K vectors by position-dependent angle
- Dot product depends on RELATIVE distance, not absolute position
- Low dims = fast rotation (nearby words), High dims = slow (distant words)

### GQA (Grouped Query Attention)
- MHA: 32 Q heads, 32 K heads, 32 V heads (expensive)
- GQA: 32 Q heads, 8 K heads, 8 V heads (groups of 4 Q share 1 KV)
- MQA: 32 Q heads, 1 K head, 1 V head (too aggressive)
- GQA = sweet spot: 4x memory savings, ~0% quality loss

### SwiGLU
```python
gate = SiLU(x @ W_gate)     # SiLU = x * sigmoid(x), smooth curve
signal = x @ W_up            # Raw transformation
output = gate * signal       # Gate controls what passes through
```

## Quick Reference: What Each Part Does

| Part | One-Sentence Job |
|------|-----------------|
| Token Embedding | Convert word IDs to vectors |
| Position Encoding | Tell the model where each word is |
| Self-Attention | Figure out which words are relevant to each other |
| Residual Connection | Preserve original info (skip road) |
| Normalization | Keep values in a stable range |
| Feed-Forward Network | Process and transform attended information |
| Output Head | Convert final vectors to word probabilities |
