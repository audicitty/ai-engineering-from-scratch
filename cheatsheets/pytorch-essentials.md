# 🔥 PyTorch Essentials Cheatsheet

## Creating Tensors

```python
import torch

x = torch.tensor(5.0)                    # Scalar (0D)
v = torch.tensor([1.0, 2.0, 3.0])        # Vector (1D)
m = torch.tensor([[1, 2], [3, 4]])        # Matrix (2D)
t = torch.zeros(2, 3, 4)                 # 3D tensor of zeros
r = torch.randn(3, 4)                    # Random normal values
o = torch.ones(3, 4)                     # All ones
```

## Essential Operations

```python
# Shape
x.shape                          # Check shape (do this constantly!)
x.view(6, 2)                     # Reshape (total elements must match)
x.permute(0, 2, 1)               # Swap axes

# Math
A @ B                            # Matrix multiplication
torch.matmul(A, B)               # Same thing, explicit
A * B                            # Element-wise multiplication
A + B                            # Element-wise addition

# Useful functions
torch.softmax(x, dim=-1)         # Convert to probabilities
torch.relu(x)                    # Zero out negatives
x.mean(), x.sum(), x.max()       # Reductions
```

## GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)                 # Move tensor to GPU
model = model.to(device)         # Move model to GPU
```

## Autograd (Automatic Gradients)

```python
x = torch.tensor(3.0, requires_grad=True)   # Track this tensor
y = x ** 2 + 5                               # y = 14
y.backward()                                  # Compute gradients
print(x.grad)                                 # dy/dx = 2x = 6.0

# During inference (no training):
with torch.no_grad():
    output = model(input)
```

## Building Models

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
```

## The Training Loop (Memorize This!)

```python
model = MyModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    y_pred = model(x_batch)           # 1. Forward
    loss = criterion(y_pred, y_batch)  # 2. Loss
    optimizer.zero_grad()              # 3. Clear gradients
    loss.backward()                    # 4. Backward
    optimizer.step()                   # 5. Update weights
```

## Common Shape Rules

```
Matrix multiply: (a, b) @ (b, c) → (a, c)    # Inner dims must match
Broadcasting:    (3, 4) + (4,)   → (3, 4)     # Smaller stretches
View/Reshape:    elements must stay the same    # (2,3,4) → (6,4) ✓
```

## Common Errors & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| RuntimeError: mat1 and mat2 shapes cannot be multiplied | Inner dimensions don't match | Check `.shape` of both tensors |
| RuntimeError: expected device cuda but got cpu | Tensors on different devices | Move both to same device with `.to(device)` |
| Gradients exploding (loss = NaN) | Forgot `zero_grad()` or learning rate too high | Add `optimizer.zero_grad()`, reduce lr |
| Model not learning (loss stuck) | Learning rate too low or wrong loss function | Increase lr, check loss function choice |
