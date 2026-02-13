# MIT 6.S191 — Lab 1, Part 1

## Tensor Basics, Tensor Operations, and Simple Neural Networks

This note summarizes the core concepts learned in **PyTorch Part 1 — Introduction**.

---

# 1. What is a Tensor?

A tensor is a **multi-dimensional container of numbers**.

It generalizes:

- Scalar → single number (0D)
- Vector → list of numbers (1D)
- Matrix → grid of numbers (2D)
- Higher-order tensors → multi-dimensional arrays (3D, 4D, ...)

In deep learning, **all data is represented as tensors**:

- Images
- Text
- Audio
- Model parameters

PyTorch tensors support:

- GPU acceleration
- Automatic differentiation (autograd)
- Efficient math operations

---

# 2. Tensor Dimensions vs Shape vs Size

These terms are often confused.

## Dimension (Rank)

Number of axes a tensor has.

Examples:

- Scalar → 0D
- Vector → 1D
- Matrix → 2D
- Image batch → 4D

## Shape

Size along each axis.

Example: (2, 4)
means:

- 2 rows
- 4 columns

## Size in PyTorch

`tensor.size()` and `tensor.shape` mean the same thing.

---

# 3. Why Images are 4D Tensors

Images in deep learning use the format:
(batch_size, channels, height)

Example: (10, 3, 256, 256)

Meaning:

- 10 images in a batch
- 3 color channels (RGB)
- 256 × 256 resolution

Why batch?
Neural networks process **multiple samples at once** for efficiency.

---

# 4. Tensor Creation

Examples:

matrix = torch.tensor([[1,2],[3,4]])
images = torch.zeros(10, 3, 256, 256)

Tensors can be:

manually created

filled with zeros / ones / random values

---

# 5. Tensor Operations

Basic elementwise math:

c = a + b
d = b - 1
e = c \* d

PyTorch records this as a computational graph, enabling gradient calculation.

Operators are equivalent to:

- → torch.add

* → torch.mul

- → torch.sub
  / → torch.div

---

# 7. Linear (Dense) Layer

Core building block of neural networks:
z = xW + b
y = activation(z)

In PyTorch:
z = torch.matmul(x, W) + bias
y = torch.sigmoid(z)
Where:

- W = learnable weights
- b = bias
- sigmoid = activation function

---

# 8. Sequential Model

A simple neural network can be built using:
nn.Sequential(
nn.Linear(input_size, output_size),
nn.Sigmoid()
)
This creates: Input → Linear → Sigmoid → Output

---

# 9. Subclassing Models

Instead of Sequential, models can be manually defined:

class Model(nn.Module):
def **init**(self):
self.linear = nn.Linear(...)
self.activation = nn.Sigmoid()

This allows flexible architectures.

---

# 10. Identity Behavior

A model can optionally return the input unchanged:

if isidentity:
return inputs

---

This demonstrates that neural networks are simply functions.

---

# 11. Loss Function

loss = (x - x_pred)\*\*2

Loss measures prediction error. Training aims to minimize loss.

---

# 12. Gradient Descent Intuition

Neural networks learn by:

- Predicting
- Measuring loss
- Computing gradients
- Updating weights to reduce loss
  This process is called optimization.
