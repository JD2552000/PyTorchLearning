# main.py

import torch
import numpy as np
import time

# This script provides a comprehensive overview of fundamental PyTorch tensor operations.
# It's structured to be a clear, executable guide for beginners.

# ======================================================================================
# 1. Tensor Creation
# ======================================================================================
print("--- 1. Tensor Creation ---")

# Create a tensor from a Python list
# A tensor is a multi-dimensional array, the primary data structure in PyTorch.
basic_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Basic Tensor:\n", basic_tensor)

# Create an identity matrix
# The eye function creates a 2D tensor with ones on the diagonal and zeros elsewhere.
identity_tensor = torch.eye(5)
print("\nIdentity Tensor (5x5):\n", identity_tensor)

# Create a tensor filled with a single value
# The full function creates a tensor of a given shape, filled with a specified value.
filled_tensor = torch.full((3, 3), 5)
print("\nTensor filled with 5s (3x3):\n", filled_tensor)


# ======================================================================================
# 2. Tensor Properties (Shape & Data Type)
# ======================================================================================
print("\n--- 2. Tensor Properties ---")

# Get the shape of a tensor
# The .shape attribute returns the size of the tensor along each dimension.
shape_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\nShape of tensor: {shape_tensor.shape}") # Expected: torch.Size([2, 3])

# Get the data type of a tensor
# The .dtype attribute shows the data type of the elements in the tensor.
print(f"Data type of tensor: {shape_tensor.dtype}") # Expected: torch.int64 by default for integers

# Create a tensor with a specific data type
float_tensor = torch.tensor([1, 2, 3], dtype=torch.float64)
print(f"\nFloat tensor's dtype: {float_tensor.dtype}")

# Change the data type of an existing tensor
# The .to() method can be used to cast a tensor to a new data type.
int_tensor_to_float = shape_tensor.to(torch.float32)
print(f"Original dtype: {shape_tensor.dtype}, Converted dtype: {int_tensor_to_float.dtype}")


# ======================================================================================
# 3. Creating Tensors from Other Tensors (Sharing Properties)
# ======================================================================================
print("\n--- 3. Creating Tensors with Inherited Properties ---")

# Create tensors with the same shape as an existing tensor
template_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"\nTemplate tensor shape: {template_tensor.shape}")

empty_like_tensor = torch.empty_like(template_tensor) # Contains uninitialized data
zeros_like_tensor = torch.zeros_like(template_tensor) # Filled with zeros
ones_like_tensor = torch.ones_like(template_tensor)   # Filled with ones

print("Zeros-like tensor:\n", zeros_like_tensor)

# Create a tensor with random values of the same shape
# Note: rand_like creates floats. The input tensor must have a floating-point dtype.
# The original code had an error here because the template was an integer tensor.
template_float_tensor = template_tensor.to(torch.float32)
rand_like_tensor = torch.rand_like(template_float_tensor)
print("\nRandom-like float tensor:\n", rand_like_tensor)


# ======================================================================================
# 4. Mathematical & Logical Operations
# ======================================================================================
print("\n--- 4. Mathematical & Logical Operations ---")

# --- Scalar Operations ---
# Operations between a tensor and a single number (scalar).
print("\n-- Scalar Operations --")
scalar_op_tensor = torch.rand(2, 2)
print("Original tensor:\n", scalar_op_tensor)
print("Addition (+ 2):\n", scalar_op_tensor + 2)
print("Multiplication (* 2):\n", scalar_op_tensor * 2)
print("Power (** 2):\n", scalar_op_tensor ** 2)

# --- Element-wise Tensor Operations ---
# Operations between two tensors of the same shape. The operation is applied element by element.
print("\n-- Element-wise Operations --")
tensor_op_a = torch.rand(2, 3)
tensor_op_b = torch.rand(2, 3)
print("Tensor A:\n", tensor_op_a)
print("Tensor B:\n", tensor_op_b)
print("A + B:\n", tensor_op_a + tensor_op_b)
print("A * B (Element-wise):\n", tensor_op_a * tensor_op_b)

# --- Other Common Math Functions ---
print("\n-- Common Math Functions --")
math_func_tensor = torch.tensor([1.5, -2.3, 3.7])
print("Original tensor for functions:", math_func_tensor)
print("Absolute values:", torch.abs(math_func_tensor)) # torch.abs(tensor)
print("Ceiling (round up):", torch.ceil(math_func_tensor)) # torch.ceil(tensor)
print("Floor (round down):", torch.floor(math_func_tensor)) # torch.floor(tensor)

# Clamp: Limits all elements to be within a specified min/max range.
clamped_tensor = torch.clamp(math_func_tensor, min=-2.0, max=3.0)
print("Clamped tensor (min=-2, max=3):", clamped_tensor)

# --- Reduction Operations ---
# Operations that reduce a tensor to a single value.
print("\n-- Reduction Operations --")
reduction_tensor = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)
print("Original tensor for reduction:\n", reduction_tensor)
print("Sum of all elements:", torch.sum(reduction_tensor))
print("Mean of all elements:", torch.mean(reduction_tensor))
print("Sum along columns (dim=0):", torch.sum(reduction_tensor, dim=0))
print("Mean along rows (dim=1):", torch.mean(reduction_tensor, dim=1))
print("Index of max element (flattened):", torch.argmax(reduction_tensor))
print("Index of min element (flattened):", torch.argmin(reduction_tensor))

# --- Matrix Operations ---
print("\n-- Matrix Operations --")
matrix_a = torch.randint(size=(2, 3), low=0, high=10)
matrix_b = torch.randint(size=(3, 4), low=0, high=10)
print("Matrix A (2x3):\n", matrix_a)
print("Matrix B (3x4):\n", matrix_b)

# Matrix Multiplication: (m x n) @ (n x p) -> (m x p)
matrix_mul_result = torch.matmul(matrix_a, matrix_b)
print("Matrix Multiplication (A @ B):\n", matrix_mul_result)

# Transpose: Swaps dimensions. Here, we swap dim 0 and 1.
transposed_matrix_a = torch.transpose(matrix_a, 0, 1)
print("Transposed Matrix A (3x2):\n", transposed_matrix_a)

# Determinant and Inverse: Only for square matrices of float/complex type.
square_matrix = torch.tensor([[3., 2.], [5., 4.]], dtype=torch.float32)
print("\nSquare Matrix:\n", square_matrix)
print("Determinant:", torch.det(square_matrix))
print("Inverse:\n", torch.inverse(square_matrix))

# Dot Product: For two 1D tensors (vectors).
vector_1 = torch.tensor([1, 2, 3])
vector_2 = torch.tensor([4, 5, 6])
dot_product = torch.dot(vector_1, vector_2)
print("\nDot product of [1,2,3] and [4,5,6]:", dot_product) # 1*4 + 2*5 + 3*6 = 32

# --- Comparison Operations ---
# These return a boolean tensor.
print("\n-- Comparison Operations --")
comp_tensor_a = torch.tensor([[1, 2], [3, 4]])
comp_tensor_b = torch.tensor([[1, 5], [0, 4]])
print("A > B:\n", comp_tensor_a > comp_tensor_b)
print("A == B:\n", comp_tensor_a == comp_tensor_b)

# --- Special Functions (Common in ML) ---
print("\n-- Special Functions --")
special_func_tensor = torch.arange(-5., 6., dtype=torch.float32).reshape(11, 1)
print("Original tensor for special functions:\n", special_func_tensor.T) # .T is a shortcut for transpose
print("Sigmoid:\n", torch.sigmoid(special_func_tensor).T)
print("ReLU (Rectified Linear Unit):\n", torch.relu(special_func_tensor).T)

# Softmax: Converts values into a probability distribution. Applied along a dimension.
softmax_tensor = torch.rand(1, 4)
print("\nOriginal tensor for softmax:", softmax_tensor)
print("Softmax (dim=1):", torch.softmax(softmax_tensor, dim=1))
print("Sum after softmax:", torch.sum(torch.softmax(softmax_tensor, dim=1))) # Sums to 1


# ======================================================================================
# 5. In-place Operations
# ======================================================================================
print("\n--- 5. In-place Operations ---")
# In-place operations modify the tensor directly without creating a new one.
# They are denoted by a trailing underscore `_` (e.g., `add_`). This saves memory.

inplace_tensor = torch.rand(2, 3)
tensor_to_add = torch.ones(2, 3)
print("Original inplace tensor:\n", inplace_tensor)

inplace_tensor.add_(tensor_to_add) # Modifies inplace_tensor directly
print("Tensor after inplace addition:\n", inplace_tensor)


# ======================================================================================
# 6. Copying Tensors
# ======================================================================================
print("\n--- 6. Copying Tensors ---")
# Simple assignment creates a reference, not a copy. Modifying one affects the other.
original_tensor = torch.ones(3)
ref_tensor = original_tensor
ref_tensor[0] = 999
print(f"Original tensor after modifying reference: {original_tensor}") # It changed!

# Use .clone() to create a true, independent copy.
original_tensor_2 = torch.ones(3)
cloned_tensor = original_tensor_2.clone()
cloned_tensor[0] = 555
print(f"\nOriginal tensor: {original_tensor_2}") # Unchanged
print(f"Cloned tensor after modification: {cloned_tensor}")


# ======================================================================================
# 7. Tensor Operations on GPU
# ======================================================================================
print("\n--- 7. GPU Operations ---")

# Check if a CUDA-enabled GPU is available
if torch.cuda.is_available():
    print("GPU is available!")
    device = torch.device("cuda")

    # Move an existing tensor to the GPU
    cpu_tensor = torch.rand(2, 3)
    print("\nCPU Tensor:\n", cpu_tensor)
    gpu_tensor = cpu_tensor.to(device)
    print("GPU Tensor:\n", gpu_tensor) # Note the device='cuda:0'

    # --- Compare CPU vs GPU Speed ---
    # GPU excels at large-scale parallel computations like matrix multiplication.
    # For small tensors, the overhead of moving data to the GPU can make it slower.
    
    # Increase matrix size for a more noticeable difference
    size = 5000 
    matrix_1_cpu = torch.rand(size, size)
    matrix_2_cpu = torch.rand(size, size)

    # Time CPU
    start_time = time.time()
    result_cpu = torch.matmul(matrix_1_cpu, matrix_2_cpu)
    cpu_time = time.time() - start_time
    print(f"\nCPU time for {size}x{size} matrix multiplication: {cpu_time:.4f} seconds")

    # Move matrices to GPU
    matrix_1_gpu = matrix_1_cpu.to(device)
    matrix_2_gpu = matrix_2_cpu.to(device)
    
    # Time GPU
    start_time = time.time()
    result_gpu = torch.matmul(matrix_1_gpu, matrix_2_gpu)
    torch.cuda.synchronize()  # Wait for GPU operations to complete
    gpu_time = time.time() - start_time
    print(f"GPU time for {size}x{size} matrix multiplication: {gpu_time:.4f} seconds")

    print(f"Speedup (CPU time / GPU time): {cpu_time / gpu_time:.2f}x")

else:
    print("GPU not available, skipping GPU tests.")


# ======================================================================================
# 8. Reshaping Tensors
# ======================================================================================
print("\n--- 8. Reshaping Tensors ---")
reshape_tensor = torch.ones(4, 4)
print("Original 4x4 tensor shape:", reshape_tensor.shape)

# Reshape: The total number of elements must remain the same (4*4 = 16).
reshaped = reshape_tensor.reshape(2, 2, 4)
print("Reshaped to (2, 2, 4):", reshaped.shape)

# Flatten: Collapses the tensor into a single dimension.
flattened = reshape_tensor.flatten()
print("Flattened tensor shape:", flattened.shape)

# Permute: Rearranges dimensions.
permute_tensor = torch.rand(2, 3, 4) # Dims: 0, 1, 2
permuted = permute_tensor.permute(2, 0, 1) # New order: 4, 2, 3
print(f"\nOriginal shape (2,3,4), Permuted shape (4,2,3): {permuted.shape}")

# Unsqueeze: Adds a new dimension of size 1 at a specified position.
# Useful for adding a batch dimension to a single instance.
image_tensor = torch.rand(224, 224, 3) # (height, width, channels)
batched_tensor = image_tensor.unsqueeze(0) # Add batch dimension at the start
print(f"\nOriginal image shape: {image_tensor.shape}, With batch dim: {batched_tensor.shape}")

# Squeeze: Removes dimensions of size 1.
squeezed_tensor = batched_tensor.squeeze(0)
print(f"Squeezed tensor shape: {squeezed_tensor.shape}")


# ======================================================================================
# 9. NumPy and PyTorch Bridge
# ======================================================================================
print("\n--- 9. NumPy and PyTorch Bridge ---")
# PyTorch tensors and NumPy arrays can be converted back and forth efficiently.
# They share the underlying memory location on the CPU, so changes to one affect the other.

# Convert tensor to NumPy array
tensor_to_np = torch.tensor([1, 2, 3])
numpy_array = tensor_to_np.numpy()
print(f"NumPy array: {numpy_array}, Type: {type(numpy_array)}")

# Convert NumPy array to tensor
numpy_to_tensor = np.array([4, 5, 6])
tensor_from_np = torch.from_numpy(numpy_to_tensor)
print(f"Tensor from NumPy: {tensor_from_np}, Type: {type(tensor_from_np)}")

# Demonstrate shared memory
tensor_from_np[0] = 99
print(f"\nOriginal NumPy array after modifying tensor: {numpy_to_tensor}")