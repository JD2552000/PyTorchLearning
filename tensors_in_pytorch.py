import torch
import numpy as np
import time

# ======================================================================================
# 0. Setup and Device Configuration
# ======================================================================================
# This script provides a comprehensive overview of fundamental PyTorch tensor operations.
# It's structured to be a clear, executable guide for beginners.

print(f"PyTorch Version: {torch.__version__}")

# Set the device for computation (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
elif torch.backends.mps.is_available(): # <-- This line detects your M3 GPU
    device = torch.device("mps")
    print("Using Apple Metal (MPS) GPU")
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU")

# ======================================================================================
# 1. Tensor Creation
# ======================================================================================
print("\n--- 1. Tensor Creation ---")

# From a Python list: The most basic way to create a tensor with specific data.
tensor_from_list = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("Tensor from list:\n", tensor_from_list)

# Using `torch.empty`: Creates a tensor with uninitialized data.
# It allocates memory of the given shape and returns whatever was already in that memory location.
uninitialized_tensor = torch.empty(2, 3)
print("\nUninitialized tensor (from empty):\n", uninitialized_tensor)

# Using `torch.zeros` and `torch.ones`: Create tensors filled with 0s or 1s.
zeros_tensor = torch.zeros(2, 3)
ones_tensor = torch.ones(2, 3)
print("\nZeros tensor:\n", zeros_tensor)
print("Ones tensor:\n", ones_tensor)

# Using `torch.full`: Creates a tensor of a given shape, filled with a specified value.
fives_tensor = torch.full((2, 3), 5)
print("\nTensor filled with 5s (from full):\n", fives_tensor)

# Using `torch.rand`: Creates a tensor with random values from a uniform distribution on [0, 1).
# Useful for initializing weights in neural networks.
random_tensor = torch.rand(2, 3)
print("\nRandom tensor (from rand):\n", random_tensor)

# To ensure reproducibility, set the random seed before creating the tensor.
torch.manual_seed(42)
reproducible_random_tensor = torch.rand(2, 3)
print("Reproducible random tensor (seed=42):\n", reproducible_random_tensor)

# Using `torch.arange`: Creates a 1D tensor with values in a given range and a step.
range_tensor = torch.arange(start=0, end=10, step=2)
print("\nTensor from arange (0 to 10, step 2):", range_tensor)

# Using `torch.eye`: Creates a 2D identity matrix (ones on the diagonal, zeros elsewhere).
identity_matrix = torch.eye(4)
print("\nIdentity matrix (4x4):\n", identity_matrix)

# ======================================================================================
# 2. Tensor Properties (Shape & Data Type)
# ======================================================================================
print("\n--- 2. Tensor Properties ---")

properties_tensor = torch.tensor([[10, 20, 30], [40, 50, 60]])

# The `.shape` attribute returns the size of the tensor along each dimension.
print(f"\nShape of tensor: {properties_tensor.shape}")  # Expected: torch.Size([2, 3])

# The `.dtype` attribute shows the data type of the elements.
print(f"Data type of tensor: {properties_tensor.dtype}")  # Expected: torch.int64

# You can create a tensor with a specific data type.
tensor_with_float64 = torch.tensor([1, 2, 3], dtype=torch.float64)
print(f"\nFloat tensor's dtype: {tensor_with_float64.dtype}")

# The `.to()` method can cast a tensor to a new data type.
int_to_float_tensor = properties_tensor.to(torch.float32)
print(f"Original dtype: {properties_tensor.dtype}, Converted dtype: {int_to_float_tensor.dtype}")

# ======================================================================================
# 3. Creating Tensors from Other Tensors (Sharing Properties)
# ======================================================================================
print("\n--- 3. Creating Tensors with Inherited Properties ---")

# These methods create new tensors that inherit the shape and dtype of an existing tensor.
tensor_for_likeness = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
print(f"\nTemplate tensor shape: {tensor_for_likeness.shape}, dtype: {tensor_for_likeness.dtype}")

# Create tensors with the same properties as the template tensor.
zeros_like_tensor = torch.zeros_like(tensor_for_likeness)
rand_like_tensor = torch.rand_like(tensor_for_likeness)

print("Zeros-like tensor:\n", zeros_like_tensor)
print("Random-like tensor:\n", rand_like_tensor)

# ======================================================================================
# 4. Mathematical & Logical Operations
# ======================================================================================
print("\n--- 4. Mathematical & Logical Operations ---")

# --- Scalar Operations ---
print("\n-- Scalar Operations --")
tensor_for_scalar_ops = torch.rand(2, 2)
print("Original tensor:\n", tensor_for_scalar_ops)
print("Addition (+ 2):\n", tensor_for_scalar_ops + 2)
print("Multiplication (* 2):\n", tensor_for_scalar_ops * 2)

# --- Element-wise Tensor Operations ---
print("\n-- Element-wise Operations --")
elementwise_tensor_a = torch.tensor([[1, 2], [3, 4]])
elementwise_tensor_b = torch.tensor([[5, 6], [7, 8]])
print("Tensor A:\n", elementwise_tensor_a)
print("Tensor B:\n", elementwise_tensor_b)
print("A + B:\n", elementwise_tensor_a + elementwise_tensor_b)
print("A * B (Element-wise):\n", elementwise_tensor_a * elementwise_tensor_b)

# --- Reduction Operations (reduce a tensor to a single value) ---
print("\n-- Reduction Operations --")
tensor_for_reduction = torch.tensor([[1., 5., 2.], [8., 3., 6.]])
print("Original tensor for reduction:\n", tensor_for_reduction)
print("Sum of all elements:", torch.sum(tensor_for_reduction))
print("Mean of all elements:", torch.mean(tensor_for_reduction))
print("Sum along columns (dim=0):", torch.sum(tensor_for_reduction, dim=0))
print("Max value along rows (dim=1):", torch.max(tensor_for_reduction, dim=1).values)
print("Index of max value (flattened):", torch.argmax(tensor_for_reduction))

# --- Matrix Operations ---
print("\n-- Matrix Operations --")
matrix_for_mul_a = torch.randint(size=(2, 3), low=0, high=10)
matrix_for_mul_b = torch.randint(size=(3, 4), low=0, high=10)
print("Matrix A (2x3):\n", matrix_for_mul_a)
print("Matrix B (3x4):\n", matrix_for_mul_b)
# Matrix Multiplication: (m x n) @ (n x p) -> (m x p)
matrix_mul_result = torch.matmul(matrix_for_mul_a, matrix_for_mul_b)
print("Matrix Multiplication (A @ B):\n", matrix_mul_result)

# --- Special Functions (Common in ML) ---
print("\n-- Special Functions --")
tensor_for_activation = torch.arange(-3., 4., dtype=torch.float32)
print("Original tensor for activations:", tensor_for_activation)
print("Sigmoid:", torch.sigmoid(tensor_for_activation))
print("ReLU (Rectified Linear Unit):", torch.relu(tensor_for_activation))

tensor_for_softmax = torch.rand(1, 4)
print("\nOriginal tensor for softmax:", tensor_for_softmax)
print("Softmax (probabilities):", torch.softmax(tensor_for_softmax, dim=1))

# ======================================================================================
# 5. In-place Operations
# ======================================================================================
print("\n--- 5. In-place Operations ---")
# In-place operations modify the tensor directly, saving memory. They end with an underscore `_`.
tensor_for_inplace_op = torch.ones(2, 3)
tensor_to_add = torch.full((2, 3), 5)
print("Original tensor:\n", tensor_for_inplace_op)

tensor_for_inplace_op.add_(tensor_to_add) # Modifies the tensor directly
print("Tensor after in-place addition of 5s:\n", tensor_for_inplace_op)

# ======================================================================================
# 6. Copying vs. Referencing Tensors
# ======================================================================================
print("\n--- 6. Copying vs. Referencing Tensors ---")
# Simple assignment creates a reference. Modifying one affects the other.
original_tensor_ref = torch.ones(3)
referenced_tensor = original_tensor_ref # Both variables point to the same memory
referenced_tensor[0] = 999
print(f"Original tensor changed after modifying its reference: {original_tensor_ref}")

# Use .clone() to create a true, independent copy.
original_tensor_clone = torch.ones(3)
cloned_tensor = original_tensor_clone.clone() # `cloned_tensor` is a separate object
cloned_tensor[0] = 555
print(f"\nOriginal tensor is unchanged: {original_tensor_clone}")
print(f"Cloned tensor was modified independently: {cloned_tensor}")

# ======================================================================================
# 7. Reshaping Tensors
# ======================================================================================
print("\n--- 7. Reshaping Tensors ---")
tensor_for_reshaping = torch.arange(1, 17) # 16 elements
print("Original 1D tensor with 16 elements:\n", tensor_for_reshaping)

# Reshape to a 4x4 matrix. The total number of elements must remain the same.
reshaped_4x4_tensor = tensor_for_reshaping.reshape(4, 4)
print("\nReshaped to 4x4:\n", reshaped_4x4_tensor)

# Flatten collapses all dimensions into one.
flattened_tensor = reshaped_4x4_tensor.flatten()
print("\nFlattened back to 1D:", flattened_tensor)

# Unsqueeze adds a new dimension of size 1, useful for adding a batch dimension.
sample_image_tensor = torch.rand(28, 28) # e.g., a single grayscale image
print(f"\nOriginal image shape: {sample_image_tensor.shape}")
batched_image_tensor = sample_image_tensor.unsqueeze(0) # Add batch dimension
print(f"With batch dimension: {batched_image_tensor.shape}") # Shape is now (1, 28, 28)

# ======================================================================================
# 8. NumPy and PyTorch Bridge
# ======================================================================================
print("\n--- 8. NumPy and PyTorch Bridge ---")
# Tensors on the CPU and NumPy arrays can share the underlying memory location.
# This means conversions are fast, but changes to one will affect the other.

# Convert tensor to NumPy array
tensor_for_numpy_conversion = torch.tensor([1, 2, 3])
converted_numpy_array = tensor_for_numpy_conversion.numpy()
print(f"NumPy array: {converted_numpy_array}, Type: {type(converted_numpy_array)}")

# Convert NumPy array to tensor
numpy_array_for_tensor_conversion = np.array([4, 5, 6])
converted_tensor = torch.from_numpy(numpy_array_for_tensor_conversion)
print(f"Tensor from NumPy: {converted_tensor}, Type: {type(converted_tensor)}")

# Demonstrate shared memory: modifying the tensor also changes the NumPy array
converted_tensor[0] = 99
print(f"\nOriginal NumPy array after modifying tensor: {numpy_array_for_tensor_conversion}")

# ======================================================================================
# 9. CPU vs. GPU Speed Comparison
# ======================================================================================
print("\n--- 9. CPU vs. GPU Speed Comparison ---")
# GPUs excel at large-scale parallel computations like matrix multiplication.
# For small tensors, the overhead of moving data can make the CPU faster.

print("\n--- 9. CPU vs. GPU Speed Comparison ---")

if device.type != 'cpu': # This will be TRUE on your Mac
    size = 4096
    print(f"\nComparing CPU vs {device.type.upper()} for {size}x{size} matrix multiplication...")

    # -- CPU Computation --
    cpu_matrix_a = torch.rand(size, size, device='cpu')
    cpu_matrix_b = torch.rand(size, size, device='cpu')
    
    start_time = time.time()
    result_cpu = torch.matmul(cpu_matrix_a, cpu_matrix_b)
    cpu_time = time.time() - start_time
    print(f"CPU Time: {cpu_time:.4f} seconds")

    # -- GPU Computation --
    gpu_matrix_a = cpu_matrix_a.to(device)
    gpu_matrix_b = cpu_matrix_b.to(device)

    # First operation on GPU can have some startup overhead, so we sync here.
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
        
    start_time = time.time()
    result_gpu = torch.matmul(gpu_matrix_a, gpu_matrix_b)
    
    # Wait for the GPU operation to complete before stopping the timer
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

    gpu_time = time.time() - start_time
    print(f"{device.type.upper()} Time: {gpu_time:.4f} seconds")
    
    if gpu_time > 0:
        print(f"Speedup ({device.type.upper()}): {cpu_time / gpu_time:.2f}x")
else:
    print("\nGPU not available, skipping speed comparison.")