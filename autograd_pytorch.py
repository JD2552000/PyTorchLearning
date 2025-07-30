import torch
import math

# ==============================================================================
# Section 1: Basic Autograd Example
#
# This section demonstrates the fundamental capability of PyTorch's autograd
# to compute the derivative of a simple function y = x^2.
# ==============================================================================

print("--- Section 1: Basic Autograd ---")

# Define a tensor 'x_basic' with an initial value of 3.0.
# 'requires_grad=True' tells PyTorch to track all operations on this tensor
# so that it can automatically compute gradients later.
x_basic = torch.tensor(3.0, requires_grad=True)

# Define a new tensor 'y_basic' as the square of 'x_basic'.
# PyTorch builds a computation graph where 'y_basic' is a result of an operation on 'x_basic'.
y_basic = x_basic ** 2

# The analytical derivative of y = x^2 is dy/dx = 2x.
# For x = 3, the gradient should be 2 * 3 = 6.
print(f"Initial Tensor (x): {x_basic}")
print(f"Resulting Tensor (y = x^2): {y_basic}")

# To compute the gradients, we call the .backward() method on the output tensor 'y_basic'.
# This initiates the backpropagation process from 'y_basic' through the computation graph.
y_basic.backward()

# The computed gradient (dy/dx) is stored in the .grad attribute of the input tensor 'x_basic'.
print(f"Computed Gradient (dy/dx at x=3): {x_basic.grad}\n")


# ==============================================================================
# Section 2: Autograd with the Chain Rule
#
# This section shows how autograd handles more complex functions by applying
# the chain rule. We compute the derivative of z = sin(x^2).
# The derivative dz/dx is 2x * cos(x^2).
# ==============================================================================

print("--- Section 2: Chain Rule ---")

# Define a tensor 'x_chain' and track its gradients.
x_chain = torch.tensor(4.0, requires_grad=True)

# Define intermediate tensor 'y_chain'.
# PyTorch tracks this operation as part of the graph.
y_chain = x_chain ** 2

# Define the final output tensor 'z_chain'.
z_chain = torch.sin(y_chain)

# The analytical derivative is dz/dx = dz/dy * dy/dx = cos(y) * 2x = cos(x^2) * 2x.
# For x = 4, the gradient is cos(16) * 8.
# math.cos(16) * 8 is approximately -7.66.
print(f"Initial Tensor (x): {x_chain}")
print(f"Final Tensor (z = sin(x^2)): {z_chain}")

# Perform backpropagation from the final output 'z_chain'.
z_chain.backward()

# The gradient dz/dx is stored in 'x_chain.grad'.
print(f"Computed Gradient (dz/dx at x=4): {x_chain.grad}")

# Note: By default, PyTorch only retains the gradients for leaf nodes of the
# computation graph that have requires_grad=True. Intermediate tensors like
# 'y_chain' will have their gradients calculated but not stored to save memory.
# Therefore, 'y_chain.grad' will be None.
print(f"Gradient of intermediate tensor 'y_chain' is: {y_chain.grad}\n")


# ==============================================================================
# Section 3: Gradients for a Simple Model (Manual vs. Automatic)
#
# This section demonstrates a common use case: computing gradients of a loss
# function with respect to model parameters (weights 'w' and bias 'b').
# It first calculates them manually and then verifies with autograd.
# ==============================================================================

print("--- Section 3: Gradients for a Simple Model ---")

# --- Manual Gradient Calculation ---

# Define scalar inputs and parameters
x_input_manual = torch.tensor(6.7)
true_label_manual = torch.tensor(0.0)
weight_manual = torch.tensor(1.0)
bias_manual = torch.tensor(0.0)

# Define a stable binary cross-entropy loss function
def binary_cross_entropy_loss(prediction, target):
    """
    Calculates the binary cross-entropy loss.
    Args:
        prediction (torch.Tensor): The predicted probability (between 0 and 1).
        target (torch.Tensor): The true label (0 or 1).
    Returns:
        torch.Tensor: The calculated loss.
    """
    epsilon = 1e-8  # A small value to prevent taking the log of zero
    # Clamp prediction to avoid log(0) or log(1) which are undefined or infinity
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))

# Manual Forward Pass
linear_output_manual = weight_manual * x_input_manual + bias_manual
prediction_manual = torch.sigmoid(linear_output_manual)
loss_manual = binary_cross_entropy_loss(prediction_manual, true_label_manual)
print(f"Manual Loss: {loss_manual}")

# Manual Backward Pass (using the chain rule to find derivatives)
# 1. Derivative of loss with respect to prediction
dloss_dprediction = (prediction_manual - true_label_manual) / (prediction_manual * (1 - prediction_manual))
# 2. Derivative of prediction (sigmoid) with respect to linear output
dprediction_dlinear = prediction_manual * (1 - prediction_manual)
# 3. Derivative of linear output with respect to weight
dlinear_dweight = x_input_manual
# 4. Derivative of linear output with respect to bias
dlinear_dbias = 1

# Combine derivatives using the chain rule to get the final gradients
dloss_dweight = dloss_dprediction * dprediction_dlinear * dlinear_dweight
dloss_dbias = dloss_dprediction * dprediction_dlinear * dlinear_dbias

print(f"Manual Gradient of loss w.r.t weight (dw): {dloss_dweight}")
print(f"Manual Gradient of loss w.r.t bias (db): {dloss_dbias}\n")


# --- Automatic Gradient Calculation with Autograd ---

# Define inputs and parameters, setting requires_grad=True for the parameters
# because we want to compute gradients with respect to them.
x_input_auto = torch.tensor(6.7)
true_label_auto = torch.tensor(0.0)
weight_auto = torch.tensor(1.0, requires_grad=True)
bias_auto = torch.tensor(0.0, requires_grad=True)

# Autograd Forward Pass (PyTorch tracks these operations to build the computation graph)
linear_output_auto = weight_auto * x_input_auto + bias_auto
prediction_auto = torch.sigmoid(linear_output_auto)
loss_auto = binary_cross_entropy_loss(prediction_auto, true_label_auto)
print(f"Autograd Loss: {loss_auto}")


# Autograd Backward Pass: This single call computes all gradients automatically.
loss_auto.backward()

# Gradients are automatically computed and stored in the .grad attribute of the parameters.
print(f"Autograd Gradient of loss w.r.t weight (dw): {weight_auto.grad}")
print(f"Autograd Gradient of loss w.r.t bias (db): {bias_auto.grad}\n")


# ==============================================================================
# Section 4: Gradients of a Tensor with Multiple Elements
#
# This section shows how .backward() behaves when the input is a vector
# and the output is a scalar (e.g., the mean of a function of the vector).
# ==============================================================================

print("--- Section 4: Gradients of a Tensor with Multiple Elements ---")

# Define an input tensor with multiple elements.
x_vector = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# Define the output 'y_mean' as the mean of the squares of the elements in 'x_vector'.
# y = (1/3) * (x_1^2 + x_2^2 + x_3^2)
y_mean = (x_vector ** 2).mean()
print(f"Input Vector (x): {x_vector}")
print(f"Result (y = mean(x^2)): {y_mean}")


# The partial derivative of y with respect to each x_i is dy/dx_i = (1/3) * 2 * x_i.
# So the gradient vector should be [2/3 * 1, 2/3 * 2, 2/3 * 3] = [0.66, 1.33, 2.0].
y_mean.backward()

# The gradients are stored in the .grad attribute of 'x_vector'.
print(f"Computed Gradient (dy/dx): {x_vector.grad}\n")


# ==============================================================================
# Section 5: Gradient Accumulation and Clearing
#
# PyTorch accumulates gradients by default. This means that every time .backward()
# is called, the new gradients are added to the existing ones. This is useful
# in scenarios like Recurrent Neural Networks (RNNs), but requires manual
# resetting in typical feedforward network training loops.
# ==============================================================================

print("--- Section 5: Gradient Accumulation and Clearing ---")

# Define a tensor and perform backpropagation
x_accum = torch.tensor(2.0, requires_grad=True)
y_accum_1 = x_accum ** 2
y_accum_1.backward() # Computes dy/dx = 2*x = 4
print(f"Gradient after first backward pass: {x_accum.grad}")

# If we run another backward pass on a new function, the gradients add up.
y_accum_2 = x_accum ** 3
y_accum_2.backward() # Computes dy/dx = 3*x^2 = 12
# The new gradient will be 4 (from before) + 12 (new) = 16
print(f"Gradient after second backward pass (accumulation): {x_accum.grad}")


# To prevent this accumulation, we must manually zero out the gradients
# using .grad.zero_(). This is typically done at the start of each training iteration.
x_accum.grad.zero_()
print(f"Gradient after clearing with .zero_(): {x_accum.grad}\n")


# ==============================================================================
# Section 6: Disabling Gradient Tracking
#
# Sometimes, we don't need PyTorch to track gradients, such as during model
# inference or when updating weights. This saves memory and computation.
# There are three common ways to achieve this.
# ==============================================================================

print("--- Section 6: Disabling Gradient Tracking ---")

x_track_off = torch.tensor(2.0, requires_grad=True)

# --- Option 1: .requires_grad_(False) ---
# This is an in-place operation that permanently disables gradient tracking for the tensor.
print(f"Before disabling: x.requires_grad = {x_track_off.requires_grad}")
x_track_off.requires_grad_(False)
print(f"After disabling with .requires_grad_(False): x.requires_grad = {x_track_off.requires_grad}")
try:
    y_no_grad_1 = x_track_off ** 2
    y_no_grad_1.backward() # This will raise a runtime error because tracking is off
except RuntimeError as e:
    print(f"Error when calling .backward(): {e}\n")


# --- Option 2: .detach() ---
# This creates a new tensor that shares the same data but is detached from the
# computation graph. The original tensor remains unchanged.
x_original = torch.tensor(2.0, requires_grad=True)
x_detached = x_original.detach()

print(f"Original tensor requires_grad: {x_original.requires_grad}")
print(f"Detached tensor requires_grad: {x_detached.requires_grad}")

y_original = x_original ** 2 # Still part of the graph
y_detached = x_detached ** 2 # Not part of the graph

y_original.backward() # This works because the original tensor tracks gradients
print(f"Gradient on original tensor: {x_original.grad}")

try:
    y_detached.backward() # This will fail because the detached tensor has no graph
except RuntimeError as e:
    print(f"Error on detached tensor: {e}\n")


# --- Option 3: with torch.no_grad(): ---
# This is a context manager that temporarily disables gradient tracking for all
# operations within its scope. It's the most common and recommended way to
# perform inference without tracking gradients.
x_context = torch.tensor(2.0, requires_grad=True)
print(f"Tensor requires_grad before 'no_grad' block: {x_context.requires_grad}")

with torch.no_grad():
    print("Inside 'with torch.no_grad()' block:")
    y_no_grad_2 = x_context ** 2
    # Any tensor created inside this block will have requires_grad=False
    print(f"  - New tensor 'y' requires_grad: {y_no_grad_2.requires_grad}")

print(f"Tensor requires_grad after 'no_grad' block: {x_context.requires_grad}")

try:
    y_no_grad_2.backward() # This will fail as it was created without a computation graph
except RuntimeError as e:
    print(f"Error on tensor created in 'no_grad' block: {e}")