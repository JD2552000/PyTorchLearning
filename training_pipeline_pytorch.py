# Step 1: Import Necessary Libraries
#
# We import libraries for data manipulation (pandas), numerical operations (numpy),
# machine learning utilities (scikit-learn), and deep learning (PyTorch).
# ==============================================================================
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# Step 2: Load and Prepare the Dataset
#
# We will use the Breast Cancer Wisconsin (Diagnostic) dataset. This is a classic
# binary classification problem: predict whether a tumor is malignant or benign.
# ==============================================================================

# Load the dataset from a URL into a pandas DataFrame
data_url = 'https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv'
breast_cancer_df = pd.read_csv(data_url)

print("Initial data shape:", breast_cancer_df.shape)

# Data Cleaning: The 'id' and 'Unnamed: 32' columns are not useful features
# for prediction, so we drop them from our DataFrame.
breast_cancer_df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

print("Data shape after dropping unnecessary columns:", breast_cancer_df.shape)

# ==============================================================================
# Step 3: Split Data into Features (X) and Target (y) and then into Training/Testing Sets
#
# - X (Features): All columns except the first one ('diagnosis'). These are the inputs to our model.
# - y (Target): The first column ('diagnosis'), which we want to predict.
# - train_test_split: We divide the data to train the model on one part and test its performance
#   on another, unseen part. `random_state` ensures the split is the same every time we run the code.
# ==============================================================================

# The first column 'diagnosis' is our target, the rest are features.
features = breast_cancer_df.iloc[:, 1:]
target = breast_cancer_df.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# ==============================================================================
# Step 4: Feature Scaling
#
# Neural networks perform best when input features are on a similar scale.
# StandardScaler transforms the data to have a mean of 0 and a standard deviation of 1.
# - We use `fit_transform` on the training data to learn the scaling parameters (mean, std).
# - We use only `transform` on the test data to apply the same scaling learned from the training data,
#   preventing any information from the test set from "leaking" into the training process.
# ==============================================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# Step 5: Label Encoding
#
# The target variable 'diagnosis' is categorical ('M' for malignant, 'B' for benign).
# The model needs numerical inputs, so we convert these labels into numbers (e.g., 1 and 0).
# ==============================================================================

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# ==============================================================================
# Step 6: Convert Data from NumPy Arrays to PyTorch Tensors
#
# PyTorch operates on its own data structure called tensors, which are similar to NumPy arrays
# but are optimized for GPU computations and automatic differentiation (autograd).
# We also ensure the data type is float32, the standard for deep learning models.
# The target tensors are reshaped to be column vectors to match the model's output shape.
# ==============================================================================

X_train_tensor = torch.from_numpy(X_train_scaled).float()
X_test_tensor = torch.from_numpy(X_test_scaled).float()
y_train_tensor = torch.from_numpy(y_train_encoded).float().view(-1, 1)
y_test_tensor = torch.from_numpy(y_test_encoded).float().view(-1, 1)

print("\nShape of training features tensor:", X_train_tensor.shape)
print("Shape of training labels tensor:", y_train_tensor.shape)

# ==============================================================================
# Step 7: Define the Neural Network Model
#
# We define our model as a Python class. This simple model has one layer.
# It takes the input features and produces a single output probability.
# ==============================================================================
class BreastCancerClassifier:
    def __init__(self, num_features):
        # Learnable Parameters:
        # - Weights: One weight for each input feature. The shape is (num_features, 1) because
        #   we want to map the input features to a single output logit.
        # - Bias: A single value that is added to the weighted sum.
        # `requires_grad=True` tells PyTorch to track these tensors for gradient calculation.
        self.weights = torch.rand(num_features, 1, dtype=torch.float32, requires_grad=True)
        self.bias = torch.zeros(1, dtype=torch.float32, requires_grad=True)

    def forward(self, input_features):
        # The forward pass defines how the model computes its output.
        # 1. Linear Transformation: We compute a weighted sum of the inputs and add the bias.
        #    This is the core linear equation: z = Xw + b
        linear_combination = torch.matmul(input_features, self.weights) + self.bias
        
        # 2. Activation Function: We apply the sigmoid function to the result.
        #    The sigmoid function squashes any real number into the range (0, 1), which is
        #    perfect for representing the probability of the positive class in binary classification.
        predicted_probability = torch.sigmoid(linear_combination)
        return predicted_probability
    
    def loss_function(self, predicted_prob, true_labels):
        # Binary Cross-Entropy (BCE) Loss: This is the standard loss function for binary classification.
        # It measures the difference between the predicted probability and the true label (0 or 1).
        
        # Clamp function: We clamp the predictions to avoid `log(0)`, which is undefined and would
        # result in NaN (Not a Number) values, crashing the training.
        epsilon = 1e-7
        predicted_prob = torch.clamp(predicted_prob, epsilon, 1 - epsilon)

        # BCE Formula: -[y*log(p) + (1-y)*log(1-p)]
        # It penalizes the model heavily when it is confidently wrong.
        loss = -(true_labels * torch.log(predicted_prob) + (1 - true_labels) * torch.log(1 - predicted_prob)).mean()
        return loss

# ==============================================================================
# Step 8: Define Training Hyperparameters
# ==============================================================================
learning_rate = 0.1  # Controls how much we adjust the model's parameters during each update.
epochs = 50          # The number of times the model will see the entire training dataset.

# ==============================================================================
# Step 9: The Training Pipeline
# ==============================================================================

# Instantiate the model
num_features = X_train_tensor.shape[1]
classifier_model = BreastCancerClassifier(num_features)

print("\n--- Starting Training ---")
# The training loop iterates for a specified number of epochs.
for epoch in range(epochs):
    # 1) Forward Pass: Compute the model's prediction for the current parameters.
    y_predicted = classifier_model.forward(X_train_tensor)

    # 2) Calculate Loss: Measure how wrong the model's predictions are.
    loss = classifier_model.loss_function(y_predicted, y_train_tensor)

    # 3) Backward Pass: This is where PyTorch's magic happens.
    #    `loss.backward()` automatically computes the gradient of the loss with respect
    #    to all parameters that have `requires_grad=True` (i.e., our weights and bias).
    loss.backward()

    # 4) Update Parameters (Gradient Descent):
    #    We adjust the weights and bias in the opposite direction of their gradients to minimize the loss.
    #    `torch.no_grad()` is used as a context manager to turn off gradient tracking for this block,
    #    as we are manually updating the parameters and don't want this operation to be part of the
    #    computation graph for the next iteration.
    with torch.no_grad():
        classifier_model.weights -= learning_rate * classifier_model.weights.grad
        classifier_model.bias -= learning_rate * classifier_model.bias.grad

    # Zero the Gradients: It's crucial to reset the gradients to zero after each epoch.
    # PyTorch accumulates gradients by default, so failing to zero them would cause the gradients
    # from previous epochs to interfere with the current one.
    classifier_model.weights.grad.zero_()
    classifier_model.bias.grad.zero_()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")

print("--- Training Finished ---")

# ==============================================================================
# Step 10: Evaluation
#
# Now we test the trained model on the unseen test data to see how well it generalizes.
# ==============================================================================
with torch.no_grad(): # We don't need to calculate gradients for evaluation.
    # Get the model's probability predictions on the test set.
    test_predictions_prob = classifier_model.forward(X_test_tensor)
    
    # Convert probabilities to binary class predictions (0 or 1) using a 0.5 threshold.
    # If probability > 0.5, predict class 1 (Malignant); otherwise, predict 0 (Benign).
    test_predictions_class = (test_predictions_prob > 0.5).float()
    
    # Calculate accuracy by comparing predicted classes to the true classes.
    # (test_predictions_class == y_test_tensor) creates a tensor of 1s (correct) and 0s (incorrect).
    # .mean() calculates the proportion of correct predictions.
    accuracy = (test_predictions_class == y_test_tensor).float().mean()
    print(f"\nAccuracy on Test Data: {accuracy.item() * 100:.2f}%")