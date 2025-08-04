#We will build a simple neural network using nn.Module where we will have one neural and 5 inputs

import torch
import torch.nn as nn

#Create neural network
class SimpleModelNN(nn.Module):
    def __init__(self, num_features):
        #num_features is the number of inputs to the neural network

        #We will invoke the constructor of the parent class using super()
        super().__init__()

        self.linear = nn.Linear(num_features, 1)  # Linear layer with num_features inputs and 1 output
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation function

    def forward(self, features):
        
        out = self.linear(features)  # Apply linear transformation
        out = self.sigmoid(out)  # Apply sigmoid activation function
        return out  # Return the output of the neural network   
    
#Create a simple dataset
features = torch.rand(10, 5)  # 10 samples, each with 5 features

#Create a model
model = SimpleModelNN(features.shape[1])  # Pass the number of features to the model
#Forward pass through the model
y_pred = model(features)  # Get the output of the model by calling forward function.
#We dont need to used mode.forward explicitly as it is overrided by magic functions concept in python.
#As soon as we create object of the class and we call it as function then the forward method will be called.

#Show weights
print(model.linear.weight)  # Access the weights of the linear layer
print(model.linear.bias)  # Access the bias of the linear layer

#Now if we want to visualize the model, we can use a library called torchinfo which is extension of torch.
from torchinfo import summary

summary(model, input_size=(10, 5))  # Print the summary of the model with input size (10 samples, each with 5 features)