#Here we will build the neural network that includes hidden layer and output layer too.
#We will have 5 inputs in input layer and 3 inputs in hidden layer and 1 output in output layer.
#So total paramter in neural network will be 5*3 + 3*1 + 3 + 1= 15 + 3 + 3 + 1 = 22 parameters.
#And we will be using Relu as an activation function in hidden layer and Sigmoid in output layer.
#The main thing is that when the neural network keeps on growing in complexity then the forward pass code will
# keep on grwoing and there should some extended function or class to minimize the code here.

import torch
import torch.nn as nn
from torchinfo import summary



