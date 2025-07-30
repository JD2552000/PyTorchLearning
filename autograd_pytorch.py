import torch

x = torch.tensor(3.0, requires_grad=True)
#Through requires_grad=True, we are telling PyTorch to track all operations on this tensor and that we will be calculating derivates of this tensor in future
# and then pytorch will create a computation graph internally using mathematical functions and inputs so it will store it in memory
# and when during backpropagation for calculating derivate it will realize what functions were using during forward pass and then calculate derivate.

y = x**2
#Here, y is a function of x, so we can calculate the derivative of y with respect to x.
print(x)
print(y)

#Differentiation using autograd
y.backward() #calculates the derivative of y with respect to x

#To check the derivate and its output.
x.grad

a = torch.tensor(3.0, requires_grad=True)
b = a**2
c = torch.sin(b)

c.backward()  # This will compute the gradient of c with respect to a
print(a.grad)  # This will print the gradient of c with respect to a

#The main thing is pytorch makes a computation graph using input and functions so here the input are the leaf nodes
# The derivate of wich we want to calculate is root node and between them are the intermediate nodes and we cannot calculate the derivate
# of the intermediate nodes as derivative calculation always start from root so y.backward() will not work.