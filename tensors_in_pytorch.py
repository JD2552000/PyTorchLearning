import torch
print(torch.__version__)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS GPU")
else:
    device = torch.device("cpu")
    print("MPS not available, using CPU")

#Creating & Manipulating Tensors with the help of PyTorch

#Create tensor using empty function and give the shape to that function.
a = torch.empty(2,3)
"""So empty function create a space of the needed shape in memory and assigns the value
which is already present previously at that location.It does not create new values."""

#To check the type
type(a)

#Creating tensor using zeros function. All values are initialized to 0.
torch.zeros(2,3)

#Creating tensor using ones function. All values are initialized to 1.
torch.ones(2,3)

#Using rand function. It creates tensor of the given shape with random values between 0 and 1.
# This can be used in weight initialization in neural network.
torch.rand(2,3)

"""The main thing is each time we run the rand method the tensor will initilize with different random values
and not the same as before it was initilized.

Now to use the rand function and to make sure each time the random values remain as it is we need to make use of
seed.This also assures reproducibility"""
torch.manual_seed(100)
torch.rand(2,3)

#Another way is using torch.Tensor.It can produce custom values by giving python iterables.
torch.tensor([[1,2,3],[4,5,6]])

"""Above are 5 main methods, some of the other methods are as follows"""
#Using arange method, same as numpy where we provide the range and step along with the range
#So, here tensor is created from 0 to 10 with skipping 2 steps i.e [0,2,4,6,8]
print("Using arange ->", torch.arange(0,10,2))