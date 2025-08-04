#Building Artificial Neural Networks using PyTorch on FashinMNIST Dataset.
#This dataset has 70k image so first before going on GPU, currently here we will train it through CPU
#So, for that we will fetch 6k images from above dataset.

"""Architecure:
Input Layer: 784 features(neuron)
1st Hidden Layer: 128 neurons -> ReLU activation
2nd Hidden Layer: 64 neurons -> ReLU activation
Output Layer: 10 neurons -> Softmax activation"""

"""Worfklow:
-> DataLoader Object for both training and testing data.
-> Training Loop to train the model.
-> Evaluation our model on testing data."""

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#Set random seed for reproducibility so that if any other developer runs the same code we get exact result due to same tensors if rand is used.
torch.manual_seed(42)

#Load the read the dataset
df = pd.read_csv('fmnist_small.csv')
df.head()

#Fetch first 16 images from the dataset for visualization.
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
fig.suptitle('First 16 Images', fontsize=16)

#Plot the first 16 images from the dataset.
for i, ax in enumerate(axes.flat):
    imf = df.iloc[i, 1:].values.reshape(28, 28)  # Reshape the image data to 28x28
    ax.imshow(imf) # Display the image
    ax.axis('off') # Hide the axes
    ax.set_title(f'Label: {df.iloc[i, 0]}') # Set the title with the label

plt.tight_layout(rect=[0, 0, 1, 0, 0.96]) # Adjust layout to make room for the suptitle
plt.show()

#Perform the train-test split
X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scaling the features
"""We divided our train and test data by 255 because the pixel values range between 0 to 255 and in our dataset while there are 
0 pixel value, there are also higher pixel values like 142, 180 and we are training neural network and so its good if we give the input
in equal scale or range"""

X_train = X_train / 255.0
X_test = X_test / 255.0

#Create CustomeDataset class
class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32) # Features should be in Float
        self.labels = torch.tensor(labels, dtype=torch.long) #Labels should in Long dataset

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, index):
        return self.features[index], self.labels[index] #Index of features and labels to access them
    
#Create train Dataset Object
train_dataset = CustomDataset(X_train, y_train)

#Create test Dataset Object
test_dataset = CustomDataset(X_test, y_test)

#Create DataLoader for train and test dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True) #Shuffle the data for better training
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False) #No need to shuffle test data as it is prediction phase and not training phase.

#Define the Neural Network Class
class MyANN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )
    
    def forward(self, x):
        return self.model(x)
    
#Set the learning rate and the number of epochs
learning_rate = 0.1
epochs = 100

#Now, we need to instantiate the model, define the loss function and the optimizer
model = MyANN(num_features=X_train.shape[1]) #Number of features is 784

#Loss function defintion
criterion = nn.CrossEntropyLoss() #CrossEntropyLoss is used for multi-class classification problems

#Optimizer definition
optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Stochastic Gradient Descent optimizer with learning rate of 0.1

#Now we will perform the training loop
for epoch in range(epochs):

    total_epoch_loss = 0.0 #To keep track of the loss for each epoch

    for batch_features, batch_labels in train_loader:
        #batch_features means input data
        #batch_labels means original true data\
        #batch_features and batch_labels are tensors of size (batch_size, 784) and (batch_size,) respectively

        #Foward pass
        outputs = model(batch_features) #Pass the batch features through the model

        #Calculate loss
        loss = criterion(outputs, batch_labels)

        #Backpropagation
        #Clear the gradients first
        optimizer.zero_grad()
        loss.backward()

        #Update Paramters
        optimizer.step()

        total_epoch_loss += loss.item() #Add the loss of this batch to the total epoch loss
    
    #Print the Average Loss for all the epoch
    total_epoch_loss /= len(train_loader) #Average loss for the epoch
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_epoch_loss:.4f}') #Print the loss for the epoch

#Evaluation Code
"""The main requirement of the evaluation code is during the whole process there are some features in deep learning
that does not behave in similar manner during both the training and testing phase.
Foe example the dropouts, during training we simply drop some neuron but during testing phase
all the neurons are required so there should be some method to know that now we are in testing phase.

Another example is batch normlization, during train phase we calculate mean, median and other statistical properties but during
testing phase we take into account same mean and statistical properties so different behavior"""
total = 0.0
correct = 0.0 
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)

        total += batch_labels.shape[0]

        correct = correct + (predicted == batch_labels).sum().items()

print(correct/total)