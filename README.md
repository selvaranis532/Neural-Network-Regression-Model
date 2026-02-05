# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: SELVARANI S
### Register Number:212224040301
```
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("/dataset.csv")

X = torch.tensor(data.iloc[:,0].values, dtype=torch.float32).view(-1,1)
Y = torch.tensor(data.iloc[:,1].values, dtype=torch.float32).view(-1,1)

X = X / X.max()

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,1)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(ai_brain.parameters(), lr=0.01)

def train_model(ai_brain, X, Y, criterion, optimizer, epochs=500):
    losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = ai_brain(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

losses = train_model(ai_brain, X, Y, criterion, optimizer)

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss vs Epochs")
plt.show()


```
## Dataset Information

<img width="278" height="468" alt="image" src="https://github.com/user-attachments/assets/372ab362-14a4-4bbe-9f2f-d4673b4e371b" />

## OUTPUT

### Training Loss Vs Iteration Plot

<img width="759" height="580" alt="image" src="https://github.com/user-attachments/assets/83fd499d-5f50-483e-abdc-6f1d1973a814" />


### New Sample Data Prediction

<img width="643" height="131" alt="image" src="https://github.com/user-attachments/assets/8eaf885a-ec07-426f-a730-8ebb8246ee49" />


## RESULT

Include your result here
