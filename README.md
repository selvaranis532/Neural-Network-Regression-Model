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
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 8)
        self.fc2 = nn.Linear(8, 10)
        self.fc3 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.history = {'loss': []}

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  
        return x


# Initialize the Model, Loss Function, and Optimizer

sel_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sel_brain.parameters(), lr=0.001)

def train_model(sel_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = criterion(sel_brain(X_train), y_train)
        loss.backward()
        optimizer.step()

        sel_brain.history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

       

```
## Dataset Information

<img width="278" height="468" alt="image" src="https://github.com/user-attachments/assets/372ab362-14a4-4bbe-9f2f-d4673b4e371b" />

## OUTPUT

<img width="1049" height="337" alt="image" src="https://github.com/user-attachments/assets/99b19340-cec9-4efb-97e4-87c1ee454f81" />


### Training Loss Vs Iteration Plot

<img width="833" height="633" alt="image" src="https://github.com/user-attachments/assets/ce3c2340-2d8e-4cd5-bc8e-c03fb65049ff" />



### New Sample Data Prediction

<img width="1350" height="148" alt="image" src="https://github.com/user-attachments/assets/99b9c49d-fcd2-4da2-83cb-0def7941a3f7" />



## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
