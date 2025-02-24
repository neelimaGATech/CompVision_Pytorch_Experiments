# Steps
# 1. Design model (input size, output size, forward pass)
# 2. Construct loss and optimizers
# 3, Create training loop:
#   - forward pass : computer the predicted output and hence the loss using it
#   - backward pass: to calculate auto- gradients
#   - update weights or parameters using gradients

import torch
import torch.nn as nn

#for data transformations use numpy
import numpy as np
#dataset
from sklearn import datasets
import matplotlib.pyplot as plt

# Prepare data
# generate samples using numpy methods
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

#Create Tensors using numpy arrays
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

X_test = torch.tensor([5], dtype=torch.float32)

# y is a row vector , convert it to column vector
y = y.view(y.shape[0], 1)

# convert y to column vector

# Design model
n_samples, n_features = X.shape
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    #forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # backward pass
    loss.backward()

    #udpate weights
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'epoch:{epoch+1}, loss:{loss.item():.4f}')
        #print(epoch, loss.item())


#plot all predicted values
predicted = model(X).detach().numpy() # keep this operation detached from the computational graph that is maintained
plt.plot(X_numpy, y_numpy, 'ro', label='data')
plt.plot(X_numpy,predicted, 'b-', label='prediction' )
plt.show()


output = model(X_test)
print(output.shape)
print(f'model output for input 5 : {output.item():.3f}')



