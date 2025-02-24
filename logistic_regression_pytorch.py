# Steps
# 1. Design model (input size, output size, forward pass)
# 2. Construct loss and optimizers
# 3, Create training loop:
#   - forward pass : computer the predicted output and hence the loss using it
#   - backward pass: to calculate auto- gradients
#   - update weights or parameters using gradients

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# prepare data
# load the binary classification training dataset for breast cancer
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

n_samples, n_features = X.shape
#print(n_samples, n_features)
print(y.shape)
#print(y[:100])

#split the dsta
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the data values - data values will have mean zero
# recommended for logistic regression
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#convert to torch tensors because we wish to use pytorch library
X_train = torch.from_numpy(X_train.astype(np.float32))

X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))
#print(y_train.shape)

# reshape y tesors to column vectors
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)
#print(y_train.shape)


# model
# f = wx + b, sigmoid at the end
class LogisticRegression(nn.Module):

    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1 )

    def forward(self, x):
        y_pred = torch.sigmoid((self.linear(x)))
        return y_pred



model = LogisticRegression(n_features)
# loss and optimizer
# Binary cross entropy loss used for regression
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# training loop
num_epochs = 100

for epoch in range(num_epochs):
    #forward pass and calculate loss
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    # backward pass to auto calculate gradients
    loss.backward()

    #update weights
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch+1}, loss: {loss.item():.4f}')

#evaluate the model, these calculations should not be part of grad history
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'Accuracy: {acc*100:.2f}%')




