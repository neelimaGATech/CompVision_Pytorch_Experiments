import numpy as np
from numpy import dtype

# Linear Regression
#f = w * x, f = 2 *x

# assume given values of input vector X and output vector Y
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

# Assume initial weights to be 0
w = 0.0

# model prediction - forward method
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_pred):
    return ((y_pred - y) ** 2).mean()

#gradient of loss
# MSE 1/N * (w*x - y) ** 2
# dJ/dw = 1/N * 2x * (w*x - y)
# gradient calculated manually
def gradient(x,y,y_pred):
    return np.dot(2*x, y_pred - y).mean()  # for division by N take mean

print(f'Prediciton before training f(5) = {forward(5):.3f}')

#Training
learning_rate = 0.01
num_iters = 100

for epoch in range(num_iters):
    y_pred = forward(X)

    l = loss(Y, y_pred)

    dw = gradient(X, Y, y_pred)

    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f'epoch {epoch+1}: w={w:.3f}, loss={l: .8f}')

print(f'Prediciton after training f(5) = {forward(5):.3f}')