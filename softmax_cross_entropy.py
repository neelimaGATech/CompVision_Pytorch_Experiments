# USe of Softmax and Cross Entropy Loss

import torch
import torch.nn as nn
import numpy as np

# custom defined softmax function using numpy methods
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print('softmax numpy: ', outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim=0) # computes along the first axis i.e. rows
print('softmax torch: ', outputs)

def cross_entropy(actual, predicted):
    loss = - np.sum(actual * np.log(predicted))
    return loss

# true labels
Y = np.array([0, 1, 0])
y_pred_good_softmax_outputs = np.array([0.2, 0.7, 0.1])
y_pred_bad_softmax_outputs = np.array([0.7, 0.2, 0.1])

l1 = cross_entropy(Y, y_pred_good_softmax_outputs)
l2 = cross_entropy(Y, y_pred_bad_softmax_outputs)

print(f'Loss 1: {l1:.4f}')
print(f'Loss 2: {l2:.4f}')

# Cross entroy usig Pytorch
# Pytorch cross entopy loss applies softmax on its own
# Y has class labels
# Y_pred would have raw labels (no softmax applied)
loss = nn.CrossEntropyLoss()

# class 0 is the actual class
Y = torch.tensor([0])
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[1.0, 2.0, 0.1]])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)
print(f'Loss 1: {l1:.4f}')
print(f'Loss 2: {l2:.4f}')

# get predictions made by model based on imaginary outputs stored in Y_pred_good and Y_pred_bad
_ , prediction1 = torch.max(Y_pred_good, 1)
_ , prediction2 = torch.max(Y_pred_bad, 1)

print(f'Prediction 1: {prediction1}')
print(f'Prediction 2: {prediction2}')