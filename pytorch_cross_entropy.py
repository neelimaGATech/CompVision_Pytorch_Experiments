import torch
import torch.nn as nn
import numpy as np

loss = nn.CrossEntropyLoss()

# output tensor
Y = torch.tensor([2, 0 , 1])

# predicted outputs
# number of samples = 3, number of classes = 3, total elememts = 3x3 = 9
y_pred_good = torch.tensor([[0.2, 1.8, 2.5], [2.8, 1.9, 0.6],[0.2, 3.2, 1.7]])
y_pred_bad = torch.tensor([[0.2, 2.5, 1.8], [ 1.9, 2.8,0.6],[0.2, 1.7, 3.2]])

l1 = loss(y_pred_good, Y)
l2 = loss(y_pred_bad, Y)

print(l1)
print(l2)

print(l1.item())
print(l2.item())