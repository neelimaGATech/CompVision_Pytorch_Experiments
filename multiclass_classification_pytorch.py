#Define and Use a Neural Network using PyTorch for Multi Class Classification

import torch
import torch.nn as nn
import numpy as np

class CustomNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomNeuralNet, self).__init__()

        # define / add custom layers to create a customized NN
        # input layer : input size and hidden size
        self.linear1 = nn.Linear(input_size, hidden_size)
        # activation function after the linear layer
        self.relu = nn.ReLU()
        #output layer that gets hidden layer size number of inputs and produces num_class count of outputs
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        # no softmax needed at the end because we will you
        # pytorch cross entropy which does its own softmax
        return out

model = CustomNeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
# For binary classification we can change the output layer to output just 1 value
# based on the probability value it is termed as True or False
# Example is the image that of a dog - output would be a probability and that gets translated to a
# 1 or 0 / True or False
criterion = nn.CrossEntropyLoss()   # applies softmax on its own


