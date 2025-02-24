import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetworkActivation1(nn.Module):
    # constructor
    # design the layers of your neural network here
    # if binary classifier we do not need to provide the output size
    def __init__(self, input_size, hidden_size):
        #call super constructor
        super(NeuralNetworkActivation1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()



    # method to overload in order to
    def forward(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.linear2(output)
        output = self.sigmoid(output)
        return output

class NeuralNetworkActivation2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetworkActivation2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        #out = torch.relu(self.linear1(x))
        #out = torch.sigmoid(self.linear2(out))

        out = F.relu(self.linear1(x))
        out = F.sigmoid(self.linear2(out))
        return out


