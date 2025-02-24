import torch
import torch.nn as nn
from save_model import CustomModel

input_size = 6
class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

model = CustomModel(input_size)
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(optimizer.state_dict())
#print(model.state_dict())

# create a checkpoint - dictionary of current state of training
checkpoint = {
    "epoch": 90,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
}


#torch.save(checkpoint, "checkpoint.pth")

# loaded the checkpoint
loaded_checkpoint = torch.load("checkpoint.pth")
epoch = loaded_checkpoint["epoch"]
loaded_model = CustomModel(6)
optimizer = torch.optim.SGD(loaded_model.parameters(), lr=learning_rate)

loaded_model.load_state_dict(loaded_checkpoint["model_state"])
optimizer.load_state_dict(loaded_checkpoint["optimizer_state"])

print(optimizer.state_dict())

