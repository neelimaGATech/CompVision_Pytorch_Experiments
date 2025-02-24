import torch
import torch.nn as nn

input_size = 6
class CustomModel(nn.Module):
    def __init__(self, input_size):
        super(CustomModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

model = CustomModel(input_size)
# train

for param in model.parameters():
   print(param)
# OPTION 1
# save the entire model

FILE = "model.pth"
#torch.save(model, FILE)

# load the saved model
#model1 = torch.load(FILE)
#model1.eval()



# OPTION 2 - PReferred way
# only save parameters by saving state dictionary
#torch.save(model.load_state_dict(), PATH)
torch.save(model.state_dict(), FILE)
#model = Model(*args, **kargs)
#model.load_state_dict(torch.load(PATH))

loaded_model = CustomModel(input_size)
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.eval()

for param in loaded_model.parameters():
   print(param)

