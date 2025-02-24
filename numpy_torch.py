import torch
import numpy as np

#torch tensor to numpy array
# Mark that this tensor would require automatic calculation of gradients
a = torch.ones(5, requires_grad=True)
print(a.dtype)

b = a.numpy()
print(b.dtype)

#numpy array to torch tensor
c = np.ones(5)
print(c)
d = torch.from_numpy(c)
print(d)