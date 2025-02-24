import torch

x = torch.rand(5,3)
# print(x)
# print(x[:, 0])
# print(x[2, :])
# print(x[1,1].item())

# Reshape the tensor dimensions using view method
y = torch.rand(4,4)
z = y.view(16)
print("y =", y)
print("Y Reshaped to 1 dimension", z)
# only give one target dimension for reshape
q = y.view(-1,8)
print(q.size())

