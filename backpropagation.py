import torch

x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)

# forward pass

y_hat = w * x
loss = (y_hat - y) ** 2

    # perform backpropagation
loss.backward()
print(w.grad.dtype)

#update w with value of w.grad
learning_rate = 0.01
epochs = 100
for step in range(epochs):
    w=w - learning_rate * w.grad
    print(w.grad.dtype)