# 1) design model - (input size, output size, layers of forward pass )
# 2) construct loss and optimizer function
# 3) Training loop:
#       - forward pass
#       - backward pass : gradients
#       - update weights
import torch
import torch.nn as nn

# modify the input and ouput to convert to tensors of 2 D shape (num of samples, num of features)
X = torch.tensor([[1],[2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# TEst value can not be a scalar so have to define a tensor for testing
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

#Weights not needed for pytorch model
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
input_size = n_features
output_size = n_features
model = nn.Linear(input_size, output_size)

# forward function
def forward_manual(x):
    return w * x

# MSE = 1/N *
def loss_manual(y, y_pred):
    return (y_pred - y ).pow(2).mean()



print(f'Prediciton before training f(5) = {model(X_test).item():.3f}')
#training
learning_rate = 0.01
iters_num = 100

# loss is a callable function from neural network directory
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(iters_num):
    #y_pred = forward(X)
    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward()
    # no manual weight update required just do optimization step
    #with torch.no_grad():
    #    w -= learning_rate * w.grad
    optimizer.step()

    # zero the gradiwents using optimizer
    optimizer.zero_grad()
    #w.grad.zero_()


    if epoch % 2 == 0:
        [w, b] = model.parameters()
        #first
        print(f'epoch: {epoch}: loss: {l:.3f}, w: {w[0][0].item():.3f}')

print(f'Prediciton after training f(5) = {model(X_test).item():.3f}')