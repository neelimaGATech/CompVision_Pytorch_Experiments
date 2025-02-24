import torch
import torch.nn as nn

class LinearRegression(nn.Module):

    # implement init method like a  constructor
    # call the constructoer of super class -- required
    def __init__(self, in_features, out_features):
        super(LinearRegression, self).__init__()

        #define the layers of model in this .
        # For this model we will only have one linear layer
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.lin(x)

X = torch.tensor([[1],[2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

# TEst value can not be a scalar so have to define a tensor for testing
X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
input_size = n_features
output_size = n_features

model = LinearRegression(in_features=n_features, out_features=output_size)

print(f'Prediciton before training f(5) = {model(X_test).item():.3f}')
#training
learning_rate = 0.01
iters_num = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(iters_num):
    y_pred = model(X)
    l = loss(y_pred, Y)

    l.backward()
    optimizer.step()

    optimizer.zero_grad()

    if epoch % 5 == 0:
        [w, d] = model.parameters()
        
        print(f'epoch: {epoch}: loss: {l:.3f}, w: {w[0][0].item():.3f}')

print(f'Prediciton after training f(5) = {model(X_test).item():.3f}')
