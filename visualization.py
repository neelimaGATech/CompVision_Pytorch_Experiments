# Visulization NN training using TensorBoard
# loss, accuracy, ops, layers, histograms of weights, biases, tensors
# We will use MNIST dataset, input images are 28* 28 ---> flattened to 784 vector
# Output size / number of classes across which image would be scored = 10


import torch
import traceback
import torch.nn as nn
import torchvision
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#from binary_classification_pytorch import CustomNeuralNet
# Tensor board
import sys
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/mnist1')

try:
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
except Exception:
    traceback.print_exc()
# hyper parameters for the model
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 10
batch_size = 64
learning_rate = 0.001

# load the data
train_dataset = tv.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = tv.datasets.MNIST(root='./data',train=False, download=True, transform=transforms.ToTensor())

# Data loader objects
train_loader= torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

examples = iter(train_loader)
images, labels = next(examples)
#print(images.shape)
#print(len(train_loader))

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i][0], cmap='gray')
#commenting this to use tensorboard writer instead
#plt.show()

#### TensorBoard
img_grid = torchvision.utils.make_grid(images)
writer.add_image('mnist_image', img_grid)

class CustomNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CustomNeuralNet, self).__init__()
        self.input_size = input_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.relu1(self.linear1(x))
        out = self.linear2(out)
        return out

# training pipeline
model = CustomNeuralNet(input_size, hidden_size, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

### Tensorboard
writer.add_graph(model, images.reshape(-1, input_size).to(device))

n_total_steps =len(train_loader)
for epoch in range(num_epochs):
    for i, (images,labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')


# Evaluate model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predictions - labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy:{accuracy:.2f}%')

