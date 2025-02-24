# Design a Convolutional Neural Network to train on CIFAR image dataset
# torchvision.datasets.CIFAR10 data is stored as pickled file .pkl in batches of 10000 image data and labels
# Each label corresponds to one of the 10 classes
# when loaded using PyTorch, dataset is automatically unpickled and converted to PIL/PILLOW images
# PIL (Python Image Library) provides python tools for opening, manipulating and saving image files
# PIL images need to be transformed to Tensors because Pytorch works with Tensors and also normalized
# transforms.ToTensor():
#       converts PIL array to tensor ,
#       Converts the image from shape (H, W, C) (height, width, channels) →
#                               (C, H, W) (channel-first format, required by PyTorch).
#       Converts pixel values from [0, 255] to [0, 1] (float values).
# transforms.Normalize()
#       - Shifts and scales pixel values from [0, 1] (after ToTensor()) to [-1, 1].
#       - this is a common practice in deep learning models because it
#           Keeps the data distribution centered around zero, which benefits gradient-based optimization.
# CNN is a sequence of convolutional, pooling (mac pool) and finally a fully connected layer

# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#configure the device to appropriate type
device = torch.device('mps' if torch.backends.mps.is_available()
                      else ('cuda' if torch.cuda.is_available() else 'cpu'))

#hyper parameters for the model
num_epochs = 4
batch_size = 4
learning_rate = 0.001

transform = tv.transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

#load datasets
train_dataset = tv.datasets.CIFAR10(root='./data', train=True,
                                    download=True, transform=transform)
test_dataset = tv.datasets.CIFAR10(root='./data', train=False,
                                   download=True, transform=transform)

# create dataloaders
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# plot the images to show
def imshow(img):
    img = img / 2 + 0.5 # unnormalize the values
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#get some training images to show
iter1 = iter(train_dataloader)
images, labels = next(iter1)

# show some images
#imshow(tv.utils.make_grid(images))


# classes from CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#Design network

class CustomConNet(nn.Module):
    def __init__(self):
        super(CustomConNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # input size for the first fully connected layer would be the flattened size of output from last
        # convolution / pooling layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # input -> conv1+Relu - pooling - conv2+Relu - pooling - fc+relu-fc+relu-fc-softmax -> output
    def forward(self, x):
        out = self.pool(F.relu(self.conv1(x)))

        out = self.pool(F.relu(self.conv2(out)))
        #Flatten the output before sending to the fully connected layer
        out = out.view(-1, 16 * 5 * 5)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # no sigmoid function at the last layer because we are using torch's cross entropy loss
        return out

#create model, loss function and optimizer
model = CustomConNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_dataloader)
# training loop
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step: [{i+1} /{n_total_steps}], Loss: {loss.item():.4f}')

# Finished training

# Test trained model and calculate accuracy
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images,labels in test_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        #update total number of samples by the size of batch read by dataloader
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        # count the correct predictions for each class
        for i in range(batch_size):
            label = labels[i]
            prediction = predicted[i]

            if label == prediction:
                n_class_correct[label] += 1

            n_class_samples[label] += 1

    #Accuracy of the total network
    acc = 100 * n_correct / n_samples
    print(f'Accuracy of the network = {acc:.2f}')

    # Accuracy for each class
    for i in range(10):
        acc = 100 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of the {classes[i]} class= {acc:.2f}')



