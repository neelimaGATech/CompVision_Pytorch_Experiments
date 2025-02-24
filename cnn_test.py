# import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#configure the device to appropriate type
device = torch.device('mps' if torch.backends.mps.is_available()
                      else ('cuda' if torch.cuda.is_available() else 'cpu'))

#hyper parameters for the model
num_epochs = 0
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

# show images
#imshow(torchvision.utils.make_grid(images))

conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)
pool1 = nn.MaxPool2d(2, 2)
conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

print(images.shape)
x = conv1(images)
print(x.shape)
x = pool1(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x= pool1(x)
print(x.shape)
#imshow(torchvision.utils.make_grid(x))
