# Apply Transforms to Torch Vision Dataset

import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np


class WineDataset(Dataset):

    def __init__(self, transform=None):
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, [0]]
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.n_samples

# Custom Transform classes
#custom class to convert inputs and targets to tensors from numpy arrays
class ToTensor(object):
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

#custom transform to multiple features by a factor
class MulTransform(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

dataset = WineDataset(transform=ToTensor())
firstData = dataset[0]
features, labels = firstData
print(f'features: {features}')
print(f'labels: {labels}')


# apply composed transformation to second wine dataset
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(0.5)])
dataset_new = WineDataset(transform=composed)
firstData_new = dataset_new[0]
features, labels = firstData_new
print(f'features compose: {features}')
print(f'labels composed: {labels}')





#dataset = torchvision.datasets.MNIST(
#    root='./data', train=True, transform=torchvision.transforms.ToTensor())