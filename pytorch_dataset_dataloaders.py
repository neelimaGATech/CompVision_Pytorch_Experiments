#Large datasets
# divide the dataset over batches  - pytorch can do batch calculations and iterations

# terms for batch processing
# 1 epoch = 1 forward and 1 backward pass over all training samples (across all batches)
# batch_size = number of training samples in one forward and backward pass
# number of iterations = number of passes, each pass using [batch_sizes] number of samples
# each epoch would have iterations = total samples / [batch_size] will make up 1 epoch
# Processing one batch makes up one iteration
# So for a sample size of 100, batch_size of 20, 100/20 = 5 iterations would make up 1 epoch

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math as mt

# Custom Dataset
class WineDataset(Dataset):
    def __init__(self):
        # load data
        #print("inside the init method", title)
        xy = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0])
        self.n_samples = self.x.shape[0]



    def __getitem__(self,index):
        # get single dataset item dataset[0]
        #print("inside the getitem method")
        return self.x[index], self.y[index]

    def __len__(self):
        # len(dataset)
        #print("inside the __len__ method")

        return self.n_samples

# create dataset
dataset = WineDataset()

#print(len(dataset))
#firstdata = dataset[0]
#features, labels = firstdata

#print(f'Features are: {features}')

#print(f'labels are: {labels}')

# Use dataloader class to use loaded dataset into batchs
batch_size = 4
dataloader = DataLoader(dataset = dataset, batch_size =batch_size, shuffle=True)

#now we can use iterator object to iterate through the batches
num_epochs = 2
total_samples = len(dataset)
n_iterations = mt.ceil(total_samples / batch_size)

for epoch in range(num_epochs):
    #for iteration in range(n_iterations):
    for index, (inputs, labels) in enumerate(dataloader):
        # forward pass, backward pass and update
        if index % 5 ==0:
            print(f'Epoch = {epoch+1}/ {num_epochs} , Current Step: {index +1}/{n_iterations}')
            print(f'inputs: {inputs.shape}, labels: {labels.shape}')

#torch vision has computer vision datasets





