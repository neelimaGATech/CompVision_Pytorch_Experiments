# In Transfer Learning, a model trained on another set of data and classfication lables
# can be used to predict the classification labels for an entirely different dataset.
# Some  changes are made to the last fully connected layers
# Pretrained network from Pytorch would be used for an image data of bees and ants (2 classes)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision as tv
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import time
import os
import copy

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

means = np.array([0.485, 0.456, 0.406])
stds = np.array([0.229, 0.224, 0.225])

data_transforms =  {
    'train' : transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ]),
    'val' : transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
}

# data import
data_dir = 'data/transfer_learning'
sets = ['train', 'val']
# load datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in sets}

# create dataloaders for
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True)
               for x in sets}

dataset_sizes = {x: len(dataloaders[x]) for x in sets}
class_names = image_datasets['train'].classes

print(class_names)

# training loop
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch +1 }/{num_epochs}')
        print('-' * 10)

        # each epoch will have training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            #running_accuracy = 0.0
            running_corrects = 0

            # iterate over every batch of data given by dataloader
            for inputs, labels in dataloaders[phase]:
                # set device for inputs
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # optimizer is called only in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                #calculate run statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            # print epoch statistics
            print(f'{phase} Loss: {epoch_loss} | Acc: {epoch_acc}')

            if (phase == 'val') and (epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

    #load model with the best model weights
    model.load_state_dict(best_model_wts)
    return model

########
# Load a pre-trained model from pytorch and reset its fully connected layer
# RESNET 18 ---- Option 1 - Train weights of all the layers
#model = models.resnet18(pretrained=True) ,
model = models.resnet18(weights = ResNet18_Weights.DEFAULT).to(device)
num_input_features = model.fc.in_features


model.fc = nn.Linear(num_input_features, 2)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# perform training of this pretrained model with replaced output layer on ant and bees training set
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=25)

# Option 2
# Train only the last layer - freeze the weights of all other layers (require grad to false)

model_conv = models.resnet18(weights = ResNet18_Weights.DEFAULT).to(device)
for param in model_conv.parameters():
    param.requires_grad = False

num_input_features = model_conv.fc.in_features
# add new FC layer - this layer would have default requires grad true
model_conv.fc = nn.Linear(num_input_features, 2)
model_conv.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model_conv.parameters(), lr=0.001, momentum=0.9)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model_conv = train_model(model_conv, criterion, optimizer, step_lr_scheduler, num_epochs=25)





