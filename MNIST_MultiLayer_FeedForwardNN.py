# DataSet: MNIST
# MNIST (Modified National Institute of Standards and Technology) dataset contains
# handwritten digits from 0 to 9
# MNIST is used for training and testing machine learning models, especially in image classification
# Pytorch DataLoader and Transformation

import torch
# for nn module
import torch.nn as nn
# for datasets for computer vision training
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# GPU device
if torch.backends.mps.is_available():
    device = torch.device('mps')
    x = torch.ones(1, device=device)
    print(x)
else:
    device = torch.device('cpu')
    print("MPS not available")

# Parameters to train on MNIST images
input_size = 784 # images are size 28*28 flattened later to a vector for input
hidden_size = 100 # hyper parameter
num_classes = 10  # MNIST is for 10 different classes 0 -9
num_epochs = 2
batch_size = 100
learning_rate = 0.001


# import MNIST data
# download MNIST training data set, download if need be, immediately apply transform to Tensors
train_dataset = tv.datasets.MNIST(root='./data', train=True,
                                  transform=transforms.ToTensor(), download=True)

# get the test dataset from the MNIST , perform transformation to Tensor
# dont bother to download
test_dataset = tv.datasets.MNIST(root='./data', train= False,
                                 transform=transforms.ToTensor())

# use data loader to load the training dataset in form of batches of size 100
# shuffle is done during training mostly
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batch_size)

# use iterator on top of dataloader object to pick data on batch size at a time
examples = iter(train_loader)
samples, labels = next(examples)
print(type(samples), samples.shape)
print(type(labels), labels.shape)

plt.cla()
# use matplotlib to plot / view first 5 images in the samples list
for i in range(6):
    # plt.subplot(rows, cols, index)
    # create a grid of subplots with 2 rows , 3 columns and which subplot to use
    plt.subplot(2, 3 , i+1)

    # samples is a list of containing images data
    #
    print(type(samples[i][0]), samples[i][0].shape)
    plt.imshow(samples[i][0], cmap='gray')
    #plt.show()

# Define a Feedforward method
# do not apply softmax as we would use cross entropy loss function from pytorch
# CE loss function from pytorch applies the Softmax function to the output implicitely
class NeuralNetworkTrainMNIST(nn.Module):
    def __init__(self, input_s, hidden_s, n_classes):
        super(NeuralNetworkTrainMNIST,self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size,num_classes )

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        return out

model = NeuralNetworkTrainMNIST(input_size, hidden_size, num_classes).to(device)

# criterion for loss would be cross entropy loss
# would apply the sigmoid to the network output
criterion = nn.CrossEntropyLoss()

# method to optimize the above loss criterion -- minimize
# uses Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

# training loop
num_steps = len(train_loader)
for epoch in range(num_epochs):
    # give a iterator i to the elements of epoch containing batch_size number of samples
    for i, (images,labels) in enumerate(train_loader):
        # shape of the images data is 100 * 1 * 28 * 28
        # reshape each image to a 1-D vector for efficient processing 784 element vector
        # new shape would be 100 * 784
        # push the images tensors to gpu device
        images = images.reshape(-1,28*28).to(device)
        labels = labels.to(device)

        # forward pass - run model on images and calculate loss
        outputs = model(images)
        loss = criterion(outputs, labels)


        #backward pass - run backward pass, run a step of optimizer to update weight parameters
        # set the gradients to zero
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print information every 100 epochs
        if (i+1) % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step: {i+1} / {num_steps}, Loss: {loss.item():.4f}')


# testing code
# gradient calculation should be off during testing
with torch.no_grad():
    num_correct_predictions = 0
    num_samples = 0

    # no enumeration needed as we are not tracking steps
    for images, labels in test_loader:
        # reshape 28*28  size input images to a vector of length 784
        images = images.reshape(-1,28*28).to(device)

        labels = labels.to(device)

        outputs = model(images)
        print(outputs.shape)
        # outputs would usually have shape (batch_size, num_classes )
        # outputs would contain the raw scores for each class along dimension 1 because
        # our model does not have sigmoid at the last layer
        # the maximum of these scores indicates which class has been found as most probable class
        # max method returns value and index of the maximum value in the given output vector
        # we are only interested in the index of max since that gives the class predicted by model
        _ , predictions = torch.max(outputs, 1)
        num_samples += labels.shape[0]
        num_correct_predictions += (predictions == labels).sum().item()

accuracy = (num_correct_predictions/num_samples)*100

print('Accuracy:', accuracy)


