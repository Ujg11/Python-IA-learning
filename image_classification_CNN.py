# Image Classification with a Multi-Layer Perceptron

import numpy as np
np.random.seed(123)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import random
import torch.optim as optim


# Defining the Hyper-parameters

hparams = {
	'batch_size':64,#we'll process 64 samples before actualize weights
	'num_epochs':10,
	'test_batch_size':64,#test size
	'hidden_size':128,#size of neurons in the layer
	'num_classes':10,
	'num_inputs':784,#image 28x28
	'learning_rate':1e-3,
	'log_interval':100,
}

# we select to work on GPU if it is available in the machine, otherwise will run on CPU
hparams['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


# DEFINING THE PYTORCH DATASET AND THE DATALOADER

# The PyTorch Dataset is an inheritable class that helps us defining 
# what source of data do we have (image, audio, text, ...) and how to
# load it (overriding the __getitem__ function)

mnist_trainset = datasets.MNIST('data', train=True, download=True,
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
							]))
mnist_testset = datasets.MNIST('data', train=False,
							transform=transforms.Compose([
								transforms.ToTensor(),
								transforms.Normalize((0.1307,), (0.3081,))
							]))

train_loader = torch.utils.data.DataLoader(
	mnist_trainset,
	batch_size=hparams['batch_size'],
	shuffle=True)

test_loader = torch.utils.data.DataLoader(
	mnist_testset,
	batch_size=hparams['test_batch_size'],
	shuffle=False)

# Now we can use the data

img, label = mnist_trainset[0]
print('Img shape: ', img.shape)
print('Label: ', label)

# Similarly, we can sample a BATCH from the dataloader by running over its iterator
iter_ = iter(train_loader)
bimg, blabel = next(iter_)
print('Batch Img shape: ', bimg.shape)
print('Batch Label shape: ', blabel.shape)
print('The Batched tensors return a collection of {} grayscale images ({} channel, {} height pixels, {} width pixels)'.format(bimg.shape[0],
																															bimg.shape[1],
																															bimg.shape[2],
																															bimg.shape[3]))
print('In the case of the labels, we obtain {} batched integers, one per image'.format(blabel.shape[0]))


# Let's see the images we are working with:
def plot_samples(images,N=5):
	'''
	Plots N**2 randomly selected images from training data in a NxN grid
	'''
	# Randomly select NxN images and save them in ps
	ps = random.sample(range(0,images.shape[0]), N**2)

	# Allocates figure f divided in subplots contained in an NxN axarr
	# https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.pyplot.subplots.html
	f, axarr = plt.subplots(N, N)
	
	# Index for the images in ps to be plotted
	p = 0
	# Scan the NxN positions of the grid
	for i in range(N):
		for j in range(N):

			# Load the image pointed by p
			im = images[ps[p]]

			# If images are encoded in grayscale
			# (a tensor of 3 dimensions: width x height x luma)...
			if len(images.shape) == 3:
				axarr[i,j].imshow(im,cmap='gray')
			else:
				axarr[i,j].imshow(im)

			# Remove axis
			axarr[i,j].axis('off')

			# Point to the next image from the random selection
			p+=1
	# Show the plotted figure
	plt.show()

# convert the dataloader output tensors from the previous cell to numpy arrays
# The channel dimension has to be squeezed in order for matplotlib to work
# with grayscale images
img = bimg.squeeze(1).data.numpy()
plot_samples(img)


# TRAINING A MULTI-LAYER PERCEPTRON (MLP)

# create the network:
network=nn.Sequential(
	nn.Linear(hparams['num_inputs'], hparams['hidden_size']),
	nn.ReLU(),
	nn.Linear(hparams['hidden_size'], hparams['num_classes']),
	nn.LogSoftmax(dim=1)
)

network.to(hparams['device'])
'''Sequential(
	(0): Linear(in_features=784, out_features=128, bias=True)
	(1): ReLU()
	(2): Linear(in_features=128, out_features=10, bias=True)
	(3): LogSoftmax(dim=1)
)'''

# We create this function in order to see the number of parameters
def get_nn_nparams(net):
	pp=0
	for p in list(net.parameters()):
		nn=1
		for s in list(p.size()):
			nn = nn*s
		pp += nn
	return pp

print(network)
print('Num params: ', get_nn_nparams(network))
'''Num params:  101770'''

# Optimizer: RMS Prop (use the hparams['learning_rate'] previously defined)
optimizer = optim.RMSprop(network.parameters(),lr=hparams['learning_rate'])

# Negative Log Likelihood (NLL) Loss criterion from the functional API
criterion = F.nll_loss

# Define the Accuracy metric in the function below by:
def correct_predictions(predicted_batch, label_batch):
	pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
	acum = pred.eq(label_batch.view_as(pred)).sum().item()
	return acum


# TRAIN NETWORK FOR AN EPOCH WITH train_loader:

def train_epoch(train_loader, network, optimizer, criterion, hparams):
	# Activate the train=True flag inside the model
	network.train()

	device = hparams['device']
	avg_loss = None
	avg_weight = 0.1

	# For each batch
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()

		data = data.view(data.shape[0], -1)
		output = network(data)
		loss = criterion(output, target)
		loss.backward()
		if avg_loss:
			avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
		else:
			avg_loss = loss.item()
		optimizer.step()
		if batch_idx % hparams['log_interval'] == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
	            epoch, batch_idx * len(data), len(train_loader.dataset),
	            100. * batch_idx / len(train_loader), loss.item()))
	return avg_loss


# Let's do the test function:

def test_epoch(test_loader, network, hparams):
    # Dectivate the train=True flag inside the model
    network.eval()

    device = hparams['device']
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:

            # Load data and feed it through the neural network
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)
            output = network(data)

            test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            # WARNING: If you are using older Torch versions, the previous call may need to be replaced by
            # test_loss += criterion(output, target, size_average=False).item()

            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)

    # Average accuracy across all correct predictions batches now
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc


# NOW WE WILL DO THE TRAINING:

# Init lists to save the evolution of the training & test losses/accuracy.
train_losses = []
test_losses = []
test_accs = []

# For each epoch
for epoch in range(1, hparams['num_epochs'] + 1):
	# Compute & save the average training loss for the current epoch
	train_loss = train_epoch(train_loader, network, optimizer, criterion, hparams)
	train_losses.append(train_loss)

	# TODO: Compute & save the average test loss & accuracy for the current epoch
	# TIP: Review the functions previously defined to implement the train/test epochs
	test_loss,test_accuracy=test_epoch(test_loader, network, hparams)
	test_losses.append(test_loss)
	test_accs.append(test_accuracy)


plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')
plt.plot(train_losses, label='train')
plt.plot(test_losses, label='test')
plt.legend()
plt.subplot(2,1,2)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy [%]')
plt.plot(test_accs)






