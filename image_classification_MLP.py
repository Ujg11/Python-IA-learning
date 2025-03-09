# Image Classification with a Convolutional Neural Network

import numpy as np
np.random.seed(1)
import torch
import torch.optim as optim
torch.manual_seed(1)
if torch.cuda.is_available():
	torch.cuda.manual_seed_all(1)
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# Let's define some hyper-parameters
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


# CONVOLUTIONAL NEURAL NETWORKS

# Let's first define a 2D convolutional layer with 1 input channel, 3 output channels and (height=3, width=3) kernel size
conv = nn.Conv2d(1, 3, 3) # 3x3 amb imatges en blanc i negre (1 canal)

#Importantly, a convnet takes input tensors of shape (batch, num_channels, image_height, image_width)
x = torch.rand(1, 1, 28, 28) #MNIST image:28x28
y = conv(x)
print('Output shape: {} = conv({})'.format(y.shape, x.shape))

# For an other example, let's forward an image of size (11, 11)
x = torch.rand(1, 1, 11, 11)
y = conv(x)
print('Output shape: {} = conv({})'.format(y.shape, x.shape))

# Exercise: define the conv layer below and ensure that the output tensor shape in dimensions {H, W}
#( as in [1, channels, H, W] ) will be the same as the input in both cases.

conv=nn.Conv2d(1, 3, 3, padding= 1)

x = torch.rand(1, 1, 20, 20)
y = conv(x)
print('Output shape: {} = conv({})'.format(y.shape, x.shape))
assert y.shape[2:] == x.shape[2:], 'Err: conv not well specified!'
x = torch.rand(1, 1, 11, 11)
y = conv(x)
print('Output shape: {} = conv({})'.format(y.shape, x.shape))
assert y.shape[2:] == x.shape[2:], 'Err: conv not well specified!'


# ABOUT POOLING

# Pooling refers to a block where downsampling happens. In the case of CNNs,
# as they process full images throughout a certain stack of layers that can get
# quite deep, they occupy a lot of memory to store the so called feature maps.
# Feature maps are the intermediate hidden activations of a CNN. 

# Let's define a small CNN without pooling and another one with pooling, and
# let's check the amount of memory used by each in terms of feature map usage
# and the time it takes to forward an image of 512x512 pixels with just 1 input channel (hence greyscale).

NUM_BITS_FLOAT32 = 32

# Class that extract the amount of memory
class CNNMemAnalyzer(nn.Module):
	def __init__(self, layers):
		super().__init__()
		self.layers = layers

	def forward(self, x):
		tot_mbytes = 0
		spat_res = []
		for layer in self.layers:
			h = layer(x)
			mem_h_bytes = np.cumprod(h.shape)[-1] * NUM_BITS_FLOAT32 // 8
			mem_h_mb = mem_h_bytes / 1e6
			print('-' * 30)
			print('New feature map of shape: ', h.shape)
			print('Mem usage: {} MB'.format(mem_h_mb))
			x = h
			if isinstance(layer, nn.Conv2d):
				# keep track of the current spatial width for conv layers
				spat_res.append(h.shape[-1])
			tot_mbytes += mem_h_mb
		print('=' * 30)
		print('Total used memory: {:.2f} MB'.format(tot_mbytes))
		return tot_mbytes, spat_res


# Forwarding the 512x512 image through a non-pooled CNN
cnn = CNNMemAnalyzer(nn.ModuleList([nn.Conv2d(1, 32, 3), #(canals d'entrada, canals de sortida, mida filtre)
                                    nn.Conv2d(32, 64, 3),
                                    nn.Conv2d(64, 64, 3),
                                    nn.Conv2d(64, 128, 3),
                                    nn.Conv2d(128, 512, 3)]))

# Let's work with a realistic 512x512 image size
# Also, keep track of time to make forward
beg_t = timer()
nopool_mbytes, nopool_res = cnn(torch.randn(1, 1, 512, 512))
end_t = timer()
nopool_time = end_t - beg_t
print('Total inference time for non-pooled CNN: {:.2f} s'.format(nopool_time))


# Now, let's make a stack of convlayers combined with MaxPoolings

cnn = CNNMemAnalyzer(nn.ModuleList([nn.Conv2d(1, 32, 3),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(32, 64, 3),
                                    nn.MaxPool2d(2),
                                    nn.Conv2d(64, 64, 3),
                                    nn.Conv2d(64, 128, 3),
                                    nn.Conv2d(128, 512, 3)]))

beg_t = timer()
pool_mbytes, pool_res = cnn(torch.randn(1, 1, 512, 512))
end_t = timer()
pool_time = end_t - beg_t
print('Total inference time for pooled CNN: {:.2f} s'.format(pool_time))


mem_ratio = 1. - pool_mbytes / nopool_mbytes
print('Total saved memory with poolings: ', 100. * mem_ratio)

time_ratio = nopool_time / pool_time
print('Total inference speed increase with poolings: x{:.1f}'.format(time_ratio))

# Let's plot the width of each feature map as we get deeper into the network
_ = plt.plot(nopool_res, label='No pooling')
_ = plt.plot(pool_res, label='Pooling')
_ = plt.legend()


# Key Observations
# We save 87.3% of memory having a model which is pooling after the first couple of conv layers.
# The model that contains pooling runs 11.1 times faster in inference than the other one.
# The width dimension decreases exponentially when inserting the poolings, compared to the one without those poolings.


# BUILDING A PSEUDO LeNet MODEL:

# Make the ConvBlock class to properly do: Conv2d, ReLU, and MaxPool2d.

class ConvBlock(nn.Module):

	def __init__(self, num_inp_channels, num_out_fmaps, kernel_size, pool_size=2):
		super().__init__()
		# TODO: define the 3 modules needed
		self.conv = nn.Conv2d(num_inp_channels, num_out_fmaps, kernel_size)
		self.relu = nn.ReLU()
		self.maxpool = nn.MaxPool2d(pool_size)

	def forward(self, x):
		return self.maxpool(self.relu(self.conv(x)))

x = torch.randn(1, 1, 32, 32)
y = ConvBlock(1, 6, 5, 2)(x)
assert y.shape[1] == 6, 'The amount of feature maps is not correct!'
assert y.shape[2] == 14 and y.shape[3] == 14, 'The spatial dimensions are not correct!'
print('Input shape: {}'.format(x.shape))
print('ConvBlock output shape (S2 level in Figure): {}'.format(y.shape))

class PseudoLeNet(nn.Module):

	def __init__(self):
		super().__init__()
		# Define the zero-padding
		self.pad = nn.ConstantPad2d(2, 0) #2 pixels de pàdding a cada costat 28x28 -> 32x32
		self.conv1 = ConvBlock(1, 6, 5)
		self.conv2 = ConvBlock(6, 16, 5) #sortida de 16 capes
		# Define the MLP at the deepest layers
		self.mlp = nn.Sequential(
			nn.Linear(16*5*5, 120),
			nn.ReLU(),
			nn.Linear(120, 84),
			nn.ReLU(),
			nn.Linear(84, 10),
			nn.LogSoftmax(dim=1)
		)

	def forward(self, x):
		x = self.pad(x)
		x = self.conv1(x)
		x = self.conv2(x)
		# Obtain the parameters of the tensor in terms of:
		# 1) batch size
		# 2) number of channels
		# 3) spatial "height"
		# 4) spatial "width"
		bsz, nch, height, width = x.shape
		# Flatten the feature map with the view() operator so that
		# the batches are preserved. You can achive this by calling view
		# and provide it with the batch size as the first parameter
		x= x.view(bsz, -1)
		y = self.mlp(x)
		return y

# Let's forward a toy example emulating the MNIST image size
plenet = PseudoLeNet()
y = plenet(torch.randn(1, 1, 28, 28))
print(y.shape)


# TRAIN AND TEST:

def correct_predictions(predicted_batch, label_batch):
	pred = predicted_batch.argmax(dim=1, keepdim=True) # get the index of the max log-probability
	acum = pred.eq(label_batch.view_as(pred)).sum().item()
	return acum

def train_epoch(train_loader, network, optimizer, criterion, hparams):
	# Activate the train=True flag inside the model
	network.train()
	device = hparams['device']
	avg_loss = None
	avg_weight = 0.1
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
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

def test_epoch(test_loader, network, hparams):
    network.eval()
    device = hparams['device']
    test_loss = 0
    acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += criterion(output, target, reduction='sum').item() # sum up batch loss
            # compute number of correct predictions in the batch
            acc += correct_predictions(output, target)
    # Average acc across all correct predictions batches now
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * acc / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, acc, len(test_loader.dataset), test_acc,
        ))
    return test_loss, test_acc


# The training:

tr_losses = []
te_losses = []
te_accs = []
network = PseudoLeNet()
network.to(hparams['device'])
optimizer = optim.RMSprop(network.parameters(), lr=hparams['learning_rate'])
criterion = F.nll_loss

for epoch in range(1, hparams['num_epochs'] + 1):
	tr_losses.append(train_epoch(train_loader, network, optimizer, criterion, hparams))
	te_loss, te_acc = test_epoch(test_loader, network, hparams)
	te_losses.append(te_loss)
	te_accs.append(te_acc)


# Let's see the results:

plt.figure(figsize=(10, 8))
plt.subplot(2,1,1)
plt.xlabel('Epoch')
plt.ylabel('NLLLoss')
plt.plot(tr_losses, label='train')
plt.plot(te_losses, label='test')
plt.legend()
plt.subplot(2,1,2)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy [%]')
plt.plot(te_accs)

# The final result is slightly above 99%, better than the MLP
# model for a comparable amount of training.
