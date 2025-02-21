import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt
from timeit import default_timer as timer

# The torch.Tensor class has an attribute ".requires_grad". If you set it to True, it starts tracking 
# all operations on it. When you finish your computations you can call .backward() and have all the
# gradients computed automatically. The gradient for this tensor will be accumulated into the .grad
# attribute.

# CREATE A TENSOR, OPERATING ON IT, AND COMPUTING DERIVATIVES
def describe_tensor(tensor, name=''):
	# Helper function to explore the attributes of a tensor object
	print('-' * 30)
	print('Name: ', name)
	print('-' * 30)
	print('data : ', tensor.data)
	print('requires_grad : ', tensor.requires_grad)
	print('grad: ', tensor.grad)
	print('grad_fn: ', tensor.grad_fn)
	print('is_leaf: ', tensor.is_leaf)
	print('=' * 30)

x = torch.tensor(1.0)
y = torch.tensor(2.0)

describe_tensor(x, name='x')
describe_tensor(y, name='y')

Z = x * y
describe_tensor(Z, name='Z')

# Now we can't do the z.backward() because there is no node requiring gradients
# Let's fix it:
x = torch.tensor(1.0, requires_grad=True)
Z = x * y
describe_tensor(Z, name='Z')

Z.backward()

describe_tensor(x, 'x')
describe_tensor(y, 'y')
describe_tensor(z, 'z')

# Now, in the tensor 'x', in the grad field we have tensor(2.)
# That's because d(x * y)/dx = y = 2


# N-DIMENSIONAL BACKPROP:

x = torch.ones(3, requires_grad=True) # tensor([1., 1., 1.])
y = 3 * torch.ones(3) # tensor([3., 3., 3.])
z = x * y # tensor([3., 3., 3.])
describe_tensor(x, 'x')
describe_tensor(y, 'y')
describe_tensor(z, 'z')
# Backprop with the explicit gradient
z.backward(torch.ones(3))
print('AFTER BACKWARD:')
describe_tensor(x, 'x') 
# now 'x' have grad value of [3, 3, 3]

# Once the backward computation is done the DCG is removed and so we cannot
# perform backprop anymore. Unless you specify you want to retain the graph 
# to do as many backwards as desired

# If we want to fix it:
x = torch.ones(3, requires_grad=True)
y = 3 * torch.ones(3)
z = x * y
z.backward(x,retain_graph=True)
z.backward(torch.ones(3))

describe_tensor(x, 'x')
# After that, the grad field would be: tensor([6., 6., 6.]), 3+3


# BUILDING A NEURAL NETWORK AND TRAINING IT
import torch.nn as nn

class MyNet(nn.Module):

	def __init__(self):
		super().__init__() # must call the superclass init first
		# First fully-connected layer (3 inputs, 20 hidden neurons)
		self.fc1 = nn.Linear(3, 20)
		# First hidden activation
		self.act1 = nn.Tanh()
		# Second fully-connected layer (20 hidden neurons, 3 outputs)
		self.fc2 = nn.Linear(20, 3)
		# No activation as we make it a linear output

	def forward(self, x):
		# activation of first layer is Tanh(FC1(x))
		h1 = self.act1(self.fc1(x))
		# output activation
		y = self.fc2(h1)
		return y

net = MyNet()
print(net)
describe_tensor(net.fc1.weight, 'FC1 weight')

params = list(net.parameters())
for p in params:
	print(p.shape)

# Let's instantiate MSE loss function:
loss_fn = nn.MSELoss()

def train(network, optimizer, loss_fn, num_iters):
	""" Training function """

	loss_history = []

	for niter in range(1, num_iters + 1):
		# Reset the gradients computed in previous iterations (if any)
		optimizer.zero_grad()
		# Sample 10 (minibatch size) random samples of dimension expected by NN
		x = torch.rand(10, 3)
		# 1) Forward the data through the network
		y_ = network(x)
		# 2) Compute the loss wrt to a zero label
		loss = loss_fn(y_, torch.zeros(y_.shape))
		# 3) Backprop with respect to the loss function
		loss.backward()
		# 4) Apply the optimizer with a learning step
		optimizer.step()
		# Store the loss log to plot
		loss_history.append(loss.item())

		if niter % 50 == 0:
			print('Step {:2d} loss: {:.3f}'.format(niter, loss_history[-1]))

	plt.plot(loss_history)
	plt.xlabel('Niter')
	plt.ylabel('Loss')

net = MyNet()
# we will take stochastic gradient descent (SGD) to exemplify the training loop of a neural network
# We first need to handle the parameters that the optimizer will tune, and then we must specify the learning rate (lr) of each
# update step
opt = optim.SGD(net.parameters(), lr=0.01)
train(net, opt, loss_fn, 500)


# EXERCISE: instantiate a network like the one shown earlier and train it to map simple uniform noise to zeros.
# We will track the loss value, which must decrease, and will plot it.

x_sample = torch.rand(5, 3)
non_trained = MyNet()
print('Non-trained result: ', torch.mean(non_trained(x_sample)).item())
print('Trained result: ', torch.mean(net(x_sample)).item())
# Trained result should be closer to zero than the non-trained one (if training went well)


# WE DO NOT WANT GRADIENTS
# We can avoid the computation of gradients through the neural network forward pass by enclosing it into the with
# torch.no_grad() context (which speeds up evaluation process by x2 or x3 normally). 

x = torch.zeros(10, 3)
with torch.no_grad():
	y_ = net(x)
	loss = loss_fn(y_, torch.zeros(x.shape))
	print('Loss: {:.2f}'.format(loss))
	describe_tensor(loss, 'loss')
	print('NOTE THAT requires_grad=False NOW IN THE LOSS TENSOR')


# Finally, we can also cut the graph at any point we want (if we want) with the .detach() function of a Tensor.

class MyNetWithDetach(nn.Module):

	def __init__(self):
		super().__init__() # must call the superclass init first
		# First fully-connected layer (3 inputs, 20 hidden neurons)
		self.fc1 = nn.Linear(3, 20)
		# First hidden activation
		self.act1 = nn.Tanh()
		# Second fully-connected layer (20 hidden neurons, 3 outputs)
		self.fc2 = nn.Linear(20, 3)
		# No activation as we make it a linear output

	def forward(self, x):
		# activation of first layer is Tanh(FC1(x))
		h1 = self.act1(self.fc1(x))
		# DETACH
		h1 = h1.detach()
		# output activation
		y = self.fc2(h1)
		return y

# Now we can train this network
net = MyNetWithDetach()
# Now we can observe the difference of gradients in the biases of the 2 layers
# between this network and the regular one
# in terms of computed gradients

def forward_backward(network, net_name=''):
	x = torch.zeros(10, 3)
	y_ = network(x)
	loss = loss_fn(y_, torch.zeros(x.shape))
	loss.backward()
	describe_tensor(network.fc1.bias, '{}:FC1 bias'.format(net_name))
	describe_tensor(network.fc2.bias, '{}:FC2 bias'.format(net_name))

# Try with a non-detached network
forward_backward(MyNet(), 'Non-Detached Net')
# Try with a detached network
forward_backward(MyNetWithDetach(), 'Detached Net')
