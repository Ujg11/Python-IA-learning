# WHAT IS A TENSOR?
	# A tensor is the generalization of a vector into k dimensions.
	# Because of this, a tensor is any k-dimensional structure, including matrices, vectors and scalars. 
	# PyTorch is a deep learning framework (https://pytorch.org) widely used for both research and production. 
	# As in any other deep learning framework, its core data structure is the tensor.

# We have to install PyTorch first: pip install torch torchvision torchaudio
import torch #PyTorch: work with neural networks and process GPU&CPU data
import numpy #Numpy (Numerical Python): work with multidimensional arrays

torch.manual_seed(1)
a = torch.empty(5, 7) # We initialize an empty structure with certain dimensions (rows, columns)
print(a.shape)
print(a.size())
print(a)
a.fill_(10) # Fill the tensor with all 10
print(a)
print('\n\n\n')

# Now we create an empty tensor
b = torch.empty(2, 4, 6, 8)
b.fill_(8)
# We have 2 elements in the first dimension (dim=0), 4 in the second (dim=1), 6 in the third (dim=2) and 8 in the fourth (dim=3)
print(b)
# Tenim 2 blocs principals, dintre de cada bloc hi ha 4 matrius i cada matriu té 6 files i 8 columnes
print('\n\n\n')

# Functions:
print("Random Gaussian distribution")
c = torch.randn(2, 3) # samples from a Gaussian distribution (mean=0, std=1)
print(f"{c}\n")
print("Random Uniform distribution")
d = torch.rand(4, 5) # samples from a uniform distribution [0, 1)
print(f"{d}\n")
print("Tensor with ones")
e = torch.ones(4, 4) # creates a tensor with 1s
print(f"{e}\n")
print("Tensor with zeros")
f = torch.zeros(2, 2) # creates a tensor with 0s
print(f"{f}\n")

# EXERCISE 1: Create a tensor z drawn from a Gaussian distribution of dimensions (16, 1024)
z = torch.randn(16, 1024) # Gaussian distribution (mean=0, std=1)

if z.mean().round().int().item() == 0 and z.std().round().int().item() == 1 \
	and z.shape[0] == 16 and z.shape[1] == 1024:
	print('Well done! The elements in the tensor are mean centered with unit variance, and it is 16x1024!\n\n')
else:
	print('Wrong :( Did you use the proper torch function? Did you supply the correct tensor shape?\n\n')


# Tensors have data type. We can check the type with tensor.dtyp atribute and also change with simple casts:
g = torch.ones(5)
print(g.dtype)
print(a.double().dtype) # change to float64
print(a.half().dtype) # change to float16 (aka. half)
print(a.short().dtype) # change to int16 (aka. short)
print(a.long().dtype) # change to int64 (aka. long)
print(a.int().dtype) # change to int32 (aka. long)


# To create a tensor with specific data type:
a = torch.empty(5, 7, dtype=torch.short) # Initialize with type
print(a.dtype)

a = torch.ShortTensor(5, 7) # Directly create a short tensor
print(a.dtype)


# NUMPY AND PYTHON LISTS:

# Creating a 1-D tensor from the Numpy array [1, 2, 3]
a = torch.tensor(numpy.array([1, 2, 3]))

# Creating a 1-D tensor from the Python list [1, 2, 3]
a = torch.tensor([1, 2, 3])

# Values 1, 2, 3
print('Tensor a values: ', a)
# 1 dimension of size 3
print('Tensor a shape: ', a.shape)

# k-dimensional arrays are also turned into PyTorch tensors:
A = torch.tensor(numpy.ones((16, 1024)))
print(A.dtype)
print(A.shape)


# CONVERTING TENSORS BACK TO NUMPY:

A = torch.rand(10, 10) # Create a tensor
Anpy = A.data.numpy() # Convert it to Numpy
print('A type: ', type(A)) # torch.Tensor
print('Anpy type: ', type(Anpy)) # numpy.ndarray


# EXERCISE 2: Create an int16 it tensor in PyTorch (however you want) from the following numpy array na

na = 10 * numpy.random.rand(8,8)
it = torch.tensor(na, dtype=short.short)
assert it.dtype == torch.int16, 'this tensor is not of dtype short? =('


# OPERATIONS WITH TENSORS

	# 1 In-place operations (function contain an undescore '_')
a = torch.empty(2, 2)
a.fill_(1)
a.add_(1)


# APLIED TO DEEP LEARNING:

# A Neuron is defined as a linear operation of weighted sums followed by a non-linearity.
# We thus have a tensor of weights w, a scalar with the bias b, and a non-linearity 
# (like ReLU max(0, x) that just allows the positive components to go forth in the y values).

x = torch.ones(1, 100_000)
w = 0.02 * torch.randn(100_000, 1)
b = 10 * torch.ones(1)

# Function that will perform the operation of a neuron:
def forward_neuron(x, w, b):
	v = x.mm(w) + b # .mm -> multiplicate matrix x & w
	y = v.clamp(min=0) # Aplies ReLU function
	return y

print(forward_neuron(x, w, b))
b = -10 * torch.ones(1)
print(forward_neuron(x, w, b))


# EXERCISE 4:  change the forward_neuron function to apply the ReLU in-place. 
# This is very useful to save memory when constructing very deep nets.

def forward_neuron(x, w, b):
	v = x.mm(w) + b
	return v.clamp_(min=0)


# TRANSPOSITION AND BEYOND

A = torch.empty(10, 20, 5)

# Change the dim=2 with dim=1
A_21 = A.transpose(2, 1) # torch.Size([10, 20, 5]) transposed axis (2, 1) to: torch.Size([10, 5, 20])
# Change the dim=2 with dim=0
A_20 = A.transpose(2, 0) # torch.Size([10, 20, 5]) transposed axis (2, 0) to: torch.Size([5, 20, 10])

# We can merge axis without change the content, for example 10 * 20 = 200
B = A.view(200, 5) # torch.Size([10, 20, 5]) axis (0, 1) merged to: torch.Size([200, 5])


# We can chunk the tensor (split)
Achunk = torch.chunk(A, 5, dim=1)
# to see it:
for i, achunk in enumerate(Achunks):
	print('Chunk {} shape: {}'.format(i, achunk.shape))

# We can merge it back:
Amerged = torch.cat(Achunks, dim=1)


# ADD ADDITIONAL DIMENSIONS OR REMOVE THEM:

A = torch.empty(2, 2)

# 1) Add an extra dimension in axis 0 (unsqueeze)
A = A.unsqueeze(0)
# 2) Add an extra dimension in axis 2
A = A.unsqueeze(2)
# 3) Add an extra dimension in axis 2 again
A = A.unsqueeze(2)
print(A) # to dimensions=(0, 2, 2) -> torch.Size([1, 2, 1, 1, 2])

# 4) Remove the dimension 0 from step (1)
A = A.squeeze(0)
# 5) We will remove all remaining dimensions of size 1 ("useless") when we do not specify the dimension
A = A.squeeze()


# EXERCISE 5: Unsqueezing and squeezing dimensions can also be achieved with the .view() function.
# "View" the tensor A to achieve the same shape as the one after step (3)
A = torch.empty(2, 2)
A = A.view(1, 2, 1, 1, 2)


# EXERCISE 6: Given the tensor A, shuffle each of the elements of the first dimension
# with the random.shuffle Python function.

import random
random.seed(1)
from random import shuffle

A = torch.rand(4, 2, 4)
print('A before shuffling:\n ', A)
A = torch.chunk(A, 4, dim=0)
A = list(A) # convert into a list

random.shuffle(A) #operate with shuffle over the list
A= torch.cat(A, dim=0) #concatenate the sub-tensors in list "A" back to tensor "A"
