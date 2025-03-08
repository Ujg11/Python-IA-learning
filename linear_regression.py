import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from random import shuffle
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# inline ensures we will automatically see the plots as soon as we operate with plot() calls

# DATA LOADING:

# We'll build a toy dataset based on the line: y = 5x + 3 with additive noise N(0, 3)
# that will add some distortioning values.
# We will use the lineal perceptron model: y^ = w * x + b

NUM_SAMPLES = 200 #200 mostres
X_SPAN = 10
train_X = np.random.rand(NUM_SAMPLES) * X_SPAN #generate 200 numbers between 0 and 10
noise = np.random.randn(NUM_SAMPLES) * 3 #generate 200 numbers with normal distribution
train_Y = 5 * train_X + 3 + noise #function with m=5, b=3, with some noise
n_samples = train_X.shape[0] #keep the 200 numbers
plt.scatter(train_X, train_Y) #generate a map with the dispersion
plt.show()

# We also build a testset to try new predictions with out model once trained
test_X = np.random.rand(NUM_SAMPLES) * X_SPAN
test_Y = 5 * test_X + 3 + noise
plt.scatter(test_X, test_Y)
plt.show()


# BUILDING LINEAR REGRESSOR:

class LinearRegression(nn.Module): #nn.Module is the base class for neural network models
	def __init__(self):
		super().__init__()
		self.w = nn.Parameter(torch.randn(1)) #Initialize weigth 
		self.b = nn.Parameter(torch.randn(1)) #Initialize bias
	
	def forward(self, x): # Forward propagation
		y = self.w * x + self.b
		return y

# Let's try the model:
plt.scatter(train_X, train_Y) #draw data without train

lreg = LinearRegression() #Intantiate the class
test_X = torch.FloatTensor(test_X) # We create the tensor
y_ = lreg(test_X) #We call the method forward
plt.scatter(test_X.data.numpy(), y_.data.numpy()) #draw the predictions
print('w initial value: ', lreg.w.item())
print('b initial value: ', lreg.b.item())
plt.show()

# As we can see, the line above look a bit out of fit with respect to the data as the
# parameters w and b in our model are initialized randomly


# BACKPROPAGATION and STOCHASTIC GRADIENT DESCENT

def shuffle_dataset(X, Y):
	joint = torch.cat((X, Y), dim=1) #concatenate X & Y
	joint = list(torch.chunk(joint, len(joint), dim=0)) #Split the joint tensor into a list of individual rows (chunks)
	shuffle(joint)
	joint = torch.cat(joint, dim=0) #Concatenate the shuffled rows back together
	return torch.chunk(joint, 2, dim=1) #Split the concatenated tensor back into two parts (X and Y)

lreg = LinearRegression()
# Fit all training data
X = torch.FloatTensor(train_X).view(-1, 1)
Y = torch.FloatTensor(train_Y).view(-1, 1)
NUM_EPOCHS = 10 #number of iterations
BATCH_SIZE=5
LR = 1e-2 #learning rate

# define MSE as the cost function
cost = F.mse_loss

opt = optim.SGD(lreg.parameters(), lr=LR) #stochastic gradient descendent
avg_loss = None
avg_weight = 0.1
losses = []
for epoch in range(1, NUM_EPOCHS + 1): #desde 1 fins a NUM_EPOCHS

	# Iterate like this: [0, BATCH_SIZE), [BATCH_SIZE, 2 * BATCH_SIZE), ...
	for (beg_i, end_i) in zip(range(0, len(X) - BATCH_SIZE + 1, BATCH_SIZE), range(BATCH_SIZE, len(X), BATCH_SIZE)): #process the batchs
		#We take the data set of BATCH_SIZE
		x = X[beg_i:end_i] 
		y = Y[beg_i:end_i]

		y_ = lreg(x) #call the class and predict
		loss = cost(y, y_) #calculate the loss
		loss.backward() #calculate the gradient with backward propagation
		opt.step() #do the stochastic gradient descendent and update parameters
		opt.zero_grad() #put the acomulate gradients to 0

		# Smooth the loss value that is saved to be plotted later
		if avg_loss:
			avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
		else:
			avg_loss = loss.item() #extract the scalar value
		losses.append(avg_loss)

	# Shuffle the data in the training batch to regularize training
	X, Y = shuffle_dataset(X, Y)

plt.ylabel('Smoothed loss by factor {:.2f}'.format(1 - avg_weight))
plt.xlabel('Iteration step')
plt.plot(losses) #we draw the loss and print it
plt.show()


# EXERCISE: write the code to predict the y values x=[0.5,5,8.75] and plot them overlayed with the previous plot.

x = torch.FloatTensor([0.5, 5, 8.75]) # we create the tensor
y = lreg(x) # we calculate the predictions
print(y.data.numpy())

plt.scatter(train_X, train_Y)
plt.scatter(test_X.data.numpy(), y_.data.numpy())
plt.scatter(x.data.numpy(), y.data.numpy(), s=200)
plt.show()
# We are going to print the original points, the predictions and the predictions obteined for new values x=[0.5,5,8.75]


# LOGISTIC REGRESSION:

# we use it to establish a boundary between two classes.
# Hence, it can be used to determine if for an input, we have either one outcome or 
# the other one, so used in binary classification problems.

# We first generate some training data points
NUM_SAMPLES = 1000

# Binary classification
class_0 = np.random.randn(NUM_SAMPLES, 2) # center (0,0)
class_1 = np.random.randn(NUM_SAMPLES,2 ) + 2.5 #center (2.5, 2.5)

train_X = np.concatenate((class_0, class_1), axis=0)
train_Y = np.concatenate((np.zeros((NUM_SAMPLES,)), np.ones((NUM_SAMPLES,))), axis=0)

_ = plt.scatter(class_0[:, 0], class_0[:, 1], alpha=0.15)
_ = plt.scatter(class_1[:, 0], class_1[:, 1], alpha=0.15)
_ = plt.scatter([0], [0], s=200, color='blue')
_ = plt.scatter([2.5], [2.5], s=200, color='red')


class LogisticRegression(nn.Module):

	def __init__(self):
		super().__init__()
		# Linear projection
		self.proj = nn.Linear(2, 1) #y=x*A^T + b

		# Sigmoid activation
		self.act = nn.Sigmoid()

	def forward(self, x):
		# Combine the linear layer with the sigmoid activation
		y = self.act(self.proj(x))
		return y	


# Create an instance of a logistic regressor
loreg = LogisticRegression()


# Fit all training data
X = torch.FloatTensor(train_X).view(-1, 2)
Y = torch.FloatTensor(train_Y).view(-1, 1)
NUM_EPOCHS = 100
BATCH_SIZE=15
LR = 1e-1

# define binary corss entropy as the cost function
cost = F.binary_cross_entropy

opt = optim.SGD(loreg.parameters(), lr=LR)
avg_loss = None
avg_weight = 0.1
losses = []

for epoch in range(1, NUM_EPOCHS + 1): #desde 0 fins a NUM_EPOCHS
    for (beg_i, end_i) in zip(range(0, len(X) - BATCH_SIZE + 1, BATCH_SIZE), range(BATCH_SIZE, len(X), BATCH_SIZE)):
        x = X[beg_i:end_i]
        y = Y[beg_i:end_i]        
        y_ = loreg(x) #cridem a la classe
        loss = cost(y_, y) #calculem els costos crdidant a la funcio del mse
        loss.backward() #fem la retropropagació
        opt.step() #frm l'algoritme del gradient descendent i actualitzem els parametres
        opt.zero_grad() #tornem a inicialitzar a 0        
        # Smooth the loss value that is saved to be plotted later
        if avg_loss:
          avg_loss = avg_weight * loss.item() + (1 - avg_weight) * avg_loss
        else:
            avg_loss = loss.item()
        losses.append(avg_loss)
    # Shuffle the data in the training batch to regularize training
    X, Y = shuffle_dataset(X, Y)

plt.ylabel('Smoothed loss by factor {:.2f}'.format(1 - avg_weight))
plt.xlabel('Iteration step')
plt.plot(losses)
# Plot the trainig curve of the loss function
plt.ylabel('Smoothed loss by factor {:.2f}'.format(1 - avg_weight))
plt.xlabel('Iteration step')
plt.plot(losses)


# We are going to define a function make_logistic_surface(logistic) to visualize the decision
# boundary of logistic regresion model in 3D.

def make_logistic_surface(logistic):
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.set_title('Logistic regression probability surface. Blue dot: zero-class centroid. Orange dot: one-class centroid.')
	X = []
	y_coords = np.linspace(-4, 6, 100)
	x_coords = np.linspace(-4, 6, 100)
	for n in y_coords:
		for m in x_coords:
			X.append([n, m])
	X = torch.FloatTensor(X)
	Y_ = logistic(X)
	Y_ = Y_.data.numpy()
	sidx = 0
	surface = np.zeros((100, 100))
	xc, yc = np.meshgrid(x_coords, y_coords)
	for n in range(100):
		for m in range(100):
			surface[n, m] = Y_[sidx]
			sidx += 1

	surf = ax.plot_surface(xc, yc, surface, cmap=cm.coolwarm,
		linewidth=0, antialiased=True, alpha=0.5)
	_ = ax.scatter([0], [0], [1], s=200)
	_ = ax.plot([0, 1e-3], [0, 1e-3], [0, 1], linewidth=3, color='blue')
	_ = ax.scatter([2.5], [2.5], [1], s=200)
	_ = ax.plot([2.5, 2.5 + 1e-3], [2.5, 2.5 + 1e-3], [0, 1], linewidth=3, color='red')


# RANDOM LOGISTIC REGRESSION:

# We put a random seed that we know can give a very bad initialization
# completely giving zeros to class one, and vice-versa with class zero

_ = torch.manual_seed(3)
loreg_nt = LogisticRegression()
make_logistic_surface(loreg_nt)


# TRAINED LOGISTIC REGRESSION:

make_logistic_surface(loreg)