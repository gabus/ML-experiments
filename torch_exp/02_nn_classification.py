import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

from helper_functions import plot_decision_boundary


# multi-label classification vs multi class classification
# multi-label - one object will contain multiple tags or labels (red, round, small - tomato)
# multi-class - multiple objects to classify. But only one tag or label (car, dog, chicken)

# Neural network classification
# (distinguish between objects)

def draw_circles(x, y, c, block=False):
	plt.scatter(x=x, y=y, c=c, cmap=plt.cm.RdYlBu)
	plt.show(block=block)


n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

logger.info(X[:4])
logger.info(y[:4])

logger.info(X[:3, 0])

circles_df = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'label': y})

draw_circles(x=X[:, 0], y=X[:, 1], c=y)

# convert data into tensors
logger.info(type(X))
X_tensor = torch.from_numpy(X).type(torch.float32)
y_tensor = torch.from_numpy(y).type(torch.float32)

logger.info(X_tensor)
logger.info(y_tensor[:5])

# split data into training and testing
train_test_split_ration = 0.8
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

logger.info([len(X_train), len(X_test), len(y_train), len(y_test)])

# build a model
# 1. device agnostic code
# 2. construct a model
#     2 nn.Linear() layers (works for this problem)
# 3. define loss function, optimiser
# 4. Create a training and testing loop

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CirclesModelV0(nn.Module):

	def __init__(self):
		super().__init__()

		self.layer_1 = nn.Linear(2, 10)
		self.layer_2 = nn.Linear(10, 1)

	def forward(self, x):
		return self.layer_2(self.layer_1(x))


class CirclesModelV1(nn.Module):

	def __init__(self):
		super().__init__()

		self.layers = nn.Sequential(
			nn.Linear(in_features=2, out_features=5),
			nn.Linear(in_features=5, out_features=1),
		)

	def forward(self, x):
		return self.layers(x)


model_0 = CirclesModelV0().to(device=device)
# check which device it's running on
logger.info(next(model_0.parameters()).device)

# another way to create simple model
model_inline = nn.Sequential(
	nn.Linear(in_features=2, out_features=5),
	nn.Linear(in_features=5, out_features=1),
).to(device=device)

loss_fn = nn.BCEWithLogitsLoss()  # it contains Sigmoid activation function
optimisation_fn = torch.optim.SGD(params=model_0.parameters(), lr=0.1)  # adam is another common optimiser


# todo why do i need this as there's loss function for this?
def accuracy_fn(y_true, y_pred):
	correct = torch.eq(y_true, torch.round(y_pred)).sum().item()
	return correct / len(y_pred) * 100


# Train the model
# training loop
# testing loop
# logits is the data before activation function (raw node data from the model)

logger.info(model_inline)

model_inline.eval()
with torch.inference_mode():
	y_logit = model_inline(X_test.to(device))

logger.info(y_logit[:8])

# convert logits to prediction probabilities
y_test_probs = torch.sigmoid(y_logit)
logger.info(y_test_probs[:8])

# find prediction labels
y_preds = torch.round(y_test_probs)
logger.info(y_preds[:8])

# building a training loop
epochs = 100

# X_train - training data
# y_train - known answers to the data..
for epoch in range(epochs):
	model_inline.train()

	y_logits = model_inline(X_train).squeeze()

	y_pred = torch.round(torch.sigmoid(y_logits))

	# BCEWithLogitsLoss - this loss functions expects raw input (logits)
	# BCELoss - this loss functions input which is run thought activation function first
	loss_train = loss_fn(y_logits, y_train)

	accuracy = accuracy_fn(y_train, y_pred)

	optimisation_fn.zero_grad()

	# backpropagation
	loss_train.backward()

	# gradient descent (apply values)
	optimisation_fn.step()

	model_inline.eval()
	with torch.inference_mode():
		# forward pass
		test_logits = model_0(X_test).squeeze()
		test_pred = torch.round(torch.sigmoid(test_logits))

		# calculate accuracy
		test_loss = loss_fn(test_logits, y_test)
		test_acc = accuracy_fn(y_test, test_pred)

	if epoch % 10 == 0:
		logger.warning(
			f'Epoch: {epoch} | Loss: {loss_train:.5f} | Accuracy: {accuracy:0.5f} | Test Loss: {test_loss:0.5f} | Test Accuracy: {test_acc:0.5f}')

# Epoch: 0 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 10 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 20 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 30 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 40 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 50 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 60 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 70 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 80 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000
# Epoch: 90 | Loss: 0.75261 | Accuracy: 52.25000 | Test Loss: 0.70420 | Test Accuracy: 49.50000

# model isn't learning anything..
# let's visualise predictions
plt.figure()
plt.subplot(1, 3, 1)
plt.title("Train")
plot_decision_boundary(model_inline, X_train, y_train)
plt.subplot(1, 3, 2)
plt.title("Test")
plot_decision_boundary(model_inline, X_test, y_test)


# plt.show(block=False)


# improving the model (hyperparameters)
# add more layers
# add more nodes
# more epochs
# change activation function
# change learning rate (exploding gradiant problem, vanishing gradiant problem - goes to 0 too quickly)
# improvement from data perspective - have more data so the model has more opportunity to learn

# !!!! import torch.utils.tensorboard - a tool to monitor training

# todo if BCEWithLogitsLoss is used, do i need to define any activaiton functions at all????

class CirclesModelV3(nn.Module):

	def __init__(self):
		super().__init__()

		self.layer_1 = nn.Linear(in_features=2, out_features=10)
		self.layer_2 = nn.Linear(in_features=10, out_features=10)
		self.layer_3 = nn.Linear(in_features=10, out_features=1)

	def forward(self, x):
		return self.layer_3(self.layer_2(self.layer_1(x)))


model_3 = CirclesModelV3()
model_3.to(device=device)

loss_fn_BCELoss = nn.BCELoss()
optimisation_fn_model_v3 = torch.optim.SGD(model_3.parameters(), lr=0.1)

epochs = 1000
for epoch in range(epochs):
	model_3.train()

	y_logit = model_3(X_train).squeeze()

	# todo do i need activation funciton?
	#  no, since loss function is BCEWithLogitsLoss (it has sigmoid activation function integrated)
	#  update: loss function changed, now i need activation function

	y_train_pred = torch.sigmoid(y_logit)

	loss = loss_fn_BCELoss(y_train_pred, y_train)

	optimisation_fn_model_v3.zero_grad()

	loss.backward()

	optimisation_fn_model_v3.step()

	# test the model
	if epoch % 10 == 0:
		model_3.eval()
		with torch.inference_mode():
			y_test_pred = model_3(X_test).squeeze()

			test_loss = loss_fn(y_test_pred, y_test)

			logger.info(f'Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}')

plt.subplot(1, 3, 3)
plt.title("v3 model test")
plot_decision_boundary(model_3, X_test, y_test_pred)
plt.show()
