from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from torch import nn  # neural networks stuff

from matlib_wapper import MatlibWrapper

t = torch.rand(size=(7, 3))
logger.info(t)

# data can be anything.. images, videos, excel spreadsheets, DNA, text...
# 1. convert data in numerical represntation
# 2. build the model to learn patterns in that numerical representation

# Create a model with known data using linear regression formula

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.02

# practical example when unsqueeze is useful. Turn array into array of arrays with one element each
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

# get first :x elements
logger.info(X[:10])
logger.info(y[:10])

# 36. -------------
# splitting data into training and test data

# 3 data sets!
# training set (60-80%), validation test, test data (10-20%)
# 80/20 is the most used

data_split_index = int(0.8 * len(X))
X_train, y_train = X[:data_split_index], y[:data_split_index]
X_test, y_test = X[data_split_index:], y[data_split_index:]

logger.info({f'{len(X_train)=}', f'{len(y_train)=}', f'{len(X_test)=}', f'{len(y_test)=}'})


# Visualise data
def plot_predictions(
		train_data: torch.Tensor,
		train_labels: torch.Tensor,
		test_data: torch.Tensor,
		test_labels: torch.Tensor,
		predictions: torch.Tensor = None,
		block_graph: bool = False,
		title=''
):
	plt.figure(figsize=(10, 7))
	plt.scatter(train_data, train_labels, c='blue', s=4, label='Training data')
	plt.scatter(test_data, test_labels, c='green', s=4, label='Testing data')

	if predictions is not None:
		plt.scatter(test_data, predictions, c='red', s=5, label='Predictions data')

	plt.legend(prop={'size': 10})
	plt.title(title)
	plt.show(block=block_graph)


# plot_predictions(X_train, y_train, X_test, y_test, block_graph=False, title='Training and Testing data')


def plot_loss(train_loss_values, test_loss_values):
	plt.clf()
	plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label='Train loss')
	plt.plot(epoch_count, test_loss_values, label='Test loss')
	plt.title('Loss curve')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend()
	plt.show()


# 38. build a model

# create linear regression model class
class LinearRegressionModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))
		self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float32))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.weights * x + self.bias


# NN adapts its parameters through two algorithms:
#   1. gradiant descent
#   2. backpropagation

# !!!!!!!!!!!!!!!!!!!!!!!
# https://youtu.be/Ilg3gGewQ5U - backpropagation explanation
# backpropagation finds the most rapid decrease to the cost in cost function
# stochastic gradient descent - like a drunk descent which is caused by running backpropagation in batches of data,
# rather than backpropagating after each iteration (forward function).
# stochastic gradient descent needs to have "step" defined which runs SGD only after certain steps

# ---------------------
# pytorch building essentials
#  torch.nn - all building blocks for computation graphs
#  torch.nn.Parameter - what parameters our model should try and learn.
#  torch.nn.Module - base class for all NN modules. If you inherit, you must override forward() method
#  torch.optim - pytorch optimisers; helps with gradient descent
#  def forward() - this method defines what happens in forward computation

# pytorch cheat sheet!!!
# https://pytorch.org/tutorials/beginner/ptcheat.html

# ------------------------
torch.manual_seed(42)

model_0 = LinearRegressionModel()
logger.info(model_0)
logger.info(list(model_0.parameters()))
logger.info(model_0.state_dict())  # .state_dict() gives named parameters

# Using current state of model, check prediction
# using torch.inference_mode()
# pass data through the model and it'll run through forward() method

with torch.inference_mode():
	y_preds = model_0(X_test)

# another way (not as good) to run the model without triggering backpropagation is no_grad()
# with torch.no_grad():
# 	y_preds = model_0(X_test)

logger.info(y_preds)
# todo uncomment
# plot_predictions(X_train, y_train, X_test, y_test, y_preds, True, title='Training, Testing and Prediction data')

# 43. ----------- training the model
# Transfer learning is just parameters coppying from another model
# to measure models performance use a loss function

# !!!!!!!!!!!!
# Loss function - tells me how badly model performs. This is just a calculation. Usually the lower, the better
# Optimizer - an algorithm which uses loss function calculation and updates model parameters to better match the problem
#   https://pytorch.org/docs/stable/optim.html#module-torch.optim

# 44. loss function and optimiser in Torch
# nn.L1Loss - mean absolute error loss function
# nn.MSELoss - squared mean absolute error (L2Loss)

# setup a loss function
loss_fn = nn.L1Loss()

# setup an optimiser (stochastic gradient descent)
optimzer_fn = torch.optim.SGD(
	model_0.parameters(),  # what parameters optimiser will work with
	lr=0.001  # learning rate (hyperparameter)
)

# 45. Training loop (and testing loop) in Torch
# 1. loop though the data
# 2. forward pass (forward propagation)
# 3. calculate the loss
# 4. optimiser zero grad
# 5. loss backward (backpropagation)
# 6. optimiser step (gradient descent)

# todo i thought backpropagation uses gradient descent. Why these two steps are separate in model training workflow?

# epoch - how many times the same data will be fed

eopchs = 2000  # another hyperparameter

epoch_count = []
train_loss_values = []
test_loss_values = []

graph_1 = MatlibWrapper()  # imidiatelly draws

for epoch in range(eopchs):
	model_0.train()  # set the model to training mode. Sets parameters to gradient true

	# 2. forward pass
	y_pred = model_0(X_train)

	# 3. calculate loss
	loss = loss_fn(y_pred, y_train)  # (predictions, labels)
	# logger.info(f'{loss=}')

	# 4. optimiser zero grad
	optimzer_fn.zero_grad()

	# 5. backpropagation on the loss with respect with parameters of the model
	loss.backward()

	# 6. step the optimiser (perform gradient descent). Updates model parameters
	optimzer_fn.step()

	# todo do i need both model_0.eval() and torch.inference_mode() for testing? Why there are two? What's the exact difference?
	# testing code
	model_0.eval()  # turns off gradient tracking (used for testing) (turns off dropout/bath norm layers)
	with torch.inference_mode():
		# 1. do forward pass
		test_pred = model_0(X_test)

		# 2. Calculate the loss
		test_loss = loss_fn(test_pred, y_test)

	if epoch % 10 == 0:
		epoch_count.append(epoch)
		train_loss_values.append(loss)
		test_loss_values.append(test_loss)
		logger.info(f'Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss}')
		logger.info(f'{model_0.state_dict()=}')

plot_loss(train_loss_values, test_loss_values)
plot_predictions(X_train, y_train, X_test, y_test, test_pred, True, 'Prediction after model training')

# Saving a model!!
# There are 3 main methods for saving and loading models
# 1. torch.save() - saves in torch pickle format
# 2. torch.load() - loads
# 3. torch.nn.Module.load_state_dict() - loads model parameters

MODEL_PATH = Path('models')
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = 'model_0.pth'  # pytorch .pt or .pth file extensions
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


def save() -> None:
	torch.save(model_0.state_dict(), MODEL_SAVE_PATH)


def load() -> LinearRegressionModel:
	model = LinearRegressionModel()
	model.load_state_dict(torch.load(MODEL_SAVE_PATH))
	return model


save()

loaded_model_0 = load()

logger.info(f'{loaded_model_0.state_dict()=}')

# 6 ---------------------------- putting it all together ------------------------------
