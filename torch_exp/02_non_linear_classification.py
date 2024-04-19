import torch.optim
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from torch import nn

from helper_functions import plot_decision_boundary

n_data_samples = 1000
random_seed = 42


class NonLinearModelV0(nn.Module):

	def __init__(self):
		super().__init__()

		self.layer_1 = nn.Linear(in_features=2, out_features=10)
		self.layer_2 = nn.Linear(in_features=10, out_features=10)
		self.layer_3 = nn.Linear(in_features=10, out_features=1)
		self.hidden_layers_activation_fn = nn.ReLU()

		# self.__loss_fn = nn.BCEWithLogitsLoss()
		self.__loss_fn = nn.BCELoss()
		self.__optimiser_fn = torch.optim.SGD(self.parameters(), lr=0.1)
		self.__output_activation_fn = nn.Sigmoid()

	def forward(self, x):
		return self.layer_3(self.hidden_layers_activation_fn(
			self.layer_2(self.hidden_layers_activation_fn(
				self.layer_1(x)
			))
		))

	def activation_fn(self, x):
		return self.__output_activation_fn(x)

	def loss_fn(self, y_preds, y_true_label):
		return self.__loss_fn(y_preds, y_true_label)

	def optimiser_fn(self):
		return self.__optimiser_fn


class LinearModelV0(nn.Module):
	def __init__(self):
		super().__init__()

		self.layer_1 = nn.Linear(in_features=1, out_features=1)

		self.__loss_fn = nn.L1Loss()
		self.__optimiser_fn = torch.optim.SGD(self.parameters(), lr=0.01)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layer_1(x)

	@staticmethod
	def activation_fn(x):
		return x

	def loss_fn(self, y_pred, y_true):
		return self.__loss_fn(y_pred, y_true)

	def optimiser_fn(self):
		return self.__optimiser_fn


def draw_circles(x, y, c, label: str, block: bool = False):
	plt.figure()
	plt.title(label)
	plt.scatter(x, y, c=c, cmap=plt.cm.Dark2, s=1)
	# plt.scatter(x, y, c=c, cmap=plt.cm.RdYlBu)
	plt.show(block=block)


def draw_linear_graph(X_train, X_test, y_train, y_test, label: str, y_pred=None, block: bool = False):
	plt.figure(figsize=(10, 7))
	plt.title(label)
	plt.scatter(X_train, y_train, c='red', s=3)
	plt.scatter(X_test, y_test, c='blue', s=3)

	if y_pred is not None:
		plt.scatter(X_test, y_pred, c='grey', s=4)

	plt.show(block=block)


def generate_linear_data():
	start = 0
	end = 1
	step = 0.01

	weight = 0.7
	bias = 0.3

	X = torch.arange(start, end, step).unsqueeze(dim=1)
	y = X * weight + bias

	data_split = int(0.8 * len(X))
	X_train, y_train = X[:data_split], y[:data_split]
	X_test, y_test = X[data_split:], y[data_split:]

	# logger.info([len(X_train), len(y_train), len(X_test), len(y_test)])
	# draw_linear_graph(X_train, X_test, y_train, y_test, 'Train/Test data')

	return X_train, X_test, y_train, y_test


def generate_test_train_circle_data(n_samples: int):
	X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=random_seed)
	X = torch.from_numpy(X).type(torch.float32)
	y = torch.from_numpy(y).type(torch.float32)

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

	# draw_circles(X_train[:, 0], X_train[:, 1], y_train, 'Train data')
	# draw_circles(X_test[:, 0], X_test[:, 1], y_test, 'Test data')

	return X_train, X_test, y_train, y_test


def accuracy_fn(y_true, y_pred):
	correct = torch.eq(y_true, torch.round(y_pred)).sum().item()
	return correct / len(y_pred) * 100


def non_linear_training_loop(model: NonLinearModelV0, epochs: int, X_train, X_test, y_train, y_test):
	logger.error(X_train.shape)
	logger.error(y_train.shape)

	for epoch in range(epochs):
		model.train()

		y_logit = model(X_train).squeeze()

		y_pred = model.activation_fn(y_logit)

		# todo be very careful which loss function is used.. if it's BCEWithLogitsLoss don't use output activation function!
		#   idea: don't use BCEWithLogitsLoss.. just define both loss function and output activation function all the time
		train_loss = model.loss_fn(y_pred, y_train)

		model.optimiser_fn().zero_grad()

		train_loss.backward()

		model.optimiser_fn().step()

		if epoch % 100 == 0:
			model.eval()
			with torch.inference_mode():
				y_test_logit = model(X_test).squeeze()

				y_test_pred = model.activation_fn(y_test_logit)

				test_loss = model.loss_fn(y_test_pred, y_test)

				test_accuracy = accuracy_fn(y_test, y_test_pred)

				logger.info(
					f'Epoch: {epoch} | Train loss: {train_loss} | Test loss: {test_loss} | Test accuracy: {test_accuracy}')


def train_linear_model(model: LinearModelV0, epochs: int, X_train, X_test, y_train, y_test):
	for epoch in range(epochs):
		model.train()

		y_pred = model(X_train)

		train_loss = model.loss_fn(y_pred, y_train)

		model.optimiser_fn().zero_grad()

		train_loss.backward()

		model.optimiser_fn().step()

		if epoch % 100 == 0:
			model.eval()
			with torch.inference_mode():
				y_test_pred = model(X_test)
				test_loss = model.loss_fn(y_test_pred, y_test)

				logger.info(f'Epoch: {epoch} | Train loss: {train_loss} | Test loss: {test_loss}')


# ================ Non Linear model training ================

X_train, X_test, y_train, y_test = generate_test_train_circle_data(n_data_samples)
model_0 = NonLinearModelV0()
non_linear_training_loop(model_0, 2000, X_train, X_test, y_train, y_test)

plot_decision_boundary(model_0, X_test, y_test)

# ================ Non Linear model training END ================


# ================ Linear model training ================

# X_train, X_test, y_train, y_test = generate_linear_data()
#
# model_1 = LinearModelV0()
# logger.warning(model_1.state_dict())
# train_linear_model(model_1, 1000, X_train, X_test, y_train, y_test)
# logger.warning(model_1.state_dict())
#
# model_1.eval()
# with torch.inference_mode():
# 	y_pred = model_1(X_test)
# 	draw_linear_graph(X_train, X_test, y_train, y_test, 'Prediction on trained model', y_pred)

# ================ Linear model training END ================


plt.show(block=True)
