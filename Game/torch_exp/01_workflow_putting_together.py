import matplotlib.pyplot as plt
import torch
from loguru import logger
from torch import nn


# 45. Training loop (and testing loop) in Torch
# 1. loop though the data
# 2. forward pass (forward propagation)
# 3. calculate the loss
# 4. optimiser zero grad
# 5. loss backward (backpropagation)
# 6. optimiser step (gradient descent)
class MyNNClass(nn.Module):

	def __init__(self):
		super().__init__()
		self.linear_layer = nn.Linear(in_features=1, out_features=1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.linear_layer(x)


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


def generate_data():
	torch.manual_seed(42)

	bias = 0.3
	weight = 0.7

	# todo why big steps make model training fail?
	start = 0
	end = 1
	step = 0.01

	X = torch.arange(start, end, step).unsqueeze(dim=1)  # there are hidden functions which work with arrays
	y = X * weight + bias

	train_test_split = int(0.8 * len(X))
	X_train, y_train = X[:train_test_split], y[:train_test_split]
	X_test, y_test = X[train_test_split:], y[train_test_split:]

	return X_train, y_train, X_test, y_test


def training_loop(model: nn.Module):
	epochs = 200

	loss_fn = nn.L1Loss()
	optimizer_fn = torch.optim.SGD(params=model.parameters(), lr=0.01)

	for epoch in range(epochs):
		model.train()

		y_pred = model(X_train)

		loss = loss_fn(y_pred, y_train)

		optimizer_fn.zero_grad()

		loss.backward()

		optimizer_fn.step()

		model.eval()
		with torch.inference_mode():
			test_pred = model(X_test)

			test_loss = loss_fn(test_pred, y_test)

			if epoch % 10 == 0:
				logger.info(f'Epoch: {epoch} | Loss: {loss}, Test Loss: {test_loss}')


X_train, y_train, X_test, y_test = generate_data()
plot_predictions(X_train, y_train, X_test, y_test, title='Training and Testing data')

model_0 = MyNNClass()

logger.info(f'{model_0.state_dict()=}')

with torch.inference_mode():
	y_pred = model_0(X_test)

plot_predictions(X_train, y_train, X_test, y_test, y_pred, title='Predictions on untrained model')

training_loop(model_0)
logger.info(f'{model_0.state_dict()=}')

with torch.inference_mode():
	y_pred = model_0(X_test)

plot_predictions(X_train, y_train, X_test, y_test, y_pred, title='Predictions on trained model')

torch.save(model_0.state_dict(), 'models/wrapping_up_model.pth')

model_1 = MyNNClass()
loaded_model = torch.load('models/wrapping_up_model.pth')
model_1.load_state_dict(loaded_model)

with torch.inference_mode():
	y_pred = model_1(X_test)

plot_predictions(X_train, y_train, X_test, y_test, y_pred, title='Predictions on loaded model')

plt.show(block=True)
