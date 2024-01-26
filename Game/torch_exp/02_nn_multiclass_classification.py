from sklearn.metrics import accuracy_score
import torch
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch import nn
from loguru import logger
from helper_functions import plot_decision_boundary
import time

NUM_CLASSES = 10
NUM_FEATURES = 2
RANDOM_SEED = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MultiClassModel(nn.Module):

	def __init__(self, in_features: int, out_features: int, hidden_nodes: int = 8):
		super().__init__()

		# todo for this particular problem non linear model is not needed! Just use straight lines!
		self.layer_stack = nn.Sequential(
			nn.Linear(in_features=in_features, out_features=hidden_nodes),
			# nn.ReLU(),  # todo <-------------------- don't need it! Amazing
			nn.Linear(in_features=hidden_nodes, out_features=hidden_nodes),
			# nn.ReLU(),  # todo <-------------------- don't need it! Amazing
			nn.Linear(in_features=hidden_nodes, out_features=out_features),
		)

		self.__output_activation_function = nn.Softmax(dim=1)
		self.__loss_fn = nn.CrossEntropyLoss()
		self.__optimiser_fn = torch.optim.SGD(params=self.parameters(), lr=0.1)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layer_stack(x)

	def output_activation_fn(self, x: torch.Tensor) -> torch.Tensor:
		# return torch.softmax(x, dim=1)
		return self.__output_activation_function(x)

	def loss_fn(self, pred, actual):
		return self.__loss_fn(pred, actual)

	def optimisation_fn(self):
		return self.__optimiser_fn


def generate_blob_data(n_samples: int = 3000):
	X_blob, y_blob = make_blobs(
		n_samples=n_samples,
		n_features=NUM_FEATURES,
		cluster_std=2,
		random_state=RANDOM_SEED,
		centers=NUM_CLASSES,
		shuffle=False,
	)

	X_blob = torch.from_numpy(X_blob).type(torch.float32)
	y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)

	X_train, X_test, y_train, y_test = train_test_split(X_blob, y_blob, train_size=0.8)

	logger.warning(torch.unique(y_train))

	return X_train, y_train, X_test, y_test


def plot_blobs(X_train, y_train, c, label: str, block: bool = False):
	plt.figure(figsize=(10, 7))
	plt.scatter(X_train, y_train, cmap=plt.cm.Dark2, c=c, s=5, label=label)
	plt.legend(prop={'size': 10})
	plt.title('Multi class classification')
	plt.show(block=block)


def calculate_accuracy(model, y_pred_logit, y_label):
	# todo in order to evaluate and train the model, we need to convert mode's logits into prediction probabilities
	#   and then prediction labels
	#   Logits -> pred probs -> pred labels
	#   output activation function is used to determine which class model this is correct
	y_pred = model.output_activation_fn(y_pred_logit)

	# this converts prediction probabilities into prediction labels. Just find model's best choice and use it as label
	# todo define dim... figure out what it does exactly
	y_pred_labels = y_pred.argmax(dim=1)
	y_pred_labels = torch.round(y_pred_labels)
	correct = torch.eq(y_label, y_pred_labels).sum()
	return correct / len(y_label) * 100


def training_loop(model: MultiClassModel, X_train, y_train, X_test, y_test, epochs):
	torch.manual_seed(RANDOM_SEED)
	torch.cuda.manual_seed(RANDOM_SEED)

	X_train, y_train = X_train.to(device=device), y_train.to(device=device)
	X_test, y_test = X_test.to(device=device), y_test.to(device=device)

	for epoch in range(epochs):
		model.train()

		y_pred_logit = model(X_train)

		# todo !!!!!!! loss function works with LOGITS!!!!!
		loss = model.loss_fn(y_pred_logit, y_train)

		model.optimisation_fn().zero_grad()

		loss.backward()

		model.optimisation_fn().step()

		if epoch % 10 == 0:
			model.eval()

			with torch.inference_mode():
				y_test_logits = model(X_test)
				test_loss = model.loss_fn(y_test_logits, y_test)
				test_acc = calculate_accuracy(model, y_test_logits, y_test)

			logger.info([f'Epoch: {epoch} | Train Loss: {loss} | Test loss: {test_loss} | Test accuracy: {test_acc}%'])


X_train, y_train, X_test, y_test = generate_blob_data()
plot_blobs(X_train[:, 0], X_train[:, 1], y_train, 'Train data')

model_0 = MultiClassModel(in_features=NUM_FEATURES, out_features=NUM_CLASSES, hidden_nodes=10)
model_0.to(device=device)

start_time = time.time()
training_loop(model_0, X_train, y_train, X_test, y_test, 10000)
logger.warning(f'Training time: {time.time() - start_time:0.2f}s')

model_0.eval()
with torch.inference_mode():
	plot_decision_boundary(model_0, X_test, y_test)

plt.show()

# Classification metrics
# Accuracy - how many samples it got right?
# Precision
# Recall
# F1-score
# Confusion matrix
# Classification report
