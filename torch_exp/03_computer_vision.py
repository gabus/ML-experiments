import torch
from torch import nn
from torchvision import transforms  # functions to manipulate image data to be suitable for training model
from torchvision import models  # pretrained computer vision models
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from loguru import logger
from torchvision.transforms import v2
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from tqdm.auto import tqdm

# Convolutional neural network work best with computer vision problems
# Transformers is a new a potentially better approach


# todo
#   [] is it possible to add another class without retraining entire model?
#   [] how to train a model with only few images?
#   [] how to load a model and use it? Do i need to create a class and specify its parameters?

BATCH_SIZE = 32


class ComputerVisionModelV1(nn.Module):
	"""
		input: batch_size, width, height, colour_ch (NHWC)
		output: as many as there are classification classes
	"""

	def __init__(self, in_features: int, out_features: int, hidden_nodes: int):
		# todo
		#   use baseline model and train on top

		# nn.Flatten layer flattens x,y into one dimension

		super().__init__()

		self.layer_stack = nn.Sequential(
			nn.Flatten(),
			nn.Linear(in_features=in_features, out_features=hidden_nodes),
			nn.Linear(in_features=hidden_nodes, out_features=out_features),
		)

		self.__output_activation_fn = nn.Softmax()
		self.__optimisation_fn = torch.optim.SGD(self.parameters(), lr=0.05)
		self.__loss_fn = nn.CrossEntropyLoss()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.layer_stack(x)

	def loss_fn(self, pred, actual):
		return self.__loss_fn(pred, actual)

	def output_activation_fn(self, x):
		return self.__output_activation_fn(x)

	def optimisation_fn(self):
		return self.__optimisation_fn


def dataset() -> tuple[datasets.mnist.FashionMNIST, datasets.mnist.FashionMNIST]:
	"""
	FashionMNIST
	"""
	train_data = datasets.FashionMNIST(
		root='data',  # download folder
		train=True,  # training or testing dataset?
		download=True,
		transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # how to transform the data
		target_transform=None,  # how to transform the labels/targets
	)

	test_data = datasets.FashionMNIST(
		root='data',  # download folder
		train=False,  # training or testing dataset?
		download=True,
		transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),  # how to transform the data
		target_transform=None,  # how to transform the labels/targets
	)
	# logger.warning(type(train_data))

	return train_data, test_data


def draw_random_items(items: datasets.mnist.FashionMNIST, seed: int = None):
	if seed:
		torch.manual_seed(seed)

	fig = plt.figure(figsize=(10, 10))

	rows, cols = 4, 4
	for i in range(1, rows * cols + 1):
		random_index = torch.randint(0, len(items), size=[1]).item()
		img, label = items[random_index]
		fig.add_subplot(rows, cols, i)
		plt.imshow(img.squeeze(), cmap='gray')
		plt.title(items.classes[label])
		plt.show(block=False)
		plt.axis(False)


def train_loop(model, train_dataset, test_dataset, epochs: int):
	for epoch in tqdm(range(epochs)):
		logger.warning(f'Epoch: {epoch}')

		for batch, (image, label) in enumerate(train_dataset):
			model.train()

			train_logit = model(image)

			train_loss = model.loss_fn(train_logit, label)

			model.optimisation_fn().zero_grad()

			train_loss.backward()

			model.optimisation_fn().step()

		if batch % 400 == 0:
			eval_model(model, test_dataset)


def eval_model(model: nn.Module, data_loader: torch.utils.data.DataLoader, accuracy_fn) -> dict:
	model.eval()

	with torch.inference_mode():
		for test_image, test_label in data_loader:
			test_logit = model(test_image)

			test_loss = model.loss_fn(test_logit, test_label)

			acc = accuracy_fn()

	return {
		'model': model.__class__.__name__,
		'accuracy': acc,
	}


train_data, test_data = dataset()
draw_random_items(train_data, 42)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

logger.info(f'Train Data Loader length: {len(train_dataloader)} in batches of {train_dataloader.batch_size}')

train_features_batch, train_labels_batch = next(iter(train_dataloader))
logger.warning(train_features_batch.size())
logger.warning(len(train_features_batch))

model_0 = ComputerVisionModelV1(in_features=28 * 28, out_features=len(train_data.classes), hidden_nodes=10)
model_0.to('cpu')

train_loop(model_0, train_dataloader, test_dataloader, 3)

plt.show(block=False)
