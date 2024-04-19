import torch
from loguru import logger
from torch import nn
from torch.nn import Module, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from torch_class import TorchClass

train_data = datasets.FashionMNIST(
	root='data',
	train=True,
	download=True,
	transform=ToTensor()
)

test_data = datasets.FashionMNIST(
	root='data',
	train=False,
	download=True,
	transform=ToTensor()
)

batch_size = 64
device = 'cpu'

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# logger.info(test_dataloader[0])

# todo what's x and y?
for x, y in train_dataloader:
	logger.info(x.shape)
	logger.info(x.dtype)
	break

my_network = TorchClass()
my_network.to(device)

logger.info(my_network)


# my_network.train()


def train(dataloader: DataLoader, model: Module, loss_fn: CrossEntropyLoss, optimizer):
	size = len(dataloader.dataset)
	model.train()

	for batch, (x, y) in enumerate(dataloader):
		x = x.to(device)
		y = y.to(device)

		pred = model(x)  # todo what uses forward function then?
		loss = loss_fn(pred, y)

		# Backpropagation
		loss.backward()  # todo why there is no IDE autocomplete support?
		optimizer.step()
		optimizer.zero_grad()

		if batch % 100 == 0:
			loss = loss.item()
			current = (batch + 1) * len(x)
			logger.warning("loss: {:>7f} current: {:>5d} / {}".format(loss, current, size))


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_network.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
	logger.info("---------- Epoch: {} ----------".format(t + 1))
	train(train_dataloader, my_network, loss_fn, optimizer)

logger.success("done")
