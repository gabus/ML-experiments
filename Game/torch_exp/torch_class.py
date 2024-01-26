from torch import nn


class TorchClass(nn.Module):

	def __init__(self):
		super().__init__()

		self.flatten = nn.Flatten()

		self.linear_relu_stack = nn.Sequential(
			nn.Linear(28 * 28, 512),  # input - images are 28x28 pixels, output goes into 512 hidden layer
			nn.ReLU(),
			nn.Linear(512, 512),  # hidden layer - in/out 512 nodes
			nn.ReLU(),
			nn.Linear(512, 10),  # comes in 512 from hidden layer, goes out into 10 possible labels
			# 	todo what defines last activation function?
		)

	def forward(self, x):
		x = self.flatten(x)  # todo does this flattens picture into one line of numbers? (for first layer)
		logits = self.linear_relu_stack(x)  # todo what's this???
		return logits
