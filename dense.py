import numpy as np

from layer import Layer


class Dense(Layer):

	def __init__(self, input_size: int, output_size: int):
		self.input = None
		self.weights = np.random.randn(output_size, input_size)
		self.bias = np.random.randn(input_size)

	def forward(self, input):
		self.input = input
		return np.dot(self.weights, input) + self.bias

	def backward(self, output_gradient, learning_rate):
		# calculate error
		# 
		pass
