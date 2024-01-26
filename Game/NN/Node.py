import random


class Node:

	def __init__(self):
		self.node_value = 0
		self.input_weights = []
		self.bias = random.uniform(-1, 1)
		self.meta_data = ''

	# def propagate_forward(self):
	# 	return np.dot(self.input_weights, self.weight) + self.bias

	def as_dict(self):
		return {
			'input_weights': self.input_weights,
			'bias': self.bias,
			'node_value': self.node_value,
		}
