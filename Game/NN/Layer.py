import random

from NN.Node import Node


class Layer:

	def __init__(self, nodes_count):
		self.nodes = []
		self.name = ''

		# create nodes arrays in a layer
		for node in range(nodes_count):
			self.nodes.append(Node())

	def initialize_weights(self, previous_layer_count: int):
		for node in self.nodes:
			for c in range(previous_layer_count):
				node.input_weights.append(random.uniform(-1, 1))

	def as_dict(self):
		return {
			'name': self.name,
			'nodes': [n.as_dict() for n in self.nodes],
		}
