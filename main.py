from pprint import pprint

import numpy as np

# VOCABULARY
# mnist database - big db of hand written digits
# gradient descent
# derivative
# transposed - flipped. column becomes row and vice versa
# activation function - sigmoid,tanh, ReLu, softmax (for output layer) function

# WORKFLOW
# feed network with data
# compare output with desired output -- calculate error
# adjust parameters for neural network to reduce error rate. using "gradient descent" algorithm for example

# LAYERS
# fully connected
# convolutional
# drop out
# activation

"""
 x1   y1
 x2   y2
      y3
"""

if __name__ == '__main__':
	# input = np.matrix([
	# 	[1],  # x1
	# 	[2],  # 2x
	# ])

	weights = np.random.randn(3, 2)

	# weights = np.matrix([
	# 	[1, 1],  # w11, w12 (x1 -> y1, x2 -> y1)
	# 	[1, 1],  # w21, w22
	# 	[1, 1],  # w31, w32
	# ])

	# output = np.matrix([
	# 	[0],
	# 	[0],
	# 	[0], 
	# ])

	# rand_matrix = np.random.randn(3, 2)

	# logger.info(input)
	# logger.info(output)
	# logger.info(weights)

	input = np.random.randn(1, 2)

	pprint(input)
	# pprint(output)
	pprint(weights)

	output1 = np.dot(300, weights)
	output2 = weights * 300

# logger.info(np.dot(inputs, inputs2))
