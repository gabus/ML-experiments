import torch
from loguru import logger

# 1.

# 2.
rand_tensor = torch.rand((7, 7))
logger.info(rand_tensor)

# 3.
rand_tensor_1 = torch.rand((7, 7))
rand_tensor_2 = torch.rand((1, 7))
rands_mult = torch.mm(rand_tensor_1, rand_tensor_2.T)
logger.info(rands_mult)

# 4.
torch.manual_seed(0)
rand_tensor_1 = torch.rand((7, 7))
rand_tensor_2 = torch.rand((1, 7))
rands_mult = torch.mm(rand_tensor_1, rand_tensor_2.T)
logger.info(rands_mult)
logger.info(rands_mult.size())

# 1.
# 1.
# 1.
# 1.
# 1.
# 1.
