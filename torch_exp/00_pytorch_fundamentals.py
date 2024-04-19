import numpy as np
import torch
from loguru import logger

logger.info(torch.__version__)

scalar = torch.tensor(7.1)
# scalar = torch.Tensor(7.1)  # todo why there are Tensor and tensor??

logger.info(scalar)
logger.info(scalar.data)
logger.info(type(scalar.item()))
logger.info(scalar.item())  # get tensor back as python data type

vector = torch.tensor([4, 6])
logger.info(vector.ndim)  # get vector dimension
logger.info(vector.shape)  # get vector shape

matrix = torch.tensor([[7, 8], [10, 11]])
logger.info(matrix)
logger.info(matrix.ndim)  # get vector dimension
logger.info(matrix.shape)  # get vector shape

tensor = torch.tensor([[[1, 2, 3], [3, 5, 6], [7, 8, 9]]])
logger.info(f'{tensor.ndim=}')  # get vector dimension
logger.info(f'{tensor.shape=}')  # get vector shape

# random tensors
random_tensor = torch.rand(3, 4)
logger.info(f'{random_tensor=}')  # get vector dimension
logger.info(f'{random_tensor.ndim=}')  # get vector dimension
logger.info(f'{random_tensor.shape=}')  # get vector shape

random_image_size_tensor = torch.rand(size=(224, 224, 3))  # height, width, colour channel (r,g,b)
# logger.info(f'{random_image_size_tensor=}')  # get vector dimension
logger.info(f'{random_image_size_tensor.ndim=}')  # get vector dimension
logger.info(f'{random_image_size_tensor.shape=}')  # get vector shape

# zeros or ones tensors
zeros = torch.zeros(size=(3, 4))
logger.info(f'{zeros=}')  # get vector dimension

ones = torch.ones(size=(3, 4))
logger.info(f'{ones=}')  # get vector dimension
logger.info(f'{ones.dtype=}')  # get tensor data type

# create range of tensors and tensors-like
arange = torch.arange(start=0, end=10, step=2)
logger.info(f'{arange=}')

like = torch.zeros_like(input=arange)
logger.info(f'{like=}')

# tensor data type
t_float_32 = torch.tensor(
	[3, 4, 5],
	dtype=torch.float32,
	device=None,  # 'cpu', 'cuda'
	requires_grad=False,
)
logger.info(f'{t_float_32=}')
logger.info(f'{t_float_32.dtype=}')
logger.info(f'{t_float_32.device=}')
logger.info(f'{t_float_32.requires_grad=}')
logger.info(f'{t_float_32.shape=}')
logger.info(f'{t_float_32.size()=}')

# manipulating tensors
# * addition
# * subtraction
# * multiplication
# * division
# * matrix multiplication

tensor = torch.tensor([1, 2, 3])
tensor = tensor + 10
logger.info(f'{tensor=}')

tensor = torch.tensor([1, 2, 3])
tensor = tensor * 10
logger.info(f'{tensor=}')
logger.info(f'{tensor.dtype=}')

tensor = torch.tensor([1, 2, 3])
tensor = tensor / 10
logger.info(f'{tensor=}')
logger.info(f'{tensor.dtype=}')

tensor = torch.mul(tensor, 100)  # built in torch multiplication function
logger.info(f'{tensor=}')

tensor = torch.add(tensor, 100)  # built in torch addition function
logger.info(f'{tensor=}')

# Matrix multiplication (do product)
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
logger.info(f'{t1=}')

t2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
logger.info(f'{t2=}')

t_m = torch.matmul(t1, t2)  # there's dot product and multiplication
logger.info(f'{t_m=}')

# Matrix multiplication (do product)
t1 = torch.tensor([[1, 2], [4, 6]])
logger.info(f'{t1=}')

t2 = torch.tensor([[7, 8], [9, 10]])
logger.info(f'{t2=}')

t_m = torch.matmul(t1, t2)  # there's dot product and multiplication
t_m_p = t1 * t2  # there's dot product and multiplication
logger.info(f'{t_m=}')
logger.info(f'{t_m_p=}')

# Matrix multiplication (do product)
t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
logger.info(f'{t1=}')

t2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
logger.info(f'{t2=}')

t_m_p = t1 @ t2  # there's dot product and multiplication
logger.info(f'{t_m=}')

# todo what's the difference between python * and matmul() and @ ??
#   python * just multiplies each element from matrix with the same index of another matrix.
#   That's why shape has to be the same
#   matmul() is actual dot product of the matrices
#   python @ symbol is actual python dot product -- just like matmul()

# 21. matrix multiplication

# Rules:
#   * the inner dimensions must match
#   * the resulting matrix has the shape of the outer dimensions

#        inner - v                 v - inner
# torch.rand([2, 3]) @ torch.rand([2, 3])

#     outer - v                       v - outer
# torch.rand([2, 3]) @ torch.rand([2, 3])

# torch.rand([2, 3]) @ torch.rand([2, 3])  # won't work
torch.rand([3, 2]) @ torch.rand([2, 3])  # will work
torch.rand([2, 3]) @ torch.rand([3, 2])  # will work

#  multiplication errors

t_a = torch.tensor([[1, 2], [3, 4], [5, 6]])
t_b = torch.tensor([[7, 10], [8, 11], [9, 12]])
t_b_trans = t_b.T
t_a_trans = t_a.T

logger.info(f'{torch.mm(t_a, t_b_trans)=}')
logger.info(f'{torch.mm(t_a_trans, t_b)=}')

# todo which matrix should be transposed for multiplication???

#  tensor aggregation
#    min, max, mean, sum

x = torch.arange(5, 100, 10)
logger.info(f'{x.min()=}')
logger.info(f'{x.max()=}')
logger.info(f'{x.type(torch.float32).mean()=}')  # change data type only for calculation. don't change original variable

logger.info(f'{x.sum()=}')

# positional min or max value -- get value of max
logger.info(f'{x.argmin()=}')  # get index of min value
logger.info(f'{x.argmax()=}')  # get index of max value
logger.info(f'{x[9]=}')  # get value by index

# ----------------- experimenting

t1 = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
t2 = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])

logger.info(f'{t1 * t2=}')
logger.info(f'{t1 @ t2=}')
logger.info(f'{t1 @ t2.T=}')
logger.info(f'{t1.T @ t2=}')
logger.info(f'{t1.T @ t2.T=}')
logger.info(f'{t1 @ t2.T.T.T.T=}')

# reshaping, stacking, squeezing, un-squeezing
# reshape - reshape an input tensor to a defined shape todo (what for?)
# view - return a view of an input tensor of certain shape but keep the same memory as original tensor
# stacking - stacking two tensors on top of each other. hstack and vstack
# squeeze - removes dimensions of 1 from a tensor
# unsqueeze - add a dimension to a tensor
# permute - return a view of the input with dimensions permuted (swapped) in a cetrain way

multi_dim_t = torch.rand(size=(1, 3, 5))
logger.info(f'{multi_dim_t=}')

logger.info(f'{multi_dim_t[0][0]=}')
logger.info(f'{multi_dim_t[0,0]=}')  # get all items 1st dim
logger.info(f'{multi_dim_t[:,:, 0]=}')  # gets all 0 index items from 1st and 2nd dim

# ------------------ numpy ------------------

arr = np.arange(1.0, 8.0)
tensor = torch.from_numpy(arr)  # converting arrays from numpy defaults to float64 dtype
logger.info(f'{arr=}')
logger.info(f'{tensor=}')

tensor = torch.rand(15)
numpy_tens = tensor.numpy()
logger.info(f'{numpy_tens=}')
logger.info(f'{numpy_tens.dtype=}')

# --------- reproducibility --------------
# take out random out of random.
# Random seed!

rand_a = torch.rand(3, 4)
rand_b = torch.rand(3, 4)

logger.info(f'{rand_a=}')
logger.info(f'{rand_b=}')
logger.info(f'{rand_a == rand_b=}')

# let's make random but reproducible tensors
RANDOM_SEED = 1321

torch.manual_seed(RANDOM_SEED)  # seed is used only for one randomisation function
rand_c = torch.rand(3, 4)

torch.manual_seed(RANDOM_SEED)  # seed is used only for one randomisation function
rand_d = torch.rand(3, 4)

logger.info(f'{rand_c=}')
logger.info(f'{rand_d=}')
logger.info(f'{rand_c == rand_d=}')

# ========== GPU =================

# check if pytorch sees GPU
logger.info(f'{torch.cuda.is_available()=}')

# device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'

tensor = torch.tensor([1, 2, 3], device=device)
logger.info(f'{tensor=}')
logger.info(f'{tensor.device=}')

# numpy works only with CPU. Need to move tensor back to CPU
# rule of thumb - to make code hardware-agnostic, always move tensor to CPU

# --------------------------- permutation - changing order
org_t = torch.rand((225, 225, 3))  # height, width, colour_c
permutated = org_t.permute(2, 0, 1)  # colour_c, height, width
