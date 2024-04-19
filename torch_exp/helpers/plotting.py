from matplotlib import pyplot as plt
import torch

# plt.scatter(x=x, y=y, c=c, cmap=plt.cm.RdYlBu)

# todo consider interactive mode? any benefits?
# todo how to draw on top of existing plot?
# todo implicit vs explicit declaration

plt.ion()


def plot(data: torch.Tensor, labels, title: str, block: bool = False):
	plt.clf()  # clear and redraw new plot
	plt.style.use('dark_background')
	plt.figure(figsize=(10, 7))
	plt.title(title, fontsize=12)
	plt.legend(prop={'size': 10})
	plt.xlabel('x label', fontsize=12)
	plt.ylabel('y label', fontsize=12)
	# plt.text()
	plt.scatter(data, labels, c='blue', s=4, label='Training data')
	plt.show(block=block)
