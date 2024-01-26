import matplotlib.pyplot as plt
from loguru import logger


class MatlibWrapper:

	def __init__(self):
		"""
		todo specs
			[] separate process (threading) (https://stackoverflow.com/a/19846691/4061254)
			[] non blocking graph
			[] near-real-time animation
			[] hold the graph after code finish (might be solved by default because of separate thread??)
			[] consistent cross platform behaviour
			[] ability to *add* data and set *entire dataset* for each frame
			[] smooth animation in case lots of data is loaded
			[] update graph only when data is updated
			[] support multiple graphs in one window
			[] support multiple windows (instances. should be by default because of multiprocessing)

		todo questions
			scatter vs plot?
			plt.show() vs plt.draw() ??
			separate thread cannot support interactive mode (Note that (for the case where you are working with an interactive backend) most GUI backends require being run from the main thread as well.)
			matlib does not support partial data update -- redraw all entire graph
			setx_data sety_data requires interactive mode, wihhc is not supported in multiprocessing
			[] does title, ylable, xlable or legends persist between .clf() ?

			print(plt.style.available)
			['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']
			plt.style.use('fivethirtyeight')

		"""
		self.plt = plt
		self.is_drawing = False
		self.fig, self.ax = plt.subplots(figsize=(10, 7))

	# self.fig.canvas.draw()
	# plt.show(block=False)

	def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
		# todo how to plot two graphs without clearing one another
		# self.ax.plot(*args, scalex=True, scaley=True, data=None, **kwargs)
		plt.clf()
		plt.plot(*args, scalex=True, scaley=True, data=None, **kwargs)

		plt.title('Loss curve')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend()
		# self.ax.relim()
		# self.ax.autoscale_view(True, True, True)
		# self.fig.canvas.draw()
		plt.pause(0.1)

	# if not self.is_drawing:
	# 	self.__draw()
	# 	self.is_drawing = False

	def __draw(self):
		logger.info("dwagin")
		self.is_drawing = True  # todo in multiprocessing this variable will not be accessible
# plt.scatter()
# plt.clf()  # todo might need to clear the plot to avoid memory leak or "plt.cla()"
# plt.draw()  # might not need it at all  https://stackoverflow.com/a/63702988/4061254
# plt.pause(0.1)
# self.fig.canvas.draw()  # draw and show it

# todo example how to make graph draw real time
# 	plt.clf()
# 	plt.plot(epoch_count, np.array(torch.tensor(train_loss_values).numpy()), label='Train loss')
# 	plt.plot(epoch_count, test_loss_values, label='Test loss')
# 	plt.title('Loss curve')
# 	plt.ylabel('Loss')
# 	plt.xlabel('Epoch')
# 	plt.legend()
# 	plt.pause(0.01)
