from PyGame.PyGame import PyGame


class Game:

	def __init__(self):
		pg = PyGame()

		# with cProfile.Profile() as pr:
		pg.main_loop()

	# stats = pstats.Stats(pr)
	# stats.dump_stats(filename='profiling.prof')


Game()
