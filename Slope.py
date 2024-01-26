import random


class Slope:

	def __init__(self, problem_formula):
		self.problem_formula = problem_formula
		self.solve_x = random.randint(-3, 3)
		self.h = 0.0000001

		self.all_x = []
		self.all_y = []

		# steepness and direction
		self.slope = 0

	def get_next_solve_x(self):
		solve_y = self.problem_formula(self.solve_x)
		self.all_x.append(self.solve_x)
		self.all_y.append(solve_y)

		solve_h_y = self.problem_formula(self.solve_x + self.h)

		slope_direction = solve_h_y - solve_y
		move = slope_direction * 3000000

		debug = {
			'solve_x': self.solve_x,
			'solve_y': solve_y,
			'self.solve_x + self.h': self.solve_x + self.h,
			'solve_h_y': solve_h_y,
			'slope_direction': slope_direction,
			'move': move,
		}

		print(debug)

		self.solve_x = self.solve_x + move * -1

	def find_next_point(self):
		# print((self.solve_x))
		self.find_slope(self.solve_x)
		return self.get_line(self.solve_x)

	# approximate slope of the function objective at x
	def find_slope(self, x: float):
		y = self.problem_formula(x)
		self.all_x.append(x)
		self.all_y.append(y)

		# positive - going up, negative - going down
		slope_direction = self.problem_formula(x + self.h) - y
		# print(slope_direction)

		# If the slope is very steep, make bigger intervals for the next point check
		# When slope evens out, that means you're close the minimum, so take more samples
		slope_steepness = slope_direction * -1 / self.h

		self.solve_x = self.solve_x + slope_steepness

	def get_line(self, x):
		line_length = 0.001

		return {
			'x': [
				(x + self.h) * line_length,
				x,
				(x + self.h) * -line_length,
			], 'y': [
				self.problem_formula(x + self.h) * line_length,
				self.problem_formula(x),
				self.problem_formula(x + self.h) * -line_length,
			]
		}
