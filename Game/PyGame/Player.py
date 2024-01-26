import random


class Player:

	def __init__(self, screen_width, screen_height):
		self.colours = [
			'#f2f07e', '#a0db8e', '#ffa69e', '#c0ffee', '#ffa32d', '#fff4b9', '#741313', '#8fc1e3', '#EF9A9A',
			'#CE93D8', '#B39DDB', '#9FA8DA', '#90CAF9', '#80CBC4', '#A5D6A7', '#C5E1A5', '#E6EE9C', '#FFF59D',
			'#FFE082', '#FFCC80', '#FFAB91', '#BCAAA4'
		]

		self.x = random.randrange(50, screen_width - 50)
		self.y = random.randrange(50, screen_height - 50)
		self.size = random.randrange(30, 60)
		self.speed = 1  # todo inverse proportional to the size (0.1 - 2.0)
		self.dash = 1  # doubles the speed, but doubles energy consumption
		self.energy = 100 * self.size
		self.colour = self.colours[random.randrange(0, len(self.colours) - 1)]
		self.alive = True
		self.energy_dissipation = 1
		self.sensing_distance = random.randrange(self.size + 80, self.size + 200)
		self.negative_feedback = {
			'left': False,
			'right': False,
			'top': False,
			'bottom': False,
		}
		self.positive_feedback = {
			'left': False,
			'right': False,
			'top': False,
			'bottom': False,
		}

		self.alive_cycles = 0

	def add_to_size(self, amount):
		self.size += amount

		if self.size < 10:
			self.size = 10

		if self.size > 100:
			self.size = 100

	def add_to_energy(self, amount):
		self.energy += amount

		if self.energy < 1000:
			self.energy = 1000

		if self.energy > 5000:
			self.energy = 5000

	def reset_feedbacks(self):
		self.negative_feedback = {
			'left': False,
			'right': False,
			'top': False,
			'bottom': False,
		}
		self.positive_feedback = {
			'left': False,
			'right': False,
			'top': False,
			'bottom': False,
		}

	def up(self):
		self.y += self.speed * self.dash

	def down(self):
		self.y -= self.speed * self.dash

	def left(self):
		self.x -= self.speed * self.dash

	def right(self):
		self.x += self.speed * self.dash

	def die(self):
		self.alive = False

	def dissipate_energy(self):
		self.alive_cycles += 1
		self.energy -= self.energy_dissipation

	def set_energy_dissipation(self, case: str):
		if case == 'walking':
			self.energy_dissipation = 1

		if case == 'running':
			self.energy_dissipation = 2

		if case == 'stationary':
			self.energy_dissipation = 50

		if case == 'conflicting-inputs':
			self.energy_dissipation = 45

		if case == 'out-of-bounds':
			self.energy_dissipation = 60

	def roam_randomly(self):
		# 4th so there's a chance it just stays at one place
		direction = random.randrange(0, 30)

		if direction == 0:
			self.up()

		if direction == 1:
			self.down()

		if direction == 2:
			self.left()

		if direction == 3:
			self.right()

	def nn_control(self, *args):

		if args[0] > 0:
			self.dash = 2
		else:
			self.dash = 1

		if args[1] > 0:
			self.up()

		if args[2] > 0:
			self.down()

		if args[3] > 0:
			self.left()

		if args[4] > 0:
			self.right()

		dissipation_state = 'stationary'

		if args[1] > 0 or args[2] > 0 or args[3] > 0 or args[4] > 0:
			dissipation_state = 'walking'

		if args[0] > 0 and dissipation_state == 'walking':
			dissipation_state = 'running'

		if args[1] > 0 and args[2] > 0:
			dissipation_state = 'conflicting-inputs'

		if args[3] > 0 and args[4] > 0:
			dissipation_state = 'conflicting-inputs'

		self.set_energy_dissipation(dissipation_state)

	def as_dict(self):
		return {
			'x': self.x,
			'y': self.y,
			'speed': self.speed,
			'size': self.size,
			'energy': self.energy,
			'energy_dissipation': self.energy_dissipation,
			'alive': self.alive,
			'sensing_distance': self.sensing_distance,
			'positive_feedback': self.positive_feedback,
			'negative_feedback': self.negative_feedback,
			'alive_cycles': self.alive_cycles,
		}

	def gain_energy(self, amount: int):
		self.energy += amount
