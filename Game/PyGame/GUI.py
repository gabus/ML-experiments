import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox


class GUI:

	def __init__(self, screen, scree_width, screen_height):
		self.screen = screen
		self.screen_width = scree_width
		self.screen_height = screen_height

	def fps_slider(self):
		return Slider(
			self.screen,
			self.screen_width - 600,
			self.screen_height - 150,
			500,
			20,
			min=5,
			max=500,
			step=1,
			initial=144
		)

	def fps_slider_text(self):
		return TextBox(self.screen, self.screen_width - 800, self.screen_height - 170, 130, 50, fontSize=30)

	def food_count_slider(self):
		return Slider(
			self.screen,
			self.screen_width - 600,
			self.screen_height - 200,
			500,
			20,
			min=10,
			max=300,
			step=1,
			initial=100
		)

	def food_count_text(self):
		return TextBox(self.screen, self.screen_width - 800, self.screen_height - 230, 130, 50, fontSize=30)

	def players_count_slider(self):
		return Slider(
			self.screen,
			self.screen_width - 600,
			self.screen_height - 100,
			500,
			20,
			min=5,
			max=99,
			step=1,
			initial=40
		)

	def players_count_text(self):
		return TextBox(self.screen, self.screen_width - 800, self.screen_height - 110, 130, 50, fontSize=30)

	def keyboard_map_manual(self):
		manual_text_1 = TextBox(
			self.screen,
			self.screen_width - 300,
			self.screen_height - 300,
			130,
			30,
			fontSize=20,
			borderThickness=0
		)
		manual_text_1.setText("json dump NN: n")

		manual_text_2 = TextBox(
			self.screen,
			self.screen_width - 300,
			self.screen_height - 270,
			130,
			30,
			fontSize=20,
			borderThickness=0
		)
		manual_text_2.setText("delete agent: k")

		manual_text_3 = TextBox(
			self.screen,
			self.screen_width - 300,
			self.screen_height - 240,
			130,
			30,
			fontSize=20,
			borderThickness=0
		)
		manual_text_3.setText("duplicate: m")

	def display_debug(self, text, line_position: int):
		line_margin = 30
		font = pygame.freetype.SysFont(pygame.font.get_default_font(), 20)
		font.render_to(self.screen, (10, line_margin * line_position), text)
