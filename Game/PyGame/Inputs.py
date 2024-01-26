import pygame


class Inputs:

	def __init__(self):
		self.left_mouse_down = False
		self.n_pressed = False
		self.k_pressed = False
		self.m_pressed = False

	def check_inputs(self):
		keys = pygame.key.get_pressed()

		if keys[pygame.K_n]:
			self.n_pressed = True
		else:
			self.n_pressed = False

		if keys[pygame.K_k]:
			self.k_pressed = True
		else:
			self.k_pressed = False

		if keys[pygame.K_m]:
			self.m_pressed = True
		else:
			self.m_pressed = False

		if keys[pygame.MOUSEBUTTONDOWN]:
			self.left_mouse_down = True
		else:
			self.left_mouse_down = False
