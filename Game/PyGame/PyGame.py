import json
import random

import numpy as np
import pygame
import pygame_widgets
from loguru import logger

from NN.Network import Network
from PyGame.GUI import GUI
from PyGame.Inputs import Inputs
from PyGame.Player import Player

SCREEN_WIDTH = 3000
SCREEN_HEIGHT = 1600


# todo write a function to load NN into a model
# smart nn
#  {'network_structure': [7, 20, 5], 'layers': [{'name': 'input', 'nodes': [{'input_weights': [], 'bias': -0.15583249319393913, 'node_value': 3.286}, {'input_weights': [], 'bias': -0.9050432081823819, 'node_value': 0.2}, {'input_weights': [], 'bias': -0.10224050252296374, 'node_value': 1}, {'input_weights': [], 'bias': -0.6185474561460247, 'node_value': 0}, {'input_weights': [], 'bias': 0.9672188210130018, 'node_value': 2}, {'input_weights': [], 'bias': 0.7297372925350829, 'node_value': 2}, {'input_weights': [], 'bias': -0.5166614715594635, 'node_value': 0}]}, {'name': 'hidden_1', 'nodes': [{'input_weights': [-0.03994701575014009, 0.06998455197034925, 0.7119713898485545, 0.7235594661141198, -0.42553930755809655, -0.40228711463691913, -0.34018752157791954], 'bias': 0.2665993251380303, 'node_value': -0.7860273507741912}, {'input_weights': [-0.890644138680517, 0.6600095708971923, -0.3071869398644831, -0.5104352779164094, 0.811051028310213, 0.7883207150109783, 0.2885378698213869], 'bias': -0.8633608273971277, 'node_value': 0.09659965463160886}, {'input_weights': [-0.20299059567898173, 0.7126940493946898, -0.4999686162766064, -0.4735957095371489, -0.6015984293196208, -0.4065840091943844, -0.0665435622886058], 'bias': -0.25511567240949484, 'node_value': -0.995441575387023}, {'input_weights': [-0.8439722348628306, 0.9366568583984003, 0.7133614214471957, -0.02286196294877635, -0.300016566913508, 0.6519968580250235, 0.15943793025083763], 'bias': -0.6039393059335479, 'node_value': -0.823835502291189}, {'input_weights': [0.09267753928174183, -0.030920769484204913, -0.48121742069626927, -0.15015111785551105, -0.0019128754435271955, -0.8618749217169391, 1.0651857014800903], 'bias': 0.6195838755397124, 'node_value': -0.9571222566936811}, {'input_weights': [-0.5306871683813166, 0.02340503228924229, -0.8777135817465106, 0.7217303977169267, 0.2987291968821117, 1.0609055118565476, -0.2934808023065554], 'bias': -0.9556634338848712, 'node_value': 0.10204240022531097}, {'input_weights': [-0.743134068124216, 0.5967105603919898, 0.004261852167969538, 0.36062538030007585, -0.7595408762179332, 0.2617722158619228, 0.38380329924196044], 'bias': -1.1603887912108388, 'node_value': -0.9973572068202107}, {'input_weights': [-0.6661915563377598, 0.489634994161283, 0.6355009998402266, 0.9698301446096484, -0.7765052895546729, 0.856973282829655, 0.12285040840463374], 'bias': -0.20318621650800986, 'node_value': -0.8603632892651168}, {'input_weights': [0.6914201165249787, 0.08196672838497188, 0.8529118566830065, -0.7141414464775473, -0.15830551804805515, -0.6999045005332933, 0.9864843095132596], 'bias': 0.012115972452279383, 'node_value': 0.890614964152183}, {'input_weights': [-0.2030436318550675, 0.03816530532145146, -0.041060953950943946, 0.9455963013439577, -0.692234353464809, 0.6393512169279196, 0.7071985021568221], 'bias': 0.6599308742292569, 'node_value': -0.6675970614377563}, {'input_weights': [0.08152804937209313, 0.8990344331686768, 0.8787814805518928, 0.07412711948011075, -0.3016296484665488, -0.4823473929149851, 0.3404492172207292], 'bias': 0.4945685347378982, 'node_value': -0.2368785958360942}, {'input_weights': [-0.07633420237112631, 0.8504115085812196, -0.5732661902085363, 0.36418136303010185, 0.024301555930149155, 0.44038283649229326, 0.011852944500945206], 'bias': 0.1967492280679847, 'node_value': 0.2685966185428828}, {'input_weights': [1.0382073235963445, 0.15387913726233393, 1.0332432099956559, 0.395818916775986, 0.952565139042107, -0.5055521151052902, 0.5522598217549217], 'bias': 0.9835638211284313, 'node_value': 0.9999566439021468}, {'input_weights': [0.6961900659490974, 0.3835121513995787, 0.18002155640311984, 0.5304724566847283, -0.5889420809592933, 1.0383383435680644, 1.0573620884983284], 'bias': -0.10136034727902077, 'node_value': 0.9979589033174985}, {'input_weights': [0.9569984651392562, 0.08742920821642297, 0.20525336236933894, 0.7580925058959056, 0.14226503343313207, 1.001232790329042, -0.8938900595015281], 'bias': -0.6375375015526259, 'node_value': 0.9999754728207726}, {'input_weights': [-0.20021867891100456, 0.10294169926429664, -0.060663348959271, 0.6020141721795784, -0.08215705580694499, 0.4724385962351504, -0.23125349853841506], 'bias': -0.5186752349880626, 'node_value': 0.08238235790619129}, {'input_weights': [0.9808437875168985, 0.07270047794424528, 0.5782579897787956, 0.3130949137071846, 0.346298365614214, -0.8400514727334022, -0.7581896049783756], 'bias': -0.14995058311497514, 'node_value': 0.9930361995320371}, {'input_weights': [1.0165581273941122, 0.6223212584265101, -0.7965946552358276, 0.6325334870515751, -0.30393798088880786, 1.0247271214220992, -0.738316982925209], 'bias': 0.9614044662713448, 'node_value': 0.9994615618063881}, {'input_weights': [0.9235926320861358, 0.2514290086888623, 0.6347503532950527, -0.8767995767017497, 0.1519702730340925, 0.978983324328353, 0.26354240884199986], 'bias': 0.4034227133619016, 'node_value': 0.999987257869188}, {'input_weights': [-0.2712166452651683, -0.5254528579298572, 1.0222708838070649, -0.07266042704069309, 1.0304126119780979, -0.583226344748045, 0.06290051150659527], 'bias': 0.5647233638113924, 'node_value': 0.7260558322686838}]}, {'name': 'output', 'nodes': [{'input_weights': [0.28288059499805684, 0.46039063798930946, 0.1386487278301582, 0.2858593226243767, -0.08756807688550705, 0.5218982640493414, 1.0034247862122943, -0.029067850387791527, -0.29322577916014314, 0.06972563690377111, -0.8467928060132791, -0.3496219578571922, -0.8679226906478595, -0.01963556681820028, -0.7160409989298488, -0.04177486063254171, -0.22078559227372865, -0.4137090946329587, 1.0526711823414965, -0.6631927751928248], 'bias': 0.8248215681221512, 'node_value': -0.9970549305171298}, {'input_weights': [0.17472164563987458, 0.2586486680929022, -0.20981178362272737, 0.32896582001883873, 0.3364249792821643, 0.4725538346729624, -0.7114035187369325, -0.7104674059206046, -0.4111751429946533, -0.5972143088575774, 0.9984341066597402, -0.7083354931621424, 0.4701728448018898, -0.6515177521171023, -0.38137949422295825, -0.7875303693317273, 0.025081361256984236, -0.892016388592671, -0.8300865545010434, -0.7916852502883476], 'bias': -0.4672708464367722, 'node_value': -0.9842964508752577}, {'input_weights': [1.005494825737884, 0.435052601461027, 0.2370101230975846, 0.6865076314004421, -0.7969442017856616, -0.20171318427470866, 0.11120881663602036, 0.6765110302406612, -0.19654183841083486, 0.37710350781102386, -0.12245945269008074, -0.41147344611418357, 0.6792592413369452, -0.0383716188042485, -0.4132266127916828, 0.8351230415461843, 0.05337554556715583, 0.8981642233711196, 0.1713653945786704, -0.8132941465559809], 'bias': 0.5051909746476883, 'node_value': -0.8277126893983454}, {'input_weights': [0.33527542427577295, 0.6410517301284233, 0.8951967155890388, 0.22483823609290973, -0.7905110448537278, 0.37809810061477234, 0.494443207810078, 0.6755553362439897, 0.4162424343809066, 0.976153036626658, 0.8883496537307491, -0.05571521886268399, 0.8687972453714319, 0.15324485273281663, 0.07665950942600985, 0.9028410887413612, -0.9038872052258121, -0.10104125890089916, -0.6645617143352276, -0.2183471149073635], 'bias': 0.730651555577924, 'node_value': -0.9912270471200367}, {'input_weights': [-0.24696728797972992, 0.9580697242468434, 0.0846688520012584, -0.6422140738318833, 0.7611294730324583, 0.7372890745285041, 0.9306424997796079, -0.10445425419990878, -0.836477085041886, -0.36757812926993516, -0.8502652728073491, -0.5240504671379842, 0.9106452841110091, -0.48992187180733615, -0.8054135016145447, -0.8479649575900435, 0.1372638395593639, 0.6622541459325244, 0.4512095267863825, 0.6266673629081181], 'bias': -0.25061843723587673, 'node_value': 0.05175235304738101}]}]}

class PyGame:

	def __init__(self):
		logger.info("---")
		pygame.init()
		self.running: bool = True
		self.clock = pygame.time.Clock()
		self.dt: float = 0.0001

		self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

		# ------------ GUI ------------
		self.gui = GUI(self.screen, SCREEN_WIDTH, SCREEN_HEIGHT)

		self.fps_slider = self.gui.fps_slider()
		self.fps_text = self.gui.fps_slider_text()

		self.food_count_slider = self.gui.food_count_slider()
		self.food_count_text = self.gui.food_count_text()

		self.players_count_slider = self.gui.players_count_slider()
		self.players_count_text = self.gui.players_count_text()

		# self.fps_text = self.gui.fps_text()

		self.gui.keyboard_map_manual()
		# ------------ End of GUI ------------

		self.inputs = Inputs()

		# self.network_structure = [7, 20, 5]  todo this so far yielded best results
		self.network_structure: list = [7, 25, 5]

		self.players_count = self.players_count_slider.getValue()
		self.players: list = []
		self.dead_players: list = []
		self.surviving_record: int = 0

		self.food_count: float = self.food_count_slider.getValue()
		self.food: list = []
		self.eating_food_reward: int = 1000
		self.eating_other_player_reward: int = 5000

		self.add_players(self.players_count)

		[p['nn_controller'].set_input_layer_meta([
			# 'speed',
			# 'size',
			'ener',
			'ener_d',
			'dash',
			# 'n_left',
			# 'n_right',
			# 'n_top',
			# 'n_bottom',
			'p_left',
			'p_right',
			'p_top',
			'p_bottom',
		]) for p in self.players]

	def add_players(self, player_count):
		for i in range(player_count):
			player_obj = Player(SCREEN_WIDTH, SCREEN_HEIGHT)
			self.players.append({
				'obj': player_obj,
				'draw': self.draw_player(player_obj),
				'sensing_shield': self.draw_sensing_zone(player_obj),
				'nn_controller': Network(self.network_structure),
			})

	def main_loop(self):

		while self.running:

			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.running = False
					return

			self.inputs.check_inputs()

			self.screen.fill("#799997")
			# self.draw_dead_players()

			if self.food_count > len(self.food):
				self.spawn_food()

			for food in self.food:
				pygame.draw.rect(self.screen, food['obj'].colour, food['draw'])

			self.draw_and_run_physics()

			if len(self.players) < self.players_count:
				self.spawn_fittest_player()

			self.mouse_debugger()
			# self.gui.display_debug("Players alive: {} Players dead: {}".format(len(self.players), len(self.dead_players)), 1)
			# self.debug_longest_survival()
			# self.debug_nn(self.players[0]['nn_controller'])
			self.display_players_count_slider()

			# pygame.display.flip()
			pygame_widgets.update(pygame.event.get())
			pygame.display.update()

			self.dt = self.clock.tick(self.fps_slider.getValue()) / 1000

		# todo remove after profiling
		# self.running = False

		pygame.quit()

	# todo save historical model to a file?
	#    fitest agent might be lost due mutation
	def spawn_fittest_player(self, player=None):
		chance_to_spawn_new = 0.4
		fittest = None

		if player:
			fittest = player
			chance_to_spawn_new = 0
		else:
			last_longest_lived = -1
			for player in self.players:
				if player['obj'].alive_cycles > last_longest_lived:
					last_longest_lived = player['obj'].alive_cycles
					fittest = player

		mutation_rate = 0.1
		mutation_chance = 0.5  # %

		player_copy = Player(SCREEN_WIDTH, SCREEN_HEIGHT)

		should_be_new = random.uniform(0, 1)
		if should_be_new < chance_to_spawn_new:
			nn_controller = Network(self.network_structure)
		else:
			# todo this directly affects already existing agent
			# fittest['obj'].add_to_size(random.randrange(-2, 2))
			# player_copy.size = fittest['obj'].size

			player_copy.add_to_energy(random.randrange(-30, 30))
			player_copy.colour = fittest['obj'].colour
			player_copy.sensing_distance = fittest['obj'].sensing_distance
			nn_controller = fittest['nn_controller'].copy_network_with_mutation(mutation_rate, mutation_chance)

		self.players.append({
			'obj': player_copy,
			'draw': self.draw_player(player_copy),
			'sensing_shield': self.draw_sensing_zone(player_copy),
			'nn_controller': nn_controller,
		})

	def draw_dead_players(self):
		for p in self.dead_players:
			player_draw = pygame.Rect(p['obj'].x, p['obj'].y, p['obj'].size, p['obj'].size)
			pygame.draw.rect(self.screen, "#ABA9BF", player_draw)

	def draw_and_run_physics(self):
		for player in self.players:

			player_obj = player['obj']  # type: Player

			if player_obj.energy <= 0:
				player_obj.alive = False
				self.dead_players.append(id(1))
				self.players.remove(player)
				continue

			# pass in all input player states:
			player['nn_controller'].set_input_values(
				# player_obj.speed,
				# player_obj.size,
				player_obj.energy / 1000,
				player_obj.energy_dissipation / 5,
				player_obj.dash,
				# 1 if player_obj.negative_feedback['left'] else 0,
				# 1 if player_obj.negative_feedback['right'] else 0,
				# 1 if player_obj.negative_feedback['top'] else 0,
				# 1 if player_obj.negative_feedback['bottom'] else 0,
				2 if player_obj.positive_feedback['left'] else 0,
				2 if player_obj.positive_feedback['right'] else 0,
				2 if player_obj.positive_feedback['top'] else 0,
				2 if player_obj.positive_feedback['bottom'] else 0,
			)

			inputs = player['nn_controller'].run_network()
			player_obj.nn_control(*inputs)

			# drawing player
			player['draw'] = self.draw_player(player_obj)

			# drawing player shield
			player['sensing_shield'] = self.draw_sensing_zone(player_obj)

			player_obj.reset_feedbacks()

			# todo can be done on multithreading
			# Player vs Food

			self.check_for_food(player)

			# Player VS Player
			# self.check_player_vs_player(player)

			# todo can be done on multithreading
			# punish player for leaving the area
			self.check_leaving_boundaries(player_obj)

			player_obj.dissipate_energy()

	# mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()
	# if mouse_pos_x > 0 and mouse_pos_y > 0 and not self.inputs.left_mouse_down:
	# 	time.sleep(1)

	def check_player_vs_player(self, player):
		for other in self.players:
			if id(other) == id(player):
				continue

			if player['draw'].colliderect(other['draw']):
				if player['obj'].size <= other['obj'].size:
					player['obj'].alive = False
					self.dead_players.append(id(player))
					self.players.remove(player)
					return
				else:
					# you just absorbed another player and gained its energy
					player['obj'].gain_energy(self.eating_other_player_reward)

			if player['sensing_shield'].colliderect(other['sensing_shield']):
				if player['obj'].size <= other['obj'].size:

					if other['obj'].x > player['obj'].x:
						player['obj'].negative_feedback['right'] = True

					if other['obj'].x < player['obj'].x:
						player['obj'].negative_feedback['left'] = True

					if other['obj'].y > player['obj'].y:
						player['obj'].negative_feedback['bottom'] = True

					if other['obj'].y < player['obj'].y:
						player['obj'].negative_feedback['top'] = True

				else:

					if other['obj'].x > player['obj'].x:
						player['obj'].positive_feedback['right'] = True

					if other['obj'].x < player['obj'].x:
						player['obj'].positive_feedback['left'] = True

					if other['obj'].y > player['obj'].y:
						player['obj'].positive_feedback['bottom'] = True

					if other['obj'].y < player['obj'].y:
						player['obj'].positive_feedback['top'] = True

	def check_leaving_boundaries(self, player_obj):
		if player_obj.x < 0 or player_obj.x > SCREEN_WIDTH or player_obj.y < 0 or player_obj.y > SCREEN_HEIGHT:
			player_obj.set_energy_dissipation('out-of-bounds')

			if player_obj.x < 0:
				player_obj.negative_feedback['left'] = True

			if player_obj.x > SCREEN_WIDTH:
				player_obj.negative_feedback['right'] = True

			if player_obj.y < 0:
				player_obj.negative_feedback['top'] = True

			if player_obj.y > SCREEN_HEIGHT:
				player_obj.negative_feedback['bottom'] = True

	def check_for_food(self, player):
		for food in self.food:
			if player['draw'].colliderect(food['draw']):
				player['obj'].gain_energy(self.eating_food_reward)
				self.food.remove(food)
				return

			if player['sensing_shield'].colliderect(food['draw']):
				if food['obj'].x > player['obj'].x:
					player['obj'].positive_feedback['right'] = True

				if food['obj'].x < player['obj'].x:
					player['obj'].positive_feedback['left'] = True

				if food['obj'].y > player['obj'].y:
					player['obj'].positive_feedback['bottom'] = True

				if food['obj'].y < player['obj'].y:
					player['obj'].positive_feedback['top'] = True

	def spawn_food(self):
		food = Player(SCREEN_WIDTH, SCREEN_HEIGHT)
		food.colour = 'green'
		food.size = 10
		draw = pygame.Rect(food.x, food.y, food.size, food.size)
		self.food.append({'obj': food, 'draw': draw})

	def mouse_debugger(self):
		mouse_pos_x, mouse_pos_y = pygame.mouse.get_pos()

		# optimisation
		if mouse_pos_x <= 0 or mouse_pos_x >= SCREEN_WIDTH - 1 or mouse_pos_y <= 0 or mouse_pos_y >= SCREEN_HEIGHT - 1:
			return

		mouse_square = pygame.Rect(mouse_pos_x - 15, mouse_pos_y - 15, 30, 30)
		pygame.draw.rect(self.screen, 'yellow', mouse_square, 2)

		for player in self.players:
			if mouse_square.colliderect(player['draw']):
				if self.inputs.n_pressed:
					logger.error(player['nn_controller'].as_dict())

				if self.inputs.k_pressed:
					self.players.remove(player)
					continue

				if self.inputs.m_pressed:
					self.spawn_fittest_player(player)
					continue

				self.gui.display_debug(json.dumps(player['obj'].as_dict()), 6)

	def draw_player(self, player_obj):
		player_draw = pygame.Rect(player_obj.x, player_obj.y, player_obj.size, player_obj.size)
		pygame.draw.rect(self.screen, player_obj.colour, player_draw)
		return player_draw

	def draw_sensing_zone(self, player_obj):
		player_draw_shield = pygame.Rect(
			player_obj.x - (player_obj.sensing_distance - player_obj.size) / 2,
			player_obj.y - (player_obj.sensing_distance - player_obj.size) / 2,
			player_obj.sensing_distance,
			player_obj.sensing_distance,
		)
		pygame.draw.rect(self.screen, player_obj.colour, player_draw_shield, 2)
		return player_draw_shield

	def debug_longest_survival(self):
		longest_time = -1
		survival = None

		for player in self.players:
			if player['obj'].alive_cycles > longest_time:
				longest_time = player['obj'].alive_cycles
				survival = player

		player_draw = pygame.Rect(
			survival['obj'].x + survival['obj'].size / 2 - 45,
			survival['obj'].y + survival['obj'].size / 2 - 45,
			90,
			90
		)
		pygame.draw.rect(self.screen, 'red', player_draw, 3)

		if self.surviving_record < longest_time:
			self.surviving_record = longest_time

		self.gui.display_debug("Longest survival: {} Energy: {} dis: {} x{}:y{}".format(
			longest_time,
			survival['obj'].energy,
			survival['obj'].energy_dissipation,
			survival['obj'].x,
			survival['obj'].y,
		), 3)
		self.gui.display_debug("Survival: {}".format(survival['obj'].as_dict()), 4)
		self.gui.display_debug("Survival record: {}".format(self.surviving_record), 5)

	def debug_nn(self, nn: Network):
		smallfont = pygame.font.SysFont('Corbel', 20)
		layer_height = 200
		for layer in nn.layers:
			layer_height += 50
			node_horizontal_position = 0
			for node in layer.nodes:
				node_horizontal_position += 90
				colour = node.node_value * 255
				# logger.warning(colour)

				colour = np.clip(colour, 0, 255)

				pygame.draw.rect(
					self.screen,
					(colour, colour, colour),
					[node_horizontal_position, layer_height, 80, 40]
				)
				meta_text = smallfont.render(node.meta_data, True, 'red')
				text = smallfont.render(str(round(node.node_value, 2)), True, 'red')
				self.screen.blit(meta_text, (node_horizontal_position + 5, layer_height))
				self.screen.blit(text, (node_horizontal_position + 5, layer_height + 15))

	def display_players_count_slider(self):
		self.players_count = int(self.players_count_slider.getValue())
		self.food_count = int(self.food_count_slider.getValue())
		self.food_count_text.setText("Bananas: {}".format(self.food_count_slider.getValue()))
		self.players_count_text.setText("Minions: {}".format(self.players_count_slider.getValue()))
		self.fps_text.setText("FPS: {}".format(int(1 / self.dt)))
