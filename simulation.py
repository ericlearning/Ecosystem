import os, cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random, time
from multiprocessing import Pool

cv2.namedWindow('Rendered')

world_x, world_y, disp_mult = 50, 50, 20
world_prey = np.zeros((world_y, world_x)).astype('uint8')
world_predator = np.zeros((world_y, world_x)).astype('uint8')

class Prey():
	global world_prey, world_x, world_y

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.hunger = 0
		self.movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

		self.daily_hunger = 1
		self.birth_min_hunger = random.randint(30, 40)
		self.birth_hunger_set = 0


	def restrict(self, x, y):
		new_x = min(max(x, 0), world_x - 1)
		new_y = min(max(y, 0), world_y - 1)
		return new_x, new_y

	def random_move(self):
		move_direction = random.randint(0, len(self.movements)-1)
		self.x += self.movements[move_direction][0]
		self.y += self.movements[move_direction][1]
		self.x, self.y = self.restrict(self.x, self.y)

	def interaction(self):
		self.hunger += self.daily_hunger

		if(self.hunger < 0):
			return (0, self.x, self.y)

		if(self.hunger >= self.birth_min_hunger):
			self.hunger = self.birth_hunger_set
			return (1, self.x, self.y)

		return (-1, self.x, self.y)

class Predator():
	global world_prey, world_predator, world_x, world_y

	def __init__(self, x, y):
		self.x = x
		self.y = y
		self.hunger = random.randint(30, 40)
		self.movements = [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]

		self.hunt_min_num = 1
		self.daily_hunger = -1

	def restrict(self, x, y):
		new_x = min(max(x, 0), world_x - 1)
		new_y = min(max(y, 0), world_y - 1)
		return new_x, new_y

	def random_move(self):
		move_direction = random.randint(0, len(self.movements)-1)
		self.x += self.movements[move_direction][0]
		self.y += self.movements[move_direction][1]
		self.x, self.y = self.restrict(self.x, self.y)

	def interaction(self):
		self.hunger += self.daily_hunger

		if(self.hunger < 0):
			return (0, self.x, self.y)

		if(world_prey[self.y, self.x] >= self.hunt_min_num):
			return (1, self.x, self.y)

		return (-1, self.x, self.y)

def draw_on(target, l, population_control):
	pos = []
	new_l = []
	for i in range(len(l)):
		draw_x, draw_y = l[i].x, l[i].y
		if(target[draw_y, draw_x] >= population_control):
			continue
		new_l.append(l[i])
		target[draw_y, draw_x] += 1

	return target, new_l

def random_moves(l):
	for i in range(len(l)):
		l[i].random_move()

def predator_interactions(inputs):
	predator, act_predator = inputs
	(act, x, y) = act_predator
	outputs = []
	eat_outputs = []
	if(act == 0):
		return ([], [])
	elif(act == 1):
		return ([predator], [(predator.x, predator.y)])
	else:
		return ([predator], [None])

def prey_interactions(inputs, is_killed):
	global preys
	prey, act_prey = inputs
	(act, x, y) = act_prey
	outputs = []
	eat_outputs = []
	hunger = prey.hunger

	if(is_killed):
		return ([], hunger)
	else:
		if(act == 0):
			return ([], None)
		elif(act == 1):
			return ([prey, Prey(prey.x, prey.y)], None)
		else:
			return ([prey], None)

def is_killed(eat_pos, prey):
	is_killed = 0
	for i, e in enumerate(eat_pos):
		if(e is not None):
			if(e[0] == prey.x and e[1] == prey.y):
				is_killed = 1
				return is_killed, i
	return is_killed, None

def interactions(preys, predators):
	acts_predator = []
	acts_prey = []

	random.shuffle(preys)
	random.shuffle(predators)

	for i in range(len(predators)):
		act = predators[i].interaction()
		acts_predator.append(act)
	for i in range(len(preys)):
		act = preys[i].interaction()
		acts_prey.append(act)


	new_predators = []
	new_preys = []
	eat_pos = []
	for i in zip(predators, acts_predator):
		p, e = predator_interactions(i)
		new_predators.extend(p)
		eat_pos.extend(e)

	for i in zip(preys, acts_prey):
		c, idx = is_killed(eat_pos, i[0])
		p, prey_hunger = prey_interactions(i, c)
		new_preys.extend(p)

		if(c):
			eat_pos[idx] = None
			who_caught = new_predators[idx]
			who_caught.hunger += prey_hunger
			child = Predator(who_caught.x, who_caught.y)
			new_predators.append(child)
		
	return new_preys, new_predators

def render(world_prey, world_predator):
	rendered = np.zeros((world_x, world_y, 3)).astype('uint8')
	for y in range(world_y):
		for x in range(world_x):
			if(world_prey[y, x]>0):
				rendered[y, x] = [0, 255, 0]
			if(world_predator[y, x]>0):
				rendered[y, x] = [0, 0, 255]
	return rendered

init_predator_num = 300
predators = []
for _ in range(init_predator_num):
	predator = Predator(random.randint(0, world_x - 1), random.randint(0, world_y - 1))
	predators.append(predator)

init_prey_num = 300
preys = []
for _ in range(init_prey_num):
	prey = Prey(random.randint(0, world_x - 1), random.randint(0, world_y - 1))
	preys.append(prey)

records_prey = []
records_predator = []

max_record_len = 50
fig = plt.gcf()

while True:
	print(len(preys), len(predators))
	world_prey = np.zeros((world_y, world_x)).astype('uint8')
	world_predator = np.zeros((world_y, world_x)).astype('uint8')	

	world_prey, new_preys = draw_on(world_prey, preys, 3)
	world_predator, new_predators = draw_on(world_predator, predators, 2)
	preys, predators = new_preys, new_predators

	records_prey.append(len(preys))
	records_predator.append(len(predators))
	if(len(records_prey)>max_record_len):
		records_prey.pop(0)
	if(len(records_predator)>max_record_len):
		records_predator.pop(0)
	plt.plot(records_prey, color = 'green')
	plt.plot(records_predator, color = 'red')

	fig.canvas.draw()
	plt.pause(0.0001)
	plt.cla()

	random_moves(preys)
	random_moves(predators)

	preys, predators = interactions(preys, predators)

	world_render = render(world_prey, world_predator)

	world_render = cv2.resize(world_render, (world_x * disp_mult, world_y * disp_mult), interpolation = cv2.INTER_NEAREST)

	cv2.imshow('Rendered', world_render)

	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break



