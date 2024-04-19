import torch
from torch import nn


# Task:
# Reinforcement learning using CNN
# Agent which controls Mario in level 1 - 1
# Using 4 consecutive frames
# +1 reward for every movement to the right
# -1 reward for standing still
# -15 reward for dying

# Agent gives env action
# episode - one attempt at the level (dies, wins)
# policy: function mapping state to action
# value function: takes state and returns valueability
# action-value function:

# epsilon greedy approach: a strategy to choose actions. Explore-exploit dilemma.
# reply buffer. A storage for action-reward.
#   Takes in (current_State, Action_taken, Reward_received, Next_State, episode_Done)
# Action-value function. Bellman equation.
# Q(s, a) = r + y * max(s', a')
# y (gama) - discount factor


class DDQN(nn.Module):

	def __init__(self):
		super(DDQN, self).__init__()

	def forward(self, x):
		pass

# /imagine asdfasdf
