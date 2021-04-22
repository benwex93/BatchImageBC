import gym
import numpy as np

import glfw
import gym
from types import MethodType
class PickPlaceWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		self.env_timesteps = 1
		# self.env._max_episode_steps = 100

		self.distance_threshold = 0.08

		self.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
		self.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
		#zoom in
		self.env.env._get_viewer('rgb_array').move_camera(5,0,0.22)
		#rotate
		self.env.env._get_viewer('rgb_array').move_camera(1,0.19,0)

	def step(self, action):

		next_state, reward, done, info = self.env.step(action)

		info = True
		# modify ...
		achieved_goal = next_state['achieved_goal']
		goal = next_state['desired_goal']
		assert achieved_goal.shape == goal.shape
		# Compute distance between goal and the achieved goal.
		d = np.linalg.norm(achieved_goal - goal, axis=-1)

		# print('env reward: ', reward)
		# print('caclutated distance: ', d)
		# if self.sparse:
		reward = -(d > self.distance_threshold).astype(np.float32)
		# print('caclutated reward: ', reward)
		# else:
		# 	reward = -d


		#makes sure not to start in done state and if does then exits so as to not bias
		if reward == 0 and self.env_timesteps == 1:
			reward = -1.0
			done = True
			info = False
		#+50 for no negatives
		elif self.env_timesteps < 100:
			done = False
		else:
			done = True
		self.env_timesteps+=1

		return next_state, reward + 1, done, info
	def close(self):
		if self.viewer is not None:
			glfw.destroy_window(self.viewer.window)
			self.viewer = None
	def reset(self):
		self.env_timesteps = 1

		return self.env.reset()

class ReachWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.env = env
		self.env_timesteps = 1

		self.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
		self.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
		#zoom in
		self.env.env._get_viewer('rgb_array').move_camera(5,0,0.22)
		#rotate
		self.env.env._get_viewer('rgb_array').move_camera(1,0.19,0)

	def step(self, action):

		next_state, reward, done, info = self.env.step(action)

		# modify ...
		achieved_goal = next_state['achieved_goal']
		goal = next_state['desired_goal']
		assert achieved_goal.shape == goal.shape
		# Compute distance between goal and the achieved goal.
		d = np.linalg.norm(achieved_goal - goal, axis=-1)

		reward = -(d > self.distance_threshold).astype(np.float32)

		return next_state, reward + 1, done, info
	def close(self):
		if self.viewer is not None:
			glfw.destroy_window(self.viewer.window)
			self.viewer = None

from gym.utils import seeding
from mujoco_py.generated import const
def init_env(env_name, gen_exp=False):
	env = gym.make(env_name)
	# import pdb
	# pdb.set_trace()
	env.seed(4)
	env.np_random, seed = seeding.np_random(4)
	env.action_space.seed(4)


	if env_name == 'FetchPickAndPlace-v1':

		env = PickPlaceWrapper(env)
		if gen_exp:
			from gym.wrappers import FilterObservation, FlattenObservation
			env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
		env._max_episode_steps = 100

	if env_name == 'FetchReach-v1':

		env = ReachWrapper(env)
		if gen_exp:
			from gym.wrappers import FilterObservation, FlattenObservation
			env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))

	return env