from envs import PickPlaceWrapper, ReachWrapper, init_env
import numpy as np
import gym

def init_expert(env_name, expl_noise):
	expert = None

	if env_name == 'FetchReach-v1':	
		expert_env = gym.make(env_name)
		expert_env = ReachWrapper(expert_env)
		expert_env._max_episode_steps = 50
		expert = FetchReachExpert(expert_env)

	elif env_name == 'FetchPickAndPlace-v1':
		expert_env = gym.make(env_name)
		expert_env = PickPlaceWrapper(expert_env)
		expert_env._max_episode_steps = 100
		expert = FetchPickPlaceExpert(expert_env)

	return expert

class FetchReachExpert():
	def __init__(self, env):
		self.env = env
		self.env.reset()
	def play_episode(self):

		state = self.env.reset()

		goal = state['desired_goal']

		gripperPos = state['observation'][:3]

		object_oriented_goal = (goal-gripperPos)

		ep_images, ep_state, ep_actions, ep_next_state, ep_reward, ep_done_bool = \
																		[],[],[],[],[],[]
		timeStep = 0

		self.env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
		self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
		img1 = self.env.render(mode='rgb_array')

		self.env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
		self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
		img2 = self.env.render(mode='rgb_array')

		ep_images.append((img1,img2))

		
		state = np.append(state['observation'], goal)
		
		while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= self.env._max_episode_steps:
			# self.env.render()
			action = [0, 0, 0, 0]

			for i in range(len(object_oriented_goal)):
				if np.abs(object_oriented_goal[i]) >= 0.01:
					# action[i] = object_oriented_goal[i]*6

					action[i] = np.sign(object_oriented_goal[i])*0.5
					break


			next_state, reward, done, info = self.env.step(action)
			
			if info is False:
				print('bad episode')
				return None, None, None, None, None, None, None

			done_bool = float(done) if timeStep < self.env._max_episode_steps else 0.0

			timeStep += 1
			
			gripperPos = next_state['observation'][:3]
			object_oriented_goal = (goal-gripperPos)

			next_state = np.append(next_state['observation'], goal)

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
			img1 = self.env.render(mode='rgb_array')

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
			img2 = self.env.render(mode='rgb_array')

			ep_images.append((img1,img2))

			ep_state.append(state)
			ep_actions.append(action_to_save)
			ep_next_state.append(next_state)
			ep_reward.append(reward)
			ep_done_bool.append(done_bool)

			state = next_state

		saved_state = [None for _ in ep_state]
		ep_done_bool[-1] = 1
		return ep_images, ep_state, ep_actions, ep_next_state, \
						ep_reward, ep_done_bool, saved_state

from mujoco_py.generated import const
from trans_fetch_img import transform_image
class FetchPickPlaceExpert():
	def __init__(self, env):
		self.env = env
		self.env.reset()

	def play_episode(self):

		state = self.env.reset()

		# goal = state['observation'][25:28]
		goal = state['desired_goal']

		#objectPosition
		objectPos = state['observation'][3:6]
		gripperPos = state['observation'][:3]
		object_rel_pos = state['observation'][6:9]


		object_oriented_goal = (goal-gripperPos)
		object_oriented_goal[2] += 0.03

		timeStep = 0

		ep_images, ep_state, ep_actions, ep_next_state, ep_reward, ep_done_bool, ep_saved_state = \
																		[],[],[],[],[],[],[]
		
		self.env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
		self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
		img1 = self.env.render(mode='rgb_array')

		self.env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
		self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
		img2 = self.env.render(mode='rgb_array')

		ep_images.append((img1,img2))

		
		state = np.append(state['observation'], goal)
		
		while np.linalg.norm(object_oriented_goal) >= 0.005 and timeStep <= self.env._max_episode_steps:
			# self.env.render()
			action = [0, 0, 0, 0]

			object_oriented_goal = object_rel_pos.copy()
			object_oriented_goal[2] += 0.03

			for i in range(len(object_oriented_goal)):
				action[i] = object_oriented_goal[i]*6

			action[len(action)-1] = 0.05

			next_state, reward, done, info = self.env.step(action)
			
			if info is False:
				print('bad episode')
				return None, None, None, None, None, None, None

			done_bool = float(done) if timeStep < self.env._max_episode_steps else 0.0

			timeStep += 1
			
			saved_state = self.env.sim.get_state().flatten().copy()
			
			objectPos = next_state['observation'][3:6]
			gripperPos = next_state['observation'][:3]
			object_rel_pos = next_state['observation'][6:9]
			#set goal to correct one since loading state in mujoco only loads robot positions 
			next_state = np.append(next_state['observation'], goal)

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
			img1 = self.env.render(mode='rgb_array')

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
			img2 = self.env.render(mode='rgb_array')

			ep_images.append((img1,img2))

			ep_state.append(state)
			ep_actions.append(action)
			ep_next_state.append(next_state)
			ep_reward.append(reward)
			ep_done_bool.append(done_bool)
			ep_saved_state.append(saved_state)

			state = next_state


		while np.linalg.norm(object_rel_pos) >= 0.005 and timeStep <= self.env._max_episode_steps :
			# self.env.render()
			action = [0, 0, 0, 0]

			for i in range(len(object_rel_pos)):
				action[i] = object_rel_pos[i]*6

			action[len(action)-1] = -0.05

			next_state, reward, done, info = self.env.step(action)
			done_bool = float(done) if timeStep < self.env._max_episode_steps else 0.0

			timeStep += 1

			saved_state = self.env.sim.get_state().flatten().copy()

			objectPos = next_state['observation'][3:6]
			gripperPos = next_state['observation'][:3]
			object_rel_pos = next_state['observation'][6:9]

			#set goal to correct one since loading state in mujoco only loads robot positions 
			next_state = np.append(next_state['observation'], goal)

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
			img1 = self.env.render(mode='rgb_array')

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
			img2 = self.env.render(mode='rgb_array')

			ep_images.append((img1,img2))

			ep_state.append(state)
			ep_actions.append(action)
			ep_next_state.append(next_state)
			ep_reward.append(reward)
			ep_done_bool.append(done_bool)
			ep_saved_state.append(saved_state)

			state = next_state



		while np.linalg.norm(goal - objectPos) >= 0.01 and timeStep <= self.env._max_episode_steps :
			# self.env.render()
			action = [0, 0, 0, 0]

			for i in range(len(goal - objectPos)):
				action[i] = (goal - objectPos)[i]*6

			action[len(action)-1] = -0.05

			next_state, reward, done, info = self.env.step(action)

			done_bool = float(done) if timeStep < self.env._max_episode_steps else 0.0

			timeStep += 1

			saved_state = self.env.sim.get_state().flatten().copy()

			objectPos = next_state['observation'][3:6]
			gripperPos = next_state['observation'][:3]
			object_rel_pos = next_state['observation'][6:9]

			#set goal to correct one since loading state in mujoco only loads robot positions 
			next_state = np.append(next_state['observation'], goal)

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
			img1 = self.env.render(mode='rgb_array')

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
			img2 = self.env.render(mode='rgb_array')

			ep_images.append((img1,img2))

			ep_state.append(state)
			ep_actions.append(action)
			ep_next_state.append(next_state)
			ep_reward.append(reward)
			ep_done_bool.append(done_bool)
			ep_saved_state.append(saved_state)

			state = next_state


		while timeStep <= self.env._max_episode_steps:
			# self.env.render()
			action = [0, 0, 0, 0]

			action[len(action)-1] = -0.05

			next_state, reward, done, info = self.env.step(action)

			reward = 1.0

			done_bool = float(done) if timeStep < self.env._max_episode_steps else 0.0

			timeStep += 1

			saved_state = self.env.sim.get_state().flatten().copy()
			
			objectPos = next_state['observation'][3:6]
			gripperPos = next_state['observation'][:3]
			object_rel_pos = next_state['observation'][6:9]
			
			#set goal to correct one since loading state in mujoco only loads robot positions 
			next_state = np.append(next_state['observation'], goal)

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
			img1 = self.env.render(mode='rgb_array')

			self.env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
			self.env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
			img2 = self.env.render(mode='rgb_array')

			ep_images.append((img1,img2))

			ep_state.append(state)
			ep_actions.append(action)
			ep_next_state.append(next_state)
			ep_reward.append(reward)
			ep_done_bool.append(done_bool)
			ep_saved_state.append(saved_state)

			state = next_state



			# if timeStep >= self.env._max_episode_steps: break
		# print(sum(ep_reward))
		saved_state = [None for _ in ep_state]
		ep_done_bool[-1] = 1
		return ep_images, ep_state, ep_actions, ep_next_state, \
						ep_reward, ep_done_bool, saved_state
