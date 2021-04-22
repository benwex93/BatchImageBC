import numpy as np
import torch


class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e5)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.ep_num = np.zeros(max_size)
		self.ep_step = np.zeros(max_size)

		self.state = np.zeros((max_size, state_dim))

		self.action = np.zeros((max_size, action_dim))
		
		self.next_state = np.zeros((max_size, state_dim))

		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def add_episode(self, ep_num, ep_step, \
			ep_state, ep_action, ep_next_state, ep_reward, ep_done_bool):
		for i in range(len(ep_state)):
			self.add(ep_num[i], ep_step[i], ep_state[i],\
				ep_action[i], ep_next_state[i],	ep_reward[i], ep_done_bool[i])

	def add(self, ep_num, ep_step, \
				state, action, next_state, reward, done):

		self.ep_num[self.ptr] = ep_num
		self.ep_step[self.ptr] = ep_step
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def can_sample(self, batch_size):
		if self.size > batch_size:
			return True
		return False

	def sample(self, batch_size):

		ind = np.random.randint(0, self.size-1, size=batch_size)
		return (
			self.ep_num[ind],
			self.ep_step[ind],
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
			self.reward[ind],
			self.not_done[ind],
		)

	def sample_ind(self, ind):

		return (
			self.ep_num[ind],
			self.ep_step[ind],
			self.state[ind],
			self.action[ind],
			self.next_state[ind],
		)

	def delete_first_quarter(self):


		first_quarter = round(self.size*(1/4))

		self.ep_num = self.ep_num[first_quarter:self.size]
		self.ep_step = self.ep_step[first_quarter:self.size]
		self.state = self.state[first_quarter:self.size]
		self.action = self.action[first_quarter:self.size]
		self.next_state = self.next_state[first_quarter:self.size]
		self.reward = self.reward[first_quarter:self.size]
		self.not_done = self.not_done[first_quarter:self.size]

		self.size = self.size - first_quarter

	def retain_first_quarter(self):

		first_quarter = round(self.size*(1/4))
		
		self.ep_num = self.ep_num[:first_quarter]
		self.ep_step = self.ep_step[:first_quarter]
		self.state = self.state[:first_quarter]
		self.action = self.action[:first_quarter]
		self.next_state = self.next_state[:first_quarter]
		self.reward = self.reward[:first_quarter]
		self.not_done = self.not_done[:first_quarter]

		self.size = first_quarter
