import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.l3(a)

class StateVectorBC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		lr,
	):

		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.bc_actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.max_action = max_action

		self.total_train_it = 0
		self.total_dev_it = 0


	def select_action(self, state):
		(state_img, state_vector) = state

		with torch.no_grad():
			self.actor.eval()
			state = np.append(state_vector['observation'], state_vector['desired_goal'])
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)
			action = self.actor(state).cpu().data.numpy().flatten()

			return action

	def train(self, batch):
		self.total_train_it += 1

		cam_images, state, action, next_state = batch

		# bc_loss = F.mse_loss(self.actor(state.to(device)), action.to(device)).mean()
		bc_loss = F.smooth_l1_loss(self.actor(state.to(device)), action.to(device)).sum()

		self.bc_actor_optimizer.zero_grad()
		bc_loss.backward()
		self.bc_actor_optimizer.step()

		return bc_loss

	def validate(self, batch):
		self.total_dev_it += 1

		cam_images, state, action, next_state = batch

		#img representation
		with torch.no_grad():

			# # Compute BC losse
			bc_loss = F.smooth_l1_loss(self.actor(state.to(device)), action.to(device)).sum()
		
		return bc_loss

	def save(self, filename):
		print('saving')
		# torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.bc_actor_optimizer.state_dict(), filename + "_actor_optimizer")
