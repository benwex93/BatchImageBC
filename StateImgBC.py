import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models,transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import math
class ConvActor(nn.Module):

	def __init__(self, action_dim):
		super(ConvActor, self).__init__()

		self.aux_conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=2)
		self.head_conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=2)

		self.aux_conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=2)
		self.head_conv2 = nn.Conv2d(10, 20, kernel_size=3, stride=2)
		self.conv2_drop = nn.Dropout2d(p=0.2)
		self.fc1 = nn.Linear(3920, 128)
		self.fc2 = nn.Linear(128, action_dim)
		self.apply(self.init_weights)
	def init_weights(self, m):
		if type(m) == nn.Linear :
			torch.nn.init.xavier_uniform(m.weight)
			m.bias.data.fill_(0.01)

		if type(m) == nn.Conv2d :
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
	def forward(self, x):
		# aux_cam_images
		#stack images on top of each other
		aux_cam_images = x[:,0,:,:,:]
		head_cam_images = x[:,1,:,:,:]
		start_aux_cam_images = x[:,2,:,:,:]
		start_head_cam_images = x[:,3,:,:,:]

		aux_cam_images = F.leaky_relu(F.max_pool2d(self.aux_conv1(aux_cam_images), 2))
		head_cam_images = F.leaky_relu(F.max_pool2d(self.head_conv1(head_cam_images), 2))
		start_aux_cam_images = F.leaky_relu(F.max_pool2d(self.aux_conv1(start_aux_cam_images), 2))
		start_head_cam_images = F.leaky_relu(F.max_pool2d(self.head_conv1(start_head_cam_images), 2))

		aux_cam_images = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.aux_conv2(aux_cam_images)), 2))
		head_cam_images = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.head_conv2(head_cam_images)), 2))
		start_aux_cam_images = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.aux_conv2(start_aux_cam_images)), 2))
		start_head_cam_images = F.leaky_relu(F.max_pool2d(self.conv2_drop(self.head_conv2(start_head_cam_images)), 2))

		aux_cam_images = aux_cam_images.view(aux_cam_images.shape[0],-1)
		head_cam_images = head_cam_images.view(head_cam_images.shape[0],-1)
		start_aux_cam_images = start_aux_cam_images.view(start_aux_cam_images.shape[0],-1)
		start_head_cam_images = start_head_cam_images.view(start_head_cam_images.shape[0],-1)

		x = torch.cat((aux_cam_images,head_cam_images,start_aux_cam_images,start_head_cam_images),dim=-1)

		x = F.leaky_relu(self.fc1(x))
		return torch.tanh(self.fc2(x))

class ImageActor(nn.Module):
	def __init__(self, action_dim):
		super(ImageActor, self).__init__()
		#128
		self.aux_cam_net = nn.Sequential(
			nn.Linear(49152, 1000),
			nn.ReLU(True),
			nn.Linear(8000, 2000),
			nn.ReLU(True),
			nn.Linear(2000, 100),
			nn.ReLU(True),
		)
		self.head_cam_net = nn.Sequential(
			nn.Linear(49152, 1000),
			nn.ReLU(True),
			nn.Linear(8000, 2000),
			nn.ReLU(True),
			nn.Linear(2000, 100),
			nn.ReLU(True),
		)
		self.cat_net = nn.Sequential(
			nn.Linear(100*4, 50),
			nn.ReLU(True),
			nn.Linear(50, action_dim),
			# nn.Tanh(),
		)

	def forward(self, x):

		aux_cam_images = x[:,0,:,:,:].flatten(1)
		head_cam_images = x[:,1,:,:,:].flatten(1)
		start_aux_cam_images = x[:,2,:,:,:].flatten(1)
		start_head_cam_images = x[:,3,:,:,:].flatten(1)

		aux_cam_images = self.aux_cam_net(aux_cam_images)
		head_cam_images = self.head_cam_net(head_cam_images)
		start_aux_cam_images = self.aux_cam_net(start_aux_cam_images)
		start_head_cam_images = self.head_cam_net(start_head_cam_images)

		x = torch.cat((aux_cam_images, head_cam_images, start_head_cam_images, start_aux_cam_images),dim=1)

		return self.cat_net(x)

class StateImgBC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		lr,
	):
		self.actor = ConvActor(action_dim).to(device)
		# self.actor = ImageActor(action_dim).to(device)
		self.bc_actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)

		self.max_action = max_action

		self.total_train_it = 0
		self.total_dev_it = 0


	def select_action(self, state):

		(state_img, state_vector) = state
		
		with torch.no_grad():
			self.actor.eval()

			#img representation
			state_img = state_img.unsqueeze(0).to(device)

			action = self.actor(state_img).cpu().data.numpy().flatten()

			return action


	def train(self, batch):
		self.total_train_it += 1

		cam_images, state, action, next_state = batch


		out = self.actor(cam_images.to(device))

		bc_loss = F.smooth_l1_loss(out, action.to(device)).sum()
		# bc_loss = F.mse_loss(out, action.to(device)).mean()
		
		# Optimize the actor 
		self.bc_actor_optimizer.zero_grad()
		bc_loss.backward()
		
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 40, norm_type=2)

		self.bc_actor_optimizer.step()

		return bc_loss


	def validate(self, batch):
		self.total_dev_it += 1

		cam_images, state, action, next_state = batch

		#img representation
		with torch.no_grad():

			out = self.actor(cam_images.to(device))

			bc_loss = F.smooth_l1_loss(out, action.to(device)).sum()
			# bc_loss = F.mse_loss(out, action.to(device)).mean()

		return bc_loss
	def save(self, filename):
		
		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")
