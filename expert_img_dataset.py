import os

import torch
import cv2
from torchvision import transforms
from PIL import Image
from time import sleep
class SaveEpisodeImages():

	def __init__(self, save_location):

		self.save_location = save_location
	def save_image_to_dir(self, ep_images, episode_num):


		episode_path = os.path.join(self.save_location,'episode'+str(episode_num))  
		aux_cam_path = os.path.join(episode_path, 'aux_cam')  
		head_cam_path = os.path.join(episode_path, 'head_cam')  
		os.mkdir(episode_path)  
		os.mkdir(aux_cam_path)  
		os.mkdir(head_cam_path)  

		ep_num = []
		ep_step = []

		for cam_idx, (aux_cam_img, head_cam_img) in enumerate(ep_images):

			aux_cam_filename = os.path.join(aux_cam_path,str(cam_idx)+'.jpg')
			cv2.imwrite(aux_cam_filename, aux_cam_img)

			head_cam_filename = os.path.join(head_cam_path,str(cam_idx)+'.jpg')
			cv2.imwrite(head_cam_filename, head_cam_img)

			ep_num.append(episode_num)
			ep_step.append(cam_idx)

		return (ep_num, ep_step)



class ExpertImageDataset(torch.utils.data.Dataset):
	"""Face Landmarks dataset."""

	def __init__(self, demo_dir):
		self.demo_dir = demo_dir
		self.transform=transforms.Compose([
											   # transforms.CenterCrop(128),
											   # transforms.Resize(128),
											   transforms.Resize(128,interpolation=Image.HAMMING),
											   # transforms.Resize(256,interpolation=Image.HAMMING),
											   # transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=(0.5,0.5)),

											   # transforms.Resize(80),
											   transforms.Grayscale(num_output_channels=1),
											   # transforms.RandomPerspective(p=1.0),
											   transforms.RandomAffine(45,shear=45),
											   transforms.ToTensor(),
											   # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
										   ])

	def __len__(self):
		return self.replay_buffer.size

	def __getitem__(self, sample):

		ep_num, ep_step, state, action, next_state = sample
		cam_images = []
		for i, j in zip(ep_num, ep_step):
			start_aux_cam_image = Image.open(os.path.join(self.demo_dir, \
				'episode'+str(int(i)),'aux_cam',\
				str(0)+'.jpg')).convert("RGB")

			start_head_cam_image = Image.open(os.path.join(self.demo_dir, \
				'episode'+str(int(i)),'head_cam',\
				str(0)+'.jpg')).convert("RGB")
			
			aux_cam_image = Image.open(os.path.join(self.demo_dir, \
				'episode'+str(int(i)),'aux_cam',\
				str(int(j))+'.jpg')).convert("RGB")

			head_cam_image = Image.open(os.path.join(self.demo_dir, \
				'episode'+str(int(i)),'head_cam',\
				str(int(j))+'.jpg')).convert("RGB")


			cam_images.append(torch.stack(
								(self.transform(start_aux_cam_image), \
								self.transform(aux_cam_image), \
								self.transform(start_head_cam_image), \
								self.transform(head_cam_image))))

		cam_images = torch.stack(cam_images)

		action = torch.FloatTensor(action)
		state = torch.FloatTensor(state)
		return (cam_images, state, action, next_state)
import numpy as np
class DemoBatchSampler(object):

	def __init__(self, replay_buffer, batch_size, demo_dir, eval_freq):


		self.replay_buffer = replay_buffer
		self.batch_size = batch_size
		self.demo_dir = demo_dir
		self.eval_freq = eval_freq

	def __iter__(self):
		
		minibatches = int(self.replay_buffer.size / self.batch_size)
		for epoch in range(self.eval_freq):
			shuffle_indeces = np.random.choice(np.arange(self.replay_buffer.size), \
				(minibatches, self.batch_size), \
				replace=True)

			for i in range(minibatches):
				yield self.replay_buffer.sample_ind([shuffle_indeces[i]])

	def __len__(self):
		return np.inf
		# return self.replay_buffer.size / self.batch_size