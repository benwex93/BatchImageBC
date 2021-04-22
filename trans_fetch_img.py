import gym
import numpy as np


import cv2
cv2.ocl.setUseOpenCL(False)
import numpy as np
WIDTH = 128
# WIDTH = 256
# HEIGHT = 256
HEIGHT = 128
import os
from torchvision import transforms
aux_cam_i = 0
head_cam_i = 0
mixed_cam_i = 0
# def transform_image(image, cam_type):
data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.ToTensor(),
])
def transform_image(image):
	# image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
	# image = np.true_divide(image, 255)

	global data_transforms
	image = data_transforms(image)
	image /= 255
	# global aux_cam_i
	# global head_cam_i
	# global mixed_cam_i
	# if cam_type=='aux_cam':
	# 	filename = os.path.join(os.getcwd(),'episode_aux_cam_images','savedImage'+str(aux_cam_i)+'.jpg')
	# 	aux_cam_i+=1
	# elif cam_type=='head_cam':
	# 	filename = os.path.join(os.getcwd(),'episode_head_cam_images','savedImage'+str(head_cam_i)+'.jpg')
	# 	head_cam_i+=1
	
	# mixed_filename = os.path.join(os.getcwd(),'episode_mixed_cam_images','savedImage'+str(mixed_cam_i)+'.jpg')
	# mixed_cam_i+=1
	# cv2.imwrite(filename, image)
	# cv2.imwrite(mixed_filename, image)
	# import pdb
	# pdb.set_trace()

	return image
