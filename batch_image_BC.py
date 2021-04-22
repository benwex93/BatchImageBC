import numpy as np
import torch
import gym
import argparse
import os

import utils
import StateImgBC
import StateVectorBC

from envs import init_env
from moviepy.editor import ImageSequenceClip

from expert_img_dataset import ExpertImageDataset, DemoBatchSampler

from mujoco_py.generated import const
from torchvision import transforms
from tqdm import tqdm
def get_images(eval_env, start_aux_image, start_head_image):

	eval_env.env.env._get_viewer('rgb_array').cam.fixedcamid= -1
	eval_env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FREE
	aux_cam_image = eval_env.env.env.render(mode='rgb_array')

	eval_env.env.env._get_viewer('rgb_array').cam.fixedcamid=3
	eval_env.env.env._get_viewer('rgb_array').cam.type = const.CAMERA_FIXED
	head_cam_image = eval_env.env.env.render(mode='rgb_array')
	
	aux_cam_image = transforms.ToPILImage()(aux_cam_image)
	head_cam_image = transforms.ToPILImage()(head_cam_image)

	aux_cam_image = train_dataset.transform(aux_cam_image)
	head_cam_image = train_dataset.transform(head_cam_image)

	if start_aux_image is None and start_head_image is None:
		state = torch.stack((aux_cam_image, aux_cam_image, head_cam_image, head_cam_image))
	else:
		state = torch.stack((start_aux_image, aux_cam_image, start_head_image, head_cam_image))

	return state

def eval_policy(train_dataset, policy, env_name, seed, num_env_it, eval_episodes=10, policy_name=None):
	eval_episodes=10
	eval_env = init_env(env_name)
	avg_reward = 0.
	gif = []
	frames = 0
	rate = 10
	for episode_num in range(eval_episodes):
		state, done = eval_env.reset(), False

		if policy_name == 'StateActionBC':
			state = (None, state)
		else:
			state = (get_images(eval_env, start_aux_image=None, start_head_image=None), state)
			start_aux_image = state[0][0]
			start_head_image = state[0][2]

		episode_reward = 0

		while not done:

			action = policy.select_action(state)
			# if episode_num == 0:
			if episode_num >= 0:
				frames+=1
				if frames % rate == 0:

					arr = eval_env.render(mode='rgb_array')
					gif.append(arr)

			state, reward, done, _ = eval_env.step(action) 

			if policy_name == 'StateActionBC':
				state = (None, state)
			else:
				state = (get_images(eval_env, start_aux_image, start_head_image), state)

			avg_reward += reward

			episode_reward += reward
		# if episode_num == 0:
		if episode_num == 9:
			clip = ImageSequenceClip(gif, fps=5)
			clip.write_videofile(os.path.join(os.getcwd(),'sample.mp4')) 


	avg_reward /= eval_episodes

	if env_name != 'FetchPickAndPlace-v1':
		eval_env.env.close()

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	return avg_reward

def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})

parser = argparse.ArgumentParser()
if __name__ == "__main__":
	
	parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	boolean_feature("log", False, 'to log to tensorboard')
	boolean_feature("csil", False, 'to log to tensorboard')
	boolean_feature("load_pretrained", False, 'to log to tensorboard')
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds

	parser.add_argument("--eval_freq", default=20, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--cpu_workers", default=4, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
	parser.add_argument("--batch_size", default=32, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--lr", default=0.0003, type=float)       
	parser.add_argument("--save-model", action="store_true")        # Save model and optimizer parameters
	args = parser.parse_args()

	# Set seeds before init_env (env seeds set in init_env)
	# env.seed(args.seed)
	# env.action_space.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.enabled=False
	torch.backends.cudnn.benchmark = False
	np.random.seed(args.seed)
	
	env = init_env(args.env)

	state_dim = env.observation_space['observation'].shape[0] + \
				env.observation_space['desired_goal'].shape[0]

	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])


	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
	}

	# Target policy smoothing is scaled wrt the action scale
	kwargs["lr"] = args.lr

	if args.policy == "StateImgBC":
		policy = StateImgBC.StateImgBC(**kwargs)
	if args.policy == "StateVectorBC":
		policy = StateVectorBC.StateVectorBC(**kwargs)

	evaluations = []
	losses = []

	demo_dir = './exp_img_demos/FetchPickAndPlace-v1_episodes:_101_2021-04-19 01:26:33.798615'
	replay_buffer = None

	import pickle
	for filename in os.listdir(demo_dir):
		if filename.endswith('.pckl'):

			with open(os.path.join(demo_dir,filename), 'rb') as filehandler: 
				training_replay_buffer = pickle.load(filehandler)
				training_replay_buffer.delete_first_quarter()

			with open(os.path.join(demo_dir,filename), 'rb') as filehandler: 
				validation_replay_buffer = pickle.load(filehandler)
				validation_replay_buffer.retain_first_quarter()

	file_name = f"{args.policy}_{args.env}"

	train_dataset = ExpertImageDataset(demo_dir)
	train_sampler = DemoBatchSampler(training_replay_buffer, args.batch_size, \
											demo_dir, args.eval_freq)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None,\
		sampler=train_sampler, num_workers=30, \
		# sampler=train_sampler, num_workers=0, \
		pin_memory=True, drop_last=False)

	dev_dataset = ExpertImageDataset(demo_dir)
	dev_sampler = DemoBatchSampler(validation_replay_buffer, args.batch_size, \
											demo_dir, args.eval_freq)
	dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=None,\
		sampler=dev_sampler, num_workers=30, \
		# sampler=dev_sampler, num_workers=0, \
		pin_memory=True, drop_last=False)

	# evaluations.append(\
	# eval_policy(train_dataset, policy, args.env, args.seed, num_env_it=0, \
	# policy_name=str(args.policy)))

	for t in range(int(args.max_timesteps)):
		# Train agent after collecting sufficient data
		for (i_train_batch, train_batch), (i_dev_batch, dev_batch) in \
				zip(enumerate(tqdm(train_loader)), enumerate(tqdm(dev_loader))):
			train_loss = policy.train(train_batch)
			dev_loss = policy.validate(dev_batch)
			
			if i_train_batch % 100 == 0:
				print('training loss:', train_loss)
				print('validation loss:', dev_loss)
			if args.log:
				losses.append((train_loss,dev_loss))
		print('epoch: ',t)
		evaluations.append(\
			eval_policy(train_dataset, policy, args.env, args.seed, num_env_it=(t + 1), \
			policy_name=str(args.policy)))


		if args.save_model:
			policy.save(f"./models/{file_name}")
		if args.log:
			np.save(f"./fc_results/{file_name}losses", losses)
			np.save(f"./fc_results/{file_name}evaluations", evaluations)

