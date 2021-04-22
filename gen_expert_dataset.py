import numpy as np
import gym
import argparse
import os

import utils
from envs import init_env
from env_experts import init_expert
from expert_img_dataset import SaveEpisodeImages
from datetime import datetime

def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})

parser = argparse.ArgumentParser()
if __name__ == "__main__":
	boolean_feature("log", True, 'to log to tensorboard')

	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--num_episodes", default=1e4, type=int)   # Max time steps to run environment
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	args = parser.parse_args()


	
	env = init_env(args.env, gen_exp=True)
	state_dim = env.observation_space.shape[0]

	action_dim = env.action_space.shape[0]
	max_action = float(env.action_space.high[0])
	
	demos_dir_name = f"{args.env}_episodes:_{args.num_episodes}_{str(datetime.now())}"
	demos_save_location = os.path.join("./exp_img_demos/",demos_dir_name)
	os.mkdir(demos_save_location)

	img_saver = SaveEpisodeImages(demos_save_location)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	expert = init_expert(args.env, args.expl_noise)
	evaluations = []
	overall_states = 0
	for t in range(int(args.num_episodes)):
		# Reset environment
		# ep_state, ep_actions, ep_next_state, ep_reward, ep_done_bool, ep_saved_state = \
		# 																	[],[],[],[],[],[]
		ep_images, ep_state, ep_actions, ep_next_state, ep_reward, ep_done_bool, \
			ep_saved_state = expert.play_episode()
			
		if ep_state == ep_actions == ep_next_state == ep_reward == ep_done_bool == \
			ep_saved_state == None:
			continue

		ep_num, ep_step = img_saver.save_image_to_dir(ep_images, t)

		replay_buffer.add_episode(ep_num, ep_step,
			ep_state, ep_actions, ep_next_state, ep_reward, ep_done_bool)

		# replay_buffer.add_episode(ep_state, ep_actions, ep_reward, ep_done_bool)

		overall_states += len(ep_state)

		evaluations.append(sum(ep_reward))
		# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
		print(f"Total states: {overall_states} Episode Num: {t+1} Episode T: Reward: {sum(ep_reward):.3f}")

	# if args.log:
	# 	np.save(f"./results/{file_name}", evaluations)
	replay_file = os.path.join(demos_save_location,'replay_buff.pckl')
	import pickle 
	with open(replay_file, 'wb') as file_pi:
		pickle.dump(replay_buffer, file_pi)
	# np.save(replay_file, replay_buffer, allow_pickle=True)