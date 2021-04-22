import numpy as np
import os
from matplotlib import pyplot as plt
import copy

class DrawGraphs:
	def __init__(self):
		pass
	def draw(self):
		env = 'fetch'
		# env='cheetah'
		Fetch_Expert_dir = './results/FetchPickAndPlace-v1_episodes:_101_states:_9393.npy'
		Fetch_Expert_files = np.load(Fetch_Expert_dir, allow_pickle=True)

		StateVectorBC_losses_file = './fc_results/StateActionBC_FetchPickAndPlace-v1losses.npy'
		StateVectorBC_evaluations_file = './fc_results/StateActionBC_FetchPickAndPlace-v1evaluations.npy'

		# StateVectorBC_losses_file = np.load(StateVectorBC_losses_file, allow_pickle=True)
		StateVectorBC_evaluations_file = np.load(StateVectorBC_evaluations_file, allow_pickle=True)
	

		StateVectorBC_evaluations_file = np.array(StateVectorBC_evaluations_file)
		StateVectorBC_evaluations_file = np.array(StateVectorBC_evaluations_file).T
		StateVectorBC_evaluations_reward_mean = [np.mean(step) for step in StateVectorBC_evaluations_file]
		StateVectorBC_evaluations_reward_var = [np.var(step) for step in StateVectorBC_evaluations_file]

		Fetch_Expert_mean = [np.mean(Fetch_Expert_files) for _ in range(len(StateVectorBC_evaluations_file))]
		Fetch_Images_reward = [0 for _ in range(len(StateVectorBC_evaluations_file))]

		StateVectorBC_evaluations_range = range(0, len(StateVectorBC_evaluations_file))
		Fetch_Expert_range = range(0, len(StateVectorBC_evaluations_file))

		plt.title('Fetch_Expert_evaluation_average vs. StateVectorBC_evaluation \n vs. StateImagesBC_evaluation')
		plt.xlabel("training epochs")
		plt.ylabel("evaluation rewards")


		plt.plot(Fetch_Expert_range, Fetch_Expert_mean, label="Fetch_Expert_evaluation_average", color='b')
		plt.plot(StateVectorBC_evaluations_range, StateVectorBC_evaluations_reward_mean, label="StateVectorBC_evaluation", color='g')
		plt.plot(Fetch_Expert_range, Fetch_Images_reward, label="StateImagesBC_evaluation", color='r')

		plt.legend()
		plt.savefig("Fetch_Expert_evaluation_average StateImagesBC_evaluation StateVectorBC_evaluations_reward_mean.png")
		plt.close() 

if __name__ == "__main__":

	dg = DrawGraphs()
	dg.draw()