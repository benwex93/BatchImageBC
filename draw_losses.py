import numpy as np
import os
from matplotlib import pyplot as plt
import copy

class DrawGraphs:
	def __init__(self):
		pass
	def draw(self):
		env = 'fetch'
		# StateVectorBC_losses_file = './fc_results/StateActionBC_FetchPickAndPlace-v1losses.npy'

		# StateVectorBC_losses_file = np.load(StateVectorBC_losses_file, allow_pickle=True)

		# StateVectorBC_losses_file = StateVectorBC_losses_file[:5000]

		# StateVectorBC_evaluations_range = range(0, len(StateVectorBC_losses_file))

		# StateVectorBC_losses_file = np.array(StateVectorBC_losses_file)

		# StateVectorBC_train_losses = StateVectorBC_losses_file[:,0]
		# StateVectorBC_dev_losses = StateVectorBC_losses_file[:,1]

		# StateVectorBC_train_losses = [i.cpu().detach().numpy().item() for i in StateVectorBC_train_losses]
		# StateVectorBC_dev_losses = [i.cpu().detach().numpy().item() for i in StateVectorBC_dev_losses]

		
		# plt.title('StateVectorBC train vs dev losses')
		# plt.xlabel("backwards passes")
		# plt.ylabel("SmoothL1Loss Sum")

		# StateVectorBC_train_losses_range = range(0, len(StateVectorBC_train_losses))
		# StateVectorBC_dev_losses_range = range(0, len(StateVectorBC_dev_losses))


		# plt.plot(StateVectorBC_train_losses_range, StateVectorBC_train_losses, label="StateVectorBC_train_losses", color='b', linewidth=0.1)
		# plt.plot(StateVectorBC_dev_losses_range, StateVectorBC_dev_losses, label="StateVectorBC_dev_losses", color='r', linewidth=0.1)

		# plt.legend()
		# plt.savefig("StateVectorBC train vs dev losses.png")
		# plt.close() 


		# ################################
		DetImgBC_losses_file = './fc_results/DetImgBC_FetchPickAndPlace-v1losses.npy'

		DetImgBC_losses_file = np.load(DetImgBC_losses_file, allow_pickle=True)

		DetImgBC_losses_file = DetImgBC_losses_file[:5000]

		DetImgBC_evaluations_range = range(0, len(DetImgBC_losses_file))

		DetImgBC_losses_file = np.array(DetImgBC_losses_file)

		DetImgBC_train_losses = DetImgBC_losses_file[:,0]
		DetImgBC_dev_losses = DetImgBC_losses_file[:,1]

		DetImgBC_train_losses = [i.cpu().detach().numpy().item() for i in DetImgBC_train_losses]
		DetImgBC_dev_losses = [i.cpu().detach().numpy().item() for i in DetImgBC_dev_losses]

		DetImgBC_train_losses_range = range(0, len(DetImgBC_train_losses))
		DetImgBC_dev_losses_range = range(0, len(DetImgBC_dev_losses))

		plt.title('StateImagesBC train vs dev losses')
		plt.xlabel("backwards passes")
		plt.ylabel("SmoothL1Loss Sum")

		axes = plt.gca()
		# axes.set_xlim([xmin,xmax])
		axes.set_ylim([0,0.1])

		plt.plot(DetImgBC_train_losses_range, DetImgBC_train_losses, label="StateImgBC_train_losses", color='b', linewidth=0.1)
		plt.plot(DetImgBC_dev_losses_range, DetImgBC_dev_losses, label="StateImgBC_dev_losses", color='r', linewidth=0.1)
		
		plt.legend()
		plt.savefig("StateImagesBC train vs dev losses.png")
		plt.close() 

if __name__ == "__main__":

	dg = DrawGraphs()
	dg.draw()