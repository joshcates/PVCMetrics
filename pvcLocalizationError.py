import argparse
import pvcMetrics as pvc
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parse command line args
parser = argparse.ArgumentParser(description='Compute the average correlation in time between two sets of time series electrical potential signals.')
parser.add_argument('-gt', '--ground_truth', help='filename of the ground truth .mat file', required=True)
parser.add_argument('-sol', '--solution', help='filename of the solution .mat file', required=True)

args = parser.parse_args()

# Load ground truth data
ground_truth_dict = {}
ground_truth_dict.update(sio.loadmat(args.ground_truth))

print(ground_truth_dict.keys())

gt_potentials_array  = ground_truth_dict['pacingCoordinates_GT']
gt_number_of_nodes   = (gt_potentials_array.shape)[0]

# Load solution data
solution_dict = {}
solution_dict.update(sio.loadmat(args.solution))

sol_potentials_array  = solution_dict['pacingCoordinates']
sol_number_of_nodes   = (sol_potentials_array.shape)[0]

n = np.arange(0, gt_number_of_nodes, 1);
X_GT = gt_potentials_array
X_SOL = sol_potentials_array

print("ground truth shape is",X_GT.shape)

#vect1 = X_GT[:,0];
#print("vect1 = ", vect1)

print( "Ground truth has %d nodes" % gt_number_of_nodes )
print( "Solution has %d nodes"  % sol_number_of_nodes )

print(X_SOL)

# c = pvc.corrACTT(X_GT,X_SOL)

# print ("corrACTT = %f" % c)

# ## Plot
# fig, (ax1, ax2) = plt.subplots(2, 1, sharex="all")

# ax1.plot(n, X_GT)
# ax1.grid(True)
# ax1.axhline(0,color = 'black', lw=3)
# ax1.set_title('GT')

# ax2.plot(n, X_SOL)
# ax2.grid(True)
# ax2.axhline(0,color = 'black', lw=3)
# ax2.set_title('Sol')

# plt.show()
