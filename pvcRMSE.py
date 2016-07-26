import argparse
import pvcMetrics as pvc
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parse command line args
parser = argparse.ArgumentParser(description='Compute relative mean squared error in space, averaged across time, of two sets of time series electrical potential signals.')
parser.add_argument('-gt', '--ground_truth', help='filename of the ground truth .mat file', required=True)
parser.add_argument('-sol', '--solution', help='filename of the solution .mat file', required=True)

args = parser.parse_args()

# Load ground truth data
ground_truth_dict = {}
ground_truth_dict.update(sio.loadmat(args.ground_truth))

gt_potentials_array  = ground_truth_dict['X_GT']
gt_number_of_nodes   = (gt_potentials_array.shape)[0]
gt_number_of_samples = (gt_potentials_array.shape)[1]

# Load solution data
solution_dict = {}
solution_dict.update(sio.loadmat(args.solution))

sol_potentials_array  = solution_dict['X']
sol_number_of_nodes   = (sol_potentials_array.shape)[0]
sol_number_of_samples = (sol_potentials_array.shape)[1]

t = np.arange(0, gt_number_of_samples, 1);
X_GT = gt_potentials_array
X_SOL = sol_potentials_array

print( "Ground truth has %d nodes and %d samples per node"
       % (gt_number_of_nodes,gt_number_of_samples) )

print( "Solution has %d nodes and %d samples per node"
       % (sol_number_of_nodes,sol_number_of_samples) )

(avgRMSE, sdevRMSE, mse) = pvc.RMSE(X_GT,X_SOL)

print ("Average RMSE = %f", avgRMSE)
print ("Std Deviation RMSE = %f", sdevRMSE)

x_mean = np.mean(X_GT, axis = 0)
y_mean = np.mean(X_SOL, axis = 0)

# Plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex="all")

ax1.plot(t, x_mean)
ax1.grid(True)
ax1.axhline(0,color = 'black', lw=3)
ax1.set_title('Mean GT')

ax2.plot(t, y_mean)
ax2.grid(True)
ax2.axhline(0,color = 'black', lw=3)
ax2.set_title('Mean Sol')

ax3.plot(t, mse)
ax3.grid(True)
ax3.axhline(0,color = 'black', lw=3)
ax3.set_title('Mean squared error')

plt.show()



