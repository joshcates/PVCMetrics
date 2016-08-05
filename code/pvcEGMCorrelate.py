import argparse
import pvcMetrics as pvc
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import sys

# Parse command line args
parser = argparse.ArgumentParser(description='Compute the average correlation in time between two sets of time series electrical potential signals.')
parser.add_argument('-gt', '--ground_truth', help='filename of the ground truth .mat file', required=True)
parser.add_argument('-sol', '--solution', help='filename of the solution .mat file', required=True)
parser.add_argument('-v','--verbose',help='verbose = produces graphs and other information')

args = parser.parse_args()

if args.verbose:
    verbose = True
else:
    verbose = False


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

(corrEGM, x_mean, y_mean, cor) = pvc.corrEGM(X_GT,X_SOL)

if (verbose == True):

    print( "Ground truth has %d nodes and %d samples per node"
           % (gt_number_of_nodes,gt_number_of_samples) )
    
    print( "Solution has %d nodes and %d samples per node"
           % (sol_number_of_nodes,sol_number_of_samples) )
    
    print ("corrEGM = %f", corrEGM)
    
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
    
    ax3.plot(t, cor)
    ax3.grid(True)
    ax3.axhline(0,color = 'black', lw=3)
    ax3.set_title('Correlation')
    
    plt.show()

sys.stdout.write('%f' % corrEGM)
sys.stdout.flush()
sys.exit(0)    
