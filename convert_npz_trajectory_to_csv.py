'''Load an npz containing a trajectory for the pancake and flipper, extract
the flipper trajectory only, then save it as a csv containing t, x, z, theta
for the flipper only

Also save a csv containing t, x, z, theta for the pancake only
'''
import numpy as np
import argparse


# Define the command line arguments
parser = argparse.ArgumentParser(
    description=('Convert an NPZ trajectory trace to CSV for use with the '
                 'arm_visualizer. Yields one CSV for the flipper and one '
                 'for the pancake.'))
parser.add_argument('datafile',
                    help=('Path to the NPZ file containing the '
                          'trajectory trace.'))
args = parser.parse_args()

# datafile = 'results/arm_viz2_trace_umax40_ceiling5_mu0.0_T60.npz'
datafile = args.datafile
npzfile = np.load(datafile)
h_opt = npzfile['h_opt']
t = np.cumsum(h_opt)
t = np.hstack(([0], t))
T = len(t)
# Just load the states (we're only trying to visualize here, not actually
# stabilize a planned control trajectory).
q_opt = npzfile['q_opt']

# This set indexes into the pancake x, z, theta in q_opt
pancake_dof_idxs = [1, 3, 5]
# This set indexes into the flipper x, z, theta in q_opt
flipper_dof_idxs = [0, 2, 4]

# Get pancake trajectory
t = t.reshape(T, 1)
q_pancake = np.hstack((t, q_opt[:, pancake_dof_idxs]))
# and save
np.savetxt(datafile + "_pancake_to.csv", q_pancake, delimiter=",")

# Get the flipper trajectory
q_flipper = np.hstack((t, q_opt[:, flipper_dof_idxs]))
np.savetxt(datafile + "_flipper_to.csv", q_flipper, delimiter=",")
