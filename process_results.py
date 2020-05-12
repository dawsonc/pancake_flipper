import numpy as np
import matplotlib.pyplot as plt
import argparse


# Define the command line arguments
parser = argparse.ArgumentParser(
    description=('Load and plot a pancake-manipulation trajectory '
                 'found by pancake_flipper_trajopt.py'))
parser.add_argument('datafile',
                    help=('Path to the NPZ file containing the '
                          'trajectory trace.'))
args = parser.parse_args()

# datafile = 'results/slide_trace_umax50_ceiling5_T60.npz'
datafile = args.datafile
npzfile = np.load(datafile)

# Construct time from time intervals
t = np.cumsum(npzfile['h_opt'])
T = len(t)

# Extract the other quantities of interest
q_opt = npzfile['q_opt']
qd_opt = npzfile['qd_opt']
qdd_opt = npzfile['qdd_opt']
u_opt = npzfile['u_opt']
f_ll_opt = npzfile['f_ll_opt']
f_lr_opt = npzfile['f_lr_opt']
f_ur_opt = npzfile['f_ur_opt']
f_ul_opt = npzfile['f_ul_opt']

plt.plot(t, u_opt[:, 0], label='Flipper x force')
plt.plot(t, u_opt[:, 1], label='Flipper z force')
plt.plot(t, u_opt[:, 2], label='Flipper theta torque')
# plt.plot(t, f_ul_opt[:, 1], label='ul z', marker='o')
plt.plot(t, f_ll_opt[:, 0], label='ll x', marker='x')
plt.plot(t, f_ll_opt[:, 1], label='ll z', marker='x')
plt.plot(t, f_lr_opt[:, 0], label='lr x', marker='x')
plt.plot(t, f_lr_opt[:, 1], label='lr z', marker='x')
# plt.plot(t, qd_opt[:T, 0], label='Velocity x')
# plt.plot(t, qd_opt[:T, 2], label='Velocity y')
# plt.plot(t, qd_opt[:T, 4], label='Velocity theta')
plt.xlabel('Time (s)')
# plt.ylabel('Velocity')
plt.ylim([-60, 60])
plt.grid(True)
plt.legend()
plt.show()
