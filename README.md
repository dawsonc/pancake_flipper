Pancake flipping via trajectory optimization
============================================

Unlike some breakfast preparation tasks (e.g. pouring orange juice), pancakes cannot be prepared using only traditional prehensile manipulation techniques (put simply, you can't just pick up the half-cooked pancake, turn it over, and put it back down). Instead, a pancake can only be flipped using underactuated (or "non-prehensile") techniques, where we have to flip the pancake by tossing it into the air or sliding it around on the pan. This is a special case of the more general non-prehensile manipulation problem.
Here, we solve this problem using non-convex optimization with implicit contact mechanics. Those interested in the details can watch the [demonstration video](https://www.youtube.com/watch?v=q2j53m65DKU) or read the full report in `technical_report.pdf`.

It's always important to be up-front about your assumptions and the caveats of your approach. My assumptions are:

- Frictionless contact between the pan and pancake (i.e. I assume you've put enough butter in the pan). I've added some of the constraints for friction, but left them commented out (or set the friction coefficient to 0, which is effectively the same); feel free to try adding them back in if you want!
- Rigid pancakes. It's very hard to model and simulate a non-rigid pancake, so maybe this is really latke-flipping, or flipping a very dense whole-wheat pancake.
- Inelastic collisions between the pan and pancake. If you ever see a pancake bounce let me know (and probably don't eat that pancake).
- The pan is fully actuated, while the pancake is unactuated.

The main caveat of this approach is that it uses non-convex optimization, which is a local method that is not guaranteed to converge, is not complete, and can be highly sensitive to initial conditions, the specific formulation of your constraints, the phase of the moon, and the alignment of the planets.

This project is open-source (under the MIT license). I won't commit to maintaining it in the long-term, but I'm happy to have a conversation about the code (email me at cbd [at] mit [dot] edu or open an issue here).

Installation
------------

This project depends on `numpy`, `matplotlib`, and `pydrake`, and is written in Python 3 and C++. I recommend installing Drake and PyDrake using pre-compiled binaries, by following the directions [here](https://drake.mit.edu/from_binary.html).

Most of the code can be run using Python (I used 3.6.9), but to visualize trajectories being executed on a robot arm in 3D, you'll need to build the C++ code for that visualization. This has the additional dependencies of `cmake` and `gflags` (see [here](https://github.com/gflags/gflags/blob/master/INSTALL.md) for installation instructions for `gflags`).

```
mkdir build && cd build
cmake ..
cmake --build .
cd ..
# To run (see instructions below):
./build/arm_visualizer
```

Using the code
--------------
There are five ways you can interact with this code. The first four all provide simple plots and 2D animations of the trajectory.

- Running `python3 pancake_flipper_trajopt.py` to solve the trajectory optimization problem defined in `pancake_flipper_trajopt.py`. By default, we solve for a trajectory that flips the pancake, but if you want to see a trajectory that slides the pancake, add the `--slide` option. After optimizing, `pancake_flipper_trajopt.py` will save both an MP4 and NPZ file to the `results` directory, saving an animation of the simulated trajectory and the details of the optimized trajectory respectively. The NPZ file may be loaded using NumPy. Run `python3 pancake_flipper_trajopt.py --help` for more command-line options, or dig into the code to see the details of the trajectory optimization problem.
- Running `python3 process_results.py` to plot the trajectory trace NPZ file saved by `pancake_flipper_trajopt.py`. To customize the resulting plot, you'll need to modify the `process_results.py` script, but it's a pretty straightforward wrapper around `matplotlib`.
- Running `python3 trajectory_execute.py` to see what it would look like to execute the open-loop trajectory without any controller. This will almost certainly fail, but...
- Running either `python3 trajectory_execute_inverse_dynamics_controller_flipping.py` or `python3 trajectory_execute_inverse_dynamics_controller_sliding.py` will execute the trajectory found by `pancake_flipper_trajopt.py` using a partial feedback linearization state-feedback controller to track the nominal trajectory. This simulates the contact dynamics between the pan and pancake using Drake, so the results are fairly "real" (up to the fidelity of the simulation). Both of these scripts will plot the tracking performance and control effort used during execution. We need a separate script for flipping and sliding because the PID gains for the state-feedback controller need to be slightly different (and yes, unfortunately you may need to do a lot of PID tuning to get this to work; this is a drawback of using a relatively simple controller in this context).

The fifth way provides a 3D visualization of a KUKA arm flipping the pancake, but it's a little more involved. You'll need to build the C++ file for visualizing the trajectory (see instructions above).

1. Run `python3 convert_npz_trajectory_to_csv.py` to (as the name says) convert the NPZ trajectory trace generated by `pancake_flipper_trajopt.py` to a CSV file.
2. Run `{path-to-drake}/bin/drake-visualizer` (on my machine, `{path-to-drake}=/opt/drake`).
3. Run `arm_visualizer` to visualize the flipping motion on an arm! Check out the command-line flags to load specific datafiles (the defaults visualize a good example of flipping, run `arm_visualizer --help` for more details).
