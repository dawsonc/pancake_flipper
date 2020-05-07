'''
Executes an optimized trajectory.

Written by C. Dawson (cbd@mit.edu)
'''
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rc
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser,
    PlanarSceneGraphVisualizer, Simulator, VectorSystem,
    Multiplexer, Demultiplexer, MatrixGain, LogOutput, plot_system_graphviz,
    TrajectorySource, PiecewisePolynomial, InverseDynamicsController,
    MultibodyPlant, LogOutput
)


def build_pancake_flipper_plant(builder):
    '''Creates a pancake_flipper MultibodyPlant. Returns the plant and
    corresponding scene graph.

    Inputs:
        builder -- a DiagramBuilder
    Outputs:
        the MultibodyPlant object and corresponding SceneGraph representing
        the pancake flipper
    '''
    # Instantiate the pancake flipper plant and the scene graph.
    # The scene graph is a container for the geometries of all
    # the physical systems in our diagram
    pancake_flipper, scene_graph = AddMultibodyPlantSceneGraph(
        builder,
        time_step=0.0  # discrete update period, or zero for continuous systems
    )

    # parse the urdf and populate the plant
    urdf_path = './models/pancake_flipper_massless_links_nofriction.urdf'
    Parser(pancake_flipper).AddModelFromFile(urdf_path)

    # Finalize the plant so we can use it
    pancake_flipper.Finalize()

    # Return the plant and scene graph, as promised
    return pancake_flipper, scene_graph


# Create the builder to house our system
builder = DiagramBuilder()
# Create the plant and scene graph
pancake_flipper, scene_graph = build_pancake_flipper_plant(builder)

# Load the nominal state trajectory
datafile = 'results/arm_viz2_trace_umax40_ceiling5_mu0.0_T60.npz'
npzfile = np.load(datafile)
h_opt = npzfile['h_opt']
t = np.cumsum(h_opt)
# Add a zero at the start for t
t = np.hstack(([0], t))
T = len(t) - 1
q_opt = npzfile['q_opt']
qd_opt = npzfile['qd_opt']
# Also get the initial state
q_opt_0 = q_opt[0, :]
# get the nominal control input as well for comparison at the end
u_opt = npzfile['u_opt']

# Extract the flipper trajectory
q_opt_flipper = q_opt[:, [0, 2, 4]]
q_opt_poly = PiecewisePolynomial.FirstOrderHold(t, q_opt_flipper.T)
control_source_q = builder.AddSystem(TrajectorySource(q_opt_poly))
# Extract the flipper velocity
qd_opt_flipper = qd_opt[:, [0, 2, 4]]
qd_opt_poly = PiecewisePolynomial.FirstOrderHold(t, qd_opt_flipper.T)
control_source_v = builder.AddSystem(TrajectorySource(qd_opt_poly))

# Make an inverse dynamics controller for tracking the trajectory
# We need to make a new plant that has only the flipper, since the pancake
# is underactuated
flipper_only_plant = MultibodyPlant(time_step=0)
file_name = './models/pancake_flipper_only.urdf'
Parser(flipper_only_plant).AddModelFromFile(file_name)
flipper_only_plant.Finalize()
# And we can make an inverse dynamics controller based on that plant,
# with some specified PID gains
Kp = np.array([1, 10, 10])
Ki = np.array([1, 50, 1])
Kd = np.array([50, 50, 50])
controller = builder.AddSystem(InverseDynamicsController(flipper_only_plant,
                                                         Kp, Ki, Kd,
                                                         False))

# Connect the desired trajectory source to the controller. We need to use a
# multiplexer to squish the desired states and velocities together
nominal_trajectory_mux = builder.AddSystem(Multiplexer([3, 3]))
builder.Connect(control_source_q.get_output_port(0),
                nominal_trajectory_mux.get_input_port(0))
builder.Connect(control_source_v.get_output_port(0),
                nominal_trajectory_mux.get_input_port(1))
builder.Connect(nominal_trajectory_mux.get_output_port(0),
                controller.get_input_port_desired_state())

# Also provide the controller with the direct state measurements from the
# MultiBodyPlant. However, we need to extract the flipper states and velocities
# from this, so we run them through a demux and then a mux to repackage
# only the desired states
estimated_state_demux = builder.AddSystem(Demultiplexer(12))
estimated_state_mux = builder.AddSystem(Multiplexer(6))
builder.Connect(pancake_flipper.get_state_output_port(),
                estimated_state_demux.get_input_port(0))
builder.Connect(estimated_state_demux.get_output_port(0),
                estimated_state_mux.get_input_port(0))
builder.Connect(estimated_state_demux.get_output_port(2),
                estimated_state_mux.get_input_port(1))
builder.Connect(estimated_state_demux.get_output_port(4),
                estimated_state_mux.get_input_port(2))
# Velocities
builder.Connect(estimated_state_demux.get_output_port(6),
                estimated_state_mux.get_input_port(3))
builder.Connect(estimated_state_demux.get_output_port(8),
                estimated_state_mux.get_input_port(4))
builder.Connect(estimated_state_demux.get_output_port(10),
                estimated_state_mux.get_input_port(5))

# Now we can feed the repackaged states to the controller
builder.Connect(estimated_state_mux.get_output_port(0),
                controller.get_input_port_estimated_state())

# Close the loop by connecting the output of the controller to the plant
builder.Connect(controller.get_output_port_control(),
                pancake_flipper.get_actuation_input_port())

# We also want to measure the control input for funsies
u_logger = LogOutput(controller.get_output_port_control(), builder)
# Also measure the states
q_logger = LogOutput(pancake_flipper.get_state_output_port(), builder)

# Add a visualizer so we can see what's going on
max_flipper_x = np.max(q_opt[:, 0])
max_pancake_x = np.max(q_opt[:, 1])
min_flipper_x = np.min(q_opt[:, 0])
min_pancake_x = np.min(q_opt[:, 1])
max_x = max(max_flipper_x, max_pancake_x)
min_x = min(min_flipper_x, min_pancake_x)

max_flipper_y = np.max(q_opt[:, 2])
max_pancake_y = np.max(q_opt[:, 3])
min_flipper_y = np.min(q_opt[:, 2])
min_pancake_y = np.min(q_opt[:, 3])
max_y = max(max_flipper_y, max_pancake_y)
min_y = min(min_flipper_y, min_pancake_y)

xlim = [min_x - 3, max_x + 3]
ylim = [min_y - 2, max_y + 2]
visualizer = builder.AddSystem(
    PlanarSceneGraphVisualizer(scene_graph, xlim=xlim, ylim=ylim))

# Connect the output of the scene graph to the visualizer
builder.Connect(scene_graph.get_pose_bundle_output_port(),
                visualizer.get_input_port(0))

# Build the diagram!
diagram = builder.Build()

# Reset the loggers
u_logger.reset()
q_logger.reset()

# start recording the video for the animation of the simulation
visualizer.start_recording()

# set up a simulator
simulator = Simulator(diagram)

context = simulator.get_mutable_context()
context.SetTime(0.0)  # reset current time
context.SetContinuousState(np.array([
    q_opt_0[0], q_opt_0[1],  # Initial states
    q_opt_0[2], q_opt_0[3],
    q_opt_0[4], q_opt_0[5],
    0, 0,                    # Initial velocity
    0, 0,
    0, 0,
    0, 0, 0                  # Initial state of controller
]))

# generate and display animation
visualizer.start_recording()
# simulator.AdvanceTo(q_opt_poly.end_time())
simulator.AdvanceTo(3.2)
ani = visualizer.get_recording_as_animation()

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Charles Dawson'), bitrate=1800)
stamp = datafile[8:-4]
ani.save('results/closedloop_inv_dyn_foh_' + stamp + '.mp4', writer=writer)

# Set some font parameters for matplotlib
# rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# for Palatino and other serif fonts use:
rc('font', **{'family': 'serif', 'serif': ['Palatino'], 'size': 22})
rc('text', usetex=True)

# Plot the control outputs over time
plt.figure()
plt.plot(t[1:], u_opt[:, 0], 'k--',
         label='$f_x$ (nominal)')
plt.plot(t[1:], u_opt[:, 1], 'b--',
         label='$f_z$ (nominal)')
plt.plot(t[1:], u_opt[:, 2], 'r--',
         label='$f_\\theta$ (nominal)')

plt.plot(u_logger.sample_times(), u_logger.data().T[:, 0], 'k',
         label='$f_x$ (tracking)')
plt.plot(u_logger.sample_times(), u_logger.data().T[:, 1], 'b',
         label='$f_z$ (tracking)')
plt.plot(u_logger.sample_times(), u_logger.data().T[:, 2], 'r',
         label='$f_\\theta$ (tracking)')

plt.xlabel("Time")
plt.ylabel("Generalized force")
plt.title("Flipping behavior (control effort)")
plt.grid(True)
plt.legend(loc="upper center", ncol=2)
plt.show()

# Plot the tracking error over time
plt.figure()
plt.plot(t, q_opt[:, 0], 'k--',
         label='$x$ (nominal)')
plt.plot(q_logger.sample_times(), q_logger.data().T[:, 0], 'k',
         label='$x$ (tracking)')
plt.plot(t, q_opt[:, 2], 'b--',
         label='$z$ (nominal)')
plt.plot(q_logger.sample_times(), q_logger.data().T[:, 2], 'b',
         label='$z$ (tracking)')
plt.plot(t, q_opt[:, 4], 'r--',
         label='$\\theta$ (nominal)')
plt.plot(q_logger.sample_times(), q_logger.data().T[:, 4], 'r',
         label='$\\theta$ (tracking)')

plt.xlabel("Time")
plt.ylabel("State")
plt.title("Flipping behavior (tracking performance)")
plt.grid(True)
plt.legend(loc="lower left")
plt.show()
