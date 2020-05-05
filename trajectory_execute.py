'''
Executes an optimized trajectory.

Written by C. Dawson (cbd@mit.edu)
'''
import numpy as np
import matplotlib.animation as animation
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser,
    PlanarSceneGraphVisualizer, Simulator, VectorSystem,
    Multiplexer, MatrixGain, LogOutput, plot_system_graphviz,
    TrajectorySource, PiecewisePolynomial
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

# Load the control trajectory
# datafile = 'results/slide_trace_umax50_ceiling5_T60.npz'
# datafile = 'results/flip_trace_umax40_ceiling5_mu0.1_T60.npz'
datafile = 'results/slide_trace_umax40_ceiling5_mu0.0_T60.npz'
npzfile = np.load(datafile)
h_opt = npzfile['h_opt']
t = np.cumsum(h_opt)
T = len(t)
u_opt = npzfile['u_opt']

# Load initial conditions as well
q_opt = npzfile['q_opt']
q_opt_0 = q_opt[0, :]
print(q_opt_0)

# Create a control trajectory
u_opt_poly = PiecewisePolynomial.ZeroOrderHold(t, u_opt.T)
control_source = builder.AddSystem(TrajectorySource(u_opt_poly))

# Close the loop by connecting the output of the controller to the plant
builder.Connect(control_source.get_output_port(0),
                pancake_flipper.get_actuation_input_port())

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

# start recording the video for the animation of the simulation
visualizer.start_recording()

# set up a simulator
simulator = Simulator(diagram)

context = simulator.get_mutable_context()
context.SetTime(0.0)  # reset current time
context.SetContinuousState((
    q_opt_0[0], q_opt_0[1],
    q_opt_0[2], q_opt_0[3],
    q_opt_0[4], q_opt_0[5],
    0, 0,
    0, 0,
    0, 0
))

# generate and display animation
visualizer.start_recording()
simulator.AdvanceTo(u_opt_poly.end_time())
ani = visualizer.get_recording_as_animation()

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Charles Dawson'), bitrate=1800)
stamp = datafile[8:-4]
ani.save('results/openloop_zoh_' + stamp + '.mp4', writer=writer)
