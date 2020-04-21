#!/usr/bin/python3
'''
Much of this code is patterned on that in the vibrating_pendulum.ipynb file
from the Underactuated Robotics homeworks.

Written by C. Dawson (cbd@mit.edu)
'''
# Python libraries
import numpy as np
import matplotlib.pyplot as plt

# PyDrake imports
import pydrake
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser,
    PlanarSceneGraphVisualizer, Simulator, VectorSystem,
    JacobianWrtVariable
)

from utils import ManipulatorDynamics


# Record the *half* extents of the flipper and pancake geometry
FLIPPER_H = 0.1
FLIPPER_W = 0.75
PANCAKE_H = 0.05
PANCAKE_W = 0.25


def get_pancake_corners():
    '''Returns a list of the (x, y, z) coordinates of the four corners
    of the pancake, expressed in the pancake frame

    @returns: a list of 4 1x3 np.arrays, containing the lower left, lower
              right, upper right, and upper left corner coordinates.
    '''
    # Calculate the positions of all four corners of the pancake in the
    # pancake frame
    #
    #       d---------c
    #       |         |
    #       |         |
    #       a---------b
    x_a = np.array([-PANCAKE_W, 0, -PANCAKE_H])
    x_b = np.array([PANCAKE_W, 0, -PANCAKE_H])
    x_c = np.array([PANCAKE_W, 0, PANCAKE_H])
    x_d = np.array([-PANCAKE_W, 0, PANCAKE_H])

    return [x_a, x_b, x_c, x_d]


def manipulator_equations(vars, pancake_flipper):
    '''Returns a vector that must vanish in order to enforce dynamics
    consistent with the manipulator equations for the pancake flipper.
    Based on the code provided in compass_gait_limit_cycle.ipynb (Assignment 5)

    @param vars: the concatenation of
                    - pancake flipper configuration
                    - configuration velocity
                    - configuration acceleration
                    - contact forces (x and z) at each corner of the pancake,
                      arranged as [lower left, lower right, upper right,
                      upper left]
    @param pancake_flipper: a MultibodyPlant representing the pancake flipper
    @returns: a vector whose components must go to zero in order for the
              dynamics to be consistent with the manipulator equations
    '''
    # Split vars into subvariables
    n_states = 6
    n_corners = 4
    n_forces = 2
    assert vars.size == 3 * n_states + n_corners * n_forces
    split_at_indices = [n_states, 2 * n_states, 3 * n_states]
    q, qdot, qddot, f = np.split(vars, split_at_indices)
    split_at_indices = [n_forces, 2 * n_forces, 3 * n_forces]
    f_ll, f_lr, f_ur, f_ul = np.split(f, split_at_indices)

    # Set plant state
    context = pancake_flipper.CreateDefaultContext()
    pancake_flipper.SetPositions(context, q)
    pancake_flipper.SetVelocities(context, qdot)

    # Calculate the manipulator equation matrices
    M = pancake_flipper.CalcMassMatrixViaInverseDynamics(context)
    Cv = pancake_flipper.CalcBiasTerm(context)
    tauG = pancake_flipper.CalcGravityGeneralizedForces(context)

    # Calculate the contact Jacobian for each corner
    J_ll = get_pancake_corner_jacobian(pancake_flipper, context, 0)
    J_lr = get_pancake_corner_jacobian(pancake_flipper, context, 1)
    J_ur = get_pancake_corner_jacobian(pancake_flipper, context, 2)
    J_ul = get_pancake_corner_jacobian(pancake_flipper, context, 3)

    # Return the violation of the manipulator equation
    return M.dot(qddot) + Cv - tauG - J_ll.T.dot(f_ll) \
                                    - J_lr.T.dot(f_lr) \
                                    - J_ur.T.dot(f_ur) \
                                    - J_ul.T.dot(f_ul)


def collision_guards(pancake_flipper, q):
    '''Given the state of the flipper and pancake, returns a vector of
    signed distances from each corner of the pancake to the top surface
    of the flipper.

    @param pancake_flipper: a MultibodyPlant representing the pancake flipper
    @param q: 1x6 np.array [x_f, x_p, y_f, y_p, theta_f, theta_p]
    @returns phi: 1x4 np.array of signed distances from the top surface
                 of the flipper to the bottom left, bottom right, top
                 right, and top left corners of the pancake. Positive
                 when no contact, zero at contact.
    '''
    # Get the frames for the flipper and pancake
    pancake_frame = pancake_flipper.GetBodyByName("pancake").body_frame()
    flipper_frame = pancake_flipper.GetBodyByName("flipper").body_frame()

    # Now get the positions of all four corners of the pancake in the
    # pancake frame
    #
    #       d---------c
    #       |         |
    #       |         |
    #       a---------b
    x_a, x_b, x_c, x_d = get_pancake_corners()

    # Transform each corner point into the flipper frame
    context = pancake_flipper.CreateDefaultContext()
    pancake_flipper.SetPositions(context, q)
    x_a_flipper_frame = pancake_flipper.CalcPointsPosition(
        context,
        pancake_frame,
        x_a,
        flipper_frame
    )
    x_b_flipper_frame = pancake_flipper.CalcPointsPosition(
        context,
        pancake_frame,
        x_b,
        flipper_frame
    )
    x_c_flipper_frame = pancake_flipper.CalcPointsPosition(
        context,
        pancake_frame,
        x_c,
        flipper_frame
    )
    x_d_flipper_frame = pancake_flipper.CalcPointsPosition(
        context,
        pancake_frame,
        x_d,
        flipper_frame
    )

    # We assume that the flipper is infinitely wide, so the signed
    # distance from each corner to the flipper is simply its z coordinate
    # in the flipper frame offset by the half height of the flipper
    # TODO: relax this assumption
    phi_a = x_a_flipper_frame[2] - np.sign(x_a_flipper_frame[2]) * FLIPPER_H
    phi_b = x_b_flipper_frame[2] - np.sign(x_b_flipper_frame[2]) * FLIPPER_H
    phi_c = x_c_flipper_frame[2] - np.sign(x_c_flipper_frame[2]) * FLIPPER_H
    phi_d = x_d_flipper_frame[2] - np.sign(x_d_flipper_frame[2]) * FLIPPER_H

    # Return the signed distances
    return np.array([phi_a, phi_b, phi_c, phi_d])


def reset_velocity_on_corner_impact(vars, pancake_flipper, corner_idx):
    '''Returns a vector that must vanish in order to enforce an impulsive
    collision between the specified corner and the flipper. Based on the
    code provided in compass_gait_limit_cycle.ipynb (Assignment 5).

    @param vars: the concatenation of
                    - pancake flipper configuration
                    - configuration velocities before and after the impact
                    - the impulse (in x and z) delivered to the pancake by
                      the flipper, transmitted through the corner
    @param pancake_flipper: a MultibodyPlant representing the pancake flipper
    @param corner_idx: an integer index into the corners of the pancake,
                       arranged as [lower left, lower right, upper right,
                       upper left]
    @returns: a vector whose components must go to zero in order for the
              collision to be inelastic and the velocity jump to be consistent
              with the impulse delivered.
    '''
    # Extract the subvariables from the input vector
    assert vars.size == 20   # 3*6 states + 2 impulses
    q = vars[:6]             # configuration
    qdot_pre = vars[6:12]    # configuration velocity before impact
    qdot_post = vars[12:18]  # configuration velocity after impact
    impulse = vars[18:]      # (x, z) impulse delivered to pancake

    # Set the configuration
    context = pancake_flipper.CreateDefaultContext()
    pancake_flipper.SetPositions(context, q)

    # Get the mass matrix and Jacobian for this corner
    M = pancake_flipper.CalcMassMatrixViaInverseDynamics(context)
    J = get_pancake_corner_jacobian(pancake_flipper, context, corner_idx)

    # This vector must vanish in order for the impact to be valid
    arrest_x_motion = 0  # set to 1 to zero the x velocity on impact
    arrest_motion = np.array([[arrest_x_motion, 0], [0, 1]])  # masks x
    return np.concatenate((
        M.dot(qdot_post - qdot_pre) - J.T.dot(impulse),  # momentum conserved
        arrest_motion.dot(J.dot(qdot_pre))               # stick the landing
    ))


def get_pancake_corner_jacobian(pancake_flipper, context, corner_idx):
    '''Returns the Jacobian of the specified corner of the pancake in the
    flipper frame.

    @param pancake_flipper: a MultibodyPlant representing the pancake flipper
    @param context: the drake context specifying the state of the plant
    @param corner_idx: an integer index into the corners of the pancake,
                       arranged as [lower left, lower right, upper right,
                       upper left]
    @returns: the Jacobian of the translational position of the specified
              corner of the pancake in the flipper frame, as a 2x6 matrix
              where the first row represents the gradient of the x position
              w.r.t. the state, and the second row represents the gradient of
              the z position w.r.t. the state.
    '''
    # Get the frames for the flipper and pancake
    pancake_frame = pancake_flipper.GetBodyByName("pancake").body_frame()
    flipper_frame = pancake_flipper.GetBodyByName("flipper").body_frame()

    # Now get the positions of all four corners of the pancake in the
    # pancake frame
    #
    #       d---------c
    #       |         |
    #       |         |
    #       a---------b
    corners_list = get_pancake_corners()

    # Select the one we want
    corner = corners_list[corner_idx]
    # Calculate the Jacobian using the plant
    corner_jacobian = pancake_flipper.CalcJacobianTranslationVelocity(
        context,
        JacobianWrtVariable(0),
        pancake_frame,
        corner,
        flipper_frame,
        flipper_frame
    )

    # Discard the y component since we're in 2D
    corner_jacobian = corner_jacobian[[0, 2]]


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

    # parse the urdf and populate the vibrating pendulum
    urdf_path = './models/pancake_flipper_massless_links.urdf'
    Parser(pancake_flipper).AddModelFromFile(urdf_path)

    # Finalize the plant so we can use it
    pancake_flipper.Finalize()

    # Return the plant and scene graph, as promised
    return pancake_flipper, scene_graph


# Create the builder to house our system
builder = DiagramBuilder()
# Create the plant and scene graph
pancake_flipper, scene_graph = build_pancake_flipper_plant(builder)

# Instantiate the controller
controller = builder.AddSystem(DropController(pancake_flipper))

# Connect the state output from the plant to the controller
builder.Connect(pancake_flipper.get_state_output_port(),
                controller.get_input_port(0))

# Close the loop by connecting the output of the controller to the plant
builder.Connect(controller.get_output_port(0),
                pancake_flipper.get_actuation_input_port())

# Add a visualizer so we can see what's going on
visualizer = builder.AddSystem(
    PlanarSceneGraphVisualizer(scene_graph, xlim=[-4., 4.], ylim=[-4., 4.]))

# Connect the output of the scene graph to the visualizer
builder.Connect(scene_graph.get_pose_bundle_output_port(),
                visualizer.get_input_port(0))

# Build the diagram!
diagram = builder.Build()

# start recording the video for the animation of the simulation
visualizer.start_recording()

# set up a simulator
simulator = Simulator(diagram)
simulator.set_publish_every_time_step(False)

# We should be fine with the default initial conditions, so don't change them
context = simulator.get_mutable_context()
context.SetTime(0.0)  # reset current time
context.SetContinuousState((
    0, 0.2,
    0, 0.5,
    0, 0.4,
    0, 0,
    0, 0,
    0, 0
))

# simulate from zero to sim_time
simulator.Initialize()
sim_time = 5
simulator.AdvanceTo(sim_time)

# stop the video and build the animation
visualizer.stop_recording()
