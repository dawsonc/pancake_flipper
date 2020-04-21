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
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser,
    MathematicalProgram, eq, ge,
    JacobianWrtVariable
)

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

# Create the optimization problem (based on UR HW 5)

# The number of time steps in the trajectory optimization
T = 50

# The minimum and maximum time interval is seconds
h_min = .005
h_max = .05

# Initialize the optimization program
prog = MathematicalProgram()

# Define the vector of the time intervals as a decision variable
# (distances between the T + 1 break points)
h = prog.NewContinuousVariables(T, name='h')

nq = 6  # number of states
# Define decision variables for the system configuration, generalized
# velocities, and accelerations
q = prog.NewContinuousVariables(rows=T + 1, cols=nq, name='q')
qdot = prog.NewContinuousVariables(rows=T + 1, cols=nq, name='qdot')
qddot = prog.NewContinuousVariables(rows=T, cols=nq, name='qddot')

# Also define decision variables for the contact forces (x and z in flipper
# frame) at each corner of the pancake at each timestep
n_forces = 2  # x and z
f_ll = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ll')
f_lr = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_lr')
f_ur = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ur')
f_ul = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ul')

# Now we get to the fun part: defining our constraints!

# Add a bounding box on the length of individual timesteps
prog.AddBoundingBoxConstraint([h_min] * T, [h_max] * T, h)

# Using the implicit Euler method, constrain the configuration,
# velocity, and accelerations to be self-consistent.
# Note: we're not worried about satisfying the dynamics here, just
# making sure that the velocity is the rate of change of configuration,
# and likewise for acceleration and velocity
for t in range(T):
    prog.AddConstraint(eq(q[t + 1], q[t] + h[t] * qdot[t + 1]))
    prog.AddConstraint(eq(qdot[t + 1], qdot[t] + h[t] * qddot[t + 1]))

# Now we add the constraint enforcing the manipulator dynamics at all timesteps
for t in range(T):
    # Assemble the relevant variables:
    #   - configuration
    #   - velocity
    #   - acceleration
    #   - contact forces (ll, lr, ur, ul)
    vars = np.concatenate((q[t + 1], qdot[t + 1], qddot[t],
                           f_ll[t], f_lr[t], f_ur[t], f_ul[t]))
    prog.AddConstraint(manipulator_equations,
                       lb=[0] * nq,
                       ub=[0] * nq,
                       vars=vars)

# Now we have the (only somewhat onerous) task of constraining the contact
# forces at all timesteps. More specifically, we need to constrain:
#   1.) The contact guards are nonnegative
#   2.) Contact forces in the z direction are nonnegative
#   3.) Contact forces in the x direction are within the friction cone
#   4.) Contact forces in z are complementary with the contact guards
#   5.) Sliding only occurs when contact forces are at the limit of the
#       friction cone (i.e. the friction cone constraint and tangential
#       velocity of the contact point are complementary)

# TODO: Add decision variables for fx+ and fx- for each corner
# TODO: Add linear inequality enforcing the friction cone constraint
# TODO: write function to enforce the nonlinear complementarity constraints

# TODO: Add initial guess for timesteps
# TODO: Add initial guess for trajectory (how to make a feasible one?)
#           Maybe just up, flip, down?
# TODO: Add initial guess for forces (0?)

# TODO: Solve?
# TODO: Hope it works?
# TODO: Delete previous question marks!

# TODO: Extract optimal solution
# TODO: Animate
