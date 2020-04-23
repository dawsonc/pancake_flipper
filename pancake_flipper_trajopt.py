#!/usr/bin/python3
'''
Much of this code is patterned on that in the vibrating_pendulum.ipynb file
from the Underactuated Robotics homeworks.

Written by C. Dawson (cbd@mit.edu)
'''
# Python libraries
import numpy as np

# PyDrake imports
from pydrake.all import (
    AddMultibodyPlantSceneGraph, DiagramBuilder, Parser,
    MathematicalProgram, SnoptSolver, eq, ge,
    JacobianWrtVariable,
    PiecewisePolynomial,
    TrajectorySource, MultibodyPositionToGeometryPose,
    PlanarSceneGraphVisualizer, Simulator,
    MultibodyPlant, SceneGraph
)

import time
import matplotlib.animation as animation

# Record the *half* extents of the flipper and pancake geometry
FLIPPER_H = 0.1
FLIPPER_W = 0.75
PANCAKE_H = 0.05
PANCAKE_W = 0.25

# Friction coefficient between pan and cake (0 for the well-buttered case)
MU = 0.0


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


def manipulator_equations(pancake_flipper, vars):
    '''Returns a vector that must vanish in order to enforce dynamics
    consistent with the manipulator equations for the pancake flipper.
    Based on the code provided in compass_gait_limit_cycle.ipynb (Assignment 5)

    @param vars: the concatenation of
                    - pancake flipper configuration
                    - configuration velocity
                    - configuration acceleration
                    - generalized forces
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
    assert vars.size == 4 * n_states + n_corners * n_forces
    split_at_indices = [n_states, 2 * n_states, 3 * n_states, 4 * n_states]
    q, qdot, qddot, u, f = np.split(vars, split_at_indices)
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
    B = np.eye(n_states)

    # Calculate the contact Jacobian for each corner
    J_ll = get_pancake_corner_jacobian(pancake_flipper, context, 0)
    J_lr = get_pancake_corner_jacobian(pancake_flipper, context, 1)
    J_ur = get_pancake_corner_jacobian(pancake_flipper, context, 2)
    J_ul = get_pancake_corner_jacobian(pancake_flipper, context, 3)

    # Return the violation of the manipulator equation
    return M.dot(qddot) + Cv - tauG - B.dot(u) - J_ll.T.dot(f_ll) \
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
    x_a_flipper_frame = pancake_flipper.CalcPointsPositions(
        context,
        pancake_frame,
        x_a,
        flipper_frame
    )
    x_b_flipper_frame = pancake_flipper.CalcPointsPositions(
        context,
        pancake_frame,
        x_b,
        flipper_frame
    )
    x_c_flipper_frame = pancake_flipper.CalcPointsPositions(
        context,
        pancake_frame,
        x_c,
        flipper_frame
    )
    x_d_flipper_frame = pancake_flipper.CalcPointsPositions(
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


def guard_force_complementarity(pancake_flipper, vars):
    '''Returns the inner product of the collision guards with the contact
    forces in the z direction.

    @param vars: the concatenation of
                    - pancake flipper configuration
                    - contact forces (x and z) at each corner of the pancake,
                      arranged as [lower left, lower right, upper right,
                      upper left]
    @param pancake_flipper: a MultibodyPlant representing the pancake flipper
    @returns the inner product of the four contact guards and the z components
             of the four forces.
    '''
    # Split vars into subvariables
    n_states = 6
    n_corners = 4
    n_forces = 2
    assert vars.size == n_states + n_corners * n_forces
    split_at_indices = [n_states]
    q, f = np.split(vars, split_at_indices)
    split_at_indices = [n_forces, 2 * n_forces, 3 * n_forces]
    f_ll, f_lr, f_ur, f_ul = np.split(f, split_at_indices)

    # Extract z components of forces
    f_ll_z = f_ll[1]
    f_lr_z = f_lr[1]
    f_ur_z = f_ur[1]
    f_ul_z = f_ul[1]
    # and assemble them into an np array
    fz = np.array([f_ll_z, f_lr_z, f_ur_z, f_ul_z])

    # Compute contact guards
    guards = collision_guards(pancake_flipper, q)

    # Return the inner product of the two (which must be zero to enforce
    # complementarity)
    return guards.T.dot(fz)


def calc_tangential_velocities(pancake_flipper, q, qdot):
    '''Compute the tangential velocity of each corner of the pancake relative
    to the flipper surface.

    @param pancake_flipper: the MultibodyPlant representing the pancake flipper
    @param q: the configuration of the pancake flipper
    @param qdot: the configuration velocity of the pancake flipper
    @returns a vector of the 4 tangential velocities (one for each corner,
             ordered as [lower left, lower right, upper right, upper left])
    '''
    # Set plant state
    context = pancake_flipper.CreateDefaultContext()
    pancake_flipper.SetPositions(context, q)
    pancake_flipper.SetVelocities(context, qdot)

    # Calculate the contact Jacobian for each corner
    J_ll = get_pancake_corner_jacobian(pancake_flipper, context, 0)
    J_lr = get_pancake_corner_jacobian(pancake_flipper, context, 1)
    J_ur = get_pancake_corner_jacobian(pancake_flipper, context, 2)
    J_ul = get_pancake_corner_jacobian(pancake_flipper, context, 3)

    # Compute tangential velocity as J*qdot
    v_tangent_ll = J_ll.dot(qdot)
    v_tangent_lr = J_lr.dot(qdot)
    v_tangent_ur = J_ur.dot(qdot)
    v_tangent_ul = J_ul.dot(qdot)

    # Extract the components tangent to the flipper surface (i.e. the local
    # x coordinate)
    tangential_velocities = np.array([
        v_tangent_ll[0], v_tangent_lr[0], v_tangent_ur[0], v_tangent_ul[0]])

    return tangential_velocities


def gamma_vs_abs_tangential_velocity(pancake_flipper, vars):
    '''Returns a vector whose components should be nonnegative in order to
    enforce the constraint that the gamma slack variables are greater than
    the magnitude of the tangential velocity of each corner point on the
    pancake (expressed in the flipper frame).

    @param pancake_flipper: a MultibodyPlant representing the pancake flipper
    @param vars: the concatenation of
                    - pancake flipper configuration
                    - pancake flipper configuration velocity
                    - gamma (scalar) at each corner of the pancake,
                      arranged as [lower left, lower right, upper right,
                      upper left]
    @returns a vector of which all components should be constrained to be
             nonnegative
    '''
    # Split vars into subvariables
    n_states = 6
    n_corners = 4
    assert vars.size == 2 * n_states + n_corners
    split_at_indices = [n_states, 2 * n_states]
    q, qdot, gamma = np.split(vars, split_at_indices)

    # Get the tangential velocities
    tangent_velocities = calc_tangential_velocities(pancake_flipper, q, qdot)

    # We want to constrain gamma to be greater than tangent_velocities AND
    # greater than - tangent_velocities (i.e. greater than the magnitude), so
    # we return a vector concatenating these two
    return np.concatenate((gamma - tangent_velocities,
                           gamma + tangent_velocities))


def friction_cone_complementarity(pancake_flipper, vars):
    '''Returns a vector whose components should vanish to ensure that sliding
    only occurs when the contact forces are at the edge of the friction cone.

    @param vars: the concatenation of
                    - pancake flipper configuration
                    - pancake flipper velocity
                    - gamma (scalar) at each corner of the pancake,
                      arranged as [lower left, lower right, upper right,
                      upper left]
                    - contact forces (x and z) at each corner of the pancake,
                      arranged as [lower left, lower right, upper right,
                      upper left]
                    - x-direction contact force components (+ and -) at each
                      corner of the pancake, arranged as [lower left,
                      lower right, upper right, upper left]
    @param pancake_flipper: a MultibodyPlant representing the pancake flipper
    @returns a vector whose components must be zero in order for the solution
             to be dynamically valid.
    '''
    # Split vars into subvariables
    n_states = 6
    n_corners = 4
    n_forces = 2
    assert vars.size == 2 * n_states + n_corners + 2 * n_corners * n_forces
    split_at_indices = [n_states, 2 * n_states]
    q, qdot, rest = np.split(vars, split_at_indices)
    split_at_indices = [n_corners]
    gamma, rest = np.split(vars, split_at_indices)
    split_at_indices = [n_forces, 2 * n_forces, 3 * n_forces, 4 * n_forces]
    f_ll, f_lr, f_ur, f_ul, rest = np.split(vars, split_at_indices)
    split_at_indices = [n_forces, 2 * n_forces, 3 * n_forces]
    f_ll_x, f_lr_x, f_ur_x, f_ul_x = np.split(vars, split_at_indices)

    # Extract the z components of the contact force
    f_z = np.array([f_ll[1], f_lr[1], f_ur[1], f_ul[1]])

    # Reformulate positive and negative components of the x component of the
    # contact forces for easier use later
    f_x_pos = np.array([f_ll_x[0], f_lr_x[0], f_ur_x[0], f_ul_x[0]])
    f_x_neg = np.array([f_ll_x[1], f_lr_x[1], f_ur_x[1], f_ul_x[1]])

    # Compute the inner product of the friction cone constraint with the
    # tangential velocity magnitude
    complementarity_friction_cone = np.multiply((MU * f_z - f_x_pos - f_x_neg),
                                                gamma)

    # Get the tangential velocities
    tangent_velocities = calc_tangential_velocities(pancake_flipper, q, qdot)

    # Compute complementarity between the positive x contact force and
    # (gamma + tangent_velocities), so that we only get friction in the +x
    # direction if we're sliding in the -x direction.
    complementarity_sliding_negx = np.multiply((gamma + tangent_velocities),
                                               f_x_pos)
    # Likewise for the negative x contact force
    complementarity_sliding_posx = np.multiply((gamma - tangent_velocities),
                                               f_x_neg)

    # Return a vector containing these constraints. All components must go to
    # zero in order for the constraints to be enforced.
    return np.concatenate([complementarity_friction_cone,
                           complementarity_sliding_negx,
                           complementarity_sliding_posx])


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
    corner_jacobian = pancake_flipper.CalcJacobianTranslationalVelocity(
        context,
        JacobianWrtVariable(0),
        pancake_frame,
        corner,
        flipper_frame,
        flipper_frame
    )

    # Discard the y component since we're in 2D
    corner_jacobian = corner_jacobian[[0, 2]]
    return corner_jacobian


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

# Overwrite the original MultibodyPlant with its autodiff copy
pancake_flipper = pancake_flipper.ToAutoDiffXd()

# Create the optimization problem (based on UR HW 5)

# The number of time steps in the trajectory optimization
T = 100

# The minimum and maximum time interval is seconds
h_min = 0.0005
h_max = 0.05

# Initialize the optimization program
prog = MathematicalProgram()

# Define the vector of the time intervals as a decision variable
# (distances between the T + 1 break points)
h = prog.NewContinuousVariables(T, name='h')

num_states = 6
# Define decision variables for the system configuration, generalized
# velocities, and accelerations
q = prog.NewContinuousVariables(rows=T + 1, cols=num_states, name='q')
qdot = prog.NewContinuousVariables(rows=T + 1, cols=num_states, name='qdot')
qddot = prog.NewContinuousVariables(rows=T, cols=num_states, name='qddot')
# Control inputs: x, y, and theta for the flipper
u = prog.NewContinuousVariables(rows=T, cols=num_states, name='u')

# Also define decision variables for the contact forces (x and z in flipper
# frame) at each corner of the pancake at each timestep
n_forces = 2  # x and z
f_ll = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ll')
f_lr = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_lr')
f_ur = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ur')
f_ul = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ul')

# Now we get to the fun part: defining our constraints!

# Constrain our start and end states.
# Remember that because of how we've structured the URDF, the states are
# ordered q: 1x6 np.array [x_f, x_p, y_f, y_p, theta_f, theta_p]
start_state = np.array([0, 0.0, 0, 0.15, 0, 0])
final_state = np.array([0, 0.0, 0, 0.15, 0, -np.pi])

zeros = np.zeros(num_states)
# Constrain start state
prog.AddLinearConstraint(eq(q[0], start_state))
prog.AddLinearConstraint(eq(qdot[0], zeros))
prog.AddLinearConstraint(eq(qddot[0], zeros))
# Constrain end state
prog.AddLinearConstraint(eq(q[-1], final_state))
prog.AddLinearConstraint(eq(qdot[-1], zeros))
prog.AddLinearConstraint(eq(qddot[-1], zeros))


# Add a bounding box on the length of individual timesteps
prog.AddBoundingBoxConstraint([h_min] * T, [h_max] * T, h)
# Also constrain the time steps to form pairs of equal duration (per Posa13)
for j in range(int((T - 3) / 2)):
    prog.AddConstraint(
        h[2 * j - 1] + h[2 * j] == h[2 * j + 1] + h[2 * j + 2])

# We can only control x, y, and theta generalized forces on flipper,
# so we need to zero out everything else.
#
# The controls we do have are bounded, so add those constraints too
u_abs_max = 10
for t in range(T):
    prog.AddConstraint(u[t, 1] == 0)
    prog.AddConstraint(u[t, 3] == 0)
    prog.AddConstraint(u[t, 5] == 0)

    # prog.AddConstraint(u[t, 0] <= u_abs_max)
    # prog.AddConstraint(-u[t, 0] <= u_abs_max)
    # prog.AddConstraint(u[t, 2] <= u_abs_max)
    # prog.AddConstraint(-u[t, 2] <= u_abs_max)
    # prog.AddConstraint(u[t, 4] <= u_abs_max)
    # prog.AddConstraint(-u[t, 4] <= u_abs_max)

# Using the implicit Euler method, constrain the configuration,
# velocity, and accelerations to be self-consistent.
# Note: we're not worried about satisfying the dynamics here, just
# making sure that the velocity is the rate of change of configuration,
# and likewise for acceleration and velocity
for t in range(T):
    prog.AddConstraint(eq(q[t + 1], q[t] + h[t] * qdot[t + 1]))
    prog.AddConstraint(eq(qdot[t + 1], qdot[t] + h[t] * qddot[t]))

# Now we add the constraint enforcing the manipulator dynamics at all timesteps
for t in range(T):
    # Assemble the relevant variables:
    #   - configuration
    #   - velocity
    #   - acceleration
    #   - contact forces (ll, lr, ur, ul)
    vars = np.concatenate((q[t + 1], qdot[t + 1], qddot[t], u[t],
                           f_ll[t], f_lr[t], f_ur[t], f_ul[t]))
    # Make an anonymous function that passes the pancake_flipper plant along
    # with the assembled variables, and use that in the constraint
    prog.AddConstraint(
        lambda vars: manipulator_equations(pancake_flipper, vars),
        lb=[0] * num_states,
        ub=[0] * num_states,
        vars=vars
    )

# Now we have the (only somewhat onerous) task of constraining the contact
# forces at all timesteps. More specifically, we need to constrain:
#   1.) The contact guards are nonnegative
#   2.) Contact forces in the z direction are nonnegative
#   3.) Contact forces in the x direction are within the friction cone
#   4.) Contact forces in z are complementary with the contact guards
#   5.) Sliding only occurs when contact forces are at the limit of the
#       friction cone (i.e. the friction cone constraint and tangential
#       velocity of the contact point are complementary)

# 1.) Contact guards are nonnegative at all times
for t in range(T):
    prog.AddConstraint(lambda vars: collision_guards(pancake_flipper, vars),
                       lb=[0] * 4, ub=[np.inf] * 4, vars=q[t])

# To express the contact constraints, it is useful to break the tangential
# force into two trictly non-negative components (i.e. fx = fx^+ - fx^-).
# At time t, f_ll_x[t, 0] will be the contribution to f_ll in the +x direction,
# and f_ll_x[t, 1] will be the contribution in the -x direction,
f_ll_x = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ll_x')
f_lr_x = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_lr_x')
f_ur_x = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ur_x')
f_ul_x = prog.NewContinuousVariables(rows=T, cols=n_forces, name='f_ul_x')

# 2.), 3.), and 4.) for all timesteps
for t in range(T):
    # We need to constrain these relative to the total force in the x direction
    prog.AddLinearConstraint(f_ll_x[t, 0] - f_ll_x[t, 1] == f_ll[t, 0])
    prog.AddLinearConstraint(f_lr_x[t, 0] - f_lr_x[t, 1] == f_lr[t, 0])
    prog.AddLinearConstraint(f_ur_x[t, 0] - f_ur_x[t, 1] == f_ur[t, 0])
    prog.AddLinearConstraint(f_ul_x[t, 0] - f_ul_x[t, 1] == f_ul[t, 0])

    # 2.) The force in the z direction and each component in the +/- x
    # directions have to be nonnegative
    zeros = np.zeros(2)
    prog.AddLinearConstraint(f_ll[t, 1] >= 0)
    prog.AddLinearConstraint(ge(f_ll_x[t], zeros))

    prog.AddLinearConstraint(f_lr[t, 1] >= 0)
    prog.AddLinearConstraint(ge(f_lr_x[t], zeros))

    prog.AddLinearConstraint(f_ur[t, 1] >= 0)
    prog.AddLinearConstraint(ge(f_ur_x[t], zeros))

    prog.AddLinearConstraint(f_ul[t, 1] >= 0)
    prog.AddLinearConstraint(ge(f_ul_x[t], zeros))

    # 3.) Forces in the x direction fall within the friction cone
    prog.AddLinearConstraint(
        MU * f_ll[t, 1] - f_ll_x[t, 0] - f_ll_x[t, 1] >= 0)
    prog.AddLinearConstraint(
        MU * f_lr[t, 1] - f_lr_x[t, 0] - f_lr_x[t, 1] >= 0)
    prog.AddLinearConstraint(
        MU * f_ur[t, 1] - f_ur_x[t, 0] - f_ur_x[t, 1] >= 0)
    prog.AddLinearConstraint(
        MU * f_ul[t, 1] - f_ul_x[t, 0] - f_ul_x[t, 1] >= 0)

    # 4.) Contact guards are complimentary with contact force in z
    vars = np.concatenate((q[t], f_ll[t], f_lr[t], f_ur[t], f_ul[t]))
    prog.AddConstraint(
        lambda vars: guard_force_complementarity(pancake_flipper, vars),
        lb=[0], ub=[0], vars=vars
    )

# When we have friction, we need to enforce that sliding only occurs at the
# edge of the friction cone. To do this, it's helpful to add a slack variable
# for the magnitude of the tangential velocity of each corner
gamma = prog.NewContinuousVariables(rows=T, cols=4, name='gamma')

# Enforce 5.) for all timesteps
for t in range(T):
    # All gammas should be nonnegative
    zeros = np.zeros(4)
    prog.AddLinearConstraint(ge(gamma[t], zeros))

    # All gammas should be greater than the absolute value of the tangential
    # velocity of the relevant corner of the pancake (in the flipper frame)
    vars = np.concatenate((q[t], qdot[t], gamma[t]))
    prog.AddConstraint(
        lambda vars: gamma_vs_abs_tangential_velocity(pancake_flipper, vars),
        lb=[0] * 8, ub=[np.inf] * 8, vars=vars
    )

    # 5.) Sliding only occurs when we're at the edge of the friction cone
    # This means that gamma is complementary with the friction cone constraint,
    # with additional constraints to distinguish between being on the +x and -x
    # edge of the friction cone
    vars = np.concatenate((q[t], qdot[t], gamma[t],
                           f_ll[t], f_lr[t], f_ur[t], f_ul[t],
                           f_ll_x[t], f_lr_x[t], f_ur_x[t], f_ul_x[t]))
    # prog.AddConstraint(
    #     lambda vars: friction_cone_complementarity(pancake_flipper, vars),
    #     lb=[0] * 12, ub=[0] * 12, vars=vars
    # )

# Now we should have defined all of our constraints, so we just need to
# seed the solver with an initial guess

# Create a vector to store the initial guess
# Unless we explicitly set a guess here, our initial guess will just be zero
initial_guess = np.zeros(prog.num_vars())

# Add our initial guess for the time step: each timestep takes its maximum
# duration.
h_guess = h_max
prog.SetDecisionVariableValueInVector(h, [h_guess] * T, initial_guess)

# For now, we'll guess an initial trajectory that is just a linear
# interpolation of the configuration between the start and target states.
# This trajectory will be INFEASIBLE initiall, so we'll see if we need a more
# refined initial guess
q_guess_poly = PiecewisePolynomial.FirstOrderHold(
    [0, T * h_guess],
    np.column_stack((start_state, final_state))
)
qdot_guess_poly = q_guess_poly.derivative()
qddot_guess_poly = qdot_guess_poly.derivative()

# Set our initial guess for configuration, velocity, and acceleration based
# on the linear interpolation
q_guess = np.hstack([q_guess_poly.value(t * h_guess) for t in range(T + 1)]).T
qd_guess = np.hstack(
    [qdot_guess_poly.value(t * h_guess) for t in range(T + 1)]).T
qdd_guess = np.hstack(
    [qddot_guess_poly.value(t * h_guess) for t in range(T)]).T
prog.SetDecisionVariableValueInVector(q, q_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qdot, qd_guess, initial_guess)
prog.SetDecisionVariableValueInVector(qddot, qdd_guess, initial_guess)

# Solve the mathematical program with our initial guess
solver = SnoptSolver()
result = solver.Solve(prog, initial_guess)

# Check if a solution was found
assert result.is_success(), "Solution not found :("

# Extract the optimal solution
h_opt = result.GetSolution(h)
q_opt = result.GetSolution(q)
qd_opt = result.GetSolution(qdot)
qdd_opt = result.GetSolution(qddot)
u_opt = result.GetSolution(u)
f_ll_opt = result.GetSolution(f_ll)
f_lr_opt = result.GetSolution(f_lr)
f_ur_opt = result.GetSolution(f_ur)
f_ul_opt = result.GetSolution(f_ul)

# stack states
x_opt = np.hstack((q_opt, qd_opt))

# interpolate state values for animation
time_breaks_opt = np.array([sum(h_opt[:t]) for t in range(T + 1)])
x_opt_poly = PiecewisePolynomial.FirstOrderHold(time_breaks_opt, x_opt.T)

# Get a new plant and scene graph to animate
pancake_flipper = MultibodyPlant(time_step=0)
scene_graph = SceneGraph()
pancake_flipper.RegisterAsSourceForSceneGraph(scene_graph)
file_name = './models/pancake_flipper_massless_links.urdf'
Parser(pancake_flipper).AddModelFromFile(file_name)
pancake_flipper.Finalize()

# build block diagram and drive system state with
# the trajectory from the optimization problem
builder = DiagramBuilder()
source = builder.AddSystem(TrajectorySource(x_opt_poly))
builder.AddSystem(scene_graph)
pos_to_pose = builder.AddSystem(
    MultibodyPositionToGeometryPose(
        pancake_flipper, input_multibody_state=True))
builder.Connect(source.get_output_port(0), pos_to_pose.get_input_port())
builder.Connect(
    pos_to_pose.get_output_port(),
    scene_graph.get_source_pose_port(pancake_flipper.get_source_id()))

# add visualizer
xlim = [-3, 3.]
ylim = [-3, 7]
visualizer = builder.AddSystem(
    PlanarSceneGraphVisualizer(scene_graph, xlim=xlim, ylim=ylim))
builder.Connect(
    scene_graph.get_pose_bundle_output_port(), visualizer.get_input_port(0))
simulator = Simulator(builder.Build())

# generate and display animation
visualizer.start_recording()
simulator.AdvanceTo(x_opt_poly.end_time())
ani = visualizer.get_recording_as_animation()

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Charles Dawson'), bitrate=1800)
time_save = str(time.now())
ani.save('results/animation' + time_save + '.mp4', writer=writer)

np.savez('results/trace' + 'time_save' + '.npz',
         h_opt=h_opt,
         q_opt=q_opt,
         qd_opt=qd_opt,
         qdd_opt=qdd_opt,
         u_opt=u_opt,
         f_ll_opt=f_ll_opt,
         f_lr_opt=f_lr_opt,
         f_ur_opt=f_ur_opt,
         f_ul_opt=f_ul_opt)
