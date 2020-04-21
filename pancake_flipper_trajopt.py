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
    Multiplexer, MatrixGain, LogOutput, plot_system_graphviz
)

from utils import ManipulatorDynamics


# Record the *half* extents of the flipper and pancake geometry
FLIPPER_H = 0.1
FLIPPER_W = 0.75
PANCAKE_H = 0.05
PANCAKE_W = 0.25


def collision_guards(q):
    '''Given the state of the flipper and pancake, returns a vector of
    signed distances from each corner of the pancake to the top surface
    of the flipper.

    @input q: 1x6 np.array [x_f, x_p, y_f, y_p, theta_f, theta_p]
    @output phi: 1x4 np.array of signed distances from the top surface
                 of the flipper to the bottom left, bottom right, top
                 right, and top left corners of the pancake. Positive
                 when no contact, zero at contact.
    '''
    # Extract state for flipper and pancake as well
    q_f = np.array([q[0], q[2], q[4]])
    q_p = np.array([q[1], q[3], q[5]])

    # For convenience, extract state further
    x_f = q_f[:2]                   # 2D location of flipper
    x_p = q_p[:2]                   # 2D location of pancake
    theta_f = q_f[2]                # Angle of flipper
    theta_p = q_p[2]                # Angle of pancake
    c_theta_f = np.cos(theta_f)     # Trig functions, for convenience
    s_theta_f = np.sin(theta_f)
    c_theta_p = np.cos(theta_p)
    s_theta_p = np.sin(theta_p)

    # Now get the unit vectors normal to the pancake and flipper
    # and the unit normal tangent to the flipper
    n_f = np.array([s_theta_f, c_theta_f])  # normal to flipper
    n_p = np.array([s_theta_p, c_theta_p])  # normal to pancake
    t_p = np.array([c_theta_p, -s_theta_p])  # tangent to pancake

    # We need to make sure the flipper normal is pointing towards the
    # pancake, so keep track of whether we need to flip the sign.
    n_f_sign = 1
    if n_f.dot(x_p - x_f) < 0:
        n_f_sign = -1

    # Now calculate the positions of all four corners of the pancake
    #
    #       d---------c
    #       |         |
    #       |         |
    #       a---------b
    x_a = x_p - PANCAKE_H * n_p - PANCAKE_W * t_p
    x_b = x_p - PANCAKE_H * n_p + PANCAKE_W * t_p
    x_c = x_p + PANCAKE_H * n_p + PANCAKE_W * t_p
    x_d = x_p + PANCAKE_H * n_p - PANCAKE_W * t_p

    # Now calculate the guards (signed distance from each corner of the
    # pancake to the flipper)
    phi_a = n_f_sign * n_f.transpose().dot(x_a - x_f) - FLIPPER_H
    phi_b = n_f_sign * n_f.transpose().dot(x_b - x_f) - FLIPPER_H
    phi_c = n_f_sign * n_f.transpose().dot(x_c - x_f) - FLIPPER_H
    phi_d = n_f_sign * n_f.transpose().dot(x_d - x_f) - FLIPPER_H

    # Return the signed distances
    return np.array([phi_a, phi_b, phi_c, phi_d])


class DropController(VectorSystem):

    def __init__(self, pancake_flipper):
        # 12 inputs: state of robot and pancake (x, y, and theta for each)
        # 3 outputs: x_ddot, y_ddot, and theta_ddot for the robot
        VectorSystem.__init__(self, 12, 3)
        self.pancake_flipper = pancake_flipper
        self.phi_b = 100

        # Record the *half* extents of the flipper and pancake geometry
        self.FLIPPER_H = 0.1
        self.FLIPPER_W = 0.75
        self.PANCAKE_H = 0.05
        self.PANCAKE_W = 0.25

    def DoCalcVectorOutput(
            self,
            context,
            controller_input,  # state of robot flipper and pancake
            controller_state,  # unused input (static controller)
            controller_output):  # robot flipper acceleration

        # unpack state
        q = controller_input[:6]
        q_dot = controller_input[6:]  # time derivative of q

        # Also extract state for flipper and pancake as well
        q_f = np.array([q[0], q[2], q[4]])
        q_dot_f = np.array([q_dot[0], q_dot[2], q_dot[4]])

        q_p = np.array([q[1], q[3], q[5]])
        q_dot_p = np.array([q_dot[1], q_dot[3], q_dot[5]])

        # extract manipulator equations: M*a + Cv = tauG + B*u + tauExt
        M, Cv, tauG, B, tauExt = ManipulatorDynamics(self.pancake_flipper,
                                                     q, q_dot)

        # Consolidate gravitational and coriolis forces
        tau = tauG - Cv

        # Check the contact guard conditions (this requires some setup first)
        x_f = q_f[:2]
        x_p = q_p[:2]
        theta_f = q_f[2]
        theta_p = q_p[2]
        c_theta_f = np.cos(theta_f)
        s_theta_f = np.sin(theta_f)
        c_theta_p = np.cos(theta_p)
        s_theta_p = np.sin(theta_p)

        # Now get the unit vectors normal to the pancake and flipper
        # and the unit normal tangent to the flipper
        n_f = np.array([s_theta_f, c_theta_f])  # normal to flipper
        n_p = np.array([s_theta_p, c_theta_p])  # normal to pancake
        t_p = np.array([c_theta_p, -s_theta_p])  # tangent to pancake

        # We need to make sure the flipper normal is pointing towards the
        # pancake
        n_f_sign = 1
        if n_f.dot(x_p - x_f) < 0:
            n_f_sign = -1

        # Now calculate the positions of all four corners of the pancake
        #
        #       d---------c
        #       |         |
        #       |         |
        #       a---------b
        x_a = x_p - self.PANCAKE_H * n_p - self.PANCAKE_W * t_p
        x_b = x_p - self.PANCAKE_H * n_p + self.PANCAKE_W * t_p
        x_c = x_p + self.PANCAKE_H * n_p + self.PANCAKE_W * t_p
        x_d = x_p + self.PANCAKE_H * n_p - self.PANCAKE_W * t_p

        # Now calculate the guards (signed distance from each corner of the
        # pancake to the flipper)
        phi_a = n_f_sign * n_f.transpose().dot(x_a - x_f) - self.FLIPPER_H
        phi_b = n_f_sign * n_f.transpose().dot(x_b - x_f) - self.FLIPPER_H
        phi_c = n_f_sign * n_f.transpose().dot(x_c - x_f) - self.FLIPPER_H
        phi_d = n_f_sign * n_f.transpose().dot(x_d - x_f) - self.FLIPPER_H

        # We would also like the Jacobian of each signed distance for use
        # in the manipulator equations:
        #
        #       M q_ddot = tau + J_phi^T [lambda_a, lambda_b, ...]^T
        #
        # J_phi^T = [ \nabla phi_a ^T, \nabla phi_b ^T, ... ]
        #
        # \nabla phi_i = (x_i-x_f)^T(\nabla n_f) + n_f^T(\nabla x_i-\nabla x_f)
        # (\nabla = gradient)
        nabla_n_f = np.array([[0, 0, c_theta_f, 0, 0, 0],
                              [0, 0, -s_theta_f, 0, 0, 0]])
        nabla_x_f = np.array([[1, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0]])

        h, w = (self.PANCAKE_H, self.PANCAKE_W)
        nabla_x_a = np.array([[0, 0, 0, 1, 0, -h * c_theta_p + w * s_theta_p],
                              [0, 0, 0, 0, 1, h * s_theta_p + w * c_theta_p]])
        nabla_x_b = np.array([[0, 0, 0, 1, 0, -h * c_theta_p - w * s_theta_p],
                              [0, 0, 0, 0, 1, h * s_theta_p - w * c_theta_p]])
        nabla_x_c = np.array([[0, 0, 0, 1, 0, h * c_theta_p - w * s_theta_p],
                              [0, 0, 0, 0, 1, -h * s_theta_p - w * c_theta_p]])
        nabla_x_d = np.array([[0, 0, 0, 1, 0, h * c_theta_p + w * s_theta_p],
                              [0, 0, 0, 0, 1, -h * s_theta_p + w * c_theta_p]])

        nabla_phi_a = (x_a - x_f).transpose().dot(nabla_n_f) \
            + n_f.transpose().dot(nabla_x_a - nabla_x_f)
        nabla_phi_b = (x_b - x_f).transpose().dot(nabla_n_f) \
            + n_f.transpose().dot(nabla_x_b - nabla_x_f)
        nabla_phi_c = (x_c - x_f).transpose().dot(nabla_n_f) \
            + n_f.transpose().dot(nabla_x_c - nabla_x_f)
        nabla_phi_d = (x_d - x_f).transpose().dot(nabla_n_f) \
            + n_f.transpose().dot(nabla_x_d - nabla_x_f)

        J_phi_T = np.hstack((nabla_phi_a.transpose(),
                             nabla_phi_b.transpose(),
                             nabla_phi_c.transpose(),
                             nabla_phi_d.transpose()))

        # control signal
        # Dummy controller just counteracts gravity and coriolis
        controller_output[:3] = -100*q_f - np.array([tau[0], tau[2], tau[4]])
        # controller_output[2] = 0


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
