import argparse
import os
import sys

import numpy as np

from pydrake.all import (
    TrajectorySource, PiecewisePolynomial, DiagramBuilder,
    Parser, MultibodyPlant, SceneGraph, FindResourceOrThrow,
    MultibodyPositionToGeometryPose, Simulator,
    ConnectDrakeVisualizer
)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.manipulation.planner import (
    DifferentialInverseKinematicsParameters
)

from differential_ik import DifferentialIK


TIME_STEP = 0.01

# Create the builder to house our system
builder = DiagramBuilder()

# Instantiate the arm plant and the scene graph.
arm_plant = MultibodyPlant(time_step=TIME_STEP)
scene_graph = SceneGraph()
arm_plant.RegisterAsSourceForSceneGraph(scene_graph)
# parse the urdf and populate the plant
sdf_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
Parser(arm_plant).AddModelFromFile(sdf_path)

# Weld the base link of the arm to the world
joint_arm_root_frame = arm_plant.GetFrameByName("iiwa_link_0")
arm_plant.WeldFrames(
    arm_plant.world_frame(),
    joint_arm_root_frame,
    RigidTransform.Identity())

# Attach the pancake flipper
flipper_model_file = "./models/flipper_arm_scale.urdf"
Parser(arm_plant).AddModelFromFile(flipper_model_file)
# Weld the end effector to the pancake flipper
joint_arm_eef_frame = arm_plant.GetFrameByName("iiwa_link_7")
flipper_frame = arm_plant.GetFrameByName("flipper")
flipper_xform = RigidTransform.Identity()
flipper_xform.set_translation(np.array([0, 0, 0.05]))
arm_plant.WeldFrames(
    joint_arm_eef_frame,
    flipper_frame,
    flipper_xform)

# Finalize the plant so we can use it
arm_plant.Finalize()

# Also make a plant for the pancake
pancake_plant = MultibodyPlant(time_step=TIME_STEP)
pancake_plant.RegisterAsSourceForSceneGraph(scene_graph)
pancake_model_file = "./models/pancake_arm_scale.urdf"
Parser(pancake_plant).AddModelFromFile(pancake_model_file)
pancake_plant.Finalize()

# Load the trajectory from the save file
datafile = 'results/test_slide_trace_umax40_ceiling5_mu0.0_T60.npz'
npzfile = np.load(datafile)
h_opt = npzfile['h_opt']
t = np.cumsum(h_opt)
t = np.hstack(([0], t))
T = len(t)
# Just load the states (we're only trying to visualize here, not actually
# stabilize a planned control trajectory).
q_opt = npzfile['q_opt'].T

# We scaled down the geometry by 4x, so scale the trajectory accordingly
TRAJ_SCALE = 1/4.0
q_opt *= TRAJ_SCALE

TRAJ_OFFSET_Z = 0.8
q_opt[[2, 3], :] += TRAJ_OFFSET_Z

# WARNING: fancy footwork required here.
# We want to define a trajectory source for the pancake that traces the
# pancake's path over time.
#
# To define the robot's path over time, we need to solve an IK problem to
# figure out the arm joint angles needed to move the flipper as desired.

# This set indexes into the pancake x, z, theta in q_opt
pancake_dof_idxs = [1, 3, 5]
# This set indexes into the flipper x, z, theta in q_opt
flipper_dof_idxs = [0, 2, 4]

# Get the pancake trajectory
q_pancake = np.zeros((pancake_plant.num_multibody_states(), T))
q_pancake[:3, :] = q_opt[pancake_dof_idxs, :]
q_pancake_poly = PiecewisePolynomial.FirstOrderHold(t, q_pancake)
q_pancake_source = builder.AddSystem(TrajectorySource(q_pancake_poly))

# Get the flipper trajectory in (roll, pitch, yaw, x, y, z)
rpy_xyz_flipper = np.zeros((6, T))
rpy_xyz_flipper[1, :] = q_opt[4, :]  # theta = pitch
rpy_xyz_flipper[3, :] = q_opt[0, :]  # x = x
rpy_xyz_flipper[5, :] = q_opt[2, :]  # z = z

# Now we need to compute the arm joint angles that achieve the desired
# orientation of the flipper frame at each timestep.
# Unfortunately, the Drake Python bindings don't support the function
# we want (ConstraintRelaxingIk.PlanSequentialTrajectory), so we'll do
# this offline in C++. OR CONVERT THIS WHOLE THING TO C++?


arm_poly = PiecewisePolynomial.FirstOrderHold(t, rpy_xyz_flipper)
arm_source = builder.AddSystem(TrajectorySource(arm_poly))

# Wire up the diagram
# We can drive the pancake's pose directly from q_pancake_source
builder.AddSystem(scene_graph)
pancake_pos_to_pose = builder.AddSystem(
    MultibodyPositionToGeometryPose(
        pancake_plant, input_multibody_state=True))
builder.Connect(
    q_pancake_source.get_output_port(0),
    pancake_pos_to_pose.get_input_port())
builder.Connect(
    pancake_pos_to_pose.get_output_port(),
    scene_graph.get_source_pose_port(pancake_plant.get_source_id()))

# Connect the arm
arm_pos_to_pose = builder.AddSystem(
    MultibodyPositionToGeometryPose(
        arm_plant, input_multibody_state=True))
builder.Connect(
    arm_source.get_output_port(0),
    arm_pos_to_pose.get_input_port())
builder.Connect(
    arm_pos_to_pose.get_output_port(),
    scene_graph.get_source_pose_port(arm_plant.get_source_id()))

# Make the visualizer
visualizer = ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.Initialize()

# generate and display animation
simulator.set_publish_every_time_step(False)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(q_pancake_poly.end_time())
