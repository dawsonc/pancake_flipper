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
from pydrake.math import RigidTransform


# Create the builder to house our system
builder = DiagramBuilder()

# Instantiate the arm plant and the scene graph.
arm_plant = MultibodyPlant(time_step=0)
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
pancake_model_file = "./models/pancake_flipper_arm_scale.urdf"
Parser(arm_plant).AddModelFromFile(pancake_model_file)
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

# Define the trajectory
# This set indexes into the pancake DOFs
pancake_dof_idxs = [0, 2, 4]
# This set indexes into the arm DOFs
arm_dof_idxs = [1, 2, 5, 6, 7, 8, 9]

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

q_traj = np.zeros((arm_plant.num_multibody_states(), T))
q_traj[pancake_dof_idxs, :] = q_opt[[1, 3, 5], :]
q_poly = PiecewisePolynomial.FirstOrderHold(t, q_traj)
q_source = builder.AddSystem(TrajectorySource(q_poly))

# Wire up the diagram
builder.AddSystem(scene_graph)
pos_to_pose = builder.AddSystem(
    MultibodyPositionToGeometryPose(
        arm_plant, input_multibody_state=True))
builder.Connect(q_source.get_output_port(0), pos_to_pose.get_input_port())
builder.Connect(
    pos_to_pose.get_output_port(),
    scene_graph.get_source_pose_port(arm_plant.get_source_id()))

# Make the visualizer
visualizer = ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
diagram = builder.Build()
simulator = Simulator(diagram)
simulator.Initialize()

# generate and display animation
simulator.set_publish_every_time_step(False)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(q_poly.end_time())
