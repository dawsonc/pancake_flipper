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
urdf_path = FindResourceOrThrow(
      "drake/manipulation/models/iiwa_description/sdf/iiwa14_no_collision.sdf")
Parser(arm_plant).AddModelFromFile(urdf_path)

# # Weld the base link of the arm to the world
joint_arm_root_frame = arm_plant.GetFrameByName("iiwa_link_0")
arm_plant.WeldFrames(
    arm_plant.world_frame(),
    joint_arm_root_frame,
    RigidTransform.Identity())

# Finalize the plant so we can use it
arm_plant.Finalize()

# Define the trajectory
t_knot = np.array([0, 1])
q = np.zeros((arm_plant.num_multibody_states(), 2))
q[1, 0] = 1.0
q[1, 1] = -1.0
q_poly = PiecewisePolynomial.FirstOrderHold(t_knot, q)
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
simulator = Simulator(builder.Build())
simulator.Initialize()

# generate and display animation
simulator.set_publish_every_time_step(False)
simulator.set_target_realtime_rate(1.0)
simulator.AdvanceTo(q_poly.end_time())
