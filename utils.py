'''
This module provides various helper functions for computational
dynamics and control.

Written by C. Dawson (cbd@mit.edu)
'''
from pydrake.multibody.tree import MultibodyForces_


def ManipulatorDynamics(plant, q, v=None):
    '''Calculate the mass, C, actuation, gravitational torque,
    and external torque matrices/vectors for the given MultibodyPlant.

    Copied from the underactuated library by Russ Tedrake
    (transcribing code helps me learn)
    '''
    # Create a local context
    context = plant.CreateDefaultContext()

    # Set the state and state velocities (if given),
    # working inside the local context
    plant.SetPositions(context, q)
    if v is not None:
        plant.SetVelocities(context, v)

    # Get the matrices for the manipulator equations
    M = plant.CalcMassMatrixViaInverseDynamics(context)
    Cv = plant.CalcBiasTerm(context)
    tauG = plant.CalcGravityGeneralizedForces(context)
    B = plant.MakeActuationMatrix()
    forces = MultibodyForces_(plant)
    plant.CalcForceElementsContribution(context, forces)
    tauExt = forces.generalized_forces()

    return (M, Cv, tauG, B, tauExt)
