# Assignment AE4135 Rotor/wake aerodynamics
# BEM Assignment
#
# Author: Tomas Jakus
#
# This code implements the Blade Element Momentum (BEM) method for calculating the aerodynamic performance of a wind turbine rotor.
# including Glauert correction for high axial induction factors and Prandtl's tip loss correction.

import numpy as np



if __name__ == "__main__":

    # Geometric
    tipRadius = 50.0  # m
    bladeStart = 0.2 # -
    bladeEnd = 1 # -
    bladeNumber = 3
    rotorYaw = 0.0 # deg
    # Aerodynamic    
    freeStreamVelocity = 10.0 # m/s
    tipSpeedRatio = 6.0 # TODO include all three
    rotationalSpeed = tipSpeedRatio * freeStreamVelocity / tipRadius # rad/s
    # Fluid properties for air at sea level
    airDensity = 1.225 # kg/m^3
    airViscosity = 1.81e-5 # kg/m/s

