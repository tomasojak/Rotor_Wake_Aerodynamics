# Assignment AE4135 Rotor/wake aerodynamics
# BEM Assignment
#
# Author: Tomas Jakus
#
# This code implements the Blade Element Momentum (BEM) method for calculating the aerodynamic performance of a wind turbine rotor.
# including Glauert correction for high axial induction factors and Prandtl's tip loss correction.

import numpy as np
import pandas as pd

# We want a handy way to load airfoil data from a file (or keep local capabilityto define it in code)
class Airfoil:
    def __init__(self, alpha, cl, cd, cm):
        self.alpha = alpha
        self.cl = cl
        self.cd = cd
        self.cm = cm

    # Overload the constructor to allow loading from a file
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        # Input in CSV is in degrees; convert once so internals use radians.
        self.alpha = np.deg2rad(self.data['Alfa'].values)
        self.cl = self.data['Cl'].values
        self.cd = self.data['Cd'].values
        self.cm = self.data['Cm'].values

    # We will want to lookup the values
    def lookup(self, alpha, variable: str): # Performs linear interpolation
        if variable == 'cl':
            return np.interp(alpha, self.alpha, self.cl)
        elif variable == 'cd':
            return np.interp(alpha, self.alpha, self.cd)
        elif variable == 'cm':
            return np.interp(alpha, self.alpha, self.cm)
        else:
            raise ValueError("Variable must be 'cl', 'cd', or 'cm'")
        
    def lookupAll(self, alpha) -> tuple:
        return self.lookup(alpha, 'cl'), self.lookup(alpha, 'cd'), self.lookup(alpha, 'cm')

#region Parameters
# I do not like keeping magic numbers here but seems most convinient to keep away from main

# Geometric
tipRadius = 50.0  # m
bladeStart = 0.2 # -
bladeEnd = 1 # -
bladeNumber = 3

dimensionlessRadialPosition = lambda radius: radius / tipRadius

# Aerodynamic    
freeStreamVelocity = 10.0 # m/s
tipSpeedRatio = 6.0 # TODO include all three TSRs
rotationalSpeed = tipSpeedRatio * freeStreamVelocity / tipRadius # rad/s
# Fluid properties for air at sea level
airDensity = 1.225 # kg/m^3
airViscosity = 1.81e-5 # kg/m/s

# Load airfoil data
airfoil = Airfoil('AirfoilPolarPlot.csv')

bladeTwist = lambda radius: 14.0 * (1 - dimensionlessRadialPosition(radius)) # deg
bladePitch = lambda radius: -2.0 # deg
chordFunction = lambda radius: 3.0 * (1 - dimensionlessRadialPosition(radius)) + 1.0 # m

#endregion

def PitchAngle(radius):
    return np.deg2rad(bladeTwist(radius) + bladePitch(radius)) # rad

def PrandtlTipLoss(radius, a):
    d = ((2 * np.pi * radius) / bladeNumber) * (1 - a) / np.sqrt(tipSpeedRatio**2 + (1 - a)**2)
    f = (2/np.pi) * np.arccos(np.exp(-np.pi * (1-radius/tipRadius) / d))
    return f

# Induction will decrease the axial velocity at the rotor plane by (1 - a)
# and increase the swirl by a factor of (1 + aPrime)
def Inflowangle(radius, a, aPrime):
    phi = np.arctan(freeStreamVelocity * (1 - a) / (rotationalSpeed(radius) * radius * (1 + aPrime)))
    return phi # rad

def AngleOfAttack(radius, a, aPrime):
    return PitchAngle(radius) - Inflowangle(radius, a, aPrime) # rad

def Solidity(radius):
    return bladeNumber * chordFunction(radius) / (2 * np.pi * radius)

def BEMElement(radius, a, aPrime)-> tuple:
    alpha = AngleOfAttack(radius, a, aPrime)
    cl, cd, _ = airfoil.lookupAll(alpha)
    inflowAngle = Inflowangle(radius, a, aPrime)
    
    # Axial and tangential forces per unit length
    dCaxial = cl * np.cos(inflowAngle) + cd * np.sin(inflowAngle)
    dCtangential = cl * np.sin(inflowAngle) - cd * np.cos(inflowAngle)

    return dCaxial, dCtangential

def GlauertCorrection(a):

    return

CT1 = 1.816
CT2 = 2 * np.sqrt(CT1) - CT1 # NOTE: Does optimizer resolve this?

# We will want to solve for a and aPrime iteratively
def SolveInduction(radius, maxIterations=100, tol=1e-6, relaxation=0.5):

    a = 0.3 # Initial guess for axial induction factor
    aPrime = 0.01 # Initial guess for tangential induction factor

    solidity = Solidity(radius)

    for iteration in range(maxIterations):

        # Find the forces on the blade element
        dCax, dCtg = BEMElement(radius, a, aPrime)

        # Correct induction factors
        kappa = solidity * dCax / (4 * PrandtlTipLoss(radius, a) * np.sin(Inflowangle(radius, a, aPrime))**2)
        aMomentum = kappa / (1 + kappa)

        kappaPrime = solidity * dCtg / (4 * PrandtlTipLoss(radius, a) * np.sin(Inflowangle(radius, a, aPrime)) * np.cos(Inflowangle(radius, a, aPrime)))
        aPrimeMomentum = kappaPrime / (1 - kappaPrime)

        # Glauert correction
        relativeVelocity = np.sqrt((freeStreamVelocity * (1 - a))**2 + (rotationalSpeed(radius) * radius * (1 + aPrime))**2)
        thrustCoefficient = solidity * dCax * (relativeVelocity / freeStreamVelocity)**2

        if thrustCoefficient > CT2:
            aMomentum = 1 + (thrustCoefficient - CT1) / (4 * np.sqrt(CT1) - 4)

        # Check for convergence
        if np.abs(a - aMomentum) < tol and np.abs(aPrime - aPrimeMomentum) < tol:
            break

        # Assign new values with relaxation
        a = relaxation * aMomentum + (1 - relaxation) * a
        aPrime = relaxation * aPrimeMomentum + (1 - relaxation) * aPrime

    return a, aPrime

def solveElement(radius):
    a, aPrime = SolveInduction(radius)
    dCax, dCtg = BEMElement(radius, a, aPrime)

    relativeVelocity = np.sqrt((freeStreamVelocity * (1 - a))**2 + (rotationalSpeed(radius) * radius * (1 + aPrime))**2)

    dT = 0.5 * airDensity * relativeVelocity**2 * bladeNumber * chordFunction(radius) * dCax
    dQ = 0.5 * airDensity * relativeVelocity**2 * bladeNumber * chordFunction(radius) * radius * dCtg

    return dT, dQ