# Assignment 1; AE4135 Rotor/wake aerodynamics
# BEM Assignment
#
# Authors: - Tomas Jakus
#          - 
#          - 
#
# This code implements the Blade Element Momentum (BEM) method for calculating the aerodynamic performance of a wind turbine rotor.
# including Glauert correction for high axial induction factors and Prandtl's tip loss correction.

import numpy as np
import pandas as pd

EPSILON = 1e-6

# We want a handy way to load airfoil data from a file (or keep local capability to define it in code)
class Airfoil:

	# Loads the values from an external file
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
	
class SolutionProperties:
	# Store convergence history
	def __init__(self):
		self.elements = []
		# NOTE: Maybe this should be kept separately from the solution
		self.maxIterations = 100
		self.tolerance = 1e-6
		self.relaxation = 0.5
		self.elementCount = 10

		self.initialGuess = {
			"a": 0.3,
			"aPrime": 0.01
			}
	
	def setParameters(self, maxIterations, tolerance, relaxation, elementCount):
		self.maxIterations = maxIterations
		self.tolerance = tolerance
		self.relaxation = relaxation
		self.elementCount = elementCount

	def addSolutionIteration(self, radius, a, aPrime, iterations, precision):
		self.elements.append({
			"radius": radius,
			"a": a,
			"aPrime": aPrime,
			"iterations": iterations,
			"precision": precision,
			"converged": precision < self.tolerance
		})

	# TODO: We want to keep all of the solution variables in one struct to then have easy access to solutions of all TSRs, to do this we should:
	#           - Create a handy struct to hold the information
	#           - Hold each TSR solution in a separate "solutionProperties" instance

	def getConvergence(self, radius):
		for element in self.elements:
			if element["radius"] == radius:
				return element
		return None # Not found

#region Parameters
# I do not like keeping magic numbers here but seems most convenient to keep away from main

# Geometric
tipRadius = 50.0  # m
bladeStart = 0.2 # -
bladeEnd = 1 # -
bladeNumber = 3

dimensionlessRadialPosition = lambda radius: radius / tipRadius

# Aerodynamic    
freeStreamVelocity = 10.0 # m/s
tipSpeedRatio = 8.0 # TODO include all three TSRs
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
	if radius == tipRadius:
		return 0.0 # At the tip, the correction converges to zero
	
	d = ((2 * np.pi * radius) / bladeNumber) * (1 - a) / np.sqrt(tipSpeedRatio**2 + (1 - a)**2)
	argument = np.exp(-np.pi * (1-radius/tipRadius) / d)
	# HACK: Clamp argument to avoid numerical issues with arccos
	# We might be hitting some unities here around the tip
	argument = np.clip(argument, EPSILON, 1 - EPSILON)
	f = (2/np.pi) * np.arccos(argument)
	return f

# Induction will decrease the axial velocity at the rotor plane by (1 - a)
# and increase the swirl by a factor of (1 + aPrime)
def Inflowangle(radius, a, aPrime):
	# TODO: Check this formula
	return np.arctan(freeStreamVelocity * (1 - a) / (rotationalSpeed * radius * (1 + aPrime))) # rad

def AngleOfAttack(radius, a, aPrime):
	return Inflowangle(radius, a, aPrime) - PitchAngle(radius) # rad

def Solidity(radius):
	return bladeNumber * chordFunction(radius) / (2 * np.pi * radius)

# Calculates the differential force coefficient at a given radius already projected in axial and tangential directions
# Does not apply any corrections on its own
def BEMElement(radius, a, aPrime)-> tuple:
	alpha = AngleOfAttack(radius, a, aPrime)
	cl, cd, _ = airfoil.lookupAll(alpha)
	inflowAngle = Inflowangle(radius, a, aPrime)
	
	# Axial and tangential forces per unit length
	dCAxial = cl * np.cos(inflowAngle) + cd * max(np.sin(inflowAngle), EPSILON)
	dCTangential = cl * max(np.sin(inflowAngle), EPSILON) - cd * np.cos(inflowAngle)

	return dCAxial, dCTangential

# We will want to solve for a and aPrime iteratively
# Solves the induction factors at a given radius stage, applies Prandtl and Glauert corrections
# Takes in settings for iterations and initial guesses
# To do this the following algorithm is applied:
# 	1) Takes an initial guess for induction factors from the previous solution (or default values if solution is not available)
#	2) Enter Iterative Loop:
#	3) Solve the bare BEM element thrust and tangential force coefficient
#	4) Apply Prandtl tip loss correction ont the element
#	5) For the given coefficients, use Rankine theory to find the induction
#	6) Apply Glauert correction if applicable (if rotor is heavily loaded)
#	7) Assign the momentum theory induction to aMomentum
#	8) Correct the induction with the induction suggested by momentum theory with a given relaxation factor
#	9) Repeat 3-8 until convergence is reached
#	10) TODO: Write data to solution struct
def SolveInduction(
		radius, 
		solutionProperties:SolutionProperties) -> tuple:

	# Load iteration and convergence values from properties set for the solver
	maxIterations = solutionProperties.maxIterations
	tolerance = solutionProperties.tolerance
	relaxation = solutionProperties.relaxation

	# Initial guess
	if solutionProperties.elements.__len__() > 0:
		# TODO: Do not use prev values if we didn't converge
		# We have a previous solution, use it as initial guess
		previousSolution = solutionProperties.elements[-1]
		a = previousSolution["a"]
		aPrime = previousSolution["aPrime"]
	else:
		# This is the first solution, use default initial guess
		a = solutionProperties.initialGuess["a"]
		aPrime = solutionProperties.initialGuess["aPrime"]

	solidity = Solidity(radius)

	for iteration in range(maxIterations):

		# Find the forces on the blade element
		dCax, dCtg = BEMElement(radius, a, aPrime)
		prandtlCorrection = PrandtlTipLoss(radius, a)

		# At tip, Prandtl correction will converge to 0 and we reach unity (DIV0)
		# HACK: We can assume minimal change and return values from previous solution
		# TODO: Is this the problem? Maybe just return zeros?
		if prandtlCorrection < 1e-6:
			solutionProperties.addSolutionIteration(radius, a, aPrime, iteration, 0.0)
			return a, aPrime

		inflowAngle = Inflowangle(radius, a, aPrime)

		# Correct induction factors
		kappa = solidity * dCax / (4 * prandtlCorrection * max(np.sin(inflowAngle), EPSILON)**2)
		aMomentum = kappa / (1 + kappa)

		kappaPrime = solidity * dCtg / (4 * prandtlCorrection * max(np.sin(inflowAngle), EPSILON) * np.cos(inflowAngle))
		aPrimeMomentum = kappaPrime / (1 - kappaPrime)

		# Glauert correction
		# TODO: Move this to a separate function
		# Magic number correction thresholds from lecture slides
		CT1 = 1.816
		CT2 = 2 * np.sqrt(CT1) - CT1 # NOTE: Does optimizer resolve this?

		relativeVelocity = np.sqrt((freeStreamVelocity * (1 - a))**2 + (rotationalSpeed * radius * (1 + aPrime))**2)
		thrustCoefficient = solidity * dCax * (relativeVelocity / freeStreamVelocity)**2
		
		if thrustCoefficient > CT2: # We need to apply Glauert correction
			aMomentum = 1 + (thrustCoefficient - CT1) / (4 * np.sqrt(CT1) - 4)

		# Check for convergence
		if np.abs(a - aMomentum) < tolerance and np.abs(aPrime - aPrimeMomentum) < tolerance:
			# We have converged
			solutionProperties.addSolutionIteration(radius, aMomentum, aPrimeMomentum, iteration, max(np.abs(a - aMomentum), np.abs(aPrime - aPrimeMomentum)))
			return aMomentum, aPrimeMomentum

		# Assign new values with relaxation
		a = relaxation * aMomentum + (1 - relaxation) * a
		aPrime = relaxation * aPrimeMomentum + (1 - relaxation) * aPrime

	print("No convergence reached after maximum iterations")
	return a, aPrime

# Computes the corrected converged differential axial thrust and torque multiplied by the bladeNumber
def solveElement(radius, solutionProperties:SolutionProperties=None):
	a, aPrime = SolveInduction(radius, solutionProperties=solutionProperties)
	dCax, dCtg = BEMElement(radius, a, aPrime)

	relativeVelocity = np.sqrt((freeStreamVelocity * (1 - a))**2 + (rotationalSpeed * radius * (1 + aPrime))**2)

	dT = 0.5 * airDensity * relativeVelocity**2 * bladeNumber * chordFunction(radius) * dCax
	dQ = 0.5 * airDensity * relativeVelocity**2 * bladeNumber * chordFunction(radius) * radius * dCtg

	return dT, dQ

if __name__ == "__main__":

	solution = SolutionProperties()
	solution.setParameters(
		maxIterations=500, 
		tolerance=1e-6, 
		relaxation=0.5,
		elementCount=100)
	
	radius = np.linspace(bladeStart * tipRadius, bladeEnd * tipRadius - 0.001 * tipRadius, num=solution.elementCount) # Avoid tip singularities

	dT = np.array([])
	dQ = np.array([])

	thrust = 0.0
	torque = 0.0
	
	for radiusIterator in radius:
		print("-" * 20)
		print(f"Solving for radius: {radiusIterator:.2f} m")

		dTCurrent, dQCurrent = solveElement(radiusIterator, solutionProperties=solution)

		dT = np.append(dT, dTCurrent)
		dQ = np.append(dQ, dQCurrent)

		thrust += dTCurrent * (radius[1] - radius[0]) # Integration
		torque += dQCurrent * (radius[1] - radius[0]) # Integration

		print(f"Solution converged: {solution.getConvergence(radiusIterator)['converged']}, iterations: {solution.getConvergence(radiusIterator)['iterations']}, precision: {solution.getConvergence(radiusIterator)['precision']}")
		print(f"Differential thrust: {dTCurrent:.2f} N/m, Differential torque: {dQCurrent:.2f} Nm/m")

	print("=" * 40)
	print("Final results:")
	print(f"Total thrust: {thrust:.2f} N, Total torque: {torque:.2f} Nm")

	#region Plotting
	import matplotlib.pyplot as plt

	convergencePrecision = np.array([element["precision"] for element in solution.elements])
	convergenceIterations = np.array([element["iterations"] for element in solution.elements])

	fig, (ax_top, ax_bottom) = plt.subplots(
		2, 1, figsize=(12, 9), sharex=True,
		gridspec_kw={"height_ratios": [2, 1]}
	)

	# Top subplot: dT and dQ with secondary axis
	ax_top_secondary = ax_top.twinx()
	line_dT, = ax_top.plot(radius, dT, marker='o', color='tab:blue', label='dT')
	line_dQ, = ax_top_secondary.plot(radius, dQ, marker='s', color='tab:orange', label='dQ')

	ax_top.set_ylabel('Differential Thrust dT (N/m)', color='tab:blue')
	ax_top_secondary.set_ylabel('Differential Torque dQ (Nm/m)', color='tab:orange')
	ax_top.tick_params(axis='y', labelcolor='tab:blue')
	ax_top_secondary.tick_params(axis='y', labelcolor='tab:orange')
	ax_top.grid(True)
	ax_top.legend([line_dT, line_dQ], ['dT', 'dQ'], loc='best')

	# Bottom subplot: convergence precision and iterations
	ax_bottom_secondary = ax_bottom.twinx()
	line_precision, = ax_bottom.plot(
		radius, convergencePrecision, marker='o', color='tab:green', label='Convergence precision'
	)
	line_iterations, = ax_bottom_secondary.plot(
		radius, convergenceIterations, marker='^', color='tab:red', label='Iterations'
	)

	ax_bottom.set_xlabel('Radius (m)')
	ax_bottom.set_ylabel('Convergence precision (-)', color='tab:green')
	ax_bottom_secondary.set_ylabel('Iterations (-)', color='tab:red')
	ax_bottom.tick_params(axis='y', labelcolor='tab:green')
	ax_bottom_secondary.tick_params(axis='y', labelcolor='tab:red')
	ax_bottom.grid(True)
	ax_bottom.legend([line_precision, line_iterations], ['Convergence precision', 'Iterations'], loc='best')

	plt.tight_layout()
	plt.show()
	#endregion