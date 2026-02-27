# Assignment 1; AE4135 Rotor/wake aerodynamics
# BEM Assignment
#
# Authors: - Tomas Jakus
#          - 
#          - 
#
	#region Pitch Variations
	
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

	def addSolutionIteration(self, radius, a, aPrime, iterations, precision, pradtlCorrection):
		self.elements.append({
			"radius": radius,
			"a": a,
			"aPrime": aPrime,
			"iterations": iterations,
			"precision": precision,
			"converged": precision < self.tolerance,
			"prandtlCorrection": pradtlCorrection
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
bladePitch = -2.0 # deg
chordFunction = lambda radius: 3.0 * (1 - dimensionlessRadialPosition(radius)) + 1.0 # m

#endregion

def PitchAngle(radius, pitchOverrideDeg=None):
	pitchDeg = bladePitch if pitchOverrideDeg is None else pitchOverrideDeg
	return np.deg2rad(bladeTwist(radius) + pitchDeg) # rad

# TODO: Apparently theres also a root loss
def PrandtlTipLoss(radius, a):
	if radius == tipRadius:
		return 0.0 # At the tip, use the full correction value (not zero)
	
	# r_R is the non-dimensioned radius position
	r_R = dimensionlessRadialPosition(radius)
	
	# Tip loss factor calculation
	# Use (1-a)**2 as denominator, but guard against division by zero
	denominator = (1 - a)**2
	if denominator < 1e-6:
		denominator = 1e-6
	
	argument_tip = -bladeNumber/2 * (1 - r_R) / r_R * np.sqrt(1 + (tipSpeedRatio * r_R)**2 / denominator)
	argument_tip = np.clip(argument_tip, -700, 0)
	exp_tip = np.clip(np.exp(argument_tip), -1, 1)
	fTip = 2/np.pi * np.arccos(exp_tip)
	
	# Root loss factor calculation
	argument_root = -bladeNumber/2 * (r_R - dimensionlessRadialPosition(bladeStart * tipRadius)) / (1 - dimensionlessRadialPosition(bladeStart * tipRadius)) * np.sqrt(1 + (tipSpeedRatio * r_R)**2 / denominator)
	argument_root = np.clip(argument_root, -700, 0)
	exp_root = np.clip(np.exp(argument_root), -1, 1)
	fRoot = 2/np.pi * np.arccos(exp_root)

	# Combined loss factor
	f = fTip * fRoot

	return f

# Induction will decrease the axial velocity at the rotor plane by (1 - a)
# and increase the swirl by a factor of (1 + aPrime)
def Inflowangle(radius, a, aPrime):
	# TODO: Check this formula
	return np.arctan(freeStreamVelocity * (1 - a) / (rotationalSpeed * radius * (1 + aPrime))) # rad

def AngleOfAttack(radius, a, aPrime, pitchOverrideDeg=None):
	return Inflowangle(radius, a, aPrime) - PitchAngle(radius, pitchOverrideDeg=pitchOverrideDeg) # rad

def Solidity(radius):
	return bladeNumber * chordFunction(radius) / (2 * np.pi * radius)

# Calculates the differential force coefficient at a given radius already projected in axial and tangential directions
# Does not apply any corrections on its own
def BEMElement(radius, a, aPrime, pitchOverrideDeg=None)-> tuple:
	alpha = AngleOfAttack(radius, a, aPrime, pitchOverrideDeg=pitchOverrideDeg)
	cl, cd, _ = airfoil.lookupAll(alpha)
	inflowAngle = Inflowangle(radius, a, aPrime)
	
	# Axial and tangential forces per unit length
	dCAxial = cl * np.cos(inflowAngle) + cd * np.maximum(np.sin(inflowAngle), EPSILON)
	dCTangential = cl * np.maximum(np.sin(inflowAngle), EPSILON) - cd * np.cos(inflowAngle)

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
		solutionProperties:SolutionProperties,
		pitchOverrideDeg=None) -> tuple:

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
		dCax, dCtg = BEMElement(radius, a, aPrime, pitchOverrideDeg=pitchOverrideDeg)
		prandtlCorrection = PrandtlTipLoss(radius, a)

		# At tip, Prandtl correction will converge to 0 and we reach unity (DIV0)
		# HACK: We can assume minimal change and return values from previous solution
		# TODO: Is this the problem? Maybe just return zeros?
		if prandtlCorrection < 1e-6:
			solutionProperties.addSolutionIteration(radius, a, aPrime, iteration, 0.0, prandtlCorrection)
			return a, aPrime

		inflowAngle = Inflowangle(radius, a, aPrime)

		# Correct induction factors
		sinInflow = np.maximum(np.sin(inflowAngle), EPSILON)
		cosInflow = np.cos(inflowAngle)
		kappa = solidity * dCax / (4 * prandtlCorrection * sinInflow**2)
		aMomentum = kappa / (1 + kappa)

		kappaPrime = solidity * dCtg / (4 * prandtlCorrection * sinInflow * cosInflow)
		aPrimeMomentum = kappaPrime / (1 - kappaPrime)

		# Glauert correction
		# TODO: Move this to a separate function
		# TODO: Compare to the solution in the slides
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
			solutionProperties.addSolutionIteration(
				radius, 
				aMomentum, 
				aPrimeMomentum, 
				iteration, 
				max(np.abs(a - aMomentum), 
				np.abs(aPrime - aPrimeMomentum)), 
				prandtlCorrection)
			
			return aMomentum, aPrimeMomentum

		# Assign new values with relaxation
		a = relaxation * aMomentum + (1 - relaxation) * a
		aPrime = relaxation * aPrimeMomentum + (1 - relaxation) * aPrime

	print("No convergence reached after maximum iterations") # TODO: Debug when this is called and find the problem
	return a, aPrime

# Computes the corrected converged differential axial thrust and torque multiplied by the bladeNumber
def solveElement(radius, solutionProperties:SolutionProperties=None, pitchOverrideDeg=None):
	a, aPrime = SolveInduction(radius, solutionProperties=solutionProperties, pitchOverrideDeg=pitchOverrideDeg)
	dCax, dCtg = BEMElement(radius, a, aPrime, pitchOverrideDeg=pitchOverrideDeg)

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

		convergenceData = solution.getConvergence(radiusIterator)
		if convergenceData:
			print(f"Solution converged: {convergenceData['converged']}, iterations: {convergenceData['iterations']}, precision: {convergenceData['precision']}")
		else:
			print(f"No convergence data found for radius: {radiusIterator:.2f} m")
		print(f"Differential thrust: {dTCurrent:.2f} N/m, Differential torque: {dQCurrent:.2f} Nm/m")

	print("=" * 40)
	print("Final results:")
	print(f"Total thrust: {thrust:.2f} N, Total torque: {torque:.2f} Nm")
	
	# Calculate coefficient of power
	rotorArea = np.pi * tipRadius**2
	power = torque * rotationalSpeed  # Power = Torque * angular velocity
	windPower = 0.5 * airDensity * rotorArea * freeStreamVelocity**3  # Available wind power
	cP = power / windPower if windPower > 0 else 0
	print(f"Total power: {power:.2f} W, Coefficient of power cP: {cP:.4f}")

	import matplotlib.pyplot as plt
	#region Plotting

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

	fig2 = plt.figure(figsize=(10, 6))
	fig2_ax = fig2.add_subplot(111)

	prandtlCorrection = np.array([element['prandtlCorrection'] for element in solution.elements])
	axialInduction = np.array([element['a'] for element in solution.elements])
	
	# Plot Prandtl correction
	ax2_main = fig2_ax
	ax2_main.plot(radius, prandtlCorrection, marker='o', color='tab:purple', label='Prandtl correction', linewidth=2)
	ax2_main.set_xlabel('Radius (m)')
	ax2_main.set_ylabel('Prandtl Tip Loss Correction (-)', color='tab:purple')
	ax2_main.tick_params(axis='y', labelcolor='tab:purple')
	ax2_main.grid(True)
	
	# Plot axial induction on secondary axis
	ax2_secondary = ax2_main.twinx()
	ax2_secondary.plot(radius, axialInduction, marker='s', color='tab:cyan', label='Axial induction a', linewidth=2)	
	ax2_secondary.set_ylabel('Axial induction a (-)', color='tab:cyan')
	ax2_secondary.tick_params(axis='y', labelcolor='tab:cyan')
	
	ax2_main.set_title('Prandtl Tip Loss Correction vs Radius')
	ax2_main.legend(loc='upper left')
	ax2_secondary.legend(loc='upper right')
	plt.grid(True)

	plt.show()
	#endregion

	#region Pitch Variations
	# Pitch as a function of radius
	pitchByRadius = lambda radiusValue: -2.0 + 4.0 * (1 - dimensionlessRadialPosition(radiusValue)) # deg
	radiusSweep = np.linspace(bladeStart * tipRadius, bladeEnd * tipRadius - 0.001 * tipRadius, num=80)

	# Discretized map where each cell is one (radius, pitch) region
	pitchMinDeg = -10.0
	pitchMaxDeg = 10.0
	pitchBins = 50
	radiusBins = 80

	pitchGrid = np.linspace(pitchMinDeg, pitchMaxDeg, num=pitchBins)
	radiusGrid = np.linspace(bladeStart * tipRadius, bladeEnd * tipRadius - 0.001 * tipRadius, num=radiusBins)

	dTMap = np.zeros((pitchBins, radiusBins))
	dQMap = np.zeros((pitchBins, radiusBins))
	cPMap = np.zeros((pitchBins, radiusBins))

	for pitchIndex, pitchValue in enumerate(pitchGrid):
		for radiusIndex, radiusValue in enumerate(radiusGrid):
			bladePitch = pitchValue # Override the blade pitch for this solution
			localSolution = SolutionProperties()
			localSolution.setParameters(
				maxIterations=1000,
				tolerance=1e-4,
				relaxation=0.5,
				elementCount=1
			)

			dTValue, dQValue = solveElement(radiusValue, solutionProperties=localSolution, pitchOverrideDeg=pitchValue)
			dTMap[pitchIndex, radiusIndex] = dTValue
			dQMap[pitchIndex, radiusIndex] = dQValue
			
			# Calculate local cP
			localPower = dQValue * rotationalSpeed
			localRotorArea = np.pi * tipRadius**2
			localWindPower = 0.5 * airDensity * localRotorArea * freeStreamVelocity**3
			cPMap[pitchIndex, radiusIndex] = localPower / localWindPower if localWindPower > 0 else 0

	# Coefficient of power map
	fig_pitch = plt.figure(figsize=(12, 8))
	
	targetVariavle = cPMap

	# cP performance map
	ax_cP = fig_pitch.add_subplot(111)
	heatmap_cP = ax_cP.pcolormesh(
		radiusGrid,
		pitchGrid,
		targetVariavle,
		shading='auto',
		cmap='viridis'
	)
	ax_cP.set_xlabel('Radius (m)')
	ax_cP.set_ylabel('Pitch (deg)')
	ax_cP.set_title('Coefficient of Power (cP) Performance Map')
	ax_cP.grid(False)
	colorbar_cP = fig_pitch.colorbar(heatmap_cP, ax=ax_cP)
	colorbar_cP.set_label('cP (-)')
	
	# Add markers for maximum cP value in each column (each radius)
	max_indices = np.argmax(targetVariavle, axis=0)  # Find max pitch index for each radius
	max_pitches = pitchGrid[max_indices]  # Get the pitch values at those indices
	ax_cP.plot(radiusGrid, max_pitches, 'r*', markersize=15, label='Maximum cP per radius')
	ax_cP.legend(loc='best')

	plt.tight_layout()
	plt.show()
	#endregion