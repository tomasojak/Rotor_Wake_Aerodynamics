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
import SolverTools as st
from Plotting import *

EPSILON = 1e-6
DEBUG_PRINT = False

def PitchAngle(radius, solverProperties:st.SolverData):
	bladeTwist = np.deg2rad(solverProperties.geometry.bladeTwist(radius))
	bladePitchDeg = np.deg2rad(solverProperties.geometry.bladePitch)
	return bladeTwist + bladePitchDeg # rad

def PrandtlCorrection(radius, a, solverProperties:st.SolverData):
	"""
	Calculate Prandtl's tip and root loss correction factor.
	
	This accounts for the finite number of blades and the flow around the blade tips
	and roots, which reduces the effective forces compared to an infinite number of blades.
	
	Args:
		radius: Radial position along the blade (m)
		a: Axial induction factor
		solverProperties: SolverData instance containing geometry and solver settings
		
	Returns:
		Combined Prandtl correction factor (0 to 1)
	"""
	geometry = solverProperties.geometry

	if radius == geometry.tipRadius:
		return 0.0 # At the tip, use the full correction value (not zero)
	
	# r_R is the non-dimensioned radius position
	r_R = geometry.dimensionlessRadialPosition(radius)
	
	# Tip loss factor calculation
	# Use (1-a)**2 as denominator, but guard against division by zero
	denominator = (1 - a)**2
	if denominator < 1e-6:
		denominator = 1e-6
	
	argument_tip = -geometry.bladeNumber/2 * (1 - r_R) / r_R * np.sqrt(1 + (geometry.tipSpeedRatio * r_R)**2 / denominator)
	argument_tip = np.clip(argument_tip, -700, 0)
	exp_tip = np.clip(np.exp(argument_tip), -1, 1)
	fTip = 2/np.pi * np.arccos(exp_tip)
	
	# Root loss factor calculation
	argument_root = -geometry.bladeNumber/2 * (r_R - geometry.dimensionlessRadialPosition(geometry.bladeStart * geometry.tipRadius)) / (1 - geometry.dimensionlessRadialPosition(geometry.bladeStart * geometry.tipRadius)) * np.sqrt(1 + (geometry.tipSpeedRatio * r_R)**2 / denominator)
	argument_root = np.clip(argument_root, -700, 0)
	exp_root = np.clip(np.exp(argument_root), -1, 1)
	fRoot = 2/np.pi * np.arccos(exp_root)

	# Combined loss factor
	f = fTip * fRoot

	return f

def GlauertCorrection(aMomentum, radius, induction, inductionPrime, solidity, dCax, solverProperties:st.SolverData):
	"""
	Apply Glauert's empirical correction for high axial induction factors.
	
	For heavily loaded rotors (high thrust coefficients), momentum theory becomes
	inaccurate. This correction uses an empirical relationship based on experimental data.
	
	Args:
		aMomentum: Axial induction factor from momentum theory
		radius: Radial position along the blade (m)
		induction: Current axial induction factor
		inductionPrime: Current tangential induction factor
		solidity: Local solidity
		dCax: Differential axial force coefficient
		solverProperties: SolverData instance containing geometry and solver settings
		
	Returns:
		Corrected axial induction factor
	"""
	geometry = solverProperties.geometry

	CT1 = 1.816
	CT2 = 2 * np.sqrt(CT1) - CT1

	relativeVelocity = np.sqrt((geometry.freeStreamVelocity * (1 - induction))**2 + (geometry.rotationalSpeed * radius * (1 + inductionPrime))**2)
	thrustCoefficient = solidity * dCax * (relativeVelocity / geometry.freeStreamVelocity)**2
	
	if thrustCoefficient > CT2: # We need to apply Glauert correction
		aMomentum = 1 + (thrustCoefficient - CT1) / (4 * np.sqrt(CT1) - 4)
	return aMomentum
	
def Inflowangle(radius, a, aPrime, solverProperties:st.SolverData):
	freeStreamVelocity = solverProperties.geometry.freeStreamVelocity
	rotationalSpeed = solverProperties.geometry.rotationalSpeed
	return np.arctan(freeStreamVelocity * (1 - a) / (rotationalSpeed * radius * (1 + aPrime))) # rad

def AngleOfAttack(radius, a, aPrime, solverProperties:st.SolverData):
	return Inflowangle(radius, a, aPrime, solverProperties) - PitchAngle(radius, solverProperties) # rad

def Solidity(radius, solverProperties:st.SolverData):
	bladeNumber = solverProperties.geometry.bladeNumber
	chord = solverProperties.geometry.chordFunction(radius)
	return bladeNumber * chord / (2 * np.pi * radius)

def BEMElement(radius, a, aPrime, solverProperties:st.SolverData)-> tuple:
	"""
	Calculate differential force coefficients at a given radius.
	
	Computes the axial and tangential force coefficients projected from lift and drag,
	but does not apply Prandtl or Glauert corrections.
	
	Args:
		radius: Radial position along the blade (m)
		a: Axial induction factor
		aPrime: Tangential induction factor
		solverProperties: SolverData instance containing geometry and solver settings
		
	Returns:
		Tuple of (dCAxial, dCTangential) - differential force coefficients
	"""
	alpha = AngleOfAttack(radius, a, aPrime, solverProperties)
	cl, cd, _ = solverProperties.geometry.airfoil.lookupAll(alpha)
	inflowAngle = Inflowangle(radius, a, aPrime, solverProperties)
	
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
#	10) Write data to solution struct
def SolveInduction(
		radius,
		solutionProperties:st.SolverData) -> tuple:
	"""
	Iteratively solve for induction factors at a given radial position.
	
	Applies the BEM algorithm with Prandtl tip/root loss and Glauert corrections:
	1. Start with initial guess from previous solution or default values
	2. Iterative loop:
	   - Calculate BEM element forces
	   - Apply Prandtl tip/root loss correction
	   - Use momentum theory to find new induction factors
	   - Apply Glauert correction for high induction
	   - Update with relaxation factor
	3. Repeat until convergence or max iterations reached
	
	Args:
		radius: Radial position along the blade (m)
		solutionProperties: SolverData instance containing geometry, solver settings,
		                    and convergence history
		
	Returns:
		Tuple of (a, aPrime) - converged axial and tangential induction factors
	"""

	# Load iteration and convergence values from properties set for the solver
	maxIterations = solutionProperties.maxIterations
	tolerance = solutionProperties.tolerance
	relaxation = solutionProperties.relaxation

	# Initial guess - start with default values
	a = solutionProperties.initialGuess["a"]
	aPrime = solutionProperties.initialGuess["aPrime"]
	
	# If we have a previous converged solution, use it as initial guess
	if solutionProperties.elementSolutions.__len__() > 0:
		if solutionProperties.elementSolutions[-1].converged:
			previousSolution = solutionProperties.elementSolutions[-1]
			a = previousSolution.a
			aPrime = previousSolution.aPrime

	solidity = Solidity(radius, solutionProperties)

	for iteration in range(maxIterations):

		# Find the forces on the blade element
		dCax, dCtg = BEMElement(radius, a, aPrime, solutionProperties)

		prandtlCorrection = PrandtlCorrection(radius, a, solutionProperties)

		# At tip, Prandtl correction will converge to 0 and we reach unity (DIV0)
		# HACK: We can assume minimal change and return values from previous solution
		if prandtlCorrection < 1e-6:
			solutionProperties.addSolutionIteration(st.IterationSolution(radius, a, aPrime, iteration, 0.0, prandtlCorrection))
			return a, aPrime

		inflowAngle = Inflowangle(radius, a, aPrime, solutionProperties)

		# Correct induction factors
		sinInflow = np.maximum(np.sin(inflowAngle), EPSILON)
		cosInflow = np.cos(inflowAngle)
		kappa = solidity * dCax / (4 * prandtlCorrection * sinInflow**2)
		aMomentum = kappa / (1 + kappa)

		kappaPrime = solidity * dCtg / (4 * prandtlCorrection * sinInflow * cosInflow)
		aPrimeMomentum = kappaPrime / (1 - kappaPrime)

		# Glauert correction
		aMomentum = GlauertCorrection(aMomentum, radius, a, aPrime, solidity, dCax, solutionProperties)

		# Check for convergence
		if np.abs(a - aMomentum) < tolerance and np.abs(aPrime - aPrimeMomentum) < tolerance:

			# We have converged!
			solution = st.IterationSolution(
				radius, 
				aMomentum, 
				aPrimeMomentum, 
				iteration, 
				max(np.abs(a - aMomentum), 
				np.abs(aPrime - aPrimeMomentum)), 
				prandtlCorrection)
			
			solutionProperties.addSolutionIteration(solution)
			
			return aMomentum, aPrimeMomentum

		# Not converged yet
		# Assign new values with relaxation
		a = relaxation * aMomentum + (1 - relaxation) * a
		aPrime = relaxation * aPrimeMomentum + (1 - relaxation) * aPrime

	if DEBUG_PRINT: print("No convergence reached after maximum iterations")
	return a, aPrime

# Computes the corrected converged differential axial thrust and torque multiplied by the bladeNumber
def solveElement(radius:float, solutionProperties:st.SolverData=None) -> tuple:
	"""
	Solves the BEM element at a given radius
	---
		radius (float): The radius at which to solve the element.
		solutionProperties (SolutionProperties): An instance of SolutionProperties containing settings for the solver and convergence history
	"""
	a, aPrime = SolveInduction(radius, solutionProperties)
	dCax, dCtg = BEMElement(radius, a, aPrime, solutionProperties)

	geometry = solutionProperties.geometry
	airDensity = solutionProperties.airDensity

	relativeVelocity = np.sqrt((geometry.freeStreamVelocity * (1 - a))**2 + (geometry.rotationalSpeed * radius * (1 + aPrime))**2)

	dT = 0.5 * airDensity * relativeVelocity**2 * geometry.bladeNumber * geometry.chordFunction(radius) * dCax
	dQ = 0.5 * airDensity * relativeVelocity**2 * geometry.bladeNumber * geometry.chordFunction(radius) * radius * dCtg

	return dT, dQ

def SolveBEM(solutionProperties:st.SolverData):
	"""
	Solve the complete BEM problem for all radial stations.
	
	Iterates through radial positions from root to tip, solving for induction
	factors and calculating thrust and torque distributions. Integrates to obtain
	total thrust, torque, and power coefficient.
	
	Args:
		solutionProperties: SolverData instance containing geometry, solver settings,
		                    and will be updated with results
		
	Returns:
		None (results are stored in solutionProperties.result)
	"""
	geometry = solutionProperties.geometry
	
	radius = np.linspace(geometry.bladeStart * geometry.tipRadius, geometry.bladeEnd * geometry.tipRadius - 0.001 * geometry.tipRadius, num=solutionProperties.elementCount) # Avoid tip singularities

	dT = np.array([])
	dQ = np.array([])

	thrust = 0.0
	torque = 0.0
	
	for radiusIterator in radius:
		if DEBUG_PRINT: print("-" * 20)
		if DEBUG_PRINT: print(f"Solving for radius: {radiusIterator:.2f} m")

		dTCurrent, dQCurrent = solveElement(radiusIterator, solutionProperties)

		dT = np.append(dT, dTCurrent)
		dQ = np.append(dQ, dQCurrent)

		thrust += dTCurrent * (radius[1] - radius[0]) # Integration
		torque += dQCurrent * (radius[1] - radius[0]) # Integration

		convergenceData = solutionProperties.getElementSolution(radiusIterator)
		if convergenceData:
			if DEBUG_PRINT: print(f"Solution converged: {convergenceData.converged}, iterations: {convergenceData.iterations}, precision: {convergenceData.precision}")
		else:
			if DEBUG_PRINT: print(f"No convergence data found for radius: {radiusIterator:.2f} m")
		if DEBUG_PRINT: print(f"Differential thrust: {dTCurrent:.2f} N/m, Differential torque: {dQCurrent:.2f} Nm/m")


	print("=" * 40)
	print("Final results:")
	print(f"Total thrust: {thrust:.2f} N, Total torque: {torque:.2f} Nm")
	
	# Calculate coefficient of power
	rotorArea = np.pi * geometry.tipRadius**2
	power = torque * geometry.rotationalSpeed  # Power = Torque * angular velocity
	windPower = 0.5 * solutionProperties.airDensity * rotorArea * geometry.freeStreamVelocity**3  # Available wind power
	cP = power / windPower if windPower > 0 else 0
	print(f"Total power: {power:.2f} W, Coefficient of power cP: {cP:.4f}")

	solutionProperties.result = st.Result(radius, dT, dQ, cP, thrust, torque)

	return

if __name__ == "__main__":

	###### PART 1: SOLVE BEM AND PLOT RESULTS ######

	for tsr in [8.0]:
		solution1 = st.SolverData()
		solution1.setParameters(
			maxIterations=500, 
			tolerance=1e-6, 
			relaxation=0.5,
			elementCount=100)
		
		solution1.geometry.tipSpeedRatio = tsr
		
		SolveBEM(solution1)

		# Plot main BEM results
		plot_bem_results(solution1.result.radius, solution1.result.dT, solution1.result.dQ, solution1.elementSolutions)

	# Pitch as a function of radius
	localSolution = st.SolverData()
	localSolution.setParameters(
		maxIterations=1000,
		tolerance=1e-4,
		relaxation=0.5,
		elementCount=1
	)

	geometry = localSolution.geometry

	pitchByRadius = lambda radiusValue: -2.0 + 4.0 * (1 - geometry.dimensionlessRadialPosition(radiusValue)) # deg
	radiusSweep = np.linspace(geometry.bladeStart * geometry.tipRadius, geometry.bladeEnd * geometry.tipRadius - 0.001 * geometry.tipRadius, num=80)

	# Discretized map where each cell is one (radius, pitch) region
	pitchMinDeg = -10.0
	pitchMaxDeg = 10.0
	pitchBins = 50
	radiusBins = 80

	pitchGrid = np.linspace(pitchMinDeg, pitchMaxDeg, num=pitchBins)
	radiusGrid = np.linspace(geometry.bladeStart * geometry.tipRadius, geometry.bladeEnd * geometry.tipRadius - 0.001 * geometry.tipRadius, num=radiusBins)

	dTMap = np.zeros((pitchBins, radiusBins))
	dQMap = np.zeros((pitchBins, radiusBins))
	cPMap = np.zeros((pitchBins, radiusBins))

	for pitchIndex, pitchValue in enumerate(pitchGrid):
		for radiusIndex, radiusValue in enumerate(radiusGrid):
			

			# Override the blade pitch for this solution
			localSolution.geometry.bladePitch = pitchValue

			dTValue, dQValue = solveElement(radiusValue, localSolution)
			dTMap[pitchIndex, radiusIndex] = dTValue
			dQMap[pitchIndex, radiusIndex] = dQValue
			
			# Calculate local cP
			localPower = dQValue * localSolution.geometry.rotationalSpeed
			localRotorArea = np.pi * localSolution.geometry.tipRadius**2
			localWindPower = 0.5 * localSolution.airDensity * localRotorArea * localSolution.geometry.freeStreamVelocity**3
			cPMap[pitchIndex, radiusIndex] = localPower / localWindPower if localWindPower > 0 else 0

	plot_pitch_variations(radiusGrid, pitchGrid, cPMap)

	ShowPlots()