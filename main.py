"""
Main Execution Script for BEM Solver
"""

import SolverTools as st
import Plotting
import numpy as np
import BEM

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
		
		BEM.SolveBEM(solution1)

		# Plot main BEM results
		# Plotting.plot_bem_results(solution1.result.radius, solution1.elementSolutions)

	# Pitch as a function of radius
	localSolution = st.SolverData()
	localSolution.setParameters(
		maxIterations=1000,
		tolerance=1e-4,
		relaxation=0.5,
		elementCount=1
	)

	geometry = localSolution.geometry

	def pitchByRadius(radiusValue):
		return -2.0 + 4.0 * (1 - geometry.dimensionlessRadialPosition(radiusValue)) # deg
	
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

			# NOTE: Pitch angle should be constant along the blade???
			# NOTE: There is an optimal inflow angle for maximum power, see cp of the airfoil
			#	Maybe we just need to enforce the optimal angle of attack for all radii to get best performance?
			# 	Could be compared to this approach I guess 

			dTValue, dQValue = BEM.solveElement(radiusValue, localSolution)
			dTMap[pitchIndex, radiusIndex] = dTValue
			dQMap[pitchIndex, radiusIndex] = dQValue
			
			# Calculate local cP
			localPower = dQValue * localSolution.geometry.rotationalSpeed
			localRotorArea = np.pi * localSolution.geometry.tipRadius**2
			localWindPower = 0.5 * localSolution.airDensity * localRotorArea * localSolution.geometry.freeStreamVelocity**3
			cPMap[pitchIndex, radiusIndex] = localPower / localWindPower if localWindPower > 0 else 0

	# Plotting.plot_pitch_variations(radiusGrid, pitchGrid, cPMap)
	Plotting.PlotResults(solution1)
	Plotting.ShowPlots()